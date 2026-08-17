"""Qwen3 backbone with dLLM (block diffusion) inference support.

Re-uses sglang's Qwen3 model class (which is known to produce correct logits
for Qwen3-based checkpoints).  The only difference from Qwen3ForCausalLM is
that the attention layer is declared ``ENCODER_ONLY`` so the ``dllm_extend``
forward mode can provide block-diffusion-friendly masks (current block
bidirectional; previously committed blocks served via the paged KV cache).

Use together with ``--dllm-algorithm {LowConfidence,JointThreshold}``.
The SDARForCausalLM class is not used here because it adds a bunch of
unrelated SDAR-specific paths that, in practice, produce garbage logits for
our Qwen3-trained dLLM checkpoints.
"""

from typing import Any, Dict, Optional

import torch
from torch import nn

from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.models.qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
)
from sglang.srt.utils import add_prefix


class Qwen3DLLMAttention(Qwen3Attention):
    """Qwen3Attention with ``ENCODER_ONLY`` attention type for dLLM prefill."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        head_dim: Optional[int] = None,
        max_position_embeddings: int = 32768,
        quant_config=None,
        rms_norm_eps: Optional[float] = None,
        attention_bias: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            prefix=prefix,
            alt_stream=alt_stream,
        )
        # Re-create attn with ENCODER_ONLY so dllm_extend prefill passes a
        # non-causal mask (SDAR-style block-diffusion forward).
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            attn_type=AttentionType.ENCODER_ONLY,
            prefix=add_prefix("attn", prefix),
        )


class Qwen3DLLMDecoderLayer(Qwen3DecoderLayer):
    def __init__(
        self,
        config,
        layer_id: int = 0,
        quant_config=None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            alt_stream=alt_stream,
        )
        # Replace the self-attention with the dLLM-flavored one.
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_pos = getattr(config, "max_position_embeddings", 32768)
        head_dim = getattr(config, "head_dim", None)
        attention_bias = getattr(config, "attention_bias", False)
        self.self_attn = Qwen3DLLMAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            max_position_embeddings=max_pos,
            quant_config=quant_config,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=attention_bias,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
        )


class Qwen3DLLMModel(Qwen3Model):
    def __init__(self, config, quant_config=None, prefix: str = "", alt_stream=None):
        # Qwen3Model.__init__ -> Qwen2Model.__init__ which takes decoder_layer_type
        super(Qwen3Model, self).__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=Qwen3DLLMDecoderLayer,
            alt_stream=alt_stream,
        )


class Qwen3DLLMForCausalLM(Qwen3ForCausalLM):
    """Qwen3 backbone CausalLM with ENCODER_ONLY attention for dLLM decoding."""

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        # Invoke nn.Module.__init__ and then build submodules ourselves, so we
        # can swap in Qwen3DLLMModel (which uses our dLLM-attention decoder
        # layer) without duplicating Qwen3ForCausalLM's lm_head / pooler setup.
        nn.Module.__init__(self)
        from sglang.srt.distributed import get_pp_group
        from sglang.srt.layers.logits_processor import LogitsProcessor
        from sglang.srt.layers.pooler import Pooler, PoolingType
        from sglang.srt.layers.utils import PPMissingLayer
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
        from sglang.srt.server_args import get_global_server_args

        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3DLLMModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            self.lm_head = PPMissingLayer()

        # dLLM needs full-sequence logits, not just the last position
        self.logits_processor = LogitsProcessor(config, return_full_logits=True)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.capture_aux_hidden_states = False


EntryClass = Qwen3DLLMForCausalLM
