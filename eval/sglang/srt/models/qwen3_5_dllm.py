"""Qwen3.5 backbone with dLLM (block diffusion) inference support.

Hybrid model: softmax attention layers + GDN (Gated Delta Net) linear attention
layers. For dLLM, softmax layers use ENCODER_ONLY attention (non-causal masks
controlled by dllm_force_causal / dllm_force_bidir_mask flags). GDN layers use
a dLLM-aware backend that handles state snapshot/restore.

Use with ``--dllm-algorithm {LowConfidence,JointThreshold,HybridDiffusionSelfSpec}``.

Architecture follows the same pattern as qwen3_dllm.py (Qwen3 dLLM variant):
inherit from the base Qwen3.5 model, override only the attention declarations.
"""

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.models.qwen3_5 import (
    Qwen3_5AttentionDecoderLayer,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ForCausalLM,
    Qwen3_5LinearDecoderLayer,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix


class Qwen3_5DLLMAttentionDecoderLayer(Qwen3_5AttentionDecoderLayer):
    """Qwen3.5 softmax attention decoder layer with ENCODER_ONLY attention for dLLM.

    Only change vs parent: self.attn is re-created with
    ``attn_type=AttentionType.ENCODER_ONLY`` so that the flashinfer backend
    allows non-causal / custom block masks during DLLM_EXTEND.
    """

    def __init__(
        self,
        config,
        layer_id: int = 0,
        quant_config=None,
        prefix: str = "",
        alt_stream=None,
        is_nextn: bool = False,
    ) -> None:
        super().__init__(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            alt_stream=alt_stream,
            is_nextn=is_nextn,
        )
        # Override the attention with ENCODER_ONLY so dllm_extend can provide
        # non-causal masks (block diffusion V1-V4).
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            attn_type=AttentionType.ENCODER_ONLY,
            prefix=add_prefix("attn", prefix),
        )


# Layer type mapping for dLLM.
# Softmax layers: replaced with ENCODER_ONLY variant.
# GDN layers: unchanged — dLLM behavior comes from the backend (GDNDllmBackend),
# not from the model layer class.
DLLM_DECODER_LAYER_TYPES = {
    "attention": Qwen3_5DLLMAttentionDecoderLayer,
    "linear_attention": Qwen3_5LinearDecoderLayer,
}


class Qwen3_5DLLMForCausalLM(Qwen3_5ForCausalLM):
    """Qwen3.5 text backbone with dLLM (block diffusion) support.

    Changes from Qwen3_5ForCausalLM:
    1. Softmax attention layers use ENCODER_ONLY so dLLM masks can be applied.
    2. GDN layers are dLLM-aware via GDNDllmBackend.

    This mirrors the naming in qwen3_5.py: Qwen3_5ForCausalLM is the language
    backbone used inside Qwen3_5ForConditionalGeneration, not the top-level
    serving module with lm_head/logits processing.
    """

    def __init__(
        self,
        config,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        if DllmConfig.from_server_args(get_global_server_args()) is None:
            return
        for layer in self.layers:
            if isinstance(layer, Qwen3_5AttentionDecoderLayer):
                layer.attn = RadixAttention(
                    layer.num_heads,
                    layer.head_dim,
                    layer.scaling,
                    num_kv_heads=layer.num_kv_heads,
                    layer_id=layer.layer_id,
                    attn_type=AttentionType.ENCODER_ONLY,
                    prefix=add_prefix(
                        f"layers.{layer.layer_id}.self_attn.attn", prefix
                    ),
                )


class Qwen3_5DLLMForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Top-level Qwen3.5 serving class for dLLM/self-spec inference.

    Qwen3.5 checkpoints are exported with `model.language_model.*` weights and
    the `Qwen3_5ForConditionalGeneration` architecture.  This wrapper preserves
    that outer structure while swapping the text backbone to the dLLM variant and
    returning full block logits for HybridDiffusionSelfSpec verification.
    """

    def __init__(
        self,
        config,
        quant_config=None,
        prefix: str = "",
        language_model_cls=Qwen3_5DLLMForCausalLM,
    ):
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            language_model_cls=language_model_cls,
        )
        self.logits_processor = LogitsProcessor(
            self.config, return_full_logits=True
        )


EntryClass = [Qwen3_5DLLMForConditionalGeneration, Qwen3_5DLLMForCausalLM]
