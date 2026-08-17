# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import List, Optional

from torch import nn

from torchtitan.config import JobConfig
from torchtitan.models.moe import MoEArgs
from torchtitan.protocols.train_spec import BaseModelArgs
from torchtitan.tools.logging import logger


@dataclass
class Qwen3_5ModelArgs(BaseModelArgs):
    """
    Qwen3.5 hybrid architecture args: full_attention (GQA + gated output +
    partial RoPE) interleaved with linear_attention (Gated DeltaNet + 1D
    causal conv).  Default layer pattern is 3:1 (every 4th layer is
    full_attention).

    Supports dense and MoE variants:
    - When ``moe_enabled`` is True the dense FFN is replaced by a
      Mixture-of-Experts layer (Qwen3.5-MoE family, e.g. 35B-A3B).
    """

    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 16
    n_kv_heads: int = 4
    vocab_size: int = 248320
    head_dim: int = 256
    hidden_dim: int = 12288

    norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    partial_rotary_factor: float = 0.25
    qk_norm: bool = True
    max_seq_len: int = 32768
    depth_init: bool = True

    attn_type: str = "sdpa"
    attn_mask_type: str = "causal"
    eos_id: int = 248044

    enable_weight_tying: bool = False

    # Linear attention (Gated DeltaNet) params
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32

    layer_types: Optional[List[str]] = None
    full_attention_interval: int = 4

    # MoE params
    moe_enabled: bool = False
    moe_inter_dim: int = 512
    moe_args: MoEArgs = field(default_factory=MoEArgs)

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention"
                if bool((i + 1) % self.full_attention_interval)
                else "full_attention"
                for i in range(self.n_layers)
            ]
        assert len(self.layer_types) == self.n_layers, (
            f"layer_types length {len(self.layer_types)} != n_layers {self.n_layers}"
        )

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

        self.moe_args._debug_force_load_balance = (
            job_config.debug.moe_force_load_balance
        )

        enable_packing = (
            getattr(job_config.dllm, "enable_packing", False)
            if hasattr(job_config, "dllm")
            else False
        )
        if enable_packing:
            if self.attn_mask_type != "doc_causal":
                self.attn_mask_type = "doc_causal"
            if self.attn_type not in ("flex", "varlen"):
                self.attn_type = "varlen"
            logger.info(
                f"Packing enabled: using {self.attn_type} attention with "
                "doc_causal masking for cross-document isolation."
            )

    def _get_hybrid_flops_breakdown(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, int, int]:
        """Return total params plus token-wise and full-attention FLOPs/token."""
        nparams_embedding = 0

        if self.moe_enabled:
            nparams_moe_router = 0
            nparams_shared_experts = 0
            nparams_experts = 0
            nparams_dense = 0

            for name, p in model.named_parameters():
                if "embedding" in name:
                    nparams_embedding += p.numel()
                    nparams_dense += p.numel()
                elif "moe.shared_experts" in name:
                    nparams_shared_experts += p.numel()
                elif "moe.router" in name:
                    nparams_moe_router += p.numel()
                elif "moe.experts" in name:
                    nparams_experts += p.numel()
                else:
                    nparams_dense += p.numel()

            nparams_sparse = (
                nparams_moe_router + nparams_shared_experts + nparams_experts
            )
            nparams_sparse_active = (
                nparams_moe_router
                + nparams_shared_experts
                + nparams_experts * self.moe_args.top_k // self.moe_args.num_experts
            )
            nparams = nparams_dense + nparams_sparse
            active_non_embedding_params = (
                nparams_dense - nparams_embedding + nparams_sparse_active
            )
        else:
            nparams = sum(p.numel() for p in model.parameters())
            nparams_embedding = sum(
                sum(p.numel() for p in module.parameters())
                for module in model.modules()
                if isinstance(module, nn.Embedding)
            )
            active_non_embedding_params = nparams - nparams_embedding

        tokenwise_flops_per_token = 6 * active_non_embedding_params
        num_full_attention_layers = sum(
            layer_type == "full_attention" for layer_type in self.layer_types
        )
        full_attention_flops_per_token = (
            6 * num_full_attention_layers * self.n_heads * (2 * self.head_dim) * seq_len
        )

        if self.enable_weight_tying:
            nparams = nparams - nparams_embedding

        return nparams, tokenwise_flops_per_token, full_attention_flops_per_token

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, int]:
        nparams, tokenwise_flops_per_token, full_attention_flops_per_token = (
            self._get_hybrid_flops_breakdown(model, seq_len)
        )
        num_flops_per_token = (
            tokenwise_flops_per_token + full_attention_flops_per_token
        )
        return nparams, num_flops_per_token
