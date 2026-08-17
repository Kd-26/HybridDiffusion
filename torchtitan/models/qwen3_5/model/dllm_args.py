from dataclasses import dataclass

import torch.nn as nn

from torchtitan.config import JobConfig
from torchtitan.models.qwen3_5.model.args import Qwen3_5ModelArgs
from torchtitan.tools.logging import logger


@dataclass
class Qwen3_5DLLMModelArgs(Qwen3_5ModelArgs):
    """Qwen3.5 args extended with block diffusion training controls."""

    mask_token_id: int = -1
    pad_token_id: int = -1
    block_size: int = 4
    block_train_method: str = "auto"
    block_train_conv_method: str = "auto"
    chunk_wy_bwd_split_enabled: bool | None = None
    chunk_wy_bwd_parallel_groups: int = 0
    chunk_wy_bwd_checkpoint_stride: int = 0
    chunk_wy_bwd_bv: int = 0
    chunk_wy_bwd_store_b: bool = False
    antithetic_sampling: bool = True
    complementary_mask: bool = False
    serial_complementary_mask: bool = False
    use_flex_attention: bool = True
    flex_attention_q_block_size: int = -1
    flex_attention_kv_block_size: int = -1
    use_fused_ce: bool = False
    enable_weight_tying: bool = False

    causal_x0: bool = False
    causal_xt: bool = False
    all_masked: bool = False
    logit_shift: bool = False
    ar_loss_weight: float = 0.0
    loss_auto_balance: bool = False
    loss_normalize: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.mask_token_id < 0:
            self.mask_token_id = self.vocab_size - 1
        if self.pad_token_id < 0:
            self.pad_token_id = self.eos_id
        self._validate_dllm()

    def _validate_dllm(self) -> None:
        if self.ar_loss_weight > 0 and not self.causal_x0:
            raise ValueError(
                "ar_loss_weight > 0 requires causal_x0 = True. "
                "Block-causal x_0 lets tokens see the future within a block, "
                "which invalidates AR next-token prediction."
            )
        if self.causal_xt and not self.causal_x0:
            raise ValueError(
                "causal_xt = True requires causal_x0 = True."
            )
        if self.loss_auto_balance and self.ar_loss_weight <= 0:
            raise ValueError(
                "loss_auto_balance = True requires ar_loss_weight > 0 "
                "(need two losses to balance)."
            )
        if self.all_masked and self.complementary_mask:
            logger.warning(
                "all_masked=True makes complementary_mask meaningless "
                "(flipping all-masked gives all-unmasked). "
                "Forcing complementary_mask=False."
            )
            self.complementary_mask = False

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        super().update_from_config(job_config, **kwargs)
        if hasattr(job_config, "dllm"):
            cfg = job_config.dllm
            if hasattr(cfg, "block_size"):
                self.block_size = cfg.block_size
            if hasattr(cfg, "mask_token_id") and cfg.mask_token_id >= 0:
                self.mask_token_id = cfg.mask_token_id
            if hasattr(cfg, "pad_token_id") and cfg.pad_token_id >= 0:
                self.pad_token_id = cfg.pad_token_id
            if hasattr(cfg, "antithetic_sampling"):
                self.antithetic_sampling = cfg.antithetic_sampling
            if hasattr(cfg, "complementary_mask"):
                self.complementary_mask = cfg.complementary_mask
            if hasattr(cfg, "serial_complementary_mask"):
                self.serial_complementary_mask = cfg.serial_complementary_mask
            if hasattr(cfg, "use_flex_attention"):
                self.use_flex_attention = cfg.use_flex_attention
            if hasattr(cfg, "flex_attention_q_block_size"):
                self.flex_attention_q_block_size = cfg.flex_attention_q_block_size
            if hasattr(cfg, "flex_attention_kv_block_size"):
                self.flex_attention_kv_block_size = cfg.flex_attention_kv_block_size
            if hasattr(cfg, "block_train_method"):
                self.block_train_method = cfg.block_train_method
            if hasattr(cfg, "block_train_conv_method"):
                self.block_train_conv_method = cfg.block_train_conv_method
            if hasattr(cfg, "chunk_wy_bwd_split_enabled"):
                self.chunk_wy_bwd_split_enabled = cfg.chunk_wy_bwd_split_enabled
            if hasattr(cfg, "chunk_wy_bwd_parallel_groups"):
                self.chunk_wy_bwd_parallel_groups = cfg.chunk_wy_bwd_parallel_groups
            if hasattr(cfg, "chunk_wy_bwd_checkpoint_stride"):
                self.chunk_wy_bwd_checkpoint_stride = cfg.chunk_wy_bwd_checkpoint_stride
            if hasattr(cfg, "chunk_wy_bwd_bv"):
                self.chunk_wy_bwd_bv = cfg.chunk_wy_bwd_bv
            if hasattr(cfg, "chunk_wy_bwd_store_b"):
                self.chunk_wy_bwd_store_b = cfg.chunk_wy_bwd_store_b
            if hasattr(cfg, "use_fused_ce"):
                self.use_fused_ce = cfg.use_fused_ce
            if hasattr(cfg, "enable_weight_tying"):
                self.enable_weight_tying = cfg.enable_weight_tying
            if hasattr(cfg, "causal_x0"):
                self.causal_x0 = cfg.causal_x0
            if hasattr(cfg, "causal_xt"):
                self.causal_xt = cfg.causal_xt
            if hasattr(cfg, "all_masked"):
                self.all_masked = cfg.all_masked
            if hasattr(cfg, "logit_shift"):
                self.logit_shift = cfg.logit_shift
            if hasattr(cfg, "ar_loss_weight"):
                self.ar_loss_weight = cfg.ar_loss_weight
            if hasattr(cfg, "loss_auto_balance"):
                self.loss_auto_balance = cfg.loss_auto_balance
            if hasattr(cfg, "loss_normalize"):
                self.loss_normalize = cfg.loss_normalize
        self._validate_dllm()

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, int]:
        (
            nparams,
            tokenwise_flops_per_token,
            full_attention_flops_per_token,
        ) = self._get_hybrid_flops_breakdown(model, seq_len)
        complementary_factor = 2 if self.complementary_mask else 1
        dllm_flops_per_input_token = complementary_factor * (
            2 * tokenwise_flops_per_token + 2 * full_attention_flops_per_token
        )
        return nparams, dllm_flops_per_input_token
