# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.hf_datasets.text_datasets import build_ar_sft_dataloader, build_text_dataloader
from torchtitan.models.moe import MoEArgs
from torchtitan.protocols.train_spec import TrainSpec, register_train_spec

from .infra.parallelize import parallelize_qwen3
from .model.args import Qwen3ModelArgs
from .model.model import Qwen3Model
from .model.model_dllm import DLLM, DLLMModelArgs
from .model.state_dict_adapter import Qwen3StateDictAdapter
from .text_datasets import build_dllm_dataloader, build_dllm_sft_dataloader

__all__ = [
    "parallelize_qwen3",
    "Qwen3ModelArgs",
    "Qwen3Model",
    "DLLM",
    "DLLMModelArgs",
    "qwen3_args",
    "dllm_args",
]

# Adding different variants of the model

qwen3_args = {
    "debugmodel": Qwen3ModelArgs(
        vocab_size=2048,
        max_seq_len=4096,
        head_dim=128,
        dim=256,
        n_layers=8,
        n_heads=16,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=3072,
        rope_theta=1000000,
        enable_weight_tying=True,
    ),
    "0.6B": Qwen3ModelArgs(
        vocab_size=151936,
        max_seq_len=4096,
        head_dim=128,
        dim=1024,
        n_layers=28,
        n_heads=16,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=3072,
        rope_theta=1000000,
        enable_weight_tying=True,
    ),
    "1.7B": Qwen3ModelArgs(
        vocab_size=151936,
        max_seq_len=4096,
        head_dim=128,
        dim=2048,
        n_layers=28,
        n_heads=16,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=6144,
        rope_theta=1000000,
        enable_weight_tying=True,
    ),
    "4B": Qwen3ModelArgs(
        vocab_size=151936,
        max_seq_len=4096,
        head_dim=128,
        dim=2560,
        n_layers=36,
        n_heads=32,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=9728,
        rope_theta=1000000,
        enable_weight_tying=True,
    ),
    "8B": Qwen3ModelArgs(
        vocab_size=151936,
        max_seq_len=4096,
        head_dim=128,
        dim=4096,
        n_layers=36,
        n_heads=32,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=12288,
        rope_theta=1000000,
    ),
    "14B": Qwen3ModelArgs(
        vocab_size=151936,
        max_seq_len=4096,
        head_dim=128,
        dim=5120,
        n_layers=40,
        n_heads=40,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=17408,
        rope_theta=1000000,
    ),
    "32B": Qwen3ModelArgs(
        vocab_size=151936,
        max_seq_len=4096,
        head_dim=128,
        dim=5120,
        n_layers=64,
        n_heads=64,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=25600,
        rope_theta=1000000,
    ),
    # Qwen3-MoE models
    "debugmodel_moe": Qwen3ModelArgs(
        vocab_size=2048,
        max_seq_len=4096,
        head_dim=128,
        dim=256,
        n_layers=8,
        n_heads=16,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=3072,
        rope_theta=1000000,
        moe_enabled=True,
        moe_inter_dim=768,
        moe_args=MoEArgs(
            num_experts=64,
            num_shared_experts=0,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            score_before_experts=False,
        ),
    ),
    "30B-A3B": Qwen3ModelArgs(
        vocab_size=151936,
        max_seq_len=262144,
        head_dim=128,
        dim=2048,
        n_layers=48,
        n_heads=32,
        n_kv_heads=4,
        qk_norm=True,
        hidden_dim=6144,
        rope_theta=1000000,
        moe_enabled=True,
        moe_inter_dim=768,
        moe_args=MoEArgs(
            num_experts=128,
            num_shared_experts=0,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            score_before_experts=False,
        ),
    ),
    "235B-A22B": Qwen3ModelArgs(
        vocab_size=151936,
        max_seq_len=4096,
        head_dim=128,
        dim=4096,
        n_layers=94,
        n_heads=64,
        n_kv_heads=4,
        qk_norm=True,
        hidden_dim=12288,
        rope_theta=5000000,
        moe_enabled=True,
        moe_inter_dim=1536,
        moe_args=MoEArgs(
            num_experts=128,
            num_shared_experts=0,  # no shared experts, double check
            top_k=8,  # num_experts_per_tok
            score_func="softmax",  # need double check
            route_norm=True,
            route_scale=1.0,  # not needed, need double check
            score_before_experts=False,
        ),
    ),
}


# DLLM configurations (block diffusion)
# Note: mask_token_id=-1 defaults to vocab_size-1 (backward compat).
# New experiments: set mask_token_id=151669, pad_token_id=151643 in [dllm] config.
dllm_args = {
    "debugmodel": DLLMModelArgs(
        dim=256,
        n_layers=6,
        n_heads=16,
        vocab_size=2048,
        head_dim=128,
        hidden_dim=3072,
        block_size=16,
        enable_weight_tying=True,
        complementary_mask=True,
    ),
    "0.6B": DLLMModelArgs(
        dim=1024,
        n_layers=28,
        n_heads=16,
        n_kv_heads=8,
        head_dim=128,
        hidden_dim=3072,
        vocab_size=151936,
        qk_norm=True,
        rope_theta=1000000,
        block_size=16,
        enable_weight_tying=True,
        complementary_mask=True,
    ),
    "1.7B": DLLMModelArgs(
        dim=2048,
        n_layers=28,
        n_heads=16,
        n_kv_heads=8,
        head_dim=128,
        hidden_dim=6144,
        vocab_size=151936,
        qk_norm=True,
        rope_theta=1000000,
        block_size=16,
        enable_weight_tying=True,
        complementary_mask=True,
    ),
}


def _build_dllm_loss(job_config, **kwargs):
    return DLLM.compute_loss


def _build_fast_dllm_loss(job_config, **kwargs):
    return DLLM.compute_loss_fast_dllm


def _build_hybrid_diffusion_sft_loss(job_config, **kwargs):
    return DLLM.compute_loss_hybrid_diffusion_sft


def get_train_spec() -> TrainSpec:
    """Train spec for standard Qwen3 (CLM)."""
    return TrainSpec(
        model_cls=Qwen3Model,
        model_args=qwen3_args,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Qwen3StateDictAdapter,
    )


def get_dllm_train_spec() -> TrainSpec:
    """Train spec for DLLM (Qwen3 + diffusion)."""
    return TrainSpec(
        model_cls=DLLM,
        model_args=dllm_args,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dllm_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=_build_dllm_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Qwen3StateDictAdapter,
    )

def get_fast_dllm_train_spec() -> TrainSpec:
    """Train spec for DLLM with Fast-dLLM v2 style training (eps floor + HF loss)."""
    return TrainSpec(
        model_cls=DLLM,
        model_args=dllm_args,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dllm_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=_build_fast_dllm_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Qwen3StateDictAdapter,
    )


def get_dllm_sft_train_spec() -> TrainSpec:
    """Train spec for DLLM SFT (Qwen3 + diffusion + supervised fine-tuning)."""
    return TrainSpec(
        model_cls=DLLM,
        model_args=dllm_args,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dllm_sft_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=_build_dllm_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Qwen3StateDictAdapter,
    )

def get_fast_dllm_sft_train_spec() -> TrainSpec:
    """Train spec for Fast-dLLM SFT (eps floor + HF loss + supervised fine-tuning)."""
    return TrainSpec(
        model_cls=DLLM,
        model_args=dllm_args,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dllm_sft_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=_build_fast_dllm_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Qwen3StateDictAdapter,
    )

def get_hybrid_diffusion_sft_train_spec() -> TrainSpec:
    """Train spec for Qwen3 HybridDiffusion SFT."""
    return TrainSpec(
        model_cls=DLLM,
        model_args=dllm_args,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dllm_sft_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=_build_hybrid_diffusion_sft_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Qwen3StateDictAdapter,
    )

# Register qwen3_dllm so it can be used via name="qwen3_dllm" in config
register_train_spec("qwen3_dllm", get_dllm_train_spec())
register_train_spec("qwen3_fast_dllm", get_fast_dllm_train_spec())
register_train_spec("qwen3_dllm_sft", get_dllm_sft_train_spec())
register_train_spec("qwen3_fast_dllm_sft", get_fast_dllm_sft_train_spec())
register_train_spec("qwen3_hybrid_diffusion_sft", get_hybrid_diffusion_sft_train_spec())


def _ar_sft_loss(pred, labels):
    """AR SFT loss: passthrough for fused CE, standard CE for logits."""
    if isinstance(pred, dict) and "loss" in pred:
        return pred["loss"]
    import torch
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(), labels.flatten(0, 1)
    )


def _build_ar_sft_loss(job_config, **kwargs):
    return _ar_sft_loss


def get_ar_sft_train_spec() -> TrainSpec:
    """Train spec for standard AR SFT (Qwen3 + ChatML + prompt masking)."""
    return TrainSpec(
        model_cls=Qwen3Model,
        model_args=qwen3_args,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_ar_sft_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=_build_ar_sft_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Qwen3StateDictAdapter,
    )

register_train_spec("qwen3_ar_sft", get_ar_sft_train_spec())
