# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.hf_datasets.text_datasets import (
    build_ar_sft_dataloader,
    build_dllm_dataloader,
    build_dllm_sft_dataloader,
    build_text_dataloader,
)
from torchtitan.models.moe import MoEArgs
from torchtitan.protocols.train_spec import TrainSpec, register_train_spec

from .infra.parallelize import parallelize_qwen3_5
from .model.args import Qwen3_5ModelArgs
from .model.dllm_args import Qwen3_5DLLMModelArgs
from .model.dllm_model import Qwen3_5DLLMModel
from .model.model import Qwen3_5Model
from .model.state_dict_adapter import Qwen3_5StateDictAdapter

__all__ = [
    "parallelize_qwen3_5",
    "Qwen3_5ModelArgs",
    "Qwen3_5DLLMModelArgs",
    "Qwen3_5Model",
    "Qwen3_5DLLMModel",
    "qwen3_5_args",
    "qwen3_5_dllm_args",
]


def _to_qwen3_5_dllm_args(
    base_args: Qwen3_5ModelArgs, **overrides
) -> Qwen3_5DLLMModelArgs:
    values = dict(vars(base_args))
    values.update(overrides)
    return Qwen3_5DLLMModelArgs(**values)

qwen3_5_args = {
    "debugmodel": Qwen3_5ModelArgs(
        vocab_size=2048,
        max_seq_len=4096,
        head_dim=64,
        dim=256,
        n_layers=8,
        n_heads=4,
        n_kv_heads=2,
        qk_norm=True,
        hidden_dim=512,
        rope_theta=1000000,
        partial_rotary_factor=0.25,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        full_attention_interval=4,
        enable_weight_tying=True,
    ),
    # Qwen3.5-0.8B (dense, weight-tied, from Qwen/Qwen3.5-0.8B)
    "0.8B": Qwen3_5ModelArgs(
        vocab_size=248320,
        max_seq_len=262144,
        head_dim=256,
        dim=1024,
        n_layers=24,
        n_heads=8,
        n_kv_heads=2,
        qk_norm=True,
        hidden_dim=3584,
        rope_theta=10000000,
        partial_rotary_factor=0.25,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        full_attention_interval=4,
        enable_weight_tying=True,
    ),
    # Qwen3.5-2B (dense, weight-tied, from Qwen/Qwen3.5-2B)
    "2B": Qwen3_5ModelArgs(
        vocab_size=248320,
        max_seq_len=262144,
        head_dim=256,
        dim=2048,
        n_layers=24,
        n_heads=8,
        n_kv_heads=2,
        qk_norm=True,
        hidden_dim=6144,
        rope_theta=10000000,
        partial_rotary_factor=0.25,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        full_attention_interval=4,
        enable_weight_tying=True,
    ),
    # Qwen3.5-4B (dense, weight-tied, from Qwen/Qwen3.5-4B)
    "4B": Qwen3_5ModelArgs(
        vocab_size=248320,
        max_seq_len=262144,
        head_dim=256,
        dim=2560,
        n_layers=32,
        n_heads=16,
        n_kv_heads=4,
        qk_norm=True,
        hidden_dim=9216,
        rope_theta=10000000,
        partial_rotary_factor=0.25,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        full_attention_interval=4,
        enable_weight_tying=True,
    ),
    # Qwen3.5-9B (dense)
    "9B": Qwen3_5ModelArgs(
        vocab_size=248320,
        max_seq_len=262144,
        head_dim=256,
        dim=4096,
        n_layers=32,
        n_heads=16,
        n_kv_heads=4,
        qk_norm=True,
        hidden_dim=12288,
        rope_theta=10000000,
        partial_rotary_factor=0.25,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        full_attention_interval=4,
    ),
    # Qwen3.5-27B (dense)
    "27B": Qwen3_5ModelArgs(
        vocab_size=248320,
        max_seq_len=262144,
        head_dim=256,
        dim=5120,
        n_layers=64,
        n_heads=24,
        n_kv_heads=4,
        qk_norm=True,
        hidden_dim=17408,
        rope_theta=10000000,
        partial_rotary_factor=0.25,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        full_attention_interval=4,
    ),
    # MoE debug model
    "debugmodel_moe": Qwen3_5ModelArgs(
        vocab_size=2048,
        max_seq_len=4096,
        head_dim=64,
        dim=256,
        n_layers=8,
        n_heads=4,
        n_kv_heads=2,
        qk_norm=True,
        hidden_dim=512,
        rope_theta=1000000,
        partial_rotary_factor=0.25,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        full_attention_interval=4,
        moe_enabled=True,
        moe_inter_dim=256,
        moe_args=MoEArgs(
            num_experts=8,
            num_shared_experts=0,
            top_k=2,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            score_before_experts=False,
        ),
    ),
    # Qwen3.5-35B-A3B (MoE)
    "35B-A3B": Qwen3_5ModelArgs(
        vocab_size=248320,
        max_seq_len=262144,
        head_dim=256,
        dim=2048,
        n_layers=40,
        n_heads=16,
        n_kv_heads=2,
        qk_norm=True,
        hidden_dim=6144,
        rope_theta=10000000,
        partial_rotary_factor=0.25,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        full_attention_interval=4,
        moe_enabled=True,
        moe_inter_dim=512,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=1,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            score_before_experts=False,
        ),
    ),
}


qwen3_5_dllm_args = {
    name: _to_qwen3_5_dllm_args(args)
    for name, args in qwen3_5_args.items()
}


def _build_dllm_loss(job_config, **kwargs):
    return Qwen3_5DLLMModel.compute_loss


def _build_hybrid_diffusion_sft_loss(job_config, **kwargs):
    return Qwen3_5DLLMModel.compute_loss


def _ar_sft_loss(pred, labels):
    """AR SFT loss: passthrough for fused CE, standard CE for logits."""
    if isinstance(pred, dict) and "loss" in pred:
        return pred["loss"]
    import torch

    valid = labels != -100
    if not valid.any():
        return pred.sum() * 0.0
    return torch.nn.functional.cross_entropy(
        pred[valid].float(), labels[valid], reduction="mean"
    )


def _build_ar_sft_loss(job_config, **kwargs):
    return _ar_sft_loss


def get_train_spec() -> TrainSpec:
    """Train spec for standard Qwen3.5 (CLM, language-only)."""
    return TrainSpec(
        model_cls=Qwen3_5Model,
        model_args=qwen3_5_args,
        parallelize_fn=parallelize_qwen3_5,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Qwen3_5StateDictAdapter,
    )


def get_ar_sft_train_spec() -> TrainSpec:
    """Train spec for standard Qwen3.5 AR SFT (ChatML + prompt masking)."""
    return TrainSpec(
        model_cls=Qwen3_5Model,
        model_args=qwen3_5_args,
        parallelize_fn=parallelize_qwen3_5,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_ar_sft_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=_build_ar_sft_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Qwen3_5StateDictAdapter,
    )


register_train_spec("qwen3_5_ar_sft", get_ar_sft_train_spec())


def get_dllm_train_spec() -> TrainSpec:
    """Train spec for native Qwen3.5 DLLM (language-only)."""
    return TrainSpec(
        model_cls=Qwen3_5DLLMModel,
        model_args=qwen3_5_dllm_args,
        parallelize_fn=parallelize_qwen3_5,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dllm_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=_build_dllm_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Qwen3_5StateDictAdapter,
    )


def get_dllm_sft_train_spec() -> TrainSpec:
    """Train spec for native Qwen3.5 DLLM SFT (language-only)."""
    return TrainSpec(
        model_cls=Qwen3_5DLLMModel,
        model_args=qwen3_5_dllm_args,
        parallelize_fn=parallelize_qwen3_5,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dllm_sft_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=_build_dllm_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Qwen3_5StateDictAdapter,
    )


def get_hybrid_diffusion_sft_train_spec() -> TrainSpec:
    """Train spec for Qwen3.5 HybridDiffusion SFT (language-only).

    All training behavior (logit_shift, causal_x0/xt, AR loss, etc.)
    is controlled by [dllm] config flags inside forward(). With
    use_fused_ce=true the loss function is a passthrough.
    """
    return TrainSpec(
        model_cls=Qwen3_5DLLMModel,
        model_args=qwen3_5_dllm_args,
        parallelize_fn=parallelize_qwen3_5,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dllm_sft_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=_build_hybrid_diffusion_sft_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Qwen3_5StateDictAdapter,
    )
