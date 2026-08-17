# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

_supported_models = frozenset(
    [
        "llama3",
        "llama4",
        "qwen3",
        "qwen3_ar_sft",
        "qwen3_dllm",
        "qwen3_dllm_sft",
        "qwen3_fast_dllm",
        "qwen3_fast_dllm_sft",
        "qwen3_hybrid_diffusion_sft",
        "qwen3_5",
        "qwen3_5_ar_sft",
        "qwen3_5_dllm",
        "qwen3_5_dllm_sft",
        "qwen3_5_hybrid_diffusion_sft",
    ]
)

_model_aliases = {
    "qwen3_ar_sft": ("torchtitan.models.qwen3", "get_ar_sft_train_spec"),
    "qwen3_dllm": ("torchtitan.models.qwen3", "get_dllm_train_spec"),
    "qwen3_dllm_sft": ("torchtitan.models.qwen3", "get_dllm_sft_train_spec"),
    "qwen3_fast_dllm": ("torchtitan.models.qwen3", "get_fast_dllm_train_spec"),
    "qwen3_fast_dllm_sft": (
        "torchtitan.models.qwen3",
        "get_fast_dllm_sft_train_spec",
    ),
    "qwen3_hybrid_diffusion_sft": (
        "torchtitan.models.qwen3",
        "get_hybrid_diffusion_sft_train_spec",
    ),
    "qwen3_5_ar_sft": ("torchtitan.models.qwen3_5", "get_ar_sft_train_spec"),
    "qwen3_5_dllm": ("torchtitan.models.qwen3_5", "get_dllm_train_spec"),
    "qwen3_5_dllm_sft": ("torchtitan.models.qwen3_5", "get_dllm_sft_train_spec"),
    "qwen3_5_hybrid_diffusion_sft": (
        "torchtitan.models.qwen3_5",
        "get_hybrid_diffusion_sft_train_spec",
    ),
}
