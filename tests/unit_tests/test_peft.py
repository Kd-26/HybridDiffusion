# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch
import torch.nn as nn

from torchtitan.components.peft.lora import LoRALinear
from torchtitan.components.peft.utils import canonicalize_fqn
from torchtitan.config import ConfigManager
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import build_model_converters


def build_parallel_dims(job_config, world_size):
    parallelism_config = job_config.parallelism
    return ParallelDims(
        dp_shard=parallelism_config.data_parallel_shard_degree,
        dp_replicate=parallelism_config.data_parallel_replicate_degree,
        cp=parallelism_config.context_parallel_degree,
        tp=parallelism_config.tensor_parallel_degree,
        pp=parallelism_config.pipeline_parallel_degree,
        ep=parallelism_config.expert_parallel_degree,
        etp=parallelism_config.expert_tensor_parallel_degree,
        world_size=world_size,
    )


def build_config(extra_args: list[str] | None = None):
    args = ["--model.converters", "peft", "--peft.lora.rank", "4"]
    if extra_args:
        args.extend(extra_args)
    config_manager = ConfigManager()
    return config_manager.parse_args(args)


class TinyPeftModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(32, 8)
        self.proj = nn.Linear(8, 8)
        self.output = nn.Linear(8, 32, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.tok_embeddings(tokens)
        hidden = self.proj(hidden)
        return self.output(hidden)


def test_canonicalize_fqn_strips_compile_wrapper_segments():
    assert (
        canonicalize_fqn("layers.0._orig_mod.attention.wq.weight")
        == "layers.0.attention.wq.weight"
    )


def test_peft_checkpoint_validation_rejects_hf_load():
    config = build_config(["--checkpoint.initial_load_in_hf"])
    parallel_dims = build_parallel_dims(config, 1)

    with pytest.raises(ValueError, match="HF initial checkpoint loading"):
        build_model_converters(config, parallel_dims)


def test_peft_checkpoint_validation_requires_model_only_load_plan():
    config = build_config(
        ["--checkpoint.initial_load_path", "/tmp/base-ckpt"]
    )
    parallel_dims = build_parallel_dims(config, 1)

    with pytest.raises(ValueError, match="initial_ckpt_load_plan"):
        build_model_converters(config, parallel_dims)


def test_lora_conversion_preserves_base_weight_key_and_adds_adapter_keys():
    config = build_config(["--peft.target_globs", "proj"])
    parallel_dims = build_parallel_dims(config, 1)
    model_converters = build_model_converters(config, parallel_dims)
    model = TinyPeftModel()

    model_converters.convert(model)

    assert isinstance(model.proj, LoRALinear)
    state_dict_keys = set(model.state_dict().keys())
    assert "proj.weight" in state_dict_keys
    assert "proj.bias" in state_dict_keys
    assert "proj.lora_a" in state_dict_keys
    assert "proj.lora_b" in state_dict_keys


def test_lora_zero_start_keeps_forward_identical_to_base_model():
    torch.manual_seed(0)
    baseline_model = TinyPeftModel().eval()
    peft_model = copy.deepcopy(baseline_model).eval()

    config = build_config(["--peft.target_globs", "proj"])
    parallel_dims = build_parallel_dims(config, 1)
    model_converters = build_model_converters(config, parallel_dims)

    model_converters.convert(peft_model)
    model_converters.post_model_init([peft_model])

    tokens = torch.randint(0, 32, (2, 4))
    baseline_out = baseline_model(tokens)
    peft_out = peft_model(tokens)

    torch.testing.assert_close(peft_out, baseline_out, rtol=0, atol=0)
    torch.testing.assert_close(
        peft_model.proj.lora_b,
        torch.zeros_like(peft_model.proj.lora_b),
        rtol=0,
        atol=0,
    )


def test_peft_freezes_base_params_and_can_unfreeze_embeddings():
    config = build_config(
        [
            "--peft.target_globs",
            "proj",
            "--peft.trainable_base_globs",
            "tok_embeddings.weight",
        ]
    )
    parallel_dims = build_parallel_dims(config, 1)
    model_converters = build_model_converters(config, parallel_dims)
    model = TinyPeftModel()

    model_converters.convert(model)
    model_converters.post_model_init([model])

    assert model.tok_embeddings.weight.requires_grad is True
    assert model.output.weight.requires_grad is False
    assert model.proj.weight.requires_grad is False
    assert model.proj.bias.requires_grad is False
    assert model.proj.lora_a.requires_grad is True
    assert model.proj.lora_b.requires_grad is True


def test_peft_strict_mode_errors_when_no_modules_match():
    config = build_config(["--peft.target_globs", "missing*"])
    parallel_dims = build_parallel_dims(config, 1)
    model_converters = build_model_converters(config, parallel_dims)
    model = TinyPeftModel()

    with pytest.raises(ValueError, match="matched no supported modules"):
        model_converters.convert(model)


def test_peft_rejects_adapting_shared_weight_after_tying():
    config = build_config(["--peft.target_globs", "output"])
    parallel_dims = build_parallel_dims(config, 1)
    model_converters = build_model_converters(config, parallel_dims)
    model = TinyPeftModel()

    model_converters.convert(model)
    model.output.weight = model.tok_embeddings.weight

    with pytest.raises(ValueError, match="shared base weight"):
        model_converters.post_model_init([model])


def test_peft_post_model_init_errors_if_adapter_params_are_still_meta():
    config = build_config(["--peft.target_globs", "proj"])
    parallel_dims = build_parallel_dims(config, 1)
    model_converters = build_model_converters(config, parallel_dims)

    with torch.device("meta"):
        model = TinyPeftModel()

    model_converters.convert(model)

    with pytest.raises(RuntimeError, match="still on the meta device"):
        model_converters.post_model_init([model])


def test_peft_meta_to_empty_then_post_init_materializes_and_initializes_lora():
    config = build_config(
        [
            "--peft.target_globs",
            "proj",
            "--peft.trainable_base_globs",
            "tok_embeddings.weight",
        ]
    )
    parallel_dims = build_parallel_dims(config, 1)
    model_converters = build_model_converters(config, parallel_dims)

    with torch.device("meta"):
        model = TinyPeftModel()

    model_converters.convert(model)
    model.to_empty(device="cpu")
    model_converters.post_model_init([model])

    assert isinstance(model.proj, LoRALinear)
    assert model.proj.lora_a.is_meta is False
    assert model.proj.lora_b.is_meta is False
    torch.testing.assert_close(
        model.proj.lora_b,
        torch.zeros_like(model.proj.lora_b),
        rtol=0,
        atol=0,
    )
    assert torch.isfinite(model.proj.lora_a).all()
