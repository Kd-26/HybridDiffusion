# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch.nn as nn

from torchtitan.components.peft.converter import PeftConverter
from torchtitan.components.quantization.float8 import Float8LinearConverter
from torchtitan.config import ConfigManager
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import (
    _registry_model_converter_cls,
    build_model_converters,
    ModelConvertersContainer,
    register_model_converter,
)


def build_parallel_dims(job_config, world_size):
    parallelism_config = job_config.parallelism
    parallel_dims = ParallelDims(
        dp_shard=parallelism_config.data_parallel_shard_degree,
        dp_replicate=parallelism_config.data_parallel_replicate_degree,
        cp=parallelism_config.context_parallel_degree,
        tp=parallelism_config.tensor_parallel_degree,
        pp=parallelism_config.pipeline_parallel_degree,
        ep=parallelism_config.expert_parallel_degree,
        etp=parallelism_config.expert_tensor_parallel_degree,
        world_size=world_size,
    )
    return parallel_dims


def test_build_model_converters_empty_list():
    config_manager = ConfigManager()
    config = config_manager.parse_args([])
    parallel_dims = build_parallel_dims(config, 1)

    model_converters = build_model_converters(config, parallel_dims)
    assert isinstance(model_converters, ModelConvertersContainer)
    assert model_converters.converters == []


def test_build_model_converters_float8_converter():
    pytest.importorskip("torchao")
    config_manager = ConfigManager()
    config = config_manager.parse_args(
        [
            "--model.converters",
            "quantize.linear.float8",
            "--quantize.linear.float8.emulate",
        ]
    )
    parallel_dims = build_parallel_dims(config, 1)

    model_converters = build_model_converters(config, parallel_dims)
    assert isinstance(model_converters, ModelConvertersContainer)
    assert len(model_converters.converters) == 1
    assert isinstance(model_converters.converters[0], Float8LinearConverter)


def test_build_model_converters_peft_converter():
    config_manager = ConfigManager()
    config = config_manager.parse_args(["--model.converters", "peft"])
    parallel_dims = build_parallel_dims(config, 1)

    model_converters = build_model_converters(config, parallel_dims)
    assert isinstance(model_converters, ModelConvertersContainer)
    assert len(model_converters.converters) == 1
    assert isinstance(model_converters.converters[0], PeftConverter)


def test_model_converters_forward_post_model_init_hook():
    calls = []
    converter_name = "tests.dummy_post_model_init"

    class DummyConverter:
        def __init__(self, job_config, parallel_dims):
            pass

        def convert(self, model):
            calls.append("convert")

        def post_model_init(self, model):
            calls.append("post_model_init")

        def post_optimizer_hook(self, model):
            calls.append("post_optimizer_hook")

    register_model_converter(DummyConverter, converter_name)
    try:
        config_manager = ConfigManager()
        config = config_manager.parse_args(["--model.converters", converter_name])
        parallel_dims = build_parallel_dims(config, 1)

        model_converters = build_model_converters(config, parallel_dims)
        model = nn.Linear(2, 2)
        model_converters.convert(model)
        model_converters.post_model_init(model)

        assert calls == ["convert", "post_model_init"]
    finally:
        _registry_model_converter_cls.pop(converter_name, None)
