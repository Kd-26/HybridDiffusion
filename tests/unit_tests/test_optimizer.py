# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch.nn as nn

from torchtitan.components.optimizer import build_optimizers
from torchtitan.config import Optimizer as OptimizerConfig
from torchtitan.config import OptimizerModuleLROverride
from torchtitan.distributed.parallel_dims import ParallelDims


class ToyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wq = nn.Linear(4, 4, bias=False)
        self.wo = nn.Linear(4, 4, bias=False)


class ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = ToyAttention()
        self.feed_forward = nn.Linear(4, 4, bias=False)
        self.ffn_norm = nn.LayerNorm(4)


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleDict(
            {
                "0": ToyBlock(),
                "1": ToyBlock(),
            }
        )
        self.norm = nn.LayerNorm(4)


class NoAttentionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feed_forward = nn.Linear(4, 4, bias=False)
        self.norm = nn.LayerNorm(4)


def _parallel_dims(pp: int = 1, world_size: int = 1) -> ParallelDims:
    return ParallelDims(
        dp_replicate=1,
        dp_shard=1,
        cp=1,
        tp=1,
        pp=pp,
        ep=1,
        etp=1,
        world_size=world_size,
    )


def test_build_optimizers_with_module_lr_overrides():
    model = ToyModel()
    config = OptimizerConfig(
        lr=1e-4,
        implementation="for-loop",
        module_lr_overrides=[
            OptimizerModuleLROverride(
                name="attention",
                target_globs=["layers.*.attention.*"],
                lr_multiplier=3.0,
            )
        ],
    )

    optimizers = build_optimizers([model], config, _parallel_dims())
    optimizer = optimizers.optimizers[0]

    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [3e-4, 1e-4]
    )

    param_name_by_id = {id(param): name for name, param in model.named_parameters()}
    attention_param_names = {
        name
        for name in param_name_by_id.values()
        if name.startswith("layers.") and ".attention." in name
    }
    grouped_attention_names = {
        param_name_by_id[id(param)] for param in optimizer.param_groups[0]["params"]
    }
    grouped_default_names = {
        param_name_by_id[id(param)] for param in optimizer.param_groups[1]["params"]
    }

    assert grouped_attention_names == attention_param_names
    assert grouped_attention_names.isdisjoint(grouped_default_names)
    assert grouped_attention_names | grouped_default_names == set(
        param_name_by_id.values()
    )


def test_build_optimizers_rejects_overlapping_module_lr_overrides():
    model = ToyModel()
    config = OptimizerConfig(
        lr=1e-4,
        implementation="for-loop",
        module_lr_overrides=[
            OptimizerModuleLROverride(
                name="attention",
                target_globs=["layers.*.attention.*"],
                lr_multiplier=3.0,
            ),
            OptimizerModuleLROverride(
                name="layer0",
                target_globs=["layers.0.*"],
                lr_multiplier=2.0,
            ),
        ],
    )

    with pytest.raises(ValueError, match="matched multiple optimizer.module_lr_overrides"):
        build_optimizers([model], config, _parallel_dims())


def test_build_optimizers_rejects_pipeline_group_layout_mismatch():
    config = OptimizerConfig(
        lr=1e-4,
        implementation="for-loop",
        module_lr_overrides=[
            OptimizerModuleLROverride(
                name="attention",
                target_globs=["layers.*.attention.*"],
                lr_multiplier=3.0,
            )
        ],
    )

    with pytest.raises(ValueError, match="same non-empty param-group layout"):
        build_optimizers(
            [ToyModel(), NoAttentionModel()],
            config,
            _parallel_dims(pp=2, world_size=2),
        )
