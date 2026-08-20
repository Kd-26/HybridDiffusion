import importlib.util
import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest


PATH = Path(__file__).parents[1] / "eval/sglang/srt/dllm/region/execution_spec.py"
SPEC = importlib.util.spec_from_file_location("cluster1_execution_spec", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

HybridExecutionSpec = MODULE.HybridExecutionSpec
PositionInterval = MODULE.PositionInterval


def test_valid_spec_serialization_and_immutable_advance():
    original = HybridExecutionSpec.prefix_diffusion(
        ar_boundary=64, sequence_length=80, diffusion_steps=4
    )
    restored = HybridExecutionSpec.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original
    advanced = original.advance_boundary(68, sequence_length=84)
    assert original.ar_boundary == 64
    assert advanced.ar_boundary == 68
    assert advanced.region_versions == (1, 1)
    assert advanced.stable_positions == (PositionInterval(0, 68),)
    assert advanced.active_positions == (PositionInterval(68, 84),)
    with pytest.raises(FrozenInstanceError):
        original.ar_boundary = 65


@pytest.mark.parametrize(
    "change",
    [
        {"region_ids": ("same", "same")},
        {"region_versions": (0, -1)},
        {"parent_regions": ((), ())},
        {"stable_positions": (PositionInterval(1, 64),)},
        {"active_positions": (PositionInterval(65, 80),)},
        {"active_positions": (PositionInterval(64, 64),)},
        {"diffusion_steps": 0},
        {"attention_contract_id": "unknown"},
    ],
)
def test_invalid_cluster1_specs_are_rejected(change):
    values = HybridExecutionSpec.prefix_diffusion(
        ar_boundary=64, sequence_length=80, diffusion_steps=4
    ).to_dict()
    values.update(change)
    with pytest.raises(ValueError):
        HybridExecutionSpec.from_dict(values)


def test_multispan_execution_is_rejected():
    base = HybridExecutionSpec.prefix_diffusion(
        ar_boundary=64, sequence_length=80, diffusion_steps=2
    ).to_dict()
    base["active_positions"] = [
        {"start": 64, "end": 72},
        {"start": 76, "end": 80},
    ]
    with pytest.raises(ValueError, match="one stable span"):
        HybridExecutionSpec.from_dict(base)
