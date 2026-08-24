import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[1]
REGION_PATH = ROOT / "eval/sglang/srt/dllm/region"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if "sglang.srt.dllm.region.execution_spec" in sys.modules:
    EXECUTION = sys.modules["sglang.srt.dllm.region.execution_spec"]
else:
    EXECUTION = load_module(
        "sglang.srt.dllm.region.execution_spec",
        REGION_PATH / "execution_spec.py",
    )
POSITIONS = load_module(
    "cluster3_position_manager", REGION_PATH / "position_manager.py"
)

Contract = EXECUTION.RegionDAGExecutionSpec
Region = EXECUTION.RegionDAGRegion
Status = EXECUTION.RegionStatus
PositionMap = POSITIONS.RegionPositionMap


def region(region_id, start, end, status, parents=()):
    return Region(
        region_id=region_id,
        region_version=0,
        start=start,
        end=end,
        status=status,
        parent_region_ids=tuple(parents),
        recorded_parent_versions=tuple((parent, 0) for parent in parents),
        token_hash=f"tokens:{region_id}",
        position_hash=f"positions:{start}:{end}",
    )


def distant_spec():
    return Contract(
        sequence_length=1024,
        diffusion_steps=2,
        regions=(
            region("S0", 0, 100, Status.STABLE),
            region("A0", 100, 120, Status.ACTIVE, ("S0",)),
            region("S1", 120, 900, Status.STABLE, ("S0",)),
            region("A1", 900, 920, Status.ACTIVE, ("S1",)),
            region("S2", 920, 1024, Status.STABLE, ("S1",)),
        ),
    )


def short_spec():
    return Contract(
        sequence_length=8,
        diffusion_steps=1,
        regions=(
            region("P", 0, 2, Status.STABLE),
            region("X", 2, 4, Status.ACTIVE, ("P",)),
            region("Q", 4, 6, Status.STABLE, ("P",)),
            region("Y", 6, 8, Status.ACTIVE, ("X",)),
        ),
    )


def test_two_distant_active_spans_preserve_absolute_positions():
    position_map = POSITIONS.build_position_map([distant_spec()])
    expected = tuple(range(100, 120)) + tuple(range(900, 920))
    assert position_map.active_positions == (expected,)
    assert position_map.flattened_active_positions == expected
    assert position_map.request_start_offsets == (0, 40)
    positions = POSITIONS.gather_active_positions(position_map)
    preserved = POSITIONS.preserve_original_position_ids(position_map)
    assert positions.dtype is torch.int64
    assert positions.is_contiguous()
    assert torch.equal(positions, torch.tensor(expected, dtype=torch.int64))
    assert torch.equal(preserved, positions)
    assert preserved.tolist()[20] == 900


def test_gather_scatter_round_trip_preserves_every_stable_bit():
    position_map = POSITIONS.build_position_map([short_spec()])
    full = torch.arange(8 * 3, dtype=torch.int64).reshape(1, 8, 3)
    original = full.clone()
    gathered = POSITIONS.gather_active_states(full, position_map)
    expected = torch.cat((full[0, 2:4], full[0, 6:8]), dim=0)
    assert torch.equal(gathered, expected)

    replacement = gathered + 10_000
    scattered = POSITIONS.scatter_active_states(full, replacement, position_map)
    assert torch.equal(full, original)
    assert torch.equal(scattered[0, 2:4], replacement[:2])
    assert torch.equal(scattered[0, 6:8], replacement[2:])
    stable = torch.tensor([0, 1, 4, 5], dtype=torch.int64)
    assert torch.equal(
        scattered[0].index_select(0, stable), original[0].index_select(0, stable)
    )


def test_batched_gather_order_and_request_offsets_are_deterministic():
    first = short_spec()
    second = Contract(
        sequence_length=6,
        diffusion_steps=1,
        regions=(
            region("R", 0, 1, Status.STABLE),
            region("Z", 1, 5, Status.ACTIVE, ("R",)),
            region("T", 5, 6, Status.STABLE, ("R",)),
        ),
    )
    position_map = POSITIONS.build_position_map([first, second])
    assert position_map.request_start_offsets == (0, 4, 8)
    assert position_map.flattened_active_positions == (2, 3, 6, 7, 1, 2, 3, 4)
    states = (
        torch.arange(8).reshape(8, 1),
        (torch.arange(6) + 100).reshape(6, 1),
    )
    gathered = POSITIONS.gather_active_states(states, position_map)
    assert gathered.flatten().tolist() == [2, 3, 6, 7, 101, 102, 103, 104]
    scattered = POSITIONS.scatter_active_states(states, gathered + 1000, position_map)
    assert isinstance(scattered, tuple)
    assert states[0].flatten().tolist() == list(range(8))
    assert states[1].flatten().tolist() == list(range(100, 106))


def test_region_membership_covers_every_position_exactly_once():
    position_map = POSITIONS.build_position_map([short_spec()])
    assert position_map.region_membership == (("P", "P", "X", "X", "Q", "Q", "Y", "Y"),)
    assert set(position_map.active_positions[0]).isdisjoint(
        position_map.stable_positions[0]
    )
    assert set(position_map.active_positions[0]).union(
        position_map.stable_positions[0]
    ) == set(range(8))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"active_positions": ((2, 2, 6, 7),)}, "duplicate active"),
        ({"active_positions": ((-1, 3, 6, 7),)}, "negative active"),
        ({"active_positions": ((2, 3, 6, 8),)}, "out-of-range active"),
        ({"stable_positions": ((0, 1, 2, 4, 5),)}, "overlap"),
        ({"stable_positions": ((0, 1, 4),)}, "missing positions"),
        ({"flattened_active_positions": (0, 1, 2, 3)}, "flattened"),
        ({"request_start_offsets": (0, 3)}, "offsets"),
        ({"region_membership": (("P",) * 7,)}, "membership length"),
    ],
)
def test_malformed_position_maps_fail_closed(change, message):
    valid = POSITIONS.build_position_map([short_spec()])
    with pytest.raises(ValueError, match=message):
        replace(valid, **change)


def test_gather_and_scatter_shape_mismatches_fail_closed():
    position_map = POSITIONS.build_position_map([short_spec()])
    with pytest.raises(ValueError, match="batch size"):
        POSITIONS.gather_active_states(torch.zeros(2, 8, 3), position_map)
    with pytest.raises(ValueError, match="active state rows"):
        POSITIONS.scatter_active_states(
            torch.zeros(1, 8, 3), torch.zeros(3, 3), position_map
        )
    with pytest.raises(ValueError, match="active state rows"):
        POSITIONS.scatter_active_states(
            torch.zeros(1, 8, 3), torch.tensor(1.0), position_map
        )
    with pytest.raises(ValueError, match="feature shape"):
        POSITIONS.scatter_active_states(
            torch.zeros(1, 8, 3), torch.zeros(4, 4), position_map
        )


def test_position_map_requires_at_least_one_request():
    with pytest.raises(ValueError, match="at least one request"):
        POSITIONS.build_position_map([])
