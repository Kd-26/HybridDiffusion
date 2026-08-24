"""Deterministic gather/scatter primitives for absolute Region-DAG positions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

import torch

from sglang.srt.dllm.region.execution_spec import RegionDAGExecutionSpec


@dataclass(frozen=True)
class RegionPositionMap:
    """Immutable CPU description of batched absolute Region-DAG positions."""

    sequence_lengths: Tuple[int, ...]
    active_positions: Tuple[Tuple[int, ...], ...]
    flattened_active_positions: Tuple[int, ...]
    request_start_offsets: Tuple[int, ...]
    region_membership: Tuple[Tuple[str, ...], ...]
    stable_positions: Tuple[Tuple[int, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sequence_lengths",
            tuple(int(value) for value in self.sequence_lengths),
        )
        object.__setattr__(
            self,
            "active_positions",
            tuple(
                tuple(int(value) for value in values)
                for values in self.active_positions
            ),
        )
        object.__setattr__(
            self,
            "flattened_active_positions",
            tuple(int(value) for value in self.flattened_active_positions),
        )
        object.__setattr__(
            self,
            "request_start_offsets",
            tuple(int(value) for value in self.request_start_offsets),
        )
        object.__setattr__(
            self,
            "region_membership",
            tuple(
                tuple(str(value) for value in values)
                for values in self.region_membership
            ),
        )
        object.__setattr__(
            self,
            "stable_positions",
            tuple(
                tuple(int(value) for value in values)
                for values in self.stable_positions
            ),
        )
        validate_position_map(self)

    @property
    def batch_size(self) -> int:
        return len(self.sequence_lengths)

    @property
    def active_count(self) -> int:
        return len(self.flattened_active_positions)


def build_position_map(
    specs: Sequence[RegionDAGExecutionSpec],
) -> RegionPositionMap:
    """Build one canonical position map without renumbering absolute positions."""
    specs = tuple(specs)
    if not specs:
        raise ValueError("Region-DAG position map requires at least one request")
    for request_index, spec in enumerate(specs):
        if not isinstance(spec, RegionDAGExecutionSpec):
            raise TypeError(
                f"request {request_index} does not contain a RegionDAGExecutionSpec"
            )
        spec.validate()

    sequence_lengths = []
    active_by_request = []
    stable_by_request = []
    memberships = []
    flattened = []
    offsets = [0]
    for spec in specs:
        active = tuple(
            position
            for region in spec.regions
            if region.is_active
            for position in range(region.start, region.end)
        )
        stable = tuple(
            position
            for region in spec.regions
            if region.is_stable
            for position in range(region.start, region.end)
        )
        membership = [""] * spec.sequence_length
        for region in spec.regions:
            membership[region.start : region.end] = [region.region_id] * (
                region.end - region.start
            )
        sequence_lengths.append(spec.sequence_length)
        active_by_request.append(active)
        stable_by_request.append(stable)
        memberships.append(tuple(membership))
        flattened.extend(active)
        offsets.append(len(flattened))

    return RegionPositionMap(
        sequence_lengths=tuple(sequence_lengths),
        active_positions=tuple(active_by_request),
        flattened_active_positions=tuple(flattened),
        request_start_offsets=tuple(offsets),
        region_membership=tuple(memberships),
        stable_positions=tuple(stable_by_request),
    )


def validate_position_map(position_map: RegionPositionMap) -> None:
    """Fail closed on malformed, renumbered, duplicated, or missing positions."""
    batch_size = len(position_map.sequence_lengths)
    fields = {
        "active_positions": position_map.active_positions,
        "region_membership": position_map.region_membership,
        "stable_positions": position_map.stable_positions,
    }
    for name, values in fields.items():
        if len(values) != batch_size:
            raise ValueError(
                f"position map {name} request count {len(values)} != {batch_size}"
            )
    if len(position_map.request_start_offsets) != batch_size + 1:
        raise ValueError("position map request_start_offsets has the wrong length")
    if (
        not position_map.request_start_offsets
        or position_map.request_start_offsets[0] != 0
    ):
        raise ValueError("position map request_start_offsets must begin at zero")

    expected_flattened = []
    expected_offsets = [0]
    for request_index, sequence_length in enumerate(position_map.sequence_lengths):
        if sequence_length <= 0:
            raise ValueError(
                f"request {request_index} sequence length must be positive"
            )
        active = position_map.active_positions[request_index]
        stable = position_map.stable_positions[request_index]
        membership = position_map.region_membership[request_index]
        for name, positions in (("active", active), ("stable", stable)):
            if len(set(positions)) != len(positions):
                raise ValueError(
                    f"request {request_index} has duplicate {name} positions"
                )
            if tuple(sorted(positions)) != positions:
                raise ValueError(
                    f"request {request_index} {name} positions are not sorted"
                )
            negatives = [position for position in positions if position < 0]
            if negatives:
                raise ValueError(
                    f"request {request_index} has negative {name} positions {negatives[:4]}"
                )
            out_of_range = [
                position for position in positions if position >= sequence_length
            ]
            if out_of_range:
                raise ValueError(
                    f"request {request_index} has out-of-range {name} positions "
                    f"{out_of_range[:4]} for sequence length {sequence_length}"
                )
        overlap = sorted(set(active).intersection(stable))
        if overlap:
            raise ValueError(
                f"request {request_index} stable/active positions overlap at "
                f"{overlap[:4]}"
            )
        covered = set(active).union(stable)
        expected = set(range(sequence_length))
        missing = sorted(expected - covered)
        if missing:
            raise ValueError(
                f"request {request_index} is missing positions {missing[:4]}"
            )
        if covered - expected:
            raise ValueError(f"request {request_index} contains invalid positions")
        if len(membership) != sequence_length:
            raise ValueError(
                f"request {request_index} region membership length "
                f"{len(membership)} != {sequence_length}"
            )
        if any(not region_id for region_id in membership):
            raise ValueError(f"request {request_index} has missing region membership")
        expected_flattened.extend(active)
        expected_offsets.append(len(expected_flattened))

    if tuple(expected_flattened) != position_map.flattened_active_positions:
        raise ValueError(
            "position map flattened active positions do not match request order"
        )
    if tuple(expected_offsets) != position_map.request_start_offsets:
        raise ValueError("position map request start offsets are inconsistent")


def gather_active_positions(
    position_map: RegionPositionMap,
    *,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Materialize deterministic flattened absolute positions as int64 indices."""
    validate_position_map(position_map)
    return torch.tensor(
        position_map.flattened_active_positions,
        dtype=torch.int64,
        device=device,
    ).contiguous()


def preserve_original_position_ids(
    position_map: RegionPositionMap,
    *,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Return original absolute active positions, never zero-based gathered rows."""
    return gather_active_positions(position_map, device=device)


def _state_rows(
    states: Any, position_map: RegionPositionMap, label: str
) -> tuple[list[torch.Tensor], bool]:
    if torch.is_tensor(states):
        if states.ndim < 2:
            raise ValueError(f"{label} tensor must have batch and sequence dimensions")
        if int(states.shape[0]) != position_map.batch_size:
            raise ValueError(
                f"{label} batch size {states.shape[0]} != {position_map.batch_size}"
            )
        rows = [states[request_index] for request_index in range(states.shape[0])]
        is_tensor = True
    else:
        rows = list(states)
        if len(rows) != position_map.batch_size:
            raise ValueError(
                f"{label} request count {len(rows)} != {position_map.batch_size}"
            )
        if any(not torch.is_tensor(value) for value in rows):
            raise TypeError(f"every {label} request value must be a tensor")
        is_tensor = False
    for request_index, (row, sequence_length) in enumerate(
        zip(rows, position_map.sequence_lengths)
    ):
        if row.ndim < 1 or int(row.shape[0]) < sequence_length:
            raise ValueError(
                f"{label} request {request_index} has {row.shape[0]} sequence "
                f"rows, expected at least {sequence_length}"
            )
    return rows, is_tensor


def gather_active_states(states: Any, position_map: RegionPositionMap) -> torch.Tensor:
    """Gather active rows in request order and ascending absolute position order."""
    validate_position_map(position_map)
    rows, _ = _state_rows(states, position_map, "state")
    gathered = []
    for row, positions in zip(rows, position_map.active_positions):
        index = torch.tensor(positions, dtype=torch.int64, device=row.device)
        gathered.append(row.index_select(0, index))
    return torch.cat(gathered, dim=0).contiguous()


def scatter_active_states(
    stable_states: Any,
    active_states: torch.Tensor,
    position_map: RegionPositionMap,
) -> Any:
    """Return a full-state copy with active rows replaced; never mutate the input."""
    validate_position_map(position_map)
    if not torch.is_tensor(active_states):
        raise TypeError("active states must be a tensor")
    active_rows = int(active_states.shape[0]) if active_states.ndim >= 1 else 0
    if active_rows != position_map.active_count:
        raise ValueError(
            f"active state rows {active_rows} != "
            f"position-map active count {position_map.active_count}"
        )
    rows, tensor_input = _state_rows(stable_states, position_map, "stable state")
    result_rows = [row.clone() for row in rows]
    for request_index, (row, positions) in enumerate(
        zip(result_rows, position_map.active_positions)
    ):
        start = position_map.request_start_offsets[request_index]
        end = position_map.request_start_offsets[request_index + 1]
        values = active_states[start:end]
        if row.device != values.device or row.dtype != values.dtype:
            raise ValueError(
                f"request {request_index} active/stable state device or dtype differs"
            )
        if tuple(row.shape[1:]) != tuple(values.shape[1:]):
            raise ValueError(
                f"request {request_index} active/stable state feature shape differs"
            )
        index = torch.tensor(positions, dtype=torch.int64, device=row.device)
        row.index_copy_(0, index, values)
    if tensor_input:
        return torch.stack(result_rows, dim=0)
    return tuple(result_rows)
