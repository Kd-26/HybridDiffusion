"""Production metadata contract for conservative Region-DAG execution.

This module intentionally contains no CUDA work. It turns one immutable
Region-DAG contract and an explicit edit set into the exact logical and
recurrent work sets that the scheduler and model executor must preserve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Tuple

from sglang.srt.dllm.region.dependency_graph import DependencyGraph
from sglang.srt.dllm.region.execution_spec import (
    REGION_DAG_CONSERVATIVE_GDN_V1,
    RegionDAGExecutionSpec,
)


def _ordered_unique_region_ids(
    spec: RegionDAGExecutionSpec, values: Iterable[str]
) -> Tuple[str, ...]:
    selected = tuple(str(value) for value in values)
    if not selected:
        raise ValueError("Region-DAG edited_regions must be nonempty")
    if len(set(selected)) != len(selected):
        raise ValueError("Region-DAG edited_regions must not contain duplicates")
    known = set(spec.region_ids)
    missing = sorted(set(selected) - known)
    if missing:
        raise ValueError(f"Region-DAG edited_regions are unknown: {missing}")
    selected_set = set(selected)
    return tuple(
        region.region_id for region in spec.regions if region.region_id in selected_set
    )


@dataclass(frozen=True)
class RegionDAGRuntimePlan:
    """Immutable distinction between graph invalidation and recurrent replay."""

    selected_contract: str
    edited_regions: Tuple[str, ...]
    logical_invalidation_regions: Tuple[str, ...]
    gdn_replay_start: int
    gdn_replay_positions: Tuple[int, ...]
    attention_query_positions: Tuple[int, ...]
    actually_recomputed_positions: Tuple[int, ...]
    reused_regions: Tuple[str, ...]
    reused_positions: Tuple[int, ...]
    replayed_but_logically_valid_regions: Tuple[str, ...]
    invalidated_ranges: Tuple[Tuple[int, int], ...]
    full_replay: bool = False

    def __post_init__(self) -> None:
        tuple_fields = (
            "edited_regions",
            "logical_invalidation_regions",
            "gdn_replay_positions",
            "attention_query_positions",
            "actually_recomputed_positions",
            "reused_regions",
            "reused_positions",
            "replayed_but_logically_valid_regions",
            "invalidated_ranges",
        )
        for field_name in tuple_fields:
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))
        if self.selected_contract != REGION_DAG_CONSERVATIVE_GDN_V1:
            raise ValueError(
                f"unsupported Region-DAG runtime contract {self.selected_contract!r}"
            )
        replay = self.gdn_replay_positions
        if not replay or replay[0] != self.gdn_replay_start:
            raise ValueError("Region-DAG replay positions must begin at replay_start")
        if replay != tuple(range(replay[0], replay[-1] + 1)):
            raise ValueError("Region-DAG GDN replay must be contiguous and ordered")
        if self.attention_query_positions != replay:
            raise ValueError(
                "conservative Region-DAG attention queries must equal GDN replay rows"
            )
        if self.actually_recomputed_positions != replay:
            raise ValueError(
                "conservative Region-DAG recomputation must cover every replay row"
            )
        if set(self.reused_positions).intersection(replay):
            raise ValueError("Region-DAG reused and replayed positions overlap")

    @property
    def query_count(self) -> int:
        return len(self.attention_query_positions)

    @property
    def gdn_replay_tokens(self) -> int:
        return len(self.gdn_replay_positions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_contract": self.selected_contract,
            "edited_regions": list(self.edited_regions),
            "logical_invalidation_regions": list(self.logical_invalidation_regions),
            "gdn_replay_start": self.gdn_replay_start,
            "gdn_replay_positions": list(self.gdn_replay_positions),
            "attention_query_positions": list(self.attention_query_positions),
            "actually_recomputed_positions": list(self.actually_recomputed_positions),
            "reused_regions": list(self.reused_regions),
            "reused_positions": list(self.reused_positions),
            "replayed_but_logically_valid_regions": list(
                self.replayed_but_logically_valid_regions
            ),
            "invalidated_ranges": [list(value) for value in self.invalidated_ranges],
            "full_replay": self.full_replay,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegionDAGRuntimePlan":
        return cls(
            selected_contract=str(value["selected_contract"]),
            edited_regions=tuple(value["edited_regions"]),
            logical_invalidation_regions=tuple(value["logical_invalidation_regions"]),
            gdn_replay_start=int(value["gdn_replay_start"]),
            gdn_replay_positions=tuple(
                int(position) for position in value["gdn_replay_positions"]
            ),
            attention_query_positions=tuple(
                int(position) for position in value["attention_query_positions"]
            ),
            actually_recomputed_positions=tuple(
                int(position) for position in value["actually_recomputed_positions"]
            ),
            reused_regions=tuple(value["reused_regions"]),
            reused_positions=tuple(
                int(position) for position in value["reused_positions"]
            ),
            replayed_but_logically_valid_regions=tuple(
                value["replayed_but_logically_valid_regions"]
            ),
            invalidated_ranges=tuple(
                (int(start), int(end)) for start, end in value["invalidated_ranges"]
            ),
            full_replay=bool(value.get("full_replay", False)),
        )


def build_region_dag_runtime_plan(
    spec: RegionDAGExecutionSpec,
    edited_regions: Iterable[str],
    *,
    force_full_replay: bool = False,
) -> RegionDAGRuntimePlan:
    """Build the exact work plan for one edit without estimating runtime work."""
    if not isinstance(spec, RegionDAGExecutionSpec):
        raise TypeError("Region-DAG runtime requires RegionDAGExecutionSpec")
    spec.validate()
    edited = _ordered_unique_region_ids(spec, edited_regions)
    graph = DependencyGraph(spec)
    logical = graph.invalidation_closure(edited)
    logical_set = set(logical)
    replay_start = (
        0 if force_full_replay else graph.earliest_invalidated_position(edited)
    )
    replay_positions = tuple(range(replay_start, spec.sequence_length))
    reused_positions = tuple(range(replay_start))
    reused_regions = tuple(
        region.region_id
        for region in spec.regions
        if region.end <= replay_start and region.region_id not in logical_set
    )
    replayed_valid = tuple(
        region.region_id
        for region in spec.regions
        if region.end > replay_start and region.region_id not in logical_set
    )
    invalidated_ranges = tuple(
        (region.start, region.end)
        for region in spec.regions
        if region.region_id in logical_set
    )
    return RegionDAGRuntimePlan(
        selected_contract=spec.attention_contract_id,
        edited_regions=edited,
        logical_invalidation_regions=logical,
        gdn_replay_start=replay_start,
        gdn_replay_positions=replay_positions,
        attention_query_positions=replay_positions,
        actually_recomputed_positions=replay_positions,
        reused_regions=reused_regions,
        reused_positions=reused_positions,
        replayed_but_logically_valid_regions=replayed_valid,
        invalidated_ranges=invalidated_ranges,
        full_replay=bool(force_full_replay),
    )


@dataclass(frozen=True)
class RegionDAGFrontierKey:
    """Complete identity of a layer-local GDN state before one position."""

    request_id: str
    request_pool_idx: int
    request_slot_generation: int
    boundary: int
    model_identity: str
    model_revision: str
    adapter_identity: str
    adapter_revision: str
    attention_contract_id: str
    preceding_regions: Tuple[
        Tuple[str, int, Tuple[Tuple[str, int], ...], str, str], ...
    ]

    def __post_init__(self) -> None:
        preceding_regions = tuple(
            (
                str(region_id),
                int(version),
                tuple((str(parent), int(value)) for parent, value in parents),
                str(token_hash),
                str(position_hash),
            )
            for region_id, version, parents, token_hash, position_hash in self.preceding_regions
        )
        object.__setattr__(self, "preceding_regions", preceding_regions)
        if not self.request_id:
            raise ValueError("Region-DAG frontier request_id must be nonempty")
        if self.request_pool_idx < 0 or self.request_slot_generation < 0:
            raise ValueError("Region-DAG frontier request slot is invalid")
        if self.boundary < 0:
            raise ValueError("Region-DAG frontier boundary must be nonnegative")
        if self.attention_contract_id != REGION_DAG_CONSERVATIVE_GDN_V1:
            raise ValueError("Region-DAG frontier attention contract is invalid")
        for region_id, version, parents, token_hash, position_hash in preceding_regions:
            if not region_id or version < 0 or not token_hash or not position_hash:
                raise ValueError("Region-DAG frontier region identity is invalid")
            if any(parent_version < 0 for _, parent_version in parents):
                raise ValueError("Region-DAG frontier parent version is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "request_pool_idx": self.request_pool_idx,
            "request_slot_generation": self.request_slot_generation,
            "boundary": self.boundary,
            "model_identity": self.model_identity,
            "model_revision": self.model_revision,
            "adapter_identity": self.adapter_identity,
            "adapter_revision": self.adapter_revision,
            "attention_contract_id": self.attention_contract_id,
            "preceding_regions": [
                [
                    region_id,
                    version,
                    [list(parent) for parent in parents],
                    token_hash,
                    position_hash,
                ]
                for region_id, version, parents, token_hash, position_hash in self.preceding_regions
            ],
        }


def build_region_dag_frontier_key(
    *,
    spec: RegionDAGExecutionSpec,
    boundary: int,
    request_id: str,
    request_pool_idx: int,
    request_slot_generation: int,
    model_identity: str,
    model_revision: str,
    adapter_identity: str,
    adapter_revision: str,
) -> RegionDAGFrontierKey:
    """Bind a frontier to every region that can influence its GDN state."""
    spec.validate()
    boundary = int(boundary)
    if boundary < 0 or boundary > spec.sequence_length:
        raise ValueError(
            f"Region-DAG frontier {boundary} is outside [0, {spec.sequence_length}]"
        )
    valid_boundaries = {0, spec.sequence_length, *(r.start for r in spec.regions)}
    if boundary not in valid_boundaries:
        raise ValueError(
            f"Region-DAG frontier {boundary} must be an exact region boundary"
        )
    preceding = tuple(
        (
            region.region_id,
            region.region_version,
            tuple(region.recorded_parent_versions),
            region.token_hash,
            region.position_hash,
        )
        for region in spec.regions
        if region.end <= boundary
    )
    return RegionDAGFrontierKey(
        request_id=str(request_id),
        request_pool_idx=int(request_pool_idx),
        request_slot_generation=int(request_slot_generation),
        boundary=boundary,
        model_identity=str(model_identity),
        model_revision=str(model_revision),
        adapter_identity=str(adapter_identity),
        adapter_revision=str(adapter_revision),
        attention_contract_id=spec.attention_contract_id,
        preceding_regions=preceding,
    )


@dataclass
class RegionDAGInstrumentation:
    """Observed counters/timings; unavailable evidence remains ``None``."""

    selected_contract: str
    region_count: int
    stable_regions: Tuple[str, ...]
    active_regions: Tuple[str, ...]
    edited_regions: Tuple[str, ...]
    logical_invalidation_regions: Tuple[str, ...]
    gdn_replay_start: int
    gdn_replay_tokens: int
    reused_regions: Tuple[str, ...]
    replayed_but_logically_valid_regions: Tuple[str, ...]
    attention_query_token_layer_positions: int = 0
    gdn_replayed_token_layer_positions: int = 0
    kv_cache_hits: int = 0
    kv_cache_misses: int = 0
    gdn_state_restores: int = 0
    invalidated_ranges: Tuple[Tuple[int, int], ...] = ()
    fallback_count: int = 0
    recovery_replays: int = 0
    mask_build_time: Optional[float] = None
    gather_scatter_time: Optional[float] = None
    attention_time: Optional[float] = None
    gdn_time: Optional[float] = None
    model_forward_time: Optional[float] = None
    total_latency: Optional[float] = None
    peak_memory: Optional[int] = None

    @classmethod
    def from_plan(
        cls, spec: RegionDAGExecutionSpec, plan: RegionDAGRuntimePlan
    ) -> "RegionDAGInstrumentation":
        return cls(
            selected_contract=plan.selected_contract,
            region_count=len(spec.regions),
            stable_regions=tuple(region.region_id for region in spec.stable_regions),
            active_regions=tuple(region.region_id for region in spec.active_regions),
            edited_regions=plan.edited_regions,
            logical_invalidation_regions=plan.logical_invalidation_regions,
            gdn_replay_start=plan.gdn_replay_start,
            gdn_replay_tokens=plan.gdn_replay_tokens,
            reused_regions=plan.reused_regions,
            replayed_but_logically_valid_regions=(
                plan.replayed_but_logically_valid_regions
            ),
            invalidated_ranges=plan.invalidated_ranges,
        )
