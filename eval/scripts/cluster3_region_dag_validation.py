#!/usr/bin/env python3
"""Controlled Region-DAG full-recompute versus conservative-cache validation.

The artifact executes the same loaded Qwen3.5 HybridDiffusion model twice:
one complete Region-DAG recomputation and one production metadata path that
reuses only pre-frontier KV/GDN state and replays every textual row from the
earliest logically invalidated position. Missing evidence fails the case.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import logging
import math
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "cluster3_region_dag_validation"
CONTRACT_ID = "region_dag_conservative_gdn_v1"
NUMERICAL_TOLERANCE = 1e-2
DEFAULT_SEED = 20260825
MAX_COMPARE_CHUNK_ELEMENTS = 1 << 20
ATTENTION_ROW_OBSERVATION_POINT = "qkv_projection_input"
ROOT = Path(__file__).resolve().parents[2]
CLUSTER1_PATH = Path(__file__).with_name("cluster1_exact_handoff_trace.py")
CLUSTER2_PATH = Path(__file__).with_name("cluster2_active_only_validation.py")


@contextlib.contextmanager
def _nvtx_range(torch_module: Any, name: str) -> Iterator[None]:
    """Emit an NVTX range on CUDA builds and remain inert in CPU unit tests."""
    pushed = False
    try:
        torch_module.cuda.nvtx.range_push(f"cluster3::{name}")
        pushed = True
    except RuntimeError:
        pass
    try:
        yield
    finally:
        if pushed:
            torch_module.cuda.nvtx.range_pop()


@dataclass(frozen=True)
class DeferredCudaTiming:
    """A CUDA event pair resolved only after the enclosing request sync."""

    started: Any
    finished: Any

    def milliseconds(self) -> float:
        return float(self.started.elapsed_time(self.finished))

    def __float__(self) -> float:
        return self.milliseconds()

    def __add__(self, other: Any) -> float:
        return self.milliseconds() + float(other)

    def __radd__(self, other: Any) -> float:
        return float(other) + self.milliseconds()


REQUIRED_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "case_id",
        "revision",
        "model_scale",
        "dtype",
        "tp_size",
        "sequence_length",
        "region_contract",
        "edited_regions",
        "expected_invalidation_closure",
        "observed_invalidation_closure",
        "expected_gdn_replay_start",
        "observed_gdn_replay_start",
        "original_active_positions",
        "selected_mask_backend",
        "mask_hash",
        "mask_dimensions",
        "paired_tensor_hashes",
        "max_logits_error",
        "max_hidden_error",
        "max_gdn_error",
        "top1_identical",
        "reused_stable_hash_before",
        "reused_stable_hash_after",
        "negative_lookup_results",
        "stale_state_reuse_count",
        "fallback_count",
        "recovery_replays",
        "nan_or_inf_detected",
        "component_timings_ms",
        "timing_protocol",
        "work_counters",
        "positions_preserved",
        "attention_row_observation_point",
        "peak_memory_bytes",
        "peak_memory_by_path_bytes",
        "case_pass",
        "failure_reasons",
    }
)


@dataclass(frozen=True)
class RegionShape:
    region_id: str
    start: int
    end: int
    status: str
    parents: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationCase:
    case_id: str
    profile: str
    token_seed: int
    sequence_length: int
    regions: tuple[RegionShape, ...]
    edited_regions: tuple[str, ...]
    diffusion_steps: int
    batch_size: int = 1

    @property
    def active_positions(self) -> tuple[int, ...]:
        return tuple(
            position
            for region in self.regions
            if region.status == "active"
            for position in range(region.start, region.end)
        )

    @property
    def layout_identity(self) -> tuple[Any, ...]:
        return (
            self.sequence_length,
            tuple(
                (
                    region.start,
                    region.end,
                    region.status,
                    region.parents,
                )
                for region in self.regions
            ),
            self.edited_regions,
            self.diffusion_steps,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Region-DAG logical invalidation, custom-paged attention, "
            "and conservative ordered GDN replay on one TP=1 GPU."
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument(
        "--profile",
        required=True,
        choices=(
            "one1",
            "smoke16",
            "paper100",
            "effectiveness",
            "efficiency_one",
        ),
    )
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16",))
    parser.add_argument("--tp-size", default=1, type=int)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int)
    parser.add_argument("--max-total-tokens", default=4096, type=int)
    parser.add_argument("--timed-repetitions", default=10, type=int)
    parser.add_argument("--debug-sync-stages", action="store_true")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.tp_size != 1:
        raise ValueError("--tp-size must be 1 until rank-aware evidence exists")
    if args.device < 0:
        raise ValueError("--device must be non-negative")
    if args.max_total_tokens <= 0:
        raise ValueError("--max-total-tokens must be positive")
    if args.timed_repetitions < 10:
        raise ValueError("--timed-repetitions must be at least 10")
    return args


def _one_layout() -> tuple[RegionShape, ...]:
    return (
        RegionShape("A", 0, 64, "stable"),
        RegionShape("B", 64, 96, "active", ("A",)),
        RegionShape("C", 96, 192, "stable", ("A",)),
        RegionShape("D", 192, 224, "active", ("B",)),
        RegionShape("E", 224, 256, "stable", ("A",)),
    )


def _layout(
    sequence_length: int,
    active_count: int,
    active_width: int,
    placement: str,
    *,
    chain: bool,
) -> tuple[RegionShape, ...]:
    if active_count not in (1, 2, 4):
        raise ValueError("active_count must be one, two, or four")
    if placement == "early":
        centers = [
            sequence_length // 8 + i * sequence_length // (active_count + 2)
            for i in range(active_count)
        ]
    elif placement == "middle":
        span = sequence_length // 2
        start = sequence_length // 4
        centers = [
            start + (i + 1) * span // (active_count + 1) for i in range(active_count)
        ]
    elif placement == "late":
        centers = [
            sequence_length // 2
            + (i + 1) * (sequence_length // 2) // (active_count + 1)
            for i in range(active_count)
        ]
    else:
        raise ValueError(f"unsupported placement {placement!r}")
    spans = []
    cursor = 1
    for center in centers:
        start = max(cursor, min(center - active_width // 2, sequence_length - 1))
        end = min(start + active_width, sequence_length - 1)
        if end <= start:
            raise ValueError("active region is empty")
        spans.append((start, end))
        cursor = end + 1

    regions = []
    cursor = 0
    stable_index = 0
    previous_active = None
    root_id = "S0"
    for active_index, (start, end) in enumerate(spans):
        if cursor < start:
            region_id = f"S{stable_index}"
            parents = () if not regions else (root_id,)
            regions.append(RegionShape(region_id, cursor, start, "stable", parents))
            stable_index += 1
        active_id = f"X{active_index}"
        parents = (root_id,)
        if chain and previous_active is not None:
            parents = (previous_active,)
        regions.append(RegionShape(active_id, start, end, "active", parents))
        previous_active = active_id
        cursor = end
    if cursor < sequence_length:
        regions.append(
            RegionShape(
                f"S{stable_index}", cursor, sequence_length, "stable", (root_id,)
            )
        )
    if not regions or regions[0].region_id != root_id or regions[0].status != "stable":
        raise RuntimeError("generated Region-DAG layout lacks a stable root")
    return tuple(regions)


def _case(
    profile: str,
    index: int,
    seed: int,
    sequence_length: int,
    regions: tuple[RegionShape, ...],
    edited_regions: tuple[str, ...],
    steps: int,
) -> ValidationCase:
    return ValidationCase(
        case_id=f"{profile}-{index:03d}-n{sequence_length}-e{'_'.join(edited_regions)}-s{steps}",
        profile=profile,
        token_seed=seed + index * 104729,
        sequence_length=sequence_length,
        regions=regions,
        edited_regions=edited_regions,
        diffusion_steps=steps,
    )


def _smoke_cases(seed: int) -> list[ValidationCase]:
    combinations = (
        (256, 1, "early", 2),
        (256, 2, "middle", 4),
        (256, 4, "late", 8),
        (512, 1, "middle", 8),
        (512, 2, "late", 2),
        (512, 4, "early", 4),
        (1024, 1, "late", 4),
        (1024, 2, "early", 8),
        (1024, 4, "middle", 2),
        (256, 1, "late", 4),
        (256, 4, "middle", 2),
        (512, 2, "early", 8),
        (512, 4, "late", 4),
        (1024, 1, "early", 2),
        (1024, 2, "middle", 8),
    )
    cases = []
    for index, (sequence_length, active_count, placement, steps) in enumerate(
        combinations, 1
    ):
        width = max(4, 64 // active_count)
        regions = _layout(
            sequence_length,
            active_count,
            width,
            placement,
            chain=(index % 2 == 0),
        )
        active_ids = tuple(
            region.region_id for region in regions if region.status == "active"
        )
        cases.append(
            _case(
                "smoke16",
                index,
                seed,
                sequence_length,
                regions,
                (active_ids[0],),
                steps,
            )
        )
    cases.append(
        _case(
            "smoke16",
            16,
            seed,
            256,
            (RegionShape("X0", 0, 256, "active"),),
            ("X0",),
            2,
        )
    )
    return cases


def _paper_cases(seed: int) -> list[ValidationCase]:
    rng = random.Random(seed ^ 0xC3A30)
    candidates = []
    for sequence_length in (256, 512, 1024):
        for active_count in (1, 2, 4):
            for width in (8, 16, 32, 64):
                if active_count * width >= sequence_length // 2:
                    continue
                for placement in ("early", "middle", "late"):
                    for chain in (False, True):
                        regions = _layout(
                            sequence_length,
                            active_count,
                            width,
                            placement,
                            chain=chain,
                        )
                        active_ids = tuple(
                            region.region_id
                            for region in regions
                            if region.status == "active"
                        )
                        edit_count = 1 if len(active_ids) == 1 else rng.randint(1, 2)
                        edited = tuple(sorted(rng.sample(active_ids, edit_count)))
                        steps = rng.choice((2, 4, 8))
                        candidates.append((sequence_length, regions, edited, steps))
    unique_candidates = {}
    for sequence_length, regions, edited, steps in candidates:
        identity = (
            sequence_length,
            tuple(
                (
                    region.start,
                    region.end,
                    region.status,
                    region.parents,
                )
                for region in regions
            ),
            edited,
            steps,
        )
        unique_candidates.setdefault(
            identity, (sequence_length, regions, edited, steps)
        )
    candidates = list(unique_candidates.values())
    rng.shuffle(candidates)
    selected = candidates[:100]
    cases = [
        _case("paper100", index, seed, *values)
        for index, values in enumerate(selected, 1)
    ]
    if len(cases) != 100 or len({case.layout_identity for case in cases}) != 100:
        raise RuntimeError("paper100 manifest is not 100 unique Region-DAG cases")
    return cases


def _effectiveness_cases(seed: int) -> list[ValidationCase]:
    cases = []
    index = 1
    for active_count, width in ((1, 64), (2, 32), (4, 16)):
        for placement in ("early", "middle", "late"):
            regions = _layout(1024, active_count, width, placement, chain=False)
            first_active = next(
                region.region_id for region in regions if region.status == "active"
            )
            cases.append(
                _case(
                    "effectiveness",
                    index,
                    seed,
                    1024,
                    regions,
                    (first_active,),
                    2,
                )
            )
            index += 1
    return cases


def build_manifest(profile: str, seed: int = DEFAULT_SEED) -> list[ValidationCase]:
    if profile == "one1":
        return [_case("one1", 1, seed, 256, _one_layout(), ("B",), 2)]
    if profile == "smoke16":
        return _smoke_cases(seed)
    if profile == "paper100":
        return _paper_cases(seed)
    if profile == "effectiveness":
        return _effectiveness_cases(seed)
    if profile == "efficiency_one":
        return [
            _case(
                "efficiency_one",
                1,
                seed,
                2112,
                (
                    RegionShape("S0", 0, 2048, "stable"),
                    RegionShape("X0", 2048, 2112, "active", ("S0",)),
                ),
                ("X0",),
                4,
            )
        ]
    raise ValueError(f"unsupported profile {profile!r}")


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _token_hash(token_ids: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _apply_reference_top1_at_absolute_positions(
    edited_tokens: list[int],
    diffusion_positions: Iterable[int],
    reference_trace: Mapping[str, Any],
) -> None:
    """Scatter compact query-row predictions back to absolute token positions."""
    query_positions = tuple(int(position) for position in reference_trace["positions"])
    reference_top1 = reference_trace["top1"]
    if len(query_positions) != len(reference_top1):
        raise RuntimeError(
            "reference top-1 rows do not match the traced absolute positions"
        )
    row_by_position = {
        position: row_index for row_index, position in enumerate(query_positions)
    }
    if len(row_by_position) != len(query_positions):
        raise RuntimeError("reference trace contains duplicate absolute positions")
    requested_positions = tuple(int(position) for position in diffusion_positions)
    missing_positions = [
        position for position in requested_positions if position not in row_by_position
    ]
    if missing_positions:
        raise RuntimeError(
            "diffusion positions are absent from the compact reference trace: "
            f"{missing_positions[:8]}"
        )
    for position in requested_positions:
        edited_tokens[position] = int(reference_top1[row_by_position[position]])


def _position_hash(start: int, end: int) -> str:
    digest = hashlib.sha256()
    digest.update(int(start).to_bytes(8, "little", signed=True))
    digest.update(int(end).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def build_execution_spec(
    case: ValidationCase,
    token_ids: Sequence[int],
    *,
    edited: bool,
) -> Any:
    from sglang.srt.dllm.region.dependency_graph import DependencyGraph
    from sglang.srt.dllm.region.execution_spec import (
        RegionDAGExecutionSpec,
        RegionDAGRegion,
        RegionStatus,
    )

    if len(token_ids) != case.sequence_length:
        raise ValueError("case token count differs from sequence_length")
    initial_versions = {region.region_id: 0 for region in case.regions}

    def provisional(versions: Mapping[str, int]) -> Any:
        return RegionDAGExecutionSpec(
            sequence_length=case.sequence_length,
            diffusion_steps=case.diffusion_steps,
            regions=tuple(
                RegionDAGRegion(
                    region_id=region.region_id,
                    region_version=versions[region.region_id],
                    start=region.start,
                    end=region.end,
                    status=RegionStatus(region.status),
                    parent_region_ids=region.parents,
                    recorded_parent_versions=tuple(
                        (parent, versions[parent]) for parent in region.parents
                    ),
                    token_hash=_token_hash(token_ids[region.start : region.end]),
                    position_hash=_position_hash(region.start, region.end),
                )
                for region in case.regions
            ),
        )

    base = provisional(initial_versions)
    if not edited:
        return base
    invalidated = DependencyGraph(base).invalidation_closure(case.edited_regions)
    versions = {
        region_id: int(region_id in invalidated) for region_id in initial_versions
    }
    return provisional(versions)


def expected_plan(spec: Any, edited_regions: Sequence[str]) -> Any:
    from sglang.srt.dllm.region.runtime import build_region_dag_runtime_plan

    return build_region_dag_runtime_plan(spec, edited_regions)


def validate_case_record(record: Mapping[str, Any], case: ValidationCase) -> list[str]:
    missing = sorted(REQUIRED_RECORD_FIELDS - set(record))
    if missing:
        return [f"missing required fields: {missing}"]
    reasons = []
    if record["case_id"] != case.case_id:
        reasons.append("case identity mismatch")
    if record["model_scale"] != "2B":
        reasons.append("validation did not identify HybridDiffusion-2B")
    if record["dtype"] != "bfloat16" or record["tp_size"] != 1:
        reasons.append("validation did not use native BF16 TP=1")
    if record["sequence_length"] != case.sequence_length:
        reasons.append("sequence length mismatch")
    try:
        from sglang.srt.dllm.region.execution_spec import RegionDAGExecutionSpec

        observed_spec = RegionDAGExecutionSpec.from_dict(record["region_contract"])
        observed_spec.validate()
        independently_expected = expected_plan(observed_spec, case.edited_regions)
    except (KeyError, TypeError, ValueError) as exc:
        reasons.append(f"complete region contract is invalid: {exc}")
        independently_expected = None
    if record["selected_mask_backend"] != "custom_paged":
        reasons.append("Region-DAG did not use custom_paged")
    if (
        record["expected_invalidation_closure"]
        != record["observed_invalidation_closure"]
    ):
        reasons.append("logical invalidation closure mismatch")
    if record["expected_gdn_replay_start"] != record["observed_gdn_replay_start"]:
        reasons.append("GDN replay frontier mismatch")
    if independently_expected is not None:
        if (
            list(independently_expected.logical_invalidation_regions)
            != record["expected_invalidation_closure"]
        ):
            reasons.append("expected closure was not independently reproduced")
        if (
            independently_expected.gdn_replay_start
            != record["expected_gdn_replay_start"]
        ):
            reasons.append("expected replay frontier was not independently reproduced")
    active_evidence = record["original_active_positions"]
    active_positions = list(case.active_positions)
    if (
        not isinstance(active_evidence, Mapping)
        or active_evidence.get("count") != len(active_positions)
        or active_evidence.get("sha256") != _sha256_json(active_positions)
    ):
        reasons.append("active-position evidence is missing")
    if not record["mask_hash"] or not record["mask_dimensions"]:
        reasons.append("mask evidence is missing")
    elif any(
        not isinstance(shape, list)
        or len(shape) != 2
        or int(shape[1]) != case.sequence_length
        or int(shape[0]) <= 0
        for shape in record["mask_dimensions"]
    ):
        reasons.append("mask dimensions are invalid")
    if not record["paired_tensor_hashes"]:
        reasons.append("paired tensor hashes are missing")
    elif any(
        not entry.get("reference")
        or not entry.get("cached")
        or entry.get("reused_reference_state") != entry.get("reused_cached_state")
        for entry in record["paired_tensor_hashes"]
    ):
        reasons.append("paired reference/cached tensor evidence is incomplete")
    for field in ("max_logits_error", "max_hidden_error", "max_gdn_error"):
        value = record[field]
        if value is None or not math.isfinite(float(value)):
            reasons.append(f"{field} is unavailable or non-finite")
        elif float(value) >= NUMERICAL_TOLERANCE:
            reasons.append(f"{field} exceeds BF16 tolerance")
    if not record["top1_identical"]:
        reasons.append("top-1 tokens differ")
    if record["reused_stable_hash_before"] != record["reused_stable_hash_after"]:
        reasons.append("reused pre-frontier state changed")
    negatives = record["negative_lookup_results"]
    if not negatives or not all(negatives.values()):
        reasons.append("one or more stale cache identities did not miss")
    if record["stale_state_reuse_count"] != 0:
        reasons.append("stale state was reused")
    if record["fallback_count"] != 0:
        reasons.append("fallback occurred")
    if record["recovery_replays"] != 0:
        reasons.append("unreported recovery replay occurred")
    if record["nan_or_inf_detected"]:
        reasons.append("NaN or Inf detected")
    if not record["positions_preserved"]:
        reasons.append("absolute query positions changed")
    if record["attention_row_observation_point"] != ATTENTION_ROW_OBSERVATION_POINT:
        reasons.append("attention row observation point is invalid")
    if record["peak_memory_bytes"] is None or int(record["peak_memory_bytes"]) <= 0:
        reasons.append("peak-memory evidence is unavailable")
    path_peaks = record["peak_memory_by_path_bytes"]
    if not isinstance(path_peaks, Mapping) or any(
        not path_peaks.get(name) or any(int(value) <= 0 for value in path_peaks[name])
        for name in ("reference_full", "cached")
    ):
        reasons.append("per-path peak-memory evidence is unavailable")
    timing_names = {
        "reference_full_ms",
        "cached_total_ms",
        "canonical_frontier_establishment_ms",
        "warm_cached_suffix_ms",
        "full_attention_ms",
        "gdn_replay_ms",
        "mlp_forward_ms",
        "prefix_snapshot_ms",
        "mask_build_ms",
        "gather_scatter_ms",
        "cache_lookup_restore_ms",
    }
    timings = record["component_timings_ms"]
    if not isinstance(timings, Mapping) or set(timings) != timing_names:
        reasons.append("component timing evidence is unavailable")
    else:
        for name in sorted(timing_names):
            evidence = timings[name]
            samples = evidence.get("samples") if isinstance(evidence, Mapping) else None
            unavailable = (
                evidence.get("unavailable_reason")
                if isinstance(evidence, Mapping)
                else None
            )
            if not samples:
                if not (
                    name == "cache_lookup_restore_ms"
                    and record["expected_gdn_replay_start"] == 0
                    and unavailable
                ):
                    reasons.append(f"{name} has no observed timing samples")
                continue
            if not all(
                math.isfinite(float(value)) and float(value) >= 0.0 for value in samples
            ):
                reasons.append(f"{name} has invalid timing samples")
            if case.profile in ("effectiveness", "efficiency_one"):
                if len(samples) < 10:
                    reasons.append(f"{name} has fewer than ten timed repetitions")
                if not all(
                    key in evidence for key in ("median", "mad", "bootstrap_ci_95")
                ):
                    reasons.append(f"{name} lacks robust timing statistics")
    protocol = record["timing_protocol"]
    if not isinstance(protocol, Mapping) or not protocol.get("cuda_events"):
        reasons.append("CUDA timing protocol evidence is missing")
    elif case.profile in ("effectiveness", "efficiency_one") and (
        int(protocol.get("warmup_repetitions", 0))
        < (3 if case.profile == "efficiency_one" else 1)
        or int(protocol.get("timed_repetitions", 0)) < 10
    ):
        reasons.append("effectiveness timing protocol lacks warm-up/repetitions")
    expected_repetitions = (
        int(protocol.get("timed_repetitions", 0))
        if isinstance(protocol, Mapping)
        else 0
    )
    if (
        expected_repetitions > 0
        and isinstance(path_peaks, Mapping)
        and any(
            len(path_peaks.get(name, ())) != expected_repetitions
            for name in ("reference_full", "cached")
        )
    ):
        reasons.append("per-path peak memory does not cover every repetition")
    work = record["work_counters"]
    work_names = {
        "reference_full_attention_query_token_layer_positions",
        "cached_full_attention_query_token_layer_positions",
        "reference_gdn_token_layer_positions",
        "cached_gdn_replay_token_layer_positions",
        "kv_cache_hits",
        "kv_cache_misses",
        "gdn_state_restores",
    }
    if not isinstance(work, Mapping) or any(
        not isinstance(work.get(name), list) or not work[name] for name in work_names
    ):
        reasons.append("observed work-counter evidence is incomplete")
    elif expected_repetitions > 0 and any(
        len(work[name]) != expected_repetitions for name in work_names
    ):
        reasons.append("work counters do not cover every timed repetition")
    if isinstance(work, Mapping) and all(
        isinstance(work.get(name), list)
        for name in ("kv_cache_hits", "gdn_state_restores")
    ):
        if record["expected_gdn_replay_start"] == 0:
            if any(work["kv_cache_hits"]) or any(work["gdn_state_restores"]):
                reasons.append("entirely-active replay unexpectedly reused cache state")
            cache_timing = (
                timings.get("cache_lookup_restore_ms", {})
                if isinstance(timings, Mapping)
                else {}
            )
            if cache_timing.get("samples") or not cache_timing.get(
                "unavailable_reason"
            ):
                reasons.append(
                    "entirely-active replay did not report restoration as unavailable"
                )
        elif not all(value > 0 for value in work["gdn_state_restores"]):
            reasons.append("nonzero replay frontier lacks an observed GDN restore")
    return reasons


def build_summary(
    cases: Sequence[ValidationCase],
    records: Sequence[Mapping[str, Any]],
    *,
    revision: str,
    hardware: Mapping[str, Any],
    profile: str,
) -> dict[str, Any]:
    expected_ids = [case.case_id for case in cases]
    observed_ids = [str(record.get("case_id")) for record in records]
    one_record = (
        len(observed_ids) == len(expected_ids)
        and len(set(observed_ids)) == len(observed_ids)
        and set(observed_ids) == set(expected_ids)
    )
    passed = sum(bool(record.get("case_pass")) for record in records)
    checks = {
        "one_record_per_case": one_record,
        "all_cases_pass": passed == len(cases),
        "custom_paged_only": bool(records)
        and all(
            record.get("selected_mask_backend") == "custom_paged" for record in records
        ),
        "zero_stale_fallback_recovery": bool(records)
        and all(
            record.get("stale_state_reuse_count") == 0
            and record.get("fallback_count") == 0
            and record.get("recovery_replays") == 0
            for record in records
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "cluster3_revision": revision,
        "hardware": dict(hardware),
        "profile": profile,
        "requested_cases": len(cases),
        "received_records": len(records),
        "passed_cases": passed,
        "checks": checks,
        "strict_pass": all(checks.values()),
        "failed_cases": [
            {
                "case_id": record.get("case_id"),
                "failure_reasons": record.get("failure_reasons", []),
            }
            for record in records
            if not record.get("case_pass")
        ],
    }


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load validation dependency {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _trace_tensor_to_cpu(value: Any) -> Any:
    torch = __import__("torch")
    if not torch.is_tensor(value):
        raise TypeError("trace evidence must be a tensor")
    return value.detach().to(device="cpu", copy=True).contiguous()


def _max_abs(left: Any, right: Any) -> float:
    if left.shape != right.shape or left.device != right.device:
        raise RuntimeError("paired tensor shape/device mismatch")
    if not left.numel():
        raise RuntimeError("paired tensor is empty")
    maximum = 0.0
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    for start in range(0, int(left_flat.numel()), MAX_COMPARE_CHUNK_ELEMENTS):
        stop = min(start + MAX_COMPARE_CHUNK_ELEMENTS, int(left_flat.numel()))
        left_chunk = left_flat[start:stop].float()
        right_chunk = right_flat[start:stop].float()
        if not bool(left_chunk.isfinite().all()) or not bool(
            right_chunk.isfinite().all()
        ):
            raise RuntimeError("paired tensor contains NaN or Inf")
        maximum = max(maximum, float((left_chunk - right_chunk).abs().max()))
    return maximum


class RuntimeLogEvidence(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.fallback_count = 0
        self.recovery_replays = 0

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage().lower()
        self.fallback_count += int("fallback" in message)
        self.recovery_replays += int(
            "recovery_replay" in message or "recovery replay" in message
        )

    @contextlib.contextmanager
    def installed(self) -> Iterator["RuntimeLogEvidence"]:
        root = logging.getLogger()
        root.addHandler(self)
        try:
            yield self
        finally:
            root.removeHandler(self)


def _median_mad(values: Sequence[float]) -> dict[str, float]:
    if len(values) < 10:
        raise ValueError("timing statistics require at least ten repetitions")
    median = float(statistics.median(values))
    mad = float(statistics.median(abs(value - median) for value in values))
    return {"median": median, "mad": mad}


def _bootstrap_median_ci(
    values: Sequence[float],
    *,
    seed: int,
    repetitions: int = 2000,
) -> list[float]:
    """Return a deterministic non-parametric 95% CI for the sample median."""
    if len(values) < 10:
        raise ValueError("bootstrap timing statistics require at least ten samples")
    if repetitions < 100:
        raise ValueError("bootstrap timing statistics require at least 100 resamples")
    samples = [float(value) for value in values]
    if not all(math.isfinite(value) and value >= 0.0 for value in samples):
        raise ValueError("timing samples must be finite and non-negative")
    rng = random.Random(int(seed))
    medians = sorted(
        statistics.median(rng.choices(samples, k=len(samples)))
        for _ in range(repetitions)
    )
    low = medians[int(0.025 * (repetitions - 1))]
    high = medians[int(0.975 * (repetitions - 1))]
    return [float(low), float(high)]


def _timing_evidence(
    values: Sequence[float], *, seed: int, require_statistics: bool
) -> dict[str, Any]:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("timing evidence cannot be empty")
    if not all(math.isfinite(value) and value >= 0.0 for value in samples):
        raise ValueError("timing evidence must be finite and non-negative")
    evidence: dict[str, Any] = {"samples": samples}
    if require_statistics:
        evidence.update(_median_mad(samples))
        evidence["bootstrap_ci_95"] = _bootstrap_median_ci(samples, seed=seed)
    return evidence


class ScopedBackendTimers:
    """Validator-only CUDA event probes removed on every exit path.

    Qwen's full-attention helper is not an ``nn.Module``. Timing the concrete
    FlashInfer and GDN backend entry points observes the complete kernel paths
    without changing production code or relying on Python-method row hooks.
    """

    def __init__(self, runtime: "Cluster3ValidationRuntime") -> None:
        self.torch = __import__("torch")
        attention_backend = runtime.model_runner.attn_backend
        self.full_backend = getattr(
            attention_backend, "full_attn_backend", attention_backend
        )
        self.gdn_backend = runtime.backend
        self._targets: list[tuple[Any, str, bool, Any]] = []
        self._events: dict[str, list[tuple[Any, Any]]] = {
            "full_attention": [],
            "gdn_replay": [],
            "cache_lookup_restore": [],
            "prefix_snapshot": [],
            "mlp": [],
        }
        try:
            self._install(
                self.full_backend,
                "forward_extend",
                "full_attention",
            )
            self._install(self.gdn_backend, "forward_extend", "gdn_replay")
            self._install(
                self.gdn_backend,
                "_restore_region_dag_layer_snapshot",
                "cache_lookup_restore",
            )
            self._install(
                self.gdn_backend,
                "_put_region_dag_layer_snapshot",
                "prefix_snapshot",
            )
            model = runtime.cluster1.ModelTraceHooks._language_model(
                runtime.model_runner.model
            )
            for layer in model.layers:
                self._install(layer.mlp, "forward", "mlp")
        except BaseException:
            self.close()
            raise

    def _install(self, target: Any, name: str, kind: str) -> None:
        original = getattr(target, name, None)
        if not callable(original):
            raise RuntimeError(f"Region-DAG timer cannot observe {name}")
        namespace = getattr(target, "__dict__", {})
        had_instance_value = name in namespace
        instance_value = namespace.get(name)

        def measured(*args: Any, **kwargs: Any) -> Any:
            started = self.torch.cuda.Event(enable_timing=True)
            finished = self.torch.cuda.Event(enable_timing=True)
            nvtx_name = {
                "full_attention": "attention_forward",
                "gdn_replay": "gdn_forward",
                "cache_lookup_restore": "gdn_restore",
                "prefix_snapshot": "prefix_snapshot",
                "mlp": "mlp_forward",
            }[kind]
            with _nvtx_range(self.torch, nvtx_name):
                started.record()
                result = original(*args, **kwargs)
                finished.record()
            self._events[kind].append((started, finished))
            return result

        setattr(target, name, measured)
        self._targets.append((target, name, had_instance_value, instance_value))

    def snapshot(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for kind, pairs in self._events.items():
            values[kind] = (
                sum(float(start.elapsed_time(end)) for start, end in pairs)
                if pairs
                else None
            )
            values[f"{kind}_calls"] = len(pairs)
        return values

    def close(self) -> None:
        for target, name, had_instance_value, instance_value in reversed(self._targets):
            if had_instance_value:
                setattr(target, name, instance_value)
            else:
                delattr(target, name)
        self._targets.clear()
        self._events.clear()

    @property
    def released(self) -> bool:
        return not self._targets

    def __enter__(self) -> "ScopedBackendTimers":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


class Cluster3ValidationRuntime:
    """Real-model paired execution using production Region-DAG metadata."""

    def __init__(self, args: argparse.Namespace):
        torch = __import__("torch")
        if not torch.cuda.is_available():
            raise RuntimeError("Cluster-3 dynamic validation requires CUDA")
        torch.cuda.set_device(args.device)
        self.args = args
        self.cluster1 = _load_module("cluster3_cluster1_runtime", CLUSTER1_PATH)
        self.cluster2 = _load_module("cluster3_cluster2_runtime", CLUSTER2_PATH)

        class Runtime(self.cluster1.Cluster1ModelRuntime):
            @staticmethod
            def _server_args_kwargs(
                runtime_args: Any, config_path: Path
            ) -> dict[str, Any]:
                values = Runtime.__mro__[1]._server_args_kwargs(
                    runtime_args, config_path
                )
                values.update(
                    max_running_requests=1,
                    max_total_tokens=runtime_args.max_total_tokens,
                )
                return values

        runtime_args = SimpleNamespace(**vars(args))
        runtime_args.model_dir = args.model_path
        self.runtime = Runtime(runtime_args)
        self.model_runner = self.runtime.model_runner
        self.backend = self.runtime.backend
        self.device = self.runtime.device
        self.revision = _git_revision()
        self.model_scale = self.cluster2.infer_model_scale(self.model_runner)

    def _clear(self) -> None:
        self.runtime._clear_pools()
        self.backend._region_dag_layer_snapshots.clear()

    def _tokens(self, case: ValidationCase) -> list[int]:
        torch = __import__("torch")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(case.token_seed)
        vocab = int(self.model_runner.model_config.vocab_size)
        mask_id = int(self.runtime.dllm_config.mask_id)
        values = torch.randint(
            0,
            vocab,
            (case.sequence_length,),
            generator=generator,
            dtype=torch.int64,
        ).tolist()
        return [0 if value == mask_id else int(value) for value in values]

    def _attach_request(
        self,
        req: Any,
        spec: Any,
        plan: Any,
        *,
        mode: str,
        initialized: bool,
    ) -> None:
        from sglang.srt.dllm.region.runtime import RegionDAGInstrumentation

        req.hybrid_execution_spec = None
        req.region_dag_execution_spec = spec
        req.region_dag_runtime_plan = plan
        req.region_dag_instrumentation = RegionDAGInstrumentation.from_plan(spec, plan)
        req.region_dag_mode = mode
        req.region_dag_allow_full_replay = False
        req.region_dag_initialized = initialized
        req.region_dag_restore_required = initialized and plan.gdn_replay_start > 0
        req.region_dag_model_identity = str(Path(self.args.model_path).resolve())
        req.region_dag_model_revision = str(
            getattr(self.model_runner.server_args, "revision", "") or "local-checkpoint"
        )
        req.region_dag_adapter_revision = ""
        req.region_dag_frontier_keys = {}

    def _bind_frontiers(self, req: Any) -> None:
        from sglang.srt.dllm.region.runtime import build_region_dag_frontier_key

        if req.req_pool_idx is None:
            raise RuntimeError("controlled Region-DAG request has no real pool slot")
        spec = req.region_dag_execution_spec
        boundaries = sorted(
            {0, spec.sequence_length, *(region.start for region in spec.regions)}
        )
        req.region_dag_frontier_keys = {
            boundary: build_region_dag_frontier_key(
                spec=spec,
                boundary=boundary,
                request_id=str(req.rid),
                request_pool_idx=int(req.req_pool_idx),
                request_slot_generation=int(req.hybrid_request_slot_generation),
                model_identity=req.region_dag_model_identity,
                model_revision=req.region_dag_model_revision,
                adapter_identity="",
                adapter_revision="",
            )
            for boundary in boundaries
        }

    def _new_batch(self, req: Any) -> Any:
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        tree_cache = SimpleNamespace(
            page_size=self.model_runner.server_args.page_size,
            device=self.device,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            supports_swa=lambda: False,
            supports_mamba=lambda: False,
            is_chunk_cache=lambda: False,
            is_tree_cache=lambda: True,
            evict=lambda _params: None,
        )
        return ScheduleBatch.init_new(
            reqs=[req],
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            dllm_config=self.runtime.dllm_config,
        )

    def _cuda_timed(
        self, operation: Callable[[], Any], *, nvtx_phase: str
    ) -> tuple[Any, DeferredCudaTiming]:
        torch = __import__("torch")
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        with _nvtx_range(torch, nvtx_phase):
            started.record()
            result = operation()
            finished.record()
        return result, DeferredCudaTiming(started, finished)

    def _prepare_reference(
        self,
        rid: str,
        token_ids: list[int],
        spec: Any,
        case: ValidationCase,
    ) -> tuple[Any, Any, dict[str, float]]:
        torch = __import__("torch")
        from sglang.srt.dllm.region.runtime import build_region_dag_runtime_plan
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        req = self.runtime._make_req(rid, token_ids)
        plan = build_region_dag_runtime_plan(
            spec, case.edited_regions, force_full_replay=True
        )
        self._attach_request(req, spec, plan, mode="reference", initialized=False)
        req.prefix_indices = torch.empty(0, dtype=torch.int64, device=self.device)
        req.fill_ids = list(token_ids)
        req.set_extend_input_len(len(token_ids))
        batch = self._new_batch(req)
        _, gather_scatter_ms = self._cuda_timed(
            batch.prepare_for_extend, nvtx_phase="request_setup"
        )
        self._bind_frontiers(req)
        worker_batch = batch.get_model_worker_batch()
        forward_batch, mask_build_ms = self._cuda_timed(
            lambda: ForwardBatch.init_new(worker_batch, self.model_runner),
            nvtx_phase="region_mask_build",
        )
        return (
            req,
            forward_batch,
            {
                "gather_scatter": gather_scatter_ms,
                "mask_build": mask_build_ms,
            },
        )

    def _prepare_canonical_frontier(
        self,
        rid: str,
        token_ids: list[int],
        spec: Any,
        case: ValidationCase,
    ) -> tuple[Any, Any, dict[str, float]]:
        """Schedule only the stable topological prefix that owns the frontier."""
        torch = __import__("torch")
        from sglang.srt.dllm.region.runtime import (
            build_canonical_frontier_execution_spec,
            build_region_dag_runtime_plan,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        plan = build_region_dag_runtime_plan(spec, case.edited_regions)
        boundary = int(plan.gdn_replay_start)
        frontier_spec = build_canonical_frontier_execution_spec(spec, boundary)
        req = self.runtime._make_req(rid, token_ids)
        self._attach_request(
            req, spec, plan, mode="canonical_frontier", initialized=False
        )
        req.region_dag_frontier_establishing = True
        req.prefix_indices = torch.empty(0, dtype=torch.int64, device=self.device)
        req.fill_ids = list(token_ids[:boundary])
        req.set_extend_input_len(boundary)
        batch = self._new_batch(req)
        _, gather_scatter_ms = self._cuda_timed(
            batch.prepare_for_extend, nvtx_phase="request_setup"
        )
        self._bind_frontiers(req)
        batch.region_dag_execution_specs_cpu = [frontier_spec]
        batch.region_dag_query_positions_cpu = [tuple(range(boundary))]
        batch.region_dag_frontier_keys_cpu = [
            {
                position: key
                for position, key in req.region_dag_frontier_keys.items()
                if position <= boundary
            }
        ]
        batch.region_dag_restore_required_cpu = [False]
        batch.region_dag_reference_cpu = [True]
        worker_batch = batch.get_model_worker_batch()
        forward_batch, mask_build_ms = self._cuda_timed(
            lambda: ForwardBatch.init_new(worker_batch, self.model_runner),
            nvtx_phase="region_mask_build",
        )
        if tuple(forward_batch.region_dag_query_positions_cpu[0]) != tuple(
            range(boundary)
        ):
            raise RuntimeError("canonical frontier changed original prefix positions")
        if int(forward_batch.input_ids.numel()) != boundary:
            raise RuntimeError("canonical frontier did not schedule exactly its prefix")
        return (
            req,
            forward_batch,
            {
                "gather_scatter": gather_scatter_ms,
                "mask_build": mask_build_ms,
            },
        )

    def _prepare_frontier_suffix(
        self,
        req: Any,
        token_ids: list[int],
        spec: Any,
        case: ValidationCase,
        *,
        restore: bool,
    ) -> tuple[Any, dict[str, float]]:
        """Attach the full suffix to a canonical live or committed frontier."""
        from sglang.srt.dllm.region.runtime import build_region_dag_runtime_plan
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        plan = build_region_dag_runtime_plan(spec, case.edited_regions)
        boundary = int(plan.gdn_replay_start)
        req.prefix_indices = self.runtime._canonical_prefix_locations(
            req.req_pool_idx, boundary
        )
        req.origin_input_ids = list(token_ids)
        req.fill_ids = list(token_ids)
        req.set_extend_input_len(len(token_ids) - boundary)
        self._attach_request(
            req,
            spec,
            plan,
            mode="canonical_cached" if restore else "canonical_segmented",
            initialized=True,
        )
        req.region_dag_frontier_established = True
        req.region_dag_restore_required = bool(restore)
        self._bind_frontiers(req)
        batch = self._new_batch(req)
        _, gather_scatter_ms = self._cuda_timed(
            batch.prepare_for_extend,
            nvtx_phase="kv_restore" if restore else "request_setup",
        )
        batch.region_dag_query_positions_cpu = [plan.attention_query_positions]
        batch.region_dag_restore_required_cpu = [bool(restore)]
        batch.region_dag_reference_cpu = [False]
        worker_batch = batch.get_model_worker_batch()
        forward_batch, mask_build_ms = self._cuda_timed(
            lambda: ForwardBatch.init_new(worker_batch, self.model_runner),
            nvtx_phase="region_mask_build",
        )
        if not restore:
            forward_batch.region_dag_diagnostic_live_prefix_cpu = [True]
        if bool(forward_batch.region_dag_restore_required_cpu[0]) != bool(restore):
            raise RuntimeError("canonical suffix restore contract was not preserved")
        if int(forward_batch.input_ids.numel()) != plan.query_count:
            raise RuntimeError("canonical suffix did not schedule exact replay rows")
        return forward_batch, {
            "gather_scatter": gather_scatter_ms,
            "mask_build": mask_build_ms,
        }

    def _prepare_cached(
        self,
        req: Any,
        token_ids: list[int],
        spec: Any,
        case: ValidationCase,
    ) -> tuple[Any, dict[str, float]]:
        from sglang.srt.dllm.region.runtime import build_region_dag_runtime_plan
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        plan = build_region_dag_runtime_plan(spec, case.edited_regions)
        req.origin_input_ids = list(token_ids)
        req.fill_ids = list(token_ids)
        self._attach_request(req, spec, plan, mode="cached", initialized=True)
        self._bind_frontiers(req)
        batch = self._new_batch(req)
        _, gather_scatter_ms = self._cuda_timed(
            batch.prepare_for_region_dag_replay, nvtx_phase="kv_restore"
        )
        worker_batch = batch.get_model_worker_batch()
        forward_batch, mask_build_ms = self._cuda_timed(
            lambda: ForwardBatch.init_new(worker_batch, self.model_runner),
            nvtx_phase="region_mask_build",
        )
        return forward_batch, {
            "gather_scatter": gather_scatter_ms,
            "mask_build": mask_build_ms,
        }

    def _run_forward(
        self, forward_batch: Any, hooks: Any
    ) -> tuple[dict[str, Any], float, dict[str, Any]]:
        torch = __import__("torch")
        with _nvtx_range(torch, "flashinfer_plan_build"):
            self.model_runner.attn_backend.init_forward_metadata(forward_batch)
        mamba_slots = [
            self.backend._current_mamba_slot(int(value))
            for value in forward_batch.req_pool_indices.detach().cpu().tolist()
        ]
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        with ScopedBackendTimers(self) as timers:
            with _nvtx_range(torch, "model_forward_total"):
                started.record()
                with hooks.capture(mamba_slots):
                    logits = self.runtime._forward(
                        forward_batch, metadata_prepared=True
                    )
            finished.record()
            self.runtime._synchronize()
            model_forward_ms = float(started.elapsed_time(finished))
            backend_timings = timers.snapshot()
        if not timers.released:
            raise RuntimeError("temporary backend timers survived model execution")
        trace = hooks.snapshot()
        trace.update(
            logits=_trace_tensor_to_cpu(logits),
            top1=_trace_tensor_to_cpu(logits.argmax(dim=-1)),
            positions=tuple(
                int(value) for value in forward_batch.positions.cpu().tolist()
            ),
            selected_mask_backend=forward_batch.dllm_selected_mask_backend,
            mask=_trace_tensor_to_cpu(forward_batch.region_dag_custom_mask),
        )
        if not trace["hidden"] or not trace["gdn_states"]:
            raise RuntimeError("model hooks produced incomplete layer evidence")
        tensors = [trace["logits"], *trace["hidden"].values()]
        tensors.extend(
            value for states in trace["gdn_states"].values() for value in states
        )
        if any(not bool(value.isfinite().all()) for value in tensors):
            raise RuntimeError("model execution produced NaN or Inf evidence")
        return trace, model_forward_ms, backend_timings

    def _kv_hash(self, req: Any, positions: Sequence[int]) -> str:
        torch = __import__("torch")
        locations = self.model_runner.req_to_token_pool.req_to_token[
            int(req.req_pool_idx), list(positions)
        ].to(dtype=torch.int64)
        named = []
        text_config = self.model_runner.model_config.hf_text_config
        full_layers = set(getattr(text_config, "full_attention_layer_ids", []) or [])
        if not full_layers:
            full_layers = {
                index
                for index, kind in enumerate(text_config.layers_block_type)
                if kind == "attention"
            }
        for layer_id in sorted(full_layers):
            key, value = self.model_runner.token_to_kv_pool.get_kv_buffer(layer_id)
            named.extend(
                (
                    (f"kv.{layer_id}.key", key.index_select(0, locations)),
                    (f"kv.{layer_id}.value", value.index_select(0, locations)),
                )
            )
        return self.cluster1.hash_tensors(named)

    def _frontier_hash(self, req: Any, boundary: int) -> str:
        named = []
        key = req.region_dag_frontier_keys[boundary]
        for layer_id in self.backend.gdn_layer_ids:
            snapshot = self.backend._region_dag_layer_snapshots.get((key, layer_id))
            if snapshot is None:
                raise RuntimeError(
                    f"missing GDN snapshot at boundary={boundary}, layer={layer_id}"
                )
            named.extend(
                (
                    (f"gdn.{layer_id}.conv", snapshot.conv_state),
                    (f"gdn.{layer_id}.recurrent", snapshot.recurrent_state),
                )
            )
        return self.cluster1.hash_tensors(named)

    def _reused_hash(self, req: Any, replay_start: int) -> str:
        digest = hashlib.sha256()
        digest.update(self._kv_hash(req, range(replay_start)).encode())
        digest.update(self._frontier_hash(req, replay_start).encode())
        return digest.hexdigest()

    def _negative_lookups(self, req: Any, replay_start: int) -> dict[str, bool]:
        key = req.region_dag_frontier_keys[replay_start]
        layer_id = self.backend.gdn_layer_ids[0]
        cache = self.backend._region_dag_layer_snapshots
        preceding = list(key.preceding_regions)
        if preceding:
            region_id, version, parents, token_hash, position_hash = preceding[-1]
            wrong_parents = tuple(parents) + (("wrong-parent", 999),)
            preceding[-1] = (
                region_id,
                version,
                wrong_parents,
                token_hash,
                position_hash,
            )
        else:
            preceding.append(
                ("__negative__", 999, (), "negative-token", "negative-position")
            )

        def rejected_or_missed(**changes: Any) -> bool:
            try:
                mutated = replace(key, **changes)
            except (TypeError, ValueError):
                return True
            return (mutated, layer_id) not in cache

        return {
            "wrong_model_miss": rejected_or_missed(
                model_identity=key.model_identity + ":wrong"
            ),
            "wrong_model_revision_miss": rejected_or_missed(
                model_revision=key.model_revision + ":wrong"
            ),
            "wrong_adapter_miss": rejected_or_missed(
                adapter_identity=key.adapter_identity + ":wrong"
            ),
            "wrong_adapter_revision_miss": rejected_or_missed(
                adapter_revision=key.adapter_revision + ":wrong"
            ),
            "wrong_contract_miss": rejected_or_missed(
                attention_contract_id=key.attention_contract_id + ":wrong"
            ),
            "wrong_request_generation_miss": rejected_or_missed(
                request_slot_generation=key.request_slot_generation + 1
            ),
            "wrong_parent_version_miss": rejected_or_missed(
                preceding_regions=tuple(preceding)
            ),
        }

    def _slice_reference(self, trace: Mapping[str, Any], start: int) -> dict[str, Any]:
        return {
            **trace,
            "logits": trace["logits"][start:],
            "top1": trace["top1"][start:],
            "hidden": {
                layer: value[start:] for layer, value in trace["hidden"].items()
            },
        }

    @staticmethod
    def _validate_trace_rows(
        trace: Mapping[str, Any], *, expected_rows: int, label: str
    ) -> None:
        for kind in ("attention", "gdn", "mlp"):
            observed = trace["rows"].get(kind)
            if not observed:
                raise RuntimeError(f"{label} has no observed {kind} rows")
            if any(int(value) != expected_rows for value in observed):
                raise RuntimeError(
                    f"{label} {kind} rows differ from scheduled query rows: "
                    f"observed={observed}, expected={expected_rows}"
                )

    def _compare(
        self, reference: Mapping[str, Any], cached: Mapping[str, Any]
    ) -> tuple[float, float, float, bool]:
        if set(reference["hidden"]) != set(cached["hidden"]):
            raise RuntimeError("paired executions captured different model layers")
        if set(reference["gdn_states"]) != set(cached["gdn_states"]):
            raise RuntimeError("paired executions captured different GDN layers")
        for layer in reference["gdn_states"]:
            if len(reference["gdn_states"][layer]) != len(cached["gdn_states"][layer]):
                raise RuntimeError(
                    f"paired GDN layer {layer} captured different state counts"
                )
        logits = _max_abs(reference["logits"], cached["logits"])
        hidden = max(
            _max_abs(reference["hidden"][layer], cached["hidden"][layer])
            for layer in reference["hidden"]
        )
        gdn_errors = [
            _max_abs(left, right)
            for layer in reference["gdn_states"]
            for left, right in zip(
                reference["gdn_states"][layer], cached["gdn_states"][layer]
            )
        ]
        if not gdn_errors:
            raise RuntimeError("GDN comparison evidence is empty")
        top1 = bool((reference["top1"] == cached["top1"]).all())
        return logits, hidden, max(gdn_errors), top1

    def _trace_hashes(self, trace: Mapping[str, Any]) -> dict[str, str]:
        final_layer = max(trace["hidden"])
        return {
            "logits": self.cluster1.hash_tensors((("logits", trace["logits"]),)),
            "hidden": self.cluster1.hash_tensors(
                (f"hidden.{layer}", value)
                for layer, value in sorted(trace["hidden"].items())
            ),
            "gdn": self.cluster1.hash_tensors(
                (f"gdn.{layer}.{index}", value)
                for layer, values in sorted(trace["gdn_states"].items())
                for index, value in enumerate(values)
            ),
            "final_hidden": self.cluster1.hash_tensors(
                (("final_hidden", trace["hidden"][final_layer]),)
            ),
            "top1": self.cluster1.hash_token_ids(trace["top1"]),
        }

    def run_case(self, case: ValidationCase) -> dict[str, Any]:
        torch = __import__("torch")
        if case.sequence_length > self.args.max_total_tokens:
            raise ValueError("case exceeds --max-total-tokens")
        original_tokens = self._tokens(case)
        initial_plan = expected_plan(
            build_execution_spec(case, original_tokens, edited=True),
            case.edited_regions,
        )
        invalidated_ids = set(initial_plan.logical_invalidation_regions)
        diffusion_positions = tuple(
            position
            for region in case.regions
            if region.region_id in invalidated_ids
            for position in range(region.start, region.end)
        )
        if not diffusion_positions:
            raise RuntimeError("Region-DAG edit produced no diffusion positions")
        performance_profile = case.profile in ("effectiveness", "efficiency_one")
        measurement_repetitions = (
            self.args.timed_repetitions if performance_profile else 1
        )
        warmup_repetitions = (
            3 if case.profile == "efficiency_one" else 1 if performance_profile else 0
        )
        paired_hashes: list[dict[str, Any]] = []
        comparisons: list[tuple[float, float, float, bool]] = []
        selected_backends: list[str] = []
        mask_hashes: list[str] = []
        mask_dimensions: list[list[int]] = []
        reused_before: list[str] = []
        reused_after: list[str] = []
        negative_results: list[dict[str, bool]] = []
        cached_row_evidence: list[dict[str, list[int]]] = []
        observed_positions: list[tuple[int, ...]] = []
        observed_closures: list[tuple[str, ...]] = []
        observed_replay_starts: list[int] = []
        reference_peak_samples: list[int] = []
        cached_peak_samples: list[int] = []
        work_samples: dict[str, list[int]] = {
            "reference_full_attention_query_token_layer_positions": [],
            "cached_full_attention_query_token_layer_positions": [],
            "reference_gdn_token_layer_positions": [],
            "cached_gdn_replay_token_layer_positions": [],
            "kv_cache_hits": [],
            "kv_cache_misses": [],
            "gdn_state_restores": [],
        }
        timing_samples: dict[str, list[float]] = {
            "reference_full_ms": [],
            "cached_total_ms": [],
            "canonical_frontier_establishment_ms": [],
            "warm_cached_suffix_ms": [],
            "full_attention_ms": [],
            "gdn_replay_ms": [],
            "mlp_forward_ms": [],
            "prefix_snapshot_ms": [],
            "mask_build_ms": [],
            "gather_scatter_ms": [],
            "cache_lookup_restore_ms": [],
        }
        cache_restore_unavailable = False
        peak = 0
        started = time.perf_counter()
        with (
            RuntimeLogEvidence().installed() as logs,
            self.cluster2.ScopedRowHooks(self.model_runner) as hooks,
        ):
            for repetition in range(-warmup_repetitions, measurement_repetitions):
                collect = repetition >= 0
                edited_tokens = list(original_tokens)
                for position in diffusion_positions:
                    edited_tokens[position] = int(self.runtime.dllm_config.mask_id)
                repetition_timings = {name: 0.0 for name in timing_samples}
                repetition_work = {name: 0 for name in work_samples}
                repetition_has_restore = False
                repetition_reference_peak = 0
                repetition_cached_peak = 0

                for step in range(case.diffusion_steps):
                    edited_spec = build_execution_spec(case, edited_tokens, edited=True)
                    plan = expected_plan(edited_spec, case.edited_regions)
                    replay_start = plan.gdn_replay_start

                    self._clear()
                    torch.cuda.reset_peak_memory_stats(self.device)
                    if replay_start == 0:
                        reference_req, reference_batch, reference_prepare = (
                            self._prepare_reference(
                                f"{case.case_id}:reference:{repetition}:{step}",
                                edited_tokens,
                                edited_spec,
                                case,
                            )
                        )
                        reference_trace, reference_forward_ms, _ = self._run_forward(
                            reference_batch, hooks
                        )
                        reference_frontier_trace = None
                    else:
                        (
                            reference_req,
                            reference_frontier_batch,
                            reference_frontier_prepare,
                        ) = self._prepare_canonical_frontier(
                            f"{case.case_id}:reference:{repetition}:{step}",
                            edited_tokens,
                            edited_spec,
                            case,
                        )
                        (
                            reference_frontier_trace,
                            reference_frontier_ms,
                            _,
                        ) = self._run_forward(reference_frontier_batch, hooks)
                        self.backend._region_dag_layer_snapshots.clear()
                        reference_batch, reference_suffix_prepare = (
                            self._prepare_frontier_suffix(
                                reference_req,
                                edited_tokens,
                                edited_spec,
                                case,
                                restore=False,
                            )
                        )
                        reference_trace, reference_suffix_ms, _ = self._run_forward(
                            reference_batch, hooks
                        )
                        reference_prepare = {
                            "frontier": sum(reference_frontier_prepare.values()),
                            "suffix": sum(reference_suffix_prepare.values()),
                        }
                        reference_forward_ms = (
                            reference_frontier_ms + reference_suffix_ms
                        )
                    reference_reused_hash = self._reused_hash(
                        reference_req, replay_start
                    )
                    self._validate_trace_rows(
                        reference_trace,
                        expected_rows=plan.query_count,
                        label="canonical segmented Region-DAG reference",
                    )
                    if reference_trace["positions"] != tuple(
                        plan.attention_query_positions
                    ):
                        raise RuntimeError(
                            "Region-DAG reference changed absolute positions"
                        )
                    repetition_reference_peak = max(
                        repetition_reference_peak,
                        int(torch.cuda.max_memory_allocated(self.device)),
                    )

                    self._clear()
                    base_spec = build_execution_spec(
                        case, original_tokens, edited=False
                    )
                    prepare_baseline = (
                        self._prepare_reference
                        if replay_start == 0
                        else self._prepare_canonical_frontier
                    )
                    baseline_req, baseline_batch, baseline_prepare = prepare_baseline(
                        f"{case.case_id}:cached:{repetition}:{step}",
                        original_tokens,
                        base_spec,
                        case,
                    )
                    baseline_trace, baseline_forward_ms, _ = self._run_forward(
                        baseline_batch, hooks
                    )
                    self._validate_trace_rows(
                        baseline_trace,
                        expected_rows=(
                            case.sequence_length if replay_start == 0 else replay_start
                        ),
                        label="canonical Region-DAG frontier establishment",
                    )
                    before = self._reused_hash(baseline_req, replay_start)
                    if reference_reused_hash != before:
                        raise RuntimeError(
                            "reused pre-frontier KV/GDN state disagrees with "
                            "complete Region-DAG recomputation"
                        )
                    torch.cuda.reset_peak_memory_stats(self.device)
                    if replay_start == 0:
                        cached_batch, cached_prepare = self._prepare_cached(
                            baseline_req, edited_tokens, edited_spec, case
                        )
                    else:
                        cached_batch, cached_prepare = self._prepare_frontier_suffix(
                            baseline_req,
                            edited_tokens,
                            edited_spec,
                            case,
                            restore=True,
                        )
                    (
                        cached_trace,
                        cached_forward_ms,
                        cached_backend_ms,
                    ) = self._run_forward(cached_batch, hooks)
                    self._validate_trace_rows(
                        cached_trace,
                        expected_rows=plan.query_count,
                        label="Region-DAG cached replay",
                    )
                    if cached_trace["positions"] != tuple(
                        plan.attention_query_positions
                    ):
                        raise RuntimeError(
                            "Region-DAG cached replay changed absolute positions"
                        )
                    after = self._reused_hash(baseline_req, replay_start)
                    repetition_cached_peak = max(
                        repetition_cached_peak,
                        int(torch.cuda.max_memory_allocated(self.device)),
                    )

                    comparison = self._compare(reference_trace, cached_trace)
                    if collect:
                        comparisons.append(comparison)
                        paired_hashes.append(
                            {
                                "repetition": repetition + 1,
                                "step": step + 1,
                                "reference": self._trace_hashes(reference_trace),
                                "cached": self._trace_hashes(cached_trace),
                                "reused_reference_state": reference_reused_hash,
                                "reused_cached_state": before,
                            }
                        )
                        selected_backends.extend(
                            (
                                reference_trace["selected_mask_backend"],
                                cached_trace["selected_mask_backend"],
                            )
                        )
                        mask_hashes.append(
                            self.cluster1.hash_tensors(
                                (("mask", cached_trace["mask"]),)
                            )
                        )
                        mask_dimensions.append([plan.query_count, case.sequence_length])
                        reused_before.append(before)
                        reused_after.append(after)
                        negative_results.append(
                            self._negative_lookups(baseline_req, replay_start)
                        )
                        cached_row_evidence.append(
                            {
                                name: [int(value) for value in values]
                                for name, values in cached_trace["rows"].items()
                            }
                        )
                        observed_positions.append(cached_trace["positions"])

                    repetition_timings["reference_full_ms"] += (
                        sum(reference_prepare.values()) + reference_forward_ms
                    )
                    frontier_establishment_ms = (
                        sum(baseline_prepare.values()) + baseline_forward_ms
                        if replay_start > 0
                        else 0.0
                    )
                    warm_cached_suffix_ms = (
                        cached_prepare["gather_scatter"]
                        + cached_prepare["mask_build"]
                        + cached_forward_ms
                    )
                    repetition_timings[
                        "canonical_frontier_establishment_ms"
                    ] += frontier_establishment_ms
                    repetition_timings["warm_cached_suffix_ms"] += warm_cached_suffix_ms
                    repetition_timings["cached_total_ms"] += warm_cached_suffix_ms
                    for target, source in (
                        ("full_attention_ms", "full_attention"),
                        ("gdn_replay_ms", "gdn_replay"),
                        ("mlp_forward_ms", "mlp"),
                        ("prefix_snapshot_ms", "prefix_snapshot"),
                    ):
                        value = cached_backend_ms[source]
                        if value is None:
                            raise RuntimeError(
                                f"cached execution produced no {source} timing"
                            )
                        repetition_timings[target] += float(value)
                    repetition_timings["mask_build_ms"] += cached_prepare["mask_build"]
                    repetition_timings["gather_scatter_ms"] += cached_prepare[
                        "gather_scatter"
                    ]
                    restore_ms = cached_backend_ms["cache_lookup_restore"]
                    restore_calls = int(cached_backend_ms["cache_lookup_restore_calls"])
                    if replay_start > 0:
                        if restore_ms is None or restore_calls <= 0:
                            raise RuntimeError(
                                "cached replay did not observe the required exact "
                                "GDN frontier restoration"
                            )
                        repetition_timings["cache_lookup_restore_ms"] += float(
                            restore_ms
                        )
                        repetition_has_restore = True
                    elif restore_ms is not None or restore_calls != 0:
                        raise RuntimeError(
                            "full Region-DAG replay unexpectedly restored GDN state"
                        )

                    repetition_work[
                        "reference_full_attention_query_token_layer_positions"
                    ] += sum(reference_trace["rows"]["attention"])
                    if reference_frontier_trace is not None:
                        repetition_work[
                            "reference_full_attention_query_token_layer_positions"
                        ] += sum(reference_frontier_trace["rows"]["attention"])
                    repetition_work[
                        "cached_full_attention_query_token_layer_positions"
                    ] += sum(cached_trace["rows"]["attention"])
                    repetition_work["reference_gdn_token_layer_positions"] += sum(
                        reference_trace["rows"]["gdn"]
                    )
                    if reference_frontier_trace is not None:
                        repetition_work["reference_gdn_token_layer_positions"] += sum(
                            reference_frontier_trace["rows"]["gdn"]
                        )
                    repetition_work["cached_gdn_replay_token_layer_positions"] += sum(
                        cached_trace["rows"]["gdn"]
                    )
                    metrics = baseline_req.region_dag_instrumentation
                    if metrics is None:
                        raise RuntimeError(
                            "cached request lost Region-DAG instrumentation"
                        )
                    if collect:
                        observed_closures.append(
                            tuple(metrics.logical_invalidation_regions)
                        )
                        observed_replay_starts.append(int(metrics.gdn_replay_start))
                    repetition_work["kv_cache_hits"] += int(metrics.kv_cache_hits)
                    repetition_work["kv_cache_misses"] += len(
                        plan.actually_recomputed_positions
                    )
                    repetition_work["gdn_state_restores"] += restore_calls

                    _apply_reference_top1_at_absolute_positions(
                        edited_tokens,
                        diffusion_positions,
                        reference_trace,
                    )

                if collect:
                    for name, value in repetition_timings.items():
                        if (
                            name == "cache_lookup_restore_ms"
                            and not repetition_has_restore
                        ):
                            cache_restore_unavailable = True
                            continue
                        timing_samples[name].append(value)
                    for name, value in repetition_work.items():
                        work_samples[name].append(value)
                    reference_peak_samples.append(repetition_reference_peak)
                    cached_peak_samples.append(repetition_cached_peak)
                    peak = max(
                        peak,
                        repetition_reference_peak,
                        repetition_cached_peak,
                    )
        if not hooks.released:
            raise RuntimeError("temporary row hooks survived Cluster-3 validation")

        if not comparisons or not negative_results or not cached_row_evidence:
            raise RuntimeError("Cluster-3 validation produced no measured evidence")
        final_spec = build_execution_spec(case, edited_tokens, edited=True)
        final_plan = expected_plan(final_spec, case.edited_regions)
        active_positions = list(case.active_positions)
        negatives = {
            name: all(result[name] for result in negative_results)
            for name in negative_results[0]
        }
        observed_closure = (
            list(observed_closures[0])
            if observed_closures
            and all(value == observed_closures[0] for value in observed_closures)
            else ["__inconsistent__"]
        )
        observed_replay_start = (
            observed_replay_starts[0]
            if observed_replay_starts
            and all(
                value == observed_replay_starts[0] for value in observed_replay_starts
            )
            else -1
        )
        require_statistics = performance_profile
        component_timings = {
            name: _timing_evidence(
                values,
                seed=case.token_seed ^ int(_sha256_json(name)[:16], 16),
                require_statistics=require_statistics,
            )
            for name, values in timing_samples.items()
            if values
        }
        if cache_restore_unavailable:
            component_timings["cache_lookup_restore_ms"] = {
                "samples": [],
                "unavailable_reason": (
                    "earliest invalidation is zero, so the entirely-active "
                    "reduction performs no cache lookup or GDN restore"
                ),
            }
        record = {
            "schema_version": SCHEMA_VERSION,
            "case_id": case.case_id,
            "profile": case.profile,
            "revision": self.revision,
            "model_scale": self.model_scale,
            "dtype": self.args.dtype,
            "tp_size": self.args.tp_size,
            "sequence_length": case.sequence_length,
            "diffusion_steps": case.diffusion_steps,
            "region_contract": final_spec.to_dict(),
            "edited_regions": list(case.edited_regions),
            "expected_invalidation_closure": list(
                final_plan.logical_invalidation_regions
            ),
            "observed_invalidation_closure": observed_closure,
            "expected_gdn_replay_start": final_plan.gdn_replay_start,
            "observed_gdn_replay_start": observed_replay_start,
            "gdn_replay_positions": list(final_plan.gdn_replay_positions),
            "replayed_but_logically_valid_regions": list(
                final_plan.replayed_but_logically_valid_regions
            ),
            "original_active_positions": {
                "count": len(active_positions),
                "sha256": _sha256_json(active_positions),
                "preview": active_positions[:16],
            },
            "selected_mask_backend": (
                "custom_paged"
                if selected_backends
                and all(value == "custom_paged" for value in selected_backends)
                else "inconsistent"
            ),
            "mask_hash": _sha256_json(mask_hashes),
            "mask_dimensions": mask_dimensions,
            "paired_tensor_hashes": paired_hashes,
            "max_logits_error": max(value[0] for value in comparisons),
            "max_hidden_error": max(value[1] for value in comparisons),
            "max_gdn_error": max(value[2] for value in comparisons),
            "top1_identical": all(value[3] for value in comparisons),
            "reused_stable_hash_before": _sha256_json(reused_before),
            "reused_stable_hash_after": _sha256_json(reused_after),
            "negative_lookup_results": negatives,
            "stale_state_reuse_count": sum(
                not missed for result in negative_results for missed in result.values()
            ),
            "fallback_count": logs.fallback_count,
            "recovery_replays": logs.recovery_replays,
            "nan_or_inf_detected": False,
            "component_timings_ms": component_timings,
            "timing_protocol": {
                "warmup_repetitions": warmup_repetitions,
                "timed_repetitions": measurement_repetitions,
                "cuda_events": True,
                "synchronized_measurement_boundaries": True,
                "cuda_synchronizations_per_measured_forward": 1,
                "debug_sync_stages": bool(self.args.debug_sync_stages),
                "timing_scope": "request_exclusive",
                "host_paired_total_ms": (time.perf_counter() - started) * 1000.0,
            },
            "work_counters": {
                **work_samples,
                "invalidated_ranges": [
                    list(value) for value in final_plan.invalidated_ranges
                ],
                "reused_regions": list(final_plan.reused_regions),
                "replayed_but_logically_valid_regions": list(
                    final_plan.replayed_but_logically_valid_regions
                ),
            },
            "positions_preserved": all(
                positions == tuple(final_plan.attention_query_positions)
                for positions in observed_positions
            ),
            "attention_row_observation_point": (ATTENTION_ROW_OBSERVATION_POINT),
            "attention_query_rows_per_layer": cached_row_evidence[0]["attention"],
            "gdn_rows_per_layer": cached_row_evidence[0]["gdn"],
            "mlp_rows_per_layer": cached_row_evidence[0]["mlp"],
            "peak_memory_bytes": peak,
            "peak_memory_by_path_bytes": {
                "reference_full": reference_peak_samples,
                "cached": cached_peak_samples,
            },
            "case_pass": False,
            "failure_reasons": [],
        }
        reasons = validate_case_record(record, case)
        record["failure_reasons"] = reasons
        record["case_pass"] = not reasons
        return record

    def hardware(self) -> dict[str, Any]:
        torch = __import__("torch")
        properties = torch.cuda.get_device_properties(self.args.device)
        return {
            "device_index": self.args.device,
            "device_name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "cuda_version": torch.version.cuda,
        }

    def close(self) -> None:
        self.runtime.close()


def run_validation(
    args: argparse.Namespace,
    *,
    runtime_factory: Callable[[argparse.Namespace], Any] = Cluster3ValidationRuntime,
) -> dict[str, Any]:
    cases = build_manifest(args.profile, args.seed)
    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    runtime = runtime_factory(args)
    records = []
    hardware: dict[str, Any] = {"unavailable_reason": "runtime did not report"}
    primary_error: Optional[BaseException] = None
    try:
        with output_path.open("w", encoding="utf-8") as destination:
            for case in cases:
                try:
                    record = runtime.run_case(case)
                    missing = sorted(REQUIRED_RECORD_FIELDS - set(record))
                    if missing:
                        raise RuntimeError(
                            f"case {case.case_id} returned incomplete evidence: {missing}"
                        )
                    strict_reasons = validate_case_record(record, case)
                    if strict_reasons:
                        record = dict(record)
                        record["failure_reasons"] = strict_reasons
                        record["case_pass"] = False
                except BaseException as exc:
                    primary_error = exc
                    break
                records.append(record)
                destination.write(json.dumps(record, sort_keys=True) + "\n")
                destination.flush()
    finally:
        try:
            hardware = runtime.hardware()
        except BaseException as exc:
            if primary_error is None:
                primary_error = exc
        try:
            runtime.close()
        except BaseException as exc:
            if primary_error is None:
                primary_error = exc
    summary = build_summary(
        cases,
        records,
        revision=_git_revision(),
        hardware=hardware,
        profile=args.profile,
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if primary_error is not None:
        raise primary_error
    if not summary["strict_pass"]:
        raise RuntimeError("Cluster-3 validation failed; inspect the summary JSON")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_validation(parse_args(argv))


if __name__ == "__main__":
    main()
