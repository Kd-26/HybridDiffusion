#!/usr/bin/env python3
"""Versioned manifest normalization for the Cluster-3 production benchmark.

This module is deliberately CPU-only.  It converts external JSON into the
validated Region-DAG case representation before a checkpoint or CUDA runtime
is loaded.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA_VERSION = 1
SUPPORTED_BATCH_SIZES = (1,)


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if result != value:
        raise ValueError(f"{label} must be an integer")
    if positive and result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _normalize_spans(value: Any, total_tokens: int) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError("active_spans must be a nonempty array")
    spans = []
    for index, span in enumerate(value):
        if (
            not isinstance(span, Sequence)
            or isinstance(span, (str, bytes))
            or len(span) != 2
        ):
            raise ValueError(f"active_spans[{index}] must be [start, end]")
        start = _integer(span[0], f"active_spans[{index}].start")
        end = _integer(span[1], f"active_spans[{index}].end")
        if start < 0 or end <= start or end > total_tokens:
            raise ValueError(
                f"active_spans[{index}] is outside [0, {total_tokens}) or reversed"
            )
        spans.append((start, end))
    spans.sort()
    for previous, current in zip(spans, spans[1:]):
        if current[0] < previous[1]:
            raise ValueError("active_spans must not overlap")
    return tuple(spans)


def _simple_suffix_regions(
    validation_module: Any,
    *,
    total_tokens: int,
    prefix_tokens: int,
    active_spans: tuple[tuple[int, int], ...],
) -> tuple[Any, ...]:
    if prefix_tokens <= 0 or prefix_tokens >= total_tokens:
        raise ValueError("prefix_tokens must be inside the sequence")
    expected = ((prefix_tokens, total_tokens),)
    if active_spans != expected:
        raise ValueError(
            "prefix_tokens shorthand requires one active suffix [prefix_tokens, total)"
        )
    return (
        validation_module.RegionShape("S0", 0, prefix_tokens, "stable"),
        validation_module.RegionShape(
            "X0", prefix_tokens, total_tokens, "active", ("S0",)
        ),
    )


def _explicit_regions(
    validation_module: Any,
    value: Any,
    *,
    total_tokens: int,
    active_spans: tuple[tuple[int, int], ...],
) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError("arbitrary active spans require a nonempty regions array")
    regions = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"regions[{index}] must be an object")
        region_id = str(item.get("region_id", ""))
        if not region_id:
            raise ValueError(f"regions[{index}].region_id must be nonempty")
        start = _integer(item.get("start"), f"regions[{index}].start")
        end = _integer(item.get("end"), f"regions[{index}].end")
        stable = item.get("stable")
        if not isinstance(stable, bool):
            raise ValueError(f"regions[{index}].stable must be boolean")
        parents = item.get("parents", ())
        if (
            not isinstance(parents, Sequence)
            or isinstance(parents, (str, bytes))
            or any(not isinstance(parent, str) or not parent for parent in parents)
        ):
            raise ValueError(f"regions[{index}].parents must contain nonempty IDs")
        regions.append(
            validation_module.RegionShape(
                region_id,
                start,
                end,
                "stable" if stable else "active",
                tuple(parents),
            )
        )
    regions.sort(key=lambda region: (region.start, region.end, region.region_id))
    cursor = 0
    for region in regions:
        if region.start != cursor or region.end <= region.start:
            raise ValueError("regions must form a nonoverlapping, gap-free layout")
        cursor = region.end
    if cursor != total_tokens:
        raise ValueError("regions must cover total_tokens_per_request exactly")
    region_active_spans = tuple(
        (region.start, region.end) for region in regions if region.status == "active"
    )
    if region_active_spans != active_spans:
        raise ValueError("active_spans must exactly match non-stable regions")
    known_ids = {region.region_id for region in regions}
    missing_parents = sorted(
        {
            parent
            for region in regions
            for parent in region.parents
            if parent not in known_ids
        }
    )
    if missing_parents:
        raise ValueError(f"regions contain unknown parent IDs: {missing_parents}")
    return tuple(regions)


def normalize_case(
    value: Mapping[str, Any],
    *,
    validation_module: Any,
    seed: int,
    index: int,
    requested_routes: Sequence[str],
) -> tuple[Any, dict[str, Any]]:
    case_id = str(value.get("case_id", ""))
    if not case_id:
        raise ValueError(f"cases[{index}].case_id must be nonempty")
    total_tokens = _integer(
        value.get("total_tokens_per_request"),
        f"case {case_id} total_tokens_per_request",
        positive=True,
    )
    steps = _integer(
        value.get("diffusion_steps"),
        f"case {case_id} diffusion_steps",
        positive=True,
    )
    batch_size = _integer(
        value.get("batch_size"), f"case {case_id} batch_size", positive=True
    )
    if batch_size not in SUPPORTED_BATCH_SIZES:
        raise ValueError(
            f"case {case_id} batch_size={batch_size} is unsupported: the production "
            "harness has one request-pool slot and does not emulate batching"
        )
    active_spans = _normalize_spans(value.get("active_spans"), total_tokens)
    prefix_value = value.get("prefix_tokens")
    if prefix_value is not None:
        prefix_tokens = _integer(prefix_value, f"case {case_id} prefix_tokens")
        if value.get("regions") is not None:
            raise ValueError("prefix_tokens shorthand cannot be combined with regions")
        regions = _simple_suffix_regions(
            validation_module,
            total_tokens=total_tokens,
            prefix_tokens=prefix_tokens,
            active_spans=active_spans,
        )
    else:
        regions = _explicit_regions(
            validation_module,
            value.get("regions"),
            total_tokens=total_tokens,
            active_spans=active_spans,
        )
        prefix_tokens = min(start for start, _ in active_spans)

    active_region_ids = tuple(
        region.region_id for region in regions if region.status == "active"
    )
    case = validation_module.ValidationCase(
        case_id=case_id,
        profile="production_manifest",
        token_seed=int(seed) + index * 104729,
        sequence_length=total_tokens,
        regions=tuple(regions),
        edited_regions=active_region_ids,
        diffusion_steps=steps,
        batch_size=batch_size,
    )
    # Reuse the repository contract for parent existence, cycles, interval layout,
    # stable ancestry, and diffusion-step validation.
    validation_module.build_execution_spec(
        case, [0] * total_tokens, edited=False
    ).validate()
    cache_requested = any(route != "full_replay" for route in requested_routes)
    if cache_requested and prefix_tokens <= 0:
        raise ValueError(
            f"case {case_id} cannot construct a cached route before position zero"
        )
    normalized = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "case_id": case_id,
        "total_tokens_per_request": total_tokens,
        "prefix_tokens": prefix_tokens,
        "active_spans": [list(span) for span in active_spans],
        "diffusion_steps": steps,
        "batch_size": batch_size,
        "regions": [
            {
                "region_id": region.region_id,
                "start": region.start,
                "end": region.end,
                "stable": region.status == "stable",
                "parents": list(region.parents),
            }
            for region in regions
        ],
        "edited_regions": list(active_region_ids),
        "attention_contract": "region_dag_conservative_gdn_v1",
    }
    return case, normalized


def normalize_manifest(
    value: Mapping[str, Any],
    *,
    validation_module: Any,
    seed: int,
    requested_routes: Sequence[str],
) -> tuple[list[Any], list[dict[str, Any]]]:
    if not isinstance(value, Mapping):
        raise ValueError("case manifest must be a JSON object")
    if value.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported case manifest schema_version={value.get('schema_version')!r}"
        )
    raw_cases = value.get("cases")
    if (
        not isinstance(raw_cases, Sequence)
        or isinstance(raw_cases, (str, bytes))
        or not raw_cases
    ):
        raise ValueError("case manifest requires a nonempty cases array")
    cases = []
    normalized = []
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, Mapping):
            raise ValueError(f"cases[{index}] must be an object")
        case, record = normalize_case(
            raw_case,
            validation_module=validation_module,
            seed=seed,
            index=index,
            requested_routes=requested_routes,
        )
        cases.append(case)
        normalized.append(record)
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("case IDs must be unique")
    return cases, normalized


def load_manifest(
    path: Path,
    *,
    validation_module: Any,
    seed: int,
    requested_routes: Sequence[str],
) -> tuple[list[Any], list[dict[str, Any]]]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read case manifest {path}: {exc}") from exc
    return normalize_manifest(
        value,
        validation_module=validation_module,
        seed=seed,
        requested_routes=requested_routes,
    )


def legacy_production_case(
    validation_module: Any, *, seed: int, requested_routes: Sequence[str]
) -> tuple[list[Any], list[dict[str, Any]]]:
    value = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "cases": [
            {
                "case_id": "p2048-a64-s4-b1",
                "total_tokens_per_request": 2112,
                "prefix_tokens": 2048,
                "active_spans": [[2048, 2112]],
                "diffusion_steps": 4,
                "batch_size": 1,
            }
        ],
    }
    return normalize_manifest(
        value,
        validation_module=validation_module,
        seed=seed,
        requested_routes=requested_routes,
    )
