#!/usr/bin/env python3
"""Build a fail-closed hierarchical Cluster-3 profiling report."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

ROUTES = ("full_replay", "cold_handoff_build", "warm_cached_suffix")
OUTER_SUFFIX_CHILDREN = (
    "suffix_prepare_total",
    "model_forward_total",
    "suffix_finalize_total",
)
MODEL_FORWARD_CHILDREN = (
    "token_hidden_input_preparation",
    "decoder_layer_total",
    "final_normalization",
    "lm_head_projection",
)
SUFFIX_PREPARE_HOST_CHILDREN = (
    "runtime_plan_build",
    "request_metadata_attachment",
    "frontier_key_construction",
    "schedule_batch_initialization",
    "worker_batch_construction",
)
SUFFIX_PREPARE_CUDA_CHILDREN = (
    "canonical_prefix_location_materialization",
    "schedule_batch_prepare",
    "forward_batch_initialization",
    "flashinfer_plan_build",
    "model_request_slot_materialization",
)
DECODER_LAYER_CHILDREN = (
    "pre_attention_normalization",
    "attention_block_total",
    "gdn_block_total",
    "residual_post_attention_normalization",
    "mlp_forward",
    "residual_connection",
)
ATTENTION_BLOCK_CHILDREN = (
    "attention_qkv_projection",
    "rope_attention_preparation",
    "attention_forward",
    "attention_output_projection",
)
GDN_BLOCK_CHILDREN = (
    "gdn_input_projection",
    "region_cache_lookup",
    "gdn_restore",
    "gdn_forward",
    "prefix_snapshot",
    "cache_commit",
    "gdn_output_projection",
)
EXPECTED_REQUEST_TOTAL_CALLS = {
    "full_replay": 16,
    "cold_handoff_build": 24,
    "warm_cached_suffix": 20,
}
COVERAGE_TARGET_PERCENT = 90.0

EXPECTED_HIERARCHY = {
    "request_total": (None, "envelope"),
    "active_suffix_forward": ("request_total", "envelope"),
    **{phase: ("active_suffix_forward", "envelope") for phase in OUTER_SUFFIX_CHILDREN},
    **{
        phase: ("suffix_prepare_total", "leaf")
        for phase in SUFFIX_PREPARE_HOST_CHILDREN + SUFFIX_PREPARE_CUDA_CHILDREN
    },
    "runtime_plan_build": ("suffix_prepare_total", "envelope"),
    "schedule_batch_prepare": ("suffix_prepare_total", "envelope"),
    "forward_batch_initialization": ("suffix_prepare_total", "envelope"),
    **{phase: ("model_forward_total", "leaf") for phase in MODEL_FORWARD_CHILDREN},
    "decoder_layer_total": ("model_forward_total", "envelope"),
    **{phase: ("decoder_layer_total", "leaf") for phase in DECODER_LAYER_CHILDREN},
    "attention_block_total": ("decoder_layer_total", "envelope"),
    "gdn_block_total": ("decoder_layer_total", "envelope"),
    **{phase: ("attention_block_total", "leaf") for phase in ATTENTION_BLOCK_CHILDREN},
    **{phase: ("gdn_block_total", "leaf") for phase in GDN_BLOCK_CHILDREN},
}


def _stats(
    samples: Sequence[float], *, seed: int = 3, allow_negative: bool = False
) -> dict[str, Any]:
    values = [float(value) for value in samples]
    if not values or any(
        not math.isfinite(value) or (value < 0 and not allow_negative)
        for value in values
    ):
        qualifier = "finite" if allow_negative else "finite and nonnegative"
        raise RuntimeError(f"timing samples must be {qualifier}")
    median = statistics.median(values)
    rng = random.Random(seed)
    boot = sorted(
        statistics.median(rng.choices(values, k=len(values))) for _ in range(2000)
    )
    return {
        "samples": values,
        "median": median,
        "mean": statistics.fmean(values),
        "mad": statistics.median(abs(value - median) for value in values),
        "minimum": min(values),
        "maximum": max(values),
        "bootstrap_95_ci": [boot[49], boot[1949]],
    }


def _validate_snapshot(snapshot: Mapping[str, Any], route: str, index: int) -> None:
    if int(snapshot.get("schema_version", 0)) != 3 or not snapshot.get("finalized"):
        raise RuntimeError(f"{route}[{index}] is not a finalized schema-v3 snapshot")
    if snapshot.get("timing_scope") != "request_exclusive":
        raise RuntimeError(f"{route}[{index}] is not request-exclusive")
    if snapshot.get("debug_sync"):
        raise RuntimeError("debug synchronization is invalid for performance evidence")
    if int(snapshot.get("synchronization_count", -1)) != 1:
        raise RuntimeError(f"{route}[{index}] violates the one-sync timing protocol")
    metadata = snapshot.get("metadata") or {}
    expected_metadata = {
        "route": route,
        "repetition_index": index,
        "prefix_length": 2048,
        "active_length": 64,
        "diffusion_steps": 4,
        "batch_size": 1,
        "cache_status": "hit" if route == "warm_cached_suffix" else "miss",
        "request_total_semantics": "segmented_route_wall_aggregate",
        "expected_request_total_calls": EXPECTED_REQUEST_TOTAL_CALLS[route],
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise RuntimeError(
                f"{route}[{index}] has invalid {field}: "
                f"expected={expected!r} observed={metadata.get(field)!r}"
            )
    if not metadata.get("case_id"):
        raise RuntimeError(f"{route}[{index}] is missing case identity")
    phases = snapshot.get("phases")
    hierarchy = snapshot.get("hierarchy")
    if not isinstance(phases, Mapping) or not isinstance(hierarchy, Mapping):
        raise RuntimeError(f"{route}[{index}] has no phase hierarchy")
    for name, (expected_parent, expected_kind) in EXPECTED_HIERARCHY.items():
        definition = hierarchy.get(name)
        if not isinstance(definition, Mapping):
            raise RuntimeError(f"missing required phase: {name}")
        if (
            definition.get("parent") != expected_parent
            or definition.get("kind") != expected_kind
        ):
            raise RuntimeError(f"invalid schema-v3 hierarchy definition: {name}")
    if (
        int(phases.get("request_total", {}).get("calls", -1))
        != EXPECTED_REQUEST_TOTAL_CALLS[route]
    ):
        raise RuntimeError(
            f"{route}[{index}] has an unexpected request-total call count"
        )
    for name, phase in phases.items():
        if not isinstance(phase, Mapping) or name not in hierarchy:
            raise RuntimeError(f"invalid phase hierarchy evidence: {name}")
        definition = hierarchy[name]
        if phase.get("parent_phase") != definition.get("parent") or phase.get(
            "hierarchy_kind"
        ) != definition.get("kind"):
            raise RuntimeError(f"phase hierarchy mismatch: {route}[{index}].{name}")
        for field in ("host_ms", "cuda_ms"):
            value = phase.get(field)
            if value is not None and (
                not math.isfinite(float(value)) or float(value) < 0
            ):
                raise RuntimeError(
                    "timing samples must be finite and nonnegative: "
                    f"{route}[{index}].{name}.{field}"
                )
        if int(phase.get("calls", -1)) < 0:
            raise RuntimeError(f"invalid call count: {route}[{index}].{name}")
        if not phase.get("available", True) and not phase.get("availability_reason"):
            raise RuntimeError(f"unavailable phase lacks a reason: {name}")


def _phase_samples(snapshots, phase: str, domain: str) -> list[float]:
    field = "cuda_ms" if domain == "cuda" else "host_ms"
    values = []
    for snapshot in snapshots:
        evidence = snapshot["phases"].get(phase)
        if not isinstance(evidence, Mapping):
            raise RuntimeError(f"missing required phase: {phase}")
        value = evidence.get(field)
        values.append(0.0 if value is None else float(value))
    return values


def _phase_row(snapshots, phase: str, domain: str, seed: int) -> dict[str, Any]:
    evidence = [snapshot["phases"][phase] for snapshot in snapshots]
    return {
        "phase": phase,
        "parent_phase": evidence[0].get("parent_phase"),
        "hierarchy_kind": evidence[0].get("hierarchy_kind"),
        "timing_domain": domain,
        "timing_scope": "request_exclusive",
        "calls": [int(value["calls"]) for value in evidence],
        "timing_ms": _stats(_phase_samples(snapshots, phase, domain), seed=seed),
        "availability_reasons": sorted(
            {
                str(value["availability_reason"])
                for value in evidence
                if value.get("availability_reason")
            }
        ),
    }


def _direct_decomposition(snapshots, children, domain: str, seed: int):
    return [
        _phase_row(snapshots, phase, domain, seed + index)
        for index, phase in enumerate(children)
    ]


def _coverage(snapshots, parent: str, children, *, seed: int) -> dict[str, Any]:
    totals = _phase_samples(snapshots, parent, "cuda")
    child_samples = {
        child: _phase_samples(snapshots, child, "cuda") for child in children
    }
    attributed = [
        sum(child_samples[child][index] for child in children)
        for index in range(len(totals))
    ]
    residuals = [total - attributed[index] for index, total in enumerate(totals)]
    percentages = [
        100.0 * attributed[index] / total if total else 0.0
        for index, total in enumerate(totals)
    ]
    tolerances = [max(0.05, total * 0.005) for total in totals]
    overlap_violations = [
        residual < -tolerances[index] for index, residual in enumerate(residuals)
    ]
    return {
        "parent_phase": parent,
        "direct_children": list(children),
        "parent_ms": _stats(totals, seed=seed),
        "attributed_ms": _stats(attributed, seed=seed + 1),
        "residual_ms": _stats(residuals, seed=seed + 2, allow_negative=True),
        "percent": _stats(percentages, seed=seed + 3),
        "overlap_violation": overlap_violations,
        "pass": min(percentages) >= COVERAGE_TARGET_PERCENT
        and not any(overlap_violations),
    }


def _host_decomposition(snapshots):
    rows = []
    for name, definition in snapshots[0]["hierarchy"].items():
        if definition.get("kind") != "leaf":
            continue
        evidence = [snapshot["phases"][name] for snapshot in snapshots]
        if any(value.get("timing_domain") == "cuda" for value in evidence):
            continue
        rows.append(_phase_row(snapshots, name, "host", len(rows) + 101))
    return sorted(rows, key=lambda row: row["timing_ms"]["median"], reverse=True)


def summarize(record: Mapping[str, Any]) -> dict[str, Any]:
    if record.get("profile") != "efficiency_one":
        raise RuntimeError("hierarchical report requires the efficiency_one profile")
    if not record.get("case_pass"):
        raise RuntimeError("cannot report profiling coverage from a failed case")
    if record.get("dtype") != "bfloat16" or int(record.get("tp_size", 0)) != 1:
        raise RuntimeError("hierarchical report requires BF16 and TP=1")
    protocol = record.get("timing_protocol") or {}
    if protocol.get("debug_sync_stages", False):
        raise RuntimeError("debug synchronization is invalid for performance evidence")
    if int(protocol.get("warmup_repetitions", 0)) != 3:
        raise RuntimeError("efficiency profile requires exactly three warmups")
    if int(protocol.get("timed_repetitions", 0)) != 10:
        raise RuntimeError("efficiency profile requires exactly ten repetitions")
    profiles = record.get("request_phase_profiles")
    if not isinstance(profiles, Mapping):
        raise RuntimeError("missing schema-v3 request_phase_profiles")

    routes = {}
    for route in ROUTES:
        snapshots = profiles.get(route)
        if not isinstance(snapshots, list) or len(snapshots) != 10:
            raise RuntimeError(f"{route} requires exactly ten finalized snapshots")
        for index, snapshot in enumerate(snapshots):
            _validate_snapshot(snapshot, route, index)
        routes[route] = snapshots

    warm = routes["warm_cached_suffix"]
    if any(int(s["counters"].get("recovery_replay_count", 0)) for s in warm):
        raise RuntimeError("valid warm hits contain unexplained recovery replay")
    required_warm_phases = {
        "active_suffix_forward",
        *OUTER_SUFFIX_CHILDREN,
        *MODEL_FORWARD_CHILDREN,
        *EXPECTED_HIERARCHY,
    }
    for snapshot in warm:
        for phase in required_warm_phases:
            if phase not in snapshot["phases"]:
                raise RuntimeError(f"missing required phase: {phase}")
    for phase, calls in {
        "active_suffix_forward": 4,
        "suffix_prepare_total": 4,
        "model_forward_total": 4,
        "suffix_finalize_total": 4,
        "token_hidden_input_preparation": 4,
        "decoder_layer_total": 96,
        "final_normalization": 4,
        "lm_head_projection": 4,
        "attention_block_total": 24,
        "gdn_block_total": 72,
        "attention_forward": 24,
        "gdn_forward": 72,
        "mlp_forward": 96,
        "prefix_snapshot": 144,
    }.items():
        observed = [int(s["phases"][phase]["calls"]) for s in warm]
        if any(value != calls for value in observed):
            raise RuntimeError(f"unexpected warm call count for {phase}: {observed}")

    outer = _coverage(warm, "active_suffix_forward", OUTER_SUFFIX_CHILDREN, seed=211)
    model = _coverage(warm, "model_forward_total", MODEL_FORWARD_CHILDREN, seed=223)
    if not outer["pass"] or not model["pass"]:
        raise RuntimeError(
            "hierarchical CUDA coverage below 90%: "
            f"outer_min={outer['percent']['minimum']:.2f}% "
            f"model_min={model['percent']['minimum']:.2f}%"
        )

    host_rows = _host_decomposition(warm)
    host_total = _phase_samples(warm, "request_total", "host")
    host_attributed = [
        sum(row["timing_ms"]["samples"][index] for row in host_rows)
        for index in range(len(warm))
    ]
    host_residual = [
        total - host_attributed[index] for index, total in enumerate(host_total)
    ]
    hierarchy = dict(warm[0]["hierarchy"])

    snapshot_contract = record.get("warm_snapshot_behavior")
    if not isinstance(snapshot_contract, Mapping) or not snapshot_contract.get("pass"):
        raise RuntimeError("warm snapshot behavior contract failed")
    if snapshot_contract.get("expected_zero") is not False or not snapshot_contract.get(
        "reason"
    ):
        raise RuntimeError("warm snapshots require an explicit nonzero contract")
    observed_snapshot_calls = sum(
        int(snapshot["counters"].get("snapshot_count", 0)) for snapshot in warm
    )
    if int(snapshot_contract.get("calls", -1)) != observed_snapshot_calls:
        raise RuntimeError("warm snapshot behavior call count is inconsistent")

    return {
        "schema_version": 3,
        "case_id": record.get("case_id"),
        "revision": record.get("revision"),
        "hardware": record.get("hardware"),
        "hierarchy": hierarchy,
        "request_total_semantics": {
            "kind": "segmented_route_wall_aggregate",
            "expected_calls": EXPECTED_REQUEST_TOTAL_CALLS,
        },
        "route_totals": {
            route: {
                "request_wall_ms": _stats(
                    _phase_samples(snapshots, "request_total", "host"), seed=251
                ),
                "model_cuda_ms": _stats(
                    _phase_samples(snapshots, "model_forward_total", "cuda"),
                    seed=257,
                ),
            }
            for route, snapshots in routes.items()
        },
        "outer_suffix_decomposition": _direct_decomposition(
            warm, OUTER_SUFFIX_CHILDREN, "cuda", 271
        ),
        "suffix_prepare_decomposition": {
            "host": _direct_decomposition(
                warm, SUFFIX_PREPARE_HOST_CHILDREN, "host", 275
            ),
            "cuda": _direct_decomposition(
                warm, SUFFIX_PREPARE_CUDA_CHILDREN, "cuda", 277
            ),
        },
        "model_forward_decomposition": _direct_decomposition(
            warm, MODEL_FORWARD_CHILDREN, "cuda", 281
        ),
        "decoder_layer_decomposition": _direct_decomposition(
            warm, DECODER_LAYER_CHILDREN, "cuda", 283
        ),
        "attention_block_decomposition": _direct_decomposition(
            warm, ATTENTION_BLOCK_CHILDREN, "cuda", 287
        ),
        "gdn_block_decomposition": _direct_decomposition(
            warm, GDN_BLOCK_CHILDREN, "cuda", 289
        ),
        "host_orchestration_decomposition": host_rows,
        "outer_suffix_coverage_percent": outer["percent"]["median"],
        "model_forward_coverage_percent": model["percent"]["median"],
        "outer_suffix_residual_ms": outer["residual_ms"],
        "model_forward_residual_ms": model["residual_ms"],
        "host_orchestration_coverage": {
            "parent_phase": "request_total",
            "attributed_ms": _stats(host_attributed, seed=293),
            "residual_ms": _stats(host_residual, seed=307, allow_negative=True),
            "interpretation": (
                "host-only leaf coverage of a segmented route-wall aggregate; "
                "CUDA durations are excluded"
            ),
        },
        "coverage": {
            "outer_suffix_percent": outer["percent"]["median"],
            "model_forward_percent": model["percent"]["median"],
            "outer_suffix_percent_distribution": outer["percent"],
            "model_forward_percent_distribution": model["percent"],
            "outer_suffix_pass": outer["pass"],
            "model_forward_pass": model["pass"],
        },
        "phase_call_counts": {
            route: {
                phase: [int(s["phases"][phase]["calls"]) for s in snapshots]
                for phase in snapshots[0]["phases"]
            }
            for route, snapshots in routes.items()
        },
        "raw_phase_inventory": {
            route: [dict(snapshot["phases"]) for snapshot in snapshots]
            for route, snapshots in routes.items()
        },
        "route_counters": {
            route: [dict(snapshot["counters"]) for snapshot in snapshots]
            for route, snapshots in routes.items()
        },
        "synchronization_counts": {
            route: [int(snapshot["synchronization_count"]) for snapshot in snapshots]
            for route, snapshots in routes.items()
        },
        "warm_snapshot_behavior": dict(snapshot_contract),
        "unsupported_claims": [
            "No speedup or optimization claim is supported by instrumentation-only evidence.",
            "Host and CUDA durations are separate domains and are never added.",
            "Nested model leaves are not added to model_forward_total at the outer suffix level.",
        ],
    }


def markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Cluster-3 hierarchical profiling coverage",
        "",
        f"Revision: `{summary['revision']}`  ",
        f"Case: `{summary['case_id']}`  ",
        "",
        "## Outer suffix decomposition",
        "",
        "| Phase | CUDA median ms |",
        "|---|---:|",
    ]
    for row in summary["outer_suffix_decomposition"]:
        lines.append(f"| {row['phase']} | {row['timing_ms']['median']:.4f} |")
    lines.extend(
        (
            "",
            "## Model-forward decomposition",
            "",
            "| Phase | CUDA median ms |",
            "|---|---:|",
        )
    )
    for row in summary["model_forward_decomposition"]:
        lines.append(f"| {row['phase']} | {row['timing_ms']['median']:.4f} |")
    coverage = summary["coverage"]
    lines.extend(
        (
            "",
            "## Coverage gates",
            "",
            f"- Outer suffix: {coverage['outer_suffix_percent']:.2f}% — PASS",
            f"- Model forward: {coverage['model_forward_percent']:.2f}% — PASS",
            "- Host and CUDA timings are reported separately and never summed.",
            "",
            "## Unsupported claims",
            "",
        )
    )
    lines.extend(f"- {value}" for value in summary["unsupported_claims"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in args.record.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 1:
        raise RuntimeError("efficiency_one JSONL must contain exactly one record")
    summary = summarize(records[0])
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
