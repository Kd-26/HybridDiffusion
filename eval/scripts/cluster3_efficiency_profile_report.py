#!/usr/bin/env python3
"""Build a fail-closed report from request-scoped Cluster-3 profiles."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

ROUTES = ("full_replay", "cold_handoff_build", "warm_cached_suffix")
ENVELOPES = frozenset(("request_total", "model_forward_total", "active_suffix_forward"))


def _stats(samples: Sequence[float], *, seed: int = 3) -> dict[str, Any]:
    values = [float(value) for value in samples]
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise RuntimeError("timing samples must be finite and nonnegative")
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
    if int(snapshot.get("schema_version", 0)) < 2 or not snapshot.get("finalized"):
        raise RuntimeError(f"{route}[{index}] is not a finalized profiling snapshot")
    if snapshot.get("timing_scope") != "request_exclusive":
        raise RuntimeError(f"{route}[{index}] is not request-exclusive")
    if snapshot.get("debug_sync"):
        raise RuntimeError("debug synchronization is invalid for performance evidence")
    if int(snapshot.get("synchronization_count", -1)) != 1:
        raise RuntimeError(f"{route}[{index}] violates the one-sync timing protocol")
    metadata = snapshot.get("metadata") or {}
    if (
        metadata.get("route") != route
        or int(metadata.get("repetition_index", -1)) != index
    ):
        raise RuntimeError(f"{route}[{index}] has inconsistent route metadata")
    expected_metadata = {
        "prefix_length": 2048,
        "active_length": 64,
        "diffusion_steps": 4,
        "batch_size": 1,
        "cache_status": "hit" if route == "warm_cached_suffix" else "miss",
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
    if not isinstance(phases, Mapping):
        raise RuntimeError(f"{route}[{index}] has no phase evidence")
    for name, phase in phases.items():
        if not isinstance(phase, Mapping):
            raise RuntimeError(f"invalid phase evidence: {name}")
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


def _phase_samples(snapshots, phase: str, domain: str) -> list[float]:
    field = "cuda_ms" if domain == "cuda" else "host_ms"
    return [
        (
            0.0
            if snapshot["phases"][phase].get(field) is None
            else float(snapshot["phases"][phase][field])
        )
        for snapshot in snapshots
    ]


def _ranking(snapshots, domain: str) -> list[dict[str, Any]]:
    field = "cuda_ms" if domain == "cuda" else "host_ms"
    rows = []
    for name in snapshots[0]["phases"]:
        if name in ENVELOPES:
            continue
        evidence = [snapshot["phases"][name] for snapshot in snapshots]
        if domain == "cuda" and not any(
            value.get(field) is not None for value in evidence
        ):
            continue
        if domain == "host" and any(
            value.get("timing_domain") == "cuda" for value in evidence
        ):
            continue
        rows.append(
            {
                "phase": name,
                "timing_domain": domain,
                "timing_scope": "request_exclusive",
                "calls": [int(value["calls"]) for value in evidence],
                "timing_ms": _stats(
                    _phase_samples(snapshots, name, domain), seed=len(rows) + 7
                ),
                "availability_reasons": sorted(
                    {
                        str(value["availability_reason"])
                        for value in evidence
                        if value.get("availability_reason")
                    }
                ),
            }
        )
    return sorted(rows, key=lambda row: row["timing_ms"]["median"], reverse=True)


def summarize(record: Mapping[str, Any]) -> dict[str, Any]:
    if record.get("profile") != "efficiency_one":
        raise RuntimeError("bottleneck report requires the efficiency_one profile")
    if not record.get("case_pass"):
        raise RuntimeError("cannot report bottlenecks from a failed correctness case")
    if record.get("dtype") != "bfloat16" or int(record.get("tp_size", 0)) != 1:
        raise RuntimeError("bottleneck report requires BF16 and TP=1")
    protocol = record.get("timing_protocol") or {}
    if protocol.get("debug_sync_stages", False):
        raise RuntimeError("debug synchronization is invalid for performance evidence")
    if int(protocol.get("warmup_repetitions", 0)) < 3:
        raise RuntimeError("efficiency profile requires three warmups")
    measured = int(protocol.get("timed_repetitions", 0))
    if measured != 10:
        raise RuntimeError(
            "efficiency profile requires exactly ten measured repetitions"
        )
    profiles = record.get("request_phase_profiles")
    if not isinstance(profiles, Mapping):
        raise RuntimeError(
            "missing request_phase_profiles; legacy timings are not substituted"
        )

    routes = {}
    for route in ROUTES:
        snapshots = profiles.get(route)
        if not isinstance(snapshots, list) or len(snapshots) != measured:
            raise RuntimeError(f"{route} requires exactly ten finalized snapshots")
        for index, snapshot in enumerate(snapshots):
            _validate_snapshot(snapshot, route, index)
        routes[route] = snapshots

    warm = routes["warm_cached_suffix"]
    if any(int(s["counters"].get("recovery_replay_count", 0)) for s in warm):
        raise RuntimeError("valid warm hits contain unexplained recovery replay")
    route_totals = {
        route: {
            "request_wall_ms": _stats(
                _phase_samples(s, "request_total", "host"), seed=31
            ),
            "model_cuda_ms": _stats(
                _phase_samples(s, "model_forward_total", "cuda"), seed=37
            ),
        }
        for route, s in routes.items()
    }
    host_ranking = _ranking(warm, "host")
    cuda_ranking = _ranking(warm, "cuda")
    gpu_total = _phase_samples(warm, "active_suffix_forward", "cuda")
    gpu_leaf_names = [row["phase"] for row in cuda_ranking]
    gpu_residuals = [
        max(
            total
            - sum(_phase_samples(warm, phase, "cuda")[i] for phase in gpu_leaf_names),
            0.0,
        )
        for i, total in enumerate(gpu_total)
    ]
    host_total = _phase_samples(warm, "request_total", "host")
    host_leaf_names = [row["phase"] for row in host_ranking]
    host_residuals = [
        max(
            total
            - sum(_phase_samples(warm, phase, "host")[i] for phase in host_leaf_names),
            0.0,
        )
        for i, total in enumerate(host_total)
    ]
    gpu_residual = _stats(gpu_residuals, seed=41)
    gpu_total_stats = _stats(gpu_total, seed=43)
    residual_percent = (
        100.0 * gpu_residual["median"] / gpu_total_stats["median"]
        if gpu_total_stats["median"]
        else 0.0
    )
    snapshot_contract = record.get("warm_snapshot_behavior")
    if not isinstance(snapshot_contract, Mapping):
        raise RuntimeError("missing fail-closed warm_snapshot_behavior contract")
    observed_snapshot_calls = sum(
        int(snapshot["counters"].get("snapshot_count", 0)) for snapshot in warm
    )
    if int(snapshot_contract.get("calls", -1)) != observed_snapshot_calls:
        raise RuntimeError("warm snapshot contract disagrees with finalized profiles")
    if not snapshot_contract.get("pass"):
        raise RuntimeError("warm snapshot behavior contract failed")
    if snapshot_contract.get("expected_zero") is not False:
        raise RuntimeError(
            "warm snapshot behavior must use the explicit nonzero contract"
        )
    if not snapshot_contract.get("reason"):
        raise RuntimeError("nonzero warm snapshot behavior requires an explicit reason")

    return {
        "schema_version": 2,
        "case_id": record.get("case_id"),
        "revision": record.get("revision"),
        "hardware": record.get("hardware"),
        "route_totals": route_totals,
        "host_orchestration_decomposition": host_ranking,
        "gpu_model_critical_path_decomposition": cuda_ranking,
        "per_repetition_residuals": {
            "host_ms": _stats(host_residuals, seed=47),
            "cuda_ms": gpu_residual,
        },
        "phase_call_counts": {
            route: {
                phase: [int(s["phases"][phase]["calls"]) for s in snapshots]
                for phase in snapshots[0]["phases"]
            }
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
        "profiling_coverage": {
            "warm_attributed_ms": max(
                gpu_total_stats["median"] - gpu_residual["median"], 0.0
            ),
            "warm_residual_ms": gpu_residual["median"],
            "warm_residual_percent": residual_percent,
            "coverage_pass": residual_percent <= 10.0,
        },
        "unsupported_claims": [
            "No speedup or optimization claim is supported by instrumentation-only evidence.",
            "Host and CUDA durations are separate domains and must not be added.",
            "Request wall time can include host/GPU overlap and synchronization outside a leaf.",
        ],
    }


def markdown(summary: Mapping[str, Any]) -> str:
    coverage = summary["profiling_coverage"]
    lines = [
        "# Cluster-3 production-efficiency profiling coverage",
        "",
        f"Revision: `{summary['revision']}`  ",
        f"Case: `{summary['case_id']}`  ",
        "",
        "## GPU/model critical path (warm route)",
        "",
        "| Rank | Phase | CUDA median ms |",
        "|---:|---|---:|",
    ]
    for index, row in enumerate(summary["gpu_model_critical_path_decomposition"], 1):
        lines.append(f"| {index} | {row['phase']} | {row['timing_ms']['median']:.4f} |")
    lines.extend(
        (
            "",
            "## Host/orchestration (warm route)",
            "",
            "| Rank | Phase | Host median ms |",
            "|---:|---|---:|",
        )
    )
    for index, row in enumerate(summary["host_orchestration_decomposition"], 1):
        lines.append(f"| {index} | {row['phase']} | {row['timing_ms']['median']:.4f} |")
    lines.extend(
        (
            "",
            "## Profiling coverage",
            "",
            f"- Warm attributed CUDA median: {coverage['warm_attributed_ms']:.4f} ms",
            f"- Warm residual CUDA median: {coverage['warm_residual_ms']:.4f} ms",
            f"- Warm residual: {coverage['warm_residual_percent']:.2f}%",
            f"- Coverage gate: {'PASS' if coverage['coverage_pass'] else 'FAIL'}",
            "",
            "Host and CUDA timings are reported separately; they are not summed.",
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
