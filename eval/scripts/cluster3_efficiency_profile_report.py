#!/usr/bin/env python3
"""Build a fail-closed bottleneck report from the A30 efficiency-one record."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


REQUIRED_TIMINGS = (
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
)


def _median(timings: Mapping[str, Any], name: str) -> float:
    evidence = timings.get(name)
    if not isinstance(evidence, Mapping):
        raise RuntimeError(f"missing timing evidence: {name}")
    value = evidence.get("median")
    samples = evidence.get("samples")
    if value is None or not isinstance(samples, list) or len(samples) < 10:
        raise RuntimeError(f"timing lacks ten samples and a median: {name}")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise RuntimeError(f"invalid timing median: {name}={value}")
    return value


def summarize(record: Mapping[str, Any]) -> dict[str, Any]:
    if record.get("profile") != "efficiency_one":
        raise RuntimeError("bottleneck report requires the efficiency_one profile")
    if not record.get("case_pass"):
        raise RuntimeError("cannot report bottlenecks from a failed correctness case")
    if record.get("dtype") != "bfloat16" or int(record.get("tp_size", 0)) != 1:
        raise RuntimeError("bottleneck report requires BF16 and TP=1")
    protocol = record.get("timing_protocol") or {}
    if protocol.get("debug_sync_stages", False):
        raise RuntimeError("debug-synchronized evidence is not valid for performance")
    if int(protocol.get("warmup_repetitions", 0)) < 3:
        raise RuntimeError("efficiency profile requires three warmups")
    if int(protocol.get("timed_repetitions", 0)) < 10:
        raise RuntimeError("efficiency profile requires ten measured repetitions")

    timings = record.get("component_timings_ms") or {}
    values = {name: _median(timings, name) for name in REQUIRED_TIMINGS}
    warm = values["warm_cached_suffix_ms"]
    full = values["reference_full_ms"]
    cold = values["canonical_frontier_establishment_ms"] + warm
    restore = values["cache_lookup_restore_ms"]
    snapshot = values["prefix_snapshot_ms"]
    # GDN backend timing includes its nested restore/snapshot calls.  Rank the
    # residual kernel work so the rows remain additive instead of double-counted.
    gdn_exclusive = max(values["gdn_replay_ms"] - restore - snapshot, 0.0)
    phases = {
        "attention_forward": values["full_attention_ms"],
        "gdn_forward_exclusive": gdn_exclusive,
        "mlp_forward": values["mlp_forward_ms"],
        "prefix_snapshot": snapshot,
        "gdn_restore": restore,
        "region_mask_build": values["mask_build_ms"],
        "kv_restore_and_batch_assembly": values["gather_scatter_ms"],
    }
    attributed = sum(phases.values())
    phases["unattributed_scheduler_and_model_work"] = max(warm - attributed, 0.0)
    ranking = sorted(
        (
            {
                "phase": name,
                "warm_ms": value,
                "percent_of_warm_cached_suffix": (
                    100.0 * value / warm if warm else 0.0
                ),
                "cold_ms": None,
                "difference_from_full_replay_ms": None,
            }
            for name, value in phases.items()
        ),
        key=lambda row: row["warm_ms"],
        reverse=True,
    )
    return {
        "schema_version": 1,
        "case_id": record.get("case_id"),
        "revision": record.get("revision"),
        "hardware": record.get("hardware"),
        "route": "single_stable_prefix_one_active_suffix",
        "prefix_length": 2048,
        "active_length": 64,
        "diffusion_steps": 4,
        "batch_size": 1,
        "cold_handoff_build_ms": cold,
        "warm_cached_suffix_ms": warm,
        "full_replay_ms": full,
        "warm_minus_full_replay_ms": warm - full,
        "warm_reduction_vs_full_replay_pct": 100.0 * (1.0 - warm / full),
        "ranking": ranking,
        "cold_phase_breakdown_available": False,
        "unsupported_claims": [
            "No end-to-end speedup claim is valid from this profiling-only record.",
            "Cold per-phase attribution is unavailable in the frozen exporter.",
            "Optimization selection requires review of the retained A30/Nsight evidence.",
        ],
    }


def markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Cluster-3 production-efficiency bottleneck profile",
        "",
        f"Revision: `{summary['revision']}`  ",
        f"Case: `{summary['case_id']}`  ",
        "Route: stable prefix 2048 | active suffix 64, 4 steps, batch size 1",
        "",
        "| Rank | Phase | Warm ms | % of warm suffix |",
        "|---:|---|---:|---:|",
    ]
    for index, row in enumerate(summary["ranking"], 1):
        lines.append(
            f"| {index} | {row['phase']} | {row['warm_ms']:.4f} | "
            f"{row['percent_of_warm_cached_suffix']:.2f}% |"
        )
    lines.extend(
        (
            "",
            "## Route totals",
            "",
            f"- Full replay median: {summary['full_replay_ms']:.4f} ms",
            f"- Cold handoff build median: {summary['cold_handoff_build_ms']:.4f} ms",
            f"- Warm cached suffix median: {summary['warm_cached_suffix_ms']:.4f} ms",
            f"- Warm minus full replay: {summary['warm_minus_full_replay_ms']:.4f} ms",
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
