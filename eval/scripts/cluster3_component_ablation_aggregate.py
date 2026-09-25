#!/usr/bin/env python3
"""Aggregate isolated component-ablation JSONL with paired 95% intervals."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


FIXED_POLICIES = (
    "full_replay",
    "kv_only",
    "gdn_only",
    "kv_gdn_conservative",
    "kv_gdn_oracle",
)
BOOTSTRAP_REPETITIONS = 10_000


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def paired_bootstrap_ci(
    reference: Sequence[float],
    candidate: Sequence[float],
    statistic: Callable[[Sequence[float], Sequence[float]], float],
    *,
    seed: int,
) -> list[float]:
    if not reference or len(reference) != len(candidate):
        raise ValueError("paired bootstrap requires equal nonempty samples")
    rng = random.Random(seed)
    count = len(reference)
    values = []
    for _ in range(BOOTSTRAP_REPETITIONS):
        indices = [rng.randrange(count) for _ in range(count)]
        left = [float(reference[index]) for index in indices]
        right = [float(candidate[index]) for index in indices]
        values.append(float(statistic(left, right)))
    return [_percentile(values, 0.025), _percentile(values, 0.975)]


def _mean_difference(reference, candidate):
    return statistics.mean(reference) - statistics.mean(candidate)


def _mean_speedup(reference, candidate):
    return statistics.mean(left / right for left, right in zip(reference, candidate))


def _mean_reduction(reference, candidate):
    return statistics.mean(
        1.0 - right / left for left, right in zip(reference, candidate)
    )


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _by_repetition(records, policy):
    selected = {
        int(record["repetition_index"]): record
        for record in records
        if record.get("variant") == "uninstrumented"
        and record.get("ablation_policy") == policy
    }
    if set(selected) != set(range(10)):
        raise RuntimeError(f"{policy} lacks ten measured paired repetitions")
    return [selected[index] for index in range(10)]


def aggregate_job(job_dir: Path, policy: str) -> dict[str, Any]:
    records = _read_jsonl(job_dir / "measurements.jsonl")
    candidate = _by_repetition(records, policy)
    reference = (
        candidate if policy == "full_replay" else _by_repetition(records, "full_replay")
    )
    for expected, actual in zip(reference, candidate):
        if (
            expected["output_hash"] != actual["output_hash"]
            or expected["generated_top1_token_ids"]
            != actual["generated_top1_token_ids"]
        ):
            raise RuntimeError(f"{job_dir} output differs from canonical full replay")
        if int(actual["fallback_count"]) or int(actual["recovery_replay_count"]):
            raise RuntimeError(f"{job_dir} recovered or fell back")

    def values(rows, name):
        return [float(row[name]) for row in rows]

    uninstrumented = values(candidate, "uninstrumented_cuda_latency_ms")
    minimally_profiled = values(candidate, "minimally_profiled_cuda_latency_ms")
    full_latency = values(reference, "uninstrumented_cuda_latency_ms")
    full_attention = values(reference, "executed_attention_query_positions")
    attention = values(candidate, "executed_attention_query_positions")
    full_gdn = values(reference, "executed_gdn_replay_positions")
    gdn = values(candidate, "executed_gdn_replay_positions")
    full_memory = values(reference, "peak_allocated_gpu_memory_bytes")
    memory = values(candidate, "peak_allocated_gpu_memory_bytes")
    seed = int.from_bytes(f"{job_dir}:{policy}".encode()[:16], "little")
    return {
        "case_id": str(candidate[0]["case_id"]),
        "policy": policy,
        "sample_count": len(candidate),
        "uninstrumented_cuda_latency_ms": {
            "samples": uninstrumented,
            "median": statistics.median(uninstrumented),
        },
        "minimally_profiled_cuda_latency_ms": {
            "samples": minimally_profiled,
            "median": statistics.median(minimally_profiled),
        },
        "speedup_vs_full": {
            "paired_mean": _mean_speedup(full_latency, uninstrumented),
            "paired_bootstrap_95_ci": paired_bootstrap_ci(
                full_latency, uninstrumented, _mean_speedup, seed=seed ^ 1
            ),
        },
        "latency_reduction_ms": {
            "paired_mean": _mean_difference(full_latency, uninstrumented),
            "paired_bootstrap_95_ci": paired_bootstrap_ci(
                full_latency, uninstrumented, _mean_difference, seed=seed ^ 2
            ),
        },
        "attention_reduction_fraction": {
            "paired_mean": _mean_reduction(full_attention, attention),
            "paired_bootstrap_95_ci": paired_bootstrap_ci(
                full_attention, attention, _mean_reduction, seed=seed ^ 3
            ),
        },
        "gdn_reduction_fraction": {
            "paired_mean": _mean_reduction(full_gdn, gdn),
            "paired_bootstrap_95_ci": paired_bootstrap_ci(
                full_gdn, gdn, _mean_reduction, seed=seed ^ 4
            ),
        },
        "peak_allocated_gpu_memory_bytes": {
            "samples": memory,
            "median": statistics.median(memory),
            "paired_delta_vs_full_mean": -_mean_difference(full_memory, memory),
            "paired_delta_bootstrap_95_ci": paired_bootstrap_ci(
                full_memory,
                memory,
                lambda left, right: -_mean_difference(left, right),
                seed=seed ^ 5,
            ),
        },
        "correctness_pass": True,
    }


def aggregate(root: Path) -> dict[str, Any]:
    jobs_root = root / "jobs"
    cases = sorted(path for path in jobs_root.iterdir() if path.is_dir())
    results = []
    unsupported = set()
    failed = set()
    completed = set()
    for case_dir in cases:
        for policy in FIXED_POLICIES:
            job_dir = case_dir / policy
            summary_path = job_dir / "summary.json"
            records_path = job_dir / "measurements.jsonl"
            if not summary_path.exists():
                failed.add(policy)
                continue
            summary = _read_json(summary_path)
            if summary.get("unsupported_policies"):
                unsupported.add(policy)
                continue
            if (
                not records_path.exists()
                or summary.get("publication_acceptable") is not True
            ):
                failed.add(policy)
                continue
            results.append(aggregate_job(job_dir, policy))
            completed.add(policy)
    all_complete = (
        len(cases) == 12
        and len(results) == 12 * len(FIXED_POLICIES)
        and not unsupported
        and not failed
    )
    complete_policies = sorted(
        policy
        for policy in completed
        if policy not in unsupported
        and policy not in failed
        and sum(result["policy"] == policy for result in results) == len(cases)
    )
    failed_policies = sorted(policy for policy in failed if policy not in unsupported)
    return {
        "schema_version": 1,
        "artifact_type": "cluster3_component_ablation_aggregation",
        "case_count": len(cases),
        "completed_policies": complete_policies,
        "failed_policies": failed_policies,
        "unsupported_policies": sorted(unsupported),
        "correctness_pass": all_complete
        and all(result["correctness_pass"] for result in results),
        "publication_gate_pass": all_complete,
        "all_publication_gates_pass": all_complete,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    result = aggregate(args.input_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
