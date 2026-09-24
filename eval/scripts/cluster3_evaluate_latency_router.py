#!/usr/bin/env python3
"""Evaluate a conservative latency policy on held-out benchmark cases."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence


ROUTER_PATH = Path(__file__).parents[1] / "sglang/srt/dllm/region/latency_router.py"
_ROUTER_SPEC = importlib.util.spec_from_file_location(
    "cluster3_evaluate_latency_router_policy", ROUTER_PATH
)
if _ROUTER_SPEC is None or _ROUTER_SPEC.loader is None:
    raise RuntimeError(f"cannot load latency router: {ROUTER_PATH}")
_ROUTER = importlib.util.module_from_spec(_ROUTER_SPEC)
sys.modules[_ROUTER_SPEC.name] = _ROUTER
_ROUTER_SPEC.loader.exec_module(_ROUTER)
ConservativeLatencyRouter = _ROUTER.ConservativeLatencyRouter
router_inputs_from_normalized_case = _ROUTER.router_inputs_from_normalized_case


def read_jsonl(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, Mapping):
                        raise ValueError(f"{path} contains a non-object JSONL row")
                    records.append(dict(value))
    return records


def aggregate_measurements(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    samples: dict[tuple[str, str], list[float]] = defaultdict(list)
    cases = {}
    hashes: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        if record.get("variant") != "uninstrumented":
            continue
        case_id = str(record.get("case_id", ""))
        route = str(record.get("route", ""))
        case = record.get("normalized_case")
        if not case_id or not route or not isinstance(case, Mapping):
            raise ValueError("evaluation record lacks case, route, or normalized_case")
        cases[case_id] = dict(case)
        latency = float(record["timing"]["production_route_total_cuda_ms"])
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError("evaluation latency must be finite and positive")
        samples[(case_id, route)].append(latency)
        hashes[(case_id, route)].add(str(record.get("output_hash", "")))
    result = {}
    for case_id, case in cases.items():
        latencies = {
            route: statistics.median(values)
            for (sample_case, route), values in samples.items()
            if sample_case == case_id
        }
        if "full_replay" not in latencies:
            raise ValueError(f"case {case_id} is missing full_replay")
        route_hashes = {
            route: next(iter(values)) if len(values) == 1 else None
            for (sample_case, route), values in hashes.items()
            if sample_case == case_id
        }
        if len(set(route_hashes.values())) != 1 or None in route_hashes.values():
            raise ValueError(f"case {case_id} output hashes differ across routes")
        result[case_id] = {
            "case": case,
            "latencies": latencies,
            "output_hash": next(iter(route_hashes.values())),
        }
    return result


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def evaluate_policy(
    policy: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    decision_repetitions: int = 100,
) -> dict[str, Any]:
    measured = aggregate_measurements(records)
    test_ids = list((policy.get("data_split") or {}).get("test_case_ids") or ())
    if not test_ids:
        raise ValueError("policy has no held-out test case IDs")
    missing = sorted(set(test_ids) - set(measured))
    if missing:
        raise ValueError(f"held-out benchmark data is missing cases: {missing}")
    router = ConservativeLatencyRouter(policy)
    rows = []
    decision_samples = []
    for case_id in test_ids:
        value = measured[case_id]
        latencies = value["latencies"]
        inputs = router_inputs_from_normalized_case(
            value["case"],
            cache_available="warm_cached_suffix" in latencies,
            cache_state=("warm" if "warm_cached_suffix" in latencies else "absent"),
            cache_constructible="cold_handoff_build" in latencies,
            compatible=True,
        )
        started = time.perf_counter_ns()
        decisions = [router.decide(inputs) for _ in range(decision_repetitions)]
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        if any(decision != decisions[0] for decision in decisions[1:]):
            raise RuntimeError("router decision is nondeterministic")
        decision = decisions[0]
        decision_samples.append(elapsed_ms / decision_repetitions)
        valid = dict(latencies)
        oracle_route = min(valid, key=lambda route: (valid[route], route))
        selected_route = decision.selected_route
        actual_route = selected_route if selected_route in valid else "full_replay"
        selected_latency = valid[actual_route]
        oracle_latency = valid[oracle_route]
        regret_ms = selected_latency - oracle_latency
        regret_fraction = regret_ms / oracle_latency
        rows.append(
            {
                "case_id": case_id,
                "selected_route": selected_route,
                "actual_route": actual_route,
                "oracle_route": oracle_route,
                "selected_latency_ms": selected_latency,
                "oracle_latency_ms": oracle_latency,
                "regret_ms": regret_ms,
                "regret_fraction": regret_fraction,
                "full_replay_latency_ms": valid["full_replay"],
                "cold_handoff_latency_ms": valid.get("cold_handoff_build"),
                "warm_cache_latency_ms": valid.get("warm_cached_suffix"),
                "output_hash": value["output_hash"],
                "fallback_reason": decision.fallback_reason,
                "decision": decision.to_dict(),
            }
        )

    def total_for(policy_name: str) -> float:
        total = 0.0
        for row in rows:
            if policy_name == "adaptive":
                total += row["selected_latency_ms"]
            elif policy_name == "oracle":
                total += row["oracle_latency_ms"]
            elif policy_name == "full_replay":
                total += row["full_replay_latency_ms"]
            elif policy_name == "cold_handoff_build":
                total += row["cold_handoff_latency_ms"] or row["full_replay_latency_ms"]
            elif policy_name == "warm_cached_suffix":
                total += row["warm_cache_latency_ms"] or row["full_replay_latency_ms"]
        return total

    regrets = [row["regret_fraction"] for row in rows]
    exact = sum(row["actual_route"] == row["oracle_route"] for row in rows)
    adaptive_total = total_for("adaptive")
    oracle_total = total_for("oracle")
    full_total = total_for("full_replay")
    overhead_median = statistics.median(decision_samples)
    selected_median = statistics.median(row["selected_latency_ms"] for row in rows)
    metrics = {
        "case_count": len(rows),
        "exact_route_selection_accuracy": exact / len(rows),
        "aggregate_latency_ms": adaptive_total,
        "aggregate_regret_ms": adaptive_total - oracle_total,
        "aggregate_regret_fraction": (adaptive_total - oracle_total) / oracle_total,
        "median_regret_fraction": statistics.median(regrets),
        "p95_regret_fraction": _percentile(regrets, 0.95),
        "worst_case_regret_fraction": max(regrets),
        "cases_within_2pct_of_oracle": sum(value <= 0.02 for value in regrets)
        / len(regrets),
        "router_decision_overhead_ms_median": overhead_median,
        "router_decision_overhead_ms_p95": _percentile(decision_samples, 0.95),
        "router_overhead_fraction_of_selected_median": overhead_median
        / selected_median,
        "fallback_count": sum(row["fallback_reason"] is not None for row in rows),
        "out_of_domain_count": sum(
            str(row["fallback_reason"]).startswith("out_of_domain") for row in rows
        ),
        "fixed_policy_aggregate_latency_ms": {
            "always_full_replay": full_total,
            "always_cold_handoff_when_valid": total_for("cold_handoff_build"),
            "always_warm_cache_when_valid": total_for("warm_cached_suffix"),
            "oracle_fastest_valid": oracle_total,
        },
    }
    gates = {
        "oracle_agreement_at_least_90pct": metrics["exact_route_selection_accuracy"]
        >= 0.90,
        "aggregate_latency_within_2pct_of_oracle": metrics["aggregate_regret_fraction"]
        <= 0.02,
        "router_overhead_below_1pct": metrics[
            "router_overhead_fraction_of_selected_median"
        ]
        < 0.01,
        "router_overhead_preferably_below_0_1ms": overhead_median < 0.1,
        "adaptive_beats_full_replay": adaptive_total < full_total,
    }
    return {
        "schema_version": 1,
        "policy_version": policy.get("policy_version"),
        "held_out_case_ids": test_ids,
        "metrics": metrics,
        "acceptance_gates": gates,
        "acceptance_pass": all(gates.values()),
        "cases": rows,
    }


def execute_adaptive_cases(
    policy: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    route_executor: Callable[[Mapping[str, Any], str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select then actually execute one route per case through a supplied harness."""
    router = ConservativeLatencyRouter(policy)
    results = []
    for case in cases:
        inputs = router_inputs_from_normalized_case(
            case,
            cache_available=True,
            cache_state="warm",
            cache_constructible=True,
            compatible=True,
        )
        started = time.perf_counter_ns()
        decision = router.decide(inputs)
        overhead_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        actual = dict(route_executor(case, decision.selected_route))
        actual_route = str(actual.get("route", ""))
        if actual_route != decision.selected_route:
            raise RuntimeError("adaptive executor did not run the selected route")
        if int(actual.get("fallback_count", 0)) or int(
            actual.get("recovery_replay_count", 0)
        ):
            raise RuntimeError("adaptive execution recovered or fell back")
        results.append(
            {
                "case_id": case["case_id"],
                "policy_decision": decision.to_dict(),
                "selected_route": decision.selected_route,
                "actual_route": actual_route,
                "actual_latency_ms": float(
                    actual["timing"]["production_route_total_cuda_ms"]
                ),
                "output_hash": actual["output_hash"],
                "fallback_count": int(actual.get("fallback_count", 0)),
                "recovery_replay_count": int(actual.get("recovery_replay_count", 0)),
                "router_decision_overhead_ms": overhead_ms,
            }
        )
    return results


def write_csv(path: Path, summary: Mapping[str, Any]) -> None:
    fields = (
        "case_id",
        "selected_route",
        "actual_route",
        "oracle_route",
        "selected_latency_ms",
        "oracle_latency_ms",
        "regret_ms",
        "regret_fraction",
        "full_replay_latency_ms",
        "cold_handoff_latency_ms",
        "warm_cache_latency_ms",
        "output_hash",
        "fallback_reason",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row in summary["cases"]:
            writer.writerow({field: row.get(field) for field in fields})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-json", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--decision-repetitions", type=int, default=100)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    policy = json.loads(args.policy_json.read_text(encoding="utf-8"))
    summary = evaluate_policy(
        policy,
        read_jsonl(args.input_jsonl),
        decision_repetitions=args.decision_repetitions,
    )
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(args.output_csv, summary)
    if not summary["acceptance_pass"]:
        raise SystemExit("router acceptance gates failed; artifacts were preserved")


if __name__ == "__main__":
    main()
