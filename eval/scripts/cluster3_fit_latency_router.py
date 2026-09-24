#!/usr/bin/env python3
"""Fit an inspectable per-route ridge latency policy from benchmark JSONL."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np


ROUTER_PATH = Path(__file__).parents[1] / "sglang/srt/dllm/region/latency_router.py"
_ROUTER_SPEC = importlib.util.spec_from_file_location(
    "cluster3_fit_latency_router_policy", ROUTER_PATH
)
if _ROUTER_SPEC is None or _ROUTER_SPEC.loader is None:
    raise RuntimeError(f"cannot load latency router: {ROUTER_PATH}")
_ROUTER = importlib.util.module_from_spec(_ROUTER_SPEC)
sys.modules[_ROUTER_SPEC.name] = _ROUTER
_ROUTER_SPEC.loader.exec_module(_ROUTER)
DEFAULT_FEATURE_SCHEMA = _ROUTER.DEFAULT_FEATURE_SCHEMA
DEFAULT_SAFETY_MARGIN = _ROUTER.DEFAULT_SAFETY_MARGIN
POLICY_SCHEMA_VERSION = _ROUTER.POLICY_SCHEMA_VERSION
ROUTES = _ROUTER.ROUTES
feature_values = _ROUTER.feature_values
router_inputs_from_normalized_case = _ROUTER.router_inputs_from_normalized_case


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                records.append(dict(value))
    return records


def split_case_ids(case_ids: Sequence[str], *, seed: int) -> dict[str, list[str]]:
    unique = sorted(set(str(case_id) for case_id in case_ids))
    if not unique:
        raise ValueError("router fitting requires at least one case")
    rng = random.Random(seed)
    rng.shuffle(unique)
    if len(unique) == 1:
        return {"train": unique, "validation": [], "test": []}
    test_count = max(1, round(len(unique) * 0.15))
    validation_count = max(1, round(len(unique) * 0.15)) if len(unique) >= 3 else 0
    if test_count + validation_count >= len(unique):
        validation_count = max(0, len(unique) - test_count - 1)
    train_count = len(unique) - validation_count - test_count
    return {
        "train": sorted(unique[:train_count]),
        "validation": sorted(unique[train_count : train_count + validation_count]),
        "test": sorted(unique[train_count + validation_count :]),
    }


def aggregate_case_routes(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    cases: dict[str, Mapping[str, Any]] = {}
    revisions: dict[str, str] = {}
    for record in records:
        if record.get("variant") != "uninstrumented":
            continue
        case = record.get("normalized_case")
        case_id = str(record.get("case_id", ""))
        route = str(record.get("route", ""))
        if not case_id or route not in ROUTES or not isinstance(case, Mapping):
            raise ValueError("benchmark record lacks normalized case or valid route")
        if case_id in cases and dict(cases[case_id]) != dict(case):
            raise ValueError(f"case {case_id} has inconsistent normalized metadata")
        cases[case_id] = case
        revisions[case_id] = str(record.get("benchmark_revision", "unknown"))
        latency = float(record["timing"]["production_route_total_cuda_ms"])
        if not np.isfinite(latency) or latency <= 0:
            raise ValueError("training latency must be finite and positive")
        grouped[(case_id, route)].append(latency)
    result = {}
    for case_id, case in cases.items():
        route_latencies = {
            route: statistics.median(samples)
            for (group_case, route), samples in grouped.items()
            if group_case == case_id
        }
        if "full_replay" not in route_latencies:
            raise ValueError(f"case {case_id} has no full_replay measurement")
        inputs = router_inputs_from_normalized_case(
            case,
            cache_available=True,
            cache_state="warm",
            cache_constructible=True,
            compatible=True,
        )
        result[case_id] = {
            "case": dict(case),
            "features": feature_values(inputs),
            "latencies": route_latencies,
            "benchmark_revision": revisions[case_id],
        }
    if not result:
        raise ValueError("no uninstrumented benchmark records were found")
    return result


def _fit_route(
    rows: Sequence[Mapping[str, Any]],
    route: str,
    feature_schema: Sequence[str],
    ridge_alpha: float,
) -> dict[str, Any]:
    selected = [row for row in rows if route in row["latencies"]]
    if not selected:
        raise ValueError(f"training split contains no {route} measurements")
    matrix = np.asarray(
        [[row["features"][feature] for feature in feature_schema] for row in selected],
        dtype=np.float64,
    )
    targets = np.asarray(
        [row["latencies"][route] for row in selected], dtype=np.float64
    )
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales == 0.0] = 1.0
    normalized = (matrix - means) / scales
    design = np.column_stack((np.ones(len(normalized)), normalized))
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge_alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    predictions = design @ coefficients
    return {
        "intercept": float(coefficients[0]),
        "coefficients": [float(value) for value in coefficients[1:]],
        "feature_means": [float(value) for value in means],
        "feature_scales": [float(value) for value in scales],
        "ridge_alpha": float(ridge_alpha),
        "training_case_count": len(selected),
        "training_mae_ms": float(np.mean(np.abs(predictions - targets))),
    }


def fit_policy(
    records: Sequence[Mapping[str, Any]],
    *,
    source_hashes: Mapping[str, str],
    seed: int = 20260924,
    ridge_alpha: float = 1.0,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
    fit_timestamp: Optional[str] = None,
) -> dict[str, Any]:
    if ridge_alpha < 0 or not 0 <= safety_margin < 1:
        raise ValueError("ridge alpha and safety margin are invalid")
    aggregated = aggregate_case_routes(records)
    split = split_case_ids(tuple(aggregated), seed=seed)
    train_rows = [aggregated[case_id] for case_id in split["train"]]
    feature_schema = list(DEFAULT_FEATURE_SCHEMA)
    route_models = {
        route: _fit_route(train_rows, route, feature_schema, ridge_alpha)
        for route in ROUTES
        if all(route in row["latencies"] for row in train_rows)
    }
    support_domain = {
        feature: [
            min(float(row["features"][feature]) for row in train_rows),
            max(float(row["features"][feature]) for row in train_rows),
        ]
        for feature in feature_schema
    }
    revisions = sorted(
        {
            str(row["benchmark_revision"])
            for row in aggregated.values()
            if row["benchmark_revision"] != "unknown"
        }
    )
    timestamp = fit_timestamp or datetime.now(timezone.utc).isoformat()
    provenance_payload = {
        "source_jsonl_sha256": dict(sorted(source_hashes.items())),
        "train_case_ids": split["train"],
        "feature_schema": feature_schema,
        "ridge_alpha": ridge_alpha,
        "seed": seed,
    }
    provenance = hashlib.sha256(
        json.dumps(provenance_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "policy_version": f"cluster3-ridge-v1-{provenance[:12]}",
        "model_provenance": f"sha256:{provenance}",
        "safety_margin": float(safety_margin),
        "feature_schema": feature_schema,
        "route_models": route_models,
        "support_domain": support_domain,
        "data_split": {
            "seed": int(seed),
            "split_unit": "normalized_case_id",
            "train_case_ids": split["train"],
            "validation_case_ids": split["validation"],
            "test_case_ids": split["test"],
        },
        "source_jsonl_sha256": dict(sorted(source_hashes.items())),
        "fit_timestamp": timestamp,
        "benchmark_revision": revisions,
        "normalization": "per-route training mean and population standard deviation",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--output-policy", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--safety-margin", type=float, default=0.02)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    records = read_jsonl(args.input_jsonl)
    hashes = {str(path): _sha256_file(path) for path in args.input_jsonl}
    policy = fit_policy(
        records,
        source_hashes=hashes,
        seed=args.seed,
        ridge_alpha=args.ridge_alpha,
        safety_margin=args.safety_margin,
    )
    args.output_policy.parent.mkdir(parents=True, exist_ok=True)
    args.output_policy.write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
