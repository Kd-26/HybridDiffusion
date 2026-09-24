import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ROUTER = load(
    "cluster3_latency_router_tested",
    ROOT / "eval/sglang/srt/dllm/region/latency_router.py",
)
FIT = load(
    "cluster3_latency_fit_tested",
    ROOT / "eval/scripts/cluster3_fit_latency_router.py",
)
EVALUATE = load(
    "cluster3_latency_evaluate_tested",
    ROOT / "eval/scripts/cluster3_evaluate_latency_router.py",
)


def policy(full=100.0, cold=60.0, warm=40.0, test_ids=()):
    features = list(ROUTER.DEFAULT_FEATURE_SCHEMA)

    def model(intercept):
        return {
            "intercept": intercept,
            "coefficients": [0.0] * len(features),
            "feature_means": [0.0] * len(features),
            "feature_scales": [1.0] * len(features),
        }

    return {
        "schema_version": 1,
        "policy_version": "test-v1",
        "model_provenance": "synthetic-test",
        "safety_margin": 0.02,
        "feature_schema": features,
        "route_models": {
            "full_replay": model(full),
            "cold_handoff_build": model(cold),
            "warm_cached_suffix": model(warm),
        },
        "support_domain": {feature: [0.0, 100000.0] for feature in features},
        "data_split": {
            "seed": 7,
            "split_unit": "normalized_case_id",
            "train_case_ids": [],
            "validation_case_ids": [],
            "test_case_ids": list(test_ids),
        },
    }


def inputs(**updates):
    value = {
        "total_tokens": 1024,
        "stable_tokens": 960,
        "active_tokens": 64,
        "stable_fraction": 0.9375,
        "diffusion_steps": 4,
        "active_region_count": 1,
        "batch_size": 1,
        "cache_available": True,
        "cache_state": "warm",
        "cache_constructible": False,
        "contract_compatible": True,
        "version_compatible": True,
        "position_hash_compatible": True,
        "parent_versions_compatible": True,
    }
    value.update(updates)
    return value


def normalized_case(case_id="case"):
    return {
        "schema_version": 1,
        "case_id": case_id,
        "total_tokens_per_request": 1024,
        "prefix_tokens": 960,
        "active_spans": [[960, 1024]],
        "diffusion_steps": 4,
        "batch_size": 1,
        "regions": [],
    }


def test_missing_policy_cache_and_incompatible_identity_fail_closed():
    missing_policy = ROUTER.ConservativeLatencyRouter(None)
    assert missing_policy.decide(inputs()).selected_route == "full_replay"
    assert missing_policy.decide(inputs()).fallback_reason == "policy_missing"

    router = ROUTER.ConservativeLatencyRouter(policy())
    missing_cache = router.decide(
        inputs(cache_available=False, cache_state="absent", cache_constructible=False)
    )
    assert missing_cache.selected_route == "full_replay"
    assert missing_cache.fallback_reason == "cache_unavailable"
    incompatible = router.decide(inputs(position_hash_compatible=False))
    assert incompatible.selected_route == "full_replay"
    assert incompatible.fallback_reason == "cache_incompatible"


@pytest.mark.parametrize(
    "changes",
    (
        {"active_tokens": float("nan")},
        {"diffusion_steps": float("inf")},
        {"total_tokens": 200000},
    ),
)
def test_missing_nonfinite_and_out_of_domain_features_fail_closed(changes):
    decision = ROUTER.ConservativeLatencyRouter(policy()).decide(inputs(**changes))
    assert decision.selected_route == "full_replay"
    assert decision.fallback_reason


def test_clearly_faster_warm_route_is_selected():
    decision = ROUTER.ConservativeLatencyRouter(policy()).decide(inputs())
    assert decision.selected_route == "warm_cached_suffix"
    assert decision.fallback_reason is None
    assert decision.predicted_latency_ms["warm_cached_suffix"] == 40.0


def test_marginal_cached_prediction_inside_margin_selects_full():
    decision = ROUTER.ConservativeLatencyRouter(policy(warm=99.0)).decide(inputs())
    assert decision.selected_route == "full_replay"
    assert decision.fallback_reason == "cached_prediction_inside_safety_margin"


def test_policy_json_round_trip(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy()))
    decision = ROUTER.ConservativeLatencyRouter.from_path(path).decide(inputs())
    assert decision.selected_route == "warm_cached_suffix"
    assert decision.policy_version == "test-v1"


def test_checked_in_example_policy_is_valid_but_labeled_example():
    path = ROOT / "eval/artifacts/cluster3_latency_router_policy.example.json"
    router = ROUTER.ConservativeLatencyRouter.from_path(path)
    decision = router.decide(inputs())
    assert decision.selected_route == "warm_cached_suffix"
    assert decision.model_provenance == "example-only-not-fitted-evidence"


def synthetic_fit_records(case_count=12):
    records = []
    for index in range(case_count):
        total = 512 + 64 * index
        active = 32 + 8 * (index % 4)
        case = {
            "schema_version": 1,
            "case_id": f"c{index}",
            "total_tokens_per_request": total,
            "prefix_tokens": total - active,
            "active_spans": [[total - active, total]],
            "diffusion_steps": 2 + index % 3,
            "batch_size": 1,
            "regions": [],
        }
        for route, factor in (
            ("full_replay", 1.0),
            ("cold_handoff_build", 0.65),
            ("warm_cached_suffix", 0.4),
        ):
            for repetition in range(2):
                records.append(
                    {
                        "case_id": case["case_id"],
                        "normalized_case": case,
                        "route": route,
                        "variant": "uninstrumented",
                        "benchmark_revision": "revision",
                        "timing": {
                            "production_route_total_cuda_ms": factor
                            * total
                            * case["diffusion_steps"]
                            / 10.0
                            + repetition * 0.01
                        },
                    }
                )
    return records


def test_fitting_is_deterministic_and_split_has_no_case_leakage():
    kwargs = {
        "source_hashes": {"synthetic": "abc"},
        "seed": 19,
        "fit_timestamp": "2026-09-24T00:00:00Z",
    }
    first = FIT.fit_policy(synthetic_fit_records(), **kwargs)
    second = FIT.fit_policy(synthetic_fit_records(), **kwargs)
    assert first == second
    split = first["data_split"]
    groups = [
        set(split["train_case_ids"]),
        set(split["validation_case_ids"]),
        set(split["test_case_ids"]),
    ]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert set.union(*groups) == {f"c{index}" for index in range(12)}


def evaluation_records(case_id="held-out"):
    case = normalized_case(case_id)
    records = []
    for route, latency in (
        ("full_replay", 100.0),
        ("warm_cached_suffix", 40.0),
    ):
        records.append(
            {
                "case_id": case_id,
                "normalized_case": case,
                "route": route,
                "variant": "uninstrumented",
                "output_hash": "a" * 64,
                "timing": {"production_route_total_cuda_ms": latency},
            }
        )
    return records


def test_oracle_fixed_policy_metrics_and_overhead_are_reported():
    result = EVALUATE.evaluate_policy(
        policy(test_ids=("held-out",)),
        evaluation_records(),
        decision_repetitions=2,
    )
    metrics = result["metrics"]
    assert metrics["exact_route_selection_accuracy"] == 1.0
    assert metrics["aggregate_latency_ms"] == 40.0
    assert metrics["aggregate_regret_ms"] == 0.0
    assert metrics["fixed_policy_aggregate_latency_ms"]["always_full_replay"] == 100.0
    assert metrics["fixed_policy_aggregate_latency_ms"]["oracle_fastest_valid"] == 40.0
    assert metrics["router_decision_overhead_ms_median"] >= 0.0


def test_paper_csv_exporter_writes_one_row_per_case(tmp_path):
    result = EVALUATE.evaluate_policy(
        policy(test_ids=("held-out",)),
        evaluation_records(),
        decision_repetitions=1,
    )
    destination = tmp_path / "paper.csv"
    EVALUATE.write_csv(destination, result)
    lines = destination.read_text().splitlines()
    assert len(lines) == 2
    assert "oracle_route" in lines[0]
    assert "warm_cached_suffix" in lines[1]


def test_adaptive_execution_invokes_selected_route_and_keeps_hash_gates():
    calls = []

    def execute(case, route):
        calls.append((case["case_id"], route))
        return {
            "route": route,
            "timing": {"production_route_total_cuda_ms": 40.0},
            "output_hash": "a" * 64,
            "fallback_count": 0,
            "recovery_replay_count": 0,
        }

    result = EVALUATE.execute_adaptive_cases(
        policy(), [normalized_case("adaptive")], execute
    )
    assert calls == [("adaptive", "warm_cached_suffix")]
    assert result[0]["selected_route"] == result[0]["actual_route"]
    assert result[0]["output_hash"] == "a" * 64
