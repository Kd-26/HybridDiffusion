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


PROFILING = load(
    "cluster3_profiling_schema",
    ROOT / "eval/sglang/srt/dllm/region/profiling.py",
)
MODULE = load(
    "cluster3_efficiency_profile_report",
    ROOT / "eval/scripts/cluster3_efficiency_profile_report.py",
)

CUDA_VALUES = {
    "active_suffix_forward": 100.0,
    "suffix_prepare_total": 20.0,
    "model_forward_total": 60.0,
    "suffix_finalize_total": 15.0,
    "token_hidden_input_preparation": 2.0,
    "decoder_layer_total": 52.0,
    "final_normalization": 2.0,
    "lm_head_projection": 3.0,
    # Nested model phases are deliberately large enough to catch accidental
    # parent-plus-child double counting.
    "attention_block_total": 18.0,
    "gdn_block_total": 26.0,
    "mlp_forward": 12.0,
    "attention_forward": 8.0,
    "gdn_forward": 10.0,
    "trace_materialization": 14.0,
}


def snapshot(route, repetition, *, debug=False):
    phases = {}
    for phase in PROFILING.PROFILE_PHASES:
        cuda = CUDA_VALUES.get(phase)
        calls = 1
        if phase == "request_total":
            calls = MODULE.EXPECTED_REQUEST_TOTAL_CALLS[route]
        elif phase in (
            "active_suffix_forward",
            "suffix_prepare_total",
            "model_forward_total",
            "suffix_finalize_total",
        ):
            calls = 4
        elif phase in (
            "token_hidden_input_preparation",
            "final_normalization",
            "lm_head_projection",
        ):
            calls = 4
        elif phase == "decoder_layer_total":
            calls = 96
        elif phase in ("attention_block_total", "attention_forward"):
            calls = 24
        elif phase in ("gdn_block_total", "gdn_forward"):
            calls = 72
        elif phase == "mlp_forward":
            calls = 96
        elif phase == "prefix_snapshot":
            calls = 144
        elif phase == "recovery_replay":
            calls = 0
        phases[phase] = {
            "calls": calls,
            "host_ms": 120.0 if phase == "request_total" else 0.1,
            "cuda_ms": cuda,
            "timing_domain": "cuda" if cuda is not None else "host",
            "timing_scope": "request_exclusive",
            "envelope": phase in PROFILING.ENVELOPE_PHASES,
            "parent_phase": PROFILING.PHASE_PARENTS.get(phase),
            "hierarchy_kind": PROFILING.PHASE_HIERARCHY[phase]["kind"],
            "available": True,
            "availability_reason": (
                "not_executed_for_route" if phase == "recovery_replay" else None
            ),
        }
    return {
        "schema_version": 3,
        "finalized": True,
        "request_id": f"case:{route}:{repetition}",
        "timing_scope": "request_exclusive",
        "debug_sync": debug,
        "metadata": {
            "route": route,
            "repetition_index": repetition,
            "case_id": "efficiency-one",
            "prefix_length": 2048,
            "active_length": 64,
            "diffusion_steps": 4,
            "batch_size": 1,
            "cache_status": "hit" if route == "warm_cached_suffix" else "miss",
            "request_total_semantics": "segmented_route_wall_aggregate",
            "expected_request_total_calls": MODULE.EXPECTED_REQUEST_TOTAL_CALLS[route],
        },
        "hierarchy": json.loads(json.dumps(PROFILING.PHASE_HIERARCHY)),
        "phases": phases,
        "counters": {"recovery_replay_count": 0, "snapshot_count": 2},
        "synchronization_count": 1,
        "synchronizations": [{"classification": "final_timing"}],
        "peak_allocated_bytes": 1024,
        "peak_reserved_bytes": 2048,
    }


def record():
    return {
        "profile": "efficiency_one",
        "case_pass": True,
        "dtype": "bfloat16",
        "tp_size": 1,
        "case_id": "efficiency-one",
        "revision": "revision",
        "timing_protocol": {
            "warmup_repetitions": 3,
            "timed_repetitions": 10,
            "debug_sync_stages": False,
        },
        "request_phase_profiles": {
            route: [snapshot(route, index) for index in range(10)]
            for route in MODULE.ROUTES
        },
        "warm_snapshot_behavior": {
            "calls": 20,
            "expected_zero": False,
            "pass": True,
            "reason": "conservative replay republishes frontiers",
        },
    }


def test_schema_v3_hierarchical_coverage_and_serialization():
    summary = MODULE.summarize(record())
    assert summary["schema_version"] == 3
    assert summary["coverage"]["outer_suffix_percent"] == 95.0
    assert summary["coverage"]["model_forward_percent"] == pytest.approx(98.3333333333)
    assert summary["outer_suffix_coverage_percent"] == 95.0
    assert summary["model_forward_coverage_percent"] == pytest.approx(98.3333333333)
    assert summary["outer_suffix_residual_ms"]["median"] == 5.0
    assert summary["model_forward_residual_ms"]["median"] == 1.0
    assert [row["phase"] for row in summary["outer_suffix_decomposition"]] == list(
        MODULE.OUTER_SUFFIX_CHILDREN
    )
    assert [row["phase"] for row in summary["model_forward_decomposition"]] == list(
        MODULE.MODEL_FORWARD_CHILDREN
    )
    assert [
        row["phase"] for row in summary["suffix_prepare_decomposition"]["host"]
    ] == list(MODULE.SUFFIX_PREPARE_HOST_CHILDREN)
    assert [row["phase"] for row in summary["decoder_layer_decomposition"]] == list(
        MODULE.DECODER_LAYER_CHILDREN
    )
    json.dumps(summary)


def test_parent_and_nested_children_are_not_double_counted():
    value = record()
    warm = value["request_phase_profiles"]["warm_cached_suffix"]
    for snapshot_value in warm:
        snapshot_value["phases"]["attention_forward"]["cuda_ms"] = 1000.0
        snapshot_value["phases"]["gdn_forward"]["cuda_ms"] = 1000.0
    summary = MODULE.summarize(value)
    assert summary["coverage"]["outer_suffix_percent"] == 95.0
    assert summary["coverage"]["model_forward_percent"] == pytest.approx(98.3333333333)


def test_report_rejects_hierarchy_and_snapshot_contract_mismatches():
    value = record()
    warm = value["request_phase_profiles"]["warm_cached_suffix"]
    warm[0]["hierarchy"]["decoder_layer_total"] = {
        "parent": "active_suffix_forward",
        "kind": "envelope",
    }
    with pytest.raises(RuntimeError, match="hierarchy definition"):
        MODULE.summarize(value)

    value = record()
    value["warm_snapshot_behavior"]["calls"] += 1
    with pytest.raises(RuntimeError, match="snapshot behavior call count"):
        MODULE.summarize(value)


def test_residual_is_calculated_per_repetition_and_not_clamped():
    value = record()
    warm = value["request_phase_profiles"]["warm_cached_suffix"]
    warm[0]["phases"]["active_suffix_forward"]["cuda_ms"] = 101.0
    coverage = MODULE._coverage(
        warm, "active_suffix_forward", MODULE.OUTER_SUFFIX_CHILDREN, seed=1
    )
    assert coverage["residual_ms"]["samples"][0] == 6.0
    warm[0]["phases"]["active_suffix_forward"]["cuda_ms"] = 90.0
    coverage = MODULE._coverage(
        warm, "active_suffix_forward", MODULE.OUTER_SUFFIX_CHILDREN, seed=1
    )
    assert coverage["residual_ms"]["samples"][0] == -5.0
    assert coverage["overlap_violation"][0] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("old", "schema-v3"),
        ("few", "exactly ten"),
        ("debug", "debug synchronization"),
        ("scope", "request-exclusive"),
        ("sync", "one-sync"),
        ("calls", "request-total call count"),
        ("missing", "missing required phase"),
        ("reason", "lacks a reason"),
        ("recovery", "recovery replay"),
        ("negative", "finite and nonnegative"),
    ),
)
def test_report_fails_closed_on_invalid_evidence(mutation, message):
    value = record()
    warm = value["request_phase_profiles"]["warm_cached_suffix"]
    if mutation == "old":
        del value["request_phase_profiles"]
    elif mutation == "few":
        value["request_phase_profiles"]["full_replay"].pop()
    elif mutation == "debug":
        warm[0]["debug_sync"] = True
    elif mutation == "scope":
        warm[0]["timing_scope"] = "shared_batch"
    elif mutation == "sync":
        warm[0]["synchronization_count"] = 2
    elif mutation == "calls":
        warm[0]["phases"]["request_total"]["calls"] = 1
    elif mutation == "missing":
        del warm[0]["phases"]["suffix_finalize_total"]
    elif mutation == "reason":
        warm[0]["phases"]["recovery_replay"]["available"] = False
        warm[0]["phases"]["recovery_replay"]["availability_reason"] = None
    elif mutation == "recovery":
        warm[0]["counters"]["recovery_replay_count"] = 1
    elif mutation == "negative":
        warm[0]["phases"]["model_forward_total"]["cuda_ms"] = -1.0
    with pytest.raises(RuntimeError, match=message):
        MODULE.summarize(value)


@pytest.mark.parametrize("target", ("outer", "model"))
def test_cuda_coverage_gates_fail_closed_below_ninety_percent(target):
    value = record()
    warm = value["request_phase_profiles"]["warm_cached_suffix"]
    phase = "suffix_finalize_total" if target == "outer" else "decoder_layer_total"
    for snapshot_value in warm:
        snapshot_value["phases"][phase]["cuda_ms"] = 0.0
    with pytest.raises(RuntimeError, match="coverage below 90%"):
        MODULE.summarize(value)


def test_host_and_cuda_domains_remain_separate():
    summary = MODULE.summarize(record())
    assert all(
        row["timing_domain"] == "host"
        for row in summary["host_orchestration_decomposition"]
    )
    assert (
        "CUDA durations are excluded"
        in summary["host_orchestration_coverage"]["interpretation"]
    )
