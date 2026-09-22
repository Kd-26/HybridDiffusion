import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "eval/scripts/cluster3_efficiency_profile_report.py"
SPEC = importlib.util.spec_from_file_location(
    "cluster3_efficiency_profile_report", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

PHASES = (
    "request_setup",
    "identity_token_hash",
    "identity_position_hash",
    "dependency_validation",
    "region_cache_lookup",
    "region_mask_build",
    "flashinfer_plan_build",
    "prefix_snapshot",
    "cache_commit",
    "kv_restore",
    "gdn_restore",
    "recovery_replay",
    "attention_forward",
    "gdn_forward",
    "mlp_forward",
    "active_suffix_forward",
    "verification",
    "scheduler_postprocess",
    "model_forward_total",
    "request_total",
)
CUDA_LEAVES = {
    "request_setup": 0.5,
    "region_mask_build": 1.0,
    "flashinfer_plan_build": 0.5,
    "prefix_snapshot": 0.5,
    "kv_restore": 0.5,
    "gdn_restore": 0.5,
    "attention_forward": 2.0,
    "gdn_forward": 3.0,
    "mlp_forward": 1.0,
}


def snapshot(route, repetition, *, debug=False):
    phases = {}
    for phase in PHASES:
        cuda = CUDA_LEAVES.get(phase)
        host = 0.1
        if phase == "active_suffix_forward":
            cuda, host = 10.0, 10.5
        elif phase == "model_forward_total":
            cuda, host = 7.0, 7.5
        elif phase == "request_total":
            cuda, host = None, 12.0
        phases[phase] = {
            "calls": 0 if phase == "recovery_replay" else 1,
            "host_ms": host,
            "cuda_ms": cuda,
            "timing_domain": "cuda" if cuda is not None else "host",
            "timing_scope": "request_exclusive",
            "envelope": phase in MODULE.ENVELOPES,
            "available": True,
            "availability_reason": (
                "not_executed_for_route" if phase == "recovery_replay" else None
            ),
        }
    return {
        "schema_version": 2,
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
        },
        "phases": phases,
        "counters": {
            "recovery_replay_count": 0,
            "snapshot_count": 2 if route == "warm_cached_suffix" else 1,
        },
        "synchronization_count": 1,
        "synchronizations": [{"classification": "final_timing"}],
        "peak_allocated_bytes": 1024,
        "peak_reserved_bytes": 2048,
    }


def record():
    profiles = {
        route: [snapshot(route, index) for index in range(10)]
        for route in MODULE.ROUTES
    }
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
        "component_timings_ms": {},
        "request_phase_profiles": profiles,
        "warm_snapshot_behavior": {
            "calls": 20,
            "expected_zero": False,
            "pass": True,
            "reason": "conservative replay republishes frontiers",
        },
    }


def test_summary_separates_domains_and_uses_per_repetition_residuals():
    summary = MODULE.summarize(record())
    assert (
        summary["route_totals"]["warm_cached_suffix"]["request_wall_ms"]["median"]
        == 12.0
    )
    assert summary["profiling_coverage"]["warm_residual_ms"] == 0.5
    assert summary["profiling_coverage"]["warm_residual_percent"] == 5.0
    assert summary["profiling_coverage"]["coverage_pass"] is True
    assert all(
        row["timing_domain"] == "cuda"
        for row in summary["gpu_model_critical_path_decomposition"]
    )
    assert all(
        row["timing_domain"] == "host"
        for row in summary["host_orchestration_decomposition"]
    )
    assert "not summed" in MODULE.markdown(summary)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("old", "missing request_phase_profiles"),
        ("few", "exactly ten"),
        ("debug", "debug synchronization"),
        ("scope", "request-exclusive"),
        ("sync", "one-sync"),
        ("recovery", "recovery replay"),
        ("negative", "finite and nonnegative"),
    ),
)
def test_report_fails_closed_on_invalid_phase_evidence(mutation, message):
    value = record()
    if mutation == "old":
        del value["request_phase_profiles"]
    elif mutation == "few":
        value["request_phase_profiles"]["full_replay"].pop()
    elif mutation == "debug":
        value["request_phase_profiles"]["warm_cached_suffix"][0]["debug_sync"] = True
    elif mutation == "scope":
        value["request_phase_profiles"]["warm_cached_suffix"][0][
            "timing_scope"
        ] = "shared_batch"
    elif mutation == "sync":
        value["request_phase_profiles"]["warm_cached_suffix"][0][
            "synchronization_count"
        ] = 2
    elif mutation == "recovery":
        value["request_phase_profiles"]["warm_cached_suffix"][0]["counters"][
            "recovery_replay_count"
        ] = 1
    elif mutation == "negative":
        value["request_phase_profiles"]["warm_cached_suffix"][0]["phases"][
            "gdn_forward"
        ]["cuda_ms"] = -1.0
    with pytest.raises(RuntimeError, match=message):
        MODULE.summarize(value)


def test_residual_is_computed_per_repetition_before_summary():
    value = record()
    warm = value["request_phase_profiles"]["warm_cached_suffix"]
    warm[0]["phases"]["active_suffix_forward"]["cuda_ms"] = 20.0
    warm[0]["phases"]["gdn_forward"]["cuda_ms"] = 13.0
    summary = MODULE.summarize(value)
    assert summary["per_repetition_residuals"]["cuda_ms"]["samples"][0] == 0.5
