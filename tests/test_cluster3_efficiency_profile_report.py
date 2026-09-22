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


def record():
    medians = {
        "reference_full_ms": 12.0,
        "cached_total_ms": 10.0,
        "canonical_frontier_establishment_ms": 20.0,
        "warm_cached_suffix_ms": 10.0,
        "full_attention_ms": 2.0,
        "gdn_replay_ms": 4.0,
        "mlp_forward_ms": 1.0,
        "prefix_snapshot_ms": 0.5,
        "mask_build_ms": 0.5,
        "gather_scatter_ms": 0.5,
        "cache_lookup_restore_ms": 0.5,
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
        "component_timings_ms": {
            name: {"median": value, "samples": [value] * 10}
            for name, value in medians.items()
        },
    }


def test_summary_ranks_additive_exclusive_overhead():
    summary = MODULE.summarize(record())
    assert summary["cold_handoff_build_ms"] == 30.0
    assert summary["warm_minus_full_replay_ms"] == -2.0
    rows = {row["phase"]: row for row in summary["ranking"]}
    assert rows["gdn_forward_exclusive"]["warm_ms"] == 3.0
    assert rows["unattributed_scheduler_and_model_work"]["warm_ms"] == 2.0
    assert sum(row["warm_ms"] for row in summary["ranking"]) == 10.0
    assert "No end-to-end speedup claim" in MODULE.markdown(summary)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("profile", "efficiency_one"),
        ("failed", "failed correctness"),
        ("sync", "debug-synchronized"),
        ("warmup", "three warmups"),
        ("samples", "ten samples"),
    ),
)
def test_report_fails_closed_on_invalid_evidence(mutation, message):
    value = record()
    if mutation == "profile":
        value["profile"] = "one1"
    elif mutation == "failed":
        value["case_pass"] = False
    elif mutation == "sync":
        value["timing_protocol"]["debug_sync_stages"] = True
    elif mutation == "warmup":
        value["timing_protocol"]["warmup_repetitions"] = 2
    elif mutation == "samples":
        value["component_timings_ms"]["full_attention_ms"]["samples"] = [1.0]
    with pytest.raises(RuntimeError, match=message):
        MODULE.summarize(value)
