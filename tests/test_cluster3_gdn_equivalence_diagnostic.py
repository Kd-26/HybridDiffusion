import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "eval/scripts/cluster3_gdn_equivalence_diagnostic.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "cluster3_gdn_equivalence_diagnostic_test", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


def test_cli_help_does_not_load_model_or_cuda():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--output-dir" in completed.stdout
    assert "freshly segmented" in completed.stdout


def test_segmented_reference_freshly_recomputes_prefix_every_time():
    events = []
    generation = {"value": 0}

    def run_once():
        def clear_state():
            events.append("clear")

        def create_request():
            generation["value"] += 1
            request = f"request-{generation['value']}"
            events.append(("create", request))
            return request

        def recompute_prefix(request):
            events.append(("prefix", request))
            return "fresh-prefix"

        def prepare_suffix(request, prefix):
            events.append(("prepare", request, prefix))
            return type("Prepared", (), {"restore_required": False})()

        def execute_suffix(request, _prepared):
            events.append(("suffix", request))
            return type("Result", (), {"restore_calls": 0, "request": request})()

        return MODULE.execute_fresh_segmented_reference(
            clear_state=clear_state,
            create_request=create_request,
            recompute_prefix=recompute_prefix,
            region_cache_entry_count=lambda: 0,
            prepare_suffix=prepare_suffix,
            execute_suffix=execute_suffix,
        )

    first = run_once()
    second = run_once()
    assert first.request != second.request
    assert events.count("clear") == 2
    assert [
        event for event in events if isinstance(event, tuple) and event[0] == "prefix"
    ] == [
        ("prefix", "request-1"),
        ("prefix", "request-2"),
    ]


def test_segmented_reference_fails_if_prefix_publishes_region_cache_state():
    cache_entries = {"value": 0}
    suffix_called = False

    def recompute_prefix(_request):
        cache_entries["value"] = 1

    def execute_suffix(_request, _prepared):
        nonlocal suffix_called
        suffix_called = True

    with pytest.raises(RuntimeError, match="unexpectedly published"):
        MODULE.execute_fresh_segmented_reference(
            clear_state=lambda: cache_entries.update(value=0),
            create_request=lambda: object(),
            recompute_prefix=recompute_prefix,
            region_cache_entry_count=lambda: cache_entries["value"],
            prepare_suffix=lambda _request, _prefix: None,
            execute_suffix=execute_suffix,
        )
    assert not suffix_called


def test_segmented_reference_fails_on_restore_request_or_restore_call():
    prepared = type("Prepared", (), {"restore_required": True})()
    with pytest.raises(RuntimeError, match="requested a cache restoration"):
        MODULE.execute_fresh_segmented_reference(
            clear_state=lambda: None,
            create_request=lambda: object(),
            recompute_prefix=lambda _request: None,
            region_cache_entry_count=lambda: 0,
            prepare_suffix=lambda _request, _prefix: prepared,
            execute_suffix=lambda _request, _prepared: None,
        )

    prepared.restore_required = False
    restored = type("Result", (), {"restore_calls": 1})()
    with pytest.raises(RuntimeError, match="observed a cache restoration"):
        MODULE.execute_fresh_segmented_reference(
            clear_state=lambda: None,
            create_request=lambda: object(),
            recompute_prefix=lambda _request: None,
            region_cache_entry_count=lambda: 0,
            prepare_suffix=lambda _request, _prefix: prepared,
            execute_suffix=lambda _request, _prepared: restored,
        )


def test_tensor_diagnostic_records_required_numerical_evidence():
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    right = left.clone()
    right[1, 0] += torch.tensor(0.25, dtype=torch.bfloat16)
    evidence = MODULE._tensor_difference(left, right)
    assert evidence["reference"]["shape"] == [2, 2]
    assert evidence["reference"]["dtype"] == "torch.bfloat16"
    assert len(evidence["reference"]["sha256"]) == 64
    assert not evidence["exact_equal"]
    assert evidence["max_abs_error"] == 0.25
    assert evidence["mean_abs_error"] == 0.0625
    assert evidence["first_mismatching_flat_index"] == 2
    assert evidence["values_above_1e_2"] == 1
