import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "eval/scripts/cluster2_active_only_validation.py"
SPEC = importlib.util.spec_from_file_location("cluster2_active_only_validation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_case(prefix=64, active=32, steps=2, batch=2, profile="smoke16"):
    return MODULE.ValidationCase(
        case_id=f"case-p{prefix}-a{active}-s{steps}-b{batch}",
        profile=profile,
        token_seed=MODULE.DEFAULT_SEED,
        prefix_length=prefix,
        active_length=active,
        diffusion_steps=steps,
        batch_size=batch,
    )


def make_record(case=None):
    case = case or make_case()
    expected = case.expected_active_tokens
    positions = list(range(case.prefix_length, case.prefix_length + case.active_length))
    record = {
        "schema_version": MODULE.SCHEMA_VERSION,
        "case_id": case.case_id,
        "profile": case.profile,
        "revision": "revision",
        "model_scale": "qwen3_5-4b",
        "dtype": "bfloat16",
        "tp_size": 1,
        "batch_size": case.batch_size,
        "prefix_tokens_per_request": case.prefix_length,
        "active_tokens_per_request": case.active_length,
        "diffusion_steps": case.diffusion_steps,
        "scheduled_input_tokens": expected,
        "expected_active_tokens": expected,
        "stable_query_tokens": 0,
        "attention_query_rows_per_layer": [expected, expected],
        "total_kv_tokens_per_request": [case.total_tokens_per_request]
        * case.batch_size,
        "mlp_rows_per_layer": [expected] * 4,
        "gdn_rows_per_layer": [expected, expected],
        "active_position_ids_per_request": [
            MODULE._position_evidence(positions) for _ in range(case.batch_size)
        ],
        "positions_preserved": True,
        "stable_kv_unchanged": True,
        "stable_gdn_unchanged": True,
        "max_logits_error": 0.0,
        "max_hidden_error": 0.0,
        "max_gdn_error": 0.0,
        "top1_identical": True,
        "fallback_count": 0,
        "recovery_replays": 0,
        "nan_or_inf_detected": False,
        "stage_latency_ms": {"paired_case_total": 1.0},
        "peak_memory_bytes": 1024,
        "unavailable_reasons": {},
        "case_pass": True,
        "failure_reasons": [],
    }
    if case.prefix_length == 0:
        record["stable_kv_unchanged"] = None
        record["stable_gdn_unchanged"] = None
        record["unavailable_reasons"] = {
            "stable_kv_unchanged": "no stable prefix exists",
            "stable_gdn_unchanged": "no stable prefix state exists",
        }
    return record


def test_cli_help_without_cuda_or_model_loading():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--profile {one1,smoke16,paper100}" in completed.stdout
    assert "--debug-sync-stages" in completed.stdout


@pytest.mark.parametrize(
    ("profile", "count"), (("one1", 1), ("smoke16", 16), ("paper100", 100))
)
def test_manifests_are_deterministic_and_have_required_coverage(profile, count):
    first = MODULE.build_manifest(profile, MODULE.DEFAULT_SEED)
    second = MODULE.build_manifest(profile, MODULE.DEFAULT_SEED)
    assert first == second
    assert len(first) == count
    assert len({case.case_id for case in first}) == count
    coverage = MODULE.manifest_coverage(first)
    required = MODULE.required_coverage(profile)
    for dimension, expected in required.items():
        assert coverage[dimension] == expected
    assert all(
        case.batch_size
        * (case.prefix_length + case.active_length * case.diffusion_steps)
        <= 8192
        for case in first
    )


def test_paper_manifest_is_stratified_not_cartesian():
    cases = MODULE.build_manifest("paper100")
    assert len(cases) == 100
    assert len(cases) < 9 * 5 * 3 * 3
    assert all(
        not (case.batch_size == 4 and case.prefix_length >= 512) for case in cases
    )


def test_valid_synthetic_record_is_accepted():
    case = make_case()
    assert MODULE.validate_case_record(make_record(case), case) == []


def test_zero_prefix_has_explicit_inapplicable_stable_state_evidence():
    case = make_case(prefix=0)
    assert MODULE.validate_case_record(make_record(case), case) == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda r: r.update(stable_query_tokens=1), "stable query"),
        (lambda r: r.update(scheduled_input_tokens=1), "submitted input"),
        (lambda r: r.update(attention_query_rows_per_layer=[1]), "attention row"),
        (lambda r: r.update(mlp_rows_per_layer=[1]), "MLP row"),
        (lambda r: r.update(gdn_rows_per_layer=[1]), "GDN row"),
        (lambda r: r.update(total_kv_tokens_per_request=[1, 1]), "readable KV"),
        (lambda r: r.update(positions_preserved=False), "positions drifted"),
        (lambda r: r.update(stable_kv_unchanged=False), "stable KV"),
        (lambda r: r.update(stable_gdn_unchanged=False), "stable GDN"),
        (lambda r: r.update(max_logits_error=0.01), "BF16 tolerance"),
        (lambda r: r.update(max_hidden_error=float("nan")), "BF16 tolerance"),
        (lambda r: r.update(max_gdn_error=None), "unavailable"),
        (lambda r: r.update(top1_identical=False), "top-1"),
        (lambda r: r.update(fallback_count=1), "fallback"),
        (lambda r: r.update(recovery_replays=1), "recovery replay"),
        (lambda r: r.update(nan_or_inf_detected=True), "NaN/Inf"),
        (lambda r: r.update(attention_query_rows_per_layer=[]), "attention"),
    ],
)
def test_invalid_synthetic_records_are_rejected(mutation, message):
    case = make_case()
    record = make_record(case)
    mutation(record)
    assert any(
        message in reason for reason in MODULE.validate_case_record(record, case)
    )


def test_missing_required_trace_field_is_rejected():
    case = make_case()
    record = make_record(case)
    del record["gdn_rows_per_layer"]
    assert "missing required fields" in MODULE.validate_case_record(record, case)[0]


def test_missing_and_duplicate_records_fail_summary():
    cases = [make_case(batch=1), make_case(batch=2)]
    missing = MODULE.build_summary(
        cases,
        [make_record(cases[0])],
        revision="revision",
        hardware={},
        dtype="bfloat16",
        tp_size=1,
        profile="smoke16",
        runtime_diff_is_empty=True,
    )
    duplicate = MODULE.build_summary(
        cases,
        [make_record(cases[0]), make_record(cases[0]), make_record(cases[1])],
        revision="revision",
        hardware={},
        dtype="bfloat16",
        tp_size=1,
        profile="smoke16",
        runtime_diff_is_empty=True,
    )
    assert missing["checks"]["one_record_per_case"] is False
    assert duplicate["checks"]["one_record_per_case"] is False
    assert missing["strict_pass"] is False
    assert duplicate["strict_pass"] is False


def test_tp_size_other_than_one_is_rejected():
    with pytest.raises(ValueError, match="tp-size must be 1"):
        MODULE.parse_args(
            [
                "--model-path",
                "model",
                "--output-jsonl",
                "out.jsonl",
                "--summary-json",
                "summary.json",
                "--profile",
                "one1",
                "--tp-size",
                "2",
            ]
        )


class FakeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = torch.nn.Identity()
        self.mlp = torch.nn.Identity()


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([FakeLayer()])


def fake_runner():
    return type(
        "Runner",
        (),
        {
            "model": FakeModel(),
            "req_to_token_pool": type("Pool", (), {"mamba_map": {}})(),
        },
    )()


def test_scoped_hooks_removed_after_success():
    hooks = MODULE.ScopedRowHooks(fake_runner())
    assert hooks.handles
    with hooks:
        pass
    assert hooks.released


def test_scoped_hooks_removed_after_exception():
    hooks = MODULE.ScopedRowHooks(fake_runner())
    with pytest.raises(RuntimeError, match="primary"):
        with hooks:
            raise RuntimeError("primary")
    assert hooks.released


def _git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )


def test_prohibited_runtime_diff_detection(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "cluster2@example.invalid")
    _git(tmp_path, "config", "user.name", "Cluster 2 Test")
    allowed = tmp_path / "docs/cluster2_active_only_audit.md"
    protected = tmp_path / "eval/sglang/srt/runtime.py"
    allowed.parent.mkdir(parents=True)
    protected.parent.mkdir(parents=True)
    allowed.write_text("base\n", encoding="utf-8")
    protected.write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "base")
    base = _git(tmp_path, "rev-parse", "HEAD").stdout.strip()
    allowed.write_text("allowed\n", encoding="utf-8")
    assert MODULE.prohibited_runtime_diffs(tmp_path, base) == []
    protected.write_text("changed\n", encoding="utf-8")
    assert MODULE.prohibited_runtime_diffs(tmp_path, base) == [
        "eval/sglang/srt/runtime.py"
    ]


def test_summary_rejects_runtime_diff_and_bad_batch_coverage():
    cases = [make_case(batch=1)]
    summary = MODULE.build_summary(
        cases,
        [make_record(cases[0])],
        revision="revision",
        hardware={},
        dtype="bfloat16",
        tp_size=1,
        profile="smoke16",
        runtime_diff_is_empty=False,
        runtime_diff_files=["eval/sglang/srt/runtime.py"],
    )
    assert summary["checks"]["runtime_files_unchanged"] is False
    assert summary["checks"]["required_batch_coverage"] is False
    assert summary["strict_pass"] is False


def test_run_validation_writes_one_record_and_summary_without_cuda(
    tmp_path, monkeypatch
):
    class Runtime:
        def __init__(self, _args):
            self.closed = False

        def run_case(self, case):
            return make_record(case)

        def hardware(self):
            return {"device_name": "synthetic"}

        def close(self):
            self.closed = True

    monkeypatch.setattr(MODULE, "prohibited_runtime_diffs", lambda: [])
    args = MODULE.parse_args(
        [
            "--model-path",
            "unused",
            "--output-jsonl",
            str(tmp_path / "records.jsonl"),
            "--summary-json",
            str(tmp_path / "summary.json"),
            "--profile",
            "one1",
        ]
    )
    summary = MODULE.run_validation(args, runtime_factory=Runtime)
    records = (tmp_path / "records.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(records) == 1
    assert summary["strict_pass"] is True
    assert (
        json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))[
            "received_records"
        ]
        == 1
    )


def test_run_validation_preserves_primary_error_when_cleanup_fails(
    tmp_path, monkeypatch
):
    primary = RuntimeError("primary CUDA failure")

    class Runtime:
        def __init__(self, _args):
            pass

        def run_case(self, _case):
            raise primary

        def hardware(self):
            raise RuntimeError("hardware cleanup failure")

        def close(self):
            raise RuntimeError("process-group cleanup failure")

    monkeypatch.setattr(MODULE, "prohibited_runtime_diffs", lambda: [])
    args = MODULE.parse_args(
        [
            "--model-path",
            "unused",
            "--output-jsonl",
            str(tmp_path / "records.jsonl"),
            "--summary-json",
            str(tmp_path / "summary.json"),
            "--profile",
            "one1",
        ]
    )
    with pytest.raises(RuntimeError, match="primary CUDA failure") as caught:
        MODULE.run_validation(args, runtime_factory=Runtime)
    assert caught.value is primary
