import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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
        "model_scale": "2B",
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
        "attention_row_observation_point": MODULE.ATTENTION_ROW_OBSERVATION_POINT,
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


def make_prefix_runtime(canonical_locations=None):
    runtime = MODULE.Cluster2ValidationRuntime.__new__(MODULE.Cluster2ValidationRuntime)
    canonical = Mock(
        return_value=(
            canonical_locations
            if canonical_locations is not None
            else torch.tensor([11, 12], dtype=torch.int64)
        )
    )
    runtime.runtime = SimpleNamespace(_canonical_prefix_locations=canonical)
    runtime.model_runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros((4, 16), dtype=torch.int32)
        )
    )
    return runtime, canonical


def test_zero_prefix_reference_preserves_canonical_empty_without_lookup():
    runtime, canonical = make_prefix_runtime()
    empty = torch.empty((0,), dtype=torch.int64)
    req = SimpleNamespace(req_pool_idx=None, prefix_indices=empty)

    runtime._prepare_prefix_indices([req], 0)

    canonical.assert_not_called()
    assert req.prefix_indices is empty
    assert req.prefix_indices.dtype == torch.int64
    assert req.prefix_indices.ndim == 1
    assert req.prefix_indices.is_contiguous()
    assert (
        req.prefix_indices.device
        == runtime.model_runner.req_to_token_pool.req_to_token.device
    )
    assert req.prefix_indices.numel() == 0
    assert req.req_pool_idx is None


@pytest.mark.parametrize(
    "bad_prefix",
    [
        torch.empty((0,), dtype=torch.int32),
        torch.empty((1, 0), dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
    ],
)
def test_zero_prefix_rejects_noncanonical_empty_tensors(bad_prefix):
    runtime, canonical = make_prefix_runtime()
    req = SimpleNamespace(req_pool_idx=None, prefix_indices=bad_prefix)
    with pytest.raises(RuntimeError, match="invalid zero-prefix req.prefix_indices"):
        runtime._prepare_prefix_indices([req], 0)
    canonical.assert_not_called()
    assert req.req_pool_idx is None


def test_positive_prefix_requires_real_request_pool_slot():
    runtime, canonical = make_prefix_runtime()
    req = SimpleNamespace(
        req_pool_idx=None, prefix_indices=torch.empty((0,), dtype=torch.int64)
    )
    with pytest.raises(RuntimeError, match="no allocated req_pool_idx"):
        runtime._prepare_prefix_indices([req], 2)
    canonical.assert_not_called()
    assert req.req_pool_idx is None


def test_positive_prefix_reads_canonical_locations_from_real_slot():
    locations = torch.tensor([21, 22], dtype=torch.int64)
    runtime, canonical = make_prefix_runtime(locations)
    req = SimpleNamespace(
        req_pool_idx=3, prefix_indices=torch.empty((0,), dtype=torch.int64)
    )
    runtime._prepare_prefix_indices([req], 2)
    canonical.assert_called_once_with(3, 2)
    assert req.prefix_indices is locations
    assert req.req_pool_idx == 3


def make_entirely_active_trace(state_value):
    value = torch.tensor([[float(state_value)]])
    return {
        "logits": value.clone(),
        "top1": torch.tensor([0]),
        "hidden": {0: value.clone()},
        "gdn_states": {0: (value.clone(),)},
        "rows": {"attention": [1], "mlp": [1], "gdn": [1]},
        "positions": torch.tensor([0]),
        "scheduled_rows": 1,
        "kv_lengths": [1],
    }


def test_zero_prefix_candidate_uses_fresh_requests_without_state_or_restore():
    runtime, canonical = make_prefix_runtime()
    state = {"gdn": 99, "next_slot": 0}
    clear_calls = []
    created_requests = []

    def clear_pools():
        clear_calls.append(True)
        state["gdn"] = 0

    def make_reqs(_case, path, _prefixes):
        req = SimpleNamespace(
            rid=path,
            req_pool_idx=None,
            prefix_indices=torch.empty((0,), dtype=torch.int64),
        )
        created_requests.append(req)
        return [req]

    def run_active(reqs, values, _hooks):
        assert reqs[0].req_pool_idx is None
        state["next_slot"] += 1
        reqs[0].req_pool_idx = state["next_slot"]
        state["gdn"] += int(values[0][0])
        return make_entirely_active_trace(state["gdn"]), None, None

    runtime.runtime._clear_pools = Mock(side_effect=clear_pools)
    runtime._make_reqs = Mock(side_effect=make_reqs)
    runtime._run_active_batch = Mock(side_effect=run_active)
    runtime._seal_prefixes = Mock(side_effect=AssertionError("must not seal"))
    runtime._restore = Mock(side_effect=AssertionError("must not restore"))
    references = [make_entirely_active_trace(1), make_entirely_active_trace(1)]
    case = make_case(prefix=0, active=1, steps=2, batch=1)

    traces, comparisons, *stable_hashes = runtime._run_candidate_steps(
        case,
        prefixes=[[]],
        active_inputs=[[[1]], [[1]]],
        references=references,
        hooks=object(),
    )

    assert len(clear_calls) == case.diffusion_steps
    assert len(created_requests) == case.diffusion_steps
    assert created_requests[0] is not created_requests[1]
    assert [trace["gdn_states"][0][0].item() for trace in traces] == [1.0, 1.0]
    assert comparisons == [(0.0, 0.0, 0.0, True)] * case.diffusion_steps
    assert stable_hashes == [None, None, None, None]
    canonical.assert_not_called()
    runtime._seal_prefixes.assert_not_called()
    runtime._restore.assert_not_called()


def qwen35_runner(hidden_size=2048, layers=24, intermediate_size=6144, identity=True):
    text_config = SimpleNamespace(
        model_type="qwen3_5_text" if identity else "unknown",
        hidden_size=hidden_size,
        num_hidden_layers=layers,
        intermediate_size=intermediate_size,
    )
    hf_config = SimpleNamespace(
        model_type="qwen3_5" if identity else "unknown",
        architectures=["Qwen3_5DLLMForConditionalGeneration"] if identity else [],
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=text_config,
            hf_config=hf_config,
        )
    )


def test_qwen35_2b_checkpoint_scale_is_recorded_as_2b():
    assert MODULE.infer_model_scale(qwen35_runner()) == "2B"
    case = make_case()
    record = make_record(case)
    assert record["model_scale"] == "2B"
    assert MODULE.validate_case_record(record, case) == []


@pytest.mark.parametrize(
    "runner",
    [
        qwen35_runner(hidden_size=2560, layers=32, intermediate_size=9216),
        qwen35_runner(identity=False),
        SimpleNamespace(model_config=SimpleNamespace(hf_text_config=None)),
    ],
)
def test_unsupported_or_ambiguous_checkpoint_scale_fails_closed(runner):
    with pytest.raises(RuntimeError, match="checkpoint scale|hf_text_config"):
        MODULE.infer_model_scale(runner)


def test_trace_tensor_snapshot_is_detached_contiguous_cpu_copy():
    source = torch.arange(12.0, requires_grad=True).reshape(3, 4).transpose(0, 1)
    snapshot = MODULE._trace_tensor_to_cpu(source)

    assert snapshot.device.type == "cpu"
    assert snapshot.is_contiguous()
    assert snapshot.requires_grad is False
    assert snapshot.data_ptr() != source.data_ptr()
    expected = snapshot.clone()
    with torch.no_grad():
        source.add_(100)
    assert torch.equal(snapshot, expected)


def test_chunked_max_abs_preserves_fp32_definition(monkeypatch):
    monkeypatch.setattr(MODULE, "MAX_COMPARE_CHUNK_ELEMENTS", 2)
    left = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
    right = torch.tensor([0.0, 1.5, 2.0, -5.0, 4.0], dtype=torch.bfloat16)

    assert MODULE._max_abs(left, right) == 8.0


def test_chunked_max_abs_rejects_nonfinite_value_in_later_chunk(monkeypatch):
    monkeypatch.setattr(MODULE, "MAX_COMPARE_CHUNK_ELEMENTS", 2)
    left = torch.tensor([0.0, 1.0, 2.0, float("nan")])
    right = torch.zeros(4)

    with pytest.raises(RuntimeError, match="NaN or Inf"):
        MODULE._max_abs(left, right)


class RealisticAttentionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = torch.nn.Identity()
        self.mlp = torch.nn.Identity()

    def self_attention(self, hidden_states):
        return hidden_states


class LegacyAttentionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = torch.nn.Identity()
        self.mlp = torch.nn.Identity()


class FakeGDNLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_attn = torch.nn.Identity()
        self.mlp = torch.nn.Identity()


class UnclassifiedLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Identity()


class FakeModel(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layers = torch.nn.ModuleList([layer])


def fake_runner(layer=None):
    layer = layer or LegacyAttentionLayer()
    return type(
        "Runner",
        (),
        {
            "model": FakeModel(layer),
            "req_to_token_pool": type("Pool", (), {"mamba_map": {}})(),
        },
    )()


def assert_no_registered_hooks(layer):
    assert not layer._forward_hooks
    for name in ("qkv_proj", "self_attention", "linear_attn", "mlp"):
        module = getattr(layer, name, None)
        if isinstance(module, torch.nn.Module):
            assert not module._forward_pre_hooks


def test_real_qwen_attention_method_uses_qkv_projection_rows():
    layer = RealisticAttentionLayer()
    assert callable(layer.self_attention)
    assert not isinstance(layer.self_attention, torch.nn.Module)
    inputs = torch.zeros((13, 8))
    with MODULE.ScopedRowHooks(fake_runner(layer)) as hooks:
        with hooks.capture([]):
            layer.qkv_proj(inputs)
            layer.mlp(inputs)
        snapshot = hooks.snapshot()
        assert snapshot["rows"]["attention"] == [13]
        assert snapshot["rows"]["mlp"] == [13]
        assert snapshot["rows"]["gdn"] == []
    assert hooks.released
    assert_no_registered_hooks(layer)


def test_hook_snapshot_offloads_evidence_and_releases_source_references():
    layer = RealisticAttentionLayer()
    source_hidden = torch.arange(6.0).reshape(2, 3)
    source_gdn = torch.arange(4.0)
    with MODULE.ScopedRowHooks(fake_runner(layer)) as hooks:
        hooks.hidden = {0: source_hidden}
        hooks.gdn_states = {0: (source_gdn,)}
        snapshot = hooks.snapshot()

        assert snapshot["hidden"][0].device.type == "cpu"
        assert snapshot["gdn_states"][0][0].device.type == "cpu"
        assert snapshot["hidden"][0].data_ptr() != source_hidden.data_ptr()
        assert snapshot["gdn_states"][0][0].data_ptr() != source_gdn.data_ptr()
        assert hooks.hidden == {}
        assert hooks.gdn_states == {}
    assert_no_registered_hooks(layer)


def test_gdn_layer_records_linear_attention_rows():
    layer = FakeGDNLayer()
    inputs = torch.zeros((17, 8))
    with MODULE.ScopedRowHooks(fake_runner(layer)) as hooks:
        with hooks.capture([]):
            layer.linear_attn(inputs)
            layer.mlp(inputs)
        snapshot = hooks.snapshot()
        assert snapshot["rows"]["gdn"] == [17]
        assert snapshot["rows"]["mlp"] == [17]
        assert snapshot["rows"]["attention"] == []
    assert_no_registered_hooks(layer)


def test_legacy_module_self_attention_remains_supported():
    layer = LegacyAttentionLayer()
    inputs = torch.zeros((5, 8))
    with MODULE.ScopedRowHooks(fake_runner(layer)) as hooks:
        with hooks.capture([]):
            layer.self_attention(inputs)
        assert hooks.snapshot()["rows"]["attention"] == [5]
    assert_no_registered_hooks(layer)


def test_unclassified_layer_fails_closed_and_removes_partial_hooks():
    layer = UnclassifiedLayer()
    with pytest.raises(
        RuntimeError, match="layer 0 has no hookable full-attention projection"
    ):
        MODULE.ScopedRowHooks(fake_runner(layer))
    assert_no_registered_hooks(layer)


def test_scoped_hooks_removed_after_success():
    layer = LegacyAttentionLayer()
    hooks = MODULE.ScopedRowHooks(fake_runner(layer))
    assert hooks.handles
    with hooks:
        pass
    assert hooks.released
    assert_no_registered_hooks(layer)


def test_scoped_hooks_removed_after_exception():
    layer = LegacyAttentionLayer()
    hooks = MODULE.ScopedRowHooks(fake_runner(layer))
    with pytest.raises(RuntimeError, match="primary"):
        with hooks:
            with hooks.capture([]):
                layer.self_attention(torch.zeros((3, 8)))
                raise RuntimeError("primary")
    assert hooks.released
    assert_no_registered_hooks(layer)


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
