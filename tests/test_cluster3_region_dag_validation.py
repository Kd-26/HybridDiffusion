import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).parents[1]
REGION = ROOT / "eval/sglang/srt/dllm/region"
SCRIPT = ROOT / "eval/scripts/cluster3_region_dag_validation.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load the pure Region-DAG modules without importing SGLang's optional runtime
# dependencies. The controlled GPU process imports the installed runtime normally.
load_module("sglang.srt.dllm.region.execution_spec", REGION / "execution_spec.py")
load_module("sglang.srt.dllm.region.dependency_graph", REGION / "dependency_graph.py")
load_module("sglang.srt.dllm.region.runtime", REGION / "runtime.py")
MODULE = load_module("cluster3_region_dag_validation", SCRIPT)


@pytest.fixture(autouse=True)
def install_consistent_region_modules():
    """Undo canonical-module replacement performed by sibling pure tests."""
    load_module("sglang.srt.dllm.region.execution_spec", REGION / "execution_spec.py")
    load_module(
        "sglang.srt.dllm.region.dependency_graph", REGION / "dependency_graph.py"
    )
    load_module("sglang.srt.dllm.region.runtime", REGION / "runtime.py")


def make_record(case=None):
    case = case or MODULE.build_manifest("one1")[0]
    tokens = list(range(case.sequence_length))
    spec = MODULE.build_execution_spec(case, tokens, edited=True)
    plan = MODULE.expected_plan(spec, case.edited_regions)
    samples = [float(index) for index in range(1, 11)]
    require_statistics = case.profile == "effectiveness"
    repetition_count = 10 if require_statistics else 1
    timings = {
        name: MODULE._timing_evidence(
            samples if require_statistics else samples[:1],
            seed=index,
            require_statistics=require_statistics,
        )
        for index, name in enumerate(
            (
                "reference_full_ms",
                "cached_total_ms",
                "canonical_frontier_establishment_ms",
                "warm_cached_suffix_ms",
                "full_attention_ms",
                "gdn_replay_ms",
                "mlp_forward_ms",
                "prefix_snapshot_ms",
                "mask_build_ms",
                "gather_scatter_ms",
                "cache_lookup_restore_ms",
            ),
            1,
        )
    }
    if plan.gdn_replay_start == 0:
        timings["cache_lookup_restore_ms"] = {
            "samples": [],
            "unavailable_reason": "full replay performs no restore",
        }
    active_positions = list(case.active_positions)
    return {
        "schema_version": MODULE.SCHEMA_VERSION,
        "case_id": case.case_id,
        "profile": case.profile,
        "revision": "revision",
        "model_scale": "2B",
        "dtype": "bfloat16",
        "tp_size": 1,
        "sequence_length": case.sequence_length,
        "region_contract": spec.to_dict(),
        "edited_regions": list(case.edited_regions),
        "expected_invalidation_closure": list(plan.logical_invalidation_regions),
        "observed_invalidation_closure": list(plan.logical_invalidation_regions),
        "expected_gdn_replay_start": plan.gdn_replay_start,
        "observed_gdn_replay_start": plan.gdn_replay_start,
        "original_active_positions": {
            "count": len(active_positions),
            "sha256": MODULE._sha256_json(active_positions),
            "preview": active_positions[:16],
        },
        "selected_mask_backend": "custom_paged",
        "mask_hash": "mask-hash",
        "mask_dimensions": [[plan.query_count, case.sequence_length]],
        "paired_tensor_hashes": [
            {
                "reference": {"hidden": "reference"},
                "cached": {"hidden": "cached"},
                "reused_reference_state": "same",
                "reused_cached_state": "same",
            }
        ],
        "max_logits_error": 0.0,
        "max_hidden_error": 0.0,
        "max_gdn_error": 0.0,
        "top1_identical": True,
        "reused_stable_hash_before": "stable",
        "reused_stable_hash_after": "stable",
        "negative_lookup_results": {
            "wrong_model_miss": True,
            "wrong_model_revision_miss": True,
            "wrong_adapter_miss": True,
            "wrong_adapter_revision_miss": True,
            "wrong_contract_miss": True,
            "wrong_request_generation_miss": True,
            "wrong_parent_version_miss": True,
        },
        "stale_state_reuse_count": 0,
        "fallback_count": 0,
        "recovery_replays": 0,
        "nan_or_inf_detected": False,
        "component_timings_ms": timings,
        "timing_protocol": {
            "warmup_repetitions": 1 if require_statistics else 0,
            "timed_repetitions": 10 if require_statistics else 1,
            "cuda_events": True,
            "synchronized_measurement_boundaries": True,
        },
        "work_counters": {
            "reference_full_attention_query_token_layer_positions": [100]
            * repetition_count,
            "cached_full_attention_query_token_layer_positions": [50]
            * repetition_count,
            "reference_gdn_token_layer_positions": [100] * repetition_count,
            "cached_gdn_replay_token_layer_positions": [50] * repetition_count,
            "kv_cache_hits": [plan.gdn_replay_start] * repetition_count,
            "kv_cache_misses": [plan.query_count] * repetition_count,
            "gdn_state_restores": [int(plan.gdn_replay_start > 0)] * repetition_count,
        },
        "positions_preserved": True,
        "attention_row_observation_point": (MODULE.ATTENTION_ROW_OBSERVATION_POINT),
        "peak_memory_bytes": 1024,
        "peak_memory_by_path_bytes": {
            "reference_full": [1024] * repetition_count,
            "cached": [768] * repetition_count,
        },
        "case_pass": True,
        "failure_reasons": [],
    }


def test_cli_help_does_not_load_cuda_or_model():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (
        "--profile {one1,smoke16,paper100,effectiveness,efficiency_one}"
        in completed.stdout
    )
    assert "--timed-repetitions" in completed.stdout
    assert "--debug-sync-stages" in completed.stdout


@pytest.mark.parametrize(
    ("profile", "count"),
    (("one1", 1), ("smoke16", 16), ("paper100", 100), ("effectiveness", 9)),
)
def test_manifests_are_deterministic_unique_and_bounded(profile, count):
    first = MODULE.build_manifest(profile)
    second = MODULE.build_manifest(profile)
    assert first == second
    assert len(first) == count
    assert len({case.case_id for case in first}) == count
    assert len({case.layout_identity for case in first}) == count
    assert all(case.sequence_length <= 1024 for case in first)


def test_one1_layout_and_logical_replay_distinction_are_exact():
    case = MODULE.build_manifest("one1")[0]
    assert [(r.region_id, r.start, r.end, r.status) for r in case.regions] == [
        ("A", 0, 64, "stable"),
        ("B", 64, 96, "active"),
        ("C", 96, 192, "stable"),
        ("D", 192, 224, "active"),
        ("E", 224, 256, "stable"),
    ]
    spec = MODULE.build_execution_spec(case, list(range(256)), edited=True)
    plan = MODULE.expected_plan(spec, case.edited_regions)
    assert plan.logical_invalidation_regions == ("B", "D")
    assert plan.gdn_replay_start == 64
    assert plan.replayed_but_logically_valid_regions == ("C", "E")
    versions = {region.region_id: region.region_version for region in spec.regions}
    assert versions == {"A": 0, "B": 1, "C": 0, "D": 1, "E": 0}
    region_d = next(region for region in spec.regions if region.region_id == "D")
    assert region_d.recorded_parent_versions == (("B", 1),)


def test_smoke_covers_all_sizes_shapes_placements_and_full_reduction():
    cases = MODULE.build_manifest("smoke16")
    assert {case.sequence_length for case in cases} == {256, 512, 1024}
    assert {
        sum(region.status == "active" for region in case.regions) for case in cases
    } == {1, 2, 4}
    assert {case.diffusion_steps for case in cases} == {2, 4, 8}
    full = cases[-1]
    assert full.regions == (MODULE.RegionShape("X0", 0, 256, "active"),)
    spec = MODULE.build_execution_spec(full, list(range(256)), edited=True)
    plan = MODULE.expected_plan(spec, full.edited_regions)
    assert plan.gdn_replay_start == 0
    assert plan.reused_positions == ()
    assert plan.gdn_replay_positions == tuple(range(256))


def test_effectiveness_manifest_has_requested_nine_shapes():
    cases = MODULE.build_manifest("effectiveness")
    shapes = {
        (
            sum(region.status == "active" for region in case.regions),
            sum(
                region.end - region.start
                for region in case.regions
                if region.status == "active"
            ),
        )
        for case in cases
    }
    assert shapes == {(1, 64), (2, 64), (4, 64)}
    assert all(case.sequence_length == 1024 for case in cases)


def test_efficiency_one_is_the_primary_long_prefix_profile():
    cases = MODULE.build_manifest("efficiency_one")
    assert len(cases) == 1
    case = cases[0]
    assert case.sequence_length == 2112
    assert case.diffusion_steps == 4
    assert case.regions == (
        MODULE.RegionShape("S0", 0, 2048, "stable"),
        MODULE.RegionShape("X0", 2048, 2112, "active", ("S0",)),
    )
    spec = MODULE.build_execution_spec(
        case, list(range(case.sequence_length)), edited=True
    )
    plan = MODULE.expected_plan(spec, case.edited_regions)
    assert plan.gdn_replay_start == 2048
    assert plan.query_count == 64


def test_efficiency_one_instantiates_separate_measured_route_profilers():
    source = inspect.getsource(MODULE.Cluster3ValidationRuntime.run_case)
    assert 'case.profile == "efficiency_one" and collect' in source
    assert "RequestScopedProfiler(" in source
    for route in (
        "full_replay",
        "cold_handoff_build",
        "warm_cached_suffix",
    ):
        assert f'"{route}": []' in source


def test_exception_cleanup_detaches_request_owned_profilers():
    runtime = object.__new__(MODULE.Cluster3ValidationRuntime)
    req = SimpleNamespace(region_dag_profilers=(object(),))
    runtime._profiled_requests = [req]
    with pytest.raises(RuntimeError, match="abort"):
        with runtime._profiling_request_cleanup():
            raise RuntimeError("abort")
    assert req.region_dag_profilers == ()
    assert runtime._profiled_requests == []


def test_unedited_independent_active_region_keeps_its_version():
    case = next(
        case
        for case in MODULE.build_manifest("effectiveness")
        if sum(region.status == "active" for region in case.regions) == 2
    )
    spec = MODULE.build_execution_spec(
        case, list(range(case.sequence_length)), edited=True
    )
    plan = MODULE.expected_plan(spec, case.edited_regions)
    active = [region for region in spec.regions if region.is_active]
    assert plan.logical_invalidation_regions == (active[0].region_id,)
    assert active[0].region_version == 1
    assert active[1].region_version == 0


def test_bootstrap_timing_statistics_are_deterministic_and_fail_closed():
    values = [float(index) for index in range(10)]
    first = MODULE._timing_evidence(values, seed=17, require_statistics=True)
    second = MODULE._timing_evidence(values, seed=17, require_statistics=True)
    assert first == second
    assert first["median"] == 4.5
    assert first["mad"] == 2.5
    assert first["bootstrap_ci_95"][0] <= 4.5 <= first["bootstrap_ci_95"][1]
    with pytest.raises(ValueError, match="at least ten"):
        MODULE._timing_evidence(values[:9], seed=17, require_statistics=True)


def test_compact_top1_rows_scatter_to_absolute_suffix_positions():
    edited_tokens = [0] * 256
    query_positions = tuple(range(145, 256))
    reference_top1 = list(range(1000, 1000 + len(query_positions)))

    MODULE._apply_reference_top1_at_absolute_positions(
        edited_tokens,
        (145, 200, 255),
        {"positions": query_positions, "top1": reference_top1},
    )

    assert edited_tokens[145] == 1000
    assert edited_tokens[200] == 1055
    assert edited_tokens[255] == 1110
    assert edited_tokens[144] == 0


def test_compact_top1_scatter_fails_closed_on_inconsistent_trace():
    with pytest.raises(RuntimeError, match="do not match"):
        MODULE._apply_reference_top1_at_absolute_positions(
            [0] * 8,
            (4,),
            {"positions": (4, 5), "top1": [7]},
        )
    with pytest.raises(RuntimeError, match="absent"):
        MODULE._apply_reference_top1_at_absolute_positions(
            [0] * 8,
            (3,),
            {"positions": (4, 5), "top1": [7, 8]},
        )


def test_complete_synthetic_record_passes_strict_validation():
    case = MODULE.build_manifest("one1")[0]
    assert MODULE.validate_case_record(make_record(case), case) == []


def test_timing_schema_requires_canonical_frontier_and_warm_suffix_metrics():
    case = MODULE.build_manifest("one1")[0]
    expected = {
        "reference_full_ms",
        "cached_total_ms",
        "canonical_frontier_establishment_ms",
        "warm_cached_suffix_ms",
        "full_attention_ms",
        "gdn_replay_ms",
        "mlp_forward_ms",
        "prefix_snapshot_ms",
        "mask_build_ms",
        "gather_scatter_ms",
        "cache_lookup_restore_ms",
    }
    record = make_record(case)
    assert set(record["component_timings_ms"]) == expected
    assert MODULE.validate_case_record(record, case) == []

    for metric in (
        "canonical_frontier_establishment_ms",
        "warm_cached_suffix_ms",
    ):
        missing = make_record(case)
        del missing["component_timings_ms"][metric]
        assert "component timing evidence is unavailable" in (
            MODULE.validate_case_record(missing, case)
        )


def test_entirely_active_record_has_no_cache_restore_or_fake_reuse():
    case = MODULE.build_manifest("smoke16")[-1]
    record = make_record(case)
    assert MODULE.validate_case_record(record, case) == []
    record["work_counters"]["gdn_state_restores"] = [1]
    assert any(
        "entirely-active" in reason
        for reason in MODULE.validate_case_record(record, case)
    )


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    (
        ("selected_mask_backend", "full_paged", "custom_paged"),
        ("max_logits_error", 1e-2, "tolerance"),
        ("max_hidden_error", float("nan"), "non-finite"),
        ("top1_identical", False, "top-1"),
        ("stale_state_reuse_count", 1, "stale"),
        ("fallback_count", 1, "fallback"),
        ("recovery_replays", 1, "recovery"),
        ("positions_preserved", False, "positions"),
        ("peak_memory_bytes", None, "peak-memory"),
    ),
)
def test_corrupt_or_missing_evidence_fails_closed(field, bad_value, message):
    case = MODULE.build_manifest("one1")[0]
    record = make_record(case)
    record[field] = bad_value
    assert any(
        message in reason for reason in MODULE.validate_case_record(record, case)
    )


def test_missing_field_and_negative_identity_failure_are_rejected():
    case = MODULE.build_manifest("one1")[0]
    missing = make_record(case)
    del missing["mask_hash"]
    assert "missing required fields" in MODULE.validate_case_record(missing, case)[0]
    negative = make_record(case)
    negative["negative_lookup_results"]["wrong_parent_version_miss"] = False
    assert any(
        "stale cache identities" in reason
        for reason in MODULE.validate_case_record(negative, case)
    )


def test_effectiveness_record_requires_ten_samples_and_robust_statistics():
    case = MODULE.build_manifest("effectiveness")[0]
    record = make_record(case)
    assert MODULE.validate_case_record(record, case) == []
    record["component_timings_ms"]["cached_total_ms"] = {"samples": [1.0]}
    reasons = MODULE.validate_case_record(record, case)
    assert any("fewer than ten" in reason for reason in reasons)
    assert any("robust timing statistics" in reason for reason in reasons)


class FakeEvent:
    def record(self):
        return None

    def elapsed_time(self, _other):
        return 1.25


def fake_timer_runtime():
    full = SimpleNamespace(forward_extend=lambda value: value + 1)
    gdn = SimpleNamespace(
        forward_extend=lambda value: value + 2,
        _restore_region_dag_layer_snapshot=lambda **_kwargs: "restored",
        _put_region_dag_layer_snapshot=lambda **_kwargs: "snapshotted",
    )
    mlp = SimpleNamespace(forward=lambda value: value + 3)
    language_model = SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)])
    trace_hooks = SimpleNamespace(_language_model=lambda _model: language_model)
    return SimpleNamespace(
        model_runner=SimpleNamespace(
            attn_backend=SimpleNamespace(full_attn_backend=full),
            model=object(),
        ),
        backend=gdn,
        cluster1=SimpleNamespace(ModelTraceHooks=trace_hooks),
    )


def test_backend_timing_probes_capture_calls_and_cleanup(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", lambda **_kwargs: FakeEvent())
    runtime = fake_timer_runtime()
    original_full = runtime.model_runner.attn_backend.full_attn_backend.forward_extend
    original_gdn = runtime.backend.forward_extend
    with MODULE.ScopedBackendTimers(runtime) as timers:
        assert (
            runtime.model_runner.attn_backend.full_attn_backend.forward_extend(1) == 2
        )
        assert runtime.backend.forward_extend(1) == 3
        assert runtime.backend._restore_region_dag_layer_snapshot() == "restored"
        assert runtime.backend._put_region_dag_layer_snapshot() == "snapshotted"
        assert runtime.model_runner.model is not None
        evidence = timers.snapshot()
        assert evidence["full_attention"] == 1.25
        assert evidence["gdn_replay"] == 1.25
        assert evidence["cache_lookup_restore"] == 1.25
        assert evidence["cache_lookup_restore_calls"] == 1
        assert evidence["prefix_snapshot"] == 1.25
    assert timers.released
    assert (
        runtime.model_runner.attn_backend.full_attn_backend.forward_extend
        is original_full
    )
    assert runtime.backend.forward_extend is original_gdn


def test_backend_timing_probes_cleanup_after_exception(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", lambda **_kwargs: FakeEvent())
    runtime = fake_timer_runtime()
    original = runtime.backend.forward_extend
    with pytest.raises(RuntimeError, match="primary"):
        with MODULE.ScopedBackendTimers(runtime) as timers:
            raise RuntimeError("primary")
    assert timers.released
    assert runtime.backend.forward_extend is original


class FakeRuntime:
    def __init__(self, args):
        self.args = args
        self.closed = False

    def run_case(self, case):
        return make_record(case)

    def hardware(self):
        return {"device_name": "unit-test GPU"}

    def close(self):
        self.closed = True


def validation_args(tmp_path, profile="one1"):
    return SimpleNamespace(
        profile=profile,
        seed=MODULE.DEFAULT_SEED,
        output_jsonl=str(tmp_path / "records.jsonl"),
        summary_json=str(tmp_path / "summary.json"),
    )


def test_export_writes_exactly_one_record_per_completed_case(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "_git_revision", lambda: "revision")
    args = validation_args(tmp_path)
    summary = MODULE.run_validation(args, runtime_factory=FakeRuntime)
    records = [
        json.loads(line) for line in Path(args.output_jsonl).read_text().splitlines()
    ]
    assert len(records) == 1
    assert records[0]["case_id"] == MODULE.build_manifest("one1")[0].case_id
    assert summary["strict_pass"]
    assert json.loads(Path(args.summary_json).read_text()) == summary


def test_summary_rejects_missing_and_duplicate_case_records():
    cases = MODULE.build_manifest("smoke16")[:2]
    record = make_record(cases[0])
    missing = MODULE.build_summary(
        cases, [record], revision="r", hardware={}, profile="smoke16"
    )
    duplicate = MODULE.build_summary(
        cases,
        [record, record],
        revision="r",
        hardware={},
        profile="smoke16",
    )
    assert not missing["checks"]["one_record_per_case"]
    assert not duplicate["checks"]["one_record_per_case"]
    assert not missing["strict_pass"]
    assert not duplicate["strict_pass"]


def test_primary_case_exception_survives_hardware_and_cleanup_errors(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(MODULE, "_git_revision", lambda: "revision")

    class BrokenRuntime(FakeRuntime):
        def run_case(self, _case):
            raise LookupError("primary CUDA failure")

        def hardware(self):
            raise RuntimeError("secondary hardware failure")

        def close(self):
            raise RuntimeError("secondary cleanup failure")

    with pytest.raises(LookupError, match="primary CUDA failure"):
        MODULE.run_validation(validation_args(tmp_path), runtime_factory=BrokenRuntime)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert not summary["strict_pass"]
    assert (tmp_path / "records.jsonl").read_text() == ""
