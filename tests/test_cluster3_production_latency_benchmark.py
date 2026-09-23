import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VALIDATION = load(
    "cluster3_production_benchmark_validation",
    ROOT / "eval/scripts/cluster3_region_dag_validation.py",
)
MODULE = load(
    "cluster3_production_latency_benchmark_tested",
    ROOT / "eval/scripts/cluster3_production_latency_benchmark.py",
)


class FakeEvent:
    def __init__(self, cuda):
        self.cuda = cuda
        self.timestamp = None

    def record(self, stream):
        assert stream is self.cuda.stream
        self.timestamp = self.cuda.clock
        self.cuda.clock += 1.0

    def elapsed_time(self, other):
        return other.timestamp - self.timestamp


class FakeStream:
    def __init__(self):
        self.synchronizations = 0

    def synchronize(self):
        self.synchronizations += 1


class FakeCuda:
    def __init__(self):
        self.clock = 0.0
        self.stream = FakeStream()
        self.events = []

    def Event(self, **_kwargs):
        event = FakeEvent(self)
        self.events.append(event)
        return event

    def current_stream(self, device=None):
        assert device == 0
        return self.stream


def fake_torch():
    return SimpleNamespace(cuda=FakeCuda())


def raw_record(route, variant, repetition, *, output_hash="a" * 64, recovery=0):
    base = {
        "full_replay": 100.0,
        "cold_handoff_build": 60.0,
        "warm_cached_suffix": 50.0,
    }[route]
    profiled = variant == "minimally_profiled"
    total = base * (1.01 if profiled else 1.0)
    return {
        "schema_version": 1,
        "profile": "production_efficiency",
        "route": route,
        "variant": variant,
        "repetition_index": repetition,
        "timing": {
            "production_request_wall_host_ms": total + 1.0,
            "production_scheduler_host_ms": 2.0,
            "production_route_total_cuda_ms": total,
            "production_prepare_cuda_ms": 10.0 if profiled else None,
            "production_model_cuda_ms": total - 11.0 if profiled else None,
            "production_state_commit_cuda_ms": 0.0 if profiled else None,
            "production_state_commit_unavailable_reason": "inside model",
            "production_residual_cuda_ms": 1.0 if profiled else None,
            "synchronization_count": 1,
            "component_event_pairs": 8 if profiled else 0,
        },
        "output_hash": output_hash,
        "generated_top1_token_ids": [1, 2, 3, 4] * 64,
        "cache_status": "hit" if route == "warm_cached_suffix" else "miss",
        "cache_hits": 64 if route == "warm_cached_suffix" else 0,
        "cache_misses": 64,
        "gdn_state_restores": 72 if route == "warm_cached_suffix" else 0,
        "recovery_replay_count": recovery,
        "fallback_count": 0,
        "active_token_count": 64,
        "full_attention_query_token_layer_positions": 1536,
        "gdn_replay_token_layer_positions": 4608,
        "peak_allocated_gpu_memory_bytes": 1024,
        "workload_fingerprint": "same-work",
        "attention_contract": "region_dag_conservative_gdn_v1",
        "diffusion_steps": 4,
        "positions_preserved": True,
        "stable_queries_absent": True,
        "full_replay_processed_prefix": route == "full_replay",
        "warm_restored_valid_prefix": True,
        "trace_hooks_installed": False,
    }


def raw_records():
    return [
        raw_record(route, variant, repetition)
        for route in MODULE.ROUTES
        for repetition in range(10)
        for variant in MODULE.VARIANTS
    ]


def summarize(records=None):
    return MODULE.summarize(
        records or raw_records(),
        revision="revision",
        hardware={"device_name": "NVIDIA A30"},
        checkpoint={"model_scale": "2B"},
        correctness_prerequisite_pass=True,
    )


def test_parser_and_manifest_add_production_without_changing_efficiency_one():
    production = VALIDATION.parse_args(
        (
            "--model-path",
            "/model",
            "--output-jsonl",
            "/tmp/raw.jsonl",
            "--summary-json",
            "/tmp/summary.json",
            "--profile",
            "production_efficiency",
            "--correctness-artifact",
            "/tmp/correctness.json",
            "--preflight-json",
            "/tmp/preflight.json",
        )
    )
    assert production.profile == "production_efficiency"
    assert VALIDATION.build_manifest("production_efficiency")[0].sequence_length == 2112
    correctness_case = VALIDATION.build_manifest("efficiency_one")[0]
    assert correctness_case.profile == "efficiency_one"
    assert correctness_case.case_id.startswith("efficiency_one-")
    assert correctness_case.diffusion_steps == 4


def test_production_source_excludes_trace_and_full_validation_work():
    source = inspect.getsource(MODULE.execute_production_measurement)
    for forbidden in (
        "ScopedRowHooks",
        "ScopedBackendTimers",
        "hooks.snapshot",
        "_trace_tensor_to_cpu",
        "_compare(",
        "_reused_hash",
        ".isfinite(",
        "RuntimeLogEvidence",
    ):
        assert forbidden not in source
    assert source.index("timer.result()") < source.index(
        "compact_output_hash_after_timing"
    )
    assert "ScopedRowHooks" in inspect.getsource(
        VALIDATION.Cluster3ValidationRuntime.run_case
    )


def test_output_hashing_requires_finalized_timing():
    timer = MODULE.ProductionTimingEnvelope(fake_torch(), 0, profiled=False)
    with pytest.raises(RuntimeError, match="after production timing"):
        MODULE.compact_output_hash_after_timing([torch.tensor([1, 2])], timer)
    with timer.route():
        pass
    digest, token_ids = MODULE.compact_output_hash_after_timing(
        [torch.tensor([1, 2])], timer
    )
    assert token_ids == [1, 2]
    assert digest == MODULE._hash_token_ids(token_ids)


def test_production_environment_rejects_debug_and_trace_modes(monkeypatch):
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "")
    with pytest.raises(RuntimeError, match="CUDA_LAUNCH_BLOCKING"):
        MODULE._validate_production_environment()
    monkeypatch.delenv("CUDA_LAUNCH_BLOCKING")
    monkeypatch.setenv("SGLANG_DLLM_REQUEST_METRICS", "1")
    with pytest.raises(RuntimeError, match="SGLANG_DLLM_REQUEST_METRICS"):
        MODULE._validate_production_environment()


def test_timing_phases_are_nonoverlapping_and_use_one_terminal_sync():
    torch_module = fake_torch()
    timer = MODULE.ProductionTimingEnvelope(torch_module, 0, profiled=True)
    with timer.route():
        with timer.cuda_phase("production_prepare_cuda"):
            with pytest.raises(RuntimeError, match="must not overlap"):
                with timer.cuda_phase("production_model_cuda"):
                    pass
        with timer.cuda_phase("production_model_cuda"):
            pass
    result = timer.result()
    assert result["synchronization_count"] == 1
    assert torch_module.cuda.stream.synchronizations == 1
    assert result["component_event_pairs"] == 2
    assert result["production_state_commit_cuda_ms"] == 0.0
    assert result["production_state_commit_unavailable_reason"]


def test_uninstrumented_timer_has_only_route_events():
    torch_module = fake_torch()
    timer = MODULE.ProductionTimingEnvelope(torch_module, 0, profiled=False)
    with timer.route():
        with timer.cuda_phase("production_prepare_cuda"):
            pass
        with timer.cuda_phase("production_model_cuda"):
            pass
    result = timer.result()
    assert len(torch_module.cuda.events) == 2
    assert result["component_event_pairs"] == 0
    assert result["production_prepare_cuda_ms"] is None


def test_exception_cleanup_still_performs_one_terminal_sync():
    torch_module = fake_torch()
    timer = MODULE.ProductionTimingEnvelope(torch_module, 0, profiled=True)
    with pytest.raises(RuntimeError, match="primary"):
        with timer.route():
            with timer.cuda_phase("production_model_cuda"):
                raise RuntimeError("primary")
    assert timer.finalized
    assert timer.result()["synchronization_count"] == 1
    assert torch_module.cuda.stream.synchronizations == 1


def test_summary_computes_overhead_and_comparisons():
    summary = summarize()
    assert summary["schema_version"] == 1
    assert summary["comparisons"]["profiling_overhead_pct"] == pytest.approx(
        {route: 1.0 for route in MODULE.ROUTES}
    )
    assert summary["comparisons"]["warm_speedup"] == 2.0
    assert summary["comparisons"]["warm_latency_reduction_pct"] == 50.0
    assert summary["comparisons"]["cold_amortization_steps"] == 0.8
    assert summary["gates"]["profiling_overhead_acceptable"] is True
    assert summary["publication_acceptable"] is True


def test_profiled_and_uninstrumented_workloads_must_match():
    records = raw_records()
    records[0]["workload_fingerprint"] = "different"
    summary = summarize(records)
    assert (
        summary["workload_sanity"]["profiled_and_uninstrumented_workloads_identical"]
        is False
    )
    assert summary["publication_acceptable"] is False


def test_profiled_and_uninstrumented_work_counters_must_match():
    records = raw_records()
    records[0]["gdn_replay_token_layer_positions"] += 1
    summary = summarize(records)
    assert (
        summary["workload_sanity"]["profiled_and_uninstrumented_workloads_identical"]
        is False
    )
    assert summary["publication_acceptable"] is False


def test_full_and_warm_output_hash_comparison_is_enforced():
    records = raw_records()
    warm = next(record for record in records if record["route"] == "warm_cached_suffix")
    warm["output_hash"] = "b" * 64
    summary = summarize(records)
    assert summary["gates"]["output_hashes_identical"] is False
    assert summary["publication_acceptable"] is False


def test_warm_recovery_replay_fails_the_gate():
    records = raw_records()
    warm = next(record for record in records if record["route"] == "warm_cached_suffix")
    warm["recovery_replay_count"] = 1
    summary = summarize(records)
    assert summary["gates"]["zero_warm_recovery_replay"] is False
    assert summary["publication_acceptable"] is False


def test_fallback_fails_publication():
    records = raw_records()
    records[0]["fallback_count"] = 1
    summary = summarize(records)
    assert summary["gates"]["zero_fallback"] is False
    assert summary["publication_acceptable"] is False


def test_incorrect_route_labels_fail_closed():
    records = raw_records()
    records[0]["route"] = "not-a-route"
    with pytest.raises(RuntimeError, match="incorrect route label"):
        summarize(records)


def test_profiling_overhead_above_five_percent_fails_publication():
    records = raw_records()
    for record in records:
        if record["variant"] == "minimally_profiled":
            record["timing"]["production_route_total_cuda_ms"] *= 1.06 / 1.01
    summary = summarize(records)
    assert summary["gates"]["profiling_overhead_acceptable"] is False
    assert summary["publication_acceptable"] is False


def test_bootstrap_improvement_interval_is_deterministic():
    full = [100.0 + index for index in range(10)]
    warm = [50.0 + index for index in range(10)]
    first = MODULE.bootstrap_improvement_interval(full, warm, seed=7)
    second = MODULE.bootstrap_improvement_interval(full, warm, seed=7)
    assert first == second
    assert first[0] > 0.0


def test_correctness_prerequisite_rejects_wrong_revision():
    correctness = {
        "cluster3_revision": "wrong",
        "profile": "efficiency_one",
        "strict_pass": True,
    }
    with pytest.raises(RuntimeError, match="accepted efficiency_one"):
        MODULE.validate_correctness_prerequisite(
            correctness,
            {},
            {},
            {"device_name": "NVIDIA A30"},
        )


def test_benchmark_runtime_closes_after_measurement_exception(tmp_path, monkeypatch):
    correctness = {
        "cluster3_revision": MODULE.ACCEPTED_CORRECTNESS_REVISION,
        "profile": "efficiency_one",
        "strict_pass": True,
        "hardware": {"device_name": "NVIDIA A30", "cuda_version": "12.8"},
    }
    checkpoint = {
        "files": {"config.json": {"sha256": "a"}},
        "model_scale": "2B",
    }
    preflight = {
        "repository": {"head": MODULE.ACCEPTED_CORRECTNESS_REVISION},
        "checkpoint_hashes_match": True,
        "a30_required": True,
        "environment": {
            "gpus": [{"name": "NVIDIA A30"}],
            "cuda": "12.8",
            "pytorch": "2.8",
        },
        "checkpoint": checkpoint,
    }
    correctness_path = tmp_path / "correctness.json"
    preflight_path = tmp_path / "preflight.json"
    correctness_path.write_text(json.dumps(correctness))
    preflight_path.write_text(json.dumps(preflight))
    hardware = {
        "device_name": "NVIDIA A30",
        "visible_device_count": 1,
        "cuda_version": "12.8",
        "pytorch_version": "2.8",
    }
    monkeypatch.setattr(MODULE, "production_hardware", lambda _device: hardware)

    runtime = SimpleNamespace(closed=False)
    runtime.close = lambda: setattr(runtime, "closed", True)
    validation = SimpleNamespace(
        _load_module=lambda *_args: SimpleNamespace(
            checkpoint_identity=lambda _path: checkpoint
        ),
        Cluster3ValidationRuntime=lambda _args: runtime,
        build_manifest=lambda *_args: [SimpleNamespace()],
        _git_revision=lambda: "revision",
    )
    args = SimpleNamespace(
        profile="production_efficiency",
        timed_repetitions=10,
        debug_sync_stages=False,
        dtype="bfloat16",
        tp_size=1,
        max_total_tokens=4096,
        model_path="/model",
        correctness_artifact=correctness_path,
        preflight_json=preflight_path,
        device=0,
        seed=1,
        output_jsonl=tmp_path / "raw.jsonl",
        summary_json=tmp_path / "summary.json",
    )

    def fail(*_args):
        raise RuntimeError("measurement failed")

    with pytest.raises(RuntimeError, match="measurement failed"):
        MODULE.run_production_benchmark(
            args,
            validation_module=validation,
            measurement_runner=fail,
        )
    assert runtime.closed


def test_runner_uses_three_warmups_and_ten_measured_repetitions(tmp_path, monkeypatch):
    correctness = {
        "cluster3_revision": MODULE.ACCEPTED_CORRECTNESS_REVISION,
        "profile": "efficiency_one",
        "strict_pass": True,
        "hardware": {"device_name": "NVIDIA A30", "cuda_version": "12.8"},
    }
    checkpoint = {
        "files": {"config.json": {"sha256": "a"}},
        "model_scale": "2B",
    }
    preflight = {
        "repository": {"head": MODULE.ACCEPTED_CORRECTNESS_REVISION},
        "checkpoint_hashes_match": True,
        "a30_required": True,
        "environment": {
            "gpus": [{"name": "NVIDIA A30"}],
            "cuda": "12.8",
            "pytorch": "2.8",
        },
        "checkpoint": checkpoint,
    }
    correctness_path = tmp_path / "correctness.json"
    preflight_path = tmp_path / "preflight.json"
    correctness_path.write_text(json.dumps(correctness))
    preflight_path.write_text(json.dumps(preflight))
    hardware = {
        "device_name": "NVIDIA A30",
        "visible_device_count": 1,
        "cuda_version": "12.8",
        "pytorch_version": "2.8",
    }
    monkeypatch.setattr(MODULE, "production_hardware", lambda _device: hardware)

    runtime = SimpleNamespace(closed=False)
    runtime.close = lambda: setattr(runtime, "closed", True)
    validation = SimpleNamespace(
        _load_module=lambda *_args: SimpleNamespace(
            checkpoint_identity=lambda _path: checkpoint
        ),
        Cluster3ValidationRuntime=lambda _args: runtime,
        build_manifest=lambda *_args: [SimpleNamespace()],
        _git_revision=lambda: "revision",
    )
    args = SimpleNamespace(
        profile="production_efficiency",
        timed_repetitions=10,
        debug_sync_stages=False,
        dtype="bfloat16",
        tp_size=1,
        max_total_tokens=4096,
        model_path="/model",
        correctness_artifact=correctness_path,
        preflight_json=preflight_path,
        device=0,
        seed=1,
        output_jsonl=tmp_path / "raw.jsonl",
        summary_json=tmp_path / "summary.json",
    )
    calls = []

    def measure(_runtime, _validation, _case, route, variant, repetition):
        calls.append((route, variant, repetition))
        return raw_record(route, variant, repetition)

    summary = MODULE.run_production_benchmark(
        args,
        validation_module=validation,
        measurement_runner=measure,
    )
    assert len(calls) == 3 * 2 * (3 + 10)
    assert sum(repetition < 0 for _, _, repetition in calls) == 3 * 2 * 3
    assert len(args.output_jsonl.read_text().splitlines()) == 3 * 2 * 10
    assert summary["publication_acceptable"] is True
    assert runtime.closed
