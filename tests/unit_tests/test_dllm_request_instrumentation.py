import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch


_REPO_ROOT = Path(__file__).parents[2]
_MODULE_PATH = _REPO_ROOT / "eval/sglang/srt/dllm/algorithm/instrumentation.py"
_SPEC = importlib.util.spec_from_file_location(
    "hybrid_diffusion_request_instrumentation", _MODULE_PATH
)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

DllmRequestMetrics = _MODULE.DllmRequestMetrics
emit_metrics_record = _MODULE.emit_metrics_record
detect_model_scale = _MODULE.detect_model_scale


def _load_base_module(monkeypatch):
    modules = {
        "sglang": types.ModuleType("sglang"),
        "sglang.srt": types.ModuleType("sglang.srt"),
        "sglang.srt.dllm": types.ModuleType("sglang.srt.dllm"),
        "sglang.srt.dllm.algorithm": types.ModuleType("sglang.srt.dllm.algorithm"),
        "sglang.srt.dllm.algorithm.instrumentation": _MODULE,
        "sglang.srt.dllm.config": types.ModuleType("sglang.srt.dllm.config"),
        "sglang.srt.server_args": types.ModuleType("sglang.srt.server_args"),
    }
    modules["sglang.srt.dllm.algorithm"].get_algorithm = lambda config: config
    modules["sglang.srt.dllm.config"].DllmConfig = type("DllmConfig", (), {})
    modules["sglang.srt.server_args"].ServerArgs = type("ServerArgs", (), {})
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    base_path = _MODULE_PATH.with_name("base.py")
    spec = importlib.util.spec_from_file_location(
        "hybrid_diffusion_algorithm_base", base_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ForwardMode:
    def __init__(self, decode=False):
        self.decode = decode

    def is_decode(self):
        return self.decode


class _FakeRunner:
    def __init__(self, model_path="Dream-org/HybridDiffusion-2B", output_ids=None):
        text_config = types.SimpleNamespace(num_hidden_layers=32)
        hf_config = types.SimpleNamespace(
            architectures=["Qwen3_5DLLMForConditionalGeneration"],
            _name_or_path=model_path,
        )
        self.model_config = types.SimpleNamespace(
            model_path=model_path,
            hf_text_config=text_config,
            hf_config=hf_config,
        )
        self.model = None
        self.calls = 0
        self.output_ids = list(output_ids or [11, 12])
        self.invoke_restore = False
        self.fail_before_restore = False

    def forward(self, forward_batch, pp_proxy_tensors=None):
        self.calls += 1
        if self.fail_before_restore:
            raise RuntimeError("backend failed")
        if self.invoke_restore:
            forward_batch.dllm_metrics_gdn_restore_callback([0])
        return list(self.output_ids)


def _fake_forward_batch(rid="request-7", rpx=7, decode=False):
    return types.SimpleNamespace(
        batch_size=1,
        input_ids=torch.tensor([1, 2, 3]),
        positions=torch.tensor([10, 11, 12]),
        req_pool_indices=torch.tensor([rpx]),
        dllm_rpx_cpu=[rpx],
        extend_seq_lens_cpu=[3],
        forward_mode=_ForwardMode(decode),
        rids=[rid],
    )


def _metric(route="ar_trust", scale="2B"):
    return DllmRequestMetrics(
        request_id="request-1",
        req_pool_idx=7,
        selected_mode=route,
        model_scale=scale,
    )


def test_01_schema_serialization_contains_required_fields():
    record = _metric().to_record()
    required = {
        "schema_version",
        "request_id",
        "req_pool_idx",
        "model_scale",
        "selected_mode",
        "phase_counts",
        "prompt_tokens",
        "ar_tokens",
        "stable_tokens",
        "active_tokens",
        "diffusion_steps",
        "kv_cache_hits",
        "gdn_state_restores",
        "invalidated_regions",
        "recomputed_token_layer_positions",
        "attention_time_ms",
        "gdn_time_ms",
        "mlp_time_ms",
        "model_forward_time_ms",
        "verification_time_ms",
        "total_latency_ms",
        "peak_memory_bytes",
        "component_timed_forwards",
        "component_untimed_forwards",
        "timing_scope",
        "memory_scope",
        "other_model_time_ms",
        "forward_timer_coverage",
        "model_component_coverage",
        "model_forward_to_total_latency_ratio",
        "unavailable_metrics",
    }
    assert required <= record.keys()
    assert record["schema_version"] == 2


def test_02_model_scale_2b_is_detected_without_shape_guessing(monkeypatch):
    monkeypatch.delenv("SGLANG_DLLM_MODEL_SCALE", raising=False)
    assert detect_model_scale(_FakeRunner()) == "2B"
    assert detect_model_scale(_FakeRunner("Dream-org/HybridDiffusion-4B")) == "4B"
    assert detect_model_scale(_FakeRunner("Dream-org/HybridDiffusion-9B")) == "9B"


def test_03_selected_mode_is_immutable_request_route():
    metric = _metric("ar_trust")
    metric.record_forward(
        phases=("self_spec_verify", "self_spec_draft"),
        active_tokens=7,
        diffusion_step=True,
    )
    assert metric.selected_mode == "ar_trust"


def test_04_phase_counts_are_separate_from_route():
    metric = _metric("ar_trust")
    metric.record_forward(
        phases=("self_spec_verify", "self_spec_draft"),
        active_tokens=7,
        diffusion_step=True,
    )
    metric.record_phase("self_spec_correction")
    assert metric.phase_counts["self_spec_verify"] == 1
    assert metric.phase_counts["self_spec_draft"] == 1
    assert metric.phase_counts["self_spec_correction"] == 1


def test_05_token_counters_clamp_invalid_values():
    metric = _metric()
    metric.record_forward(phases="prefill", active_tokens=-9, diffusion_step=False)
    metric.record_consumed_tokens(stable=2, ar=9)
    metric.set_final_token_counts(stable=-1, ar=-4)
    assert metric.active_tokens == 0
    assert metric.stable_tokens == 0
    assert metric.ar_tokens == 0


def test_06_exactly_one_record_is_emitted_after_pop(monkeypatch, tmp_path):
    base = _load_base_module(monkeypatch)
    recorder = base.RequestMetricsRecorder("causal")
    recorder._request_metrics[7] = _metric("causal")
    path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("SGLANG_DLLM_REQUEST_METRICS_PATH", str(path))
    first = recorder.pop_request_metrics(7)
    assert first is not None
    emit_metrics_record(first, tp_rank=0)
    assert recorder.pop_request_metrics(7) is None
    assert len(path.read_text().splitlines()) == 1


def test_07_nonzero_tp_rank_does_not_emit(monkeypatch, tmp_path):
    path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("SGLANG_DLLM_REQUEST_METRICS_PATH", str(path))
    assert emit_metrics_record(_metric(), tp_rank=1) is None
    assert not path.exists()


def test_08_pool_slot_reuse_replaces_old_request_state(monkeypatch):
    base = _load_base_module(monkeypatch)
    recorder = base.RequestMetricsRecorder("causal")
    recorder._model_scale = "2B"
    first = recorder._metrics_for_batch(_fake_forward_batch("old", 7))[0]
    first.active_tokens = 99
    second = recorder._metrics_for_batch(_fake_forward_batch("new", 7))[0]
    assert second.request_id == "new"
    assert second.active_tokens == 0


def test_09_aborted_request_cleanup_removes_state(monkeypatch):
    base = _load_base_module(monkeypatch)
    recorder = base.RequestMetricsRecorder("causal")
    recorder._request_metrics[7] = _metric("causal")
    recorder.cleanup_request(7)
    assert recorder._request_metrics == {}


def test_10_ar_trust_uses_instrumented_forwards(monkeypatch):
    source = (_MODULE_PATH.with_name("hybrid_diffusion_self_spec.py")).read_text()
    assert source.count("self._forward_with_metrics(") >= 2
    assert "out = model_runner.forward(forward_batch" not in source
    base = _load_base_module(monkeypatch)
    config = types.SimpleNamespace(
        block_size=7, mask_id=99, algorithm="HybridDiffusionSelfSpec"
    )
    assert base.DllmAlgorithm(config)._instrumentation.selected_mode == "ar_trust"


def test_11_diffusion_trust_uses_instrumented_forwards():
    for name in (
        "low_confidence.py",
        "joint_threshold.py",
        "low_confidence_shift_hybrid_diffusion.py",
    ):
        source = (_MODULE_PATH.with_name(name)).read_text()
        assert "self._forward_with_metrics(" in source


def test_12_causal_recorder_produces_route_level_metric(monkeypatch):
    base = _load_base_module(monkeypatch)
    monkeypatch.setenv("SGLANG_DLLM_REQUEST_METRICS", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    recorder = base.RequestMetricsRecorder("causal")
    batch = _fake_forward_batch(decode=True)
    recorder.enabled(_FakeRunner())
    recorder.forward(_FakeRunner(), batch, phases="causal_decode")
    metric = recorder.pop_request_metrics(7)
    metric.set_final_token_counts(stable=2)
    assert metric.selected_mode == "causal"
    assert metric.ar_tokens == metric.stable_tokens == 2
    assert metric.diffusion_steps == 0


def test_13_gdn_restore_increments_only_after_backend_success(monkeypatch):
    base = _load_base_module(monkeypatch)
    monkeypatch.setenv("SGLANG_DLLM_REQUEST_METRICS", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    recorder = base.RequestMetricsRecorder("diffusion_trust")
    runner = _FakeRunner()
    runner.invoke_restore = True
    recorder.enabled(runner)
    recorder.forward(runner, _fake_forward_batch(), phases="diffusion_denoise")
    assert recorder.get_request_metrics(7).gdn_state_restores == 1

    failed = base.RequestMetricsRecorder("diffusion_trust")
    runner.fail_before_restore = True
    failed.enabled(runner)
    with pytest.raises(RuntimeError, match="backend failed"):
        failed.forward(runner, _fake_forward_batch(), phases="diffusion_denoise")
    assert failed._request_metrics == {}


def test_14_invalidation_requires_real_recording_call():
    metric = _metric()
    assert metric.invalidated_regions == []
    metric.record_invalidation(start=10, length=0)
    assert metric.invalidated_regions == []
    metric.record_invalidation(start=10, length=2)
    assert metric.invalidated_regions == [{"start": 10, "end": 12, "length": 2}]


def test_15_recomputation_requires_later_actual_revisit():
    metric = _metric()
    metric.record_invalidation(start=10, length=3)
    assert metric.recomputed_token_layer_positions == 0
    metric.record_revisited_positions([9, 11, 20], layers=32)
    assert metric.recomputed_token_layer_positions == 32
    metric.record_revisited_positions([11], layers=32)
    assert metric.recomputed_token_layer_positions == 32


def test_16_peak_uses_max_memory_not_current_allocation(monkeypatch):
    base = _load_base_module(monkeypatch)
    monkeypatch.setenv("SGLANG_DLLM_REQUEST_METRICS", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 999)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *_args: 123)

    class FakeEvent:
        def __init__(self, enable_timing=True):
            pass

        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, _other):
            return 5.0

    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    recorder = base.RequestMetricsRecorder("causal")
    runner = _FakeRunner()
    recorder.enabled(runner)
    recorder.start_request_memory(
        "request-7", device=torch.device("cpu"), exclusive_request=True
    )
    recorder.forward(runner, _fake_forward_batch(), phases="prefill")
    recorder.flush_forward_timings()
    assert recorder.get_request_metrics(7).peak_memory_bytes == 999


def test_17_disabled_instrumentation_is_direct_forward(monkeypatch):
    base = _load_base_module(monkeypatch)
    monkeypatch.delenv("SGLANG_DLLM_REQUEST_METRICS", raising=False)
    config = types.SimpleNamespace(block_size=7, mask_id=99, algorithm="low_confidence")
    algorithm = base.DllmAlgorithm(config)
    runner = _FakeRunner()
    batch = _fake_forward_batch()
    assert algorithm._forward_with_metrics(runner, batch, modes="prefill") == [11, 12]
    assert runner.calls == 1
    assert algorithm._request_metrics == {}


def test_18_enabled_and_disabled_return_identical_tokens(monkeypatch):
    base = _load_base_module(monkeypatch)
    config = types.SimpleNamespace(
        block_size=7, mask_id=99, algorithm="hybrid_diffusion_self_spec"
    )
    runner = _FakeRunner(output_ids=[42, 43, 44])
    batch = _fake_forward_batch()
    monkeypatch.delenv("SGLANG_DLLM_REQUEST_METRICS", raising=False)
    disabled = base.DllmAlgorithm(config)._forward_with_metrics(
        runner, batch, modes="prefill"
    )
    monkeypatch.setenv("SGLANG_DLLM_REQUEST_METRICS", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    enabled = base.DllmAlgorithm(config)._forward_with_metrics(
        runner, batch, modes="prefill"
    )
    assert enabled == disabled == [42, 43, 44]


def test_19_timer_components_are_nonnegative_and_nonoverlapping():
    metric = _metric()
    metric.add_forward_time(
        model_forward_ms=10.0,
        attention_ms=6.0,
        gdn_ms=3.0,
        mlp_ms=4.0,
        timing_scope="cuda_events_request",
    )
    record = metric.to_record()
    components = [
        record["attention_time_ms"],
        record["gdn_time_ms"],
        record["mlp_time_ms"],
        record["other_model_time_ms"],
    ]
    assert all(value >= 0 for value in components)
    assert sum(components) == pytest.approx(record["model_forward_time_ms"])
    assert record["model_component_coverage"] == pytest.approx(1.0)


def test_20_jsonl_stays_valid_for_sequential_requests(monkeypatch, tmp_path):
    path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("SGLANG_DLLM_REQUEST_METRICS_PATH", str(path))
    for index in range(3):
        metric = DllmRequestMetrics(
            request_id=f"request-{index}",
            req_pool_idx=index,
            selected_mode="causal",
            model_scale="2B",
        )
        emit_metrics_record(metric, tp_rank=0)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["request_id"] for record in records] == [
        "request-0",
        "request-1",
        "request-2",
    ]
