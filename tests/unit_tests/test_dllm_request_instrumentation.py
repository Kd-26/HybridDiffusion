import importlib.util
import sys
import types
from pathlib import Path

import torch


_MODULE_PATH = (
    Path(__file__).parents[2] / "eval/sglang/srt/dllm/algorithm/instrumentation.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "hybrid_diffusion_request_instrumentation", _MODULE_PATH
)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

DllmRequestMetrics = _MODULE.DllmRequestMetrics


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


class _FakeRunner:
    def __init__(self, hidden_size=2560):
        text_config = types.SimpleNamespace(
            hidden_size=hidden_size,
            num_hidden_layers=32,
        )
        hf_config = types.SimpleNamespace(
            architectures=["Qwen3_5DLLMForConditionalGeneration"]
        )
        self.model_config = types.SimpleNamespace(
            hf_text_config=text_config,
            hf_config=hf_config,
        )
        self.model = None
        self.calls = 0

    def forward(self, forward_batch, pp_proxy_tensors=None):
        self.calls += 1
        return (forward_batch, pp_proxy_tensors)


def _fake_forward_batch():
    return types.SimpleNamespace(
        batch_size=1,
        input_ids=torch.tensor([1, 2, 3]),
        req_pool_indices=torch.tensor([7]),
        dllm_rpx_cpu=[7],
        dllm_request_token_counts=[3],
        dllm_gdn_layer_count=24,
        dllm_model_layer_count=32,
        rids=["request-7"],
    )


def test_request_metrics_accounting_and_schema():
    metrics = DllmRequestMetrics(request_id="request-1", req_pool_idx=7)

    metrics.record_forward(
        mode="self_spec_verify",
        active_tokens=7,
        diffusion_step=True,
        gdn_state_restores=24,
        recomputed_token_layer_positions=96,
    )
    metrics.record_finalized_tokens(stable=4, ar=1)
    metrics.record_invalidation(start=103, length=3)
    metrics.add_component_time(attention_ms=1.25, gdn_ms=2.5)
    metrics.verification_time_ms += 0.75

    record = metrics.to_record()
    assert record["schema_version"] == 1
    assert record["model_scale"] == "4B"
    assert record["selected_mode"] == "self_spec_verify"
    assert record["selected_mode_counts"] == {"self_spec_verify": 1}
    assert record["active_tokens"] == 7
    assert record["diffusion_steps"] == 1
    assert record["gdn_state_restores"] == 24
    assert record["stable_tokens"] == 4
    assert record["ar_tokens"] == 1
    assert record["invalidated_regions"] == [{"start": 103, "length": 3}]
    # Invalidating a region does not claim it was recomputed; that is recorded
    # only if a later forward actually revisits it.
    assert record["recomputed_token_layer_positions"] == 96
    assert record["attention_time_ms"] == 1.25
    assert record["gdn_time_ms"] == 2.5
    assert record["verification_time_ms"] == 0.75
    assert record["component_timed_forwards"] == 1


def test_request_metrics_clamps_token_counts_and_marks_timing_gaps():
    metrics = DllmRequestMetrics(request_id="request-2", req_pool_idx=8)

    metrics.record_finalized_tokens(stable=2, ar=5)
    metrics.record_invalidation(start=-2, length=-1)
    metrics.add_component_time(attention_ms=None, gdn_ms=None)

    assert metrics.stable_tokens == 2
    assert metrics.ar_tokens == 2
    assert metrics.invalidated_regions == []
    assert metrics.attention_time_ms is None
    assert metrics.gdn_time_ms is None
    assert metrics.component_timed_forwards == 0
    assert metrics.component_untimed_forwards == 1


def test_disabled_instrumentation_is_a_direct_forward(monkeypatch):
    base = _load_base_module(monkeypatch)
    monkeypatch.delenv("SGLANG_DLLM_REQUEST_METRICS", raising=False)

    config = types.SimpleNamespace(block_size=7, mask_id=99)
    algorithm = base.DllmAlgorithm(config)
    runner = _FakeRunner()
    batch = _fake_forward_batch()

    result = algorithm._forward_with_metrics(
        runner,
        batch,
        modes=["would_fail_if_expanded_with_zero_batch"],
    )

    assert result == (batch, None)
    assert runner.calls == 1
    assert algorithm._request_metrics == {}


def test_4b_gate_and_request_local_recompute_accounting(monkeypatch):
    base = _load_base_module(monkeypatch)
    monkeypatch.setenv("SGLANG_DLLM_REQUEST_METRICS", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    config = types.SimpleNamespace(block_size=7, mask_id=99)
    algorithm = base.DllmAlgorithm(config)
    runner = _FakeRunner()
    batch = _fake_forward_batch()

    algorithm._forward_with_metrics(
        runner,
        batch,
        modes="self_spec_verify",
        diffusion_steps=True,
        gdn_restores=True,
    )
    algorithm._set_output_token_modes(batch, [[False, True]])
    algorithm.record_consumed_tokens(7, 1)
    algorithm._record_invalidation(batch, batch_index=0, start=10, length=2)
    algorithm._forward_with_metrics(
        runner,
        batch,
        modes="self_spec_cold_start",
        diffusion_steps=True,
        gdn_restores=True,
    )

    metrics = algorithm.pop_request_metrics(7)
    assert metrics.request_id == "request-7"
    assert metrics.active_tokens == 6
    assert metrics.diffusion_steps == 2
    assert metrics.gdn_state_restores == 48
    assert metrics.stable_tokens == 1
    assert metrics.ar_tokens == 0
    assert metrics.invalidated_regions == [{"start": 10, "length": 2}]
    assert metrics.recomputed_token_layer_positions == 64
    assert runner.calls == 2

    non_4b_algorithm = base.DllmAlgorithm(config)
    non_4b_runner = _FakeRunner(hidden_size=2048)
    non_4b_algorithm._forward_with_metrics(
        non_4b_runner,
        batch,
        modes="self_spec_verify",
    )
    assert non_4b_algorithm._request_metrics == {}
