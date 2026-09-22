import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
PATH = ROOT / "eval/sglang/srt/dllm/region/profiling.py"
SPEC = importlib.util.spec_from_file_location("cluster3_production_profiling", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeEvent:
    def __init__(self, clock):
        self.clock = clock
        self.value = None

    def record(self, _stream):
        self.value = self.clock.tick()

    def elapsed_time(self, other):
        return float(other.value - self.value)


class FakeClock:
    def __init__(self):
        self.value = 0

    def tick(self):
        self.value += 2
        return self.value


class FakeStream:
    def __init__(self):
        self.sync_calls = 0

    def synchronize(self):
        self.sync_calls += 1


class FakeNvtx:
    def __init__(self):
        self.ranges = []
        self.depth = 0

    def range_push(self, name):
        self.ranges.append(name)
        self.depth += 1

    def range_pop(self):
        self.depth -= 1


class FakeCuda:
    def __init__(self):
        self.clock = FakeClock()
        self.stream = FakeStream()
        self.nvtx = FakeNvtx()

    @staticmethod
    def is_available():
        return True

    def Event(self, enable_timing):
        assert enable_timing is True
        return FakeEvent(self.clock)

    def current_stream(self, device=None):
        assert device == "cuda:0"
        return self.stream

    @staticmethod
    def max_memory_allocated(_device):
        return 1024

    @staticmethod
    def max_memory_reserved(_device):
        return 2048


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()


def make_cuda_profiler(*, debug_sync=False):
    fake_torch = FakeTorch()
    profiler = MODULE.RequestScopedProfiler(
        request_id="request-1",
        cuda_enabled=True,
        device="cuda:0",
        debug_sync=debug_sync,
        metadata={"selected_route": "cached"},
        torch_module=fake_torch,
    )
    return profiler, fake_torch


def test_declares_every_required_phase_once():
    assert MODULE.PROFILE_PHASES == (
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
    assert len(MODULE.PROFILE_PHASES) == len(set(MODULE.PROFILE_PHASES))


def test_cuda_events_resolve_with_one_terminal_synchronization():
    profiler, fake_torch = make_cuda_profiler()
    with profiler.phase("request_total", cuda=True):
        with profiler.phase("attention_forward", cuda=True):
            pass
        with profiler.phase("gdn_forward", cuda=True):
            pass

    assert fake_torch.cuda.stream.sync_calls == 0
    record = profiler.finalize()
    assert fake_torch.cuda.stream.sync_calls == 1
    assert record["synchronization_count"] == 1
    assert record["synchronizations"][0]["classification"] == "final_timing"
    assert record["phases"]["attention_forward"]["cuda_ms"] == 2.0
    assert record["phases"]["gdn_forward"]["cuda_ms"] == 2.0
    assert record["peak_allocated_bytes"] == 1024
    assert record["peak_reserved_bytes"] == 2048
    assert fake_torch.cuda.nvtx.depth == 0
    assert all(
        value.startswith("cluster3::request-1::")
        for value in fake_torch.cuda.nvtx.ranges
    )


def test_debug_sync_is_explicit_and_counted():
    profiler, fake_torch = make_cuda_profiler(debug_sync=True)
    with profiler.phase("attention_forward", cuda=True):
        pass
    record = profiler.finalize()
    assert fake_torch.cuda.stream.sync_calls == 2
    assert [entry["classification"] for entry in record["synchronizations"]] == [
        "diagnostic",
        "final_timing",
    ]


def test_leaf_phases_cannot_overlap_but_totals_can_enclose_them():
    profiler = MODULE.RequestScopedProfiler(
        request_id="cpu-request", cuda_enabled=False
    )
    with profiler.phase("model_forward_total"):
        with profiler.phase("mlp_forward"):
            with pytest.raises(RuntimeError, match="must not overlap"):
                with profiler.phase("attention_forward"):
                    pass


def test_cpu_profile_has_no_synchronization_and_reports_missing_phases():
    profiler = MODULE.RequestScopedProfiler(
        request_id="cpu-request",
        cuda_enabled=False,
        timing_scope="shared_batch",
    )
    profiler.increment("recovery_replay_count", 0)
    with MODULE.optional_profile_phase(profiler, "dependency_validation"):
        pass
    record = profiler.finalize()
    assert record["timing_scope"] == "shared_batch"
    assert record["synchronization_count"] == 0
    assert record["phases"]["dependency_validation"]["calls"] == 1
    assert record["phases"]["dependency_validation"]["cuda_ms"] is None
    assert record["phases"]["kv_restore"]["calls"] == 0
    assert record["peak_allocated_bytes"] is None
    assert profiler.finalize() is record


def test_unknown_phase_and_post_finalize_mutation_fail_closed():
    profiler = MODULE.RequestScopedProfiler(request_id="x", cuda_enabled=False)
    with pytest.raises(ValueError, match="unknown profiling phase"):
        with profiler.phase("not-a-phase"):
            pass
    profiler.finalize()
    with pytest.raises(RuntimeError, match="finalized"):
        profiler.increment("cache_hit")
