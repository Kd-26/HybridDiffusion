"""Request-scoped profiling for exact Region-DAG handoff.

The profiler is deliberately opt-in.  It records host-only work with a
monotonic clock and CUDA work with events on the current execution stream.
CUDA events are resolved after one terminal synchronization; phase contexts
never synchronize unless ``debug_sync`` is explicitly enabled.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import time
from collections import Counter
from typing import Any, Iterator, Mapping, Optional


PROFILE_SCHEMA_VERSION = 1

PROFILE_PHASES = (
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

# Totals are envelopes.  Every other phase is exclusive so the overhead table
# can sum its rows without double counting.
ENVELOPE_PHASES = frozenset(("model_forward_total", "request_total"))


@dataclasses.dataclass
class _PhaseSample:
    name: str
    host_started_ns: int
    host_finished_ns: int
    cuda_started: Any = None
    cuda_finished: Any = None


class RequestScopedProfiler:
    """Collect one request's production handoff evidence.

    ``cuda_enabled`` should only be true when the profiled operations execute
    on a CUDA stream.  Supplying a torch-like object is supported for focused
    unit tests; production callers leave ``torch_module`` unset.
    """

    def __init__(
        self,
        *,
        request_id: str,
        cuda_enabled: bool,
        device: Any = None,
        debug_sync: bool = False,
        timing_scope: str = "request_exclusive",
        metadata: Optional[Mapping[str, Any]] = None,
        torch_module: Any = None,
    ) -> None:
        if not request_id:
            raise ValueError("profile request_id must be nonempty")
        if timing_scope not in ("request_exclusive", "shared_batch"):
            raise ValueError("invalid profiling timing_scope")
        if debug_sync and not cuda_enabled:
            raise ValueError("debug synchronization requires CUDA timing")

        self.request_id = str(request_id)
        self.cuda_enabled = bool(cuda_enabled)
        self.device = device
        self.debug_sync = bool(debug_sync)
        self.timing_scope = timing_scope
        self.metadata = dict(metadata or {})
        self.counters: Counter[str] = Counter()
        self.synchronizations: list[dict[str, Any]] = []
        self._samples: list[_PhaseSample] = []
        self._active_leaf: Optional[str] = None
        self._active_envelopes: list[str] = []
        self._finalized = False
        self._snapshot: Optional[dict[str, Any]] = None
        self._torch = torch_module
        if self.cuda_enabled:
            if self._torch is None:
                self._torch = __import__("torch")
            if not bool(self._torch.cuda.is_available()):
                raise RuntimeError("CUDA profiling requested without CUDA")

    def set_metadata(self, **values: Any) -> None:
        if self._finalized:
            raise RuntimeError("cannot update a finalized profile")
        self.metadata.update(values)

    def increment(self, name: str, value: int = 1) -> None:
        if self._finalized:
            raise RuntimeError("cannot update a finalized profile")
        self.counters[str(name)] += int(value)

    def _event(self) -> Any:
        return self._torch.cuda.Event(enable_timing=True)

    def _record_event(self, event: Any) -> None:
        stream = self._torch.cuda.current_stream(device=self.device)
        event.record(stream)

    def _nvtx_push(self, name: str) -> bool:
        if not self.cuda_enabled:
            return False
        nvtx = getattr(self._torch.cuda, "nvtx", None)
        push = getattr(nvtx, "range_push", None)
        if not callable(push):
            return False
        push(f"cluster3::{self.request_id}::{name}")
        return True

    def _nvtx_pop(self, pushed: bool) -> None:
        if pushed:
            self._torch.cuda.nvtx.range_pop()

    def _synchronize(self, *, source: str, classification: str) -> None:
        if classification not in ("diagnostic", "final_timing"):
            raise ValueError("invalid synchronization classification")
        self._torch.cuda.current_stream(device=self.device).synchronize()
        self.synchronizations.append(
            {"source": source, "classification": classification}
        )

    @contextlib.contextmanager
    def phase(self, name: str, *, cuda: bool = False) -> Iterator[None]:
        if name not in PROFILE_PHASES:
            raise ValueError(f"unknown profiling phase {name!r}")
        if self._finalized:
            raise RuntimeError("cannot record a finalized profile")
        is_envelope = name in ENVELOPE_PHASES
        if is_envelope:
            if name in self._active_envelopes:
                raise RuntimeError(f"profiling phase {name!r} is already active")
            self._active_envelopes.append(name)
        elif self._active_leaf is not None:
            raise RuntimeError(
                "profiling leaf phases must not overlap: "
                f"{self._active_leaf!r} and {name!r}"
            )
        else:
            self._active_leaf = name

        use_cuda = bool(cuda and self.cuda_enabled)
        started_event = self._event() if use_cuda else None
        finished_event = self._event() if use_cuda else None
        pushed = self._nvtx_push(name)
        host_started_ns = time.perf_counter_ns()
        if started_event is not None:
            self._record_event(started_event)
        try:
            yield
        finally:
            if finished_event is not None:
                self._record_event(finished_event)
            host_finished_ns = time.perf_counter_ns()
            self._nvtx_pop(pushed)
            self._samples.append(
                _PhaseSample(
                    name=name,
                    host_started_ns=host_started_ns,
                    host_finished_ns=host_finished_ns,
                    cuda_started=started_event,
                    cuda_finished=finished_event,
                )
            )
            if self.debug_sync and use_cuda:
                self._synchronize(
                    source=f"{__file__}:{inspect.currentframe().f_lineno}",
                    classification="diagnostic",
                )
            if is_envelope:
                popped = self._active_envelopes.pop()
                if popped != name:
                    raise RuntimeError("profiling envelope phases closed out of order")
            else:
                self._active_leaf = None

    def finalize(self) -> dict[str, Any]:
        """Resolve all events after one terminal stream synchronization."""
        if self._snapshot is not None:
            return self._snapshot
        if self._active_leaf is not None or self._active_envelopes:
            raise RuntimeError("cannot finalize while a profiling phase is active")
        if self.cuda_enabled and any(
            sample.cuda_finished is not None for sample in self._samples
        ):
            self._synchronize(
                source=f"{__file__}:{inspect.currentframe().f_lineno}",
                classification="final_timing",
            )

        phase_values: dict[str, dict[str, Any]] = {}
        for name in PROFILE_PHASES:
            samples = [sample for sample in self._samples if sample.name == name]
            host_ms = sum(
                (sample.host_finished_ns - sample.host_started_ns) / 1_000_000.0
                for sample in samples
            )
            gpu_samples = [
                float(sample.cuda_started.elapsed_time(sample.cuda_finished))
                for sample in samples
                if sample.cuda_started is not None
            ]
            phase_values[name] = {
                "calls": len(samples),
                "host_ms": host_ms,
                "cuda_ms": sum(gpu_samples) if gpu_samples else None,
                "timing_domain": "cuda" if gpu_samples else "host",
                "envelope": name in ENVELOPE_PHASES,
            }

        peak_allocated = None
        peak_reserved = None
        if self.cuda_enabled:
            peak_allocated = int(self._torch.cuda.max_memory_allocated(self.device))
            peak_reserved = int(self._torch.cuda.max_memory_reserved(self.device))

        self._finalized = True
        self._snapshot = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "request_id": self.request_id,
            "timing_scope": self.timing_scope,
            "debug_sync": self.debug_sync,
            "metadata": dict(self.metadata),
            "phases": phase_values,
            "counters": dict(sorted(self.counters.items())),
            "synchronization_count": len(self.synchronizations),
            "synchronizations": list(self.synchronizations),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        }
        return self._snapshot


@contextlib.contextmanager
def optional_profile_phase(
    profiler: Optional[RequestScopedProfiler],
    name: str,
    *,
    cuda: bool = False,
) -> Iterator[None]:
    """A zero-behavior-change profiling context for optional call sites."""
    if profiler is None:
        yield
        return
    with profiler.phase(name, cuda=cuda):
        yield
