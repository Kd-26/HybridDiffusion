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
from contextlib import ExitStack
from typing import Any, Iterator, Mapping, Optional


PROFILE_SCHEMA_VERSION = 3

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
    "suffix_prepare_total",
    "suffix_finalize_total",
    "runtime_plan_build",
    "canonical_prefix_location_materialization",
    "request_metadata_attachment",
    "frontier_key_construction",
    "schedule_batch_initialization",
    "schedule_batch_prepare",
    "worker_batch_construction",
    "forward_batch_initialization",
    "model_request_slot_materialization",
    "trace_materialization",
    "token_hidden_input_preparation",
    "decoder_layer_total",
    "pre_attention_normalization",
    "attention_block_total",
    "attention_qkv_projection",
    "rope_attention_preparation",
    "attention_output_projection",
    "gdn_block_total",
    "gdn_input_projection",
    "gdn_output_projection",
    "residual_post_attention_normalization",
    "residual_connection",
    "final_normalization",
    "lm_head_projection",
)

# The primary parent identifies the hierarchy used by the warm-route report.
# Some phases also run on full/cold routes without that parent being active;
# the snapshot still records the intended hierarchy so consumers never sum an
# envelope with its descendants.
PHASE_PARENTS = {
    "request_total": None,
    "active_suffix_forward": "request_total",
    "suffix_prepare_total": "active_suffix_forward",
    "model_forward_total": "active_suffix_forward",
    "suffix_finalize_total": "active_suffix_forward",
    "runtime_plan_build": "suffix_prepare_total",
    "canonical_prefix_location_materialization": "suffix_prepare_total",
    "request_metadata_attachment": "suffix_prepare_total",
    "frontier_key_construction": "suffix_prepare_total",
    "schedule_batch_initialization": "suffix_prepare_total",
    "schedule_batch_prepare": "suffix_prepare_total",
    "worker_batch_construction": "suffix_prepare_total",
    "forward_batch_initialization": "suffix_prepare_total",
    "region_mask_build": "forward_batch_initialization",
    "kv_restore": "schedule_batch_prepare",
    "flashinfer_plan_build": "suffix_prepare_total",
    "model_request_slot_materialization": "suffix_prepare_total",
    "trace_materialization": "suffix_finalize_total",
    "token_hidden_input_preparation": "model_forward_total",
    "decoder_layer_total": "model_forward_total",
    "final_normalization": "model_forward_total",
    "lm_head_projection": "model_forward_total",
    "pre_attention_normalization": "decoder_layer_total",
    "attention_block_total": "decoder_layer_total",
    "gdn_block_total": "decoder_layer_total",
    "residual_post_attention_normalization": "decoder_layer_total",
    "mlp_forward": "decoder_layer_total",
    "residual_connection": "decoder_layer_total",
    "attention_qkv_projection": "attention_block_total",
    "rope_attention_preparation": "attention_block_total",
    "attention_forward": "attention_block_total",
    "attention_output_projection": "attention_block_total",
    "gdn_input_projection": "gdn_block_total",
    "region_cache_lookup": "gdn_block_total",
    "gdn_restore": "gdn_block_total",
    "gdn_forward": "gdn_block_total",
    "prefix_snapshot": "gdn_block_total",
    "cache_commit": "gdn_block_total",
    "gdn_output_projection": "gdn_block_total",
    "identity_token_hash": "request_total",
    "identity_position_hash": "request_total",
    "dependency_validation": "runtime_plan_build",
    "request_setup": "request_total",
    "recovery_replay": "active_suffix_forward",
    "verification": "request_total",
    "scheduler_postprocess": "request_total",
}

ENVELOPE_PHASES = frozenset(
    (
        "request_total",
        "active_suffix_forward",
        "suffix_prepare_total",
        "model_forward_total",
        "suffix_finalize_total",
        "runtime_plan_build",
        "schedule_batch_prepare",
        "forward_batch_initialization",
        "decoder_layer_total",
        "attention_block_total",
        "gdn_block_total",
    )
)

PHASE_HIERARCHY = {
    name: {
        "parent": PHASE_PARENTS.get(name),
        "kind": "envelope" if name in ENVELOPE_PHASES else "leaf",
    }
    for name in PROFILE_PHASES
}


class FrozenDict(dict):
    """JSON-serializable immutable mapping used for finalized evidence."""

    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("finalized profiling snapshots are immutable")

    __delitem__ = _immutable
    __ior__ = _immutable
    __setitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenDict({str(key): _freeze(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(child) for child in value)
    return value


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
        self.counters: Counter[str] = Counter(
            {
                "cache_hit_count": 0,
                "cache_miss_count": 0,
                "gdn_state_restore_count": 0,
                "recovery_replay_count": 0,
                "snapshot_count": 0,
            }
        )
        self.synchronizations: list[dict[str, Any]] = []
        self._samples: list[_PhaseSample] = []
        self._active_leaf: Optional[str] = None
        self._active_envelopes: list[str] = []
        self._phase_stack: list[str] = []
        self._unavailable: dict[str, str] = {}
        self._peak_allocated = 0
        self._peak_reserved = 0
        self._finalized = False
        self._snapshot: Optional[FrozenDict] = None
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

    def mark_unavailable(self, phase: str, reason: str) -> None:
        if self._finalized:
            raise RuntimeError("cannot update a finalized profile")
        if phase not in PROFILE_PHASES:
            raise ValueError(f"unknown profiling phase {phase!r}")
        if not reason:
            raise ValueError("profiling availability reason must be nonempty")
        self._unavailable[phase] = str(reason)

    def observe_peak_memory(self) -> None:
        if self._finalized:
            raise RuntimeError("cannot update a finalized profile")
        if not self.cuda_enabled:
            return
        self._peak_allocated = max(
            self._peak_allocated,
            int(self._torch.cuda.max_memory_allocated(self.device)),
        )
        self._peak_reserved = max(
            self._peak_reserved,
            int(self._torch.cuda.max_memory_reserved(self.device)),
        )

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
            if self._active_envelopes:
                active = self._active_envelopes[-1]
                parent = PHASE_PARENTS.get(name)
                active_parent = PHASE_PARENTS.get(active)
                if active != parent and active_parent == parent:
                    raise RuntimeError(
                        "profiling sibling envelopes must not overlap: "
                        f"{active!r} and {name!r}"
                    )
            self._active_envelopes.append(name)
        elif self._active_leaf is not None:
            raise RuntimeError(
                "profiling leaf phases must not overlap: "
                f"{self._active_leaf!r} and {name!r}"
            )
        else:
            self._active_leaf = name
        self._phase_stack.append(name)

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
            if not self._phase_stack or self._phase_stack[-1] != name:
                self._phase_stack.clear()
                self._active_envelopes.clear()
                self._active_leaf = None
                self._nvtx_pop(pushed)
                raise RuntimeError("profiling phases closed out of stack order")
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
            self._phase_stack.pop()

    def finalize(self) -> FrozenDict:
        """Resolve all events after one terminal stream synchronization."""
        if self._snapshot is not None:
            return self._snapshot
        if self._active_leaf is not None or self._active_envelopes or self._phase_stack:
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
                "timing_scope": self.timing_scope,
                "envelope": name in ENVELOPE_PHASES,
                "parent_phase": PHASE_PARENTS.get(name),
                "hierarchy_kind": PHASE_HIERARCHY[name]["kind"],
                "available": name not in self._unavailable,
                "availability_reason": self._unavailable.get(
                    name,
                    "not_executed_for_route" if not samples else None,
                ),
            }

        peak_allocated = None
        peak_reserved = None
        if self.cuda_enabled:
            self.observe_peak_memory()
            peak_allocated = self._peak_allocated
            peak_reserved = self._peak_reserved

        self._finalized = True
        self._snapshot = _freeze(
            {
                "schema_version": PROFILE_SCHEMA_VERSION,
                "finalized": True,
                "request_id": self.request_id,
                "timing_scope": self.timing_scope,
                "debug_sync": self.debug_sync,
                "metadata": dict(self.metadata),
                "hierarchy": PHASE_HIERARCHY,
                "phases": phase_values,
                "counters": dict(sorted(self.counters.items())),
                "synchronization_count": len(self.synchronizations),
                "synchronizations": list(self.synchronizations),
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
            }
        )
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


@contextlib.contextmanager
def profile_many_phase(
    profilers: Any,
    name: str,
    *,
    cuda: bool = False,
) -> Iterator[None]:
    """Enter the same real operation for each attached route profiler."""
    selected = tuple(profiler for profiler in (profilers or ()) if profiler is not None)
    if len({id(profiler) for profiler in selected}) != len(selected):
        raise RuntimeError("duplicate request profiler attachment")
    with ExitStack() as stack:
        for profiler in selected:
            stack.enter_context(optional_profile_phase(profiler, name, cuda=cuda))
        yield


def profilers_from_forward_batch(
    forward_batch: Any, bid: Optional[int] = None
) -> tuple:
    """Return request-owned profilers without creating state for unprofiled work."""
    by_request = getattr(forward_batch, "region_dag_profilers_cpu", None)
    if not by_request:
        return ()
    if len(by_request) > 1:
        attached = tuple(
            profiler
            for request_profilers in by_request
            for profiler in request_profilers
        )
        if any(profiler.timing_scope != "shared_batch" for profiler in attached):
            raise RuntimeError(
                "multi-request profiling must be labeled with shared_batch timing"
            )
    if bid is not None:
        return tuple(by_request[int(bid)] or ())
    return tuple(
        profiler for request_profilers in by_request for profiler in request_profilers
    )
