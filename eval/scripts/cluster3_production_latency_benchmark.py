#!/usr/bin/env python3
"""Production-only Cluster-3 latency benchmark and report.

This path deliberately excludes the controlled validator's row hooks, tensor
traces, full-device copies, comparisons, and finite scans.  It measures the
same fixed Region-DAG workload with either one route-level CUDA event pair or
that pair plus three non-overlapping production component envelopes.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence


SCHEMA_VERSION = 1
ACCEPTED_CORRECTNESS_REVISION = "057dcab2261895a0e32357f5d57c65e1eb4b3b8c"
ROUTES = ("full_replay", "cold_handoff_build", "warm_cached_suffix")
VARIANTS = ("uninstrumented", "minimally_profiled")
CUDA_PHASES = (
    "production_prepare_cuda",
    "production_model_cuda",
    "production_state_commit_cuda",
)
WARMUP_REPETITIONS = 3
MEASURED_REPETITIONS = 10
BOOTSTRAP_REPETITIONS = 2000
MAX_PROFILING_OVERHEAD_PERCENT = 5.0
REFERENCE_EXECUTION = "canonical_segmented_no_cache"
FORBIDDEN_PRODUCTION_ENVIRONMENT = (
    "CUDA_LAUNCH_BLOCKING",
    "SGLANG_DLLM_REQUEST_METRICS",
    "SGLANG_DLLM_REQUEST_METRICS_ALL_MODELS",
    "SGLANG_HYBRID_DIFFUSION_SELF_SPEC_EXTRA_BUFFER_TRACE",
    "SGLANG_HYBRID_DIFFUSION_SELF_SPEC_DEBUG_STEPS",
    "SGLANG_HYBRID_DIFFUSION_SELF_SPEC_TRACE_PATH",
    "SGLANG_HYBRID_EXACT_HANDOFF_DEBUG",
    "SGLANG_HYBRID_EXACT_HANDOFF_DEBUG_SYNC",
)


def _load_local_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _validate_production_environment() -> None:
    enabled = []
    for name in FORBIDDEN_PRODUCTION_ENVIRONMENT:
        if name == "CUDA_LAUNCH_BLOCKING":
            forbidden = name in os.environ
        elif name.endswith("TRACE_PATH"):
            forbidden = bool(os.environ.get(name))
        else:
            forbidden = os.environ.get(name, "0") == "1"
        if forbidden:
            enabled.append(name)
    if enabled:
        raise RuntimeError(
            "production timing forbids debug/trace instrumentation environment: "
            f"{enabled}"
        )


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _hash_token_ids(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _first_differing_token_index(
    reference: Sequence[int], candidate: Sequence[int]
) -> Optional[int]:
    for index, (expected, actual) in enumerate(zip(reference, candidate)):
        if int(expected) != int(actual):
            return index
    if len(reference) != len(candidate):
        return min(len(reference), len(candidate))
    return None


def _output_mismatch_diagnostic(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    route: str,
    variant: str,
    repetition: int,
) -> dict[str, Any]:
    reference_ids = [
        int(value) for value in reference.get("generated_top1_token_ids", ())
    ]
    candidate_ids = [
        int(value) for value in candidate.get("generated_top1_token_ids", ())
    ]
    return {
        "paired_full_replay_token_ids": reference_ids,
        "candidate_token_ids": candidate_ids,
        "first_differing_flattened_token_index": _first_differing_token_index(
            reference_ids, candidate_ids
        ),
        "route": route,
        "variant": variant,
        "repetition": repetition,
        "workload_fingerprint": candidate.get("workload_fingerprint"),
        "reference_execution": candidate.get("reference_execution"),
    }


def _require_paired_output_hash(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    route: str,
    variant: str,
    repetition: int,
) -> None:
    if str(candidate.get("output_hash", "")) == str(reference.get("output_hash", "")):
        return
    diagnostic = _output_mismatch_diagnostic(
        reference,
        candidate,
        route=route,
        variant=variant,
        repetition=repetition,
    )
    raise RuntimeError(
        "production output hash differs from paired full replay: "
        f"route={route} variant={variant} repetition={repetition} "
        f"diagnostic={json.dumps(diagnostic, sort_keys=True)}"
    )


def _stats(
    samples: Sequence[float],
    *,
    seed: int,
    allow_negative: bool = False,
    expected_count: Optional[int] = None,
) -> dict[str, Any]:
    values = [float(value) for value in samples]
    expected_count = MEASURED_REPETITIONS if expected_count is None else expected_count
    if len(values) != expected_count or any(
        not math.isfinite(value) or (value < 0 and not allow_negative)
        for value in values
    ):
        qualifier = "finite" if allow_negative else "finite nonnegative"
        raise RuntimeError(
            f"latency evidence requires {expected_count} {qualifier} samples"
        )
    median = statistics.median(values)
    rng = random.Random(seed)
    medians = sorted(
        statistics.median(rng.choices(values, k=len(values)))
        for _ in range(BOOTSTRAP_REPETITIONS)
    )
    lower_index = max(0, int(0.025 * BOOTSTRAP_REPETITIONS) - 1)
    upper_index = min(BOOTSTRAP_REPETITIONS - 1, int(0.975 * BOOTSTRAP_REPETITIONS) - 1)
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "samples": values,
        "median": median,
        "p50": median,
        "p95": ordered[p95_index],
        "mad": statistics.median(abs(value - median) for value in values),
        "minimum": min(values),
        "maximum": max(values),
        "bootstrap_95_ci": [medians[lower_index], medians[upper_index]],
    }


def bootstrap_improvement_interval(
    full_samples: Sequence[float],
    warm_samples: Sequence[float],
    *,
    seed: int = 20260923,
) -> list[float]:
    full = [float(value) for value in full_samples]
    warm = [float(value) for value in warm_samples]
    if not full or len(full) != len(warm):
        raise RuntimeError("bootstrap comparison requires equal nonempty route samples")
    rng = random.Random(seed)
    improvements = sorted(
        statistics.median(rng.choices(full, k=len(full)))
        - statistics.median(rng.choices(warm, k=len(warm)))
        for _ in range(BOOTSTRAP_REPETITIONS)
    )
    lower_index = max(0, int(0.025 * BOOTSTRAP_REPETITIONS) - 1)
    upper_index = min(BOOTSTRAP_REPETITIONS - 1, int(0.975 * BOOTSTRAP_REPETITIONS) - 1)
    return [improvements[lower_index], improvements[upper_index]]


class ProductionTimingEnvelope:
    """One route-level event pair with optional non-overlapping child pairs."""

    def __init__(self, torch_module: Any, device: Any, *, profiled: bool) -> None:
        self.torch = torch_module
        self.device = device
        self.profiled = bool(profiled)
        self._active_phase: Optional[str] = None
        self._active_route = False
        self._finalized = False
        self._total_pair: Optional[tuple[Any, Any]] = None
        self._phase_pairs: dict[str, list[tuple[Any, Any]]] = {
            phase: [] for phase in CUDA_PHASES
        }
        self._scheduler_host_ms = 0.0
        self._wall_host_ms: Optional[float] = None
        self.synchronization_count = 0

    def _event(self) -> Any:
        return self.torch.cuda.Event(enable_timing=True)

    def _record(self, event: Any) -> None:
        event.record(self.torch.cuda.current_stream(device=self.device))

    @contextlib.contextmanager
    def route(self) -> Iterator[None]:
        if self._active_route or self._finalized:
            raise RuntimeError("production route timer cannot be reused")
        self._active_route = True
        started = self._event()
        finished = self._event()
        wall_started = time.perf_counter_ns()
        self._record(started)
        try:
            yield
        finally:
            if self._active_phase is not None:
                self._active_phase = None
                raise RuntimeError("production CUDA phase leaked past route exit")
            self._record(finished)
            self.torch.cuda.current_stream(device=self.device).synchronize()
            self.synchronization_count += 1
            self._wall_host_ms = (time.perf_counter_ns() - wall_started) / 1_000_000.0
            self._total_pair = (started, finished)
            self._active_route = False
            self._finalized = True

    @contextlib.contextmanager
    def cuda_phase(self, phase: str) -> Iterator[None]:
        if phase not in CUDA_PHASES:
            raise ValueError(f"unknown production CUDA phase: {phase}")
        if not self._active_route:
            raise RuntimeError("production CUDA phase requires an active route")
        if self._active_phase is not None:
            raise RuntimeError(
                "production CUDA phases must not overlap: "
                f"{self._active_phase!r} and {phase!r}"
            )
        self._active_phase = phase
        pair = None
        if self.profiled:
            pair = (self._event(), self._event())
            self._record(pair[0])
        try:
            yield
        finally:
            if pair is not None:
                self._record(pair[1])
                self._phase_pairs[phase].append(pair)
            self._active_phase = None

    @contextlib.contextmanager
    def scheduler_host(self) -> Iterator[None]:
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            self._scheduler_host_ms += (time.perf_counter_ns() - started) / 1_000_000.0

    @property
    def finalized(self) -> bool:
        return self._finalized

    def result(self) -> dict[str, Any]:
        if (
            not self._finalized
            or self._total_pair is None
            or self._wall_host_ms is None
        ):
            raise RuntimeError("production route timing is not finalized")
        total = float(self._total_pair[0].elapsed_time(self._total_pair[1]))
        phases = {
            phase: sum(float(start.elapsed_time(end)) for start, end in pairs)
            for phase, pairs in self._phase_pairs.items()
        }
        state_reason = None
        if self.profiled and not self._phase_pairs["production_state_commit_cuda"]:
            state_reason = (
                "Region-DAG state publication occurs inside production model forward; "
                "no separate post-forward state commit executes"
            )
        attributed = sum(phases.values()) if self.profiled else None
        return {
            "production_request_wall_host_ms": self._wall_host_ms,
            "production_scheduler_host_ms": self._scheduler_host_ms,
            "production_route_total_cuda_ms": total,
            "production_prepare_cuda_ms": (
                phases["production_prepare_cuda"] if self.profiled else None
            ),
            "production_model_cuda_ms": (
                phases["production_model_cuda"] if self.profiled else None
            ),
            "production_state_commit_cuda_ms": (
                phases["production_state_commit_cuda"] if self.profiled else None
            ),
            "production_state_commit_unavailable_reason": state_reason,
            "production_residual_cuda_ms": (
                total - float(attributed) if attributed is not None else None
            ),
            "synchronization_count": self.synchronization_count,
            "component_event_pairs": (
                sum(len(pairs) for pairs in self._phase_pairs.values())
                if self.profiled
                else 0
            ),
        }


def compact_output_hash_after_timing(
    outputs: Sequence[Any], timer: ProductionTimingEnvelope
) -> tuple[str, list[int]]:
    if not timer.finalized:
        raise RuntimeError("output hashing must happen after production timing")
    token_ids = []
    for output in outputs:
        value = output.detach().reshape(-1).to(device="cpu").tolist()
        token_ids.extend(int(token_id) for token_id in value)
    if not token_ids:
        raise RuntimeError("production route emitted no output tokens")
    return _hash_token_ids(token_ids), token_ids


def _query_positions(forward_batch: Any) -> tuple[int, ...]:
    positions = getattr(forward_batch, "region_dag_query_positions_cpu", None)
    if not positions or len(positions) != 1:
        raise RuntimeError(
            "production benchmark requires one request's query positions"
        )
    return tuple(int(value) for value in positions[0])


def _compact_top1(logits: Any, positions: Sequence[int], active_positions: set[int]):
    torch = __import__("torch")
    top1 = logits.argmax(dim=-1)
    rows = [
        index
        for index, position in enumerate(positions)
        if position in active_positions
    ]
    if len(rows) != len(active_positions):
        raise RuntimeError("production output does not cover every active position")
    indices = torch.tensor(rows, dtype=torch.long, device=top1.device)
    return top1.index_select(0, indices).contiguous()


def _layer_counts(runtime: Any) -> tuple[int, int]:
    config = runtime.model_runner.model_config.hf_text_config
    kinds = tuple(str(value) for value in config.layers_block_type)
    attention = sum(value == "attention" for value in kinds)
    gdn = sum(value != "attention" for value in kinds)
    if attention <= 0 or gdn <= 0:
        raise RuntimeError("production workload requires mixed attention/GDN layers")
    return attention, gdn


def _production_forward(
    runtime: Any,
    forward_batch: Any,
    timer: ProductionTimingEnvelope,
    outputs: list[Any],
    active_positions: set[int],
    work: dict[str, Any],
) -> None:
    with timer.cuda_phase("production_prepare_cuda"):
        with timer.scheduler_host():
            runtime.model_runner.attn_backend.init_forward_metadata(forward_batch)
    positions = _query_positions(forward_batch)
    with timer.cuda_phase("production_model_cuda"):
        logits = runtime.runtime._forward(forward_batch, metadata_prepared=True)
        if active_positions:
            outputs.append(_compact_top1(logits, positions, active_positions))
    attention_layers, gdn_layers = _layer_counts(runtime)
    work["query_counts"].append(len(positions))
    work["full_attention_query_token_layer_positions"] += (
        len(positions) * attention_layers
    )
    work["gdn_replay_token_layer_positions"] += len(positions) * gdn_layers


def _prepare_call(timer: ProductionTimingEnvelope, operation: Callable[[], Any]):
    with timer.cuda_phase("production_prepare_cuda"):
        with timer.scheduler_host():
            return operation()


def _untimed_frontier_setup(runtime: Any, validation: Any, case: Any, token_ids):
    started = runtime.torch.cuda.Event(enable_timing=True)
    finished = runtime.torch.cuda.Event(enable_timing=True)
    stream = runtime.torch.cuda.current_stream(device=runtime.device)
    started.record(stream)
    spec = validation.build_execution_spec(case, token_ids, edited=False)
    req, batch, _ = runtime._prepare_canonical_frontier(
        f"{case.case_id}:production-warm-setup",
        token_ids,
        spec,
        case,
        collect_component_timing=False,
    )
    runtime.model_runner.attn_backend.init_forward_metadata(batch)
    runtime.runtime._forward(batch, metadata_prepared=True)
    finished.record(stream)
    stream.synchronize()
    return req, float(started.elapsed_time(finished))


def _discard_temporary_layer_snapshots(runtime: Any) -> None:
    """Drop prefix publications while preserving the request's live continuation."""
    snapshots = getattr(runtime.backend, "_region_dag_layer_snapshots", None)
    if snapshots is None or not hasattr(snapshots, "clear"):
        raise RuntimeError(
            "canonical segmented reference requires Region-DAG layer snapshots"
        )
    snapshots.clear()


def _execute_canonical_segmented_step(
    runtime: Any,
    case: Any,
    route_identity: str,
    edited_tokens: list[int],
    edited_spec: Any,
    replay_start: int,
    expected_suffix_positions: tuple[int, ...],
    timer: ProductionTimingEnvelope,
    outputs: list[Any],
    active: set[int],
    work: dict[str, Any],
) -> tuple[Any, tuple[int, ...], tuple[int, ...]]:
    """Recompute one uncached canonical prefix plus its live suffix."""
    runtime._clear()
    req, prefix_batch, _ = _prepare_call(
        timer,
        lambda: runtime._prepare_canonical_frontier(
            route_identity,
            edited_tokens,
            edited_spec,
            case,
            collect_component_timing=False,
        ),
    )
    prefix_positions = _query_positions(prefix_batch)
    if prefix_positions != tuple(range(replay_start)):
        raise RuntimeError(
            "canonical segmented full replay changed absolute prefix positions"
        )
    _production_forward(runtime, prefix_batch, timer, [], set(), work)

    # The controlled validator removes only these temporary publications.  The
    # live recurrent continuation owned by req must remain intact for suffix B.
    _discard_temporary_layer_snapshots(runtime)
    suffix_batch, _ = _prepare_call(
        timer,
        lambda: runtime._prepare_frontier_suffix(
            req,
            edited_tokens,
            edited_spec,
            case,
            restore=False,
            collect_component_timing=False,
        ),
    )
    suffix_positions = _query_positions(suffix_batch)
    if suffix_positions != expected_suffix_positions:
        raise RuntimeError(
            "canonical segmented full replay changed absolute suffix positions"
        )
    if len(prefix_positions) + len(suffix_positions) != case.sequence_length:
        raise RuntimeError(
            "canonical segmented full replay did not execute full sequence work"
        )
    _production_forward(runtime, suffix_batch, timer, outputs, active, work)
    return req, prefix_positions, suffix_positions


def _prepare_restored_suffix(
    runtime: Any,
    req: Any,
    edited_tokens: list[int],
    edited_spec: Any,
    case: Any,
    timer: ProductionTimingEnvelope,
) -> tuple[Any, dict[str, float]]:
    """Attach a suffix to the cached frontier used by cold and warm routes."""
    return _prepare_call(
        timer,
        lambda: runtime._prepare_frontier_suffix(
            req,
            edited_tokens,
            edited_spec,
            case,
            restore=True,
            collect_component_timing=False,
        ),
    )


def _capture_prefix_reuse_observation(req: Any, forward_batch: Any) -> dict[str, Any]:
    """Retain device evidence without inspecting or synchronizing it in timing."""
    restore_flags = getattr(forward_batch, "region_dag_restore_required_cpu", None)
    restore_required = bool(
        restore_flags and len(restore_flags) == 1 and bool(restore_flags[0])
    )
    return {
        "req_pool_idx": getattr(req, "req_pool_idx", None),
        "prefix_indices": getattr(req, "prefix_indices", None),
        "restore_required": restore_required,
    }


def verify_prefix_kv_reuse_after_timing(
    runtime: Any,
    timer: ProductionTimingEnvelope,
    observations: Sequence[Mapping[str, Any]],
    *,
    replay_start: int,
    expected_restore_steps: int,
) -> dict[str, Any]:
    """Prove physical prefix reuse after the route's terminal synchronization."""
    if not timer.finalized:
        raise RuntimeError("physical KV reuse must be verified after route timing")
    if replay_start <= 0:
        raise RuntimeError("physical KV reuse requires a positive prefix boundary")
    if len(observations) != expected_restore_steps:
        raise RuntimeError(
            "restored suffix count differs from the diffusion-step contract: "
            f"observed={len(observations)} expected={expected_restore_steps}"
        )

    torch = __import__("torch")
    allocator = runtime.model_runner.token_to_kv_pool_allocator
    pool_size = int(allocator.size)
    canonical_lookup = runtime.runtime._canonical_prefix_locations
    verified_steps = 0
    for step, observation in enumerate(observations):
        if observation.get("restore_required") is not True:
            raise RuntimeError(f"restored suffix step {step} lacks its restore flag")
        req_pool_idx = observation.get("req_pool_idx")
        if req_pool_idx is None:
            raise RuntimeError(f"restored suffix step {step} has no request-pool slot")
        prefix_indices = observation.get("prefix_indices")
        if not isinstance(prefix_indices, torch.Tensor):
            raise RuntimeError(
                f"restored suffix step {step} has no prefix-index tensor"
            )
        if prefix_indices.dtype is not torch.int64:
            raise RuntimeError(
                f"restored suffix step {step} prefix indices are not int64"
            )
        if prefix_indices.ndim != 1 or int(prefix_indices.numel()) != replay_start:
            raise RuntimeError(
                f"restored suffix step {step} prefix length differs from boundary"
            )
        if not prefix_indices.is_contiguous():
            raise RuntimeError(
                f"restored suffix step {step} prefix indices are not contiguous"
            )

        canonical = canonical_lookup(int(req_pool_idx), replay_start)
        if not isinstance(canonical, torch.Tensor):
            raise RuntimeError("canonical prefix lookup did not return a tensor")
        if (
            canonical.dtype is not torch.int64
            or canonical.ndim != 1
            or int(canonical.numel()) != replay_start
            or not canonical.is_contiguous()
        ):
            raise RuntimeError("authoritative canonical prefix locations are invalid")
        if canonical.device != prefix_indices.device:
            raise RuntimeError("canonical prefix lookup changed tensor device")
        if not bool(((canonical >= 0) & (canonical < pool_size)).all().item()):
            raise RuntimeError("canonical prefix contains an invalid physical location")
        if not torch.equal(prefix_indices, canonical):
            raise RuntimeError(
                f"restored suffix step {step} changed canonical physical KV locations"
            )
        verified_steps += 1

    observed_reused_positions = replay_start * verified_steps
    return {
        "method": "canonical_prefix_page_table_identity",
        "restore_steps": verified_steps,
        "prefix_positions_per_step": replay_start,
        "physical_locations_match": True,
        "observed_reused_positions": observed_reused_positions,
    }


def execute_production_measurement(
    runtime: Any,
    validation: Any,
    case: Any,
    route: str,
    variant: str,
    repetition: int,
) -> dict[str, Any]:
    if route not in ROUTES:
        raise RuntimeError(f"invalid production route label: {route}")
    if variant not in VARIANTS:
        raise RuntimeError(f"invalid production benchmark variant: {variant}")
    _validate_production_environment()

    torch = __import__("torch")
    runtime.torch = torch
    original_tokens = runtime._tokens(case)
    active = set(case.active_positions)
    edited_tokens = list(original_tokens)
    for position in active:
        edited_tokens[position] = int(runtime.runtime.dllm_config.mask_id)
    edited_spec = validation.build_execution_spec(case, edited_tokens, edited=True)
    plan = validation.expected_plan(edited_spec, case.edited_regions)
    replay_start = int(plan.gdn_replay_start)
    if replay_start <= 0 or replay_start >= case.sequence_length:
        raise RuntimeError(
            "production cached routes require an internal replay frontier"
        )

    runtime._clear()
    warm_req = None
    warm_cache_build_cuda_ms = None
    if route == "warm_cached_suffix":
        warm_req, warm_cache_build_cuda_ms = _untimed_frontier_setup(
            runtime, validation, case, original_tokens
        )

    torch.cuda.reset_peak_memory_stats(runtime.device)
    timer = ProductionTimingEnvelope(
        torch,
        runtime.device,
        profiled=variant == "minimally_profiled",
    )
    outputs: list[Any] = []
    work: dict[str, Any] = {
        "query_counts": [],
        "full_attention_query_token_layer_positions": 0,
        "gdn_replay_token_layer_positions": 0,
    }
    instrumentation_kv_cache_hits = 0
    gdn_restores = 0
    recovery_replays = 0
    fallback_count = 0
    positions_preserved = True
    stable_queries_absent = True
    full_replay_processed_prefix = route == "full_replay"
    full_replay_prefix_positions_preserved = route == "full_replay"
    full_replay_suffix_positions_preserved = route == "full_replay"
    full_replay_full_sequence_work = route == "full_replay"
    restore_contract_observed = True
    prefix_reuse_observations: list[dict[str, Any]] = []
    cold_req = None

    with timer.route():
        if route == "cold_handoff_build":
            base_spec = validation.build_execution_spec(
                case, original_tokens, edited=False
            )
            cold_req, cold_batch, _ = _prepare_call(
                timer,
                lambda: runtime._prepare_canonical_frontier(
                    f"{case.case_id}:{route}:{variant}:{repetition}:frontier",
                    original_tokens,
                    base_spec,
                    case,
                    collect_component_timing=False,
                ),
            )
            _production_forward(runtime, cold_batch, timer, [], set(), work)

        for step in range(case.diffusion_steps):
            if route == "full_replay":
                req, prefix_positions, suffix_positions = (
                    _execute_canonical_segmented_step(
                        runtime,
                        case,
                        f"{case.case_id}:{route}:{variant}:{repetition}:{step}",
                        edited_tokens,
                        edited_spec,
                        replay_start,
                        tuple(plan.attention_query_positions),
                        timer,
                        outputs,
                        active,
                        work,
                    )
                )
                full_replay_processed_prefix = (
                    full_replay_processed_prefix
                    and prefix_positions == tuple(range(replay_start))
                )
                full_replay_prefix_positions_preserved = (
                    full_replay_prefix_positions_preserved
                    and prefix_positions == tuple(range(replay_start))
                )
                full_replay_suffix_positions_preserved = (
                    full_replay_suffix_positions_preserved
                    and suffix_positions == tuple(plan.attention_query_positions)
                )
                full_replay_full_sequence_work = (
                    full_replay_full_sequence_work
                    and len(prefix_positions) + len(suffix_positions)
                    == case.sequence_length
                )
                positions_preserved = (
                    positions_preserved
                    and full_replay_prefix_positions_preserved
                    and full_replay_suffix_positions_preserved
                )
            else:
                req = warm_req if route == "warm_cached_suffix" else cold_req
                if req is None:
                    raise RuntimeError("production suffix route has no valid frontier")
                batch, _ = _prepare_restored_suffix(
                    runtime,
                    req,
                    edited_tokens,
                    edited_spec,
                    case,
                    timer,
                )
                positions = _query_positions(batch)
                stable_queries_absent = stable_queries_absent and all(
                    position >= replay_start for position in positions
                )
                positions_preserved = positions_preserved and positions == tuple(
                    plan.attention_query_positions
                )
                restore_flags = getattr(batch, "region_dag_restore_required_cpu", None)
                restore_contract_observed = (
                    restore_contract_observed
                    and bool(restore_flags)
                    and len(restore_flags) == 1
                    and bool(restore_flags[0])
                )
                prefix_reuse_observations.append(
                    _capture_prefix_reuse_observation(req, batch)
                )
                if bool(restore_flags) and bool(restore_flags[0]):
                    _, gdn_layers = _layer_counts(runtime)
                    gdn_restores += gdn_layers
                _production_forward(runtime, batch, timer, outputs, active, work)
            metrics = req.region_dag_instrumentation
            if metrics is None:
                raise RuntimeError("production request lost Region-DAG instrumentation")
            instrumentation_kv_cache_hits += int(metrics.kv_cache_hits)
            recovery_replays += int(metrics.recovery_replays)
            fallback_count += int(metrics.fallback_count)

    timing = timer.result()
    if route in ("cold_handoff_build", "warm_cached_suffix"):
        kv_reuse_evidence = verify_prefix_kv_reuse_after_timing(
            runtime,
            timer,
            prefix_reuse_observations,
            replay_start=replay_start,
            expected_restore_steps=case.diffusion_steps,
        )
    else:
        if prefix_reuse_observations:
            raise RuntimeError("full replay unexpectedly recorded restored KV state")
        kv_reuse_evidence = {
            "method": "not_applicable_full_replay",
            "restore_steps": 0,
            "prefix_positions_per_step": 0,
            "physical_locations_match": True,
            "observed_reused_positions": 0,
        }
    cache_hits = int(kv_reuse_evidence["observed_reused_positions"])
    output_hash, top1_ids = compact_output_hash_after_timing(outputs, timer)
    peak_memory = int(torch.cuda.max_memory_allocated(runtime.device))
    expected_query_positions = {
        "full_replay": case.sequence_length * case.diffusion_steps,
        "cold_handoff_build": replay_start
        + len(plan.attention_query_positions) * case.diffusion_steps,
        "warm_cached_suffix": len(plan.attention_query_positions)
        * case.diffusion_steps,
    }[route]
    executed_query_positions = sum(int(value) for value in work["query_counts"])
    attention_layers, gdn_layers = _layer_counts(runtime)
    logical_active_attention_work = (
        len(active) * case.diffusion_steps * attention_layers
    )
    work_accounting_matches = (
        executed_query_positions == expected_query_positions
        and work["full_attention_query_token_layer_positions"]
        == expected_query_positions * attention_layers
        and work["gdn_replay_token_layer_positions"]
        == expected_query_positions * gdn_layers
    )
    if not work_accounting_matches:
        raise RuntimeError("production route token-layer accounting is inconsistent")
    workload_fingerprint = _sha256_json(
        {
            "input_token_hash": _hash_token_ids(edited_tokens),
            "active_positions": sorted(active),
            "diffusion_steps": case.diffusion_steps,
            "attention_contract": edited_spec.attention_contract_id,
            "reference_execution": REFERENCE_EXECUTION,
        }
    )
    if route == "warm_cached_suffix":
        if replay_start > 0 and cache_hits <= 0:
            raise RuntimeError("warm cached route did not reuse the stable prefix")
        if int(kv_reuse_evidence["restore_steps"]) != case.diffusion_steps:
            raise RuntimeError("warm cached route did not restore every suffix step")
        if recovery_replays or fallback_count:
            raise RuntimeError("warm cached route recovered or fell back")
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": "production_efficiency",
        "route": route,
        "variant": variant,
        "repetition_index": repetition,
        "timing": timing,
        "output_hash": output_hash,
        "generated_top1_token_ids": top1_ids,
        "cache_status": "hit" if route == "warm_cached_suffix" else "miss",
        "cache_hits": cache_hits,
        "instrumentation_kv_cache_hits": instrumentation_kv_cache_hits,
        "instrumentation_kv_cache_hits_unavailable_reason": (
            (
                "canonical prepare_for_extend does not increment the legacy "
                "prepare_for_region_dag_replay counter"
            )
            if route in ("cold_handoff_build", "warm_cached_suffix")
            else "not applicable to canonical segmented full replay"
        ),
        "kv_reuse_evidence": kv_reuse_evidence,
        "cache_misses": sum(int(value) for value in work["query_counts"]),
        "gdn_state_restores": gdn_restores,
        "recovery_replay_count": recovery_replays,
        "fallback_count": fallback_count,
        "active_token_count": len(active),
        "active_region_count": len(edited_spec.active_regions),
        "reusable_prefix_positions": replay_start,
        "full_attention_active_token_layer_positions": logical_active_attention_work,
        "full_attention_query_token_layer_positions": work[
            "full_attention_query_token_layer_positions"
        ],
        "gdn_replay_token_layer_positions": work["gdn_replay_token_layer_positions"],
        "peak_allocated_gpu_memory_bytes": peak_memory,
        "cache_build_cuda_ms": warm_cache_build_cuda_ms,
        "workload_fingerprint": workload_fingerprint,
        "reference_execution": REFERENCE_EXECUTION,
        "attention_contract": edited_spec.attention_contract_id,
        "diffusion_steps": case.diffusion_steps,
        "positions_preserved": positions_preserved,
        "stable_queries_absent": stable_queries_absent,
        "full_replay_processed_prefix": full_replay_processed_prefix,
        "full_replay_prefix_positions_preserved": (
            full_replay_prefix_positions_preserved
        ),
        "full_replay_suffix_positions_preserved": (
            full_replay_suffix_positions_preserved
        ),
        "full_replay_full_sequence_work": full_replay_full_sequence_work,
        "executed_query_positions": executed_query_positions,
        "expected_query_positions": expected_query_positions,
        "work_accounting_matches": work_accounting_matches,
        "warm_restored_valid_prefix": (
            route != "warm_cached_suffix"
            or (replay_start > 0 and cache_hits > 0 and restore_contract_observed)
        ),
        "trace_hooks_installed": False,
    }


def _route_records(
    records: Sequence[Mapping[str, Any]],
    route: str,
    variant: str,
    *,
    expected_count: int = MEASURED_REPETITIONS,
):
    values = [
        record
        for record in records
        if record.get("route") == route and record.get("variant") == variant
    ]
    if len(values) != expected_count:
        raise RuntimeError(
            f"{route}/{variant} requires exactly {expected_count} production measurements"
        )
    if {int(value.get("repetition_index", -1)) for value in values} != set(
        range(expected_count)
    ):
        raise RuntimeError(f"{route}/{variant} has invalid repetition indices")
    if any(
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("profile") != "production_efficiency"
        for value in values
    ):
        raise RuntimeError(f"{route}/{variant} has invalid production record identity")
    return sorted(values, key=lambda value: int(value["repetition_index"]))


def summarize(
    records: Sequence[Mapping[str, Any]],
    *,
    revision: str,
    hardware: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    correctness_prerequisite_pass: bool,
    normalized_case: Optional[Mapping[str, Any]] = None,
    measured_repetitions: int = MEASURED_REPETITIONS,
    warmups: int = WARMUP_REPETITIONS,
    routes: Sequence[str] = ROUTES,
) -> dict[str, Any]:
    routes = tuple(routes)
    if not routes or any(route not in ROUTES for route in routes):
        raise RuntimeError("summary routes are empty or invalid")
    if any(record.get("route") not in ROUTES for record in records):
        raise RuntimeError("production evidence contains an incorrect route label")
    if any(record.get("variant") not in VARIANTS for record in records):
        raise RuntimeError("production evidence contains an incorrect variant label")

    route_summaries = {}
    overhead = {}
    all_hashes = set()
    all_sync_counts = []
    all_workloads_identical = True
    all_workload_fingerprints = set()
    for route_index, route in enumerate(routes):
        uninstrumented = _route_records(
            records,
            route,
            "uninstrumented",
            expected_count=measured_repetitions,
        )
        profiled = _route_records(
            records,
            route,
            "minimally_profiled",
            expected_count=measured_repetitions,
        )
        route_hashes = {
            str(record["output_hash"]) for record in (*uninstrumented, *profiled)
        }
        all_hashes.update(route_hashes)
        fingerprints = {
            str(record["workload_fingerprint"])
            for record in (*uninstrumented, *profiled)
        }
        all_workload_fingerprints.update(fingerprints)
        workload_signatures = {
            (
                str(record["workload_fingerprint"]),
                str(record["reference_execution"]),
                int(record["active_token_count"]),
                int(record["full_attention_query_token_layer_positions"]),
                int(record["gdn_replay_token_layer_positions"]),
                str(record["attention_contract"]),
                int(record["diffusion_steps"]),
            )
            for record in (*uninstrumented, *profiled)
        }
        all_workloads_identical = (
            all_workloads_identical
            and len(fingerprints) == 1
            and len(workload_signatures) == 1
        )
        uninstrumented_cuda = [
            float(record["timing"]["production_route_total_cuda_ms"])
            for record in uninstrumented
        ]
        profiled_cuda = [
            float(record["timing"]["production_route_total_cuda_ms"])
            for record in profiled
        ]
        uninstrumented_stats = _stats(
            uninstrumented_cuda,
            seed=101 + route_index,
            expected_count=measured_repetitions,
        )
        profiled_stats = _stats(
            profiled_cuda,
            seed=111 + route_index,
            expected_count=measured_repetitions,
        )
        if uninstrumented_stats["median"] <= 0:
            raise RuntimeError("production CUDA latency median must be positive")
        overhead[route] = (
            100.0
            * (profiled_stats["median"] - uninstrumented_stats["median"])
            / uninstrumented_stats["median"]
        )
        all_sync_counts.extend(
            int(record["timing"]["synchronization_count"])
            for record in (*uninstrumented, *profiled)
        )
        route_summaries[route] = {
            "uninstrumented_wall_latency_ms": _stats(
                [
                    float(record["timing"]["production_request_wall_host_ms"])
                    for record in uninstrumented
                ],
                seed=121 + route_index,
                expected_count=measured_repetitions,
            ),
            "uninstrumented_cuda_latency_ms": uninstrumented_stats,
            "minimally_profiled_cuda_latency_ms": profiled_stats,
            "production_prepare_cuda_ms": _stats(
                [
                    float(record["timing"]["production_prepare_cuda_ms"])
                    for record in profiled
                ],
                seed=131 + route_index,
                expected_count=measured_repetitions,
            ),
            "production_model_cuda_ms": _stats(
                [
                    float(record["timing"]["production_model_cuda_ms"])
                    for record in profiled
                ],
                seed=141 + route_index,
                expected_count=measured_repetitions,
            ),
            "production_state_commit_cuda_ms": _stats(
                [
                    float(record["timing"]["production_state_commit_cuda_ms"])
                    for record in profiled
                ],
                seed=151 + route_index,
                expected_count=measured_repetitions,
            ),
            "production_residual_cuda_ms": _stats(
                [
                    float(record["timing"]["production_residual_cuda_ms"])
                    for record in profiled
                ],
                seed=161 + route_index,
                allow_negative=True,
                expected_count=measured_repetitions,
            ),
            "production_scheduler_host_ms": _stats(
                [
                    float(record["timing"]["production_scheduler_host_ms"])
                    for record in profiled
                ],
                seed=171 + route_index,
                expected_count=measured_repetitions,
            ),
            "peak_allocated_gpu_memory_bytes": _stats(
                [
                    float(record["peak_allocated_gpu_memory_bytes"])
                    for record in uninstrumented
                ],
                seed=181 + route_index,
                expected_count=measured_repetitions,
            ),
            "output_hash": next(iter(route_hashes)) if len(route_hashes) == 1 else None,
            "cache_hits": [int(record["cache_hits"]) for record in uninstrumented],
            "instrumentation_kv_cache_hits": [
                int(record["instrumentation_kv_cache_hits"])
                for record in uninstrumented
            ],
            "instrumentation_kv_cache_hits_unavailable_reason": str(
                uninstrumented[0]["instrumentation_kv_cache_hits_unavailable_reason"]
            ),
            "kv_reuse_evidence": [
                dict(record["kv_reuse_evidence"]) for record in uninstrumented
            ],
            "cache_misses": [int(record["cache_misses"]) for record in uninstrumented],
            "recovery_replay_count": [
                int(record["recovery_replay_count"]) for record in uninstrumented
            ],
            "fallback_count": [
                int(record["fallback_count"]) for record in uninstrumented
            ],
            "active_token_count": int(uninstrumented[0]["active_token_count"]),
            "active_region_count": int(uninstrumented[0].get("active_region_count", 1)),
            "reusable_prefix_positions": int(
                uninstrumented[0].get("reusable_prefix_positions", 2048)
            ),
            "full_attention_active_token_layer_positions": [
                int(
                    record.get(
                        "full_attention_active_token_layer_positions",
                        record["full_attention_query_token_layer_positions"],
                    )
                )
                for record in uninstrumented
            ],
            "full_attention_query_token_layer_positions": [
                int(record["full_attention_query_token_layer_positions"])
                for record in uninstrumented
            ],
            "gdn_replay_token_layer_positions": [
                int(record["gdn_replay_token_layer_positions"])
                for record in uninstrumented
            ],
            "profiling_overhead_pct": overhead[route],
            "production_state_commit_unavailable_reason": profiled[0]["timing"].get(
                "production_state_commit_unavailable_reason"
            ),
            "workload_fingerprint": (
                next(iter(fingerprints)) if len(fingerprints) == 1 else None
            ),
            "reference_execution": str(uninstrumented[0]["reference_execution"]),
            "cache_build_cuda_ms": (
                _stats(
                    [float(record["cache_build_cuda_ms"]) for record in uninstrumented],
                    seed=191 + route_index,
                    expected_count=measured_repetitions,
                )
                if all(
                    record.get("cache_build_cuda_ms") is not None
                    for record in uninstrumented
                )
                else None
            ),
        }

    first_record = records[0]
    diffusion_steps = int(first_record["diffusion_steps"])
    active_tokens = int(first_record["active_token_count"])
    reusable_prefix = int(first_record.get("reusable_prefix_positions", 2048))
    full = route_summaries["full_replay"]["uninstrumented_cuda_latency_ms"]
    cold = (
        route_summaries["cold_handoff_build"]["uninstrumented_cuda_latency_ms"]
        if "cold_handoff_build" in route_summaries
        else None
    )
    warm = (
        route_summaries["warm_cached_suffix"]["uninstrumented_cuda_latency_ms"]
        if "warm_cached_suffix" in route_summaries
        else None
    )
    improvement_ci = (
        bootstrap_improvement_interval(full["samples"], warm["samples"])
        if warm is not None
        else None
    )
    reliable = improvement_ci is not None and improvement_ci[0] > 0.0
    warm_speedup = full["median"] / warm["median"] if warm is not None else None
    warm_reduction = (
        100.0 * (full["median"] - warm["median"]) / full["median"]
        if warm is not None
        else None
    )
    per_step_saving = (
        (full["median"] - warm["median"]) / diffusion_steps
        if warm is not None
        else None
    )
    cold_extra = (
        cold["median"] - warm["median"]
        if cold is not None and warm is not None
        else None
    )
    cold_amortization = (
        cold_extra / per_step_saving
        if cold_extra is not None
        and per_step_saving is not None
        and per_step_saving > 0
        else None
    )

    warm_records = (
        _route_records(
            records,
            "warm_cached_suffix",
            "uninstrumented",
            expected_count=measured_repetitions,
        )
        if "warm_cached_suffix" in routes
        else []
    )
    output_hashes_identical = (
        len(all_hashes) == 1
        and len(next(iter(all_hashes))) == 64
        and all(character in "0123456789abcdef" for character in next(iter(all_hashes)))
    )
    zero_warm_recovery = all(
        int(record["recovery_replay_count"]) == 0 for record in warm_records
    )
    zero_fallback = all(int(record["fallback_count"]) == 0 for record in records)
    one_sync = all(value == 1 for value in all_sync_counts)
    overhead_ok = all(
        value <= MAX_PROFILING_OVERHEAD_PERCENT for value in overhead.values()
    )
    sanity = {
        "full_replay_processes_prefix": all(
            bool(record["full_replay_processed_prefix"])
            for record in _route_records(
                records,
                "full_replay",
                "uninstrumented",
                expected_count=measured_repetitions,
            )
        ),
        "full_replay_prefix_positions_preserved": all(
            bool(record["full_replay_prefix_positions_preserved"])
            for record in _route_records(
                records,
                "full_replay",
                "uninstrumented",
                expected_count=measured_repetitions,
            )
        ),
        "full_replay_suffix_positions_preserved": all(
            bool(record["full_replay_suffix_positions_preserved"])
            for record in _route_records(
                records,
                "full_replay",
                "uninstrumented",
                expected_count=measured_repetitions,
            )
        ),
        "full_replay_executes_full_sequence_each_step": all(
            bool(record["full_replay_full_sequence_work"])
            for record in _route_records(
                records,
                "full_replay",
                "uninstrumented",
                expected_count=measured_repetitions,
            )
        ),
        "route_work_accounting_matches": all(
            bool(record["work_accounting_matches"])
            and int(record["executed_query_positions"])
            == int(record["expected_query_positions"])
            for record in records
        ),
        "warm_restores_valid_prefix": all(
            bool(record["warm_restored_valid_prefix"]) for record in warm_records
        ),
        "physical_kv_reuse_proven": all(
            record["kv_reuse_evidence"].get("method")
            == "canonical_prefix_page_table_identity"
            and record["kv_reuse_evidence"].get("physical_locations_match") is True
            and int(record["kv_reuse_evidence"].get("restore_steps", 0))
            == diffusion_steps
            and int(record["kv_reuse_evidence"].get("prefix_positions_per_step", 0))
            == reusable_prefix
            and int(record["kv_reuse_evidence"].get("observed_reused_positions", 0))
            == reusable_prefix * diffusion_steps
            and int(record["cache_hits"]) == reusable_prefix * diffusion_steps
            for record in records
            if record["route"] in ("cold_handoff_build", "warm_cached_suffix")
        ),
        "stable_queries_absent_from_warm": all(
            bool(record["stable_queries_absent"]) for record in warm_records
        ),
        "active_positions_preserved": all(
            bool(record["positions_preserved"]) for record in records
        ),
        "same_attention_contract_and_steps": len(
            {
                (record["attention_contract"], int(record["diffusion_steps"]))
                for record in records
            }
        )
        == 1,
        "normalized_production_workload": all(
            record["attention_contract"] == "region_dag_conservative_gdn_v1"
            and record["reference_execution"] == REFERENCE_EXECUTION
            and int(record["diffusion_steps"]) == diffusion_steps
            and int(record["active_token_count"]) == active_tokens
            and len(record["generated_top1_token_ids"])
            == active_tokens * diffusion_steps
            for record in records
        ),
        "same_tokens_and_spec_across_routes": len(all_workload_fingerprints) == 1,
        "profiled_and_uninstrumented_workloads_identical": all_workloads_identical,
        "trace_hooks_absent": all(
            record.get("trace_hooks_installed") is False for record in records
        ),
        "uninstrumented_component_events_absent": all(
            int(record["timing"]["component_event_pairs"]) == 0
            for record in records
            if record["variant"] == "uninstrumented"
        ),
        "state_commit_absence_explained": all(
            float(record["timing"]["production_state_commit_cuda_ms"]) == 0.0
            and bool(record["timing"].get("production_state_commit_unavailable_reason"))
            for record in records
            if record["variant"] == "minimally_profiled"
        ),
    }
    gates = {
        "output_hashes_identical": output_hashes_identical,
        "zero_warm_recovery_replay": zero_warm_recovery,
        "zero_fallback": zero_fallback,
        "one_sync_per_route": one_sync,
        "profiling_overhead_acceptable": overhead_ok,
        "statistically_reliable_improvement": reliable,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "revision": revision,
        "hardware": dict(hardware),
        "checkpoint": dict(checkpoint),
        "correctness_prerequisite_pass": bool(correctness_prerequisite_pass),
        "timing_protocol": {
            "warmups": warmups,
            "measured_repetitions": measured_repetitions,
            "cuda_events": True,
            "debug_synchronization": False,
            "headline_metric": "uninstrumented_cuda_latency_ms",
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "batch_size": int((normalized_case or {}).get("batch_size", 1)),
            "prefix_tokens": reusable_prefix,
            "active_tokens": active_tokens,
            "diffusion_steps": diffusion_steps,
            "selected_routes": list(routes),
            "reference_execution": REFERENCE_EXECUTION,
        },
        "routes": route_summaries,
        "normalized_case": dict(normalized_case or {}),
        "comparisons": {
            "warm_speedup": warm_speedup,
            "warm_latency_reduction_pct": warm_reduction,
            "cold_amortization_steps": cold_amortization,
            "profiling_overhead_pct": overhead,
            "warm_improvement_ms_bootstrap_95_ci": improvement_ci,
            "speedup_claim_supported": reliable,
        },
        "workload_sanity": sanity,
        "gates": gates,
        "publication_acceptable": (
            bool(correctness_prerequisite_pass)
            and all(sanity.values())
            and output_hashes_identical
            and zero_warm_recovery
            and zero_fallback
            and one_sync
            and overhead_ok
        ),
        "unsupported_claims": [
            "This benchmark implements no runtime optimization.",
            "A speedup claim is unsupported when the improvement confidence interval includes zero.",
        ],
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"artifact is not a JSON object: {path}")
    return value


def validate_correctness_prerequisite(
    correctness: Mapping[str, Any],
    preflight: Mapping[str, Any],
    actual_checkpoint: Mapping[str, Any],
    hardware: Mapping[str, Any],
) -> None:
    if (
        correctness.get("cluster3_revision") != ACCEPTED_CORRECTNESS_REVISION
        or correctness.get("profile") != "efficiency_one"
        or correctness.get("strict_pass") is not True
    ):
        raise RuntimeError(
            "correctness prerequisite is not the accepted efficiency_one run"
        )
    repository = preflight.get("repository") or {}
    environment = preflight.get("environment") or {}
    correctness_hardware = correctness.get("hardware") or {}
    if repository.get("head") != ACCEPTED_CORRECTNESS_REVISION:
        raise RuntimeError(
            "preflight does not belong to the accepted correctness revision"
        )
    if preflight.get("checkpoint_hashes_match") is not True or not preflight.get(
        "a30_required"
    ):
        raise RuntimeError("preflight did not validate the frozen checkpoint on A30")
    gpus = environment.get("gpus") or []
    if len(gpus) != 1 or gpus[0].get("name") != "NVIDIA A30":
        raise RuntimeError("correctness prerequisite hardware is not one NVIDIA A30")
    if correctness_hardware.get(
        "device_name"
    ) != "NVIDIA A30" or correctness_hardware.get("cuda_version") != environment.get(
        "cuda"
    ):
        raise RuntimeError("correctness artifact hardware differs from its preflight")
    if (
        hardware.get("device_name") != "NVIDIA A30"
        or hardware.get("visible_device_count") != 1
    ):
        raise RuntimeError("production benchmark hardware is not NVIDIA A30")
    if environment.get("cuda") != hardware.get("cuda_version"):
        raise RuntimeError("production CUDA version differs from correctness preflight")
    if environment.get("pytorch") != hardware.get("pytorch_version"):
        raise RuntimeError(
            "production PyTorch version differs from correctness preflight"
        )
    expected_files = (preflight.get("checkpoint") or {}).get("files") or {}
    observed_files = actual_checkpoint.get("files") or {}
    expected_hashes = {
        name: details.get("sha256") for name, details in expected_files.items()
    }
    observed_hashes = {
        name: details.get("sha256") for name, details in observed_files.items()
    }
    if not expected_hashes or expected_hashes != observed_hashes:
        raise RuntimeError("production checkpoint differs from correctness preflight")


def production_hardware(device: int) -> dict[str, Any]:
    torch = __import__("torch")
    properties = torch.cuda.get_device_properties(device)
    device_count = int(torch.cuda.device_count())
    try:
        driver = subprocess.check_output(
            (
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
                f"--id={device}",
            ),
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        driver = "unavailable"
    return {
        "device_index": int(device),
        "device_name": properties.name,
        "visible_device_count": device_count,
        "visible_device_names": [
            torch.cuda.get_device_properties(index).name
            for index in range(device_count)
        ],
        "total_memory_bytes": int(properties.total_memory),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "driver_version": driver,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
    }


def run_production_benchmark(
    args: Any,
    *,
    validation_module: Any,
    runtime_factory: Optional[Callable[[Any], Any]] = None,
    measurement_runner: Callable[
        ..., Mapping[str, Any]
    ] = execute_production_measurement,
) -> dict[str, Any]:
    if args.profile != "production_efficiency":
        raise RuntimeError(
            "production benchmark requires --profile production_efficiency"
        )
    measured_repetitions = int(args.timed_repetitions)
    warmups = int(getattr(args, "warmups", WARMUP_REPETITIONS))
    routes = tuple(getattr(args, "routes", ROUTES))
    if measured_repetitions <= 0 or warmups < 0:
        raise RuntimeError(
            "production repetitions must be positive and warmups nonnegative"
        )
    if (
        not routes
        or len(set(routes)) != len(routes)
        or any(route not in ROUTES for route in routes)
    ):
        raise RuntimeError("production routes are empty, duplicated, or invalid")
    if "full_replay" not in routes:
        raise RuntimeError("production route comparison requires full_replay")
    if args.dtype != "bfloat16" or int(args.tp_size) != 1:
        raise RuntimeError("production benchmark requires BF16 and TP=1")
    if bool(args.debug_sync_stages):
        raise RuntimeError("production benchmark forbids debug synchronization")
    _validate_production_environment()

    # Parse and validate all cases before checkpoint inspection or runtime/model load.
    manifest_module = _load_local_module(
        "cluster3_benchmark_manifest",
        Path(__file__).with_name("cluster3_benchmark_manifest.py"),
    )
    case_manifest = getattr(args, "case_manifest", None)
    if case_manifest is None:
        if hasattr(validation_module, "RegionShape"):
            cases, normalized_cases = manifest_module.legacy_production_case(
                validation_module, seed=int(args.seed), requested_routes=routes
            )
        else:
            # Compatibility for injected unit-test validation doubles.  Real
            # execution always uses the concrete validation module above.
            cases = validation_module.build_manifest(args.profile, args.seed)
            normalized_cases = [
                {
                    "schema_version": 1,
                    "case_id": "p2048-a64-s4-b1",
                    "total_tokens_per_request": 2112,
                    "prefix_tokens": 2048,
                    "active_spans": [[2048, 2112]],
                    "diffusion_steps": 4,
                    "batch_size": 1,
                }
            ]
    else:
        cases, normalized_cases = manifest_module.load_manifest(
            Path(case_manifest),
            validation_module=validation_module,
            seed=int(args.seed),
            requested_routes=routes,
        )
    required_capacity = max(
        int(value["total_tokens_per_request"]) * int(value["batch_size"])
        for value in normalized_cases
    )
    if int(args.max_total_tokens) < required_capacity:
        raise RuntimeError(
            "production benchmark token capacity is too small: "
            f"required={required_capacity} configured={args.max_total_tokens}"
        )

    preflight_module = validation_module._load_module(
        "cluster3_production_preflight",
        Path(__file__).with_name("cluster3_efficiency_preflight.py"),
    )
    actual_checkpoint = preflight_module.checkpoint_identity(Path(args.model_path))
    correctness = _read_json(Path(args.correctness_artifact))
    preflight = _read_json(Path(args.preflight_json))
    hardware = production_hardware(int(args.device))
    validate_correctness_prerequisite(
        correctness, preflight, actual_checkpoint, hardware
    )

    factory = runtime_factory or validation_module.Cluster3ValidationRuntime
    runtime = factory(args)
    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    full_replay_records: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    primary_error: Optional[BaseException] = None
    try:
        with output_path.open("w", encoding="utf-8") as destination:
            for case, normalized_case in zip(cases, normalized_cases):
                case_id = str(normalized_case["case_id"])
                for route in routes:
                    for repetition in range(-warmups, measured_repetitions):
                        for variant in VARIANTS:
                            measured = measurement_runner(
                                runtime,
                                validation_module,
                                case,
                                route,
                                variant,
                                repetition,
                            )
                            if repetition < 0:
                                continue
                            record = dict(measured)
                            record["case_id"] = case_id
                            record["normalized_case"] = dict(normalized_case)
                            comparison_key = (case_id, variant, repetition)
                            if route == "full_replay":
                                full_replay_records[comparison_key] = record
                            else:
                                _require_paired_output_hash(
                                    full_replay_records.get(comparison_key, {}),
                                    record,
                                    route=route,
                                    variant=variant,
                                    repetition=repetition,
                                )
                            records.append(record)
                            destination.write(json.dumps(record, sort_keys=True) + "\n")
                            destination.flush()
    except BaseException as exc:
        primary_error = exc
    finally:
        try:
            runtime.close()
        except BaseException as exc:
            if primary_error is None:
                primary_error = exc
    if primary_error is not None:
        raise primary_error

    revision = validation_module._git_revision()
    case_summaries = []
    for case, normalized_case in zip(cases, normalized_cases):
        case_id = str(normalized_case["case_id"])
        case_records = [record for record in records if record["case_id"] == case_id]
        case_summary = summarize(
            case_records,
            revision=revision,
            hardware=hardware,
            checkpoint=actual_checkpoint,
            correctness_prerequisite_pass=True,
            normalized_case=normalized_case,
            measured_repetitions=measured_repetitions,
            warmups=warmups,
            routes=routes,
        )
        case_summaries.append(case_summary)
    if len(case_summaries) == 1:
        summary = case_summaries[0]
    else:
        summary = {
            "schema_version": SCHEMA_VERSION,
            "revision": revision,
            "hardware": dict(hardware),
            "checkpoint": dict(actual_checkpoint),
            "case_count": len(case_summaries),
            "case_ids": [value["case_id"] for value in normalized_cases],
            "cases": case_summaries,
            "publication_acceptable": all(
                value["publication_acceptable"] for value in case_summaries
            ),
        }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not summary["publication_acceptable"]:
        raise RuntimeError("production benchmark failed a publication acceptance gate")
    return summary
