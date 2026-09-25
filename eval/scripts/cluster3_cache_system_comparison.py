#!/usr/bin/env python3
"""Fail-closed same-runtime cache-system comparison for Cluster 3.

Manifest and method availability are validated before checkpoint inspection or
model construction.  Unsupported labels never produce repetition records.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence


SCHEMA_VERSION = 1
BASE_REVISION = "c57970dca909ac8fa77260ea75da1580797c22ad"
ORIGINAL_FLARE_REVISION = "6ca547aebb72bfe897e80e0a683c776789e5f38c"
METHODS = (
    "controlled_full_replay",
    "upstream_sglang_radix",
    "safe_kv_only_diffusion",
    "kv_gdn_handoff_full_rows",
    "complete_region_state_execution",
)
VARIANTS = ("uninstrumented", "minimally_profiled")
MIN_COMPONENT_COVERAGE = 0.90

METHOD_REGISTRY = {
    "controlled_full_replay": {
        "method": "controlled_full_replay",
        "supported": True,
        "availability_reason": None,
        "semantic_contract": "same_runtime_canonical_segmented_full_replay",
        "uses_attention_kv_reuse": False,
        "uses_gdn_restore": False,
        "submits_stable_query_rows": True,
        "submits_stable_mlp_rows": True,
        "uses_region_dag": True,
    },
    "upstream_sglang_radix": {
        "method": "upstream_sglang_radix",
        "supported": False,
        "availability_reason": "native_radix_lacks_exact_hybrid_recurrent_state",
        "semantic_contract": "bundled_native_sglang_radix_behavior",
        "uses_attention_kv_reuse": True,
        "uses_gdn_restore": False,
        "submits_stable_query_rows": False,
        "submits_stable_mlp_rows": False,
        "uses_region_dag": False,
    },
    "safe_kv_only_diffusion": {
        "method": "safe_kv_only_diffusion",
        "supported": True,
        "availability_reason": None,
        "semantic_contract": "retained_attention_kv_with_fresh_full_prefix_gdn_reconstruction",
        "uses_attention_kv_reuse": True,
        "uses_gdn_restore": False,
        "submits_stable_query_rows": True,
        "submits_stable_mlp_rows": True,
        "uses_region_dag": True,
    },
    "kv_gdn_handoff_full_rows": {
        "method": "kv_gdn_handoff_full_rows",
        "supported": False,
        "availability_reason": "state_handoff_and_active_rows_structurally_coupled",
        "semantic_contract": "exact_kv_gdn_handoff_followed_by_full_token_rows",
        "uses_attention_kv_reuse": True,
        "uses_gdn_restore": True,
        "submits_stable_query_rows": True,
        "submits_stable_mlp_rows": True,
        "uses_region_dag": True,
    },
    "complete_region_state_execution": {
        "method": "complete_region_state_execution",
        "supported": True,
        "availability_reason": None,
        "semantic_contract": "validated_exact_region_state_conservative_suffix_execution",
        "uses_attention_kv_reuse": True,
        "uses_gdn_restore": True,
        "submits_stable_query_rows": False,
        "submits_stable_mlp_rows": False,
        "uses_region_dag": True,
    },
}


def _load_local_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-total-tokens", type=int, default=4096)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--timed-repetitions", type=int, default=10)
    parser.add_argument("--correctness-artifact", type=Path, required=True)
    parser.add_argument("--preflight-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--original-flare-jsonl", type=Path)
    parser.add_argument("--require-all-methods", action="store_true")
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument(
        "--debug-sync-stages", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--profile", default="production_efficiency", help=argparse.SUPPRESS
    )
    return parser


def validate_requested_methods(
    methods: Sequence[str], *, require_all: bool
) -> tuple[dict[str, Any], ...]:
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("requested methods must be nonempty and unique")
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown comparison methods: {unknown}")
    entries = tuple(dict(METHOD_REGISTRY[name]) for name in methods)
    unsupported = [entry for entry in entries if not entry["supported"]]
    if require_all and unsupported:
        reasons = {
            entry["method"]: entry["availability_reason"] for entry in unsupported
        }
        raise RuntimeError(f"requested methods are unavailable: {reasons}")
    return entries


def production_route_for_method(method: str) -> str:
    routes = {
        "controlled_full_replay": "full_replay",
        "complete_region_state_execution": "warm_cached_suffix",
    }
    if method not in routes:
        raise ValueError(f"method {method!r} is not an existing production route alias")
    return routes[method]


def _hash_token_ids(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _kv_hash_at_locations(runtime: Any, locations: Any) -> str:
    named = []
    text_config = runtime.model_runner.model_config.hf_text_config
    full_layers = set(getattr(text_config, "full_attention_layer_ids", []) or [])
    if not full_layers:
        full_layers = {
            index
            for index, kind in enumerate(text_config.layers_block_type)
            if kind == "attention"
        }
    long_locations = locations.to(dtype=__import__("torch").long)
    for layer_id in sorted(full_layers):
        key, value = runtime.model_runner.token_to_kv_pool.get_kv_buffer(layer_id)
        named.extend(
            (
                (f"kv.{layer_id}.key", key.index_select(0, long_locations)),
                (f"kv.{layer_id}.value", value.index_select(0, long_locations)),
            )
        )
    return runtime.cluster1.hash_tensors(named)


def _region_state_guard_factory(runtime: Any, req: Any, boundary: int):
    before = runtime._reused_hash(req, boundary)

    def verify() -> bool:
        return runtime._reused_hash(req, boundary) == before

    return verify


class ComparisonComponentTimers:
    """Temporary comparison-only probes resolved by the route's existing sync."""

    def __init__(self, validation: Any, runtime: Any) -> None:
        self.torch = __import__("torch")
        self.runtime = runtime
        self.backend_timers = validation.ScopedBackendTimers(runtime)
        self._installations = []
        self._event_pairs = {"planning": [], "kv_lookup": []}
        self._deferred = {"mask_build": []}
        self._snapshot = None
        try:
            self._install_event(
                runtime.model_runner.attn_backend,
                "init_forward_metadata",
                "planning",
            )
            self._install_event(
                runtime.runtime,
                "_canonical_prefix_locations",
                "kv_lookup",
            )
            self._install_deferred_prepare(runtime)
        except BaseException:
            self.close(preserve_primary=True)
            raise

    def _remember(self, target: Any, name: str) -> tuple[bool, Any]:
        namespace = getattr(target, "__dict__", {})
        had_value = name in namespace
        value = namespace.get(name)
        self._installations.append((target, name, had_value, value))
        return had_value, value

    def _install_event(self, target: Any, name: str, kind: str) -> None:
        original = getattr(target, name)
        self._remember(target, name)

        def measured(*args: Any, **kwargs: Any) -> Any:
            started = self.torch.cuda.Event(enable_timing=True)
            finished = self.torch.cuda.Event(enable_timing=True)
            started.record()
            result = original(*args, **kwargs)
            finished.record()
            self._event_pairs[kind].append((started, finished))
            return result

        setattr(target, name, measured)

    def _install_deferred_prepare(self, runtime: Any) -> None:
        original = runtime._maybe_cuda_timed
        self._remember(runtime, "_maybe_cuda_timed")

        def measured(
            operation: Callable[[], Any],
            *,
            nvtx_phase: str,
            collect_component_timing: bool,
        ) -> tuple[Any, Any]:
            result, timing = original(
                operation,
                nvtx_phase=nvtx_phase,
                collect_component_timing=True,
            )
            if nvtx_phase == "region_mask_build":
                self._deferred["mask_build"].append(timing)
            return result, timing if collect_component_timing else 0.0

        setattr(runtime, "_maybe_cuda_timed", measured)

    def snapshot(self) -> dict[str, Any]:
        if self._snapshot is not None:
            return dict(self._snapshot)
        values = self.backend_timers.snapshot()
        for kind, pairs in self._event_pairs.items():
            values[kind] = (
                sum(float(start.elapsed_time(end)) for start, end in pairs)
                if pairs
                else None
            )
            values[f"{kind}_calls"] = len(pairs)
        for kind, timings in self._deferred.items():
            values[kind] = sum(float(timing) for timing in timings) if timings else None
            values[f"{kind}_calls"] = len(timings)
        return values

    def close(self, *, preserve_primary: bool = False) -> None:
        errors = []
        for target, name, had_value, value in reversed(self._installations):
            try:
                if had_value:
                    setattr(target, name, value)
                else:
                    delattr(target, name)
            except BaseException as exc:
                errors.append(exc)
        self._installations.clear()
        try:
            self.backend_timers.close(preserve_primary=preserve_primary)
        except BaseException as exc:
            errors.append(exc)
        if errors and not preserve_primary:
            raise RuntimeError(
                "failed to remove comparison component probes"
            ) from errors[0]

    def __enter__(self) -> "ComparisonComponentTimers":
        return self

    def __exit__(self, exc_type: Any, _value: Any, _traceback: Any) -> None:
        if exc_type is None:
            self._snapshot = self.snapshot()
        self.close(preserve_primary=exc_type is not None)


def _release_request(runtime: Any, req: Any, token_locations: Any) -> None:
    runtime.model_runner.token_to_kv_pool_allocator.free(token_locations)
    req_pool = runtime.model_runner.req_to_token_pool
    req_pool.free_mamba_cache(req)
    req_pool.free(req)


def validate_safe_prefix_evidence(
    runtime: Any,
    observations: Sequence[Mapping[str, Any]],
    *,
    boundary: int,
    expected_steps: int,
) -> dict[str, Any]:
    torch = __import__("torch")
    if len(observations) != expected_steps:
        raise RuntimeError("safe KV-only evidence does not cover every diffusion step")
    allocator_size = int(runtime.model_runner.token_to_kv_pool_allocator.size)
    for step, observation in enumerate(observations):
        prefix = observation.get("prefix_indices")
        authoritative = observation.get("authoritative_locations")
        fresh = observation.get("fresh_reconstruction_locations")
        req_pool_idx = observation.get("req_pool_idx")
        if req_pool_idx is None:
            raise RuntimeError(
                f"safe KV-only step {step} has no real request-pool slot"
            )
        if not isinstance(prefix, torch.Tensor) or prefix.dtype is not torch.int64:
            raise RuntimeError(f"safe KV-only step {step} prefix dtype is not int64")
        if prefix.ndim != 1 or int(prefix.numel()) != boundary:
            raise RuntimeError(f"safe KV-only step {step} prefix length is invalid")
        if not prefix.is_contiguous():
            raise RuntimeError(f"safe KV-only step {step} prefix is not contiguous")
        if not isinstance(authoritative, torch.Tensor) or not torch.equal(
            prefix, authoritative
        ):
            raise RuntimeError(
                f"safe KV-only step {step} changed authoritative physical locations"
            )
        if not bool(((prefix > 0) & (prefix <= allocator_size)).all().item()):
            raise RuntimeError(f"safe KV-only step {step} has invalid pool locations")
        if not isinstance(fresh, torch.Tensor) or bool(
            torch.isin(prefix, fresh.to(device=prefix.device)).any().item()
        ):
            raise RuntimeError("GDN reconstruction can overwrite reused KV locations")
        if observation.get("restore_required") is not False:
            raise RuntimeError("safe KV-only unexpectedly requested a GDN restore")
    return {
        "method": "authoritative_retained_page_table_identity",
        "restore_steps": 0,
        "prefix_positions_per_step": boundary,
        "physical_locations_match": True,
        "observed_reused_positions": boundary * expected_steps,
        "fresh_reconstruction_pages_disjoint": True,
    }


def execute_safe_kv_measurement(
    runtime: Any,
    validation: Any,
    production: Any,
    case: Any,
    variant: str,
    repetition: int,
) -> dict[str, Any]:
    """Rebuild GDN exactly, then attach separately retained immutable KV pages."""
    if variant not in VARIANTS:
        raise RuntimeError(f"invalid comparison variant: {variant}")
    torch = __import__("torch")
    runtime.torch = torch
    original_tokens = runtime._tokens(case)
    active = set(case.active_positions)
    edited_tokens = list(original_tokens)
    for position in active:
        edited_tokens[position] = int(runtime.runtime.dllm_config.mask_id)
    edited_spec = validation.build_execution_spec(case, edited_tokens, edited=True)
    plan = validation.expected_plan(edited_spec, case.edited_regions)
    boundary = int(plan.gdn_replay_start)
    if boundary <= 0 or boundary >= case.sequence_length:
        raise RuntimeError("safe KV-only requires an internal reusable boundary")

    runtime._clear()
    authoritative_req, cache_build_ms = production._untimed_frontier_setup(
        runtime, validation, case, original_tokens
    )
    authoritative = (
        runtime.runtime._canonical_prefix_locations(
            int(authoritative_req.req_pool_idx), boundary
        )
        .detach()
        .clone()
        .contiguous()
    )
    stable_hash_before = _kv_hash_at_locations(runtime, authoritative)
    production._discard_temporary_layer_snapshots(runtime)
    # Retain only the allocated attention-KV pages.  Release request-local GDN
    # and the single request slot so reconstruction cannot consume cached GDN.
    req_pool = runtime.model_runner.req_to_token_pool
    req_pool.free_mamba_cache(authoritative_req)
    req_pool.free(authoritative_req)

    torch.cuda.reset_peak_memory_stats(runtime.device)
    timer = production.ProductionTimingEnvelope(
        torch, runtime.device, profiled=variant == "minimally_profiled"
    )
    outputs = []
    work = {
        "query_counts": [],
        "full_attention_query_token_layer_positions": 0,
        "gdn_replay_token_layer_positions": 0,
    }
    observations = []
    fallback_count = 0
    recovery_replays = 0
    component_context = (
        ComparisonComponentTimers(validation, runtime)
        if variant == "minimally_profiled"
        else contextlib.nullcontext(None)
    )
    component_timing = None
    with component_context as component_timer, timer.route():
        for step in range(case.diffusion_steps):
            base_spec = validation.build_execution_spec(
                case, edited_tokens, edited=True
            )
            req, prefix_batch, _ = production._prepare_call(
                timer,
                lambda step=step, base_spec=base_spec: runtime._prepare_canonical_frontier(
                    f"{case.case_id}:safe-kv:{variant}:{repetition}:{step}",
                    edited_tokens,
                    base_spec,
                    case,
                    collect_component_timing=False,
                ),
            )
            prefix_positions = production._query_positions(prefix_batch)
            if prefix_positions != tuple(range(boundary)):
                raise RuntimeError("safe KV-only changed prefix absolute positions")
            production._production_forward(
                runtime, prefix_batch, timer, [], set(), work
            )
            production._discard_temporary_layer_snapshots(runtime)
            fresh = (
                runtime.runtime._canonical_prefix_locations(
                    int(req.req_pool_idx), boundary
                )
                .detach()
                .clone()
                .contiguous()
            )
            runtime.model_runner.token_to_kv_pool_allocator.free(fresh)
            table = runtime.model_runner.req_to_token_pool.req_to_token
            table[int(req.req_pool_idx), :boundary].copy_(
                authoritative.to(dtype=table.dtype, device=table.device)
            )
            suffix_batch, _ = production._prepare_call(
                timer,
                lambda req=req: runtime._prepare_frontier_suffix(
                    req,
                    edited_tokens,
                    edited_spec,
                    case,
                    restore=False,
                    collect_component_timing=False,
                ),
            )
            suffix_positions = production._query_positions(suffix_batch)
            if suffix_positions != tuple(plan.attention_query_positions):
                raise RuntimeError("safe KV-only changed suffix absolute positions")
            observations.append(
                {
                    "req_pool_idx": req.req_pool_idx,
                    "prefix_indices": req.prefix_indices,
                    "authoritative_locations": authoritative,
                    "fresh_reconstruction_locations": fresh,
                    "restore_required": bool(
                        suffix_batch.region_dag_restore_required_cpu[0]
                    ),
                }
            )
            production._production_forward(
                runtime, suffix_batch, timer, outputs, active, work
            )
            instrumentation = req.region_dag_instrumentation
            fallback_count += int(instrumentation.fallback_count)
            recovery_replays += int(instrumentation.recovery_replays)
            suffix_locations = (
                table[int(req.req_pool_idx), boundary : case.sequence_length]
                .detach()
                .to(dtype=torch.int64, copy=True)
            )
            _release_request(runtime, req, suffix_locations)
    if component_timer is not None:
        component_timing = component_timer.snapshot()
        if int(component_timing.get("cache_lookup_restore_calls", -1)) != 0:
            raise RuntimeError("safe KV-only consumed a Region-DAG GDN snapshot")

    timing = timer.result()
    evidence = validate_safe_prefix_evidence(
        runtime,
        observations,
        boundary=boundary,
        expected_steps=case.diffusion_steps,
    )
    output_hash, token_ids = production.compact_output_hash_after_timing(outputs, timer)
    stable_hash_after = _kv_hash_at_locations(runtime, authoritative)
    if stable_hash_before != stable_hash_after:
        raise RuntimeError("safe KV-only mutated retained attention KV tensors")
    attention_layers, gdn_layers = production._layer_counts(runtime)
    expected_rows = case.sequence_length * case.diffusion_steps
    executed_rows = sum(work["query_counts"])
    if executed_rows != expected_rows:
        raise RuntimeError(
            "safe KV-only omitted required full-prefix reconstruction work"
        )
    if fallback_count or recovery_replays:
        raise RuntimeError("safe KV-only recovered or fell back")
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": "production_efficiency",
        "route": "safe_kv_only_diffusion",
        "variant": variant,
        "repetition_index": repetition,
        "timing": timing,
        "component_timing": component_timing,
        "output_hash": output_hash,
        "generated_top1_token_ids": token_ids,
        "kv_reuse_evidence": evidence,
        "cache_hits": int(evidence["observed_reused_positions"]),
        "gdn_state_restores": 0,
        "recovery_replay_count": recovery_replays,
        "fallback_count": fallback_count,
        "active_token_count": len(active),
        "reusable_prefix_positions": boundary,
        "full_attention_query_token_layer_positions": executed_rows * attention_layers,
        "gdn_replay_token_layer_positions": executed_rows * gdn_layers,
        "executed_query_positions": executed_rows,
        "expected_query_positions": expected_rows,
        "peak_allocated_gpu_memory_bytes": int(
            torch.cuda.max_memory_allocated(runtime.device)
        ),
        "cache_build_cuda_ms": cache_build_ms,
        "positions_preserved": True,
        "stable_state_unchanged": True,
        "full_replay_full_sequence_work": True,
        "work_accounting_matches": True,
        "attention_contract": edited_spec.attention_contract_id,
        "diffusion_steps": case.diffusion_steps,
    }


def _component_value(
    component: Optional[Mapping[str, Any]], name: str
) -> Optional[float]:
    if not component:
        return None
    value = component.get(name)
    return None if value is None else float(value)


def component_attribution(raw: Mapping[str, Any]) -> dict[str, Any]:
    timing = raw["timing"]
    profiled = raw["variant"] == "minimally_profiled"
    unavailable = None if profiled else "uninstrumented_headline_repetition"
    component = raw.get("component_timing") or {}
    attention = _component_value(component, "full_attention") if profiled else None
    gdn = _component_value(component, "gdn_replay") if profiled else None
    mlp = _component_value(component, "mlp") if profiled else None
    snapshot = _component_value(component, "prefix_snapshot") if profiled else None
    restore = _component_value(component, "cache_lookup_restore") if profiled else None
    mask = _component_value(component, "mask_build") if profiled else None
    planning = _component_value(component, "planning") if profiled else None
    kv_lookup = _component_value(component, "kv_lookup") if profiled else None
    model = timing.get("production_model_cuda_ms") if profiled else None
    total = float(timing["production_route_total_cuda_ms"])
    prepare = timing.get("production_prepare_cuda_ms") if profiled else None
    model_residual = None
    component_coverage = None
    route_residual = None
    route_coverage = None
    if profiled:
        required = (model, attention, gdn, mlp, prepare)
        if any(value is None for value in required):
            raise RuntimeError("profiled comparison record lacks core CUDA components")
        model_residual = float(model) - float(attention) - float(gdn) - float(mlp)
        if model_residual < -1e-3:
            raise RuntimeError("exclusive model component intervals double-count")
        component_coverage = (
            float(attention) + float(gdn) + float(mlp) + model_residual
        ) / float(model)
        route_residual = total - float(prepare) - float(model)
        route_coverage = (float(prepare) + float(model) + route_residual) / total
    values = {
        "model_forward_cuda_ms": None if model is None else float(model),
        "attention_cuda_ms": attention,
        "gdn_cuda_ms": gdn,
        "mlp_cuda_ms": mlp,
        "mask_cuda_ms": mask,
        "planning_cuda_ms": planning,
        "kv_lookup_cuda_ms": kv_lookup,
        "snapshot_cuda_ms": snapshot,
        "restore_cuda_ms": restore,
        "state_validation_cuda_ms": None,
        "state_commit_cuda_ms": None,
        "residual_cuda_ms": model_residual,
        "raw_unclamped_residual_cuda_ms": model_residual,
        "route_residual_cuda_ms": route_residual,
        "component_coverage_ratio": component_coverage,
        "route_coverage_ratio": route_coverage,
        "timing_domain": "cuda_events",
        "attribution": {
            "model_forward_cuda_ms": "parent_envelope",
            "attention_cuda_ms": "exclusive_model_child",
            "gdn_cuda_ms": "exclusive_model_child",
            "mlp_cuda_ms": "exclusive_model_child",
            "snapshot_cuda_ms": "shared_child_of_gdn",
            "restore_cuda_ms": "shared_child_of_gdn",
            "mask_cuda_ms": "shared_child_of_prepare",
            "planning_cuda_ms": "shared_child_of_prepare",
            "kv_lookup_cuda_ms": "shared_child_of_prepare",
            "residual_cuda_ms": "exclusive_unattributed_model_time",
        },
    }
    reasons = {
        "state_validation_cuda_ms_availability_reason": unavailable
        or "dependency_validation_is_host_only",
        "state_commit_cuda_ms_availability_reason": unavailable
        or "state_publication_is_nested_in_gdn_forward",
    }
    if profiled:
        for name, value in (
            ("mask_cuda_ms", mask),
            ("planning_cuda_ms", planning),
            ("kv_lookup_cuda_ms", kv_lookup),
            ("snapshot_cuda_ms", snapshot),
            ("restore_cuda_ms", restore),
        ):
            if value is None:
                reasons[f"{name}_availability_reason"] = (
                    "component_does_not_execute_for_method"
                )
    if not profiled:
        for key, value in values.items():
            if key.endswith("_cuda_ms") or key.endswith("_ratio"):
                values[key] = None
        reasons.update(
            {
                f"{name}_availability_reason": unavailable
                for name in (
                    "model_forward_cuda_ms",
                    "attention_cuda_ms",
                    "gdn_cuda_ms",
                    "mlp_cuda_ms",
                    "mask_cuda_ms",
                    "planning_cuda_ms",
                    "kv_lookup_cuda_ms",
                    "snapshot_cuda_ms",
                    "restore_cuda_ms",
                    "residual_cuda_ms",
                )
            }
        )
    values.update(reasons)
    return values


def validate_component_coverage(record: Mapping[str, Any]) -> None:
    if record.get("variant") != "minimally_profiled":
        return
    for name in ("component_coverage_ratio", "route_coverage_ratio"):
        value = record.get(name)
        if (
            value is None
            or not math.isfinite(float(value))
            or float(value) < MIN_COMPONENT_COVERAGE
        ):
            raise RuntimeError(f"{name} is below the 90% fail-closed threshold")


def require_output_identity(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], *, method: str
) -> None:
    if candidate.get("output_token_ids") != reference.get("output_token_ids"):
        raise RuntimeError(f"{method} output tokens differ from full replay")
    if candidate.get("output_hash") != reference.get("output_hash"):
        raise RuntimeError(f"{method} output hash differs from full replay")


def normalize_record(
    raw: Mapping[str, Any],
    *,
    method: str,
    case_id: str,
    revision: str,
    warmup: bool,
) -> dict[str, Any]:
    metadata = METHOD_REGISTRY[method]
    timing = raw["timing"]
    record = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_revision": revision,
        "case_id": case_id,
        "method": method,
        "variant": raw["variant"],
        "repetition_index": int(raw["repetition_index"]),
        "warmup": bool(warmup),
        "semantic_contract": metadata["semantic_contract"],
        "output_token_ids": list(raw["generated_top1_token_ids"]),
        "output_hash": str(raw["output_hash"]),
        "positions_preserved": bool(raw["positions_preserved"]),
        "stable_state_unchanged": bool(raw.get("stable_state_unchanged", True)),
        "full_sequence_active_reduces_to_reference": bool(
            raw.get("work_accounting_matches", False)
        ),
        "total_cuda_ms": float(timing["production_route_total_cuda_ms"]),
        "submitted_query_rows": int(raw["executed_query_positions"]),
        "submitted_mlp_rows": int(raw["executed_query_positions"]),
        "attention_token_layer_positions": int(
            raw["full_attention_query_token_layer_positions"]
        ),
        "gdn_token_layer_positions": int(raw["gdn_replay_token_layer_positions"]),
        "physical_kv_reused_positions": int(raw["cache_hits"]),
        "kv_reuse_evidence": dict(raw["kv_reuse_evidence"]),
        "gdn_restore_count": int(raw["gdn_state_restores"]),
        "gdn_replay_start": int(raw["reusable_prefix_positions"]),
        "fallback_count": int(raw["fallback_count"]),
        "recovery_replay_count": int(raw["recovery_replay_count"]),
        "synchronization_count": int(timing["synchronization_count"]),
        "peak_allocated_gpu_memory_bytes": int(raw["peak_allocated_gpu_memory_bytes"]),
    }
    record.update(component_attribution(raw))
    if record["synchronization_count"] != 1:
        raise RuntimeError(
            "comparison route did not use exactly one terminal synchronization"
        )
    if record["fallback_count"] or record["recovery_replay_count"]:
        raise RuntimeError("comparison route recovered or fell back")
    if not record["positions_preserved"]:
        raise RuntimeError("comparison route changed original absolute positions")
    if not record["stable_state_unchanged"]:
        raise RuntimeError("comparison route mutated stable cached tensors")
    if not record["full_sequence_active_reduces_to_reference"]:
        raise RuntimeError("comparison route failed its work-accounting contract")
    if method == "controlled_full_replay" and record["physical_kv_reused_positions"]:
        raise RuntimeError("controlled full replay consumed cached KV state")
    if method == "safe_kv_only_diffusion" and record["gdn_restore_count"] != 0:
        raise RuntimeError("safe KV-only consumed a GDN snapshot")
    validate_component_coverage(record)
    return record


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def _summarize_method(
    records: Sequence[Mapping[str, Any]], control_median: float
) -> dict[str, Any]:
    headline = [record for record in records if record["variant"] == "uninstrumented"]
    profiled = [
        record for record in records if record["variant"] == "minimally_profiled"
    ]
    samples = [float(record["total_cuda_ms"]) for record in headline]
    median = statistics.median(samples)
    mad = statistics.median(abs(value - median) for value in samples)
    component_names = (
        "model_forward_cuda_ms",
        "attention_cuda_ms",
        "gdn_cuda_ms",
        "mlp_cuda_ms",
        "snapshot_cuda_ms",
        "restore_cuda_ms",
        "residual_cuda_ms",
        "component_coverage_ratio",
        "route_coverage_ratio",
    )
    return {
        "supported": True,
        "median_total_cuda_ms": median,
        "p95_total_cuda_ms": _percentile(samples, 0.95),
        "relative_mad": mad / median if median else None,
        "speedup_vs_controlled_full_replay": control_median / median,
        "latency_reduction_percent": 100.0 * (control_median - median) / control_median,
        "median_peak_memory_bytes": statistics.median(
            int(record["peak_allocated_gpu_memory_bytes"]) for record in headline
        ),
        "work_accounting": {
            "submitted_query_rows": sorted(
                {record["submitted_query_rows"] for record in headline}
            ),
            "attention_token_layer_positions": sorted(
                {record["attention_token_layer_positions"] for record in headline}
            ),
            "gdn_token_layer_positions": sorted(
                {record["gdn_token_layer_positions"] for record in headline}
            ),
            "physical_kv_reused_positions": sorted(
                {record["physical_kv_reused_positions"] for record in headline}
            ),
        },
        "component_timing": {
            name: (
                statistics.median(
                    float(record[name])
                    for record in profiled
                    if record.get(name) is not None
                )
                if any(record.get(name) is not None for record in profiled)
                else None
            )
            for name in component_names
        },
    }


def validate_original_flare_artifact(
    path: Path, *, expected_input_hashes: Optional[Mapping[str, str]] = None
) -> dict[str, Any]:
    records = []
    required = {
        "software_provenance",
        "hardware",
        "checkpoint_identity",
        "token_input_hash",
        "diffusion_steps",
        "warmup",
        "repetition_index",
        "total_cuda_ms",
    }
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if value.get("method") != "original_flare_external_runtime":
            raise RuntimeError(
                f"original FLARE artifact line {line_number} is mislabeled"
            )
        if value.get("runtime_revision") != ORIGINAL_FLARE_REVISION:
            raise RuntimeError(
                f"original FLARE artifact line {line_number} has wrong revision"
            )
        if value.get("case_id") == "four-fragmented-regions":
            raise RuntimeError(
                "original FLARE cannot claim the fragmented Region-DAG case"
            )
        missing = sorted(required - set(value))
        if missing:
            raise RuntimeError(
                f"original FLARE artifact line {line_number} lacks provenance: {missing}"
            )
        if int(value.get("diffusion_steps", 0)) != 4:
            raise RuntimeError("original FLARE artifact changed diffusion-step count")
        case_id = str(value.get("case_id", ""))
        if expected_input_hashes is not None and value.get(
            "token_input_hash"
        ) != expected_input_hashes.get(case_id):
            raise RuntimeError("original FLARE artifact token input differs")
        records.append(value)
    return {
        "label": "original_flare_external_runtime",
        "runtime_revision": ORIGINAL_FLARE_REVISION,
        "record_count": len(records),
        "excluded_from_same_runtime_statistics": True,
    }


def run(
    args: argparse.Namespace, *, runtime_factory: Optional[Callable[[Any], Any]] = None
) -> dict[str, Any]:
    validation = _load_local_module(
        "cluster3_cache_comparison_validation",
        Path(__file__).with_name("cluster3_region_dag_validation.py"),
    )
    manifest = _load_local_module(
        "cluster3_cache_comparison_manifest",
        Path(__file__).with_name("cluster3_benchmark_manifest.py"),
    )
    production = _load_local_module(
        "cluster3_cache_comparison_production",
        Path(__file__).with_name("cluster3_production_latency_benchmark.py"),
    )
    registry = validate_requested_methods(
        args.methods, require_all=bool(args.require_all_methods)
    )
    cases, normalized_cases = manifest.load_manifest(
        args.case_manifest,
        validation_module=validation,
        seed=int(args.seed),
        requested_routes=args.methods,
    )
    if len(cases) != 3:
        raise RuntimeError("cache-system comparison requires exactly three cases")
    required_capacity = max(case.sequence_length * case.batch_size for case in cases)
    if int(args.max_total_tokens) < required_capacity:
        raise RuntimeError("max-total-tokens is smaller than the largest manifest case")
    if args.dtype != "bfloat16" or int(args.tp_size) != 1:
        raise RuntimeError("cache-system comparison requires BF16 and TP=1")
    if args.warmups < 0 or args.timed_repetitions <= 0:
        raise RuntimeError("warmups must be nonnegative and repetitions positive")
    if args.debug_sync_stages:
        raise RuntimeError("debug synchronization is forbidden")

    # Nothing above this line inspects the checkpoint or constructs the model.
    preflight_module = validation._load_module(
        "cluster3_cache_comparison_preflight",
        Path(__file__).with_name("cluster3_efficiency_preflight.py"),
    )
    checkpoint = preflight_module.checkpoint_identity(Path(args.model_path))
    correctness = production._read_json(args.correctness_artifact)
    preflight = production._read_json(args.preflight_json)
    hardware = production.production_hardware(int(args.device))
    production.validate_correctness_prerequisite(
        correctness, preflight, checkpoint, hardware
    )
    factory = runtime_factory or validation.Cluster3ValidationRuntime
    runtime = factory(args)
    revision = validation._git_revision()
    supported = [entry["method"] for entry in registry if entry["supported"]]
    if "controlled_full_replay" not in supported:
        raise RuntimeError("supported comparison requires controlled_full_replay")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    records = []
    references = {}
    case_inputs = {}
    try:
        with args.output_jsonl.open("w", encoding="utf-8") as destination:
            for case, normalized_case in zip(cases, normalized_cases):
                case_id = normalized_case["case_id"]
                input_token_ids = runtime._tokens(case)
                for position in case.active_positions:
                    input_token_ids[position] = int(runtime.runtime.dllm_config.mask_id)
                case_inputs[case_id] = {
                    "token_input_ids": input_token_ids,
                    "token_input_hash": _hash_token_ids(input_token_ids),
                    "diffusion_steps": int(case.diffusion_steps),
                }
                for method in supported:
                    for repetition in range(-args.warmups, args.timed_repetitions):
                        for variant in VARIANTS:
                            if method == "safe_kv_only_diffusion":
                                raw = execute_safe_kv_measurement(
                                    runtime,
                                    validation,
                                    production,
                                    case,
                                    variant,
                                    repetition,
                                )
                            else:
                                route = production_route_for_method(method)
                                raw = production.execute_production_measurement(
                                    runtime,
                                    validation,
                                    case,
                                    route,
                                    variant,
                                    repetition,
                                    stable_state_guard_factory=(
                                        _region_state_guard_factory
                                    ),
                                    component_timer_factory=(
                                        lambda value: ComparisonComponentTimers(
                                            validation, value
                                        )
                                    ),
                                )
                            record = normalize_record(
                                raw,
                                method=method,
                                case_id=case_id,
                                revision=revision,
                                warmup=repetition < 0,
                            )
                            if repetition < 0:
                                continue
                            key = (case_id, variant, repetition)
                            if method == "controlled_full_replay":
                                references[key] = record
                            else:
                                reference = references[key]
                                require_output_identity(
                                    reference, record, method=method
                                )
                            records.append(record)
                            destination.write(json.dumps(record, sort_keys=True) + "\n")
                            destination.flush()
    finally:
        runtime.close()

    results = {}
    for case in normalized_cases:
        case_id = case["case_id"]
        case_records = [record for record in records if record["case_id"] == case_id]
        control = [
            record["total_cuda_ms"]
            for record in case_records
            if record["method"] == "controlled_full_replay"
            and record["variant"] == "uninstrumented"
        ]
        control_median = statistics.median(control)
        results[case_id] = {
            method: _summarize_method(
                [record for record in case_records if record["method"] == method],
                control_median,
            )
            for method in supported
        }
        for entry in registry:
            if not entry["supported"]:
                results[case_id][entry["method"]] = {
                    "supported": False,
                    "availability_reason": entry["availability_reason"],
                }

    hashes_by_case = {
        case["case_id"]: {
            record["output_hash"]
            for record in records
            if record["case_id"] == case["case_id"]
        }
        for case in normalized_cases
    }
    external = (
        validate_original_flare_artifact(
            args.original_flare_jsonl,
            expected_input_hashes={
                case_id: value["token_input_hash"]
                for case_id, value in case_inputs.items()
            },
        )
        if args.original_flare_jsonl
        else {
            "label": "original_flare_external_runtime",
            "runtime_revision": ORIGINAL_FLARE_REVISION,
            "status": "not_ingested",
            "excluded_from_same_runtime_statistics": True,
        }
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_revision": revision,
        "base_revision": BASE_REVISION,
        "model_identity": checkpoint,
        "hardware": hardware,
        "protocol": {
            "warmups": int(args.warmups),
            "uninstrumented_repetitions": int(args.timed_repetitions),
            "minimally_profiled_repetitions": int(args.timed_repetitions),
            "dtype": args.dtype,
            "tp_size": int(args.tp_size),
            "batch_size": 1,
            "terminal_synchronizations_per_route": 1,
            "component_coverage_threshold": MIN_COMPONENT_COVERAGE,
        },
        "method_registry": {entry["method"]: entry for entry in registry},
        "case_count": len(cases),
        "supported_method_count": len(supported),
        "publication_matrix_complete": all(entry["supported"] for entry in registry),
        "all_outputs_identical": all(
            len(values) == 1 for values in hashes_by_case.values()
        ),
        "all_correctness_gates_pass": all(
            record["positions_preserved"]
            and record["stable_state_unchanged"]
            and record["fallback_count"] == 0
            and record["recovery_replay_count"] == 0
            and record["synchronization_count"] == 1
            for record in records
        ),
        "original_flare_external_runtime": external,
        "case_inputs": case_inputs,
        "results": results,
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
