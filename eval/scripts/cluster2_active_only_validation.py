#!/usr/bin/env python3
"""Validate the production active-only suffix layout without changing it.

HybridDiffusion already submits active suffix rows to the model while exposing
the stable prefix through paged KV and restored GDN state.  This artifact adds
only reproducible manifests, temporary observation hooks, strict validation,
and bounded JSON reports around that pre-existing execution path.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import logging
import math
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "cluster2_active_only_validation"
RUNTIME_BASE_REVISION = "1e21bdec792a665419b290bf89f75c14c730dc94"
NUMERICAL_TOLERANCE = 1e-2
MAX_COMPARE_CHUNK_ELEMENTS = 1 << 20
DEFAULT_SEED = 20260825
ATTENTION_ROW_OBSERVATION_POINT = "qkv_projection_input"
QWEN35_2B_FINGERPRINT = {
    "hidden_size": 2048,
    "num_hidden_layers": 24,
    "intermediate_size": 6144,
}
ROOT = Path(__file__).resolve().parents[2]
CLUSTER1_PATH = Path(__file__).with_name("cluster1_exact_handoff_trace.py")

CHECK_NAMES = (
    "one_record_per_case",
    "submitted_equals_active",
    "zero_stable_queries",
    "attention_rows_equal_active",
    "mlp_rows_equal_active",
    "gdn_rows_equal_active",
    "kv_contains_stable_plus_active",
    "positions_preserved",
    "stable_states_unchanged",
    "bf16_numerical_tolerance",
    "top1_identical",
    "no_nan_or_inf",
    "zero_fallback",
    "zero_recovery_replay",
    "required_batch_coverage",
    "runtime_files_unchanged",
)

REQUIRED_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "case_id",
        "profile",
        "revision",
        "model_scale",
        "dtype",
        "tp_size",
        "batch_size",
        "prefix_tokens_per_request",
        "active_tokens_per_request",
        "diffusion_steps",
        "scheduled_input_tokens",
        "expected_active_tokens",
        "stable_query_tokens",
        "attention_query_rows_per_layer",
        "total_kv_tokens_per_request",
        "mlp_rows_per_layer",
        "gdn_rows_per_layer",
        "active_position_ids_per_request",
        "positions_preserved",
        "stable_kv_unchanged",
        "stable_gdn_unchanged",
        "max_logits_error",
        "max_hidden_error",
        "max_gdn_error",
        "top1_identical",
        "fallback_count",
        "recovery_replays",
        "nan_or_inf_detected",
        "stage_latency_ms",
        "peak_memory_bytes",
        "case_pass",
        "failure_reasons",
    }
)


@dataclass(frozen=True)
class ValidationCase:
    case_id: str
    profile: str
    token_seed: int
    prefix_length: int
    active_length: int
    diffusion_steps: int
    batch_size: int

    @property
    def expected_active_tokens(self) -> int:
        return self.active_length * self.batch_size

    @property
    def total_tokens_per_request(self) -> int:
        return self.prefix_length + self.active_length


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the pre-existing HybridDiffusion active-only suffix "
            "execution layout and numerical equivalence."
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument(
        "--profile", required=True, choices=("one1", "smoke16", "paper100")
    )
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16",))
    parser.add_argument("--tp-size", default=1, type=int)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int)
    parser.add_argument("--max-total-tokens", default=8192, type=int)
    parser.add_argument("--debug-sync-stages", action="store_true")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.tp_size != 1:
        raise ValueError("--tp-size must be 1 until rank-aware proof collection exists")
    if args.device < 0:
        raise ValueError("--device must be non-negative")
    if args.max_total_tokens <= 0:
        raise ValueError("--max-total-tokens must be positive")
    return args


def _shape_case(
    profile: str,
    index: int,
    seed: int,
    prefix: int,
    active: int,
    steps: int,
    batch: int,
) -> ValidationCase:
    return ValidationCase(
        case_id=f"{profile}-{index:03d}-p{prefix}-a{active}-s{steps}-b{batch}",
        profile=profile,
        token_seed=seed + index * 104729,
        prefix_length=prefix,
        active_length=active,
        diffusion_steps=steps,
        batch_size=batch,
    )


def _smoke_shapes() -> list[tuple[int, int, int, int]]:
    return [
        (0, 1, 2, 1),
        (1, 32, 4, 2),
        (6, 64, 8, 4),
        (7, 128, 2, 1),
        (8, 256, 4, 2),
        (15, 1, 8, 4),
        (16, 32, 2, 1),
        (63, 64, 4, 2),
        (64, 128, 8, 4),
        (65, 256, 2, 1),
        (255, 1, 4, 2),
        (256, 32, 8, 4),
        (257, 64, 2, 1),
        (511, 128, 4, 2),
        (512, 256, 8, 1),
        (1024, 1, 2, 1),
    ]


def _paper_shapes(seed: int) -> list[tuple[int, int, int, int]]:
    prefixes = (0, 1, 7, 8, 64, 256, 512, 1024, 2048)
    actives = (1, 32, 64, 128, 256)
    steps = (2, 4, 8)
    batches = (1, 2, 4)
    candidates = []
    for prefix in prefixes:
        for active in actives:
            for step in steps:
                for batch in batches:
                    # Keep the controlled replay allocation below the configured
                    # A30 token budget, including all cached-path denoising steps.
                    footprint = batch * (prefix + active * step)
                    if footprint <= 8192 and not (batch == 4 and prefix >= 512):
                        candidates.append((prefix, active, step, batch))

    required = [
        (0, 1, 2, 1),
        (1, 32, 4, 2),
        (7, 64, 8, 4),
        (8, 128, 2, 1),
        (64, 256, 4, 2),
        (256, 1, 8, 4),
        (512, 32, 2, 1),
        (1024, 64, 4, 1),
        (2048, 128, 8, 1),
        (2048, 256, 2, 1),
    ]
    chosen = list(dict.fromkeys(required))
    remaining = [shape for shape in candidates if shape not in chosen]
    random.Random(seed).shuffle(remaining)
    chosen.extend(remaining[: 100 - len(chosen)])
    if len(chosen) != 100:
        raise RuntimeError("paper100 shape construction did not produce 100 cases")
    random.Random(seed ^ 0xC2A30).shuffle(chosen)
    return chosen


def build_manifest(profile: str, seed: int = DEFAULT_SEED) -> list[ValidationCase]:
    if profile == "one1":
        shapes = [(64, 1, 2, 1)]
    elif profile == "smoke16":
        shapes = _smoke_shapes()
    elif profile == "paper100":
        shapes = _paper_shapes(seed)
    else:
        raise ValueError(f"unsupported profile: {profile}")
    return [
        _shape_case(profile, index, seed, *shape)
        for index, shape in enumerate(shapes, 1)
    ]


def required_coverage(profile: str) -> dict[str, set[int]]:
    if profile == "one1":
        return {
            "batch": {1},
            "prefix": {64},
            "active": {1},
            "steps": {2},
        }
    if profile == "smoke16":
        return {
            "batch": {1, 2, 4},
            "prefix": {
                0,
                1,
                6,
                7,
                8,
                15,
                16,
                63,
                64,
                65,
                255,
                256,
                257,
                511,
                512,
                1024,
            },
            "active": {1, 32, 64, 128, 256},
            "steps": {2, 4, 8},
        }
    if profile == "paper100":
        return {
            "batch": {1, 2, 4},
            "prefix": {0, 1, 7, 8, 64, 256, 512, 1024, 2048},
            "active": {1, 32, 64, 128, 256},
            "steps": {2, 4, 8},
        }
    raise ValueError(f"unsupported profile: {profile}")


def manifest_coverage(cases: Iterable[ValidationCase]) -> dict[str, set[int]]:
    values = list(cases)
    return {
        "batch": {case.batch_size for case in values},
        "prefix": {case.prefix_length for case in values},
        "active": {case.active_length for case in values},
        "steps": {case.diffusion_steps for case in values},
    }


def _all_equal(values: Any, expected: int) -> bool:
    return (
        isinstance(values, list)
        and bool(values)
        and all(isinstance(value, int) and value == expected for value in values)
    )


def validate_case_record(record: Mapping[str, Any], case: ValidationCase) -> list[str]:
    reasons: list[str] = []
    missing = sorted(REQUIRED_RECORD_FIELDS - record.keys())
    if missing:
        reasons.append(f"missing required fields: {missing}")
        return reasons
    expected_active = case.expected_active_tokens
    expected_kv = case.total_tokens_per_request
    identity = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case.case_id,
        "profile": case.profile,
        "batch_size": case.batch_size,
        "prefix_tokens_per_request": case.prefix_length,
        "active_tokens_per_request": case.active_length,
        "diffusion_steps": case.diffusion_steps,
        "expected_active_tokens": expected_active,
        "dtype": "bfloat16",
        "tp_size": 1,
        "model_scale": "2B",
    }
    for field, expected in identity.items():
        if record.get(field) != expected:
            reasons.append(f"{field}={record.get(field)!r}, expected {expected!r}")
    if record.get("scheduled_input_tokens") != expected_active:
        reasons.append("submitted input rows do not equal expected active rows")
    if record.get("stable_query_tokens") != 0:
        reasons.append("stable query rows are nonzero")
    for field, label in (
        ("attention_query_rows_per_layer", "attention"),
        ("mlp_rows_per_layer", "MLP"),
        ("gdn_rows_per_layer", "GDN"),
    ):
        if not _all_equal(record.get(field), expected_active):
            reasons.append(f"{label} row trace is missing or differs from active rows")
    kv_lengths = record.get("total_kv_tokens_per_request")
    if (
        not isinstance(kv_lengths, list)
        or len(kv_lengths) != case.batch_size
        or any(value != expected_kv for value in kv_lengths)
    ):
        reasons.append("readable KV lengths do not equal stable plus active")
    position_evidence = record.get("active_position_ids_per_request")
    if (
        not isinstance(position_evidence, list)
        or len(position_evidence) != case.batch_size
        or any(
            not isinstance(value, Mapping)
            or value.get("count") != case.active_length
            or value.get("first") != case.prefix_length
            or value.get("last") != case.prefix_length + case.active_length - 1
            or not value.get("sha256")
            for value in position_evidence
        )
    ):
        reasons.append("active position trace is missing or incomplete")
    if record.get("positions_preserved") is not True:
        reasons.append("active positions drifted")
    if case.prefix_length == 0:
        unavailable = record.get("unavailable_reasons", {})
        for field in ("stable_kv_unchanged", "stable_gdn_unchanged"):
            if record.get(field) is not None or not unavailable.get(field):
                reasons.append(
                    f"{field} must be explicitly inapplicable for prefix zero"
                )
    else:
        if record.get("stable_kv_unchanged") is not True:
            reasons.append("stable KV state changed or is unobserved")
        if record.get("stable_gdn_unchanged") is not True:
            reasons.append("stable GDN state changed or is unobserved")
    for field in ("max_logits_error", "max_hidden_error", "max_gdn_error"):
        value = record.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            reasons.append(f"{field} is unavailable")
        elif (
            not math.isfinite(float(value)) or value < 0 or value >= NUMERICAL_TOLERANCE
        ):
            reasons.append(f"{field}={value} exceeds BF16 tolerance")
    if record.get("top1_identical") is not True:
        reasons.append("top-1 tokens differ")
    if record.get("fallback_count") != 0:
        reasons.append("fallback was observed")
    if record.get("recovery_replays") != 0:
        reasons.append("recovery replay was observed")
    if record.get("nan_or_inf_detected") is not False:
        reasons.append("NaN/Inf status is missing or positive")
    latency = record.get("stage_latency_ms")
    if (
        not isinstance(latency, Mapping)
        or not latency
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value < 0
            for value in latency.values()
        )
    ):
        reasons.append("stage latency evidence is missing")
    peak = record.get("peak_memory_bytes")
    if not isinstance(peak, int) or peak < 0:
        reasons.append("peak memory evidence is missing")
    if not record.get("attention_query_rows_per_layer"):
        reasons.append("attention traces are missing")
    if not record.get("mlp_rows_per_layer"):
        reasons.append("MLP traces are missing")
    if not record.get("gdn_rows_per_layer"):
        reasons.append("GDN traces are missing")
    return reasons


def _check_record(record: Mapping[str, Any], case: ValidationCase, name: str) -> bool:
    expected = case.expected_active_tokens
    checks = {
        "submitted_equals_active": record.get("scheduled_input_tokens") == expected,
        "zero_stable_queries": record.get("stable_query_tokens") == 0,
        "attention_rows_equal_active": _all_equal(
            record.get("attention_query_rows_per_layer"), expected
        ),
        "mlp_rows_equal_active": _all_equal(record.get("mlp_rows_per_layer"), expected),
        "gdn_rows_equal_active": _all_equal(record.get("gdn_rows_per_layer"), expected),
        "kv_contains_stable_plus_active": record.get("total_kv_tokens_per_request")
        == [case.total_tokens_per_request] * case.batch_size,
        "positions_preserved": record.get("positions_preserved") is True,
        "stable_states_unchanged": (
            (
                record.get("stable_kv_unchanged") is True
                and record.get("stable_gdn_unchanged") is True
            )
            if case.prefix_length > 0
            else (
                record.get("stable_kv_unchanged") is None
                and record.get("stable_gdn_unchanged") is None
                and bool(
                    record.get("unavailable_reasons", {}).get("stable_kv_unchanged")
                )
                and bool(
                    record.get("unavailable_reasons", {}).get("stable_gdn_unchanged")
                )
            )
        ),
        "bf16_numerical_tolerance": all(
            isinstance(record.get(field), (int, float))
            and not isinstance(record.get(field), bool)
            and math.isfinite(float(record[field]))
            and 0 <= record[field] < NUMERICAL_TOLERANCE
            for field in ("max_logits_error", "max_hidden_error", "max_gdn_error")
        ),
        "top1_identical": record.get("top1_identical") is True,
        "no_nan_or_inf": record.get("nan_or_inf_detected") is False,
        "zero_fallback": record.get("fallback_count") == 0,
        "zero_recovery_replay": record.get("recovery_replays") == 0,
    }
    return checks[name]


def build_summary(
    cases: Sequence[ValidationCase],
    records: Sequence[Mapping[str, Any]],
    *,
    revision: str,
    hardware: Mapping[str, Any],
    dtype: str,
    tp_size: int,
    profile: str,
    runtime_diff_is_empty: bool,
    runtime_diff_files: Sequence[str] = (),
) -> dict[str, Any]:
    expected = {case.case_id: case for case in cases}
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("case_id")), []).append(record)
    one_record = set(grouped) == set(expected) and all(
        len(values) == 1 for values in grouped.values()
    )
    valid_pairs = [
        (expected[case_id], values[0])
        for case_id, values in grouped.items()
        if case_id in expected and len(values) == 1
    ]
    coverage = manifest_coverage(cases)
    required = required_coverage(profile)
    checks = {"one_record_per_case": one_record}
    for name in CHECK_NAMES[1:-2]:
        checks[name] = len(valid_pairs) == len(cases) and all(
            _check_record(record, case, name) for case, record in valid_pairs
        )
    checks["required_batch_coverage"] = coverage["batch"] == required["batch"]
    checks["runtime_files_unchanged"] = runtime_diff_is_empty
    failed_cases = []
    passed_cases = 0
    for case, record in valid_pairs:
        reasons = validate_case_record(record, case)
        if (
            reasons
            or record.get("case_pass") is not True
            or record.get("failure_reasons")
        ):
            failed_cases.append(
                {
                    "case_id": case.case_id,
                    "failure_reasons": reasons or record.get("failure_reasons", []),
                }
            )
        else:
            passed_cases += 1
    numeric = lambda field: [
        float(record[field])
        for _, record in valid_pairs
        if isinstance(record.get(field), (int, float))
        and not isinstance(record.get(field), bool)
        and math.isfinite(float(record[field]))
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "cluster2_revision": revision,
        "runtime_base_revision": RUNTIME_BASE_REVISION,
        "runtime_diff_empty": runtime_diff_is_empty,
        "runtime_diff_files": list(runtime_diff_files),
        "hardware": dict(hardware),
        "dtype": dtype,
        "tp_size": tp_size,
        "profile": profile,
        "requested_cases": len(cases),
        "received_records": len(records),
        "passed_cases": passed_cases,
        "batch_sizes_covered": sorted(coverage["batch"]),
        "prefix_lengths_covered": sorted(coverage["prefix"]),
        "active_lengths_covered": sorted(coverage["active"]),
        "steps_covered": sorted(coverage["steps"]),
        "maximum_logits_error": max(numeric("max_logits_error"), default=None),
        "maximum_hidden_error": max(numeric("max_hidden_error"), default=None),
        "maximum_gdn_error": max(numeric("max_gdn_error"), default=None),
        "checks": checks,
        "strict_pass": all(checks.values()) and passed_cases == len(cases),
        "failed_cases": failed_cases,
    }


def prohibited_runtime_diffs(
    repo_root: Path = ROOT, base_revision: str = RUNTIME_BASE_REVISION
) -> list[str]:
    completed = subprocess.run(
        ["git", "diff", "--name-only", base_revision, "--"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    changed = {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    for line in status.stdout.splitlines():
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        changed.add(path)

    def protected(path: str) -> bool:
        return (
            path.startswith("eval/sglang/")
            or path.startswith("torchtitan/")
            or path == "eval/scripts/cluster1_exact_handoff_trace.py"
            or path.endswith(".ipynb")
        )

    return sorted(path for path in changed if protected(path))


def _load_cluster1_module() -> Any:
    name = "cluster2_cluster1_runtime"
    spec = importlib.util.spec_from_file_location(name, CLUSTER1_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the Cluster-1 controlled replay runtime")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tensor_rows(inputs: tuple[Any, ...], kwargs: Mapping[str, Any]) -> int:
    torch = __import__("torch")
    candidates = [kwargs.get("hidden_states"), *inputs]
    for value in candidates:
        if torch.is_tensor(value) and value.ndim >= 1:
            return int(value.shape[0])
    raise RuntimeError("row hook did not receive a hidden-state tensor")


def _trace_tensor_to_cpu(value: Any) -> Any:
    """Copy validator evidence off CUDA without retaining its source storage."""
    torch = __import__("torch")
    if not torch.is_tensor(value):
        raise TypeError("trace evidence must be a torch tensor")
    return value.detach().to(device="cpu", copy=True).contiguous()


class ScopedRowHooks:
    """Test-local model hooks that are removed on every exit path."""

    def __init__(self, model_runner: Any):
        self.model_runner = model_runner
        self.handles: list[Any] = []
        self.capturing = False
        self.rows = {"attention": [], "mlp": [], "gdn": []}
        self.hidden: dict[int, Any] = {}
        self.gdn_states: dict[int, tuple[Any, ...]] = {}
        self.mamba_slots: list[int] = []
        self.cluster1 = _load_cluster1_module()
        model = self.cluster1.ModelTraceHooks._language_model(model_runner.model)
        self.num_layers = len(model.layers)
        self.mamba_map = getattr(model_runner.req_to_token_pool, "mamba_map", {})
        torch = __import__("torch")
        nn = torch.nn
        try:
            for layer_id, layer in enumerate(model.layers):
                if not isinstance(layer, nn.Module):
                    raise RuntimeError(
                        f"layer {layer_id} is not a hookable torch.nn.Module"
                    )
                self.handles.append(
                    layer.register_forward_hook(self._layer_output_hook(layer_id))
                )

                mlp_module = getattr(layer, "mlp", None)
                if not isinstance(mlp_module, nn.Module):
                    raise RuntimeError(f"layer {layer_id} has no hookable MLP module")
                self.handles.append(
                    mlp_module.register_forward_pre_hook(
                        self._row_hook("mlp"), with_kwargs=True
                    )
                )

                gdn_module = getattr(layer, "linear_attn", None)
                if isinstance(gdn_module, nn.Module):
                    self.handles.append(
                        gdn_module.register_forward_pre_hook(
                            self._row_hook("gdn"), with_kwargs=True
                        )
                    )
                    continue

                # Qwen3.5 self_attention is a bound helper method, not an
                # nn.Module. qkv_proj is the real projection module and its
                # input has one row for every generated attention query row.
                attention_module = getattr(layer, "qkv_proj", None)
                if not isinstance(attention_module, nn.Module):
                    legacy_attention = getattr(layer, "self_attention", None)
                    if isinstance(legacy_attention, nn.Module):
                        attention_module = legacy_attention
                if not isinstance(attention_module, nn.Module):
                    raise RuntimeError(
                        f"layer {layer_id} has no hookable full-attention projection"
                    )
                self.handles.append(
                    attention_module.register_forward_pre_hook(
                        self._row_hook("attention"), with_kwargs=True
                    )
                )
        except BaseException:
            self.close()
            raise

    def _row_hook(self, kind: str) -> Callable[..., None]:
        def hook(
            _module: Any, inputs: tuple[Any, ...], kwargs: Mapping[str, Any]
        ) -> None:
            if self.capturing:
                self.rows[kind].append(_tensor_rows(inputs, kwargs))

        return hook

    def _layer_output_hook(self, layer_id: int) -> Callable[..., None]:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if not self.capturing:
                return
            torch = __import__("torch")
            hidden = output[0] if isinstance(output, tuple) else output
            residual = (
                output[1] if isinstance(output, tuple) and len(output) > 1 else None
            )
            value = hidden.detach().contiguous().clone()
            if (
                residual is not None
                and torch.is_tensor(residual)
                and residual.shape == hidden.shape
            ):
                value = value + residual.detach().contiguous()
            self.hidden[layer_id] = value
            gdn_index = self.mamba_map.get(layer_id)
            if gdn_index is not None:
                cache = self.model_runner.req_to_token_pool.mamba_pool.mamba_cache
                slots = torch.tensor(
                    self.mamba_slots, dtype=torch.long, device=value.device
                )
                states = [
                    conv[int(gdn_index)].index_select(0, slots).detach().clone()
                    for conv in cache.conv
                ]
                states.append(
                    cache.temporal[int(gdn_index)]
                    .index_select(0, slots)
                    .detach()
                    .clone()
                )
                self.gdn_states[layer_id] = tuple(states)

        return hook

    @contextlib.contextmanager
    def capture(self, mamba_slots: Sequence[int]) -> Iterator[None]:
        if self.capturing:
            raise RuntimeError("nested row capture is not allowed")
        self.rows = {"attention": [], "mlp": [], "gdn": []}
        self.hidden = {}
        self.gdn_states = {}
        self.mamba_slots = [int(value) for value in mamba_slots]
        self.capturing = True
        try:
            yield
        finally:
            self.capturing = False

    def snapshot(self) -> dict[str, Any]:
        snapshot = {
            "rows": {name: list(values) for name, values in self.rows.items()},
            "hidden": {
                key: _trace_tensor_to_cpu(value) for key, value in self.hidden.items()
            },
            "gdn_states": {
                key: tuple(_trace_tensor_to_cpu(value) for value in values)
                for key, values in self.gdn_states.items()
            },
        }
        # Do not let the hook object retain CUDA evidence between paired
        # forwards. The returned snapshot owns independent CPU copies.
        self.hidden.clear()
        self.gdn_states.clear()
        return snapshot

    def close(self) -> None:
        self.capturing = False
        self.hidden.clear()
        self.gdn_states.clear()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @property
    def released(self) -> bool:
        return not self.handles and not self.capturing

    def __enter__(self) -> "ScopedRowHooks":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


class RuntimeLogEvidence(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.fallback_count = 0
        self.recovery_replays = 0

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage().lower()
        if "fallback" in message:
            self.fallback_count += 1
        if "recovery_replay" in message or "recovery replay" in message:
            self.recovery_replays += 1

    @contextlib.contextmanager
    def installed(self) -> Iterator["RuntimeLogEvidence"]:
        root = logging.getLogger()
        root.addHandler(self)
        try:
            yield self
        finally:
            root.removeHandler(self)


def _max_abs(left: Any, right: Any) -> float:
    torch = __import__("torch")
    if left.shape != right.shape:
        raise RuntimeError(
            f"paired tensor shape mismatch: {left.shape} != {right.shape}"
        )
    if left.device != right.device:
        raise RuntimeError(
            f"paired tensor device mismatch: {left.device} != {right.device}"
        )
    if int(left.numel()) == 0:
        raise RuntimeError("paired tensor is empty")

    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    maximum = 0.0
    # Full-vocabulary logits can be hundreds of MiB. Converting both complete
    # tensors to FP32 at once creates several equally large temporaries. Work
    # in bounded chunks while preserving the exact maximum-absolute-error
    # definition and finite-value validation.
    for start in range(0, int(left_flat.numel()), MAX_COMPARE_CHUNK_ELEMENTS):
        stop = min(start + MAX_COMPARE_CHUNK_ELEMENTS, int(left_flat.numel()))
        left_chunk = left_flat[start:stop].float()
        right_chunk = right_flat[start:stop].float()
        if not bool(torch.isfinite(left_chunk).all().item()) or not bool(
            torch.isfinite(right_chunk).all().item()
        ):
            raise RuntimeError("paired tensor contains NaN or Inf")
        maximum = max(
            maximum,
            float((left_chunk - right_chunk).abs().max().item()),
        )
    return maximum


def _position_evidence(values: Sequence[int]) -> dict[str, Any]:
    encoded = json.dumps(list(values), separators=(",", ":")).encode("ascii")
    return {
        "count": len(values),
        "first": values[0] if values else None,
        "last": values[-1] if values else None,
        "preview": list(values[:8]),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def infer_model_scale(model_runner: Any) -> str:
    """Identify the supported TP=1 checkpoint from loaded model metadata."""
    model_config = getattr(model_runner, "model_config", None)
    text_config = getattr(model_config, "hf_text_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    if text_config is None:
        raise RuntimeError("cannot identify model scale: hf_text_config is missing")

    identifiers = [
        getattr(text_config, "model_type", None),
        type(text_config).__name__,
        getattr(hf_config, "model_type", None),
        type(hf_config).__name__ if hf_config is not None else None,
        *(getattr(hf_config, "architectures", None) or []),
    ]
    normalized_identity = " ".join(
        str(value).lower() for value in identifiers if value is not None
    ).replace("-", "_")
    observed = {
        name: getattr(text_config, name, None) for name in QWEN35_2B_FINGERPRINT
    }
    qwen35_identity = (
        "qwen3_5" in normalized_identity or "qwen35" in normalized_identity
    )
    if qwen35_identity and observed == QWEN35_2B_FINGERPRINT:
        return "2B"
    raise RuntimeError(
        "unsupported or ambiguous checkpoint scale; expected Qwen3.5-2B "
        f"metadata={QWEN35_2B_FINGERPRINT}, observed={observed}, "
        f"identity={normalized_identity!r}"
    )


class Cluster2ValidationRuntime:
    """Controlled replay versus the existing batched production tensor layout."""

    def __init__(self, args: argparse.Namespace):
        torch = __import__("torch")
        if args.tp_size != 1:
            raise RuntimeError("rank-aware proof collection is not implemented")
        if not torch.cuda.is_available():
            raise RuntimeError("Cluster-2 dynamic validation requires CUDA")
        torch.cuda.set_device(args.device)
        self.args = args
        self.cluster1 = _load_cluster1_module()

        class Runtime(self.cluster1.Cluster1ModelRuntime):
            @staticmethod
            def _server_args_kwargs(
                runtime_args: argparse.Namespace, config_path: Path
            ) -> dict[str, Any]:
                values = Runtime.__mro__[1]._server_args_kwargs(
                    runtime_args, config_path
                )
                values.update(
                    max_running_requests=4,
                    max_total_tokens=runtime_args.max_total_tokens,
                )
                return values

        runtime_args = SimpleNamespace(**vars(args))
        runtime_args.model_dir = args.model_path
        self.runtime = Runtime(runtime_args)
        self.model_runner = self.runtime.model_runner
        self.device = self.runtime.device
        self.revision = _git_revision()
        self.model_scale = infer_model_scale(self.model_runner)

    def _tokens(self, case: ValidationCase) -> tuple[list[list[int]], list[list[int]]]:
        torch = __import__("torch")
        prefixes, actives = [], []
        vocab = int(self.model_runner.model_config.vocab_size)
        mask_id = int(self.runtime.dllm_config.mask_id)
        for bid in range(case.batch_size):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(case.token_seed + bid)
            values = torch.randint(
                0,
                vocab,
                (case.prefix_length + case.active_length,),
                generator=generator,
                dtype=torch.int64,
            ).tolist()
            values = [0 if value == mask_id else int(value) for value in values]
            prefixes.append(values[: case.prefix_length])
            actives.append(values[case.prefix_length :])
        return prefixes, actives

    def _prepare_batch(self, reqs: Sequence[Any], *, bidir: bool) -> tuple[Any, Any]:
        from sglang.srt.dllm.config import (
            DLLM_ATTN_MASK_BIDIR_BLOCK,
            DLLM_ATTN_MASK_CAUSAL_PREFILL,
        )
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        tree_cache = SimpleNamespace(
            page_size=self.model_runner.server_args.page_size,
            device=self.device,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            supports_swa=lambda: False,
            supports_mamba=lambda: False,
            is_chunk_cache=lambda: False,
            is_tree_cache=lambda: True,
            evict=lambda _params: None,
        )
        for req in reqs:
            self.runtime._validate_req_prefix_indices(
                req, int(req.dllm_block_offset) if bidir else 0
            )
        batch = ScheduleBatch.init_new(
            reqs=list(reqs),
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            dllm_config=self.runtime.dllm_config,
        )
        batch.prepare_for_extend()
        batch._dllm_attn_mask_types_cpu = [
            DLLM_ATTN_MASK_BIDIR_BLOCK if bidir else DLLM_ATTN_MASK_CAUSAL_PREFILL
        ] * len(reqs)
        forward_batch = ForwardBatch.init_new(
            batch.get_model_worker_batch(), self.model_runner
        )
        if bidir:
            active = int(reqs[0].extend_input_len)
            self.runtime._configure_active_mask(
                forward_batch, active_length=active, device=self.device
            )
            forward_batch.dllm_gdn_causal_mode = 0
            forward_batch.dllm_gdn_block_size = active
        else:
            forward_batch.dllm_force_causal = True
            forward_batch.dllm_gdn_causal_mode = 1
        forward_batch.dllm_gdn_persist_state = True
        return batch, forward_batch

    def _run_prefix_batch(self, reqs: Sequence[Any]) -> None:
        if not reqs or not reqs[0].origin_input_ids:
            return
        _, forward_batch = self._prepare_batch(reqs, bidir=False)
        self.runtime._forward(forward_batch)
        self.runtime._synchronize()

    def _prepare_prefix_indices(self, reqs: Sequence[Any], boundary: int) -> None:
        """Preserve canonical empties or capture real allocated prefix locations."""
        torch = __import__("torch")
        boundary = int(boundary)
        if boundary < 0:
            raise RuntimeError("prefix boundary must be non-negative")
        expected_device = self.model_runner.req_to_token_pool.req_to_token.device
        for request_index, req in enumerate(reqs):
            if boundary == 0:
                locations = getattr(req, "prefix_indices", None)
                errors = []
                if not torch.is_tensor(locations):
                    errors.append("not a tensor")
                else:
                    if locations.dtype != torch.int64:
                        errors.append(f"dtype={locations.dtype}, expected=torch.int64")
                    if locations.ndim != 1:
                        errors.append(f"ndim={locations.ndim}, expected=1")
                    if not locations.is_contiguous():
                        errors.append("tensor is not contiguous")
                    if int(locations.numel()) != 0:
                        errors.append(f"numel={locations.numel()}, expected=0")
                    if locations.device != expected_device:
                        errors.append(
                            f"device={locations.device}, expected={expected_device}"
                        )
                if errors:
                    raise RuntimeError(
                        "invalid zero-prefix req.prefix_indices for request "
                        f"{request_index}: " + "; ".join(errors)
                    )
                # Keep the canonical empty from _make_req(). Do not assign a
                # synthetic request slot before the real ScheduleBatch runs.
                continue

            if getattr(req, "req_pool_idx", None) is None:
                raise RuntimeError(
                    "positive-prefix validator request has no allocated "
                    f"req_pool_idx before boundary={boundary} lookup"
                )
            req.prefix_indices = self.runtime._canonical_prefix_locations(
                req.req_pool_idx, boundary
            )

    def _run_active_batch(
        self,
        reqs: Sequence[Any],
        active_tokens: Sequence[Sequence[int]],
        hooks: ScopedRowHooks,
    ) -> tuple[dict[str, Any], Any, Any]:
        torch = __import__("torch")
        prefix = len(reqs[0].prefix_indices)
        active = len(active_tokens[0])
        for req, tokens in zip(reqs, active_tokens):
            req.dllm_block_offset = prefix
            req.fill_ids = list(req.origin_input_ids[:prefix]) + list(tokens)
            req.set_extend_input_len(active)
        batch, forward_batch = self._prepare_batch(reqs, bidir=True)
        expected_rows = len(reqs) * active
        if int(forward_batch.input_ids.numel()) != expected_rows:
            raise RuntimeError("scheduled input rows include non-active positions")
        expected_positions = torch.tensor(
            [position for _ in reqs for position in range(prefix, prefix + active)],
            dtype=forward_batch.positions.dtype,
            device=forward_batch.positions.device,
        )
        if not torch.equal(forward_batch.positions, expected_positions):
            raise RuntimeError("active absolute positions were not preserved")
        self.model_runner.attn_backend.init_forward_metadata(forward_batch)
        slots = [
            self.runtime.backend._current_mamba_slot(req.req_pool_idx) for req in reqs
        ]
        with hooks.capture(slots):
            logits = self.runtime._forward(forward_batch, metadata_prepared=True)
        self.runtime._synchronize()
        trace = hooks.snapshot()
        top1 = logits.argmax(dim=-1)
        trace.update(
            logits=_trace_tensor_to_cpu(logits),
            top1=_trace_tensor_to_cpu(top1),
            positions=_trace_tensor_to_cpu(forward_batch.positions),
            scheduled_rows=int(forward_batch.input_ids.numel()),
            kv_lengths=[int(value) for value in forward_batch.seq_lens.tolist()],
        )
        return trace, batch, forward_batch

    def _make_reqs(
        self, case: ValidationCase, path: str, prefixes: Sequence[Sequence[int]]
    ) -> list[Any]:
        return [
            self.runtime._make_req(f"{case.case_id}:{path}:{bid}", list(prefix))
            for bid, prefix in enumerate(prefixes)
        ]

    def _seal_prefixes(
        self,
        case: ValidationCase,
        reqs: Sequence[Any],
        prefixes: Sequence[Sequence[int]],
    ) -> tuple[list[Any], list[Any]]:
        if case.prefix_length == 0:
            return [], []
        keys, references = [], []
        for bid, (req, prefix) in enumerate(zip(reqs, prefixes)):
            c1_case = self.cluster1.ManifestCase(
                schema_version=1,
                case_id=f"{case.case_id}:{bid}",
                token_seed=case.token_seed + bid,
                prefix_length=case.prefix_length,
                active_length=case.active_length,
                diffusion_steps=case.diffusion_steps,
                attention_contract_id=self.cluster1.ATTENTION_CONTRACT,
            )
            key = self.runtime._make_key(c1_case, req, list(prefix))
            reference = self.runtime._kv_reference(key)
            slot = self.runtime.backend._current_mamba_slot(req.req_pool_idx)
            self.runtime.backend.commit_region_state(
                state_key=key, mamba_cache_idx=slot, kv_prefix=reference
            )
            req.prefix_indices = reference.locations.detach().clone()
            keys.append(key)
            references.append(reference)
        self.runtime._synchronize()
        return keys, references

    def _stable_hashes(self, keys: Sequence[Any]) -> tuple[str, str]:
        if not keys:
            return "", ""
        kv_tensors, gdn_tensors = [], []
        for bid, key in enumerate(keys):
            lookup = self.runtime.backend.region_state_cache.get(
                key, current_slot_generation=key.request_slot_generation
            )
            if not lookup.hit:
                raise RuntimeError("committed stable state disappeared")
            for name, tensor in self.runtime._stable_tensors(lookup.state):
                target = gdn_tensors if name.startswith("gdn.") else kv_tensors
                target.append((f"request.{bid}.{name}", tensor))
        return self.cluster1.hash_tensors(kv_tensors), self.cluster1.hash_tensors(
            gdn_tensors
        )

    def _restore(self, keys: Sequence[Any], reqs: Sequence[Any]) -> None:
        for key, req in zip(keys, reqs):
            result = self.runtime.backend.restore_region_state(
                state_key=key,
                mamba_cache_idx=self.runtime.backend._current_mamba_slot(
                    req.req_pool_idx
                ),
                current_slot_generation=key.request_slot_generation,
            )
            if not result.hit:
                raise RuntimeError(f"exact prefix restore missed: {result.miss_reason}")
        self.runtime._synchronize()

    def _run_candidate_steps(
        self,
        case: ValidationCase,
        prefixes: Sequence[Sequence[int]],
        active_inputs: Sequence[Sequence[Sequence[int]]],
        references: Sequence[Mapping[str, Any]],
        hooks: ScopedRowHooks,
    ) -> tuple[
        list[dict[str, Any]],
        list[tuple[float, float, float, bool]],
        Optional[str],
        Optional[str],
        Optional[str],
        Optional[str],
    ]:
        comparisons = []
        candidate_traces = []
        if case.prefix_length == 0:
            # No stable state exists to seal or restore. Every entirely-active
            # reevaluation gets cleared pools and fresh requests so GDN state
            # from the previous diffusion step cannot persist.
            for step, values in enumerate(active_inputs):
                self.runtime._clear_pools()
                candidate_reqs = self._make_reqs(
                    case, f"zero-prefix-candidate:{step}", prefixes
                )
                self._prepare_prefix_indices(candidate_reqs, 0)
                trace, _, _ = self._run_active_batch(candidate_reqs, values, hooks)
                candidate_traces.append(trace)
                comparisons.append(self._compare_traces(references[step], trace))
            return candidate_traces, comparisons, None, None, None, None

        self.runtime._clear_pools()
        cached_reqs = self._make_reqs(case, "cached", prefixes)
        self._run_prefix_batch(cached_reqs)
        self._prepare_prefix_indices(cached_reqs, case.prefix_length)
        keys, _ = self._seal_prefixes(case, cached_reqs, prefixes)
        stable_kv_before, stable_gdn_before = self._stable_hashes(keys)
        for step, values in enumerate(active_inputs):
            self._restore(keys, cached_reqs)
            trace, _, _ = self._run_active_batch(cached_reqs, values, hooks)
            candidate_traces.append(trace)
            comparisons.append(self._compare_traces(references[step], trace))
        stable_kv_after, stable_gdn_after = self._stable_hashes(keys)
        return (
            candidate_traces,
            comparisons,
            stable_kv_before,
            stable_kv_after,
            stable_gdn_before,
            stable_gdn_after,
        )

    @staticmethod
    def _compare_traces(
        reference: Mapping[str, Any], cached: Mapping[str, Any]
    ) -> tuple[float, float, float, bool]:
        logits = _max_abs(reference["logits"], cached["logits"])
        hidden = max(
            _max_abs(reference["hidden"][layer], cached["hidden"][layer])
            for layer in reference["hidden"]
        )
        gdn_values = []
        for layer in reference["gdn_states"]:
            gdn_values.extend(
                _max_abs(left, right)
                for left, right in zip(
                    reference["gdn_states"][layer], cached["gdn_states"][layer]
                )
            )
        if not gdn_values:
            raise RuntimeError("GDN state traces are missing")
        top1 = bool((reference["top1"] == cached["top1"]).all().item())
        return logits, hidden, max(gdn_values), top1

    def _trace_hashes(self, trace: Mapping[str, Any]) -> dict[str, str]:
        hidden = [
            (f"layer.{layer}", value)
            for layer, value in sorted(trace["hidden"].items())
        ]
        gdn = [
            (f"layer.{layer}.state.{index}", value)
            for layer, states in sorted(trace["gdn_states"].items())
            for index, value in enumerate(states)
        ]
        return {
            "active_logits": self.cluster1.hash_tensors(
                (("active_logits", trace["logits"]),)
            ),
            "active_hidden": self.cluster1.hash_tensors(hidden),
            "active_gdn": self.cluster1.hash_tensors(gdn),
            "top1_tokens": self.cluster1.hash_token_ids(trace["top1"]),
        }

    def run_case(self, case: ValidationCase) -> dict[str, Any]:
        torch = __import__("torch")
        if (
            case.batch_size
            * (case.prefix_length + case.active_length * case.diffusion_steps)
            > self.args.max_total_tokens
        ):
            raise ValueError(
                "case exceeds the configured controlled-replay token budget"
            )
        prefixes, initial_active = self._tokens(case)
        references = []
        active_inputs = []
        active = [list(values) for values in initial_active]
        peak = 0
        started = time.perf_counter()
        cuda_started = torch.cuda.Event(enable_timing=True)
        cuda_finished = torch.cuda.Event(enable_timing=True)
        torch.cuda.reset_peak_memory_stats(self.device)
        cuda_started.record()
        with (
            RuntimeLogEvidence().installed() as logs,
            ScopedRowHooks(self.model_runner) as hooks,
        ):
            for step in range(case.diffusion_steps):
                self.runtime._clear_pools()
                reqs = self._make_reqs(case, f"reference:{step}", prefixes)
                self._run_prefix_batch(reqs)
                self._prepare_prefix_indices(reqs, case.prefix_length)
                active_inputs.append([list(values) for values in active])
                trace, _, _ = self._run_active_batch(reqs, active, hooks)
                references.append(trace)
                active = (
                    trace["top1"].reshape(case.batch_size, case.active_length).tolist()
                )

            (
                cached_traces,
                comparisons,
                stable_kv_before,
                stable_kv_after,
                stable_gdn_before,
                stable_gdn_after,
            ) = self._run_candidate_steps(
                case, prefixes, active_inputs, references, hooks
            )
            cuda_finished.record()
            self.runtime._synchronize()
            cuda_elapsed_ms = float(cuda_started.elapsed_time(cuda_finished))
            peak = int(torch.cuda.max_memory_allocated(self.device))
        if not hooks.released:
            raise RuntimeError("temporary row hooks survived validation")

        expected_rows = case.expected_active_tokens

        def consistent_rows(kind: str) -> list[int]:
            first = cached_traces[0]["rows"][kind]
            if not first or any(value != expected_rows for value in first):
                return []
            if any(trace["rows"][kind] != first for trace in cached_traces[1:]):
                return []
            return first

        attention_rows = consistent_rows("attention")
        mlp_rows = consistent_rows("mlp")
        gdn_rows = consistent_rows("gdn")
        all_attention_rows = [
            value for trace in cached_traces for value in trace["rows"]["attention"]
        ]
        stable_query_tokens = (
            max(value - expected_rows for value in all_attention_rows)
            if all_attention_rows
            else None
        )
        scheduled_rows = (
            cached_traces[0]["scheduled_rows"]
            if all(
                trace["scheduled_rows"] == cached_traces[0]["scheduled_rows"]
                for trace in cached_traces
            )
            else -1
        )
        kv_lengths = (
            cached_traces[0]["kv_lengths"]
            if all(
                trace["kv_lengths"] == cached_traces[0]["kv_lengths"]
                for trace in cached_traces
            )
            else []
        )
        expected_positions = list(
            range(case.prefix_length, case.prefix_length + case.active_length)
        )
        unavailable = {}
        if case.prefix_length == 0:
            stable_kv_unchanged = stable_gdn_unchanged = None
            unavailable = {
                "stable_kv_unchanged": "no stable prefix exists for prefix length zero",
                "stable_gdn_unchanged": "no stable prefix state exists for prefix length zero",
            }
        else:
            stable_kv_unchanged = stable_kv_before == stable_kv_after
            stable_gdn_unchanged = stable_gdn_before == stable_gdn_after
        record = {
            "schema_version": SCHEMA_VERSION,
            "case_id": case.case_id,
            "profile": case.profile,
            "revision": self.revision,
            "model_scale": self.model_scale,
            "dtype": self.args.dtype,
            "tp_size": self.args.tp_size,
            "batch_size": case.batch_size,
            "prefix_tokens_per_request": case.prefix_length,
            "active_tokens_per_request": case.active_length,
            "diffusion_steps": case.diffusion_steps,
            "scheduled_input_tokens": scheduled_rows,
            "expected_active_tokens": case.expected_active_tokens,
            "stable_query_tokens": stable_query_tokens,
            "attention_query_rows_per_layer": attention_rows,
            "attention_row_observation_point": ATTENTION_ROW_OBSERVATION_POINT,
            "total_kv_tokens_per_request": kv_lengths,
            "mlp_rows_per_layer": mlp_rows,
            "gdn_rows_per_layer": gdn_rows,
            "active_position_ids_per_request": [
                _position_evidence(expected_positions) for _ in range(case.batch_size)
            ],
            "positions_preserved": all(
                trace["positions"].tolist() == expected_positions * case.batch_size
                for trace in cached_traces
            ),
            "stable_kv_unchanged": stable_kv_unchanged,
            "stable_gdn_unchanged": stable_gdn_unchanged,
            "stable_kv_hash_before": stable_kv_before,
            "stable_kv_hash_after": stable_kv_after,
            "stable_gdn_hash_before": stable_gdn_before,
            "stable_gdn_hash_after": stable_gdn_after,
            "paired_active_hashes": [
                {
                    "step": step + 1,
                    "controlled_reference": self._trace_hashes(reference),
                    "cached_production_layout": self._trace_hashes(cached),
                }
                for step, (reference, cached) in enumerate(
                    zip(references, cached_traces)
                )
            ],
            "max_logits_error": max(value[0] for value in comparisons),
            "max_hidden_error": max(value[1] for value in comparisons),
            "max_gdn_error": max(value[2] for value in comparisons),
            "top1_identical": all(value[3] for value in comparisons),
            "fallback_count": logs.fallback_count,
            "recovery_replays": logs.recovery_replays,
            "nan_or_inf_detected": False,
            "stage_latency_ms": {
                "cuda_paired_case_total": cuda_elapsed_ms,
                "host_paired_case_total": (time.perf_counter() - started) * 1000.0,
            },
            "peak_memory_bytes": peak,
            "unavailable_reasons": unavailable,
            "case_pass": False,
            "failure_reasons": [],
        }
        reasons = validate_case_record(record, case)
        record["failure_reasons"] = reasons
        record["case_pass"] = not reasons
        return record

    def hardware(self) -> dict[str, Any]:
        torch = __import__("torch")
        properties = torch.cuda.get_device_properties(self.args.device)
        return {
            "device_index": self.args.device,
            "device_name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "cuda_version": torch.version.cuda,
        }

    def close(self) -> None:
        self.runtime.close()


def _git_revision(repo_root: Path = ROOT) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_validation(
    args: argparse.Namespace,
    *,
    runtime_factory: Callable[[argparse.Namespace], Any] = Cluster2ValidationRuntime,
) -> dict[str, Any]:
    cases = build_manifest(args.profile, args.seed)
    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    runtime = runtime_factory(args)
    records = []
    primary_error: Optional[BaseException] = None
    hardware: dict[str, Any] = {"unavailable_reason": "runtime did not report hardware"}
    try:
        with output_path.open("w", encoding="utf-8") as destination:
            for case in cases:
                try:
                    record = runtime.run_case(case)
                except BaseException as exc:
                    primary_error = exc
                    break
                records.append(record)
                destination.write(json.dumps(record, sort_keys=True) + "\n")
                destination.flush()
    finally:
        try:
            hardware = runtime.hardware()
        except BaseException:
            if primary_error is None:
                primary_error = sys.exception()
        try:
            runtime.close()
        except BaseException:
            if primary_error is None:
                primary_error = sys.exception()
    diffs = prohibited_runtime_diffs()
    summary = build_summary(
        cases,
        records,
        revision=_git_revision(),
        hardware=hardware,
        dtype=args.dtype,
        tp_size=args.tp_size,
        profile=args.profile,
        runtime_diff_is_empty=not diffs,
        runtime_diff_files=diffs,
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if primary_error is not None:
        raise primary_error
    if not summary["strict_pass"]:
        raise RuntimeError("Cluster-2 validation failed; inspect the summary JSON")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_validation(parse_args(argv))


if __name__ == "__main__":
    main()
