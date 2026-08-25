#!/usr/bin/env python3
"""Three-way A30 diagnostic for Cluster-3 layer-0 GDN equivalence."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]
CLUSTER3_PATH = Path(__file__).with_name("cluster3_region_dag_validation.py")
OUTPUT_NAME = "one1-three-way-diagnostic.json"
TENSOR_TOLERANCE = 1e-2


def _load_cluster3() -> Any:
    name = "cluster3_gdn_equivalence_runtime"
    spec = importlib.util.spec_from_file_location(name, CLUSTER3_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Cluster-3 validator {CLUSTER3_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run monolithic, freshly segmented, and cached Cluster-3 one1 paths "
            "with layer-0 tensor evidence."
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--max-total-tokens", type=int, default=4096)
    parser.add_argument("--debug-sync-stages", action="store_true")
    return parser.parse_args(argv)


def execute_fresh_segmented_reference(
    *,
    clear_state: Callable[[], None],
    create_request: Callable[[], Any],
    recompute_prefix: Callable[[Any], Any],
    region_cache_entry_count: Callable[[], int],
    prepare_suffix: Callable[[Any, Any], Any],
    execute_suffix: Callable[[Any, Any], Any],
) -> Any:
    """Run a prefix from zero and continue its live state without a cache hit."""
    clear_state()
    if region_cache_entry_count() != 0:
        raise RuntimeError(
            "fresh segmented reference did not start with an empty cache"
        )
    request = create_request()
    prefix_result = recompute_prefix(request)
    if region_cache_entry_count() != 0:
        raise RuntimeError(
            "fresh segmented prefix unexpectedly published Region-DAG cache state"
        )
    suffix = prepare_suffix(request, prefix_result)
    if bool(getattr(suffix, "restore_required", True)):
        raise RuntimeError("fresh segmented reference requested a cache restoration")
    result = execute_suffix(request, suffix)
    if int(getattr(result, "restore_calls", -1)) != 0:
        raise RuntimeError("fresh segmented reference observed a cache restoration")
    return result


def _first_tensor(value: Any) -> Any:
    torch = __import__("torch")
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _resolve_hook_tensor(
    inputs: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    preferred_keyword: Optional[str] = None,
    evidence_name: str,
) -> Any:
    """Resolve hook evidence across positional and keyword module calls."""
    if preferred_keyword is not None and preferred_keyword in kwargs:
        tensor = _first_tensor(kwargs[preferred_keyword])
        if tensor is not None:
            return tensor

    tensor = _first_tensor(inputs)
    if tensor is not None:
        return tensor

    # Controlled compatibility fallback for modules with a different keyword
    # parameter name. Forward-hook kwargs are ordered by the Python call.
    tensor = _first_tensor(tuple(kwargs.values()))
    if tensor is not None:
        return tensor

    raise RuntimeError(
        f"{evidence_name} hook received no tensor input; "
        f"positional_count={len(inputs)}, keyword_keys={sorted(kwargs)}"
    )


def _cpu_tensor(value: Any) -> Any:
    torch = __import__("torch")
    if not torch.is_tensor(value):
        raise TypeError("diagnostic evidence must be a tensor")
    return value.detach().to(device="cpu", copy=True).contiguous()


def _tensor_sha256(value: Any) -> str:
    tensor = _cpu_tensor(value)
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.view(dtype=__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def _tensor_description(value: Any) -> dict[str, Any]:
    tensor = _cpu_tensor(value)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": _tensor_sha256(tensor),
    }


def _tensor_provenance(value: Any) -> dict[str, Any]:
    if not hasattr(value, "stride") or not hasattr(value, "storage_offset"):
        raise TypeError("frontier provenance requires a tensor")
    return {
        **_tensor_description(value),
        "device": str(value.device),
        "stride": list(value.stride()),
        "storage_offset": int(value.storage_offset()),
        "contiguous": bool(value.is_contiguous()),
    }


def _frontier_identity(frontier_key: Any) -> dict[str, Any]:
    preceding = tuple(getattr(frontier_key, "preceding_regions", ()))
    return {
        "request_id": str(frontier_key.request_id),
        "request_pool_slot": int(frontier_key.request_pool_idx),
        "boundary_position": int(frontier_key.boundary),
        "frontier_key": (
            frontier_key.to_dict()
            if callable(getattr(frontier_key, "to_dict", None))
            else repr(frontier_key)
        ),
        "region_versions": [
            [str(region_id), int(version)]
            for region_id, version, _parents, _token_hash, _position_hash in preceding
        ],
        "parent_versions": [
            [
                str(region_id),
                [[str(parent), int(version)] for parent, version in parents],
            ]
            for region_id, _version, parents, _token_hash, _position_hash in preceding
        ],
        "token_hashes": [
            [str(region_id), str(token_hash)]
            for region_id, _version, _parents, token_hash, _position_hash in preceding
        ],
        "position_hashes": [
            [str(region_id), str(position_hash)]
            for region_id, _version, _parents, _token_hash, position_hash in preceding
        ],
    }


def _tensor_difference(left: Any, right: Any) -> dict[str, Any]:
    torch = __import__("torch")
    left = _cpu_tensor(left)
    right = _cpu_tensor(right)
    evidence = {
        "reference": _tensor_description(left),
        "candidate": _tensor_description(right),
        "shape_equal": tuple(left.shape) == tuple(right.shape),
        "dtype_equal": left.dtype == right.dtype,
    }
    if not evidence["shape_equal"]:
        evidence.update(
            exact_equal=False,
            max_abs_error=None,
            mean_abs_error=None,
            first_mismatching_flat_index=None,
            values_above_1e_2=None,
        )
        return evidence
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    maximum = 0.0
    absolute_sum = 0.0
    first_mismatch = None
    above = 0
    chunk_size = 1 << 20
    for start in range(0, int(left_flat.numel()), chunk_size):
        stop = min(start + chunk_size, int(left_flat.numel()))
        delta = (left_flat[start:stop].float() - right_flat[start:stop].float()).abs()
        if not bool(torch.isfinite(delta).all().item()):
            raise RuntimeError("diagnostic tensor difference contains NaN or Inf")
        maximum = max(maximum, float(delta.max().item()))
        absolute_sum += float(delta.double().sum().item())
        above += int((delta >= TENSOR_TOLERANCE).sum().item())
        if first_mismatch is None:
            indices = torch.nonzero(delta != 0, as_tuple=False)
            if indices.numel():
                first_mismatch = start + int(indices[0].item())
    evidence.update(
        exact_equal=bool(torch.equal(left, right)),
        max_abs_error=maximum,
        mean_abs_error=absolute_sum / max(int(left_flat.numel()), 1),
        first_mismatching_flat_index=first_mismatch,
        values_above_1e_2=above,
    )
    return evidence


class LayerZeroEvidence:
    """Scoped real-module hooks for the first Qwen3.5 GDN decoder layer."""

    ROW_TENSORS = frozenset(
        {
            "decoder_layer_input_hidden_states",
            "input_layernorm_output",
            "in_proj_qkvz_output",
            "in_proj_ba_output",
            "mixed_qkv",
            "z",
            "a",
            "b",
            "gdn_core_output",
            "gdn_output_before_output_projection",
            "output_projection_result",
            "decoder_layer_output_hidden",
            "decoder_layer_output_residual",
            "decoder_layer_output",
        }
    )
    REQUIRED = ROW_TENSORS | frozenset(
        {"pre_suffix_convolution_state", "pre_suffix_recurrent_state"}
    )

    def __init__(self, runtime: Any):
        torch = __import__("torch")
        nn = torch.nn
        self.runtime = runtime
        model = runtime.cluster1.ModelTraceHooks._language_model(
            runtime.model_runner.model
        )
        self.layer = model.layers[0]
        self.linear = getattr(self.layer, "linear_attn", None)
        if not isinstance(self.layer, nn.Module) or not isinstance(
            self.linear, nn.Module
        ):
            raise RuntimeError("model layer 0 is not a hookable GDN decoder layer")
        self.input_norm = getattr(self.layer, "input_layernorm", None)
        self.qkvz = getattr(self.linear, "in_proj_qkvz", None)
        self.ba = getattr(self.linear, "in_proj_ba", None)
        self.attn = getattr(self.linear, "attn", None)
        self.out_proj = getattr(self.linear, "out_proj", None)
        for name, target in (
            ("input_layernorm", self.input_norm),
            ("in_proj_qkvz", self.qkvz),
            ("in_proj_ba", self.ba),
            ("attn", self.attn),
            ("out_proj", self.out_proj),
        ):
            if not isinstance(target, nn.Module):
                raise RuntimeError(f"layer 0 has no hookable {name} module")
        self.layer_id = int(getattr(self.layer, "layer_id", 0))
        if self.layer_id != 0:
            raise RuntimeError(f"first decoder layer has unexpected ID {self.layer_id}")
        self.handles: list[Any] = []
        self.values: dict[str, dict[str, Any]] = {}
        self.rows: dict[str, int] = {}
        self.restore_calls: dict[str, int] = {}
        self.active_label: Optional[str] = None
        self._inside_linear = False
        try:
            self._install_hooks()
            self._install_fused_capture()
            self._install_restore_capture()
        except BaseException:
            self.close()
            raise

    def _canonical_rows(self, value: Any, rows: int) -> Any:
        tensor = _cpu_tensor(value)
        if tensor.ndim > 0 and int(tensor.shape[0]) == rows:
            return tensor.reshape(rows, -1).contiguous()
        if (
            tensor.ndim > 1
            and int(tensor.shape[0]) == 1
            and int(tensor.shape[1]) == rows
        ):
            return tensor.squeeze(0).reshape(rows, -1).contiguous()
        raise RuntimeError(
            f"layer-0 row tensor shape {tuple(tensor.shape)} does not contain {rows} rows"
        )

    def _record(self, name: str, value: Any) -> None:
        if self.active_label is None:
            return
        tensor = _first_tensor(value)
        if tensor is None:
            raise RuntimeError(f"layer-0 hook {name} produced no tensor")
        if name in self.ROW_TENSORS:
            tensor = self._canonical_rows(tensor, self.rows[self.active_label])
        else:
            tensor = _cpu_tensor(tensor)
        self.values[self.active_label][name] = tensor

    def record_state(self, label: str, conv: Any, recurrent: Any) -> None:
        self.values.setdefault(label, {})["pre_suffix_convolution_state"] = _cpu_tensor(
            conv
        )
        self.values.setdefault(label, {})["pre_suffix_recurrent_state"] = _cpu_tensor(
            recurrent
        )

    def _install_hooks(self) -> None:
        def layer_pre(_module: Any, inputs: tuple[Any, ...], kwargs: Mapping[str, Any]):
            if self.active_label is not None:
                self._record(
                    "decoder_layer_input_hidden_states",
                    _resolve_hook_tensor(
                        inputs,
                        kwargs,
                        preferred_keyword="hidden_states",
                        evidence_name="decoder_layer_input_hidden_states",
                    ),
                )

        def layer_post(_module: Any, _inputs: Any, output: Any):
            if self.active_label is None:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            self._record("decoder_layer_output_hidden", hidden)
            if isinstance(output, tuple) and len(output) > 1:
                residual = _first_tensor(output[1])
                if residual is not None and residual.shape == hidden.shape:
                    self._record("decoder_layer_output_residual", residual)
                    self._record("decoder_layer_output", hidden + residual)
                    return
            self._record(
                "decoder_layer_output_residual", hidden.new_zeros(hidden.shape)
            )
            self._record("decoder_layer_output", hidden)

        def output_hook(name: str) -> Callable[..., None]:
            def hook(_module: Any, _inputs: Any, output: Any):
                self._record(name, output)

            return hook

        def pre_hook(name: str) -> Callable[..., None]:
            def hook(
                _module: Any,
                inputs: tuple[Any, ...],
                kwargs: Mapping[str, Any],
            ):
                if self.active_label is not None:
                    self._record(
                        name,
                        _resolve_hook_tensor(
                            inputs,
                            kwargs,
                            evidence_name=name,
                        ),
                    )

            return hook

        def linear_pre(
            _module: Any, _inputs: tuple[Any, ...], _kwargs: Mapping[str, Any]
        ):
            self._inside_linear = self.active_label is not None

        def linear_post(_module: Any, _inputs: Any, _output: Any):
            self._inside_linear = False

        self.handles.extend(
            (
                self.layer.register_forward_pre_hook(layer_pre, with_kwargs=True),
                self.layer.register_forward_hook(layer_post),
                self.input_norm.register_forward_hook(
                    output_hook("input_layernorm_output")
                ),
                self.qkvz.register_forward_hook(output_hook("in_proj_qkvz_output")),
                self.ba.register_forward_hook(output_hook("in_proj_ba_output")),
                self.linear.register_forward_pre_hook(linear_pre, with_kwargs=True),
                self.linear.register_forward_hook(linear_post),
                self.attn.register_forward_hook(output_hook("gdn_core_output")),
                self.out_proj.register_forward_pre_hook(
                    pre_hook("gdn_output_before_output_projection"), with_kwargs=True
                ),
                self.out_proj.register_forward_hook(
                    output_hook("output_projection_result")
                ),
            )
        )

    def _install_fused_capture(self) -> None:
        module = sys.modules[type(self.linear).__module__]
        self._qwen_module = module
        self._original_fused = getattr(
            module, "fused_qkvzba_split_reshape_cat_contiguous", None
        )
        if not callable(self._original_fused):
            raise RuntimeError("Qwen layer-0 fused qkvzba splitter is unavailable")

        def captured(*args: Any, **kwargs: Any) -> Any:
            result = self._original_fused(*args, **kwargs)
            if self._inside_linear and self.active_label is not None:
                for name, value in zip(("mixed_qkv", "z", "b", "a"), result):
                    self._record(name, value)
            return result

        setattr(module, "fused_qkvzba_split_reshape_cat_contiguous", captured)

    def _install_restore_capture(self) -> None:
        target = self.runtime.backend
        name = "_restore_region_dag_layer_snapshot"
        original = getattr(target, name)
        namespace = getattr(target, "__dict__", {})
        self._restore_had_instance_value = name in namespace
        self._restore_instance_value = namespace.get(name)
        self._original_restore = original

        def captured_restore(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            label = self.active_label
            if label is not None:
                self.restore_calls[label] = self.restore_calls.get(label, 0) + 1
                layer_id = int(kwargs.get("layer_id", -1))
                if layer_id == self.layer_id:
                    self.record_state(
                        label,
                        kwargs["conv_destination"],
                        kwargs["recurrent_destination"],
                    )
            return result

        setattr(target, name, captured_restore)

    @contextlib.contextmanager
    def capture(self, label: str, rows: int) -> Iterator[None]:
        if self.active_label is not None:
            raise RuntimeError("nested layer-0 diagnostic capture is not allowed")
        self.active_label = label
        self.rows[label] = int(rows)
        # A fresh segmented path records its live pre-suffix state immediately
        # before entering the model. Preserve only those state tensors when the
        # row-capture scope begins.
        existing = self.values.get(label, {}) if label == "B_segmented" else {}
        self.values[label] = {
            name: value
            for name, value in existing.items()
            if name in ("pre_suffix_convolution_state", "pre_suffix_recurrent_state")
        }
        self.restore_calls[label] = 0
        try:
            yield
        finally:
            self._inside_linear = False
            self.active_label = None

    def validate(self, label: str) -> None:
        missing = sorted(self.REQUIRED - set(self.values.get(label, {})))
        if missing:
            raise RuntimeError(f"{label} is missing layer-0 evidence: {missing}")

    def close(self) -> None:
        self.active_label = None
        self._inside_linear = False
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if hasattr(self, "_qwen_module"):
            setattr(
                self._qwen_module,
                "fused_qkvzba_split_reshape_cat_contiguous",
                self._original_fused,
            )
        if hasattr(self, "_original_restore"):
            target = self.runtime.backend
            name = "_restore_region_dag_layer_snapshot"
            if self._restore_had_instance_value:
                setattr(target, name, self._restore_instance_value)
            else:
                delattr(target, name)

    @property
    def released(self) -> bool:
        return not self.handles and self.active_label is None

    def __enter__(self) -> "LayerZeroEvidence":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


class FrontierStateProvenance:
    """Scoped B/L/S/R/D trace around canonical frontier state operations."""

    def __init__(self, runtime: Any, boundary: int):
        self.runtime = runtime
        self.backend = runtime.backend
        self.boundary = int(boundary)
        self.active_step: Optional[int] = None
        self.active_phase: Optional[str] = None
        self._generation = 0
        self._records: dict[int, dict[int, dict[str, dict[str, Any]]]] = {}
        self._tensors: dict[int, dict[int, dict[str, dict[str, Any]]]] = {}
        self._snapshot_generations: dict[tuple[int, int], int] = {}
        self._targets: list[tuple[Any, str, bool, Any]] = []
        try:
            self._install_publication_trace()
            self._install_restore_trace()
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _storage_pointer(tensor: Any) -> int:
        return int(tensor.untyped_storage().data_ptr())

    def _checkpoint(
        self,
        *,
        checkpoint: str,
        state_kind: str,
        tensor: Any,
        frontier_key: Any,
        layer_id: int,
        mamba_state_slot: int,
        snapshot_generation: int,
        source_slot_generation: int,
        destination_slot_generation: int,
    ) -> None:
        if self.active_step is None:
            raise RuntimeError("frontier provenance has no active diffusion step")
        step_records = self._records.setdefault(self.active_step, {})
        layer_records = step_records.setdefault(int(layer_id), {})
        state_records = layer_records.setdefault(state_kind, {})
        if checkpoint in state_records:
            raise RuntimeError(
                f"duplicate frontier checkpoint {checkpoint} for layer {layer_id}"
            )
        state_records[checkpoint] = {
            **_tensor_provenance(tensor),
            **_frontier_identity(frontier_key),
            "checkpoint": checkpoint,
            "state_kind": state_kind,
            "mamba_state_slot": int(mamba_state_slot),
            "layer_id": int(layer_id),
            "snapshot_generation": int(snapshot_generation),
            "source_slot_generation": int(source_slot_generation),
            "destination_slot_generation": int(destination_slot_generation),
        }
        step_tensors = self._tensors.setdefault(self.active_step, {})
        layer_tensors = step_tensors.setdefault(int(layer_id), {})
        layer_tensors.setdefault(state_kind, {})[checkpoint] = _cpu_tensor(tensor)

    def _install(self, name: str, replacement_factory: Callable[[Any], Any]) -> None:
        namespace = getattr(self.backend, "__dict__", {})
        had_instance_value = name in namespace
        instance_value = namespace.get(name)
        original = getattr(self.backend, name, None)
        if not callable(original):
            raise RuntimeError(f"frontier provenance cannot observe {name}")
        setattr(self.backend, name, replacement_factory(original))
        self._targets.append((self.backend, name, had_instance_value, instance_value))

    def _install_publication_trace(self) -> None:
        def replacement(original: Callable[..., Any]) -> Callable[..., Any]:
            def traced(*args: Any, **kwargs: Any) -> Any:
                frontier_key = kwargs["frontier_key"]
                layer_id = int(kwargs["layer_id"])
                should_capture = (
                    self.active_phase == "snapshot"
                    and int(frontier_key.boundary) == self.boundary
                )
                if should_capture:
                    self._generation += 1
                    generation = self._generation
                    assert self.active_step is not None
                    self._snapshot_generations[(self.active_step, layer_id)] = (
                        generation
                    )
                    mamba_slot = self.backend._current_mamba_slot(
                        int(frontier_key.request_pool_idx)
                    )
                    for state_kind, source in (
                        ("convolution", kwargs["conv_state"]),
                        ("recurrent", kwargs["recurrent_state"]),
                    ):
                        self._checkpoint(
                            checkpoint="L",
                            state_kind=state_kind,
                            tensor=source,
                            frontier_key=frontier_key,
                            layer_id=layer_id,
                            mamba_state_slot=mamba_slot,
                            snapshot_generation=generation,
                            source_slot_generation=int(
                                frontier_key.request_slot_generation
                            ),
                            destination_slot_generation=int(
                                frontier_key.request_slot_generation
                            ),
                        )
                result = original(*args, **kwargs)
                if not should_capture:
                    return result
                snapshot = self.backend._region_dag_layer_snapshots.get(
                    (frontier_key, layer_id)
                )
                if snapshot is None:
                    raise RuntimeError("published frontier snapshot is unavailable")
                for state_kind, source, stored in (
                    ("convolution", kwargs["conv_state"], snapshot.conv_state),
                    (
                        "recurrent",
                        kwargs["recurrent_state"],
                        snapshot.recurrent_state,
                    ),
                ):
                    if self._storage_pointer(source) == self._storage_pointer(stored):
                        raise RuntimeError(
                            f"{state_kind} frontier snapshot aliases its live source"
                        )
                    self._checkpoint(
                        checkpoint="S",
                        state_kind=state_kind,
                        tensor=stored,
                        frontier_key=frontier_key,
                        layer_id=layer_id,
                        mamba_state_slot=mamba_slot,
                        snapshot_generation=generation,
                        source_slot_generation=int(
                            frontier_key.request_slot_generation
                        ),
                        destination_slot_generation=int(
                            frontier_key.request_slot_generation
                        ),
                    )
                return result

            return traced

        self._install("_put_region_dag_layer_snapshot", replacement)

    def _install_restore_trace(self) -> None:
        def replacement(original: Callable[..., Any]) -> Callable[..., Any]:
            def traced(*args: Any, **kwargs: Any) -> Any:
                frontier_key = kwargs["frontier_key"]
                layer_id = int(kwargs["layer_id"])
                should_capture = (
                    self.active_phase == "restore"
                    and int(frontier_key.boundary) == self.boundary
                )
                if not should_capture:
                    return original(*args, **kwargs)
                snapshot = self.backend._region_dag_layer_snapshots.get(
                    (frontier_key, layer_id)
                )
                if snapshot is None:
                    raise RuntimeError("authoritative frontier snapshot is unavailable")
                assert self.active_step is not None
                generation = self._snapshot_generations.get(
                    (self.active_step, layer_id)
                )
                if generation is None:
                    raise RuntimeError(
                        "restored snapshot has no publication generation"
                    )
                mamba_slot = self.backend._current_mamba_slot(
                    int(frontier_key.request_pool_idx)
                )
                for state_kind, stored in (
                    ("convolution", snapshot.conv_state),
                    ("recurrent", snapshot.recurrent_state),
                ):
                    self._checkpoint(
                        checkpoint="R",
                        state_kind=state_kind,
                        tensor=stored,
                        frontier_key=frontier_key,
                        layer_id=layer_id,
                        mamba_state_slot=mamba_slot,
                        snapshot_generation=generation,
                        source_slot_generation=int(
                            frontier_key.request_slot_generation
                        ),
                        destination_slot_generation=int(
                            frontier_key.request_slot_generation
                        ),
                    )
                result = original(*args, **kwargs)
                for state_kind, stored, destination in (
                    (
                        "convolution",
                        snapshot.conv_state,
                        kwargs["conv_destination"],
                    ),
                    (
                        "recurrent",
                        snapshot.recurrent_state,
                        kwargs["recurrent_destination"],
                    ),
                ):
                    if self._storage_pointer(stored) == self._storage_pointer(
                        destination
                    ):
                        raise RuntimeError(
                            f"restored {state_kind} destination aliases cache storage"
                        )
                    self._checkpoint(
                        checkpoint="D",
                        state_kind=state_kind,
                        tensor=destination,
                        frontier_key=frontier_key,
                        layer_id=layer_id,
                        mamba_state_slot=mamba_slot,
                        snapshot_generation=generation,
                        source_slot_generation=int(
                            frontier_key.request_slot_generation
                        ),
                        destination_slot_generation=int(
                            frontier_key.request_slot_generation
                        ),
                    )
                return result

            return traced

        self._install("_restore_region_dag_layer_snapshot", replacement)

    def start_step(self, step: int) -> None:
        if self.active_phase is not None:
            raise RuntimeError("cannot change provenance step during an active phase")
        self.active_step = int(step)
        self._records[self.active_step] = {}
        self._tensors[self.active_step] = {}

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if name not in ("snapshot", "restore"):
            raise ValueError(f"unsupported frontier provenance phase {name}")
        if self.active_step is None or self.active_phase is not None:
            raise RuntimeError("invalid frontier provenance phase lifecycle")
        self.active_phase = name
        try:
            yield
        finally:
            self.active_phase = None

    def record_reference_frontier(self, req: Any) -> None:
        if self.active_step is None:
            raise RuntimeError("frontier provenance has no active diffusion step")
        frontier_key = req.region_dag_frontier_keys.get(self.boundary)
        if frontier_key is None:
            raise RuntimeError("live prefix has no exact frontier identity")
        mamba_slot = self.backend._current_mamba_slot(int(req.req_pool_idx))
        for layer_id in self.backend.gdn_layer_ids:
            cache = self.runtime.model_runner.req_to_token_pool.mamba2_layer_cache(
                layer_id
            )
            for state_kind, tensor in (
                ("convolution", cache.conv[0][mamba_slot]),
                ("recurrent", cache.temporal[mamba_slot]),
            ):
                self._checkpoint(
                    checkpoint="B",
                    state_kind=state_kind,
                    tensor=tensor,
                    frontier_key=frontier_key,
                    layer_id=layer_id,
                    mamba_state_slot=mamba_slot,
                    snapshot_generation=0,
                    source_slot_generation=int(req.hybrid_request_slot_generation),
                    destination_slot_generation=int(
                        frontier_key.request_slot_generation
                    ),
                )

    def result(self) -> dict[str, Any]:
        if self.active_step is None:
            raise RuntimeError("frontier provenance has no completed step")
        records = self._records[self.active_step]
        tensors = self._tensors[self.active_step]
        required_layers = sorted(int(value) for value in self.backend.gdn_layer_ids)
        missing_layers = sorted(set(required_layers) - set(records))
        if missing_layers:
            raise RuntimeError(
                f"frontier provenance is missing layers {missing_layers}"
            )
        comparisons: dict[str, Any] = {}
        first_unequal = None
        for layer_id in required_layers:
            comparisons[str(layer_id)] = {}
            for state_kind in ("convolution", "recurrent"):
                checkpoints = tensors[layer_id].get(state_kind, {})
                missing = [
                    name
                    for name in ("B", "L", "S", "R", "D")
                    if name not in checkpoints
                ]
                if missing:
                    raise RuntimeError(
                        f"frontier provenance layer {layer_id} {state_kind} "
                        f"is missing checkpoints {missing}"
                    )
                state_comparisons = {}
                for left, right in (
                    ("B", "L"),
                    ("L", "S"),
                    ("S", "R"),
                    ("R", "D"),
                    ("D", "B"),
                ):
                    name = f"{left}_vs_{right}"
                    evidence = _tensor_difference(checkpoints[left], checkpoints[right])
                    state_comparisons[name] = evidence
                    if first_unequal is None and not evidence["exact_equal"]:
                        first_unequal = {
                            "layer_id": layer_id,
                            "state_kind": state_kind,
                            "comparison": name,
                        }
                comparisons[str(layer_id)][state_kind] = state_comparisons
        return {
            "boundary": self.boundary,
            "checkpoints": {
                str(layer): values for layer, values in sorted(records.items())
            },
            "comparisons": comparisons,
            "first_unequal_checkpoint": first_unequal,
            "all_bitwise_equal": first_unequal is None,
        }

    def close(self) -> None:
        self.active_phase = None
        self.active_step = None
        for target, name, had_instance_value, instance_value in reversed(self._targets):
            if had_instance_value:
                setattr(target, name, instance_value)
            else:
                delattr(target, name)
        self._targets.clear()

    @property
    def released(self) -> bool:
        return not self._targets and self.active_phase is None

    def __enter__(self) -> "FrontierStateProvenance":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def _suffix_capture(values: Mapping[str, Any], boundary: int) -> dict[str, Any]:
    return {
        name: (value[boundary:] if name in LayerZeroEvidence.ROW_TENSORS else value)
        for name, value in values.items()
    }


def _capture_comparison(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    names = sorted(set(left) | set(right))
    return {
        name: (
            _tensor_difference(left[name], right[name])
            if name in left and name in right
            else {"missing_from": "reference" if name not in left else "candidate"}
        )
        for name in names
    }


def _top1_evidence(left: Any, right: Any, positions: Sequence[int]) -> dict[str, Any]:
    torch = __import__("torch")
    left = _cpu_tensor(left).reshape(-1)
    right = _cpu_tensor(right).reshape(-1)
    indices = torch.nonzero(left != right, as_tuple=False).reshape(-1).tolist()
    return {
        "identical": not indices,
        "mismatch_count": len(indices),
        "mismatch_preview": [
            {
                "row": int(index),
                "absolute_position": int(positions[int(index)]),
                "reference_token": int(left[int(index)]),
                "candidate_token": int(right[int(index)]),
            }
            for index in indices[:32]
        ],
    }


class ThreeWayDiagnostic:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cluster3 = _load_cluster3()
        runtime_args = SimpleNamespace(**vars(args))
        runtime_args.output_jsonl = ""
        runtime_args.summary_json = ""
        runtime_args.profile = "one1"
        runtime_args.timed_repetitions = 1
        self.runtime = self.cluster3.Cluster3ValidationRuntime(runtime_args)
        self.torch = __import__("torch")

    def _layer_zero_snapshot(self, req: Any, boundary: int) -> tuple[Any, Any]:
        key = req.region_dag_frontier_keys[int(boundary)]
        snapshot = self.runtime.backend._region_dag_layer_snapshots.get((key, 0))
        if snapshot is None:
            raise RuntimeError("monolithic reference has no layer-0 boundary snapshot")
        return snapshot.conv_state, snapshot.recurrent_state

    def _layer_zero_live_state(self, req: Any) -> tuple[Any, Any]:
        slot = self.runtime.backend._current_mamba_slot(int(req.req_pool_idx))
        cache = self.runtime.model_runner.req_to_token_pool.mamba2_layer_cache(0)
        return cache.conv[0][slot], cache.temporal[slot]

    def _path_metadata(
        self, forward_batch: Any, trace: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "input_ids": _cpu_tensor(forward_batch.input_ids),
            "positions": _cpu_tensor(forward_batch.positions),
            "mask": _cpu_tensor(trace["mask"]),
        }

    def run(self) -> dict[str, Any]:
        case = self.cluster3.build_manifest("one1", self.args.seed)[0]
        original_tokens = self.runtime._tokens(case)
        initial_plan = self.cluster3.expected_plan(
            self.cluster3.build_execution_spec(case, original_tokens, edited=True),
            case.edited_regions,
        )
        provenance_boundary = int(initial_plan.gdn_replay_start)
        invalidated = set(initial_plan.logical_invalidation_regions)
        diffusion_positions = tuple(
            position
            for region in case.regions
            if region.region_id in invalidated
            for position in range(region.start, region.end)
        )
        edited_tokens = list(original_tokens)
        for position in diffusion_positions:
            edited_tokens[position] = int(self.runtime.runtime.dllm_config.mask_id)

        steps = []
        with (
            self.cluster3.RuntimeLogEvidence().installed() as logs,
            self.runtime.cluster2.ScopedRowHooks(
                self.runtime.model_runner
            ) as row_hooks,
            LayerZeroEvidence(self.runtime) as layer_zero,
            FrontierStateProvenance(
                self.runtime, provenance_boundary
            ) as frontier_provenance,
        ):
            for step in range(case.diffusion_steps):
                frontier_provenance.start_step(step)
                spec = self.cluster3.build_execution_spec(
                    case, edited_tokens, edited=True
                )
                plan = self.cluster3.expected_plan(spec, case.edited_regions)
                boundary = int(plan.gdn_replay_start)
                rows = int(plan.query_count)
                if boundary != provenance_boundary:
                    raise RuntimeError(
                        "one1 provenance boundary changed across diffusion steps"
                    )

                self.runtime._clear()
                reference_req, reference_batch, reference_prepare = (
                    self.runtime._prepare_reference(
                        f"{case.case_id}:A:{step}", edited_tokens, spec, case
                    )
                )
                with layer_zero.capture("A_monolithic", case.sequence_length):
                    reference_trace, reference_forward_ms, _ = (
                        self.runtime._run_forward(reference_batch, row_hooks)
                    )
                layer_zero.record_state(
                    "A_monolithic", *self._layer_zero_snapshot(reference_req, boundary)
                )
                layer_zero.validate("A_monolithic")
                reference_meta = self._path_metadata(reference_batch, reference_trace)

                def clear_segmented() -> None:
                    self.runtime._clear()

                def create_segmented_request() -> Any:
                    req, prefix_batch, prefix_prepare = (
                        self.runtime._prepare_canonical_frontier(
                            f"{case.case_id}:B:{step}", edited_tokens, spec, case
                        )
                    )
                    return SimpleNamespace(
                        req=req,
                        prefix_batch=prefix_batch,
                        prefix_prepare=prefix_prepare,
                    )

                def recompute_prefix(state: Any) -> dict[str, float]:
                    _, prefix_ms, _ = self.runtime._run_forward(
                        state.prefix_batch, row_hooks
                    )
                    frontier_provenance.record_reference_frontier(state.req)
                    self.runtime.backend._region_dag_layer_snapshots.clear()
                    return {
                        **state.prefix_prepare,
                        "frontier_forward": prefix_ms,
                    }

                def cache_entries() -> int:
                    return len(self.runtime.backend._region_dag_layer_snapshots)

                def prepare_suffix(
                    state: Any, prefix_timing: Mapping[str, float]
                ) -> Any:
                    cache_entries_before_suffix = cache_entries()
                    forward_batch, prepare = self.runtime._prepare_frontier_suffix(
                        state.req,
                        edited_tokens,
                        spec,
                        case,
                        restore=False,
                    )
                    conv, recurrent = self._layer_zero_live_state(state.req)
                    layer_zero.record_state("B_segmented", conv, recurrent)
                    return SimpleNamespace(
                        restore_required=False,
                        forward_batch=forward_batch,
                        timings={**prefix_timing, **prepare},
                        cache_entries_before_suffix=cache_entries_before_suffix,
                    )

                def execute_suffix(_state: Any, prepared: Any) -> Any:
                    with layer_zero.capture("B_segmented", rows):
                        segmented_trace, segmented_forward_ms, _ = (
                            self.runtime._run_forward(prepared.forward_batch, row_hooks)
                        )
                    return SimpleNamespace(
                        trace=segmented_trace,
                        forward_batch=prepared.forward_batch,
                        timings={
                            **prepared.timings,
                            "forward": segmented_forward_ms,
                        },
                        cache_entries_before_suffix=(
                            prepared.cache_entries_before_suffix
                        ),
                        restore_calls=layer_zero.restore_calls["B_segmented"],
                    )

                segmented = execute_fresh_segmented_reference(
                    clear_state=clear_segmented,
                    create_request=create_segmented_request,
                    recompute_prefix=recompute_prefix,
                    region_cache_entry_count=cache_entries,
                    prepare_suffix=prepare_suffix,
                    execute_suffix=execute_suffix,
                )
                layer_zero.validate("B_segmented")
                segmented_meta = self._path_metadata(
                    segmented.forward_batch, segmented.trace
                )

                self.runtime._clear()
                base_spec = self.cluster3.build_execution_spec(
                    case, original_tokens, edited=False
                )
                baseline_req, baseline_batch, frontier_prepare = (
                    self.runtime._prepare_canonical_frontier(
                        f"{case.case_id}:C:{step}",
                        original_tokens,
                        base_spec,
                        case,
                    )
                )
                with frontier_provenance.phase("snapshot"):
                    _, frontier_forward_ms, _ = self.runtime._run_forward(
                        baseline_batch, row_hooks
                    )
                stable_before = self.runtime._reused_hash(baseline_req, boundary)
                cached_batch, cached_prepare = self.runtime._prepare_frontier_suffix(
                    baseline_req,
                    edited_tokens,
                    spec,
                    case,
                    restore=True,
                )
                with (
                    frontier_provenance.phase("restore"),
                    layer_zero.capture("C_cached", rows),
                ):
                    cached_trace, cached_forward_ms, _ = self.runtime._run_forward(
                        cached_batch, row_hooks
                    )
                layer_zero.validate("C_cached")
                stable_after = self.runtime._reused_hash(baseline_req, boundary)
                cached_meta = self._path_metadata(cached_batch, cached_trace)
                provenance_result = frontier_provenance.result()

                a_suffix = self.runtime._slice_reference(reference_trace, boundary)
                a_capture = _suffix_capture(layer_zero.values["A_monolithic"], boundary)
                b_capture = dict(layer_zero.values["B_segmented"])
                c_capture = dict(layer_zero.values["C_cached"])
                a_b_trace = self.runtime._compare(a_suffix, segmented.trace)
                b_c_trace = self.runtime._compare(segmented.trace, cached_trace)
                positions = tuple(plan.attention_query_positions)

                a_mask = reference_meta["mask"].reshape(
                    case.sequence_length, case.sequence_length
                )[boundary:]
                b_mask = segmented_meta["mask"].reshape(rows, case.sequence_length)
                c_mask = cached_meta["mask"].reshape(rows, case.sequence_length)
                input_contract = {
                    "suffix_input_ids_exact": bool(
                        self.torch.equal(
                            segmented_meta["input_ids"], cached_meta["input_ids"]
                        )
                    ),
                    "absolute_positions_exact": bool(
                        self.torch.equal(
                            segmented_meta["positions"], cached_meta["positions"]
                        )
                    ),
                    "row_order_exact": tuple(segmented.trace["positions"])
                    == tuple(cached_trace["positions"]),
                    "attention_mask_exact": bool(self.torch.equal(b_mask, c_mask)),
                    "monolithic_mask_subset_exact": bool(
                        self.torch.equal(a_mask, b_mask)
                    ),
                    "projection_shapes_exact": all(
                        tuple(b_capture[name].shape) == tuple(c_capture[name].shape)
                        for name in ("in_proj_qkvz_output", "in_proj_ba_output")
                    ),
                    "segmented_region_cache_entries_before_suffix": (
                        segmented.cache_entries_before_suffix
                    ),
                    "segmented_restore_calls": layer_zero.restore_calls["B_segmented"],
                    "cached_restore_calls": layer_zero.restore_calls["C_cached"],
                }
                detailed_bc = _capture_comparison(b_capture, c_capture)
                projection_inputs_match = all(
                    detailed_bc[name]["exact_equal"]
                    for name in (
                        "decoder_layer_input_hidden_states",
                        "input_layernorm_output",
                    )
                )
                projections_match = all(
                    detailed_bc[name]["exact_equal"]
                    for name in ("in_proj_qkvz_output", "in_proj_ba_output")
                )
                states_match = all(
                    detailed_bc[name]["exact_equal"]
                    for name in (
                        "pre_suffix_convolution_state",
                        "pre_suffix_recurrent_state",
                    )
                )
                gdn_output_matches = detailed_bc["gdn_core_output"]["exact_equal"]
                input_identity_pass = (
                    input_contract["suffix_input_ids_exact"]
                    and input_contract["absolute_positions_exact"]
                    and input_contract["row_order_exact"]
                    and input_contract["attention_mask_exact"]
                    and input_contract["monolithic_mask_subset_exact"]
                    and input_contract["projection_shapes_exact"]
                )
                input_contract_pass = (
                    input_identity_pass
                    and input_contract["segmented_region_cache_entries_before_suffix"]
                    == 0
                    and input_contract["segmented_restore_calls"] == 0
                    and input_contract["cached_restore_calls"] > 0
                )
                strict_bc = (
                    input_contract_pass
                    and projection_inputs_match
                    and max(b_c_trace[:3]) < TENSOR_TOLERANCE
                    and bool(b_c_trace[3])
                    and stable_before == stable_after
                )
                a_b_detailed = _capture_comparison(a_capture, b_capture)
                monolithic_projection_drift = any(
                    not a_b_detailed[name]["exact_equal"]
                    for name in ("in_proj_qkvz_output", "in_proj_ba_output")
                )
                if not input_identity_pass or not projection_inputs_match:
                    decision = "case_4_suffix_gather_or_row_order"
                elif not projections_match:
                    decision = "unresolved_projection_difference_between_B_and_C"
                elif not states_match:
                    decision = "case_2_frontier_snapshot_restore"
                elif not gdn_output_matches:
                    decision = "case_3_gdn_kernel_restart_equivalence"
                elif monolithic_projection_drift and strict_bc:
                    decision = "case_1_shape_dependent_bf16_projection"
                else:
                    decision = "unresolved"

                steps.append(
                    {
                        "step": step + 1,
                        "boundary": boundary,
                        "input_contract_B_vs_C": input_contract,
                        "layer0_A_vs_B": a_b_detailed,
                        "layer0_B_vs_C": detailed_bc,
                        "trace_A_vs_B": {
                            "max_logits_error": a_b_trace[0],
                            "max_hidden_error": a_b_trace[1],
                            "max_gdn_error": a_b_trace[2],
                            "top1": _top1_evidence(
                                a_suffix["top1"], segmented.trace["top1"], positions
                            ),
                        },
                        "trace_B_vs_C": {
                            "max_logits_error": b_c_trace[0],
                            "max_hidden_error": b_c_trace[1],
                            "max_gdn_error": b_c_trace[2],
                            "top1": _top1_evidence(
                                segmented.trace["top1"], cached_trace["top1"], positions
                            ),
                        },
                        "timings_ms": {
                            "monolithic_numerical_audit_ms": (
                                reference_prepare["gather_scatter"]
                                + reference_prepare["mask_build"]
                                + reference_forward_ms
                            ),
                            "reference_full_ms": sum(segmented.timings.values()),
                            "canonical_frontier_establishment_ms": (
                                sum(frontier_prepare.values()) + frontier_forward_ms
                            ),
                            "cached_suffix_ms": (
                                cached_prepare["gather_scatter"]
                                + cached_prepare["mask_build"]
                                + cached_forward_ms
                            ),
                        },
                        "stable_pre_frontier_unchanged": stable_before == stable_after,
                        "frontier_state_provenance": provenance_result,
                        "strict_B_vs_C_pass": strict_bc,
                        "decision": decision,
                    }
                )
                reference_top1 = reference_trace["top1"]
                for position in diffusion_positions:
                    edited_tokens[position] = int(reference_top1[position])

        if (
            not row_hooks.released
            or not layer_zero.released
            or not frontier_provenance.released
        ):
            raise RuntimeError("diagnostic hooks survived three-way execution")
        strict_steps = all(step["strict_B_vs_C_pass"] for step in steps)
        return {
            "schema_version": 1,
            "artifact_type": "cluster3_gdn_equivalence_three_way_diagnostic",
            "revision": self.runtime.revision,
            "model_scale": self.runtime.model_scale,
            "dtype": self.args.dtype,
            "tp_size": self.args.tp_size,
            "case_id": case.case_id,
            "paths": {
                "A": "monolithic full reference",
                "B": "fresh segmented full recomputation with live prefix state",
                "C": "cached exact-frontier restoration",
            },
            "steps": steps,
            "all_strict_B_vs_C_pass": (
                strict_steps and logs.fallback_count == 0 and logs.recovery_replays == 0
            ),
            "decisions_consistent": bool(steps)
            and all(step["decision"] == steps[0]["decision"] for step in steps),
            "fallback_count": logs.fallback_count,
            "recovery_replays": logs.recovery_replays,
            "a30_acceptance_claimed": False,
        }

    def close(self) -> None:
        self.runtime.close()


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / OUTPUT_NAME
    runner = ThreeWayDiagnostic(args)
    primary_error: Optional[BaseException] = None
    try:
        artifact = runner.run()
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            runner.close()
        except BaseException:
            if primary_error is None:
                raise
    destination.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return artifact


def main(argv: Optional[Sequence[str]] = None) -> None:
    artifact = run_diagnostic(parse_args(argv))
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
