#!/usr/bin/env python3
"""Export paired traces for the Cluster-1 exact-prefix handoff.

The reference path causally replays the stable prefix before every diffusion
step.  The cached path restores the committed Cluster-1 prefix state and sends
only the active suffix through the model.  This script never treats decoded
text as a correctness signal; all comparisons are made directly on tensors.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence


SCHEMA_VERSION = 1
ATTENTION_CONTRACT = "causal_prefix_diffusion_suffix_v1"
NEGATIVE_CHECK_NAMES = (
    "wrong_model_miss",
    "wrong_model_revision_miss",
    "wrong_token_hash_miss",
    "wrong_position_hash_miss",
    "wrong_attention_contract_miss",
    "wrong_region_version_miss",
    "wrong_parent_version_miss",
    "recycled_request_slot_miss",
)


@dataclass(frozen=True)
class ManifestCase:
    schema_version: int
    case_id: str
    token_seed: int
    prefix_length: int
    active_length: int
    diffusion_steps: int
    attention_contract_id: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ManifestCase":
        required = {field.name for field in dataclasses.fields(cls)}
        missing = sorted(required - value.keys())
        extra = sorted(value.keys() - required)
        if missing or extra:
            raise ValueError(
                f"invalid manifest fields: missing={missing}, extra={extra}"
            )
        case = cls(**{name: value[name] for name in required})
        if case.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema: {case.schema_version}")
        if not case.case_id:
            raise ValueError("case_id must be nonempty")
        if case.prefix_length <= 0 or case.active_length <= 0:
            raise ValueError("prefix_length and active_length must be positive")
        if case.diffusion_steps <= 0:
            raise ValueError("diffusion_steps must be positive")
        if case.attention_contract_id != ATTENTION_CONTRACT:
            raise ValueError(
                f"unsupported attention contract: {case.attention_contract_id}"
            )
        return case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export real paired Cluster-1 exact-prefix traces."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", required=True, choices=("float16", "bfloat16"))
    parser.add_argument("--tp-size", required=True, type=int)
    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help="Required: trace hooks are intentionally incompatible with CUDA graphs.",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.tp_size <= 0:
        raise ValueError("--tp-size must be positive")
    if not args.disable_cuda_graph:
        raise ValueError("--disable-cuda-graph is required for exact trace capture")
    return args


def read_manifest(path: str | Path) -> list[ManifestCase]:
    cases: list[ManifestCase] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                case = ManifestCase.from_mapping(value)
            except Exception as exc:
                raise ValueError(f"invalid manifest line {line_number}: {exc}") from exc
            if case.case_id in seen:
                raise ValueError(f"duplicate case_id: {case.case_id}")
            seen.add(case.case_id)
            cases.append(case)
    if not cases:
        raise ValueError("manifest contains no cases")
    return cases


def _import_torch():
    import torch

    return torch


def _require_finite(tensor: Any, label: str) -> None:
    torch = _import_torch()
    if not torch.is_tensor(tensor):
        raise TypeError(f"{label} is not a tensor")
    if not bool(torch.isfinite(tensor.float()).all().item()):
        raise ValueError(f"{label} contains NaN or Inf")


def tensor_max_abs(reference: Any, cached: Any, label: str) -> float:
    if tuple(reference.shape) != tuple(cached.shape):
        raise ValueError(
            f"{label} shape mismatch: {tuple(reference.shape)} != {tuple(cached.shape)}"
        )
    _require_finite(reference, f"reference {label}")
    _require_finite(cached, f"cached {label}")
    if reference.numel() == 0:
        raise ValueError(f"{label} is empty")
    return float((reference.float() - cached.float()).abs().max().item())


def hash_tensors(named_tensors: Iterable[tuple[str, Any]]) -> str:
    """Hash tensor metadata and exact device values in deterministic name order."""
    torch = _import_torch()
    digest = hashlib.sha256()
    count = 0
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        if not torch.is_tensor(tensor):
            raise TypeError(f"stable value {name} is not a tensor")
        _require_finite(tensor, f"stable tensor {name}")
        value = tensor.detach().contiguous()
        raw = value.view(torch.uint8).cpu().numpy().tobytes()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(raw)
        count += 1
    if count == 0:
        raise ValueError("no committed stable tensors were available to hash")
    return digest.hexdigest()


def hash_token_ids(token_ids: Any) -> str:
    torch = _import_torch()
    if not torch.is_tensor(token_ids):
        token_ids = torch.tensor(token_ids, dtype=torch.int64)
    return hash_tensors((("top1", token_ids.to(dtype=torch.int64)),))


@dataclass
class CapturedLayer:
    layer_id: int
    has_gdn: bool
    active_hidden: Any
    gdn_conv: Optional[Any]
    gdn_recurrent: Optional[Any]

    def release(self) -> None:
        self.active_hidden = None
        self.gdn_conv = None
        self.gdn_recurrent = None


@dataclass
class CapturedStep:
    step: int
    active_logits: Any
    top1_tokens: Any
    layers: list[CapturedLayer]
    execution_signature: str = ""

    def release(self) -> None:
        self.active_logits = None
        self.top1_tokens = None
        for layer in self.layers:
            layer.release()
        self.layers.clear()


class ModelTraceHooks:
    """Temporary opt-in hooks; normal serving never constructs this object."""

    def __init__(self, model_runner: Any):
        self.model_runner = model_runner
        self._handles: list[Any] = []
        self._capturing = False
        self._active_length = 0
        self._mamba_cache_idx: Optional[int] = None
        self._layers: dict[int, CapturedLayer] = {}
        model = self._language_model(model_runner.model)
        self.num_layers = len(model.layers)
        self._mamba_map = getattr(model_runner.req_to_token_pool, "mamba_map", {})
        for layer_id, layer in enumerate(model.layers):
            handle = layer.register_forward_hook(self._make_hook(layer_id))
            self._handles.append(handle)

    @staticmethod
    def _language_model(model: Any) -> Any:
        candidates = (model, getattr(model, "model", None))
        for candidate in candidates:
            if candidate is None:
                continue
            if hasattr(candidate, "layers"):
                return candidate
            language_model = getattr(candidate, "language_model", None)
            if language_model is not None and hasattr(language_model, "layers"):
                return language_model
            nested = getattr(candidate, "model", None)
            if nested is not None and hasattr(nested, "layers"):
                return nested
        raise RuntimeError("cannot locate Qwen3.5 language-model layers")

    def _make_hook(self, layer_id: int) -> Callable[..., None]:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if not self._capturing:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            residual = (
                output[1] if isinstance(output, tuple) and len(output) > 1 else None
            )
            if hidden.shape[0] < self._active_length:
                raise RuntimeError(
                    f"layer {layer_id} returned fewer rows than the active suffix"
                )
            active_hidden = hidden[-self._active_length :]
            if residual is not None and tuple(residual.shape) == tuple(hidden.shape):
                active_hidden = active_hidden + residual[-self._active_length :]
            active_hidden = active_hidden.detach().clone()
            gdn_index = self._mamba_map.get(layer_id)
            conv = recurrent = None
            if gdn_index is not None:
                if self._mamba_cache_idx is None:
                    raise RuntimeError("GDN trace is missing the Mamba cache slot")
                cache = self.model_runner.req_to_token_pool.mamba_pool.mamba_cache
                conv = tuple(
                    value[int(gdn_index), int(self._mamba_cache_idx)].detach().clone()
                    for value in cache.conv
                )
                recurrent = (
                    cache.temporal[int(gdn_index), int(self._mamba_cache_idx)]
                    .detach()
                    .clone()
                )
            self._layers[layer_id] = CapturedLayer(
                layer_id=layer_id,
                has_gdn=gdn_index is not None,
                active_hidden=active_hidden,
                gdn_conv=conv,
                gdn_recurrent=recurrent,
            )

        return hook

    @contextlib.contextmanager
    def capture(self, *, active_length: int, mamba_cache_idx: int) -> Iterator[None]:
        if self._capturing:
            raise RuntimeError("nested trace capture is not allowed")
        self._layers.clear()
        self._active_length = int(active_length)
        self._mamba_cache_idx = int(mamba_cache_idx)
        self._capturing = True
        try:
            yield
        finally:
            self._capturing = False
            self._mamba_cache_idx = None

    def finish_step(self, step: int, logits: Any) -> CapturedStep:
        if self._capturing:
            raise RuntimeError("finish_step called while hooks are active")
        if len(self._layers) != self.num_layers:
            raise RuntimeError(
                f"missing layer traces: captured {len(self._layers)} of {self.num_layers}"
            )
        if logits is None or logits.shape[0] < self._active_length:
            raise RuntimeError("active logits trace is missing")
        active_logits = logits[-self._active_length :].detach().clone()
        _require_finite(active_logits, "active logits")
        top1 = active_logits.argmax(dim=-1).detach().clone()
        layers = [self._layers[index] for index in range(self.num_layers)]
        self._layers = {}
        return CapturedStep(step, active_logits, top1, layers)

    def close(self) -> None:
        self._capturing = False
        for layer in self._layers.values():
            layer.release()
        self._layers.clear()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @property
    def released(self) -> bool:
        return not self._handles and not self._layers and not self._capturing


def compare_steps(reference: CapturedStep, cached: CapturedStep) -> dict[str, Any]:
    if reference.step != cached.step:
        raise ValueError("diffusion step numbers differ")
    if reference.execution_signature != cached.execution_signature:
        raise ValueError("paired executions used different tokens, positions, or masks")
    if len(reference.layers) != len(cached.layers):
        raise ValueError("layer counts differ")
    if not reference.layers:
        raise ValueError("step trace has no layers")
    top1_equal = bool((reference.top1_tokens == cached.top1_tokens).all().item())
    layer_records: list[dict[str, Any]] = []
    for ref_layer, cached_layer in zip(reference.layers, cached.layers):
        if ref_layer.layer_id != cached_layer.layer_id:
            raise ValueError("layer IDs differ")
        if ref_layer.has_gdn != cached_layer.has_gdn:
            raise ValueError(f"layer {ref_layer.layer_id} GDN classification differs")
        record = {
            "layer_id": ref_layer.layer_id,
            "has_gdn": ref_layer.has_gdn,
            "active_hidden_max_abs": tensor_max_abs(
                ref_layer.active_hidden,
                cached_layer.active_hidden,
                f"layer {ref_layer.layer_id} active hidden",
            ),
            "gdn_conv_max_abs": None,
            "gdn_recurrent_max_abs": None,
        }
        if ref_layer.has_gdn:
            if ref_layer.gdn_conv is None or cached_layer.gdn_conv is None:
                raise ValueError(
                    f"layer {ref_layer.layer_id} convolution trace is missing"
                )
            if len(ref_layer.gdn_conv) != len(cached_layer.gdn_conv):
                raise ValueError(
                    f"layer {ref_layer.layer_id} convolution counts differ"
                )
            if not ref_layer.gdn_conv:
                raise ValueError(
                    f"layer {ref_layer.layer_id} convolution trace is empty"
                )
            record["gdn_conv_max_abs"] = max(
                tensor_max_abs(left, right, f"layer {ref_layer.layer_id} GDN conv")
                for left, right in zip(ref_layer.gdn_conv, cached_layer.gdn_conv)
            )
            if ref_layer.gdn_recurrent is None or cached_layer.gdn_recurrent is None:
                raise ValueError(
                    f"layer {ref_layer.layer_id} recurrent trace is missing"
                )
            record["gdn_recurrent_max_abs"] = tensor_max_abs(
                ref_layer.gdn_recurrent,
                cached_layer.gdn_recurrent,
                f"layer {ref_layer.layer_id} GDN recurrent",
            )
        layer_records.append(record)
    return {
        "step": reference.step,
        "active_logits_max_abs": tensor_max_abs(
            reference.active_logits, cached.active_logits, "active logits"
        ),
        "top1_tokens_identical": top1_equal,
        "reference_top1_hash": hash_token_ids(reference.top1_tokens),
        "cached_top1_hash": hash_token_ids(cached.top1_tokens),
        "layers": layer_records,
    }


def validate_result(result: Mapping[str, Any], case: ManifestCase) -> None:
    if result.get("case_id") != case.case_id:
        raise ValueError("result case_id mismatch")
    if result.get("cache_hit") is not True:
        raise ValueError("cached path did not report a genuine exact cache hit")
    steps = result.get("steps")
    if not isinstance(steps, list) or len(steps) != case.diffusion_steps:
        raise ValueError("missing diffusion-step traces")
    expected_layers = int(result.get("num_layers", 0))
    if expected_layers <= 0:
        raise ValueError("num_layers must be positive")
    for expected_step, step in enumerate(steps, 1):
        if step.get("step") != expected_step:
            raise ValueError("step trace numbering is incomplete")
        if len(step.get("layers", [])) != expected_layers:
            raise ValueError(f"step {expected_step} is missing layer traces")
    if result.get("stable_hash_before") != result.get("stable_hash_after"):
        raise ValueError("committed stable state mutated during cached execution")
    if result.get("gdn_snapshot_restore_deterministic") is not True:
        raise ValueError("GDN snapshot restore was not deterministic")
    if result.get("stale_state_reuse_count") != 0:
        raise ValueError("stale GDN state was reused")
    checks = result.get("negative_cache_checks")
    if not isinstance(checks, Mapping):
        raise ValueError("negative cache checks are missing")
    failed = [name for name in NEGATIVE_CHECK_NAMES if checks.get(name) is not True]
    if failed:
        raise ValueError(f"negative cache checks failed: {failed}")


class Cluster1ModelRuntime:
    """One loaded HybridDiffusion model and its real SGLang memory pools."""

    def __init__(self, args: argparse.Namespace):
        if args.tp_size != 1:
            raise RuntimeError(
                "model-side trace collection currently requires --tp-size 1; "
                "sharded traces must not be silently compared as full tensors"
            )
        if not _import_torch().cuda.is_available():
            raise RuntimeError("the real paired exporter requires CUDA")
        self.args = args
        self._load_model()

    def _load_model(self) -> None:
        eval_root = Path(__file__).resolve().parents[1]
        if str(eval_root) not in sys.path:
            sys.path.insert(0, str(eval_root))
        from sglang.bench_one_batch import _set_envs_and_config, load_model
        from sglang.srt.server_args import PortArgs, ServerArgs

        config_path = (
            eval_root / "configs/hybrid_diffusion_self_spec_b7_g4_exact_handoff.yaml"
        )
        self.server_args = ServerArgs(
            model_path=self.args.model_dir,
            dtype=self.args.dtype,
            tp_size=self.args.tp_size,
            disable_cuda_graph=True,
            disable_overlap_schedule=True,
            max_running_requests=8,
            dllm_algorithm="HybridDiffusionSelfSpec",
            dllm_algorithm_config=str(config_path),
        )
        _set_envs_and_config(self.server_args)
        port_args = PortArgs.init_new(self.server_args)
        wrapper, _tokenizer = load_model(self.server_args, port_args, 0, 0)
        self.model_runner = wrapper.torch_runner
        self.device = self.model_runner.device
        from sglang.srt.dllm.config import DllmConfig

        self.dllm_config = DllmConfig.from_server_args(self.server_args)
        if self.dllm_config is None:
            raise RuntimeError("failed to initialize the dLLM execution contract")
        self.backend = self._region_backend()
        self.backend.configure_region_state_cache(
            max_entries=128, strict_validation=True
        )

    def _region_backend(self) -> Any:
        backend = getattr(self.model_runner.attn_backend, "linear_attn_backend", None)
        if backend is None or not all(
            hasattr(backend, name)
            for name in (
                "commit_region_state",
                "restore_region_state",
                "region_state_cache",
            )
        ):
            raise RuntimeError("HybridDiffusion-2B did not initialize GDNDllmBackend")
        return backend

    def _synchronize(self) -> None:
        _import_torch().cuda.synchronize(self.device)

    def _clear_pools(self) -> None:
        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()

    def _make_req(self, rid: str, tokens: list[int]) -> Any:
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=list(tokens),
            sampling_params=SamplingParams(
                temperature=0, max_new_tokens=1, ignore_eos=True
            ),
        )
        req.init_diffusion_llm(self.dllm_config)
        req.fill_ids = list(tokens)
        req.logprob_start_len = -1
        req.set_extend_input_len(len(tokens))
        return req

    def _prepare_extend(self, req: Any, *, bidir: bool) -> tuple[Any, Any]:
        from types import SimpleNamespace
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
        batch = ScheduleBatch.init_new(
            reqs=[req],
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            dllm_config=self.dllm_config,
        )
        batch.prepare_for_extend()
        batch._dllm_attn_mask_types_cpu = [
            DLLM_ATTN_MASK_BIDIR_BLOCK if bidir else DLLM_ATTN_MASK_CAUSAL_PREFILL
        ]
        worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(worker_batch, self.model_runner)
        if bidir:
            active_length = int(req.extend_input_len)
            torch = _import_torch()
            forward_batch.dllm_force_bidir_mask = True
            forward_batch.dllm_bidir_custom_mask = torch.ones(
                (active_length, active_length), dtype=torch.bool, device=self.device
            )
            forward_batch.dllm_gdn_causal_mode = 0
            forward_batch.dllm_gdn_block_size = active_length
        else:
            forward_batch.dllm_force_causal = True
            forward_batch.dllm_gdn_causal_mode = 1
        forward_batch.dllm_gdn_persist_state = True
        return batch, forward_batch

    def _forward(self, forward_batch: Any) -> Any:
        output = self.model_runner.forward(forward_batch).logits_output
        logits = output.full_logits
        if logits is None:
            logits = output.next_token_logits
        if logits is None:
            raise RuntimeError("model did not return logits required by the exporter")
        return logits

    def _run_prefix(self, req: Any) -> None:
        _batch, forward_batch = self._prepare_extend(req, bidir=False)
        self._forward(forward_batch)
        self._synchronize()

    def _run_active(
        self, req: Any, active_tokens: list[int], hooks: ModelTraceHooks, step: int
    ) -> CapturedStep:
        prefix_length = len(req.prefix_indices)

        # Bidirectional dLLM positions are constructed from dllm_block_offset.
        # Every denoising step operates at the fixed sealed-prefix boundary.
        req.dllm_block_offset = prefix_length

        req.fill_ids = list(req.origin_input_ids[:prefix_length]) + list(active_tokens)
        req.set_extend_input_len(len(active_tokens))
        _batch, forward_batch = self._prepare_extend(req, bidir=True)
        if int(forward_batch.input_ids.numel()) != len(active_tokens):
            raise RuntimeError(
                "active forward unexpectedly contains replayed prefix tokens"
            )
        if int(forward_batch.extend_prefix_lens[0].item()) != prefix_length:
            raise RuntimeError("active forward lost the exact sealed prefix boundary")
        self._validate_active_positions(
            forward_batch,
            prefix_length=prefix_length,
            active_length=len(active_tokens),
        )
        mamba_idx = self.backend._current_mamba_slot(req.req_pool_idx)
        self._synchronize()
        with hooks.capture(active_length=len(active_tokens), mamba_cache_idx=mamba_idx):
            logits = self._forward(forward_batch)
        self._synchronize()
        trace = hooks.finish_step(step, logits)
        trace.execution_signature = hash_tensors(
            (
                ("input_ids", forward_batch.input_ids),
                ("positions", forward_batch.positions),
                ("attention_mask", forward_batch.dllm_bidir_custom_mask),
            )
        )
        return trace

    @staticmethod
    def _validate_active_positions(
        forward_batch: Any, *, prefix_length: int, active_length: int
    ) -> None:
        torch = _import_torch()
        expected_positions = torch.arange(
            prefix_length,
            prefix_length + active_length,
            dtype=forward_batch.positions.dtype,
            device=forward_batch.positions.device,
        )
        if not torch.equal(forward_batch.positions, expected_positions):
            raise RuntimeError(
                "active positions mismatch: "
                f"boundary={prefix_length}, "
                f"active_length={active_length}, "
                f"expected_head={expected_positions[:8].tolist()}, "
                f"observed_head={forward_batch.positions[:8].tolist()}"
            )

    def _tokens(self, case: ManifestCase) -> tuple[list[int], list[int]]:
        torch = _import_torch()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(case.token_seed)
        vocab_size = int(self.model_runner.model_config.vocab_size)
        values = torch.randint(
            0,
            vocab_size,
            (case.prefix_length + case.active_length,),
            generator=generator,
            dtype=torch.int64,
        ).tolist()
        mask_id = int(self.dllm_config.mask_id)
        values = [0 if value == mask_id else int(value) for value in values]
        return values[: case.prefix_length], values[case.prefix_length :]

    def _make_key(self, case: ManifestCase, req: Any, prefix: list[int]) -> Any:
        from sglang.srt.dllm.region.execution_spec import hash_positions, hash_token_ids
        from sglang.srt.mem_cache.region_state_cache import RegionStateKey

        revision = str(
            getattr(self.model_runner.model_config.hf_config, "_commit_hash", None)
            or getattr(self.server_args, "revision", None)
            or "local-checkpoint"
        )
        return RegionStateKey(
            request_id=str(req.rid),
            request_pool_idx=int(req.req_pool_idx),
            request_slot_generation=int(req.hybrid_request_slot_generation),
            region_id="causal_prefix",
            region_version=0,
            boundary=len(prefix),
            token_hash=hash_token_ids(prefix),
            position_hash=hash_positions(0, len(prefix)),
            model_identity=str(Path(self.args.model_dir).resolve()),
            model_revision=revision,
            adapter_identity="",
            adapter_revision="",
            attention_contract_id=case.attention_contract_id,
            parent_region_versions=(),
        )

    def _kv_reference(self, key: Any) -> Any:
        from sglang.srt.mem_cache.region_state_cache import KVPrefixReference

        pool = self.model_runner.req_to_token_pool
        return KVPrefixReference(
            request_id=key.request_id,
            request_pool_idx=key.request_pool_idx,
            request_slot_generation=key.request_slot_generation,
            pool_identity=id(pool),
            locations=pool.req_to_token[key.request_pool_idx, : key.boundary]
            .detach()
            .clone(),
            valid_length=key.boundary,
        )

    def _stable_tensors(self, state: Any) -> list[tuple[str, Any]]:
        tensors: list[tuple[str, Any]] = []
        for index, value in enumerate(state.gdn_conv_states):
            tensors.append((f"gdn.conv.{index}", value))
        tensors.append(("gdn.recurrent", state.gdn_recurrent_states))
        locations = state.kv_prefix.locations.to(dtype=_import_torch().long)
        text_config = self.model_runner.model_config.hf_text_config
        full_layers = set(getattr(text_config, "full_attention_layer_ids", []) or [])
        if not full_layers:
            full_layers = {
                layer_id
                for layer_id, layer_type in enumerate(
                    getattr(text_config, "layers_block_type", []) or []
                )
                if layer_type == "attention"
            }
        if not full_layers:
            raise RuntimeError("model exposes no full-attention layers to hash")
        for layer_id in sorted(full_layers):
            key, value = self.model_runner.token_to_kv_pool.get_kv_buffer(layer_id)
            tensors.append((f"kv.{layer_id}.key", key.index_select(0, locations)))
            tensors.append((f"kv.{layer_id}.value", value.index_select(0, locations)))
        return tensors

    def _negative_checks(self, key: Any) -> dict[str, bool]:
        cache = self.backend.region_state_cache
        mutations = {
            "wrong_model_miss": replace(
                key, model_identity=key.model_identity + ":wrong"
            ),
            "wrong_model_revision_miss": replace(
                key, model_revision=key.model_revision + ":wrong"
            ),
            "wrong_token_hash_miss": replace(key, token_hash=key.token_hash + ":wrong"),
            "wrong_position_hash_miss": replace(
                key, position_hash=key.position_hash + ":wrong"
            ),
            "wrong_attention_contract_miss": replace(
                key, attention_contract_id=key.attention_contract_id + ":wrong"
            ),
            "wrong_region_version_miss": replace(
                key, region_version=key.region_version + 1
            ),
            "wrong_parent_version_miss": replace(
                key, parent_region_versions=(("parent", 1),)
            ),
        }
        checks = {
            name: not cache.get(mutated).hit for name, mutated in mutations.items()
        }
        checks["recycled_request_slot_miss"] = not cache.get(
            key, current_slot_generation=key.request_slot_generation + 1
        ).hit
        return checks

    def _gdn_restore_tests(self, key: Any, mamba_idx: int) -> tuple[bool, int]:
        torch = _import_torch()
        pool_cache = self.model_runner.req_to_token_pool.mamba_pool.mamba_cache

        def destination() -> list[Any]:
            return [
                *(value[:, mamba_idx].detach().clone() for value in pool_cache.conv),
                pool_cache.temporal[:, mamba_idx].detach().clone(),
            ]

        for value in pool_cache.conv:
            value[:, mamba_idx].zero_()
        pool_cache.temporal[:, mamba_idx].zero_()
        first = self.backend.restore_region_state(
            state_key=key,
            mamba_cache_idx=mamba_idx,
            current_slot_generation=key.request_slot_generation,
        )
        self._synchronize()
        first_values = destination()
        for value in pool_cache.conv:
            value[:, mamba_idx].normal_()
        pool_cache.temporal[:, mamba_idx].normal_()
        second = self.backend.restore_region_state(
            state_key=key,
            mamba_cache_idx=mamba_idx,
            current_slot_generation=key.request_slot_generation,
        )
        self._synchronize()
        second_values = destination()
        deterministic = (
            first.hit
            and second.hit
            and all(
                torch.equal(left, right)
                for left, right in zip(first_values, second_values)
            )
        )

        stale_count = 0
        attempts = [key.request_slot_generation, key.request_slot_generation + 1] * 4
        random.Random(key.request_slot_generation).shuffle(attempts)
        for generation in attempts:
            before = destination()
            lookup = self.backend.restore_region_state(
                state_key=key,
                mamba_cache_idx=mamba_idx,
                current_slot_generation=generation,
            )
            self._synchronize()
            after = destination()
            if generation != key.request_slot_generation:
                if lookup.hit or any(
                    not torch.equal(left, right) for left, right in zip(before, after)
                ):
                    stale_count += 1
        return deterministic, stale_count

    def run_case(self, case: ManifestCase) -> dict[str, Any]:
        torch = _import_torch()
        prefix, initial_active = self._tokens(case)
        reference_steps: list[CapturedStep] = []
        cached_steps: list[CapturedStep] = []
        active_inputs: list[list[int]] = []
        active = list(initial_active)

        hooks = ModelTraceHooks(self.model_runner)
        try:
            # Reference: every step starts clean, replays the causal prefix, and
            # then executes the active diffusion suffix.
            for step in range(1, case.diffusion_steps + 1):
                self._clear_pools()
                req = self._make_req(f"{case.case_id}:reference:{step}", prefix)
                self._run_prefix(req)
                prefix_locations = (
                    self.model_runner.req_to_token_pool.req_to_token[
                        req.req_pool_idx, : len(prefix)
                    ]
                    .detach()
                    .clone()
                )
                req.prefix_indices = prefix_locations
                active_inputs.append(list(active))
                trace = self._run_active(req, active, hooks, step)
                reference_steps.append(trace)
                active = trace.top1_tokens.detach().cpu().tolist()

            # Build and publish the stable state only after the reference path.
            self._clear_pools()
            cached_req = self._make_req(f"{case.case_id}:cached", prefix)
            self._run_prefix(cached_req)
            key = self._make_key(case, cached_req, prefix)
            kv_reference = self._kv_reference(key)
            mamba_idx = self.backend._current_mamba_slot(cached_req.req_pool_idx)
            self.backend.commit_region_state(
                state_key=key,
                mamba_cache_idx=mamba_idx,
                kv_prefix=kv_reference,
            )
            self._synchronize()
            committed_lookup = self.backend.region_state_cache.get(
                key, current_slot_generation=key.request_slot_generation
            )
            if not committed_lookup.hit:
                raise RuntimeError("committed prefix was not published to the cache")
            stable_hash_before = hash_tensors(
                self._stable_tensors(committed_lookup.state)
            )
            negative_checks = self._negative_checks(key)
            restore_deterministic, stale_count = self._gdn_restore_tests(key, mamba_idx)

            cached_req.prefix_indices = kv_reference.locations.detach().clone()
            cache_hit = True
            for step, step_active in enumerate(active_inputs, 1):
                lookup = self.backend.restore_region_state(
                    state_key=key,
                    mamba_cache_idx=mamba_idx,
                    current_slot_generation=key.request_slot_generation,
                )
                cache_hit = cache_hit and bool(lookup.hit)
                if not lookup.hit:
                    raise RuntimeError(
                        f"cached execution step {step} missed exact prefix state: "
                        f"{lookup.miss_reason}"
                    )
                # prefix_indices fixes extend_prefix_len at the sealed boundary;
                # only step_active is present in forward_batch.input_ids.
                cached_req.prefix_indices = kv_reference.locations.detach().clone()
                cached_steps.append(
                    self._run_active(cached_req, step_active, hooks, step)
                )

            self._synchronize()
            committed_after = self.backend.region_state_cache.get(
                key, current_slot_generation=key.request_slot_generation
            )
            if not committed_after.hit:
                raise RuntimeError(
                    "committed prefix disappeared during cached execution"
                )
            stable_hash_after = hash_tensors(
                self._stable_tensors(committed_after.state)
            )
            compared = [
                compare_steps(reference, cached)
                for reference, cached in zip(reference_steps, cached_steps)
            ]
            result = {
                "schema_version": SCHEMA_VERSION,
                "case_id": case.case_id,
                "prefix_length": case.prefix_length,
                "active_length": case.active_length,
                "diffusion_steps": case.diffusion_steps,
                "attention_contract_id": case.attention_contract_id,
                "num_layers": hooks.num_layers,
                "cache_hit": cache_hit,
                "steps": compared,
                "stable_hash_before": stable_hash_before,
                "stable_hash_after": stable_hash_after,
                "gdn_snapshot_restore_deterministic": restore_deterministic,
                "stale_state_reuse_count": stale_count,
                "negative_cache_checks": negative_checks,
            }
            validate_result(result, case)
            return result
        finally:
            for trace in reference_steps + cached_steps:
                trace.release()
            hooks.close()
            if not hooks.released:
                raise RuntimeError(
                    "temporary hooks or retained tensors survived the case"
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def close(self) -> None:
        try:
            backend = getattr(self, "backend", None)
            cache = getattr(backend, "region_state_cache", None)
            if cache is not None:
                cache.clear()
        finally:
            torch = _import_torch()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()


def export_manifest(
    args: argparse.Namespace,
    *,
    runtime_factory: Callable[[argparse.Namespace], Any] = Cluster1ModelRuntime,
) -> None:
    cases = read_manifest(args.manifest)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runtime = runtime_factory(args)
    try:
        with output_path.open("w", encoding="utf-8") as destination:
            for case in cases:
                result = runtime.run_case(case)
                validate_result(result, case)
                destination.write(json.dumps(result, sort_keys=True) + "\n")
                destination.flush()
    finally:
        runtime.close()


def main(argv: Optional[Sequence[str]] = None) -> None:
    export_manifest(parse_args(argv))


if __name__ == "__main__":
    main()
