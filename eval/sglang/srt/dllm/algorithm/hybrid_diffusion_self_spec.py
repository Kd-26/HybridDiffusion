"""HybridDiffusion AR-Trust decoding: up to N tokens per forward with verification.

This is the serving implementation of the paper's AR-Trust path. Each forward
has two modes:
  Cold start:  [t0, M, M, ..., M]        (1 + 2(N-1) positions, padded)
  Verify:      [pending, spec0..specK, M, M, ..., M]  (1 + K + (N-1) positions)

The SGLang block_size must be set to 2*N - 1 (the max input size for verify rounds).
N is controlled by the `gen_block_size` config key.

Speculative tokens are verified left-to-right using the standard spec decoding criterion:
    r = p(x) / q(x)
    if r >= 1: always accept
    if r <  1: accept with probability r, else resample from max(0, p-q)

Config keys (passed via --dllm-algorithm-config YAML):
  block_size:             int,   MUST be 2*gen_block_size - 1
  gen_block_size:         int,   tokens per step (1 clean + N-1 spec). Default 3.
  confidence_threshold:   float, default 0.0 (disabled, rely on spec verify)
  temperature:            float, default 1.0
  top_k:                  int,   default 50
  top_p:                  float, default 0.95
  use_spec_verify:        bool,  default true
  draft_mode:             str,   default strict_truncated
    - strict_truncated: sample draft tokens from the same top-k/top-p proposal
      used by verify. This is the mathematically consistent temp>0 path.
    - argmax_softmax_verify: draft tokens are argmax and verification uses the
      full-softmax target. This is the paper's Softmax-Argmax policy.
    - argmax_truncated_verify: draft tokens are argmax, verify uses sparse
      top-k/top-p p/q. This is an experimental faster approximation.
"""

import hashlib
import importlib.util
import logging
import json
import os
import time
from dataclasses import replace
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm

try:
    from sglang.srt.dllm.algorithm.fused_verify_kernel import (
        fused_spec_verify,
        fused_spec_verify_from_logits,
        fused_sparse_spec_verify,
        sample_sparse_probs,
    )
    _HAS_FUSED_VERIFY = True
except ImportError:
    _HAS_FUSED_VERIFY = False
from sglang.srt.dllm.config import (
    DLLM_ATTN_MASK_CAUSAL_PREFILL,
    DllmConfig,
    SelfSpecVariant,
)
from sglang.srt.dllm.region.execution_spec import (
    HybridBoundaryCommit,
    hash_positions,
    hash_token_ids,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.mem_cache.region_state_cache import (
    KVPrefixReference,
    RegionStateKey,
    canonicalize_kv_prefix_locations,
)

logger = logging.getLogger(__name__)
_EXTRA_BUFFER_TRACE = (
    os.getenv("SGLANG_HYBRID_DIFFUSION_SELF_SPEC_EXTRA_BUFFER_TRACE", "0") == "1"
)


try:
    _HAS_FLASHINFER_SAMPLING = (
        importlib.util.find_spec("flashinfer.sampling") is not None
    )
    if _HAS_FLASHINFER_SAMPLING:
        from sglang.srt.layers.flashinfer_sampling_compat import (
            flashinfer_top_k_top_p_sampling_from_probs as _fi_sample,
        )
except (ImportError, ModuleNotFoundError):
    _HAS_FLASHINFER_SAMPLING = False


# Pre-allocated sampling parameter tensors (lazily initialized per device)
_SAMPLE_BUFS: Dict[torch.device, Dict[str, torch.Tensor]] = {}


def _get_sample_bufs(n: int, top_k: int, top_p: float, device: torch.device):
    """Get or create pre-allocated top_k/top_p tensors for flashinfer sampling."""
    bufs = _SAMPLE_BUFS.get(device)
    if bufs is None or bufs["size"] < n:
        new_size = max(n, 128)
        bufs = {
            "size": new_size,
            "top_ks": torch.full((new_size,), top_k, dtype=torch.int32, device=device),
            "top_ps": torch.full((new_size,), top_p, dtype=torch.float32, device=device),
        }
        _SAMPLE_BUFS[device] = bufs
    return bufs["top_ks"][:n], bufs["top_ps"][:n]


def _sampling_probs(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """Return the exact proposal distribution used for temp>0 sampling."""
    scaled = logits if temperature == 1.0 else logits / temperature
    if top_k > 0:
        k = min(top_k, scaled.shape[-1])
        topk_vals, _ = scaled.topk(k, dim=-1)
        scaled = scaled.masked_fill(scaled < topk_vals[:, -1:], float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = scaled.sort(dim=-1, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cum_probs = sorted_probs.cumsum(dim=-1)
        # Keep the token that crosses top_p, matching the existing sampler.
        mask = (cum_probs - sorted_probs) >= top_p
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        scaled = torch.full_like(scaled, float("-inf")).scatter(
            1, sorted_idx, sorted_logits
        )
    return F.softmax(scaled, dim=-1)


def _batched_sample(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    return_probs: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Batched sampling from [N, vocab_size] logits on GPU.

    Returns (token_ids [N], token_probs [N]) — both stay on GPU, no .item() sync.
    If return_probs=True, also returns the full probs [N, vocab_size] matrix.
    """
    if temperature <= 0:
        token_ids = logits.argmax(dim=-1)
        if return_probs:
            probs = F.softmax(logits, dim=-1)
            token_probs = probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
            return token_ids, token_probs, probs
        # Skip softmax when probs aren't needed (greedy path)
        token_probs = torch.ones(token_ids.shape[0], device=logits.device)
        return token_ids, token_probs

    if return_probs:
        probs = _sampling_probs(logits, temperature, top_k, top_p)
        token_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
        token_probs = probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
        return token_ids, token_probs, probs

    scaled = logits if temperature == 1.0 else logits / temperature
    probs = F.softmax(scaled, dim=-1)

    if _HAS_FLASHINFER_SAMPLING:
        n = probs.shape[0]
        top_ks, top_ps = _get_sample_bufs(n, top_k, top_p, probs.device)
        token_ids = _fi_sample(
            probs.contiguous(), top_ks, top_ps, filter_apply_order="joint"
        )
        token_probs = probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
        return token_ids, token_probs

    # Fallback: manual implementation
    if top_k > 0:
        topk_vals, _ = scaled.topk(top_k, dim=-1)
        scaled = scaled.masked_fill(scaled < topk_vals[:, -1:], float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = scaled.sort(dim=-1, descending=True)
        cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        mask = (cum_probs - sorted_logits.softmax(dim=-1)) >= top_p
        sorted_logits[mask] = float("-inf")
        scaled = sorted_logits.scatter(1, sorted_idx, sorted_logits)
    probs = F.softmax(scaled, dim=-1)
    token_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
    token_probs = probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
    if return_probs:
        return token_ids, token_probs, probs
    return token_ids, token_probs


class HybridDiffusionSelfSpec(DllmAlgorithm):
    """
    Block-N speculative generation: 1 forward → 1 to N output tokens.

    Two modes per round (blk = 2*N - 1 positions):
      Verify:     [pending, spec0, ..., spec(N-2), M, M, ..., M]
                  → verify specs left-to-right, then sample N new tokens
      Cold start: [t0, M, M, ..., M]
                  → sample 1 clean + (N-1) spec, hold specs for next verify

    MASK KV is always freed after the forward (never persists in cache).
    Rejected specs cause a fallback to cold start with the corrected token.

    Communicates with the output processor via instance dicts:
      _dllm_write_override: per-request tokens to write to dllm_ids
      _kv_trim_info: per-request KV pool indices to free (GPU tensor)
      _advance_override: per-request variable dllm_block_offset advance
    """

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.variant = config.variant
        self.gen_block_size: int = config.algorithm_config.get("gen_block_size", 3)
        self.num_masks: int = self.gen_block_size - 1  # N-1 draft tokens (all variants)
        N = self.gen_block_size
        if self.variant in (
            SelfSpecVariant.CAUSAL_SHIFT,
            SelfSpecVariant.BD_BIDIR_SHIFT,
        ):
            expected_blk = 2 * N - 1
        else:
            expected_blk = 2 * N
        assert self.block_size == expected_blk, (
            f"HybridDiffusionSelfSpec variant={self.variant.value} gen_block_size={N} requires "
            f"block_size={expected_blk}, got {self.block_size}"
        )
        self.temperature: float = config.algorithm_config.get("temperature", 1.0)
        self.top_k: int = config.algorithm_config.get("top_k", 50)
        self.top_p: float = config.algorithm_config.get("top_p", 0.95)
        self.confidence_threshold: float = config.algorithm_config.get(
            "confidence_threshold", 0.0
        )
        self.use_spec_verify: bool = config.algorithm_config.get(
            "use_spec_verify", True
        )
        # Relaxed verification: accept with prob min(1, p/(alpha*q))
        # alpha=1.0: standard verify, alpha<1: more lenient, alpha=0: always accept
        self.verify_alpha: float = config.algorithm_config.get(
            "verify_alpha", 1.0
        )
        # Number of speculative tokens to verify (default: all = num_masks)
        # Setting to 1 verifies only the first spec, auto-accepts the rest.
        self.verify_num_specs: int = min(
            config.algorithm_config.get("verify_num_specs", self.num_masks),
            self.num_masks,
        )
        # Fast verify mode: use logit-based threshold instead of full p/q ratio
        # Skips softmax computation for significant speedup
        self.fast_verify: bool = config.algorithm_config.get("fast_verify", False)
        # Logit threshold for fast verify: accept if logit(spec) is in top-K logits
        self.fast_verify_topk: int = config.algorithm_config.get("fast_verify_topk", 5)
        # Per-spec topk values for tiered verification (e.g., [7, 50] for strict spec0, lenient spec1)
        # If set, overrides fast_verify_topk for individual specs
        self.fast_verify_topk_per_spec: List[int] = config.algorithm_config.get(
            "fast_verify_topk_per_spec", []
        )
        # Output correction: when a spec token is accepted (in top-K) but isn't the
        # clean argmax, replace the OUTPUT token with the argmax for higher quality.
        # KV cache retains the original spec token (small mismatch, usually negligible).
        self.output_correction: bool = config.algorithm_config.get(
            "output_correction", False
        )
        self.draft_mode: str = config.algorithm_config.get(
            "draft_mode",
            config.algorithm_config.get("draft_model", "strict_truncated"),
        )
        valid_draft_modes = {
            "strict_truncated",
            "argmax_softmax_verify",
            "argmax_truncated_verify",
        }
        if self.draft_mode not in valid_draft_modes:
            raise ValueError(
                f"Unsupported HybridDiffusionSelfSpec draft_mode={self.draft_mode!r}; "
                f"expected one of {sorted(valid_draft_modes)}"
            )
        self._argmax_draft = (
            self.temperature > 0
            and self.draft_mode in (
                "argmax_softmax_verify",
                "argmax_truncated_verify",
            )
        )
        self._full_softmax_verify_from_logits = (
            self.temperature > 0
            and self.draft_mode == "argmax_softmax_verify"
        )
        self._sparse_truncated_verify = (
            self.temperature > 0
            and self.draft_mode in (
                "strict_truncated",
                "argmax_truncated_verify",
            )
        )
        self._uses_truncated_sampling = (
            self.temperature > 0
            and (self.top_k > 0 or self.top_p < 1.0)
        )
        self._use_sparse_topk_logits = (
            self.temperature > 0
            and self.top_k > 0
            and not self.fast_verify
            and self._sparse_truncated_verify
        )
        allow_approx = (
            os.getenv("SGLANG_HYBRID_DIFFUSION_SELF_SPEC_ALLOW_APPROX_VERIFY", "0") == "1"
        )
        if self.temperature > 0 and not allow_approx:
            if self.draft_mode == "strict_truncated":
                if not self.use_spec_verify:
                    raise RuntimeError(
                        "HybridDiffusionSelfSpec temp>0 production path requires exact speculative "
                        "verify. Set SGLANG_HYBRID_DIFFUSION_SELF_SPEC_ALLOW_APPROX_VERIFY=1 only "
                        "for debug."
                    )
                if self.fast_verify:
                    raise RuntimeError(
                        "HybridDiffusionSelfSpec fast_verify is an approximate temp>0 path. Set "
                        "SGLANG_HYBRID_DIFFUSION_SELF_SPEC_ALLOW_APPROX_VERIFY=1 only for debug."
                    )
                if self.output_correction:
                    raise RuntimeError(
                        "HybridDiffusionSelfSpec output_correction changes accepted temp>0 tokens. "
                        "Set SGLANG_HYBRID_DIFFUSION_SELF_SPEC_ALLOW_APPROX_VERIFY=1 only for debug."
                    )
                if self.verify_alpha != 1.0:
                    raise RuntimeError(
                        "HybridDiffusionSelfSpec temp>0 production path requires verify_alpha=1.0 "
                        "for exact p/q verification."
                    )
            elif not self.use_spec_verify:
                raise RuntimeError(
                    f"HybridDiffusionSelfSpec draft_mode={self.draft_mode} requires "
                    "use_spec_verify=true."
                )
        self._requires_fused_verify = (
            self.temperature > 0
            and self.use_spec_verify
            and self.verify_alpha > 0
            and self.verify_num_specs > 0
            and not self.fast_verify
            and (
                not self._uses_truncated_sampling
                or self._full_softmax_verify_from_logits
            )
        )
        if (
            self._requires_fused_verify
            and not _HAS_FUSED_VERIFY
            and os.getenv("SGLANG_HYBRID_DIFFUSION_SELF_SPEC_ALLOW_UNFUSED_VERIFY", "0")
            != "1"
        ):
            raise RuntimeError(
                "HybridDiffusionSelfSpec stochastic verify requires fused_verify_kernel on the "
                "production path. Set SGLANG_HYBRID_DIFFUSION_SELF_SPEC_ALLOW_UNFUSED_VERIFY=1 "
                "only for debug."
            )

        # V3/V4: precompute block-internal attention mask
        # Clean rows (0..N-1): causal; MASK rows (N..blk-1): attend to all
        if self.variant in (
            SelfSpecVariant.BD_BIDIR,
            SelfSpecVariant.BD_BIDIR_SHIFT,
        ):
            blk = self.block_size
            N = self.gen_block_size
            mask = torch.zeros(blk, blk, dtype=torch.bool)
            for i in range(N):
                mask[i, :i + 1] = True
            mask[N:, :] = True
            self._bidir_mask = mask
        else:
            self._bidir_mask = None

        # Per-request state (keyed by req_pool_idx)
        self._prev_last_logits: Dict[int, torch.Tensor] = {}
        # Greedy shortcut: store argmax token id directly instead of full logits
        self._prev_last_argmax: Dict[int, int] = {}
        # Truncated sampling shortcut: store compact top-k distribution for the
        # initial clean token after prefill instead of a full vocab row.
        self._prev_last_topk: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._pending: Dict[int, int] = {}
        self._spec_tokens: Dict[int, List[int]] = {}
        # Store full draft probability distributions for correct max(0, p-q) correction.
        # Each entry is a list of GPU tensors of shape [vocab_size], one per spec token.
        self._spec_draft_probs: Dict[int, List[torch.Tensor]] = {}
        self._force_next_token: Dict[int, int] = {}
        # Pre-allocated draft probs buffer: [max_slots, vocab_size] indexed by rpx
        # Lazily initialized on first use. Eliminates torch.stack overhead.
        self._draft_probs_buf: torch.Tensor = None  # [max_slots, vocab_size]
        self._draft_probs_buf_rpx: set = set()  # track which rpx slots are valid
        self._gumbel_seed_counter: int = 0  # incrementing seed for Gumbel RNG

        # Per-round signals to the output processor
        self._dllm_write_override: Dict[int, List[int]] = {}
        self._kv_trim_info: Dict[int, dict] = {}
        self._advance_override: Dict[int, int] = {}
        self._mamba_track_commit_info: Dict[int, int] = {}
        self.exact_prefix_handoff = bool(
            getattr(config, "exact_prefix_handoff", False)
        )
        self._region_state_cache_max_entries = int(
            getattr(config, "region_state_cache_max_entries", 128)
        )
        self._strict_region_state_validation = bool(
            getattr(config, "strict_region_state_validation", True)
        )
        self._hybrid_state_keys: Dict[int, RegionStateKey] = {}
        self._hybrid_boundary_commits: Dict[int, HybridBoundaryCommit] = {}
        self._exact_handoff_debug = (
            os.getenv("SGLANG_HYBRID_EXACT_HANDOFF_DEBUG", "0") == "1"
        )
        self._exact_handoff_debug_sync = (
            os.getenv("SGLANG_HYBRID_EXACT_HANDOFF_DEBUG_SYNC", "0") == "1"
        )

        self._stats = {
            "total_forwards": 0,
            "prefill_forwards": 0,
            "decode_forwards": 0,
            "total_tokens": 0,
            "accept_count": 0,
            "reject_count": 0,
            "fused_verify_calls": 0,
            "torch_verify_calls": 0,
            "draft_sample_batches": 0,
            "draft_argmax_batches": 0,
            "verify_decisions": 0,
            "accepted_spec_tokens": 0,
            "accept_token_hist": [0] * (self.num_masks + 1),
            "logits_mode_full": 0,
            "logits_mode_topk": 0,
            "logits_mode_topk_dense": 0,
            "logits_mode_topk_tiled": 0,
            "logits_mode_argmax": 0,
            "logits_mode_unknown": 0,
            "logits_rows_total": 0,
            "logits_rows_consumed": 0,
            "logits_rows_verify": 0,
            "logits_rows_sample": 0,
        }
        self._timing = {
            "phase1_classify": 0.0,
            "phase2_forward": 0.0,
            "phase3_verify_sample": 0.0,
            "phase4_trim_assemble": 0.0,
            "timing_count": 0,
        }
        # Conditional LoRA: base-only for verify/committed, base+LoRA for MASK positions
        self.conditional_lora: bool = config.algorithm_config.get(
            "conditional_lora", False
        )

        # Optional profiling switch. This adds GPU synchronizations, so keep it
        # disabled unless explicitly benchmarking self-spec phase costs.
        self._timing_enabled = (
            os.getenv("SGLANG_HYBRID_DIFFUSION_SELF_SPEC_TIMING", "0") == "1"
        )
        self._timing_log_interval = int(
            os.getenv("SGLANG_HYBRID_DIFFUSION_SELF_SPEC_TIMING_INTERVAL", "100")
        )
        self._debug_steps = (
            os.getenv("SGLANG_HYBRID_DIFFUSION_SELF_SPEC_DEBUG_STEPS", "0") == "1"
        )
        self._trace_path = os.getenv("SGLANG_HYBRID_DIFFUSION_SELF_SPEC_TRACE_PATH")
        self._trace_topk = int(
            os.getenv("SGLANG_HYBRID_DIFFUSION_SELF_SPEC_TRACE_TOPK", "5")
        )
        logger.info(
            f"[HybridDiffusionSelfSpec] gen_block_size={self.gen_block_size}, "
            f"block_size={self.block_size}, num_masks={self.num_masks}, "
            f"spec_verify={self.use_spec_verify}, verify_alpha={self.verify_alpha}, "
            f"draft_mode={self.draft_mode}, "
            f"truncated_sampling={self._uses_truncated_sampling}, "
            f"conditional_lora={self.conditional_lora}, "
            f"fused_verify_available={_HAS_FUSED_VERIFY}, "
            f"requires_fused_verify={self._requires_fused_verify}"
        )

    def cleanup_request(self, req_pool_idx: int):
        """Remove all per-request state for a finished request.

        Must be called when a request finishes to prevent stale state from
        being picked up by a new request that reuses the same req_pool_idx.
        """
        if self._debug_steps:
            had_state = {
                "prev_logits": req_pool_idx in self._prev_last_logits,
                "prev_argmax": req_pool_idx in self._prev_last_argmax,
                "prev_topk": req_pool_idx in self._prev_last_topk,
                "pending": req_pool_idx in self._pending,
                "specs": req_pool_idx in self._spec_tokens,
                "draft_probs": req_pool_idx in self._spec_draft_probs,
                "force": req_pool_idx in self._force_next_token,
            }
            logger.info(
                "[HybridDiffusionSelfSpec] cleanup rpx=%s state=%s", req_pool_idx, had_state
            )
        self._prev_last_logits.pop(req_pool_idx, None)
        self._prev_last_argmax.pop(req_pool_idx, None)
        self._prev_last_topk.pop(req_pool_idx, None)
        self._pending.pop(req_pool_idx, None)
        self._spec_tokens.pop(req_pool_idx, None)
        self._spec_draft_probs.pop(req_pool_idx, None)
        self._force_next_token.pop(req_pool_idx, None)
        self._mamba_track_commit_info.pop(req_pool_idx, None)
        self._hybrid_boundary_commits.pop(req_pool_idx, None)
        key = self._hybrid_state_keys.pop(req_pool_idx, None)
        backend = getattr(self, "_configured_region_backend", None)
        if key is not None and backend is not None:
            backend.invalidate_request_state(key.request_id)

    def _write_trace_records(
        self,
        records: List[Dict[str, Any]],
        full_logits: torch.Tensor,
        device: torch.device,
    ) -> None:
        if not self._trace_path or not records:
            return

        row_records = [
            rec for rec in records
            if rec.get("logit_row") is not None
            and full_logits is not None
            and 0 <= rec["logit_row"] < full_logits.shape[0]
        ]
        if row_records:
            rows = torch.tensor(
                [rec["logit_row"] for rec in row_records],
                dtype=torch.long,
                device=device,
            )
            k = min(self._trace_topk, full_logits.shape[-1])
            vals, idx = full_logits[rows].float().topk(k, dim=-1)
            vals_cpu = vals.detach().cpu().tolist()
            idx_cpu = idx.detach().cpu().tolist()
            for rec, top_vals, top_ids in zip(row_records, vals_cpu, idx_cpu):
                rec["top_tokens"] = top_ids
                rec["top_logits"] = top_vals
                rec["argmax"] = top_ids[0] if top_ids else None
                rec["argmax_margin"] = (
                    top_vals[0] - top_vals[1] if len(top_vals) > 1 else None
                )

        os.makedirs(os.path.dirname(self._trace_path) or ".", exist_ok=True)
        with open(self._trace_path, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, sort_keys=True) + "\n")

    def _get_gdn_dllm_backend(self, model_runner):
        """Get the GDNDllmBackend if the model uses one, else None."""
        backend = getattr(model_runner, 'attn_backend', None)
        if backend is None:
            return None
        linear_backend = getattr(backend, 'linear_attn_backend', None)
        if linear_backend is None:
            return None
        from sglang.srt.layers.attention.linear.gdn_dllm_backend import GDNDllmBackend
        if isinstance(linear_backend, GDNDllmBackend):
            if self.exact_prefix_handoff:
                linear_backend.configure_region_state_cache(
                    max_entries=self._region_state_cache_max_entries,
                    strict_validation=self._strict_region_state_validation,
                )
                self._configured_region_backend = linear_backend
            return linear_backend
        return None

    def _hybrid_key_for_bid(
        self, forward_batch: ForwardBatch, bid: int, req_pool_idx: int
    ) -> RegionStateKey:
        def field(name, default):
            values = getattr(forward_batch, name, None)
            return values[bid] if values is not None else default

        boundary = int(field("hybrid_ar_boundaries_cpu", -1))
        if boundary < 0:
            raise RuntimeError("exact prefix handoff is missing ar_boundary metadata")
        request_ids = getattr(forward_batch, "rids", None) or []
        request_id = (
            str(request_ids[bid])
            if bid < len(request_ids)
            else f"req_pool_idx:{req_pool_idx}"
        )
        return RegionStateKey(
            request_id=request_id,
            request_pool_idx=int(req_pool_idx),
            request_slot_generation=int(
                field("hybrid_request_slot_generations_cpu", 0)
            ),
            region_id="causal_prefix",
            region_version=int(field("hybrid_region_versions_cpu", 0)),
            boundary=boundary,
            token_hash=str(field("hybrid_token_hashes_cpu", "")),
            position_hash=str(field("hybrid_position_hashes_cpu", "")),
            model_identity=str(field("hybrid_model_identities_cpu", "")),
            model_revision=str(field("hybrid_model_revisions_cpu", "")),
            adapter_identity=str(field("hybrid_adapter_identities_cpu", "")),
            adapter_revision=str(field("hybrid_adapter_revisions_cpu", "")),
            attention_contract_id=str(
                field("hybrid_attention_contract_ids_cpu", "")
            ),
            parent_region_versions=(),
        )

    @staticmethod
    def _hybrid_location_hash(locations: torch.Tensor) -> str:
        """Return a compact diagnostic digest without changing tensor ownership."""
        digest = hashlib.sha256()
        for location in locations.detach().to(device="cpu", dtype=torch.int64).tolist():
            digest.update(int(location).to_bytes(8, "little", signed=True))
        return digest.hexdigest()[:16]

    @staticmethod
    def _hybrid_key_log_fields(key: RegionStateKey) -> Dict[str, Any]:
        return {
            "request_id": key.request_id,
            "request_pool_idx": key.request_pool_idx,
            "request_slot_generation": key.request_slot_generation,
            "boundary": key.boundary,
            "region_version": key.region_version,
            "token_hash": key.token_hash,
            "position_hash": key.position_hash,
            "model_identity": key.model_identity,
            "model_revision": key.model_revision,
            "adapter_identity": key.adapter_identity,
            "adapter_revision": key.adapter_revision,
            "attention_contract": key.attention_contract_id,
        }

    def _hybrid_debug_log(self, event: str, **fields: Any) -> None:
        if self._exact_handoff_debug or self._exact_handoff_debug_sync:
            logger.info(
                "[HybridExactHandoff] %s %s",
                event,
                json.dumps(fields, sort_keys=True, default=str),
            )

    def _hybrid_debug_synchronize(
        self, stage: str, device: Union[str, torch.device]
    ) -> None:
        if not self._exact_handoff_debug_sync:
            return
        device = torch.device(device)
        if device.type != "cuda":
            return
        self._hybrid_debug_log("cuda_sync_begin", stage=stage, device=str(device))
        torch.cuda.synchronize(device)
        self._hybrid_debug_log("cuda_sync_complete", stage=stage, device=str(device))

    @staticmethod
    def _canonical_replay_kv_locations(
        model_runner: ModelRunner,
        key: RegionStateKey,
        required_device: torch.device,
    ) -> torch.Tensor:
        """Capture replay write locations using ForwardBatch's int64 contract."""
        locations = canonicalize_kv_prefix_locations(
            model_runner.req_to_token_pool.req_to_token[
                key.request_pool_idx, : key.boundary
            ],
            key.boundary,
            pool_size=model_runner.token_to_kv_pool.size,
        )
        required_device = torch.device(required_device)
        if locations.device != required_device:
            raise RuntimeError(
                "recovery replay KV locations are on the wrong device: "
                f"{locations.device} != {required_device}"
            )
        if (
            locations.dtype != torch.int64
            or locations.ndim != 1
            or not locations.is_contiguous()
            or locations.numel() != key.boundary
        ):
            raise RuntimeError("recovery replay KV locations are not canonical")
        return locations

    @staticmethod
    def _validate_recovery_forward_batch(
        replay: ForwardBatch,
        model_runner: ModelRunner,
        key: RegionStateKey,
        mamba_idx: int,
    ) -> None:
        """Fail before CUDA launch if a recovery causal-prefill is inconsistent."""
        locations = replay.out_cache_loc
        pool_size = int(model_runner.token_to_kv_pool.size)
        checks = (
            (replay.batch_size == 1, "recovery batch_size must be one"),
            (replay.input_ids.numel() == key.boundary, "input length mismatch"),
            (replay.positions.numel() == key.boundary, "position length mismatch"),
            (replay.req_pool_indices.numel() == 1, "request index mismatch"),
            (replay.seq_lens.numel() == 1, "sequence metadata mismatch"),
            (int(replay.seq_lens[0].item()) == key.boundary, "sequence length mismatch"),
            (replay.seq_lens_sum == key.boundary, "sequence sum mismatch"),
            (replay.extend_num_tokens == key.boundary, "extend token mismatch"),
            (
                replay.extend_seq_lens.numel() == 1
                and int(replay.extend_seq_lens[0].item()) == key.boundary,
                "GPU extend length mismatch",
            ),
            (
                replay.extend_prefix_lens.numel() == 1
                and int(replay.extend_prefix_lens[0].item()) == 0,
                "GPU extend prefix mismatch",
            ),
            (replay.extend_seq_lens_cpu == [key.boundary], "CPU extend mismatch"),
            (replay.extend_prefix_lens_cpu == [0], "replay prefix must be zero"),
            (
                replay.dllm_request_token_counts == [key.boundary],
                "dLLM token-count mismatch",
            ),
            (
                replay.dllm_attn_mask_types_cpu
                == [DLLM_ATTN_MASK_CAUSAL_PREFILL],
                "recovery mask must describe one causal prefill",
            ),
            (
                replay.dllm_attn_mask_types.numel() == 1
                and int(replay.dllm_attn_mask_types[0].item())
                == DLLM_ATTN_MASK_CAUSAL_PREFILL,
                "GPU recovery mask mismatch",
            ),
            (locations.dtype == torch.int64, "KV locations must be int64"),
            (locations.ndim == 1, "KV locations must be one-dimensional"),
            (locations.is_contiguous(), "KV locations must be contiguous"),
            (locations.numel() == key.boundary, "KV location count mismatch"),
        )
        for condition, message in checks:
            if not condition:
                raise RuntimeError(f"invalid recovery ForwardBatch: {message}")
        if locations.numel() and not bool(
            ((locations >= 0) & (locations < pool_size)).all().item()
        ):
            raise RuntimeError("invalid recovery ForwardBatch: KV location out of range")
        request_pool_size = int(model_runner.req_to_token_pool.req_to_token.shape[0])
        request_pool_idx = int(replay.req_pool_indices[0].item())
        if request_pool_idx != key.request_pool_idx or not (
            0 <= request_pool_idx < request_pool_size
        ):
            raise RuntimeError("invalid recovery ForwardBatch: request slot out of range")
        mamba_cache = model_runner.req_to_token_pool.mamba_pool.mamba_cache
        mamba_pool_size = int(mamba_cache.temporal.shape[1])
        if not 0 <= int(mamba_idx) < mamba_pool_size:
            raise RuntimeError("invalid recovery ForwardBatch: Mamba slot out of range")
        mapped_mamba_idx = int(
            model_runner.req_to_token_pool.req_index_to_mamba_index_mapping[
                request_pool_idx
            ].item()
        )
        if mapped_mamba_idx != int(mamba_idx):
            raise RuntimeError("invalid recovery ForwardBatch: Mamba mapping mismatch")

    @staticmethod
    def _hybrid_kv_reference(
        model_runner: ModelRunner, key: RegionStateKey
    ) -> KVPrefixReference:
        locations = canonicalize_kv_prefix_locations(
            model_runner.req_to_token_pool.req_to_token[
                key.request_pool_idx, : key.boundary
            ],
            key.boundary,
            pool_size=model_runner.token_to_kv_pool.size,
        )
        return KVPrefixReference(
            request_id=key.request_id,
            request_pool_idx=key.request_pool_idx,
            request_slot_generation=key.request_slot_generation,
            pool_identity=id(model_runner.req_to_token_pool),
            locations=locations,
            valid_length=key.boundary,
        )

    def _snapshot_hybrid_boundaries(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        bids: List[int],
        req_pool_indices_cpu: List[int],
    ) -> None:
        if not self.exact_prefix_handoff or not bids:
            return
        backend = self._get_gdn_dllm_backend(model_runner)
        if backend is None:
            raise RuntimeError("exact_prefix_handoff requires the dLLM GDN backend")
        for bid in bids:
            rpx = int(req_pool_indices_cpu[bid])
            key = self._hybrid_key_for_bid(forward_batch, bid, rpx)
            seq_lens = getattr(forward_batch, "seq_lens_cpu", None)
            if seq_lens is not None and int(seq_lens[bid]) < key.boundary:
                # Chunked inline prefill has not reached the canonical boundary.
                self._hybrid_debug_log(
                    "snapshot_skipped",
                    reason="sequence_before_boundary",
                    sequence_length=int(seq_lens[bid]),
                    **self._hybrid_key_log_fields(key),
                )
                continue
            mamba_idx = backend._current_mamba_slot(rpx)
            kv_prefix = self._hybrid_kv_reference(model_runner, key)
            self._hybrid_debug_synchronize(
                "snapshot_kv_reference_constructed", kv_prefix.locations.device
            )
            backend.snapshot_region_state(
                state_key=key,
                mamba_cache_idx=mamba_idx,
                kv_prefix=kv_prefix,
            )
            self._hybrid_debug_synchronize(
                "snapshot_gdn_state_published", kv_prefix.locations.device
            )
            self._hybrid_state_keys[rpx] = key
            if self._exact_handoff_debug or self._exact_handoff_debug_sync:
                self._hybrid_debug_log(
                    "snapshot_created",
                    mamba_slot=mamba_idx,
                    kv_location_hash=self._hybrid_location_hash(
                        kv_prefix.locations
                    ),
                    **self._hybrid_key_log_fields(key),
                )

    def _restore_hybrid_boundaries(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        bids: List[int],
        req_pool_indices_cpu: List[int],
    ) -> None:
        if not self.exact_prefix_handoff or not bids:
            return
        backend = self._get_gdn_dllm_backend(model_runner)
        if backend is None:
            raise RuntimeError("exact_prefix_handoff requires the dLLM GDN backend")
        for bid in bids:
            rpx = int(req_pool_indices_cpu[bid])
            requested = self._hybrid_key_for_bid(forward_batch, bid, rpx)
            current = self._hybrid_state_keys.get(rpx)
            debug_enabled = (
                self._exact_handoff_debug or self._exact_handoff_debug_sync
            )
            current_locations = None
            if debug_enabled:
                stored_state = backend.region_state_cache.peek(
                    current if current is not None else requested
                )
                current_locations = canonicalize_kv_prefix_locations(
                    model_runner.req_to_token_pool.req_to_token[
                        rpx, : requested.boundary
                    ],
                    requested.boundary,
                    pool_size=model_runner.token_to_kv_pool.size,
                )
                self._hybrid_debug_log(
                    "restore_requested",
                    stored_key=(
                        self._hybrid_key_log_fields(current)
                        if current is not None
                        else None
                    ),
                    requested_key=self._hybrid_key_log_fields(requested),
                    stored_kv_location_hash=(
                        self._hybrid_location_hash(
                            stored_state.kv_prefix.locations
                        )
                        if stored_state is not None
                        else None
                    ),
                    current_kv_location_hash=self._hybrid_location_hash(
                        current_locations
                    ),
                    current_mamba_slot=backend._current_mamba_slot(rpx),
                )
            # A differing indexed key is lifecycle divergence, not a cache
            # miss. Preserve the committed snapshot for diagnosis and fail
            # before any recovery kernel can obscure the model/scheduler bug.
            if current is not None and current != requested:
                stored_fields = self._hybrid_key_log_fields(current)
                requested_fields = self._hybrid_key_log_fields(requested)
                logger.error(
                    "Exact-prefix lifecycle divergence before restore: "
                    "stored=%s requested=%s",
                    json.dumps(stored_fields, sort_keys=True),
                    json.dumps(requested_fields, sort_keys=True),
                )
                raise RuntimeError(
                    "exact-prefix committed key differs from scheduler-requested key"
                )
            lookup = backend.restore_region_state(
                state_key=requested,
                mamba_cache_idx=backend._current_mamba_slot(rpx),
                current_slot_generation=requested.request_slot_generation,
            )
            if lookup.hit:
                self._hybrid_state_keys[rpx] = requested
                assert lookup.state is not None
                if debug_enabled:
                    self._hybrid_debug_log(
                        "restore_hit",
                        stored_kv_location_hash=self._hybrid_location_hash(
                            lookup.state.kv_prefix.locations
                        ),
                        current_kv_location_hash=self._hybrid_location_hash(
                            current_locations
                        ),
                        current_mamba_slot=backend._current_mamba_slot(rpx),
                        **self._hybrid_key_log_fields(requested),
                    )
                continue
            if debug_enabled:
                self._hybrid_debug_log(
                    "restore_miss",
                    miss_reason=(
                        lookup.miss_reason.value
                        if lookup.miss_reason is not None
                        else None
                    ),
                    current_kv_location_hash=self._hybrid_location_hash(
                        current_locations
                    ),
                    current_mamba_slot=backend._current_mamba_slot(rpx),
                    **self._hybrid_key_log_fields(requested),
                )
            backend.region_state_cache.invalidate_key(requested)
            self._hybrid_state_keys.pop(rpx, None)
            self._recompute_hybrid_boundary(
                model_runner, forward_batch, bid, requested, backend
            )

    def _recompute_hybrid_boundary(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        bid: int,
        key: RegionStateKey,
        backend,
    ) -> None:
        """Safely rebuild a missing boundary with the native causal path."""
        stable_tokens_by_request = getattr(
            forward_batch, "hybrid_stable_token_ids_cpu", None
        )
        if stable_tokens_by_request is None or bid >= len(stable_tokens_by_request):
            raise RuntimeError("safe prefix recomputation is missing stable token IDs")
        stable_tokens = list(stable_tokens_by_request[bid])
        if len(stable_tokens) != key.boundary:
            raise RuntimeError(
                "safe prefix recomputation token count does not match ar_boundary"
            )
        if hash_token_ids(stable_tokens) != key.token_hash:
            raise RuntimeError(
                "safe prefix recomputation token hash does not match the state key"
            )

        device = forward_batch.input_ids.device
        mamba_idx = backend._current_mamba_slot(key.request_pool_idx)
        self._hybrid_debug_log(
            "recovery_replay_invoked",
            batch_index=bid,
            mamba_slot=mamba_idx,
            **self._hybrid_key_log_fields(key),
        )
        mamba_cache = backend.req_to_token_pool.mamba_pool.mamba_cache
        for conv_state in mamba_cache.conv:
            conv_state[:, mamba_idx].zero_()
        mamba_cache.temporal[:, mamba_idx].zero_()
        self._hybrid_debug_synchronize("recovery_mamba_state_zeroed", device)

        # req_to_token uses an int32 page table internally, but all
        # ForwardBatch KV write locations are int64.  Keep this owned copy
        # reachable through replay for the entire forward.
        replay_locations = self._canonical_replay_kv_locations(
            model_runner, key, device
        )

        replay_input_ids = torch.tensor(
            stable_tokens, dtype=torch.int64, device=device
        )
        replay_req_pool_indices = (
            forward_batch.req_pool_indices[bid : bid + 1]
            .detach()
            .clone()
            .contiguous()
        )
        replay_seq_lens = torch.tensor(
            [key.boundary], dtype=forward_batch.seq_lens.dtype, device=device
        )
        replay_positions = torch.arange(
            key.boundary,
            dtype=(
                forward_batch.positions.dtype
                if forward_batch.positions is not None
                else torch.int64
            ),
            device=device,
        )
        replay_extend_seq_lens = torch.tensor(
            [key.boundary], dtype=torch.int32, device=device
        )
        replay_extend_prefix_lens = torch.zeros(
            1, dtype=torch.int32, device=device
        )
        replay_extend_start_loc = torch.zeros(
            1, dtype=torch.int32, device=device
        )
        replay_lora_ids = (
            [forward_batch.lora_ids[bid]]
            if getattr(forward_batch, "lora_ids", None) is not None
            else None
        )
        # Construct a new dataclass so active-decode tensors, graph buffers,
        # attention metadata and mutable per-request lists cannot leak into the
        # recovery causal-prefill.
        replay = ForwardBatch(
            forward_mode=ForwardMode.DLLM_MIXED,
            batch_size=1,
            input_ids=replay_input_ids,
            req_pool_indices=replay_req_pool_indices,
            seq_lens=replay_seq_lens,
            out_cache_loc=replay_locations,
            seq_lens_sum=key.boundary,
            orig_seq_lens=replay_seq_lens.to(dtype=torch.int32),
            seq_lens_cpu=torch.tensor([key.boundary], dtype=torch.int64),
            positions=replay_positions,
            extend_num_tokens=key.boundary,
            extend_seq_lens=replay_extend_seq_lens,
            extend_prefix_lens=replay_extend_prefix_lens,
            extend_start_loc=replay_extend_start_loc,
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[key.boundary],
            lora_ids=replay_lora_ids,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool=model_runner.token_to_kv_pool,
            attn_backend=model_runner.attn_backend,
            capture_hidden_mode=getattr(
                forward_batch, "capture_hidden_mode", None
            ),
            global_forward_mode=ForwardMode.DLLM_MIXED,
        )
        replay.dllm_request_token_counts = [key.boundary]
        replay.dllm_attn_mask_types_cpu = [DLLM_ATTN_MASK_CAUSAL_PREFILL]
        replay.dllm_attn_mask_types = torch.tensor(
            [DLLM_ATTN_MASK_CAUSAL_PREFILL], dtype=torch.int32, device=device
        )
        replay.dllm_force_causal = True
        replay.dllm_force_bidir_mask = False
        replay.dllm_bidir_custom_mask = None
        replay.dllm_gdn_persist_state = True
        replay.dllm_gdn_causal_mode = 1
        replay.dllm_gdn_num_clean = key.boundary
        replay.dllm_gdn_block_size = key.boundary
        replay.dllm_gdn_save_for_commit = False
        replay.dllm_gdn_cache_intermediate_for_commit = False
        replay.dllm_gdn_use_graph_save_buffer = False
        replay.dllm_return_argmax_only = False
        replay.dllm_return_topk_probs = False
        replay.rids = [key.request_id]
        replay.return_logprob = False
        replay.input_embeds = None
        replay.replace_embeds = None
        replay.replace_positions = None
        replay.mamba_track_mask = None
        replay.mamba_track_indices = None
        replay.mamba_track_seqlens = None
        replay.dllm_mamba_track_indices_cpu = None
        replay.dllm_mamba_track_steps_cpu = None
        replay.dllm_mamba_track_boundaries_cpu = None
        if self.conditional_lora and getattr(replay, "lora_ids", None) is not None:
            replay.lora_segment_ids = [None]
            replay.lora_segment_lens_cpu = [key.boundary]
        self._validate_recovery_forward_batch(
            replay, model_runner, key, mamba_idx
        )
        lora_manager = getattr(model_runner, "lora_manager", None)
        primary_exception = None
        try:
            if (
                lora_manager is not None
                and getattr(replay, "lora_ids", None) is not None
            ):
                lora_manager.prepare_lora_batch(replay)
            # Initialize full-paged/GDN metadata for the replay shape explicitly;
            # the normal decode call that follows initializes its own metadata.
            model_runner.attn_backend.init_forward_metadata(replay)
            self._hybrid_debug_synchronize(
                "recovery_attention_metadata_prepared", device
            )
            model_runner.forward(
                replay,
                skip_attn_backend_init=True,
                pp_proxy_tensors=None,
            )
            self._hybrid_debug_synchronize("recovery_model_forward", device)
        except BaseException as exc:
            primary_exception = exc
            raise
        finally:
            if (
                lora_manager is not None
                and getattr(forward_batch, "lora_ids", None) is not None
            ):
                try:
                    lora_manager.prepare_lora_batch(forward_batch)
                except BaseException:
                    if primary_exception is None:
                        raise
                    logger.exception(
                        "Failed to restore active decode LoRA metadata after "
                        "a recovery replay failure; preserving the primary exception"
                    )

        kv_prefix = self._hybrid_kv_reference(model_runner, key)
        self._hybrid_debug_synchronize(
            "recovery_kv_reference_constructed", kv_prefix.locations.device
        )
        backend.snapshot_region_state(
            state_key=key,
            mamba_cache_idx=mamba_idx,
            kv_prefix=kv_prefix,
        )
        self._hybrid_debug_synchronize(
            "recovery_gdn_snapshot_published", kv_prefix.locations.device
        )
        self._hybrid_state_keys[key.request_pool_idx] = key

    def _commit_hybrid_boundaries(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        req_pool_indices_cpu: List[int],
        decode_bids: List[int],
        advances: List[int],
        committed_tokens: List[List[int]],
    ) -> None:
        if not self.exact_prefix_handoff or not decode_bids:
            return
        backend = self._get_gdn_dllm_backend(model_runner)
        assert backend is not None
        for bid in decode_bids:
            rpx = int(req_pool_indices_cpu[bid])
            old_key = self._hybrid_state_keys[rpx]
            advance = int(advances[bid])
            new_boundary = old_key.boundary + advance
            stable_tokens_by_request = getattr(
                forward_batch, "hybrid_stable_token_ids_cpu", None
            )
            if stable_tokens_by_request is None:
                raise RuntimeError("accepted-state commit is missing stable token IDs")
            stable_tokens = list(stable_tokens_by_request[bid])
            if (
                len(stable_tokens) != old_key.boundary
                or hash_token_ids(stable_tokens) != old_key.token_hash
            ):
                raise RuntimeError(
                    "accepted-state commit origin does not match the sealed prefix"
                )
            committed_token_ids = tuple(committed_tokens[bid][:advance])
            if len(committed_token_ids) != advance:
                raise RuntimeError(
                    "accepted-state commit does not contain committed_advance tokens"
                )
            new_key = replace(
                old_key,
                region_version=old_key.region_version + 1,
                boundary=new_boundary,
                token_hash=hash_token_ids(
                    stable_tokens + list(committed_token_ids)
                ),
                position_hash=hash_positions(0, new_boundary),
            )
            backend.commit_region_state(
                state_key=new_key,
                mamba_cache_idx=backend._current_mamba_slot(rpx),
                kv_prefix=self._hybrid_kv_reference(model_runner, new_key),
            )
            backend.region_state_cache.invalidate_key(old_key)
            self._hybrid_state_keys[rpx] = new_key
            self._hybrid_boundary_commits[rpx] = HybridBoundaryCommit(
                request_id=new_key.request_id,
                request_pool_idx=rpx,
                request_slot_generation=new_key.request_slot_generation,
                region_id=new_key.region_id,
                previous_boundary=old_key.boundary,
                previous_region_version=old_key.region_version,
                previous_token_hash=old_key.token_hash,
                previous_position_hash=old_key.position_hash,
                committed_advance=advance,
                committed_token_ids=committed_token_ids,
                new_boundary=new_key.boundary,
                new_region_version=new_key.region_version,
                token_hash=new_key.token_hash,
                position_hash=new_key.position_hash,
                model_identity=new_key.model_identity,
                model_revision=new_key.model_revision,
                adapter_identity=new_key.adapter_identity,
                adapter_revision=new_key.adapter_revision,
                attention_contract_id=new_key.attention_contract_id,
            )

    def _get_gdn_layer(self, model_runner, layer_id):
        """Get the RadixLinearAttention layer for a GDN layer_id.

        Model structure: model.model.layers[layer_id].linear_attn.attn
        where linear_attn is Qwen3_5GatedDeltaNet and .attn is RadixLinearAttention.
        """
        model = model_runner.model
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layer_module = model.model.layers[layer_id]
            gdn = getattr(layer_module, 'linear_attn', None)
            if gdn is not None:
                return getattr(gdn, 'attn', None)
        return None

    @staticmethod
    def _dense_proposal(prop, vocab_size: int, device: torch.device) -> torch.Tensor:
        """Convert a sparse `(ids, probs)` proposal row to dense probabilities."""
        if not isinstance(prop, tuple):
            return prop
        ids, probs = prop
        dense = torch.zeros(vocab_size, dtype=probs.dtype, device=device)
        dense.scatter_(0, ids.to(device=device, dtype=torch.long), probs.to(device))
        return dense

    def _setup_conditional_lora(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        extend_lens_cpu: List[int],
        is_prefill: List[bool],
        case_types: List[str],
        old_specs: List,
    ):
        """Set per-token LoRA mask: 0 for verify tokens, 1 for MASK tokens.

        Matches reference generate.py L3137-3142:
          lora_mask = [0]*n_verify + [1]*num_masks
        """
        lora_backend = model_runner.lora_manager.lora_backend

        # Fast path: when cuBLAS graph is captured and batch fits in mask buffer,
        # just update the mask tensor (avoids expensive segment computation).
        if lora_backend.cublas_graph_captured:
            total_tokens = sum(el for el in extend_lens_cpu if el > 0)
            if lora_backend.lora_mask is not None and total_tokens <= lora_backend.lora_mask.shape[0]:
                mask_values = []
                for bid, extend_len in enumerate(extend_lens_cpu):
                    if extend_len <= 0:
                        continue
                    req_lora_id = forward_batch.lora_ids[bid]
                    if req_lora_id is None or is_prefill[bid] or case_types[bid] == "P":
                        mask_values.extend([0.0] * extend_len)
                        continue
                    if case_types[bid] == "V":
                        base_len = 1 + len(old_specs[bid] or [])
                    else:
                        base_len = 1
                    draft_len = self.num_masks
                    pad_len = extend_len - base_len - draft_len
                    mask_values.extend([0.0] * base_len)
                    mask_values.extend([1.0] * draft_len)
                    mask_values.extend([0.0] * max(0, pad_len))
                lora_backend.update_lora_mask(mask_values)
                return
            # else: fall through to segment-based routing for large batches

        # Non-graph path: use segment-based routing (original csgmv path)
        lora_ids = forward_batch.lora_ids
        segment_ids: List[Union[str, None]] = []
        segment_lens: List[int] = []

        for bid, extend_len in enumerate(extend_lens_cpu):
            req_lora_id = lora_ids[bid]
            if extend_len <= 0:
                continue

            # Prefill or no LoRA → single segment, all base
            if req_lora_id is None or is_prefill[bid] or case_types[bid] == "P":
                segment_ids.append(None)
                segment_lens.append(extend_len)
                continue

            # Compute base_len (verify positions) and draft_len (MASK positions)
            if case_types[bid] == "V":
                base_len = 1 + len(old_specs[bid] or [])
            else:  # Cold start
                base_len = 1
            draft_len = self.num_masks
            pad_len = extend_len - base_len - draft_len

            # Base segment (committed + specs): no LoRA
            if base_len > 0:
                segment_ids.append(None)
                segment_lens.append(base_len)
            # MASK segment: apply LoRA
            if draft_len > 0:
                segment_ids.append(req_lora_id)
                segment_lens.append(draft_len)
            # Padding segment: no LoRA
            if pad_len > 0:
                segment_ids.append(None)
                segment_lens.append(pad_len)

        forward_batch.lora_segment_ids = segment_ids
        forward_batch.lora_segment_lens_cpu = segment_lens
        model_runner.lora_manager.prepare_lora_batch(forward_batch)

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        overlap_fn=None,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        # Result-scoped handoff publications must never leak into a later
        # scheduler result when the algorithm object is reused.
        self._hybrid_boundary_commits.clear()
        if self._timing_enabled:
            _t_run_start = time.perf_counter()
        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device
        blk = self.block_size  # 2*N - 1

        # Compute per-request extend lengths and cumulative offsets
        # for mixed decode+prefill batches (variable-length ragged layout)
        _el = forward_batch.extend_seq_lens
        if _el is not None:
            extend_lens_cpu = (
                _el.tolist()
                if isinstance(_el, torch.Tensor)
                else list(_el)
            )
        else:
            extend_lens_cpu = [blk] * batch_size
        # Decode batches are flat [batch_size, block_size]. Some cached/mixed
        # scheduler paths can leave extend_seq_lens carrying prefill lengths;
        # use the actual input layout as the source of truth for self-spec offsets.
        if sum(extend_lens_cpu) != forward_batch.input_ids.numel():
            extend_lens_cpu = [blk] * batch_size
        forward_batch.dllm_request_token_counts = extend_lens_cpu
        # Cumulative offsets: base[bid] = sum(extend_lens[:bid])
        base_offsets = [0] * batch_size
        for bid in range(1, batch_size):
            base_offsets[bid] = base_offsets[bid - 1] + extend_lens_cpu[bid - 1]

        attn_mask_types_cpu = getattr(forward_batch, "dllm_attn_mask_types_cpu", None)
        if attn_mask_types_cpu is not None and len(attn_mask_types_cpu) == batch_size:
            is_prefill = [
                attn_mask_types_cpu[bid] == DLLM_ATTN_MASK_CAUSAL_PREFILL
                for bid in range(batch_size)
            ]
        else:
            # Fallback for batches produced before explicit dLLM mask metadata.
            is_prefill = [extend_lens_cpu[bid] != blk for bid in range(batch_size)]

        # Use cached CPU values from prepare_for_dllm_decode when available
        # (avoids GPU→CPU sync via torch.cat + tolist)
        _cached_rpx = getattr(forward_batch, 'dllm_rpx_cpu', None)
        if _cached_rpx is not None and len(_cached_rpx) == batch_size:
            req_pool_indices_cpu = _cached_rpx
            seq_lens_cpu = forward_batch.dllm_seq_lens_cpu
        else:
            # Fallback: GPU→CPU sync (first forward or non-decode-loop)
            _combined = torch.cat([
                forward_batch.req_pool_indices[:batch_size],
                forward_batch.seq_lens[:batch_size].to(forward_batch.req_pool_indices.dtype),
            ])
            _combined_cpu = _combined.tolist()
            req_pool_indices_cpu = _combined_cpu[:batch_size]
            seq_lens_cpu = [int(x) for x in _combined_cpu[batch_size:]]
        has_any_decode = not all(is_prefill)

        # ── Pure prefill ──────────────────────────────────────────────
        if not has_any_decode:
            # Conditional LoRA: prefill is base-only (no LoRA)
            if self.conditional_lora and forward_batch.lora_ids is not None:
                forward_batch.lora_segment_ids = [None] * batch_size
                forward_batch.lora_segment_lens_cpu = extend_lens_cpu
                model_runner.lora_manager.prepare_lora_batch(forward_batch)
            prefix_lens = forward_batch.extend_prefix_lens
            total_new = 0
            for bid in range(batch_size):
                seq_len = int(forward_batch.seq_lens[bid].item())
                prefix_len = (
                    int(prefix_lens[bid].item()) if prefix_lens is not None else 0
                )
                total_new += max(0, seq_len - prefix_len)
            if total_new == 0:
                return LogitsProcessorOutput(next_token_logits=None), [], False

            forward_batch.dllm_return_argmax_only = (
                self.temperature <= 0 and not self._trace_path
            )
            forward_batch.dllm_return_topk_probs = (
                self._use_sparse_topk_logits and not self._trace_path
            )
            forward_batch.dllm_top_k = self.top_k
            forward_batch.dllm_top_p = self.top_p
            forward_batch.dllm_temperature = self.temperature
            out = self._forward_with_metrics(
                model_runner,
                forward_batch,
                modes="prefill",
            )
            forward_batch.dllm_return_argmax_only = False
            forward_batch.dllm_return_topk_probs = False
            full_logits = out.logits_output.full_logits
            full_argmax = getattr(out.logits_output, "full_argmax", None)
            full_topk_ids = getattr(out.logits_output, "full_topk_ids", None)
            full_topk_probs = getattr(out.logits_output, "full_topk_probs", None)

            offset = 0
            for bid in range(batch_size):
                seq_len = int(forward_batch.seq_lens[bid].item())
                prefix_len = (
                    int(prefix_lens[bid].item()) if prefix_lens is not None else 0
                )
                n_new = seq_len - prefix_len
                rpx = req_pool_indices_cpu[bid]
                if n_new <= 0:
                    offset += max(n_new, 0)
                    continue
                last_idx = offset + n_new - 1
                if (
                    self.temperature <= 0
                    and full_argmax is not None
                    and last_idx >= 0
                    and last_idx < full_argmax.shape[0]
                ):
                    self._prev_last_argmax[rpx] = int(full_argmax[last_idx].item())
                elif (
                    full_topk_ids is not None
                    and full_topk_probs is not None
                    and last_idx >= 0
                    and last_idx < full_topk_ids.shape[0]
                ):
                    self._prev_last_topk[rpx] = (
                        full_topk_ids[last_idx].detach().clone(),
                        full_topk_probs[last_idx].detach().clone(),
                    )
                elif (
                    full_logits is not None
                    and last_idx >= 0
                    and last_idx < full_logits.shape[0]
                ):
                    self._prev_last_logits[rpx] = (
                        full_logits[last_idx].detach().clone()
                    )
                elif out.logits_output.next_token_logits is not None:
                    self._prev_last_logits[rpx] = (
                        out.logits_output.next_token_logits[bid].detach().clone()
                    )
                else:
                    raise RuntimeError(
                        "HybridDiffusionSelfSpec pure prefill requires full_logits or next_token_logits"
                    )
                offset += n_new

            self._stats["total_forwards"] += 1
            self._stats["prefill_forwards"] += 1
            self._flush_forward_timings()
            self._snapshot_hybrid_boundaries(
                model_runner,
                forward_batch,
                list(range(batch_size)),
                req_pool_indices_cpu,
            )
            return out.logits_output, [], out.can_run_graph

        # ── Decode (possibly mixed with inline prefill) ───────────────
        self._dllm_write_override.clear()
        self._kv_trim_info.clear()
        self._advance_override.clear()
        self._mamba_track_commit_info.clear()

        num_masks = self.num_masks  # N - 1

        # Phase 1: Classify requests and fill input_ids
        case_types = []
        t0_tokens = []
        was_forced = [False] * batch_size
        old_specs = [None] * batch_size
        old_draft_probs = [None] * batch_size  # full draft prob distributions per spec

        pre_sample_logits = []
        pre_sample_bids = []
        pre_sample_topk_ids = []
        pre_sample_topk_probs = []
        pre_sample_topk_bids = []

        for bid in range(batch_size):
            rpx = req_pool_indices_cpu[bid]

            # Skip inline prefill requests — they don't participate in decode
            if is_prefill[bid]:
                case_types.append('P')  # prefill
                t0_tokens.append(None)
                continue

            pending = self._pending.pop(rpx, None)
            specs = self._spec_tokens.pop(rpx, None)
            draft_probs = self._spec_draft_probs.pop(rpx, None)
            base = base_offsets[bid]

            if pending is not None and specs:
                forward_batch.input_ids[base + 0] = pending
                for si, sv in enumerate(specs):
                    forward_batch.input_ids[base + 1 + si] = sv
                case_types.append('V')
                t0_tokens.append(pending)
                old_specs[bid] = specs
                old_draft_probs[bid] = draft_probs
            else:
                forced = self._force_next_token.pop(rpx, None)
                if forced is not None:
                    forward_batch.input_ids[base] = forced
                    t0_tokens.append(forced)
                    was_forced[bid] = True
                else:
                    # Greedy shortcut: use cached argmax directly (no GPU tensor needed)
                    cached_argmax = self._prev_last_argmax.pop(rpx, None)
                    if cached_argmax is not None:
                        forward_batch.input_ids[base] = cached_argmax
                        t0_tokens.append(cached_argmax)
                    else:
                        prev_topk = self._prev_last_topk.get(rpx)
                        if prev_topk is not None:
                            ids, probs = prev_topk
                            pre_sample_topk_ids.append(ids)
                            pre_sample_topk_probs.append(probs)
                            pre_sample_topk_bids.append(bid)
                            t0_tokens.append(None)
                        prev = self._prev_last_logits.get(rpx)
                        if prev is not None and prev_topk is None:
                            pre_sample_logits.append(prev)
                            pre_sample_bids.append(bid)
                            t0_tokens.append(None)
                        elif prev_topk is None:
                            forward_batch.input_ids[base] = self.mask_id
                            t0_tokens.append(self.mask_id)
                case_types.append('C')

        # Batched pre-forward sampling for cold start t0
        if pre_sample_logits:
            sampled, _ = _batched_sample(
                torch.stack(pre_sample_logits),
                self.temperature, self.top_k, self.top_p,
            )
            sampled_cpu = sampled.tolist()
            for i, bid in enumerate(pre_sample_bids):
                tok = sampled_cpu[i]
                forward_batch.input_ids[base_offsets[bid]] = tok
                t0_tokens[bid] = tok
        if pre_sample_topk_ids:
            sampled = sample_sparse_probs(
                torch.stack(pre_sample_topk_ids),
                torch.stack(pre_sample_topk_probs),
            )
            sampled_cpu = sampled.tolist()
            for i, bid in enumerate(pre_sample_topk_bids):
                tok = sampled_cpu[i]
                forward_batch.input_ids[base_offsets[bid]] = tok
                t0_tokens[bid] = tok

        if self._timing_enabled:
            _t_phase1_end = time.perf_counter()

        # Phase 2: Forward (GPU async — returns before GPU finishes)
        # Set up conditional LoRA segments: base-only for verify, base+LoRA for MASK
        if self.conditional_lora and forward_batch.lora_ids is not None:
            self._setup_conditional_lora(
                model_runner, forward_batch, extend_lens_cpu,
                is_prefill, case_types, old_specs,
            )
        if self.variant in (
            SelfSpecVariant.BD_BIDIR,
            SelfSpecVariant.BD_BIDIR_SHIFT,
        ):
            forward_batch.dllm_force_causal = False
            forward_batch.dllm_force_bidir_mask = True
            forward_batch.dllm_bidir_custom_mask = self._bidir_mask
            forward_batch.dllm_gdn_causal_mode = 2
            forward_batch.dllm_gdn_num_clean = self.gen_block_size
        else:
            forward_batch.dllm_force_causal = True
            forward_batch.dllm_gdn_causal_mode = 1
        use_intermediate_commit = (
            has_any_decode
            and not any(is_prefill)
        )
        forward_batch.dllm_gdn_persist_state = False
        forward_batch.dllm_gdn_cache_intermediate_for_commit = use_intermediate_commit
        forward_batch.dllm_gdn_save_for_commit = not use_intermediate_commit
        forward_batch.dllm_gdn_block_size = self.block_size
        forward_batch.dllm_return_argmax_only = (
            self.temperature <= 0 and not self._trace_path
        )
        forward_batch.dllm_return_topk_probs = (
            self._use_sparse_topk_logits
            and not any(is_prefill)
            and not self._trace_path
        )
        forward_batch.dllm_top_k = self.top_k
        forward_batch.dllm_top_p = self.top_p
        forward_batch.dllm_temperature = self.temperature
        request_modes = [
            (
                "prefill"
                if is_prefill[bid]
                else (
                    "self_spec_verify"
                    if case_types[bid] == "V"
                    else "self_spec_cold_start"
                )
            )
            for bid in range(batch_size)
        ]
        self._restore_hybrid_boundaries(
            model_runner,
            forward_batch,
            [bid for bid in range(batch_size) if not is_prefill[bid]],
            req_pool_indices_cpu,
        )
        out = self._forward_with_metrics(
            model_runner,
            forward_batch,
            modes=request_modes,
            diffusion_steps=[not value for value in is_prefill],
            gdn_restores=[not value for value in is_prefill],
        )
        forward_batch.dllm_force_causal = False
        forward_batch.dllm_force_bidir_mask = False
        forward_batch.dllm_bidir_custom_mask = None
        forward_batch.dllm_gdn_persist_state = None
        forward_batch.dllm_gdn_save_for_commit = False
        forward_batch.dllm_gdn_cache_intermediate_for_commit = False
        forward_batch.dllm_return_argmax_only = False
        forward_batch.dllm_return_topk_probs = False

        # ── Overlap window: GPU is still computing, run CPU callback ──
        if overlap_fn is not None:
            overlap_fn()

        # Phase 3 starts here — first access to full_logits blocks until GPU done
        logits_output = out.logits_output
        full_logits = logits_output.full_logits
        full_argmax = getattr(logits_output, "full_argmax", None)
        full_topk_ids = getattr(logits_output, "full_topk_ids", None)
        full_topk_probs = getattr(logits_output, "full_topk_probs", None)
        logits_mode = getattr(logits_output, "dllm_logits_mode", None)
        if logits_mode is None:
            if full_logits is not None:
                logits_mode = "full"
            elif full_topk_ids is not None and full_topk_probs is not None:
                logits_mode = "topk"
            elif full_argmax is not None:
                logits_mode = "argmax"
        if logits_mode in ("topk_dense", "topk_tiled"):
            self._stats[f"logits_mode_{logits_mode}"] += 1
            self._stats["logits_mode_topk"] += 1
        elif logits_mode in ("full", "topk", "argmax"):
            self._stats[f"logits_mode_{logits_mode}"] += 1
        else:
            self._stats["logits_mode_unknown"] += 1
        if self._timing_enabled:
            torch.cuda.synchronize()  # For accurate phase timing (only when profiling)
            _t_phase2_end = time.perf_counter()

        # Phase 3: Batched post-forward — verify + sample + trim
        # seq_lens_cpu already computed above (combined tolist with req_pool_indices)
        req_to_token = model_runner.req_to_token_pool.req_to_token

        # Separate decode bids from prefill bids
        decode_bids = [bid for bid in range(batch_size) if not is_prefill[bid]]
        prefill_bids = [bid for bid in range(batch_size) if is_prefill[bid]]

        # ── Handle inline prefill requests ─────────────────────────
        # Save last logits for each prefill request (like pure prefill path)
        if prefill_bids:
            prefill_logit_indices = [
                base_offsets[bid] + extend_lens_cpu[bid] - 1
                for bid in prefill_bids
            ]
            prefill_argmax = None
            if (
                self.temperature <= 0
                and full_argmax is not None
                and prefill_logit_indices
                and max(prefill_logit_indices) < full_argmax.shape[0]
            ):
                prefill_idx_t = torch.tensor(
                    prefill_logit_indices, dtype=torch.long, device=device
                )
                prefill_argmax = full_argmax[prefill_idx_t].tolist()
                prefill_logits = None
            elif (
                full_logits is not None
                and prefill_logit_indices
                and max(prefill_logit_indices) < full_logits.shape[0]
            ):
                prefill_idx_t = torch.tensor(
                    prefill_logit_indices, dtype=torch.long, device=device
                )
                prefill_logits = full_logits[prefill_idx_t].detach().clone()
            elif logits_output.next_token_logits is not None:
                prefill_logits = logits_output.next_token_logits[
                    torch.tensor(prefill_bids, dtype=torch.long, device=device)
                ].detach().clone()
            else:
                raise RuntimeError(
                    "HybridDiffusionSelfSpec prefill branch requires full_logits or next_token_logits"
                )
            for k, bid in enumerate(prefill_bids):
                rpx = req_pool_indices_cpu[bid]
                if prefill_argmax is not None:
                    self._prev_last_argmax[rpx] = int(prefill_argmax[k])
                else:
                    self._prev_last_logits[rpx] = prefill_logits[k]

        # ── Steps 1+2: Verify + sample (all GPU work, then single CPU sync) ──
        verify_bids = [bid for bid in decode_bids if case_types[bid] == 'V']
        cold_bids = [bid for bid in decode_bids if case_types[bid] == 'C']
        request_verify_start = (
            time.perf_counter()
            if self._instrumentation_enabled and verify_bids
            else None
        )

        reject_at = {}       # bid -> rejected spec index
        corrected = {}       # bid -> corrected token id
        spec_corrections = {}  # bid -> list of corrected tokens per spec (output correction mode)
        gs = 1 + num_masks
        nv = 0
        vns = self.verify_num_specs

        # Build speculative sampling indices for ALL verify + cold bids
        # V2/V3 skip the discard position (first MASK overlaps last clean logit)
        # V1/V4 have uniform logit_shift → no discard
        _has_discard = self.variant in (
            SelfSpecVariant.BD_CAUSAL,
            SelfSpecVariant.BD_BIDIR,
        )
        sample_logit_indices = []
        sample_bid_roles = []
        for bid in verify_bids:
            n_specs = len(old_specs[bid])
            clean_idx = base_offsets[bid] + n_specs
            sample_logit_indices.append(clean_idx)  # clean position
            for j in range(1, gs):
                sample_logit_indices.append(clean_idx + j + (1 if _has_discard else 0))
            sample_bid_roles.append(bid)
        for bid in cold_bids:
            base = base_offsets[bid]
            sample_logit_indices.append(base)  # clean position 0
            for j in range(1, gs):
                sample_logit_indices.append(base + j + (1 if _has_discard else 0))
            sample_bid_roles.append(bid)

        if self.block_size == 7 and has_any_decode:
            verify_logit_indices_for_stats = []
            if self.use_spec_verify:
                for bid in verify_bids:
                    base = base_offsets[bid]
                    for si in range(vns):
                        verify_logit_indices_for_stats.append(base + si)
            consumed_rows = len(set(verify_logit_indices_for_stats + sample_logit_indices))
            if full_logits is not None:
                logits_rows = int(full_logits.shape[0])
            elif full_topk_ids is not None:
                logits_rows = int(full_topk_ids.shape[0])
            elif full_argmax is not None:
                logits_rows = int(full_argmax.shape[0])
            else:
                logits_rows = 0
            self._stats["logits_rows_total"] += logits_rows
            self._stats["logits_rows_consumed"] += consumed_rows
            self._stats["logits_rows_verify"] += len(verify_logit_indices_for_stats)
            self._stats["logits_rows_sample"] += len(sample_logit_indices)

        # --- All GPU work: verify + correction + sample ---
        all_accepted_gpu = None
        all_corr_tokens_gpu = None
        all_sample_ids_gpu = None
        draft_probs_all = None
        use_sparse_topk = (
            self.temperature > 0
            and full_topk_ids is not None
            and full_topk_probs is not None
        )
        if self._timing_enabled:
            _t_vs_start = time.perf_counter()

        # ── Fused greedy path: single gather + argmax for verify+sample ──
        if self.temperature <= 0 and not self.fast_verify:
            nv = len(verify_bids) if (self.use_spec_verify and verify_bids) else 0

            # Build combined index list: [verify_indices... | sample_indices...]
            all_gather_idx = []
            all_spec_vals = []
            if nv > 0:
                for bid in verify_bids:
                    base = base_offsets[bid]
                    for si in range(vns):
                        all_gather_idx.append(base + si)
                        all_spec_vals.append(old_specs[bid][si])

            verify_count = len(all_gather_idx)  # nv * vns
            combined_indices = all_gather_idx + sample_logit_indices

            if combined_indices:
                max_idx = max(combined_indices)
                if (
                    full_argmax is not None
                    and max_idx < full_argmax.shape[0]
                ):
                    combined_argmax = full_argmax[
                        torch.tensor(combined_indices, dtype=torch.long, device=device)
                    ]
                elif full_logits is not None and max_idx < full_logits.shape[0]:
                    # Single gather + single argmax for everything
                    combined_logits = full_logits[
                        torch.tensor(combined_indices, dtype=torch.long, device=device)
                    ]
                    combined_argmax = combined_logits.argmax(dim=-1)
                else:
                    got = None if full_logits is None else tuple(full_logits.shape)
                    got_argmax = (
                        None if full_argmax is None else tuple(full_argmax.shape)
                    )
                    nxt = (
                        None
                        if logits_output.next_token_logits is None
                        else tuple(logits_output.next_token_logits.shape)
                    )
                    raise RuntimeError(
                        "HybridDiffusionSelfSpec requires per-token full_logits for verify/sample "
                        f"indices up to {max_idx}, got full={got}, "
                        f"argmax={got_argmax}, next={nxt}, "
                        f"input_tokens={forward_batch.input_ids.numel()}, "
                        f"extend_lens={extend_lens_cpu}, graph={bool(out.can_run_graph)}"
                    )

                # Split results
                if verify_count > 0:
                    all_corr_tokens_gpu = combined_argmax[:verify_count]
                    all_spec_vals_t = torch.tensor(
                        all_spec_vals, dtype=torch.long, device=device
                    )
                    all_accepted_gpu = all_spec_vals_t == all_corr_tokens_gpu

                if sample_logit_indices:
                    all_sample_ids_gpu = combined_argmax[verify_count:]

            if self._timing_enabled:
                _t_vs_after_verify = time.perf_counter()
                _t_vs_after_sample = time.perf_counter()
        else:
            # ── Non-greedy path: separate verify + sample ──
            if self.use_spec_verify and verify_bids:
                nv = len(verify_bids)

                all_gather_idx = []
                all_spec_vals = []
                all_draft_prob_list = []
                _need_draft_probs = (
                    self.temperature > 0
                    and not self.fast_verify
                )
                for bid in verify_bids:
                    base = base_offsets[bid]
                    for si in range(vns):
                        all_gather_idx.append(base + si)
                        all_spec_vals.append(old_specs[bid][si])
                        if _need_draft_probs:
                            all_draft_prob_list.append(old_draft_probs[bid][si])

                gather_idx_t = torch.tensor(
                    all_gather_idx, dtype=torch.long, device=device
                )
                all_spec_vals_t = torch.tensor(
                    all_spec_vals, dtype=torch.long, device=device
                )

                if use_sparse_topk:
                    all_clean_ids_t = full_topk_ids[gather_idx_t]
                    all_clean_probs_t = full_topk_probs[gather_idx_t]
                    draft_ids_rows = []
                    draft_probs_rows = []
                    for prop in all_draft_prob_list:
                        if isinstance(prop, tuple):
                            draft_ids_rows.append(prop[0])
                            draft_probs_rows.append(prop[1])
                        else:
                            vals, ids = prop.topk(
                                min(self.top_k, prop.shape[-1]), dim=-1
                            )
                            draft_ids_rows.append(ids)
                            draft_probs_rows.append(vals / vals.sum())
                    all_draft_ids_t = torch.stack(draft_ids_rows)
                    all_draft_probs_t = torch.stack(draft_probs_rows)
                    all_accepted_gpu, all_corr_tokens_gpu = fused_sparse_spec_verify(
                        all_clean_ids_t,
                        all_clean_probs_t,
                        all_draft_ids_t,
                        all_draft_probs_t,
                        all_spec_vals_t,
                    )
                    self._stats["fused_verify_calls"] += 1
                else:
                    all_clean_logits = full_logits[gather_idx_t]

                    if self.fast_verify:
                        spec_logit_vals = all_clean_logits.gather(
                            1, all_spec_vals_t.unsqueeze(1)
                        ).squeeze(1)

                        if self.fast_verify_topk_per_spec and vns > 1:
                            per_spec_topk = self.fast_verify_topk_per_spec
                            all_accepted_gpu = torch.zeros(nv * vns, dtype=torch.bool, device=device)
                            for si in range(vns):
                                k = per_spec_topk[si] if si < len(per_spec_topk) else self.fast_verify_topk
                                si_indices = list(range(si, nv * vns, vns))
                                si_logits = all_clean_logits[si_indices]
                                si_spec_vals = spec_logit_vals[si_indices]
                                topk_vals_si, _ = si_logits.topk(k, dim=-1)
                                thresholds_si = topk_vals_si[:, -1]
                                accepted_si = si_spec_vals >= thresholds_si
                                for j, idx in enumerate(si_indices):
                                    all_accepted_gpu[idx] = accepted_si[j]
                        else:
                            topk_vals, _ = all_clean_logits.topk(
                                self.fast_verify_topk, dim=-1
                            )
                            topk_thresholds = topk_vals[:, -1]
                            all_accepted_gpu = spec_logit_vals >= topk_thresholds
                        all_corr_tokens_gpu = all_clean_logits.argmax(dim=-1)
                    elif self._full_softmax_verify_from_logits:
                        # AR-Trust Softmax-Argmax policy: the draft token is
                        # argmax and the verifier uses the full-softmax target.
                        if not _HAS_FUSED_VERIFY:
                            raise RuntimeError(
                                "argmax_softmax_verify requires fused_verify_kernel"
                            )
                        all_draft_logits_t = torch.stack(all_draft_prob_list)
                        self._gumbel_seed_counter += 1
                        all_accepted_gpu, all_corr_tokens_gpu = (
                            fused_spec_verify_from_logits(
                                all_clean_logits,
                                all_draft_logits_t,
                                all_spec_vals_t,
                                temperature=self.temperature,
                                alpha=self.verify_alpha,
                                gumbel_seed=self._gumbel_seed_counter,
                            )
                        )
                        self._stats["fused_verify_calls"] += 1
                    elif (
                        _HAS_FUSED_VERIFY
                        and self.verify_alpha > 0
                        and not self._uses_truncated_sampling
                    ):
                        # Exact full-softmax verify. For top-k/top-p sampling we must
                        # use the torch path below because both p and q are truncated.
                        all_draft_probs_t = torch.stack([
                            self._dense_proposal(prop, all_clean_logits.shape[-1], device)
                            for prop in all_draft_prob_list
                        ])
                        self._gumbel_seed_counter += 1
                        all_accepted_gpu, all_corr_tokens_gpu = fused_spec_verify(
                            all_clean_logits, all_draft_probs_t, all_spec_vals_t,
                            temperature=self.temperature,
                            alpha=self.verify_alpha,
                            gumbel_seed=self._gumbel_seed_counter,
                        )
                        self._stats["fused_verify_calls"] += 1
                    else:
                        # Standard verify: softmax/top-k/top-p + p/q ratio + correction
                        self._stats["torch_verify_calls"] += 1
                        all_draft_probs_t = torch.stack([
                            self._dense_proposal(prop, all_clean_logits.shape[-1], device)
                            for prop in all_draft_prob_list
                        ])
                        all_clean_probs = _sampling_probs(
                            all_clean_logits,
                            self.temperature,
                            self.top_k,
                            self.top_p,
                        )
                        all_p = all_clean_probs.gather(
                            1, all_spec_vals_t.unsqueeze(1)
                        ).squeeze(1)
                        all_q = all_draft_probs_t.gather(
                            1, all_spec_vals_t.unsqueeze(1)
                        ).squeeze(1)
                        all_ratios = torch.where(
                            all_q > 0, all_p / all_q, torch.zeros_like(all_p)
                        )
                        all_rands = torch.rand(nv * vns, device=device)
                        all_accepted_gpu = (all_ratios >= 1.0) | (
                            all_rands < all_ratios
                        )
                        corrected_dist = torch.clamp(
                            all_clean_probs - all_draft_probs_t, min=0
                        )
                        corrected_sums = corrected_dist.sum(dim=-1, keepdim=True)
                        corrected_dist = torch.where(
                            corrected_sums > 0,
                            corrected_dist / corrected_sums,
                            all_clean_probs,
                        )
                        all_corr_tokens_gpu = torch.multinomial(
                            corrected_dist, num_samples=1
                        ).squeeze(1)

            if self._timing_enabled:
                _t_vs_after_verify = time.perf_counter()

            # Speculative sampling for ALL verify + cold bids (before knowing verify results)
            if sample_logit_indices:
                sample_idx_t = torch.tensor(
                    sample_logit_indices, dtype=torch.long, device=device
                )

                if self.temperature > 0:
                    # Split clean and draft positions. The default mode samples
                    # both from the strict truncated proposal; argmax modes only
                    # sample the clean target token and use argmax for specs.
                    clean_indices = list(range(0, len(sample_logit_indices), gs))
                    draft_indices = []
                    for group_start in range(0, len(sample_logit_indices), gs):
                        for m in range(num_masks):
                            draft_indices.append(group_start + 1 + m)

                    if use_sparse_topk:
                        sample_topk_ids = full_topk_ids[sample_idx_t]
                        sample_topk_probs = full_topk_probs[sample_idx_t]
                        clean_probs = sample_topk_probs[clean_indices]
                        clean_ids = sample_sparse_probs(
                            sample_topk_ids[clean_indices],
                            clean_probs,
                        )
                        all_sample_ids_gpu = torch.empty(
                            len(sample_logit_indices),
                            dtype=clean_ids.dtype,
                            device=device,
                        )
                        all_sample_ids_gpu[clean_indices] = clean_ids
                        if draft_indices:
                            draft_probs = sample_topk_probs[draft_indices]
                            if self._argmax_draft:
                                draft_ids = sample_topk_ids[draft_indices, 0]
                                self._stats["draft_argmax_batches"] += 1
                            else:
                                draft_ids = sample_sparse_probs(
                                    sample_topk_ids[draft_indices],
                                    draft_probs,
                                )
                                self._stats["draft_sample_batches"] += 1
                            all_sample_ids_gpu[draft_indices] = draft_ids.to(
                                clean_ids.dtype
                            )
                            draft_probs_all = (
                                sample_topk_ids[draft_indices],
                                sample_topk_probs[draft_indices],
                            )
                    else:
                        all_sample_logits = full_logits[sample_idx_t]
                        clean_logits = all_sample_logits[clean_indices]
                        clean_ids, _ = _batched_sample(
                            clean_logits, self.temperature, self.top_k, self.top_p
                        )
                        all_sample_ids_gpu = torch.empty(
                            len(sample_logit_indices),
                            dtype=clean_ids.dtype,
                            device=device,
                        )
                        all_sample_ids_gpu[clean_indices] = clean_ids
                        if draft_indices:
                            draft_logits = all_sample_logits[draft_indices]
                            if self._argmax_draft:
                                draft_ids = draft_logits.argmax(dim=-1)
                                if self._full_softmax_verify_from_logits:
                                    draft_probs_all = draft_logits
                                else:
                                    draft_probs_all = _sampling_probs(
                                        draft_logits,
                                        self.temperature,
                                        self.top_k,
                                        self.top_p,
                                    )
                                self._stats["draft_argmax_batches"] += 1
                            else:
                                draft_ids, _, draft_probs_all = _batched_sample(
                                    draft_logits,
                                    self.temperature,
                                    self.top_k,
                                    self.top_p,
                                    return_probs=True,
                                )
                                self._stats["draft_sample_batches"] += 1
                            all_sample_ids_gpu[draft_indices] = draft_ids.to(
                                clean_ids.dtype
                            )
                else:
                    all_sample_logits = full_logits[sample_idx_t]
                    all_sample_ids_gpu, _ = _batched_sample(
                        all_sample_logits, self.temperature, self.top_k, self.top_p
                    )

            if self._timing_enabled:
                _t_vs_after_sample = time.perf_counter()

        # --- Single GPU→CPU sync: pack everything ---
        gpu_parts = []
        verify_len = 0
        if all_accepted_gpu is not None:
            verify_len = nv * vns
            gpu_parts.append(all_accepted_gpu.to(torch.int32))
            gpu_parts.append(all_corr_tokens_gpu)
        sample_len = 0
        if all_sample_ids_gpu is not None:
            sample_len = all_sample_ids_gpu.shape[0]
            gpu_parts.append(all_sample_ids_gpu)

        if gpu_parts:
            _mega_packed = torch.cat(gpu_parts)
            _mega_cpu = _mega_packed.tolist()
        else:
            _mega_cpu = []

        if self._timing_enabled:
            _t_vs_after_sync = time.perf_counter()

        # --- CPU unpack ---
        if verify_len > 0:
            all_accepted_cpu = _mega_cpu[:verify_len]
            all_corr_cpu = _mega_cpu[verify_len:2 * verify_len]

            for k, bid in enumerate(verify_bids):
                base_k = k * vns
                for si in range(vns):
                    if not all_accepted_cpu[base_k + si]:
                        reject_at[bid] = si
                        corrected[bid] = all_corr_cpu[base_k + si]
                        break
                if self.output_correction:
                    spec_corrections[bid] = [
                        all_corr_cpu[base_k + si] for si in range(vns)
                    ]

        sampled_results = {}
        draft_probs_map = {}
        if sample_len > 0:
            all_ids_cpu = _mega_cpu[2 * verify_len:]

            offset = 0
            draft_offset = 0
            for bid in sample_bid_roles:
                if bid not in reject_at:
                    sampled_results[bid] = all_ids_cpu[offset:offset + gs]
                    if draft_probs_all is not None:
                        if isinstance(draft_probs_all, tuple):
                            draft_ids_all, draft_prob_all = draft_probs_all
                            draft_probs_map[bid] = [
                                (
                                    draft_ids_all[draft_offset + m],
                                    draft_prob_all[draft_offset + m],
                                )
                                for m in range(num_masks)
                            ]
                        else:
                            draft_probs_map[bid] = [
                                draft_probs_all[draft_offset + m]
                                for m in range(num_masks)
                            ]
                    else:
                        draft_probs_map[bid] = []
                offset += gs
                draft_offset += num_masks

        if request_verify_start is not None:
            self._record_verification_time(
                forward_batch,
                verify_bids,
                (time.perf_counter() - request_verify_start) * 1000.0,
            )

        if self._timing_enabled:
            _t_phase3_end = time.perf_counter()

        # ── Step 3: Batched KV trim index lookup (decode only) ─────
        trim_counts = {}
        free_counts = {}
        advances = {}
        page_size = getattr(model_runner.server_args, "page_size", 1)
        for bid in decode_bids:
            if bid in reject_at:
                si = reject_at[bid]
                adv = 1 + si
            elif case_types[bid] == 'V':
                adv = 1 + len(old_specs[bid])
            else:
                adv = 1
            advances[bid] = adv
            trim_counts[bid] = blk - adv
            if page_size > 1:
                prefix_len = seq_lens_cpu[bid] - blk
                aligned_keep_len = (
                    (prefix_len + adv + page_size - 1) // page_size
                ) * page_size
                keep_slots = min(max(aligned_keep_len - prefix_len, 0), blk)
                free_counts[bid] = blk - keep_slots
            else:
                free_counts[bid] = trim_counts[bid]

        # Vectorized KV trim index computation (avoid per-element append)
        total_trim = sum(free_counts[bid] for bid in decode_bids)
        if total_trim > 0:
            all_trim_rpx = [0] * total_trim
            all_trim_pos = [0] * total_trim
            _off = 0
            for bid in decode_bids:
                rpx = req_pool_indices_cpu[bid]
                sl = seq_lens_cpu[bid]
                tc = free_counts[bid]
                for t in range(tc):
                    all_trim_rpx[_off] = rpx
                    all_trim_pos[_off] = sl - 1 - t
                    _off += 1
            all_kv_indices = req_to_token[all_trim_rpx, all_trim_pos]
        else:
            all_kv_indices = None
        if self._timing_enabled:
            _t_phase4_trim_lookup = time.perf_counter()

        # ── Step 4: Assemble outputs ──────────────────────────────
        next_token_ids_list = []
        committed_token_ids_by_bid = [[] for _ in range(batch_size)]
        output_token_modes = [[] for _ in range(batch_size)]
        kv_offset = 0
        trace_records: List[Dict[str, Any]] = []

        # Batched gather of logits to save for decode requests
        _logit_save_indices = []
        _logit_save_bids = []
        for bid in decode_bids:
            base = base_offsets[bid]
            if bid in reject_at:
                _logit_save_indices.append(base + reject_at[bid])
            elif case_types[bid] == 'V':
                _logit_save_indices.append(base + len(old_specs[bid]))
            else:
                _logit_save_indices.append(base)
            _logit_save_bids.append(bid)

        _saved_logits = {}
        _saved_argmax = {}  # bid -> argmax token (greedy shortcut)
        if _logit_save_indices:
            _idx_t = torch.tensor(_logit_save_indices, dtype=torch.long, device=device)
            if self.temperature <= 0:
                # Greedy shortcut: just compute argmax, skip expensive clone
                if full_argmax is not None:
                    _all_argmax = full_argmax[_idx_t].tolist()
                else:
                    _all_argmax = full_logits[_idx_t].argmax(dim=-1).tolist()
                for k, bid in enumerate(_logit_save_bids):
                    _saved_argmax[bid] = _all_argmax[k]
            else:
                if full_logits is not None:
                    _all_saved = full_logits[_idx_t].detach().clone()
                    for k, bid in enumerate(_logit_save_bids):
                        _saved_logits[bid] = _all_saved[k]
        if self._timing_enabled:
            _t_phase4_save_logits = time.perf_counter()

        for bid in range(batch_size):
            rpx = req_pool_indices_cpu[bid]

            # Prefill requests: no output tokens, handled above
            if is_prefill[bid]:
                next_token_ids_list.append([])
                continue

            tc = trim_counts[bid]
            free_tc = free_counts[bid]
            adv = advances[bid]

            if bid in reject_at:
                si = reject_at[bid]
                specs = old_specs[bid]
                ct = corrected[bid]
                # Output correction: replace accepted specs before rejection with clean argmax
                if bid in spec_corrections:
                    corr = spec_corrections[bid]
                    out_pre = [corr[j] if j < len(corr) else specs[j] for j in range(si)]
                else:
                    out_pre = list(specs[:si])
                output_tokens = out_pre + [ct]
                committed_token_ids_by_bid[bid] = [t0_tokens[bid]] + list(
                    specs[:si]
                )
                output_token_modes[bid] = [False] * len(out_pre) + [True]
                dllm_tokens = [t0_tokens[bid]] + out_pre + [ct]
                dllm_tokens += [self.mask_id] * (blk - len(dllm_tokens))
                self._force_next_token[rpx] = ct
                # Reject path: force_next_token handles t0, but still need logits
                # for verify round after the forced cold start
                if bid in _saved_argmax:
                    self._prev_last_argmax[rpx] = _saved_argmax[bid]
                elif bid in _saved_logits:
                    self._prev_last_logits[rpx] = _saved_logits[bid]
                self._stats["reject_count"] += 1
                self._stats["verify_decisions"] += 1
                self._stats["accepted_spec_tokens"] += si
                if si < len(self._stats["accept_token_hist"]):
                    self._stats["accept_token_hist"][si] += 1
                if self._trace_path:
                    for out_pos, tok in enumerate(output_tokens):
                        accepted = out_pos < si
                        trace_records.append({
                            "req_pool_idx": rpx,
                            "forward_id": self._stats["total_forwards"],
                            "batch_idx": bid,
                            "case_type": case_types[bid],
                            "source": "accepted_spec" if accepted else "correction",
                            "output_pos": out_pos,
                            "token_id": tok,
                            "logit_row": base_offsets[bid] + min(out_pos, si),
                            "seq_len": seq_lens_cpu[bid],
                            "extend_len": extend_lens_cpu[bid],
                            "base_offset": base_offsets[bid],
                            "accepted": accepted,
                            "reject_at": si,
                            "can_run_graph": bool(out.can_run_graph),
                        })

            elif case_types[bid] == 'V':
                specs = old_specs[bid]
                sr = sampled_results[bid]
                clean_token = sr[0]
                new_spec_tokens = sr[1:]
                # Output correction: replace spec tokens with clean argmax for quality
                if bid in spec_corrections:
                    corr = spec_corrections[bid]
                    # Replace verified specs with their clean corrections
                    out_specs = [corr[si] if si < len(corr) else specs[si]
                                 for si in range(len(specs))]
                else:
                    out_specs = list(specs)
                output_tokens = out_specs + [clean_token]
                committed_token_ids_by_bid[bid] = [t0_tokens[bid]] + list(specs)
                output_token_modes[bid] = [False] * len(out_specs) + [True]
                dllm_tokens = [t0_tokens[bid]] + out_specs + [clean_token]
                dllm_tokens += [self.mask_id] * (blk - len(dllm_tokens))
                self._pending[rpx] = clean_token
                self._spec_tokens[rpx] = new_spec_tokens
                self._spec_draft_probs[rpx] = draft_probs_map.get(bid, [])
                if bid in _saved_argmax:
                    self._prev_last_argmax[rpx] = _saved_argmax[bid]
                elif bid in _saved_logits:
                    self._prev_last_logits[rpx] = _saved_logits[bid]
                self._stats["accept_count"] += 1
                accepted_specs = len(specs)
                self._stats["verify_decisions"] += 1
                self._stats["accepted_spec_tokens"] += accepted_specs
                if accepted_specs < len(self._stats["accept_token_hist"]):
                    self._stats["accept_token_hist"][accepted_specs] += 1
                if self._trace_path:
                    for out_pos, tok in enumerate(output_tokens):
                        is_clean = out_pos >= len(out_specs)
                        trace_records.append({
                            "req_pool_idx": rpx,
                            "forward_id": self._stats["total_forwards"],
                            "batch_idx": bid,
                            "case_type": case_types[bid],
                            "source": "clean_sample" if is_clean else "accepted_spec",
                            "output_pos": out_pos,
                            "token_id": tok,
                            "logit_row": (
                                base_offsets[bid] + len(old_specs[bid])
                                if is_clean
                                else base_offsets[bid] + out_pos
                            ),
                            "seq_len": seq_lens_cpu[bid],
                            "extend_len": extend_lens_cpu[bid],
                            "base_offset": base_offsets[bid],
                            "accepted": True,
                            "reject_at": None,
                            "can_run_graph": bool(out.can_run_graph),
                        })

            else:
                t0 = t0_tokens[bid]
                sr = sampled_results[bid]
                clean_token = sr[0]
                new_spec_tokens = sr[1:]
                if was_forced[bid]:
                    output_tokens = [clean_token]
                else:
                    output_tokens = [t0, clean_token]
                committed_token_ids_by_bid[bid] = [t0]
                output_token_modes[bid] = [True] * len(output_tokens)
                dllm_tokens = [t0, clean_token] + new_spec_tokens
                dllm_tokens += [self.mask_id] * (blk - len(dllm_tokens))
                self._pending[rpx] = clean_token
                self._spec_tokens[rpx] = new_spec_tokens
                self._spec_draft_probs[rpx] = draft_probs_map.get(bid, [])
                if bid in _saved_argmax:
                    self._prev_last_argmax[rpx] = _saved_argmax[bid]
                elif bid in _saved_logits:
                    self._prev_last_logits[rpx] = _saved_logits[bid]
                if self._trace_path:
                    for out_pos, tok in enumerate(output_tokens):
                        is_clean = was_forced[bid] or out_pos == len(output_tokens) - 1
                        trace_records.append({
                            "req_pool_idx": rpx,
                            "forward_id": self._stats["total_forwards"],
                            "batch_idx": bid,
                            "case_type": case_types[bid],
                            "source": "clean_sample" if is_clean else "t0",
                            "output_pos": out_pos,
                            "token_id": tok,
                            "logit_row": base_offsets[bid] if is_clean else None,
                            "seq_len": seq_lens_cpu[bid],
                            "extend_len": extend_lens_cpu[bid],
                            "base_offset": base_offsets[bid],
                            "accepted": None,
                            "reject_at": None,
                            "can_run_graph": bool(out.can_run_graph),
                        })

            next_token_ids_list.append(output_tokens)
            self._dllm_write_override[rpx] = dllm_tokens
            self._advance_override[rpx] = adv
            self._kv_trim_info[rpx] = {
                "kv_indices_gpu": (
                    all_kv_indices[kv_offset:kv_offset + free_tc]
                    if all_kv_indices is not None and free_tc > 0
                    else None
                ),
                "trim_count": tc,
                "logical_trim_count": trim_counts[bid],
                "physical_free_count": free_tc,
            }
            self._record_invalidation(
                forward_batch,
                batch_index=bid,
                start=seq_lens_cpu[bid] - tc,
                length=tc,
            )
            kv_offset += free_tc
        self._set_output_token_modes(
            forward_batch,
            token_is_ar=output_token_modes,
        )
        self._snapshot_hybrid_boundaries(
            model_runner,
            forward_batch,
            prefill_bids,
            req_pool_indices_cpu,
        )
        self._write_trace_records(trace_records, full_logits, device)
        if self._timing_enabled:
            _t_phase4_output_assembled = time.perf_counter()

        # ── GDN state commit for accepted tokens ──
        gdn_dllm_backend = self._get_gdn_dllm_backend(model_runner)
        if gdn_dllm_backend is not None:
            use_graph_saved = bool(
                out.can_run_graph
                and getattr(forward_batch, "dllm_gdn_use_graph_save_buffer", False)
            )
            if use_intermediate_commit:
                commit_advances = [advances[bid] for bid in decode_bids]
                track_indices_cpu = getattr(
                    forward_batch, "dllm_mamba_track_indices_cpu", None
                )
                track_steps_cpu = getattr(
                    forward_batch, "dllm_mamba_track_steps_cpu", None
                )
                track_boundaries_cpu = getattr(
                    forward_batch, "dllm_mamba_track_boundaries_cpu", None
                )
                mamba_track_indices = None
                mamba_steps_to_track = None
                if track_steps_cpu is not None and track_indices_cpu is not None:
                    track_indices_out = []
                    track_steps_out = []
                    has_track_commit = False
                    for bid in decode_bids:
                        step = track_steps_cpu[bid]
                        boundary = (
                            track_boundaries_cpu[bid]
                            if track_boundaries_cpu is not None
                            else -1
                        )
                        if step >= 0 and step < advances[bid]:
                            track_indices_out.append(track_indices_cpu[bid])
                            track_steps_out.append(step)
                            has_track_commit = True
                            self._mamba_track_commit_info[
                                req_pool_indices_cpu[bid]
                            ] = boundary
                            if _EXTRA_BUFFER_TRACE:
                                logger.warning(
                                    "[HYBRID_DIFFUSION_SELF_SPEC_EXTRA_BUFFER] event=track_commit "
                                    "req_pool_idx=%s bid=%s advance=%s step=%s boundary=%s "
                                    "track_idx=%s",
                                    req_pool_indices_cpu[bid],
                                    bid,
                                    advances[bid],
                                    step,
                                    boundary,
                                    track_indices_cpu[bid],
                                )
                        else:
                            track_indices_out.append(0)
                            track_steps_out.append(-1)
                            if _EXTRA_BUFFER_TRACE and step >= 0:
                                logger.warning(
                                    "[HYBRID_DIFFUSION_SELF_SPEC_EXTRA_BUFFER] event=track_skip "
                                    "req_pool_idx=%s bid=%s advance=%s step=%s boundary=%s "
                                    "track_idx=%s",
                                    req_pool_indices_cpu[bid],
                                    bid,
                                    advances[bid],
                                    step,
                                    boundary,
                                    track_indices_cpu[bid],
                                )
                    if has_track_commit:
                        mamba_track_indices = track_indices_out
                        mamba_steps_to_track = track_steps_out
                gdn_dllm_backend.commit_cached_intermediate_states_batch(
                    mamba_cache_indices=(
                        gdn_dllm_backend.forward_metadata.mamba_cache_indices[
                            : len(decode_bids)
                        ]
                    ),
                    num_accepted=commit_advances,
                    batch_indices=decode_bids,
                    mamba_track_indices=mamba_track_indices,
                    mamba_steps_to_track=mamba_steps_to_track,
                )
            else:
                mamba_cache_indices_cpu = (
                    gdn_dllm_backend.forward_metadata.mamba_cache_indices.detach()
                    .cpu()
                    .tolist()
                )
                if use_graph_saved:
                    commit_cache_indices = [
                        mamba_cache_indices_cpu[bid] for bid in decode_bids
                    ]
                    commit_advances = [advances[bid] for bid in decode_bids]
                    commit_query_starts = [base_offsets[bid] for bid in decode_bids]
                    for layer_id in gdn_dllm_backend.gdn_layer_ids:
                        layer = self._get_gdn_layer(model_runner, layer_id)
                        if layer is not None:
                            gdn_dllm_backend.commit_accepted_tokens_batch(
                                layer=layer,
                                mamba_cache_indices=commit_cache_indices,
                                num_accepted=commit_advances,
                                query_starts=commit_query_starts,
                                batch_indices=decode_bids,
                                use_graph_saved=True,
                            )
                else:
                    for bid in decode_bids:
                        mamba_cache_idx = mamba_cache_indices_cpu[bid]
                        adv = advances[bid]
                        # query_start: offset of this request's tokens in the batch
                        query_start = base_offsets[bid]
                        for layer_id in gdn_dllm_backend.gdn_layer_ids:
                            layer = self._get_gdn_layer(model_runner, layer_id)
                            if layer is not None:
                                gdn_dllm_backend.commit_accepted_tokens(
                                    layer=layer,
                                    mamba_cache_idx=mamba_cache_idx,
                                    num_accepted=adv,
                                    query_start=query_start,
                                    batch_idx=bid,
                                    use_graph_saved=False,
                                )
                    mamba_cache_idx_list = [
                        mamba_cache_indices_cpu[bid] for bid in decode_bids
                    ]
                    gdn_dllm_backend.discard_saved_batch(mamba_cache_idx_list)
        self._commit_hybrid_boundaries(
            model_runner,
            forward_batch,
            req_pool_indices_cpu,
            decode_bids,
            advances,
            committed_token_ids_by_bid,
        )
        if self._timing_enabled:
            _t_phase4_gdn_commit = time.perf_counter()

        # Debug logging for single request
        if self._debug_steps and batch_size == 1 and not is_prefill[0]:
            rpx0 = req_pool_indices_cpu[0]
            ct = case_types[0]
            out_toks = next_token_ids_list[0] if next_token_ids_list else []
            adv = self._advance_override.get(rpx0, '?')
            tc = self._kv_trim_info.get(rpx0, {}).get('trim_count', '?')
            n_pend = self._pending.get(rpx0, '∅')
            n_specs = self._spec_tokens.get(rpx0, [])
            logger.info(
                f"[STEP] {ct} t0={t0_tokens[0]} "
                f"→ out={out_toks} adv={adv} trim={tc} "
                f"next: pend={n_pend} specs={n_specs[:3]}..."
            )

        # Stats + optional timing
        self._stats["total_forwards"] += 1
        self._stats["decode_forwards"] += 1
        self._stats["total_tokens"] += sum(len(t) for t in next_token_ids_list)
        if self._timing_enabled and has_any_decode:
            _t_run_end = time.perf_counter()
            self._timing["phase1_classify"] += (_t_phase1_end - _t_run_start)
            self._timing["phase2_forward"] += (_t_phase2_end - _t_phase1_end)
            self._timing["phase3_verify_sample"] += (_t_phase3_end - _t_phase2_end)
            self._timing["phase4_trim_assemble"] += (_t_run_end - _t_phase3_end)
            self._timing.setdefault("phase4_trim_lookup", 0.0)
            self._timing.setdefault("phase4_save_logits", 0.0)
            self._timing.setdefault("phase4_output_assemble", 0.0)
            self._timing.setdefault("phase4_gdn_commit", 0.0)
            self._timing["phase4_trim_lookup"] += (
                _t_phase4_trim_lookup - _t_phase3_end
            )
            self._timing["phase4_save_logits"] += (
                _t_phase4_save_logits - _t_phase4_trim_lookup
            )
            self._timing["phase4_output_assemble"] += (
                _t_phase4_output_assembled - _t_phase4_save_logits
            )
            self._timing["phase4_gdn_commit"] += (
                _t_phase4_gdn_commit - _t_phase4_output_assembled
            )
            # Micro-timing for verify_sample breakdown
            self._timing.setdefault("vs_verify_gpu", 0.0)
            self._timing.setdefault("vs_sample_gpu", 0.0)
            self._timing.setdefault("vs_sync", 0.0)
            self._timing["vs_verify_gpu"] += (_t_vs_after_verify - _t_vs_start)
            self._timing["vs_sample_gpu"] += (_t_vs_after_sample - _t_vs_after_verify)
            self._timing["vs_sync"] += (_t_vs_after_sync - _t_vs_after_sample)
            self._timing["timing_count"] += 1
        if self._stats["total_forwards"] % self._timing_log_interval == 0:
            s = self._stats
            n = s["total_forwards"]
            tok_per_fwd = s["total_tokens"] / max(n, 1)
            total_decisions = s["accept_count"] + s["reject_count"]
            accept_rate = (
                s["accept_count"] / total_decisions * 100
                if total_decisions > 0
                else 0
            )
            verify_decisions = s.get("verify_decisions", 0)
            avg_accept_tokens = (
                s.get("accepted_spec_tokens", 0) / verify_decisions
                if verify_decisions > 0
                else 0.0
            )
            accept_hist = s.get("accept_token_hist", [])
            hist_str = ",".join(
                f"{i}:{count}" for i, count in enumerate(accept_hist)
            )
            timing_str = ""
            if self._timing_enabled:
                t = self._timing
                tc = max(t["timing_count"], 1)
                vs_detail = ""
                if t.get("vs_verify_gpu"):
                    vs_detail = (
                        f" [verify_gpu={t['vs_verify_gpu']/tc*1000:.2f}"
                        f" sample_gpu={t['vs_sample_gpu']/tc*1000:.2f}"
                        f" sync={t['vs_sync']/tc*1000:.2f}]"
                    )
                forward_detail = ""
                phase4_detail = ""
                if t.get("phase4_trim_lookup"):
                    phase4_detail = (
                        f" [trim_lookup={t['phase4_trim_lookup']/tc*1000:.2f}"
                        f" save_logits={t['phase4_save_logits']/tc*1000:.2f}"
                        f" output={t['phase4_output_assemble']/tc*1000:.2f}"
                        f" gdn_commit={t['phase4_gdn_commit']/tc*1000:.2f}]"
                    )
                timing_str = (
                    f" timing(ms): classify={t['phase1_classify']/tc*1000:.2f} "
                    f"forward={t['phase2_forward']/tc*1000:.2f}{forward_detail} "
                    f"verify_sample={t['phase3_verify_sample']/tc*1000:.2f}{vs_detail} "
                    f"trim_assemble={t['phase4_trim_assemble']/tc*1000:.2f}"
                    f"{phase4_detail}"
                )
            row_str = ""
            if s.get("logits_rows_total", 0):
                row_den = max(
                    s.get("logits_mode_full", 0)
                    + s.get("logits_mode_topk", 0)
                    + s.get("logits_mode_argmax", 0)
                    + s.get("logits_mode_unknown", 0),
                    1,
                )
                row_str = (
                    f", logits_rows=total:{s['logits_rows_total']/row_den:.1f}"
                    f"/consumed:{s['logits_rows_consumed']/row_den:.1f}"
                    f"/verify:{s['logits_rows_verify']/row_den:.1f}"
                    f"/sample:{s['logits_rows_sample']/row_den:.1f}"
                )
            logger.info(
                f"[HybridDiffusionSelfSpec] N={self.gen_block_size}, fwd={n}, bs={batch_size}, "
                f"tok/fwd={tok_per_fwd:.2f}, accept={accept_rate:.1f}%, "
                f"accept_hist={hist_str}, avg_accept_tokens={avg_accept_tokens:.2f}, "
                f"avg_verify_emit_tokens={avg_accept_tokens + 1:.2f}, "
                f"fused_verify={s.get('fused_verify_calls', 0)}, "
                f"torch_verify={s.get('torch_verify_calls', 0)}, "
                f"draft_sample={s.get('draft_sample_batches', 0)}, "
                f"draft_argmax={s.get('draft_argmax_batches', 0)}, "
                f"logits_mode=full:{s.get('logits_mode_full', 0)},"
                f"topk:{s.get('logits_mode_topk', 0)},"
                f"topk_dense:{s.get('logits_mode_topk_dense', 0)},"
                f"topk_tiled:{s.get('logits_mode_topk_tiled', 0)},"
                f"argmax:{s.get('logits_mode_argmax', 0)},"
                f"unknown:{s.get('logits_mode_unknown', 0)}{row_str}{timing_str}"
            )

        self._flush_forward_timings()
        return logits_output, next_token_ids_list, out.can_run_graph


Algorithm = HybridDiffusionSelfSpec
