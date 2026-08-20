"""GDN (Gated Delta Net) backend for dLLM block diffusion inference.

Wraps GDNAttnBackend with dLLM-specific state management:
- persist_state control (don't write dirty S/conv during iterative denoising)
- causal_mode control (token-causal, block-causal, or mixed readout)
- projection saving + commit for HybridDiffusion self-speculative decoding

Reads dLLM signals from forward_batch dynamic attributes:
- dllm_gdn_persist_state (bool): write S/conv to MambaPool after forward
- dllm_gdn_causal_mode (int): 0=block-causal, 1=token-causal, 2=mixed
- dllm_gdn_num_clean (int): for causal_mode=2, how many clean positions
- dllm_gdn_save_for_commit (bool): save projections for self-spec commit
"""

from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch

from sglang.srt.dllm.config import (
    DLLM_ATTN_MASK_BIDIR_BLOCK,
    DLLM_ATTN_MASK_CAUSAL_PREFILL,
)
from sglang.srt.layers.attention.linear.gdn_backend import (
    GDNAttnBackend,
    fused_gdn_gating,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import (
    fused_mamba_state_scatter_with_mask,
)
from sglang.srt.layers.attention.block_gdn import (
    fused_recurrent_block_causal_gated_delta_rule,
    fused_recurrent_block_causal_gated_delta_rule_packed,
)
from sglang.srt.layers.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)
from sglang.srt.mem_cache.region_state_cache import (
    KVPrefixReference,
    RegionState,
    RegionStateCache,
    RegionStateKey,
    RegionStateLookup,
    RegionStateMissReason,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class GDNDllmBackend:
    """dLLM-aware wrapper for GDN linear attention.

    Composition over GDNAttnBackend: reuses its conv1d, MambaPool access,
    and metadata init. Replaces the kernel dispatch for dLLM forwards.
    """

    def __init__(self, gdn_backend: GDNAttnBackend, model_config):
        self.gdn_backend = gdn_backend
        self.req_to_token_pool = gdn_backend.req_to_token_pool

        # Determine which layers are GDN (complement of full_attention_layer_ids)
        full_attn_ids = set(getattr(model_config, 'full_attention_layer_ids', []))
        num_layers = getattr(model_config, 'num_hidden_layers', 0)
        if hasattr(model_config, 'get_text_config'):
            text_cfg = model_config.get_text_config()
            num_layers = getattr(text_cfg, 'num_hidden_layers', num_layers)
        self.gdn_layer_ids = [i for i in range(num_layers) if i not in full_attn_ids]

        # Per-request saved projections for HybridDiffusion self-spec commit
        # {req_pool_idx: {layer_id: (pre_conv_mixed_qkv, a, b, conv_snapshot)}}
        self._saved: Dict[int, Dict[int, Tuple]] = {}
        self._graph_save_max_bs = 0
        self._graph_save_max_num_tokens = 0
        self._graph_saved: Dict[int, Dict[str, torch.Tensor]] = {}
        self._block_query_start_cache: Dict[
            Tuple[torch.device, int, int, torch.dtype], torch.Tensor
        ] = {}
        self.region_state_cache = RegionStateCache(max_entries=128)
        self.strict_region_state_validation = True

    def configure_region_state_cache(
        self, *, max_entries: int, strict_validation: bool
    ) -> None:
        if self.region_state_cache.max_entries != int(max_entries):
            if len(self.region_state_cache):
                raise RuntimeError("cannot resize a nonempty region-state cache")
            self.region_state_cache = RegionStateCache(max_entries=int(max_entries))
        self.strict_region_state_validation = bool(strict_validation)

    # ------------------------------------------------------------------ #
    #  Delegate standard methods to wrapped backend                       #
    # ------------------------------------------------------------------ #

    def init_forward_metadata(self, forward_batch):
        self.gdn_backend.init_forward_metadata(forward_batch)

    @property
    def forward_metadata(self):
        return self.gdn_backend.forward_metadata

    def _get_block_query_start_loc(
        self,
        device: torch.device,
        batch_size: int,
        block_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (device, batch_size, block_size, dtype)
        cached = self._block_query_start_cache.get(key)
        if cached is None:
            cached = torch.arange(
                0,
                (batch_size + 1) * block_size,
                block_size,
                dtype=dtype,
                device=device,
            )
            self._block_query_start_cache[key] = cached
        return cached

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.gdn_backend.init_cuda_graph_state(max_bs, max_num_tokens)
        self._graph_save_max_bs = max_bs
        self._graph_save_max_num_tokens = max_num_tokens

    def _get_graph_save_buffers(
        self,
        layer_id: int,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        max_num_tokens = max(self._graph_save_max_num_tokens, mixed_qkv.shape[0])
        max_bs = max(self._graph_save_max_bs, 1)
        expected = {
            "mixed_qkv": (max_num_tokens, *mixed_qkv.shape[1:]),
            "a": (max_num_tokens, *a.shape[1:]),
            "b": (max_num_tokens, *b.shape[1:]),
            "conv_snapshot": (max_bs, *conv_states.shape[1:]),
        }

        buffers = self._graph_saved.get(layer_id)
        needs_alloc = buffers is None
        if buffers is not None:
            for name, shape in expected.items():
                buf = buffers[name]
                src = {
                    "mixed_qkv": mixed_qkv,
                    "a": a,
                    "b": b,
                    "conv_snapshot": conv_states,
                }[name]
                if (
                    tuple(buf.shape) != shape
                    or buf.device != src.device
                    or buf.dtype != src.dtype
                ):
                    needs_alloc = True
                    break

        if needs_alloc:
            buffers = {
                "mixed_qkv": torch.empty(
                    expected["mixed_qkv"],
                    device=mixed_qkv.device,
                    dtype=mixed_qkv.dtype,
                ),
                "a": torch.empty(expected["a"], device=a.device, dtype=a.dtype),
                "b": torch.empty(expected["b"], device=b.device, dtype=b.dtype),
                "conv_snapshot": torch.empty(
                    expected["conv_snapshot"],
                    device=conv_states.device,
                    dtype=conv_states.dtype,
                ),
            }
            self._graph_saved[layer_id] = buffers

        return buffers

    def _save_graph_commit_tensors(
        self,
        layer_id: int,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_states: torch.Tensor,
        cache_indices: torch.Tensor,
    ) -> None:
        buffers = self._get_graph_save_buffers(layer_id, mixed_qkv, a, b, conv_states)
        num_tokens = mixed_qkv.shape[0]
        batch_size = cache_indices.shape[0]
        buffers["mixed_qkv"][:num_tokens].copy_(mixed_qkv)
        buffers["a"][:num_tokens].copy_(a)
        buffers["b"][:num_tokens].copy_(b)
        buffers["conv_snapshot"][:batch_size].copy_(conv_states[cache_indices])

    def init_cpu_graph_state(self, *args, **kwargs):
        self.gdn_backend.init_cpu_graph_state(*args, **kwargs)

    def init_forward_metadata_capture_cuda_graph(self, *args, **kwargs):
        self.gdn_backend.init_forward_metadata_capture_cuda_graph(*args, **kwargs)

    def init_forward_metadata_capture_cpu_graph(self, *args, **kwargs):
        self.gdn_backend.init_forward_metadata_capture_cpu_graph(*args, **kwargs)

    def init_forward_metadata_replay_cuda_graph(self, *args, **kwargs):
        self.gdn_backend.init_forward_metadata_replay_cuda_graph(*args, **kwargs)

    def get_cuda_graph_seq_len_fill_value(self):
        return self.gdn_backend.get_cuda_graph_seq_len_fill_value()

    def get_cpu_graph_seq_len_fill_value(self):
        return self.gdn_backend.get_cpu_graph_seq_len_fill_value()

    # ------------------------------------------------------------------ #
    #  AR decode: delegate unchanged                                      #
    # ------------------------------------------------------------------ #

    def forward_decode(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
        return self.gdn_backend.forward_decode(
            layer=layer, forward_batch=forward_batch,
            mixed_qkv=mixed_qkv, a=a, b=b, **kwargs,
        )

    @staticmethod
    def _token_ranges(seq_lens_cpu: List[int]) -> List[Tuple[int, int]]:
        ranges = []
        offset = 0
        for seq_len in seq_lens_cpu:
            end = offset + int(seq_len)
            ranges.append((offset, end))
            offset = end
        return ranges

    def _subset_forward_batch(
        self,
        forward_batch: "ForwardBatch",
        req_indices: List[int],
        token_indices: torch.Tensor,
        extend_lens_cpu: List[int],
        prefix_lens_cpu: List[int],
    ) -> "ForwardBatch":
        subset = copy.copy(forward_batch)
        device = forward_batch.input_ids.device
        req_index_tensor = torch.tensor(req_indices, dtype=torch.long, device=device)

        subset.batch_size = len(req_indices)
        subset.input_ids = forward_batch.input_ids[token_indices]
        subset.req_pool_indices = forward_batch.req_pool_indices[req_index_tensor]
        subset.seq_lens = forward_batch.seq_lens[req_index_tensor]
        seq_lens_cpu = [
            int(prefix_len) + int(extend_len)
            for prefix_len, extend_len in zip(prefix_lens_cpu, extend_lens_cpu)
        ]
        subset.seq_lens_cpu = torch.tensor(seq_lens_cpu, dtype=torch.int64)
        subset.seq_lens_sum = int(sum(seq_lens_cpu))
        subset.extend_num_tokens = int(sum(extend_lens_cpu))
        subset.extend_seq_lens = torch.tensor(
            extend_lens_cpu, dtype=torch.int32, device=device
        )
        subset.extend_prefix_lens = torch.tensor(
            prefix_lens_cpu, dtype=torch.int32, device=device
        )
        subset.extend_start_loc = torch.empty(
            len(req_indices), dtype=torch.int32, device=device
        )
        if len(req_indices) > 0:
            subset.extend_start_loc[0] = 0
        if len(req_indices) > 1:
            subset.extend_start_loc[1:] = torch.cumsum(
                subset.extend_seq_lens[:-1], dim=0
            )
        subset.extend_seq_lens_cpu = extend_lens_cpu
        subset.extend_prefix_lens_cpu = prefix_lens_cpu
        subset.dllm_attn_mask_types_cpu = [
            forward_batch.dllm_attn_mask_types_cpu[i] for i in req_indices
        ]
        subset.dllm_attn_mask_types = (
            forward_batch.dllm_attn_mask_types[req_index_tensor]
            if getattr(forward_batch, "dllm_attn_mask_types", None) is not None
            else None
        )
        if forward_batch.mamba_track_mask is not None:
            subset.mamba_track_mask = forward_batch.mamba_track_mask[req_index_tensor]
            subset.mamba_track_indices = forward_batch.mamba_track_indices[
                req_index_tensor
            ]
            subset.mamba_track_seqlens = forward_batch.mamba_track_seqlens[
                req_index_tensor
            ]
        return subset

    def _forward_mixed_dllm_extend(
        self,
        layer: "RadixLinearAttention",
        forward_batch: "ForwardBatch",
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        causal_mode: int,
        num_clean: int,
        save_for_commit: bool,
        block_size: int,
    ) -> torch.Tensor:
        """Run dLLM mixed inline prefill without sending prefill through decode kernels."""
        forward_metadata = self.gdn_backend.forward_metadata
        cache_indices = forward_metadata.mamba_cache_indices
        mask_types = forward_batch.dllm_attn_mask_types_cpu
        seq_lens_cpu = list(forward_batch.extend_seq_lens_cpu)
        prefix_lens_cpu = [
            int(x) for x in forward_batch.extend_prefix_lens_cpu
        ]
        expected_num_tokens = sum(seq_lens_cpu)
        actual_num_tokens = int(forward_batch.input_ids.shape[0])
        if expected_num_tokens != actual_num_tokens:
            raise RuntimeError(
                "Mixed dLLM batch metadata does not match flattened input_ids: "
                f"expected_num_tokens={expected_num_tokens}, "
                f"actual_num_tokens={actual_num_tokens}, "
                f"extend_seq_lens_cpu={seq_lens_cpu}, "
                f"prefix_lens_cpu={prefix_lens_cpu}, "
                f"mask_types={mask_types}"
            )
        ranges = self._token_ranges(seq_lens_cpu)

        prefill_bids = [
            i
            for i, mask_type in enumerate(mask_types)
            if mask_type == DLLM_ATTN_MASK_CAUSAL_PREFILL
        ]
        decode_bids = [
            i
            for i, mask_type in enumerate(mask_types)
            if mask_type == DLLM_ATTN_MASK_BIDIR_BLOCK
        ]

        outputs_by_bid: Dict[int, torch.Tensor] = {}

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal

        if prefill_bids:
            prefill_token_indices = torch.cat(
                [
                    torch.arange(start, end, device=mixed_qkv.device)
                    for start, end in (ranges[i] for i in prefill_bids)
                ]
            )
            prefill_fb = self._subset_forward_batch(
                forward_batch,
                prefill_bids,
                prefill_token_indices,
                [seq_lens_cpu[i] for i in prefill_bids],
                [prefix_lens_cpu[i] for i in prefill_bids],
            )
            old_metadata = self.gdn_backend.forward_metadata
            try:
                self.gdn_backend.init_forward_metadata(prefill_fb)
                prefill_out = self.gdn_backend.forward_extend(
                    layer=layer,
                    forward_batch=prefill_fb,
                    mixed_qkv=mixed_qkv[prefill_token_indices],
                    a=a[prefill_token_indices],
                    b=b[prefill_token_indices],
                )
            finally:
                self.gdn_backend.forward_metadata = old_metadata

            offset = 0
            for bid in prefill_bids:
                seg_len = int(seq_lens_cpu[bid])
                outputs_by_bid[bid] = prefill_out[:, offset : offset + seg_len]
                offset += seg_len

        if decode_bids:
            decode_token_indices = torch.cat(
                [
                    torch.arange(start, end, device=mixed_qkv.device)
                    for start, end in (ranges[i] for i in decode_bids)
                ]
            )
            decode_lens_cpu = [seq_lens_cpu[i] for i in decode_bids]
            decode_cache_indices = cache_indices[
                torch.tensor(decode_bids, dtype=torch.long, device=cache_indices.device)
            ]
            decode_has_initial = forward_batch.extend_prefix_lens[
                torch.tensor(
                    decode_bids,
                    dtype=torch.long,
                    device=forward_batch.extend_prefix_lens.device,
                )
            ] > 0
            decode_query_start = torch.empty(
                len(decode_bids) + 1, dtype=torch.int32, device=mixed_qkv.device
            )
            decode_query_start[0] = 0
            if decode_lens_cpu:
                decode_query_start[1:] = torch.cumsum(
                    torch.tensor(
                        decode_lens_cpu, dtype=torch.int32, device=mixed_qkv.device
                    ),
                    dim=0,
                )

            if save_for_commit:
                pre_conv_mixed_qkv = mixed_qkv.clone()
                a_saved = a.clone()
                b_saved = b.clone()
                conv_snapshots = conv_states[cache_indices].clone()

            conv_state_backup = conv_states[decode_cache_indices].clone()
            decode_mixed = causal_conv1d_fn(
                mixed_qkv[decode_token_indices].transpose(0, 1),
                layer.conv_weights,
                layer.bias,
                activation=layer.activation,
                conv_states=conv_states,
                has_initial_state=decode_has_initial,
                cache_indices=decode_cache_indices,
                query_start_loc=decode_query_start,
                seq_lens_cpu=torch.tensor(decode_lens_cpu),
            ).transpose(0, 1)[: len(decode_token_indices)]
            conv_states[decode_cache_indices] = conv_state_backup

            query, key, value = torch.split(
                decode_mixed, [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1
            )
            actual_seq_len = query.shape[0]
            query = query.view(1, actual_seq_len, layer.num_q_heads, layer.head_q_dim)
            key = key.view(1, actual_seq_len, layer.num_k_heads, layer.head_k_dim)
            value = value.view(1, actual_seq_len, layer.num_v_heads, layer.head_v_dim)
            g, beta_val = fused_gdn_gating(
                layer.A_log, a[decode_token_indices], b[decode_token_indices], layer.dt_bias
            )

            if layer.num_v_heads // layer.num_k_heads > 1:
                repeat_factor = layer.num_v_heads // layer.num_k_heads
                query = query.repeat_interleave(repeat_factor, dim=2)
                key = key.repeat_interleave(repeat_factor, dim=2)

            initial_state = ssm_states[decode_cache_indices] if decode_has_initial.any() else None
            if causal_mode == 1:
                decode_out, _ = fused_recurrent_gated_delta_rule(
                    q=query,
                    k=key,
                    v=value,
                    g=g,
                    beta=beta_val,
                    initial_state=initial_state,
                    output_final_state=False,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=decode_query_start.to(dtype=torch.long),
                )
            else:
                decode_out, _ = fused_recurrent_block_causal_gated_delta_rule(
                    q=query,
                    k=key,
                    v=value,
                    g=g,
                    beta=beta_val,
                    block_size=block_size,
                    causal_mode=causal_mode,
                    num_clean=num_clean,
                    initial_state=initial_state,
                    output_final_state=False,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=decode_query_start.to(dtype=torch.long),
                )

            offset = 0
            for bid in decode_bids:
                seg_len = int(seq_lens_cpu[bid])
                outputs_by_bid[bid] = decode_out[:, offset : offset + seg_len]
                offset += seg_len
                if save_for_commit:
                    rpx = int(cache_indices[bid].item())
                    if rpx not in self._saved:
                        self._saved[rpx] = {}
                    self._saved[rpx][layer.layer_id] = (
                        pre_conv_mixed_qkv,
                        a_saved,
                        b_saved,
                        conv_snapshots[bid],
                        bid,
                    )

        return torch.cat([outputs_by_bid[i] for i in range(len(seq_lens_cpu))], dim=1)

    # ------------------------------------------------------------------ #
    #  Extend: dLLM-aware forward                                         #
    # ------------------------------------------------------------------ #

    def forward_extend(
        self,
        layer: "RadixLinearAttention",
        forward_batch: "ForwardBatch",
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # Check if this is a dLLM forward (flags set by the algorithm)
        persist_state = getattr(forward_batch, 'dllm_gdn_persist_state', None)
        if persist_state is None:
            # Not a dLLM forward — delegate to original backend
            return self.gdn_backend.forward_extend(
                layer=layer, forward_batch=forward_batch,
                mixed_qkv=mixed_qkv, a=a, b=b, **kwargs,
            )

        causal_mode = getattr(forward_batch, 'dllm_gdn_causal_mode', 1)
        num_clean = getattr(forward_batch, 'dllm_gdn_num_clean', 0)
        save_for_commit = getattr(forward_batch, 'dllm_gdn_save_for_commit', False)
        use_graph_save_buffer = getattr(
            forward_batch, "dllm_gdn_use_graph_save_buffer", False
        )
        cache_intermediate_for_commit = getattr(
            forward_batch, "dllm_gdn_cache_intermediate_for_commit", False
        )
        block_size = getattr(forward_batch, 'dllm_gdn_block_size', 4)
        assert isinstance(mixed_qkv, torch.Tensor)
        mask_types = getattr(forward_batch, "dllm_attn_mask_types_cpu", None)
        if mask_types is not None and any(
            mask_type == DLLM_ATTN_MASK_CAUSAL_PREFILL for mask_type in mask_types
        ):
            return self._forward_mixed_dllm_extend(
                layer,
                forward_batch,
                mixed_qkv,
                a,
                b,
                causal_mode,
                num_clean,
                save_for_commit,
                block_size,
            )

        seq_len = mixed_qkv.shape[0]

        forward_metadata = self.gdn_backend.forward_metadata
        cache_indices = forward_metadata.mamba_cache_indices
        query_start_loc = forward_metadata.query_start_loc
        has_initial_states = forward_batch.extend_prefix_lens > 0

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal

        # ── Save projections for HybridDiffusion self-spec commit ──
        if save_for_commit and not cache_intermediate_for_commit:
            if use_graph_save_buffer:
                self._save_graph_commit_tensors(
                    layer.layer_id, mixed_qkv, a, b, conv_states, cache_indices
                )
            else:
                pre_conv_mixed_qkv = mixed_qkv.clone()
                a_saved = a.clone()
                b_saved = b.clone()
                # Snapshot conv_state for each request
                conv_snapshots = conv_states[cache_indices].clone()

        # ── Conv1d ──
        # Snapshot conv_state BEFORE conv1d modifies it (if not persisting)
        if not persist_state:
            conv_state_backup = conv_states[cache_indices].clone()

        batch_size = int(cache_indices.shape[0])
        if block_size > 0 and seq_len == batch_size * block_size:
            query_start_loc = self._get_block_query_start_loc(
                mixed_qkv.device,
                batch_size,
                block_size,
                dtype=torch.int32,
            )
            query_start_loc_long = self._get_block_query_start_loc(
                mixed_qkv.device,
                batch_size,
                block_size,
                dtype=torch.long,
            )
        else:
            query_start_loc_long = query_start_loc.to(dtype=torch.long)
        use_intermediate_conv = (
            cache_intermediate_for_commit
            and block_size > 0
            and seq_len == batch_size * block_size
            and hasattr(mamba_cache_params, "intermediate_conv_window")
        )
        if use_intermediate_conv:
            intermediate_state_indices = (
                self.gdn_backend.verify_intermediate_state_indices[:batch_size]
            )
            mixed_qkv_post_conv = causal_conv1d_update(
                mixed_qkv.view(batch_size, block_size, -1).transpose(1, 2),
                conv_states,
                layer.conv_weights,
                layer.bias,
                layer.activation,
                conv_state_indices=cache_indices,
                intermediate_conv_window=mamba_cache_params.intermediate_conv_window[0],
                intermediate_state_indices=intermediate_state_indices,
            ).transpose(1, 2).reshape(seq_len, -1)
        else:
            mixed_qkv_transposed = mixed_qkv.transpose(0, 1)
            if forward_metadata.has_mamba_track_mask:
                mixed_qkv_to_track = mixed_qkv_transposed[
                    :, forward_metadata.track_conv_indices
                ].transpose(0, 1)
                conv_states[forward_metadata.conv_states_mask_indices] = (
                    mixed_qkv_to_track
                )

            mixed_qkv_post_conv = causal_conv1d_fn(
                mixed_qkv_transposed,
                layer.conv_weights,
                layer.bias,
                activation=layer.activation,
                conv_states=conv_states,
                has_initial_state=has_initial_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            ).transpose(0, 1)[:seq_len]

        # Restore conv_state if not persisting
        if not persist_state:
            conv_states[cache_indices] = conv_state_backup

        if (
            cache_intermediate_for_commit
            and use_intermediate_conv
            and causal_mode in (1, 2)
            and os.getenv("SGLANG_HYBRID_DIFFUSION_SELF_SPEC_GDN_SPLIT_REFERENCE", "0")
            != "1"
        ):
            if not hasattr(mamba_cache_params, "intermediate_ssm"):
                raise RuntimeError(
                    "dLLM GDN intermediate commit requires MambaPool.SpeculativeState"
                )
            output, final_state = (
                fused_recurrent_block_causal_gated_delta_rule_packed(
                    mixed_qkv=mixed_qkv_post_conv,
                    a=a,
                    b=b,
                    A_log=layer.A_log,
                    dt_bias=layer.dt_bias,
                    ssm_states=ssm_states,
                    cache_indices=cache_indices,
                    block_size=block_size,
                    causal_mode=causal_mode,
                    num_clean=num_clean,
                    output_final_state=persist_state,
                    intermediate_states_buffer=mamba_cache_params.intermediate_ssm,
                    intermediate_state_indices=intermediate_state_indices,
                    cache_steps=block_size,
                    use_qk_l2norm_in_kernel=True,
                )
            )
            if persist_state and final_state is not None:
                ssm_states[cache_indices] = final_state.to(ssm_states.dtype)
            return output

        # ── Split + gating ──
        query, key, value = torch.split(
            mixed_qkv_post_conv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )

        actual_seq_len = query.shape[0]
        query = query.view(1, actual_seq_len, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, actual_seq_len, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, actual_seq_len, layer.num_v_heads, layer.head_v_dim)

        g, beta_val = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)

        # ── GDN kernel with block-causal readout ──
        # GQA repeat if needed
        if layer.num_v_heads // layer.num_k_heads > 1:
            repeat_factor = layer.num_v_heads // layer.num_k_heads
            query = query.repeat_interleave(repeat_factor, dim=2)
            key = key.repeat_interleave(repeat_factor, dim=2)

        # CUDA graph capture cannot branch on a GPU bool reduction here. Pure
        # dLLM decode blocks always continue from cached GDN state; mixed/prefill
        # batches are not graphable and keep the precise per-request check.
        initial_state = (
            ssm_states[cache_indices]
            if use_intermediate_conv or forward_batch.forward_mode.is_cuda_graph()
            else (ssm_states[cache_indices] if has_initial_states.any() else None)
        )

        if cache_intermediate_for_commit:
            if not hasattr(mamba_cache_params, "intermediate_ssm"):
                raise RuntimeError(
                    "dLLM GDN intermediate commit requires MambaPool.SpeculativeState"
                )
            if not use_intermediate_conv:
                raise RuntimeError(
                    "dLLM GDN intermediate commit requires uniform pure-decode blocks"
                )
            output, final_state = fused_recurrent_block_causal_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta_val,
                block_size=block_size,
                causal_mode=causal_mode,
                num_clean=num_clean,
                initial_state=initial_state,
                output_final_state=persist_state,
                intermediate_states_buffer=mamba_cache_params.intermediate_ssm,
                intermediate_state_indices=intermediate_state_indices,
                cache_steps=block_size,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=query_start_loc_long,
            )
        elif causal_mode == 1:
            output, final_state = fused_recurrent_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta_val,
                initial_state=initial_state,
                output_final_state=persist_state,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=query_start_loc_long,
            )
        else:
            output, final_state = fused_recurrent_block_causal_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta_val,
                block_size=block_size,
                causal_mode=causal_mode,
                num_clean=num_clean,
                initial_state=initial_state,
                output_final_state=persist_state,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=query_start_loc_long,
            )

        # ── Persist state if requested ──
        if persist_state and final_state is not None:
            ssm_states[cache_indices] = final_state.to(ssm_states.dtype)

        # ── Save projections for commit ──
        if save_for_commit and not use_graph_save_buffer and not cache_intermediate_for_commit:
            # Store per-request data keyed by cache_indices
            indices_list = cache_indices.tolist()
            for i, rpx in enumerate(indices_list):
                if rpx not in self._saved:
                    self._saved[rpx] = {}
                self._saved[rpx][layer.layer_id] = (
                    pre_conv_mixed_qkv,  # shared across batch (will slice by request later)
                    a_saved,
                    b_saved,
                    conv_snapshots[i],
                    i,  # index within batch for slicing
                )

        return output

    # ------------------------------------------------------------------ #
    #  Commit accepted tokens (HybridDiffusion self-spec only)                      #
    # ------------------------------------------------------------------ #

    def commit_accepted_tokens(
        self,
        layer: "RadixLinearAttention",
        mamba_cache_idx: int,
        num_accepted: int,
        query_start: int,
        batch_idx: int = 0,
        use_graph_saved: bool = False,
    ) -> None:
        """Extend GDN state through accepted tokens only.

        Called per GDN layer per request after HybridDiffusionSelfSpec verify+trim.
        Uses saved projections from the prediction forward.

        Args:
            layer: The RadixLinearAttention layer (for weights).
            mamba_cache_idx: Mamba cache slot index for this request.
            num_accepted: Number of accepted tokens to commit.
            query_start: Start position of this request's tokens in the
                        saved projection tensors.
            batch_idx: Request row within the real batch.
            use_graph_saved: Read graph-replayed save buffers instead of the
                        eager Python dictionary.
        """
        if use_graph_saved:
            saved = self._graph_saved.get(layer.layer_id)
            if saved is None:
                raise RuntimeError(
                    f"Missing GDN graph save buffers for layer {layer.layer_id}"
                )
            pre_conv_mixed_qkv = saved["mixed_qkv"]
            a_saved = saved["a"]
            b_saved = saved["b"]
            conv_snapshot = saved["conv_snapshot"][batch_idx]
        else:
            saved = self._saved.get(mamba_cache_idx, {}).get(layer.layer_id)
            if saved is None:
                return
            pre_conv_mixed_qkv, a_saved, b_saved, conv_snapshot, _ = saved

        # Slice accepted tokens from saved projections
        # The saved tensors are for the full batch; extract this request's tokens
        # For simplicity, we assume single-request commit (can be extended to batched)
        start = query_start
        end = start + num_accepted
        pre_conv_acc = pre_conv_mixed_qkv[start:end]
        a_acc = a_saved[start:end]
        b_acc = b_saved[start:end]

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal

        # Restore conv_state from snapshot
        conv_states[mamba_cache_idx] = conv_snapshot

        # Run conv1d on accepted tokens. This Triton wrapper expects
        # [dim, total_tokens] plus query_start_loc for sequence boundaries.
        pre_conv_for_conv = pre_conv_acc.transpose(0, 1)  # [dim, num_accepted]

        cache_idx_tensor = torch.tensor([mamba_cache_idx], device=conv_states.device, dtype=torch.int32)
        has_initial = torch.tensor([True], device=conv_states.device)
        query_start_loc = torch.tensor([0, num_accepted], device=conv_states.device, dtype=torch.long)

        post_conv = causal_conv1d_fn(
            pre_conv_for_conv,
            layer.conv_weights,
            layer.bias,
            activation=layer.activation,
            conv_states=conv_states,
            has_initial_state=has_initial,
            cache_indices=cache_idx_tensor,
            query_start_loc=query_start_loc,
            seq_lens_cpu=torch.tensor([num_accepted]),
        ).transpose(0, 1)  # [num_accepted, dim]

        # Split + gating
        query, key, value = torch.split(
            post_conv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        query = query.view(1, num_accepted, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, num_accepted, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, num_accepted, layer.num_v_heads, layer.head_v_dim)

        g, beta_val = fused_gdn_gating(layer.A_log, a_acc, b_acc, layer.dt_bias)

        # GQA repeat
        if layer.num_v_heads // layer.num_k_heads > 1:
            repeat_factor = layer.num_v_heads // layer.num_k_heads
            query = query.repeat_interleave(repeat_factor, dim=2)
            key = key.repeat_interleave(repeat_factor, dim=2)

        # Read committed state
        initial_state = ssm_states[mamba_cache_idx].unsqueeze(0)  # [1, HV, V, K]

        # Recurrent forward (token-causal, commit)
        _, final_state = fused_recurrent_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta_val,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

        # Write committed state
        ssm_states[mamba_cache_idx] = final_state.squeeze(0).to(ssm_states.dtype)

    def commit_accepted_tokens_batch(
        self,
        layer: "RadixLinearAttention",
        mamba_cache_indices: List[int],
        num_accepted: List[int],
        query_starts: List[int],
        batch_indices: List[int],
        use_graph_saved: bool = False,
    ) -> None:
        """Commit accepted GDN state for a whole decode batch in one layer call."""
        if not use_graph_saved:
            for mamba_cache_idx, adv, query_start, batch_idx in zip(
                mamba_cache_indices, num_accepted, query_starts, batch_indices
            ):
                self.commit_accepted_tokens(
                    layer=layer,
                    mamba_cache_idx=mamba_cache_idx,
                    num_accepted=adv,
                    query_start=query_start,
                    batch_idx=batch_idx,
                    use_graph_saved=False,
                )
            return

        saved = self._graph_saved.get(layer.layer_id)
        if saved is None:
            raise RuntimeError(
                f"Missing GDN graph save buffers for layer {layer.layer_id}"
            )

        device = saved["mixed_qkv"].device
        cache_idx_t = torch.tensor(
            mamba_cache_indices, device=device, dtype=torch.int32
        )
        batch_idx_t = torch.tensor(batch_indices, device=device, dtype=torch.long)

        token_indices = torch.cat(
            [
                torch.arange(start, start + adv, device=device, dtype=torch.long)
                for start, adv in zip(query_starts, num_accepted)
            ]
        )
        cu = [0]
        for adv in num_accepted:
            cu.append(cu[-1] + adv)
        query_start_loc = torch.tensor(cu, device=device, dtype=torch.long)

        pre_conv_acc = saved["mixed_qkv"][token_indices]
        a_acc = saved["a"][token_indices]
        b_acc = saved["b"][token_indices]
        conv_snapshots = saved["conv_snapshot"][batch_idx_t]

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal

        conv_states[cache_idx_t] = conv_snapshots
        pre_conv_for_conv = pre_conv_acc.transpose(0, 1)
        has_initial = torch.ones(
            len(mamba_cache_indices), device=device, dtype=torch.bool
        )

        post_conv = causal_conv1d_fn(
            pre_conv_for_conv,
            layer.conv_weights,
            layer.bias,
            activation=layer.activation,
            conv_states=conv_states,
            has_initial_state=has_initial,
            cache_indices=cache_idx_t,
            query_start_loc=query_start_loc,
            seq_lens_cpu=num_accepted,
        ).transpose(0, 1)

        query, key, value = torch.split(
            post_conv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        total_tokens = post_conv.shape[0]
        query = query.view(1, total_tokens, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, total_tokens, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, total_tokens, layer.num_v_heads, layer.head_v_dim)

        g, beta_val = fused_gdn_gating(layer.A_log, a_acc, b_acc, layer.dt_bias)

        if layer.num_v_heads // layer.num_k_heads > 1:
            repeat_factor = layer.num_v_heads // layer.num_k_heads
            query = query.repeat_interleave(repeat_factor, dim=2)
            key = key.repeat_interleave(repeat_factor, dim=2)

        _, final_state = fused_recurrent_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta_val,
            initial_state=ssm_states[cache_idx_t.long()],
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=query_start_loc,
        )

        ssm_states[cache_idx_t.long()] = final_state.to(ssm_states.dtype)

    def commit_cached_intermediate_states_batch(
        self,
        mamba_cache_indices: Union[List[int], torch.Tensor],
        num_accepted: List[int],
        batch_indices: List[int],
        mamba_track_indices: Optional[Union[List[int], torch.Tensor]] = None,
        mamba_steps_to_track: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> None:
        """Commit accepted states from dLLM intermediate caches for all GDN layers.

        The dLLM forward writes speculative intermediate rows by batch position,
        matching the DFlash/MTP all-layer cache layout. This method scatters the
        accepted step for every request into the persistent MambaPool state.
        """
        if batch_indices != list(range(len(batch_indices))):
            raise RuntimeError(
                "dLLM intermediate commit currently requires a pure decode batch "
                "with source rows matching batch order"
            )

        mamba_caches = (
            self.req_to_token_pool.get_speculative_mamba2_params_all_layers()
        )
        device = mamba_caches.temporal.device
        if isinstance(mamba_cache_indices, torch.Tensor):
            cache_idx_t = mamba_cache_indices.to(device=device, dtype=torch.int32)
        else:
            cache_idx_t = torch.tensor(
                mamba_cache_indices, device=device, dtype=torch.int32
            )
        accepted_steps = torch.tensor(
            [adv - 1 if adv > 0 else -1 for adv in num_accepted],
            device=device,
            dtype=torch.int32,
        )

        fused_mamba_state_scatter_with_mask(
            mamba_caches.temporal,
            mamba_caches.intermediate_ssm,
            cache_idx_t,
            accepted_steps,
        )
        fused_mamba_state_scatter_with_mask(
            mamba_caches.conv[0],
            mamba_caches.intermediate_conv_window[0],
            cache_idx_t,
            accepted_steps,
        )
        if mamba_track_indices is not None:
            if mamba_steps_to_track is None:
                raise RuntimeError("mamba_steps_to_track is required for track scatter")
            if isinstance(mamba_steps_to_track, torch.Tensor):
                has_track_commit = bool((mamba_steps_to_track >= 0).any().item())
            else:
                has_track_commit = any(step >= 0 for step in mamba_steps_to_track)
            if not has_track_commit:
                mamba_track_indices = None
                mamba_steps_to_track = None
        if mamba_track_indices is not None:
            if isinstance(mamba_track_indices, torch.Tensor):
                track_idx_t = mamba_track_indices.to(device=device, dtype=torch.int32)
            else:
                track_idx_t = torch.tensor(
                    mamba_track_indices, device=device, dtype=torch.int32
                )
            if isinstance(mamba_steps_to_track, torch.Tensor):
                track_steps_t = mamba_steps_to_track.to(device=device, dtype=torch.int32)
            else:
                track_steps_t = torch.tensor(
                    mamba_steps_to_track, device=device, dtype=torch.int32
                )
            fused_mamba_state_scatter_with_mask(
                mamba_caches.temporal,
                mamba_caches.intermediate_ssm,
                track_idx_t,
                track_steps_t,
            )
            fused_mamba_state_scatter_with_mask(
                mamba_caches.conv[0],
                mamba_caches.intermediate_conv_window[0],
                track_idx_t,
                track_steps_t,
            )

    # ------------------------------------------------------------------ #
    #  Exact causal-prefix boundary handoff                               #
    # ------------------------------------------------------------------ #

    def _current_mamba_slot(self, request_pool_idx: int) -> int:
        mapping = self.req_to_token_pool.req_index_to_mamba_index_mapping
        return int(mapping[int(request_pool_idx)].item())

    def _validate_kv_reference(self, reference: KVPrefixReference) -> bool:
        if reference.pool_identity != id(self.req_to_token_pool):
            return False
        current = self.req_to_token_pool.req_to_token[
            reference.request_pool_idx, : reference.valid_length
        ]
        expected = reference.locations
        return (
            tuple(current.shape) == tuple(expected.shape)
            and current.device == expected.device
            and torch.equal(current, expected)
        )

    def snapshot_region_state(
        self,
        *,
        state_key: RegionStateKey,
        mamba_cache_idx: int,
        kv_prefix: KVPrefixReference,
    ) -> RegionState:
        """Clone every mutable GDN state at one exact canonical boundary."""
        if self._current_mamba_slot(state_key.request_pool_idx) != int(
            mamba_cache_idx
        ):
            raise RuntimeError("request/MambaPool slot mismatch while sealing boundary")
        if not self._validate_kv_reference(kv_prefix):
            raise RuntimeError("full-attention KV ownership changed while sealing boundary")
        mamba_cache = self.req_to_token_pool.mamba_pool.mamba_cache
        # Publish only after every tensor is cloned: readers cannot observe
        # convolution and recurrent state from different boundaries.
        conv = tuple(
            tensor[:, int(mamba_cache_idx)].detach().clone()
            for tensor in mamba_cache.conv
        )
        recurrent = mamba_cache.temporal[:, int(mamba_cache_idx)].detach().clone()
        state = RegionState(
            key=state_key,
            kv_prefix=kv_prefix,
            kv_valid_length=state_key.boundary,
            kv_owner_request_id=state_key.request_id,
            gdn_conv_states=conv,
            gdn_recurrent_states=recurrent,
            boundary=state_key.boundary,
        )
        self.region_state_cache.put(state)
        return state

    def restore_region_state(
        self,
        *,
        state_key: RegionStateKey,
        mamba_cache_idx: int,
        current_slot_generation: int,
    ) -> RegionStateLookup:
        """Restore a sealed state before one independent suffix forward."""
        lookup = self.region_state_cache.get(
            state_key, current_slot_generation=current_slot_generation
        )
        if not lookup.hit:
            return lookup
        state = lookup.state
        assert state is not None
        if self._current_mamba_slot(state_key.request_pool_idx) != int(
            mamba_cache_idx
        ):
            return RegionStateLookup(
                state=None,
                miss_reason=RegionStateMissReason.RECYCLED_REQUEST_SLOT,
            )
        if not self._validate_kv_reference(state.kv_prefix):
            return RegionStateLookup(
                state=None,
                miss_reason=RegionStateMissReason.POSITION_MISMATCH,
            )
        mamba_cache = self.req_to_token_pool.mamba_pool.mamba_cache
        for destination, source in zip(
            mamba_cache.conv, state.gdn_conv_states
        ):
            if destination.device != source.device or destination.dtype != source.dtype:
                raise RuntimeError("GDN convolution snapshot device/dtype mismatch")
            destination[:, int(mamba_cache_idx)].copy_(source)
        recurrent = state.gdn_recurrent_states
        destination = mamba_cache.temporal[:, int(mamba_cache_idx)]
        if destination.device != recurrent.device or destination.dtype != recurrent.dtype:
            raise RuntimeError("GDN recurrent snapshot device/dtype mismatch")
        destination.copy_(recurrent)
        return lookup

    def commit_region_state(
        self,
        *,
        state_key: RegionStateKey,
        mamba_cache_idx: int,
        kv_prefix: KVPrefixReference,
    ) -> RegionState:
        """Atomically publish the complete state after accepted-token commit."""
        return self.snapshot_region_state(
            state_key=state_key,
            mamba_cache_idx=mamba_cache_idx,
            kv_prefix=kv_prefix,
        )

    def invalidate_region_state(self, request_id: str, region_id: str) -> int:
        return self.region_state_cache.invalidate_region(request_id, region_id)

    def invalidate_request_state(self, request_id: str) -> int:
        return self.region_state_cache.invalidate_request(request_id)

    # ------------------------------------------------------------------ #
    #  Cleanup                                                            #
    # ------------------------------------------------------------------ #

    def discard_saved(self, req_pool_idx: int) -> None:
        """Free saved projections for a completed request."""
        self._saved.pop(req_pool_idx, None)

    def discard_saved_batch(self, req_pool_indices: list) -> None:
        """Free saved projections for a batch of requests."""
        for rpx in req_pool_indices:
            self._saved.pop(rpx, None)
