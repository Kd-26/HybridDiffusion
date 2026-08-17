# Copyright (c) 2026 Yuchen Zhu

"""Two-stream chunk-level WY noisy readout kernels.

This module backs the explicit ``chunk_wy_triton`` and
``chunk_wy_triton_improved`` development routes. The release auto route uses
the related fused implementation in ``chunk_fla_style_wy.py``.

The baseline backward is O(2M) per chunk. One kernel processes all chunks
sequentially to preserve inter-chunk gradient propagation. Within each chunk:

  Phase A (forward, O(M)):
    Compute S_j from S_0 via block-scan, save to scratch buffer.
    Compute b_j^n via noisy forward+backward replay, save to scratch.

  Phase B (reverse, O(M)):
    Load S_j, b_j^n from scratch.
    Reverse scan: dS_next = alpha · dS_next + b_j^n
    dg_end_j from ⟨dS_next, S_j⟩
    Clean grads from dDelta_j = dS_next

  Inter-chunk: dh propagates across chunks within the kernel.

Its HBM scratch contains two fp32 buffers of shape
``[num_programs, M, K, BV]``; the exact allocation is shape-dependent and is
reused across chunks. The improved route can instead use split kernels and
strided checkpoints.
"""

from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp


_IMPROVED_BWD_BV_OPTIONS = (8, 16, 32, 64)


def _resolve_improved_bwd_bv(V: int, bwd_bv: int | None) -> int:
    if bwd_bv is None or bwd_bv <= 0:
        return min(32, V)
    if bwd_bv not in _IMPROVED_BWD_BV_OPTIONS:
        raise ValueError(
            "QWEN35_CHUNK_WY_IMPROVED_BV must be one of "
            f"{_IMPROVED_BWD_BV_OPTIONS}, got {bwd_bv}"
        )
    return bwd_bv


def _auto_improved_bwd_parallel_groups(
    NC: int,
    cu_seqlens: torch.LongTensor | None,
) -> int:
    if cu_seqlens is None:
        return min(16, NC)
    n_docs = max(1, len(cu_seqlens) - 1)
    return min(16, max(1, NC // n_docs))


def _auto_improved_bwd_checkpoint_stride(M: int) -> int:
    """Conservative default from the first Qwen3.5-2B P/S sweep."""
    if M >= 64:
        return 8
    if M >= 16:
        return 4
    return max(1, int(math.ceil(math.sqrt(M))))


# =====================================================================
#  Forward kernel
# =====================================================================

@triton.jit(do_not_specialize=["T", "NC"])
def two_stream_wy_noisy_fwd_kernel(
    q_n, k_n, v_n, g_n, beta_n,
    k_c, v_new_c, G_c,
    h_chunks,
    o,
    T, NC, scale,
    chunk_bos,        # [total_NC] int32 — per-chunk doc token offset (varlen)
    chunk_start_arr,  # [total_NC] int32 — per-chunk local offset (varlen)
    chunk_T_arr,      # [total_NC] int32 — per-chunk doc length (varlen)
    B: tl.constexpr, H: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, CHUNK_SIZE: tl.constexpr,
    M: tl.constexpr,
    CAUSAL_MODE: tl.constexpr = 1,
    IS_VARLEN: tl.constexpr = False,
):
    i_cbh = tl.program_id(0)
    i_v = tl.program_id(1)

    if IS_VARLEN:
        i_gc = i_cbh // H
        i_h = i_cbh % H
        bos = tl.load(chunk_bos + i_gc).to(tl.int32)
        chunk_start = tl.load(chunk_start_arr + i_gc).to(tl.int32)
        T = tl.load(chunk_T_arr + i_gc).to(tl.int32)
        ckpt_flat = i_gc
    else:
        i_chunk = i_cbh // (B * H)
        i_bh = i_cbh % (B * H)
        i_b = i_bh // H
        i_h = i_bh % H
        bos = i_b * T
        chunk_start = i_chunk * CHUNK_SIZE
        ckpt_flat = i_b * NC + i_chunk

    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    stride_qk = H * K
    stride_vt = H * V
    stride_gt = H

    p_h = (h_chunks + (ckpt_flat * H + i_h) * (K * V)
           + o_k[:, None] * V + o_v[None, :])
    b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)

    base_qn = q_n + (bos * H + i_h) * K + o_k + chunk_start * stride_qk
    base_kn = k_n + (bos * H + i_h) * K + o_k + chunk_start * stride_qk
    base_vn = v_n + (bos * H + i_h) * V + o_v + chunk_start * stride_vt
    base_gn = g_n + bos * H + i_h + chunk_start * stride_gt
    base_bn = beta_n + bos * H + i_h + chunk_start * stride_gt
    base_kc = k_c + (bos * H + i_h) * K + o_k + chunk_start * stride_qk
    base_vnew = v_new_c + (bos * H + i_h) * V + o_v + chunk_start * stride_vt
    base_Gc = G_c + bos * H + i_h + chunk_start * stride_gt
    base_o = o + (bos * H + i_h) * V + o_v + chunk_start * stride_vt

    b_G_prev_end = tl.zeros([], dtype=tl.float32)
    b_noisy_first_scale = tl.where(chunk_start == 0, 0.0, 1.0).to(tl.float32)

    for j in range(M):
        ls = j * BLOCK_SIZE

        if j == 0:
            b_h_noisy = b_h * b_noisy_first_scale
        else:
            b_h_noisy = b_h

        if CAUSAL_MODE == 1:
            for t in range(BLOCK_SIZE):
                lt = ls + t
                if chunk_start + lt < T:
                    off_qk = lt * stride_qk
                    off_v = lt * stride_vt
                    off_g = lt * stride_gt

                    b_q = tl.load(base_qn + off_qk, mask=mask_k, other=0).to(tl.float32)
                    b_k = tl.load(base_kn + off_qk, mask=mask_k, other=0).to(tl.float32)
                    b_v = tl.load(base_vn + off_v, mask=mask_v, other=0).to(tl.float32)
                    b_g = tl.load(base_gn + off_g).to(tl.float32)
                    b_beta = tl.load(base_bn + off_g).to(tl.float32)

                    b_h_noisy = b_h_noisy * exp(b_g)
                    b_kTh = tl.sum(b_h_noisy * b_k[:, None], axis=0)
                    b_delta = b_beta * (b_v - b_kTh)
                    b_h_noisy = b_h_noisy + b_k[:, None] * b_delta

                    b_o = tl.sum(b_h_noisy * (b_q * scale)[:, None], axis=0)
                    tl.store(base_o + off_v, b_o.to(base_o.dtype.element_ty), mask=mask_v)
        else:
            # CAUSAL_MODE == 0: two-phase — run recurrence, then output with h_block_end
            for t in range(BLOCK_SIZE):
                lt = ls + t
                if chunk_start + lt < T:
                    off_qk = lt * stride_qk
                    off_v = lt * stride_vt
                    off_g = lt * stride_gt

                    b_k = tl.load(base_kn + off_qk, mask=mask_k, other=0).to(tl.float32)
                    b_v = tl.load(base_vn + off_v, mask=mask_v, other=0).to(tl.float32)
                    b_g = tl.load(base_gn + off_g).to(tl.float32)
                    b_beta = tl.load(base_bn + off_g).to(tl.float32)

                    b_h_noisy = b_h_noisy * exp(b_g)
                    b_kTh = tl.sum(b_h_noisy * b_k[:, None], axis=0)
                    b_delta = b_beta * (b_v - b_kTh)
                    b_h_noisy = b_h_noisy + b_k[:, None] * b_delta

            for t in range(BLOCK_SIZE):
                lt = ls + t
                if chunk_start + lt < T:
                    off_qk = lt * stride_qk
                    off_v = lt * stride_vt

                    b_q = tl.load(base_qn + off_qk, mask=mask_k, other=0).to(tl.float32)
                    b_o = tl.sum(b_h_noisy * (b_q * scale)[:, None], axis=0)
                    tl.store(base_o + off_v, b_o.to(base_o.dtype.element_ty), mask=mask_v)

        # Clean state advance — guard against out-of-bounds blocks
        block_end_pos = chunk_start + ls + BLOCK_SIZE - 1
        if block_end_pos < T:
            last_off = (ls + BLOCK_SIZE - 1) * stride_gt
            b_G_end = tl.load(base_Gc + last_off).to(tl.float32)
        else:
            b_G_end = b_G_prev_end

        b_delta_state = tl.zeros([K, BV], dtype=tl.float32)
        for t in range(BLOCK_SIZE):
            lt = ls + t
            if chunk_start + lt < T:
                b_G_t = tl.load(base_Gc + lt * stride_gt).to(tl.float32)
                b_e = exp(b_G_end - b_G_t)
                b_kc_t = tl.load(base_kc + lt * stride_qk, mask=mask_k, other=0).to(tl.float32)
                b_vnew_t = tl.load(base_vnew + lt * stride_vt, mask=mask_v, other=0).to(tl.float32)
                b_delta_state += b_kc_t[:, None] * (b_vnew_t * b_e)[None, :]

        if j == 0:
            b_alpha = exp(b_G_end)
        else:
            b_alpha = exp(b_G_end - b_G_prev_end)
        b_h = b_alpha * b_h + b_delta_state
        b_G_prev_end = b_G_end


# =====================================================================
#  Python wrappers
# =====================================================================

def _build_chunk_meta(cu_seqlens, chunk_size, device):
    """Build per-chunk metadata arrays for varlen forward kernel."""
    N_docs = len(cu_seqlens) - 1
    bos_list, start_list, T_list = [], [], []
    for d in range(N_docs):
        s = cu_seqlens[d].item()
        e = cu_seqlens[d + 1].item()
        doc_T = e - s
        doc_NC = max(1, (doc_T + chunk_size - 1) // chunk_size)
        for c in range(doc_NC):
            bos_list.append(s)
            start_list.append(c * chunk_size)
            T_list.append(doc_T)
    return (
        torch.tensor(bos_list, dtype=torch.int32, device=device),
        torch.tensor(start_list, dtype=torch.int32, device=device),
        torch.tensor(T_list, dtype=torch.int32, device=device),
    )


def _build_chunk_offsets(cu_seqlens, chunk_size, device):
    """Build per-doc chunk offsets for varlen backward kernel."""
    N_docs = len(cu_seqlens) - 1
    offsets, nc_list = [], []
    cursor = 0
    for d in range(N_docs):
        offsets.append(cursor)
        doc_T = (cu_seqlens[d + 1] - cu_seqlens[d]).item()
        doc_NC = max(1, (doc_T + chunk_size - 1) // chunk_size)
        nc_list.append(doc_NC)
        cursor += doc_NC
    return (
        torch.tensor(offsets, dtype=torch.int32, device=device),
        torch.tensor(nc_list, dtype=torch.int32, device=device),
    )


def _build_chunk_offsets_no_sync(cu_seqlens, chunk_size, device):
    """Build varlen backward metadata on-device for the improved path."""
    N_docs = len(cu_seqlens) - 1
    doc_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    doc_nc = torch.clamp(
        (doc_lens + chunk_size - 1) // chunk_size,
        min=1,
    ).to(dtype=torch.int32, device=device)
    offsets = torch.empty((N_docs,), dtype=torch.int32, device=device)
    if N_docs > 0:
        offsets[:1].zero_()
    if N_docs > 1:
        offsets[1:] = torch.cumsum(doc_nc[:-1], dim=0)
    return offsets, doc_nc


def _build_chunk_meta_no_sync(cu_seqlens, chunk_size, device, total_nc):
    """Build varlen forward metadata on-device for the improved path."""
    chunk_offsets, doc_nc = _build_chunk_offsets_no_sync(
        cu_seqlens, chunk_size, device)
    chunk_ids = torch.arange(total_nc, dtype=torch.int64, device=device)
    if len(cu_seqlens) == 2:
        doc_idx = torch.zeros_like(chunk_ids)
    else:
        doc_idx = torch.searchsorted(
            chunk_offsets[1:].to(torch.int64),
            chunk_ids,
            right=True,
        )

    chunk_offsets_i64 = chunk_offsets.to(torch.int64)
    chunk_in_doc = chunk_ids - chunk_offsets_i64[doc_idx]
    doc_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(dtype=torch.int32, device=device)
    chunk_bos = cu_seqlens[:-1].to(dtype=torch.int32, device=device)[doc_idx]
    chunk_start = (chunk_in_doc * chunk_size).to(torch.int32)
    chunk_T = doc_lens[doc_idx]
    return chunk_bos, chunk_start, chunk_T


def _two_stream_wy_noisy_fwd_impl(
    q_n: torch.Tensor, k_n: torch.Tensor, v_n: torch.Tensor,
    g_n: torch.Tensor, beta_n: torch.Tensor,
    k_c: torch.Tensor, v_new_c: torch.Tensor, G_c: torch.Tensor,
    h_chunks: torch.Tensor,
    scale: float, block_size: int, chunk_size: int,
    causal_mode: int = 1,
    cu_seqlens: torch.LongTensor | None = None,
    use_no_sync_varlen_meta: bool = False,
) -> torch.Tensor:
    q_n, k_n, v_n = q_n.contiguous(), k_n.contiguous(), v_n.contiguous()
    g_n, beta_n = g_n.contiguous(), beta_n.contiguous()
    k_c, v_new_c, G_c = k_c.contiguous(), v_new_c.contiguous(), G_c.contiguous()
    h_chunks = h_chunks.contiguous()
    B, T, H, K = q_n.shape
    V = v_n.shape[-1]
    NC = h_chunks.shape[1]
    M = chunk_size // block_size
    assert chunk_size % block_size == 0

    is_varlen = cu_seqlens is not None
    o = torch.empty_like(v_n)
    BV = min(64, V)

    if is_varlen:
        assert B == 1, "B must be 1 for cu_seqlens (doc packing)"
        if use_no_sync_varlen_meta:
            chunk_bos, chunk_start_arr, chunk_T_arr = _build_chunk_meta_no_sync(
                cu_seqlens, chunk_size, q_n.device, NC)
        else:
            chunk_bos, chunk_start_arr, chunk_T_arr = _build_chunk_meta(
                cu_seqlens, chunk_size, q_n.device)
        total_NC = chunk_bos.shape[0]
        grid = (total_NC * H, triton.cdiv(V, BV))
    else:
        chunk_bos = chunk_start_arr = chunk_T_arr = None
        grid = (NC * B * H, triton.cdiv(V, BV))

    two_stream_wy_noisy_fwd_kernel[grid](
        q_n, k_n, v_n, g_n, beta_n,
        k_c, v_new_c, G_c, h_chunks, o,
        T, NC, scale,
        chunk_bos, chunk_start_arr, chunk_T_arr,
        B=B, H=H, K=K, V=V, BV=BV,
        BLOCK_SIZE=block_size, CHUNK_SIZE=chunk_size, M=M,
        CAUSAL_MODE=causal_mode,
        IS_VARLEN=is_varlen,
    )
    return o


def two_stream_wy_noisy_fwd(
    q_n: torch.Tensor, k_n: torch.Tensor, v_n: torch.Tensor,
    g_n: torch.Tensor, beta_n: torch.Tensor,
    k_c: torch.Tensor, v_new_c: torch.Tensor, G_c: torch.Tensor,
    h_chunks: torch.Tensor,
    scale: float, block_size: int, chunk_size: int,
    causal_mode: int = 1,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    return _two_stream_wy_noisy_fwd_impl(
        q_n=q_n, k_n=k_n, v_n=v_n, g_n=g_n, beta_n=beta_n,
        k_c=k_c, v_new_c=v_new_c, G_c=G_c, h_chunks=h_chunks,
        scale=scale, block_size=block_size, chunk_size=chunk_size,
        causal_mode=causal_mode, cu_seqlens=cu_seqlens,
        use_no_sync_varlen_meta=False,
    )


def two_stream_wy_noisy_fwd_improved(
    q_n: torch.Tensor, k_n: torch.Tensor, v_n: torch.Tensor,
    g_n: torch.Tensor, beta_n: torch.Tensor,
    k_c: torch.Tensor, v_new_c: torch.Tensor, G_c: torch.Tensor,
    h_chunks: torch.Tensor,
    scale: float, block_size: int, chunk_size: int,
    causal_mode: int = 1,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    return _two_stream_wy_noisy_fwd_impl(
        q_n=q_n, k_n=k_n, v_n=v_n, g_n=g_n, beta_n=beta_n,
        k_c=k_c, v_new_c=v_new_c, G_c=G_c, h_chunks=h_chunks,
        scale=scale, block_size=block_size, chunk_size=chunk_size,
        causal_mode=causal_mode, cu_seqlens=cu_seqlens,
        use_no_sync_varlen_meta=True,
    )


# =====================================================================
#  Backward kernel — O(2M) per chunk, prefix-recompute
#  Recomputes h_decayed[t] on-the-fly instead of caching in registers.
#  Supports arbitrary BLOCK_SIZE with no register spilling.
#  Cost: O(BS^2) per block (vs O(BS) for cached), but avoids the
#  3-6 KB/thread spill that makes cached slower for BS >= 2.
# =====================================================================

@triton.jit(do_not_specialize=["T", "NC"])
def two_stream_wy_noisy_bwd_kernel(
    q_n, k_n, v_n, g_n, beta_n,
    k_c, v_new_c, G_c,
    h_chunks, do,
    dq_n, dk_n, dv_n, dg_n, dbeta_n,
    dk_c_out, dv_new_c_out, dG_c_out,
    dh_chunks_out,
    scratch_s, scratch_b,
    T, NC, scale,
    cu_seqlens,      # [N_docs+1] int64 — varlen doc boundaries (or None)
    chunk_offsets,   # [N_docs] int32 — per-doc start index in h_chunks (or None)
    doc_nc_arr,      # [N_docs] int32 — per-doc chunk count (or None)
    NUM_V_TILES: tl.constexpr,
    B: tl.constexpr, H: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, CHUNK_SIZE: tl.constexpr,
    M: tl.constexpr,
    CAUSAL_MODE: tl.constexpr = 1,
    IS_VARLEN: tl.constexpr = False,
):
    i_bh = tl.program_id(0)
    i_v = tl.program_id(1)

    if IS_VARLEN:
        i_n = i_bh // H
        i_h = i_bh % H
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        T = tl.load(cu_seqlens + i_n + 1).to(tl.int32) - bos
        chunk_base = tl.load(chunk_offsets + i_n).to(tl.int32)
        local_NC = tl.load(doc_nc_arr + i_n).to(tl.int32)
    else:
        i_b = i_bh // H
        i_h = i_bh % H
        bos = i_b * T
        chunk_base = i_b * NC
        local_NC = NC

    o_k = tl.arange(0, K)
    o_bv = tl.arange(0, BV)
    o_v = i_v * BV + o_bv
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    stride_qk = H * K
    stride_vt = H * V
    stride_gt = H

    prog_id = i_bh * NUM_V_TILES + i_v
    scratch_stride_j = K * BV
    scratch_base_s = scratch_s + prog_id * M * scratch_stride_j
    scratch_base_b = scratch_b + prog_id * M * scratch_stride_j

    b_dh = tl.zeros([K, BV], dtype=tl.float32)

    for c_rev in range(local_NC):
        c = local_NC - 1 - c_rev
        chunk_start = c * CHUNK_SIZE

        ckpt_flat = chunk_base + c
        p_h0 = (h_chunks + (ckpt_flat * H + i_h) * (K * V)
                + o_k[:, None] * V + o_v[None, :])
        b_S0 = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

        base_qn = q_n + (bos * H + i_h) * K + o_k + chunk_start * stride_qk
        base_kn = k_n + (bos * H + i_h) * K + o_k + chunk_start * stride_qk
        base_vn = v_n + (bos * H + i_h) * V + o_v + chunk_start * stride_vt
        base_gn = g_n + bos * H + i_h + chunk_start * stride_gt
        base_bn = beta_n + bos * H + i_h + chunk_start * stride_gt
        base_do = do + (bos * H + i_h) * V + o_v + chunk_start * stride_vt
        base_kc = k_c + (bos * H + i_h) * K + o_k + chunk_start * stride_qk
        base_vnew = v_new_c + (bos * H + i_h) * V + o_v + chunk_start * stride_vt
        base_Gc = G_c + bos * H + i_h + chunk_start * stride_gt
        base_dqn = dq_n + (bos * H + i_h) * K + o_k + chunk_start * stride_qk
        base_dkn = dk_n + (bos * H + i_h) * K + o_k + chunk_start * stride_qk
        base_dvn = dv_n + (bos * H + i_h) * V + o_v + chunk_start * stride_vt
        base_dgn = dg_n + bos * H + i_h + chunk_start * stride_gt
        base_dbn = dbeta_n + bos * H + i_h + chunk_start * stride_gt
        base_dkc = dk_c_out + (bos * H + i_h) * K + o_k + chunk_start * stride_qk
        base_dvnew = dv_new_c_out + (bos * H + i_h) * V + o_v + chunk_start * stride_vt
        base_dGc = dG_c_out + bos * H + i_h + chunk_start * stride_gt

        # ============== Phase A: forward scan O(M) ==============
        b_S = b_S0
        b_G_prev = tl.zeros([], dtype=tl.float32)
        b_noisy_first_scale = tl.where(c == 0, 0.0, 1.0).to(tl.float32)

        for j in range(M):
            ls = j * BLOCK_SIZE

            p_sj = scratch_base_s + j * scratch_stride_j + o_k[:, None] * BV + o_bv[None, :]
            tl.store(p_sj, b_S.to(tl.float32), mask=mask_h)

            if j == 0:
                b_Sj_noisy = b_S * b_noisy_first_scale
            else:
                b_Sj_noisy = b_S

            # ---- Noisy backward with prefix-recompute ----
            if CAUSAL_MODE == 0:
                # Mode 0: pre-accumulate dh_end, replay for h_end, compute dq,
                # then reverse-scan for recurrence gradients only.

                # Forward replay to get h_end + accumulate dh_n from all outputs
                b_dh_n = tl.zeros([K, BV], dtype=tl.float32)
                b_h_end = b_Sj_noisy
                for t_fwd in range(BLOCK_SIZE):
                    lt_fwd = ls + t_fwd
                    if chunk_start + lt_fwd < T:
                        off_qk_f = lt_fwd * stride_qk
                        off_v_f = lt_fwd * stride_vt
                        off_g_f = lt_fwd * stride_gt

                        b_g_f = tl.load(base_gn + off_g_f).to(tl.float32)
                        b_k_f = tl.load(base_kn + off_qk_f, mask=mask_k, other=0).to(tl.float32)
                        b_v_f = tl.load(base_vn + off_v_f, mask=mask_v, other=0).to(tl.float32)
                        b_beta_f = tl.load(base_bn + off_g_f).to(tl.float32)

                        b_h_end = b_h_end * exp(b_g_f)
                        b_kTh_f = tl.sum(b_h_end * b_k_f[:, None], axis=0)
                        b_delta_f = b_beta_f * (b_v_f - b_kTh_f)
                        b_h_end = b_h_end + b_k_f[:, None] * b_delta_f

                        b_q_f = tl.load(base_qn + off_qk_f, mask=mask_k, other=0).to(tl.float32)
                        b_do_f = tl.load(base_do + off_v_f, mask=mask_v, other=0).to(tl.float32)
                        b_dh_n += (b_q_f * scale)[:, None] * b_do_f[None, :]

                # Compute dq for all tokens using h_end
                for t_fwd in range(BLOCK_SIZE):
                    lt_fwd = ls + t_fwd
                    if chunk_start + lt_fwd < T:
                        off_qk_f = lt_fwd * stride_qk
                        off_v_f = lt_fwd * stride_vt
                        b_do_f = tl.load(base_do + off_v_f, mask=mask_v, other=0).to(tl.float32)
                        b_dq_f = tl.sum(b_h_end * b_do_f[None, :], axis=1) * scale
                        tl.atomic_add(base_dqn + off_qk_f,
                                      b_dq_f.to(base_dqn.dtype.element_ty), mask=mask_k)

                # Reverse scan: backprop through recurrence (no output injection)
                for t_rev in range(BLOCK_SIZE):
                    t = BLOCK_SIZE - 1 - t_rev
                    lt = ls + t
                    if chunk_start + lt < T:
                        b_h_replay = b_Sj_noisy
                        for s in range(BLOCK_SIZE):
                            ls_s = ls + s
                            if s <= t and chunk_start + ls_s < T:
                                b_g_s = tl.load(base_gn + ls_s * stride_gt).to(tl.float32)
                                b_h_replay = b_h_replay * exp(b_g_s)
                                if s < t:
                                    b_k_s = tl.load(base_kn + ls_s * stride_qk, mask=mask_k, other=0).to(tl.float32)
                                    b_v_s = tl.load(base_vn + ls_s * stride_vt, mask=mask_v, other=0).to(tl.float32)
                                    b_beta_s = tl.load(base_bn + ls_s * stride_gt).to(tl.float32)
                                    b_kTh_s = tl.sum(b_h_replay * b_k_s[:, None], axis=0)
                                    b_delta_s = b_beta_s * (b_v_s - b_kTh_s)
                                    b_h_replay = b_h_replay + b_k_s[:, None] * b_delta_s
                        b_hd = b_h_replay

                        off_qk = lt * stride_qk
                        off_v = lt * stride_vt
                        off_g = lt * stride_gt
                        b_k = tl.load(base_kn + off_qk, mask=mask_k, other=0).to(tl.float32)
                        b_v = tl.load(base_vn + off_v, mask=mask_v, other=0).to(tl.float32)
                        b_g = tl.load(base_gn + off_g).to(tl.float32)
                        b_beta = tl.load(base_bn + off_g).to(tl.float32)

                        b_kTh = tl.sum(b_hd * b_k[:, None], axis=0)
                        b_delta = b_beta * (b_v - b_kTh)

                        b_dk_1 = tl.sum(b_dh_n * b_delta[None, :], axis=1)
                        b_ddelta = tl.sum(b_dh_n * b_k[:, None], axis=0)

                        b_dbeta_p = tl.sum(b_ddelta * (b_v - b_kTh))
                        b_dv_p = b_beta * b_ddelta
                        b_dkTh = -b_beta * b_ddelta

                        b_dk_2 = tl.sum(b_hd * b_dkTh[None, :], axis=1)
                        b_dh_n += b_k[:, None] * b_dkTh[None, :]

                        b_dg_p = tl.sum(b_dh_n * b_hd)
                        b_dh_n = b_dh_n * exp(b_g)

                        tl.atomic_add(base_dkn + off_qk, (b_dk_1 + b_dk_2).to(base_dkn.dtype.element_ty), mask=mask_k)
                        tl.store(base_dvn + off_v, b_dv_p.to(base_dvn.dtype.element_ty), mask=mask_v)
                        tl.atomic_add(base_dgn + off_g, b_dg_p.to(base_dgn.dtype.element_ty))
                        tl.atomic_add(base_dbn + off_g, b_dbeta_p.to(base_dbn.dtype.element_ty))

            else:
                # Mode 1 (token-causal): current behavior
                b_dh_n = tl.zeros([K, BV], dtype=tl.float32)
                for t_rev in range(BLOCK_SIZE):
                    t = BLOCK_SIZE - 1 - t_rev
                    lt = ls + t
                    if chunk_start + lt < T:
                        b_h_replay = b_Sj_noisy
                        for s in range(BLOCK_SIZE):
                            ls_s = ls + s
                            if s <= t and chunk_start + ls_s < T:
                                b_g_s = tl.load(base_gn + ls_s * stride_gt).to(tl.float32)
                                b_h_replay = b_h_replay * exp(b_g_s)
                                if s < t:
                                    b_k_s = tl.load(base_kn + ls_s * stride_qk, mask=mask_k, other=0).to(tl.float32)
                                    b_v_s = tl.load(base_vn + ls_s * stride_vt, mask=mask_v, other=0).to(tl.float32)
                                    b_beta_s = tl.load(base_bn + ls_s * stride_gt).to(tl.float32)
                                    b_kTh_s = tl.sum(b_h_replay * b_k_s[:, None], axis=0)
                                    b_delta_s = b_beta_s * (b_v_s - b_kTh_s)
                                    b_h_replay = b_h_replay + b_k_s[:, None] * b_delta_s
                        b_hd = b_h_replay

                        off_qk = lt * stride_qk
                        off_v = lt * stride_vt
                        off_g = lt * stride_gt
                        b_q = tl.load(base_qn + off_qk, mask=mask_k, other=0).to(tl.float32)
                        b_k = tl.load(base_kn + off_qk, mask=mask_k, other=0).to(tl.float32)
                        b_v = tl.load(base_vn + off_v, mask=mask_v, other=0).to(tl.float32)
                        b_g = tl.load(base_gn + off_g).to(tl.float32)
                        b_beta = tl.load(base_bn + off_g).to(tl.float32)
                        b_do_t = tl.load(base_do + off_v, mask=mask_v, other=0).to(tl.float32)

                        b_kTh = tl.sum(b_hd * b_k[:, None], axis=0)
                        b_delta = b_beta * (b_v - b_kTh)
                        b_h_after = b_hd + b_k[:, None] * b_delta

                        b_dh_n += (b_q * scale)[:, None] * b_do_t[None, :]
                        b_dq_p = tl.sum(b_h_after * b_do_t[None, :], axis=1) * scale

                        b_dk_1 = tl.sum(b_dh_n * b_delta[None, :], axis=1)
                        b_ddelta = tl.sum(b_dh_n * b_k[:, None], axis=0)

                        b_dbeta_p = tl.sum(b_ddelta * (b_v - b_kTh))
                        b_dv_p = b_beta * b_ddelta
                        b_dkTh = -b_beta * b_ddelta

                        b_dk_2 = tl.sum(b_hd * b_dkTh[None, :], axis=1)
                        b_dh_n += b_k[:, None] * b_dkTh[None, :]

                        b_dg_p = tl.sum(b_dh_n * b_hd)
                        b_dh_n = b_dh_n * exp(b_g)

                        tl.atomic_add(base_dqn + off_qk, b_dq_p.to(base_dqn.dtype.element_ty), mask=mask_k)
                        tl.atomic_add(base_dkn + off_qk, (b_dk_1 + b_dk_2).to(base_dkn.dtype.element_ty), mask=mask_k)
                        tl.store(base_dvn + off_v, b_dv_p.to(base_dvn.dtype.element_ty), mask=mask_v)
                        tl.atomic_add(base_dgn + off_g, b_dg_p.to(base_dgn.dtype.element_ty))
                        tl.atomic_add(base_dbn + off_g, b_dbeta_p.to(base_dbn.dtype.element_ty))

            if j == 0:
                b_dh_n_save = b_dh_n * b_noisy_first_scale
            else:
                b_dh_n_save = b_dh_n
            p_bj = scratch_base_b + j * scratch_stride_j + o_k[:, None] * BV + o_bv[None, :]
            tl.store(p_bj, b_dh_n_save.to(tl.float32), mask=mask_h)

            # ---- Clean state advance ----
            block_end_pos = chunk_start + ls + BLOCK_SIZE - 1
            if block_end_pos < T:
                b_G_end = tl.load(base_Gc + (ls + BLOCK_SIZE - 1) * stride_gt).to(tl.float32)
            else:
                b_G_end = b_G_prev
            b_delta_state = tl.zeros([K, BV], dtype=tl.float32)
            for t in range(BLOCK_SIZE):
                lt = ls + t
                if chunk_start + lt < T:
                    b_G_t = tl.load(base_Gc + lt * stride_gt).to(tl.float32)
                    b_e = exp(b_G_end - b_G_t)
                    b_kc_t = tl.load(base_kc + lt * stride_qk, mask=mask_k, other=0).to(tl.float32)
                    b_vnew_t = tl.load(base_vnew + lt * stride_vt, mask=mask_v, other=0).to(tl.float32)
                    b_delta_state += b_kc_t[:, None] * (b_vnew_t * b_e)[None, :]
            if j == 0:
                b_alpha = exp(b_G_end)
            else:
                b_alpha = exp(b_G_end - b_G_prev)
            b_S = b_alpha * b_S + b_delta_state
            b_G_prev = b_G_end

        # b_G_prev now holds the G_c value of the last valid token in this
        # chunk (or 0 if the entire chunk is empty).  OOB blocks in Phase B
        # must use this as their G_c_end fallback so that alpha = exp(0) = 1
        # instead of the numerically catastrophic exp(0 - G_c[last_valid]).
        b_G_last_valid = b_G_prev

        # ============== Phase B: reverse scan O(M) ==============
        b_dS_next = b_dh
        b_dh_saved = b_dh
        b_P = tl.zeros([], dtype=tl.float32) + 1.0

        for j_rev in range(M):
            j = M - 1 - j_rev
            ls = j * BLOCK_SIZE

            p_sj = scratch_base_s + j * scratch_stride_j + o_k[:, None] * BV + o_bv[None, :]
            b_Sj = tl.load(p_sj, mask=mask_h).to(tl.float32)
            p_bj = scratch_base_b + j * scratch_stride_j + o_k[:, None] * BV + o_bv[None, :]
            b_bj = tl.load(p_bj, mask=mask_h).to(tl.float32)

            block_end_pos_j = chunk_start + ls + BLOCK_SIZE - 1
            if block_end_pos_j < T:
                b_G_end_j = tl.load(base_Gc + (ls + BLOCK_SIZE - 1) * stride_gt).to(tl.float32)
            else:
                b_G_end_j = b_G_last_valid
            if j == 0:
                b_alpha_j = exp(b_G_end_j)
            else:
                if chunk_start + (j - 1) * BLOCK_SIZE + BLOCK_SIZE - 1 < T:
                    b_G_prev_end = tl.load(base_Gc + ((j - 1) * BLOCK_SIZE + BLOCK_SIZE - 1) * stride_gt).to(tl.float32)
                else:
                    b_G_prev_end = b_G_last_valid
                b_alpha_j = exp(b_G_end_j - b_G_prev_end)

            b_dS_local = b_dS_next - b_P * b_dh_saved
            b_dalpha = tl.sum(b_dS_local * b_Sj)
            b_dG_end_acc = b_dalpha * b_alpha_j

            for t in range(BLOCK_SIZE):
                lt = ls + t
                if chunk_start + lt < T:
                    off_qk = lt * stride_qk
                    off_v = lt * stride_vt
                    off_g = lt * stride_gt
                    b_G_t = tl.load(base_Gc + off_g).to(tl.float32)
                    b_e = exp(b_G_end_j - b_G_t)
                    b_kc_t = tl.load(base_kc + off_qk, mask=mask_k, other=0).to(tl.float32)
                    b_vnew_t = tl.load(base_vnew + off_v, mask=mask_v, other=0).to(tl.float32)

                    b_dkc = tl.sum(b_dS_local * (b_vnew_t * b_e)[None, :], axis=1)
                    tl.atomic_add(base_dkc + off_qk, b_dkc.to(base_dkc.dtype.element_ty), mask=mask_k)
                    b_dvnew = b_e * tl.sum(b_dS_local * b_kc_t[:, None], axis=0)
                    tl.store(base_dvnew + off_v, b_dvnew.to(base_dvnew.dtype.element_ty), mask=mask_v)
                    b_dE = tl.sum(b_dS_local * b_kc_t[:, None] * b_vnew_t[None, :])
                    b_dG_end_acc += b_e * b_dE
                    b_dG_t = -b_e * b_dE
                    tl.atomic_add(base_dGc + off_g, b_dG_t.to(base_dGc.dtype.element_ty))

            if block_end_pos_j < T:
                tl.atomic_add(base_dGc + (ls + BLOCK_SIZE - 1) * stride_gt,
                              b_dG_end_acc.to(base_dGc.dtype.element_ty))

            if j > 0:
                prev_end_pos = chunk_start + (j - 1) * BLOCK_SIZE + BLOCK_SIZE - 1
                if prev_end_pos < T:
                    prev_last = ((j - 1) * BLOCK_SIZE + BLOCK_SIZE - 1) * stride_gt
                    tl.atomic_add(base_dGc + prev_last,
                                  (-b_dalpha * b_alpha_j).to(base_dGc.dtype.element_ty))

            b_P = b_alpha_j * b_P
            b_dS_next = b_alpha_j * b_dS_next + b_bj

        b_dh = b_dS_next
        p_dhc = (dh_chunks_out + (ckpt_flat * H + i_h) * (K * V)
                 + o_k[:, None] * V + o_v[None, :])
        tl.store(p_dhc, b_dS_next.to(p_dhc.dtype.element_ty), mask=mask_h)


@triton.jit(do_not_specialize=["T", "NC"])
def two_stream_wy_noisy_bwd_split_a_kernel(
    q_n, k_n, v_n, g_n, beta_n,
    k_c, v_new_c, G_c,
    h_chunks, do,
    dq_n, dk_n, dv_n, dg_n, dbeta_n,
    dk_c_out, dv_new_c_out, dG_c_out,
    dh_chunks_out, alpha_total,
    scratch_s, scratch_b,
    T, NC, scale,
    cu_seqlens,
    chunk_offsets,
    doc_nc_arr,
    NUM_V_TILES: tl.constexpr,
    B: tl.constexpr, H: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, CHUNK_SIZE: tl.constexpr,
    M: tl.constexpr,
    P: tl.constexpr,
    MAX_GROUP_CHUNKS: tl.constexpr,
    CHECKPOINT_STRIDE: tl.constexpr,
    NUM_CHECKPOINTS: tl.constexpr,
    CAUSAL_MODE: tl.constexpr = 1,
    IS_VARLEN: tl.constexpr = False,
    STORE_B: tl.constexpr = True,
):
    i_gbh = tl.program_id(0)
    i_v = tl.program_id(1)

    if IS_VARLEN:
        i_n = i_gbh // (P * H)
        rem = i_gbh - i_n * (P * H)
        i_group = rem // H
        i_h = rem - i_group * H
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        T = tl.load(cu_seqlens + i_n + 1).to(tl.int32) - bos
        chunk_base = tl.load(chunk_offsets + i_n).to(tl.int32)
        local_NC = tl.load(doc_nc_arr + i_n).to(tl.int32)
    else:
        i_b = i_gbh // (P * H)
        rem = i_gbh - i_b * (P * H)
        i_group = rem // H
        i_h = rem - i_group * H
        bos = i_b * T
        chunk_base = i_b * NC
        local_NC = NC

    chunks_per_group = (local_NC + P - 1) // P
    group_start = i_group * chunks_per_group
    group_end = tl.minimum(group_start + chunks_per_group, local_NC)

    o_k = tl.arange(0, K)
    o_bv = tl.arange(0, BV)
    o_v = i_v * BV + o_bv
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    stride_qk = H * K
    stride_vt = H * V
    stride_gt = H

    prog_id = i_gbh * NUM_V_TILES + i_v
    scratch_stride_j = K * BV
    scratch_base_s = scratch_s + prog_id * NUM_CHECKPOINTS * scratch_stride_j
    scratch_base_b = scratch_b + prog_id * M * scratch_stride_j

    for c_iter in range(MAX_GROUP_CHUNKS):
        c = group_start + c_iter
        if c < group_end:
            chunk_start = c * CHUNK_SIZE
            ckpt_flat = chunk_base + c
            p_h0 = (h_chunks + (ckpt_flat * H + i_h) * (K * V)
                    + o_k[:, None] * V + o_v[None, :])
            b_S0 = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

            base_qn = q_n + (bos * H + i_h) * K + o_k + chunk_start * stride_qk
            base_kn = k_n + (bos * H + i_h) * K + o_k + chunk_start * stride_qk
            base_vn = v_n + (bos * H + i_h) * V + o_v + chunk_start * stride_vt
            base_gn = g_n + bos * H + i_h + chunk_start * stride_gt
            base_bn = beta_n + bos * H + i_h + chunk_start * stride_gt
            base_do = do + (bos * H + i_h) * V + o_v + chunk_start * stride_vt
            base_kc = k_c + (bos * H + i_h) * K + o_k + chunk_start * stride_qk
            base_vnew = v_new_c + (bos * H + i_h) * V + o_v + chunk_start * stride_vt
            base_Gc = G_c + bos * H + i_h + chunk_start * stride_gt
            base_dqn = dq_n + (bos * H + i_h) * K + o_k + chunk_start * stride_qk
            base_dkn = dk_n + (bos * H + i_h) * K + o_k + chunk_start * stride_qk
            base_dvn = dv_n + (bos * H + i_h) * V + o_v + chunk_start * stride_vt
            base_dgn = dg_n + bos * H + i_h + chunk_start * stride_gt
            base_dbn = dbeta_n + bos * H + i_h + chunk_start * stride_gt
            base_dkc = dk_c_out + (bos * H + i_h) * K + o_k + chunk_start * stride_qk
            base_dvnew = dv_new_c_out + (bos * H + i_h) * V + o_v + chunk_start * stride_vt
            base_dGc = dG_c_out + bos * H + i_h + chunk_start * stride_gt

            b_S = b_S0
            b_G_prev = tl.zeros([], dtype=tl.float32)
            b_noisy_first_scale = tl.where(c == 0, 0.0, 1.0).to(tl.float32)

            for j in range(M):
                ls = j * BLOCK_SIZE

                if j % CHECKPOINT_STRIDE == 0:
                    ckpt_j = j // CHECKPOINT_STRIDE
                    p_sj = scratch_base_s + ckpt_j * scratch_stride_j + o_k[:, None] * BV + o_bv[None, :]
                    tl.store(p_sj, b_S.to(tl.float32), mask=mask_h)

                if j == 0:
                    b_Sj_noisy = b_S * b_noisy_first_scale
                else:
                    b_Sj_noisy = b_S

                if CAUSAL_MODE == 0:
                    b_dh_n = tl.zeros([K, BV], dtype=tl.float32)
                    b_h_end = b_Sj_noisy
                    for t_fwd in range(BLOCK_SIZE):
                        lt_fwd = ls + t_fwd
                        if chunk_start + lt_fwd < T:
                            off_qk_f = lt_fwd * stride_qk
                            off_v_f = lt_fwd * stride_vt
                            off_g_f = lt_fwd * stride_gt

                            b_g_f = tl.load(base_gn + off_g_f).to(tl.float32)
                            b_k_f = tl.load(base_kn + off_qk_f, mask=mask_k, other=0).to(tl.float32)
                            b_v_f = tl.load(base_vn + off_v_f, mask=mask_v, other=0).to(tl.float32)
                            b_beta_f = tl.load(base_bn + off_g_f).to(tl.float32)

                            b_h_end = b_h_end * exp(b_g_f)
                            b_kTh_f = tl.sum(b_h_end * b_k_f[:, None], axis=0)
                            b_delta_f = b_beta_f * (b_v_f - b_kTh_f)
                            b_h_end = b_h_end + b_k_f[:, None] * b_delta_f

                            b_q_f = tl.load(base_qn + off_qk_f, mask=mask_k, other=0).to(tl.float32)
                            b_do_f = tl.load(base_do + off_v_f, mask=mask_v, other=0).to(tl.float32)
                            b_dh_n += (b_q_f * scale)[:, None] * b_do_f[None, :]

                    for t_fwd in range(BLOCK_SIZE):
                        lt_fwd = ls + t_fwd
                        if chunk_start + lt_fwd < T:
                            off_qk_f = lt_fwd * stride_qk
                            off_v_f = lt_fwd * stride_vt
                            b_do_f = tl.load(base_do + off_v_f, mask=mask_v, other=0).to(tl.float32)
                            b_dq_f = tl.sum(b_h_end * b_do_f[None, :], axis=1) * scale
                            tl.atomic_add(base_dqn + off_qk_f,
                                          b_dq_f.to(base_dqn.dtype.element_ty), mask=mask_k)

                    for t_rev in range(BLOCK_SIZE):
                        t = BLOCK_SIZE - 1 - t_rev
                        lt = ls + t
                        if chunk_start + lt < T:
                            b_h_replay = b_Sj_noisy
                            for s in range(BLOCK_SIZE):
                                ls_s = ls + s
                                if s <= t and chunk_start + ls_s < T:
                                    b_g_s = tl.load(base_gn + ls_s * stride_gt).to(tl.float32)
                                    b_h_replay = b_h_replay * exp(b_g_s)
                                    if s < t:
                                        b_k_s = tl.load(base_kn + ls_s * stride_qk, mask=mask_k, other=0).to(tl.float32)
                                        b_v_s = tl.load(base_vn + ls_s * stride_vt, mask=mask_v, other=0).to(tl.float32)
                                        b_beta_s = tl.load(base_bn + ls_s * stride_gt).to(tl.float32)
                                        b_kTh_s = tl.sum(b_h_replay * b_k_s[:, None], axis=0)
                                        b_delta_s = b_beta_s * (b_v_s - b_kTh_s)
                                        b_h_replay = b_h_replay + b_k_s[:, None] * b_delta_s
                            b_hd = b_h_replay

                            off_qk = lt * stride_qk
                            off_v = lt * stride_vt
                            off_g = lt * stride_gt
                            b_k = tl.load(base_kn + off_qk, mask=mask_k, other=0).to(tl.float32)
                            b_v = tl.load(base_vn + off_v, mask=mask_v, other=0).to(tl.float32)
                            b_g = tl.load(base_gn + off_g).to(tl.float32)
                            b_beta = tl.load(base_bn + off_g).to(tl.float32)

                            b_kTh = tl.sum(b_hd * b_k[:, None], axis=0)
                            b_delta = b_beta * (b_v - b_kTh)

                            b_dk_1 = tl.sum(b_dh_n * b_delta[None, :], axis=1)
                            b_ddelta = tl.sum(b_dh_n * b_k[:, None], axis=0)

                            b_dbeta_p = tl.sum(b_ddelta * (b_v - b_kTh))
                            b_dv_p = b_beta * b_ddelta
                            b_dkTh = -b_beta * b_ddelta

                            b_dk_2 = tl.sum(b_hd * b_dkTh[None, :], axis=1)
                            b_dh_n += b_k[:, None] * b_dkTh[None, :]

                            b_dg_p = tl.sum(b_dh_n * b_hd)
                            b_dh_n = b_dh_n * exp(b_g)

                            tl.atomic_add(base_dkn + off_qk, (b_dk_1 + b_dk_2).to(base_dkn.dtype.element_ty), mask=mask_k)
                            tl.store(base_dvn + off_v, b_dv_p.to(base_dvn.dtype.element_ty), mask=mask_v)
                            tl.atomic_add(base_dgn + off_g, b_dg_p.to(base_dgn.dtype.element_ty))
                            tl.atomic_add(base_dbn + off_g, b_dbeta_p.to(base_dbn.dtype.element_ty))

                else:
                    b_dh_n = tl.zeros([K, BV], dtype=tl.float32)
                    for t_rev in range(BLOCK_SIZE):
                        t = BLOCK_SIZE - 1 - t_rev
                        lt = ls + t
                        if chunk_start + lt < T:
                            b_h_replay = b_Sj_noisy
                            for s in range(BLOCK_SIZE):
                                ls_s = ls + s
                                if s <= t and chunk_start + ls_s < T:
                                    b_g_s = tl.load(base_gn + ls_s * stride_gt).to(tl.float32)
                                    b_h_replay = b_h_replay * exp(b_g_s)
                                    if s < t:
                                        b_k_s = tl.load(base_kn + ls_s * stride_qk, mask=mask_k, other=0).to(tl.float32)
                                        b_v_s = tl.load(base_vn + ls_s * stride_vt, mask=mask_v, other=0).to(tl.float32)
                                        b_beta_s = tl.load(base_bn + ls_s * stride_gt).to(tl.float32)
                                        b_kTh_s = tl.sum(b_h_replay * b_k_s[:, None], axis=0)
                                        b_delta_s = b_beta_s * (b_v_s - b_kTh_s)
                                        b_h_replay = b_h_replay + b_k_s[:, None] * b_delta_s
                            b_hd = b_h_replay

                            off_qk = lt * stride_qk
                            off_v = lt * stride_vt
                            off_g = lt * stride_gt
                            b_q = tl.load(base_qn + off_qk, mask=mask_k, other=0).to(tl.float32)
                            b_k = tl.load(base_kn + off_qk, mask=mask_k, other=0).to(tl.float32)
                            b_v = tl.load(base_vn + off_v, mask=mask_v, other=0).to(tl.float32)
                            b_g = tl.load(base_gn + off_g).to(tl.float32)
                            b_beta = tl.load(base_bn + off_g).to(tl.float32)
                            b_do_t = tl.load(base_do + off_v, mask=mask_v, other=0).to(tl.float32)

                            b_kTh = tl.sum(b_hd * b_k[:, None], axis=0)
                            b_delta = b_beta * (b_v - b_kTh)
                            b_h_after = b_hd + b_k[:, None] * b_delta

                            b_dh_n += (b_q * scale)[:, None] * b_do_t[None, :]
                            b_dq_p = tl.sum(b_h_after * b_do_t[None, :], axis=1) * scale

                            b_dk_1 = tl.sum(b_dh_n * b_delta[None, :], axis=1)
                            b_ddelta = tl.sum(b_dh_n * b_k[:, None], axis=0)

                            b_dbeta_p = tl.sum(b_ddelta * (b_v - b_kTh))
                            b_dv_p = b_beta * b_ddelta
                            b_dkTh = -b_beta * b_ddelta

                            b_dk_2 = tl.sum(b_hd * b_dkTh[None, :], axis=1)
                            b_dh_n += b_k[:, None] * b_dkTh[None, :]

                            b_dg_p = tl.sum(b_dh_n * b_hd)
                            b_dh_n = b_dh_n * exp(b_g)

                            tl.atomic_add(base_dqn + off_qk, b_dq_p.to(base_dqn.dtype.element_ty), mask=mask_k)
                            tl.atomic_add(base_dkn + off_qk, (b_dk_1 + b_dk_2).to(base_dkn.dtype.element_ty), mask=mask_k)
                            tl.store(base_dvn + off_v, b_dv_p.to(base_dvn.dtype.element_ty), mask=mask_v)
                            tl.atomic_add(base_dgn + off_g, b_dg_p.to(base_dgn.dtype.element_ty))
                            tl.atomic_add(base_dbn + off_g, b_dbeta_p.to(base_dbn.dtype.element_ty))

                if j == 0:
                    b_dh_n_save = b_dh_n * b_noisy_first_scale
                else:
                    b_dh_n_save = b_dh_n
                if STORE_B:
                    p_bj = scratch_base_b + j * scratch_stride_j + o_k[:, None] * BV + o_bv[None, :]
                    tl.store(p_bj, b_dh_n_save.to(tl.float32), mask=mask_h)

                block_end_pos = chunk_start + ls + BLOCK_SIZE - 1
                if block_end_pos < T:
                    b_G_end = tl.load(base_Gc + (ls + BLOCK_SIZE - 1) * stride_gt).to(tl.float32)
                else:
                    b_G_end = b_G_prev
                b_delta_state = tl.zeros([K, BV], dtype=tl.float32)
                for t in range(BLOCK_SIZE):
                    lt = ls + t
                    if chunk_start + lt < T:
                        b_G_t = tl.load(base_Gc + lt * stride_gt).to(tl.float32)
                        b_e = exp(b_G_end - b_G_t)
                        b_kc_t = tl.load(base_kc + lt * stride_qk, mask=mask_k, other=0).to(tl.float32)
                        b_vnew_t = tl.load(base_vnew + lt * stride_vt, mask=mask_v, other=0).to(tl.float32)
                        b_delta_state += b_kc_t[:, None] * (b_vnew_t * b_e)[None, :]
                if j == 0:
                    b_alpha = exp(b_G_end)
                else:
                    b_alpha = exp(b_G_end - b_G_prev)
                b_S = b_alpha * b_S + b_delta_state
                b_G_prev = b_G_end

            b_G_last_valid = b_G_prev
            if i_v == 0:
                tl.store(alpha_total + (ckpt_flat * H + i_h), exp(b_G_last_valid))

            b_dS_next = tl.zeros([K, BV], dtype=tl.float32)
            b_dh_saved = tl.zeros([K, BV], dtype=tl.float32)
            b_P = tl.zeros([], dtype=tl.float32) + 1.0

            for j_rev in range(M):
                j = M - 1 - j_rev
                ls = j * BLOCK_SIZE

                ckpt_j = (j // CHECKPOINT_STRIDE) * CHECKPOINT_STRIDE
                ckpt_idx = j // CHECKPOINT_STRIDE
                p_sj = scratch_base_s + ckpt_idx * scratch_stride_j + o_k[:, None] * BV + o_bv[None, :]
                b_Sj = tl.load(p_sj, mask=mask_h).to(tl.float32)
                for r_offset in range(CHECKPOINT_STRIDE):
                    r_replay = ckpt_j + r_offset
                    if r_replay < j:
                        ls_replay = r_replay * BLOCK_SIZE
                        block_end_pos_replay = chunk_start + ls_replay + BLOCK_SIZE - 1
                        if block_end_pos_replay < T:
                            b_G_end_replay = tl.load(base_Gc + (ls_replay + BLOCK_SIZE - 1) * stride_gt).to(tl.float32)
                            if r_replay == 0:
                                b_alpha_replay = exp(b_G_end_replay)
                            else:
                                b_G_prev_replay = tl.load(base_Gc + ((r_replay - 1) * BLOCK_SIZE + BLOCK_SIZE - 1) * stride_gt).to(tl.float32)
                                b_alpha_replay = exp(b_G_end_replay - b_G_prev_replay)
                        else:
                            b_G_end_replay = tl.zeros([], dtype=tl.float32)
                            b_alpha_replay = tl.zeros([], dtype=tl.float32) + 1.0

                        b_delta_state_replay = tl.zeros([K, BV], dtype=tl.float32)
                        for t_replay in range(BLOCK_SIZE):
                            lt_replay = ls_replay + t_replay
                            if chunk_start + lt_replay < T:
                                b_G_t_replay = tl.load(base_Gc + lt_replay * stride_gt).to(tl.float32)
                                b_e_replay = exp(b_G_end_replay - b_G_t_replay)
                                b_kc_t_replay = tl.load(base_kc + lt_replay * stride_qk, mask=mask_k, other=0).to(tl.float32)
                                b_vnew_t_replay = tl.load(base_vnew + lt_replay * stride_vt, mask=mask_v, other=0).to(tl.float32)
                                b_delta_state_replay += b_kc_t_replay[:, None] * (b_vnew_t_replay * b_e_replay)[None, :]
                        b_Sj = b_alpha_replay * b_Sj + b_delta_state_replay
                if STORE_B:
                    p_bj = scratch_base_b + j * scratch_stride_j + o_k[:, None] * BV + o_bv[None, :]
                    b_bj = tl.load(p_bj, mask=mask_h).to(tl.float32)
                else:
                    if CAUSAL_MODE == 0:
                        b_dh_n_re = tl.zeros([K, BV], dtype=tl.float32)
                        for t_fwd_re in range(BLOCK_SIZE):
                            lt_fwd_re = ls + t_fwd_re
                            if chunk_start + lt_fwd_re < T:
                                off_qk_f_re = lt_fwd_re * stride_qk
                                off_v_f_re = lt_fwd_re * stride_vt

                                b_q_f_re = tl.load(base_qn + off_qk_f_re, mask=mask_k, other=0).to(tl.float32)
                                b_do_f_re = tl.load(base_do + off_v_f_re, mask=mask_v, other=0).to(tl.float32)
                                b_dh_n_re += (b_q_f_re * scale)[:, None] * b_do_f_re[None, :]

                        for t_rev_re in range(BLOCK_SIZE):
                            t_re = BLOCK_SIZE - 1 - t_rev_re
                            lt_re = ls + t_re
                            if chunk_start + lt_re < T:
                                off_qk_re = lt_re * stride_qk
                                off_g_re = lt_re * stride_gt
                                b_k_re = tl.load(base_kn + off_qk_re, mask=mask_k, other=0).to(tl.float32)
                                b_g_re = tl.load(base_gn + off_g_re).to(tl.float32)
                                b_beta_re = tl.load(base_bn + off_g_re).to(tl.float32)

                                b_ddelta_re = tl.sum(b_dh_n_re * b_k_re[:, None], axis=0)
                                b_dkTh_re = -b_beta_re * b_ddelta_re
                                b_dh_n_re += b_k_re[:, None] * b_dkTh_re[None, :]
                                b_dh_n_re = b_dh_n_re * exp(b_g_re)

                        if j == 0:
                            b_bj = b_dh_n_re * b_noisy_first_scale
                        else:
                            b_bj = b_dh_n_re
                    else:
                        b_dh_n_re = tl.zeros([K, BV], dtype=tl.float32)
                        for t_rev_re in range(BLOCK_SIZE):
                            t_re = BLOCK_SIZE - 1 - t_rev_re
                            lt_re = ls + t_re
                            if chunk_start + lt_re < T:
                                off_qk_re = lt_re * stride_qk
                                off_v_re = lt_re * stride_vt
                                off_g_re = lt_re * stride_gt
                                b_q_re = tl.load(base_qn + off_qk_re, mask=mask_k, other=0).to(tl.float32)
                                b_k_re = tl.load(base_kn + off_qk_re, mask=mask_k, other=0).to(tl.float32)
                                b_g_re = tl.load(base_gn + off_g_re).to(tl.float32)
                                b_beta_re = tl.load(base_bn + off_g_re).to(tl.float32)
                                b_do_t_re = tl.load(base_do + off_v_re, mask=mask_v, other=0).to(tl.float32)

                                b_dh_n_re += (b_q_re * scale)[:, None] * b_do_t_re[None, :]
                                b_ddelta_re = tl.sum(b_dh_n_re * b_k_re[:, None], axis=0)
                                b_dkTh_re = -b_beta_re * b_ddelta_re
                                b_dh_n_re += b_k_re[:, None] * b_dkTh_re[None, :]
                                b_dh_n_re = b_dh_n_re * exp(b_g_re)

                        if j == 0:
                            b_bj = b_dh_n_re * b_noisy_first_scale
                        else:
                            b_bj = b_dh_n_re

                block_end_pos_j = chunk_start + ls + BLOCK_SIZE - 1
                if block_end_pos_j < T:
                    b_G_end_j = tl.load(base_Gc + (ls + BLOCK_SIZE - 1) * stride_gt).to(tl.float32)
                else:
                    b_G_end_j = b_G_last_valid
                if j == 0:
                    b_alpha_j = exp(b_G_end_j)
                else:
                    if chunk_start + (j - 1) * BLOCK_SIZE + BLOCK_SIZE - 1 < T:
                        b_G_prev_end = tl.load(base_Gc + ((j - 1) * BLOCK_SIZE + BLOCK_SIZE - 1) * stride_gt).to(tl.float32)
                    else:
                        b_G_prev_end = b_G_last_valid
                    b_alpha_j = exp(b_G_end_j - b_G_prev_end)

                b_dS_local = b_dS_next - b_P * b_dh_saved
                b_dalpha = tl.sum(b_dS_local * b_Sj)
                b_dG_end_acc = b_dalpha * b_alpha_j

                for t in range(BLOCK_SIZE):
                    lt = ls + t
                    if chunk_start + lt < T:
                        off_qk = lt * stride_qk
                        off_v = lt * stride_vt
                        off_g = lt * stride_gt
                        b_G_t = tl.load(base_Gc + off_g).to(tl.float32)
                        b_e = exp(b_G_end_j - b_G_t)
                        b_kc_t = tl.load(base_kc + off_qk, mask=mask_k, other=0).to(tl.float32)
                        b_vnew_t = tl.load(base_vnew + off_v, mask=mask_v, other=0).to(tl.float32)

                        b_dkc = tl.sum(b_dS_local * (b_vnew_t * b_e)[None, :], axis=1)
                        tl.atomic_add(base_dkc + off_qk, b_dkc.to(base_dkc.dtype.element_ty), mask=mask_k)
                        b_dvnew = b_e * tl.sum(b_dS_local * b_kc_t[:, None], axis=0)
                        tl.store(base_dvnew + off_v, b_dvnew.to(base_dvnew.dtype.element_ty), mask=mask_v)
                        b_dE = tl.sum(b_dS_local * b_kc_t[:, None] * b_vnew_t[None, :])
                        b_dG_end_acc += b_e * b_dE
                        b_dG_t = -b_e * b_dE
                        tl.atomic_add(base_dGc + off_g, b_dG_t.to(base_dGc.dtype.element_ty))

                if block_end_pos_j < T:
                    tl.atomic_add(base_dGc + (ls + BLOCK_SIZE - 1) * stride_gt,
                                  b_dG_end_acc.to(base_dGc.dtype.element_ty))

                if j > 0:
                    prev_end_pos = chunk_start + (j - 1) * BLOCK_SIZE + BLOCK_SIZE - 1
                    if prev_end_pos < T:
                        prev_last = ((j - 1) * BLOCK_SIZE + BLOCK_SIZE - 1) * stride_gt
                        tl.atomic_add(base_dGc + prev_last,
                                      (-b_dalpha * b_alpha_j).to(base_dGc.dtype.element_ty))

                b_P = b_alpha_j * b_P
                b_dS_next = b_alpha_j * b_dS_next + b_bj

            p_dhc = (dh_chunks_out + (ckpt_flat * H + i_h) * (K * V)
                     + o_k[:, None] * V + o_v[None, :])
            tl.store(p_dhc, b_dS_next.to(p_dhc.dtype.element_ty), mask=mask_h)


@triton.jit(do_not_specialize=["NC"])
def two_stream_wy_noisy_bwd_split_b_kernel(
    dh_chunks_out, alpha_total,
    NC,
    cu_seqlens,
    chunk_offsets,
    doc_nc_arr,
    H: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BV: tl.constexpr,
    MAX_LOCAL_NC: tl.constexpr,
    IS_VARLEN: tl.constexpr = False,
):
    i_bh = tl.program_id(0)
    i_v = tl.program_id(1)

    if IS_VARLEN:
        i_n = i_bh // H
        i_h = i_bh % H
        chunk_base = tl.load(chunk_offsets + i_n).to(tl.int32)
        local_NC = tl.load(doc_nc_arr + i_n).to(tl.int32)
    else:
        i_b = i_bh // H
        i_h = i_bh % H
        chunk_base = i_b * NC
        local_NC = NC

    o_k = tl.arange(0, K)
    o_bv = tl.arange(0, BV)
    o_v = i_v * BV + o_bv
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_dh = tl.zeros([K, BV], dtype=tl.float32)
    for c_rev in range(MAX_LOCAL_NC):
        if c_rev < local_NC:
            c = local_NC - 1 - c_rev
            ckpt_flat = chunk_base + c
            p_dhc = (dh_chunks_out + (ckpt_flat * H + i_h) * (K * V)
                     + o_k[:, None] * V + o_v[None, :])
            b_L = tl.load(p_dhc, mask=mask_h, other=0).to(tl.float32)
            b_alpha = tl.load(alpha_total + (ckpt_flat * H + i_h)).to(tl.float32)
            b_dh = b_alpha * b_dh + b_L
            tl.store(p_dhc, b_dh.to(p_dhc.dtype.element_ty), mask=mask_h)


def two_stream_wy_noisy_bwd(
    q_n: torch.Tensor, k_n: torch.Tensor, v_n: torch.Tensor,
    g_n: torch.Tensor, beta_n: torch.Tensor,
    k_c: torch.Tensor, v_new_c: torch.Tensor, G_c: torch.Tensor,
    h_chunks: torch.Tensor, do: torch.Tensor,
    scale: float, block_size: int, chunk_size: int,
    causal_mode: int = 1,
    cu_seqlens: torch.LongTensor | None = None,
):
    """Full backward — O(2M) per chunk, prefix-recompute variant.

    Returns: dq_n, dk_n, dv_n, dg_n, dbeta_n,
             dh_chunks, dk_c, dv_new_c, dG_c
    """
    q_n, k_n, v_n = q_n.contiguous(), k_n.contiguous(), v_n.contiguous()
    g_n, beta_n = g_n.contiguous(), beta_n.contiguous()
    k_c, v_new_c, G_c = k_c.contiguous(), v_new_c.contiguous(), G_c.contiguous()
    h_chunks, do = h_chunks.contiguous(), do.contiguous()
    B, T, H, K = q_n.shape
    V = v_n.shape[-1]
    NC = h_chunks.shape[1]
    M = chunk_size // block_size
    BV = min(32, V)
    NUM_V_TILES = triton.cdiv(V, BV)

    is_varlen = cu_seqlens is not None

    dq_n = torch.zeros(B, T, H, K, dtype=torch.float32, device=q_n.device)
    dk_n = torch.zeros(B, T, H, K, dtype=torch.float32, device=q_n.device)
    dv_n = torch.zeros(B, T, H, V, dtype=torch.float32, device=q_n.device)
    dg_n = torch.zeros(B, T, H, dtype=torch.float32, device=q_n.device)
    dbeta_n = torch.zeros(B, T, H, dtype=torch.float32, device=q_n.device)
    dk_c_out = torch.zeros(B, T, H, K, dtype=torch.float32, device=q_n.device)
    dv_new_c_out = torch.zeros(B, T, H, V, dtype=torch.float32, device=q_n.device)
    dG_c_out = torch.zeros(B, T, H, dtype=torch.float32, device=q_n.device)
    dh_chunks_out = torch.empty(B, NC, H, K, V, dtype=torch.float32, device=q_n.device)

    if is_varlen:
        assert B == 1, "B must be 1 for cu_seqlens (doc packing)"
        N_seqs = len(cu_seqlens) - 1
        chunk_offsets_t, doc_nc_arr = _build_chunk_offsets(
            cu_seqlens, chunk_size, q_n.device)
        grid_dim0 = N_seqs * H
    else:
        chunk_offsets_t = doc_nc_arr = None
        grid_dim0 = B * H

    num_programs = grid_dim0 * NUM_V_TILES
    scratch_s = torch.empty(num_programs, M * K * BV, dtype=torch.float32, device=q_n.device)
    scratch_b = torch.empty(num_programs, M * K * BV, dtype=torch.float32, device=q_n.device)

    grid = (grid_dim0, NUM_V_TILES)
    two_stream_wy_noisy_bwd_kernel[grid](
        q_n, k_n, v_n, g_n, beta_n,
        k_c, v_new_c, G_c, h_chunks, do,
        dq_n, dk_n, dv_n, dg_n, dbeta_n,
        dk_c_out, dv_new_c_out, dG_c_out,
        dh_chunks_out, scratch_s, scratch_b,
        T, NC, scale,
        cu_seqlens, chunk_offsets_t, doc_nc_arr,
        NUM_V_TILES=NUM_V_TILES, B=B, H=H, K=K, V=V, BV=BV,
        BLOCK_SIZE=block_size, CHUNK_SIZE=chunk_size, M=M,
        CAUSAL_MODE=causal_mode,
        IS_VARLEN=is_varlen,
    )
    return dq_n, dk_n, dv_n, dg_n, dbeta_n, dh_chunks_out, dk_c_out, dv_new_c_out, dG_c_out


def two_stream_wy_noisy_bwd_split(
    q_n: torch.Tensor, k_n: torch.Tensor, v_n: torch.Tensor,
    g_n: torch.Tensor, beta_n: torch.Tensor,
    k_c: torch.Tensor, v_new_c: torch.Tensor, G_c: torch.Tensor,
    h_chunks: torch.Tensor, do: torch.Tensor,
    scale: float, block_size: int, chunk_size: int,
    causal_mode: int = 1,
    cu_seqlens: torch.LongTensor | None = None,
    parallel_groups: int = 0,
    checkpoint_stride: int = 0,
    bwd_bv: int | None = None,
    store_b: bool = True,
):
    """Split noisy backward path.

    Kernel A computes noisy grads, direct clean grads, and per-chunk local
    summaries in parallel chunk groups. Kernel B scans the affine summaries to
    recover the final `dh_chunks_out`.

    `store_b=True` stores all per-block `b_j` in scratch as a debug/safety
    fallback. The memory-efficient split mode recomputes `b_j` inside Phase B with noisy
    grad writes disabled, saving the `scratch_b` allocation. `bwd_bv` is a
    benchmark knob for the V tile width.
    """
    q_n, k_n, v_n = q_n.contiguous(), k_n.contiguous(), v_n.contiguous()
    g_n, beta_n = g_n.contiguous(), beta_n.contiguous()
    k_c, v_new_c, G_c = k_c.contiguous(), v_new_c.contiguous(), G_c.contiguous()
    h_chunks, do = h_chunks.contiguous(), do.contiguous()
    B, T, H, K = q_n.shape
    V = v_n.shape[-1]
    NC = h_chunks.shape[1]
    M = chunk_size // block_size
    BV = _resolve_improved_bwd_bv(V, bwd_bv)
    NUM_V_TILES = triton.cdiv(V, BV)
    parallel_groups = int(parallel_groups)
    if parallel_groups <= 0:
        parallel_groups = _auto_improved_bwd_parallel_groups(NC, cu_seqlens)
    P = max(1, min(parallel_groups, int(NC)))
    checkpoint_stride = int(checkpoint_stride)
    if checkpoint_stride <= 0:
        checkpoint_stride = _auto_improved_bwd_checkpoint_stride(M)
    checkpoint_stride = max(1, min(checkpoint_stride, M))
    num_checkpoints = triton.cdiv(M, checkpoint_stride)
    max_group_chunks = triton.cdiv(NC, P)
    max_local_nc = NC

    is_varlen = cu_seqlens is not None

    dq_n = torch.zeros(B, T, H, K, dtype=torch.float32, device=q_n.device)
    dk_n = torch.zeros(B, T, H, K, dtype=torch.float32, device=q_n.device)
    dv_n = torch.empty(B, T, H, V, dtype=torch.float32, device=q_n.device)
    dg_n = torch.zeros(B, T, H, dtype=torch.float32, device=q_n.device)
    dbeta_n = torch.zeros(B, T, H, dtype=torch.float32, device=q_n.device)
    dk_c_out = torch.zeros(B, T, H, K, dtype=torch.float32, device=q_n.device)
    dv_new_c_out = torch.empty(B, T, H, V, dtype=torch.float32, device=q_n.device)
    dG_c_out = torch.zeros(B, T, H, dtype=torch.float32, device=q_n.device)
    dh_chunks_out = torch.empty(B, NC, H, K, V, dtype=torch.float32, device=q_n.device)
    alpha_total = torch.empty(B, NC, H, dtype=torch.float32, device=q_n.device)

    if is_varlen:
        assert B == 1, "B must be 1 for cu_seqlens (doc packing)"
        N_seqs = len(cu_seqlens) - 1
        chunk_offsets_t, doc_nc_arr = _build_chunk_offsets_no_sync(
            cu_seqlens, chunk_size, q_n.device)
        grid_dim0_a = N_seqs * P * H
        grid_dim0_b = N_seqs * H
    else:
        chunk_offsets_t = doc_nc_arr = None
        grid_dim0_a = B * P * H
        grid_dim0_b = B * H

    num_programs_a = grid_dim0_a * NUM_V_TILES
    scratch_s = torch.empty(num_programs_a, num_checkpoints * K * BV, dtype=torch.float32, device=q_n.device)
    scratch_b_elems = M * K * BV if store_b else 1
    scratch_b = torch.empty(num_programs_a, scratch_b_elems, dtype=torch.float32, device=q_n.device)

    grid_a = (grid_dim0_a, NUM_V_TILES)
    two_stream_wy_noisy_bwd_split_a_kernel[grid_a](
        q_n, k_n, v_n, g_n, beta_n,
        k_c, v_new_c, G_c, h_chunks, do,
        dq_n, dk_n, dv_n, dg_n, dbeta_n,
        dk_c_out, dv_new_c_out, dG_c_out,
        dh_chunks_out, alpha_total,
        scratch_s, scratch_b,
        T, NC, scale,
        cu_seqlens, chunk_offsets_t, doc_nc_arr,
        NUM_V_TILES=NUM_V_TILES, B=B, H=H, K=K, V=V, BV=BV,
        BLOCK_SIZE=block_size, CHUNK_SIZE=chunk_size, M=M,
        P=P, MAX_GROUP_CHUNKS=max_group_chunks,
        CHECKPOINT_STRIDE=checkpoint_stride, NUM_CHECKPOINTS=num_checkpoints,
        CAUSAL_MODE=causal_mode,
        IS_VARLEN=is_varlen,
        STORE_B=store_b,
    )

    grid_b = (grid_dim0_b, NUM_V_TILES)
    two_stream_wy_noisy_bwd_split_b_kernel[grid_b](
        dh_chunks_out, alpha_total, NC,
        cu_seqlens, chunk_offsets_t, doc_nc_arr,
        H=H, K=K, V=V, BV=BV,
        MAX_LOCAL_NC=max_local_nc,
        IS_VARLEN=is_varlen,
    )

    return dq_n, dk_n, dv_n, dg_n, dbeta_n, dh_chunks_out, dk_c_out, dv_new_c_out, dG_c_out


def two_stream_wy_noisy_bwd_improved(
    q_n: torch.Tensor, k_n: torch.Tensor, v_n: torch.Tensor,
    g_n: torch.Tensor, beta_n: torch.Tensor,
    k_c: torch.Tensor, v_new_c: torch.Tensor, G_c: torch.Tensor,
    h_chunks: torch.Tensor, do: torch.Tensor,
    scale: float, block_size: int, chunk_size: int,
    causal_mode: int = 1,
    cu_seqlens: torch.LongTensor | None = None,
    split_enabled: bool | None = None,
    parallel_groups: int | None = None,
    checkpoint_stride: int | None = None,
    bwd_bv: int | None = None,
    store_b: bool | None = None,
):
    """Improved backward wrapper for benchmarking.

    By default this auto-selects the split optimized path when there are enough
    chunks to amortize the extra launch. Set
    `QWEN35_CHUNK_WY_IMPROVED_SPLIT=0` to benchmark the single-kernel improved
    path with only store-only empty allocations, or set it to `1` to force
    split.  `QWEN35_CHUNK_WY_IMPROVED_BV` selects the backward V tile width for
    the improved path.
    """
    NC = h_chunks.shape[1]
    M = chunk_size // block_size

    if split_enabled is None:
        split_env = os.environ.get("QWEN35_CHUNK_WY_IMPROVED_SPLIT")
        if split_env is None:
            if cu_seqlens is None:
                split_enabled = NC >= 4 and M > 2
            else:
                n_docs = len(cu_seqlens) - 1
                split_enabled = NC >= 4 * n_docs and M > 2
        else:
            split_enabled = split_env != "0"

    if split_enabled:
        if parallel_groups is None:
            p_env = os.environ.get("QWEN35_CHUNK_WY_IMPROVED_P")
            if p_env is None:
                parallel_groups = _auto_improved_bwd_parallel_groups(NC, cu_seqlens)
            else:
                parallel_groups = int(p_env)
        if checkpoint_stride is None:
            checkpoint_stride = int(os.environ.get("QWEN35_CHUNK_WY_IMPROVED_S", "0"))
        if bwd_bv is None:
            bwd_bv = int(os.environ.get("QWEN35_CHUNK_WY_IMPROVED_BV", "0"))
        if store_b is None:
            store_b = os.environ.get("QWEN35_CHUNK_WY_IMPROVED_STORE_B", "0") != "0"
        return two_stream_wy_noisy_bwd_split(
            q_n=q_n, k_n=k_n, v_n=v_n, g_n=g_n, beta_n=beta_n,
            k_c=k_c, v_new_c=v_new_c, G_c=G_c,
            h_chunks=h_chunks, do=do,
            scale=scale, block_size=block_size, chunk_size=chunk_size,
            causal_mode=causal_mode,
            cu_seqlens=cu_seqlens,
            parallel_groups=parallel_groups,
            checkpoint_stride=checkpoint_stride,
            bwd_bv=bwd_bv,
            store_b=store_b,
        )

    q_n, k_n, v_n = q_n.contiguous(), k_n.contiguous(), v_n.contiguous()
    g_n, beta_n = g_n.contiguous(), beta_n.contiguous()
    k_c, v_new_c, G_c = k_c.contiguous(), v_new_c.contiguous(), G_c.contiguous()
    h_chunks, do = h_chunks.contiguous(), do.contiguous()
    B, T, H, K = q_n.shape
    V = v_n.shape[-1]
    NC = h_chunks.shape[1]
    M = chunk_size // block_size
    if bwd_bv is None:
        bwd_bv = int(os.environ.get("QWEN35_CHUNK_WY_IMPROVED_BV", "0"))
    BV = _resolve_improved_bwd_bv(V, bwd_bv)
    NUM_V_TILES = triton.cdiv(V, BV)

    is_varlen = cu_seqlens is not None

    dq_n = torch.zeros(B, T, H, K, dtype=torch.float32, device=q_n.device)
    dk_n = torch.zeros(B, T, H, K, dtype=torch.float32, device=q_n.device)
    dv_n = torch.empty(B, T, H, V, dtype=torch.float32, device=q_n.device)
    dg_n = torch.zeros(B, T, H, dtype=torch.float32, device=q_n.device)
    dbeta_n = torch.zeros(B, T, H, dtype=torch.float32, device=q_n.device)
    dk_c_out = torch.zeros(B, T, H, K, dtype=torch.float32, device=q_n.device)
    dv_new_c_out = torch.empty(B, T, H, V, dtype=torch.float32, device=q_n.device)
    dG_c_out = torch.zeros(B, T, H, dtype=torch.float32, device=q_n.device)
    dh_chunks_out = torch.empty(B, NC, H, K, V, dtype=torch.float32, device=q_n.device)

    if is_varlen:
        assert B == 1, "B must be 1 for cu_seqlens (doc packing)"
        N_seqs = len(cu_seqlens) - 1
        chunk_offsets_t, doc_nc_arr = _build_chunk_offsets_no_sync(
            cu_seqlens, chunk_size, q_n.device)
        grid_dim0 = N_seqs * H
    else:
        chunk_offsets_t = doc_nc_arr = None
        grid_dim0 = B * H

    num_programs = grid_dim0 * NUM_V_TILES
    scratch_s = torch.empty(num_programs, M * K * BV, dtype=torch.float32, device=q_n.device)
    scratch_b = torch.empty(num_programs, M * K * BV, dtype=torch.float32, device=q_n.device)

    grid = (grid_dim0, NUM_V_TILES)
    two_stream_wy_noisy_bwd_kernel[grid](
        q_n, k_n, v_n, g_n, beta_n,
        k_c, v_new_c, G_c, h_chunks, do,
        dq_n, dk_n, dv_n, dg_n, dbeta_n,
        dk_c_out, dv_new_c_out, dG_c_out,
        dh_chunks_out, scratch_s, scratch_b,
        T, NC, scale,
        cu_seqlens, chunk_offsets_t, doc_nc_arr,
        NUM_V_TILES=NUM_V_TILES, B=B, H=H, K=K, V=V, BV=BV,
        BLOCK_SIZE=block_size, CHUNK_SIZE=chunk_size, M=M,
        CAUSAL_MODE=causal_mode,
        IS_VARLEN=is_varlen,
    )
    return dq_n, dk_n, dv_n, dg_n, dbeta_n, dh_chunks_out, dk_c_out, dv_new_c_out, dG_c_out


class TwoStreamWYNoisyTritonFunctionV2(torch.autograd.Function):
    """Autograd wrapper for the Triton two-stream noisy fwd+bwd kernels.

    Receives WY intermediates (v_new_c, G_c) and h_chunks as graph-connected
    inputs from ChunkBlockCausalGDAWithHFunctionV2, so gradients for clean
    parameters flow back through V2's existing WY chain rule automatically.

    Forward inputs:
        q_n, k_n, v_n, g_n, beta_n  — noisy inputs (requires_grad)
        k_c                           — clean key (requires_grad, shared with V2)
        v_new_c                       — WY intermediate from V2 (graph-connected)
        G_c                           — cumulative gate from V2 (graph-connected)
        h_chunks                      — chunk-start states from V2 (graph-connected)
        scale, block_size, chunk_size — non-differentiable config

    Backward outputs gradients for all 9 differentiable inputs above.
    Autograd accumulates dk_c with V2's dk, and routes dv_new_c/dG_c/dh_chunks
    back through V2's backward for the WY chain rule.
    """

    @staticmethod
    def forward(
        ctx,
        q_n, k_n, v_n, g_n, beta_n,
        k_c, v_new_c, G_c, h_chunks,
        scale, block_size, chunk_size, causal_mode,
        cu_seqlens=None,
    ):
        NC = h_chunks.shape[1]
        o = two_stream_wy_noisy_fwd(
            q_n=q_n, k_n=k_n, v_n=v_n, g_n=g_n, beta_n=beta_n,
            k_c=k_c, v_new_c=v_new_c, G_c=G_c, h_chunks=h_chunks[:, :NC],
            scale=scale, block_size=block_size, chunk_size=chunk_size,
            causal_mode=causal_mode,
            cu_seqlens=cu_seqlens,
        )
        ctx.save_for_backward(
            q_n, k_n, v_n, g_n, beta_n,
            k_c, v_new_c, G_c, h_chunks,
            *([cu_seqlens] if cu_seqlens is not None else []),
        )
        ctx.scale = scale
        ctx.block_size = block_size
        ctx.chunk_size = chunk_size
        ctx.causal_mode = causal_mode
        ctx._has_cu_seqlens = cu_seqlens is not None
        return o

    @staticmethod
    def backward(ctx, do):
        saved = ctx.saved_tensors
        if ctx._has_cu_seqlens:
            (q_n, k_n, v_n, g_n, beta_n,
             k_c, v_new_c, G_c, h_chunks, cu_seqlens) = saved
        else:
            (q_n, k_n, v_n, g_n, beta_n,
             k_c, v_new_c, G_c, h_chunks) = saved
            cu_seqlens = None

        NC = h_chunks.shape[1]
        do = do.contiguous()

        (dq_n, dk_n, dv_n, dg_n, dbeta_n,
         dh_chunks, dk_c_direct, dv_new_c, dG_c) = two_stream_wy_noisy_bwd(
            q_n=q_n, k_n=k_n, v_n=v_n, g_n=g_n, beta_n=beta_n,
            k_c=k_c, v_new_c=v_new_c, G_c=G_c,
            h_chunks=h_chunks[:, :NC], do=do,
            scale=ctx.scale, block_size=ctx.block_size, chunk_size=ctx.chunk_size,
            causal_mode=ctx.causal_mode,
            cu_seqlens=cu_seqlens,
        )

        return (
            dq_n.to(q_n.dtype), dk_n.to(k_n.dtype), dv_n.to(v_n.dtype),
            dg_n.to(g_n.dtype), dbeta_n.to(beta_n.dtype),
            dk_c_direct.to(k_c.dtype),
            dv_new_c.to(v_new_c.dtype),
            dG_c.to(G_c.dtype),
            dh_chunks.to(h_chunks.dtype).contiguous(),
            None, None, None, None, None,
        )


class TwoStreamWYNoisyTritonImprovedFunction(torch.autograd.Function):
    """Benchmark-isolated improved two-stream noisy autograd wrapper."""

    @staticmethod
    def forward(
        ctx,
        q_n, k_n, v_n, g_n, beta_n,
        k_c, v_new_c, G_c, h_chunks,
        scale, block_size, chunk_size, causal_mode,
        cu_seqlens=None,
        split_enabled=None,
        parallel_groups=None,
        checkpoint_stride=None,
        bwd_bv=None,
        store_b=None,
    ):
        NC = h_chunks.shape[1]
        o = two_stream_wy_noisy_fwd_improved(
            q_n=q_n, k_n=k_n, v_n=v_n, g_n=g_n, beta_n=beta_n,
            k_c=k_c, v_new_c=v_new_c, G_c=G_c, h_chunks=h_chunks[:, :NC],
            scale=scale, block_size=block_size, chunk_size=chunk_size,
            causal_mode=causal_mode,
            cu_seqlens=cu_seqlens,
        )
        ctx.save_for_backward(
            q_n, k_n, v_n, g_n, beta_n,
            k_c, v_new_c, G_c, h_chunks,
            *([cu_seqlens] if cu_seqlens is not None else []),
        )
        ctx.scale = scale
        ctx.block_size = block_size
        ctx.chunk_size = chunk_size
        ctx.causal_mode = causal_mode
        ctx._has_cu_seqlens = cu_seqlens is not None
        ctx.split_enabled = split_enabled
        ctx.parallel_groups = parallel_groups
        ctx.checkpoint_stride = checkpoint_stride
        ctx.bwd_bv = bwd_bv
        ctx.store_b = store_b
        return o

    @staticmethod
    def backward(ctx, do):
        saved = ctx.saved_tensors
        if ctx._has_cu_seqlens:
            (q_n, k_n, v_n, g_n, beta_n,
             k_c, v_new_c, G_c, h_chunks, cu_seqlens) = saved
        else:
            (q_n, k_n, v_n, g_n, beta_n,
             k_c, v_new_c, G_c, h_chunks) = saved
            cu_seqlens = None

        NC = h_chunks.shape[1]
        do = do.contiguous()

        (dq_n, dk_n, dv_n, dg_n, dbeta_n,
         dh_chunks, dk_c_direct, dv_new_c, dG_c) = two_stream_wy_noisy_bwd_improved(
            q_n=q_n, k_n=k_n, v_n=v_n, g_n=g_n, beta_n=beta_n,
            k_c=k_c, v_new_c=v_new_c, G_c=G_c,
            h_chunks=h_chunks[:, :NC], do=do,
            scale=ctx.scale, block_size=ctx.block_size, chunk_size=ctx.chunk_size,
            causal_mode=ctx.causal_mode,
            cu_seqlens=cu_seqlens,
            split_enabled=ctx.split_enabled,
            parallel_groups=ctx.parallel_groups,
            checkpoint_stride=ctx.checkpoint_stride,
            bwd_bv=ctx.bwd_bv,
            store_b=ctx.store_b,
        )

        return (
            dq_n.to(q_n.dtype), dk_n.to(k_n.dtype), dv_n.to(v_n.dtype),
            dg_n.to(g_n.dtype), dbeta_n.to(beta_n.dtype),
            dk_c_direct.to(k_c.dtype),
            dv_new_c.to(v_new_c.dtype),
            dG_c.to(G_c.dtype),
            dh_chunks.to(h_chunks.dtype).contiguous(),
            None, None, None, None, None, None, None, None, None, None,
        )
