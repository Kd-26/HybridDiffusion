# Copyright (c) 2026 Yuchen Zhu

"""Fused two-stream Triton kernels for HybridDiffusion's small-block Route II.

``block_train_method='auto'`` selects this implementation for block sizes below
16 through the historical route name ``chunk_wy_triton_fla_style``. The name
refers to the clean stream's FLA-style chunkwise WY representation; the actual
execution is a fused clean/noisy two-stream forward and backward.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp

from .chunk_two_stream_wy import _build_chunk_meta_no_sync
from .chunk_local_refine import local_refine_fwd


@triton.jit(do_not_specialize=["T", "NC"])
def fla_style_cross_fwd_kernel(
    q_n,
    k_n,
    g_n,
    beta_n,
    k_c,
    v_new_c,
    G_c,
    h_chunks,
    o_cross,
    T,
    NC,
    scale,
    chunk_bos,
    chunk_start_arr,
    chunk_T_arr,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    M: tl.constexpr,
    CAUSAL_MODE: tl.constexpr = 1,
    IS_VARLEN: tl.constexpr = False,
):
    """Compute the clean-prefix cross term with a fused state recurrence.

    This is equivalent to running the noisy GDN recurrence with ``v_n = 0`` and
    the current clean-prefix state as initial state, while advancing the clean
    prefix state after each block.
    """
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

    p_h = (
        h_chunks
        + (ckpt_flat * H + i_h) * (K * V)
        + o_k[:, None] * V
        + o_v[None, :]
    )
    b_h_clean = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)

    base_qn = q_n + (bos * H + i_h) * K + o_k + chunk_start * stride_qk
    base_kn = k_n + (bos * H + i_h) * K + o_k + chunk_start * stride_qk
    base_gn = g_n + bos * H + i_h + chunk_start * stride_gt
    base_bn = beta_n + bos * H + i_h + chunk_start * stride_gt
    base_kc = k_c + (bos * H + i_h) * K + o_k + chunk_start * stride_qk
    base_vnew = v_new_c + (bos * H + i_h) * V + o_v + chunk_start * stride_vt
    base_Gc = G_c + bos * H + i_h + chunk_start * stride_gt
    base_o = o_cross + (bos * H + i_h) * V + o_v + chunk_start * stride_vt

    b_G_prev_end = tl.zeros([], dtype=tl.float32)
    b_first_doc_chunk = chunk_start == 0

    for j in range(M):
        ls = j * BLOCK_SIZE

        # The first noisy block of each document cannot see the clean initial
        # state.  Later blocks use the current clean-prefix state.
        if (j == 0) & b_first_doc_chunk:
            b_h_cross = tl.zeros([K, BV], dtype=tl.float32)
        else:
            b_h_cross = b_h_clean

        if CAUSAL_MODE == 1:
            for t in range(BLOCK_SIZE):
                lt = ls + t
                if chunk_start + lt < T:
                    off_qk = lt * stride_qk
                    off_v = lt * stride_vt
                    off_g = lt * stride_gt

                    b_q = tl.load(base_qn + off_qk, mask=mask_k, other=0).to(tl.float32)
                    b_k = tl.load(base_kn + off_qk, mask=mask_k, other=0).to(tl.float32)
                    b_g = tl.load(base_gn + off_g).to(tl.float32)
                    b_beta = tl.load(base_bn + off_g).to(tl.float32)

                    b_h_cross = b_h_cross * exp(b_g)
                    b_kTh = tl.sum(b_h_cross * b_k[:, None], axis=0)
                    b_delta = -b_beta * b_kTh
                    b_h_cross = b_h_cross + b_k[:, None] * b_delta

                    b_o = tl.sum(b_h_cross * (b_q * scale)[:, None], axis=0)
                    tl.store(base_o + off_v, b_o.to(base_o.dtype.element_ty), mask=mask_v)
        else:
            for t in range(BLOCK_SIZE):
                lt = ls + t
                if chunk_start + lt < T:
                    off_qk = lt * stride_qk
                    off_g = lt * stride_gt

                    b_k = tl.load(base_kn + off_qk, mask=mask_k, other=0).to(tl.float32)
                    b_g = tl.load(base_gn + off_g).to(tl.float32)
                    b_beta = tl.load(base_bn + off_g).to(tl.float32)

                    b_h_cross = b_h_cross * exp(b_g)
                    b_kTh = tl.sum(b_h_cross * b_k[:, None], axis=0)
                    b_delta = -b_beta * b_kTh
                    b_h_cross = b_h_cross + b_k[:, None] * b_delta

            for t in range(BLOCK_SIZE):
                lt = ls + t
                if chunk_start + lt < T:
                    off_qk = lt * stride_qk
                    off_v = lt * stride_vt
                    b_q = tl.load(base_qn + off_qk, mask=mask_k, other=0).to(tl.float32)
                    b_o = tl.sum(b_h_cross * (b_q * scale)[:, None], axis=0)
                    tl.store(base_o + off_v, b_o.to(base_o.dtype.element_ty), mask=mask_v)

        # Advance clean-prefix state with the matching clean block.
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
        b_h_clean = b_alpha * b_h_clean + b_delta_state
        b_G_prev_end = b_G_end


def two_stream_fla_style_cross_fwd_triton(
    q_n: torch.Tensor,
    k_n: torch.Tensor,
    g_n: torch.Tensor,
    beta_n: torch.Tensor,
    k_c: torch.Tensor,
    v_new_c: torch.Tensor,
    G_c: torch.Tensor,
    h_chunks: torch.Tensor,
    scale: float,
    block_size: int,
    chunk_size: int,
    causal_mode: int = 1,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    """Fused Triton forward for the FLA-style cross term."""
    q_n = q_n.contiguous()
    k_n = k_n.contiguous()
    g_n = g_n.contiguous()
    beta_n = beta_n.contiguous()
    k_c = k_c.contiguous()
    v_new_c = v_new_c.contiguous()
    G_c = G_c.contiguous()
    h_chunks = h_chunks.contiguous()

    B, T, H, K = q_n.shape
    V = v_new_c.shape[-1]
    NC = h_chunks.shape[1]
    M = chunk_size // block_size
    assert chunk_size % block_size == 0
    BV = min(64, V)
    o = torch.empty(B, T, H, V, dtype=v_new_c.dtype, device=q_n.device)

    is_varlen = cu_seqlens is not None
    if is_varlen:
        assert B == 1, "B must be 1 for cu_seqlens (doc packing)"
        chunk_bos, chunk_start_arr, chunk_T_arr = _build_chunk_meta_no_sync(
            cu_seqlens, chunk_size, q_n.device, NC
        )
        total_nc = chunk_bos.shape[0]
        grid = (total_nc * H, triton.cdiv(V, BV))
    else:
        chunk_bos = chunk_start_arr = chunk_T_arr = None
        grid = (NC * B * H, triton.cdiv(V, BV))

    fla_style_cross_fwd_kernel[grid](
        q_n, k_n, g_n, beta_n,
        k_c, v_new_c, G_c, h_chunks, o,
        T, NC, scale,
        chunk_bos, chunk_start_arr, chunk_T_arr,
        B=B, H=H, K=K, V=V, BV=BV,
        BLOCK_SIZE=block_size, CHUNK_SIZE=chunk_size, M=M,
        CAUSAL_MODE=causal_mode,
        IS_VARLEN=is_varlen,
    )
    return o


def two_stream_fla_style_full_fwd_triton(
    q_n: torch.Tensor,
    k_n: torch.Tensor,
    v_n: torch.Tensor,
    g_n: torch.Tensor,
    beta_n: torch.Tensor,
    k_c: torch.Tensor,
    v_new_c: torch.Tensor,
    G_c: torch.Tensor,
    h_chunks: torch.Tensor,
    scale: float,
    block_size: int,
    chunk_size: int,
    causal_mode: int = 1,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    """Dense full noisy forward for the FLA-style path.

    This reuses the established two-stream WY noisy recurrence.  The matching
    full backward below avoids separate local/cross noisy replays and routes
    gradients from the noisy stream back into the clean transition.
    """
    from .chunk_two_stream_wy import two_stream_wy_noisy_fwd_improved

    return two_stream_wy_noisy_fwd_improved(
        q_n=q_n,
        k_n=k_n,
        v_n=v_n,
        g_n=g_n,
        beta_n=beta_n,
        k_c=k_c,
        v_new_c=v_new_c,
        G_c=G_c,
        h_chunks=h_chunks,
        scale=scale,
        block_size=block_size,
        chunk_size=chunk_size,
        causal_mode=causal_mode,
        cu_seqlens=cu_seqlens,
    )


@triton.jit
def fla_style_varlen_clean_ckpt_kernel(
    k,
    v,
    g,
    beta,
    h_chunks,
    h_ckpts,
    chunk_bos,
    chunk_start_arr,
    chunk_T_arr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    M: tl.constexpr,
    NCK: tl.constexpr,
    CKPT_STRIDE: tl.constexpr,
):
    """Build compact per-packed-chunk clean-state checkpoints.

    Output layout is ``[total_nc, NCK, H, K, V]`` flattened.  This is the
    varlen analogue of dense strided checkpoints, but it does not require
    document starts to be aligned to ``CKPT_STRIDE * BLOCK_SIZE``.
    """
    i_cbh = tl.program_id(0)
    i_v = tl.program_id(1)
    i_gc = i_cbh // H
    i_h = i_cbh % H

    bos = tl.load(chunk_bos + i_gc).to(tl.int32)
    chunk_start = tl.load(chunk_start_arr + i_gc).to(tl.int32)
    T = tl.load(chunk_T_arr + i_gc).to(tl.int32)

    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    stride_k = H * K
    stride_v = H * V
    stride_g = H

    p_h = (
        h_chunks
        + (i_gc * H + i_h) * (K * V)
        + o_k[:, None] * V
        + o_v[None, :]
    )
    b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)

    base_k = k + (bos * H + i_h) * K + o_k + chunk_start * stride_k
    base_v = v + (bos * H + i_h) * V + o_v + chunk_start * stride_v
    base_g = g + bos * H + i_h + chunk_start * stride_g
    base_b = beta + bos * H + i_h + chunk_start * stride_g

    for seg in tl.range(0, NCK, 1, loop_unroll_factor=1):
        for r in tl.range(0, CKPT_STRIDE, 1, loop_unroll_factor=1):
            j = seg * CKPT_STRIDE + r
            if j < M:
                ls = j * BLOCK_SIZE
                for tt in tl.range(0, BLOCK_SIZE, 1, loop_unroll_factor=1):
                    lt = ls + tt
                    if chunk_start + lt < T:
                        b_k = tl.load(base_k + lt * stride_k, mask=mask_k, other=0).to(tl.float32)
                        b_v = tl.load(base_v + lt * stride_v, mask=mask_v, other=0).to(tl.float32)
                        b_g = tl.load(base_g + lt * stride_g).to(tl.float32)
                        b_beta = tl.load(base_b + lt * stride_g).to(tl.float32)

                        b_h = b_h * exp(b_g)
                        b_kth = tl.sum(b_h * b_k[:, None], axis=0)
                        b_delta = b_beta * (b_v - b_kth)
                        b_h = b_h + b_k[:, None] * b_delta

        p_out = (
            h_ckpts
            + ((i_gc * NCK + seg) * H + i_h) * (K * V)
            + o_k[:, None] * V
            + o_v[None, :]
        )
        tl.store(p_out, b_h, mask=mask_h)


def fla_style_varlen_clean_ckpt_fwd(
    h_chunks: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_bos: torch.Tensor,
    chunk_start_arr: torch.Tensor,
    chunk_T_arr: torch.Tensor,
    *,
    block_size: int,
    chunk_size: int,
    checkpoint_stride: int,
) -> torch.Tensor:
    """Return compact strided clean checkpoints for packed documents."""
    _, _, H, K = k.shape
    V = v.shape[-1]
    total_nc = chunk_bos.shape[0]
    M = chunk_size // block_size
    n_ckpts = M // checkpoint_stride
    h_out = torch.empty(total_nc * n_ckpts, H, K, V, device=k.device, dtype=torch.float32)
    BV = min(8, triton.next_power_of_2(V))
    grid = (total_nc * H, triton.cdiv(V, BV))
    fla_style_varlen_clean_ckpt_kernel[grid](
        k, v, g, beta, h_chunks, h_out,
        chunk_bos, chunk_start_arr, chunk_T_arr,
        H=H, K=K, V=V, BV=BV,
        BLOCK_SIZE=block_size, CHUNK_SIZE=chunk_size, M=M,
        NCK=n_ckpts, CKPT_STRIDE=checkpoint_stride,
        num_warps=1,
        num_stages=1,
    )
    return h_out


@triton.jit(do_not_specialize=["T", "NC"])
def fla_style_full_bwd_kernel(
    q_n,
    k_n,
    v_n,
    g_n,
    beta_n,
    k_c,
    v_new_c,
    G_c,
    h_chunks,
    h_ckpts,
    do,
    dq_n,
    dk_n,
    dv_n,
    dg_n,
    dbeta_n,
    dk_c,
    dv_new_c,
    dG_c,
    dh_chunks,
    T,
    NC,
    scale,
    chunk_bos,
    chunk_start_arr,
    chunk_T_arr,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    M: tl.constexpr,
    NCK: tl.constexpr,
    CKPT_STRIDE: tl.constexpr,
    CAUSAL_MODE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Merged full noisy backward plus clean-prefix transition backward.

    Compared with the earlier split local/cross prototype, this runs the noisy
    recurrence backward only once with the true clean-prefix initial state.  The
    resulting gradient with respect to that initial state is exactly the cross
    gradient that must be routed through the clean-prefix transition.
    """
    i_cbh = tl.program_id(0)
    i_v = tl.program_id(1)

    if IS_VARLEN:
        i_gc = i_cbh // H
        i_h = i_cbh % H
        bos = tl.load(chunk_bos + i_gc).to(tl.int32)
        chunk_start = tl.load(chunk_start_arr + i_gc).to(tl.int32)
        T = tl.load(chunk_T_arr + i_gc).to(tl.int32)
        ckpt_flat = i_gc
        is_doc_first_chunk = chunk_start == 0
    else:
        i_chunk = i_cbh // (B * H)
        i_bh = i_cbh % (B * H)
        i_b = i_bh // H
        i_h = i_bh % H
        bos = i_b * T
        chunk_start = i_chunk * CHUNK_SIZE
        ckpt_flat = i_b * NC + i_chunk
        is_doc_first_chunk = i_chunk == 0

    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    stride_qk = H * K
    stride_vt = H * V
    stride_gt = H

    p_h0 = (
        h_chunks
        + (ckpt_flat * H + i_h) * (K * V)
        + o_k[:, None] * V
        + o_v[None, :]
    )

    base_qn = q_n + (bos * H + i_h) * K + o_k + chunk_start * stride_qk
    base_kn = k_n + (bos * H + i_h) * K + o_k + chunk_start * stride_qk
    base_vn = v_n + (bos * H + i_h) * V + o_v + chunk_start * stride_vt
    base_gn = g_n + bos * H + i_h + chunk_start * stride_gt
    base_bn = beta_n + bos * H + i_h + chunk_start * stride_gt
    base_kc = k_c + (bos * H + i_h) * K + o_k + chunk_start * stride_qk
    base_vnew = v_new_c + (bos * H + i_h) * V + o_v + chunk_start * stride_vt
    base_Gc = G_c + bos * H + i_h + chunk_start * stride_gt
    base_do = do + (bos * H + i_h) * V + o_v + chunk_start * stride_vt

    base_dq = dq_n + (bos * H + i_h) * K + o_k + chunk_start * stride_qk
    base_dkn = dk_n + (bos * H + i_h) * K + o_k + chunk_start * stride_qk
    base_dvn = dv_n + (bos * H + i_h) * V + o_v + chunk_start * stride_vt
    base_dgn = dg_n + bos * H + i_h + chunk_start * stride_gt
    base_dbn = dbeta_n + bos * H + i_h + chunk_start * stride_gt
    base_dkc = dk_c + (bos * H + i_h) * K + o_k + chunk_start * stride_qk
    base_dvnew = dv_new_c + (bos * H + i_h) * V + o_v + chunk_start * stride_vt
    base_dGc = dG_c + bos * H + i_h + chunk_start * stride_gt

    b_dh_from_later = tl.zeros([K, BV], dtype=tl.float32)

    for _bj in tl.range(0, M, 1, loop_unroll_factor=1):
        j = M - 1 - _bj
        block_token = chunk_start + j * BLOCK_SIZE
        if block_token < T:
            seg = j // CKPT_STRIDE
            r_start = seg * CKPT_STRIDE
            b_s = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
            if seg > 0:
                ckpt_idx = ckpt_flat * NCK + (seg - 1)
                p_ck = (
                    h_ckpts
                    + (ckpt_idx * H + i_h) * (K * V)
                    + o_k[:, None] * V
                    + o_v[None, :]
                )
                b_s = tl.load(p_ck, mask=mask_h, other=0).to(tl.float32)

            b_G_prev = tl.zeros([], dtype=tl.float32)
            if r_start > 0:
                b_G_prev = tl.load(base_Gc + (r_start * BLOCK_SIZE - 1) * stride_gt).to(tl.float32)

            for r_off in tl.range(0, CKPT_STRIDE, 1, loop_unroll_factor=1):
                r = r_start + r_off
                if r < j:
                    r_ls = r * BLOCK_SIZE
                    r_end = chunk_start + r_ls + BLOCK_SIZE - 1
                    b_G_end_r = b_G_prev
                    if r_end < T:
                        b_G_end_r = tl.load(base_Gc + (r_ls + BLOCK_SIZE - 1) * stride_gt).to(tl.float32)

                    b_delta_r = tl.zeros([K, BV], dtype=tl.float32)
                    for tt in tl.range(0, BLOCK_SIZE, 1, loop_unroll_factor=1):
                        lt = r_ls + tt
                        if chunk_start + lt < T:
                            b_G_t = tl.load(base_Gc + lt * stride_gt).to(tl.float32)
                            b_e = exp(b_G_end_r - b_G_t)
                            b_kc = tl.load(base_kc + lt * stride_qk, mask=mask_k, other=0).to(tl.float32)
                            b_vn_clean = tl.load(base_vnew + lt * stride_vt, mask=mask_v, other=0).to(tl.float32)
                            b_delta_r += b_kc[:, None] * (b_vn_clean * b_e)[None, :]

                    if r == 0:
                        b_alpha_r = exp(b_G_end_r)
                    else:
                        b_alpha_r = exp(b_G_end_r - b_G_prev)
                    b_s = b_alpha_r * b_s + b_delta_r
                    b_G_prev = b_G_end_r

            b_init = b_s
            if is_doc_first_chunk & (j == 0):
                b_init = tl.zeros([K, BV], dtype=tl.float32)

            b_dh_noisy = tl.zeros([K, BV], dtype=tl.float32)
            if CAUSAL_MODE == 0:
                b_h_final = b_init
                for u in tl.range(0, BLOCK_SIZE, 1, loop_unroll_factor=1):
                    lt_u = j * BLOCK_SIZE + u
                    if chunk_start + lt_u < T:
                        b_ku = tl.load(base_kn + lt_u * stride_qk, mask=mask_k, other=0).to(tl.float32)
                        b_vu = tl.load(base_vn + lt_u * stride_vt, mask=mask_v, other=0).to(tl.float32)
                        b_gu = tl.load(base_gn + lt_u * stride_gt).to(tl.float32)
                        b_bu = tl.load(base_bn + lt_u * stride_gt).to(tl.float32)
                        b_h_pre_u = b_h_final * exp(b_gu)
                        b_kth_u = tl.sum(b_h_pre_u * b_ku[:, None], axis=0)
                        b_delta_u = b_bu * (b_vu - b_kth_u)
                        b_h_final = b_h_pre_u + b_ku[:, None] * b_delta_u

                for t_out in tl.range(0, BLOCK_SIZE, 1, loop_unroll_factor=1):
                    lt_o = j * BLOCK_SIZE + t_out
                    if chunk_start + lt_o < T:
                        b_qo = tl.load(base_qn + lt_o * stride_qk, mask=mask_k, other=0).to(tl.float32)
                        b_do_o = tl.load(base_do + lt_o * stride_vt, mask=mask_v, other=0).to(tl.float32)
                        b_dq_o = tl.sum(b_h_final * b_do_o[None, :], axis=1) * scale
                        tl.atomic_add(base_dq + lt_o * stride_qk, b_dq_o, mask=mask_k, sem="relaxed")
                        b_dh_noisy += (b_qo * scale)[:, None] * b_do_o[None, :]

            for _rt in tl.range(0, BLOCK_SIZE, 1, loop_unroll_factor=1):
                t = BLOCK_SIZE - 1 - _rt
                lt_t = j * BLOCK_SIZE + t
                if chunk_start + lt_t < T:
                    b_h_prev_t = b_init
                    for u in tl.range(0, BLOCK_SIZE, 1, loop_unroll_factor=1):
                        if u < t:
                            lt_u = j * BLOCK_SIZE + u
                            if chunk_start + lt_u < T:
                                b_ku = tl.load(base_kn + lt_u * stride_qk, mask=mask_k, other=0).to(tl.float32)
                                b_vu = tl.load(base_vn + lt_u * stride_vt, mask=mask_v, other=0).to(tl.float32)
                                b_gu = tl.load(base_gn + lt_u * stride_gt).to(tl.float32)
                                b_bu = tl.load(base_bn + lt_u * stride_gt).to(tl.float32)
                                b_h_pre_u = b_h_prev_t * exp(b_gu)
                                b_kth_u = tl.sum(b_h_pre_u * b_ku[:, None], axis=0)
                                b_delta_u = b_bu * (b_vu - b_kth_u)
                                b_h_prev_t = b_h_pre_u + b_ku[:, None] * b_delta_u

                    b_kt = tl.load(base_kn + lt_t * stride_qk, mask=mask_k, other=0).to(tl.float32)
                    b_vt = tl.load(base_vn + lt_t * stride_vt, mask=mask_v, other=0).to(tl.float32)
                    b_gt = tl.load(base_gn + lt_t * stride_gt).to(tl.float32)
                    b_bt = tl.load(base_bn + lt_t * stride_gt).to(tl.float32)
                    b_qt = tl.load(base_qn + lt_t * stride_qk, mask=mask_k, other=0).to(tl.float32)
                    b_dot = tl.load(base_do + lt_t * stride_vt, mask=mask_v, other=0).to(tl.float32)

                    b_h_pre = b_h_prev_t * exp(b_gt)
                    b_kth_t = tl.sum(b_h_pre * b_kt[:, None], axis=0)
                    b_uval_t = b_vt - b_kth_t
                    b_delta_t = b_bt * b_uval_t
                    b_h_after = b_h_pre + b_kt[:, None] * b_delta_t

                    if CAUSAL_MODE == 1:
                        b_dq = tl.sum(b_h_after * b_dot[None, :], axis=1) * scale
                        tl.atomic_add(base_dq + lt_t * stride_qk, b_dq, mask=mask_k, sem="relaxed")
                        b_dh_noisy += (b_qt * scale)[:, None] * b_dot[None, :]

                    b_ddelta = tl.sum(b_dh_noisy * b_kt[:, None], axis=0)
                    b_dk_outer = tl.sum(b_dh_noisy * b_delta_t[None, :], axis=1)
                    b_du = b_bt * b_ddelta
                    b_dv_t = b_du
                    b_dbeta_t = tl.sum(b_ddelta * b_uval_t)
                    b_dk_kth = -tl.sum(b_h_pre * b_du[None, :], axis=1)
                    b_dh_pre = b_dh_noisy - b_kt[:, None] * b_du[None, :]
                    b_dg_t = tl.sum(b_dh_pre * b_h_pre)

                    tl.atomic_add(base_dkn + lt_t * stride_qk, b_dk_outer + b_dk_kth, mask=mask_k, sem="relaxed")
                    tl.store(base_dvn + lt_t * stride_vt, b_dv_t, mask=mask_v)
                    tl.atomic_add(base_dbn + lt_t * stride_gt, b_dbeta_t, sem="relaxed")
                    tl.atomic_add(base_dgn + lt_t * stride_gt, b_dg_t, sem="relaxed")

                    b_dh_noisy = exp(b_gt) * b_dh_pre

            b_local_dh = b_dh_noisy
            if is_doc_first_chunk & (j == 0):
                b_local_dh = tl.zeros([K, BV], dtype=tl.float32)

            b_dnext = b_dh_from_later
            ls = j * BLOCK_SIZE
            b_G_prev_j = tl.zeros([], dtype=tl.float32)
            if j > 0:
                b_G_prev_j = tl.load(base_Gc + (ls - 1) * stride_gt).to(tl.float32)

            b_G_end = b_G_prev_j
            block_end_pos = chunk_start + ls + BLOCK_SIZE - 1
            if block_end_pos < T:
                b_G_end = tl.load(base_Gc + (ls + BLOCK_SIZE - 1) * stride_gt).to(tl.float32)

            if j == 0:
                b_alpha = exp(b_G_end)
            else:
                b_alpha = exp(b_G_end - b_G_prev_j)

            b_dalpha = tl.sum(b_dnext * b_s)
            b_dG_end_acc = b_dalpha * b_alpha
            if j > 0:
                tl.atomic_add(base_dGc + (ls - 1) * stride_gt, -b_dalpha * b_alpha, sem="relaxed")

            for tt in tl.range(0, BLOCK_SIZE, 1, loop_unroll_factor=1):
                lt = ls + tt
                if chunk_start + lt < T:
                    b_G_t = tl.load(base_Gc + lt * stride_gt).to(tl.float32)
                    b_e = exp(b_G_end - b_G_t)
                    b_kc_t = tl.load(base_kc + lt * stride_qk, mask=mask_k, other=0).to(tl.float32)
                    b_vnew_t = tl.load(base_vnew + lt * stride_vt, mask=mask_v, other=0).to(tl.float32)
                    b_z = b_vnew_t * b_e
                    b_dkc_t = tl.sum(b_dnext * b_z[None, :], axis=1)
                    b_dz = tl.sum(b_dnext * b_kc_t[:, None], axis=0)
                    b_dv_t_clean = b_dz * b_e
                    b_de_scaled = tl.sum(b_dz * b_vnew_t * b_e)

                    tl.atomic_add(base_dkc + lt * stride_qk, b_dkc_t, mask=mask_k, sem="relaxed")
                    tl.store(base_dvnew + lt * stride_vt, b_dv_t_clean, mask=mask_v)
                    b_dG_end_acc += b_de_scaled
                    tl.atomic_add(base_dGc + lt * stride_gt, -b_de_scaled, sem="relaxed")

            tl.atomic_add(base_dGc + (ls + BLOCK_SIZE - 1) * stride_gt, b_dG_end_acc, sem="relaxed")
            b_dh_from_later = b_local_dh + b_alpha * b_dnext

    p_dh = (
        dh_chunks
        + (ckpt_flat * H + i_h) * (K * V)
        + o_k[:, None] * V
        + o_v[None, :]
    )
    tl.store(p_dh, b_dh_from_later, mask=mask_h)


def two_stream_fla_style_full_bwd_triton(
    q_n: torch.Tensor,
    k_n: torch.Tensor,
    v_n: torch.Tensor,
    g_n: torch.Tensor,
    beta_n: torch.Tensor,
    k_c: torch.Tensor,
    v_c: torch.Tensor,
    g_c: torch.Tensor,
    beta_c: torch.Tensor,
    v_new_c: torch.Tensor,
    G_c: torch.Tensor,
    h_chunks: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    block_size: int,
    chunk_size: int,
    causal_mode: int = 1,
    checkpoint_stride: int | None = None,
    bwd_bv: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Merged full-noisy backward for the FLA-style path."""
    if causal_mode not in (0, 1):
        raise ValueError(f"Unsupported causal_mode: {causal_mode}")

    q_n = q_n.contiguous()
    k_n = k_n.contiguous()
    v_n = v_n.contiguous()
    g_n = g_n.contiguous()
    beta_n = beta_n.contiguous()
    k_c = k_c.contiguous()
    v_c = v_c.contiguous()
    g_c = g_c.contiguous()
    beta_c = beta_c.contiguous()
    v_new_c = v_new_c.contiguous()
    G_c = G_c.contiguous()
    h_chunks = h_chunks.contiguous()
    do = do.contiguous()

    B, T, H, K = q_n.shape
    V = v_n.shape[-1]
    NC = h_chunks.shape[1]
    M = chunk_size // block_size
    assert chunk_size % block_size == 0
    if bwd_bv is None or bwd_bv <= 0:
        bwd_bv = 32
    if bwd_bv not in (8, 16, 32, 64):
        raise ValueError(f"bwd_bv must be one of 8, 16, 32, 64; got {bwd_bv}")
    BV = min(bwd_bv, V)
    if checkpoint_stride is None or checkpoint_stride <= 0:
        # Block-size-aware replay defaults used by Route II.  A larger stride
        # stores fewer clean-state checkpoints and replays more blocks in
        # registers during backward. Values match the paper for block sizes
        # 1, 2, and 4. Internal block size 3 represents the release setting
        # described as B=4 in the paper and uses a separately validated stride.
        if block_size == 1:
            checkpoint_stride = 16
        elif block_size == 2:
            # Qwen3.5 production shape has 32 value heads.  S=2 creates
            # 16 clean-state checkpoints per chunk, pushing the compact
            # checkpoint tensor past 4B fp32 elements for bs=2 local batches.
            checkpoint_stride = 8
        elif block_size == 3:
            checkpoint_stride = 3
        elif block_size == 4:
            checkpoint_stride = 2
        else:
            checkpoint_stride = 8
    checkpoint_stride = min(max(1, checkpoint_stride), M)
    while M % checkpoint_stride != 0:
        checkpoint_stride -= 1
    is_varlen = cu_seqlens is not None
    n_ckpts = M // checkpoint_stride
    if is_varlen:
        assert B == 1, "B must be 1 for cu_seqlens (doc packing)"
        chunk_bos, chunk_start_arr, chunk_T_arr = _build_chunk_meta_no_sync(
            cu_seqlens, chunk_size, q_n.device, NC
        )
        h_ckpts = fla_style_varlen_clean_ckpt_fwd(
            h_chunks=h_chunks,
            k=k_c,
            v=v_c,
            g=g_c,
            beta=beta_c,
            chunk_bos=chunk_bos,
            chunk_start_arr=chunk_start_arr,
            chunk_T_arr=chunk_T_arr,
            block_size=block_size,
            chunk_size=chunk_size,
            checkpoint_stride=checkpoint_stride,
        )
        grid = (chunk_bos.shape[0] * H, triton.cdiv(V, BV))
    else:
        checkpoint_block_size = checkpoint_stride * block_size
        h_ckpts = local_refine_fwd(
            h_checkpoints=h_chunks,
            k=k_c,
            v=v_c,
            g=g_c,
            beta=beta_c,
            chunk_size=chunk_size,
            block_size=checkpoint_block_size,
            cu_seqlens=None,
        )
        chunk_bos = chunk_start_arr = chunk_T_arr = None
        grid = (NC * B * H, triton.cdiv(V, BV))

    dq_n = torch.zeros(B, T, H, K, dtype=torch.float32, device=q_n.device)
    dk_n = torch.zeros(B, T, H, K, dtype=torch.float32, device=q_n.device)
    dv_n = torch.empty(B, T, H, V, dtype=torch.float32, device=q_n.device)
    dg_n = torch.zeros(B, T, H, dtype=torch.float32, device=q_n.device)
    dbeta_n = torch.zeros(B, T, H, dtype=torch.float32, device=q_n.device)
    dk_c = torch.zeros(B, T, H, K, dtype=torch.float32, device=q_n.device)
    dv_new_c = torch.zeros(B, T, H, V, dtype=torch.float32, device=q_n.device)
    dG_c = torch.zeros(B, T, H, dtype=torch.float32, device=q_n.device)
    dh_chunks = torch.empty_like(h_chunks, dtype=torch.float32)

    fla_style_full_bwd_kernel[grid](
        q_n, k_n, v_n, g_n, beta_n,
        k_c, v_new_c, G_c, h_chunks, h_ckpts, do,
        dq_n, dk_n, dv_n, dg_n, dbeta_n,
        dk_c, dv_new_c, dG_c, dh_chunks,
        T, NC, scale,
        chunk_bos, chunk_start_arr, chunk_T_arr,
        B=B, H=H, K=K, V=V, BV=BV,
        BLOCK_SIZE=block_size, CHUNK_SIZE=chunk_size, M=M,
        NCK=n_ckpts, CKPT_STRIDE=checkpoint_stride,
        CAUSAL_MODE=causal_mode,
        IS_VARLEN=is_varlen,
        num_warps=4,
        num_stages=1,
    )
    return dq_n, dk_n, dv_n, dg_n, dbeta_n, dk_c, dv_new_c, dG_c, dh_chunks
