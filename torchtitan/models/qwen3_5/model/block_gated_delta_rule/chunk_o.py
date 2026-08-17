# Copyright (c) 2026 Yuchen Zhu
# Portions copyright (c) 2023-2025 Songlin Yang, Yu Zhang
# Modified from Flash Linear Attention (FLA) for HybridDiffusion block-causal training.

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp
from fla.utils import is_nvidia_hopper, autotune_cache_kwargs, check_shared_mem

BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]
NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]


# ============================================================================
# Block-Causal Forward Kernel
# ============================================================================

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in BKV_LIST
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'BLOCK_SIZE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_block_causal_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    g_gamma,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CAUSAL_MODE: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Block-causal forward kernel for output computation.
    CAUSAL_MODE=0: block-causal (position i attends to all j where block(j) <= block(i))
    CAUSAL_MODE=1: token-causal (position i attends to all j where j <= i)
    Grid: (B*H, NV, NT) — B*H on dim-0 (x) to avoid 65535 limit.
    """
    i_bh, i_v, i_t = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, CHUNK_SIZE)
    else:
        NT = tl.cdiv(T, CHUNK_SIZE)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    T_chunk = tl.minimum(T, (i_t + 1) * CHUNK_SIZE)

    # offset calculation
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K*V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T_chunk, K), (H*K, 1), (i_t * CHUNK_SIZE, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T_chunk), (1, H*K), (i_k * BK, i_t * CHUNK_SIZE), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        b_o += tl.dot(b_q, b_h)
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A += tl.dot(b_q, b_k)

    # Compute positions and block indices
    o_t = i_t * CHUNK_SIZE + tl.arange(0, BT)  # Absolute positions within sequence
    local_pos = tl.arange(0, BT)  # Local positions within chunk
    m_t = o_t < T_chunk  # Valid positions mask

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T_chunk,), (H,), (i_t * CHUNK_SIZE,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))

        if CAUSAL_MODE == 0:
            block_idx = local_pos // BLOCK_SIZE
            block_end_local = (block_idx + 1) * BLOCK_SIZE - 1
            block_end_local = tl.minimum(block_end_local, CHUNK_SIZE - 1)
            block_end_abs = i_t * CHUNK_SIZE + block_end_local
            block_end_abs = tl.minimum(block_end_abs, T - 1)
            b_g_out = tl.load(g + block_end_abs * H, mask=m_t, other=0.)
        else:
            b_g_out = b_g

        b_o = b_o * exp(b_g_out)[:, None]
        b_A = b_A * exp(b_g_out[:, None] - b_g[None, :])

    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)

        if CAUSAL_MODE == 0:
            block_idx = local_pos // BLOCK_SIZE
            block_end_local = (block_idx + 1) * BLOCK_SIZE - 1
            block_end_local = tl.minimum(block_end_local, CHUNK_SIZE - 1)
            b_g_out = b_gamma * (block_end_local + 1)
        else:
            b_g_out = b_g

        b_o = b_o * exp(b_g_out)[:, None]
        b_A = b_A * exp(b_g_out[:, None] - b_g[None, :])

    if CAUSAL_MODE == 0:
        block_i = o_t[:, None] // BLOCK_SIZE
        block_j = o_t[None, :] // BLOCK_SIZE
        m_A = (block_i >= block_j) & (m_t[:, None] & m_t)
    else:
        m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)

    b_A = tl.where(m_A, b_A, 0)

    p_v = tl.make_block_ptr(v, (T_chunk, V), (H*V, 1), (i_t * CHUNK_SIZE, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o, (T_chunk, V), (H*V, 1), (i_t * CHUNK_SIZE, i_v * BV), (BT, BV), (1, 0))

    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


# ============================================================================
# Block-Causal Backward Kernels
# ============================================================================

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
    'USE_DW': lambda args: args['dw'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'BK', 'BV', 'BLOCK_SIZE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_block_causal_bwd_kernel_dqkwg(
    q,
    k,
    v,
    g,
    g_gamma,
    h,
    do,
    dh,
    dq,
    dk,
    dw,
    dv,
    dg,
    cu_seqlens,
    chunk_indices,
    scale,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CAUSAL_MODE: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_DW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Block-causal backward kernel for dq, dk, dw, dg computation.
    CAUSAL_MODE=0: block-causal, CAUSAL_MODE=1: token-causal.
    Grid: (B*H, NK, NT) — B*H on dim-0 (x) to avoid 65535 limit.
    """
    i_bh, i_k, i_t = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    all = B * T
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, CHUNK_SIZE)
    else:
        NT = tl.cdiv(T, CHUNK_SIZE)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    T_chunk = tl.minimum(T, (i_t + 1) * CHUNK_SIZE)

    # offset calculation
    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K*V
    dh += (i_tg * H + i_h).to(tl.int64) * K*V
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K

    # for delta rule only
    if USE_DW:
        dw += (bos * H + i_h) * K
        dv += (bos * H + i_h) * V

    if USE_G:
        dg += i_k * all * H
        b_dg_last = tl.zeros([1], dtype=tl.float32)
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_g_last = b_gamma * min(CHUNK_SIZE, T - i_t * CHUNK_SIZE)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32) if USE_DW else None

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v, (T_chunk, V), (H*V, 1), (i_t * CHUNK_SIZE, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T_chunk, V), (H*V, 1), (i_t * CHUNK_SIZE, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        if USE_G:
            b_dg_last += (tl.sum(b_h * b_dh))
        # [BT, BV] @ [BV, BT] -> [BT, BT]
        b_ds += tl.dot(b_do, tl.trans(b_v))
        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))
        if USE_DW:
            p_dv = tl.make_block_ptr(dv, (T_chunk, V), (H*V, 1), (i_t * CHUNK_SIZE, i_v * BV), (BT, BV), (1, 0))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_dw += tl.dot(b_dv.to(b_v.dtype), b_h.to(b_v.dtype))

    if USE_DW:
        p_dw = tl.make_block_ptr(dw, (T_chunk, K), (H*K, 1), (i_t * CHUNK_SIZE, i_k * BK), (BT, BK), (1, 0))
        tl.store(p_dw, -b_dw.to(p_dw.dtype.element_ty), boundary_check=(0, 1))

    p_q = tl.make_block_ptr(q, (T_chunk, K), (H*K, 1), (i_t * CHUNK_SIZE, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T_chunk, K), (H*K, 1), (i_t * CHUNK_SIZE, i_k * BK), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))

    p_dq = tl.make_block_ptr(dq, (T_chunk, K), (H*K, 1), (i_t * CHUNK_SIZE, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk, (T_chunk, K), (H*K, 1), (i_t * CHUNK_SIZE, i_k * BK), (BT, BK), (1, 0))

    # Compute positions and block indices
    o_t = i_t * CHUNK_SIZE + tl.arange(0, BT)
    local_pos = tl.arange(0, BT)
    m_t = o_t < T_chunk

    if CAUSAL_MODE == 0:
        block_i = o_t[:, None] // BLOCK_SIZE
        block_j = o_t[None, :] // BLOCK_SIZE
        m_A = (block_i >= block_j) & (m_t[:, None] & m_t)
    else:
        m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)

    if USE_G:
        b_dg = tl.zeros([BT], dtype=tl.float32)
        g += bos * H + i_h
        dg += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T_chunk,), (H,), (i_t * CHUNK_SIZE,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))

        if CAUSAL_MODE == 0:
            block_idx = local_pos // BLOCK_SIZE
            block_end_local = (block_idx + 1) * BLOCK_SIZE - 1
            block_end_local = tl.minimum(block_end_local, CHUNK_SIZE - 1)
            block_end_abs = i_t * CHUNK_SIZE + block_end_local
            block_end_abs = tl.minimum(block_end_abs, T - 1)
            b_g_out = tl.load(g + block_end_abs * H, mask=m_t, other=0.)
        else:
            b_g_out = b_g

        b_g_last = tl.load(g + (min(i_t * CHUNK_SIZE + CHUNK_SIZE, T) - 1) * H)
        b_dg_last *= exp(b_g_last)

        b_dq = b_dq * exp(b_g_out)[:, None] * scale
        b_dg_inter = tl.sum(b_dq * b_q, axis=1)

        b_dk = b_dk * tl.where(m_t, exp(-b_g + b_g_last), 0)[:, None]
        b_dg -= tl.sum(b_k * b_dk, axis=1)
        b_dg_last += tl.sum(b_dk * b_k)

        b_ds = tl.where(m_A, b_ds * exp(b_g_out[:, None] - b_g[None, :]), 0) * scale
        b_ds2 = b_ds * tl.dot(b_q, tl.trans(b_k))

        if CAUSAL_MODE == 0:
            b_dg_intra_block_end = tl.sum(b_ds2, axis=1)
            b_dg -= tl.sum(b_ds2, axis=0)
            b_dg_block_contrib = b_dg_inter + b_dg_intra_block_end
            chunk_len = tl.minimum(CHUNK_SIZE, T - i_t * CHUNK_SIZE)
            for blk_idx in tl.static_range(CHUNK_SIZE // BLOCK_SIZE):
                if blk_idx * BLOCK_SIZE < CHUNK_SIZE:
                    blk_mask = (block_idx == blk_idx) & m_t
                    blk_end = tl.minimum((blk_idx + 1) * BLOCK_SIZE - 1, chunk_len - 1)
                    contrib = tl.sum(tl.where(blk_mask, b_dg_block_contrib, 0.))
                    b_dg = tl.where(local_pos == blk_end, b_dg + contrib, b_dg)
        else:
            b_dg += b_dg_inter
            b_dg += tl.sum(b_ds2, axis=1)
            b_dg -= tl.sum(b_ds2, axis=0)

        b_ds = b_ds.to(b_k.dtype)
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q)
        p_dg = tl.make_block_ptr(dg, (T_chunk,), (H,), (i_t * CHUNK_SIZE,), (BT,), (0,))
        b_dg = tl.where(o_t < min(i_t * CHUNK_SIZE + CHUNK_SIZE, T) - 1, b_dg, b_dg + b_dg_last)
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))

    elif USE_G_GAMMA:
        if CAUSAL_MODE == 0:
            block_idx = local_pos // BLOCK_SIZE
            block_end_local = (block_idx + 1) * BLOCK_SIZE - 1
            block_end_local = tl.minimum(block_end_local, CHUNK_SIZE - 1)
            b_g_out = b_gamma * (block_end_local + 1)
        else:
            b_g_out = b_g

        b_dq = b_dq * exp(b_g_out)[:, None] * scale
        b_dk = b_dk * tl.where(m_t, exp(-b_g + b_g_last), 0)[:, None]
        b_ds = tl.where(m_A, b_ds * exp(b_g_out[:, None] - b_g[None, :]), 0) * scale
        b_ds = b_ds.to(b_k.dtype)
        # [BT, BK]
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q)
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    else:
        b_ds = tl.where(m_A, b_ds, 0)
        b_ds = b_ds.to(b_k.dtype)
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q) * scale
        b_dq *= scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'BK', 'BV', 'BLOCK_SIZE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_block_causal_bwd_kernel_dv(
    q,
    k,
    g,
    g_gamma,
    do,
    dv,
    dh,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CAUSAL_MODE: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Block-causal backward kernel for dv computation.
    CAUSAL_MODE=0: block-causal, CAUSAL_MODE=1: token-causal.
    Grid: (B*H, NV, NT) — B*H on dim-0 (x) to avoid 65535 limit.
    """
    i_bh, i_v, i_t = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, CHUNK_SIZE)
    else:
        NT = tl.cdiv(T, CHUNK_SIZE)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    T_chunk = tl.minimum(T, (i_t + 1) * CHUNK_SIZE)

    b_dv = tl.zeros([BT, BV], dtype=tl.float32)

    # offset calculation
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    do += (bos * H + i_h) * V
    dv += (bos * H + i_h) * V
    dh += (i_tg * H + i_h).to(tl.int64) * K*V

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k, (T_chunk, K), (H*K, 1), (i_t * CHUNK_SIZE, i_k * BK), (BT, BK), (1, 0))
        p_q = tl.make_block_ptr(q, (K, T_chunk), (1, H*K), (i_k * BK, i_t * CHUNK_SIZE), (BK, BT), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_k, b_q)
        p_dh = tl.make_block_ptr(dh, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dv += tl.dot(b_k, b_dh.to(b_k.dtype))

    # Compute positions and block indices
    o_t = i_t * CHUNK_SIZE + tl.arange(0, BT)
    local_pos = tl.arange(0, BT)
    m_t = o_t < T_chunk

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T_chunk,), (H,), (i_t * CHUNK_SIZE,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_last = tl.load(g + (min(i_t * CHUNK_SIZE + CHUNK_SIZE, T) - 1) * H)

        if CAUSAL_MODE == 0:
            block_idx = local_pos // BLOCK_SIZE
            block_end_local = (block_idx + 1) * BLOCK_SIZE - 1
            block_end_local = tl.minimum(block_end_local, CHUNK_SIZE - 1)
            block_end_abs = i_t * CHUNK_SIZE + block_end_local
            block_end_abs = tl.minimum(block_end_abs, T - 1)
            b_g_out = tl.load(g + block_end_abs * H, mask=m_t, other=0.)
        else:
            b_g_out = b_g

    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_g_last = b_gamma * min(CHUNK_SIZE, T - i_t * CHUNK_SIZE)

        if CAUSAL_MODE == 0:
            block_idx = local_pos // BLOCK_SIZE
            block_end_local = (block_idx + 1) * BLOCK_SIZE - 1
            block_end_local = tl.minimum(block_end_local, CHUNK_SIZE - 1)
            b_g_out = b_gamma * (block_end_local + 1)
        else:
            b_g_out = b_g

    if CAUSAL_MODE == 0:
        block_i = o_t[:, None] // BLOCK_SIZE
        block_j = o_t[None, :] // BLOCK_SIZE
        m_A = (block_i <= block_j) & (m_t[:, None] & m_t)
    else:
        m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)

    if USE_G or USE_G_GAMMA:
        b_A = tl.where(m_A, b_A * exp(b_g_out[None, :] - b_g[:, None]) * scale, 0).to(do.dtype.element_ty)
        b_dv *= tl.where(m_t, exp(-b_g + b_g_last), 0)[:, None]
    else:
        b_A = tl.where(m_A, b_A * scale, 0).to(do.dtype.element_ty)

    p_do = tl.make_block_ptr(do, (T_chunk, V), (H*V, 1), (i_t * CHUNK_SIZE, i_v * BV), (BT, BV), (1, 0))
    p_dv = tl.make_block_ptr(dv, (T_chunk, V), (H*V, 1), (i_t * CHUNK_SIZE, i_v * BV), (BT, BV), (1, 0))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dv += tl.dot(b_A.to(b_do.dtype), b_do)
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
    'USE_A': lambda args: args['A'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'BK', 'BV', 'BLOCK_SIZE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_block_causal_bwd_kernel_dv_local(
    q,
    k,
    g,
    g_gamma,
    A,
    do,
    dv,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CAUSAL_MODE: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_A: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Block-causal backward kernel for local dv computation.
    CAUSAL_MODE=0: block-causal, CAUSAL_MODE=1: token-causal.
    Grid: (B*H, NT) — B*H on dim-0 (x) to avoid 65535 limit.
    """
    i_bh, i_t = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    T_chunk = tl.minimum(T, (i_t + 1) * CHUNK_SIZE)

    # offset calculation
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    do += (bos * H + i_h) * V
    dv += (bos * H + i_h) * V

    # Compute positions and block indices
    o_t = i_t * CHUNK_SIZE + tl.arange(0, BT)
    local_pos = tl.arange(0, BT)
    m_t = o_t < T_chunk

    if USE_A:
        p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (BT, T_chunk), (1, H*BT), (0, i_t * CHUNK_SIZE), (BT, BT), (0, 1))
        b_A = tl.load(p_A, boundary_check=(0, 1))
    else:
        if USE_G:
            g += bos * H + i_h
            p_g = tl.make_block_ptr(g, (T_chunk,), (H,), (i_t * CHUNK_SIZE,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))

            if CAUSAL_MODE == 0:
                block_idx = local_pos // BLOCK_SIZE
                block_end_local = (block_idx + 1) * BLOCK_SIZE - 1
                block_end_local = tl.minimum(block_end_local, CHUNK_SIZE - 1)
                block_end_abs = i_t * CHUNK_SIZE + block_end_local
                block_end_abs = tl.minimum(block_end_abs, T - 1)
                b_g_out = tl.load(g + block_end_abs * H, mask=m_t, other=0.)
            else:
                b_g_out = b_g

        if USE_G_GAMMA:
            b_gamma = tl.load(g_gamma + i_h)
            b_g = b_gamma * (tl.arange(0, BT) + 1)

            if CAUSAL_MODE == 0:
                block_idx = local_pos // BLOCK_SIZE
                block_end_local = (block_idx + 1) * BLOCK_SIZE - 1
                block_end_local = tl.minimum(block_end_local, CHUNK_SIZE - 1)
                b_g_out = b_gamma * (block_end_local + 1)
            else:
                b_g_out = b_g

        b_A = tl.zeros([BT, BT], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_k = tl.make_block_ptr(k, (T_chunk, K), (H*K, 1), (i_t * CHUNK_SIZE, i_k * BK), (BT, BK), (1, 0))
            p_q = tl.make_block_ptr(q, (K, T_chunk), (1, H*K), (i_k * BK, i_t * CHUNK_SIZE), (BK, BT), (0, 1))

            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_A += tl.dot(b_k, b_q) * scale

        if USE_G or USE_G_GAMMA:
            b_A *= exp(b_g_out[None, :] - b_g[:, None])

    if CAUSAL_MODE == 0:
        block_i = o_t[:, None] // BLOCK_SIZE
        block_j = o_t[None, :] // BLOCK_SIZE
        m_A = (block_i <= block_j) & (m_t[:, None] & m_t)
    else:
        m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0).to(do.dtype.element_ty)

    for i_v in range(tl.cdiv(V, BV)):
        p_do = tl.make_block_ptr(do, (T_chunk, V), (H*V, 1), (i_t * CHUNK_SIZE, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv, (T_chunk, V), (H*V, 1), (i_t * CHUNK_SIZE, i_v * BV), (BT, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dv = tl.dot(b_A.to(b_do.dtype), b_do)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


# ============================================================================
# Block-Causal Aqk Computation Kernel
# ============================================================================

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_block_causal_aqk_kernel(
    q,
    k,
    g,
    Aqk,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CAUSAL_MODE: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Compute block/token-causal Aqk = Q @ K^T * gating * mask * scale.
    CAUSAL_MODE=0: block-causal, CAUSAL_MODE=1: token-causal.
    Grid: (B*H, NT) — B*H on dim-0 (x) to avoid 65535 limit.
    """
    i_bh, i_t = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos

    T_chunk = tl.minimum(T, (i_t + 1) * CHUNK_SIZE)

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T_chunk, K), (H * K, 1), (i_t * CHUNK_SIZE, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T_chunk), (1, H * K), (i_k * BK, i_t * CHUNK_SIZE), (BK, BT), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_q, b_k)

    o_t = i_t * CHUNK_SIZE + tl.arange(0, BT)
    local_pos = tl.arange(0, BT)
    m_t = o_t < T_chunk

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T_chunk,), (H,), (i_t * CHUNK_SIZE,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))

        if CAUSAL_MODE == 0:
            block_idx = local_pos // BLOCK_SIZE
            block_end_local = (block_idx + 1) * BLOCK_SIZE - 1
            block_end_local = tl.minimum(block_end_local, CHUNK_SIZE - 1)
            block_end_abs = i_t * CHUNK_SIZE + block_end_local
            block_end_abs = tl.minimum(block_end_abs, T - 1)
            b_g_out = tl.load(g + block_end_abs * H, mask=m_t, other=0.)
        else:
            b_g_out = b_g

        b_A = b_A * exp(b_g_out[:, None] - b_g[None, :])

    if CAUSAL_MODE == 0:
        block_i = o_t[:, None] // BLOCK_SIZE
        block_j = o_t[None, :] // BLOCK_SIZE
        m_A = (block_i >= block_j) & (m_t[:, None] & m_t)
    else:
        m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A * scale, 0.)

    p_Aqk = tl.make_block_ptr(Aqk + (bos * H + i_h) * BT, (T_chunk, BT), (H * BT, 1), (i_t * CHUNK_SIZE, 0), (BT, BT), (1, 0))
    tl.store(p_Aqk, b_A.to(p_Aqk.dtype.element_ty), boundary_check=(0, 1))


# ============================================================================
# Fused dA + dv_local Kernel
# ============================================================================

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_block_causal_bwd_kernel_dAv(
    v,
    A,
    do,
    dv,
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CAUSAL_MODE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Fused dA + dv_local: computes both in one kernel launch, sharing do/v loads.

    Given precomputed A (block-causal attention matrix with gating already applied):
      dv_local = A^T @ do        (local contribution to dv)
      dA       = do @ v^T * scale (raw gradient of attention matrix, block-masked)
    """
    i_bh, i_t = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos

    T_chunk = tl.minimum(T, (i_t + 1) * CHUNK_SIZE)

    o_t = i_t * CHUNK_SIZE + tl.arange(0, BT)
    m_t = o_t < T_chunk

    if CAUSAL_MODE == 0:
        block_ids = o_t // BLOCK_SIZE
        m_dA = (block_ids[:, None] >= block_ids[None, :]) & (m_t[:, None] & m_t[None, :])
        m_dv = (block_ids[:, None] <= block_ids[None, :]) & (m_t[:, None] & m_t[None, :])
    else:
        m_dA = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[None, :])
        m_dv = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t[None, :])

    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (BT, T_chunk), (1, H * BT), (0, i_t * CHUNK_SIZE), (BT, BT), (0, 1))
    b_AT = tl.load(p_A, boundary_check=(0, 1))
    b_AT = tl.where(m_dv, b_AT, 0).to(do.dtype.element_ty)

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        p_do = tl.make_block_ptr(do + (bos * H + i_h) * V, (T_chunk, V), (H * V, 1), (i_t * CHUNK_SIZE, i_v * BV), (BT, BV), (1, 0))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (V, T_chunk), (1, H * V), (i_v * BV, i_t * CHUNK_SIZE), (BV, BT), (0, 1))
        p_dv = tl.make_block_ptr(dv + (bos * H + i_h) * V, (T_chunk, V), (H * V, 1), (i_t * CHUNK_SIZE, i_v * BV), (BT, BV), (1, 0))

        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))

        # dv = A^T @ do
        b_dv = tl.dot(b_AT.to(b_do.dtype), b_do)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

        # dA += do @ v^T
        # Match operand dtypes explicitly; some backward paths produce fp32 do
        # while v remains bf16, and Triton tl.dot requires matching dtypes.
        b_dA += tl.dot(b_do, b_v.to(b_do.dtype))

    b_dA = tl.where(m_dA, b_dA * scale, 0.)
    p_dA = tl.make_block_ptr(dA + (bos * H + i_h) * BT, (T_chunk, BT), (H * BT, 1), (i_t * CHUNK_SIZE, 0), (BT, BT), (1, 0))
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


# ============================================================================
# Block-Causal Python Wrapper Functions
# ============================================================================

def chunk_block_causal_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    scale: float | None = None,
    block_size: int = 4,
    causal_mode: int = 0,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Block-causal forward output computation.

    Args:
        q: Query tensor of shape [B, T, H, K]
        k: Key tensor of shape [B, T, H, K]
        v: Value tensor of shape [B, T, H, V]
        h: Hidden state tensor of shape [B, NT, H, K, V]
        g: Gate tensor of shape [B, T, H] (optional)
        g_gamma: Gamma gate tensor of shape [H] (optional)
        scale: Scaling factor (default: 1/sqrt(K))
        block_size: Size of causal blocks (positions within block attend to full block)
        cu_seqlens: Cumulative sequence lengths for variable-length inputs
        chunk_size: Computational chunk size (default: 64)

    Returns:
        Output tensor of shape [B, T, H, V]
    """
    B, T, H, K, V = *q.shape, v.shape[-1]
    CHUNK_SIZE = chunk_size
    BT = 1 << (max(CHUNK_SIZE, 1) - 1).bit_length()
    chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE) if cu_seqlens is not None else None
    NT = triton.cdiv(T, CHUNK_SIZE) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.empty_like(v)
    def grid(meta): return (B * H, triton.cdiv(V, meta['BV']), NT)
    chunk_block_causal_fwd_kernel_o[grid](
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        CHUNK_SIZE=CHUNK_SIZE,
        BLOCK_SIZE=block_size,
        CAUSAL_MODE=causal_mode,
    )
    return o


def chunk_block_causal_bwd_dv(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    scale: float | None = None,
    block_size: int = 4,
    causal_mode: int = 0,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Block-causal backward computation for dv.
    """
    B, T, H, K, V = *k.shape, do.shape[-1]
    CHUNK_SIZE = chunk_size
    BT = 1 << (max(CHUNK_SIZE, 1) - 1).bit_length()
    chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE) if cu_seqlens is not None else None
    if check_shared_mem('hopper', k.device.index):
        CONST_TILING = 128
    elif check_shared_mem():
        CONST_TILING = 64
    else:
        CONST_TILING = 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NT = triton.cdiv(T, CHUNK_SIZE) if cu_seqlens is None else len(chunk_indices)
    NV = triton.cdiv(V, BV)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    dv = torch.empty_like(do)
    grid = (B * H, NV, NT)
    chunk_block_causal_bwd_kernel_dv[grid](
        q=q,
        k=k,
        g=g,
        g_gamma=g_gamma,
        do=do,
        dv=dv,
        dh=dh,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        CHUNK_SIZE=CHUNK_SIZE,
        BK=BK,
        BV=BV,
        BLOCK_SIZE=block_size,
        CAUSAL_MODE=causal_mode,
    )
    return dv


def chunk_block_causal_bwd_dv_local(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    A: torch.Tensor | None = None,
    scale: float = None,
    block_size: int = 4,
    causal_mode: int = 0,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    """
    Block-causal backward computation for local dv.
    """
    B, T, H, K, V = *k.shape, do.shape[-1]
    CHUNK_SIZE = chunk_size
    BT = 1 << (max(CHUNK_SIZE, 1) - 1).bit_length()
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
    if check_shared_mem('hopper', k.device.index):
        CONST_TILING = 128
    elif check_shared_mem():
        CONST_TILING = 64
    else:
        CONST_TILING = 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NT = triton.cdiv(T, CHUNK_SIZE) if cu_seqlens is None else len(chunk_indices)

    dv = torch.empty_like(do)
    grid = (B * H, NT)
    chunk_block_causal_bwd_kernel_dv_local[grid](
        q=q,
        k=k,
        g=g,
        g_gamma=g_gamma,
        A=A,
        do=do,
        dv=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        CHUNK_SIZE=CHUNK_SIZE,
        BK=BK,
        BV=BV,
        BLOCK_SIZE=block_size,
        CAUSAL_MODE=causal_mode,
    )
    return dv


def compute_block_causal_aqk(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float = 1.0,
    block_size: int = 4,
    causal_mode: int = 0,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Compute block-causal attention matrix Aqk = Q @ K^T * gating * block_mask * scale.

    Returns:
        Aqk of shape [B, T, H, BT] where BT = chunk_size.
    """
    B, T, H, K = q.shape
    CHUNK_SIZE = chunk_size
    BT = 1 << (max(CHUNK_SIZE, 1) - 1).bit_length()
    chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE) if cu_seqlens is not None else None
    NT = triton.cdiv(T, CHUNK_SIZE) if cu_seqlens is None else len(chunk_indices)

    Aqk = q.new_empty(B, T, H, BT, dtype=torch.float)
    grid = (B * H, NT)
    chunk_block_causal_aqk_kernel[grid](
        q=q, k=k, g=g, Aqk=Aqk,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        scale=scale, T=T, H=H, K=K, BT=BT,
        CHUNK_SIZE=CHUNK_SIZE,
        BLOCK_SIZE=block_size,
        CAUSAL_MODE=causal_mode,
    )
    return Aqk


def chunk_block_causal_bwd_dAv(
    v: torch.Tensor,
    do: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    block_size: int,
    causal_mode: int = 0,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused dA + dv_local: single kernel replacing both dA and dv_local computations.

    Computes:
        dv = A^T @ do       (local contribution to dv, using precomputed A)
        dA = do @ v^T * scale * block_mask   (attention gradient)

    Both share the same do/v tile loads, halving HBM reads compared to separate kernels.

    Returns:
        (dA, dv) where dA has shape [B, T, H, BT] and dv has shape [B, T, H, V].
    """
    B, T, H, V = do.shape
    CHUNK_SIZE = chunk_size
    BT = 1 << (max(CHUNK_SIZE, 1) - 1).bit_length()
    chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE) if cu_seqlens is not None else None
    NT = triton.cdiv(T, CHUNK_SIZE) if cu_seqlens is None else len(chunk_indices)

    dA = v.new_empty(B, T, H, BT, dtype=torch.float)
    dv = torch.empty_like(do)

    grid = (B * H, NT)
    chunk_block_causal_bwd_kernel_dAv[grid](
        v=v, A=A, do=do, dv=dv, dA=dA,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        scale=scale, T=T, H=H, V=V, BT=BT,
        CHUNK_SIZE=CHUNK_SIZE,
        BLOCK_SIZE=block_size,
        CAUSAL_MODE=causal_mode,
    )
    return dA, dv


def chunk_block_causal_bwd_dqkwg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    w: torch.Tensor | None = None,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    dv: torch.Tensor | None = None,
    scale: float | None = None,
    block_size: int = 4,
    causal_mode: int = 0,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Block-causal backward computation for dq, dk, dw, dg.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    CHUNK_SIZE = chunk_size
    BT = 1 << (max(CHUNK_SIZE, 1) - 1).bit_length()
    chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE) if cu_seqlens is not None else None
    NT = triton.cdiv(T, CHUNK_SIZE) if cu_seqlens is None else len(chunk_indices)

    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NK = triton.cdiv(K, BK)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dg = torch.empty(NK, *g.shape, dtype=torch.float32, device=g.device) if g is not None else None
    dw = torch.empty_like(w) if w is not None else None

    grid = (B * H, NK, NT)
    chunk_block_causal_bwd_kernel_dqkwg[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        h=h,
        do=do,
        dh=dh,
        dw=dw,
        dq=dq,
        dk=dk,
        dv=dv,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        CHUNK_SIZE=CHUNK_SIZE,
        BK=BK,
        BV=BV,
        BLOCK_SIZE=block_size,
        CAUSAL_MODE=causal_mode,
    )

    if dg is not None:
        dg = dg.sum(0)
    return dq, dk, dw, dg
