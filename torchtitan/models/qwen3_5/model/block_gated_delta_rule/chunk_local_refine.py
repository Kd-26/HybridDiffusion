# -*- coding: utf-8 -*-
# Copyright (c) 2026 Yuchen Zhu

"""
Chunk-parallel local state refinement for block-causal Gated Delta Rule.

Given hidden states at chunk_size boundaries (from the WY-based chunked
kernel), refine them to block_size boundaries by running the raw gated
delta rule recurrence within each chunk **in parallel**.

Forward:
    Grid: (NC * B * H, NV)  where NC = T / chunk_size.
    Each Triton program handles one (chunk, batch, head, V-tile) and
    iterates chunk_size tokens, storing h at every block_size boundary.

    Sequential depth: chunk_size (within one chunk; all chunks parallel).

Backward:
    Grid: (NC * B * H, NV).
    Within each chunk, processes blocks in reverse order: loads h at block
    start from h_blocks, forward-replays BLOCK_SIZE tokens (storing h' in
    a small replay buffer), then backward-sweeps to compute per-token
    gradients and propagate dh.

    dk/dbeta are output with an NV leading dimension and reduced in the
    Python wrapper (same pattern as _bwd_triton).
    dg has shape [NV, B, T, H] (scalar gate) and is also NV-reduced.
    dv and dh_checkpoints write to non-overlapping V-slices directly.

Key difference from KDA: GDA has scalar gate g_t in R (not per-dim g_t in R^K).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=["T"])
def local_refine_fwd_kernel(
    k,              # [B, T, H, K]
    v,              # [B, T, H, V]
    g,              # [B, T, H]  (raw scalar gate, NOT cumsum'd)
    beta,           # [B, T, H]
    h_checkpoints,  # [B, NC, H, K, V]  (float32, from WY pipeline)
    h_out,          # [B, N_blocks, H, K, V]  output (float32)
    cu_seqlens,     # [N_docs+1] or None
    chunk_offsets,  # [N_docs] or None
    chunk_indices,  # [NC_total] or None
    T,
    B: tl.constexpr,
    NC: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_cnbh = tl.program_id(0)
    i_v = tl.program_id(1)

    BLOCKS_PER_CHUNK: tl.constexpr = CHUNK_SIZE // BLOCK_SIZE

    if IS_VARLEN:
        i_chunk_global = i_cnbh // H
        i_h = i_cnbh % H
        i_doc = tl.load(chunk_indices + i_chunk_global * 2).to(tl.int32)
        bos = tl.load(cu_seqlens + i_doc).to(tl.int32)
        eos = tl.load(cu_seqlens + i_doc + 1).to(tl.int32)
        T = eos - bos
        chunk_off = tl.load(chunk_offsets + i_doc).to(tl.int32)
        i_chunk_local = i_chunk_global - chunk_off
        ckpt_flat = i_chunk_global
        block_batch_off = bos // BLOCK_SIZE
    else:
        i_chunk_local = i_cnbh // (B * H)
        i_nh = i_cnbh % (B * H)
        i_n = i_nh // H
        i_h = i_nh % H
        bos = i_n * T
        ckpt_flat = i_n * NC + i_chunk_local
        block_batch_off = i_n * (T // BLOCK_SIZE)

    chunk_start = i_chunk_local * CHUNK_SIZE
    N_blocks = T // BLOCK_SIZE
    out_block_base = i_chunk_local * BLOCKS_PER_CHUNK

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    p_ckpt = (h_checkpoints
              + ckpt_flat * (H * K * V)
              + i_h * (K * V)
              + o_k[:, None] * V
              + o_v[None, :])
    b_h = tl.load(p_ckpt, mask=mask_h, other=0).to(tl.float32)

    s_k_t = H * K
    s_v_t = H * V
    s_g_t = H

    p_k_base = k + (bos * H + i_h) * K + o_k + chunk_start * s_k_t
    p_v_base = v + (bos * H + i_h) * V + o_v + chunk_start * s_v_t
    p_g_base = g + bos * H + i_h + chunk_start * s_g_t
    p_beta_base = beta + bos * H + i_h + chunk_start * s_g_t

    for local_block in range(BLOCKS_PER_CHUNK):
        for local_t in range(BLOCK_SIZE):
            global_t = chunk_start + local_block * BLOCK_SIZE + local_t
            if global_t < T:
                b_k = tl.load(p_k_base, mask=mask_k, other=0).to(tl.float32)
                b_v_val = tl.load(p_v_base, mask=mask_v, other=0).to(tl.float32)
                b_g = tl.load(p_g_base).to(tl.float32)
                b_beta = tl.load(p_beta_base).to(tl.float32)

                b_h *= exp(b_g)

                b_kTh = tl.sum(b_h * b_k[:, None], 0)
                b_delta = b_beta * (b_v_val - b_kTh)
                b_h += b_k[:, None] * b_delta

            p_k_base += s_k_t
            p_v_base += s_v_t
            p_g_base += s_g_t
            p_beta_base += s_g_t

        block_idx = out_block_base + local_block
        if block_idx < N_blocks:
            flat_idx = block_batch_off + block_idx
            p_out = (h_out
                     + flat_idx * (H * K * V)
                     + i_h * (K * V)
                     + o_k[:, None] * V
                     + o_v[None, :])
            tl.store(p_out, b_h.to(p_out.dtype.element_ty), mask=mask_h)


def local_refine_fwd(
    h_checkpoints: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
    block_size: int,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    """
    Refine chunk-boundary states to block_size-boundary states.

    Args:
        h_checkpoints: h at the START of each chunk, shape [B, NC, H, K, V].
        k: [B, T, H, K]
        v: [B, T, H, V]
        g: [B, T, H]  raw scalar gates
        beta: [B, T, H]
        chunk_size: size of each WY chunk (typically 64).
        block_size: desired output granularity (e.g. 4, 8, 16, 32).
        cu_seqlens: [N_docs+1] cumulative sequence lengths (None = no doc packing)

    Returns:
        h_out: [B, N_blocks, H, K, V]  (float32)
    """
    from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets

    B, T, H, K = k.shape
    V = v.shape[-1]
    N_blocks = T // block_size

    assert chunk_size % block_size == 0

    if cu_seqlens is not None:
        packed_t = int(cu_seqlens[-1].item())
        has_unused_tail = packed_t < T or packed_t % block_size != 0
    else:
        has_unused_tail = False
    if has_unused_tail:
        h_out = torch.zeros(B, N_blocks, H, K, V, device=k.device, dtype=torch.float32)
    else:
        h_out = torch.empty(B, N_blocks, H, K, V, device=k.device, dtype=torch.float32)

    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V))
    NV = triton.cdiv(V, BV)

    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size)
        NC = len(chunk_indices)
        grid = (NC * H, NV)
    else:
        chunk_indices = None
        chunk_offsets = None
        NC = triton.cdiv(T, chunk_size)
        grid = (NC * B * H, NV)

    assert h_checkpoints.shape[1] >= NC

    local_refine_fwd_kernel[grid](
        k=k, v=v, g=g, beta=beta,
        h_checkpoints=h_checkpoints,
        h_out=h_out,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        chunk_indices=chunk_indices,
        T=T, B=B, NC=NC, H=H, K=K, V=V,
        BK=BK, BV=BV,
        CHUNK_SIZE=chunk_size,
        BLOCK_SIZE=block_size,
        num_warps=1,
        num_stages=1,
    )
    return h_out


# ============================================================================
# Backward kernel
# ============================================================================

@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=["T"])
def local_refine_bwd_kernel(
    k,              # [B, T, H, K]
    v,              # [B, T, H, V]
    g,              # [B, T, H]  (raw scalar gate)
    beta,           # [B, T, H]
    h0,             # [N, H, K, V]  or None
    h_blocks,       # [B, N_blocks, H, K, V]  h at END of each block
    dh_blocks,      # [B, N_blocks, H, K, V]
    dk,             # [NV, B, T, H, K]  fp32
    dv,             # [B, T, H, V]      fp32
    dg,             # [NV, B, T, H]     fp32   (scalar gate)
    dbeta,          # [NV, B, T, H]     fp32
    dh_out,         # [B, NC, H, K, V]  fp32  (gradient at chunk starts)
    h_buf,          # [num_programs, BLOCK_SIZE, BK, BV]
    cu_seqlens,     # [N_docs+1] or None
    chunk_offsets,  # [N_docs] or None
    chunk_indices,  # [NC_total] or None
    T,
    B: tl.constexpr,
    NC: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Backward through the local-refine recurrence within one chunk.
    """
    i_cnbh = tl.program_id(0)
    i_v = tl.program_id(1)

    BLOCKS_PER_CHUNK: tl.constexpr = CHUNK_SIZE // BLOCK_SIZE

    T_global = T
    if IS_VARLEN:
        i_chunk_global = i_cnbh // H
        i_h = i_cnbh % H
        i_doc = tl.load(chunk_indices + i_chunk_global * 2).to(tl.int32)
        bos = tl.load(cu_seqlens + i_doc).to(tl.int32)
        eos = tl.load(cu_seqlens + i_doc + 1).to(tl.int32)
        T = eos - bos
        chunk_off = tl.load(chunk_offsets + i_doc).to(tl.int32)
        i_chunk_local = i_chunk_global - chunk_off
        ckpt_flat = i_chunk_global
        block_batch_off = bos // BLOCK_SIZE
        i_nh = i_doc * H + i_h
    else:
        i_chunk_local = i_cnbh // (B * H)
        i_nh = i_cnbh % (B * H)
        i_n = i_nh // H
        i_h = i_nh % H
        bos = i_n * T
        ckpt_flat = i_n * NC + i_chunk_local
        block_batch_off = i_n * (T // BLOCK_SIZE)

    chunk_start = i_chunk_local * CHUNK_SIZE
    N_blocks = T // BLOCK_SIZE
    out_block_base = i_chunk_local * BLOCKS_PER_CHUNK

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    o_hbuf = o_k[:, None] * BV + tl.arange(0, BV)[None, :]

    s_k_t = H * K
    s_v_t = H * V
    s_g_t = H

    p_k_base = k + (bos * H + i_h) * K + o_k + chunk_start * s_k_t
    p_v_base = v + (bos * H + i_h) * V + o_v + chunk_start * s_v_t
    p_g_base = g + bos * H + i_h + chunk_start * s_g_t
    p_beta_base = beta + bos * H + i_h + chunk_start * s_g_t

    s_hs_block = H * K * V
    p_hb_base = (h_blocks + block_batch_off * s_hs_block
                 + i_h * (K * V)
                 + o_k[:, None] * V + o_v[None, :])
    p_dhb_base = (dh_blocks + block_batch_off * s_hs_block
                  + i_h * (K * V)
                  + o_k[:, None] * V + o_v[None, :])

    p_dk_base = dk + i_v * (B * T_global * H * K) + (bos * H + i_h) * K + o_k + chunk_start * s_k_t
    p_dv_base = dv + (bos * H + i_h) * V + o_v + chunk_start * s_v_t
    p_dg_base = dg + i_v * (B * T_global * H) + bos * H + i_h + chunk_start * s_g_t
    p_dbeta_base = dbeta + i_v * (B * T_global * H) + bos * H + i_h + chunk_start * s_g_t

    p_hbuf_base = h_buf + i_cnbh * (BLOCK_SIZE * BK * BV)

    b_h0 = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        b_h0 = tl.load(
            h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :],
            mask=mask_h, other=0,
        ).to(tl.float32)

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)

    for _bi in range(BLOCKS_PER_CHUNK):
        block_j = BLOCKS_PER_CHUNK - 1 - _bi
        local_block_idx = out_block_base + block_j
        block_offset = block_j * BLOCK_SIZE

        if local_block_idx < N_blocks:
            b_dh += tl.load(
                p_dhb_base + local_block_idx * s_hs_block,
                mask=mask_h, other=0,
            ).to(tl.float32)

            b_h_ckpt = b_h0
            if local_block_idx > 0:
                b_h_ckpt = tl.load(
                    p_hb_base + (local_block_idx - 1) * s_hs_block,
                    mask=mask_h, other=0,
                ).to(tl.float32)

            b_h = b_h_ckpt
            for _lt in range(BLOCK_SIZE):
                global_t = chunk_start + block_offset + _lt
                if global_t < T:
                    b_k_fwd = tl.load(p_k_base + (block_offset + _lt) * s_k_t,
                                      mask=mask_k, other=0).to(tl.float32)
                    b_v_fwd = tl.load(p_v_base + (block_offset + _lt) * s_v_t,
                                      mask=mask_v, other=0).to(tl.float32)
                    b_g_fwd = tl.load(p_g_base + (block_offset + _lt) * s_g_t
                                      ).to(tl.float32)
                    b_beta_fwd = tl.load(p_beta_base + (block_offset + _lt) * s_g_t
                                         ).to(tl.float32)

                    b_h_prime = b_h * exp(b_g_fwd)
                    tl.store(p_hbuf_base + _lt * (BK * BV) + o_hbuf, b_h_prime)

                    b_kTh = tl.sum(b_h_prime * b_k_fwd[:, None], 0)
                    b_delta = b_beta_fwd * (b_v_fwd - b_kTh)
                    b_h = b_h_prime + b_k_fwd[:, None] * b_delta

            for _lt_rev in range(BLOCK_SIZE):
                _lt = BLOCK_SIZE - 1 - _lt_rev
                pos = block_offset + _lt
                global_t = chunk_start + pos

                if global_t < T:
                    b_k_cur = tl.load(p_k_base + pos * s_k_t,
                                      mask=mask_k, other=0).to(tl.float32)
                    b_v_val = tl.load(p_v_base + pos * s_v_t,
                                      mask=mask_v, other=0).to(tl.float32)
                    b_g_val = tl.load(p_g_base + pos * s_g_t).to(tl.float32)
                    b_beta_val = tl.load(p_beta_base + pos * s_g_t).to(tl.float32)

                    b_h_prime = tl.load(p_hbuf_base + _lt * (BK * BV) + o_hbuf)

                    b_kTh = tl.sum(b_h_prime * b_k_cur[:, None], 0)
                    b_u = b_v_val - b_kTh

                    b_du = tl.sum(b_dh * b_k_cur[:, None], 0)

                    b_dg_val = (
                        tl.sum(b_dh * b_h_prime) -
                        b_beta_val * tl.sum(b_du * b_kTh)
                    )

                    b_dv_val = b_beta_val * b_du
                    b_dbeta_val = tl.sum(b_du * b_u)

                    b_dk_val = b_beta_val * (
                        tl.sum(b_dh * b_u[None, :], 1) -
                        tl.sum(b_h_prime * b_du[None, :], 1)
                    )

                    tl.store(p_dk_base + pos * s_k_t,
                             b_dk_val.to(p_dk_base.dtype.element_ty), mask=mask_k)
                    tl.store(p_dv_base + pos * s_v_t,
                             b_dv_val.to(p_dv_base.dtype.element_ty), mask=mask_v)
                    tl.store(p_dg_base + pos * s_g_t,
                             b_dg_val.to(p_dg_base.dtype.element_ty))
                    tl.store(p_dbeta_base + pos * s_g_t,
                             b_dbeta_val.to(p_dbeta_base.dtype.element_ty))

                    b_dh = exp(b_g_val) * (
                        b_dh - b_k_cur[:, None] * b_dv_val[None, :])

    p_dh_out = (dh_out
                + ckpt_flat * (H * K * V)
                + i_h * (K * V)
                + o_k[:, None] * V
                + o_v[None, :])
    tl.store(p_dh_out, b_dh.to(p_dh_out.dtype.element_ty), mask=mask_h)


def local_refine_bwd(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    h0: torch.Tensor | None,
    h_blocks: torch.Tensor,
    dh_blocks: torch.Tensor,
    chunk_size: int,
    block_size: int,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward through the local-refine recurrence.
    """
    from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets

    B, T, H, K = k.shape
    V = v.shape[-1]

    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V))
    NV = triton.cdiv(V, BV)

    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size)
        NC = len(chunk_indices)
        num_programs = NC * H
        grid = (num_programs, NV)
    else:
        chunk_indices = None
        chunk_offsets = None
        NC = triton.cdiv(T, chunk_size)
        num_programs = NC * B * H
        grid = (num_programs, NV)

    dk = torch.zeros(NV, B, T, H, K, device=k.device, dtype=torch.float32)
    dv = torch.zeros(B, T, H, V, device=v.device, dtype=torch.float32)
    dg = torch.zeros(NV, B, T, H, device=g.device, dtype=torch.float32)
    dbeta = torch.zeros(NV, B, T, H, device=beta.device, dtype=torch.float32)
    dh_checkpoints = torch.empty(B, NC, H, K, V, device=k.device,
                                 dtype=torch.float32)

    h_buf = torch.empty(
        num_programs, block_size, BK, BV,
        device=k.device, dtype=torch.float32,
    )

    local_refine_bwd_kernel[grid](
        k=k, v=v, g=g, beta=beta,
        h0=h0, h_blocks=h_blocks,
        dh_blocks=dh_blocks,
        dk=dk, dv=dv, dg=dg, dbeta=dbeta,
        dh_out=dh_checkpoints,
        h_buf=h_buf,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        chunk_indices=chunk_indices,
        T=T, B=B, NC=NC, H=H, K=K, V=V,
        BK=BK, BV=BV,
        CHUNK_SIZE=chunk_size,
        BLOCK_SIZE=block_size,
        num_warps=1,
        num_stages=1,
    )

    dk = dk.sum(0)
    dg = dg.sum(0)
    dbeta = dbeta.sum(0)

    return dk, dv, dg, dbeta, dh_checkpoints
