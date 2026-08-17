# Copyright (c) 2026 Yuchen Zhu
# Portions copyright (c) 2023-2025 Songlin Yang, Yu Zhang
# Modified from Flash Linear Attention (FLA) for HybridDiffusion block-causal training.
# Fused recurrent state extraction with block-boundary gradient checkpointing
# for the gated delta rule.

"""
Sequential recurrent state extraction used by explicit reference/fallback
routes. It extracts hidden states at training-block boundaries by running the
gated delta rule recurrence over the clean sequence.

Forward:
    A **Triton kernel** processes the full clean sequence token-by-token,
    running the gated delta rule recurrence and storing h at every
    ``block_size`` boundary.  Single kernel launch, no WY
    representation or chunk decomposition.

Backward:
    A **Triton kernel** that processes blocks in reverse order.  For each
    block it performs a two-phase operation:
      Phase 1 (forward replay): replay the recurrence from the stored
        checkpoint, storing h' (decayed state) at every position within
        the block into a temporary global-memory buffer.
      Phase 2 (backward sweep): iterate positions in reverse, loading h'
        from the buffer to compute dk, dv, dg, dbeta, and propagate dh.
    Memory:  O(N * B * H * K * V) for checkpoints plus
             O(block_size * BK * BV * NV * B * H) for the replay buffer.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp


def _has_unused_packed_tail(
    cu_seqlens: torch.LongTensor | None,
    total_t: int,
    block_size: int,
) -> bool:
    """True when packed storage has slots past the last real full block."""
    if cu_seqlens is None:
        return total_t % block_size != 0
    packed_t = int(cu_seqlens[-1].item())
    return packed_t < total_t or packed_t % block_size != 0


# ============================================================================
# Triton forward kernel
# ============================================================================

@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_gated_delta_state_fwd_kernel(
    k,          # [B, T, H, K]
    v,          # [B, T, H, V]
    g,          # [B, T, H]
    beta,       # [B, T, H]
    h0,         # [N, H, K, V]  or None
    h_states,   # [B, total_N, H, K, V]  output
    cu_seqlens,    # [N_docs+1] or None
    block_offsets, # [N_docs] or None
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Fused recurrent forward that stores h at every BLOCK_SIZE boundary.

    Grid: (N*H, NV) where N = B (fixed-len) or N_docs (varlen).
    Each program handles one (seq/doc, head) pair and one BV-wide slice of V.
    """
    i_nh, i_v = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        boh = tl.load(block_offsets + i_n).to(tl.int32)
    else:
        bos = i_n * T
        boh = i_n * (T // BLOCK_SIZE)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    stride_kv_t = H * K
    stride_v_t = H * V
    stride_g_t = H

    p_k_base = k + (bos * H + i_h) * K + o_k
    p_v_base = v + (bos * H + i_h) * V + o_v
    p_g_base = g + bos * H + i_h
    p_beta_base = beta + bos * H + i_h

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    N = T // BLOCK_SIZE

    for block_i in range(N):
        block_start = block_i * BLOCK_SIZE

        p_k = p_k_base + block_start * stride_kv_t
        p_v = p_v_base + block_start * stride_v_t
        p_g = p_g_base + block_start * stride_g_t
        p_beta = p_beta_base + block_start * stride_g_t

        for _local_t in range(BLOCK_SIZE):
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
            b_g = tl.load(p_g).to(tl.float32)
            b_beta = tl.load(p_beta).to(tl.float32)

            b_h *= exp(b_g)

            b_kTh = tl.sum(b_h * b_k[:, None], 0)
            b_delta = b_beta * (b_v - b_kTh)
            b_h += b_k[:, None] * b_delta

            p_k += stride_kv_t
            p_v += stride_v_t
            p_g += stride_g_t
            p_beta += stride_g_t

        p_hs = (h_states
                + (boh + block_i) * (H * K * V)
                + i_h * (K * V)
                + o_k[:, None] * V
                + o_v[None, :])
        tl.store(p_hs, b_h.to(p_hs.dtype.element_ty), mask=mask_h)


def _fwd_triton(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    h0: torch.Tensor | None,
    block_size: int,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    """Python wrapper that launches the Triton forward kernel."""
    B, T, H, K = k.shape
    V = v.shape[-1]
    N = T // block_size

    if _has_unused_packed_tail(cu_seqlens, T, block_size):
        h_states = torch.zeros(B, N, H, K, V, device=k.device, dtype=torch.float32)
    else:
        h_states = torch.empty(B, N, H, K, V, device=k.device, dtype=torch.float32)

    if cu_seqlens is not None:
        N_seqs = len(cu_seqlens) - 1
        block_offsets = (cu_seqlens[:-1] // block_size).to(torch.int32).contiguous()
    else:
        N_seqs = B
        block_offsets = None

    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V))
    NV = triton.cdiv(V, BV)

    grid = (N_seqs * H, NV)
    fused_recurrent_gated_delta_state_fwd_kernel[grid](
        k=k, v=v, g=g, beta=beta,
        h0=h0, h_states=h_states,
        cu_seqlens=cu_seqlens, block_offsets=block_offsets,
        T=T, B=B, H=H, K=K, V=V,
        BK=BK, BV=BV,
        BLOCK_SIZE=block_size,
        num_warps=1,
        num_stages=1,
    )
    return h_states


# ============================================================================
# Triton backward kernel
# ============================================================================

@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_DH0': lambda args: args['dh0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_gated_delta_state_bwd_kernel(
    # ── forward-saved inputs ──
    k,          # [B, T, H, K]
    v,          # [B, T, H, V]
    g,          # [B, T, H]
    beta,       # [B, T, H]
    h0,         # [N, H, K, V]  or None
    h_states,   # [B, total_N, H, K, V]
    # ── upstream gradient ──
    dh_states,  # [B, total_N, H, K, V]
    # ── output gradients ──
    dk,         # [B, T, H, K]  (atomic) or [NV, B, T, H, K] (legacy)
    dv,         # [B, T, H, V]
    dg,         # [B, T, H]    (atomic) or [NV, B, T, H]    (legacy)
    dbeta,      # [B, T, H]    (atomic) or [NV, B, T, H]    (legacy)
    dh0,        # [N, H, K, V]  or None
    # ── replay buffer ──
    h_buf,      # [NV * N * H, BLOCK_SIZE, BK, BV]  (flat float32)
    # ── varlen support ──
    cu_seqlens,    # [N_docs+1] or None
    block_offsets, # [N_docs] or None
    # ── dimensions ──
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_DH0: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_ATOMIC_DK: tl.constexpr = False,
):
    """
    Backward kernel for fused recurrent gated delta state extraction.

    Grid: (N*H, NV) where N = B (fixed-len) or N_docs (varlen).
    """
    i_nh, i_v = tl.program_id(0), tl.program_id(1)
    i_n = i_nh // H
    i_h = i_nh % H

    # T_global is the full sequence length (unchanged), T_local is per-doc
    T_global = T
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T_local = eos - bos
        boh = tl.load(block_offsets + i_n).to(tl.int32)
    else:
        bos = i_n * T_global
        T_local = T_global
        boh = i_n * (T_global // BLOCK_SIZE)
    N = T_local // BLOCK_SIZE

    # ── index vectors & masks ────────────────────────────────────────────
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    o_hbuf = o_k[:, None] * BV + tl.arange(0, BV)[None, :]

    # ── strides ──────────────────────────────────────────────────────────
    s_k_t = H * K
    s_v_t = H * V
    s_g_t = H

    # ── base pointers at (token bos, head i_h) ──────────────────────────
    p_k_base    = k    + (bos * H + i_h) * K + o_k
    p_v_base    = v    + (bos * H + i_h) * V + o_v
    p_g_base    = g    + bos * H + i_h
    p_beta_base = beta + bos * H + i_h

    # ── output base pointers ─────────────────────────────────────────────
    if USE_ATOMIC_DK:
        p_dk_base    = dk    + (bos * H + i_h) * K + o_k
        p_dg_base    = dg    + bos * H + i_h
        p_dbeta_base = dbeta + bos * H + i_h
    else:
        p_dk_base    = dk    + i_v * (B * T_global * H * K) + (bos * H + i_h) * K + o_k
        p_dg_base    = dg    + i_v * (B * T_global * H) + bos * H + i_h
        p_dbeta_base = dbeta + i_v * (B * T_global * H) + bos * H + i_h
    p_dv_base    = dv    + (bos * H + i_h) * V + o_v

    # ── h_states / dh_states base ────────────────────────────────────────
    s_hs_block = H * K * V
    p_hs_base  = h_states  + boh * (H * K * V) + i_h * (K * V) + o_k[:, None] * V + o_v[None, :]
    p_dhs_base = dh_states + boh * (H * K * V) + i_h * (K * V) + o_k[:, None] * V + o_v[None, :]

    # ── h_buf base for this program: [BLOCK_SIZE, BK, BV] ───────────────
    prog_id = i_v * tl.num_programs(0) + i_nh
    p_hbuf_base = h_buf + prog_id * (BLOCK_SIZE * BK * BV)

    # ── preload h0 (or zeros) for block_i == 0 ──────────────────────────
    b_h0_ckpt = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        b_h0_ckpt = tl.load(
            h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :],
            mask=mask_h, other=0,
        ).to(tl.float32)

    # ── dh accumulator (flows from later blocks to earlier ones) ─────────
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)

    # ====================================================================
    # Main loop: process blocks in reverse order
    # ====================================================================
    for _i in range(N):
        block_i = N - 1 - _i
        block_start = block_i * BLOCK_SIZE

        # ── accumulate boundary gradient from noisy pass ─────────────
        b_dh += tl.load(
            p_dhs_base + block_i * s_hs_block,
            mask=mask_h, other=0,
        ).to(tl.float32)

        # ── load checkpoint ──────────────────────────────────────────
        # block_i > 0  →  h_states[block_i − 1]
        # block_i == 0 →  h0 (or zeros)
        b_h_ckpt = b_h0_ckpt
        if block_i > 0:
            b_h_ckpt = tl.load(
                p_hs_base + (block_i - 1) * s_hs_block,
                mask=mask_h, other=0,
            ).to(tl.float32)

        # ────────────────────────────────────────────────────────────
        # Phase 1: forward replay — store h' at every position
        # ────────────────────────────────────────────────────────────
        b_h = b_h_ckpt
        for _lt in range(BLOCK_SIZE):
            pos = block_start + _lt

            b_k = tl.load(p_k_base + pos * s_k_t, mask=mask_k, other=0).to(tl.float32)
            b_v_val = tl.load(p_v_base + pos * s_v_t, mask=mask_v, other=0).to(tl.float32)
            b_g = tl.load(p_g_base + pos * s_g_t).to(tl.float32)
            b_beta = tl.load(p_beta_base + pos * s_g_t).to(tl.float32)

            # gated decay: h' = exp(g) · h_{t−1}
            b_h_prime = b_h * exp(b_g)

            # store h' to buffer
            tl.store(p_hbuf_base + _lt * (BK * BV) + o_hbuf, b_h_prime)

            # delta-rule update: h = h' + k ⊗ β·(v − k^T h')
            b_kTh = tl.sum(b_h_prime * b_k[:, None], 0)   # [BV]
            b_delta = b_beta * (b_v_val - b_kTh)            # [BV]
            b_h = b_h_prime + b_k[:, None] * b_delta         # [BK, BV]

        # barrier: ensure all h' writes are visible before phase 2
        # ────────────────────────────────────────────────────────────
        # Phase 2: backward sweep — compute gradients
        # ────────────────────────────────────────────────────────────
        for _lt_rev in range(BLOCK_SIZE):
            _lt = BLOCK_SIZE - 1 - _lt_rev
            pos = block_start + _lt

            # load inputs
            b_k = tl.load(p_k_base + pos * s_k_t, mask=mask_k, other=0).to(tl.float32)
            b_v_val = tl.load(p_v_base + pos * s_v_t, mask=mask_v, other=0).to(tl.float32)
            b_g = tl.load(p_g_base + pos * s_g_t).to(tl.float32)
            b_beta = tl.load(p_beta_base + pos * s_g_t).to(tl.float32)

            # load h' from replay buffer
            b_h_prime = tl.load(p_hbuf_base + _lt * (BK * BV) + o_hbuf)

            # recompute auxiliary from h'
            b_kTh = tl.sum(b_h_prime * b_k[:, None], 0)   # [BV]
            b_u = b_v_val - b_kTh                           # [BV]  (v − k^T h')

            # ── gradient computation ────────────────────────────────
            b_du = tl.sum(b_dh * b_k[:, None], 0)          # [BV] = k^T dh

            # dg = ⟨dh, h'⟩ − β · du · kTh'
            b_dg_val = tl.sum(b_dh * b_h_prime) - b_beta * tl.sum(b_du * b_kTh)

            # dv = β · du   [BV]
            b_dv_val = b_beta * b_du

            # dbeta = du · u   (scalar, partial sum over this V-tile)
            b_dbeta_val = tl.sum(b_du * b_u)

            # dk = β · (dh @ u − h' @ du)   [BK]
            b_dk_val = b_beta * (
                tl.sum(b_dh * b_u[None, :], 1) -
                tl.sum(b_h_prime * b_du[None, :], 1)
            )

            # store gradients
            if USE_ATOMIC_DK:
                tl.atomic_add(p_dk_base + pos * s_k_t,
                              b_dk_val.to(p_dk_base.dtype.element_ty), mask=mask_k)
                tl.atomic_add(p_dg_base + pos * s_g_t,
                              b_dg_val.to(p_dg_base.dtype.element_ty))
                tl.atomic_add(p_dbeta_base + pos * s_g_t,
                              b_dbeta_val.to(p_dbeta_base.dtype.element_ty))
            else:
                tl.store(p_dk_base + pos * s_k_t,
                         b_dk_val.to(p_dk_base.dtype.element_ty), mask=mask_k)
                tl.store(p_dg_base + pos * s_g_t,
                         b_dg_val.to(p_dg_base.dtype.element_ty))
                tl.store(p_dbeta_base + pos * s_g_t,
                         b_dbeta_val.to(p_dbeta_base.dtype.element_ty))
            tl.store(p_dv_base + pos * s_v_t,
                     b_dv_val.to(p_dv_base.dtype.element_ty), mask=mask_v)

            # propagate dh:  dh_{t−1} = exp(g) · (dh − k ⊗ dv)
            b_dh = exp(b_g) * (b_dh - b_k[:, None] * b_dv_val[None, :])

    # ── store dh0 ────────────────────────────────────────────────────────
    if STORE_DH0:
        tl.store(
            dh0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :],
            b_dh.to(tl.float32),
            mask=mask_h,
        )


def _bwd_triton(
    k: torch.Tensor,       # [B, T, H, K]
    v: torch.Tensor,       # [B, T, H, V]
    g: torch.Tensor,       # [B, T, H]
    beta: torch.Tensor,    # [B, T, H]
    h0: torch.Tensor | None,
    h_states: torch.Tensor,  # [B, N, H, K, V]
    dh_states: torch.Tensor, # [B, N, H, K, V]
    block_size: int,
    cu_seqlens: torch.LongTensor | None = None,
):
    """Python wrapper that launches the Triton backward kernel."""
    B, T, H, K = k.shape
    V = v.shape[-1]

    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V))
    NV = triton.cdiv(V, BV)

    dk    = torch.zeros(B, T, H, K, device=k.device, dtype=torch.float32)
    if _has_unused_packed_tail(cu_seqlens, T, block_size):
        dv = torch.zeros(B, T, H, V, device=v.device, dtype=torch.float32)
    else:
        dv = torch.empty(B, T, H, V, device=v.device, dtype=torch.float32)
    dg    = torch.zeros(B, T, H,    device=g.device, dtype=torch.float32)
    dbeta = torch.zeros(B, T, H,    device=beta.device, dtype=torch.float32)

    if cu_seqlens is not None:
        N_seqs = len(cu_seqlens) - 1
        block_offsets = (cu_seqlens[:-1] // block_size).to(torch.int32).contiguous()
    else:
        N_seqs = B
        block_offsets = None

    if h0 is not None:
        dh0 = torch.empty_like(h0, dtype=torch.float32)
    else:
        dh0 = None

    h_buf = torch.empty(
        NV * N_seqs * H, block_size, BK, BV,
        device=k.device, dtype=torch.float32,
    )

    grid = (N_seqs * H, NV)
    fused_recurrent_gated_delta_state_bwd_kernel[grid](
        k=k, v=v, g=g, beta=beta,
        h0=h0, h_states=h_states,
        dh_states=dh_states,
        dk=dk, dv=dv, dg=dg, dbeta=dbeta, dh0=dh0,
        h_buf=h_buf,
        cu_seqlens=cu_seqlens, block_offsets=block_offsets,
        T=T, B=B, H=H, K=K, V=V,
        BK=BK, BV=BV,
        BLOCK_SIZE=block_size,
        USE_ATOMIC_DK=True,
        num_warps=1,
        num_stages=1,
    )

    return dk, dv, dg, dbeta, dh0


# ============================================================================
# Manual backward (Python tensor ops) — kept for debugging / fallback
# ============================================================================

def _bwd_python(
    k: torch.Tensor,       # [B, T, H, K]
    v: torch.Tensor,       # [B, T, H, V]
    g: torch.Tensor,       # [B, T, H]
    beta: torch.Tensor,    # [B, T, H]
    h0: torch.Tensor | None,
    h_states: torch.Tensor,  # [B, N, H, K, V]
    dh_states: torch.Tensor, # [B, N, H, K, V]
    block_size: int,
):
    """
    Backward pass with block-boundary checkpointing (pure Python fallback).

    For each block (in reverse order):
      1. **Forward replay** from the stored checkpoint to recover h at every
         position within the block.
      2. **Backward sweep** using the replayed h to compute dk, dv, dg, dbeta
         and propagate dh to the previous block.

    The recurrence used in forward is (matching the Triton kernel):
        h' = exp(g_t) * h_{t-1}          (gated decay)
        h_t = h' + k_t ⊗ β_t*(v_t - k_t^T h')   (delta update)
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    N = T // block_size

    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    dg = torch.zeros_like(g)
    dbeta = torch.zeros_like(beta)

    # Accumulated gradient flowing to the previous block
    b_dh = torch.zeros(B, H, K, V, device=k.device, dtype=torch.float32)

    for i in range(N - 1, -1, -1):
        # Add boundary gradient from the noisy-pass path
        b_dh = b_dh + dh_states[:, i].float()

        s = i * block_size

        # ── checkpoint for this block ────────────────────────────────
        if i > 0:
            h_ckpt = h_states[:, i - 1].float()
        else:
            h_ckpt = (
                h0.float()
                if h0 is not None
                else torch.zeros(B, H, K, V, device=k.device, dtype=torch.float32)
            )

        # ── Phase 1: forward replay → collect h_{t-1} for every position ─
        h = h_ckpt
        h_prevs: list[torch.Tensor] = []     # h BEFORE decay+update at each t
        for t_off in range(block_size):
            h_prevs.append(h)
            pos = s + t_off
            k_t = k[:, pos].float()
            v_t = v[:, pos].float()
            g_t = g[:, pos].float()
            beta_t = beta[:, pos].float()

            h = h * torch.exp(g_t).unsqueeze(-1).unsqueeze(-1)        # h'
            kTh = torch.einsum('bhk,bhkv->bhv', k_t, h)              # k^T h'
            delta = beta_t.unsqueeze(-1) * (v_t - kTh)               # β*(v - k^T h')
            h = h + torch.einsum('bhk,bhv->bhkv', k_t, delta)        # h_new

        # ── Phase 2: backward sweep ──────────────────────────────────
        for t_off in range(block_size - 1, -1, -1):
            pos = s + t_off
            k_t = k[:, pos].float()                          # [B, H, K]
            v_t = v[:, pos].float()                          # [B, H, V]
            g_t = g[:, pos].float()                          # [B, H]
            beta_t = beta[:, pos].float()                    # [B, H]
            h_prev = h_prevs[t_off]                          # [B, H, K, V]

            # Recompute h' and auxiliary quantities
            exp_g = torch.exp(g_t).unsqueeze(-1).unsqueeze(-1)   # [B,H,1,1]
            h_prime = h_prev * exp_g                              # [B,H,K,V]
            kTh_prime = torch.einsum('bhk,bhkv->bhv', k_t, h_prime)  # [B,H,V]
            u = v_t - kTh_prime                                   # [B,H,V]

            # du = k^T @ dh_t
            du = torch.einsum('bhk,bhkv->bhv', k_t, b_dh)        # [B,H,V]

            # ---- dg ----
            dg_val = (
                (b_dh * h_prime).sum(dim=(-2, -1))
                - beta_t * (du * kTh_prime).sum(dim=-1)
            )
            dg[:, pos] = dg_val.to(dg.dtype)

            # ---- dv ----
            dv[:, pos] = (beta_t.unsqueeze(-1) * du).to(dv.dtype)

            # ---- dbeta ----
            dbeta[:, pos] = (du * u).sum(dim=-1).to(dbeta.dtype)

            # ---- dk ----
            dk_val = beta_t.unsqueeze(-1) * (
                (b_dh * u.unsqueeze(-2)).sum(dim=-1)
                - (h_prime * du.unsqueeze(-2)).sum(dim=-1)
            )
            dk[:, pos] = dk_val.to(dk.dtype)

            # ---- propagate dh to previous position ----
            dv_local = beta_t.unsqueeze(-1) * du
            b_dh = exp_g * (b_dh - torch.einsum('bhk,bhv->bhkv', k_t, dv_local))

    dh0 = b_dh if h0 is not None else None
    return dk, dv, dg, dbeta, dh0


# ============================================================================
# Custom _state_bwd_dhu for GDA state extraction (scalar g, dh_inject)
# ============================================================================

@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def _state_bwd_dhu_kernel(
    k,           # [B*T*H*K] contiguous
    w,           # [B*T*H*K] contiguous
    g,           # [B*T*H] contiguous (cumsum'd scalar gate)
    dh_inject,   # [B*NT*H*K*V] per-chunk dh injection
    dht,         # [B*H*K*V] or None
    dh0,         # [B*H*K*V] or None
    dh,          # [B*NT*H*K*V] output
    dv,          # [B*T*H*V] input
    dv2,         # [B*T*H*V] output
    cu_seqlens,    # [N_docs+1] or None
    chunk_offsets,  # [N_docs] or None
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    WY backward kernel for GDA state extraction dh propagation.

    Adapted from KDA's _state_bwd_dhu_kernel with scalar gating:
    - g is [B, T, H] (scalar per head) instead of [B, T, H, K]
    - Uses exp() instead of exp2()
    - dh_inject injects per-chunk gradients from block boundaries

    At each chunk step (reverse order):
      b_dh += dh_inject[chunk]
      store dh[chunk] = b_dh
      dv2 = k @ dh + dv_input
      b_dh = exp(g_last) * b_dh - w @ dv2
    """
    i_nh, i_v = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos = i_n * T
        boh = i_n * tl.cdiv(T, CHUNK_SIZE)
    NT = tl.cdiv(T, CHUNK_SIZE)

    b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

    k += (bos * H + i_h).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    g += (bos * H + i_h).to(tl.int64)       # scalar g: stride = 1 per (batch, time, head)
    dv += (bos * H + i_h).to(tl.int64) * V
    dv2 += (bos * H + i_h).to(tl.int64) * V
    dh += (boh * H + i_h).to(tl.int64) * K * V
    dh_inject += (boh * H + i_h).to(tl.int64) * K * V

    if USE_INITIAL_STATE:
        dh0 += i_nh * K * V
    if USE_FINAL_STATE_GRADIENT:
        dht += i_nh * K * V
        p_dht1 = tl.make_block_ptr(dht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_dh1 += tl.load(p_dht1, boundary_check=(0, 1))
        if K > 64:
            p_dht2 = tl.make_block_ptr(dht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_dh2 += tl.load(p_dht2, boundary_check=(0, 1))
        if K > 128:
            p_dht3 = tl.make_block_ptr(dht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_dh3 += tl.load(p_dht3, boundary_check=(0, 1))
        if K > 192:
            p_dht4 = tl.make_block_ptr(dht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_dh4 += tl.load(p_dht4, boundary_check=(0, 1))

    for i_t in range(NT - 1, -1, -1):
        i_t_int64 = i_t.to(tl.int64)

        # Inject dh from block boundaries
        p_inj1 = tl.make_block_ptr(dh_inject + i_t_int64 * H * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_dh1 += tl.load(p_inj1, boundary_check=(0, 1))
        if K > 64:
            p_inj2 = tl.make_block_ptr(dh_inject + i_t_int64 * H * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_dh2 += tl.load(p_inj2, boundary_check=(0, 1))
        if K > 128:
            p_inj3 = tl.make_block_ptr(dh_inject + i_t_int64 * H * K * V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_dh3 += tl.load(p_inj3, boundary_check=(0, 1))
        if K > 192:
            p_inj4 = tl.make_block_ptr(dh_inject + i_t_int64 * H * K * V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_dh4 += tl.load(p_inj4, boundary_check=(0, 1))

        # Store dh at chunk start
        p_dh1 = tl.make_block_ptr(dh + i_t_int64 * H * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_dh2 = tl.make_block_ptr(dh + i_t_int64 * H * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_dh3 = tl.make_block_ptr(dh + i_t_int64 * H * K * V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_dh4 = tl.make_block_ptr(dh + i_t_int64 * H * K * V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))

        # Load per-timestep and last-position cumsum'd gates
        chunk_start = i_t * CHUNK_SIZE
        chunk_end = min((i_t + 1) * CHUNK_SIZE, T)
        last_idx = chunk_end - 1
        b_g_last = tl.load(g + last_idx * H).to(tl.float32)
        o_t = chunk_start + tl.arange(0, BT)
        p_g = g + o_t * H
        m_t = o_t < chunk_end
        b_g = tl.load(p_g, mask=m_t, other=0).to(tl.float32)

        # Compute dv = k @ dh, then apply per-timestep gating, then add dv_input
        p_dv = tl.make_block_ptr(dv, (T, V), (H * V, 1), (chunk_start, i_v * BV), (BT, BV), (1, 0))
        p_dv2 = tl.make_block_ptr(dv2, (T, V), (H * V, 1), (chunk_start, i_v * BV), (BT, BV), (1, 0))

        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (chunk_start, 0), (BT, 64), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))

        if K > 64:
            p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (chunk_start, 64), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))
        if K > 128:
            p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (chunk_start, 128), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))
        if K > 192:
            p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (chunk_start, 192), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

        # Per-timestep gating: exp(g_last - g[t]) accounts for within-chunk decay
        b_dv *= tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
        b_dv += tl.load(p_dv, boundary_check=(0, 1))
        tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

        # Update dh: dh = exp(g_last) * dh - w @ dv2  (scalar gate broadcast)
        p_w = tl.make_block_ptr(w, (K, T), (1, H * K), (0, chunk_start), (64, BT), (0, 1))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_dh1 *= exp(b_g_last)
        b_dh1 -= tl.dot(b_w, b_dv.to(b_w.dtype))

        if K > 64:
            p_w = tl.make_block_ptr(w, (K, T), (1, H * K), (64, chunk_start), (64, BT), (0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_dh2 *= exp(b_g_last)
            b_dh2 -= tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(w, (K, T), (1, H * K), (128, chunk_start), (64, BT), (0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_dh3 *= exp(b_g_last)
            b_dh3 -= tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(w, (K, T), (1, H * K), (192, chunk_start), (64, BT), (0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_dh4 *= exp(b_g_last)
            b_dh4 -= tl.dot(b_w, b_dv.to(b_w.dtype))

    if USE_INITIAL_STATE:
        p_dh0 = tl.make_block_ptr(dh0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_dh02 = tl.make_block_ptr(dh0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh02, b_dh2.to(p_dh02.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_dh03 = tl.make_block_ptr(dh0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh03, b_dh3.to(p_dh03.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_dh04 = tl.make_block_ptr(dh0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh04, b_dh4.to(p_dh04.dtype.element_ty), boundary_check=(0, 1))


def _state_bwd_dhu(
    k: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    dh_inject: torch.Tensor,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    BT: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """
    WY backward for state extraction: propagate dh across chunks with injection.

    Args:
        k:  WY-transformed keys [B, T, H, K]
        w:  WY-transformed w [B, T, H, K]
        g:  cumsum'd scalar gates [B, T, H]
        dh_inject: per-chunk dh injection [B, NT, H, K, V]
        h0: initial state [B, H, K, V] or None
        dht: final state gradient [B, H, K, V] or None
        BT: chunk size
        cu_seqlens: [N_docs+1] cumulative sequence lengths (None = no doc packing)

    Returns:
        dh:  dh at start of each chunk [B, NT, H, K, V]
        dh0: gradient w.r.t. h0 or None
        dv2: updated dv [B, T, H, V]
    """
    from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets

    B, T, H, K = k.shape
    V = dh_inject.shape[-1]

    chunk_size = BT
    BT = 1 << (max(chunk_size, 1) - 1).bit_length()

    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        N_seqs = len(cu_seqlens) - 1
        NT = len(chunk_indices)
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size)
    else:
        N_seqs = B
        NT = triton.cdiv(T, chunk_size)
        chunk_offsets = None

    dh = k.new_empty(B, NT, H, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv = k.new_zeros(B, T, H, V)
    dv2 = torch.empty_like(dv)

    BV = min(32, triton.next_power_of_2(V))
    NV = triton.cdiv(V, BV)
    grid = (N_seqs * H, NV)
    _state_bwd_dhu_kernel[grid](
        k=k, w=w, g=g,
        dh_inject=dh_inject, dht=dht, dh0=dh0,
        dh=dh, dv=dv, dv2=dv2,
        cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets,
        T=T, H=H, K=K, V=V, BT=BT, CHUNK_SIZE=chunk_size, BV=BV,
        num_warps=4, num_stages=1,
    )
    return dh, dh0, dv2


# ============================================================================
# Flexible-BT wrapper for FLA's prepare_wy_repr_bwd
# ============================================================================

def _prepare_wy_repr_bwd_bt(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    BT: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Wrapper around FLA's prepare_wy_repr_bwd_kernel with flexible BT.
    FLA's Python wrapper hardcodes BT=64; the kernel accepts BT as constexpr.
    """
    from .wy_fast import prepare_wy_repr_bwd_kernel
    from fla.utils import check_shared_mem
    from fla.ops.utils import prepare_chunk_indices

    chunk_size = BT
    BT = 1 << (max(chunk_size, 1) - 1).bit_length()

    B, T, H, K = k.shape
    V = v.shape[-1]
    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)

    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        N_seqs = len(cu_seqlens) - 1
        NT = len(chunk_indices)
    else:
        chunk_indices = None
        N_seqs = B
        NT = triton.cdiv(T, chunk_size)

    dk = torch.empty_like(k, dtype=torch.float32)
    dv = torch.empty_like(v, dtype=torch.float32)
    dg = torch.empty_like(g, dtype=torch.float32) if g is not None else None
    db = torch.empty_like(beta, dtype=torch.float32)
    prepare_wy_repr_bwd_kernel[(N_seqs * H, NT)](
        k=k, v=v, beta=beta, g=g, A=A,
        dw=dw, du=du,
        dk=dk, dv=dv, db=db, dg=dg,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        T=T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV,
        CHUNK_SIZE=chunk_size,
    )
    return dk, dv, db, dg


# ============================================================================
# WY-based chunked backward for state extraction
# ============================================================================

def _bwd_wy_chunked(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    h0: torch.Tensor | None,
    dh_blocks: torch.Tensor,
    block_size: int,
    cu_seqlens: torch.LongTensor | None = None,
):
    """
    WY-based chunked backward for state extraction.

    Uses chunk_size = block_size so block boundaries align exactly
    with chunk boundaries, avoiding double-counting between local_refine
    and WY backward.

    Steps:
      1. Recompute WY intermediates at chunk_size=tbs
      2. Propagate dh across chunks via _state_bwd_dhu (with per-chunk injection)
      3. Compute per-token gradients using WY backward machinery
    """
    from .chunk_bwd_dqkwg import chunk_bwd_dqkwg
    from torchtitan.models.qwen3_5.model.block_gated_delta_rule.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    from .wy_fast import recompute_w_u_fwd
    from .chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from .cumsum import chunk_local_cumsum
    from .solve_tril import solve_tril

    B, T, H, K = k.shape
    V = v.shape[-1]
    chunk_size = block_size

    g_cumsum = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens)
    A = chunk_scaled_dot_kkt_fwd(
        k=k, g=g_cumsum, beta=beta, cu_seqlens=cu_seqlens,
        output_dtype=torch.float32, chunk_size=chunk_size,
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype, chunk_size=chunk_size)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g=g_cumsum,
                              cu_seqlens=cu_seqlens, chunk_size=chunk_size)
    h_chunks, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g_cumsum,
        initial_state=h0, output_final_state=False,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )

    dh, dh0_out, dv2 = _state_bwd_dhu(
        k=k, w=w, g=g_cumsum, dh_inject=dh_blocks,
        h0=h0, dht=None, BT=chunk_size,
        cu_seqlens=cu_seqlens,
    )

    q_zeros = torch.zeros_like(k)
    do_zeros = torch.zeros(B, T, H, V, device=k.device, dtype=k.dtype)

    _, dk_inter, dw, dg_inter = chunk_bwd_dqkwg(
        q=q_zeros, k=k, v=v_new, w=w, g=g_cumsum, h=h_chunks,
        dv=dv2, do=do_zeros, dh=dh,
        scale=1.0, chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )

    dk_wy, dv_out, db, dg_wy = _prepare_wy_repr_bwd_bt(
        k=k, v=v, beta=beta, g=g_cumsum, A=A,
        dw=dw, du=dv2, BT=chunk_size,
        cu_seqlens=cu_seqlens,
    )

    dk_total = dk_inter.float() + dk_wy
    dg_total = dg_inter + dg_wy
    dg_total = chunk_local_cumsum(dg_total, chunk_size=chunk_size, reverse=True,
                                   cu_seqlens=cu_seqlens)

    return dk_total, dv_out, dg_total, db, dh0_out


# ============================================================================
# Autograd wrapper
# ============================================================================

class FusedRecurrentGatedDeltaStateFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, k, v, g, beta, h0, block_size, cu_seqlens):
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()
        beta = beta.contiguous()
        if h0 is not None:
            h0 = h0.contiguous()

        h_states = _fwd_triton(k, v, g, beta, h0, block_size, cu_seqlens=cu_seqlens)

        if h0 is not None:
            ctx.save_for_backward(k, v, g, beta, h0, h_states)
        else:
            ctx.save_for_backward(k, v, g, beta, h_states)
        ctx.has_initial_state = h0 is not None
        ctx.block_size = block_size
        ctx.cu_seqlens = cu_seqlens
        return h_states

    @staticmethod
    def backward(ctx, dh_states):
        if ctx.has_initial_state:
            k, v, g, beta, h0, h_states = ctx.saved_tensors
        else:
            k, v, g, beta, h_states = ctx.saved_tensors
            h0 = None

        dk, dv, dg, dbeta, dh0 = _bwd_triton(
            k, v, g, beta, h0, h_states, dh_states,
            ctx.block_size, cu_seqlens=ctx.cu_seqlens,
        )
        return dk, dv, dg, dbeta, dh0, None, None


# ============================================================================
# Chunk-then-Refine state extraction
# ============================================================================

class ChunkRefineGDAStateFunction(torch.autograd.Function):
    """
    Compute hidden states at block_size boundaries using:
      Stage 1 (WY pipeline at chunk_size=64): h at chunk boundaries
      Stage 2 (local_refine): refine chunk boundaries to block boundaries

    Sequential depth: O(T/64 + 64) vs O(T) for the fused recurrent.
    Backward uses the WY-chunked path for block_size >= 16 and recomputes
    boundary states before the sequential Triton backward for smaller blocks.
    """

    @staticmethod
    def forward(ctx, k, v, g, beta, h0, block_size, chunk_size,
                h_chunks_ext, cu_seqlens):
        from torchtitan.models.qwen3_5.model.block_gated_delta_rule.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
        from .wy_fast import recompute_w_u_fwd
        from .chunk_delta_h import chunk_gated_delta_rule_fwd_h
        from .cumsum import chunk_local_cumsum
        from .solve_tril import solve_tril
        from .chunk_local_refine import local_refine_fwd

        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()
        beta = beta.contiguous()
        if h0 is not None:
            h0 = h0.contiguous()

        with torch.no_grad():
            if h_chunks_ext is not None:
                h_chunks = h_chunks_ext
            else:
                g_cumsum = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens)
                A = chunk_scaled_dot_kkt_fwd(
                    k=k, g=g_cumsum, beta=beta,
                    cu_seqlens=cu_seqlens, output_dtype=torch.float32,
                    chunk_size=chunk_size,
                )
                A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype, chunk_size=chunk_size)
                w, u = recompute_w_u_fwd(
                    k=k, v=v, beta=beta, A=A, g=g_cumsum, cu_seqlens=cu_seqlens,
                    chunk_size=chunk_size,
                )
                h_chunks, _, _ = chunk_gated_delta_rule_fwd_h(
                    k=k, w=w, u=u, g=g_cumsum,
                    initial_state=h0, output_final_state=False,
                    chunk_size=chunk_size,
                    cu_seqlens=cu_seqlens,
                )

            h_blocks = local_refine_fwd(
                h_checkpoints=h_chunks, k=k, v=v, g=g, beta=beta,
                chunk_size=chunk_size, block_size=block_size,
                cu_seqlens=cu_seqlens,
            )

        if h0 is not None:
            ctx.save_for_backward(k, v, g, beta, h0)
        else:
            ctx.save_for_backward(k, v, g, beta)
        ctx.has_initial_state = h0 is not None
        ctx.block_size = block_size
        ctx.chunk_size = chunk_size
        ctx.cu_seqlens = cu_seqlens
        return h_blocks

    @staticmethod
    def backward(ctx, dh_blocks):
        dh_blocks = dh_blocks.contiguous()
        from torchtitan.models.qwen3_5.model.block_gated_delta_rule.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
        from .wy_fast import recompute_w_u_fwd
        from .chunk_delta_h import chunk_gated_delta_rule_fwd_h
        from .cumsum import chunk_local_cumsum
        from .solve_tril import solve_tril
        from .chunk_local_refine import local_refine_fwd

        if ctx.has_initial_state:
            k, v, g, beta, h0 = ctx.saved_tensors
        else:
            k, v, g, beta = ctx.saved_tensors
            h0 = None

        bs = ctx.block_size
        _cu = ctx.cu_seqlens
        if bs >= 16:
            dk, dv, dg, dbeta, dh0 = _bwd_wy_chunked(
                k, v, g, beta, h0, dh_blocks, bs, cu_seqlens=_cu,
            )
        else:
            with torch.no_grad():
                cs = ctx.chunk_size
                g_cumsum = chunk_local_cumsum(g, chunk_size=cs, cu_seqlens=_cu)
                A = chunk_scaled_dot_kkt_fwd(
                    k=k, g=g_cumsum, beta=beta,
                    cu_seqlens=_cu, output_dtype=torch.float32, chunk_size=cs,
                )
                A = solve_tril(A=A, cu_seqlens=_cu, output_dtype=k.dtype, chunk_size=cs)
                w, u = recompute_w_u_fwd(
                    k=k, v=v, beta=beta, A=A, g=g_cumsum, cu_seqlens=_cu,
                    chunk_size=cs,
                )
                h_chunks, _, _ = chunk_gated_delta_rule_fwd_h(
                    k=k, w=w, u=u, g=g_cumsum,
                    initial_state=h0, output_final_state=False,
                    chunk_size=cs,
                    cu_seqlens=_cu,
                )
                h_blocks = local_refine_fwd(
                    h_checkpoints=h_chunks, k=k, v=v, g=g, beta=beta,
                    chunk_size=cs, block_size=bs,
                    cu_seqlens=_cu,
                )
            dk, dv, dg, dbeta, dh0 = _bwd_triton(
                k, v, g, beta, h0, h_blocks, dh_blocks, bs,
                cu_seqlens=_cu,
            )
        return dk, dv, dg, dbeta, dh0, None, None, None, None


# ============================================================================
# Public API
# ============================================================================

def chunk_refine_gda_state(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    h0: torch.Tensor | None,
    block_size: int,
    chunk_size: int = 64,
    h_chunks: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    """
    Compute hidden states at training-block boundaries using the
    chunk-then-refine strategy.

    Two-stage approach:
      1. WY pipeline produces h at chunk_size (64) boundaries.
      2. Local refinement refines to block_size boundaries.

    Mathematically equivalent to ``fused_recurrent_gated_delta_state`` but
    with lower sequential depth: O(T/chunk_size + chunk_size) vs O(T).

    Args:
        k:    [B, T, H, K]
        v:    [B, T, H, V]
        g:    [B, T, H]     gates in log space (NOT cumsum'd)
        beta: [B, T, H]
        h0:   [B, H, K, V]  or None
        block_size: block boundary interval
        chunk_size: WY chunk size (default 64)
        h_chunks: [B, NC, H, K, V] optional pre-computed h at chunk
            boundaries (to skip redundant WY pipeline computation)
        cu_seqlens: [N_docs+1] cumulative sequence lengths (None = no doc packing)

    Returns:
        h_states: [B, N, H, K, V]  (float32)
    """
    B, T, H, K = k.shape

    if T < chunk_size and h_chunks is None:
        from .chunk import compute_aligned_chunk_size
        for candidate in [32, 16]:
            cs = compute_aligned_chunk_size(block_size, max_BT=candidate)
            if cs >= block_size and cs <= T:
                chunk_size = cs
                break
        else:
            chunk_size = block_size

    assert chunk_size >= block_size, f"chunk_size={chunk_size} < block_size={block_size}"
    assert chunk_size % block_size == 0, f"chunk_size={chunk_size} not divisible by tbs={block_size}"
    return ChunkRefineGDAStateFunction.apply(
        k, v, g, beta, h0, block_size, chunk_size, h_chunks, cu_seqlens,
    )


def fused_recurrent_gated_delta_state(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    h0: torch.Tensor | None,
    block_size: int,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    """
    Compute hidden states at training-block boundaries.

    Runs the gated delta rule recurrence over the full sequence and stores
    h at every ``block_size`` boundary.

    **Forward**: single Triton kernel launch (no WY / chunk decomposition).
    **Backward**: single Triton kernel launch with per-block forward-replay
    and backward-sweep (block-boundary checkpointing).

    Args:
        k:    [B, T, H, K]  keys of the clean sequence
        v:    [B, T, H, V]  values of the clean sequence
        g:    [B, T, H]     gates in log space (NOT cumsum'd)
        beta: [B, T, H]     learning rates
        h0:   [B, H, K, V]  initial hidden state (or ``None`` → zeros)
        block_size: number of tokens per training block
        cu_seqlens: [N_docs+1] cumulative sequence lengths (None = no doc packing)

    Returns:
        h_states: [B, N, H, K, V]  (float32)
            Hidden state at the end of each of the N = T / block_size
            training blocks.
    """
    B, T, H, K = k.shape
    return FusedRecurrentGatedDeltaStateFunction.apply(
        k, v, g, beta, h0, block_size, cu_seqlens,
    )
