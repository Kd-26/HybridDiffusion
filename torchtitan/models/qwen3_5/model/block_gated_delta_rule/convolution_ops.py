# -*- coding: utf-8 -*-
# Copyright (c) 2026 Yuchen Zhu
# Portions copyright (c) 2023-2025 Songlin Yang, Yu Zhang
# Modified from Flash Linear Attention (FLA) for HybridDiffusion block-causal training.
# Local fork of FLA's Triton causal conv1d training kernels.
#
# The original FLA launch shape is (D_tiles, T_tiles, B).  In block-train
# DLLM, the noisy stream can reshape to B = batch * num_blocks, which can
# exceed CUDA's grid-y/z limit.  These kernels keep the FLA arithmetic intact
# but move the large sequence/batch dimension to grid.x.

import torch
import triton
import triton.language as tl
from einops import rearrange

from fla.ops.utils import prepare_chunk_indices
from fla.utils import IS_AMD, autotune_cache_kwargs, input_guard

from . import _profiling as _prof


NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if IS_AMD else [4, 8, 16, 32]


@triton.heuristics({
    "HAS_WEIGHT": lambda args: args["weight"] is not None,
    "HAS_BIAS": lambda args: args["bias"] is not None,
    "HAS_RESIDUAL": lambda args: args["residual"] is not None,
    "USE_INITIAL_STATE": lambda args: args["initial_state"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({"BD": BD}, num_warps=num_warps)
        for BD in [16, 32, 64, 128]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=["D", "W", "NB"],
    **autotune_cache_kwargs,
)
@triton.jit
def causal_conv1d_fwd_kernel(
    x,
    y,
    weight,
    bias,
    residual,
    cu_seqlens,
    initial_state,
    chunk_indices,
    B,
    T,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_b, i_t, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        p_x = x + bos * stride_x_t
    else:
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)
        p_x = x + i_b * stride_x_n

    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0

    if HAS_WEIGHT:
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0).to(tl.float32)

    b_y = tl.zeros((BT, BD), dtype=tl.float32)
    if not USE_INITIAL_STATE:
        for i_w in tl.static_range(-W + 1, 1):
            p_yi = tl.make_block_ptr(
                p_x, (T, D), (stride_x_t, stride_x_d),
                (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0),
            )
            b_yi = tl.load(p_yi, boundary_check=(0, 1)).to(tl.float32)
            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi
    elif i_t * BT >= W:
        for i_w in tl.static_range(-W + 1, 1):
            p_yi = tl.make_block_ptr(
                p_x, (T, D), (stride_x_t, stride_x_d),
                (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0),
            )
            b_yi = tl.load(p_yi, boundary_check=(0, 1)).to(tl.float32)
            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi
    else:
        o_t = i_t * BT + tl.arange(0, BT)
        for i_w in tl.static_range(-W + 1, 1):
            o_x = o_t + i_w
            m_x = ((o_x >= 0) & (o_x < T))[:, None] & m_d
            m_c = ((o_x + W >= 0) & (o_x < 0))[:, None] & m_d

            b_yi = tl.load(
                p_x + o_x[:, None] * stride_x_t + o_d * stride_x_d,
                mask=m_x,
                other=0,
            ).to(tl.float32)

            b_yi += tl.load(
                initial_state + i_n * D * W + o_d * W + (o_x + W)[:, None],
                mask=m_c,
                other=0,
            ).to(tl.float32)

            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi

    if HAS_BIAS:
        b_y += tl.load(bias + o_d, mask=m_d).to(tl.float32)

    if ACTIVATION == "swish" or ACTIVATION == "silu":
        b_y = b_y * tl.sigmoid(b_y)

    if HAS_RESIDUAL:
        p_residual = tl.make_block_ptr(
            residual + bos * D, (T, D), (D, 1),
            (i_t * BT, i_d * BD), (BT, BD), (1, 0),
        )
        b_residual = tl.load(p_residual, boundary_check=(0, 1))
        b_y += b_residual

    p_y = tl.make_block_ptr(
        y + bos * D, (T, D), (D, 1),
        (i_t * BT, i_d * BD), (BT, BD), (1, 0),
    )
    tl.store(p_y, tl.cast(b_y, dtype=p_y.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


@triton.heuristics({
    "HAS_WEIGHT": lambda args: args["dw"] is not None,
    "HAS_BIAS": lambda args: args["db"] is not None,
    "USE_INITIAL_STATE": lambda args: args["initial_state"] is not None,
    "USE_FINAL_STATE": lambda args: args["dht"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({"BD": BD}, num_warps=num_warps)
        for BD in [16, 32, 64, 128]
        for num_warps in [4, 8, 16, 32]
    ],
    key=["D", "W", "NB"],
    **autotune_cache_kwargs,
)
@triton.jit
def causal_conv1d_bwd_kernel(
    x,
    y,
    weight,
    initial_state,
    dht,
    dy,
    dx,
    dw,
    db,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    stride_dx_n,
    stride_dx_t,
    stride_dx_d,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_b, i_t, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        p_x = x + bos * stride_x_t
    else:
        i_tg = i_b * tl.num_programs(1) + i_t
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)
        p_x = x + i_b * stride_x_n

    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0

    if HAS_WEIGHT:
        p_x = tl.make_block_ptr(
            p_x, (T, D), (stride_x_t, stride_x_d),
            (i_t * BT, i_d * BD), (BT, BD), (1, 0),
        )
        b_x = tl.load(p_x, boundary_check=(0, 1))
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0)

    b_dx = tl.zeros((BT, BD), dtype=tl.float32)
    if HAS_BIAS:
        b_db = tl.zeros((BD,), dtype=tl.float32)

    if not USE_FINAL_STATE and not USE_INITIAL_STATE:
        for i_w in tl.static_range(0, W):
            p_dy = tl.make_block_ptr(
                dy + bos * D, (T, D), (D, 1),
                (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0),
            )
            b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
            if ACTIVATION == "swish" or ACTIVATION == "silu":
                p_y = tl.make_block_ptr(
                    y + bos * D, (T, D), (D, 1),
                    (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0),
                )
                b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
                b_ys = tl.sigmoid(b_y)
                b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))
            b_wdy = b_dy
            if HAS_WEIGHT:
                b_wdy = b_wdy * tl.sum(b_w * (o_w == (W - i_w - 1)), 1)
                b_dw = tl.sum(b_dy * b_x, 0)
                tl.store(dw + i_tg * D * W + o_d * W + W - i_w - 1, b_dw.to(dw.dtype.element_ty), mask=m_d)
            if HAS_BIAS and i_w == 0:
                b_db += tl.sum(b_dy, 0)
            b_dx += b_wdy
    elif i_t * BT >= W:
        for i_w in tl.static_range(0, W):
            p_dy = tl.make_block_ptr(
                dy + bos * D, (T, D), (D, 1),
                (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0),
            )
            b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
            if ACTIVATION == "swish" or ACTIVATION == "silu":
                p_y = tl.make_block_ptr(
                    y + bos * D, (T, D), (D, 1),
                    (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0),
                )
                b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
                b_ys = tl.sigmoid(b_y)
                b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))
            b_wdy = b_dy
            if HAS_WEIGHT:
                b_wdy = b_wdy * tl.sum(b_w * (o_w == (W - i_w - 1)), 1)
                b_dw = tl.sum(b_dy * b_x, 0)
                tl.store(dw + i_tg * D * W + o_d * W + W - i_w - 1, b_dw.to(dw.dtype.element_ty), mask=m_d)
            if HAS_BIAS and i_w == 0:
                b_db += tl.sum(b_dy, 0)
            b_dx += b_wdy
    else:
        o_t = i_t * BT + tl.arange(0, BT)
        for i_w in tl.static_range(0, W):
            p_dy = tl.make_block_ptr(
                dy + bos * D, (T, D), (D, 1),
                (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0),
            )
            b_dy_shift = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
            if ACTIVATION == "swish" or ACTIVATION == "silu":
                p_y = tl.make_block_ptr(
                    y + bos * D, (T, D), (D, 1),
                    (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0),
                )
                b_y_shift = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
                b_ys = tl.sigmoid(b_y_shift)
                b_dy_shift = b_dy_shift * b_ys * (1 + b_y_shift * (1 - b_ys))
            if HAS_WEIGHT:
                b_dw = tl.sum(b_dy_shift * b_x, 0)
                if USE_INITIAL_STATE:
                    mask_head_rows = o_t < i_w
                    b_dy_head = tl.load(
                        dy + bos * D + o_t[:, None] * D + o_d,
                        mask=(mask_head_rows[:, None] & m_d[None, :]),
                        other=0.0,
                    ).to(tl.float32)
                    if ACTIVATION == "swish" or ACTIVATION == "silu":
                        b_y_head = tl.load(
                            y + bos * D + o_t[:, None] * D + o_d,
                            mask=(mask_head_rows[:, None] & m_d[None, :]),
                            other=0.0,
                        ).to(tl.float32)
                        b_ys_head = tl.sigmoid(b_y_head)
                        b_dy_head = b_dy_head * b_ys_head * (1 + b_y_head * (1 - b_ys_head))
                    o_c = W - i_w + o_t
                    mask_c = mask_head_rows & (o_c >= 1) & (o_c < W)
                    b_xc = tl.load(
                        initial_state + i_n * D * W + o_d[None, :] * W + o_c[:, None],
                        mask=(mask_c[:, None] & m_d[None, :]),
                        other=0.0,
                    ).to(tl.float32)
                    b_dw += tl.sum(b_dy_head * b_xc, 0)
                tl.store(dw + i_tg * D * W + o_d * W + W - i_w - 1, b_dw.to(dw.dtype.element_ty), mask=m_d)

            if HAS_BIAS and i_w == 0:
                b_db += tl.sum(b_dy_shift, 0)
            b_wdy = b_dy_shift if not HAS_WEIGHT else (b_dy_shift * tl.sum(b_w * (o_w == (W - i_w - 1)), 1))
            b_dx += b_wdy

    if HAS_BIAS:
        b_db = tl.cast(b_db, dtype=db.dtype.element_ty, fp_downcast_rounding="rtne")
        tl.store(db + i_tg * D + o_d, b_db, mask=m_d)

    if USE_FINAL_STATE:
        if i_t * BT + BT >= T - W:
            start_tok = max(0, T - (W - 1))
            offset = i_t * BT + tl.arange(0, BT)
            tok_idx = offset - start_tok
            mask = (offset >= start_tok) & (offset < T)
            w_idx = 1 + tok_idx
            dht_off = i_n * D * W + o_d[None, :] * W + w_idx[:, None]
            b_dht = tl.load(dht + dht_off, mask=mask[:, None] & m_d[None, :], other=0.0).to(tl.float32)
            b_dx += b_dht

    if IS_VARLEN:
        p_dx = dx + bos * stride_dx_t
    else:
        p_dx = dx + i_b * stride_dx_n

    p_dx = tl.make_block_ptr(
        p_dx, (T, D), (stride_dx_t, stride_dx_d),
        (i_t * BT, i_d * BD), (BT, BD), (1, 0),
    )
    tl.store(p_dx, tl.cast(b_dx, dtype=p_dx.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


@triton.heuristics({
    "USE_ACTIVATION": lambda args: args["y"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.jit
def compute_dh0_kernel(
    dy,
    y,
    weight,
    dh0,
    cu_seqlens,
    stride_dy_n,
    stride_dy_t,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    USE_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_n, i_d = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        seq_len = eos - bos
        dy_base = dy + bos * stride_dy_t
    else:
        seq_len = T
        dy_base = dy + i_n * stride_dy_n

    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    for i_w in tl.static_range(1, W):
        b_dh0 = tl.zeros([BD], dtype=tl.float32)

        for t in tl.static_range(0, W - 1):
            if t < i_w:
                w_idx = i_w - 1 - t
                p_dy = dy_base + t * stride_dy_t + o_d
                m_t = (t < seq_len) & m_d
                b_dy = tl.load(p_dy, mask=m_t, other=0).to(tl.float32)

                if USE_ACTIVATION:
                    if IS_VARLEN:
                        p_y = y + bos * stride_dy_t + t * stride_dy_t + o_d
                    else:
                        p_y = y + i_n * stride_dy_n + t * stride_dy_t + o_d
                    b_y = tl.load(p_y, mask=m_t, other=0).to(tl.float32)
                    b_ys = tl.sigmoid(b_y)
                    b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))

                b_w_col = tl.load(weight + o_d * W + w_idx, mask=m_d, other=0).to(tl.float32)
                b_dh0 += tl.where(m_t, b_dy * b_w_col, 0)

        p_dh0 = dh0 + i_n * D * W + o_d * W + i_w
        tl.store(p_dh0, b_dh0.to(dh0.dtype.element_ty), mask=m_d)


@input_guard(no_guard_contiguous=["x"])
def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    activation: str | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    BT: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    shape = x.shape
    if x.shape[-1] != weight.shape[0]:
        x = rearrange(x, "b t ... -> b t (...)")
    B, T, D = x.shape[0], x.shape[1], weight.shape[0]
    W = weight.shape[1]
    stride_x_n, stride_x_t, stride_x_d = x.stride()

    BW = triton.next_power_of_2(W)
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    NB = triton.cdiv(B * T, 1024)

    y = torch.empty_like(x, memory_format=torch.contiguous_format)

    def grid(meta):
        return (B, NT, triton.cdiv(D, meta["BD"]))

    causal_conv1d_fwd_kernel[grid](
        x=x,
        y=y,
        weight=weight,
        bias=bias,
        residual=residual,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        D=D,
        W=W,
        BT=BT,
        BW=BW,
        NB=NB,
        stride_x_n=stride_x_n,
        stride_x_t=stride_x_t,
        stride_x_d=stride_x_d,
        ACTIVATION=activation,
    )

    final_state = None
    if output_final_state:
        from fla.modules.conv.triton.ops import causal_conv1d_update_states

        final_state = causal_conv1d_update_states(
            x=x,
            state_len=W,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )
    return y.view(shape), final_state


def compute_dh0_triton(
    dy: torch.Tensor,
    y: torch.Tensor | None,
    weight: torch.Tensor,
    initial_state: torch.Tensor,
    activation: str | None,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    D, W = weight.shape
    N = initial_state.shape[0]
    T = dy.shape[1]

    dh0 = torch.zeros_like(initial_state)
    BD = 32

    y_to_pass = y if activation in ("swish", "silu") else None
    stride_dy_n = dy.stride(0)
    stride_dy_t = dy.stride(1)

    grid = (N, triton.cdiv(D, BD))
    with _prof._record("block_conv_compute_dh0"):
        compute_dh0_kernel[grid](
            dy=dy,
            y=y_to_pass,
            weight=weight,
            dh0=dh0,
            cu_seqlens=cu_seqlens,
            stride_dy_n=stride_dy_n,
            stride_dy_t=stride_dy_t,
            T=T,
            D=D,
            W=W,
            BD=BD,
        )
    return dh0


def causal_conv1d_bwd(
    x: torch.Tensor,
    dy: torch.Tensor,
    dht: torch.Tensor | None,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    BT: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    shape = x.shape
    if x.shape[-1] != weight.shape[0]:
        x = rearrange(x, "b t ... -> b t (...)")
    B, T, D = x.shape
    W = weight.shape[1] if weight is not None else None

    stride_x_n, stride_x_t, stride_x_d = x.stride()

    BW = triton.next_power_of_2(W)
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    NB = triton.cdiv(B * T, 1024)

    y = None
    if activation is not None:
        y, _ = causal_conv1d_fwd(
            x=x,
            weight=weight,
            bias=bias,
            residual=None,
            initial_state=initial_state,
            activation=None,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            output_final_state=False,
            chunk_indices=chunk_indices,
        )

    dx = torch.empty_like(x)
    dw = weight.new_empty(B * NT, *weight.shape, dtype=torch.float) if weight is not None else None
    db = bias.new_empty(B * NT, *bias.shape, dtype=torch.float) if bias is not None else None
    dr = dy if residual is not None else None

    stride_dx_n, stride_dx_t, stride_dx_d = dx.stride()

    def grid(meta):
        return (B, NT, triton.cdiv(D, meta["BD"]))

    causal_conv1d_bwd_kernel[grid](
        x=x,
        y=y,
        weight=weight,
        initial_state=initial_state,
        dht=dht,
        dy=dy,
        dx=dx,
        dw=dw,
        db=db,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        D=D,
        W=W,
        BT=BT,
        BW=BW,
        NB=NB,
        stride_x_n=stride_x_n,
        stride_x_t=stride_x_t,
        stride_x_d=stride_x_d,
        stride_dx_n=stride_dx_n,
        stride_dx_t=stride_dx_t,
        stride_dx_d=stride_dx_d,
        ACTIVATION=activation,
    )

    if weight is not None:
        dw = dw.sum(0).to(weight)
    if bias is not None:
        db = db.sum(0).to(bias)

    dh0 = None
    if initial_state is not None:
        dh0 = compute_dh0_triton(
            dy=dy,
            y=y,
            weight=weight,
            initial_state=initial_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
        )

    return dx.view(shape), dw, db, dr, dh0
