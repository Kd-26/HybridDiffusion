# -*- coding: utf-8 -*-
# Copyright (c) 2026 Yuchen Zhu

"""Two-stream BlockShortConv kernels for block-train noisy stream.

The clean stream is a normal causal convolution and stays on the existing FLA
conv kernels.  The noisy stream has block-train semantics: tokens inside a
noisy block see prior noisy tokens from the same block, while prefix tokens
before the block come from the clean stream.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices

from . import _profiling as _prof


@triton.heuristics({
    "HAS_BIAS": lambda args: args["bias"] is not None,
    "USE_ACTIVATION": lambda args: args["ACTIVATION"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.jit
def _block_short_conv_twostream_fwd_kernel(
    x_clean,
    x_noisy,
    weight,
    bias,
    y_noisy,
    cu_seqlens,
    chunk_indices,
    T,
    stride_x_n: tl.constexpr,
    stride_x_t: tl.constexpr,
    stride_x_d: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    NT: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_c = tl.program_id(0)
    i_d = tl.program_id(1)

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_c * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_c * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        doc_T = (eos - bos).to(tl.int32)
        base_clean = x_clean + bos * stride_x_t
        base_noisy = x_noisy + bos * stride_x_t
        base_y = y_noisy + bos * stride_x_t
    else:
        i_n = i_c // NT
        i_t = i_c - i_n * NT
        doc_T = T
        base_clean = x_clean + i_n * stride_x_n
        base_noisy = x_noisy + i_n * stride_x_n
        base_y = y_noisy + i_n * stride_x_n

    o_t = i_t * BT + tl.arange(0, BT)
    o_d = i_d * BD + tl.arange(0, BD)
    m_t = o_t < doc_T
    m_d = o_d < D
    block_start = (o_t // BLOCK_SIZE) * BLOCK_SIZE

    acc = tl.zeros((BT, BD), dtype=tl.float32)
    for lag in tl.static_range(0, W):
        src = o_t - lag
        use_noisy = src >= block_start
        valid_src = (src >= 0) & (src < doc_T)
        x_src = tl.load(
            base_noisy + src[:, None] * stride_x_t + o_d[None, :] * stride_x_d,
            mask=(m_t[:, None] & valid_src[:, None] & use_noisy[:, None] & m_d[None, :]),
            other=0.0,
        ).to(tl.float32)
        x_src += tl.load(
            base_clean + src[:, None] * stride_x_t + o_d[None, :] * stride_x_d,
            mask=(m_t[:, None] & valid_src[:, None] & (~use_noisy)[:, None] & m_d[None, :]),
            other=0.0,
        ).to(tl.float32)
        w = tl.load(weight + o_d * W + (W - 1 - lag), mask=m_d, other=0.0).to(tl.float32)
        acc += x_src * w[None, :]

    if HAS_BIAS:
        b = tl.load(bias + o_d, mask=m_d, other=0.0).to(tl.float32)
        acc += b[None, :]

    if USE_ACTIVATION:
        acc = acc * tl.sigmoid(acc)

    tl.store(
        base_y + o_t[:, None] * stride_x_t + o_d[None, :] * stride_x_d,
        acc,
        mask=m_t[:, None] & m_d[None, :],
    )


@triton.heuristics({
    "HAS_BIAS": lambda args: args["db_partial"] is not None,
    "USE_ACTIVATION": lambda args: args["ACTIVATION"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.jit
def _block_short_conv_twostream_bwd_kernel(
    x_clean,
    x_noisy,
    dy_noisy,
    weight,
    bias,
    dx_clean,
    dx_noisy,
    dw_partial,
    db_partial,
    cu_seqlens,
    chunk_indices,
    T,
    stride_x_n: tl.constexpr,
    stride_x_t: tl.constexpr,
    stride_x_d: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    NT: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_c = tl.program_id(0)
    i_d = tl.program_id(1)

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_c * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_c * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        doc_T = (eos - bos).to(tl.int32)
        base_clean = x_clean + bos * stride_x_t
        base_noisy = x_noisy + bos * stride_x_t
        base_dy = dy_noisy + bos * stride_x_t
        base_dxc = dx_clean + bos * D
        base_dxn = dx_noisy + bos * D
    else:
        i_n = i_c // NT
        i_t = i_c - i_n * NT
        doc_T = T
        base_clean = x_clean + i_n * stride_x_n
        base_noisy = x_noisy + i_n * stride_x_n
        base_dy = dy_noisy + i_n * stride_x_n
        base_dxc = dx_clean + i_n * T * D
        base_dxn = dx_noisy + i_n * T * D

    o_t = i_t * BT + tl.arange(0, BT)
    o_d = i_d * BD + tl.arange(0, BD)
    m_t = o_t < doc_T
    m_d = o_d < D
    block_start = (o_t // BLOCK_SIZE) * BLOCK_SIZE

    preact = tl.zeros((BT, BD), dtype=tl.float32)
    for lag in tl.static_range(0, W):
        src = o_t - lag
        use_noisy = src >= block_start
        valid_src = (src >= 0) & (src < doc_T)
        x_src = tl.load(
            base_noisy + src[:, None] * stride_x_t + o_d[None, :] * stride_x_d,
            mask=(m_t[:, None] & valid_src[:, None] & use_noisy[:, None] & m_d[None, :]),
            other=0.0,
        ).to(tl.float32)
        x_src += tl.load(
            base_clean + src[:, None] * stride_x_t + o_d[None, :] * stride_x_d,
            mask=(m_t[:, None] & valid_src[:, None] & (~use_noisy)[:, None] & m_d[None, :]),
            other=0.0,
        ).to(tl.float32)
        w = tl.load(weight + o_d * W + (W - 1 - lag), mask=m_d, other=0.0).to(tl.float32)
        preact += x_src * w[None, :]

    if HAS_BIAS:
        b = tl.load(bias + o_d, mask=m_d, other=0.0).to(tl.float32)
        preact += b[None, :]

    dpre = tl.load(
        base_dy + o_t[:, None] * stride_x_t + o_d[None, :] * stride_x_d,
        mask=m_t[:, None] & m_d[None, :],
        other=0.0,
    ).to(tl.float32)
    if USE_ACTIVATION:
        sig = tl.sigmoid(preact)
        dpre = dpre * sig * (1.0 + preact * (1.0 - sig))
    dpre = tl.where(m_t[:, None] & m_d[None, :], dpre, 0.0)

    if HAS_BIAS:
        db = tl.sum(dpre, axis=0)
        tl.store(db_partial + i_c * D + o_d, db, mask=m_d)

    for lag in tl.static_range(0, W):
        src = o_t - lag
        use_noisy = src >= block_start
        valid_src = (src >= 0) & (src < doc_T)
        x_src = tl.load(
            base_noisy + src[:, None] * stride_x_t + o_d[None, :] * stride_x_d,
            mask=(m_t[:, None] & valid_src[:, None] & use_noisy[:, None] & m_d[None, :]),
            other=0.0,
        ).to(tl.float32)
        x_src += tl.load(
            base_clean + src[:, None] * stride_x_t + o_d[None, :] * stride_x_d,
            mask=(m_t[:, None] & valid_src[:, None] & (~use_noisy)[:, None] & m_d[None, :]),
            other=0.0,
        ).to(tl.float32)
        dw = tl.sum(dpre * x_src, axis=0)
        w_idx = W - 1 - lag
        tl.store(dw_partial + i_c * D * W + o_d * W + w_idx, dw, mask=m_d)

        w = tl.load(weight + o_d * W + w_idx, mask=m_d, other=0.0).to(tl.float32)
        dx_val = dpre * w[None, :]
        tl.atomic_add(
            base_dxn + src[:, None] * D + o_d[None, :],
            dx_val,
            sem="relaxed",
            mask=(m_t[:, None] & valid_src[:, None] & use_noisy[:, None] & m_d[None, :]),
        )
        tl.atomic_add(
            base_dxc + src[:, None] * D + o_d[None, :],
            dx_val,
            sem="relaxed",
            mask=(m_t[:, None] & valid_src[:, None] & (~use_noisy)[:, None] & m_d[None, :]),
        )


def _make_chunk_indices(
    cu_seqlens: torch.Tensor | None,
    T: int,
    B: int,
    BT: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, int, int]:
    if cu_seqlens is None:
        nt = triton.cdiv(T, BT)
        return None, B * nt, nt
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    return chunk_indices, len(chunk_indices), 0


def block_short_conv_twostream_fwd(
    x_clean: torch.Tensor,
    x_noisy: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    block_size: int,
    activation: str | None,
    cu_seqlens: torch.Tensor | None = None,
    BT: int = 64,
    BD: int = 64,
) -> torch.Tensor:
    x_clean = x_clean.contiguous()
    x_noisy = x_noisy.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    B, T, D = x_noisy.shape
    W = weight.shape[1]
    if activation not in (None, "silu", "swish"):
        raise ValueError(f"Unsupported activation for twostream conv: {activation!r}")
    if cu_seqlens is not None and B != 1:
        raise ValueError("twostream BlockShortConv with cu_seqlens expects flattened B=1 input")

    chunk_indices, nchunks, nt = _make_chunk_indices(cu_seqlens, T, B, BT, x_noisy.device)
    y_noisy = torch.zeros_like(x_noisy, memory_format=torch.contiguous_format)
    grid = (nchunks, triton.cdiv(D, BD))
    with _prof._record("twostream_conv_fwd_noisy"):
        _block_short_conv_twostream_fwd_kernel[grid](
            x_clean, x_noisy, weight, bias, y_noisy,
            cu_seqlens, chunk_indices, T,
            stride_x_n=x_noisy.stride(0),
            stride_x_t=x_noisy.stride(1),
            stride_x_d=x_noisy.stride(2),
            D=D,
            W=W,
            BLOCK_SIZE=block_size,
            BT=BT,
            BD=BD,
            NT=nt,
            ACTIVATION=activation,
            num_warps=4,
            num_stages=3,
        )
    return y_noisy


def block_short_conv_twostream_bwd(
    x_clean: torch.Tensor,
    x_noisy: torch.Tensor,
    dy_noisy: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    block_size: int,
    activation: str | None,
    cu_seqlens: torch.Tensor | None = None,
    BT: int = 64,
    BD: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    x_clean = x_clean.contiguous()
    x_noisy = x_noisy.contiguous()
    dy_noisy = dy_noisy.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    B, T, D = x_noisy.shape
    W = weight.shape[1]
    if activation not in (None, "silu", "swish"):
        raise ValueError(f"Unsupported activation for twostream conv: {activation!r}")
    if cu_seqlens is not None and B != 1:
        raise ValueError("twostream BlockShortConv with cu_seqlens expects flattened B=1 input")

    chunk_indices, nchunks, nt = _make_chunk_indices(cu_seqlens, T, B, BT, x_noisy.device)
    dx_clean = torch.zeros(B, T, D, device=x_noisy.device, dtype=torch.float32)
    dx_noisy = torch.zeros(B, T, D, device=x_noisy.device, dtype=torch.float32)
    dw_partial = torch.empty(nchunks, D, W, device=x_noisy.device, dtype=torch.float32)
    db_partial = torch.empty(nchunks, D, device=x_noisy.device, dtype=torch.float32) if bias is not None else None
    grid = (nchunks, triton.cdiv(D, BD))
    with _prof._record("twostream_conv_bwd_noisy"):
        _block_short_conv_twostream_bwd_kernel[grid](
            x_clean, x_noisy, dy_noisy, weight, bias,
            dx_clean, dx_noisy, dw_partial, db_partial,
            cu_seqlens, chunk_indices, T,
            stride_x_n=x_noisy.stride(0),
            stride_x_t=x_noisy.stride(1),
            stride_x_d=x_noisy.stride(2),
            D=D,
            W=W,
            BLOCK_SIZE=block_size,
            BT=BT,
            BD=BD,
            NT=nt,
            ACTIVATION=activation,
            num_warps=4,
            num_stages=3,
        )
    with _prof._record("twostream_conv_reduce_dw"):
        dw = dw_partial.sum(0).to(weight.dtype)
        db = db_partial.sum(0).to(bias.dtype) if db_partial is not None else None
    return dx_clean.to(x_clean.dtype), dx_noisy.to(x_noisy.dtype), dw, db
