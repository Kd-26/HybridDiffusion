# Fixed chunk_scaled_dot_kkt kernel (local copy for block_gated_delta_rule).
#
# FLA's original kernel places B*H in CUDA grid dim-y, which is limited to
# 65535. When the noisy batch is reshaped into (B*N, tbs, H, K) with large
# B*N (e.g. 2048 for block_size=4), B*H exceeds 65535 and the kernel launch
# fails with "CUDA: invalid argument".
#
# Fix: move B*H to grid dim-0 (x), which supports up to 2^31-1.
# The kernel body is identical to FLA's; only the grid layout and
# tl.program_id assignments are swapped.
#
# Original: fla/ops/common/chunk_scaled_dot_kkt.py (fla 0.4.x, Songlin Yang & Yu Zhang)

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp
from fla.utils import autotune_cache_kwargs


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'BT', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    # Fixed: B*H on dim-0 (x, limit 2^31-1), NT on dim-1 (y, limit 65535)
    i_bh, i_t = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    T_chunk = tl.minimum(T, (i_t + 1) * CHUNK_SIZE)
    o_t = i_t * CHUNK_SIZE + tl.arange(0, BT)
    m_t = o_t < T_chunk

    p_b = tl.make_block_ptr(beta + bos*H + i_h, (T_chunk,), (H,), (i_t * CHUNK_SIZE,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T_chunk, K), (H*K, 1), (i_t * CHUNK_SIZE, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_k, tl.trans(b_k))

    if USE_G:
        p_g = tl.make_block_ptr(g + bos*H + i_h, (T_chunk,), (H,), (i_t * CHUNK_SIZE,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_diff = b_g[:, None] - b_g[None, :]
        b_A *= exp(b_g_diff)
    b_A *= b_b[:, None]

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    p_A = tl.make_block_ptr(A + (bos*H + i_h) * BT, (T_chunk, BT), (BT*H, 1), (i_t * CHUNK_SIZE, 0), (BT, BT), (1, 0))
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""
    Compute beta * K * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`.
        g (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H]`. Default: `None`.
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`

    Returns:
        beta * K * K^T of shape `[B, T, H, BT]` where `BT` is the chunk size.
    """
    B, T, H, K = k.shape
    CHUNK_SIZE = chunk_size
    BT = max(chunk_size, 1)
    BT = 1 << (BT - 1).bit_length()
    chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE) if cu_seqlens is not None else None
    NT = triton.cdiv(T, CHUNK_SIZE) if cu_seqlens is None else len(chunk_indices)
    A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)
    # Fixed: grid (B*H, NT) instead of (NT, B*H)
    chunk_scaled_dot_kkt_fwd_kernel[(B * H, NT)](
        k=k,
        g=g,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        CHUNK_SIZE=CHUNK_SIZE,
    )
    return A
