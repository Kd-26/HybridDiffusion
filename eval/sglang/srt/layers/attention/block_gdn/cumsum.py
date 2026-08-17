# Fixed chunk_local_cumsum kernels (local copy for block_gated_delta_rule).
#
# FLA's original kernels place B*H in CUDA grid dim-y or dim-z, which are
# limited to 65535.  When the noisy batch is reshaped into (B*N, tbs, H, K)
# with large B*N (e.g. 2048 for block_size=4), B*H exceeds 65535 and the
# kernel launch fails with "CUDA: invalid argument".
#
# Fix: move B*H to grid dim-0 (x), which supports up to 2^31-1.
# The kernel bodies are identical to FLA's; only the grid layout and
# tl.program_id assignments are swapped.
#
# Original: fla/ops/utils/cumsum.py (fla 0.4.x, Songlin Yang & Yu Zhang)

import torch
import triton
import triton.language as tl

from fla.ops.utils.index import prepare_chunk_indices
from fla.utils import autotune_cache_kwargs, check_shared_mem, input_guard

BS_LIST = [32, 64] if check_shared_mem() else [16, 32]


# ---------------------------------------------------------------------------
# Scalar kernel  (3-D input: B, T, H)
# Original grid: (NT, B*H)          -- B*H on dim-1 (y), limit 65535
# Fixed grid:    (B*H, NT)          -- B*H on dim-0 (x), limit 2^31-1
# ---------------------------------------------------------------------------

@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['B', 'H', 'BT', 'IS_VARLEN', 'REVERSE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_bh, i_t = tl.program_id(0), tl.program_id(1)  # swapped vs original
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    T_chunk = tl.minimum(T, (i_t + 1) * CHUNK_SIZE)
    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + bos*H + i_h*T, (T_chunk,), (1,), (i_t * CHUNK_SIZE,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos*H + i_h*T, (T_chunk,), (1,), (i_t * CHUNK_SIZE,), (BT,), (0,))
    else:
        p_s = tl.make_block_ptr(s + bos*H + i_h, (T_chunk,), (H,), (i_t * CHUNK_SIZE,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos*H + i_h, (T_chunk,), (H,), (i_t * CHUNK_SIZE,), (BT,), (0,))
    # [BT]
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


# ---------------------------------------------------------------------------
# Vector kernel  (4-D input: B, T, H, S)
# Original grid: (ceil(S/BS), NT, B*H)  -- B*H on dim-2 (z), limit 65535
# Fixed grid:    (B*H, ceil(S/BS), NT)  -- B*H on dim-0 (x), limit 2^31-1
# ---------------------------------------------------------------------------

@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BS': BS}, num_warps=num_warps)
        for BS in BS_LIST
        for num_warps in [2, 4, 8]
    ],
    key=['B', 'H', 'S', 'BT', 'IS_VARLEN', 'REVERSE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_local_cumsum_vector_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_bh, i_s, i_t = tl.program_id(0), tl.program_id(1), tl.program_id(2)  # swapped vs original
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    T_chunk = tl.minimum(T, (i_t + 1) * CHUNK_SIZE)
    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + (bos * H + i_h*T)*S, (T_chunk, S), (S, 1), (i_t * CHUNK_SIZE, i_s * BS), (BT, BS), (1, 0))
        p_o = tl.make_block_ptr(o + (bos * H + i_h*T)*S, (T_chunk, S), (S, 1), (i_t * CHUNK_SIZE, i_s * BS), (BT, BS), (1, 0))
    else:
        p_s = tl.make_block_ptr(s + (bos * H + i_h) * S, (T_chunk, S), (H*S, 1), (i_t * CHUNK_SIZE, i_s * BS), (BT, BS), (1, 0))
        p_o = tl.make_block_ptr(o + (bos * H + i_h) * S, (T_chunk, S), (H*S, 1), (i_t * CHUNK_SIZE, i_s * BS), (BT, BS), (1, 0))
    # [BT, BS]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    if REVERSE:
        b_o = tl.cumsum(b_s, axis=0, reverse=True)
    else:
        b_o = tl.cumsum(b_s, axis=0)
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


# ---------------------------------------------------------------------------
# Python wrappers  (signatures identical to FLA originals)
# ---------------------------------------------------------------------------

def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def chunk_local_cumsum_scalar(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    CHUNK_SIZE = chunk_size
    BT = _next_power_of_2(chunk_size)
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
    NT = triton.cdiv(T, CHUNK_SIZE) if cu_seqlens is None else len(chunk_indices)
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    grid = (B * H, NT)  # fixed: B*H on dim-0
    chunk_local_cumsum_scalar_kernel[grid](
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        BT=BT,
        CHUNK_SIZE=CHUNK_SIZE,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
    )
    return g


def chunk_local_cumsum_vector(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    if head_first:
        B, H, T, S = g.shape
    else:
        B, T, H, S = g.shape
    CHUNK_SIZE = chunk_size
    BT = _next_power_of_2(chunk_size)
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
    NT = triton.cdiv(T, CHUNK_SIZE) if cu_seqlens is None else len(chunk_indices)

    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    def grid(meta): return (B * H, triton.cdiv(meta['S'], meta['BS']), NT)  # fixed: B*H on dim-0
    chunk_local_cumsum_vector_kernel[grid](
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        S=S,
        BT=BT,
        CHUNK_SIZE=CHUNK_SIZE,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
    )
    return g


@input_guard
def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
    **kwargs,
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert g.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    if len(g.shape) == 3:
        return chunk_local_cumsum_scalar(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
            chunk_indices=chunk_indices,
        )
    elif len(g.shape) == 4:
        return chunk_local_cumsum_vector(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
            chunk_indices=chunk_indices,
        )
    else:
        raise ValueError(
            f"Unsupported input shape {g.shape}, "
            f"which should be (B, T, H, D) if `head_first=False` "
            f"or (B, H, T, D) otherwise",
        )
