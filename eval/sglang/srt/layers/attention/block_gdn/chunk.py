# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Modified for block-causal gated delta rule attention
# Stripped for inference-only (block_train code removed)

import warnings
from typing import Optional

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from .chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from .wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

from .chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from .cumsum import chunk_local_cumsum
from .solve_tril import solve_tril

from .chunk_o import (
    chunk_block_causal_fwd_o,
    chunk_block_causal_bwd_dAv,
    chunk_block_causal_bwd_dqkwg,
    compute_block_causal_aqk,
)


def compute_aligned_chunk_size(block_size: int, max_BT: int = 64) -> int:
    """Compute the largest chunk_size <= max_BT that is a multiple of block_size."""
    return (max_BT // block_size) * block_size


def _compute_g_block_end(g: torch.Tensor, block_size: int, causal_mode: int = 0) -> torch.Tensor:
    """
    Compute g_block_end for each position.
    causal_mode=0: g_block_end[i] = g[block_end(b)] (block-causal)
    causal_mode=1: returns g itself (token-causal, identity)
    """
    if causal_mode == 1:
        return g
    B, T, H = g.shape
    positions = torch.arange(T, device=g.device)
    block_indices = positions // block_size
    block_end_positions = (block_indices + 1) * block_size - 1
    block_end_positions = torch.clamp(block_end_positions, max=T - 1)
    g_block_end = g[:, block_end_positions, :]
    return g_block_end


def chunk_block_causal_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    block_size: int,
    initial_state: torch.Tensor,
    output_final_state: bool,
    causal_mode: int = 0,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
):
    g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens)
    A = chunk_scaled_dot_kkt_fwd(
        k=k, g=g, beta=beta,
        cu_seqlens=cu_seqlens, output_dtype=torch.float32,
        chunk_size=chunk_size,
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype, chunk_size=chunk_size)
    w, u = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A, g=g, cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    o = chunk_block_causal_fwd_o(
        q=q, k=k, v=v_new, h=h, g=g,
        scale=scale, block_size=block_size, causal_mode=causal_mode,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    return g, o, A, final_state


def chunk_block_causal_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    block_size: int,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    causal_mode: int = 0,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    dv_new_inject: Optional[torch.Tensor] = None,
    dG_inject: Optional[torch.Tensor] = None,
    dh_inject: Optional[torch.Tensor] = None,
):
    w, u = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A, g=g, cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    Aqk = compute_block_causal_aqk(
        q=q, k=k, g=g, scale=scale,
        block_size=block_size, causal_mode=causal_mode,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dA, dv = chunk_block_causal_bwd_dAv(
        v=v_new, do=do, A=Aqk, scale=scale,
        block_size=block_size, causal_mode=causal_mode,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )

    if dv_new_inject is not None:
        dv = dv + dv_new_inject

    g_block_end = _compute_g_block_end(g, block_size, causal_mode=causal_mode)
    g_correction = g_block_end - g
    q_scaled = q * torch.exp(g_correction).unsqueeze(-1)

    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q_scaled, k=k, w=w, g=g,
        h0=initial_state, dht=dht, do=do, dv=dv,
        dh_inject=dh_inject,
        scale=scale, cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dq, dk, dw, dg = chunk_block_causal_bwd_dqkwg(
        q=q, k=k, v=v_new, w=w, g=g, h=h,
        dv=dv, do=do, dh=dh,
        scale=scale, block_size=block_size, causal_mode=causal_mode,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dk2, dv, db, dg2 = prepare_wy_repr_bwd(
        k=k, v=v, beta=beta, g=g, A=A,
        dw=dw, du=dv, cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dk.add_(dk2)
    dg.add_(dg2)
    if dG_inject is not None:
        dg = dg + dG_inject
    dg = chunk_local_cumsum(dg, chunk_size=chunk_size, reverse=True, cu_seqlens=cu_seqlens)
    return dq, dk, dv, db, dg, dh0


class ChunkBlockCausalGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        block_size: int,
        initial_state: torch.Tensor,
        output_final_state: bool,
        causal_mode: int = 0,
        cu_seqlens: Optional[torch.LongTensor] = None,
        use_qk_l2norm_in_kernel: bool = False,
        chunk_size: int = 64,
    ):
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        g, o, A, final_state = chunk_block_causal_gated_delta_rule_fwd(
            q=q, k=k, v=v, g=g, beta=beta,
            scale=scale, block_size=block_size,
            initial_state=initial_state,
            output_final_state=output_final_state,
            causal_mode=causal_mode,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens)
        ctx.scale = scale
        ctx.block_size = block_size
        ctx.causal_mode = causal_mode
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.chunk_size = chunk_size
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor,
    ):
        q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens = ctx.saved_tensors
        dq, dk, dv, db, dg, dh0 = chunk_block_causal_gated_delta_rule_bwd(
            q=q, k=k, v=v, g=g, beta=beta, A=A,
            scale=ctx.scale, block_size=ctx.block_size,
            initial_state=initial_state, do=do, dht=dht,
            causal_mode=ctx.causal_mode,
            cu_seqlens=cu_seqlens,
            chunk_size=ctx.chunk_size,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta), None, None, dh0, None, None, None, None, None


@torch.compiler.disable
def chunk_block_causal_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    block_size: int = 4,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    causal_mode: int = 0,
    **kwargs,
):
    r"""
    Block-causal gated delta rule attention (inference-only, no block_train).

    Args:
        q: queries [B, T, H, K]
        k: keys [B, T, H, K]
        v: values [B, T, H, V]
        g: gating (log space) [B, T, H]
        beta: betas [B, T, H]
        scale: scale factor (default 1/sqrt(K))
        block_size: causal block size
        initial_state: [N, H, K, V]
        output_final_state: whether to return final state
        use_qk_l2norm_in_kernel: L2-normalize q,k
        cu_seqlens: [N+1] for variable-length
        causal_mode: 0=block-causal, 1=token-causal

    Returns:
        o: [B, T, H, V], final_state: [N, H, K, V] or None
    """
    if 'head_first' in kwargs:
        warnings.warn("head_first is deprecated.")

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f"Batch size must be 1 with cu_seqlens, got {q.shape[0]}")
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"initial_state count {initial_state.shape[0]} != sequences {len(cu_seqlens) - 1}"
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    chunk_size = compute_aligned_chunk_size(block_size)
    assert chunk_size >= block_size, (
        f"block_size={block_size} too large. compute_aligned_chunk_size={chunk_size}."
    )
    assert causal_mode in (0, 1), f"causal_mode must be 0 or 1, got {causal_mode}"

    o, final_state = ChunkBlockCausalGatedDeltaRuleFunction.apply(
        q, k, v, g, beta,
        scale, block_size, initial_state, output_final_state,
        causal_mode, cu_seqlens, use_qk_l2norm_in_kernel,
        chunk_size,
    )
    return o, final_state
