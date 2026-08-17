# Copyright (c) 2026 Yuchen Zhu
# Portions copyright (c) 2023-2025 Songlin Yang, Yu Zhang
# Modified from Flash Linear Attention (FLA) for HybridDiffusion block-causal training.

import warnings

import torch
import torch.nn.functional as F

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from .chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from .wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

from . import _profiling as _prof
from .chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from .cumsum import chunk_local_cumsum
from .solve_tril import solve_tril

from .chunk_o import (
    chunk_block_causal_fwd_o,
    chunk_block_causal_bwd_dAv,
    chunk_block_causal_bwd_dqkwg,
    compute_block_causal_aqk,
)
from .fused_recurrent_state import _bwd_wy_chunked, fused_recurrent_gated_delta_state, chunk_refine_gda_state
from .chunk_two_stream_wy import (
    TwoStreamWYNoisyTritonFunctionV2,
    TwoStreamWYNoisyTritonImprovedFunction,
)
from .chunk_fla_style_wy import (
    two_stream_fla_style_full_fwd_triton,
    two_stream_fla_style_full_bwd_triton,
)


def compute_aligned_chunk_size(block_size: int, max_BT: int = 64) -> int:
    """Compute the largest chunk_size <= max_BT that is a multiple of block_size."""
    return (max_BT // block_size) * block_size


def _packed_chunk_start_grads_to_chunk_end_grads(
    dh_chunks: torch.Tensor,
    doc_lens: list[int],
    chunk_size: int,
) -> torch.Tensor:
    """Shift packed chunk-start gradients onto the previous chunk-end lattice."""
    dh_end = torch.zeros_like(dh_chunks)
    chunk_cursor = 0
    for doc_len in doc_lens:
        doc_num_chunks = max(1, (doc_len + chunk_size - 1) // chunk_size)
        if doc_num_chunks > 1:
            dh_end[:, chunk_cursor:chunk_cursor + doc_num_chunks - 1] = (
                dh_chunks[:, chunk_cursor + 1:chunk_cursor + doc_num_chunks]
            )
        chunk_cursor += doc_num_chunks
    return dh_end


def _packed_chunk_start_grads_to_chunk_end_grads_no_sync(
    dh_chunks: torch.Tensor,
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift packed chunk-start grads to chunk-end grads without CPU item syncs."""
    device = dh_chunks.device
    cu = cu_seqlens.to(device=device, non_blocking=True)
    doc_lens = cu[1:] - cu[:-1]
    doc_nc = torch.clamp((doc_lens + chunk_size - 1) // chunk_size, min=1).long()
    n_docs = doc_nc.shape[0]
    offsets = torch.empty((n_docs,), dtype=torch.long, device=device)
    offsets[:1].zero_()
    if n_docs > 1:
        offsets[1:] = torch.cumsum(doc_nc[:-1], dim=0)

    total_nc = dh_chunks.shape[1]
    chunk_ids = torch.arange(total_nc, dtype=torch.long, device=device)
    if n_docs == 1:
        doc_idx = torch.zeros_like(chunk_ids)
    else:
        doc_idx = torch.searchsorted(offsets[1:], chunk_ids, right=True)
    chunk_in_doc = chunk_ids - offsets[doc_idx]
    has_next_in_doc = chunk_in_doc + 1 < doc_nc[doc_idx]

    next_ids = torch.clamp(chunk_ids + 1, max=max(total_nc - 1, 0))
    shifted = dh_chunks.index_select(1, next_ids)
    dh_end = shifted * has_next_in_doc.to(dh_chunks.dtype).view(
        1, total_nc, *([1] * (dh_chunks.ndim - 2))
    )
    return dh_end, offsets


def _unpacked_chunk_start_grads_to_chunk_end_grads(
    dh_chunks: torch.Tensor,
) -> torch.Tensor:
    """Shift dense chunk-start gradients onto the previous chunk-end lattice."""
    dh_end = torch.zeros_like(dh_chunks)
    if dh_chunks.shape[1] > 1:
        dh_end[:, :-1] = dh_chunks[:, 1:]
    return dh_end


def _pad_block_train_inputs(q_c, k_c, v_c, g_c, beta_c,
                            q_n, k_n, v_n, g_n, beta_n,
                            half_T, block_size):
    """Zero-pad clean/noisy tensors so half_T becomes a multiple of block_size.

    Returns the (possibly padded) tensors, the new half_T, and pad_len.
    All returned tensors are guaranteed contiguous (Triton kernels require this).
    """
    pad_len = (-half_T) % block_size
    if pad_len == 0:
        return (q_c.contiguous(), k_c.contiguous(), v_c.contiguous(),
                g_c.contiguous(), beta_c.contiguous(),
                q_n.contiguous(), k_n.contiguous(), v_n.contiguous(),
                g_n.contiguous(), beta_n.contiguous(),
                half_T, 0)
    p4 = (0, 0, 0, 0, 0, pad_len)  # 4D [B, T, H, K/V]
    p3 = (0, 0, 0, pad_len)        # 3D [B, T, H]
    return (F.pad(q_c, p4), F.pad(k_c, p4), F.pad(v_c, p4),
            F.pad(g_c, p3), F.pad(beta_c, p3),
            F.pad(q_n, p4), F.pad(k_n, p4), F.pad(v_n, p4),
            F.pad(g_n, p3), F.pad(beta_n, p3),
            half_T + pad_len, pad_len)


def _validate_block_train_packed_boundaries(
    cu_seqlens: torch.LongTensor | None,
    block_size: int,
) -> None:
    """Block-train packed docs may end partial, but doc starts must align.

    The two-stream block mask is defined on block boundaries.  Therefore every
    document start must be a block boundary.  The final packed end is allowed to
    be a partial block because the implementation pads the trailing tokens.
    """
    if cu_seqlens is None or block_size <= 1:
        return
    starts = cu_seqlens[:-1]
    if starts.numel() == 0:
        return
    bad = torch.any(torch.remainder(starts, block_size) != 0)
    if bool(bad.item()):
        raise ValueError(
            "block_train with cu_seqlens requires document starts "
            f"(cu_seqlens[:-1]) to be multiples of block_size={block_size}. "
            "The final cu_seqlens[-1] may be a partial block."
        )


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
    cu_seqlens: torch.LongTensor | None = None,
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


def chunk_block_causal_gda_fwd_with_h(
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
    cu_seqlens: torch.LongTensor | None = None,
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
    return g, o, A, final_state, h, v_new


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
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    dv_new_inject: torch.Tensor | None = None,
    dG_inject: torch.Tensor | None = None,
    dh_inject: torch.Tensor | None = None,
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
        cu_seqlens: torch.LongTensor | None = None,
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


class ChunkBlockCausalGDAWithHFunction(torch.autograd.Function):
    """Same as ChunkBlockCausalGatedDeltaRuleFunction but also returns h_chunks.

    The extra output is intended for reuse by h_all state extraction
    (skipping the redundant WY pipeline).  Callers should .detach()
    h_chunks before downstream use so no extra backward cost is incurred.
    """

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
        output_final_state: bool = False,
        causal_mode: int = 0,
        cu_seqlens: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        chunk_size: int = 64,
    ):
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        g, o, A, final_state, h_chunks, _ = chunk_block_causal_gda_fwd_with_h(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
            block_size=block_size, initial_state=initial_state,
            output_final_state=output_final_state, causal_mode=causal_mode,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens)
        ctx.scale = scale
        ctx.block_size = block_size
        ctx.causal_mode = causal_mode
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.chunk_size = chunk_size
        return o.to(q.dtype), final_state, h_chunks

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht, dh_chunks):
        do = do.contiguous()
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


class ChunkBlockCausalGDAWithHFunctionV2(torch.autograd.Function):
    """Clean-side V2 with graph-connected ``h_chunks`` semantics.

    Forward reuses the fast clean chunk/WY path that emits chunk-start states.
    Backward is split into two efficient pieces:
    - the ordinary clean-output path uses the existing fast clean backward
    - the extra ``dh_chunks`` path uses fast state-extraction kernels on
      shifted chunk-end states

    This preserves the V2 autograd contract without falling back to Python
    block-by-block rematerialization.
    """

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
        output_final_state: bool = False,
        causal_mode: int = 0,
        cu_seqlens: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        chunk_size: int = 64,
    ):
        q_rstd, k_rstd = None, None
        q_eff = q
        k_eff = k
        if use_qk_l2norm_in_kernel:
            q_eff, q_rstd = l2norm_fwd(q)
            k_eff, k_rstd = l2norm_fwd(k)

        g_cumsum, o, A, final_state, h_chunks, v_new = chunk_block_causal_gda_fwd_with_h(
            q=q_eff,
            k=k_eff,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            block_size=block_size,
            initial_state=initial_state,
            output_final_state=output_final_state,
            causal_mode=causal_mode,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )
        ctx.save_for_backward(
            q, k,
            q_eff, q_rstd,
            k_eff, k_rstd,
            v, g, g_cumsum, beta, A, initial_state, cu_seqlens,
        )
        ctx.scale = scale
        ctx.block_size = block_size
        ctx.output_final_state = output_final_state
        ctx.causal_mode = causal_mode
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.chunk_size = chunk_size
        return o, final_state, h_chunks, v_new, g_cumsum

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht, dh_chunks, dv_new_ext, dG_c_ext):
        (
            q_raw, k_raw,
            q_eff, q_rstd,
            k_eff, k_rstd,
            v, g_raw, g_cumsum, beta, A, initial_state, cu_seqlens,
        ) = ctx.saved_tensors

        # Packed V2 status:
        # - ordinary clean-output / final-state gradients reuse the existing
        #   packed fast clean autograd path
        # - `dh_chunks` now also uses a packed fast exact state-gradient path
        #   for both chunk-aligned and partial trailing docs
        # - packed partial-doc support goes through the WY/state backward on
        #   the irregular chunk lattice induced by `cu_seqlens`
        if cu_seqlens is not None:
            do = do.contiguous()

            dq_eff, dk_eff, dv, dbeta, dg, dh0 = chunk_block_causal_gated_delta_rule_bwd(
                q=q_eff, k=k_eff, v=v, g=g_cumsum, beta=beta, A=A,
                scale=ctx.scale, block_size=ctx.block_size,
                initial_state=initial_state, do=do, dht=dht,
                causal_mode=ctx.causal_mode,
                cu_seqlens=cu_seqlens,
                chunk_size=ctx.chunk_size,
                dv_new_inject=dv_new_ext,
                dG_inject=dG_c_ext,
            )

            if ctx.use_qk_l2norm_in_kernel:
                dq = l2norm_bwd(q_raw, q_rstd, dq_eff)
                dk = l2norm_bwd(k_raw, k_rstd, dk_eff)
            else:
                dq = dq_eff
                dk = dk_eff

            if dh_chunks is not None:
                num_docs = len(cu_seqlens) - 1
                doc_lens = [(cu_seqlens[i + 1] - cu_seqlens[i]).item() for i in range(num_docs)]
                packed_chunk_aligned = all(doc_len % ctx.chunk_size == 0 for doc_len in doc_lens)

                dk_state = torch.zeros_like(dk)
                dv_state = torch.zeros_like(dv)
                dg_state = torch.zeros_like(dg)
                dbeta_state = torch.zeros_like(dbeta)
                dh0_state = torch.zeros_like(initial_state) if initial_state is not None else None

                if packed_chunk_aligned:
                    with torch.enable_grad():
                        if ctx.use_qk_l2norm_in_kernel:
                            k_state_base = k_raw.detach()
                            k_state_eff, k_state_rstd = l2norm_fwd(k_state_base)
                            k_state_eff = k_state_eff.detach().requires_grad_(True)
                        else:
                            k_state_base = None
                            k_state_rstd = None
                            k_state_eff = k_raw.detach().requires_grad_(True)

                        v_state_eff = v.detach().requires_grad_(True)
                        g_state_eff = g_raw.detach().requires_grad_(True)
                        beta_state_eff = beta.detach().requires_grad_(True)

                        if initial_state is not None:
                            h0_state_eff = initial_state.detach().requires_grad_(True)
                            state_inputs = (k_state_eff, v_state_eff, g_state_eff, beta_state_eff, h0_state_eff)
                        else:
                            h0_state_eff = None
                            state_inputs = (k_state_eff, v_state_eff, g_state_eff, beta_state_eff)

                        h_end = fused_recurrent_gated_delta_state(
                            k=k_state_eff,
                            v=v_state_eff,
                            g=g_state_eff,
                            beta=beta_state_eff,
                            h0=h0_state_eff,
                            block_size=ctx.chunk_size,
                            cu_seqlens=cu_seqlens,
                        )

                        dh_end = torch.zeros_like(h_end)
                        chunk_cursor = 0
                        out_cursor = 0
                        for doc_idx, doc_len in enumerate(doc_lens):
                            doc_num_chunks = max(1, doc_len // ctx.chunk_size)
                            if initial_state is not None:
                                dh0_state[doc_idx] += dh_chunks[0, chunk_cursor]
                            if doc_num_chunks > 1:
                                dh_end[:, out_cursor:out_cursor + doc_num_chunks - 1] = dh_chunks[:, chunk_cursor + 1:chunk_cursor + doc_num_chunks]
                            chunk_cursor += doc_num_chunks
                            out_cursor += doc_num_chunks

                        state_grads = torch.autograd.grad(
                            outputs=h_end,
                            inputs=state_inputs,
                            grad_outputs=dh_end,
                            allow_unused=True,
                        )

                    if initial_state is not None:
                        dk_doc, dv_doc, dg_doc, dbeta_doc, dh0_doc = state_grads
                        dh0_state += torch.zeros_like(dh0_state) if dh0_doc is None else dh0_doc
                    else:
                        dk_doc, dv_doc, dg_doc, dbeta_doc = state_grads

                    if ctx.use_qk_l2norm_in_kernel and dk_doc is not None:
                        dk_doc = l2norm_bwd(k_state_base, k_state_rstd, dk_doc)

                    if dk_doc is not None:
                        dk_state += dk_doc
                    if dv_doc is not None:
                        dv_state += dv_doc
                    if dg_doc is not None:
                        dg_state += dg_doc
                    if dbeta_doc is not None:
                        dbeta_state += dbeta_doc
                else:
                    if ctx.use_qk_l2norm_in_kernel:
                        k_state_base = k_raw
                        k_state_eff, k_state_rstd = l2norm_fwd(k_state_base)
                    else:
                        k_state_base = None
                        k_state_rstd = None
                        k_state_eff = k_raw

                    dh_end = _packed_chunk_start_grads_to_chunk_end_grads(
                        dh_chunks=dh_chunks,
                        doc_lens=doc_lens,
                        chunk_size=ctx.chunk_size,
                    )
                    dk_doc, dv_doc, dg_doc, dbeta_doc, dh0_doc = _bwd_wy_chunked(
                        k=k_state_eff,
                        v=v,
                        g=g_raw,
                        beta=beta,
                        h0=initial_state,
                        dh_blocks=dh_end,
                        block_size=ctx.chunk_size,
                        cu_seqlens=cu_seqlens,
                    )

                    if ctx.use_qk_l2norm_in_kernel:
                        dk_doc = l2norm_bwd(k_state_base, k_state_rstd, dk_doc)

                    if initial_state is not None:
                        chunk_cursor = 0
                        for doc_idx, doc_len in enumerate(doc_lens):
                            dh0_state[doc_idx] += dh_chunks[0, chunk_cursor]
                            chunk_cursor += max(1, (doc_len + ctx.chunk_size - 1) // ctx.chunk_size)
                        dh0_state += torch.zeros_like(dh0_state) if dh0_doc is None else dh0_doc

                    if dk_doc is not None:
                        dk_state += dk_doc
                    if dv_doc is not None:
                        dv_state += dv_doc
                    if dg_doc is not None:
                        dg_state += dg_doc
                    if dbeta_doc is not None:
                        dbeta_state += dbeta_doc

                dk = dk + dk_state
                dv = dv + dv_state
                dg = dg + dg_state
                dbeta = dbeta + dbeta_state
                if initial_state is not None:
                    dh0 = (torch.zeros_like(initial_state) if dh0 is None else dh0) + dh0_state

            dq_eff = dq
            dk_eff = dk
        # Unpacked trailing partial chunks stay on a fast exact path: the
        # full-chunk prefix reuses the existing fast backward/state path, and
        # the short tail is padded to one chunk so it can reuse the fast clean
        # kernels too.
        elif q_eff.shape[1] % ctx.chunk_size != 0:
            dq, dk, dv, dg, dbeta, dh0 = _clean_v2_bwd_unpacked_trailing_partial(
                q_raw=q_raw,
                k_raw=k_raw,
                v=v,
                g_raw=g_raw,
                beta=beta,
                q_eff=q_eff,
                q_rstd=q_rstd,
                k_eff=k_eff,
                k_rstd=k_rstd,
                scale=ctx.scale,
                block_size=ctx.block_size,
                initial_state=initial_state,
                output_final_state=ctx.output_final_state,
                causal_mode=ctx.causal_mode,
                use_qk_l2norm_in_kernel=ctx.use_qk_l2norm_in_kernel,
                chunk_size=ctx.chunk_size,
                do=do,
                dht=dht,
                dh_chunks=dh_chunks,
                dv_new_inject=dv_new_ext,
                dG_inject=dG_c_ext,
            )
            return dq, dk, dv, dg, dbeta, None, None, dh0, None, None, None, None, None

        else:
            do = do.contiguous()
            dq_eff, dk_eff, dv, dbeta, dg, dh0 = chunk_block_causal_gated_delta_rule_bwd(
                q=q_eff,
                k=k_eff,
                v=v,
                g=g_cumsum,
                beta=beta,
                A=A,
                scale=ctx.scale,
                block_size=ctx.block_size,
                initial_state=initial_state,
                do=do,
                dht=dht,
                causal_mode=ctx.causal_mode,
                cu_seqlens=cu_seqlens,
                chunk_size=ctx.chunk_size,
                dv_new_inject=dv_new_ext,
                dG_inject=dG_c_ext,
            )

        if dh_chunks is not None:
            if cu_seqlens is None:
                dh_end = torch.zeros_like(dh_chunks)
                if dh_chunks.shape[1] > 1:
                    dh_end[:, :-1] = dh_chunks[:, 1:]
                dh_chunks_h0 = dh_chunks[:, 0] if initial_state is not None else None
                has_dh_end = dh_chunks.shape[1] > 1

            if cu_seqlens is None and has_dh_end:
                with torch.enable_grad():
                    k_state = k_eff.detach().requires_grad_(True)
                    v_state = v.detach().requires_grad_(True)
                    g_state = g_raw.detach().requires_grad_(True)
                    beta_state = beta.detach().requires_grad_(True)

                    if initial_state is not None:
                        h0_state = initial_state.detach().requires_grad_(True)
                        state_inputs = (k_state, v_state, g_state, beta_state, h0_state)
                    else:
                        h0_state = None
                        state_inputs = (k_state, v_state, g_state, beta_state)

                    h_end = fused_recurrent_gated_delta_state(
                        k=k_state,
                        v=v_state,
                        g=g_state,
                        beta=beta_state,
                        h0=h0_state,
                        block_size=ctx.chunk_size,
                        cu_seqlens=cu_seqlens,
                    )

                    state_grads = torch.autograd.grad(
                        outputs=h_end,
                        inputs=state_inputs,
                        grad_outputs=dh_end,
                        allow_unused=True,
                    )

                if initial_state is not None:
                    dk_state, dv_state, dg_state, dbeta_state, dh0_state = state_grads
                    dh0 = (torch.zeros_like(initial_state) if dh0 is None else dh0) + dh0_state
                    if dh_chunks_h0 is not None:
                        dh0 = dh0 + dh_chunks_h0
                else:
                    dk_state, dv_state, dg_state, dbeta_state = state_grads
                    dh0 = None

                dk_eff = dk_eff + dk_state
                dv = dv + dv_state
                dg = dg + dg_state
                dbeta = dbeta + dbeta_state
            elif cu_seqlens is None and initial_state is not None:
                dh0 = (torch.zeros_like(initial_state) if dh0 is None else dh0) + dh_chunks_h0

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q_eff, q_rstd, dq_eff)
            dk = l2norm_bwd(k_eff, k_rstd, dk_eff)
        else:
            dq = dq_eff
            dk = dk_eff

        return dq.to(q_eff), dk.to(k_eff), dv.to(v), dg.to(g_raw), dbeta.to(beta), None, None, dh0, None, None, None, None, None


class ChunkBlockCausalGDAWithHFunctionV2Improved(torch.autograd.Function):
    """Clean WY path used by the improved and fused two-stream routes.

    This keeps the forward and ordinary output backward identical to V2, but
    routes gradients flowing through ``h_chunks`` into the ordinary clean
    backward's state-gradient pass instead of creating a nested autograd island
    around ``fused_recurrent_gated_delta_state``.
    """

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
        output_final_state: bool = False,
        causal_mode: int = 0,
        cu_seqlens: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        chunk_size: int = 64,
    ):
        q_rstd, k_rstd = None, None
        q_eff = q
        k_eff = k
        if use_qk_l2norm_in_kernel:
            q_eff, q_rstd = l2norm_fwd(q)
            k_eff, k_rstd = l2norm_fwd(k)

        g_cumsum, o, A, final_state, h_chunks, v_new = chunk_block_causal_gda_fwd_with_h(
            q=q_eff,
            k=k_eff,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            block_size=block_size,
            initial_state=initial_state,
            output_final_state=output_final_state,
            causal_mode=causal_mode,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )
        ctx.save_for_backward(
            q, k,
            q_eff, q_rstd,
            k_eff, k_rstd,
            v, g, g_cumsum, beta, A, initial_state, cu_seqlens,
        )
        ctx.scale = scale
        ctx.block_size = block_size
        ctx.output_final_state = output_final_state
        ctx.causal_mode = causal_mode
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.chunk_size = chunk_size
        return o, final_state, h_chunks, v_new, g_cumsum

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht, dh_chunks, dv_new_ext, dG_c_ext):
        (
            q_raw, k_raw,
            q_eff, q_rstd,
            k_eff, k_rstd,
            v, g_raw, g_cumsum, beta, A, initial_state, cu_seqlens,
        ) = ctx.saved_tensors

        if cu_seqlens is not None:
            do = do.contiguous()
            dh_inject = None
            doc_offsets = None
            if dh_chunks is not None:
                dh_inject, doc_offsets = _packed_chunk_start_grads_to_chunk_end_grads_no_sync(
                    dh_chunks=dh_chunks,
                    cu_seqlens=cu_seqlens,
                    chunk_size=ctx.chunk_size,
                )
            dq_eff, dk_eff, dv, dbeta, dg, dh0 = chunk_block_causal_gated_delta_rule_bwd(
                q=q_eff,
                k=k_eff,
                v=v,
                g=g_cumsum,
                beta=beta,
                A=A,
                scale=ctx.scale,
                block_size=ctx.block_size,
                initial_state=initial_state,
                do=do,
                dht=dht,
                causal_mode=ctx.causal_mode,
                cu_seqlens=cu_seqlens,
                chunk_size=ctx.chunk_size,
                dv_new_inject=dv_new_ext,
                dG_inject=dG_c_ext,
                dh_inject=dh_inject,
            )

            if dh_chunks is not None and initial_state is not None:
                dh0_direct = dh_chunks[0].index_select(0, doc_offsets)
                dh0 = (
                    torch.zeros_like(initial_state)
                    if dh0 is None
                    else dh0
                ) + dh0_direct

        elif q_eff.shape[1] % ctx.chunk_size != 0:
            dq, dk, dv, dg, dbeta, dh0 = _clean_v2_bwd_unpacked_trailing_partial(
                q_raw=q_raw,
                k_raw=k_raw,
                v=v,
                g_raw=g_raw,
                beta=beta,
                q_eff=q_eff,
                q_rstd=q_rstd,
                k_eff=k_eff,
                k_rstd=k_rstd,
                scale=ctx.scale,
                block_size=ctx.block_size,
                initial_state=initial_state,
                output_final_state=ctx.output_final_state,
                causal_mode=ctx.causal_mode,
                use_qk_l2norm_in_kernel=ctx.use_qk_l2norm_in_kernel,
                chunk_size=ctx.chunk_size,
                do=do,
                dht=dht,
                dh_chunks=dh_chunks,
                dv_new_inject=dv_new_ext,
                dG_inject=dG_c_ext,
                explicit_state_path=True,
            )
            return dq, dk, dv, dg, dbeta, None, None, dh0, None, None, None, None, None

        else:
            do = do.contiguous()
            dh_inject = (
                _unpacked_chunk_start_grads_to_chunk_end_grads(dh_chunks)
                if dh_chunks is not None
                else None
            )
            dq_eff, dk_eff, dv, dbeta, dg, dh0 = chunk_block_causal_gated_delta_rule_bwd(
                q=q_eff,
                k=k_eff,
                v=v,
                g=g_cumsum,
                beta=beta,
                A=A,
                scale=ctx.scale,
                block_size=ctx.block_size,
                initial_state=initial_state,
                do=do,
                dht=dht,
                causal_mode=ctx.causal_mode,
                cu_seqlens=cu_seqlens,
                chunk_size=ctx.chunk_size,
                dv_new_inject=dv_new_ext,
                dG_inject=dG_c_ext,
                dh_inject=dh_inject,
            )

            if dh_chunks is not None and initial_state is not None:
                dh0 = (
                    torch.zeros_like(initial_state)
                    if dh0 is None
                    else dh0
                ) + dh_chunks[:, 0]

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q_raw, q_rstd, dq_eff)
            dk = l2norm_bwd(k_raw, k_rstd, dk_eff)
        else:
            dq = dq_eff
            dk = dk_eff

        return dq.to(q_eff), dk.to(k_eff), dv.to(v), dg.to(g_raw), dbeta.to(beta), None, None, dh0, None, None, None, None, None


def _two_stream_noisy_reference(
    k_c: torch.Tensor,
    v_c: torch.Tensor,
    g_c: torch.Tensor,
    beta_c: torch.Tensor,
    q_n: torch.Tensor,
    k_n: torch.Tensor,
    v_n: torch.Tensor,
    g_n: torch.Tensor,
    beta_n: torch.Tensor,
    h_chunks: torch.Tensor,
    scale: float,
    block_size: int,
    use_qk_l2norm_in_kernel: bool,
    causal_mode_clean: int,
    causal_mode_noisy: int,
    chunk_size: int,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    """Exact reference two-stream noisy execution from chunk-start checkpoints.

    The first noisy block of each document uses zero state, not the clean
    stream's initial state.  Clean-prefix advancement still starts from the
    document initial state so later noisy blocks can see prior clean blocks.
    """
    if cu_seqlens is not None:
        noisy_parts = []
        chunk_cursor = 0
        blocks_per_chunk = max(1, chunk_size // block_size)
        for doc_idx in range(len(cu_seqlens) - 1):
            s = cu_seqlens[doc_idx].item()
            e = cu_seqlens[doc_idx + 1].item()
            doc_len = e - s
            doc_num_chunks = max(1, (doc_len + chunk_size - 1) // chunk_size)
            h_doc = h_chunks[:, chunk_cursor:chunk_cursor + doc_num_chunks]
            chunk_cursor += doc_num_chunks
            noisy_parts.append(_two_stream_noisy_reference(
                k_c=k_c[:, s:e],
                v_c=v_c[:, s:e],
                g_c=g_c[:, s:e],
                beta_c=beta_c[:, s:e],
                q_n=q_n[:, s:e],
                k_n=k_n[:, s:e],
                v_n=v_n[:, s:e],
                g_n=g_n[:, s:e],
                beta_n=beta_n[:, s:e],
                h_chunks=h_doc,
                scale=scale,
                block_size=block_size,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                causal_mode_clean=causal_mode_clean,
                causal_mode_noisy=causal_mode_noisy,
                chunk_size=chunk_size,
                cu_seqlens=None,
            )[:, :doc_len])
        return torch.cat(noisy_parts, dim=1)

    _, half_T, _, _ = k_c.shape
    blocks_per_chunk = max(1, chunk_size // block_size)
    num_blocks = (half_T + block_size - 1) // block_size

    noisy_blocks = []
    for chunk_idx in range(h_chunks.shape[1]):
        clean_state = h_chunks[:, chunk_idx]
        start_block = chunk_idx * blocks_per_chunk
        end_block = min(num_blocks, start_block + blocks_per_chunk)

        for block_idx in range(start_block, end_block):
            s = block_idx * block_size
            e = min(s + block_size, half_T)
            block_len = e - s
            pad_len = block_size - block_len
            q_blk = q_n[:, s:e]
            k_blk = k_n[:, s:e]
            v_blk = v_n[:, s:e]
            g_blk = g_n[:, s:e]
            beta_blk = beta_n[:, s:e]
            kc_blk = k_c[:, s:e]
            vc_blk = v_c[:, s:e]
            gc_blk = g_c[:, s:e]
            betac_blk = beta_c[:, s:e]
            if pad_len:
                p4 = (0, 0, 0, 0, 0, pad_len)
                p3 = (0, 0, 0, pad_len)
                q_blk = F.pad(q_blk, p4)
                k_blk = F.pad(k_blk, p4)
                v_blk = F.pad(v_blk, p4)
                g_blk = F.pad(g_blk, p3)
                beta_blk = F.pad(beta_blk, p3)
                kc_blk = F.pad(kc_blk, p4)
                vc_blk = F.pad(vc_blk, p4)
                gc_blk = F.pad(gc_blk, p3)
                betac_blk = F.pad(betac_blk, p3)
            noisy_state = (
                torch.zeros_like(clean_state)
                if block_idx == 0
                else clean_state
            )

            o_noisy_blk, _ = ChunkBlockCausalGatedDeltaRuleFunction.apply(
                q_blk, k_blk, v_blk,
                g_blk, beta_blk,
                scale, block_size, noisy_state, False,
                causal_mode_noisy, None, use_qk_l2norm_in_kernel,
                chunk_size,
            )
            noisy_blocks.append(o_noisy_blk[:, :block_len])

            _, clean_state = ChunkBlockCausalGatedDeltaRuleFunction.apply(
                kc_blk, kc_blk, vc_blk,
                gc_blk, betac_blk,
                scale, block_size, clean_state, True,
                causal_mode_clean, None, use_qk_l2norm_in_kernel,
                chunk_size,
            )

    return torch.cat(noisy_blocks, dim=1)


def _two_stream_noisy_fla_style_reference(
    k_c: torch.Tensor,
    v_c: torch.Tensor,
    g_c: torch.Tensor,
    beta_c: torch.Tensor,
    q_n: torch.Tensor,
    k_n: torch.Tensor,
    v_n: torch.Tensor,
    g_n: torch.Tensor,
    beta_n: torch.Tensor,
    h_chunks: torch.Tensor,
    scale: float,
    block_size: int,
    use_qk_l2norm_in_kernel: bool,
    causal_mode_clean: int,
    causal_mode_noisy: int,
    chunk_size: int,
    cu_seqlens: torch.LongTensor | None = None,
    return_parts: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact slow full/cross/local oracle for the fused two-stream path.

    This is intentionally a slow oracle, not the optimized implementation.
    For each noisy block, the current two-stream semantics are decomposed as:

        full  = noisy_block(initial_state=noisy_visible_clean_prefix_state)
        local = noisy_block(initial_state=0)
        cross = full - local

    The first noisy block of each document intentionally uses zero state even
    when the clean stream has a nonzero ``initial_state``; this matches the
    block-train mask implemented by the production dispatch modes.

    Later kernels can replace ``cross`` and ``local`` independently while
    comparing against this function.
    """
    if cu_seqlens is not None:
        full_parts = []
        cross_parts = []
        local_parts = []
        chunk_cursor = 0
        for doc_idx in range(len(cu_seqlens) - 1):
            s = cu_seqlens[doc_idx].item()
            e = cu_seqlens[doc_idx + 1].item()
            doc_len = e - s
            doc_num_chunks = max(1, (doc_len + chunk_size - 1) // chunk_size)
            h_doc = h_chunks[:, chunk_cursor:chunk_cursor + doc_num_chunks]
            chunk_cursor += doc_num_chunks

            doc_result = _two_stream_noisy_fla_style_reference(
                k_c=k_c[:, s:e],
                v_c=v_c[:, s:e],
                g_c=g_c[:, s:e],
                beta_c=beta_c[:, s:e],
                q_n=q_n[:, s:e],
                k_n=k_n[:, s:e],
                v_n=v_n[:, s:e],
                g_n=g_n[:, s:e],
                beta_n=beta_n[:, s:e],
                h_chunks=h_doc,
                scale=scale,
                block_size=block_size,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                causal_mode_clean=causal_mode_clean,
                causal_mode_noisy=causal_mode_noisy,
                chunk_size=chunk_size,
                cu_seqlens=None,
                return_parts=return_parts,
            )
            if return_parts:
                full_doc, cross_doc, local_doc = doc_result
                full_parts.append(full_doc[:, :doc_len])
                cross_parts.append(cross_doc[:, :doc_len])
                local_parts.append(local_doc[:, :doc_len])
            else:
                full_parts.append(doc_result[:, :doc_len])

        if return_parts:
            return (
                torch.cat(full_parts, dim=1),
                torch.cat(cross_parts, dim=1),
                torch.cat(local_parts, dim=1),
            )
        return torch.cat(full_parts, dim=1)

    _, half_T, _, _ = k_c.shape
    blocks_per_chunk = max(1, chunk_size // block_size)
    num_blocks = (half_T + block_size - 1) // block_size

    full_blocks = []
    cross_blocks = []
    local_blocks = []
    for chunk_idx in range(h_chunks.shape[1]):
        clean_state = h_chunks[:, chunk_idx]
        start_block = chunk_idx * blocks_per_chunk
        end_block = min(num_blocks, start_block + blocks_per_chunk)

        for block_idx in range(start_block, end_block):
            s = block_idx * block_size
            e = min(s + block_size, half_T)
            block_len = e - s
            pad_len = block_size - block_len
            q_blk = q_n[:, s:e]
            k_blk = k_n[:, s:e]
            v_blk = v_n[:, s:e]
            g_blk = g_n[:, s:e]
            beta_blk = beta_n[:, s:e]
            kc_blk = k_c[:, s:e]
            vc_blk = v_c[:, s:e]
            gc_blk = g_c[:, s:e]
            betac_blk = beta_c[:, s:e]
            if pad_len:
                p4 = (0, 0, 0, 0, 0, pad_len)
                p3 = (0, 0, 0, pad_len)
                q_blk = F.pad(q_blk, p4)
                k_blk = F.pad(k_blk, p4)
                v_blk = F.pad(v_blk, p4)
                g_blk = F.pad(g_blk, p3)
                beta_blk = F.pad(beta_blk, p3)
                kc_blk = F.pad(kc_blk, p4)
                vc_blk = F.pad(vc_blk, p4)
                gc_blk = F.pad(gc_blk, p3)
                betac_blk = F.pad(betac_blk, p3)
            noisy_state = (
                torch.zeros_like(clean_state)
                if block_idx == 0
                else clean_state
            )

            full_blk, _ = ChunkBlockCausalGatedDeltaRuleFunction.apply(
                q_blk, k_blk, v_blk,
                g_blk, beta_blk,
                scale, block_size, noisy_state, False,
                causal_mode_noisy, None, use_qk_l2norm_in_kernel,
                chunk_size,
            )
            full_blocks.append(full_blk[:, :block_len])

            if return_parts:
                zero_state = torch.zeros_like(clean_state)
                local_blk, _ = ChunkBlockCausalGatedDeltaRuleFunction.apply(
                    q_blk, k_blk, v_blk,
                    g_blk, beta_blk,
                    scale, block_size, zero_state, False,
                    causal_mode_noisy, None, use_qk_l2norm_in_kernel,
                    chunk_size,
                )
                local_blocks.append(local_blk[:, :block_len])
                cross_blocks.append((full_blk - local_blk)[:, :block_len])

            _, clean_state = ChunkBlockCausalGatedDeltaRuleFunction.apply(
                kc_blk, kc_blk, vc_blk,
                gc_blk, betac_blk,
                scale, block_size, clean_state, True,
                causal_mode_clean, None, use_qk_l2norm_in_kernel,
                chunk_size,
            )

    full = torch.cat(full_blocks, dim=1)
    if return_parts:
        return full, torch.cat(cross_blocks, dim=1), torch.cat(local_blocks, dim=1)
    return full


class TwoStreamFLAStyleFullFunction(torch.autograd.Function):
    """Merged full noisy forward/backward for the FLA-style dispatch."""

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
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
        scale: float,
        block_size: int,
        causal_mode_noisy: int,
        chunk_size: int,
        cu_seqlens: torch.LongTensor | None = None,
        checkpoint_stride: int | None = None,
        bwd_bv: int | None = None,
    ):
        out = two_stream_fla_style_full_fwd_triton(
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
            causal_mode=causal_mode_noisy,
            cu_seqlens=cu_seqlens,
        )
        ctx.save_for_backward(
            q_n, k_n, v_n, g_n, beta_n,
            k_c, v_c, g_c, beta_c, v_new_c, G_c, h_chunks,
            *([cu_seqlens] if cu_seqlens is not None else []),
        )
        ctx.scale = scale
        ctx.block_size = block_size
        ctx.causal_mode_noisy = causal_mode_noisy
        ctx.chunk_size = chunk_size
        ctx._has_cu_seqlens = cu_seqlens is not None
        ctx.checkpoint_stride = checkpoint_stride
        ctx.bwd_bv = bwd_bv
        return out

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do):
        saved = ctx.saved_tensors
        if ctx._has_cu_seqlens:
            (q_n, k_n, v_n, g_n, beta_n,
             k_c, v_c, g_c, beta_c, v_new_c, G_c, h_chunks, cu_seqlens) = saved
        else:
            (q_n, k_n, v_n, g_n, beta_n,
             k_c, v_c, g_c, beta_c, v_new_c, G_c, h_chunks) = saved
            cu_seqlens = None

        (
            dq_n,
            dk_n,
            dv_n,
            dg_n,
            dbeta_n,
            dk_c,
            dv_new_c,
            dG_c,
            dh_chunks,
        ) = two_stream_fla_style_full_bwd_triton(
            q_n=q_n,
            k_n=k_n,
            v_n=v_n,
            g_n=g_n,
            beta_n=beta_n,
            k_c=k_c,
            v_c=v_c,
            g_c=g_c,
            beta_c=beta_c,
            v_new_c=v_new_c,
            G_c=G_c,
            h_chunks=h_chunks,
            do=do,
            scale=ctx.scale,
            block_size=ctx.block_size,
            chunk_size=ctx.chunk_size,
            causal_mode=ctx.causal_mode_noisy,
            checkpoint_stride=ctx.checkpoint_stride,
            bwd_bv=ctx.bwd_bv,
            cu_seqlens=cu_seqlens,
        )
        return (
            dq_n.to(q_n), dk_n.to(k_n), dv_n.to(v_n),
            dg_n.to(g_n), dbeta_n.to(beta_n),
            dk_c.to(k_c), None, None, None, dv_new_c, dG_c,
            dh_chunks,
            None, None, None, None, None, None, None,
        )


def _doc_first_block_indices(cu_seqlens, block_size):
    """Block indices that are the first block of each document.

    Returns a LongTensor of shape ``[N_docs]`` or ``None`` when
    ``cu_seqlens`` is ``None``.
    """
    if cu_seqlens is None:
        return None
    return (cu_seqlens[:-1] // block_size).long()


def _h_all_to_noisy_init(
    h_all_states: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    block_size: int | None = None,
) -> torch.Tensor:
    """Construct noisy initial states from h_all_states [B, N, H, K, V].

    noisy_init[0] = zeros (no prior context for first noisy block)
    noisy_init[i] = h_all[i-1] (state after clean blocks 0..i-1)

    When ``cu_seqlens`` is provided (document packing), the first block
    of each document also gets zeros so that documents are isolated.
    The mask multiplication ensures backward gradients are correctly
    zeroed at document boundaries.
    """
    B, N = h_all_states.shape[:2]
    rest = h_all_states.shape[2:]
    h_zero = torch.zeros(B, 1, *rest, device=h_all_states.device, dtype=h_all_states.dtype)
    if N > 1:
        noisy_init = torch.cat([h_zero, h_all_states[:, :-1]], dim=1)
    else:
        noisy_init = h_zero

    if cu_seqlens is not None and block_size is not None:
        doc_firsts = _doc_first_block_indices(cu_seqlens, block_size)
        mask = torch.ones(N, device=h_all_states.device, dtype=h_all_states.dtype)
        mask[doc_firsts] = 0.0
        noisy_init = noisy_init * mask.view(1, N, *([1] * len(rest)))

    return noisy_init.reshape(B * N, *rest)


class HybridBlockCausalGDAFunction(torch.autograd.Function):
    """Hybrid autograd: fused-recurrent forward + chunked backward.

    The fused forward is fast and avoids materialising WY intermediates.
    The chunk backward parallelises across chunks using tensor cores.
    This combination is memory-optimal for short sequences (noisy blocks).
    """

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
        output_final_state: bool = False,
        causal_mode: int = 0,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        h_all_states: torch.Tensor | None = None,
        chunk_size: int = 64,
        doc_cu_seqlens: torch.LongTensor | None = None,
    ):
        from .fused_recurrent import fused_recurrent_block_causal_fwd

        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        g_cumsum = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens)

        o, final_state = fused_recurrent_block_causal_fwd(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
            block_size=block_size, causal_mode=causal_mode,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=cu_seqlens,
        )

        if h_all_states is not None:
            save_tensors = [q, q_rstd, k, k_rstd, v, g_cumsum, beta,
                            h_all_states, cu_seqlens]
            if doc_cu_seqlens is not None:
                save_tensors.append(doc_cu_seqlens)
            ctx.save_for_backward(*save_tensors)
        else:
            ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g_cumsum, beta,
                                  initial_state, cu_seqlens)
        ctx._has_h_all = h_all_states is not None
        ctx._has_doc_cu = doc_cu_seqlens is not None
        ctx.scale = scale
        ctx.block_size = block_size
        ctx.causal_mode = causal_mode
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.chunk_size = chunk_size
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        do = do.contiguous()
        if ctx._has_h_all:
            saved = ctx.saved_tensors
            q, q_rstd, k, k_rstd, v, g, beta, h_all_states, cu_seqlens = saved[:9]
            doc_cu = saved[9] if ctx._has_doc_cu else None
            initial_state = _h_all_to_noisy_init(
                h_all_states, cu_seqlens=doc_cu, block_size=ctx.block_size,
            )
        else:
            q, q_rstd, k, k_rstd, v, g, beta, initial_state, cu_seqlens = ctx.saved_tensors

        chunk_size = ctx.chunk_size
        A = chunk_scaled_dot_kkt_fwd(
            k=k, g=g, beta=beta,
            cu_seqlens=cu_seqlens,
            output_dtype=torch.float32,
            chunk_size=chunk_size,
        )
        A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype, chunk_size=chunk_size)

        dq, dk, dv, db, dg, dh0 = chunk_block_causal_gated_delta_rule_bwd(
            q=q, k=k, v=v, g=g, beta=beta, A=A,
            scale=ctx.scale, block_size=ctx.block_size,
            initial_state=initial_state, do=do, dht=dht,
            causal_mode=ctx.causal_mode,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        return (dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta),
                None, None, dh0, None, None, None, None, None, None, None)


def _noisy_batched_forward(
    q_n: torch.Tensor,
    k_n: torch.Tensor,
    v_n: torch.Tensor,
    g_n: torch.Tensor,
    beta_n: torch.Tensor,
    noisy_init_or_h_all: torch.Tensor,
    B: int, N: int, block_size: int, H: int, K: int, V: int,
    scale: float, use_qk_l2norm_in_kernel: bool,
    causal_mode_noisy: int = 0,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    """Batched noisy forward using hybrid (fused fwd + chunk bwd).

    ``noisy_init_or_h_all`` is either:
      - h_all_states [B, N, H, K, V] (preferred; constructs noisy_init
        with autograd tracking so gradient flows through cat/reshape), or
      - pre-constructed noisy_init [B*N, H, K, V] (legacy path).
    """
    BN = B * N

    q_n_batched = q_n.reshape(BN, block_size, H, K)
    k_n_batched = k_n.reshape(BN, block_size, H, K)
    v_n_batched = v_n.reshape(BN, block_size, H, V)
    g_n_batched = g_n.reshape(BN, block_size, H)
    beta_n_batched = beta_n.reshape(BN, block_size, H)

    if noisy_init_or_h_all.ndim == 5:
        h_all_states = noisy_init_or_h_all
        initial_state = _h_all_to_noisy_init(
            h_all_states, cu_seqlens=cu_seqlens, block_size=block_size,
        )
    else:
        h_all_states = None
        initial_state = noisy_init_or_h_all

    o_noisy_batched, _ = HybridBlockCausalGDAFunction.apply(
        q_n_batched, k_n_batched, v_n_batched,
        g_n_batched, beta_n_batched,
        scale, block_size, initial_state, False,
        causal_mode_noisy, use_qk_l2norm_in_kernel, None, h_all_states,
        chunk_size, cu_seqlens,
    )

    half_T = N * block_size
    return o_noisy_batched.reshape(B, half_T, H, V)


def _block_train_sequential(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    block_size: int,
    initial_state: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    causal_mode_clean: int = 0,
    causal_mode_noisy: int = 0,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, None]:
    """
    Block-train forward using sequential chunk_block_causal_gated_delta_rule calls
    for state extraction. This is the explicit sequential correctness route.

    Input layout: [B, T, H, K/V]  where T = 2 * half_T.
        half_T does NOT need to be a multiple of block_size (partial blocks
        are handled internally via zero-padding).
        First half:  clean tokens  [X_1, ..., X_N]
        Second half: noisy tokens  [hat_X_1, ..., hat_X_N]

    Causal dependency for noisy tokens:
        hat_X_1 attends to itself only
        hat_X_i attends to itself and X_1, ..., X_{i-1}

    The clean output is computed by processing the full clean sequence in one call.
    Intermediate states are extracted by N sequential calls (each processing one
    clean block with output_final_state=True), which gives the same mathematical
    result due to the exactness of the chunk decomposition.
    The noisy output is computed by batching all N noisy blocks into B*N batch.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    V = v.shape[-1]
    original_half_T = T // 2

    # ── Split clean / noisy ──────────────────────────────────────────────
    q_c, q_n = q[:, :original_half_T], q[:, original_half_T:]
    k_c, k_n = k[:, :original_half_T], k[:, original_half_T:]
    v_c, v_n = v[:, :original_half_T], v[:, original_half_T:]
    g_c, g_n = g[:, :original_half_T], g[:, original_half_T:]
    beta_c, beta_n = beta[:, :original_half_T], beta[:, original_half_T:]

    # ── Pad to full-block alignment if needed ─────────────────────────────
    (q_c, k_c, v_c, g_c, beta_c,
     q_n, k_n, v_n, g_n, beta_n,
     half_T, pad_len) = _pad_block_train_inputs(
        q_c, k_c, v_c, g_c, beta_c,
        q_n, k_n, v_n, g_n, beta_n,
        original_half_T, block_size)
    N = half_T // block_size

    o_clean, _ = ChunkBlockCausalGatedDeltaRuleFunction.apply(
        q_c, k_c, v_c, g_c, beta_c,
        scale, block_size, initial_state, False,
        causal_mode_clean, cu_seqlens, use_qk_l2norm_in_kernel,
        chunk_size,
    )

    doc_firsts = _doc_first_block_indices(cu_seqlens, block_size)
    doc_firsts_set = set(doc_firsts.tolist()) if doc_firsts is not None else set()

    h = initial_state
    h_states = []
    for i in range(N):
        if i > 0 and i in doc_firsts_set:
            h = None
        s = i * block_size
        e = s + block_size
        _, h = ChunkBlockCausalGatedDeltaRuleFunction.apply(
            q_c[:, s:e], k_c[:, s:e], v_c[:, s:e],
            g_c[:, s:e], beta_c[:, s:e],
            scale, block_size, h, True,
            causal_mode_clean, None, use_qk_l2norm_in_kernel,
            chunk_size,
        )
        h_states.append(h)

    # ── Build noisy initial states ───────────────────────────────────────
    h_zero = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)
    noisy_init_list = [h_zero] + h_states[:-1]  # N items
    noisy_init = torch.stack(noisy_init_list, dim=1)   # [B, N, H, K, V]

    if doc_firsts is not None:
        mask = torch.ones(N, device=q.device, dtype=noisy_init.dtype)
        mask[doc_firsts] = 0.0
        noisy_init = noisy_init * mask.view(1, N, 1, 1, 1)

    noisy_init = noisy_init.reshape(B * N, H, K, V)

    q_n_batched = q_n.reshape(B, N, block_size, H, K).reshape(B * N, block_size, H, K)
    k_n_batched = k_n.reshape(B, N, block_size, H, K).reshape(B * N, block_size, H, K)
    v_n_batched = v_n.reshape(B, N, block_size, H, V).reshape(B * N, block_size, H, V)
    g_n_batched = g_n.reshape(B, N, block_size, H).reshape(B * N, block_size, H)
    beta_n_batched = beta_n.reshape(B, N, block_size, H).reshape(B * N, block_size, H)

    o_noisy_batched, _ = ChunkBlockCausalGatedDeltaRuleFunction.apply(
        q_n_batched, k_n_batched, v_n_batched,
        g_n_batched, beta_n_batched,
        scale, block_size, noisy_init, False,
        causal_mode_noisy, None, use_qk_l2norm_in_kernel,
        chunk_size,
    )

    o_noisy = o_noisy_batched.reshape(B, half_T, H, V)

    # ── Trim padded positions and concatenate ─────────────────────────────
    o_clean = o_clean[:, :original_half_T]
    o_noisy = o_noisy[:, :original_half_T]
    o = torch.cat([o_clean, o_noisy], dim=1)
    return o, None


def _block_train_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    block_size: int,
    initial_state: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    causal_mode_clean: int = 0,
    causal_mode_noisy: int = 0,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, None]:
    """Block-train using fused recurrent state extraction + hybrid noisy.

    State extraction uses a single fused recurrent pass over the clean
    sequence with gradient checkpointing at every ``block_size``
    boundary.  Noisy forward uses ``HybridBlockCausalGDAFunction``
    (recurrent fwd + chunk bwd).

    State extraction and noisy forward are jointly wrapped in activation
    checkpointing so that ``h_all`` is freed after the forward pass and
    recomputed during backward, preventing accumulation across layers.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    original_half_T = T // 2

    q_c, q_n = q[:, :original_half_T], q[:, original_half_T:]
    k_c, k_n = k[:, :original_half_T], k[:, original_half_T:]
    v_c, v_n = v[:, :original_half_T], v[:, original_half_T:]
    g_c, g_n = g[:, :original_half_T], g[:, original_half_T:]
    beta_c, beta_n = beta[:, :original_half_T], beta[:, original_half_T:]

    # ── Pad to full-block alignment if needed ─────────────────────────────
    (q_c, k_c, v_c, g_c, beta_c,
     q_n, k_n, v_n, g_n, beta_n,
     half_T, pad_len) = _pad_block_train_inputs(
        q_c, k_c, v_c, g_c, beta_c,
        q_n, k_n, v_n, g_n, beta_n,
        original_half_T, block_size)
    N = half_T // block_size

    o_clean, _ = ChunkBlockCausalGatedDeltaRuleFunction.apply(
        q_c, k_c, v_c, g_c, beta_c,
        scale, block_size, initial_state, False,
        causal_mode_clean, cu_seqlens, use_qk_l2norm_in_kernel,
        chunk_size,
    )

    _cu = cu_seqlens

    def _noisy_with_h_recompute(k_c_, v_c_, g_c_, beta_c_,
                                q_n_, k_n_, v_n_, g_n_, beta_n_, h0_):
        h_all = fused_recurrent_gated_delta_state(
            k=k_c_, v=v_c_, g=g_c_, beta=beta_c_,
            h0=h0_, block_size=block_size,
            cu_seqlens=_cu,
        )
        return _noisy_batched_forward(
            q_n_, k_n_, v_n_, g_n_, beta_n_, h_all,
            B, N, block_size, H, K, V,
            scale, use_qk_l2norm_in_kernel,
            causal_mode_noisy=causal_mode_noisy,
            chunk_size=chunk_size,
            cu_seqlens=_cu,
        )

    if _prof._DISABLE_CHECKPOINT:
        o_noisy = _noisy_with_h_recompute(
            k_c, v_c, g_c, beta_c,
            q_n, k_n, v_n, g_n, beta_n, initial_state,
        )
    else:
        o_noisy = torch.utils.checkpoint.checkpoint(
            _noisy_with_h_recompute,
            k_c, v_c, g_c, beta_c,
            q_n, k_n, v_n, g_n, beta_n, initial_state,
            use_reentrant=False,
        )

    # ── Trim padded positions and concatenate ─────────────────────────────
    o_clean = o_clean[:, :original_half_T]
    o_noisy = o_noisy[:, :original_half_T]
    o = torch.cat([o_clean, o_noisy], dim=1)
    return o, None


def _block_train_chunk_refine(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    block_size: int,
    initial_state: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    causal_mode_clean: int = 0,
    causal_mode_noisy: int = 0,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, None]:
    """Block-train using chunk-then-refine for state extraction.

    Two-stage state extraction: WY pipeline at chunk_size=64 gives states
    at chunk boundaries, then local_refine fills in sub-chunk boundaries.
    Noisy forward uses hybrid (fused fwd + chunk bwd) via _noisy_batched_forward.
    Wrapped in activation checkpointing to save memory.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    original_half_T = T // 2

    q_c, q_n = q[:, :original_half_T], q[:, original_half_T:]
    k_c, k_n = k[:, :original_half_T], k[:, original_half_T:]
    v_c, v_n = v[:, :original_half_T], v[:, original_half_T:]
    g_c, g_n = g[:, :original_half_T], g[:, original_half_T:]
    beta_c, beta_n = beta[:, :original_half_T], beta[:, original_half_T:]

    # ── Pad to full-block alignment if needed ─────────────────────────────
    (q_c, k_c, v_c, g_c, beta_c,
     q_n, k_n, v_n, g_n, beta_n,
     half_T, pad_len) = _pad_block_train_inputs(
        q_c, k_c, v_c, g_c, beta_c,
        q_n, k_n, v_n, g_n, beta_n,
        original_half_T, block_size)
    N = half_T // block_size

    o_clean, _ = ChunkBlockCausalGatedDeltaRuleFunction.apply(
        q_c, k_c, v_c, g_c, beta_c,
        scale, block_size, initial_state, False,
        causal_mode_clean, cu_seqlens, use_qk_l2norm_in_kernel,
        chunk_size,
    )

    _cu = cu_seqlens

    def _noisy_with_h_recompute(k_c_, v_c_, g_c_, beta_c_,
                                q_n_, k_n_, v_n_, g_n_, beta_n_, h0_):
        h_all = chunk_refine_gda_state(
            k=k_c_, v=v_c_, g=g_c_, beta=beta_c_,
            h0=h0_, block_size=block_size,
            chunk_size=chunk_size,
            cu_seqlens=_cu,
        )
        return _noisy_batched_forward(
            q_n_, k_n_, v_n_, g_n_, beta_n_, h_all,
            B, N, block_size, H, K, V,
            scale, use_qk_l2norm_in_kernel,
            causal_mode_noisy=causal_mode_noisy,
            chunk_size=chunk_size,
            cu_seqlens=_cu,
        )

    if _prof._DISABLE_CHECKPOINT:
        o_noisy = _noisy_with_h_recompute(
            k_c, v_c, g_c, beta_c,
            q_n, k_n, v_n, g_n, beta_n, initial_state,
        )
    else:
        o_noisy = torch.utils.checkpoint.checkpoint(
            _noisy_with_h_recompute,
            k_c, v_c, g_c, beta_c,
            q_n, k_n, v_n, g_n, beta_n, initial_state,
            use_reentrant=False,
        )

    # ── Trim padded positions and concatenate ─────────────────────────────
    o_clean = o_clean[:, :original_half_T]
    o_noisy = o_noisy[:, :original_half_T]
    o = torch.cat([o_clean, o_noisy], dim=1)
    return o, None


def _block_train_chunk_refine_reuse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    block_size: int,
    initial_state: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    causal_mode_clean: int = 0,
    causal_mode_noisy: int = 0,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, None]:
    """Block-train that reuses h_chunks from the clean forward.

    The clean forward uses ChunkBlockCausalGDAWithHFunction which returns
    h_chunks at chunk boundaries.  These are passed (detached) to
    chunk_refine_gda_state so Stage 1 (WY pipeline) is skipped.
    Saves ~12ms per forward (~24ms with activation checkpointing).
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    original_half_T = T // 2

    q_c, q_n = q[:, :original_half_T], q[:, original_half_T:]
    k_c, k_n = k[:, :original_half_T], k[:, original_half_T:]
    v_c, v_n = v[:, :original_half_T], v[:, original_half_T:]
    g_c, g_n = g[:, :original_half_T], g[:, original_half_T:]
    beta_c, beta_n = beta[:, :original_half_T], beta[:, original_half_T:]

    # ── Pad to full-block alignment if needed ─────────────────────────────
    (q_c, k_c, v_c, g_c, beta_c,
     q_n, k_n, v_n, g_n, beta_n,
     half_T, pad_len) = _pad_block_train_inputs(
        q_c, k_c, v_c, g_c, beta_c,
        q_n, k_n, v_n, g_n, beta_n,
        original_half_T, block_size)
    N = half_T // block_size

    o_clean, _, h_chunks = ChunkBlockCausalGDAWithHFunction.apply(
        q_c, k_c, v_c, g_c, beta_c,
        scale, block_size, initial_state, False,
        causal_mode_clean, cu_seqlens, use_qk_l2norm_in_kernel,
        chunk_size,
    )

    h_chunks_detached = h_chunks.detach()
    _cu = cu_seqlens

    def _noisy_with_h_recompute(h_chunks_det, k_c_, v_c_, g_c_, beta_c_,
                                q_n_, k_n_, v_n_, g_n_, beta_n_, h0_):
        h_all = chunk_refine_gda_state(
            k=k_c_, v=v_c_, g=g_c_, beta=beta_c_,
            h0=h0_, block_size=block_size,
            h_chunks=h_chunks_det,
            chunk_size=chunk_size,
            cu_seqlens=_cu,
        )
        return _noisy_batched_forward(
            q_n_, k_n_, v_n_, g_n_, beta_n_, h_all,
            B, N, block_size, H, K, V,
            scale, use_qk_l2norm_in_kernel,
            causal_mode_noisy=causal_mode_noisy,
            chunk_size=chunk_size,
            cu_seqlens=_cu,
        )

    if _prof._DISABLE_CHECKPOINT:
        o_noisy = _noisy_with_h_recompute(
            h_chunks_detached, k_c, v_c, g_c, beta_c,
            q_n, k_n, v_n, g_n, beta_n, initial_state,
        )
    else:
        o_noisy = torch.utils.checkpoint.checkpoint(
            _noisy_with_h_recompute,
            h_chunks_detached, k_c, v_c, g_c, beta_c,
            q_n, k_n, v_n, g_n, beta_n, initial_state,
            use_reentrant=False,
        )

    # ── Trim padded positions and concatenate ─────────────────────────────
    o_clean = o_clean[:, :original_half_T]
    o_noisy = o_noisy[:, :original_half_T]
    o = torch.cat([o_clean, o_noisy], dim=1)
    return o, None


def _block_train_chunk_wy_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    block_size: int,
    initial_state: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    causal_mode_clean: int = 0,
    causal_mode_noisy: int = 0,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, None]:
    """Block-train using V2 clean path + fused Triton noisy kernels.

    The clean forward uses ChunkBlockCausalGDAWithHFunctionV2, which
    returns graph-connected h_chunks, v_new, and g_cumsum.  The noisy
    forward uses TwoStreamWYNoisyTritonFunctionV2, whose backward
    produces dh_chunks / dv_new / dG_c that flow back through V2's
    WY chain rule automatically — no separate state extraction needed.
    """
    B, T, H, K = q.shape
    original_half_T = T // 2

    q_c, q_n = q[:, :original_half_T], q[:, original_half_T:]
    k_c, k_n = k[:, :original_half_T], k[:, original_half_T:]
    v_c, v_n = v[:, :original_half_T], v[:, original_half_T:]
    g_c, g_n = g[:, :original_half_T], g[:, original_half_T:]
    beta_c, beta_n = beta[:, :original_half_T], beta[:, original_half_T:]

    (q_c, k_c, v_c, g_c, beta_c,
     q_n, k_n, v_n, g_n, beta_n,
     half_T, pad_len) = _pad_block_train_inputs(
        q_c, k_c, v_c, g_c, beta_c,
        q_n, k_n, v_n, g_n, beta_n,
        original_half_T, block_size)

    o_clean, _, h_chunks, v_new, g_c_cumsum = ChunkBlockCausalGDAWithHFunctionV2Improved.apply(
        q_c, k_c, v_c, g_c, beta_c,
        scale, block_size, initial_state, False,
        causal_mode_clean, cu_seqlens, use_qk_l2norm_in_kernel,
        chunk_size,
    )

    o_noisy = TwoStreamWYNoisyTritonFunctionV2.apply(
        q_n, k_n, v_n, g_n, beta_n,
        k_c, v_new, g_c_cumsum, h_chunks,
        scale, block_size, chunk_size, causal_mode_noisy,
        cu_seqlens,
    )

    o_clean = o_clean[:, :original_half_T]
    o_noisy = o_noisy[:, :original_half_T]
    o = torch.cat([o_clean, o_noisy], dim=1)
    return o, None


def _block_train_chunk_wy_triton_improved(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    block_size: int,
    initial_state: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    causal_mode_clean: int = 0,
    causal_mode_noisy: int = 0,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_wy_bwd_split_enabled: bool | None = None,
    chunk_wy_bwd_parallel_groups: int | None = None,
    chunk_wy_bwd_checkpoint_stride: int | None = None,
    chunk_wy_bwd_bv: int | None = None,
    chunk_wy_bwd_store_b: bool | None = None,
) -> tuple[torch.Tensor, None]:
    """Benchmark-isolated improved variant of `_block_train_chunk_wy_triton`.

    The clean path is currently identical to the baseline.  The noisy path uses
    `TwoStreamWYNoisyTritonImprovedFunction`, whose first improvement is to
    avoid zero-initializing store-only V-shaped gradient buffers.
    """
    T = q.shape[1]
    original_half_T = T // 2

    q_c, q_n = q[:, :original_half_T], q[:, original_half_T:]
    k_c, k_n = k[:, :original_half_T], k[:, original_half_T:]
    v_c, v_n = v[:, :original_half_T], v[:, original_half_T:]
    g_c, g_n = g[:, :original_half_T], g[:, original_half_T:]
    beta_c, beta_n = beta[:, :original_half_T], beta[:, original_half_T:]

    (q_c, k_c, v_c, g_c, beta_c,
     q_n, k_n, v_n, g_n, beta_n,
     _, _) = _pad_block_train_inputs(
        q_c, k_c, v_c, g_c, beta_c,
        q_n, k_n, v_n, g_n, beta_n,
        original_half_T, block_size)

    o_clean, _, h_chunks, v_new, g_c_cumsum = ChunkBlockCausalGDAWithHFunctionV2.apply(
        q_c, k_c, v_c, g_c, beta_c,
        scale, block_size, initial_state, False,
        causal_mode_clean, cu_seqlens, use_qk_l2norm_in_kernel,
        chunk_size,
    )

    o_noisy = TwoStreamWYNoisyTritonImprovedFunction.apply(
        q_n, k_n, v_n, g_n, beta_n,
        k_c, v_new, g_c_cumsum, h_chunks,
        scale, block_size, chunk_size, causal_mode_noisy,
        cu_seqlens,
        chunk_wy_bwd_split_enabled,
        chunk_wy_bwd_parallel_groups,
        chunk_wy_bwd_checkpoint_stride,
        chunk_wy_bwd_bv,
        chunk_wy_bwd_store_b,
    )

    o_clean = o_clean[:, :original_half_T]
    o_noisy = o_noisy[:, :original_half_T]
    o = torch.cat([o_clean, o_noisy], dim=1)
    return o, None


def _block_train_chunk_wy_triton_fla_style(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    block_size: int,
    initial_state: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    causal_mode_clean: int = 0,
    causal_mode_noisy: int = 0,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_wy_fla_style_checkpoint_stride: int | None = None,
    chunk_wy_fla_style_bwd_bv: int | None = None,
) -> tuple[torch.Tensor, None]:
    """Fused two-stream Route II used by automatic small-block dispatch.

    The clean half uses the chunkwise WY path and exposes chunk states; the
    noisy half uses a merged Triton autograd path whose backward replays the
    noisy recurrence once and injects its gradients into the clean recurrence.
    """
    T = q.shape[1]
    original_half_T = T // 2

    q_c, q_n = q[:, :original_half_T], q[:, original_half_T:]
    k_c, k_n = k[:, :original_half_T], k[:, original_half_T:]
    v_c, v_n = v[:, :original_half_T], v[:, original_half_T:]
    g_c, g_n = g[:, :original_half_T], g[:, original_half_T:]
    beta_c, beta_n = beta[:, :original_half_T], beta[:, original_half_T:]

    (q_c, k_c, v_c, g_c, beta_c,
     q_n, k_n, v_n, g_n, beta_n,
     _, _) = _pad_block_train_inputs(
        q_c, k_c, v_c, g_c, beta_c,
        q_n, k_n, v_n, g_n, beta_n,
        original_half_T, block_size)

    o_clean, _, h_chunks, v_new, g_c_cumsum = ChunkBlockCausalGDAWithHFunctionV2Improved.apply(
        q_c, k_c, v_c, g_c, beta_c,
        scale, block_size, initial_state, False,
        causal_mode_clean, cu_seqlens, use_qk_l2norm_in_kernel,
        chunk_size,
    )

    if use_qk_l2norm_in_kernel:
        raise NotImplementedError("chunk_wy_triton_fla_style fused path does not yet support q/k L2 norm in-kernel")

    o_noisy = TwoStreamFLAStyleFullFunction.apply(
        q_n, k_n, v_n, g_n, beta_n,
        k_c, v_c, g_c, beta_c,
        v_new, g_c_cumsum, h_chunks,
        scale, block_size, causal_mode_noisy, chunk_size, cu_seqlens,
        chunk_wy_fla_style_checkpoint_stride,
        chunk_wy_fla_style_bwd_bv,
    )

    o_clean = o_clean[:, :original_half_T]
    o_noisy = o_noisy[:, :original_half_T]
    o = torch.cat([o_clean, o_noisy], dim=1)
    return o, None


def _clean_v2_bwd_unpacked_trailing_partial(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta: torch.Tensor,
    q_eff: torch.Tensor,
    q_rstd: torch.Tensor | None,
    k_eff: torch.Tensor,
    k_rstd: torch.Tensor | None,
    scale: float,
    block_size: int,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    causal_mode: int,
    use_qk_l2norm_in_kernel: bool,
    chunk_size: int,
    do: torch.Tensor | None,
    dht: torch.Tensor | None,
    dh_chunks: torch.Tensor | None,
    dv_new_inject: torch.Tensor | None = None,
    dG_inject: torch.Tensor | None = None,
    explicit_state_path: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Fast exact backward for unpacked sequences with a trailing partial chunk.

    The long full-chunk prefix reuses the existing fast clean backward and fast
    state-gradient path. The final short chunk is padded to one full chunk so
    it can also reuse the existing fast clean kernels without falling back to
    whole-sequence reference autograd.
    """
    half_T = q_raw.shape[1]
    prefix_len = (half_T // chunk_size) * chunk_size
    tail_len = half_T - prefix_len

    if tail_len == 0:
        raise RuntimeError("_clean_v2_bwd_unpacked_trailing_partial expects a non-empty trailing partial chunk")

    dq = torch.zeros_like(q_raw)
    dk = torch.zeros_like(k_raw)
    dv = torch.zeros_like(v)
    dg = torch.zeros_like(g_raw)
    dbeta = torch.zeros_like(beta)
    dh0 = torch.zeros_like(initial_state) if initial_state is not None else None

    # Tail fast exact path: pad the short irregular tail to one full chunk so
    # it can reuse the existing fast clean kernels.
    if prefix_len > 0:
        with torch.no_grad():
            prefix_end_blocks = fused_recurrent_gated_delta_state(
                k=k_eff[:, :prefix_len].contiguous(),
                v=v[:, :prefix_len].contiguous(),
                g=g_raw[:, :prefix_len].contiguous(),
                beta=beta[:, :prefix_len].contiguous(),
                h0=initial_state,
                block_size=chunk_size,
                cu_seqlens=None,
            )
            prefix_final_state = prefix_end_blocks[:, -1].contiguous()
    else:
        prefix_final_state = initial_state

    tail_initial_state_grad = None
    tail_pad_len = chunk_size - tail_len
    p4 = (0, 0, 0, 0, 0, tail_pad_len)
    p3 = (0, 0, 0, tail_pad_len)

    q_tail_raw = F.pad(q_raw[:, prefix_len:], p4)
    k_tail_raw = F.pad(k_raw[:, prefix_len:], p4)
    v_tail = F.pad(v[:, prefix_len:], p4)
    g_tail_raw = F.pad(g_raw[:, prefix_len:], p3)
    beta_tail = F.pad(beta[:, prefix_len:], p3)
    do_tail = None if do is None else F.pad(do[:, prefix_len:], p4)
    dv_new_inject_tail = None if dv_new_inject is None else F.pad(dv_new_inject[:, prefix_len:], p4)
    dG_inject_tail = None if dG_inject is None else F.pad(dG_inject[:, prefix_len:], p3)

    if use_qk_l2norm_in_kernel:
        q_tail_eff, q_tail_rstd = l2norm_fwd(q_tail_raw)
        k_tail_eff, k_tail_rstd = l2norm_fwd(k_tail_raw)
    else:
        q_tail_eff, q_tail_rstd = q_tail_raw, None
        k_tail_eff, k_tail_rstd = k_tail_raw, None

    g_tail = chunk_local_cumsum(g_tail_raw, chunk_size=chunk_size, cu_seqlens=None)
    A_tail = chunk_scaled_dot_kkt_fwd(
        k=k_tail_eff,
        g=g_tail,
        beta=beta_tail,
        cu_seqlens=None,
        output_dtype=torch.float32,
        chunk_size=chunk_size,
    )
    A_tail = solve_tril(A=A_tail, cu_seqlens=None, output_dtype=k_tail_eff.dtype, chunk_size=chunk_size)

    dq_tail_eff, dk_tail_eff, dv_tail, dbeta_tail, dg_tail, tail_initial_state_grad = (
        chunk_block_causal_gated_delta_rule_bwd(
            q=q_tail_eff,
            k=k_tail_eff,
            v=v_tail,
            g=g_tail,
            beta=beta_tail,
            A=A_tail,
            scale=scale,
            block_size=block_size,
            initial_state=prefix_final_state,
            do=do_tail,
            dht=dht,
            causal_mode=causal_mode,
            cu_seqlens=None,
            chunk_size=chunk_size,
            dv_new_inject=dv_new_inject_tail,
            dG_inject=dG_inject_tail,
        )
    )

    if use_qk_l2norm_in_kernel:
        dq_tail = l2norm_bwd(q_tail_raw, q_tail_rstd, dq_tail_eff)
        dk_tail = l2norm_bwd(k_tail_raw, k_tail_rstd, dk_tail_eff)
    else:
        dq_tail = dq_tail_eff
        dk_tail = dk_tail_eff

    dq[:, prefix_len:] = dq_tail[:, :tail_len]
    dk[:, prefix_len:] = dk_tail[:, :tail_len]
    dv[:, prefix_len:] = dv_tail[:, :tail_len]
    dg[:, prefix_len:] = dg_tail[:, :tail_len]
    dbeta[:, prefix_len:] = dbeta_tail[:, :tail_len]

    if dh_chunks is not None and (prefix_len > 0 or initial_state is not None):
        num_prefix_chunks = prefix_len // chunk_size
        tail_chunk_grad = dh_chunks[:, num_prefix_chunks]
        tail_initial_state_grad = (
            tail_chunk_grad
            if tail_initial_state_grad is None
            else tail_initial_state_grad + tail_chunk_grad
        )

    # Full-chunk prefix fast path.
    if prefix_len > 0:
        q_prefix_eff = q_eff[:, :prefix_len].contiguous()
        k_prefix_eff = k_eff[:, :prefix_len].contiguous()
        v_prefix = v[:, :prefix_len].contiguous()
        g_prefix_raw = g_raw[:, :prefix_len].contiguous()
        beta_prefix = beta[:, :prefix_len].contiguous()
        do_prefix = None if do is None else do[:, :prefix_len].contiguous()
        dh_inject_prefix = None
        dh_chunks_h0 = None

        if dh_chunks is not None:
            num_prefix_chunks = prefix_len // chunk_size
            dh_chunks_prefix = dh_chunks[:, :num_prefix_chunks]
            dh_chunks_h0 = dh_chunks_prefix[:, 0] if initial_state is not None else None
            if explicit_state_path and num_prefix_chunks > 1:
                dh_inject_prefix = _unpacked_chunk_start_grads_to_chunk_end_grads(
                    dh_chunks_prefix
                )

        g_prefix = chunk_local_cumsum(g_prefix_raw, chunk_size=chunk_size, cu_seqlens=None)
        A_prefix = chunk_scaled_dot_kkt_fwd(
            k=k_prefix_eff,
            g=g_prefix,
            beta=beta_prefix,
            cu_seqlens=None,
            output_dtype=torch.float32,
            chunk_size=chunk_size,
        )
        A_prefix = solve_tril(A=A_prefix, cu_seqlens=None, output_dtype=k_prefix_eff.dtype, chunk_size=chunk_size)

        dv_new_inject_prefix = None if dv_new_inject is None else dv_new_inject[:, :prefix_len].contiguous()
        dG_inject_prefix = None if dG_inject is None else dG_inject[:, :prefix_len].contiguous()

        dq_prefix_eff, dk_prefix_eff, dv_prefix, dbeta_prefix, dg_prefix, dh0_prefix = (
            chunk_block_causal_gated_delta_rule_bwd(
                q=q_prefix_eff,
                k=k_prefix_eff,
                v=v_prefix,
                g=g_prefix,
                beta=beta_prefix,
                A=A_prefix,
                scale=scale,
                block_size=block_size,
                initial_state=initial_state,
                do=do_prefix,
                dht=tail_initial_state_grad,
                causal_mode=causal_mode,
                cu_seqlens=None,
                chunk_size=chunk_size,
                dv_new_inject=dv_new_inject_prefix,
                dG_inject=dG_inject_prefix,
                dh_inject=dh_inject_prefix,
            )
        )

        if dh_chunks is not None:
            if explicit_state_path:
                dk_state = dv_state = dg_state = dbeta_state = None
            else:
                num_prefix_chunks = prefix_len // chunk_size
                dh_chunks_prefix = dh_chunks[:, :num_prefix_chunks]
                dh_end_prefix = torch.zeros_like(dh_chunks_prefix)
                if num_prefix_chunks > 1:
                    dh_end_prefix[:, :-1] = dh_chunks_prefix[:, 1:]

                if num_prefix_chunks > 1:
                    with torch.enable_grad():
                        k_state = k_prefix_eff.detach().requires_grad_(True)
                        v_state = v_prefix.detach().requires_grad_(True)
                        g_state = g_prefix_raw.detach().requires_grad_(True)
                        beta_state = beta_prefix.detach().requires_grad_(True)

                        if initial_state is not None:
                            h0_state = initial_state.detach().requires_grad_(True)
                            state_inputs = (k_state, v_state, g_state, beta_state, h0_state)
                        else:
                            h0_state = None
                            state_inputs = (k_state, v_state, g_state, beta_state)

                        h_end = fused_recurrent_gated_delta_state(
                            k=k_state,
                            v=v_state,
                            g=g_state,
                            beta=beta_state,
                            h0=h0_state,
                            block_size=chunk_size,
                            cu_seqlens=None,
                        )

                        state_grads = torch.autograd.grad(
                            outputs=h_end,
                            inputs=state_inputs,
                            grad_outputs=dh_end_prefix,
                            allow_unused=True,
                        )

                    if initial_state is not None:
                        dk_state, dv_state, dg_state, dbeta_state, dh0_state = state_grads
                        dh0_prefix = (torch.zeros_like(initial_state) if dh0_prefix is None else dh0_prefix) + dh0_state
                    else:
                        dk_state, dv_state, dg_state, dbeta_state = state_grads
                else:
                    dk_state = dv_state = dg_state = dbeta_state = None

            if initial_state is not None and dh_chunks_h0 is not None:
                dh0_prefix = (torch.zeros_like(initial_state) if dh0_prefix is None else dh0_prefix) + dh_chunks_h0

            if dk_state is not None:
                dk_prefix_eff = dk_prefix_eff + dk_state
            if dv_state is not None:
                dv_prefix = dv_prefix + dv_state
            if dg_state is not None:
                dg_prefix = dg_prefix + dg_state
            if dbeta_state is not None:
                dbeta_prefix = dbeta_prefix + dbeta_state

        if use_qk_l2norm_in_kernel:
            dq_prefix = l2norm_bwd(q_prefix_eff, q_rstd[:, :prefix_len], dq_prefix_eff)
            dk_prefix = l2norm_bwd(k_prefix_eff, k_rstd[:, :prefix_len], dk_prefix_eff)
        else:
            dq_prefix = dq_prefix_eff
            dk_prefix = dk_prefix_eff

        dq[:, :prefix_len] = dq_prefix
        dk[:, :prefix_len] = dk_prefix
        dv[:, :prefix_len] = dv_prefix
        dg[:, :prefix_len] = dg_prefix
        dbeta[:, :prefix_len] = dbeta_prefix

        if initial_state is not None:
            dh0 = (torch.zeros_like(initial_state) if dh0 is None else dh0) + dh0_prefix
    elif initial_state is not None and tail_initial_state_grad is not None:
        dh0 = (torch.zeros_like(initial_state) if dh0 is None else dh0) + tail_initial_state_grad

    return dq, dk, dv, dg, dbeta, dh0


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
    cu_seqlens: torch.LongTensor | None = None,
    block_train: bool = False,
    block_train_method: str = 'auto',
    causal_mode: int = 0,
    causal_mode_clean: int | None = None,
    causal_mode_noisy: int | None = None,
    **kwargs,
):
    r"""
    Block-causal gated delta rule attention.

    This variant uses block-causal masking for the output attention:
    - Position i attends to ALL positions j where block(j) <= block(i)
    - Within a block, all positions can see the entire block
    - The delta rule hidden state update remains token-causal

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`.
        scale (Optional[float]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        block_size (int):
            Size of causal blocks for attention masking. Default: `4`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2norm to the q/k tensor internally. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]`.
        block_train (bool):
            If True, enables block-training mode. Input shape is
            `[B, T, H, K/V]` where `T = 2 * half_T`.  `half_T` does NOT
            need to be a multiple of `block_size` (partial blocks are
            zero-padded internally).  The first half contains clean tokens
            [X_1,...,X_N] and the second half noisy tokens
            [hat_X_1,...,hat_X_N]. Default: `False`.
        block_train_method (str):
            Block-train execution route:
            ``'sequential'`` uses N sequential calls;
            ``'fused'`` uses a custom fused recurrent kernel;
            ``'chunk_refine'`` uses WY pipeline + local refinement;
            ``'chunk_refine_reuse'`` reuses h_chunks from clean forward;
            ``'chunk_wy_triton'`` uses V2 clean + fused Triton noisy kernels;
            ``'chunk_wy_triton_improved'`` uses an isolated benchmark path
            for chunk_wy_triton optimizations;
            ``'chunk_wy_triton_fla_style'`` uses the merged FLA-style
            full-noisy Triton implementation;
            ``'auto'`` dispatches to chunk_wy_triton_fla_style for
            ``block_size < 16`` and chunk_refine for ``block_size >= 16``.
            Default: ``'auto'``.
        chunk_wy_bwd_split_enabled, chunk_wy_bwd_parallel_groups,
        chunk_wy_bwd_checkpoint_stride, chunk_wy_bwd_bv,
        chunk_wy_bwd_store_b:
            Optional knobs for ``block_train_method='chunk_wy_triton_improved'``.
            When omitted, the improved path falls back to its environment
            variables/defaults.
        chunk_wy_fla_style_checkpoint_stride:
            Optional clean-state checkpoint stride for the merged FLA-style
            full backward.  Smaller values reduce replay and can improve
            wallclock at the cost of more checkpoint scratch.  When omitted,
            a block-size-aware replay default is used: ``16`` for block size 1,
            ``8`` for block size 2, ``3`` for block size 3, ``2`` for block
            size 4, and ``8`` otherwise (then reduced to a divisor of the
            aligned chunk size when necessary).
        chunk_wy_fla_style_bwd_bv:
            Optional value tile size for the merged FLA-style full backward.
            Supported values are ``8, 16, 32, 64``.  Default: ``32``.
    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.
    """
    if 'head_first' in kwargs:
        warnings.warn(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead.",
        )

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    chunk_size = compute_aligned_chunk_size(block_size)
    assert chunk_size >= block_size, (
        f"block_size={block_size} is too large (max supported: 64). "
        f"compute_aligned_chunk_size returned {chunk_size}."
    )

    cm_clean = causal_mode_clean if causal_mode_clean is not None else causal_mode
    cm_noisy = causal_mode_noisy if causal_mode_noisy is not None else causal_mode

    assert causal_mode in (0, 1), f"causal_mode must be 0 or 1, got {causal_mode}"
    assert cm_clean in (0, 1), f"causal_mode_clean must be 0 or 1, got {cm_clean}"
    assert cm_noisy in (0, 1), f"causal_mode_noisy must be 0 or 1, got {cm_noisy}"

    if block_train:
        method = block_train_method
        if method == 'auto':
            method = 'chunk_wy_triton_fla_style' if block_size < 16 else 'chunk_refine'
        _validate_block_train_packed_boundaries(cu_seqlens, block_size)

        _dispatch = {
            'sequential': _block_train_sequential,
            'fused': _block_train_fused,
            'chunk_refine': _block_train_chunk_refine,
            'chunk_refine_reuse': _block_train_chunk_refine_reuse,
            'chunk_wy_triton': _block_train_chunk_wy_triton,
            'chunk_wy_triton_improved': _block_train_chunk_wy_triton_improved,
            'chunk_wy_triton_fla_style': _block_train_chunk_wy_triton_fla_style,
        }
        if method not in _dispatch:
            raise ValueError(
                f"Unknown block_train_method: {method!r}. "
                f"Use 'auto', {', '.join(repr(m) for m in _dispatch)}."
            )
        method_kwargs = {}
        if method == 'chunk_wy_triton_improved':
            method_kwargs = {
                'chunk_wy_bwd_split_enabled': kwargs.get('chunk_wy_bwd_split_enabled'),
                'chunk_wy_bwd_parallel_groups': kwargs.get('chunk_wy_bwd_parallel_groups'),
                'chunk_wy_bwd_checkpoint_stride': kwargs.get('chunk_wy_bwd_checkpoint_stride'),
                'chunk_wy_bwd_bv': kwargs.get('chunk_wy_bwd_bv'),
                'chunk_wy_bwd_store_b': kwargs.get('chunk_wy_bwd_store_b'),
            }
        elif method == 'chunk_wy_triton_fla_style':
            method_kwargs = {
                'chunk_wy_fla_style_checkpoint_stride': kwargs.get('chunk_wy_fla_style_checkpoint_stride'),
                'chunk_wy_fla_style_bwd_bv': kwargs.get('chunk_wy_fla_style_bwd_bv'),
            }
        return _dispatch[method](
            q=q, k=k, v=v, g=g, beta=beta,
            scale=scale,
            block_size=block_size,
            initial_state=initial_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            causal_mode_clean=cm_clean,
            causal_mode_noisy=cm_noisy,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            **method_kwargs,
        )

    o, final_state = ChunkBlockCausalGatedDeltaRuleFunction.apply(
        q, k, v, g, beta,
        scale, block_size, initial_state, output_final_state,
        causal_mode, cu_seqlens, use_qk_l2norm_in_kernel,
        chunk_size,
    )
    return o, final_state
