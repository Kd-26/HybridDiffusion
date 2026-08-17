# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Modified for block-causal attention

import os

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp
from fla.utils import input_guard


# ============================================================================
# Block-Causal Fused Recurrent Kernel
# ============================================================================

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_GV': lambda args: args['gv'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_block_causal_fwd_kernel(
    q,
    k,
    v,
    g,
    gk,
    gv,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    intermediate_states_buffer,
    intermediate_state_indices,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_CLEAN: tl.constexpr,
    CAUSAL_MODE: tl.constexpr,
    CACHE_STEPS: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    CACHE_INTERMEDIATE_STATES: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Fused recurrent kernel with configurable causal mode.
    
    CAUSAL_MODE=0: block-causal (two-phase: recurrence then output with h[block_end])
    CAUSAL_MODE=1: token-causal (single pass: output o_i = q_i @ h_i immediately)
    Grid: (N*HV, NV) — N*HV on dim-0 (x) to avoid 65535 limit.
    """
    i_nh, i_v = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    # Handle variable-length sequences
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    # Offset arrays for K and V dimensions
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    # Compute strides for pointer arithmetic
    stride_qk = H * K
    stride_v = HV * V
    stride_o = HV * V
    stride_g = HV
    stride_gk = HV * K
    stride_gv = HV * V
    stride_beta_hw = HV  # headwise beta
    stride_beta_full = HV * V  # full beta

    # Base pointers (pointing to start of this sequence, head, and dimension slice)
    p_q_base = q + (bos * H + i_h) * K + o_k
    p_k_base = k + (bos * H + i_h) * K + o_k
    p_v_base = v + (bos * HV + i_hv) * V + o_v
    p_o_base = o + (bos * HV + i_hv) * V + o_v

    # Gate and beta base pointers (conditional)
    p_g_base = g + bos * HV + i_hv if USE_G else None
    p_gk_base = gk + (bos * HV + i_hv) * K + o_k if USE_GK else None
    p_gv_base = gv + (bos * HV + i_hv) * V + o_v if USE_GV else None
    
    if IS_BETA_HEADWISE:
        p_beta_base = beta + bos * HV + i_hv
        stride_beta = stride_beta_hw
    else:
        p_beta_base = beta + (bos * HV + i_hv) * V + o_v
        stride_beta = stride_beta_full

    # Masks for valid K and V indices
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    # Initialize hidden state
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    cache_idx = -1
    if CACHE_INTERMEDIATE_STATES:
        cache_idx = tl.load(intermediate_state_indices + i_n).to(tl.int64)

    if CAUSAL_MODE == 1:
        p_q = p_q_base
        p_k = p_k_base
        p_v = p_v_base
        p_o = p_o_base
        p_beta = p_beta_base
        if USE_G:
            p_g = p_g_base
        if USE_GK:
            p_gk = p_gk_base
        if USE_GV:
            p_gv = p_gv_base

        for t in range(T):
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
            if USE_QK_L2NORM_IN_KERNEL:
                b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
            if IS_BETA_HEADWISE:
                b_beta = tl.load(p_beta).to(tl.float32)
            else:
                b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
            if USE_G:
                b_g = tl.load(p_g).to(tl.float32)
                b_h *= exp(b_g)
            if USE_GK:
                b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
                b_h *= exp(b_gk[None, :])
            if USE_GV:
                b_gv = tl.load(p_gv, mask=mask_v, other=0).to(tl.float32)
                b_h *= exp(b_gv[:, None])
            b_delta = b_beta * (b_v - tl.sum(b_h * b_k[None, :], 1))
            b_h += b_delta[:, None] * b_k[None, :]

            b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
            if USE_QK_L2NORM_IN_KERNEL:
                b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_q = b_q * scale
            b_o = tl.sum(b_h * b_q[None, :], 1)
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

            p_q += stride_qk
            p_k += stride_qk
            p_v += stride_v
            p_o += stride_o
            p_beta += stride_beta
            if USE_G:
                p_g += stride_g
            if USE_GK:
                p_gk += stride_gk
            if USE_GV:
                p_gv += stride_gv

            if CACHE_INTERMEDIATE_STATES:
                if t < CACHE_STEPS:
                    if cache_idx >= 0:
                        p_inter = (
                            intermediate_states_buffer
                            + cache_idx * CACHE_STEPS * HV * V * K
                            + t * HV * V * K
                            + i_hv * V * K
                            + o_v[:, None] * K
                            + o_k[None, :]
                        )
                        tl.store(
                            p_inter, b_h.to(p_inter.dtype.element_ty), mask=mask_h
                        )
    elif CAUSAL_MODE == 2:
        # Mixed readout: positions 0..NUM_CLEAN-1 use token-causal (S_t),
        # positions NUM_CLEAN..BLOCK_SIZE-1 use block-causal (S_block_end).
        # Single block assumed (self-spec processes one block per forward).
        # Phase 1: clean positions — update S AND output with S_t
        p_q = p_q_base
        p_k = p_k_base
        p_v = p_v_base
        p_o = p_o_base
        p_beta = p_beta_base
        if USE_G:
            p_g = p_g_base
        if USE_GK:
            p_gk = p_gk_base
        if USE_GV:
            p_gv = p_gv_base

        for t in range(T):
            if t < T:
                b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
                b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
                if USE_QK_L2NORM_IN_KERNEL:
                    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
                if IS_BETA_HEADWISE:
                    b_beta = tl.load(p_beta).to(tl.float32)
                else:
                    b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
                if USE_G:
                    b_g = tl.load(p_g).to(tl.float32)
                    b_h *= exp(b_g)
                if USE_GK:
                    b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
                    b_h *= exp(b_gk[None, :])
                if USE_GV:
                    b_gv = tl.load(p_gv, mask=mask_v, other=0).to(tl.float32)
                    b_h *= exp(b_gv[:, None])
                b_delta = b_beta * (b_v - tl.sum(b_h * b_k[None, :], 1))
                b_h += b_delta[:, None] * b_k[None, :]
                p_k += stride_qk
                p_v += stride_v
                p_beta += stride_beta
                if USE_G:
                    p_g += stride_g
                if USE_GK:
                    p_gk += stride_gk
                if USE_GV:
                    p_gv += stride_gv

                if CACHE_INTERMEDIATE_STATES:
                    if t < CACHE_STEPS:
                        if cache_idx >= 0:
                            p_inter = (
                                intermediate_states_buffer
                                + cache_idx * CACHE_STEPS * HV * V * K
                                + t * HV * V * K
                                + i_hv * V * K
                                + o_v[:, None] * K
                                + o_k[None, :]
                            )
                            tl.store(
                                p_inter, b_h.to(p_inter.dtype.element_ty), mask=mask_h
                            )

            # Clean positions: output immediately with current S_t
            if t < NUM_CLEAN:
                b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
                if USE_QK_L2NORM_IN_KERNEL:
                    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
                b_q = b_q * scale
                b_o = tl.sum(b_h * b_q[None, :], 1)
                tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
                p_q += stride_qk
                p_o += stride_o

        # Phase 2: draft positions — output with S_block_end (b_h is now S after all T tokens)
        p_q = p_q_base + NUM_CLEAN * stride_qk
        p_o = p_o_base + NUM_CLEAN * stride_o
        for t in range(NUM_CLEAN, T):
            if t < T:
                b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
                if USE_QK_L2NORM_IN_KERNEL:
                    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
                b_q = b_q * scale
                b_o = tl.sum(b_h * b_q[None, :], 1)
                tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
                p_q += stride_qk
                p_o += stride_o
    else:
        num_blocks = (T + BLOCK_SIZE - 1) // BLOCK_SIZE
        for block_idx in range(num_blocks):
            block_start = block_idx * BLOCK_SIZE
            p_k = p_k_base + block_start * stride_qk
            p_v = p_v_base + block_start * stride_v
            p_beta = p_beta_base + block_start * stride_beta
            if USE_G:
                p_g = p_g_base + block_start * stride_g
            if USE_GK:
                p_gk = p_gk_base + block_start * stride_gk
            if USE_GV:
                p_gv = p_gv_base + block_start * stride_gv

            for local_t in range(BLOCK_SIZE):
                t = block_start + local_t
                if t < T:
                    b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
                    b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
                    if USE_QK_L2NORM_IN_KERNEL:
                        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
                    if IS_BETA_HEADWISE:
                        b_beta = tl.load(p_beta).to(tl.float32)
                    else:
                        b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
                    if USE_G:
                        b_g = tl.load(p_g).to(tl.float32)
                        b_h *= exp(b_g)
                    if USE_GK:
                        b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
                        b_h *= exp(b_gk[None, :])
                    if USE_GV:
                        b_gv = tl.load(p_gv, mask=mask_v, other=0).to(tl.float32)
                        b_h *= exp(b_gv[:, None])
                    b_delta = b_beta * (b_v - tl.sum(b_h * b_k[None, :], 1))
                    b_h += b_delta[:, None] * b_k[None, :]
                    p_k += stride_qk
                    p_v += stride_v
                    p_beta += stride_beta
                    if USE_G:
                        p_g += stride_g
                    if USE_GK:
                        p_gk += stride_gk
                    if USE_GV:
                        p_gv += stride_gv

                    if CACHE_INTERMEDIATE_STATES:
                        if t < CACHE_STEPS:
                            if cache_idx >= 0:
                                p_inter = (
                                    intermediate_states_buffer
                                    + cache_idx * CACHE_STEPS * HV * V * K
                                    + t * HV * V * K
                                    + i_hv * V * K
                                    + o_v[:, None] * K
                                    + o_k[None, :]
                                )
                                tl.store(
                                    p_inter,
                                    b_h.to(p_inter.dtype.element_ty),
                                    mask=mask_h,
                                )

            p_q = p_q_base + block_start * stride_qk
            p_o = p_o_base + block_start * stride_o
            for local_t in range(BLOCK_SIZE):
                t = block_start + local_t
                if t < T:
                    b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
                    if USE_QK_L2NORM_IN_KERNEL:
                        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
                    b_q = b_q * scale
                    b_o = tl.sum(b_h * b_q[None, :], 1)
                    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
                    p_q += stride_qk
                    p_o += stride_o

    # Store final hidden state if requested
    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * V * K + o_v[:, None] * K + o_k[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_block_causal_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float = None,
    block_size: int = 64,
    causal_mode: int = 0,
    num_clean: int = 0,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    intermediate_states_buffer: torch.Tensor = None,
    intermediate_state_indices: torch.Tensor = None,
    cache_steps: int = 0,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Block-causal fused recurrent forward pass.
    
    Args:
        q: Query tensor of shape [B, T, H, K]
        k: Key tensor of shape [B, T, H, K]
        v: Value tensor of shape [B, T, HV, V]
        g: Gate tensor of shape [B, T, HV] (optional)
        gk: Per-K gate tensor of shape [B, T, HV, K] (optional)
        gv: Per-V gate tensor of shape [B, T, HV, V] (optional)
        beta: Learning rate tensor of shape [B, T, HV] or [B, T, HV, V]
        scale: Scale factor (default: 1/sqrt(K))
        block_size: Size of causal blocks
        initial_state: Initial hidden state of shape [N, HV, V, K]
        output_final_state: Whether to output final hidden state
        use_qk_l2norm_in_kernel: Whether to L2-normalize q and k
        cu_seqlens: Cumulative sequence lengths for variable-length
        
    Returns:
        o: Output tensor of shape [B, T, HV, V]
        final_state: Final hidden state if output_final_state=True, else None
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V)) if gv is None else triton.next_power_of_2(V)
    NV = triton.cdiv(V, BV)

    o = torch.empty_like(v)
    final_state = q.new_empty(N, HV, V, K, dtype=torch.float32) if output_final_state else None

    grid = (N * HV, NV)
    fused_recurrent_block_causal_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        gk=gk,
        gv=gv,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        intermediate_states_buffer=(
            intermediate_states_buffer
            if intermediate_states_buffer is not None
            else q
        ),
        intermediate_state_indices=(
            intermediate_state_indices
            if intermediate_state_indices is not None
            else q
        ),
        scale=scale,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        BLOCK_SIZE=block_size,
        NUM_CLEAN=num_clean,
        CAUSAL_MODE=causal_mode,
        CACHE_STEPS=cache_steps,
        IS_BETA_HEADWISE=beta.ndim != v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        CACHE_INTERMEDIATE_STATES=intermediate_states_buffer is not None,
        num_warps=1,
        num_stages=3,
    )
    return o, final_state


class FusedRecurrentBlockCausalFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        _ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        gk: torch.Tensor | None = None,
        gv: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        scale: float = None,
        block_size: int = 64,
        causal_mode: int = 0,
        num_clean: int = 0,
        initial_state: torch.Tensor = None,
        output_final_state: bool = False,
        intermediate_states_buffer: torch.Tensor = None,
        intermediate_state_indices: torch.Tensor = None,
        cache_steps: int = 0,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ):
        o, final_state = fused_recurrent_block_causal_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            gk=gk,
            gv=gv,
            beta=beta,
            scale=scale,
            block_size=block_size,
            causal_mode=causal_mode,
            num_clean=num_clean,
            initial_state=initial_state,
            output_final_state=output_final_state,
            intermediate_states_buffer=intermediate_states_buffer,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=cache_steps,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
        )
        return o, final_state

    @staticmethod
    @input_guard
    def backward(_ctx, _do, _dht):
        raise NotImplementedError(
            "Backward pass is not implemented for fused recurrent. "
            "Use chunk_block_causal_gated_delta_rule for training."
        )


def fused_recurrent_block_causal_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float = None,
    block_size: int = 64,
    causal_mode: int = 0,
    num_clean: int = 0,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    intermediate_states_buffer: torch.Tensor = None,
    intermediate_state_indices: torch.Tensor = None,
    cache_steps: int = 0,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Block-causal fused recurrent gated delta rule.

    This variant uses block-causal masking for the output:
    - causal_mode=0: all positions within a block see h[block_end]
    - causal_mode=1: each position sees h[position] (token-causal)
    - causal_mode=2: positions 0..num_clean-1 see h[position] (token-causal),
      positions num_clean..block_end see h[block_end] (block-causal).
      Used for self-spec variants where verify positions need token-causal and
      draft positions need block-causal readout.
    - The hidden state recurrence remains token-causal in all modes
    
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, HV, V]`.
            GQA is applied if `HV > H`.
        g (torch.Tensor):
            g (decays) of shape `[B, T, HV]`. Default: `None`.
        gk (torch.Tensor):
            gk (decays) of shape `[B, T, HV, K]`. Default: `None`.
        gv (torch.Tensor):
            gv (decays) of shape `[B, T, HV, V]`. Default: `None`.
        beta (torch.Tensor):
            betas of shape `[B, T, HV]` or `[B, T, HV, V]`.
        scale (Optional[float]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        block_size (int):
            Size of causal blocks. All positions within a block see the same
            hidden state (h at block end). Default: `64`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, HV, V, K]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, HV, V, K]`. Default: `False`.
        use_qk_l2norm_in_kernel (Optional[bool]):
            Whether to use L2 normalization in the kernel. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length inputs.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, V, K]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from block_gated_delta_rule.fused_recurrent import fused_recurrent_block_causal_gated_delta_rule
        >>> B, T, H, HV, K, V = 4, 2048, 4, 8, 128, 128
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, HV, V, device='cuda')
        >>> g = F.logsigmoid(torch.rand(B, T, HV, device='cuda'))
        >>> beta = torch.rand(B, T, HV, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, HV, V, K, device='cuda')
        >>> # Block size of 64: positions 0-63 all see h[63], positions 64-127 all see h[127], etc.
        >>> o, ht = fused_recurrent_block_causal_gated_delta_rule(
            q, k, v, g, beta,
            block_size=64,
            initial_state=h0,
            output_final_state=True
        )
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if beta is None:
        beta = torch.ones_like(q[..., 0])

    o, final_state = FusedRecurrentBlockCausalFunction.apply(
        q,
        k,
        v,
        g,
        gk,
        gv,
        beta,
        scale,
        block_size,
        causal_mode,
        num_clean,
        initial_state,
        output_final_state,
        intermediate_states_buffer,
        intermediate_state_indices,
        cache_steps,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
    )
    return o, final_state


@triton.jit
def fused_recurrent_block_causal_gated_delta_rule_packed_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    o,
    ssm_states,
    ht,
    cache_indices,
    intermediate_states_buffer,
    intermediate_state_indices,
    scale,
    stride_mixed_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_state_slot: tl.constexpr,
    stride_cache_indices: tl.constexpr,
    stride_intermediate_indices: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NUM_CLEAN: tl.constexpr,
    CAUSAL_MODE: tl.constexpr,
    CACHE_STEPS: tl.constexpr,
    CACHE_STORE_STEPS: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    CACHE_INTERMEDIATE_STATES: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
):
    i_nh, i_v = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    state_idx = tl.load(cache_indices + i_n * stride_cache_indices).to(tl.int64)
    p_o_base = o + (i_n * T * HV + i_hv) * V + o_v

    if state_idx < 0:
        zero = tl.zeros([BV], dtype=tl.float32).to(o.dtype.element_ty)
        for t in range(T):
            tl.store(p_o_base + t * HV * V, zero, mask=mask_v)
        return

    p_h = (
        ssm_states
        + state_idx * stride_state_slot
        + i_hv * V * K
        + o_v[:, None] * K
        + o_k[None, :]
    )
    b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)

    cache_idx = -1
    if CACHE_INTERMEDIATE_STATES:
        cache_idx = tl.load(
            intermediate_state_indices + i_n * stride_intermediate_indices
        ).to(tl.int64)

    row_base = i_n * T
    p_mixed_base = mixed_qkv + row_base * stride_mixed_tok
    q_off = i_h * K + o_k
    k_off = H * K + i_h * K + o_k
    v_off = 2 * H * K + i_hv * V + o_v
    p_a_base = a + row_base * stride_a_tok + i_hv
    p_b_base = b + row_base * stride_b_tok + i_hv

    A_log_exp = tl.exp(tl.load(A_log + i_hv).to(tl.float32))
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)

    p_inter_base = intermediate_states_buffer
    if CACHE_INTERMEDIATE_STATES:
        p_inter_base = (
            intermediate_states_buffer
            + cache_idx * CACHE_STEPS * HV * V * K
            + i_hv * V * K
            + o_v[:, None] * K
            + o_k[None, :]
        )

    if T == 7 and NUM_CLEAN == 4 and CAUSAL_MODE == 2:
        for t in range(4):
            p_mixed = p_mixed_base + t * stride_mixed_tok

            b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)
            if USE_QK_L2NORM_IN_KERNEL:
                b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)

            a_val = tl.load(p_a_base + t * stride_a_tok).to(tl.float32)
            b_val = tl.load(p_b_base + t * stride_b_tok).to(tl.float32)
            x = a_val + dt_bias_val
            softplus_x = tl.where(
                x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x
            )
            decay_val = exp(-A_log_exp * softplus_x)
            beta_val = tl.sigmoid(b_val)

            b_h *= decay_val
            b_delta = beta_val * (b_v - tl.sum(b_h * b_k[None, :], 1))
            b_h += b_delta[:, None] * b_k[None, :]

            if CACHE_INTERMEDIATE_STATES:
                if cache_idx >= 0:
                    p_inter = p_inter_base + t * HV * V * K
                    tl.store(p_inter, b_h.to(p_inter.dtype.element_ty), mask=mask_h)

            b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
            if USE_QK_L2NORM_IN_KERNEL:
                b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_q = b_q * scale
            b_o = tl.sum(b_h * b_q[None, :], 1)
            tl.store(
                p_o_base + t * HV * V,
                b_o.to(o.dtype.element_ty),
                mask=mask_v,
            )

        for t in range(4, 7):
            p_mixed = p_mixed_base + t * stride_mixed_tok

            b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)
            if USE_QK_L2NORM_IN_KERNEL:
                b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)

            a_val = tl.load(p_a_base + t * stride_a_tok).to(tl.float32)
            b_val = tl.load(p_b_base + t * stride_b_tok).to(tl.float32)
            x = a_val + dt_bias_val
            softplus_x = tl.where(
                x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x
            )
            decay_val = exp(-A_log_exp * softplus_x)
            beta_val = tl.sigmoid(b_val)

            b_h *= decay_val
            b_delta = beta_val * (b_v - tl.sum(b_h * b_k[None, :], 1))
            b_h += b_delta[:, None] * b_k[None, :]

        for t in range(4, 7):
            p_mixed = p_mixed_base + t * stride_mixed_tok
            b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
            if USE_QK_L2NORM_IN_KERNEL:
                b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_q = b_q * scale
            b_o = tl.sum(b_h * b_q[None, :], 1)
            tl.store(
                p_o_base + t * HV * V,
                b_o.to(o.dtype.element_ty),
                mask=mask_v,
            )

        if STORE_FINAL_STATE:
            p_ht = (
                ht + (i_n * HV + i_hv) * V * K + o_v[:, None] * K + o_k[None, :]
            )
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        return

    for t in range(T):
        p_mixed = p_mixed_base + t * stride_mixed_tok

        b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)

        a_val = tl.load(p_a_base + t * stride_a_tok).to(tl.float32)
        b_val = tl.load(p_b_base + t * stride_b_tok).to(tl.float32)
        x = a_val + dt_bias_val
        softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
        decay_val = exp(-A_log_exp * softplus_x)
        beta_val = tl.sigmoid(b_val)

        b_h *= decay_val
        b_delta = beta_val * (b_v - tl.sum(b_h * b_k[None, :], 1))
        b_h += b_delta[:, None] * b_k[None, :]

        if CACHE_INTERMEDIATE_STATES:
            if t < CACHE_STORE_STEPS:
                if cache_idx >= 0:
                    p_inter = p_inter_base + t * HV * V * K
                    tl.store(p_inter, b_h.to(p_inter.dtype.element_ty), mask=mask_h)

        if CAUSAL_MODE == 1 or t < NUM_CLEAN:
            b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
            if USE_QK_L2NORM_IN_KERNEL:
                b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_q = b_q * scale
            b_o = tl.sum(b_h * b_q[None, :], 1)
            tl.store(
                p_o_base + t * HV * V,
                b_o.to(o.dtype.element_ty),
                mask=mask_v,
            )

    if CAUSAL_MODE == 2:
        for t in range(NUM_CLEAN, T):
            p_mixed = p_mixed_base + t * stride_mixed_tok
            b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
            if USE_QK_L2NORM_IN_KERNEL:
                b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_q = b_q * scale
            b_o = tl.sum(b_h * b_q[None, :], 1)
            tl.store(
                p_o_base + t * HV * V,
                b_o.to(o.dtype.element_ty),
                mask=mask_v,
            )

    if STORE_FINAL_STATE:
        p_ht = ht + (i_n * HV + i_hv) * V * K + o_v[:, None] * K + o_k[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_block_causal_gated_delta_rule_packed(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    block_size: int,
    causal_mode: int,
    num_clean: int = 0,
    output_final_state: bool = False,
    intermediate_states_buffer: torch.Tensor = None,
    intermediate_state_indices: torch.Tensor = None,
    cache_steps: int = 0,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if causal_mode not in (1, 2):
        raise ValueError(
            "Packed dLLM GDN block kernel currently supports causal_mode 1 or 2."
        )
    if mixed_qkv.ndim != 2 or a.ndim != 2 or b.ndim != 2:
        raise ValueError("mixed_qkv/a/b must be 2D tensors.")
    if mixed_qkv.stride(-1) != 1 or a.stride(-1) != 1 or b.stride(-1) != 1:
        raise ValueError("mixed_qkv/a/b must be contiguous in the last dimension.")
    if cache_indices.ndim != 1:
        raise ValueError("cache_indices must be a 1D tensor.")
    if block_size <= 0 or mixed_qkv.shape[0] % block_size != 0:
        raise ValueError(
            f"Invalid block_size={block_size} for {mixed_qkv.shape[0]} rows."
        )
    if ssm_states.ndim != 4:
        raise ValueError("ssm_states must have shape [slots, HV, V, K].")

    n_seq = mixed_qkv.shape[0] // block_size
    if cache_indices.shape[0] != n_seq:
        raise ValueError(
            f"cache_indices shape mismatch: got {cache_indices.shape[0]}, expected {n_seq}."
        )
    hv, v_dim, k_dim = ssm_states.shape[-3:]
    if a.shape != (mixed_qkv.shape[0], hv) or b.shape != (mixed_qkv.shape[0], hv):
        raise ValueError(
            f"a/b must have shape {(mixed_qkv.shape[0], hv)}, got {tuple(a.shape)} and {tuple(b.shape)}."
        )

    qkv_dim = mixed_qkv.shape[1]
    qk_dim = qkv_dim - hv * v_dim
    if qk_dim <= 0 or qk_dim % 2 != 0:
        raise ValueError(f"Invalid packed qkv dim {qkv_dim}.")
    q_dim = qk_dim // 2
    if q_dim % k_dim != 0:
        raise ValueError(f"Invalid q dim {q_dim} for K={k_dim}.")
    h = q_dim // k_dim
    if h <= 0 or hv % h != 0:
        raise ValueError(f"Invalid inferred head config H={h}, HV={hv}.")

    bk = triton.next_power_of_2(k_dim)
    if triton.cdiv(k_dim, bk) != 1:
        raise ValueError(f"Packed block kernel only supports NK=1, got K={k_dim}.")
    bv = min(triton.next_power_of_2(v_dim), 32)
    num_warps = 1
    num_stages = 3
    if block_size == 7 and num_clean == 4 and causal_mode == 2:
        bv_override = int(
            os.getenv("SGLANG_HYBRID_DIFFUSION_SELF_SPEC_GDN_BLOCK7_BV", "16")
        )
        if bv_override > 0:
            bv = min(max(1, bv_override), triton.next_power_of_2(v_dim))
        num_warps = int(
            os.getenv("SGLANG_HYBRID_DIFFUSION_SELF_SPEC_GDN_BLOCK7_WARPS", "1")
        )
        num_stages = int(
            os.getenv("SGLANG_HYBRID_DIFFUSION_SELF_SPEC_GDN_BLOCK7_STAGES", "3")
        )
    out = mixed_qkv.new_empty(1, mixed_qkv.shape[0], hv, v_dim)
    final_state = (
        torch.empty(n_seq, hv, v_dim, k_dim, dtype=torch.float32, device=mixed_qkv.device)
        if output_final_state
        else None
    )

    if intermediate_states_buffer is None:
        intermediate_states_buffer_arg = ssm_states
        intermediate_state_indices_arg = cache_indices
        cache_steps_arg = 0
    else:
        if intermediate_state_indices is None:
            raise ValueError("intermediate_state_indices is required.")
        intermediate_states_buffer_arg = intermediate_states_buffer
        intermediate_state_indices_arg = intermediate_state_indices
        cache_steps_arg = cache_steps
    cache_store_steps_arg = cache_steps_arg
    if causal_mode == 2 and num_clean > 0:
        # In mixed block-7 self-spec, accepted GDN state is only selected from
        # clean positions. Keep the allocation stride while skipping unreachable
        # draft-position stores.
        cache_store_steps_arg = min(cache_steps_arg, num_clean)

    grid = (n_seq * hv, triton.cdiv(v_dim, bv))
    fused_recurrent_block_causal_gated_delta_rule_packed_kernel[grid](
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        out,
        ssm_states,
        final_state if final_state is not None else ssm_states,
        cache_indices,
        intermediate_states_buffer_arg,
        intermediate_state_indices_arg,
        k_dim ** -0.5,
        mixed_qkv.stride(0),
        a.stride(0),
        b.stride(0),
        ssm_states.stride(0),
        cache_indices.stride(0),
        intermediate_state_indices_arg.stride(0),
        block_size,
        h,
        hv,
        k_dim,
        v_dim,
        bk,
        bv,
        num_clean,
        causal_mode,
        cache_steps_arg,
        cache_store_steps_arg,
        final_state is not None,
        intermediate_states_buffer is not None,
        use_qk_l2norm_in_kernel,
        20.0,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, final_state
