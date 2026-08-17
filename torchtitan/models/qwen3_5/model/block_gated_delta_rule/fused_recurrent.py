# Copyright (c) 2026 Yuchen Zhu
# Portions copyright (c) 2023-2025 Songlin Yang, Yu Zhang
# Modified from Flash Linear Attention (FLA) for HybridDiffusion block-causal training.

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
    CAUSAL_MODE: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
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
    mask_h = mask_k[:, None] & mask_v[None, :]

    # Initialize hidden state
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

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
                b_h *= exp(b_gk[:, None])
            if USE_GV:
                b_gv = tl.load(p_gv, mask=mask_v, other=0).to(tl.float32)
                b_h *= exp(b_gv[None, :])
            b_delta = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
            b_h += b_k[:, None] * b_delta

            b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
            if USE_QK_L2NORM_IN_KERNEL:
                b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_q = b_q * scale
            b_o = tl.sum(b_h * b_q[:, None], 0)
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
                        b_h *= exp(b_gk[:, None])
                    if USE_GV:
                        b_gv = tl.load(p_gv, mask=mask_v, other=0).to(tl.float32)
                        b_h *= exp(b_gv[None, :])
                    b_delta = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
                    b_h += b_k[:, None] * b_delta
                    p_k += stride_qk
                    p_v += stride_v
                    p_beta += stride_beta
                    if USE_G:
                        p_g += stride_g
                    if USE_GK:
                        p_gk += stride_gk
                    if USE_GV:
                        p_gv += stride_gv

            p_q = p_q_base + block_start * stride_qk
            p_o = p_o_base + block_start * stride_o
            for local_t in range(BLOCK_SIZE):
                t = block_start + local_t
                if t < T:
                    b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
                    if USE_QK_L2NORM_IN_KERNEL:
                        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
                    b_q = b_q * scale
                    b_o = tl.sum(b_h * b_q[:, None], 0)
                    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
                    p_q += stride_qk
                    p_o += stride_o

    # Store final hidden state if requested
    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
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
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
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
        initial_state: Initial hidden state of shape [N, HV, K, V]
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
    final_state = q.new_empty(N, HV, K, V, dtype=torch.float32) if output_final_state else None

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
        CAUSAL_MODE=causal_mode,
        IS_BETA_HEADWISE=beta.ndim != v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        num_warps=1,
        num_stages=3,
    )
    return o, final_state


class FusedRecurrentBlockCausalFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
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
        initial_state: torch.Tensor = None,
        output_final_state: bool = False,
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
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
        )
        return o, final_state

    @staticmethod
    @input_guard
    def backward(ctx, do, dht):
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
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Block-causal fused recurrent gated delta rule.
    
    This variant uses block-causal masking for the output:
    - All positions within a block see h[block_end], not h[position]
    - The hidden state recurrence remains token-causal
    - Useful for inference with block-parallel decoding (e.g., speculative decoding)
    
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
            Initial state of shape `[N, HV, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, HV, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (Optional[bool]):
            Whether to use L2 normalization in the kernel. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length inputs.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from torchtitan.models.qwen3_5.model.block_gated_delta_rule import fused_recurrent_block_causal_gated_delta_rule
        >>> B, T, H, HV, K, V = 4, 2048, 4, 8, 128, 128
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, HV, V, device='cuda')
        >>> g = F.logsigmoid(torch.rand(B, T, HV, device='cuda'))
        >>> beta = torch.rand(B, T, HV, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, HV, K, V, device='cuda')
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
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
    )
    return o, final_state
