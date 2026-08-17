# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Qwen3.5 Model - Hybrid architecture with Gated DeltaNet linear attention
and Gated Full Attention layers.

Key differences from Qwen3:
- Hybrid layers: 75% Gated DeltaNet (linear) + 25% Gated Full Attention
- Gated attention output: attn_output * sigmoid(gate)
- RMSNorm with (1 + weight) centering (zero-initialized weights)
- Partial RoPE with partial_rotary_factor=0.25
- 1D causal convolution on QKV for linear attention layers
- Larger vocabulary (248320 tokens)
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import and_masks, BlockMask

from fla.modules.convolution import causal_conv1d as _fla_causal_conv1d
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.attention import (
    create_attention_mask,
    create_varlen_metadata_from_doc_ids,
    create_varlen_metadata_for_document,
    FlexAttentionWrapper,
    get_causal_mask_mod,
    get_doc_id_mask_mod,
    get_document_mask_mod,
    ScaledDotProductAttentionWrapper,
    VarlenAttentionWrapper,
    VarlenMetadata,
)
from torchtitan.models.moe import MoE
from torchtitan.protocols.model import AttentionMasksType
from torchtitan.protocols.train_spec import ModelProtocol

try:
    from liger_kernel.transformers.functional import (
        liger_fused_linear_cross_entropy,
    )
except ImportError:
    liger_fused_linear_cross_entropy = None

from .args import Qwen3_5ModelArgs


def precompute_rope_cache(
    dim: int, max_seq_len: int, base: float = 1_000_000.0
) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=freqs.dtype, device=freqs.device)
    idx_theta = torch.outer(t, freqs).float()
    freqs = torch.cat([idx_theta, idx_theta], dim=-1)
    rope_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    return rope_cache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def reshape_for_broadcast(
    rope_cache: torch.Tensor, x: torch.Tensor, positions: torch.Tensor | None = None
) -> torch.Tensor:
    ndim = x.ndim
    assert ndim > 1
    bz, seqlen, _, head_dim = x.shape
    if positions is None:
        rope_cache = rope_cache[0:seqlen]
        assert rope_cache.shape == (seqlen, head_dim * 2)
        shape = [-1, seqlen, 1, head_dim * 2]
        return rope_cache.view(*shape)
    elif positions.size(0) == 1:
        assert positions.shape == (1, seqlen)
        rope_cache = rope_cache[positions.squeeze(0)]
        assert rope_cache.shape == (seqlen, head_dim * 2)
        shape = [-1, seqlen, 1, head_dim * 2]
        return rope_cache.view(*shape)
    else:
        assert positions.shape == (bz, seqlen)
        rope_cache_expanded = rope_cache[None, :, None, :].expand(bz, -1, -1, -1)
        rope_cache = torch.gather(
            rope_cache_expanded,
            dim=1,
            index=positions.view(bz, seqlen, 1, 1).expand(bz, seqlen, 1, head_dim * 2),
        )
        assert rope_cache.shape == (bz, seqlen, 1, head_dim * 2)
        return rope_cache


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = xq.shape[-1]
    rope_cache = reshape_for_broadcast(rope_cache, xq, positions)
    cos = rope_cache[..., :head_dim].to(dtype=xq.dtype, device=xq.device)
    sin = rope_cache[..., head_dim:].to(dtype=xq.dtype, device=xq.device)
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_partial_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    partial_rotary_factor: float,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to only a fraction of the head dimensions."""
    head_dim = xq.shape[-1]
    rot_dim = int(head_dim * partial_rotary_factor)
    # Ensure rot_dim is even
    rot_dim = rot_dim - (rot_dim % 2)

    xq_rot, xq_pass = xq[..., :rot_dim], xq[..., rot_dim:]
    xk_rot, xk_pass = xk[..., :rot_dim], xk[..., rot_dim:]

    xq_rot, xk_rot = apply_rotary_emb(xq_rot, xk_rot, rope_cache, positions)

    xq_out = torch.cat([xq_rot, xq_pass], dim=-1)
    xk_out = torch.cat([xk_rot, xk_pass], dim=-1)
    return xq_out, xk_out


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def _to_local_if_dtensor(tensor: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.distributed, "tensor"):
        from torch.distributed.tensor import DTensor

        if isinstance(tensor, DTensor):
            return tensor.to_local()
    return tensor


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def _doc_ids_to_cu_seqlens(doc_ids: torch.Tensor) -> torch.Tensor:
    """Convert packed per-token document IDs `[B, L]` to FLA varlen offsets."""
    B, L = doc_ids.shape
    parts = []
    for row_idx in range(B):
        row = doc_ids[row_idx]
        bounds = torch.where(row[1:] != row[:-1])[0] + 1
        parts.append(torch.cat([row.new_zeros(1), bounds]) + row_idx * L)
    parts.append(doc_ids.new_tensor([B * L]))
    return torch.cat(parts).to(device=doc_ids.device, dtype=torch.long)


def _doc_ids_to_positions(doc_ids: torch.Tensor) -> torch.Tensor:
    """Convert per-token document IDs to per-document position indices."""
    B, L = doc_ids.shape
    device = doc_ids.device
    boundary = torch.cat(
        [
            torch.ones(B, 1, dtype=torch.long, device=device),
            (doc_ids[:, 1:] != doc_ids[:, :-1]).long(),
        ],
        dim=1,
    )
    global_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
    first_occurrence = boundary * global_idx
    doc_start = torch.cummax(first_occurrence, dim=1).values
    return global_idx - doc_start


class Qwen3_5RMSNorm(nn.Module):
    """RMSNorm with (1 + weight) centering, matching HF Qwen3.5 implementation."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def reset_parameters(self):
        nn.init.zeros_(self.weight)


class Qwen3_5RMSNormGated(nn.Module):
    """Gated RMSNorm for linear attention output: norm(x) * silu(z)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x_float = x.float()
        x_normed = x_float * torch.rsqrt(
            x_float.pow(2).mean(-1, keepdim=True) + self.eps
        )
        weight = _to_local_if_dtensor(self.weight).float()
        x_normed = weight * x_normed.to(dtype=x_dtype)
        return (x_normed * F.silu(z.float())).to(dtype=x_dtype)

    def reset_parameters(self):
        nn.init.ones_(self.weight)


class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet linear attention layer for Qwen3.5.

    Uses a delta rule recurrence with gating for near-linear complexity.
    On CUDA, dispatches to the fused chunk kernel with block_size=1 to
    preserve token-causal semantics. Falls back to the reference
    recurrence on non-CUDA devices.
    """

    def __init__(self, model_args: Qwen3_5ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = model_args.dim

        self.num_k_heads = model_args.linear_num_key_heads
        self.num_v_heads = model_args.linear_num_value_heads
        self.head_k_dim = model_args.linear_key_head_dim
        self.head_v_dim = model_args.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_kernel_size = model_args.linear_conv_kernel_dim
        self.delta_rule_block_size = 1

        # Projections (split into separate Q/K/V, Z, beta, alpha)
        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        # 1D causal convolution over QKV (no bias, matching HF checkpoint)
        self.conv1d = nn.Conv1d(
            in_channels=self.key_dim * 2 + self.value_dim,
            out_channels=self.key_dim * 2 + self.value_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.key_dim * 2 + self.value_dim,
            padding=self.conv_kernel_size - 1,
            bias=False,
        )

        # Learnable parameters for gating
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads))
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        # Output
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=model_args.norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def init_weights(self, init_std: float):
        for linear in (self.in_proj_qkv, self.in_proj_z, self.in_proj_b, self.in_proj_a):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.out_proj.weight, mean=0.0, std=init_std)
        nn.init.ones_(self.dt_bias)
        nn.init.uniform_(self.A_log, 0, 16)
        self.A_log.data.log_()
        self.norm.reset_parameters()
        nn.init.trunc_normal_(self.conv1d.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_mesh = None
        if hasattr(torch.distributed, "tensor"):
            from torch.distributed.tensor import DTensor, Replicate

            if isinstance(x, DTensor):
                input_mesh = x.device_mesh
                x = x.to_local()

        original_shape = x.shape
        need_unflatten = cu_seqlens is not None and x.shape[0] != 1
        if need_unflatten:
            x = x.contiguous().reshape(1, x.shape[0] * x.shape[1], x.shape[2])

        output = self._forward_impl(x, cu_seqlens=cu_seqlens)

        if need_unflatten:
            output = output.reshape(original_shape[0], original_shape[1], -1)

        if input_mesh is not None:
            replicated = tuple(Replicate() for _ in range(input_mesh.ndim))
            output = DTensor.from_local(
                output,
                input_mesh,
                replicated,
                run_check=False,
            )
        return output

    def _forward_impl(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        mixed_qkv = F.linear(x, _to_local_if_dtensor(self.in_proj_qkv.weight))

        z = F.linear(x, _to_local_if_dtensor(self.in_proj_z.weight))
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = F.linear(x, _to_local_if_dtensor(self.in_proj_b.weight))
        a = F.linear(x, _to_local_if_dtensor(self.in_proj_a.weight))

        conv_weight = _to_local_if_dtensor(self.conv1d.weight)
        if cu_seqlens is not None:
            conv_out = _fla_causal_conv1d(
                x=mixed_qkv,
                weight=conv_weight.squeeze(1),
                bias=None,
                activation="silu",
                cu_seqlens=cu_seqlens,
            )
            mixed_qkv = conv_out[0] if isinstance(conv_out, tuple) else conv_out
        else:
            mixed_qkv = mixed_qkv.transpose(1, 2)  # (B, C, L) for conv1d
            mixed_qkv = F.silu(
                F.conv1d(
                    mixed_qkv,
                    conv_weight,
                    bias=None,
                    padding=self.conv_kernel_size - 1,
                    groups=self.key_dim * 2 + self.value_dim,
                )[:, :, :seq_len]
            )
            mixed_qkv = mixed_qkv.transpose(1, 2)  # (B, L, C)

        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        a_log = _to_local_if_dtensor(self.A_log).float()
        dt_bias = _to_local_if_dtensor(self.dt_bias)
        g = -a_log.exp() * F.softplus(a.float() + dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)

        output = self._gated_delta_rule(
            query, key, value, g, beta, cu_seqlens=cu_seqlens,
        )

        output = output.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        output = self.norm(output, z)
        output = output.reshape(batch_size, seq_len, -1)

        output = F.linear(output, _to_local_if_dtensor(self.out_proj.weight))
        return output

    def _gated_delta_rule_reference(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gated delta rule recurrence (reference implementation).

        The delta rule corrects the value before writing to the state:
            v_corrected = v_t - k_t^T @ S_{t-1}
            S_t = decay_t * S_{t-1} + beta_t * k_t outer v_corrected
            o_t = q_t @ S_t

        This differs from a simple linear recurrence by subtracting the
        current state's prediction for the key, which is critical for
        correct associative memory behavior.
        """
        B, L, H, Dk = query.shape
        Dv = value.shape[-1]

        output_dtype = query.dtype
        query = query.float()
        key = key.float()
        value = value.float()

        scale = Dk ** -0.5
        query = query * scale

        decay = g.exp()  # (B, L, H)

        output = torch.zeros(B, L, H, Dv, device=query.device, dtype=torch.float32)

        if cu_seqlens is None:
            ranges = [(0, L)]
        else:
            if B != 1:
                raise ValueError("cu_seqlens reference path expects flattened B=1 input")
            ranges = [
                (int(cu_seqlens[i].item()), int(cu_seqlens[i + 1].item()))
                for i in range(len(cu_seqlens) - 1)
            ]

        for start, end in ranges:
            state = torch.zeros(B, H, Dk, Dv, device=query.device, dtype=torch.float32)
            for t in range(start, end):
                k_t = key[:, t]          # (B, H, Dk)
                v_t = value[:, t]        # (B, H, Dv)
                q_t = query[:, t]        # (B, H, Dk)
                b_t = beta[:, t]         # (B, H)
                d_t = decay[:, t]        # (B, H)

                state = d_t[:, :, None, None] * state
                v_old = torch.einsum("bhk,bhkv->bhv", k_t, state)
                v_corrected = (v_t - v_old) * b_t[:, :, None]
                state = state + torch.einsum("bhk,bhv->bhkv", k_t, v_corrected)
                output[:, t] = torch.einsum("bhk,bhkv->bhv", q_t, state)

        return output.to(dtype=output_dtype)

    def _gated_delta_rule(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not query.is_cuda:
            return self._gated_delta_rule_reference(
                query, key, value, g, beta, cu_seqlens=cu_seqlens,
            )

        output, _ = chunk_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            scale=self.head_k_dim ** -0.5,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=cu_seqlens,
        )
        return output


class GatedAttention(nn.Module):
    """
    Gated full attention for Qwen3.5.

    Standard GQA with:
    - Q/K RMSNorm
    - Partial RoPE (only partial_rotary_factor of head_dim gets rotary)
    - Gated output: attn_output * sigmoid(gate)
    """

    q_norm: Qwen3_5RMSNorm | None
    k_norm: Qwen3_5RMSNorm | None

    def __init__(self, model_args: Qwen3_5ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.head_dim
        self.partial_rotary_factor = model_args.partial_rotary_factor
        self.scaling = self.head_dim ** -0.5
        self.attn_type = getattr(model_args, "attn_type", "sdpa")

        if model_args.qk_norm:
            self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=model_args.norm_eps)
            self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=model_args.norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        # Q projection outputs 2x width for gating: [query, gate]
        self.wq = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim * 2, bias=False
        )
        self.wk = nn.Linear(
            model_args.dim, self.n_kv_heads * self.head_dim, bias=False
        )
        self.wv = nn.Linear(
            model_args.dim, self.n_kv_heads * self.head_dim, bias=False
        )
        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )

        match self.attn_type:
            case "flex":
                self.inner_attention = FlexAttentionWrapper()
            case "varlen":
                self.inner_attention = VarlenAttentionWrapper()
            case "sdpa":
                self.inner_attention = ScaledDotProductAttentionWrapper()
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_type}")

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        if self.k_norm is not None:
            self.k_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        bs, seqlen, _ = x.shape
        xq_full = self.wq(x)
        xk, xv = self.wk(x), self.wv(x)

        # Split Q projection into query and gate
        xq_full = xq_full.view(bs, seqlen, self.n_heads, self.head_dim * 2)
        xq = xq_full[..., :self.head_dim]
        gate = xq_full[..., self.head_dim:]

        xq = xq.contiguous()
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        if self.q_norm:
            xq = self.q_norm(xq)
        if self.k_norm:
            xk = self.k_norm(xk)

        # Apply partial rotary embedding
        xq, xk = apply_partial_rotary_emb(
            xq, xk, rope_cache, self.partial_rotary_factor, positions
        )

        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)

        match self.attn_type:
            case "flex":
                assert isinstance(attention_masks, BlockMask), attention_masks
                output = self.inner_attention(
                    xq, xk, xv, block_mask=attention_masks, scale=self.scaling
                )
            case "varlen":
                assert isinstance(attention_masks, VarlenMetadata), attention_masks
                output = self.inner_attention(
                    xq, xk, xv, self.head_dim, attention_masks, scale=self.scaling
                )
            case "sdpa":
                assert attention_masks is None
                output = self.inner_attention(xq, xk, xv, scale=self.scaling)
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_type}")

        output = output.transpose(1, 2).contiguous()

        # Apply gating: output * sigmoid(gate)
        gate = gate.transpose(1, 2).transpose(1, 2)  # back to (bs, seqlen, n_heads, head_dim)
        output_gated = output * torch.sigmoid(gate)

        output_gated = output_gated.view(bs, seqlen, -1)
        return self.wo(output_gated)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class TransformerBlock(nn.Module):
    """
    Hybrid transformer block for Qwen3.5.

    Each block is either a full_attention or linear_attention layer,
    determined by model_args.layer_types[layer_id].
    """

    def __init__(self, layer_id: int, model_args: Qwen3_5ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.layer_type = model_args.layer_types[layer_id]

        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(model_args, layer_id)
        elif self.layer_type == "full_attention":
            self.self_attn = GatedAttention(model_args)
        else:
            raise ValueError(f"Unknown layer type: {self.layer_type}")

        self.moe_enabled = model_args.moe_enabled
        if self.moe_enabled:
            self.moe = MoE(
                model_args.moe_args,
                dim=model_args.dim,
                hidden_dim=model_args.moe_inter_dim,
            )
        else:
            self.feed_forward = FeedForward(
                dim=model_args.dim, hidden_dim=model_args.hidden_dim
            )
        self.input_layernorm = Qwen3_5RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            model_args.dim, eps=model_args.norm_eps
        )

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ):
        residual = x
        h = self.input_layernorm(x)

        if self.layer_type == "linear_attention":
            h = self.linear_attn(h, cu_seqlens=cu_seqlens)
        else:
            h = self.self_attn(h, rope_cache, attention_masks, positions)

        x = residual + h

        residual = x
        h = self.post_attention_layernorm(x)
        if self.moe_enabled:
            h = self.moe(h)
        else:
            h = self.feed_forward(h)
        x = residual + h
        return x

    def init_weights(self, buffer_device: torch.device):
        self.input_layernorm.reset_parameters()
        self.post_attention_layernorm.reset_parameters()
        if self.layer_type == "linear_attention":
            self.linear_attn.init_weights(self.weight_init_std)
        else:
            self.self_attn.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(self.weight_init_std, buffer_device)
        else:
            self.feed_forward.init_weights(self.weight_init_std)


class Qwen3_5Model(nn.Module, ModelProtocol):
    """
    Qwen3.5 Model - Hybrid Gated DeltaNet + Gated Full Attention.
    """

    def __init__(self, model_args: Qwen3_5ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.eos_id = model_args.eos_id
        self.head_dim = model_args.head_dim

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        # RoPE cache uses partial rotary dim
        self.register_buffer(
            "rope_cache", self._precompute_rope_cache(), persistent=False
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        self.norm = Qwen3_5RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

    def init_weights(
        self,
        buffer_device: torch.device | None = None,
    ):
        buffer_device = buffer_device or self.rope_cache.device
        with torch.device(buffer_device):
            self.rope_cache = self._precompute_rope_cache()
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device)
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.model_args.dim ** -0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def _precompute_rope_cache(self) -> torch.Tensor:
        rot_dim = int(self.model_args.head_dim * self.model_args.partial_rotary_factor)
        rot_dim = rot_dim - (rot_dim % 2)
        return precompute_rope_cache(
            rot_dim,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def _get_flex_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        mask_mods = [get_causal_mask_mod()]
        match self.model_args.attn_mask_type:
            case "causal":
                B = 1
            case "block_causal":
                B = input_batch.shape[0]
                mask_mods.append(get_document_mask_mod(input_batch, tokenizer.eos_id))
            case "doc_causal":
                B = input_batch.shape[0]
                if extra_inputs and "doc_ids" in extra_inputs:
                    mask_mods.append(get_doc_id_mask_mod(extra_inputs["doc_ids"]))
                else:
                    mask_mods.append(get_document_mask_mod(input_batch, tokenizer.eos_id))
            case _:
                raise ValueError(
                    f"Unknown attention mask type: {self.model_args.attn_mask_type}"
                )
        return create_attention_mask(
            and_masks(*mask_mods), B, None, input_batch.shape[1], input_batch.shape[1]
        )

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        match self.model_args.attn_type:
            case "flex":
                return self._get_flex_attention_masks(
                    input_batch, tokenizer, extra_inputs
                )
            case "varlen":
                if (
                    self.model_args.attn_mask_type == "doc_causal"
                    and extra_inputs
                    and "doc_ids" in extra_inputs
                ):
                    return create_varlen_metadata_from_doc_ids(
                        extra_inputs["doc_ids"]
                    )
                return create_varlen_metadata_for_document(
                    input_batch, tokenizer.eos_id
                )
            case _:
                raise NotImplementedError(
                    "Only varlen and flex attn masks are supported"
                )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        doc_ids = kwargs.get("doc_ids")
        if positions is None and doc_ids is not None:
            positions = _doc_ids_to_positions(doc_ids)
        cu_seqlens = _doc_ids_to_cu_seqlens(doc_ids) if doc_ids is not None else None

        for layer in self.layers.values():
            h = layer(
                h,
                self.rope_cache,
                attention_masks,
                positions,
                cu_seqlens=cu_seqlens,
            )

        h = self.norm(h) if self.norm else h

        if labels is not None and self.training and self.output is not None:
            valid = labels != -100
            if not valid.any():
                return {"loss": (h.sum() + self.output.weight.sum()) * 0.0}
            if liger_fused_linear_cross_entropy is not None:
                loss = liger_fused_linear_cross_entropy(
                    h.reshape(-1, h.shape[-1]),
                    self.output.weight,
                    labels.reshape(-1),
                    bias=self.output.bias,
                    ignore_index=-100,
                    reduction="mean",
                )
                return {"loss": loss}

        output = self.output(h) if self.output else h
        return output

