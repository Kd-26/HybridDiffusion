# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DLLM model - extends Qwen3 with block diffusion training capabilities.

Block Diffusion Training:
- Concatenates noisy sequence x_t with clean sequence x_0: [x_t; x_0]
- Uses specialized attention mask (block_diff_mask) for proper attention patterns
- Applies independent RoPE positions: both x_t and x_0 get positions [0, 1, ..., n-1]
"""

from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
)

from torchtitan.models.qwen3.model.model import (
    Qwen3Model,
    TransformerBlock,
    Attention,
    apply_rotary_emb,
    repeat_kv,
)
from torchtitan.models.qwen3.model.args import Qwen3ModelArgs
from torchtitan.models.attention import FlexAttentionWrapper
from torchtitan.config import JobConfig


@dataclass
class DLLMModelArgs(Qwen3ModelArgs):
    """DLLM args - Qwen3 + diffusion-specific parameters."""
    mask_token_id: int = -1  # -1 = vocab_size-1 (backward compat); set via [dllm] config
    enable_weight_tying: bool = False
    block_size: int = 1  # Block size for diffusion masking (1 = token-level)
    antithetic_sampling: bool = True  # Use permuted stratified block-noise sampling for variance reduction
    complementary_mask: bool = False  # Generate complementary masks, doubling batch to 2B
    use_flex_attention: bool = True  # Use FlexAttention (faster) vs SDPA with 4D mask
    causal_x0: bool = False  # x_0 uses token-level causal attention instead of block-causal
    ar_loss_weight: float = 0.0  # Weight for AR NTP loss on x_0 outputs (0 = disabled)
    use_fused_ce: bool = False  # Use liger-kernel fused linear cross-entropy
    # Diffusion recipe flags
    all_masked: bool = False  # 100% masking rate (all tokens → MASK)
    causal_xt: bool = False  # x_t uses causal rather than bidirectional attention within blocks
    logit_shift: bool = False  # Dream-style: position i predicts token i+1
    loss_auto_balance: bool = False  # Dynamic delta = L_mask/L_clean
    loss_normalize: bool = True  # Normalize by (1+ar_weight); False uses additive dual loss

    # Qwen3 vocab layout: base vocab 0-151642, special tokens 151643-151668,
    # Reserve <|MASK|> at 151669; the embedding table size is 151936.
    # Use these in new experiment configs via [dllm] mask_token_id / pad_token_id.
    QWEN3_MASK_TOKEN_ID: int = 151669
    QWEN3_PAD_TOKEN_ID: int = 151643

    def __post_init__(self):
        if self.mask_token_id < 0:
            # Default: vocab_size - 1 for backward compatibility with old checkpoints.
            # New experiments should set mask_token_id = 151669 in [dllm] config.
            self.mask_token_id = self.vocab_size - 1

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        super().update_from_config(job_config, **kwargs)
        # Override from config if dllm section exists
        if hasattr(job_config, 'dllm'):
            if hasattr(job_config.dllm, 'block_size'):
                self.block_size = job_config.dllm.block_size
            if hasattr(job_config.dllm, 'mask_token_id') and job_config.dllm.mask_token_id >= 0:
                self.mask_token_id = job_config.dllm.mask_token_id
            if hasattr(job_config.dllm, 'antithetic_sampling'):
                self.antithetic_sampling = job_config.dllm.antithetic_sampling
            if hasattr(job_config.dllm, 'complementary_mask'):
                self.complementary_mask = job_config.dllm.complementary_mask
            if hasattr(job_config.dllm, 'use_flex_attention'):
                self.use_flex_attention = job_config.dllm.use_flex_attention
            if hasattr(job_config.dllm, 'causal_x0'):
                self.causal_x0 = job_config.dllm.causal_x0
            if hasattr(job_config.dllm, 'ar_loss_weight'):
                self.ar_loss_weight = job_config.dllm.ar_loss_weight
            if hasattr(job_config.dllm, 'use_fused_ce'):
                self.use_fused_ce = job_config.dllm.use_fused_ce
            if hasattr(job_config.dllm, 'all_masked'):
                self.all_masked = job_config.dllm.all_masked
            if hasattr(job_config.dllm, 'causal_xt'):
                self.causal_xt = job_config.dllm.causal_xt
            if hasattr(job_config.dllm, 'logit_shift'):
                self.logit_shift = job_config.dllm.logit_shift
            if hasattr(job_config.dllm, 'loss_auto_balance'):
                self.loss_auto_balance = job_config.dllm.loss_auto_balance
            if hasattr(job_config.dllm, 'loss_normalize'):
                self.loss_normalize = job_config.dllm.loss_normalize
            if hasattr(job_config.dllm, 'enable_weight_tying'):
                self.enable_weight_tying = job_config.dllm.enable_weight_tying

            # Validation
            if self.ar_loss_weight > 0 and not self.causal_x0:
                raise ValueError(
                    "ar_loss_weight > 0 requires causal_x0 = True. "
                    "Block-causal x_0 attention lets tokens see future within the block, "
                    "which invalidates AR next-token prediction."
                )
            if self.causal_xt and not self.causal_x0:
                raise ValueError(
                    "causal_xt = True requires causal_x0 = True."
                )
            if self.loss_auto_balance and self.ar_loss_weight <= 0:
                raise ValueError(
                    "loss_auto_balance = True requires ar_loss_weight > 0 "
                    "(need two losses to balance)."
                )
            if self.all_masked and self.complementary_mask:
                from torchtitan.tools.logging import logger
                logger.warning(
                    "all_masked=True with complementary_mask=True is meaningless. "
                    "Forcing complementary_mask=False."
                )
                self.complementary_mask = False

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        """
        Calculate nparams and FLOPs for DLLM, accounting for:
        1. Doubled sequence length: [x_t; x_0] = 2*seq_len tokens through transformer
        2. Complementary masking: 2x batch size when enabled
        
        The MFU formula is: MFU = num_flops_per_token * (ntokens/time) / peak_flops
        
        Since ntokens counts input tokens (B * L), but DLLM actually computes:
        - B' = 2*B if complementary_mask (batch doubling)
        - L' = 2*L for transformer (sequence doubling [x_t; x_0])
        
        We need num_flops_per_token to represent FLOPs per *input* token.
        
        Base model FLOPs per token: 6*(nparams-embed) + 6*n_layers*n_heads*head_dims*seq_len
        
        DLLM adjustments:
        - Linear/FFN: processes 2L tokens, so 2x per input token
        - Attention: (2L)^2 = 4x per input token  
        - Complementary mask: 2x batch, so 2x total FLOPs per input token
        """
        from torchtitan.models.utils import get_moe_model_nparams_and_flops
        
        # Get base nparams (this is correct for DLLM too)
        nparams, _ = get_moe_model_nparams_and_flops(
            self, model, 2 * self.head_dim, seq_len
        )
        
        # Calculate embedding params
        nparams_embedding = sum(
            sum(p.numel() for p in m.parameters())
            for m in model.modules()
            if isinstance(m, nn.Embedding)
        )
        
        # Linear/FFN FLOPs per token (base model)
        # Factor: 6 = 2 (mul+add) * 3 (fwd + 2*bwd)
        linear_flops_per_token = 6 * (nparams - nparams_embedding)
        
        # Attention FLOPs per token (base model)
        # Formula: 6 * n_layers * n_heads * head_dims * seq_len
        head_dims = 2 * self.head_dim  # Combined Q/K and V head dims
        attention_flops_per_token = 6 * self.n_layers * self.n_heads * head_dims * seq_len
        
        # DLLM scaling:
        # - Linear: 2x (processing 2L tokens through FFN)
        # - Attention: 4x (attending over (2L)^2 positions)
        dllm_flops_per_input_token = 2 * linear_flops_per_token + 4 * attention_flops_per_token
        
        # If complementary_mask is enabled, we process 2x the batch
        # This means 2x FLOPs per input token counted
        if self.complementary_mask:
            dllm_flops_per_input_token *= 2
        
        return nparams, dllm_flops_per_input_token


class DLLMAttention(Attention):
    """
    Attention module for DLLM that supports custom attention masks.
    
    Supports two attention backends:
    - FlexAttention with BlockMask (faster, optimized triton kernels)
    - SDPA with 4D attention mask (fallback)
    
    During block diffusion training, we pass a BlockMask or 4D attention mask
    instead of relying on the default causal mask.
    """
    
    def __init__(self, model_args):
        super().__init__(model_args)
        self.use_flex_attention = model_args.use_flex_attention
        if self.use_flex_attention:
            # Use compiled flex attention wrapper
            self.flex_attn = FlexAttentionWrapper()
    
    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: BlockMask | torch.Tensor | None,
        positions: torch.Tensor | None = None,
    ):
        """
        Forward pass with support for custom attention masks.
        
        Args:
            x: Input tensor, shape (B, L, D)
            rope_cache: Precomputed RoPE frequencies
            attention_masks: BlockMask for FlexAttention, 4D tensor (B, 1, L, L) for SDPA, or None for causal
            positions: Position indices for RoPE, shape (B, L) or None
        """
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        if self.q_norm:
            xq = self.q_norm(xq)
        if self.k_norm:
            xk = self.k_norm(xk)

        # Apply rotary embedding
        xq, xk = apply_rotary_emb(xq, xk, rope_cache, positions)

        # Repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)

        # Dispatch based on mask type
        if isinstance(attention_masks, BlockMask):
            # FlexAttention path - optimized for block-sparse patterns
            output = self.flex_attn(xq, xk, xv, block_mask=attention_masks, scale=self.scaling)
        elif attention_masks is not None:
            # SDPA path with explicit 4D mask
            # attention_masks is (B, 1, L, L) with -inf for masked positions
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=attention_masks,
                scale=self.scaling,
                is_causal=False,  # We provide explicit mask
            )
        else:
            # Default causal attention (inference)
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                scale=self.scaling,
                is_causal=True,
            )

        output = output.transpose(1, 2).contiguous()
        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class DLLMTransformerBlock(TransformerBlock):
    """Transformer block for DLLM with custom attention."""
    
    def __init__(self, layer_id: int, model_args: "DLLMModelArgs"):
        # Call grandparent init to skip TransformerBlock's attention creation
        nn.Module.__init__(self)
        
        from torchtitan.models.qwen3.model.model import FeedForward
        from torchtitan.models.moe import MoE
        
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.moe_enabled = model_args.moe_enabled
        
        # Use DLLM attention instead of standard attention
        self.attention = DLLMAttention(model_args)
        
        if model_args.moe_enabled:
            self.moe = MoE(
                model_args.moe_args,
                dim=model_args.dim,
                hidden_dim=model_args.moe_inter_dim,
            )
        else:
            self.feed_forward = FeedForward(
                dim=model_args.dim,
                hidden_dim=model_args.hidden_dim,
            )
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5

    def init_weights(self, buffer_device: torch.device | None = None):
        """Initialize layer weights."""
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(self.weight_init_std, buffer_device)
        else:
            self.feed_forward.init_weights(self.weight_init_std)


def block_diff_mask(
    b: torch.Tensor,
    h: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    block_size: int,
    n: int,
    causal_x0: bool = False,
    causal_xt: bool = False,
    doc_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Mask modifier function for block diffusion attention (FlexAttention compatible).

    The mask is composed of three components:
    - Block Diagonal Mask (M_BD): Self-attention within noised blocks (x_t attends to x_t in same block)
    - Offset Block Causal Mask (M_OBC): Cross-attention for conditional context (x_t attends to previous x_0 blocks)
    - Block Causal Mask (M_BC): Attention for x_0 (x_0 attends to x_0 causally)

    For a sequence [x_t; x_0] of length 2n where each half has length n:
    - Positions 0 to n-1: noisy tokens (x_t)
    - Positions n to 2n-1: clean tokens (x_0)

    Args:
        b: Batch index tensor (for FlexAttention compatibility, used with doc_ids)
        h: Head index tensor (unused, for FlexAttention compatibility)
        q_idx: Query position indices
        kv_idx: Key/value position indices
        block_size: Defines the block structure for diffusion
        n: Original sequence length (before concatenation)
        causal_x0: If True, x_0 uses token-level causal attention instead of block-causal
        causal_xt: If True, x_t uses causal attention within blocks instead of bidirectional
        doc_ids: Optional (B, 2n) tensor for per-document isolation when packing

    Returns:
        Boolean tensor - True where attention is ALLOWED
    """
    # Indicate whether token belongs to x_t (first half) or x_0 (second half)
    q_is_x0 = q_idx >= n
    kv_is_x0 = kv_idx >= n

    # Compute block indices (normalize positions to 0..n-1 range first)
    q_pos = torch.where(q_is_x0, q_idx - n, q_idx)
    kv_pos = torch.where(kv_is_x0, kv_idx - n, kv_idx)
    q_block = q_pos // block_size
    kv_block = kv_pos // block_size

    # M_OBC: x_t queries attend to previous x_0 blocks (unchanged)
    offset_block_causal = (~q_is_x0) & kv_is_x0 & (q_block > kv_block)

    # x_t mask: causal_xt selects causal attention; otherwise use bidirectional attention.
    if causal_xt:
        xt_mask = (q_block == kv_block) & (~q_is_x0) & (~kv_is_x0) & (q_pos >= kv_pos)
    else:
        xt_mask = (q_block == kv_block) & (~q_is_x0) & (~kv_is_x0)

    # x_0 mask: causal_x0 → strict token-level causal; else → block-causal
    if causal_x0:
        x0_mask = q_is_x0 & kv_is_x0 & (q_pos >= kv_pos)
    else:
        x0_mask = q_is_x0 & kv_is_x0 & (q_block >= kv_block)
        # Also need block-diagonal for x_0 within same block (original M_BD)
        x0_mask = x0_mask | ((q_block == kv_block) & q_is_x0 & kv_is_x0)

    mask = xt_mask | offset_block_causal | x0_mask

    # Document isolation: tokens from different docs cannot attend to each other
    if doc_ids is not None:
        same_doc = doc_ids[b, q_idx] == doc_ids[b, kv_idx]
        mask = mask & same_doc

    return mask


def create_block_diff_attention_mask(
    seq_len: int,
    block_size: int,
    device: torch.device,
    causal_x0: bool = False,
    causal_xt: bool = False,
    doc_ids: torch.Tensor | None = None,
) -> BlockMask:
    """
    Create a BlockMask for block diffusion attention using FlexAttention.

    Args:
        seq_len: Original sequence length (n, not 2n)
        block_size: Block size for diffusion masking
        device: Device to create the mask on
        causal_x0: If True, x_0 uses token-level causal instead of block-causal
        causal_xt: If True, x_t uses causal within blocks instead of bidirectional
        doc_ids: Optional (B, L) tensor for per-document isolation when packing

    Returns:
        BlockMask for use with FlexAttention
    """
    total_len = 2 * seq_len  # [x_t; x_0]

    if doc_ids is not None:
        extended = torch.cat([doc_ids, doc_ids], dim=1).to(device)  # (B, 2L)
        B = extended.shape[0]
    else:
        extended = None
        B = None

    mask_mod = partial(
        block_diff_mask, block_size=block_size, n=seq_len,
        causal_x0=causal_x0, causal_xt=causal_xt, doc_ids=extended,
    )

    block_mask = create_block_mask(
        mask_mod,
        B=B,
        H=None,
        Q_LEN=total_len,
        KV_LEN=total_len,
        device=device,
    )
    return block_mask


def create_block_diff_4d_mask(
    seq_len: int,
    block_size: int,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    causal_x0: bool = False,
    causal_xt: bool = False,
    doc_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Create a 4D attention mask for block diffusion (SDPA fallback).

    Args:
        seq_len: Original sequence length (n, not 2n)
        block_size: Block size for diffusion masking
        batch_size: Number of samples in the batch
        dtype: Data type for the mask (for -inf values)
        device: Device to create the mask on
        causal_x0: If True, x_0 uses token-level causal instead of block-causal
        causal_xt: If True, x_t uses causal within blocks instead of bidirectional
        doc_ids: Optional (B, L) tensor for per-document isolation when packing

    Returns:
        4D attention mask tensor of shape (B, 1, 2n, 2n) with -inf for masked positions
    """
    total_len = 2 * seq_len

    q_idx = torch.arange(total_len, device=device)
    kv_idx = torch.arange(total_len, device=device)
    q_mesh, kv_mesh = torch.meshgrid(q_idx, kv_idx, indexing='ij')

    b_dummy = torch.zeros_like(q_mesh)
    h_dummy = torch.zeros_like(q_mesh)
    # Base mask without doc_ids (computed once, same for all rows)
    mask = block_diff_mask(
        b_dummy, h_dummy, q_mesh, kv_mesh, block_size, seq_len,
        causal_x0=causal_x0, causal_xt=causal_xt,
    )

    attention_mask = torch.zeros((total_len, total_len), dtype=dtype, device=device)
    attention_mask.masked_fill_(~mask, torch.finfo(dtype).min)
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)

    # Apply per-row document isolation
    if doc_ids is not None:
        attention_mask = attention_mask.clone()  # un-expand for in-place ops
        extended = torch.cat([doc_ids, doc_ids], dim=1).to(device)  # (B, 2L)
        same_doc = extended.unsqueeze(2) == extended.unsqueeze(1)  # (B, 2L, 2L)
        same_doc = same_doc.unsqueeze(1)  # (B, 1, 2L, 2L)
        attention_mask.masked_fill_(~same_doc, torch.finfo(dtype).min)

    return attention_mask


def _doc_ids_to_positions(doc_ids: torch.Tensor) -> torch.Tensor:
    """Convert per-token document IDs to per-document position indices.

    Each document restarts positions from 0, matching what RoPE expects
    for independently packed SFT samples.

    Example:
        doc_ids = [0,0,0,0,0,0, 1,1,1,1, 1,1]
        output  = [0,1,2,3,4,5, 0,1,2,3, 4,5]
    """
    boundary = torch.cat([
        torch.ones(doc_ids.shape[0], 1, device=doc_ids.device, dtype=torch.long),
        (doc_ids[:, 1:] != doc_ids[:, :-1]).long(),
    ], dim=1)  # 1 at each doc start
    global_idx = torch.arange(
        doc_ids.shape[1], device=doc_ids.device
    ).unsqueeze(0).expand_as(doc_ids)
    first_occurrence = boundary * global_idx  # non-zero only at doc starts
    doc_start = torch.cummax(first_occurrence, dim=1).values  # propagate rightward
    return global_idx - doc_start


class DLLM(Qwen3Model):
    """
    DLLM: Qwen3 + Block Diffusion training support.
    
    Key features:
    1. Concatenates [x_t; x_0] for efficient training
    2. Uses block_diff_mask for proper attention patterns
    3. Independent RoPE for x_t and x_0 (both get positions 0 to n-1)
    """

    def __init__(self, model_args: DLLMModelArgs):
        # Call grandparent init to avoid creating standard layers
        nn.Module.__init__(self)
        
        from torchtitan.models.qwen3.model.model import precompute_rope_cache
        
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        
        # Use DLLM transformer blocks with custom attention
        self.layers = nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = DLLMTransformerBlock(layer_id, model_args)

        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        
        # Output projection - always create even with weight tying.
        # parallelize_qwen3 handles weight tying by assigning
        # model.output.weight = model.tok_embeddings.weight after TP setup.
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

        # RoPE cache - same size for both training and inference
        # During training: positions are [0..L-1, 0..L-1] (repeated, not extended)
        # During inference: positions are [0..L-1] (standard)
        # In both cases, max position index is L-1, so cache size = max_seq_len is sufficient
        self.register_buffer(
            "rope_cache",
            precompute_rope_cache(
                model_args.head_dim,
                model_args.max_seq_len,
                model_args.rope_theta,
            ),
            persistent=False,
        )

    def init_weights(self, buffer_device: torch.device | None = None):
        """Initialize model weights."""
        # Recompute rope_cache on the correct device (matching base Qwen3Model behavior)
        buffer_device = buffer_device or self.rope_cache.device
        with torch.device(buffer_device):
            self.rope_cache = self._precompute_rope_cache()
        
        # Initialize embeddings
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight, std=0.02)
        
        # Initialize transformer layers
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device)
        
        # Initialize norm
        if self.norm is not None:
            self.norm.reset_parameters()
        
        # Initialize output projection
        nn.init.normal_(self.output.weight, std=0.02)

    def get_attention_masks(self, input_batch, tokenizer, extra_inputs=None):
        """DLLM uses custom block_diff_mask, not standard causal mask."""
        return None

    def _sample_block_mask_probabilities(
        self,
        batch_size: int,
        num_blocks: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample one masking probability per block.

        Default path matches Fast-dLLM v2: iid U[0, 1] per block.

        When antithetic_sampling is enabled, use a permuted stratified sampler:
        we draw one point from each of N equal-width strata over [0, 1), where
        N = batch_size * num_blocks, and then randomly permute their assignment
        to block slots. This preserves the U[0, 1] marginal distribution for
        every block while removing the original slot-index bias.
        """
        block_probs = torch.rand((batch_size, num_blocks), device=device)
        if not self.model_args.antithetic_sampling:
            return block_probs

        flat_probs = block_probs.reshape(-1)
        num_samples = flat_probs.numel()
        strata_offsets = torch.arange(
            num_samples,
            device=device,
            dtype=flat_probs.dtype,
        )
        flat_probs = (flat_probs + strata_offsets) / num_samples
        permutation = torch.randperm(num_samples, device=device)
        return flat_probs[permutation].view(batch_size, num_blocks)

    def forward_diffusion(
        self,
        x0_embeds: torch.Tensor,
        labels: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, BlockMask | torch.Tensor]:
        """
        Apply block diffusion noise and prepare training inputs.
        
        When complementary_mask is enabled, generates both the original mask and its
        complement, effectively doubling the batch size from B to 2B.
        
        Args:
            x0_embeds: Clean token embeddings, shape (B, L, H)
            labels: Token labels, shape (B, L). Positions with -100 are padding
                    and will NOT be masked. None means all positions are valid.
            
        Returns:
            bd_inputs: Concatenated [x_t; x_0] embeddings, shape (B', 2L, H) where B'=2B if complementary
            masked_indices: Boolean mask of noised positions in x_t, shape (B', L)
            positions: Position ids for RoPE, shape (B', 2L) - [0..L-1, 0..L-1]
            attention_mask: BlockMask for FlexAttention or 4D tensor (B', 1, 2L, 2L) for SDPA
        """
        B, L, H = x0_embeds.shape
        device = x0_embeds.device
        dtype = x0_embeds.dtype
        block_size = self.model_args.block_size
        use_complementary = self.model_args.complementary_mask
        use_flex_attention = self.model_args.use_flex_attention
        
        if self.model_args.all_masked:
            # 100% masking: all non-padding tokens are masked.
            masked_indices = torch.ones(B, L, dtype=torch.bool, device=device)
            if labels is not None:
                masked_indices[labels == -100] = False
            # Complementary mask is meaningless with 100% masking
            use_complementary = False
        else:
            # Standard DLLM: random block-level masking
            num_blocks = (L + block_size - 1) // block_size
            p = self._sample_block_mask_probabilities(B, num_blocks, device)
            p = p.repeat_interleave(block_size, dim=-1)
            p = p[:, :L]
            move_probabilities = torch.rand(B, L, device=device)
            masked_indices = move_probabilities <= p
            if labels is not None:
                padding_mask = (labels == -100)
                masked_indices[padding_mask] = False
        
        # Get mask embedding: (1, 1, H) for broadcasting with (B, L, H)
        mask_embed = self.tok_embeddings(
            torch.tensor([self.model_args.mask_token_id], device=device)
        ).unsqueeze(0)  # (1,) -> embed -> (1, H) -> unsqueeze -> (1, 1, H)
        
        if use_complementary:
            # Create complementary mask (flip masked/unmasked positions)
            complementary_masked_indices = ~masked_indices
            # Also protect padding in complementary mask
            if labels is not None:
                complementary_masked_indices[padding_mask] = False
            
            # Concatenate original and complementary
            masked_indices = torch.cat([masked_indices, complementary_masked_indices], dim=0)  # (2B, L)
            x0_embeds = torch.cat([x0_embeds, x0_embeds], dim=0)  # (2B, L, H)
            if doc_ids is not None:
                doc_ids = torch.cat([doc_ids, doc_ids], dim=0)  # (2B, L)
            B = 2 * B  # Update batch size
        
        # Create noisy embeddings x_t
        xt_embeds = torch.where(
            masked_indices.unsqueeze(-1),
            mask_embed.expand(B, L, -1),
            x0_embeds
        )
        
        # Concatenate [x_t; x_0]
        bd_inputs = torch.cat([xt_embeds, x0_embeds], dim=1)  # (B, 2L, H)
        
        # Create position ids: per-document reset when packing, else global arange
        if doc_ids is not None:
            pos_ids = _doc_ids_to_positions(doc_ids)  # (B, L), per-doc 0-indexed
        else:
            pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        positions = torch.cat([pos_ids, pos_ids], dim=1)  # (B, 2L)

        # Create attention mask based on backend choice
        causal_x0 = self.model_args.causal_x0
        causal_xt = self.model_args.causal_xt
        if use_flex_attention:
            attention_mask = create_block_diff_attention_mask(
                seq_len=L,
                block_size=block_size,
                device=device,
                causal_x0=causal_x0,
                causal_xt=causal_xt,
                doc_ids=doc_ids,
            )
        else:
            attention_mask = create_block_diff_4d_mask(
                seq_len=L,
                block_size=block_size,
                batch_size=B,
                dtype=dtype,
                device=device,
                causal_x0=causal_x0,
                causal_xt=causal_xt,
                doc_ids=doc_ids,
            )
        
        return bd_inputs, masked_indices, positions, attention_mask

    def forward(self, tokens, attention_masks=None, positions=None, labels=None, doc_ids=None):
        """
        Forward pass with optional block diffusion training.
        
        During training (self.training is True):
        - Applies diffusion noise to create x_t
        - Concatenates [x_t; x_0] 
        - Uses block_diff_mask attention pattern
        - Returns dict with logits and training metadata
        - If complementary_mask is enabled, batch size is doubled (B -> 2B)
        
        During inference (self.training is False):
        - Standard forward pass with optional attention_masks and positions
        - Returns logits directly
        
        Args:
            tokens: Input token ids, shape (B, L)
            attention_masks: Optional attention mask (ignored during training, uses block_diff_mask)
            positions: Optional position ids (ignored during training, uses duplicated positions)
            labels: Optional labels, shape (B, L). Used during training to protect
                    padding positions (labels == -100) from being masked. Passed via
                    input_dict["labels"] from DLLMTextDataset. None for inference.
            
        Returns:
            Training: dict with "logits", "masked_indices"
                      If complementary_mask enabled, logits/masked_indices have batch size 2B
            Inference: logits tensor, shape (B, L, V)
        """
        # Get token embeddings
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens
        
        if self.training:
            # Block diffusion training mode
            # Note: forward_diffusion may double the batch if complementary_mask is enabled
            bd_inputs, masked_indices, bd_positions, bd_attention_mask = \
                self.forward_diffusion(h, labels=labels, doc_ids=doc_ids)
            
            # Forward through transformer layers with block diffusion setup
            h = bd_inputs
            for layer in self.layers.values():
                h = layer(h, self.rope_cache, bd_attention_mask, bd_positions)
            
            h = self.norm(h) if self.norm else h
            
            # Split hidden states into x_t and x_0 halves
            seq_len = tokens.shape[1]
            h_xt = h[:, :seq_len]
            h_x0 = h[:, seq_len:]

            use_fused = self.model_args.use_fused_ce
            ar_weight = self.model_args.ar_loss_weight
            logit_shift = self.model_args.logit_shift
            loss_auto_balance = self.model_args.loss_auto_balance
            loss_normalize = self.model_args.loss_normalize

            # Expand labels and doc_ids to match batch size (complementary mask doubles B)
            expanded_labels = labels
            expanded_doc_ids = doc_ids
            if labels is not None and labels.shape[0] != masked_indices.shape[0]:
                repeat_factor = masked_indices.shape[0] // labels.shape[0]
                expanded_labels = labels.repeat(repeat_factor, 1)
                if doc_ids is not None:
                    expanded_doc_ids = doc_ids.repeat(repeat_factor, 1)

            def _shift_labels(lbl):
                """Dream-style shift: labels[i] = original_labels[i+1]."""
                return F.pad(lbl, (0, 1), value=-100)[..., 1:].contiguous()

            def _fix_doc_boundary(shifted, d_ids):
                """Set shifted labels to -100 at document boundaries."""
                if d_ids is None:
                    return shifted
                boundary = d_ids[:, :-1] != d_ids[:, 1:]  # (B, L-1)
                shifted[:, :-1][boundary] = -100
                return shifted

            # Build diffusion labels (L_mask)
            if logit_shift:
                diff_labels = _shift_labels(expanded_labels)
                diff_labels = _fix_doc_boundary(diff_labels, expanded_doc_ids)
                diff_labels[~masked_indices] = -100
            else:
                diff_labels = expanded_labels.clone()
                diff_labels[~masked_indices] = -100

            if use_fused:
                from liger_kernel.transformers.functional import (
                    liger_fused_linear_cross_entropy,
                )

                # L_mask (fused: hidden → loss, no logits materialization)
                diff_loss = liger_fused_linear_cross_entropy(
                    h_xt.reshape(-1, h_xt.shape[-1]),
                    self.output.weight,
                    diff_labels.reshape(-1),
                    bias=self.output.bias,
                    ignore_index=-100,
                    reduction="mean",
                )

                # L_clean: AR NTP on x_0 (fused, original B only)
                ar_loss = None
                if ar_weight > 0:
                    B_orig = labels.shape[0]
                    h_x0_orig = h_x0[:B_orig]
                    ar_labels = _shift_labels(labels)  # always shifted
                    ar_labels = _fix_doc_boundary(ar_labels, doc_ids)  # always fix for packing
                    ar_loss = liger_fused_linear_cross_entropy(
                        h_x0_orig.reshape(-1, h_x0_orig.shape[-1]),
                        self.output.weight,
                        ar_labels.reshape(-1),
                        bias=self.output.bias,
                        ignore_index=-100,
                        reduction="mean",
                    )

                # Combine losses
                if ar_loss is not None:
                    if loss_auto_balance:
                        delta = diff_loss.detach() / (ar_loss.detach() + 1e-8)
                        total_loss = diff_loss + delta * ar_loss
                    elif loss_normalize:
                        total_loss = (diff_loss + ar_weight * ar_loss) / (1 + ar_weight)
                    else:
                        total_loss = diff_loss + ar_weight * ar_loss
                else:
                    total_loss = diff_loss

                return {"loss": total_loss, "masked_indices": masked_indices}

            else:
                # Non-fused path: materialize logits
                logits = self.output(h_xt)

                # L_clean on original B only (complementary half is redundant)
                ar_loss = None
                if ar_weight > 0:
                    B_orig = labels.shape[0]
                    h_x0_orig = h_x0[:B_orig]
                    ar_labels = _shift_labels(labels)  # always shifted
                    ar_labels = _fix_doc_boundary(ar_labels, doc_ids)  # always fix for packing
                    valid = ar_labels != -100
                    if valid.any():
                        logits_ar = self.output(h_x0_orig[valid])
                        ar_loss = F.cross_entropy(
                            logits_ar.float(), ar_labels[valid], reduction="mean"
                        )

                return {
                    "logits": logits,
                    "masked_indices": masked_indices,
                    "ar_loss": ar_loss,
                    "ar_loss_weight": ar_weight,
                    "loss_auto_balance": loss_auto_balance,
                    "loss_normalize": loss_normalize,
                }
        else:
            # Standard inference mode
            for layer in self.layers.values():
                h = layer(h, self.rope_cache, attention_masks, positions)
            
            h = self.norm(h) if self.norm else h
            logits = self.output(h)
            
            return logits

    @staticmethod
    def compute_loss(pred, labels) -> torch.Tensor:
        """
        Compute cross-entropy loss on masked positions only (no label shift).
        
        Same as compute_loss_fast_dllm except there is NO AR-style label shift:
        logits[i] predicts labels[i] (same position), not labels[i+1].
        
        1. Set labels to -100 at non-masked positions (only masked tokens
           contribute to loss).
        2. No shift — logits[i] predicts the token at position i.
        3. Compute CE with ignore_index=-100, reduction='mean' on valid positions.
        
        Note: The trainer passes the original labels (shape B, L) here. If the model
        doubled the batch via complementary_mask (logits shape 2B, L), we expand
        labels to match by repeating them.
        
        Args:
            pred: Model output - either logits tensor or dict with training metadata
            labels: Target token ids from dataloader, shape (B, L).
                    -100 for padding positions.
            
        Returns:
            Scalar loss tensor
        """
        # Fused CE path: loss already computed in forward
        if isinstance(pred, dict) and "loss" in pred:
            return pred["loss"]

        if isinstance(pred, dict):
            logits = pred["logits"]
            masked_indices = pred.get("masked_indices")
        else:
            logits = pred
            masked_indices = None

        # Expand labels to match batch size if forward doubled it (complementary masking)
        if logits.shape[0] != labels.shape[0]:
            repeat_factor = logits.shape[0] // labels.shape[0]
            labels = labels.repeat(repeat_factor, 1)

        # Set non-masked positions to -100 so they are ignored in loss.
        if masked_indices is not None:
            labels = labels.clone()
            labels[~masked_indices] = -100
        
        # Extract only valid (non-ignored) positions before CE to avoid
        # materializing the full (2B*L, V) float32 logits tensor.
        valid_mask = labels != -100
        num_valid = valid_mask.sum()
        
        if num_valid == 0:
            return (logits * 0).sum()
        
        logits_valid = logits[valid_mask]    # (num_valid, V)
        labels_valid = labels[valid_mask]    # (num_valid,)
        
        loss = F.cross_entropy(
            logits_valid.float(),
            labels_valid,
            reduction='mean',
        )

        # Combine with AR loss if present (non-fused path)
        if isinstance(pred, dict):
            ar_loss = pred.get("ar_loss")
            ar_weight = pred.get("ar_loss_weight", 0.0)
            auto_balance = pred.get("loss_auto_balance", False)
            normalize = pred.get("loss_normalize", True)
            if ar_loss is not None:
                if auto_balance:
                    delta = loss.detach() / (ar_loss.detach() + 1e-8)
                    loss = loss + delta * ar_loss
                elif normalize:
                    loss = (loss + ar_weight * ar_loss) / (1 + ar_weight)
                else:
                    loss = loss + ar_weight * ar_loss

        return loss

    @staticmethod
    def compute_loss_hybrid_diffusion_sft(pred, labels) -> torch.Tensor:
        """
        HybridDiffusion SFT loss: logit-shifted cross-entropy with optional dual loss.

        Like compute_loss but with Dream-style label shift (pos i predicts token i+1).
        Used by the qwen3_hybrid_diffusion_sft train spec.
        """
        # Fused path: loss already computed in forward
        if isinstance(pred, dict) and "loss" in pred:
            return pred["loss"]

        logits = pred["logits"] if isinstance(pred, dict) else pred
        masked_indices = pred.get("masked_indices") if isinstance(pred, dict) else None

        if logits.shape[0] != labels.shape[0]:
            labels = labels.repeat(logits.shape[0] // labels.shape[0], 1)

        # L_mask with logit shift: labels[i] = original_labels[i+1]
        labels = labels.clone()
        if masked_indices is not None:
            labels[~masked_indices] = -100
        labels = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()

        valid = labels != -100
        if valid.sum() == 0:
            return (logits * 0).sum()

        diff_loss = F.cross_entropy(logits[valid].float(), labels[valid], reduction="mean")

        # Combine with L_clean (AR loss)
        if isinstance(pred, dict):
            ar_loss = pred.get("ar_loss")
            ar_weight = pred.get("ar_loss_weight", 0.0)
            auto_balance = pred.get("loss_auto_balance", False)
            normalize = pred.get("loss_normalize", True)
            if ar_loss is not None:
                if auto_balance:
                    delta = diff_loss.detach() / (ar_loss.detach() + 1e-8)
                    return diff_loss + delta * ar_loss
                elif normalize:
                    return (diff_loss + ar_weight * ar_loss) / (1 + ar_weight)
                else:
                    return diff_loss + ar_weight * ar_loss

        return diff_loss

    @staticmethod
    def compute_loss_fast_dllm(pred, labels) -> torch.Tensor:
        """
        Fast-dLLM v2 style loss — identical to HuggingFace ForCausalLMLoss.
        
        Labels are ALIGNED with input_ids (no shift in the dataloader).
        The shift is applied here internally, matching HF's approach:
        
        1. Set labels to -100 at non-masked positions (only masked tokens
           contribute to loss, same as Fast-dLLM v2's `labels[mask] = -100`).
        2. Pad labels right with -100, then shift left by 1.
           This makes shift_labels[i] = labels[i+1], so logits[i] predicts
           the token at position i+1.
        3. Compute CE with ignore_index=-100, reduction='mean'.
        
        This ensures the loss is computed only where the TARGET position
        (i+1) is masked and has a valid label — exactly matching Fast-dLLM v2.
        
        Args:
            pred: Model output dict with "logits" and "masked_indices"
            labels: Target token ids from dataloader, shape (B, L), aligned
                    with input_ids (no shift). -100 for padding positions.
            
        Returns:
            Scalar loss tensor
        """
        if isinstance(pred, dict):
            logits = pred["logits"]
            masked_indices = pred.get("masked_indices")
        else:
            logits = pred
            masked_indices = None
        
        # Expand labels to match batch size if forward doubled it (complementary masking)
        if logits.shape[0] != labels.shape[0]:
            repeat_factor = logits.shape[0] // labels.shape[0]
            labels = labels.repeat(repeat_factor, 1)
        
        # Set non-masked positions to -100 so they are ignored in loss.
        # This replicates Fast-dLLM v2's: labels[noisy_input_ids != mask_id] = -100
        if masked_indices is not None:
            labels = labels.clone()
            labels[~masked_indices] = -100
        
        # HF ForCausalLMLoss style shift:
        # Pad labels right with -100, then take [..., 1:] to shift.
        # Result: shift_labels[i] = labels[i+1], shift_labels[-1] = -100
        # Logits stay at full length (B, L, V).
        labels = F.pad(labels, (0, 1), value=-100)
        shift_labels = labels[..., 1:].contiguous()
        
        # Extract only valid (non-ignored) positions before CE to avoid
        # materializing the full (2B*L, V) float32 logits tensor.
        # This matches the original compute_loss's sparse indexing strategy.
        valid_mask = shift_labels != -100
        num_valid = valid_mask.sum()
        
        if num_valid == 0:
            return (logits * 0).sum()
        
        logits_valid = logits[valid_mask]    # (num_valid, V)
        labels_valid = shift_labels[valid_mask]  # (num_valid,)
        
        loss = F.cross_entropy(
            logits_valid.float(),
            labels_valid,
            reduction='mean',
        )
        
        return loss
