# -*- coding: utf-8 -*-
# Copyright (c) 2026 Yuchen Zhu
# Portions copyright (c) 2023-2025 Songlin Yang, Yu Zhang
# Modified from Flash Linear Attention (FLA) for HybridDiffusion block-causal training.

import os
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.utils import is_amd
from fla.modules.convolution import (
    causal_conv1d,
    causal_conv1d_update,
)

from . import _profiling as _prof
from .convolution_ops import (
    causal_conv1d_bwd as _block_conv1d_bwd,
    causal_conv1d_fwd as _block_conv1d_fwd,
)
from .block_short_conv_ops import (
    block_short_conv_twostream_bwd,
    block_short_conv_twostream_fwd,
)


NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if is_amd else [4, 8, 16, 32]
STATIC_WARPS = 32 if not is_amd else 16

try:
    from causal_conv1d import causal_conv1d_fn
    from causal_conv1d import causal_conv1d_update as causal_conv1d_update_cuda
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update_cuda = None


def _validate_block_train_packed_boundaries(
    cu_seqlens: torch.LongTensor | None,
    block_size: int,
) -> None:
    """Block-train packed docs may end partial, but doc starts must align."""
    if cu_seqlens is None or block_size <= 1:
        return
    starts = cu_seqlens[:-1]
    if starts.numel() == 0:
        return
    bad = torch.any(torch.remainder(starts, block_size) != 0)
    if bool(bad.item()):
        raise ValueError(
            "block_train conv with cu_seqlens requires document starts "
            f"(cu_seqlens[:-1]) to be multiples of block_size={block_size}. "
            "The final cu_seqlens[-1] may be a partial block."
        )


class BlockTrainConvFunction(torch.autograd.Function):
    """Fused block-train convolution that bypasses autograd for state preparation.

    Eliminates ~6 autograd nodes per conv call by directly calling the Triton
    conv1d forward/backward and manually scattering dh0 back to dx_clean.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        block_size: int,
        activation: str | None,
        cu_seqlens: torch.LongTensor | None = None,
    ):
        _validate_block_train_packed_boundaries(cu_seqlens, block_size)
        with _prof._record("block_conv_forward_total"):
            B, T, D = x.shape
            W = weight.shape[1]
            original_half_T = T // 2

            x_clean = x[:, :original_half_T, :]
            x_noisy = x[:, original_half_T:, :]

            pad_len = (-original_half_T) % block_size
            if pad_len > 0:
                x_clean = F.pad(x_clean, (0, 0, 0, pad_len))
                x_noisy = F.pad(x_noisy, (0, 0, 0, pad_len))

            half_T = original_half_T + pad_len
            num_blocks = half_T // block_size

            with _prof._record("block_conv_fwd_clean"):
                y_clean, _ = _block_conv1d_fwd(
                    x=x_clean, weight=weight, bias=bias, residual=None,
                    initial_state=None, output_final_state=False, activation=activation,
                    cu_seqlens=cu_seqlens,
                )

            initial_states = x.new_zeros(B * num_blocks, D, W)
            if W > 1 and num_blocks > 1:
                block_starts = torch.arange(1, num_blocks, device=x.device) * block_size
                offsets = torch.arange(-(W - 1), 0, device=x.device)
                positions = block_starts.unsqueeze(1) + offsets.unsqueeze(0)
                valid = positions >= 0
                positions = positions.clamp(min=0)

                tails = x_clean[:, positions, :]
                tails = tails * valid.unsqueeze(0).unsqueeze(-1)
                initial_states.view(B, num_blocks, D, W)[:, 1:, :, 1:] = tails.permute(0, 1, 3, 2)

            if cu_seqlens is not None:
                doc_firsts = (cu_seqlens[:-1] // block_size).long()
                initial_states.view(B, num_blocks, D, W)[:, doc_firsts] = 0.0

            effective_bs = max(block_size, W) if W > 1 else block_size
            if effective_bs > block_size:
                x_noisy_batched = F.pad(
                    x_noisy.reshape(B, num_blocks, block_size, D),
                    (0, 0, 0, effective_bs - block_size),
                ).reshape(B * num_blocks, effective_bs, D)
            else:
                x_noisy_batched = x_noisy.reshape(B * num_blocks, block_size, D)
            with _prof._record("block_conv_fwd_noisy"):
                y_noisy_batched, _ = _block_conv1d_fwd(
                    x=x_noisy_batched, weight=weight, bias=bias, residual=None,
                    initial_state=initial_states, output_final_state=False,
                    activation=activation,
                )
            if effective_bs > block_size:
                y_noisy = y_noisy_batched[:, :block_size].contiguous().reshape(B, half_T, D)
            else:
                y_noisy = y_noisy_batched.reshape(B, half_T, D)

            y_clean = y_clean[:, :original_half_T]
            y_noisy = y_noisy[:, :original_half_T]
            y = torch.cat([y_clean, y_noisy], dim=1)

            ctx.save_for_backward(x, weight, bias, initial_states)
            ctx.block_size = block_size
            ctx.activation = activation
            ctx.original_half_T = original_half_T
            ctx.pad_len = pad_len
            ctx.cu_seqlens = cu_seqlens
            return y

    @staticmethod
    def backward(ctx, dy):
        with _prof._record("block_conv_backward_total"):
            x, weight, bias, initial_states = ctx.saved_tensors
            block_size = ctx.block_size
            activation = ctx.activation
            original_half_T = ctx.original_half_T
            pad_len = ctx.pad_len
            cu_seqlens = ctx.cu_seqlens
            B, T, D = x.shape
            W = weight.shape[1]
            half_T = original_half_T + pad_len
            num_blocks = half_T // block_size

            x_clean = x[:, :original_half_T, :]
            x_noisy = x[:, original_half_T:, :]
            dy_clean = dy[:, :original_half_T, :].contiguous()
            dy_noisy = dy[:, original_half_T:, :].contiguous()

            if pad_len > 0:
                x_clean = F.pad(x_clean, (0, 0, 0, pad_len))
                x_noisy = F.pad(x_noisy, (0, 0, 0, pad_len))
                dy_clean = F.pad(dy_clean, (0, 0, 0, pad_len))
                dy_noisy = F.pad(dy_noisy, (0, 0, 0, pad_len))

            effective_bs = max(block_size, W) if W > 1 else block_size
            if effective_bs > block_size:
                x_noisy_batched = F.pad(
                    x_noisy.reshape(B, num_blocks, block_size, D),
                    (0, 0, 0, effective_bs - block_size),
                ).reshape(B * num_blocks, effective_bs, D)
                dy_noisy_batched = F.pad(
                    dy_noisy.reshape(B, num_blocks, block_size, D),
                    (0, 0, 0, effective_bs - block_size),
                ).reshape(B * num_blocks, effective_bs, D)
            else:
                x_noisy_batched = x_noisy.reshape(B * num_blocks, block_size, D)
                dy_noisy_batched = dy_noisy.reshape(B * num_blocks, block_size, D)

            with _prof._record("block_conv_bwd_noisy"):
                dx_noisy_batched, dw_noisy, db_noisy, _, dh0 = _block_conv1d_bwd(
                    x=x_noisy_batched, dy=dy_noisy_batched, dht=None,
                    weight=weight, bias=bias, residual=None,
                    initial_state=initial_states, activation=activation,
                )

            with _prof._record("block_conv_bwd_clean"):
                dx_clean, dw_clean, db_clean, _, _ = _block_conv1d_bwd(
                    x=x_clean, dy=dy_clean, dht=None,
                    weight=weight, bias=bias, residual=None,
                    initial_state=None, activation=activation,
                    cu_seqlens=cu_seqlens,
                )

            if dh0 is not None and W > 1 and num_blocks > 1:
                dh0_4d = dh0.view(B, num_blocks, D, W)
                if cu_seqlens is not None:
                    doc_firsts = (cu_seqlens[:-1] // block_size).long()
                    dh0_4d[:, doc_firsts] = 0.0
                dh0_tails = dh0_4d[:, 1:, :, 1:]
                dh0_vals = dh0_tails.permute(0, 1, 3, 2).to(dx_clean.dtype)

                block_starts = torch.arange(1, num_blocks, device=x.device) * block_size
                offsets = torch.arange(-(W - 1), 0, device=x.device)
                positions = block_starts.unsqueeze(1) + offsets.unsqueeze(0)
                valid = positions >= 0
                positions = positions.clamp(min=0)
                dh0_vals = dh0_vals * valid.unsqueeze(0).unsqueeze(-1)

                flat_pos = positions.reshape(-1)
                flat_dh0 = dh0_vals.reshape(B, -1, D)
                idx = flat_pos.unsqueeze(0).unsqueeze(-1).expand(B, -1, D)
                dx_clean.scatter_add_(1, idx, flat_dh0)

            if effective_bs > block_size:
                dx_noisy = dx_noisy_batched[:, :block_size].contiguous().reshape(B, half_T, D)
            else:
                dx_noisy = dx_noisy_batched.reshape(B, half_T, D)
            dx_clean = dx_clean[:, :original_half_T]
            dx_noisy = dx_noisy[:, :original_half_T]
            dx = torch.cat([dx_clean, dx_noisy], dim=1)
            dw = dw_clean + dw_noisy if dw_clean is not None else dw_noisy
            db = (db_clean + db_noisy) if db_clean is not None and bias is not None else None

            return dx, dw, db, None, None, None


class BlockTrainConvTwoStreamFunction(torch.autograd.Function):
    """Block-train convolution with a direct two-stream noisy path."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        block_size: int,
        activation: str | None,
        cu_seqlens: torch.LongTensor | None = None,
        BT: int = 64,
        BD: int = 64,
    ):
        _validate_block_train_packed_boundaries(cu_seqlens, block_size)
        with _prof._record("block_conv_forward_total"):
            _, T, _ = x.shape
            original_half_T = T // 2
            x_clean = x[:, :original_half_T, :]
            x_noisy = x[:, original_half_T:, :]

            pad_len = (-original_half_T) % block_size
            if pad_len > 0:
                x_clean = F.pad(x_clean, (0, 0, 0, pad_len))
                x_noisy = F.pad(x_noisy, (0, 0, 0, pad_len))

            with _prof._record("block_conv_fwd_clean"):
                y_clean, _ = _block_conv1d_fwd(
                    x=x_clean, weight=weight, bias=bias, residual=None,
                    initial_state=None, output_final_state=False, activation=activation,
                    cu_seqlens=cu_seqlens,
                )
            y_noisy = block_short_conv_twostream_fwd(
                x_clean=x_clean,
                x_noisy=x_noisy,
                weight=weight,
                bias=bias,
                block_size=block_size,
                activation=activation,
                cu_seqlens=cu_seqlens,
                BT=BT,
                BD=BD,
            )

            y = torch.cat([
                y_clean[:, :original_half_T],
                y_noisy[:, :original_half_T],
            ], dim=1)

            ctx.save_for_backward(x, weight, bias)
            ctx.block_size = block_size
            ctx.activation = activation
            ctx.original_half_T = original_half_T
            ctx.pad_len = pad_len
            ctx.cu_seqlens = cu_seqlens
            ctx.BT = BT
            ctx.BD = BD
            return y

    @staticmethod
    def backward(ctx, dy):
        with _prof._record("block_conv_backward_total"):
            x, weight, bias = ctx.saved_tensors
            block_size = ctx.block_size
            activation = ctx.activation
            original_half_T = ctx.original_half_T
            pad_len = ctx.pad_len
            cu_seqlens = ctx.cu_seqlens
            BT = ctx.BT
            BD = ctx.BD

            x_clean = x[:, :original_half_T, :]
            x_noisy = x[:, original_half_T:, :]
            dy_clean = dy[:, :original_half_T, :].contiguous()
            dy_noisy = dy[:, original_half_T:, :].contiguous()
            if pad_len > 0:
                x_clean = F.pad(x_clean, (0, 0, 0, pad_len))
                x_noisy = F.pad(x_noisy, (0, 0, 0, pad_len))
                dy_clean = F.pad(dy_clean, (0, 0, 0, pad_len))
                dy_noisy = F.pad(dy_noisy, (0, 0, 0, pad_len))

            dx_clean_from_noisy, dx_noisy, dw_noisy, db_noisy = block_short_conv_twostream_bwd(
                x_clean=x_clean,
                x_noisy=x_noisy,
                dy_noisy=dy_noisy,
                weight=weight,
                bias=bias,
                block_size=block_size,
                activation=activation,
                cu_seqlens=cu_seqlens,
                BT=BT,
                BD=BD,
            )

            with _prof._record("block_conv_bwd_clean"):
                dx_clean, dw_clean, db_clean, _, _ = _block_conv1d_bwd(
                    x=x_clean, dy=dy_clean, dht=None,
                    weight=weight, bias=bias, residual=None,
                    initial_state=None, activation=activation,
                    cu_seqlens=cu_seqlens,
                )

            dx_clean = dx_clean + dx_clean_from_noisy
            dx_clean = dx_clean[:, :original_half_T]
            dx_noisy = dx_noisy[:, :original_half_T]
            dx = torch.cat([dx_clean, dx_noisy], dim=1)
            dw = dw_clean + dw_noisy if dw_clean is not None else dw_noisy
            db = (db_clean + db_noisy) if db_clean is not None and bias is not None else None
            return dx, dw, db, None, None, None, None, None


def _resolve_block_train_conv_method(
    method: str | None,
    block_size: int,
    kernel_size: int,
    activation: str | None,
    x: torch.Tensor,
) -> str:
    method = os.environ.get("QWEN35_BLOCK_CONV_METHOD", method or "auto")
    if method == "auto":
        if (
            x.is_cuda
            and kernel_size <= 4
            and activation in (None, "silu", "swish")
        ):
            return "twostream"
        return "fla_batched"
    if method not in ("fla_batched", "twostream"):
        raise ValueError(
            f"Unknown block_train_conv_method: {method!r}. "
            "Use 'auto', 'fla_batched', or 'twostream'."
        )
    return method


def block_train_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    block_size: int,
    activation: str | None,
    cu_seqlens: torch.LongTensor | None = None,
    method: str | None = "auto",
    BT: int = 64,
    BD: int = 64,
) -> torch.Tensor:
    """Dispatch block-train ShortConv.

    Auto mode uses the two-stream kernels on CUDA when the convolution width is
    at most 4 and the activation is None, SiLU, or Swish. Other shapes use the
    FLA-batched fallback, which also remains available as an explicit override.
    """
    _validate_block_train_packed_boundaries(cu_seqlens, block_size)
    resolved = _resolve_block_train_conv_method(
        method=method,
        block_size=block_size,
        kernel_size=weight.shape[1],
        activation=activation,
        x=x,
    )
    if resolved == "twostream":
        return BlockTrainConvTwoStreamFunction.apply(
            x, weight, bias, block_size, activation, cu_seqlens, BT, BD,
        )
    return BlockTrainConvFunction.apply(
        x, weight, bias, block_size, activation, cu_seqlens,
    )


class ShortConvolution(nn.Conv1d):
    """Short convolution layer for efficient causal convolution operations.

    This class implements a depthwise separable 1D convolution with causal padding,
    designed for efficient sequence processing. It supports multiple backends (Triton/CUDA)
    and optional activation functions.

    Args:
        hidden_size (int): Number of input/output channels (must be equal for depthwise conv)
        kernel_size (int): Size of the convolution kernel
        bias (bool, optional): Whether to include learnable bias. Defaults to False.
        activation (Optional[str], optional): Activation function ('silu' or 'swish'). Defaults to 'silu'.
        backend (Optional[str], optional): Backend implementation ('triton' or 'cuda'). Defaults to 'triton'.
        device (Optional[torch.device], optional): Device to place the layer on. Defaults to None.
        dtype (Optional[torch.dtype], optional): Data type for layer parameters. Defaults to None.
        **kwargs: Additional keyword arguments (deprecated 'use_fast_conv1d' supported for compatibility)

    Attributes:
        hidden_size (int): Number of channels
        activation (Optional[str]): Selected activation function
        backend (str): Actual backend being used (may differ from input due to availability)

    Note:
        - Uses depthwise convolution (groups=hidden_size) for efficiency
        - Applies causal padding (kernel_size-1) to ensure no future information leakage
        - Falls back to Triton backend if CUDA backend is unavailable
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: Optional[str] = 'silu',
        backend: Optional[str] = 'triton',
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
            device=device,
            dtype=dtype,
        )

        self.hidden_size = hidden_size
        self.activation = None

        if activation is not None:
            assert activation in ['silu', 'swish'], f"Activation `{activation}` not supported yet."
            self.activation = activation

        if 'use_fast_conv1d' in kwargs:
            warnings.warn(
                "The `use_fast_conv1d` parameter is deprecated and will be ignored. "
                "Please use the `backend` parameter instead."
            )
        import os
        self.backend = os.environ.get('FLA_CONV_BACKEND', backend)
        if backend not in ['cuda', 'triton']:
            raise ValueError(f"Invalid backend: {backend}, must be one of ['cuda', 'triton']")
        if backend == 'cuda':
            if causal_conv1d_fn is None:
                warnings.warn(
                    "The `backend` parameter is set to `cuda`, but `causal_conv1d_fn` is not available. "
                    "Switching to the Triton implementation instead. "
                    "Consider installing `causal_conv1d` to enable the CUDA backend."
                )
                self.backend = 'triton'

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        if self.activation is not None:
            s += ', activation={activation}'
        s += f', backend={self.backend}'
        return s.format(**self.__dict__)

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
        block_train: bool = False,
        block_size: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (`torch.Tensor`):
                Tensor of shape `[B, T, D]`. `B` must be 1 if `cu_seqlens` is provided.
            residual (`Optional[torch.Tensor]`):
                Residual tensor of shape `[B, T, D]`. Default: `None`.
            mask (`Optional[torch.Tensor]`):
                Attention mask dealing with padded positions.
            cache (`Optional[torch.Tensor]`):
                Previous cache tensor of shape `[N, D, W]`, where `W` is the kernel size.
                If provided, the cache is updated **inplace**.
            output_final_state (Optional[bool]):
                Whether to output the final state of shape `[N, D, W]`. Default: `False`.
            cu_seqlens (Optional[torch.LongTensor]):
                Cumulative sequence lengths for each batch. Used for varlen. Default: `None`.
                Shape: [B+1]
            block_train (bool):
                If True, enables HybridDiffusion clean/noisy two-stream training.
                Input x should be [B, 2*N*block_size, D] where first half is clean tokens
                (X_1, ..., X_N) and second half is noisy tokens (hat_X_1, ..., hat_X_N).
                The noisy tokens will see their preceding clean tokens:
                - hat_X_1: sees only itself
                - hat_X_2: sees X_1, then itself
                - hat_X_N: sees X_1, ..., X_{N-1}, then itself
                Default: `False`.
            block_size (Optional[int]):
                Size of each block when block_train=True. Required if block_train=True.
                The number of blocks N is computed as T // (2 * block_size).

        Returns:
            Tensor of shape `[B, T, D]`.
            If block_train=True, returns `[B, 2*N*block_size, D]` with conv applied appropriately.
        """

        B, T, *_ = x.shape
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        
        if block_train:
            return self._forward_block_train(
                x=x,
                residual=residual,
                block_size=block_size,
                cu_seqlens=cu_seqlens,
                **kwargs
            )
        
        if mask is not None:
            if cu_seqlens is not None:
                raise ValueError("`mask` and `cu_seqlens` cannot be provided at the same time")
            x = x.mul_(mask.unsqueeze(-1))

        # in decoding phase, the cache (if provided) is updated inplace
        if B * T == N:
            y, cache = self.step(
                x=x,
                residual=residual,
                cache=cache,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens
            )
            return y, cache

        # cuda backend do not support:
        # 1. both `cu_seqlens` and `cache` being provided
        # 2. both `cu_seqlens` and `output_final_state` being provided
        if self.backend == 'cuda' and (
            (cu_seqlens is not None and cache is not None) or
            (cu_seqlens is not None and output_final_state)
        ):
            warnings.warn(
                "The CUDA backend does not support both `cu_seqlens` and `cache` being provided, "
                "or both `cu_seqlens` and `output_final_state` being provided. "
                "Switching to the Triton backend instead. ",
                stacklevel=2
            )
            self.backend = 'triton'

        return causal_conv1d(
            x=x,
            weight=rearrange(self.weight, "d 1 w -> d w"),
            bias=self.bias,
            residual=residual,
            initial_state=cache,
            output_final_state=output_final_state,
            activation=self.activation,
            backend=self.backend,
            cu_seqlens=cu_seqlens,
            **kwargs
        )

    def _forward_block_train(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        block_size: Optional[int] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        """Efficient clean/noisy two-stream ShortConv for HybridDiffusion training.

        Uses BlockTrainConvFunction to bypass autograd overhead for
        state preparation (~6 fewer autograd nodes per conv call).
        """
        B, T, D = x.shape
        W = self.kernel_size[0]

        if block_size is None:
            raise ValueError("`block_size` must be provided when block_train=True")

        weight = rearrange(self.weight, "d 1 w -> d w")

        if residual is not None:
            x = x + residual

        y = block_train_conv(
            x=x,
            weight=weight,
            bias=self.bias,
            block_size=block_size,
            activation=self.activation,
            cu_seqlens=cu_seqlens,
            method=kwargs.get("block_train_conv_method", "auto"),
            BT=kwargs.get("block_train_conv_BT", 64),
            BD=kwargs.get("block_train_conv_BD", 64),
        )
        return y, None

    def step(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        cache: torch.Tensor,
        output_final_state: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None
    ):
        B, _, D, W = *x.shape, self.kernel_size[0]
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        if output_final_state and cache is None:
            cache = x.new_zeros(N, D, W)
        # NOTE: we follow the fast mode that updates the cache in-place
        if self.backend == 'triton':
            return causal_conv1d_update(
                x=x,
                cache=cache,
                residual=residual,
                weight=rearrange(self.weight, "d 1 w -> d w"),
                bias=self.bias,
                activation=self.activation,
            )

        shape = x.shape
        x = x.squeeze(0) if cu_seqlens is not None else x.squeeze(1)
        # equivalent to:
        # cache.copy_(cache.roll(shifts=-1, dims=-1))
        # cache[:, :, -1] = x
        # y = torch.sum(cache * rearrange(self.weight, "d 1 w -> d w"), dim=-1)
        y = causal_conv1d_update_cuda(
            x=x,
            conv_state=cache,
            weight=rearrange(self.weight, "d 1 w -> d w"),
            bias=self.bias,
            activation=self.activation,
        )
        y = y.view(shape)
        if residual is not None:
            y.add_(residual)
        return y, cache

    @property
    def state_size(self) -> int:
        return self.hidden_size * self.kernel_size
