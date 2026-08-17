# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        scaling: str,
        init: str,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.rank = rank
        self.alpha = alpha
        self.scaling = scaling
        self.init_mode = init
        self.lora_dropout = nn.Identity() if dropout == 0 else nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.lora_b = nn.Parameter(torch.empty(out_features, rank, device=device, dtype=dtype))
        self.reset_lora_parameters()

    @property
    def lora_scale(self) -> float:
        if self.scaling == "lora":
            return self.alpha / self.rank
        if self.scaling == "rs_lora":
            return self.alpha / math.sqrt(self.rank)
        raise ValueError(f"Unsupported LoRA scaling strategy: {self.scaling}")

    def reset_lora_parameters(self) -> None:
        if self.init_mode != "zero_start":
            raise ValueError(f"Unsupported LoRA init mode: {self.init_mode}")
        if self.lora_a.is_meta or self.lora_b.is_meta:
            return
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    @classmethod
    def from_linear(
        cls,
        base_linear: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        scaling: str,
        init: str,
    ) -> "LoRALinear":
        lora_linear = cls(
            in_features=base_linear.in_features,
            out_features=base_linear.out_features,
            bias=base_linear.bias is not None,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            scaling=scaling,
            init=init,
            device=base_linear.weight.device,
            dtype=base_linear.weight.dtype,
        )
        lora_linear.weight = base_linear.weight
        if base_linear.bias is None:
            lora_linear.register_parameter("bias", None)
        else:
            lora_linear.bias = base_linear.bias
        return lora_linear

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        base = F.linear(input, self.weight, self.bias)
        delta = F.linear(self.lora_dropout(input), self.lora_a)
        delta = F.linear(delta, self.lora_b)
        return base + (self.lora_scale * delta)
