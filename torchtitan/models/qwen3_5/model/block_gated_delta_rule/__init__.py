# Copyright (c) 2026 Yuchen Zhu
# Block-Causal Gated Delta Rule
# Modified version with block-causal attention masking

from .chunk import chunk_block_causal_gated_delta_rule
from .fused_recurrent import fused_recurrent_block_causal_gated_delta_rule

__all__ = [
    'chunk_block_causal_gated_delta_rule',
    'fused_recurrent_block_causal_gated_delta_rule',
]
