# Copyright (c) 2026 Yuchen Zhu

"""Shared kernel-timing infrastructure for profiling.

Set ``_PROFILE_TIMINGS`` to a dict (e.g. ``{}``) to start collecting
CUDA event pairs.  Set it back to ``None`` to stop.

Set ``_DISABLE_CHECKPOINT`` to ``True`` to skip activation checkpointing
in block-train helpers (useful for fair benchmarking).
"""

import torch
from contextlib import contextmanager

_PROFILE_TIMINGS: dict | None = None
_DISABLE_CHECKPOINT: bool = False


@contextmanager
def _record(name: str):
    """Record a CUDA-timed section when profiling is enabled."""
    if _PROFILE_TIMINGS is None:
        yield
        return
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    yield
    end.record()
    _PROFILE_TIMINGS.setdefault(name, []).append((start, end))
