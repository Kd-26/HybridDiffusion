"""Per-request instrumentation for the Qwen3.5-4B HybridDiffusion runtime.

Instrumentation is opt-in via ``SGLANG_DLLM_REQUEST_METRICS=1``.  A completed
request is emitted as one JSON record by the scheduler.  The counters use the
following definitions:

* ``ar_tokens``: finalized tokens produced by a causal clean/correction path.
* ``stable_tokens``: all tokens finalized into the response.
* ``active_tokens``: token positions submitted to model forwards (summed over
  forwards, but not layers).
* ``diffusion_steps``: denoising/draft model forwards.
* ``kv_cache_hits``: prompt tokens supplied by the prefix cache.
* ``gdn_state_restores``: request-layer GDN snapshot restores.
* ``invalidated_regions``: contiguous logical KV ranges discarded after a
  speculative/diffusion forward.
* ``recomputed_token_layer_positions``: invalidated or iteratively revisited
  token positions multiplied by the number of model layers.

The implementation records metadata only and never changes token selection,
cache ownership, or model inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


SCHEMA_VERSION = 1


@dataclass
class DllmRequestMetrics:
    request_id: str
    req_pool_idx: int
    selected_mode: str = "uninitialized"
    selected_mode_counts: Dict[str, int] = field(default_factory=dict)
    prompt_tokens: int = 0
    ar_tokens: int = 0
    stable_tokens: int = 0
    active_tokens: int = 0
    diffusion_steps: int = 0
    kv_cache_hits: int = 0
    gdn_state_restores: int = 0
    invalidated_regions: List[Dict[str, int]] = field(default_factory=list)
    recomputed_token_layer_positions: int = 0
    attention_time_ms: Optional[float] = None
    gdn_time_ms: Optional[float] = None
    verification_time_ms: float = 0.0
    total_latency_ms: Optional[float] = None
    peak_memory_bytes: int = 0
    component_timed_forwards: int = 0
    component_untimed_forwards: int = 0

    def select_mode(self, mode: str) -> None:
        self.selected_mode = mode
        self.selected_mode_counts[mode] = self.selected_mode_counts.get(mode, 0) + 1

    def record_forward(
        self,
        *,
        mode: str,
        active_tokens: int,
        diffusion_step: bool,
        gdn_state_restores: int,
        recomputed_token_layer_positions: int = 0,
    ) -> None:
        self.select_mode(mode)
        self.active_tokens += max(int(active_tokens), 0)
        self.diffusion_steps += int(bool(diffusion_step))
        self.gdn_state_restores += max(int(gdn_state_restores), 0)
        self.recomputed_token_layer_positions += max(
            int(recomputed_token_layer_positions), 0
        )

    def record_finalized_tokens(self, *, stable: int, ar: int) -> None:
        stable = max(int(stable), 0)
        ar = max(min(int(ar), stable), 0)
        self.stable_tokens += stable
        self.ar_tokens += ar

    def record_invalidation(
        self,
        *,
        start: int,
        length: int,
    ) -> None:
        length = max(int(length), 0)
        if length == 0:
            return
        self.invalidated_regions.append({"start": max(int(start), 0), "length": length})

    def add_component_time(
        self,
        *,
        attention_ms: Optional[float],
        gdn_ms: Optional[float],
    ) -> None:
        if attention_ms is None or gdn_ms is None:
            self.component_untimed_forwards += 1
            return
        self.attention_time_ms = (self.attention_time_ms or 0.0) + attention_ms
        self.gdn_time_ms = (self.gdn_time_ms or 0.0) + gdn_ms
        self.component_timed_forwards += 1

    def to_record(self) -> dict:
        record = asdict(self)
        record["schema_version"] = SCHEMA_VERSION
        record["model_scale"] = "4B"
        return record
