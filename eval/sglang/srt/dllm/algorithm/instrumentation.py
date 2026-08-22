"""Route-neutral, observational request instrumentation.

The feature is opt-in via ``SGLANG_DLLM_REQUEST_METRICS=1``. Counters are
updated only at the operation they describe; unavailable measurements are
serialized as ``null`` with an entry in ``unavailable_metrics``.

Metric boundaries:

* ``prompt_tokens`` is the tokenized prompt length on the scheduler request.
* ``ar_tokens`` counts consumed causal clean/correction tokens. It is all
  output tokens for the causal route and normally zero for Diffusion-Trust.
* ``stable_tokens`` is the final, post stop/max-token output length.
* ``active_tokens`` counts logical token positions submitted to forwards once
  per forward, never once per layer.
* ``diffusion_steps`` counts actual denoise/draft model forwards.
* ``kv_cache_hits`` is the initial prompt-prefix hit length returned by
  ``match_prefix`` (device plus host hits, clamped to prompt length).
* ``gdn_state_restores`` counts successful state restore assignments in the
  GDN backend. It is not inferred from a flag or architecture layer count.
* ``invalidated_regions`` contains exact half-open logical ranges passed to a
  real trim/free operation. Recomputation is counted only when later forward
  positions intersect such a range, multiplied by actual model layers.
* ``total_latency_ms`` spans scheduler receipt to scheduler completion.
* CUDA timings are batch-shared for batch sizes above one. Peak memory is the
  process CUDA allocator peak reset at scheduler acceptance when the request is
  exclusive (Cluster-0 validation uses one running request).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


SCHEMA_VERSION = 2
ROUTES = frozenset({"causal", "ar_trust", "diffusion_trust"})
PHASES = (
    "prefill",
    "causal_decode",
    "diffusion_denoise",
    "self_spec_draft",
    "self_spec_verify",
    "self_spec_correction",
)
_MODEL_SCALE_RE = re.compile(r"(?:^|[-_/])([249])b(?:$|[-_/])", re.IGNORECASE)
logger = logging.getLogger(__name__)


def detect_model_scale(model_runner) -> str:
    """Resolve 2B/4B/9B once from explicit or named runtime configuration."""
    explicit = os.getenv("SGLANG_DLLM_MODEL_SCALE")
    if explicit is not None:
        normalized = explicit.strip().upper()
        return normalized if normalized in {"2B", "4B", "9B"} else "unknown"

    model_config = getattr(model_runner, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    candidates = (
        getattr(model_config, "model_path", None),
        getattr(model_config, "model_name", None),
        getattr(hf_config, "_name_or_path", None),
        getattr(hf_config, "name_or_path", None),
    )
    for candidate in candidates:
        if not candidate:
            continue
        match = _MODEL_SCALE_RE.search(str(candidate))
        if match is not None:
            return f"{match.group(1)}B"
    return "unknown"


@dataclass
class DllmRequestMetrics:
    request_id: str
    req_pool_idx: int
    selected_mode: str
    model_scale: str = "unknown"
    phase_counts: Dict[str, int] = field(
        default_factory=lambda: {phase: 0 for phase in PHASES}
    )
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
    mlp_time_ms: Optional[float] = None
    other_model_time_ms: Optional[float] = None
    model_forward_time_ms: Optional[float] = None
    verification_time_ms: float = 0.0
    total_latency_ms: Optional[float] = None
    peak_memory_bytes: Optional[int] = None
    component_timed_forwards: int = 0
    component_untimed_forwards: int = 0
    timing_scope: str = "unavailable"
    memory_scope: str = "unavailable"
    unavailable_metrics: Dict[str, str] = field(default_factory=dict)
    _pending_invalidated_positions: set[int] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        if self.selected_mode not in ROUTES:
            raise ValueError(f"Unknown request route: {self.selected_mode}")

    def record_forward(
        self,
        *,
        phases: Iterable[str] | str,
        active_tokens: int,
        diffusion_step: bool,
    ) -> None:
        if isinstance(phases, str):
            phases = (phases,)
        for phase in phases:
            if phase not in self.phase_counts:
                raise ValueError(f"Unknown instrumentation phase: {phase}")
            self.phase_counts[phase] += 1
        self.active_tokens += max(int(active_tokens), 0)
        self.diffusion_steps += int(bool(diffusion_step))

    def record_phase(self, phase: str) -> None:
        if phase not in self.phase_counts:
            raise ValueError(f"Unknown instrumentation phase: {phase}")
        self.phase_counts[phase] += 1

    def record_consumed_tokens(self, *, stable: int, ar: int) -> None:
        stable = max(int(stable), 0)
        ar = max(min(int(ar), stable), 0)
        self.stable_tokens += stable
        self.ar_tokens += ar

    def set_final_token_counts(self, *, stable: int, ar: Optional[int] = None) -> None:
        self.stable_tokens = max(int(stable), 0)
        if self.selected_mode == "causal":
            self.ar_tokens = self.stable_tokens
        elif ar is not None:
            self.ar_tokens = max(min(int(ar), self.stable_tokens), 0)
        else:
            self.ar_tokens = min(self.ar_tokens, self.stable_tokens)

    def record_invalidation(self, *, start: int, length: int) -> None:
        start = max(int(start), 0)
        length = max(int(length), 0)
        if length == 0:
            return
        end = start + length
        self.invalidated_regions.append({"start": start, "end": end, "length": length})
        self._pending_invalidated_positions.update(range(start, end))

    def record_revisited_positions(self, positions: Iterable[int], layers: int) -> None:
        layers = max(int(layers), 0)
        if layers == 0 or not self._pending_invalidated_positions:
            return
        revisited = self._pending_invalidated_positions.intersection(
            max(int(position), 0) for position in positions
        )
        self.recomputed_token_layer_positions += len(revisited) * layers
        self._pending_invalidated_positions.difference_update(revisited)

    def record_gdn_restore(self, count: int = 1) -> None:
        self.gdn_state_restores += max(int(count), 0)

    def add_forward_time(
        self,
        *,
        model_forward_ms: Optional[float],
        attention_ms: Optional[float],
        gdn_ms: Optional[float],
        mlp_ms: Optional[float],
        timing_scope: str,
        unavailable_reason: Optional[str] = None,
    ) -> None:
        if model_forward_ms is None:
            self.component_untimed_forwards += 1
            if unavailable_reason:
                self.unavailable_metrics.setdefault("cuda_timers", unavailable_reason)
            return

        forward = max(float(model_forward_ms), 0.0)
        attention = max(float(attention_ms or 0.0), 0.0)
        gdn = max(float(gdn_ms or 0.0), 0.0)
        mlp = max(float(mlp_ms or 0.0), 0.0)
        component_sum = attention + gdn + mlp
        if component_sum > forward and component_sum > 0:
            scale = forward / component_sum
            attention *= scale
            gdn *= scale
            mlp *= scale
            component_sum = forward
        other = max(forward - component_sum, 0.0)

        self.model_forward_time_ms = (self.model_forward_time_ms or 0.0) + forward
        self.attention_time_ms = (self.attention_time_ms or 0.0) + attention
        self.gdn_time_ms = (self.gdn_time_ms or 0.0) + gdn
        self.mlp_time_ms = (self.mlp_time_ms or 0.0) + mlp
        self.other_model_time_ms = (self.other_model_time_ms or 0.0) + other
        self.component_timed_forwards += 1
        self.timing_scope = timing_scope

    def to_record(self) -> dict:
        record = asdict(self)
        record.pop("_pending_invalidated_positions", None)
        record["schema_version"] = SCHEMA_VERSION

        total_forwards = self.component_timed_forwards + self.component_untimed_forwards
        record["forward_timer_coverage"] = (
            self.component_timed_forwards / total_forwards if total_forwards else None
        )
        if not total_forwards:
            record["unavailable_metrics"].setdefault(
                "forward_timer_coverage", "no_model_forward_recorded"
            )

        forward_ms = self.model_forward_time_ms
        if (
            forward_ms is not None
            and forward_ms > 0
            and self.component_timed_forwards > 0
        ):
            component_ms = sum(
                value or 0.0
                for value in (
                    self.attention_time_ms,
                    self.gdn_time_ms,
                    self.mlp_time_ms,
                    self.other_model_time_ms,
                )
            )
            record["model_component_coverage"] = min(component_ms / forward_ms, 1.0)
        else:
            record["model_component_coverage"] = None
            record["unavailable_metrics"].setdefault(
                "model_component_coverage",
                "model_forward_cuda_time_or_component_timers_unavailable",
            )

        if (
            forward_ms is not None
            and self.total_latency_ms is not None
            and self.total_latency_ms > 0
        ):
            record["model_forward_to_total_latency_ratio"] = (
                forward_ms / self.total_latency_ms
            )
        else:
            record["model_forward_to_total_latency_ratio"] = None
            record["unavailable_metrics"].setdefault(
                "model_forward_to_total_latency_ratio",
                "model_forward_or_total_latency_unavailable",
            )
        return record


def emit_metrics_record(metric: DllmRequestMetrics, *, tp_rank: int) -> Optional[dict]:
    """Log and append one finalized record from TP rank zero."""
    if int(tp_rank) != 0:
        return None
    record = metric.to_record()
    serialized = json.dumps(record, sort_keys=True, separators=(",", ":"))
    logger.info("[DLLM_REQUEST_METRICS] %s", serialized)

    if os.getenv("SGLANG_DLLM_REQUEST_METRICS_SUPPRESS_OUTPUT", "0") == "1":
        return record
    output_path = os.getenv("SGLANG_DLLM_REQUEST_METRICS_PATH")
    if output_path:
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "a", encoding="utf-8") as output_file:
                output_file.write(serialized + "\n")
                output_file.flush()
                os.fsync(output_file.fileno())
        except OSError as exc:
            logger.warning(
                "Unable to append request metrics to %s: %s", output_path, exc
            )
    return record
