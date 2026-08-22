import copy
import os
from typing import Iterable, List, Optional, Sequence

import torch

from sglang.srt.dllm.algorithm import get_algorithm
from sglang.srt.dllm.algorithm.instrumentation import (
    DllmRequestMetrics,
    detect_model_scale,
)
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.server_args import ServerArgs


class RequestMetricsRecorder:
    """Common forward instrumentation shared by causal and dLLM routes."""

    def __init__(self, selected_mode: str):
        self.selected_mode = selected_mode
        self._request_metrics: dict[int, DllmRequestMetrics] = {}
        self._enabled: Optional[bool] = None
        self._model_scale: Optional[str] = None
        self._instrumented_model_id: Optional[int] = None
        self._hooks = []
        self._active_events = None
        self._pending_timings = []
        self._memory_started: set[int] = set()
        self._memory_started_request_ids: set[str] = set()

    def enabled(self, model_runner=None) -> bool:
        if self._enabled is None:
            self._enabled = os.getenv("SGLANG_DLLM_REQUEST_METRICS", "0") == "1"
        if self._enabled and model_runner is not None:
            if self._model_scale is None:
                self._model_scale = detect_model_scale(model_runner)
            self._install_timing_hooks(model_runner)
        return bool(self._enabled)

    @staticmethod
    def _batch_rpx(forward_batch) -> List[int]:
        rpx = getattr(forward_batch, "dllm_rpx_cpu", None)
        if rpx is None:
            rpx = forward_batch.req_pool_indices
            if torch.is_tensor(rpx):
                rpx = rpx.detach().cpu().tolist()
        return [int(value) for value in rpx[: forward_batch.batch_size]]

    @staticmethod
    def _batch_token_counts(forward_batch) -> List[int]:
        counts = getattr(forward_batch, "dllm_request_token_counts", None)
        if counts is None:
            counts = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if counts is not None:
            counts = [max(int(value), 0) for value in counts]
            if len(counts) == forward_batch.batch_size:
                return counts
        if forward_batch.batch_size <= 0:
            return []
        if forward_batch.forward_mode.is_decode():
            return [1] * forward_batch.batch_size
        total = int(forward_batch.input_ids.numel())
        if total % forward_batch.batch_size == 0:
            return [total // forward_batch.batch_size] * forward_batch.batch_size
        return [total] + [0] * (forward_batch.batch_size - 1)

    def _metrics_for_batch(self, forward_batch) -> List[DllmRequestMetrics]:
        rpx_list = self._batch_rpx(forward_batch)
        rids = getattr(forward_batch, "rids", None) or []
        metrics = []
        for bid, rpx in enumerate(rpx_list):
            rid = str(rids[bid]) if bid < len(rids) else f"req_pool_idx:{rpx}"
            metric = self._request_metrics.get(rpx)
            if metric is None or metric.request_id != rid:
                # Replacing a mismatched request ID prevents pool-slot state leaks.
                metric = DllmRequestMetrics(
                    request_id=rid,
                    req_pool_idx=rpx,
                    selected_mode=self.selected_mode,
                    model_scale=self._model_scale or "unknown",
                )
                self._request_metrics[rpx] = metric
                self._memory_started.discard(rpx)
            metrics.append(metric)
        return metrics

    @staticmethod
    def _expand_per_request(value, batch_size: int):
        if isinstance(value, str) or not isinstance(value, Sequence):
            return [value] * batch_size
        if len(value) != batch_size:
            raise ValueError(
                f"Expected {batch_size} instrumentation values, got {len(value)}"
            )
        return list(value)

    def _install_timing_hooks(self, model_runner) -> None:
        if not torch.cuda.is_available():
            return
        model = getattr(model_runner, "model", None)
        if model is None or self._instrumented_model_id == id(model):
            return

        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        self._instrumented_model_id = id(model)

        def make_hooks(kind: str):
            key = f"_request_metrics_{kind}_start_events"

            def pre_hook(module, _args):
                if self._active_events is None:
                    return
                start = torch.cuda.Event(enable_timing=True)
                start.record()
                module.__dict__.setdefault(key, []).append(start)

            def post_hook(module, _args, _output):
                starts = module.__dict__.get(key)
                start = starts.pop() if starts else None
                if self._active_events is None or start is None:
                    return
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                self._active_events.append((kind, start, end))

            return pre_hook, post_hook

        targets = []
        for module in model.modules():
            class_name = type(module).__name__
            if "AttentionDecoderLayer" in class_name:
                targets.append((module, "attention_total"))
                mlp = getattr(module, "mlp", None)
                if mlp is not None:
                    targets.append((mlp, "attention_mlp"))
                    targets.append((mlp, "mlp"))
            elif class_name == "Qwen3_5GatedDeltaNet":
                targets.append((module, "gdn"))
            elif "DecoderLayer" in class_name:
                mlp = getattr(module, "mlp", None)
                if mlp is not None:
                    targets.append((mlp, "mlp"))

        registered = set()
        for module, kind in targets:
            key = (id(module), kind)
            if key in registered:
                continue
            registered.add(key)
            pre_hook, post_hook = make_hooks(kind)
            self._hooks.append(module.register_forward_pre_hook(pre_hook))
            self._hooks.append(module.register_forward_hook(post_hook))

    @staticmethod
    def _model_layer_count(model_runner) -> int:
        model_config = getattr(model_runner, "model_config", None)
        text_config = getattr(model_config, "hf_text_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        return max(
            int(
                getattr(text_config, "num_hidden_layers", None)
                or getattr(hf_config, "num_hidden_layers", 0)
                or 0
            ),
            0,
        )

    @staticmethod
    def _positions_by_request(forward_batch, counts: List[int]) -> List[List[int]]:
        positions = getattr(forward_batch, "positions", None)
        if positions is None:
            return [[] for _ in counts]
        if torch.is_tensor(positions):
            positions = positions.detach().cpu().tolist()
        positions = list(positions)
        result = []
        offset = 0
        for count in counts:
            result.append([int(value) for value in positions[offset : offset + count]])
            offset += count
        return result

    def _start_memory_measurement(self, metrics, forward_batch) -> None:
        if not torch.cuda.is_available():
            for metric in metrics:
                metric.unavailable_metrics.setdefault(
                    "peak_memory_bytes", "cuda_unavailable"
                )
            return
        if len(metrics) != 1:
            for metric in metrics:
                metric.unavailable_metrics.setdefault(
                    "peak_memory_bytes", "per_request_attribution_unavailable_for_batch"
                )
                metric.memory_scope = "unavailable_batch_shared"
            return
        metric = metrics[0]
        if metric.req_pool_idx in self._memory_started:
            return
        if metric.request_id in self._memory_started_request_ids:
            self._memory_started.add(metric.req_pool_idx)
            metric.memory_scope = "process_peak_since_request_start"
            return
        metric.memory_scope = "unavailable_nonexclusive_request"
        metric.unavailable_metrics.setdefault(
            "peak_memory_bytes", "process_peak_cannot_be_attributed_to_this_request"
        )

    def start_request_memory(
        self, request_id: str, *, device, exclusive_request: bool
    ) -> None:
        """Reset the process peak at scheduler acceptance when attribution is valid."""
        if not self.enabled() or not torch.cuda.is_available() or not exclusive_request:
            return
        torch.cuda.reset_peak_memory_stats(device)
        self._memory_started_request_ids.add(str(request_id))

    def forward(
        self,
        model_runner,
        forward_batch,
        *,
        phases,
        diffusion_steps=False,
        forward_call=None,
    ):
        """Execute an unchanged forward while observing its inputs and CUDA work."""
        metrics = self._metrics_for_batch(forward_batch)
        batch_size = len(metrics)
        phase_list = self._expand_per_request(phases, batch_size)
        diffusion_list = self._expand_per_request(diffusion_steps, batch_size)
        token_counts = self._batch_token_counts(forward_batch)
        positions = self._positions_by_request(forward_batch, token_counts)
        layers = self._model_layer_count(model_runner)

        self._start_memory_measurement(metrics, forward_batch)
        for bid, metric in enumerate(metrics):
            metric.record_forward(
                phases=phase_list[bid],
                active_tokens=token_counts[bid],
                diffusion_step=bool(diffusion_list[bid]),
            )
            metric.record_revisited_positions(positions[bid], layers)

        old_restore_callback = getattr(
            forward_batch, "dllm_metrics_gdn_restore_callback", None
        )

        def record_restore(batch_indices):
            for bid in batch_indices:
                if 0 <= int(bid) < len(metrics):
                    metrics[int(bid)].record_gdn_restore()

        forward_batch.dllm_metrics_gdn_restore_callback = record_restore
        collect_cuda = torch.cuda.is_available()
        outer_start = outer_end = None
        if collect_cuda:
            self._active_events = []
            outer_start = torch.cuda.Event(enable_timing=True)
            outer_end = torch.cuda.Event(enable_timing=True)
            outer_start.record()
        try:
            if forward_call is None:
                out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            else:
                out = forward_call()
            if collect_cuda:
                outer_end.record()
                self._pending_timings.append(
                    (list(metrics), self._active_events, outer_start, outer_end)
                )
            else:
                for metric in metrics:
                    metric.add_forward_time(
                        model_forward_ms=None,
                        attention_ms=None,
                        gdn_ms=None,
                        mlp_ms=None,
                        timing_scope="unavailable",
                        unavailable_reason="cuda_unavailable",
                    )
            return out
        except Exception:
            for metric in metrics:
                self.cleanup_request(metric.req_pool_idx)
            raise
        finally:
            self._active_events = None
            if old_restore_callback is None:
                try:
                    delattr(forward_batch, "dllm_metrics_gdn_restore_callback")
                except AttributeError:
                    pass
            else:
                forward_batch.dllm_metrics_gdn_restore_callback = old_restore_callback

    def flush_forward_timings(self) -> None:
        pending, self._pending_timings = self._pending_timings, []
        for metrics, events, outer_start, outer_end in pending:
            outer_end.synchronize()
            forward_ms = float(outer_start.elapsed_time(outer_end))
            if not events:
                peak = int(torch.cuda.max_memory_allocated())
                for metric in metrics:
                    metric.model_forward_time_ms = (
                        metric.model_forward_time_ms or 0.0
                    ) + max(forward_ms, 0.0)
                    metric.component_untimed_forwards += 1
                    metric.timing_scope = "cuda_events_outer_only"
                    metric.unavailable_metrics.setdefault(
                        "model_components", "module_hooks_not_observed"
                    )
                    if (
                        len(metrics) == 1
                        and metric.memory_scope == "process_peak_since_request_start"
                    ):
                        metric.peak_memory_bytes = max(
                            metric.peak_memory_bytes or 0, peak
                        )
                continue
            totals = {
                "attention_total": 0.0,
                "attention_mlp": 0.0,
                "gdn": 0.0,
                "mlp": 0.0,
            }
            for kind, start, end in events:
                totals[kind] += max(float(start.elapsed_time(end)), 0.0)
            attention_ms = max(totals["attention_total"] - totals["attention_mlp"], 0.0)
            scope = (
                "cuda_events_request"
                if len(metrics) == 1
                else "cuda_events_batch_shared"
            )
            peak = int(torch.cuda.max_memory_allocated())
            for metric in metrics:
                metric.add_forward_time(
                    model_forward_ms=forward_ms,
                    attention_ms=attention_ms,
                    gdn_ms=totals["gdn"],
                    mlp_ms=totals["mlp"],
                    timing_scope=scope,
                )
                if (
                    len(metrics) == 1
                    and metric.memory_scope == "process_peak_since_request_start"
                ):
                    metric.peak_memory_bytes = max(metric.peak_memory_bytes or 0, peak)

    def record_invalidation(
        self, req_pool_idx: int, *, start: int, length: int
    ) -> None:
        metric = self._request_metrics.get(int(req_pool_idx))
        if metric is not None:
            metric.record_invalidation(start=start, length=length)

    def record_verification_time(
        self, forward_batch, batch_indices: Iterable[int], elapsed_ms: float
    ) -> None:
        metrics = self._metrics_for_batch(forward_batch)
        for bid in batch_indices:
            metrics[int(bid)].verification_time_ms += max(float(elapsed_ms), 0.0)

    def record_phase(self, req_pool_idx: int, phase: str) -> None:
        metric = self._request_metrics.get(int(req_pool_idx))
        if metric is not None:
            metric.record_phase(phase)

    def get_request_metrics(self, req_pool_idx: int) -> Optional[DllmRequestMetrics]:
        return self._request_metrics.get(int(req_pool_idx))

    def pop_request_metrics(self, req_pool_idx: int) -> Optional[DllmRequestMetrics]:
        self._memory_started.discard(int(req_pool_idx))
        metric = self._request_metrics.pop(int(req_pool_idx), None)
        if metric is not None:
            self._memory_started_request_ids.discard(metric.request_id)
        return metric

    def cleanup_request(self, req_pool_idx: int) -> None:
        self.pop_request_metrics(req_pool_idx)

    def cleanup_request_id(self, request_id: str) -> None:
        self._memory_started_request_ids.discard(str(request_id))


class DllmAlgorithm:
    def __init__(self, config: DllmConfig):
        self.block_size = config.block_size
        self.mask_id = config.mask_id
        if not hasattr(self, "_stats"):
            self._stats = {
                "total_forwards": 0,
                "prefill_forwards": 0,
                "decode_forwards": 0,
                "total_tokens": 0,
            }
        algorithm_name = config.algorithm.lower().replace("_", "").replace("-", "")
        route = "ar_trust" if "selfspec" in algorithm_name else "diffusion_trust"
        self._instrumentation = RequestMetricsRecorder(route)
        # Compatibility for tests and scheduler introspection.
        self._request_metrics = self._instrumentation._request_metrics
        self._output_token_is_ar: dict[int, List[bool]] = {}

    def get_stats(self) -> dict:
        s = copy.deepcopy(self._stats)
        n = s["total_forwards"]
        return {**s, "tpf": s["total_tokens"] / max(n, 1)}

    def reset_stats(self):
        for key, value in self._stats.items():
            if isinstance(value, list):
                self._stats[key] = [0] * len(value)
            elif isinstance(value, dict):
                self._stats[key] = {}
            else:
                self._stats[key] = 0

    def cleanup_request(self, req_pool_idx: int):
        self._output_token_is_ar.pop(int(req_pool_idx), None)
        self._instrumentation.cleanup_request(req_pool_idx)

    def pop_request_metrics(self, req_pool_idx: int) -> Optional[DllmRequestMetrics]:
        self._output_token_is_ar.pop(int(req_pool_idx), None)
        return self._instrumentation.pop_request_metrics(req_pool_idx)

    def _ensure_instrumentation(self, model_runner) -> bool:
        return self._instrumentation.enabled(model_runner)

    def _forward_with_metrics(
        self,
        model_runner,
        forward_batch,
        *,
        modes,
        diffusion_steps=False,
        **_ignored_legacy_estimates,
    ):
        if not self._ensure_instrumentation(model_runner):
            return model_runner.forward(forward_batch, pp_proxy_tensors=None)
        return self._instrumentation.forward(
            model_runner,
            forward_batch,
            phases=modes,
            diffusion_steps=diffusion_steps,
        )

    def _flush_forward_timings(self) -> None:
        self._instrumentation.flush_forward_timings()

    def _set_output_token_modes(self, forward_batch, token_is_ar) -> None:
        if not self._instrumentation.enabled():
            return
        for rpx, modes in zip(
            self._instrumentation._batch_rpx(forward_batch), token_is_ar
        ):
            self._output_token_is_ar[rpx] = [bool(value) for value in modes]

    def record_consumed_tokens(self, req_pool_idx: int, count: int) -> None:
        if not self._instrumentation.enabled():
            return
        rpx = int(req_pool_idx)
        modes = self._output_token_is_ar.pop(rpx, [])
        consumed = max(min(int(count), len(modes)), 0)
        metric = self._instrumentation.get_request_metrics(rpx)
        if metric is not None:
            metric.record_consumed_tokens(stable=consumed, ar=sum(modes[:consumed]))

    def _record_verification_time(
        self, forward_batch, batch_indices: Iterable[int], elapsed_ms: float
    ) -> None:
        if self._instrumentation.enabled():
            self._instrumentation.record_verification_time(
                forward_batch, batch_indices, elapsed_ms
            )

    def _record_invalidation(
        self, forward_batch, *, batch_index: int, start: int, length: int
    ) -> None:
        if self._instrumentation.enabled():
            rpx = self._instrumentation._batch_rpx(forward_batch)[batch_index]
            self._instrumentation.record_invalidation(rpx, start=start, length=length)

    def _record_phase(self, req_pool_idx: int, phase: str) -> None:
        if self._instrumentation.enabled():
            self._instrumentation.record_phase(req_pool_idx, phase)

    @staticmethod
    def from_server_args(server_args: ServerArgs):
        config = DllmConfig.from_server_args(server_args)
        return get_algorithm(config)
