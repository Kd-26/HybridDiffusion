import copy
import os
from typing import Iterable, List, Optional, Sequence

import torch

from sglang.srt.dllm.algorithm import get_algorithm
from sglang.srt.dllm.algorithm.instrumentation import DllmRequestMetrics
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.server_args import ServerArgs


class DllmAlgorithm:

    def __init__(
        self,
        config: DllmConfig,
    ):
        self.block_size = config.block_size
        self.mask_id = config.mask_id
        # Subclasses may override _stats with more fields; these are the common ones.
        if not hasattr(self, "_stats"):
            self._stats = {
                "total_forwards": 0,
                "prefill_forwards": 0,
                "decode_forwards": 0,
                "total_tokens": 0,
            }
        self._request_metrics: dict[int, DllmRequestMetrics] = {}
        self._instrumentation_enabled: Optional[bool] = None
        self._instrumented_model_id: Optional[int] = None
        self._instrumentation_hooks = []
        self._active_layer_events = None
        self._pending_layer_timings = []
        self._output_token_is_ar: dict[int, List[bool]] = {}
        self._pending_recompute_tokens: dict[int, int] = {}

    def get_stats(self) -> dict:
        """Return algorithm stats including TPF (tokens per forward)."""
        s = copy.deepcopy(self._stats)
        n = s["total_forwards"]
        return {
            **s,
            "tpf": s["total_tokens"] / max(n, 1),
        }

    def reset_stats(self):
        """Reset all counters to zero."""
        for key, value in self._stats.items():
            if isinstance(value, list):
                self._stats[key] = [0] * len(value)
            elif isinstance(value, dict):
                self._stats[key] = {}
            else:
                self._stats[key] = 0

    def cleanup_request(self, req_pool_idx: int):
        """Optional per-request cleanup hook for algorithms with extra state."""
        return None

    def pop_request_metrics(self, req_pool_idx: int) -> Optional[DllmRequestMetrics]:
        """Return and remove instrumentation state before a pool slot is reused."""
        self._output_token_is_ar.pop(int(req_pool_idx), None)
        self._pending_recompute_tokens.pop(int(req_pool_idx), None)
        return self._request_metrics.pop(int(req_pool_idx), None)

    @staticmethod
    def _is_qwen35_4b(model_runner) -> bool:
        model_config = getattr(model_runner, "model_config", None)
        text_config = getattr(model_config, "hf_text_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        archs = getattr(hf_config, "architectures", None) or []
        return (
            getattr(text_config, "hidden_size", None) == 2560
            and getattr(text_config, "num_hidden_layers", None) == 32
            and any("Qwen3_5DLLM" in arch for arch in archs)
        )

    def _ensure_instrumentation(self, model_runner) -> bool:
        if self._instrumentation_enabled is None:
            requested = os.getenv("SGLANG_DLLM_REQUEST_METRICS", "0") == "1"
            all_models = os.getenv("SGLANG_DLLM_REQUEST_METRICS_ALL_MODELS", "0") == "1"
            self._instrumentation_enabled = requested and (
                all_models or self._is_qwen35_4b(model_runner)
            )
        if self._instrumentation_enabled:
            self._install_layer_timing_hooks(model_runner)
        return bool(self._instrumentation_enabled)

    def _install_layer_timing_hooks(self, model_runner) -> None:
        """Install CUDA-event hooks once; CUDA-graph replays remain explicitly untimed."""
        model = getattr(model_runner, "model", None)
        if model is None or self._instrumented_model_id == id(model):
            return

        for handle in self._instrumentation_hooks:
            handle.remove()
        self._instrumentation_hooks = []
        self._instrumented_model_id = id(model)

        def make_hooks(kind: str):
            def pre_hook(module, _args):
                collector = self._active_layer_events
                if collector is None:
                    return
                start = torch.cuda.Event(enable_timing=True)
                start.record()
                module.__dict__["_dllm_metrics_start_event"] = start

            def post_hook(module, _args, _output):
                collector = self._active_layer_events
                start = module.__dict__.pop("_dllm_metrics_start_event", None)
                if collector is None or start is None:
                    return
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                collector.append((kind, start, end))

            return pre_hook, post_hook

        timed_module_ids = set()
        for module in model.modules():
            class_name = type(module).__name__
            if "AttentionDecoderLayer" in class_name:
                timing_targets = [(module, "attention_total")]
                mlp = getattr(module, "mlp", None)
                if mlp is not None:
                    timing_targets.append((mlp, "attention_mlp"))
            elif class_name == "Qwen3_5GatedDeltaNet":
                timing_targets = [(module, "gdn")]
            else:
                continue
            for timing_module, kind in timing_targets:
                target_key = (id(timing_module), kind)
                if target_key in timed_module_ids:
                    continue
                timed_module_ids.add(target_key)
                pre_hook, post_hook = make_hooks(kind)
                self._instrumentation_hooks.append(
                    timing_module.register_forward_pre_hook(pre_hook)
                )
                self._instrumentation_hooks.append(
                    timing_module.register_forward_hook(post_hook)
                )

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
        if counts is not None:
            return [int(value) for value in counts[: forward_batch.batch_size]]
        if forward_batch.batch_size <= 0:
            return []
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
                metric = DllmRequestMetrics(request_id=rid, req_pool_idx=rpx)
                self._request_metrics[rpx] = metric
            metrics.append(metric)
        return metrics

    @staticmethod
    def _expand_per_request(value, batch_size: int):
        if isinstance(value, str) or not isinstance(value, Sequence):
            return [value] * batch_size
        if len(value) != batch_size:
            raise ValueError(
                f"Expected {batch_size} per-request instrumentation values, got {len(value)}"
            )
        return list(value)

    def _forward_with_metrics(
        self,
        model_runner,
        forward_batch,
        *,
        modes,
        diffusion_steps=False,
        gdn_restores=False,
        recomputed=False,
    ):
        """Run the unchanged model forward and attach request-local counters."""
        enabled = self._ensure_instrumentation(model_runner)
        if not enabled:
            return model_runner.forward(forward_batch, pp_proxy_tensors=None)

        metrics = self._metrics_for_batch(forward_batch)
        batch_size = len(metrics)
        mode_list = self._expand_per_request(modes, batch_size)
        diffusion_list = self._expand_per_request(diffusion_steps, batch_size)
        restore_list = self._expand_per_request(gdn_restores, batch_size)
        recompute_list = self._expand_per_request(recomputed, batch_size)
        token_counts = self._batch_token_counts(forward_batch) if enabled else []
        gdn_layers = int(getattr(forward_batch, "dllm_gdn_layer_count", 0))
        model_layers = int(getattr(forward_batch, "dllm_model_layer_count", 0))

        if enabled:
            for bid, metric in enumerate(metrics):
                active = token_counts[bid]
                pending_recompute = self._pending_recompute_tokens.pop(
                    metric.req_pool_idx, 0
                )
                metric.record_forward(
                    mode=str(mode_list[bid]),
                    active_tokens=active,
                    diffusion_step=bool(diffusion_list[bid]),
                    gdn_state_restores=(gdn_layers if bool(restore_list[bid]) else 0),
                    recomputed_token_layer_positions=(
                        (active if bool(recompute_list[bid]) else 0) * model_layers
                        + pending_recompute * model_layers
                    ),
                )

        collect_cuda_timing = enabled and torch.cuda.is_available()
        self._active_layer_events = [] if collect_cuda_timing else None
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        if collect_cuda_timing:
            events = self._active_layer_events
            # An empty event list is expected for CUDA-graph replay because the
            # Python module hooks are bypassed.  We record that lack of coverage
            # instead of fabricating a component split.
            self._pending_layer_timings.append((list(metrics), events))
        self._active_layer_events = None

        if enabled and torch.cuda.is_available():
            peak = int(torch.cuda.memory_allocated(forward_batch.input_ids.device))
            for metric in metrics:
                metric.peak_memory_bytes = max(metric.peak_memory_bytes, peak)
        return out

    def _flush_forward_timings(self) -> None:
        """Resolve queued CUDA events after the algorithm has consumed GPU output."""
        pending, self._pending_layer_timings = self._pending_layer_timings, []
        for metrics, events in pending:
            if not events:
                for metric in metrics:
                    metric.add_component_time(attention_ms=None, gdn_ms=None)
                continue
            events[-1][2].synchronize()
            attention_total_ms = 0.0
            attention_mlp_ms = 0.0
            gdn_ms = 0.0
            for kind, start, end in events:
                elapsed = float(start.elapsed_time(end))
                if kind == "attention_total":
                    attention_total_ms += elapsed
                elif kind == "attention_mlp":
                    attention_mlp_ms += elapsed
                elif kind == "gdn":
                    gdn_ms += elapsed
            attention_ms = max(attention_total_ms - attention_mlp_ms, 0.0)
            for metric in metrics:
                metric.add_component_time(
                    attention_ms=attention_ms,
                    gdn_ms=gdn_ms,
                )

    def _set_output_token_modes(self, forward_batch, token_is_ar) -> None:
        """Stage token provenance; the scheduler commits only consumed entries."""
        if not self._instrumentation_enabled:
            return
        rpx_list = self._batch_rpx(forward_batch)
        for rpx, modes in zip(rpx_list, token_is_ar):
            self._output_token_is_ar[rpx] = [bool(value) for value in modes]

    def record_consumed_tokens(self, req_pool_idx: int, count: int) -> None:
        """Account tokens actually accepted by stop/max-token processing."""
        if not self._instrumentation_enabled:
            return
        rpx = int(req_pool_idx)
        modes = self._output_token_is_ar.pop(rpx, [])
        consumed = max(min(int(count), len(modes)), 0)
        metric = self._request_metrics.get(rpx)
        if metric is not None:
            metric.record_finalized_tokens(
                stable=consumed,
                ar=sum(modes[:consumed]),
            )

    def _record_verification_time(
        self, forward_batch, batch_indices: Iterable[int], elapsed_ms: float
    ) -> None:
        if not self._instrumentation_enabled:
            return
        metrics = self._metrics_for_batch(forward_batch)
        for bid in batch_indices:
            metrics[int(bid)].verification_time_ms += max(float(elapsed_ms), 0.0)

    def _record_invalidation(
        self,
        forward_batch,
        *,
        batch_index: int,
        start: int,
        length: int,
    ) -> None:
        if not self._instrumentation_enabled:
            return
        metrics = self._metrics_for_batch(forward_batch)
        metrics[batch_index].record_invalidation(
            start=start,
            length=length,
        )
        rpx = metrics[batch_index].req_pool_idx
        self._pending_recompute_tokens[rpx] = max(int(length), 0)

    @staticmethod
    def from_server_args(server_args: ServerArgs):
        config = DllmConfig.from_server_args(server_args)
        return get_algorithm(config)
