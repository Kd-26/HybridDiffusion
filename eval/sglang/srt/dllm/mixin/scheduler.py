from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING, List, Optional, Set, Union

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.dllm.region.execution_spec import (
    hash_positions,
    hash_token_ids,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler


class SchedulerDllmMixin:
    def init_diffusion_llm(self: Scheduler):
        self.dllm_config = (
            DllmConfig.from_server_args(self.server_args)
            if self.server_args.dllm_algorithm is not None
            else None
        )
        self.dllm_manager = DllmManager(dllm_config=self.dllm_config)

    def _prepare_hybrid_request(self: Scheduler, req: Req) -> None:
        """Validate and populate scheduler-owned Cluster-1 compatibility data."""
        spec = getattr(req, "hybrid_execution_spec", None)
        if spec is None:
            return
        spec.validate()
        if not req.hybrid_token_hash:
            req.hybrid_token_hash = hash_token_ids(
                req.origin_input_ids[: spec.ar_boundary]
            )
        req.hybrid_position_hash = hash_positions(0, spec.ar_boundary)
        req.hybrid_model_identity = str(
            getattr(self.server_args, "model_path", "")
        )
        req.hybrid_model_revision = str(
            getattr(self.server_args, "revision", "") or ""
        )
        req.hybrid_region_state_key = (
            str(req.rid),
            spec.region_ids[0],
            spec.region_versions[0],
            spec.ar_boundary,
            req.hybrid_token_hash,
            req.hybrid_position_hash,
            req.hybrid_model_identity,
            req.hybrid_model_revision,
            str(getattr(req, "lora_id", "") or ""),
            req.hybrid_adapter_revision,
            spec.attention_contract_id,
        )
        req.hybrid_restore_required = bool(
            req.hybrid_prefix_sealed and not req.is_dllm_prefill()
        )
        req.hybrid_commit_required = not req.is_dllm_prefill()

    @staticmethod
    def _advance_hybrid_boundary(req: Req, accepted_tokens: List[int]) -> None:
        """Publish a new scheduler spec only for tokens actually consumed."""
        spec = getattr(req, "hybrid_execution_spec", None)
        if spec is None or not accepted_tokens:
            return
        boundary = spec.ar_boundary + len(accepted_tokens)
        req.hybrid_execution_spec = spec.advance_boundary(
            boundary,
            sequence_length=boundary + req.dllm_config.block_size,
        )
        req.hybrid_token_hash = hash_token_ids(
            (req.origin_input_ids + req.output_ids)[:boundary]
        )
        req.hybrid_position_hash = hash_positions(0, boundary)
        req.hybrid_region_state_key = (
            str(req.rid),
            req.hybrid_execution_spec.region_ids[0],
            req.hybrid_execution_spec.region_versions[0],
            boundary,
            req.hybrid_token_hash,
            req.hybrid_position_hash,
            req.hybrid_model_identity,
            req.hybrid_model_revision,
            str(getattr(req, "lora_id", "") or ""),
            req.hybrid_adapter_revision,
            req.hybrid_execution_spec.attention_contract_id,
        )
        req.hybrid_prefix_sealed = True
        req.hybrid_cache_hit = True
        req.hybrid_restore_required = True
        req.hybrid_commit_required = False

    def _finalize_dllm_request_metrics(
        self: Scheduler, req: Req, dllm_algo, req_pool_idx: int
    ) -> None:
        """Emit one stable JSON record before the request-pool slot is reused."""
        if dllm_algo is None or not hasattr(dllm_algo, "pop_request_metrics"):
            return
        metric = dllm_algo.pop_request_metrics(req_pool_idx)
        if metric is None:
            return
        if getattr(self, "tp_rank", 0) != 0:
            return

        metric.prompt_tokens = len(req.origin_input_ids)
        initial_hits = getattr(req, "dllm_initial_kv_cache_hits", None)
        if initial_hits is None:
            initial_hits = min(len(req.prefix_indices), metric.prompt_tokens)
        metric.kv_cache_hits = max(int(initial_hits), 0)
        start_time = getattr(req, "dllm_metrics_start_time", None)
        completion_time = getattr(req.time_stats, "completion_time", 0.0)
        if start_time is not None and completion_time > 0:
            metric.total_latency_ms = max(
                (completion_time - start_time) * 1000.0,
                0.0,
            )

        record = metric.to_record()
        # Keep the exact emitted record available to in-process callers/tests.
        req.dllm_request_metrics = record
        serialized = json.dumps(record, sort_keys=True, separators=(",", ":"))
        logger.info("[DLLM_REQUEST_METRICS] %s", serialized)

        output_path = os.getenv("SGLANG_DLLM_REQUEST_METRICS_PATH")
        if output_path:
            try:
                with open(output_path, "a", encoding="utf-8") as output_file:
                    output_file.write(serialized + "\n")
            except OSError as exc:
                logger.warning(
                    "Unable to append DLLM request metrics to %s: %s",
                    output_path,
                    exc,
                )

    def get_new_batch_dllm(self: Scheduler) -> Optional[ScheduleBatch]:
        """Generate a new batch for DLLM (Diffusion LLM) scheduling."""
        if self.enable_priority_preemption:
            self.running_batch.batch_is_full = False

        # Early exit if batch is full or no requests available
        if self._should_skip_prefill():
            return None

        running_bs = len(self.running_batch.reqs)
        self.policy.calc_priority(self.waiting_queue)

        # Create prefill adder with resource constraints
        adder = self._create_dllm_prefill_adder(running_bs)

        # Initialize DLLM manager and transfer requests
        self.dllm_manager.init_next_round()
        self._fetch_waiting_reqs()

        # Process batches
        forward_mode = self._process_dllm_batches(adder)

        can_run_list = adder.can_run_list
        if not can_run_list:
            return None

        # Record metrics and update state
        set_time_batch(can_run_list, "set_forward_entry_time")
        self._update_state_for_batch(can_run_list, adder, running_bs)

        # Create and prepare batch
        new_batch = self._create_dllm_batch(can_run_list, forward_mode)
        return new_batch

    def _process_dllm_critical_inline(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        """Critical-path DLLM processing for the fast decode loop.

        Implements the whole-prompt dLLM behavior: free trimmed KV slots,
        update per-request
        state, append output_ids, and defer stream_output to the next loop
        iteration.
        """
        import torch

        if result.copy_done is not None:
            result.copy_done.synchronize()

        self.token_to_kv_pool_allocator.free_group_begin()

        dllm_algo = getattr(self.tp_worker, "dllm_algorithm", None)
        kv_trim_info = getattr(dllm_algo, "_kv_trim_info", {}) if dllm_algo else {}
        advance_override = (
            getattr(dllm_algo, "_advance_override", {}) if dllm_algo else {}
        )
        mamba_track_commit_info = (
            result.mamba_track_commit_info
            if result.mamba_track_commit_info is not None
            else (
                getattr(dllm_algo, "_mamba_track_commit_info", {})
                if dllm_algo
                else {}
            )
        )
        dllm_write_override = (
            getattr(dllm_algo, "_dllm_write_override", {}) if dllm_algo else {}
        )

        kv_gpu_parts = []
        for idx in range(batch.batch_size()):
            if not result.next_token_ids:
                break
            trim_info = kv_trim_info.get(batch.reqs[idx].req_pool_idx)
            if trim_info is not None:
                gpu_indices = trim_info.get("kv_indices_gpu")
                if gpu_indices is not None:
                    kv_gpu_parts.append(gpu_indices)
        if kv_gpu_parts:
            self.token_to_kv_pool_allocator.free(torch.cat(kv_gpu_parts))

        for idx in range(batch.batch_size()):
            if not result.next_token_ids:
                break

            req = batch.reqs[idx]
            req_pool_idx = req.req_pool_idx
            raw = result.next_token_ids[idx]
            next_token_ids = raw if isinstance(raw, list) else raw.tolist()

            is_inline_pf = getattr(req, "_inline_prefill", False)
            if not next_token_ids or is_inline_pf:
                if not getattr(batch, "_dllm_decode_mode", False) or is_inline_pf:
                    self.tree_cache.cache_unfinished_req(req)
                if req.is_dllm() and (req.is_dllm_prefill() or is_inline_pf):
                    origin_len = len(req.origin_input_ids)
                    cached_len = (
                        len(req.prefix_indices)
                        if req.prefix_indices is not None
                        else 0
                    )
                    if cached_len >= origin_len:
                        req.dllm_phase = DllmReqPhase.STAGING_DECODE
                        req.dllm_next_advance = origin_len
                        req._inline_prefill = False
                        if req.hybrid_execution_spec is not None:
                            req.hybrid_prefix_sealed = True
                            req.hybrid_cache_hit = True
                            req.hybrid_restore_required = True
                continue

            self.num_generated_tokens += len(next_token_ids)

            trim_info = kv_trim_info.get(req_pool_idx)
            if trim_info is not None:
                trim_count = trim_info["trim_count"]
                req.kv_committed_len -= trim_count
                req.kv_allocated_len -= trim_count
                req.dllm_kv_valid_len = req.kv_committed_len
            else:
                req.dllm_kv_valid_len = None

            adv = advance_override.get(req_pool_idx)
            if adv is not None:
                req.dllm_next_advance = adv
            track_boundary = mamba_track_commit_info.get(req_pool_idx)
            if (
                track_boundary is not None
                and req.mamba_ping_pong_track_buffer is not None
            ):
                req.mamba_next_track_idx = (
                    batch.req_to_token_pool.get_mamba_ping_pong_other_idx(
                        req.mamba_next_track_idx
                    )
                )
                req.mamba_last_track_seqlen = track_boundary

            dllm_tokens = dllm_write_override.pop(req_pool_idx, None)
            if req.dllm_ids and dllm_tokens is not None:
                write_start = req.dllm_block_offset
                req.dllm_ids[write_start : write_start + len(dllm_tokens)] = (
                    dllm_tokens
                )

            consumed_tokens = 0
            consumed_token_ids = []
            for next_token_id in next_token_ids:
                req.output_ids.append(next_token_id)
                consumed_tokens += 1
                consumed_token_ids.append(next_token_id)
                req.check_finished()
                if req.finished():
                    break
            if dllm_algo is not None:
                dllm_algo.record_consumed_tokens(req_pool_idx, consumed_tokens)
            self._advance_hybrid_boundary(req, consumed_token_ids)
            if req.finished():
                if dllm_algo is not None:
                    dllm_algo.cleanup_request(req_pool_idx)
                release_kv_cache(req, self.tree_cache)
                req.time_stats.set_completion_time()
                self._finalize_dllm_request_metrics(req, dllm_algo, req_pool_idx)

        self.token_to_kv_pool_allocator.free_group_end()

    def process_batch_result_dllm(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        import torch

        if result.copy_done is not None:
            result.copy_done.synchronize()

        decode_mode = getattr(batch, "_dllm_decode_mode", False)

        if result.next_token_ids:
            self.token_to_kv_pool_allocator.free_group_begin()

            dllm_algo = getattr(self.tp_worker, "dllm_algorithm", None)
            kv_trim_info = (
                getattr(dllm_algo, "_kv_trim_info", {}) if dllm_algo else {}
            )
            advance_override = (
                getattr(dllm_algo, "_advance_override", {}) if dllm_algo else {}
            )
            mamba_track_commit_info = (
                result.mamba_track_commit_info
                if result.mamba_track_commit_info is not None
                else (
                    getattr(dllm_algo, "_mamba_track_commit_info", {})
                    if dllm_algo
                    else {}
                )
            )
            dllm_write_override = (
                getattr(dllm_algo, "_dllm_write_override", {}) if dllm_algo else {}
            )

            kv_gpu_parts = []
            for idx in range(batch.batch_size()):
                trim_info = kv_trim_info.get(batch.reqs[idx].req_pool_idx)
                if trim_info is not None:
                    gpu_idx = trim_info.get("kv_indices_gpu")
                    if gpu_idx is not None:
                        kv_gpu_parts.append(gpu_idx)
            if kv_gpu_parts:
                self.token_to_kv_pool_allocator.free(torch.cat(kv_gpu_parts))

            for idx in range(batch.batch_size()):
                req = batch.reqs[idx]

                # HybridDiffusionSelfSpec emits per-req list[int] (variable accept_len);
                # the in-place LowConfidence / JointThreshold path emits a
                # Tensor of shape (bs, block_size). Accept both.
                tok = result.next_token_ids[idx]
                next_token_ids = tok if isinstance(tok, list) else tok.tolist()
                new_tokens = len(next_token_ids)

                is_inline_pf = getattr(req, "_inline_prefill", False)
                # Inline-prefill reqs emit no tokens on the current step; the
                # algorithm only caches last-logits for the next decode step.
                if new_tokens == 0 or is_inline_pf:
                    # If the batch is in our fast decode mode, prepare_for_dllm_decode
                    # rebuilds KV from kv_committed_len so we skip cache_unfinished_req.
                    if not decode_mode or is_inline_pf:
                        self.tree_cache.cache_unfinished_req(req)
                    if req.is_dllm() and (req.is_dllm_prefill() or is_inline_pf):
                        origin_len = len(req.origin_input_ids)
                        cached_len = (
                            len(req.prefix_indices)
                            if req.prefix_indices is not None
                            else 0
                        )
                        if cached_len >= origin_len:
                            req.dllm_phase = DllmReqPhase.STAGING_DECODE
                            req.dllm_next_advance = origin_len
                            req._inline_prefill = False
                            if req.hybrid_execution_spec is not None:
                                req.hybrid_prefix_sealed = True
                                req.hybrid_cache_hit = True
                                req.hybrid_restore_required = True
                    continue

                self.num_generated_tokens += new_tokens

                # KV trim (variable-accept verify path — see HybridDiffusionSelfSpec)
                trim_info = kv_trim_info.get(req.req_pool_idx)
                if trim_info is not None:
                    trim_count = trim_info.get("trim_count", 0)
                    req.kv_committed_len -= trim_count
                    req.kv_allocated_len -= trim_count
                    req.dllm_kv_valid_len = req.kv_committed_len
                else:
                    req.dllm_kv_valid_len = None

                adv = advance_override.get(req.req_pool_idx)
                if adv is not None:
                    req.dllm_next_advance = adv
                track_boundary = mamba_track_commit_info.get(req.req_pool_idx)
                if (
                    track_boundary is not None
                    and req.mamba_ping_pong_track_buffer is not None
                ):
                    req.mamba_next_track_idx = (
                        batch.req_to_token_pool.get_mamba_ping_pong_other_idx(
                            req.mamba_next_track_idx
                        )
                    )
                    req.mamba_last_track_seqlen = track_boundary

                # dllm_ids write-back (committed tokens for bd_bidir variants)
                dllm_tokens = dllm_write_override.pop(req.req_pool_idx, None)
                if req.dllm_ids and dllm_tokens is not None:
                    write_start = req.dllm_block_offset
                    req.dllm_ids[write_start:write_start + len(dllm_tokens)] = (
                        dllm_tokens
                    )
                else:
                    # LowConfidence / JointThreshold path: just mirror fill_ids
                    req.fill_ids[-new_tokens:] = next_token_ids[:]

                consumed_tokens = 0
                consumed_token_ids = []
                for next_token_id in next_token_ids:
                    req.output_ids.append(next_token_id)
                    consumed_tokens += 1
                    consumed_token_ids.append(next_token_id)
                    req.check_finished()
                    if req.finished():
                        break
                if dllm_algo is not None:
                    dllm_algo.record_consumed_tokens(
                        req.req_pool_idx, consumed_tokens
                    )
                self._advance_hybrid_boundary(req, consumed_token_ids)
                if req.finished():
                    if dllm_algo is not None:
                        dllm_algo.cleanup_request(req.req_pool_idx)
                    release_kv_cache(req, self.tree_cache)
                    req.time_stats.set_completion_time()
                    self._finalize_dllm_request_metrics(
                        req, dllm_algo, req.req_pool_idx
                    )

            if not decode_mode:
                self.stream_output(batch.reqs, batch.return_logprob)
            self.token_to_kv_pool_allocator.free_group_end()

        can_run_cuda_graph = getattr(result, "can_run_cuda_graph", False)
        if getattr(batch, "prefill_stats", None) is not None:
            self.report_prefill_stats(
                prefill_stats=batch.prefill_stats,
                can_run_cuda_graph=can_run_cuda_graph,
                dp_cooperation_info=batch.dp_cooperation_info,
            )

    def _fetch_waiting_reqs(self: Scheduler):
        # Calculate how many requests can be added to DLLM manager
        max_dllm_capacity = self.dllm_config.max_running_requests - len(
            self.dllm_manager.waiting_queue
        )
        # Also limit by available req pool slots (staging reqs already hold slots).
        # HybridDiffusionSelfSpec keeps many requests pending until prompt prefill finishes, so
        # admitting more than the pool can hold leads to stalls.
        req_pool_avail = self.req_to_token_pool.available_size()
        num_requests_to_add = min(
            max_dllm_capacity, len(self.waiting_queue), req_pool_avail
        )

        if num_requests_to_add > 0:
            requests_to_add = self.waiting_queue[:num_requests_to_add]
            self.dllm_manager.add_waiting_reqs(requests_to_add)
            self.waiting_queue = self.waiting_queue[num_requests_to_add:]

    def _should_skip_prefill(self: Scheduler) -> bool:
        """Check if DLLM prefill should be skipped."""
        if (
            self.running_batch.batch_is_full or not self.waiting_queue
        ) and self.dllm_manager.is_empty():
            return True

        running_bs = len(self.running_batch.reqs)
        if (
            self.get_num_allocatable_reqs(running_bs) <= 0
            and self.dllm_manager.is_empty()
            and not self.enable_priority_preemption
        ):
            self.running_batch.batch_is_full = True
            return True

        return False

    def _create_dllm_prefill_adder(self: Scheduler, running_bs: int) -> PrefillAdder:
        """Create a prefill adder configured for DLLM scheduling."""
        return PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            self.running_batch,
            self.new_token_ratio,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
            prefill_max_requests=self.server_args.prefill_max_requests,
            dllm_config=self.dllm_config,
        )

    def _process_dllm_batches(self: Scheduler, adder: PrefillAdder) -> ForwardMode:
        """Process prefill or decode batches for DLLM.

        Original prefill-first policy but with one-shot prefill:
        each new request completes prefill in 1 round, then joins
        the decode batch. This naturally builds up batch size and is
        required by HybridDiffusionSelfSpec (whole-prompt prefill, no chunking).
        """
        forward_mode = ForwardMode.DLLM_EXTEND

        # Try prefill batch first
        prefill_reqs = self.dllm_manager.get_prefill_requests()
        if prefill_reqs:
            self._process_batch_by_phase(
                adder,
                prefill_reqs,
                DllmReqPhase.STAGING_PREFILL,
                DllmReqPhase.INCOMING_PREFILL,
            )
        else:
            # Fall back to decode batch
            decode_reqs = self.dllm_manager.get_decode_requests()
            self._process_batch_by_phase(
                adder,
                decode_reqs,
                DllmReqPhase.STAGING_DECODE,
                DllmReqPhase.INCOMING_DECODE,
            )

        # Safety guard: never mix prefill and decode reqs in the same batch.
        # HybridDiffusionSelfSpec algorithm assumes either all-prefill or all-decode.
        if adder.can_run_list:
            has_prefill = any(req.is_dllm_prefill() for req in adder.can_run_list)
            has_decode = any(not req.is_dllm_prefill() for req in adder.can_run_list)
            if has_prefill and has_decode:
                adder.can_run_list = [
                    req for req in adder.can_run_list if req.is_dllm_prefill()
                ]

        return forward_mode

    def _process_batch_by_phase(
        self,
        adder: PrefillAdder,
        batch: List[Req],
        staging_phase: DllmReqPhase,
        incoming_phase: DllmReqPhase,
    ) -> None:
        """Process a batch, separating staging and incoming requests."""
        staging_reqs = [req for req in batch if req.dllm_phase == staging_phase]
        if staging_reqs:
            staging_result = self.process_dllm_staging_reqs(adder, staging_reqs)
            if staging_result != AddReqResult.CONTINUE:
                return

        incoming_reqs = [req for req in batch if req.dllm_phase == incoming_phase]
        if incoming_reqs:
            self.process_dllm_incoming_reqs(adder, incoming_reqs)

    def _update_state_for_batch(
        self: Scheduler, can_run_list: List[Req], adder: PrefillAdder, running_bs: int
    ) -> None:
        """Update state for the batch."""

        if adder.preempt_list:
            for req in adder.preempt_list:
                self._add_request_to_queue(req)

        if can_run_list:
            self.dllm_manager.add_staging_reqs(can_run_list)
            self.dllm_manager.increment_chunked_count()

        self.adder = adder
        self.can_run_list = can_run_list
        self.running_bs = len(self.running_batch.reqs)

    def _create_dllm_batch(
        self: Scheduler, can_run_list: List[Req], forward_mode: ForwardMode
    ) -> ScheduleBatch:
        """Create and prepare a new DLLM batch."""
        for req in can_run_list:
            self._prepare_hybrid_request(req)
            if req.dllm_initial_kv_cache_hits is None:
                req.dllm_initial_kv_cache_hits = min(
                    len(req.prefix_indices), len(req.origin_input_ids)
                )
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            dllm_config=self.dllm_config,
        )
        new_batch.prepare_for_extend()
        new_batch.forward_mode = forward_mode
        new_batch.decoding_reqs = None

        # Record prefill stats for logging after forward
        from sglang.srt.observability.scheduler_metrics_mixin import PrefillStats

        new_batch.prefill_stats = PrefillStats.from_adder(
            self.adder, self.running_batch.reqs, self.enable_priority_scheduling
        )

        return new_batch

    def process_dllm_incoming_reqs(
        self: Scheduler, adder: PrefillAdder, reqs: List[Req]
    ) -> AddReqResult:
        """Process incoming DLLM requests with resource allocation and preemption."""
        res = AddReqResult.CONTINUE
        for req in reqs:
            # Check if batch is full
            running_bs = len(self.running_batch.reqs)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                self.running_batch.batch_is_full = True

            # Try preemption if batch is full
            if self.running_batch.batch_is_full:
                if (
                    not self.enable_priority_preemption
                    or not adder.preempt_to_schedule(req, self.server_args)
                ):
                    break

            # Prepare and add request
            req.init_next_round_input(self.tree_cache)
            if req.dllm_initial_kv_cache_hits is None:
                req.dllm_initial_kv_cache_hits = min(
                    len(req.prefix_indices), len(req.origin_input_ids)
                )
            res = adder.add_one_req(
                req,
                has_chunked_req=True,
                truncation_align_size=self.truncation_align_size,
            )

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    self.running_batch.batch_is_full = True
                break

        return res

    def process_dllm_staging_reqs(
        self: Scheduler, adder: PrefillAdder, reqs: List[Req]
    ) -> AddReqResult:
        """Process staging DLLM requests with resource allocation."""
        for req in reqs:
            can_run_len_before = len(adder.can_run_list)
            res = adder.add_dllm_staging_req(req)
            if (
                len(adder.can_run_list) > can_run_len_before
                and adder.can_run_list[-1] is req
                and req.req_pool_idx is None
            ):
                # Fresh dLLM requests can enter this path after prefix matching
                # but before req-pool allocation. They still need the same
                # persistent radix lock that add_one_req() takes for incoming
                # requests, otherwise cache_unfinished_req() later decs an
                # unowned prefix node.
                adder._req_inc_lock_ref(req)
            if res == AddReqResult.NO_TOKEN:
                return res

        return AddReqResult.CONTINUE

    def _inline_absorb_new_requests(self: Scheduler, batch: ScheduleBatch) -> None:
        """Absorb new requests into the decode batch WITHOUT a separate forward pass.

        New requests are added to batch.reqs and marked as pending inline prefill.
        prepare_for_dllm_decode then handles mixed decode+prefill batches; the
        algorithm detects prefill requests (no MASKs in the extend slice) and
        saves last logits only, emitting zero tokens on that step.
        """
        if not self.waiting_queue:
            return

        max_reqs = self.dllm_config.max_running_requests
        current_bs = len(batch.reqs)
        capacity = max_reqs - current_bs
        if capacity <= 0:
            return

        if getattr(getattr(self, "disaggregation_mode", None), "name", "NULL") != "NULL":
            return
        max_absorb_tokens = getattr(self.server_args, "chunked_prefill_size", 0) or 4096
        # Reserve the next decode block for each running request; inline prefill
        # should never consume all headroom and force the decode batch to retract.
        max_absorb_tokens = min(
            max_absorb_tokens,
            max(0, self.token_to_kv_pool_allocator.available_size() - current_bs * self.dllm_config.block_size),
        )
        if max_absorb_tokens <= 0:
            return
        req_pool_avail = self.req_to_token_pool.available_size()

        new_reqs = []
        token_budget = max_absorb_tokens
        for req in self.waiting_queue:
            if len(new_reqs) >= min(capacity, req_pool_avail):
                break
            if (
                req.return_logprob
                or req.input_embeds is not None
                or req.multimodal_inputs is not None
            ):
                break
            prefix_len = len(req.prefix_indices)
            remaining = max(0, len(req.origin_input_ids) - prefix_len)
            token_cost = min(remaining, token_budget) if remaining > 0 else self.dllm_config.block_size
            if token_cost <= 0 or token_cost > token_budget:
                break
            req._inline_prefill_max_tokens = token_cost
            new_reqs.append(req)
            token_budget -= token_cost

        num_to_add = len(new_reqs)
        if num_to_add <= 0:
            return

        self.waiting_queue = self.waiting_queue[num_to_add:]

        try:
            req_pool_indices = self.req_to_token_pool.alloc(new_reqs)
        except Exception:
            self.waiting_queue = list(new_reqs) + self.waiting_queue
            return
        if req_pool_indices is None:
            self.waiting_queue = list(new_reqs) + self.waiting_queue
            return

        absorbed = 0
        for i, req in enumerate(new_reqs):
            req.req_pool_idx = req_pool_indices[i]
            req.init_next_round_input(self.tree_cache)
            if req.dllm_initial_kv_cache_hits is None:
                req.dllm_initial_kv_cache_hits = min(
                    len(req.prefix_indices), len(req.origin_input_ids)
                )
            prefix_len = len(req.prefix_indices)
            origin_remaining = len(req.origin_input_ids) - prefix_len
            if origin_remaining <= 0:
                req.dllm_phase = DllmReqPhase.STAGING_DECODE
                req.dllm_next_advance = len(req.origin_input_ids)
            req._inline_prefill = req.is_dllm_prefill()
            batch.reqs.append(req)
            absorbed += 1

        if absorbed > 0:
            if batch.multimodal_inputs is not None:
                batch.multimodal_inputs.extend([None] * absorbed)
            batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
                batch, self.model_config.vocab_size
            )
            self.dllm_manager.staging_queue = [
                r for r in batch.reqs if not r.finished()
            ]

    def _dllm_decode_loop(self: Scheduler, initial_batch: ScheduleBatch) -> None:
        """HybridDiffusion fast decode loop: reuse one batch, call prepare_for_dllm_decode
        each step, inline-absorb new requests, run until all finished.

        Reduces per-iteration scheduling cost by skipping the outer
        get_next_batch_to_run / event-loop machinery.
        """
        steps = 0
        exit_reason = "unknown"
        batch = initial_batch
        batch._dllm_decode_mode = True
        _recv_done = False
        _overlap_recv_reqs = []
        _deferred_stream_batch: Optional[tuple] = None
        _t_start = time.perf_counter()

        try:
            while True:
                # 1. Receive & handle new incoming requests
                if _recv_done:
                    recv_reqs = _overlap_recv_reqs
                    _recv_done = False
                    _overlap_recv_reqs = []
                else:
                    recv_reqs = self.recv_requests()
                if recv_reqs:
                    self.process_input_requests(recv_reqs)

                # 2. Flush finished requests from the batch
                if any(r.finished() for r in batch.reqs):
                    self.dllm_manager.staging_queue = [
                        r for r in self.dllm_manager.staging_queue
                        if not r.finished()
                    ]
                    batch.output_ids = None
                    batch.filter_batch()
                    if batch.is_empty():
                        exit_reason = "all_finished"
                        break

                # 3. Inline-absorb any fresh waiting-queue requests
                if self.waiting_queue:
                    self._inline_absorb_new_requests(batch)

                # 4. Lightweight batch prep (allocates KV from pool)
                success = batch.prepare_for_dllm_decode()
                if not success:
                    exit_reason = "alloc_failed"
                    break

                # 5. Forward + sample via existing run_batch path
                self.cur_batch = batch
                batch.prefill_stats = None
                batch.dp_cooperation_info = None
                batch._dllm_overlap_fn = None

                def _overlap_fn():
                    nonlocal _recv_done, _overlap_recv_reqs, _deferred_stream_batch
                    if _deferred_stream_batch is not None:
                        _reqs, _return_logprob = _deferred_stream_batch
                        self.stream_output(_reqs, _return_logprob)
                        _deferred_stream_batch = None
                    _overlap_recv_reqs = self.recv_requests()
                    _recv_done = True

                batch._dllm_overlap_fn = _overlap_fn
                result = self.run_batch(batch)
                batch._dllm_overlap_fn = None

                # 6. Critical path only. Stream output is deferred by one step.
                self._process_dllm_critical_inline(batch, result)

                # 7. Defer stream_output into the next forward's overlap window.
                _deferred_stream_batch = (list(batch.reqs), batch.return_logprob)

                self.dllm_manager.staging_queue = [
                    r for r in batch.reqs if not r.finished()
                ]
                self.last_batch = batch
                steps += 1

        except Exception as e:
            exit_reason = f"exception: {e}"
            if hasattr(batch, "_dllm_overlap_fn"):
                batch._dllm_overlap_fn = None
            logger.error(
                f"[DLLM decode loop] exception after {steps} steps: {e}"
            )
            import traceback
            traceback.print_exc()

        # Flush the final deferred stream
        if _deferred_stream_batch is not None:
            _reqs, _return_logprob = _deferred_stream_batch
            self.stream_output(_reqs, _return_logprob)

        batch._dllm_decode_mode = False

        # Stash any in-flight, unfinished requests so the outer scheduler
        # can resume them next round.
        for req in batch.reqs:
            if not req.finished():
                self.tree_cache.cache_unfinished_req(req)

        existing_rids = {r.rid for r in self.dllm_manager.waiting_queue}
        for req in batch.reqs:
            if not req.finished() and req.rid not in existing_rids:
                self.dllm_manager.waiting_queue.append(req)

        if steps > 0 and os.getenv("SGLANG_DLLM_LOG_DECODE_LOOP", "0") == "1":
            total_ms = (time.perf_counter() - _t_start) * 1000.0
            logger.info(
                f"[DLLM decode loop] ran {steps} fast steps, exit={exit_reason}, "
                f"total={total_ms:.0f}ms step={total_ms/steps:.1f}ms"
            )


class DllmManager:
    """
    Manager for Diffusion LLM request scheduling.

    Maintains two queues:
    - waiting_queue: The requests waiting to be scheduled with max running requests limit
    - staging_queue: Requests allocated resources by PrefillAdder
    """

    def __init__(self, dllm_config: Optional[DllmConfig] = None):
        self.dllm_config = dllm_config
        self.max_running_reqs = (
            dllm_config.max_running_requests if dllm_config is not None else 1
        )
        self.waiting_queue: List[Req] = []
        self.staging_queue: List[Req] = []

    def get_prefill_requests(self) -> List[Req]:
        """Get all prefill requests from waiting queue."""
        return [req for req in self.waiting_queue if req.is_dllm_prefill()]

    def get_decode_requests(self) -> List[Req]:
        """Get all decode requests from waiting queue."""
        return [req for req in self.waiting_queue if not req.is_dllm_prefill()]

    def add_waiting_reqs(self, reqs: Union[Req, List[Req]]) -> None:
        """Add requests to waiting queue with redundancy check."""
        assert self.dllm_config is not None, "Diffusion LLM config is not set."

        reqs_to_add = reqs if isinstance(reqs, list) else [reqs]

        # Check for duplicate request IDs
        if self._has_duplicate_reqs(reqs_to_add):
            raise RuntimeError("Redundant requests detected in dLLM requests.")

        self.waiting_queue.extend(reqs_to_add)

    def add_staging_reqs(self, reqs: Union[Req, List[Req]]) -> None:
        """Add requests to staging queue (allocated by PrefillAdder)."""
        reqs_to_add = reqs if isinstance(reqs, list) else [reqs]
        self.staging_queue.extend(reqs_to_add)

    def _has_duplicate_reqs(self, reqs: List[Req]) -> bool:
        """Check if any request ID already exists in waiting queue."""
        existing_rids: Set[str] = {r.rid for r in self.waiting_queue}
        return any(req.rid in existing_rids for req in reqs)

    def any_staging_reqs(self) -> bool:
        """Check if there are requests in staging queue."""
        return self.dllm_config is not None and len(self.staging_queue) > 0

    def is_empty(self) -> bool:
        """Check if both queues are empty or DLLM is not configured."""
        if self.dllm_config is None:
            return True
        return len(self.waiting_queue) == 0

    def increment_chunked_count(self) -> None:
        """Increment chunked count for all staging requests."""
        for req in self.staging_queue:
            req.is_chunked += 1

    def filter_finished_reqs(self) -> None:
        """Remove finished requests from both queues."""
        self.waiting_queue = [req for req in self.waiting_queue if not req.finished()]
        self.staging_queue = [req for req in self.staging_queue if not req.finished()]

    def init_next_round(self) -> None:
        """Initialize staging requests for next round and clear staging queue."""
        for req in self.staging_queue:
            req.init_next_round_input()
        self.staging_queue = []
