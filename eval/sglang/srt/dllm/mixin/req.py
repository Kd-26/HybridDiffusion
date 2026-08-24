from __future__ import annotations

import enum
import time
from typing import TYPE_CHECKING, Optional

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.region.execution_spec import HybridExecutionSpec

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class DllmReqPhase(str, enum.Enum):
    STAGING_PREFILL = "staging_prefill"
    STAGING_DECODE = "staging_decode"
    INCOMING_PREFILL = "incoming_prefill"
    INCOMING_DECODE = "incoming_decode"


class ReqDllmMixin:
    def init_diffusion_llm(self: Req, dllm_config: DllmConfig):
        self.dllm_phase: Optional[DllmReqPhase] = None
        self.dllm_ids = []
        self.dllm_block_offset = 0
        self.dllm_denoise_step = 0
        self.dllm_needs_commit = False
        self.dllm_config = dllm_config
        # KV trim support: when set, cache_unfinished_req uses this
        # instead of len(fill_ids) to bound prefix_indices.
        self.dllm_kv_valid_len: Optional[int] = None
        # Variable advance: override how much dllm_block_offset advances
        # in the next _init_fill_ids_for_dllm call (default = block_size).
        self.dllm_next_advance: Optional[int] = None
        # Monotonic request-local origin for behavior-neutral latency metrics.
        self.dllm_metrics_start_time = time.perf_counter()
        self.dllm_initial_kv_cache_hits: Optional[int] = None

        # Cluster-1 hybrid handoff metadata. The feature is disabled by default,
        # so these fields are inert for every existing configuration.
        self.hybrid_execution_spec: Optional[HybridExecutionSpec] = None
        self.hybrid_region_state_key = None
        self.hybrid_prefix_sealed = False
        self.hybrid_cache_hit = False
        self.hybrid_restore_required = False
        self.hybrid_commit_required = False
        self.hybrid_request_slot_generation = time.monotonic_ns()
        self.hybrid_token_hash = ""
        self.hybrid_position_hash = ""
        self.hybrid_model_identity = ""
        self.hybrid_model_revision = ""
        self.hybrid_adapter_revision = ""
        if dllm_config is not None and getattr(
            dllm_config, "exact_prefix_handoff", False
        ):
            self.hybrid_execution_spec = HybridExecutionSpec.prefix_diffusion(
                ar_boundary=len(self.origin_input_ids),
                sequence_length=len(self.origin_input_ids) + dllm_config.block_size,
                diffusion_steps=int(
                    dllm_config.algorithm_config.get("diffusion_steps", 1)
                ),
                attention_contract_id=getattr(
                    dllm_config,
                    "attention_contract",
                    "causal_prefix_diffusion_suffix_v1",
                ),
            )

        if self.dllm_config is not None:
            # Exact handoff needs one causal prefix forward to seal KV and GDN
            # state before any suffix decode can request a restore.  Short
            # prompts therefore cannot use the legacy decode-direct shortcut.
            if self.hybrid_execution_spec is not None:
                self.dllm_phase = DllmReqPhase.INCOMING_PREFILL
            elif len(self.origin_input_ids) < self.dllm_config.block_size:
                self.dllm_phase = DllmReqPhase.INCOMING_DECODE
            else:
                self.dllm_phase = DllmReqPhase.INCOMING_PREFILL

    def is_dllm(self: Req) -> bool:
        return self.dllm_config is not None

    def is_dllm_prefill(self: Req) -> bool:
        return self.dllm_phase in [
            DllmReqPhase.STAGING_PREFILL,
            DllmReqPhase.INCOMING_PREFILL,
        ]

    def determine_dllm_phase(self: Req):
        prefix_length = len(self.prefix_indices)
        min_required_length = prefix_length + self.dllm_config.block_size

        if len(self.fill_ids) < min_required_length:
            return

        input_block = self.fill_ids[prefix_length:min_required_length]
        is_prefill_phase = self.dllm_config.mask_id not in input_block

        if is_prefill_phase:
            self.dllm_phase = DllmReqPhase.STAGING_PREFILL
        else:
            self.dllm_phase = DllmReqPhase.STAGING_DECODE

    def _init_fill_ids_for_dllm(self: Req):
        if not self.dllm_ids:
            self.dllm_ids = (
                self.origin_input_ids
                + [self.dllm_config.mask_id] * self.dllm_config.block_size
            )
        else:
            # Use variable advance if set (for KV trim), else block_size.
            advance = (
                self.dllm_next_advance
                if self.dllm_next_advance is not None
                else self.dllm_config.block_size
            )
            self.dllm_next_advance = None  # reset
            self.dllm_block_offset += advance
            self.dllm_ids += [self.dllm_config.mask_id] * self.dllm_config.block_size

        self.fill_ids = self.dllm_ids

    def _update_block_offset_for_dllm(self: Req):
        """Compat shim for schedule_batch.py chunked-prefill path.

        With the whole-prompt dLLM prefill model (dllm_ids grows monotonically),
        dllm_block_offset is advanced inside _init_fill_ids_for_dllm, so this
        is a no-op. Kept so schedule_batch.py's existing call site still
        resolves without its own rewrite.
        """
        return
