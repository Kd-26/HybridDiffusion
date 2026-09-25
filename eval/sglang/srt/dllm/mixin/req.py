from __future__ import annotations

import enum
import time
from typing import TYPE_CHECKING, Optional

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.region.execution_spec import (
    HybridExecutionSpec,
    RegionDAGExecutionSpec,
)
from sglang.srt.dllm.region.runtime import (
    RegionDAGInstrumentation,
    RegionDAGRuntimePlan,
    build_region_dag_runtime_plan,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class DllmReqPhase(str, enum.Enum):
    STAGING_PREFILL = "staging_prefill"
    STAGING_DECODE = "staging_decode"
    INCOMING_PREFILL = "incoming_prefill"
    INCOMING_DECODE = "incoming_decode"


class ReqDllmMixin:
    def requires_canonical_region_frontier(self) -> bool:
        """Return whether radix matching must wait for prefix establishment."""
        plan = getattr(self, "region_dag_runtime_plan", None)
        return bool(
            getattr(self, "region_dag_execution_spec", None) is not None
            and getattr(self, "region_dag_mode", "") == "cached"
            and plan is not None
            and int(plan.gdn_replay_start) > 0
            and not getattr(self, "region_dag_frontier_established", False)
            and not getattr(self, "region_dag_initialized", False)
        )

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
        # Cluster-3 is an explicit request contract.  It never weakens or
        # auto-selects over the validated Cluster-1 prefix/suffix lifecycle.
        self.region_dag_execution_spec: Optional[RegionDAGExecutionSpec] = None
        self.region_dag_runtime_plan: Optional[RegionDAGRuntimePlan] = None
        self.region_dag_instrumentation: Optional[RegionDAGInstrumentation] = None
        self.region_dag_mode = ""
        self.region_dag_allow_full_replay = False
        self.region_dag_initialized = False
        self.region_dag_frontier_established = False
        self.region_dag_frontier_establishing = False
        self.region_dag_restore_required = False
        self.region_dag_restore_attention_state = False
        self.region_dag_restore_gdn_state = False
        self.region_dag_frontier_keys = {}
        self.region_dag_model_identity = ""
        self.region_dag_model_revision = ""
        self.region_dag_adapter_revision = ""
        # Request-owned profiling state is never inherited across slot reuse.
        # The controlled profiler attaches explicit route instances after
        # request construction; ordinary production requests keep an empty tuple.
        self.region_dag_profilers = ()

        sampling_params = getattr(self, "sampling_params", None)
        custom_params = getattr(sampling_params, "custom_params", None)
        region_request = (
            custom_params.get("region_dag") if isinstance(custom_params, dict) else None
        )
        if region_request is not None:
            if dllm_config is None:
                raise ValueError("Region-DAG execution requires a dLLM configuration")
            if not isinstance(region_request, dict):
                raise TypeError("custom_params.region_dag must be an object")
            if "execution_spec" not in region_request:
                raise ValueError("Region-DAG request is missing execution_spec")
            spec_value = region_request["execution_spec"]
            self.region_dag_execution_spec = (
                spec_value
                if isinstance(spec_value, RegionDAGExecutionSpec)
                else RegionDAGExecutionSpec.from_dict(spec_value)
            )
            if self.region_dag_execution_spec.sequence_length != len(
                self.origin_input_ids
            ):
                raise ValueError(
                    "Region-DAG sequence_length must equal the submitted token count"
                )
            self.region_dag_mode = str(region_request.get("mode", "cached"))
            if self.region_dag_mode not in ("reference", "cached"):
                raise ValueError("Region-DAG mode must be 'reference' or 'cached'")
            self.region_dag_allow_full_replay = bool(
                region_request.get("allow_full_replay", False)
            )
            edited_regions = tuple(region_request.get("edited_regions", ()))
            self.region_dag_runtime_plan = build_region_dag_runtime_plan(
                self.region_dag_execution_spec,
                edited_regions,
                force_full_replay=self.region_dag_mode == "reference",
            )
            self.region_dag_instrumentation = RegionDAGInstrumentation.from_plan(
                self.region_dag_execution_spec, self.region_dag_runtime_plan
            )

        if (
            self.region_dag_execution_spec is None
            and dllm_config is not None
            and getattr(dllm_config, "exact_prefix_handoff", False)
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
            if self.region_dag_execution_spec is not None:
                self.dllm_phase = DllmReqPhase.INCOMING_PREFILL
            elif self.hybrid_execution_spec is not None:
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
        # Region-DAG execution is a controlled paged-prefill operation.  It is
        # finalized by its own lifecycle and must never enter legacy block
        # decode based on MASK inspection.
        if self.region_dag_execution_spec is not None:
            if not self.region_dag_initialized:
                self.dllm_phase = DllmReqPhase.INCOMING_PREFILL
            return

        # An exact-handoff request cannot decode until its original causal
        # prefix has been materialized and published by the model.  In
        # particular, the diffusion MASK block appended to ``dllm_ids`` must
        # not override the initial prefill phase for short prompts.
        if self.hybrid_execution_spec is not None and not self.hybrid_prefix_sealed:
            if not self.is_dllm_prefill():
                self.dllm_phase = DllmReqPhase.INCOMING_PREFILL
            return

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
        if self.region_dag_execution_spec is not None:
            # The contract describes the complete canonical sequence.  Never
            # append the legacy contiguous diffusion block to this route.
            self.dllm_ids = list(self.origin_input_ids)
            self.fill_ids = list(self.origin_input_ids)
            return

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

        if self.hybrid_execution_spec is not None and not self.hybrid_prefix_sealed:
            # The first exact-handoff forward is a causal seal of the original
            # prompt only. Keep speculative MASKs in dllm_ids for the later
            # decode round, but never schedule them in this prefill.
            self.fill_ids = list(self.origin_input_ids)
        else:
            self.fill_ids = self.dllm_ids

    def _update_block_offset_for_dllm(self: Req):
        """Compat shim for schedule_batch.py chunked-prefill path.

        With the whole-prompt dLLM prefill model (dllm_ids grows monotonically),
        dllm_block_offset is advanced inside _init_fill_ids_for_dllm, so this
        is a no-op. Kept so schedule_batch.py's existing call site still
        resolves without its own rewrite.
        """
        return
