import ast
import copy
import importlib.util
import sys
import time
import types
import typing
from enum import Enum
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[1]
REGION_DIR = ROOT / "eval/sglang/srt/dllm/region"
REQ_PATH = ROOT / "eval/sglang/srt/dllm/mixin/req.py"
SCHEDULER_PATH = ROOT / "eval/sglang/srt/dllm/mixin/scheduler.py"
SCHEDULE_BATCH_PATH = ROOT / "eval/sglang/srt/managers/schedule_batch.py"
FLASHINFER_PATH = ROOT / "eval/sglang/srt/layers/attention/flashinfer_backend.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXECUTION = load_module(
    "sglang.srt.dllm.region.execution_spec", REGION_DIR / "execution_spec.py"
)
load_module(
    "sglang.srt.dllm.region.dependency_graph", REGION_DIR / "dependency_graph.py"
)
RUNTIME = load_module("cluster3_serving_runtime", REGION_DIR / "runtime.py")


def class_method(path, class_name, method_name, namespace):
    source = ast.parse(path.read_text())
    class_node = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    function = copy.deepcopy(
        next(
            node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == method_name
        )
    )
    function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


class Phase(str, Enum):
    STAGING_PREFILL = "staging_prefill"
    STAGING_DECODE = "staging_decode"
    INCOMING_PREFILL = "incoming_prefill"
    INCOMING_DECODE = "incoming_decode"


INIT_REQUEST = class_method(
    REQ_PATH,
    "ReqDllmMixin",
    "init_diffusion_llm",
    {
        "DllmConfig": object,
        "DllmReqPhase": Phase,
        "HybridExecutionSpec": EXECUTION.HybridExecutionSpec,
        "Optional": typing.Optional,
        "RegionDAGExecutionSpec": EXECUTION.RegionDAGExecutionSpec,
        "RegionDAGRuntimePlan": RUNTIME.RegionDAGRuntimePlan,
        "RegionDAGInstrumentation": RUNTIME.RegionDAGInstrumentation,
        "build_region_dag_runtime_plan": RUNTIME.build_region_dag_runtime_plan,
        "Req": object,
        "time": time,
    },
)
INIT_FILL = class_method(
    REQ_PATH,
    "ReqDllmMixin",
    "_init_fill_ids_for_dllm",
    {"Req": object},
)
REQUIRES_CANONICAL_FRONTIER = class_method(
    REQ_PATH,
    "ReqDllmMixin",
    "requires_canonical_region_frontier",
    {},
)


class ForwardMode:
    DLLM_EXTEND = "dllm_extend"


class SamplingBatchInfo:
    @staticmethod
    def from_schedule_batch(batch, vocab_size):
        return (batch, vocab_size)


PREPARE_REPLAY = class_method(
    SCHEDULE_BATCH_PATH,
    "ScheduleBatch",
    "prepare_for_region_dag_replay",
    {
        "ForwardMode": ForwardMode,
        "SamplingBatchInfo": SamplingBatchInfo,
        "time": time,
        "torch": torch,
    },
)
PREPARE_DLLM_DECODE = class_method(
    SCHEDULE_BATCH_PATH,
    "ScheduleBatch",
    "prepare_for_dllm_decode",
    {
        "DLLM_ATTN_MASK_BIDIR_BLOCK": 1,
        "DLLM_ATTN_MASK_CAUSAL_PREFILL": 0,
        "DllmReqPhase": Phase,
        "ForwardMode": ForwardMode,
        "SamplingBatchInfo": SamplingBatchInfo,
        "torch": torch,
    },
)

BIND_SLOT = class_method(
    SCHEDULER_PATH,
    "SchedulerDllmMixin",
    "_bind_region_dag_request_slot",
    {
        "Req": object,
        "build_region_dag_frontier_key": RUNTIME.build_region_dag_frontier_key,
    },
)
PREPARE_REGION_REQUEST = class_method(
    SCHEDULER_PATH,
    "SchedulerDllmMixin",
    "_prepare_region_dag_request",
    {
        "Scheduler": object,
        "Req": object,
        "RegionDAGInstrumentation": RUNTIME.RegionDAGInstrumentation,
        "build_canonical_frontier_execution_spec": (
            RUNTIME.build_canonical_frontier_execution_spec
        ),
        "build_region_dag_runtime_plan": RUNTIME.build_region_dag_runtime_plan,
    },
)
COMPLETE_PREFILL = class_method(
    SCHEDULER_PATH,
    "SchedulerDllmMixin",
    "_complete_dllm_prefill",
    {"Req": object, "GenerationBatchResult": object, "DllmReqPhase": Phase},
)
PROCESS_EMPTY_PREFILL = class_method(
    SCHEDULER_PATH,
    "SchedulerDllmMixin",
    "_process_empty_dllm_prefill_result",
    {
        "Scheduler": object,
        "ScheduleBatch": object,
        "GenerationBatchResult": object,
    },
)
INIT_FLASHINFER_METADATA = class_method(
    FLASHINFER_PATH,
    "FlashInferAttnBackend",
    "init_forward_metadata",
    {
        "DLLM_ATTN_MASK_BIDIR_BLOCK": 1,
        "ForwardBatch": object,
        "PrefillMetadata": lambda *args, **kwargs: types.SimpleNamespace(
            args=args, **kwargs
        ),
    },
)


def make_spec():
    region = EXECUTION.RegionDAGRegion
    status = EXECUTION.RegionStatus
    return EXECUTION.RegionDAGExecutionSpec(
        sequence_length=8,
        diffusion_steps=2,
        regions=(
            region("A", 0, 0, 2, status.STABLE, (), (), "ta", "pa"),
            region("B", 0, 2, 4, status.ACTIVE, ("A",), (("A", 0),), "tb", "pb"),
            region("C", 0, 4, 6, status.STABLE, ("A",), (("A", 0),), "tc", "pc"),
            region("D", 0, 6, 8, status.ACTIVE, ("B",), (("B", 0),), "td", "pd"),
        ),
    )


def make_config():
    return types.SimpleNamespace(
        exact_prefix_handoff=True,
        block_size=7,
        mask_id=999,
        algorithm_config={"diffusion_steps": 2},
        attention_contract=EXECUTION.HYBRID_ATTENTION_CONTRACT_V1,
    )


def make_request(mode="reference", allow_full_replay=False):
    spec = make_spec()
    req = types.SimpleNamespace(
        rid="request-a",
        origin_input_ids=list(range(8)),
        output_ids=[],
        sampling_params=types.SimpleNamespace(
            custom_params={
                "region_dag": {
                    "execution_spec": spec.to_dict(),
                    "edited_regions": ["B"],
                    "mode": mode,
                    "allow_full_replay": allow_full_replay,
                }
            }
        ),
        lora_id=None,
        prefix_indices=torch.empty(0, dtype=torch.int64),
        req_pool_idx=None,
        kv_committed_len=0,
        kv_allocated_len=0,
        extend_batch_idx=0,
        multimodal_inputs=None,
    )
    req.set_extend_input_len = lambda value: setattr(req, "extend_input_len", value)
    INIT_REQUEST(req, make_config())
    return req


def test_request_parser_selects_isolated_region_route_and_full_reference():
    req = make_request()
    assert req.hybrid_execution_spec is None
    assert req.region_dag_execution_spec == make_spec()
    assert req.region_dag_runtime_plan.logical_invalidation_regions == ("B", "D")
    assert req.region_dag_runtime_plan.gdn_replay_start == 0
    assert req.region_dag_runtime_plan.full_replay
    assert req.dllm_phase is Phase.INCOMING_PREFILL
    INIT_FILL(req)
    assert req.fill_ids == list(range(8))
    assert req.dllm_ids == list(range(8))


def test_request_parser_rejects_unknown_mode_and_length_mismatch():
    req = make_request()
    req.sampling_params.custom_params["region_dag"]["mode"] = "guess"
    with pytest.raises(ValueError, match="reference.*cached"):
        INIT_REQUEST(req, make_config())

    req = make_request()
    req.origin_input_ids.pop()
    with pytest.raises(ValueError, match="sequence_length"):
        INIT_REQUEST(req, make_config())


def bind_request(req):
    req.req_pool_idx = 1
    req.region_dag_model_identity = "model"
    req.region_dag_model_revision = "revision"
    BIND_SLOT(req)


def make_batch(req, mapping=None):
    req.region_dag_initialized = True
    req.region_dag_mode = "cached"
    req.region_dag_runtime_plan = RUNTIME.build_region_dag_runtime_plan(
        req.region_dag_execution_spec, ("B",)
    )
    req.kv_committed_len = 8
    bind_request(req)
    mapping = (
        torch.tensor(
            [[0] * 16, [11, 12, 13, 14, 15, 16, 17, 18] + [0] * 8], dtype=torch.int32
        )
        if mapping is None
        else mapping
    )
    return types.SimpleNamespace(
        reqs=[req],
        req_to_token_pool=types.SimpleNamespace(req_to_token=mapping),
        token_to_kv_pool_allocator=types.SimpleNamespace(size=64),
        model_config=types.SimpleNamespace(vocab_size=1000),
        device="cpu",
        return_logprob=False,
        has_grammar=False,
    )


def test_cached_replay_reuses_real_page_table_with_int64_absolute_rows():
    req = make_request()
    batch = make_batch(req)
    page_table_before = batch.req_to_token_pool.req_to_token.clone()
    assert PREPARE_REPLAY(batch)
    assert batch.region_dag_query_positions_cpu == [tuple(range(2, 8))]
    assert batch.input_ids.tolist() == list(range(2, 8))
    assert batch.out_cache_loc.dtype is torch.int64
    assert batch.out_cache_loc.is_contiguous()
    assert batch.out_cache_loc.tolist() == [13, 14, 15, 16, 17, 18]
    assert torch.equal(batch.req_to_token_pool.req_to_token, page_table_before)
    assert req.req_pool_idx == 1


def test_legacy_decode_entry_dispatches_region_requests_to_explicit_replay():
    req = make_request()
    batch = types.SimpleNamespace(
        reqs=[req],
        prepare_for_region_dag_replay=lambda: "region-replay",
    )
    assert PREPARE_DLLM_DECODE(batch) == "region-replay"


def test_cached_replay_fails_without_real_slot_or_complete_page_table():
    req = make_request()
    batch = make_batch(req)
    req.req_pool_idx = None
    with pytest.raises(RuntimeError, match="real request-pool slot"):
        PREPARE_REPLAY(batch)

    req = make_request()
    bad_mapping = torch.zeros((2, 16), dtype=torch.int32)
    batch = make_batch(req, bad_mapping)
    with pytest.raises(RuntimeError, match="invalid physical KV locations"):
        PREPARE_REPLAY(batch)


def test_frontier_binding_never_fabricates_a_request_slot():
    req = make_request()
    with pytest.raises(RuntimeError, match="real request-pool slot"):
        BIND_SLOT(req)
    assert req.req_pool_idx is None


def test_frontier_keys_cover_every_region_start_and_final_boundary():
    req = make_request()
    bind_request(req)
    assert tuple(req.region_dag_frontier_keys) == (0, 2, 4, 6, 8)
    assert req.region_dag_frontier_keys[2].preceding_regions[0][0] == "A"


def test_cached_request_establishes_frontier_before_attaching_suffix():
    req = make_request(mode="cached")
    scheduler = types.SimpleNamespace(
        server_args=types.SimpleNamespace(model_path="model", revision="revision")
    )

    PREPARE_REGION_REQUEST(scheduler, req)
    assert req.region_dag_frontier_establishing
    assert not req.region_dag_frontier_established
    assert not req.region_dag_restore_required
    assert req.fill_ids == [0, 1]
    assert req.extend_input_len == 2

    bind_request(req)
    req.prefix_indices = torch.tensor([11, 12], dtype=torch.int64)
    expected = {
        boundary: key
        for boundary, key in req.region_dag_frontier_keys.items()
        if boundary <= 2
    }
    req.is_dllm = lambda: True
    req.is_dllm_prefill = lambda: True
    lifecycle = types.SimpleNamespace(
        tree_cache=types.SimpleNamespace(cache_unfinished_req=lambda _req: None),
        _complete_dllm_prefill=lambda *_args: pytest.fail(
            "partial frontier establishment completed the full prefill"
        ),
    )
    PROCESS_EMPTY_PREFILL(
        lifecycle,
        types.SimpleNamespace(reqs=[req]),
        types.SimpleNamespace(
            region_dag_snapshot_publications={req.req_pool_idx: expected}
        ),
    )
    assert req.region_dag_frontier_established
    assert not req.region_dag_frontier_establishing
    assert req.region_dag_restore_required
    assert req.fill_ids == list(range(8))

    PREPARE_REGION_REQUEST(scheduler, req)
    assert req.region_dag_frontier_established
    assert not req.region_dag_initialized
    assert req.region_dag_restore_required
    assert req.fill_ids == list(range(8))


def test_radix_matching_is_deferred_only_for_unestablished_positive_frontier():
    req = make_request(mode="cached")
    assert REQUIRES_CANONICAL_FRONTIER(req)

    req.region_dag_frontier_established = True
    assert not REQUIRES_CANONICAL_FRONTIER(req)

    req = make_request(mode="reference")
    assert not REQUIRES_CANONICAL_FRONTIER(req)


def test_suffix_attachment_rejects_a_prefix_beyond_the_frontier():
    req = make_request(mode="cached")
    req.region_dag_frontier_established = True
    req.prefix_indices = torch.tensor([11, 12, 13], dtype=torch.int64)
    scheduler = types.SimpleNamespace(
        server_args=types.SimpleNamespace(model_path="model", revision="revision")
    )
    with pytest.raises(RuntimeError, match="exact frontier prefix"):
        PREPARE_REGION_REQUEST(scheduler, req)


def test_region_initialization_requires_all_published_gdn_frontiers():
    req = make_request()
    bind_request(req)
    result = types.SimpleNamespace(
        region_dag_snapshot_publications={
            req.req_pool_idx: dict(req.region_dag_frontier_keys)
        }
    )
    COMPLETE_PREFILL(req, result)
    assert req.region_dag_initialized
    assert req.dllm_phase is Phase.STAGING_DECODE

    req = make_request()
    bind_request(req)
    with pytest.raises(RuntimeError, match="every exact GDN frontier"):
        COMPLETE_PREFILL(
            req,
            types.SimpleNamespace(region_dag_snapshot_publications={}),
        )


def test_flashinfer_region_route_forces_exact_custom_paged_mask(monkeypatch):
    mask_module = types.ModuleType("sglang.srt.dllm.attention_mask")

    def validate(mask, query_counts, kv_counts):
        assert mask.dtype is torch.bool
        assert query_counts == [6]
        assert kv_counts == [8]
        assert mask.numel() == 48

    mask_module.validate_region_dag_paged_custom_mask = validate
    monkeypatch.setitem(sys.modules, "sglang.srt.dllm.attention_mask", mask_module)

    calls = []
    updater = types.SimpleNamespace(
        update=lambda *args, **kwargs: calls.append((args, kwargs))
    )
    backend = types.SimpleNamespace(
        indices_updater_prefill=updater,
        prefill_wrappers_paged=[object()],
        prefill_split_tile_size=4,
    )
    mode = types.SimpleNamespace(
        is_decode_or_idle=lambda: False,
        is_draft_extend=lambda: False,
        is_target_verify=lambda: False,
        is_dllm_mode=lambda: True,
    )
    forward_batch = types.SimpleNamespace(
        forward_mode=mode,
        region_dag_execution_specs_cpu=[make_spec()],
        region_dag_query_positions_cpu=[tuple(range(2, 8))],
        region_dag_custom_mask=torch.ones(48, dtype=torch.bool),
        extend_prefix_lens=torch.tensor([2], dtype=torch.int32),
        seq_lens=torch.tensor([8], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([8], dtype=torch.int64),
        seq_lens_sum=8,
        req_pool_indices=torch.tensor([1], dtype=torch.int64),
        encoder_lens=None,
    )
    INIT_FLASHINFER_METADATA(backend, forward_batch)
    assert len(calls) == 1
    assert calls[0][1]["custom_mask"] is forward_batch.region_dag_custom_mask
    assert not calls[0][1]["dllm_native_bidir_mask"]
    assert forward_batch.dllm_selected_mask_backend == "custom_paged"
    assert backend.forward_metadata.dllm_selected_mask_backend == "custom_paged"
