import ast
import copy
import importlib.util
import json
import logging
import sys
import types
import typing
from pathlib import Path

import torch
import pytest


ROOT = Path(__file__).parents[1]
ALGORITHM_PATH = ROOT / "eval/sglang/srt/dllm/algorithm/hybrid_diffusion_self_spec.py"
CACHE_PATH = ROOT / "eval/sglang/srt/mem_cache/region_state_cache.py"
EXECUTION_SPEC_PATH = ROOT / "eval/sglang/srt/dllm/region/execution_spec.py"
SCHEDULER_PATH = ROOT / "eval/sglang/srt/dllm/mixin/scheduler.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CACHE = _load_module("production_handoff_region_cache", CACHE_PATH)
EXECUTION_SPEC = _load_module("production_handoff_execution_spec", EXECUTION_SPEC_PATH)


def _algorithm_method(name):
    source = ast.parse(ALGORITHM_PATH.read_text())
    algorithm = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "HybridDiffusionSelfSpec"
    )
    function = copy.deepcopy(
        next(
            node
            for node in algorithm.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
    )
    function.decorator_list = []
    namespace = {
        "Any": typing.Any,
        "Dict": typing.Dict,
        "ForwardBatch": types.SimpleNamespace,
        "ForwardMode": types.SimpleNamespace(DLLM_MIXED="dllm_mixed"),
        "List": typing.List,
        "ModelRunner": object,
        "RegionStateKey": CACHE.RegionStateKey,
        "Union": typing.Union,
        "DLLM_ATTN_MASK_CAUSAL_PREFILL": 0,
        "canonicalize_kv_prefix_locations": (CACHE.canonicalize_kv_prefix_locations),
        "copy": copy,
        "hash_token_ids": EXECUTION_SPEC.hash_token_ids,
        "logger": logging.getLogger(__name__),
        "json": json,
        "torch": torch,
    }
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(ALGORITHM_PATH), "exec"), namespace)
    return namespace[name]


def _class_method(path, class_name, method_name, namespace):
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
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        )
    )
    function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


CANONICAL_REPLAY_LOCATIONS = _algorithm_method("_canonical_replay_kv_locations")
SNAPSHOT_BOUNDARIES = _algorithm_method("_snapshot_hybrid_boundaries")
RESTORE_BOUNDARIES = _algorithm_method("_restore_hybrid_boundaries")
RECOMPUTE_BOUNDARY = _algorithm_method("_recompute_hybrid_boundary")
VALIDATE_RECOVERY_BATCH = _algorithm_method("_validate_recovery_forward_batch")
COMPLETE_PREFILL = _class_method(
    SCHEDULER_PATH,
    "SchedulerDllmMixin",
    "_complete_dllm_prefill",
    {
        "DllmReqPhase": types.SimpleNamespace(STAGING_DECODE="staging_decode"),
        "GenerationBatchResult": object,
        "Req": object,
        "logger": logging.getLogger(__name__),
    },
)
PROCESS_EMPTY_PREFILL = _class_method(
    SCHEDULER_PATH,
    "SchedulerDllmMixin",
    "_process_empty_dllm_prefill_result",
    {
        "GenerationBatchResult": object,
        "Req": object,
        "ScheduleBatch": object,
        "Scheduler": object,
    },
)


def _make_key(*, generation=11, boundary=4, request_id="request-a"):
    tokens = list(range(boundary))
    return CACHE.RegionStateKey(
        request_id=request_id,
        request_pool_idx=1,
        request_slot_generation=generation,
        region_id="causal_prefix",
        region_version=0,
        boundary=boundary,
        token_hash=EXECUTION_SPEC.hash_token_ids(tokens),
        position_hash=EXECUTION_SPEC.hash_positions(0, boundary),
        model_identity="model",
        model_revision="revision",
        adapter_identity="",
        adapter_revision="",
        attention_contract_id="causal_prefix_diffusion_suffix_v1",
    )


class _ReqPool:
    def __init__(self):
        self.req_to_token = torch.tensor(
            [[0, 1, 2, 3, 4, 5], [1, 3, 5, 7, 2, 4]],
            dtype=torch.int32,
        )
        self.req_index_to_mamba_index_mapping = torch.tensor([0, 1])
        self.mamba_pool = types.SimpleNamespace(
            mamba_cache=types.SimpleNamespace(
                conv=[torch.zeros(1, 2, 3)],
                temporal=torch.zeros(1, 2, 3),
            )
        )


class _LifecycleBackend:
    def __init__(self, pool):
        self.req_to_token_pool = pool
        self.region_state_cache = CACHE.RegionStateCache()
        self.restore_calls = 0

    def _current_mamba_slot(self, request_pool_idx):
        return int(
            self.req_to_token_pool.req_index_to_mamba_index_mapping[
                request_pool_idx
            ].item()
        )

    def snapshot_region_state(self, *, state_key, mamba_cache_idx, kv_prefix):
        mamba = self.req_to_token_pool.mamba_pool.mamba_cache
        state = CACHE.RegionState(
            key=state_key,
            kv_prefix=kv_prefix,
            kv_valid_length=state_key.boundary,
            kv_owner_request_id=state_key.request_id,
            gdn_conv_states=tuple(
                tensor[:, mamba_cache_idx].clone() for tensor in mamba.conv
            ),
            gdn_recurrent_states=mamba.temporal[:, mamba_cache_idx].clone(),
            boundary=state_key.boundary,
        )
        self.region_state_cache.put(state)
        return state

    def restore_region_state(
        self, *, state_key, mamba_cache_idx, current_slot_generation
    ):
        self.restore_calls += 1
        return self.region_state_cache.get(
            state_key, current_slot_generation=current_slot_generation
        )


def _kv_reference(runner, key):
    locations = CACHE.canonicalize_kv_prefix_locations(
        runner.req_to_token_pool.req_to_token[key.request_pool_idx, : key.boundary],
        key.boundary,
        pool_size=runner.token_to_kv_pool.size,
    )
    return CACHE.KVPrefixReference(
        request_id=key.request_id,
        request_pool_idx=key.request_pool_idx,
        request_slot_generation=key.request_slot_generation,
        pool_identity=id(runner.req_to_token_pool),
        locations=locations,
        valid_length=key.boundary,
    )


def _algorithm(backend, key):
    algorithm = types.SimpleNamespace(
        exact_prefix_handoff=True,
        conditional_lora=False,
        _exact_handoff_debug=False,
        _exact_handoff_debug_sync=False,
        _hybrid_state_keys={},
        _hybrid_snapshot_publications={},
    )
    algorithm._get_gdn_dllm_backend = lambda _runner: backend
    algorithm._hybrid_key_for_bid = lambda _batch, _bid, _rpx: key
    algorithm._hybrid_kv_reference = _kv_reference
    algorithm._hybrid_key_log_fields = lambda value: {"request_id": value.request_id}
    algorithm._hybrid_debug_log = lambda *_args, **_kwargs: None
    algorithm._hybrid_debug_synchronize = lambda *_args, **_kwargs: None
    algorithm._canonical_replay_kv_locations = CANONICAL_REPLAY_LOCATIONS
    algorithm._validate_recovery_forward_batch = VALIDATE_RECOVERY_BATCH
    return algorithm


def _restore_batch(key, *, restore_required=True, sealed=True):
    return types.SimpleNamespace(
        batch_size=1,
        seq_lens_cpu=torch.tensor([key.boundary]),
        hybrid_restore_gdn_state=[restore_required],
        hybrid_prefix_sealed_cpu=[sealed],
    )


def _scheduler_req(key):
    spec = EXECUTION_SPEC.HybridExecutionSpec.prefix_diffusion(
        ar_boundary=key.boundary,
        sequence_length=key.boundary + 7,
        diffusion_steps=1,
        prefix_version=key.region_version,
        suffix_version=key.region_version,
    )
    req = types.SimpleNamespace(
        rid=key.request_id,
        req_pool_idx=key.request_pool_idx,
        origin_input_ids=list(range(key.boundary)),
        hybrid_execution_spec=spec,
        hybrid_request_slot_generation=key.request_slot_generation,
        hybrid_token_hash=key.token_hash,
        hybrid_position_hash=key.position_hash,
        hybrid_model_identity=key.model_identity,
        hybrid_model_revision=key.model_revision,
        hybrid_adapter_revision=key.adapter_revision,
        hybrid_prefix_sealed=False,
        hybrid_cache_hit=False,
        hybrid_restore_required=False,
        hybrid_commit_required=False,
        lora_id=None,
        _inline_prefill=False,
        dllm_phase="incoming_prefill",
    )
    req.is_dllm = lambda: True
    req.is_dllm_prefill = lambda: req.dllm_phase in {
        "incoming_prefill",
        "staging_prefill",
    }
    return req


def _process_first_forward(req, publication):
    tree_cache = types.SimpleNamespace(
        cache_unfinished_req=lambda value: setattr(
            value, "prefix_indices", list(range(len(value.origin_input_ids)))
        )
    )
    scheduler = types.SimpleNamespace(tree_cache=tree_cache)
    scheduler._complete_dllm_prefill = types.MethodType(
        lambda _self, value, result: COMPLETE_PREFILL(value, result),
        scheduler,
    )
    result = types.SimpleNamespace(
        next_token_ids=[],
        hybrid_snapshot_publications={req.req_pool_idx: publication},
    )
    PROCESS_EMPTY_PREFILL(scheduler, types.SimpleNamespace(reqs=[req]), result)


def test_recovery_locations_canonicalize_int32_page_table():
    pool = _ReqPool()
    runner = types.SimpleNamespace(
        req_to_token_pool=pool,
        token_to_kv_pool=types.SimpleNamespace(size=8),
    )
    locations = CANONICAL_REPLAY_LOCATIONS(runner, _make_key(), torch.device("cpu"))
    assert pool.req_to_token.dtype is torch.int32
    assert locations.dtype is torch.int64
    assert locations.ndim == 1
    assert locations.is_contiguous()
    assert locations.device == pool.req_to_token.device


@pytest.mark.parametrize("prompt_length", range(1, 7))
def test_prefill_snapshot_then_first_decode_is_a_real_restore_hit(prompt_length):
    pool = _ReqPool()
    backend = _LifecycleBackend(pool)
    key = _make_key(boundary=prompt_length)
    runner = types.SimpleNamespace(
        req_to_token_pool=pool,
        token_to_kv_pool=types.SimpleNamespace(size=8),
    )
    algorithm = _algorithm(backend, key)
    batch = _restore_batch(key)

    SNAPSHOT_BOUNDARIES(algorithm, runner, batch, [0], [key.request_pool_idx])
    req = _scheduler_req(key)
    _process_first_forward(
        req, algorithm._hybrid_snapshot_publications[key.request_pool_idx]
    )
    recomputes = []
    algorithm._recompute_hybrid_boundary = lambda *args: recomputes.append(args)
    RESTORE_BOUNDARIES(algorithm, runner, batch, [0], [key.request_pool_idx])

    assert req.hybrid_prefix_sealed is True
    assert req.dllm_phase == "staging_decode"
    assert req.hybrid_restore_required is True
    assert backend.restore_calls == 1
    assert backend.region_state_cache.hit_count == 1
    assert recomputes == []


def test_health_check_snapshot_publication_precedes_first_restore():
    pool = _ReqPool()
    backend = _LifecycleBackend(pool)
    key = _make_key(boundary=1, request_id="HEALTH_CHECK_regression")
    runner = types.SimpleNamespace(
        req_to_token_pool=pool,
        token_to_kv_pool=types.SimpleNamespace(size=8),
    )
    algorithm = _algorithm(backend, key)
    batch = _restore_batch(key)
    SNAPSHOT_BOUNDARIES(algorithm, runner, batch, [0], [key.request_pool_idx])
    req = _scheduler_req(key)
    _process_first_forward(
        req, algorithm._hybrid_snapshot_publications[key.request_pool_idx]
    )
    recomputes = []
    algorithm._recompute_hybrid_boundary = lambda *args: recomputes.append(args)

    RESTORE_BOUNDARIES(algorithm, runner, batch, [0], [key.request_pool_idx])

    assert backend.region_state_cache.hit_count == 1
    assert recomputes == []


def test_restore_before_snapshot_is_a_lifecycle_error_not_recovery():
    pool = _ReqPool()
    backend = _LifecycleBackend(pool)
    key = _make_key(boundary=1)
    runner = types.SimpleNamespace(
        req_to_token_pool=pool,
        token_to_kv_pool=types.SimpleNamespace(size=8),
    )
    algorithm = _algorithm(backend, key)
    recomputes = []
    algorithm._recompute_hybrid_boundary = lambda *args: recomputes.append(args)

    with pytest.raises(RuntimeError, match="before prefix snapshot publication"):
        RESTORE_BOUNDARIES(
            algorithm,
            runner,
            _restore_batch(key, restore_required=False, sealed=False),
            [0],
            [key.request_pool_idx],
        )

    assert backend.restore_calls == 0
    assert recomputes == []


def test_sealed_restore_without_matching_snapshot_is_a_lifecycle_error():
    pool = _ReqPool()
    backend = _LifecycleBackend(pool)
    key = _make_key(boundary=1)
    runner = types.SimpleNamespace(
        req_to_token_pool=pool,
        token_to_kv_pool=types.SimpleNamespace(size=8),
    )
    algorithm = _algorithm(backend, key)
    recomputes = []
    algorithm._recompute_hybrid_boundary = lambda *args: recomputes.append(args)

    with pytest.raises(RuntimeError, match="no matching committed snapshot"):
        RESTORE_BOUNDARIES(
            algorithm,
            runner,
            _restore_batch(key),
            [0],
            [key.request_pool_idx],
        )

    assert backend.restore_calls == 0
    assert recomputes == []


def test_scheduler_refuses_to_seal_without_model_snapshot_publication():
    key = _make_key(boundary=1)
    req = _scheduler_req(key)

    with pytest.raises(RuntimeError, match="without snapshot publication"):
        COMPLETE_PREFILL(
            req,
            types.SimpleNamespace(hybrid_snapshot_publications={}),
        )

    assert req.hybrid_prefix_sealed is False
    assert req.dllm_phase == "incoming_prefill"


def test_region_cache_hit_does_not_depend_on_local_key_index():
    pool = _ReqPool()
    backend = _LifecycleBackend(pool)
    key = _make_key()
    runner = types.SimpleNamespace(
        req_to_token_pool=pool,
        token_to_kv_pool=types.SimpleNamespace(size=8),
    )
    algorithm = _algorithm(backend, key)
    batch = _restore_batch(key)
    SNAPSHOT_BOUNDARIES(algorithm, runner, batch, [0], [key.request_pool_idx])
    algorithm._hybrid_state_keys.clear()
    recomputes = []
    algorithm._recompute_hybrid_boundary = lambda *args: recomputes.append(args)

    RESTORE_BOUNDARIES(algorithm, runner, batch, [0], [key.request_pool_idx])

    assert backend.region_state_cache.hit_count == 1
    assert algorithm._hybrid_state_keys[key.request_pool_idx] == key
    assert recomputes == []


class _AttentionBackend:
    def __init__(self):
        self.prepared = []

    def init_forward_metadata(self, replay):
        assert replay.batch_size == 1
        assert replay.out_cache_loc.dtype is torch.int64
        assert replay.out_cache_loc.is_contiguous()
        assert torch.all(replay.out_cache_loc >= 0)
        assert torch.all(replay.out_cache_loc < 8)
        self.prepared.append(replay)


class _Runner:
    def __init__(self, pool):
        self.req_to_token_pool = pool
        self.token_to_kv_pool = types.SimpleNamespace(size=8)
        self.attn_backend = _AttentionBackend()
        self.lora_manager = None
        self.forwarded = []

    def forward(self, replay, **kwargs):
        assert kwargs["skip_attn_backend_init"] is True
        assert replay.out_cache_loc.dtype is torch.int64
        self.forwarded.append(replay)


def _decode_batch(key):
    return types.SimpleNamespace(
        input_ids=torch.zeros(3, dtype=torch.int64),
        positions=torch.zeros(3, dtype=torch.int64),
        req_pool_indices=torch.tensor([key.request_pool_idx], dtype=torch.int32),
        seq_lens=torch.tensor([key.boundary + 3], dtype=torch.int32),
        hybrid_stable_token_ids_cpu=[list(range(key.boundary))],
        lora_ids=None,
    )


def test_forced_miss_recovery_uses_safe_indices_and_next_restore_hits():
    pool = _ReqPool()
    backend = _LifecycleBackend(pool)
    key = _make_key()
    runner = _Runner(pool)
    algorithm = _algorithm(backend, key)

    RECOMPUTE_BOUNDARY(algorithm, runner, _decode_batch(key), 0, key, backend)

    assert len(runner.forwarded) == 1
    assert runner.forwarded[0].out_cache_loc.dtype is torch.int64
    assert runner.forwarded[0].out_cache_loc.numel() == key.boundary
    lookup = backend.restore_region_state(
        state_key=key,
        mamba_cache_idx=1,
        current_slot_generation=key.request_slot_generation,
    )
    assert lookup.hit


def test_recovery_validation_rejects_corrupted_location_before_forward():
    pool = _ReqPool()
    backend = _LifecycleBackend(pool)
    key = _make_key()
    runner = _Runner(pool)
    algorithm = _algorithm(backend, key)
    RECOMPUTE_BOUNDARY(algorithm, runner, _decode_batch(key), 0, key, backend)
    replay = runner.forwarded[0]
    replay.out_cache_loc[0] = runner.token_to_kv_pool.size

    with pytest.raises(RuntimeError, match="KV location out of range"):
        VALIDATE_RECOVERY_BATCH(replay, runner, key, mamba_idx=1)


def test_request_slot_reuse_cannot_restore_stale_state():
    pool = _ReqPool()
    backend = _LifecycleBackend(pool)
    old_key = _make_key(generation=11)
    new_key = _make_key(generation=12)
    runner = types.SimpleNamespace(
        req_to_token_pool=pool,
        token_to_kv_pool=types.SimpleNamespace(size=8),
    )
    backend.snapshot_region_state(
        state_key=old_key,
        mamba_cache_idx=1,
        kv_prefix=_kv_reference(runner, old_key),
    )
    algorithm = _algorithm(backend, new_key)
    algorithm._hybrid_state_keys[old_key.request_pool_idx] = old_key
    with pytest.raises(RuntimeError, match="committed key differs"):
        RESTORE_BOUNDARIES(
            algorithm,
            runner,
            _restore_batch(new_key),
            [0],
            [new_key.request_pool_idx],
        )
    assert backend.region_state_cache.hit_count == 0
    assert backend.region_state_cache.contains(old_key)
