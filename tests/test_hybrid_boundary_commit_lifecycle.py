import ast
import copy
import importlib.util
import logging
import sys
import time
import types
import typing
from enum import Enum
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
EXECUTION_PATH = ROOT / "eval/sglang/srt/dllm/region/execution_spec.py"
CACHE_PATH = ROOT / "eval/sglang/srt/mem_cache/region_state_cache.py"
ALGORITHM_PATH = ROOT / "eval/sglang/srt/dllm/algorithm/hybrid_diffusion_self_spec.py"
SCHEDULER_PATH = ROOT / "eval/sglang/srt/dllm/mixin/scheduler.py"
REQ_PATH = ROOT / "eval/sglang/srt/dllm/mixin/req.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXECUTION = _load_module("boundary_commit_execution_spec", EXECUTION_PATH)
CACHE = _load_module("boundary_commit_region_cache", CACHE_PATH)


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


class _Phase(str, Enum):
    STAGING_PREFILL = "staging_prefill"
    STAGING_DECODE = "staging_decode"
    INCOMING_PREFILL = "incoming_prefill"
    INCOMING_DECODE = "incoming_decode"


INIT_DLLM_REQUEST = _class_method(
    REQ_PATH,
    "ReqDllmMixin",
    "init_diffusion_llm",
    {
        "DllmConfig": object,
        "DllmReqPhase": _Phase,
        "HybridExecutionSpec": EXECUTION.HybridExecutionSpec,
        "Optional": typing.Optional,
        "Req": object,
        "time": time,
    },
)

APPLY_COMMIT = _class_method(
    SCHEDULER_PATH,
    "SchedulerDllmMixin",
    "_apply_hybrid_boundary_commit",
    {
        "HybridBoundaryCommit": EXECUTION.HybridBoundaryCommit,
        "List": typing.List,
        "Optional": typing.Optional,
        "Req": object,
        "logger": logging.getLogger(__name__),
    },
)

COMMIT_BOUNDARIES = _class_method(
    ALGORITHM_PATH,
    "HybridDiffusionSelfSpec",
    "_commit_hybrid_boundaries",
    {
        "ForwardBatch": object,
        "HybridBoundaryCommit": EXECUTION.HybridBoundaryCommit,
        "List": typing.List,
        "ModelRunner": object,
        "RegionStateKey": CACHE.RegionStateKey,
        "hash_positions": EXECUTION.hash_positions,
        "hash_token_ids": EXECUTION.hash_token_ids,
        "replace": __import__("dataclasses").replace,
    },
)


def _config(*, exact=True, block_size=7):
    return types.SimpleNamespace(
        exact_prefix_handoff=exact,
        block_size=block_size,
        mask_id=999,
        algorithm_config={"diffusion_steps": 1},
        attention_contract=EXECUTION.HYBRID_ATTENTION_CONTRACT_V1,
    )


@pytest.mark.parametrize("prompt_length", range(1, 7))
def test_short_exact_prompts_require_initial_causal_prefill(prompt_length):
    req = types.SimpleNamespace(origin_input_ids=list(range(prompt_length)))
    INIT_DLLM_REQUEST(req, _config())
    assert req.dllm_phase is _Phase.INCOMING_PREFILL
    assert req.hybrid_execution_spec.ar_boundary == prompt_length
    assert req.hybrid_prefix_sealed is False
    assert req.hybrid_restore_required is False


def test_short_non_exact_prompt_keeps_decode_direct_behavior():
    req = types.SimpleNamespace(origin_input_ids=list(range(6)))
    INIT_DLLM_REQUEST(req, _config(exact=False))
    assert req.dllm_phase is _Phase.INCOMING_DECODE
    assert req.hybrid_execution_spec is None


def _key(boundary=4, version=0, token_ids=None):
    token_ids = list(range(boundary)) if token_ids is None else list(token_ids)
    return CACHE.RegionStateKey(
        request_id="request-a",
        request_pool_idx=1,
        request_slot_generation=17,
        region_id="causal_prefix",
        region_version=version,
        boundary=boundary,
        token_hash=EXECUTION.hash_token_ids(token_ids),
        position_hash=EXECUTION.hash_positions(0, boundary),
        model_identity="model",
        model_revision="revision",
        adapter_identity="",
        adapter_revision="",
        attention_contract_id=EXECUTION.HYBRID_ATTENTION_CONTRACT_V1,
    )


class _CommitBackend:
    def __init__(self):
        self.committed = []
        self.invalidated = []
        self.region_state_cache = types.SimpleNamespace(
            invalidate_key=self.invalidated.append
        )

    def _current_mamba_slot(self, request_pool_idx):
        return request_pool_idx

    def commit_region_state(self, **kwargs):
        self.committed.append(kwargs)


def _model_commit(old_key, emitted, advance, stable=None):
    backend = _CommitBackend()
    algorithm = types.SimpleNamespace(
        exact_prefix_handoff=True,
        _hybrid_state_keys={old_key.request_pool_idx: old_key},
        _hybrid_boundary_commits={},
        _get_gdn_dllm_backend=lambda _runner: backend,
        _hybrid_kv_reference=lambda _runner, key: ("kv", key.boundary),
    )
    stable = list(range(old_key.boundary)) if stable is None else list(stable)
    forward_batch = types.SimpleNamespace(hybrid_stable_token_ids_cpu=[stable])
    COMMIT_BOUNDARIES(
        algorithm,
        types.SimpleNamespace(),
        forward_batch,
        [old_key.request_pool_idx],
        [0],
        [advance],
        [list(emitted)],
    )
    return algorithm._hybrid_boundary_commits[old_key.request_pool_idx]


def _request(key):
    spec = EXECUTION.HybridExecutionSpec.prefix_diffusion(
        ar_boundary=key.boundary,
        sequence_length=key.boundary + 7,
        diffusion_steps=1,
        prefix_version=key.region_version,
        suffix_version=key.region_version,
    )
    return types.SimpleNamespace(
        rid=key.request_id,
        req_pool_idx=key.request_pool_idx,
        hybrid_request_slot_generation=key.request_slot_generation,
        hybrid_execution_spec=spec,
        hybrid_token_hash=key.token_hash,
        hybrid_position_hash=key.position_hash,
        hybrid_model_identity=key.model_identity,
        hybrid_model_revision=key.model_revision,
        hybrid_adapter_revision=key.adapter_revision,
        lora_id=None,
        dllm_config=_config(),
        origin_input_ids=list(range(key.boundary)),
        output_ids=[],
        finished=lambda: False,
    )


def test_two_emitted_tokens_publish_only_one_model_committed_token():
    old_key = _key()
    emitted = [101, 102]
    commit = _model_commit(old_key, emitted, advance=1)
    req = _request(old_key)
    req.output_ids.extend(emitted)

    APPLY_COMMIT(req, commit, emitted)

    assert req.output_ids == emitted
    assert commit.committed_token_ids == (101,)
    assert req.hybrid_execution_spec.ar_boundary == old_key.boundary + 1
    assert req.hybrid_execution_spec.region_versions[0] == 1
    assert req.hybrid_token_hash == commit.token_hash
    assert req.hybrid_position_hash == commit.position_hash


def test_verify_commit_includes_pending_token_from_previous_output_round():
    old_key = _key()
    committed = [100, 101, 102]
    commit = _model_commit(old_key, committed, advance=3)
    req = _request(old_key)
    req.output_ids = [100]
    newly_emitted = [101, 102, 103]
    req.output_ids.extend(newly_emitted)

    APPLY_COMMIT(req, commit, newly_emitted)

    assert commit.committed_token_ids == tuple(committed)
    assert req.hybrid_execution_spec.ar_boundary == old_key.boundary + 3
    assert req.output_ids == [100, 101, 102, 103]


def test_consecutive_variable_advances_have_no_key_drift():
    key = _key()
    req = _request(key)
    stable_ids = list(range(4))
    next_token = 100
    for expected_version, advance in enumerate((1, 2, 3, 4), start=1):
        emitted = list(range(next_token, next_token + advance))
        commit = _model_commit(key, emitted, advance, stable=stable_ids)
        req.output_ids.extend(emitted)
        APPLY_COMMIT(req, commit, emitted)
        assert req.hybrid_execution_spec.ar_boundary == commit.new_boundary
        assert req.hybrid_execution_spec.region_versions[0] == expected_version
        assert req.hybrid_token_hash == commit.token_hash
        assert req.hybrid_position_hash == commit.position_hash
        stable_ids.extend(emitted)
        key = _key(
            boundary=commit.new_boundary,
            version=commit.new_region_version,
            token_ids=stable_ids,
        )
        next_token += advance


def test_eos_truncation_never_publishes_unemitted_committed_tokens():
    old_key = _key()
    commit = _model_commit(old_key, [500, 501], advance=2)
    req = _request(old_key)
    req.finished = lambda: True
    req.output_ids.append(500)

    APPLY_COMMIT(req, commit, [500])

    assert req.hybrid_execution_spec.ar_boundary == old_key.boundary
    assert req.hybrid_token_hash == old_key.token_hash
    assert req.hybrid_restore_required is False


def test_commit_transport_is_result_scoped_in_tp_worker():
    utils_source = ast.parse((ROOT / "eval/sglang/srt/managers/utils.py").read_text())
    result_class = next(
        node
        for node in utils_source.body
        if isinstance(node, ast.ClassDef) and node.name == "GenerationBatchResult"
    )
    assert any(
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "hybrid_boundary_commits"
        for node in result_class.body
    )

    worker_source = ast.parse(
        (ROOT / "eval/sglang/srt/managers/tp_worker.py").read_text()
    )
    assert any(
        isinstance(node, ast.keyword) and node.arg == "hybrid_boundary_commits"
        for node in ast.walk(worker_source)
    )


def test_every_dllm_result_path_consumes_result_scoped_commit():
    scheduler_source = ast.parse(SCHEDULER_PATH.read_text())
    scheduler_class = next(
        node
        for node in scheduler_source.body
        if isinstance(node, ast.ClassDef) and node.name == "SchedulerDllmMixin"
    )
    for method_name in (
        "_process_dllm_critical_inline",
        "process_batch_result_dllm",
    ):
        method = next(
            node
            for node in scheduler_class.body
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        )
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_publish_hybrid_result_commit"
            for node in ast.walk(method)
        )

    output_source = ast.parse(
        (
            ROOT / "eval/sglang/srt/managers/scheduler_output_processor_mixin.py"
        ).read_text()
    )
    output_class = next(
        node
        for node in output_source.body
        if isinstance(node, ast.ClassDef)
        and node.name == "SchedulerOutputProcessorMixin"
    )
    output_method = next(
        node
        for node in output_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "process_batch_result_dllm"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_publish_hybrid_result_commit"
        for node in ast.walk(output_method)
    )
