import importlib.util
import sys
import types
from collections import namedtuple
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import torch
import pytest


ROOT = Path(__file__).parents[1]
CACHE_PATH = ROOT / "eval/sglang/srt/mem_cache/region_state_cache.py"
CACHE_SPEC = importlib.util.spec_from_file_location(
    "sglang.srt.mem_cache.region_state_cache", CACHE_PATH
)
CACHE = importlib.util.module_from_spec(CACHE_SPEC)
sys.modules[CACHE_SPEC.name] = CACHE
CACHE_SPEC.loader.exec_module(CACHE)


def _stub(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module


stub_names = (
    "sglang",
    "sglang.srt",
    "sglang.srt.dllm",
    "sglang.srt.layers",
    "sglang.srt.layers.attention",
    "sglang.srt.layers.attention.linear",
    "sglang.srt.layers.attention.mamba",
    "sglang.srt.layers.attention.fla",
    "sglang.srt.mem_cache",
)
previous_modules = {
    name: sys.modules.get(name)
    for name in stub_names
    + (
        "sglang.srt.mem_cache.region_state_cache",
        "sglang.srt.dllm.config",
        "sglang.srt.layers.attention.linear.gdn_backend",
        "sglang.srt.layers.attention.mamba.causal_conv1d_triton",
        "sglang.srt.layers.attention.mamba.mamba_state_scatter_triton",
        "sglang.srt.layers.attention.block_gdn",
        "sglang.srt.layers.attention.fla.fused_recurrent",
    )
}
for package in stub_names:
    sys.modules.setdefault(package, types.ModuleType(package))
sys.modules["sglang.srt.mem_cache.region_state_cache"] = CACHE
_stub(
    "sglang.srt.dllm.config",
    DLLM_ATTN_MASK_BIDIR_BLOCK=1,
    DLLM_ATTN_MASK_CAUSAL_PREFILL=0,
)
_stub(
    "sglang.srt.layers.attention.linear.gdn_backend",
    GDNAttnBackend=object,
    fused_gdn_gating=lambda *args, **kwargs: None,
)
_stub(
    "sglang.srt.layers.attention.mamba.causal_conv1d_triton",
    causal_conv1d_fn=lambda *args, **kwargs: None,
    causal_conv1d_update=lambda *args, **kwargs: None,
)
_stub(
    "sglang.srt.layers.attention.mamba.mamba_state_scatter_triton",
    fused_mamba_state_scatter_with_mask=lambda *args, **kwargs: None,
)
_stub(
    "sglang.srt.layers.attention.block_gdn",
    fused_recurrent_block_causal_gated_delta_rule=lambda *args, **kwargs: None,
    fused_recurrent_block_causal_gated_delta_rule_packed=lambda *args, **kwargs: None,
)
_stub(
    "sglang.srt.layers.attention.fla.fused_recurrent",
    fused_recurrent_gated_delta_rule=lambda *args, **kwargs: None,
)

BACKEND_PATH = ROOT / "eval/sglang/srt/layers/attention/linear/gdn_dllm_backend.py"
BACKEND_SPEC = importlib.util.spec_from_file_location(
    "cluster1_gdn_dllm_backend", BACKEND_PATH
)
BACKEND_MODULE = importlib.util.module_from_spec(BACKEND_SPEC)
BACKEND_SPEC.loader.exec_module(BACKEND_MODULE)
for name, old_module in previous_modules.items():
    if old_module is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = old_module


class FakeReqPool:
    def __init__(self):
        self.req_to_token = torch.arange(256, dtype=torch.int32).view(4, 64)
        self.req_index_to_mamba_index_mapping = torch.tensor([0, 2, 0, 0])
        cache = types.SimpleNamespace(
            conv=[torch.randn(2, 4, 3, 2)],
            temporal=torch.randn(2, 4, 2, 2, 3),
        )
        self.mamba_pool = types.SimpleNamespace(mamba_cache=cache)


class FakeBackend:
    def __init__(self):
        self.req_to_token_pool = FakeReqPool()


def make_key(version=0, generation=9, boundary=8):
    return CACHE.RegionStateKey(
        request_id="request-a",
        request_pool_idx=1,
        request_slot_generation=generation,
        region_id="causal_prefix",
        region_version=version,
        boundary=boundary,
        token_hash=f"tokens-{version}",
        position_hash=f"positions-{boundary}",
        model_identity="model",
        model_revision="revision",
        adapter_identity="",
        adapter_revision="",
        attention_contract_id="causal_prefix_diffusion_suffix_v1",
    )


def kv_reference(pool, key):
    return CACHE.KVPrefixReference(
        request_id=key.request_id,
        request_pool_idx=key.request_pool_idx,
        request_slot_generation=key.request_slot_generation,
        pool_identity=id(pool),
        locations=pool.req_to_token[key.request_pool_idx, : key.boundary].clone(),
        valid_length=key.boundary,
    )


def test_snapshot_restore_is_exact_and_snapshot_is_immutable():
    wrapped = FakeBackend()
    backend = BACKEND_MODULE.GDNDllmBackend(
        wrapped, types.SimpleNamespace(num_hidden_layers=2)
    )
    pool = wrapped.req_to_token_pool
    key = make_key()
    original_conv = pool.mamba_pool.mamba_cache.conv[0][:, 2].clone()
    original_recurrent = pool.mamba_pool.mamba_cache.temporal[:, 2].clone()
    state = backend.snapshot_region_state(
        state_key=key,
        mamba_cache_idx=2,
        kv_prefix=kv_reference(pool, key),
    )

    pool.mamba_pool.mamba_cache.conv[0][:, 2].add_(100)
    pool.mamba_pool.mamba_cache.temporal[:, 2].mul_(0)
    assert torch.equal(state.gdn_conv_states[0], original_conv)
    lookup = backend.restore_region_state(
        state_key=key, mamba_cache_idx=2, current_slot_generation=9
    )
    assert lookup.hit
    assert torch.equal(pool.mamba_pool.mamba_cache.conv[0][:, 2], original_conv)
    assert torch.equal(pool.mamba_pool.mamba_cache.temporal[:, 2], original_recurrent)


def test_commit_is_atomic_and_recycled_slot_cannot_restore():
    wrapped = FakeBackend()
    backend = BACKEND_MODULE.GDNDllmBackend(
        wrapped, types.SimpleNamespace(num_hidden_layers=2)
    )
    pool = wrapped.req_to_token_pool
    old_key = make_key()
    backend.snapshot_region_state(
        state_key=old_key,
        mamba_cache_idx=2,
        kv_prefix=kv_reference(pool, old_key),
    )
    new_key = replace(
        old_key,
        region_version=1,
        boundary=10,
        token_hash="tokens-1",
        position_hash="positions-10",
    )
    backend.commit_region_state(
        state_key=new_key,
        mamba_cache_idx=2,
        kv_prefix=kv_reference(pool, new_key),
    )
    assert backend.region_state_cache.contains(new_key)
    miss = backend.restore_region_state(
        state_key=new_key, mamba_cache_idx=2, current_slot_generation=10
    )
    assert not miss.hit
    assert miss.miss_reason is CACHE.RegionStateMissReason.RECYCLED_REQUEST_SLOT


def test_kv_descriptor_change_prevents_restore():
    wrapped = FakeBackend()
    backend = BACKEND_MODULE.GDNDllmBackend(
        wrapped, types.SimpleNamespace(num_hidden_layers=2)
    )
    pool = wrapped.req_to_token_pool
    key = make_key()
    backend.snapshot_region_state(
        state_key=key,
        mamba_cache_idx=2,
        kv_prefix=kv_reference(pool, key),
    )
    pool.req_to_token[1, 0] = 9999
    lookup = backend.restore_region_state(
        state_key=key, mamba_cache_idx=2, current_slot_generation=9
    )
    assert not lookup.hit
    assert lookup.miss_reason is CACHE.RegionStateMissReason.POSITION_MISMATCH


FrontierKey = namedtuple(
    "FrontierKey",
    (
        "request_id",
        "request_pool_idx",
        "request_slot_generation",
        "boundary",
        "model_identity",
        "model_revision",
        "adapter_identity",
        "adapter_revision",
        "attention_contract_id",
    ),
)


def frontier_key(boundary=8, revision="revision"):
    return FrontierKey(
        request_id="request-a",
        request_pool_idx=1,
        request_slot_generation=9,
        boundary=boundary,
        model_identity="model",
        model_revision=revision,
        adapter_identity="",
        adapter_revision="",
        attention_contract_id="region_dag_conservative_gdn_v1",
    )


def test_region_dag_layer_snapshot_is_immutable_and_restores_exactly():
    backend = BACKEND_MODULE.GDNDllmBackend(
        FakeBackend(), types.SimpleNamespace(num_hidden_layers=2)
    )
    conv = torch.randn(3, 2)
    recurrent = torch.randn(2, 2, 3)
    expected_conv = conv.clone()
    expected_recurrent = recurrent.clone()
    key = frontier_key()
    backend._put_region_dag_layer_snapshot(
        frontier_key=key,
        layer_id=1,
        conv_state=conv,
        recurrent_state=recurrent,
    )
    conv.add_(100)
    recurrent.zero_()
    backend._restore_region_dag_layer_snapshot(
        frontier_key=key,
        layer_id=1,
        conv_destination=conv,
        recurrent_destination=recurrent,
    )
    assert torch.equal(conv, expected_conv)
    assert torch.equal(recurrent, expected_recurrent)


def test_region_dag_layer_snapshot_never_uses_later_or_wrong_revision_state():
    backend = BACKEND_MODULE.GDNDllmBackend(
        FakeBackend(), types.SimpleNamespace(num_hidden_layers=2)
    )
    conv = torch.randn(3, 2)
    recurrent = torch.randn(2, 2, 3)
    backend._put_region_dag_layer_snapshot(
        frontier_key=frontier_key(boundary=16),
        layer_id=1,
        conv_state=conv,
        recurrent_state=recurrent,
    )
    with pytest.raises(RuntimeError, match="frontier snapshot miss"):
        backend._restore_region_dag_layer_snapshot(
            frontier_key=frontier_key(boundary=8),
            layer_id=1,
            conv_destination=conv,
            recurrent_destination=recurrent,
        )
    with pytest.raises(RuntimeError, match="frontier snapshot miss"):
        backend._restore_region_dag_layer_snapshot(
            frontier_key=frontier_key(boundary=16, revision="other"),
            layer_id=1,
            conv_destination=conv,
            recurrent_destination=recurrent,
        )


def test_region_dag_layer_snapshots_are_removed_with_request_cleanup():
    backend = BACKEND_MODULE.GDNDllmBackend(
        FakeBackend(), types.SimpleNamespace(num_hidden_layers=2)
    )
    key = frontier_key()
    backend._put_region_dag_layer_snapshot(
        frontier_key=key,
        layer_id=1,
        conv_state=torch.randn(3, 2),
        recurrent_state=torch.randn(2, 2, 3),
    )
    backend.invalidate_request_state("request-a")
    assert not backend._region_dag_layer_snapshots


def test_region_dag_frontier_publication_requires_every_gdn_layer():
    backend = BACKEND_MODULE.GDNDllmBackend(
        FakeBackend(), types.SimpleNamespace(num_hidden_layers=2)
    )
    key = frontier_key()
    for layer_id in backend.gdn_layer_ids:
        backend._put_region_dag_layer_snapshot(
            frontier_key=key,
            layer_id=layer_id,
            conv_state=torch.randn(3, 2),
            recurrent_state=torch.randn(2, 2, 3),
        )
    backend.validate_region_dag_frontiers([{key.boundary: key}])
    del backend._region_dag_layer_snapshots[(key, backend.gdn_layer_ids[-1])]
    with pytest.raises(RuntimeError, match="missing GDN layer snapshots"):
        backend.validate_region_dag_frontiers([{key.boundary: key}])


def _region_forward_backend():
    cache = types.SimpleNamespace(
        conv=[torch.zeros(3, 3)],
        temporal=torch.zeros(3, 1, 1, 1),
    )
    pool = types.SimpleNamespace(mamba2_layer_cache=lambda _layer_id: cache)
    backend = object.__new__(BACKEND_MODULE.GDNDllmBackend)
    backend.req_to_token_pool = pool
    backend.gdn_backend = types.SimpleNamespace(
        forward_metadata=types.SimpleNamespace(
            mamba_cache_indices=torch.tensor([1], dtype=torch.int32)
        )
    )
    backend.gdn_layer_ids = [0]
    backend._region_dag_layer_snapshots = OrderedDict()
    backend._region_dag_layer_snapshot_limit = 64
    return backend


def _region_forward_batch(reference, positions):
    boundaries = (0, 2, 4, 6, 8)
    keys = {boundary: frontier_key(boundary=boundary) for boundary in boundaries}
    regions = tuple(types.SimpleNamespace(start=start) for start in boundaries[:-1])
    spec = types.SimpleNamespace(sequence_length=8, regions=regions)
    return types.SimpleNamespace(
        region_dag_execution_specs_cpu=[spec],
        region_dag_query_positions_cpu=[tuple(positions)],
        region_dag_frontier_keys_cpu=[keys],
        region_dag_restore_required_cpu=[not reference and positions[0] > 0],
        region_dag_reference_cpu=[reference],
    )


def test_region_dag_cached_gdn_replay_matches_full_ordered_reference(monkeypatch):
    def fake_conv(
        values,
        _weights,
        _bias,
        *,
        conv_states,
        cache_indices,
        **_kwargs,
    ):
        conv_states[int(cache_indices[0])].add_(values.sum(dim=1))
        return values

    def fake_gating(_a_log, a, b, _dt_bias):
        return a, b

    def fake_recurrence(*, q, initial_state, **_kwargs):
        base = initial_state.reshape(1, 1, 1, 1)
        output = q.cumsum(dim=1) + base
        return output, output[:, -1].reshape_as(initial_state)

    monkeypatch.setattr(BACKEND_MODULE, "causal_conv1d_fn", fake_conv)
    monkeypatch.setattr(BACKEND_MODULE, "fused_gdn_gating", fake_gating)
    monkeypatch.setattr(
        BACKEND_MODULE, "fused_recurrent_gated_delta_rule", fake_recurrence
    )
    layer = types.SimpleNamespace(
        layer_id=0,
        conv_weights=None,
        bias=None,
        activation=None,
        q_dim=1,
        k_dim=1,
        v_dim=1,
        num_q_heads=1,
        num_k_heads=1,
        num_v_heads=1,
        head_q_dim=1,
        head_k_dim=1,
        head_v_dim=1,
        A_log=None,
        dt_bias=None,
    )
    original = torch.arange(24, dtype=torch.float32).view(8, 3) / 10
    a = torch.ones(8, 1)
    b = torch.ones(8, 1)
    cached_backend = _region_forward_backend()
    cached_backend._forward_region_dag(
        layer,
        _region_forward_batch(True, range(8)),
        original,
        a,
        b,
    )

    edited = original.clone()
    edited[2:, 0].add_(0.5)
    cached = cached_backend._forward_region_dag(
        layer,
        _region_forward_batch(False, range(2, 8)),
        edited[2:],
        a[2:],
        b[2:],
    )
    full_backend = _region_forward_backend()
    full = full_backend._forward_region_dag(
        layer,
        _region_forward_batch(True, range(8)),
        edited,
        a,
        b,
    )
    assert torch.equal(cached, full[:, 2:])
