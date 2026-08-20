import importlib.util
import sys
import types
from dataclasses import replace
from pathlib import Path

import torch


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
        self.req_to_token = torch.arange(256, dtype=torch.int64).view(4, 64)
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
