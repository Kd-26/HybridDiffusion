import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch


PATH = Path(__file__).parents[1] / "eval/sglang/srt/mem_cache/region_state_cache.py"
SPEC = importlib.util.spec_from_file_location("cluster1_region_state_cache", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

KVPrefixReference = MODULE.KVPrefixReference
RegionState = MODULE.RegionState
RegionStateCache = MODULE.RegionStateCache
RegionStateKey = MODULE.RegionStateKey
Miss = MODULE.RegionStateMissReason


def make_key(**changes):
    values = dict(
        request_id="request-a",
        request_pool_idx=3,
        request_slot_generation=11,
        region_id="causal_prefix",
        region_version=2,
        boundary=64,
        token_hash="tokens-a",
        position_hash="positions-a",
        model_identity="model-a",
        model_revision="rev-a",
        adapter_identity="adapter-a",
        adapter_revision="adapter-rev-a",
        attention_contract_id="causal_prefix_diffusion_suffix_v1",
        parent_region_versions=(("root", 1),),
    )
    values.update(changes)
    return RegionStateKey(**values)


def make_state(key=None):
    key = key or make_key()
    return RegionState(
        key=key,
        kv_prefix=KVPrefixReference(
            request_id=key.request_id,
            request_pool_idx=key.request_pool_idx,
            request_slot_generation=key.request_slot_generation,
            pool_identity=99,
            locations=torch.arange(key.boundary),
            valid_length=key.boundary,
        ),
        kv_valid_length=key.boundary,
        kv_owner_request_id=key.request_id,
        gdn_conv_states=(torch.ones(2, 3),),
        gdn_recurrent_states=torch.ones(2, 4),
        boundary=key.boundary,
    )


def test_exact_hit_clones_mutable_state():
    cache = RegionStateCache()
    original = make_state()
    cache.put(original)
    original.gdn_conv_states[0].zero_()
    lookup = cache.get(original.key, current_slot_generation=11)
    assert lookup.hit
    assert torch.all(lookup.state.gdn_conv_states[0] == 1)
    assert cache.hit_count == 1


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("token_hash", "tokens-b", Miss.TOKEN_MISMATCH),
        ("position_hash", "positions-b", Miss.POSITION_MISMATCH),
        ("model_identity", "model-b", Miss.MODEL_MISMATCH),
        ("adapter_identity", "adapter-b", Miss.ADAPTER_MISMATCH),
        ("attention_contract_id", "contract-b", Miss.CONTRACT_MISMATCH),
        ("region_version", 3, Miss.VERSION_MISMATCH),
        ("parent_region_versions", (("root", 2),), Miss.PARENT_VERSION_MISMATCH),
    ],
)
def test_dependency_mismatch_reports_structured_reason(field, value, reason):
    cache = RegionStateCache()
    key = make_key()
    cache.put(make_state(key))
    lookup = cache.get(replace(key, **{field: value}))
    assert not lookup.hit
    assert lookup.miss_reason is reason


def test_invalidation_cleanup_and_recycled_slot_protection():
    cache = RegionStateCache()
    key = make_key()
    cache.put(make_state(key))
    recycled = cache.get(key, current_slot_generation=12)
    assert recycled.miss_reason is Miss.RECYCLED_REQUEST_SLOT
    assert cache.invalidate_region("request-a", "causal_prefix") == 1
    assert not cache.contains(key)
    cache.put(make_state(key))
    assert cache.invalidate_request("request-a") == 1
    assert len(cache) == 0


def test_lru_capacity_evicts_complete_entries():
    cache = RegionStateCache(max_entries=1)
    first = make_key()
    second = replace(first, request_id="request-b", request_pool_idx=4)
    cache.put(make_state(first))
    cache.put(make_state(second))
    assert not cache.contains(first)
    assert cache.contains(second)
