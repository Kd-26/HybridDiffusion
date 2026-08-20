import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
region = types.ModuleType("sglang.srt.dllm.region.execution_spec")
region.HYBRID_ATTENTION_CONTRACT_V1 = "causal_prefix_diffusion_suffix_v1"
stubs = {
    "sglang": types.ModuleType("sglang"),
    "sglang.srt": types.ModuleType("sglang.srt"),
    "sglang.srt.configs": types.ModuleType("sglang.srt.configs"),
    "sglang.srt.configs.model_config": types.ModuleType(
        "sglang.srt.configs.model_config"
    ),
    "sglang.srt.server_args": types.ModuleType("sglang.srt.server_args"),
    "sglang.srt.dllm": types.ModuleType("sglang.srt.dllm"),
    "sglang.srt.dllm.region": types.ModuleType("sglang.srt.dllm.region"),
    "sglang.srt.dllm.region.execution_spec": region,
}
stubs["sglang.srt.configs.model_config"].ModelConfig = type("ModelConfig", (), {})
stubs["sglang.srt.server_args"].ServerArgs = type("ServerArgs", (), {})
previous = {name: sys.modules.get(name) for name in stubs}
sys.modules.update(stubs)

path = ROOT / "eval/sglang/srt/dllm/config.py"
spec = importlib.util.spec_from_file_location("cluster1_dllm_config", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for name, old_module in previous.items():
    if old_module is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = old_module
DllmConfig = module.DllmConfig


def make_config(**changes):
    values = dict(
        algorithm="HybridDiffusionSelfSpec",
        algorithm_config={},
        block_size=7,
        mask_id=248077,
        max_running_requests=4,
        causal_prefill=True,
        variant="bd_bidir_shift",
    )
    values.update(changes)
    return DllmConfig(**values)


def test_feature_is_disabled_by_default():
    config = make_config()
    assert config.exact_prefix_handoff is False
    assert config.attention_contract == "causal_prefix_diffusion_suffix_v1"
    assert config.strict_region_state_validation is True
    assert config.region_state_cache_max_entries == 128


def test_enabled_feature_accepts_existing_self_spec_path():
    assert make_config(exact_prefix_handoff=True).exact_prefix_handoff is True


@pytest.mark.parametrize(
    "change",
    [
        {"algorithm": "LowConfidence", "exact_prefix_handoff": True},
        {"causal_prefill": False, "exact_prefix_handoff": True},
        {"attention_contract": "unknown"},
        {"region_state_cache_max_entries": 0},
    ],
)
def test_incompatible_configuration_is_rejected(change):
    with pytest.raises(ValueError):
        make_config(**change)
