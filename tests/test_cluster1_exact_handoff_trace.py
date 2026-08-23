import argparse
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[1]
PATH = ROOT / "eval/scripts/cluster1_exact_handoff_trace.py"
SPEC = importlib.util.spec_from_file_location("cluster1_exact_handoff_trace", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_case(case_id="case-000", steps=2):
    return MODULE.ManifestCase(
        schema_version=1,
        case_id=case_id,
        token_seed=10000,
        prefix_length=8,
        active_length=2,
        diffusion_steps=steps,
        attention_contract_id=MODULE.ATTENTION_CONTRACT,
    )


def make_result(case=None):
    case = case or make_case()
    layers = [
        {
            "layer_id": 0,
            "has_gdn": True,
            "active_hidden_max_abs": 0.0,
            "gdn_conv_max_abs": 0.0,
            "gdn_recurrent_max_abs": 0.0,
        }
    ]
    return {
        "schema_version": 1,
        "case_id": case.case_id,
        "prefix_length": case.prefix_length,
        "active_length": case.active_length,
        "diffusion_steps": case.diffusion_steps,
        "attention_contract_id": case.attention_contract_id,
        "num_layers": 1,
        "cache_hit": True,
        "steps": [
            {
                "step": step,
                "active_logits_max_abs": 0.0,
                "top1_tokens_identical": True,
                "reference_top1_hash": "same",
                "cached_top1_hash": "same",
                "layers": [dict(layers[0])],
            }
            for step in range(1, case.diffusion_steps + 1)
        ],
        "stable_hash_before": "stable",
        "stable_hash_after": "stable",
        "gdn_snapshot_restore_deterministic": True,
        "stale_state_reuse_count": 0,
        "negative_cache_checks": {name: True for name in MODULE.NEGATIVE_CHECK_NAMES},
    }


def write_manifest(path, cases):
    with path.open("w", encoding="utf-8") as output:
        for case in cases:
            output.write(json.dumps(vars(case)) + "\n")


def test_cli_parsing_requires_trace_safe_cuda_mode():
    args = MODULE.parse_args(
        [
            "--model-dir",
            "/model",
            "--manifest",
            "cases.jsonl",
            "--output",
            "results.jsonl",
            "--dtype",
            "bfloat16",
            "--tp-size",
            "1",
            "--disable-cuda-graph",
        ]
    )
    assert args.model_dir == "/model"
    assert args.dtype == "bfloat16"
    assert args.tp_size == 1
    assert args.disable_cuda_graph is True

    with pytest.raises(ValueError, match="disable-cuda-graph"):
        MODULE.parse_args(
            [
                "--model-dir",
                "/model",
                "--manifest",
                "cases.jsonl",
                "--output",
                "results.jsonl",
                "--dtype",
                "float16",
                "--tp-size",
                "1",
            ]
        )


def test_export_emits_exactly_one_record_per_case(tmp_path):
    cases = [make_case("case-000", 1), make_case("case-001", 1)]
    manifest = tmp_path / "cases.jsonl"
    output = tmp_path / "results.jsonl"
    write_manifest(manifest, cases)

    class FakeRuntime:
        def __init__(self, _args):
            self.closed = False

        def run_case(self, case):
            return make_result(case)

        def close(self):
            self.closed = True

    args = argparse.Namespace(
        model_dir="/model",
        manifest=str(manifest),
        output=str(output),
        dtype="bfloat16",
        tp_size=1,
        disable_cuda_graph=True,
    )
    MODULE.export_manifest(args, runtime_factory=FakeRuntime)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["case_id"] for record in records] == [
        "case-000",
        "case-001",
    ]


def test_rejects_missing_layer_and_step_traces():
    case = make_case()
    result = make_result(case)
    result["steps"][0]["layers"] = []
    with pytest.raises(ValueError, match="missing layer traces"):
        MODULE.validate_result(result, case)

    result = make_result(case)
    result["steps"].pop()
    with pytest.raises(ValueError, match="missing diffusion-step traces"):
        MODULE.validate_result(result, case)


def test_rejects_false_cache_hit():
    case = make_case()
    result = make_result(case)
    result["cache_hit"] = False
    with pytest.raises(ValueError, match="genuine exact cache hit"):
        MODULE.validate_result(result, case)


def test_rejects_stable_state_mutation():
    case = make_case()
    result = make_result(case)
    result["stable_hash_after"] = "mutated"
    with pytest.raises(ValueError, match="stable state mutated"):
        MODULE.validate_result(result, case)


def test_rejects_stale_state_reuse():
    case = make_case()
    result = make_result(case)
    result["stale_state_reuse_count"] = 1
    with pytest.raises(ValueError, match="stale GDN state"):
        MODULE.validate_result(result, case)


def test_rejects_any_failed_negative_cache_check():
    case = make_case()
    result = make_result(case)
    result["negative_cache_checks"]["wrong_token_hash_miss"] = False
    with pytest.raises(ValueError, match="wrong_token_hash_miss"):
        MODULE.validate_result(result, case)


def make_captured_step(logits):
    layer = MODULE.CapturedLayer(
        layer_id=0,
        has_gdn=True,
        active_hidden=torch.zeros(2, 3),
        gdn_conv=(torch.zeros(2, 2),),
        gdn_recurrent=torch.zeros(2, 2),
    )
    return MODULE.CapturedStep(
        step=1,
        active_logits=logits,
        top1_tokens=logits.argmax(dim=-1),
        layers=[layer],
    )


def test_corrupted_cached_tensor_produces_nonzero_error():
    reference = make_captured_step(torch.tensor([[1.0, 2.0], [3.0, 1.0]]))
    cached = make_captured_step(torch.tensor([[1.0, 2.5], [3.0, 1.0]]))
    compared = MODULE.compare_steps(reference, cached)
    assert compared["active_logits_max_abs"] == pytest.approx(0.5)


def test_temporary_hooks_and_retained_tensors_are_released():
    class Layer(torch.nn.Module):
        def forward(self, value):
            return value + 1, value

    layers = torch.nn.ModuleList([Layer(), Layer()])
    runner = types.SimpleNamespace(
        model=types.SimpleNamespace(layers=layers),
        req_to_token_pool=types.SimpleNamespace(mamba_map={}),
    )
    hooks = MODULE.ModelTraceHooks(runner)
    value = torch.zeros(2, 3)
    with hooks.capture(active_length=2, mamba_cache_idx=1):
        for layer in layers:
            value, _residual = layer(value)
    trace = hooks.finish_step(1, torch.zeros(2, 5))
    assert len(trace.layers) == 2
    trace.release()
    hooks.close()
    assert hooks.released


def test_tensor_comparison_rejects_missing_or_nonfinite_data():
    with pytest.raises(ValueError, match="shape mismatch"):
        MODULE.tensor_max_abs(torch.zeros(1), torch.zeros(2), "probe")
    with pytest.raises(ValueError, match="NaN or Inf"):
        MODULE.tensor_max_abs(torch.tensor([math.nan]), torch.zeros(1), "probe")
