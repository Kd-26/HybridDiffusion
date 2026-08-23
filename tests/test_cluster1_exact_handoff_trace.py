import argparse
import ast
import contextlib
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

MASK_PATH = ROOT / "eval/sglang/srt/dllm/attention_mask.py"
MASK_SPEC = importlib.util.spec_from_file_location("cluster1_attention_mask", MASK_PATH)
MASK_MODULE = importlib.util.module_from_spec(MASK_SPEC)
sys.modules[MASK_SPEC.name] = MASK_MODULE
MASK_SPEC.loader.exec_module(MASK_MODULE)


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
        "attention_mask_backend": MODULE.ATTENTION_MASK_BACKEND,
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


class FakeTraceHooks:
    @contextlib.contextmanager
    def capture(self, *, active_length, mamba_cache_idx):
        del active_length, mamba_cache_idx
        yield

    def finish_step(self, step, logits):
        return MODULE.CapturedStep(
            step=step,
            active_logits=logits.detach().clone(),
            top1_tokens=logits.argmax(dim=-1),
            layers=[],
        )


def make_active_request(prefix_length=256):
    req = types.SimpleNamespace(
        prefix_indices=torch.arange(prefix_length, dtype=torch.int64),
        origin_input_ids=list(range(prefix_length)),
        dllm_block_offset=0,
        req_pool_idx=3,
    )

    def set_extend_input_len(length):
        req.extend_input_len = length

    req.set_extend_input_len = set_extend_input_len
    return req


def make_active_runtime():
    runtime = MODULE.Cluster1ModelRuntime.__new__(MODULE.Cluster1ModelRuntime)
    runtime.backend = types.SimpleNamespace(
        _current_mamba_slot=lambda req_pool_idx: req_pool_idx + 10
    )
    runtime._synchronize = lambda: None
    runtime._forward = lambda forward_batch: torch.zeros(
        forward_batch.input_ids.numel(), 4
    )
    observed_offsets = []
    observed_positions = []

    def prepare_extend(self, req, *, bidir):
        assert bidir is True
        observed_offsets.append(req.dllm_block_offset)
        positions = torch.arange(
            req.dllm_block_offset,
            req.dllm_block_offset + req.extend_input_len,
            dtype=torch.int64,
        )
        observed_positions.append(positions.clone())
        forward_batch = types.SimpleNamespace(
            input_ids=torch.tensor(
                req.fill_ids[-req.extend_input_len :], dtype=torch.int64
            ),
            extend_prefix_lens=torch.tensor(
                [len(req.prefix_indices)], dtype=torch.int64
            ),
            positions=positions,
            dllm_bidir_custom_mask=torch.ones(
                req.extend_input_len, req.extend_input_len, dtype=torch.bool
            ),
        )
        return object(), forward_batch

    runtime._prepare_extend = types.MethodType(prepare_extend, runtime)
    return runtime, observed_offsets, observed_positions


def structured_b7_g4_mask():
    mask = torch.zeros((7, 7), dtype=torch.bool)
    rows = torch.arange(4).unsqueeze(1)
    cols = torch.arange(7).unsqueeze(0)
    mask[:4, :] = cols <= rows
    mask[4:, :] = True
    return mask


def select_mask_backend(requested, *, active_length, native_available=True):
    prefix_length = 256
    return MASK_MODULE.select_bidir_mask_backend(
        requested,
        native_available=native_available,
        configured_block_size=7,
        seq_lens=torch.tensor([prefix_length + active_length]),
        prefix_lens=torch.tensor([prefix_length]),
        block_mask=(
            structured_b7_g4_mask()
            if active_length == 7
            else torch.ones((active_length, active_length), dtype=torch.bool)
        ),
        structured_mask=structured_b7_g4_mask(),
    )


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


def test_forward_batch_mask_backend_defaults_to_auto():
    source = ast.parse(
        (ROOT / "eval/sglang/srt/model_executor/forward_batch_info.py").read_text()
    )
    forward_batch = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "ForwardBatch"
    )
    field = next(
        node
        for node in forward_batch.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "dllm_bidir_mask_backend"
    )
    assert isinstance(field.value, ast.Constant)
    assert field.value.value == "auto"


def test_invalid_mask_backend_is_rejected():
    with pytest.raises(ValueError, match="invalid dllm_bidir_mask_backend"):
        select_mask_backend("surprise", active_length=32)


def test_explicit_custom_never_selects_native():
    assert select_mask_backend("custom", active_length=32) == "custom"
    assert select_mask_backend("custom", active_length=7) == "custom"


def test_explicit_native_rejects_arbitrary_active_length():
    with pytest.raises(RuntimeError, match="block_size=7"):
        select_mask_backend("native", active_length=32)


def test_auto_selects_custom_for_arbitrary_active_length():
    assert select_mask_backend("auto", active_length=32) == "custom"


def test_native_remains_selectable_for_compatible_structured_request():
    assert select_mask_backend("native", active_length=7) == "native"
    assert select_mask_backend("auto", active_length=7) == "native"


def test_auto_does_not_ignore_incompatible_seven_token_mask():
    assert (
        MASK_MODULE.select_bidir_mask_backend(
            "auto",
            native_available=True,
            configured_block_size=7,
            seq_lens=torch.tensor([263]),
            prefix_lens=torch.tensor([256]),
            block_mask=torch.ones((7, 7), dtype=torch.bool),
            structured_mask=structured_b7_g4_mask(),
        )
        == "custom"
    )


def test_native_rejects_unavailable_backend():
    with pytest.raises(RuntimeError, match="unavailable"):
        select_mask_backend("native", active_length=7, native_available=False)


def test_paged_custom_mask_has_exact_prefix_and_suffix_layout():
    prefix_length = 256
    active_length = 32
    active_mask = torch.tril(
        torch.ones((active_length, active_length), dtype=torch.bool)
    )
    flattened = MASK_MODULE.build_paged_custom_mask(
        torch.tensor([prefix_length + active_length]),
        torch.tensor([prefix_length]),
        active_mask,
        torch.tensor([1]),
        bidir_mask_type=1,
    )

    assert flattened.numel() == 32 * 288 == 9216
    logical = flattened.view(active_length, prefix_length + active_length)
    assert bool(logical[:, :prefix_length].all().item())
    assert torch.equal(logical[:, prefix_length:], active_mask)


@pytest.mark.parametrize(
    "mask, message",
    [
        (torch.ones((32, 32), dtype=torch.float32), "dtype bool"),
        (torch.ones(32, dtype=torch.bool), "2-D"),
        (torch.ones((31, 32), dtype=torch.bool), "rows"),
        (torch.ones((32, 31), dtype=torch.bool), "columns"),
    ],
    ids=("non-bool", "rank-one", "undersized-rows", "undersized-columns"),
)
def test_invalid_custom_masks_fail_before_planning(mask, message):
    with pytest.raises((TypeError, ValueError), match=message):
        MASK_MODULE.build_paged_custom_mask(
            torch.tensor([288]),
            torch.tensor([256]),
            mask,
            torch.tensor([1]),
            bidir_mask_type=1,
        )


def test_exporter_requests_custom_mask_backend():
    forward_batch = types.SimpleNamespace()
    MODULE.Cluster1ModelRuntime._configure_active_mask(
        forward_batch, active_length=32, device=torch.device("cpu")
    )
    assert forward_batch.dllm_bidir_mask_backend == "custom"
    assert forward_batch.dllm_force_bidir_mask is True
    assert forward_batch.dllm_bidir_custom_mask.dtype is torch.bool
    assert tuple(forward_batch.dllm_bidir_custom_mask.shape) == (32, 32)


def test_exporter_uses_bounded_server_memory_settings():
    args = argparse.Namespace(model_dir="/model", dtype="bfloat16", tp_size=1)
    values = MODULE.Cluster1ModelRuntime._server_args_kwargs(args, Path("config.yaml"))
    assert values["max_running_requests"] == 1
    assert values["max_total_tokens"] == 4096


def test_exporter_rejects_cases_above_token_budget():
    runtime = MODULE.Cluster1ModelRuntime.__new__(MODULE.Cluster1ModelRuntime)
    case = MODULE.ManifestCase(
        schema_version=1,
        case_id="too-large",
        token_seed=1,
        prefix_length=4090,
        active_length=32,
        diffusion_steps=1,
        attention_contract_id=MODULE.ATTENTION_CONTRACT,
    )
    with pytest.raises(ValueError, match="exceeds exporter token budget"):
        runtime.run_case(case)


def test_active_boundary_is_set_before_forward_batch_construction():
    runtime, observed_offsets, _positions = make_active_runtime()
    req = make_active_request(prefix_length=256)

    runtime._run_active(req, list(range(32)), FakeTraceHooks(), step=1)

    assert observed_offsets == [256]
    assert req.dllm_block_offset == 256


@pytest.mark.parametrize(
    "positions",
    [
        torch.arange(0, 32),
        torch.arange(255, 287),
        torch.cat((torch.tensor([256]), torch.arange(258, 289))),
        torch.arange(256, 287),
    ],
    ids=(
        "starts-at-zero",
        "starts-before-boundary",
        "non-contiguous-after-correct-first",
        "incorrect-count",
    ),
)
def test_exact_active_position_vector_rejects_invalid_positions(positions):
    forward_batch = types.SimpleNamespace(positions=positions)
    with pytest.raises(RuntimeError, match="active positions mismatch"):
        MODULE.Cluster1ModelRuntime._validate_active_positions(
            forward_batch, prefix_length=256, active_length=32
        )


def test_exact_active_position_vector_accepts_boundary_interval():
    expected = torch.arange(256, 288)
    forward_batch = types.SimpleNamespace(positions=expected.clone())
    MODULE.Cluster1ModelRuntime._validate_active_positions(
        forward_batch, prefix_length=256, active_length=32
    )
    assert torch.equal(forward_batch.positions, expected)


def test_repeated_diffusion_steps_remain_at_sealed_boundary():
    runtime, observed_offsets, observed_positions = make_active_runtime()
    req = make_active_request(prefix_length=256)
    active_tokens = list(range(32))

    runtime._run_active(req, active_tokens, FakeTraceHooks(), step=1)
    runtime._run_active(req, active_tokens, FakeTraceHooks(), step=2)

    assert observed_offsets == [256, 256]
    assert all(
        torch.equal(positions, torch.arange(256, 288))
        for positions in observed_positions
    )
    assert req.dllm_block_offset != 256 + 32


def test_reference_and_cached_requests_use_identical_active_positions():
    runtime, observed_offsets, observed_positions = make_active_runtime()
    reference_req = make_active_request(prefix_length=256)
    cached_req = make_active_request(prefix_length=256)
    active_tokens = list(range(32))

    runtime._run_active(reference_req, active_tokens, FakeTraceHooks(), step=1)
    runtime._run_active(cached_req, active_tokens, FakeTraceHooks(), step=1)

    assert observed_offsets == [256, 256]
    assert torch.equal(observed_positions[0], observed_positions[1])
    assert torch.equal(observed_positions[0], torch.arange(256, 288))


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
    assert all(record["attention_mask_backend"] == "custom_paged" for record in records)


def test_result_rejects_missing_custom_paged_backend():
    case = make_case()
    result = make_result(case)
    result.pop("attention_mask_backend")
    with pytest.raises(ValueError, match="attention_mask_backend=custom_paged"):
        MODULE.validate_result(result, case)


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


def test_hook_capture_is_contiguous_independent_and_non_mutating():
    class Layer(torch.nn.Module):
        def forward(self, hidden, residual):
            return hidden, residual

    layer = Layer()
    runner = types.SimpleNamespace(
        model=types.SimpleNamespace(layers=torch.nn.ModuleList([layer])),
        req_to_token_pool=types.SimpleNamespace(mamba_map={}),
    )
    hooks = MODULE.ModelTraceHooks(runner)
    hidden = torch.arange(24, dtype=torch.float32).view(3, 8).transpose(0, 1)
    residual = torch.arange(24, 48, dtype=torch.float32).view(3, 8).transpose(0, 1)
    hidden_before = hidden.clone()
    residual_before = residual.clone()
    expected = hidden[-2:].clone() + residual[-2:].clone()

    with hooks.capture(active_length=2, mamba_cache_idx=1):
        layer(hidden, residual)
    trace = hooks.finish_step(1, torch.zeros(2, 5))
    captured = trace.layers[0].active_hidden

    assert captured.is_contiguous()
    assert torch.equal(captured, expected)
    assert torch.equal(hidden, hidden_before)
    assert torch.equal(residual, residual_before)
    hidden[-2:].zero_()
    residual[-2:].zero_()
    assert torch.equal(captured, expected)
    assert captured.data_ptr() != hidden.data_ptr()
    trace.release()
    hooks.close()


def test_tensor_comparison_rejects_missing_or_nonfinite_data():
    with pytest.raises(ValueError, match="shape mismatch"):
        MODULE.tensor_max_abs(torch.zeros(1), torch.zeros(2), "probe")
    with pytest.raises(ValueError, match="NaN or Inf"):
        MODULE.tensor_max_abs(torch.tensor([math.nan]), torch.zeros(1), "probe")


def test_close_clears_cache_and_destroys_initialized_process_group(monkeypatch):
    events = []

    class FakeDistributed:
        initialized = True

        @staticmethod
        def is_available():
            return True

        @classmethod
        def is_initialized(cls):
            return cls.initialized

        @classmethod
        def destroy_process_group(cls):
            events.append("destroy")
            cls.initialized = False

    cache = types.SimpleNamespace(clear=lambda: events.append("clear"))
    runtime = MODULE.Cluster1ModelRuntime.__new__(MODULE.Cluster1ModelRuntime)
    runtime.backend = types.SimpleNamespace(region_state_cache=cache)
    monkeypatch.setattr(
        MODULE,
        "_import_torch",
        lambda: types.SimpleNamespace(distributed=FakeDistributed),
    )

    runtime.close()
    runtime.close()

    assert events == ["clear", "destroy", "clear"]


def test_close_is_safe_when_process_group_is_not_initialized(monkeypatch):
    events = []
    distributed = types.SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: False,
        destroy_process_group=lambda: events.append("destroy"),
    )
    runtime = MODULE.Cluster1ModelRuntime.__new__(MODULE.Cluster1ModelRuntime)
    runtime.backend = types.SimpleNamespace(
        region_state_cache=types.SimpleNamespace(clear=lambda: events.append("clear"))
    )
    monkeypatch.setattr(
        MODULE,
        "_import_torch",
        lambda: types.SimpleNamespace(distributed=distributed),
    )

    runtime.close()

    assert events == ["clear"]


def test_primary_exception_is_not_replaced_by_empty_cache_failure():
    events = []

    class Hooks:
        released = False

        def close(self):
            self.released = True
            events.append("hooks")

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: (_ for _ in ()).throw(
                RuntimeError("secondary empty_cache failure")
            ),
        )
    )

    with pytest.raises(RuntimeError, match="primary execution failure"):
        try:
            raise RuntimeError("primary execution failure")
        finally:
            MODULE.Cluster1ModelRuntime._cleanup_case_resources(
                [], [], Hooks(), fake_torch, case_completed=False
            )

    assert events == ["hooks"]
