import importlib.util
import subprocess
import sys
from collections import OrderedDict, namedtuple
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "eval/scripts/cluster3_gdn_equivalence_diagnostic.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "cluster3_gdn_equivalence_diagnostic_test", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


def fused_qkvzba_split_reshape_cat_contiguous(qkvz, ba, *_args):
    """Test stand-in patched by LayerZeroEvidence like the Qwen CUDA helper."""
    return qkvz, qkvz, ba, ba


class FakeLinearAttention(torch.nn.Module):
    def __init__(self, *, out_proj_keyword=False):
        super().__init__()
        self.in_proj_qkvz = torch.nn.Identity()
        self.in_proj_ba = torch.nn.Identity()
        self.attn = torch.nn.Identity()
        self.out_proj = torch.nn.Identity()
        self.out_proj_keyword = out_proj_keyword

    def forward(self, hidden_states):
        qkvz = self.in_proj_qkvz(hidden_states)
        ba = self.in_proj_ba(hidden_states)
        mixed_qkv, _z, _b, _a = fused_qkvzba_split_reshape_cat_contiguous(qkvz, ba)
        core = self.attn(mixed_qkv)
        if self.out_proj_keyword:
            return self.out_proj(input=core)
        return self.out_proj(core)


class FakeDecoderLayer(torch.nn.Module):
    layer_id = 0

    def __init__(self, *, out_proj_keyword=False):
        super().__init__()
        self.input_layernorm = torch.nn.Identity()
        self.linear_attn = FakeLinearAttention(out_proj_keyword=out_proj_keyword)

    def forward(self, hidden_states):
        hidden = self.input_layernorm(hidden_states)
        hidden = self.linear_attn(hidden)
        return hidden, torch.zeros_like(hidden)


class FakeBackend:
    def _restore_region_dag_layer_snapshot(self, **_kwargs):
        return None


class ProvenanceKey(
    namedtuple(
        "ProvenanceKeyBase",
        (
            "request_id",
            "request_pool_idx",
            "request_slot_generation",
            "boundary",
            "preceding_regions",
        ),
    )
):
    __slots__ = ()

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "request_pool_idx": self.request_pool_idx,
            "request_slot_generation": self.request_slot_generation,
            "boundary": self.boundary,
            "preceding_regions": self.preceding_regions,
        }


class FakeProvenanceBackend:
    def __init__(self, pool):
        self.pool = pool
        self.gdn_layer_ids = [0, 1]
        self._region_dag_layer_snapshots = OrderedDict()

    def _current_mamba_slot(self, request_pool_idx):
        return self.pool.mapping[int(request_pool_idx)]

    def _put_region_dag_layer_snapshot(
        self, *, frontier_key, layer_id, conv_state, recurrent_state
    ):
        self._region_dag_layer_snapshots[(frontier_key, int(layer_id))] = (
            SimpleNamespace(
                frontier_key=frontier_key,
                layer_id=int(layer_id),
                conv_state=conv_state.detach().clone(),
                recurrent_state=recurrent_state.detach().clone(),
            )
        )

    def _restore_region_dag_layer_snapshot(
        self,
        *,
        frontier_key,
        layer_id,
        conv_destination,
        recurrent_destination,
    ):
        snapshot = self._region_dag_layer_snapshots[(frontier_key, int(layer_id))]
        conv_destination.copy_(snapshot.conv_state)
        recurrent_destination.copy_(snapshot.recurrent_state)


class FakeProvenancePool:
    def __init__(self):
        self.mapping = {1: 2, 2: 3}
        self.layers = {
            layer_id: SimpleNamespace(
                conv=[torch.zeros(5, 2, 3)],
                temporal=torch.zeros(5, 2, 2),
            )
            for layer_id in (0, 1)
        }

    def mamba2_layer_cache(self, layer_id):
        return self.layers[int(layer_id)]


def provenance_key(request_id, request_pool_idx, generation, boundary=64):
    return ProvenanceKey(
        request_id=request_id,
        request_pool_idx=request_pool_idx,
        request_slot_generation=generation,
        boundary=boundary,
        preceding_regions=(("A", 3, (("root", 2),), "token-sha", "position-sha"),),
    )


def make_provenance_runtime():
    pool = FakeProvenancePool()
    backend = FakeProvenanceBackend(pool)
    runtime = SimpleNamespace(
        backend=backend,
        model_runner=SimpleNamespace(req_to_token_pool=pool),
    )
    return runtime, pool, backend


def make_hook_runtime(*, out_proj_keyword=False):
    layer = FakeDecoderLayer(out_proj_keyword=out_proj_keyword)

    class ModelTraceHooks:
        @staticmethod
        def _language_model(model):
            return model

    runtime = SimpleNamespace(
        cluster1=SimpleNamespace(ModelTraceHooks=ModelTraceHooks),
        model_runner=SimpleNamespace(model=SimpleNamespace(layers=[layer])),
        backend=FakeBackend(),
    )
    return runtime, layer


def run_hook_capture(*, layer_keyword=False, out_proj_keyword=False):
    runtime, layer = make_hook_runtime(out_proj_keyword=out_proj_keyword)
    hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    with MODULE.LayerZeroEvidence(runtime) as evidence:
        with evidence.capture("test", rows=3):
            if layer_keyword:
                layer(hidden_states=hidden)
            else:
                layer(hidden)
        values = dict(evidence.values["test"])
    return hidden, values, evidence, layer


def hook_count(layer):
    modules = (
        layer,
        layer.input_layernorm,
        layer.linear_attn,
        layer.linear_attn.in_proj_qkvz,
        layer.linear_attn.in_proj_ba,
        layer.linear_attn.attn,
        layer.linear_attn.out_proj,
    )
    return sum(
        len(module._forward_pre_hooks) + len(module._forward_hooks)
        for module in modules
    )


def test_cli_help_does_not_load_model_or_cuda():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--output-dir" in completed.stdout
    assert "freshly segmented" in completed.stdout


def test_decoder_layer_pre_hook_accepts_entirely_positional_invocation():
    hidden, values, _evidence, _layer = run_hook_capture()
    assert torch.equal(values["decoder_layer_input_hidden_states"], hidden)


def test_decoder_layer_pre_hook_accepts_entirely_keyword_invocation():
    hidden, values, _evidence, _layer = run_hook_capture(layer_keyword=True)
    assert torch.equal(values["decoder_layer_input_hidden_states"], hidden)


def test_hook_tensor_resolver_prioritizes_hidden_states_keyword():
    positional = torch.zeros(2, 3)
    preferred = torch.ones(2, 3)
    resolved = MODULE._resolve_hook_tensor(
        (positional,),
        {"hidden_states": preferred},
        preferred_keyword="hidden_states",
        evidence_name="decoder_layer_input_hidden_states",
    )
    assert resolved is preferred


def test_empty_positional_inputs_use_keyword_tensor_without_index_error():
    hidden = torch.ones(2, 3)
    resolved = MODULE._resolve_hook_tensor(
        (),
        {"hidden_states": hidden},
        preferred_keyword="hidden_states",
        evidence_name="decoder_layer_input_hidden_states",
    )
    assert resolved is hidden


def test_hook_tensor_resolver_fails_closed_when_no_tensor_exists():
    with pytest.raises(
        RuntimeError,
        match=(
            "decoder_layer_input_hidden_states hook received no tensor input; "
            "positional_count=0, keyword_keys=\\['hidden_states'\\]"
        ),
    ):
        MODULE._resolve_hook_tensor(
            (),
            {"hidden_states": "not-a-tensor"},
            preferred_keyword="hidden_states",
            evidence_name="decoder_layer_input_hidden_states",
        )


def test_out_proj_pre_hook_accepts_positional_input():
    hidden, values, _evidence, _layer = run_hook_capture()
    assert torch.equal(values["gdn_output_before_output_projection"], hidden)


def test_out_proj_pre_hook_accepts_keyword_input():
    hidden, values, _evidence, _layer = run_hook_capture(out_proj_keyword=True)
    assert torch.equal(values["gdn_output_before_output_projection"], hidden)


def test_layer_zero_hooks_are_removed_after_normal_completion():
    _hidden, _values, evidence, layer = run_hook_capture(
        layer_keyword=True, out_proj_keyword=True
    )
    assert evidence.released
    assert hook_count(layer) == 0


def test_layer_zero_hooks_are_removed_after_capture_exception():
    runtime, layer = make_hook_runtime()
    with pytest.raises(RuntimeError, match="received no tensor input"):
        with MODULE.LayerZeroEvidence(runtime) as evidence:
            with evidence.capture("test", rows=3):
                layer(hidden_states="not-a-tensor")
    assert evidence.released
    assert hook_count(layer) == 0


def test_frontier_provenance_records_exact_l_s_r_d_without_aliasing():
    runtime, pool, backend = make_provenance_runtime()
    live_key = provenance_key("B", 1, 11)
    cached_key = provenance_key("C", 2, 12)
    req = SimpleNamespace(
        req_pool_idx=1,
        hybrid_request_slot_generation=11,
        region_dag_frontier_keys={64: live_key},
    )
    expected = {}
    for layer_id in backend.gdn_layer_ids:
        cache = pool.mamba2_layer_cache(layer_id)
        conv = torch.arange(6, dtype=torch.float32).reshape(2, 3) + layer_id
        recurrent = torch.arange(4, dtype=torch.float32).reshape(2, 2) + layer_id
        cache.conv[0][2].copy_(conv)
        cache.temporal[2].copy_(recurrent)
        cache.conv[0][3].copy_(conv)
        cache.temporal[3].copy_(recurrent)
        expected[layer_id] = (conv.clone(), recurrent.clone())

    with MODULE.FrontierStateProvenance(runtime, 64) as provenance:
        provenance.start_step(0)
        provenance.record_live_prefix(req)
        with provenance.phase("snapshot"):
            for layer_id in backend.gdn_layer_ids:
                cache = pool.mamba2_layer_cache(layer_id)
                backend._put_region_dag_layer_snapshot(
                    frontier_key=cached_key,
                    layer_id=layer_id,
                    conv_state=cache.conv[0][3],
                    recurrent_state=cache.temporal[3],
                )

        for layer_id in backend.gdn_layer_ids:
            cache = pool.mamba2_layer_cache(layer_id)
            cache.conv[0][3].fill_(-1)
            cache.temporal[3].fill_(-1)
        with provenance.phase("restore"):
            for layer_id in reversed(backend.gdn_layer_ids):
                cache = pool.mamba2_layer_cache(layer_id)
                backend._restore_region_dag_layer_snapshot(
                    frontier_key=cached_key,
                    layer_id=layer_id,
                    conv_destination=cache.conv[0][3],
                    recurrent_destination=cache.temporal[3],
                )
        for layer_id in backend.gdn_layer_ids:
            snapshot = backend._region_dag_layer_snapshots[(cached_key, layer_id)]
            snapshot.conv_state.fill_(999)
            snapshot.recurrent_state.fill_(999)
            cache = pool.mamba2_layer_cache(layer_id)
            assert torch.equal(cache.conv[0][3], expected[layer_id][0])
            assert torch.equal(cache.temporal[3], expected[layer_id][1])

        result = provenance.result()
        assert result["all_bitwise_equal"]
        assert result["first_unequal_checkpoint"] is None
        for layer_id in backend.gdn_layer_ids:
            for state_kind in ("convolution", "recurrent"):
                checkpoints = result["checkpoints"][str(layer_id)][state_kind]
                assert set(checkpoints) == {"L", "S", "R", "D"}
                assert len({checkpoints[name]["sha256"] for name in checkpoints}) == 1
                assert checkpoints["S"]["snapshot_generation"] > 0
    assert provenance.released


def test_frontier_provenance_reports_first_bitwise_mismatch():
    runtime, pool, backend = make_provenance_runtime()
    live_key = provenance_key("B", 1, 11)
    cached_key = provenance_key("C", 2, 12)
    req = SimpleNamespace(
        req_pool_idx=1,
        hybrid_request_slot_generation=11,
        region_dag_frontier_keys={64: live_key},
    )
    for layer_id in backend.gdn_layer_ids:
        cache = pool.mamba2_layer_cache(layer_id)
        cache.conv[0][2].fill_(1)
        cache.temporal[2].fill_(1)
        cache.conv[0][3].fill_(1)
        cache.temporal[3].fill_(1)
    pool.mamba2_layer_cache(0).conv[0][3, 0, 0] = 2

    with MODULE.FrontierStateProvenance(runtime, 64) as provenance:
        provenance.start_step(0)
        provenance.record_live_prefix(req)
        with provenance.phase("snapshot"):
            for layer_id in backend.gdn_layer_ids:
                cache = pool.mamba2_layer_cache(layer_id)
                backend._put_region_dag_layer_snapshot(
                    frontier_key=cached_key,
                    layer_id=layer_id,
                    conv_state=cache.conv[0][3],
                    recurrent_state=cache.temporal[3],
                )
        with provenance.phase("restore"):
            for layer_id in backend.gdn_layer_ids:
                cache = pool.mamba2_layer_cache(layer_id)
                backend._restore_region_dag_layer_snapshot(
                    frontier_key=cached_key,
                    layer_id=layer_id,
                    conv_destination=cache.conv[0][3],
                    recurrent_destination=cache.temporal[3],
                )
        result = provenance.result()
    assert result["first_unequal_checkpoint"] == {
        "layer_id": 0,
        "state_kind": "convolution",
        "comparison": "L_vs_S",
    }


def test_segmented_reference_freshly_recomputes_prefix_every_time():
    events = []
    generation = {"value": 0}

    def run_once():
        def clear_state():
            events.append("clear")

        def create_request():
            generation["value"] += 1
            request = f"request-{generation['value']}"
            events.append(("create", request))
            return request

        def recompute_prefix(request):
            events.append(("prefix", request))
            return "fresh-prefix"

        def prepare_suffix(request, prefix):
            events.append(("prepare", request, prefix))
            return type("Prepared", (), {"restore_required": False})()

        def execute_suffix(request, _prepared):
            events.append(("suffix", request))
            return type("Result", (), {"restore_calls": 0, "request": request})()

        return MODULE.execute_fresh_segmented_reference(
            clear_state=clear_state,
            create_request=create_request,
            recompute_prefix=recompute_prefix,
            region_cache_entry_count=lambda: 0,
            prepare_suffix=prepare_suffix,
            execute_suffix=execute_suffix,
        )

    first = run_once()
    second = run_once()
    assert first.request != second.request
    assert events.count("clear") == 2
    assert [
        event for event in events if isinstance(event, tuple) and event[0] == "prefix"
    ] == [
        ("prefix", "request-1"),
        ("prefix", "request-2"),
    ]


def test_segmented_reference_fails_if_prefix_publishes_region_cache_state():
    cache_entries = {"value": 0}
    suffix_called = False

    def recompute_prefix(_request):
        cache_entries["value"] = 1

    def execute_suffix(_request, _prepared):
        nonlocal suffix_called
        suffix_called = True

    with pytest.raises(RuntimeError, match="unexpectedly published"):
        MODULE.execute_fresh_segmented_reference(
            clear_state=lambda: cache_entries.update(value=0),
            create_request=lambda: object(),
            recompute_prefix=recompute_prefix,
            region_cache_entry_count=lambda: cache_entries["value"],
            prepare_suffix=lambda _request, _prefix: None,
            execute_suffix=execute_suffix,
        )
    assert not suffix_called


def test_segmented_reference_fails_on_restore_request_or_restore_call():
    prepared = type("Prepared", (), {"restore_required": True})()
    with pytest.raises(RuntimeError, match="requested a cache restoration"):
        MODULE.execute_fresh_segmented_reference(
            clear_state=lambda: None,
            create_request=lambda: object(),
            recompute_prefix=lambda _request: None,
            region_cache_entry_count=lambda: 0,
            prepare_suffix=lambda _request, _prefix: prepared,
            execute_suffix=lambda _request, _prepared: None,
        )

    prepared.restore_required = False
    restored = type("Result", (), {"restore_calls": 1})()
    with pytest.raises(RuntimeError, match="observed a cache restoration"):
        MODULE.execute_fresh_segmented_reference(
            clear_state=lambda: None,
            create_request=lambda: object(),
            recompute_prefix=lambda _request: None,
            region_cache_entry_count=lambda: 0,
            prepare_suffix=lambda _request, _prefix: prepared,
            execute_suffix=lambda _request, _prepared: restored,
        )


def test_tensor_diagnostic_records_required_numerical_evidence():
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    right = left.clone()
    right[1, 0] += torch.tensor(0.25, dtype=torch.bfloat16)
    evidence = MODULE._tensor_difference(left, right)
    assert evidence["reference"]["shape"] == [2, 2]
    assert evidence["reference"]["dtype"] == "torch.bfloat16"
    assert len(evidence["reference"]["sha256"]) == 64
    assert not evidence["exact_equal"]
    assert evidence["max_abs_error"] == 0.25
    assert evidence["mean_abs_error"] == 0.0625
    assert evidence["first_mismatching_flat_index"] == 2
    assert evidence["values_above_1e_2"] == 1
