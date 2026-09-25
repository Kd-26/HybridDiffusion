import ast
import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VALIDATION = load(
    "cluster3_component_ablation_validation",
    ROOT / "eval/scripts/cluster3_region_dag_validation.py",
)
BENCHMARK = load(
    "cluster3_component_ablation_benchmark",
    ROOT / "eval/scripts/cluster3_production_latency_benchmark.py",
)
SWEEP = load(
    "cluster3_component_ablation_sweep_tested",
    ROOT / "eval/scripts/cluster3_component_ablation_sweep.py",
)
AGGREGATE = load(
    "cluster3_component_ablation_aggregate_tested",
    ROOT / "eval/scripts/cluster3_component_ablation_aggregate.py",
)


def production_argv(*extra):
    return (
        "--model-path",
        "/model",
        "--output-jsonl",
        "/tmp/raw.jsonl",
        "--summary-json",
        "/tmp/summary.json",
        "--profile",
        "production_efficiency",
        "--correctness-artifact",
        "/tmp/correctness.json",
        "--preflight-json",
        "/tmp/preflight.json",
        *extra,
    )


def test_policy_parsing_default_environment_and_cli_precedence(monkeypatch):
    monkeypatch.delenv("HYBRID_ABLATION_POLICY", raising=False)
    default = VALIDATION.parse_args(production_argv())
    assert default.ablation_policy is None
    assert tuple(default.routes) == (
        "full_replay",
        "cold_handoff_build",
        "warm_cached_suffix",
    )

    monkeypatch.setenv("HYBRID_ABLATION_POLICY", "gdn_only")
    assert VALIDATION.parse_args(production_argv()).ablation_policy == "gdn_only"
    assert (
        VALIDATION.parse_args(
            production_argv("--ablation-policy", "full_replay")
        ).ablation_policy
        == "full_replay"
    )


def test_invalid_environment_policy_fails_closed(monkeypatch):
    monkeypatch.setenv("HYBRID_ABLATION_POLICY", "not-a-policy")
    with pytest.raises(ValueError, match="HYBRID_ABLATION_POLICY"):
        VALIDATION.parse_args(production_argv())


def test_attention_and_gdn_restore_controls_are_independent():
    kv_only = VALIDATION.resolve_region_state_restore_controls(
        restore_attention_state=True, restore_gdn_state=False
    )
    gdn_only = VALIDATION.resolve_region_state_restore_controls(
        restore_attention_state=False, restore_gdn_state=True
    )
    assert (kv_only.attention_state, kv_only.gdn_state) == (True, False)
    assert (gdn_only.attention_state, gdn_only.gdn_state) == (False, True)
    assert VALIDATION.resolve_region_state_restore_controls(restore=True) == (
        VALIDATION.RegionStateRestoreControls(True, True)
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        VALIDATION.resolve_region_state_restore_controls(
            restore=True,
            restore_attention_state=True,
            restore_gdn_state=True,
        )


def test_ablation_prepare_passes_independent_switches_to_runtime(monkeypatch):
    observed = {}

    class Runtime:
        def _prepare_frontier_suffix(self, *_args, **kwargs):
            observed.update(kwargs)
            return object(), {}

    monkeypatch.setattr(
        BENCHMARK, "_prepare_call", lambda _timer, operation: operation()
    )
    BENCHMARK._prepare_ablation_suffix(
        Runtime(),
        object(),
        [1, 2],
        object(),
        object(),
        object(),
        attention_state_reused=True,
        gdn_state_reused=False,
    )
    assert observed["restore_attention_state"] is True
    assert observed["restore_gdn_state"] is False
    assert "restore" not in observed


@pytest.mark.parametrize(
    ("policy", "attention", "gdn", "kv", "gdn_restored"),
    (
        ("full_replay", 200, 200, 0, 0),
        ("kv_only", 40, 200, 160, 0),
        ("gdn_only", 200, 40, 0, 160),
        ("kv_gdn_conservative", 40, 40, 160, 160),
        ("kv_gdn_oracle", 40, 20, 160, 180),
    ),
)
def test_expected_work_counter_pattern_for_every_fixed_policy(
    policy, attention, gdn, kv, gdn_restored
):
    work = BENCHMARK.expected_ablation_work(
        policy,
        sequence_length=100,
        conservative_replay_positions=20,
        oracle_replay_positions=10 if policy == "kv_gdn_oracle" else None,
        diffusion_steps=2,
    )
    assert work == {
        "executed_attention_query_positions": attention,
        "executed_gdn_replay_positions": gdn,
        "restored_kv_positions": kv,
        "restored_gdn_positions": gdn_restored,
    }


def test_oracle_replay_never_exceeds_conservative_replay():
    conservative = BENCHMARK.expected_ablation_work(
        "kv_gdn_conservative",
        sequence_length=100,
        conservative_replay_positions=40,
        diffusion_steps=3,
    )
    oracle = BENCHMARK.expected_ablation_work(
        "kv_gdn_oracle",
        sequence_length=100,
        conservative_replay_positions=40,
        oracle_replay_positions=12,
        diffusion_steps=3,
    )
    assert (
        oracle["executed_gdn_replay_positions"]
        <= conservative["executed_gdn_replay_positions"]
    )
    with pytest.raises(ValueError, match="no larger than conservative"):
        BENCHMARK.expected_ablation_work(
            "kv_gdn_oracle",
            sequence_length=100,
            conservative_replay_positions=40,
            oracle_replay_positions=41,
            diffusion_steps=1,
        )


@pytest.mark.parametrize("policy", ("kv_only", "gdn_only", "kv_gdn_oracle"))
def test_unimplementable_policies_fail_with_clear_runtime_reason(policy):
    with pytest.raises(
        BENCHMARK.UnsupportedAblationPolicyError,
        match=f"ablation policy {policy!r} is not implementable",
    ):
        BENCHMARK.require_implementable_ablation_policy(policy)


def test_missing_oracle_compatible_state_is_explicit():
    with pytest.raises(
        BENCHMARK.UnsupportedAblationPolicyError,
        match="independently restorable per-region GDN state",
    ):
        BENCHMARK.expected_ablation_work(
            "kv_gdn_oracle",
            sequence_length=100,
            conservative_replay_positions=40,
            diffusion_steps=1,
        )


def test_output_equivalence_checks_ids_even_when_hash_label_matches():
    reference = {
        "output_hash": "a" * 64,
        "generated_top1_token_ids": [1, 2, 3],
    }
    BENCHMARK._require_paired_output_hash(
        reference,
        dict(reference),
        route="kv_gdn_conservative",
        variant="uninstrumented",
        repetition=0,
    )
    candidate = dict(reference)
    candidate["generated_top1_token_ids"] = [1, 9, 3]
    with pytest.raises(RuntimeError, match="token IDs differ"):
        BENCHMARK._require_paired_output_hash(
            reference,
            candidate,
            route="kv_gdn_conservative",
            variant="uninstrumented",
            repetition=0,
        )


def test_capacity_policy_avoids_the_failed_4160_slot_configuration():
    assert BENCHMARK.capacity_for_prefix(256) == 8192
    assert BENCHMARK.capacity_for_prefix(1024) == 8192
    assert BENCHMARK.capacity_for_prefix(4096) == 16384


def test_fixed_grid_is_twelve_cases_and_sixty_isolated_jobs():
    cases = SWEEP._grid_cases()
    assert len(cases) == 12
    assert len(cases) * len(SWEEP.FIXED_POLICIES) == 60
    assert {
        (
            case["prefix_tokens"],
            case["active_spans"][0][1] - case["active_spans"][0][0],
            case["diffusion_steps"],
        )
        for case in cases
    } == {
        (prefix, active, steps)
        for prefix in (256, 1024, 4096)
        for active in (16, 64)
        for steps in (2, 8)
    }
    assert all(
        SWEEP._capacity(case) == (16384 if case["prefix_tokens"] == 4096 else 8192)
        for case in cases
    )


def test_complete_sweep_refuses_to_start_before_smoke_gate(tmp_path):
    smoke = tmp_path / "smoke.json"
    smoke.write_text('{"smoke_gate_pass": false}\n')
    args = SimpleNamespace(
        mode="sweep",
        smoke_summary=smoke,
        output_dir=tmp_path / "sweep",
    )
    with pytest.raises(RuntimeError, match="refusing the 60-job sweep"):
        SWEEP.run(args)
    assert not args.output_dir.exists()


def test_unsupported_policy_writes_a_fail_closed_summary(tmp_path):
    args = SimpleNamespace(
        output_jsonl=tmp_path / "measurements.jsonl",
        summary_json=tmp_path / "summary.json",
    )
    spec = BENCHMARK.ablation_policy_spec("kv_gdn_oracle")
    BENCHMARK._write_unsupported_ablation_artifacts(args, spec)
    summary = __import__("json").loads(args.summary_json.read_text())
    assert summary["unsupported_policies"][0]["policy"] == "kv_gdn_oracle"
    assert summary["correctness_pass"] is False
    assert summary["all_publication_gates_pass"] is False
    assert args.output_jsonl.read_text() == ""


def test_paired_confidence_interval_resamples_pairs_deterministically():
    reference = [10.0, 20.0, 30.0, 40.0]
    candidate = [5.0, 15.0, 25.0, 35.0]
    first = AGGREGATE.paired_bootstrap_ci(
        reference, candidate, AGGREGATE._mean_difference, seed=7
    )
    second = AGGREGATE.paired_bootstrap_ci(
        reference, candidate, AGGREGATE._mean_difference, seed=7
    )
    assert first == second == [5.0, 5.0]


def test_allocator_diagnostic_is_safe_for_simplenamespace_cache():
    source = (ROOT / "eval/sglang/srt/mem_cache/common.py").read_text()
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.ImportFrom, ast.FunctionDef))
        and (
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
            or isinstance(node, ast.FunctionDef)
            and node.name in ("available_and_evictable_str", "safe_pretty_print")
        )
    ]
    namespace = {
        "BasePrefixCache": object,
        "logger": logging.getLogger("allocator-diagnostic-test"),
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "common.py", "exec"),
        namespace,
    )
    cache = SimpleNamespace(
        token_to_kv_pool_allocator=SimpleNamespace(available_size=lambda: 17)
    )
    assert "allocator_available=17" in namespace["available_and_evictable_str"](cache)
    namespace["safe_pretty_print"](cache)
