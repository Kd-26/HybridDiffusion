import argparse
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MODULE = load(
    "cluster3_cache_system_comparison_tested",
    ROOT / "eval/scripts/cluster3_cache_system_comparison.py",
)


def raw_record(method="controlled_full_replay", *, variant="minimally_profiled"):
    reused = method != "controlled_full_replay"
    restore = method == "complete_region_state_execution"
    return {
        "variant": variant,
        "repetition_index": 0,
        "timing": {
            "production_route_total_cuda_ms": 100.0,
            "production_prepare_cuda_ms": (
                10.0 if variant == "minimally_profiled" else None
            ),
            "production_model_cuda_ms": (
                85.0 if variant == "minimally_profiled" else None
            ),
            "synchronization_count": 1,
        },
        "component_timing": (
            {
                "full_attention": 20.0,
                "gdn_replay": 25.0,
                "mlp": 30.0,
                "prefix_snapshot": 2.0,
                "cache_lookup_restore": 1.0 if restore else None,
            }
            if variant == "minimally_profiled"
            else None
        ),
        "generated_top1_token_ids": [1, 2, 3],
        "output_hash": MODULE._hash_token_ids([1, 2, 3]),
        "positions_preserved": True,
        "stable_state_unchanged": True,
        "work_accounting_matches": True,
        "executed_query_positions": 32,
        "full_attention_query_token_layer_positions": 192,
        "gdn_replay_token_layer_positions": 576,
        "cache_hits": 128 if reused else 0,
        "kv_reuse_evidence": {"physical_locations_match": True},
        "gdn_state_restores": 18 if restore else 0,
        "reusable_prefix_positions": 32,
        "fallback_count": 0,
        "recovery_replay_count": 0,
        "peak_allocated_gpu_memory_bytes": 1024,
    }


def normalize(method="controlled_full_replay", **changes):
    raw = raw_record(method, variant=changes.pop("variant", "minimally_profiled"))
    for key, value in changes.items():
        if key.startswith("timing."):
            raw["timing"][key.split(".", 1)[1]] = value
        else:
            raw[key] = value
    return MODULE.normalize_record(
        raw, method=method, case_id="case", revision="revision", warmup=False
    )


def test_exact_cli_and_method_names():
    help_text = MODULE.build_parser().format_help()
    assert MODULE.METHODS == (
        "controlled_full_replay",
        "upstream_sglang_radix",
        "safe_kv_only_diffusion",
        "kv_gdn_handoff_full_rows",
        "complete_region_state_execution",
    )
    assert all(name in help_text for name in MODULE.METHODS)


def test_manifest_has_exact_three_primary_cases_and_valid_dag():
    value = json.loads(
        (ROOT / "eval/manifests/cluster3_cache_system_comparison.json").read_text()
    )
    assert [case["total_tokens_per_request"] for case in value["cases"]] == [
        2080,
        1152,
        1536,
    ]
    assert [case["diffusion_steps"] for case in value["cases"]] == [4, 4, 4]
    assert [case["batch_size"] for case in value["cases"]] == [1, 1, 1]
    assert value["cases"][0]["active_spans"] == [[2048, 2080]]
    assert value["cases"][1]["active_spans"] == [[1024, 1152]]
    assert value["cases"][2]["active_spans"] == [
        [256, 272],
        [640, 656],
        [1024, 1040],
        [1280, 1296],
    ]
    code = f"""
import importlib.util, sys
from pathlib import Path
root = Path({str(ROOT)!r})
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
region = root / 'eval/sglang/srt/dllm/region'
load('sglang.srt.dllm.region.execution_spec', region / 'execution_spec.py')
load('sglang.srt.dllm.region.dependency_graph', region / 'dependency_graph.py')
load('sglang.srt.dllm.region.runtime', region / 'runtime.py')
validation = load('comparison_validation', root / 'eval/scripts/cluster3_region_dag_validation.py')
manifest = load('comparison_manifest', root / 'eval/scripts/cluster3_benchmark_manifest.py')
cases, normalized = manifest.load_manifest(root / 'eval/manifests/cluster3_cache_system_comparison.json', validation_module=validation, seed=17, requested_routes={MODULE.METHODS!r})
assert len(cases) == len(normalized) == 3
"""
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_require_all_fails_before_model_factory():
    called = []
    args = argparse.Namespace(
        methods=list(MODULE.METHODS),
        require_all_methods=True,
        case_manifest=Path("missing"),
    )
    with pytest.raises(RuntimeError, match="requested methods are unavailable"):
        MODULE.run(args, runtime_factory=lambda _args: called.append(True))
    assert called == []


def test_unsupported_methods_have_metadata_and_no_route_alias():
    entries = MODULE.validate_requested_methods(MODULE.METHODS, require_all=False)
    unsupported = {entry["method"] for entry in entries if not entry["supported"]}
    assert unsupported == {"upstream_sglang_radix", "kv_gdn_handoff_full_rows"}
    for method in unsupported:
        assert MODULE.METHOD_REGISTRY[method]["availability_reason"]
        with pytest.raises(ValueError, match="not an existing production route alias"):
            MODULE.production_route_for_method(method)


def test_cold_route_is_not_accepted_as_kv_only():
    assert "cold_handoff_build" not in MODULE.METHODS
    with pytest.raises(ValueError):
        MODULE.production_route_for_method("safe_kv_only_diffusion")


def test_warm_route_is_only_complete_method_not_handoff_only():
    assert (
        MODULE.production_route_for_method("complete_region_state_execution")
        == "warm_cached_suffix"
    )
    with pytest.raises(ValueError):
        MODULE.production_route_for_method("kv_gdn_handoff_full_rows")


def test_full_replay_consumes_no_cached_state():
    record = normalize("controlled_full_replay")
    assert record["physical_kv_reused_positions"] == 0
    with pytest.raises(RuntimeError, match="consumed cached KV"):
        normalize("controlled_full_replay", cache_hits=1)


def test_native_radix_registry_disables_cluster_extensions():
    entry = MODULE.METHOD_REGISTRY["upstream_sglang_radix"]
    assert not entry["supported"]
    assert not entry["uses_gdn_restore"]
    assert not entry["uses_region_dag"]
    assert entry["semantic_contract"] == "bundled_native_sglang_radix_behavior"


def safe_runtime(pool_size=64):
    return SimpleNamespace(
        model_runner=SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(size=pool_size)
        )
    )


def observation(prefix=None, authoritative=None, fresh=None, **changes):
    value = {
        "req_pool_idx": 0,
        "prefix_indices": (
            torch.tensor([1, 2, 3], dtype=torch.int64) if prefix is None else prefix
        ),
        "authoritative_locations": (
            torch.tensor([1, 2, 3], dtype=torch.int64)
            if authoritative is None
            else authoritative
        ),
        "fresh_reconstruction_locations": (
            torch.tensor([4, 5, 6], dtype=torch.int64) if fresh is None else fresh
        ),
        "restore_required": False,
    }
    value.update(changes)
    return value


def validate_observation(value):
    return MODULE.validate_safe_prefix_evidence(
        safe_runtime(), [value], boundary=3, expected_steps=1
    )


def test_safe_kv_only_reuses_valid_physical_kv():
    evidence = validate_observation(observation())
    assert evidence["physical_locations_match"]
    assert evidence["observed_reused_positions"] == 3


def test_safe_kv_only_performs_zero_gdn_restores():
    record = normalize("safe_kv_only_diffusion")
    assert record["gdn_restore_count"] == 0
    with pytest.raises(RuntimeError, match="consumed a GDN snapshot"):
        normalize("safe_kv_only_diffusion", gdn_state_restores=1)


def test_safe_kv_only_rejects_wrong_prefix_dtype():
    with pytest.raises(RuntimeError, match="dtype"):
        validate_observation(
            observation(prefix=torch.tensor([1, 2, 3], dtype=torch.int32))
        )


def test_safe_kv_only_rejects_wrong_prefix_length():
    with pytest.raises(RuntimeError, match="length"):
        validate_observation(observation(prefix=torch.tensor([1, 2])))


def test_safe_kv_only_rejects_changed_physical_locations():
    with pytest.raises(RuntimeError, match="authoritative"):
        validate_observation(observation(authoritative=torch.tensor([1, 2, 4])))


@pytest.mark.parametrize(
    "locations", (torch.tensor([0, 2, 3]), torch.tensor([1, 2, 65]))
)
def test_safe_kv_only_rejects_invalid_pool_locations(locations):
    with pytest.raises(RuntimeError, match="invalid pool"):
        validate_observation(
            observation(prefix=locations, authoritative=locations.clone())
        )


def test_gdn_reconstruction_cannot_overwrite_reused_kv():
    with pytest.raises(RuntimeError, match="overwrite"):
        validate_observation(observation(fresh=torch.tensor([3, 4, 5])))


def test_complete_method_preserves_physical_reuse_proof():
    record = normalize("complete_region_state_execution")
    assert record["physical_kv_reused_positions"] > 0
    assert record["kv_reuse_evidence"]["physical_locations_match"]
    assert record["gdn_restore_count"] > 0


def test_output_hashes_and_tokens_must_match_full_replay():
    reference = {"output_token_ids": [1, 2], "output_hash": "same"}
    MODULE.require_output_identity(reference, dict(reference), method="safe")
    with pytest.raises(RuntimeError, match="tokens differ"):
        MODULE.require_output_identity(
            reference,
            {"output_token_ids": [2, 1], "output_hash": "same"},
            method="safe",
        )
    with pytest.raises(RuntimeError, match="hash differs"):
        MODULE.require_output_identity(
            reference,
            {"output_token_ids": [1, 2], "output_hash": "different"},
            method="safe",
        )


def test_stable_tensors_must_remain_unchanged():
    with pytest.raises(RuntimeError, match="mutated stable"):
        normalize("complete_region_state_execution", stable_state_unchanged=False)


@pytest.mark.parametrize(
    ("field", "message"),
    (("fallback_count", "fell back"), ("recovery_replay_count", "recovered")),
)
def test_fallback_and_recovery_gates_remain_mandatory(field, message):
    with pytest.raises(RuntimeError, match="recovered or fell back"):
        normalize("controlled_full_replay", **{field: 1})


def test_exactly_one_terminal_synchronization_remains():
    assert normalize()["synchronization_count"] == 1
    with pytest.raises(RuntimeError, match="exactly one"):
        normalize(**{"timing.synchronization_count": 2})


def test_equality_and_physical_verification_are_after_timer_finalization():
    source = inspect.getsource(
        load(
            "cluster3_cache_comparison_production_source",
            ROOT / "eval/scripts/cluster3_production_latency_benchmark.py",
        ).execute_production_measurement
    )
    assert source.index("timing = timer.result()") < source.index(
        "verify_prefix_kv_reuse_after_timing"
    )
    assert source.index("timing = timer.result()") < source.index(
        "compact_output_hash_after_timing"
    )


def test_component_intervals_do_not_double_count():
    raw = raw_record()
    raw["component_timing"]["gdn_replay"] = 60.0
    with pytest.raises(RuntimeError, match="double-count"):
        MODULE.component_attribution(raw)


def test_component_coverage_below_90_percent_fails():
    record = normalize()
    record["component_coverage_ratio"] = 0.89
    with pytest.raises(RuntimeError, match="90%"):
        MODULE.validate_component_coverage(record)


def test_uninstrumented_components_are_null_with_reasons():
    record = normalize(variant="uninstrumented")
    assert record["attention_cuda_ms"] is None
    assert record["attention_cuda_ms_availability_reason"] == (
        "uninstrumented_headline_repetition"
    )


def test_original_flare_artifacts_cannot_be_mislabeled(tmp_path):
    artifact = tmp_path / "flare.jsonl"
    artifact.write_text(
        json.dumps(
            {
                "method": "controlled_full_replay",
                "runtime_revision": MODULE.ORIGINAL_FLARE_REVISION,
                "case_id": "long-favorable-suffix",
            }
        )
        + "\n"
    )
    with pytest.raises(RuntimeError, match="mislabeled"):
        MODULE.validate_original_flare_artifact(artifact)


def test_original_flare_fragmented_case_is_rejected(tmp_path):
    artifact = tmp_path / "flare.jsonl"
    artifact.write_text(
        json.dumps(
            {
                "method": "original_flare_external_runtime",
                "runtime_revision": MODULE.ORIGINAL_FLARE_REVISION,
                "case_id": "four-fragmented-regions",
            }
        )
        + "\n"
    )
    with pytest.raises(RuntimeError, match="fragmented"):
        MODULE.validate_original_flare_artifact(artifact)


def test_method_registry_does_not_manufacture_five_supported_paths():
    assert sum(entry["supported"] for entry in MODULE.METHOD_REGISTRY.values()) == 3
    assert MODULE.METHOD_REGISTRY["safe_kv_only_diffusion"]["semantic_contract"] != (
        MODULE.METHOD_REGISTRY["complete_region_state_execution"]["semantic_contract"]
    )
