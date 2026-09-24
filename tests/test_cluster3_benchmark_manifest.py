import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
REGION = ROOT / "eval/sglang/srt/dllm/region"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load("sglang.srt.dllm.region.execution_spec", REGION / "execution_spec.py")
load("sglang.srt.dllm.region.dependency_graph", REGION / "dependency_graph.py")
load("sglang.srt.dllm.region.runtime", REGION / "runtime.py")
VALIDATION = load(
    "cluster3_manifest_validation",
    ROOT / "eval/scripts/cluster3_region_dag_validation.py",
)
MANIFEST = load(
    "cluster3_benchmark_manifest_tested",
    ROOT / "eval/scripts/cluster3_benchmark_manifest.py",
)


MANIFEST_ROUTES = (
    "full_replay",
    "cold_handoff_build",
    "warm_cached_suffix",
)


def normalize(cases, routes=MANIFEST_ROUTES):
    return MANIFEST.normalize_manifest(
        {"schema_version": 1, "cases": cases},
        validation_module=VALIDATION,
        seed=17,
        requested_routes=routes,
    )


def explicit_case(case_id, spans):
    total = 128
    boundaries = sorted({0, total, *(value for span in spans for value in span)})
    regions = []
    span_set = set(spans)
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        active = (start, end) in span_set
        regions.append(
            {
                "region_id": ("X" if active else "S") + str(index),
                "start": start,
                "end": end,
                "stable": not active,
                "parents": [] if index == 0 else ["S0"],
            }
        )
    return {
        "case_id": case_id,
        "total_tokens_per_request": total,
        "active_spans": [list(span) for span in spans],
        "diffusion_steps": 4,
        "batch_size": 1,
        "regions": regions,
    }


def test_legacy_case_normalizes_exactly():
    cases, normalized = MANIFEST.legacy_production_case(
        VALIDATION, seed=17, requested_routes=MANIFEST_ROUTES
    )
    assert cases[0].sequence_length == 2112
    assert cases[0].active_positions == tuple(range(2048, 2112))
    assert normalized[0]["case_id"] == "p2048-a64-s4-b1"
    assert normalized[0]["prefix_tokens"] == 2048
    assert normalized[0]["diffusion_steps"] == 4
    assert normalized[0]["batch_size"] == 1


@pytest.mark.parametrize(
    "filename, expected_count",
    (
        ("cluster3_parameterized_smoke.json", 5),
        ("cluster3_parameterized_final_sweep.example.json", 11),
    ),
)
def test_checked_in_manifests_normalize(filename, expected_count):
    cases, normalized = MANIFEST.load_manifest(
        ROOT / "eval/manifests" / filename,
        validation_module=VALIDATION,
        seed=17,
        requested_routes=MANIFEST_ROUTES,
    )
    assert len(cases) == len(normalized) == expected_count
    assert len({case.case_id for case in cases}) == expected_count


@pytest.mark.parametrize(
    "spans",
    (
        ((96, 128),),
        ((32, 48),),
        ((16, 24), (104, 112)),
        ((8, 16), (40, 48), (72, 80), (112, 120)),
    ),
)
def test_suffix_internal_two_and_four_spans_preserve_absolute_positions(spans):
    cases, normalized = normalize([explicit_case("case", spans)])
    expected = tuple(position for start, end in spans for position in range(start, end))
    assert cases[0].active_positions == expected
    assert normalized[0]["active_spans"] == [list(span) for span in spans]
    spec = VALIDATION.build_execution_spec(
        cases[0], [0] * cases[0].sequence_length, edited=True
    )
    plan = VALIDATION.expected_plan(spec, cases[0].edited_regions)
    assert plan.gdn_replay_start == spans[0][0]
    assert plan.gdn_replay_positions == tuple(range(spans[0][0], 128))


def test_overlapping_active_spans_fail_before_runtime():
    case = explicit_case("overlap", ((16, 32),))
    case["active_spans"] = [[16, 32], [24, 40]]
    with pytest.raises(ValueError, match="must not overlap"):
        normalize([case])


def test_unsupported_schema_nonpositive_tokens_and_unknown_parent_fail():
    with pytest.raises(ValueError, match="unsupported"):
        MANIFEST.normalize_manifest(
            {"schema_version": 2, "cases": []},
            validation_module=VALIDATION,
            seed=17,
            requested_routes=MANIFEST_ROUTES,
        )
    case = explicit_case("bad-total", ((16, 32),))
    case["total_tokens_per_request"] = 0
    with pytest.raises(ValueError, match="must be positive"):
        normalize([case])
    case = explicit_case("missing-parent", ((16, 32),))
    case["regions"][1]["parents"] = ["absent"]
    with pytest.raises(ValueError, match="unknown parent"):
        normalize([case])


def test_invalid_region_gap_and_cycle_fail():
    gap = explicit_case("gap", ((16, 32),))
    gap["regions"][0]["end"] = 15
    with pytest.raises(ValueError, match="gap-free"):
        normalize([gap])

    cycle = explicit_case("cycle", ((16, 32),))
    cycle["regions"][0]["parents"] = [cycle["regions"][1]["region_id"]]
    with pytest.raises(ValueError, match="cycle"):
        normalize([cycle])


def test_duplicate_ids_and_unsupported_batch_fail_closed():
    case = explicit_case("duplicate", ((16, 32),))
    with pytest.raises(ValueError, match="unique"):
        normalize([case, case])
    case = dict(case, case_id="batch-two", batch_size=2)
    with pytest.raises(ValueError, match="does not emulate batching"):
        normalize([case])


def test_warm_route_requires_constructible_positive_frontier():
    case = {
        "case_id": "starts-active",
        "total_tokens_per_request": 16,
        "active_spans": [[0, 8]],
        "diffusion_steps": 2,
        "batch_size": 1,
        "regions": [
            {
                "region_id": "X0",
                "start": 0,
                "end": 8,
                "stable": False,
                "parents": [],
            },
            {
                "region_id": "S0",
                "start": 8,
                "end": 16,
                "stable": True,
                "parents": [],
            },
        ],
    }
    with pytest.raises(ValueError, match="before position zero"):
        normalize([case])
    cases, _ = normalize([case], routes=("full_replay",))
    assert cases[0].active_positions == tuple(range(8))
