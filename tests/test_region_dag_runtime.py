import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
REGION = ROOT / "eval/sglang/srt/dllm/region"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXECUTION = load_module(
    "sglang.srt.dllm.region.execution_spec", REGION / "execution_spec.py"
)
load_module("sglang.srt.dllm.region.dependency_graph", REGION / "dependency_graph.py")
RUNTIME = load_module("cluster3_region_runtime", REGION / "runtime.py")

Contract = EXECUTION.RegionDAGExecutionSpec
Region = EXECUTION.RegionDAGRegion
Status = EXECUTION.RegionStatus


def make_region(region_id, start, end, status, parents=(), version=0):
    return Region(
        region_id=region_id,
        region_version=version,
        start=start,
        end=end,
        status=status,
        parent_region_ids=tuple(parents),
        recorded_parent_versions=tuple((parent, 0) for parent in parents),
        token_hash=f"tokens:{region_id}:{version}",
        position_hash=f"positions:{start}:{end}",
    )


def abcd_spec():
    return Contract(
        sequence_length=40,
        diffusion_steps=2,
        regions=(
            make_region("A", 0, 10, Status.STABLE),
            make_region("B", 10, 20, Status.ACTIVE, ("A",)),
            make_region("C", 20, 30, Status.STABLE, ("A",)),
            make_region("D", 30, 40, Status.ACTIVE, ("B",)),
        ),
    )


def test_runtime_plan_keeps_logical_closure_and_gdn_replay_separate():
    plan = RUNTIME.build_region_dag_runtime_plan(abcd_spec(), ("B",))
    assert plan.logical_invalidation_regions == ("B", "D")
    assert plan.gdn_replay_start == 10
    assert plan.gdn_replay_positions == tuple(range(10, 40))
    assert plan.actually_recomputed_positions == tuple(range(10, 40))
    assert plan.replayed_but_logically_valid_regions == ("C",)
    assert plan.reused_regions == ("A",)
    assert plan.reused_positions == tuple(range(10))
    assert plan.invalidated_ranges == ((10, 20), (30, 40))


@pytest.mark.parametrize(
    ("boundary", "region_ids"),
    [(10, ("A",)), (30, ("A", "B", "C"))],
)
def test_canonical_frontier_projects_exact_topological_prefix(boundary, region_ids):
    source = abcd_spec()
    frontier = RUNTIME.build_canonical_frontier_execution_spec(source, boundary)
    assert frontier.sequence_length == boundary
    assert frontier.source_sequence_length == source.sequence_length
    assert tuple(region.region_id for region in frontier.regions) == region_ids
    assert frontier.attention_contract_id == source.attention_contract_id
    assert frontier.diffusion_steps == source.diffusion_steps


def test_canonical_frontier_rejects_non_replay_boundaries():
    with pytest.raises(ValueError, match="begin an active"):
        RUNTIME.build_canonical_frontier_execution_spec(abcd_spec(), 20)


def test_multiple_edits_are_canonicalized_in_text_order():
    plan = RUNTIME.build_region_dag_runtime_plan(abcd_spec(), ("D", "B"))
    assert plan.edited_regions == ("B", "D")
    assert plan.logical_invalidation_regions == ("B", "D")


def test_explicit_full_replay_is_observable_and_reuses_nothing():
    plan = RUNTIME.build_region_dag_runtime_plan(
        abcd_spec(), ("D",), force_full_replay=True
    )
    assert plan.full_replay
    assert plan.logical_invalidation_regions == ("D",)
    assert plan.gdn_replay_start == 0
    assert plan.reused_regions == ()
    assert plan.reused_positions == ()
    assert plan.replayed_but_logically_valid_regions == ("A", "B", "C")


@pytest.mark.parametrize(
    ("edits", "message"),
    [
        ((), "nonempty"),
        (("B", "B"), "duplicates"),
        (("missing",), "unknown"),
    ],
)
def test_invalid_edit_sets_fail_closed(edits, message):
    with pytest.raises(ValueError, match=message):
        RUNTIME.build_region_dag_runtime_plan(abcd_spec(), edits)


def test_runtime_plan_round_trip_preserves_every_observable_set():
    original = RUNTIME.build_region_dag_runtime_plan(abcd_spec(), ("B",))
    assert RUNTIME.RegionDAGRuntimePlan.from_dict(original.to_dict()) == original


def test_frontier_key_contains_all_preceding_region_cache_identity():
    key = RUNTIME.build_region_dag_frontier_key(
        spec=abcd_spec(),
        boundary=30,
        request_id="request-1",
        request_pool_idx=4,
        request_slot_generation=9,
        model_identity="model",
        model_revision="revision",
        adapter_identity="adapter",
        adapter_revision="adapter-revision",
    )
    assert key.boundary == 30
    assert tuple(value[0] for value in key.preceding_regions) == ("A", "B", "C")
    assert key.preceding_regions[1][2] == (("A", 0),)


def test_frontier_key_rejects_non_region_boundary():
    with pytest.raises(ValueError, match="exact region boundary"):
        RUNTIME.build_region_dag_frontier_key(
            spec=abcd_spec(),
            boundary=11,
            request_id="request-1",
            request_pool_idx=4,
            request_slot_generation=9,
            model_identity="model",
            model_revision="revision",
            adapter_identity="",
            adapter_revision="",
        )


def test_parent_or_region_version_changes_frontier_identity():
    original = abcd_spec()
    original_key = RUNTIME.build_region_dag_frontier_key(
        spec=original,
        boundary=40,
        request_id="request-1",
        request_pool_idx=4,
        request_slot_generation=9,
        model_identity="model",
        model_revision="revision",
        adapter_identity="",
        adapter_revision="",
    )
    changed = Contract(
        sequence_length=40,
        diffusion_steps=2,
        regions=(
            make_region("A", 0, 10, Status.STABLE),
            make_region("B", 10, 20, Status.ACTIVE, ("A",), version=1),
            make_region("C", 20, 30, Status.STABLE, ("A",)),
            Region(
                **{
                    **make_region("D", 30, 40, Status.ACTIVE, ("B",)).__dict__,
                    "recorded_parent_versions": (("B", 1),),
                }
            ),
        ),
    )
    changed_key = RUNTIME.build_region_dag_frontier_key(
        spec=changed,
        boundary=40,
        request_id="request-1",
        request_pool_idx=4,
        request_slot_generation=9,
        model_identity="model",
        model_revision="revision",
        adapter_identity="",
        adapter_revision="",
    )
    assert changed_key != original_key


def test_instrumentation_starts_only_observed_counters_at_zero():
    spec = abcd_spec()
    plan = RUNTIME.build_region_dag_runtime_plan(spec, ("B",))
    metrics = RUNTIME.RegionDAGInstrumentation.from_plan(spec, plan)
    assert metrics.selected_contract == spec.attention_contract_id
    assert metrics.region_count == 4
    assert metrics.gdn_replay_tokens == 30
    assert metrics.attention_query_token_layer_positions == 0
    assert metrics.mask_build_time is None
    assert metrics.peak_memory is None
