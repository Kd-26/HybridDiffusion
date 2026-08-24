import importlib.util
import json
import random
import sys
from dataclasses import FrozenInstanceError
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
GRAPH = load_module("cluster3_dependency_graph", REGION / "dependency_graph.py")

Contract = EXECUTION.RegionDAGExecutionSpec
Region = EXECUTION.RegionDAGRegion
Status = EXECUTION.RegionStatus
DependencyGraph = GRAPH.DependencyGraph
CONTRACT_ID = EXECUTION.REGION_DAG_CONSERVATIVE_GDN_V1


def make_region(
    region_id,
    start,
    end,
    status,
    parents=(),
    versions=None,
    version=0,
):
    versions = versions or {}
    return Region(
        region_id=region_id,
        region_version=version,
        start=start,
        end=end,
        status=status,
        parent_region_ids=tuple(parents),
        recorded_parent_versions=tuple(
            (parent, versions.get(parent, 0)) for parent in parents
        ),
        token_hash=f"tokens:{region_id}:{version}",
        position_hash=f"positions:{start}:{end}",
    )


def abcd_spec():
    regions = (
        make_region("A", 0, 10, Status.STABLE),
        make_region("B", 10, 20, Status.ACTIVE, ("A",)),
        make_region("C", 20, 30, Status.STABLE, ("A",)),
        make_region("D", 30, 40, Status.ACTIVE, ("B",)),
    )
    return Contract(sequence_length=40, regions=regions, diffusion_steps=2)


def test_abcd_logical_invalidation_is_distinct_from_conservative_gdn_replay():
    graph = DependencyGraph(abcd_spec())
    assert graph.parents("D") == ("B",)
    assert graph.ancestors("D") == ("A", "B")
    assert graph.descendants("A") == ("B", "C", "D")
    assert graph.invalidation_closure("B") == ("B", "D")
    assert "C" not in graph.invalidation_closure("B")
    assert graph.earliest_invalidated_position("B") == 10
    assert graph.conservative_gdn_replay_positions("B") == tuple(range(10, 40))
    assert set(range(20, 30)).issubset(graph.conservative_gdn_replay_positions("B"))


def test_contract_serialization_is_canonical_and_immutable():
    original = abcd_spec()
    restored = Contract.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original
    assert restored.attention_contract_id == CONTRACT_ID
    with pytest.raises(FrozenInstanceError):
        restored.sequence_length = 41


def test_topological_order_is_deterministic_for_siblings_and_disconnected_roots():
    spec = Contract(
        sequence_length=50,
        diffusion_steps=2,
        regions=(
            make_region("root-a", 0, 10, Status.STABLE),
            make_region("left", 10, 20, Status.ACTIVE, ("root-a",)),
            make_region("right", 20, 30, Status.ACTIVE, ("root-a",)),
            make_region("root-b", 30, 40, Status.STABLE),
            make_region("tail", 40, 50, Status.ACTIVE, ("root-b",)),
        ),
    )
    graph = DependencyGraph(spec)
    assert graph.topological_order() == (
        "root-a",
        "left",
        "right",
        "root-b",
        "tail",
    )
    assert graph.invalidation_closure({"left", "root-b"}) == (
        "left",
        "root-b",
        "tail",
    )
    assert graph.earliest_invalidated_position({"left", "root-b"}) == 10


def test_chain_ancestors_descendants_and_multiple_edits():
    spec = Contract(
        sequence_length=32,
        diffusion_steps=1,
        regions=(
            make_region("R0", 0, 8, Status.STABLE),
            make_region("R1", 8, 16, Status.ACTIVE, ("R0",)),
            make_region("R2", 16, 24, Status.ACTIVE, ("R1",)),
            make_region("R3", 24, 32, Status.ACTIVE, ("R2",)),
        ),
    )
    graph = DependencyGraph(spec)
    assert graph.ancestors("R3") == ("R0", "R1", "R2")
    assert graph.descendants("R0") == ("R1", "R2", "R3")
    assert graph.invalidation_closure({"R1", "R3"}) == ("R1", "R2", "R3")


def test_parent_version_matching_is_exact_and_ordered():
    versions = {"A": 3, "B": 7}
    spec = Contract(
        sequence_length=30,
        diffusion_steps=2,
        regions=(
            make_region("A", 0, 10, Status.STABLE, version=3),
            make_region("B", 10, 20, Status.ACTIVE, version=7),
            make_region("D", 20, 30, Status.ACTIVE, ("A", "B"), versions=versions),
        ),
    )
    graph = DependencyGraph(spec)
    assert graph.versions_match("D", (("A", 3), ("B", 7)))
    assert graph.versions_match("D", {"B": 7, "A": 3})
    assert not graph.versions_match("D", (("B", 7), ("A", 3)))
    assert not graph.versions_match("D", (("A", 4), ("B", 7)))
    assert not graph.versions_match("D", {"A": 3})


@pytest.mark.parametrize(
    ("regions", "sequence_length", "message"),
    [
        (
            (
                make_region("A", 0, 10, Status.STABLE),
                make_region("B", 11, 20, Status.ACTIVE),
            ),
            20,
            "gap",
        ),
        (
            (
                make_region("A", 0, 11, Status.STABLE),
                make_region("B", 10, 20, Status.ACTIVE),
            ),
            20,
            "overlaps",
        ),
        (
            (
                make_region("A", 0, 10, Status.STABLE),
                make_region("A", 10, 20, Status.ACTIVE),
            ),
            20,
            "unique",
        ),
        ((make_region("A", 0, 10, Status.STABLE),), 10, "active region"),
    ],
)
def test_malformed_partitions_fail_closed(regions, sequence_length, message):
    with pytest.raises(ValueError, match=message):
        Contract(
            sequence_length=sequence_length,
            regions=regions,
            diffusion_steps=1,
        )


def test_unknown_parent_fails_closed():
    with pytest.raises(ValueError, match="unknown parents"):
        Contract(
            sequence_length=20,
            diffusion_steps=1,
            regions=(
                make_region("A", 0, 10, Status.STABLE),
                make_region("B", 10, 20, Status.ACTIVE, ("missing",)),
            ),
        )


def test_cycle_fails_closed():
    with pytest.raises(ValueError, match="cycle"):
        Contract(
            sequence_length=20,
            diffusion_steps=1,
            regions=(
                make_region("A", 0, 10, Status.ACTIVE, ("B",)),
                make_region("B", 10, 20, Status.ACTIVE, ("A",)),
            ),
        )


def test_stable_region_cannot_depend_on_active_ancestor():
    with pytest.raises(ValueError, match="stable region.*active ancestor"):
        Contract(
            sequence_length=30,
            diffusion_steps=1,
            regions=(
                make_region("A", 0, 10, Status.ACTIVE),
                make_region("B", 10, 20, Status.ACTIVE, ("A",)),
                make_region("C", 20, 30, Status.STABLE, ("B",)),
            ),
        )


def test_stale_recorded_parent_version_is_rejected_by_contract():
    with pytest.raises(ValueError, match="recorded parent versions"):
        Contract(
            sequence_length=20,
            diffusion_steps=1,
            regions=(
                make_region("A", 0, 10, Status.STABLE, version=2),
                make_region(
                    "B",
                    10,
                    20,
                    Status.ACTIVE,
                    ("A",),
                    versions={"A": 1},
                ),
            ),
        )


def test_unknown_graph_queries_and_empty_edits_fail_closed():
    graph = DependencyGraph(abcd_spec())
    with pytest.raises(ValueError, match="unknown"):
        graph.parents("missing")
    with pytest.raises(ValueError, match="nonempty"):
        graph.invalidation_closure(set())


def test_random_dags_match_independent_invalidation_oracle():
    rng = random.Random(20260825)
    operations = 0
    for case_index in range(250):
        count = rng.randint(2, 12)
        versions = {f"R{i}": rng.randint(0, 5) for i in range(count)}
        parents = {"R0": ()}
        regions = [make_region("R0", 0, 2, Status.STABLE, version=versions["R0"])]
        for index in range(1, count):
            candidates = [f"R{i}" for i in range(index)]
            selected = tuple(
                candidate for candidate in candidates if rng.random() < 0.25
            )
            parents[f"R{index}"] = selected
            regions.append(
                make_region(
                    f"R{index}",
                    index * 2,
                    index * 2 + 2,
                    Status.ACTIVE,
                    selected,
                    versions=versions,
                    version=versions[f"R{index}"],
                )
            )
        graph = DependencyGraph(
            Contract(
                sequence_length=count * 2,
                regions=tuple(regions),
                diffusion_steps=2,
            )
        )
        children = {region_id: set() for region_id in parents}
        for child, parent_ids in parents.items():
            for parent in parent_ids:
                children[parent].add(child)

        for _ in range(10):
            edit_count = rng.randint(1, min(3, count))
            edits = set(rng.sample(list(parents), edit_count))
            expected = set(edits)
            pending = list(edits)
            while pending:
                current = pending.pop()
                for child in children[current]:
                    if child not in expected:
                        expected.add(child)
                        pending.append(child)
            observed = set(graph.invalidation_closure(edits))
            assert observed == expected, (case_index, edits, expected, observed)
            expected_start = min(int(region_id[1:]) * 2 for region_id in expected)
            assert graph.earliest_invalidated_position(edits) == expected_start
            assert graph.conservative_gdn_replay_positions(edits) == tuple(
                range(expected_start, count * 2)
            )
            operations += 1
    assert operations == 2500
