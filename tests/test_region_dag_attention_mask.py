import ast
import importlib.util
import random
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[1]
EXECUTION_PATH = ROOT / "eval/sglang/srt/dllm/region/execution_spec.py"
MASK_PATH = ROOT / "eval/sglang/srt/dllm/attention_mask.py"
TRAINING_PATH = ROOT / "torchtitan/models/qwen3_5/model/dllm_model.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if "sglang.srt.dllm.region.execution_spec" in sys.modules:
    EXECUTION = sys.modules["sglang.srt.dllm.region.execution_spec"]
else:
    EXECUTION = load_module("sglang.srt.dllm.region.execution_spec", EXECUTION_PATH)
MASKS = load_module("cluster3_attention_mask", MASK_PATH)


def load_training_mask_helpers():
    tree = ast.parse(TRAINING_PATH.read_text(encoding="utf-8"))
    names = {
        "_region_dag_reference_value",
        "create_region_dag_attention_mask",
        "create_region_dag_4d_mask",
    }
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id == "REGION_DAG_CONSERVATIVE_GDN_V1"
            for target in node.targets
        ):
            nodes.append(node)
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in names
        ):
            nodes.append(node)
    assert {node.name for node in nodes if isinstance(node, ast.FunctionDef)} == names
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(TRAINING_PATH), "exec"),
        namespace,
    )
    return namespace


TRAINING = load_training_mask_helpers()
Contract = EXECUTION.RegionDAGExecutionSpec
Region = EXECUTION.RegionDAGRegion
Status = EXECUTION.RegionStatus


def region(region_id, start, end, status, parents=(), versions=None, version=0):
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
    return Contract(
        sequence_length=8,
        diffusion_steps=2,
        regions=(
            region("A", 0, 2, Status.STABLE),
            region("B", 2, 4, Status.ACTIVE, ("A",)),
            region("C", 4, 6, Status.STABLE, ("A",)),
            region("D", 6, 8, Status.ACTIVE, ("B",)),
        ),
    )


def expected_abcd_full_mask():
    return torch.tensor(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0, 1, 1],
            [1, 1, 1, 1, 0, 0, 1, 1],
        ],
        dtype=torch.bool,
    )


def test_hand_computed_abcd_mask_and_transitive_parent_visibility():
    observed = MASKS.create_region_dag_boolean_mask(abcd_spec())
    assert observed.dtype is torch.bool
    assert observed.is_contiguous()
    assert torch.equal(observed, expected_abcd_full_mask())
    assert observed[6, 0]
    assert not observed[6, 4]


def test_stable_queries_never_see_active_keys_and_remain_causal_in_region():
    observed = MASKS.create_region_dag_boolean_mask(abcd_spec())
    assert not observed[4:6, 2:4].any()
    assert not observed[4, 5]
    assert observed[5, 4]


def test_active_siblings_are_isolated_without_an_ancestor_edge():
    spec = Contract(
        sequence_length=6,
        diffusion_steps=1,
        regions=(
            region("A", 0, 2, Status.STABLE),
            region("B", 2, 4, Status.ACTIVE, ("A",)),
            region("C", 4, 6, Status.ACTIVE, ("A",)),
        ),
    )
    observed = MASKS.create_region_dag_boolean_mask(spec)
    assert not observed[2:4, 4:6].any()
    assert not observed[4:6, 2:4].any()
    assert observed[2:4, 0:2].all()
    assert observed[4:6, 0:2].all()


def test_single_active_region_and_entire_sequence_active_reductions():
    one_active = Contract(
        sequence_length=4,
        diffusion_steps=1,
        regions=(
            region("S", 0, 2, Status.STABLE),
            region("X", 2, 4, Status.ACTIVE, ("S",)),
        ),
    )
    expected = torch.tensor(
        [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
        dtype=torch.bool,
    )
    assert torch.equal(MASKS.create_region_dag_boolean_mask(one_active), expected)

    entirely_active = Contract(
        sequence_length=5,
        diffusion_steps=1,
        regions=(region("all", 0, 5, Status.ACTIVE),),
    )
    assert MASKS.create_region_dag_boolean_mask(entirely_active).all()


def test_serving_active_rows_equal_full_oracle_rows_at_original_positions():
    positions = torch.tensor([2, 3, 6, 7], dtype=torch.int64)
    active_mask = MASKS.create_region_dag_boolean_mask(abcd_spec(), positions)
    assert torch.equal(
        active_mask, expected_abcd_full_mask().index_select(0, positions)
    )


def test_batched_flattened_paged_mask_uses_exact_query_and_kv_counts():
    first = abcd_spec()
    second = Contract(
        sequence_length=3,
        diffusion_steps=1,
        regions=(region("all", 0, 3, Status.ACTIVE),),
    )
    first_positions = torch.tensor([2, 3, 6, 7], dtype=torch.int64)
    second_positions = torch.arange(3, dtype=torch.int64)
    flattened = MASKS.build_region_dag_paged_custom_mask(
        [first, second], [first_positions, second_positions]
    )
    assert flattened.dtype is torch.bool
    assert flattened.ndim == 1
    assert flattened.is_contiguous()
    assert flattened.numel() == 4 * 8 + 3 * 3
    assert torch.equal(
        flattened[: 4 * 8].view(4, 8),
        expected_abcd_full_mask().index_select(0, first_positions),
    )
    assert flattened[4 * 8 :].view(3, 3).all()


def test_region_route_is_forced_to_custom_paged():
    selected = MASKS.select_region_dag_mask_backend(
        EXECUTION.REGION_DAG_CONSERVATIVE_GDN_V1
    )
    assert selected == "custom_paged"
    custom = torch.ones(8, dtype=torch.bool)
    planned, native, full = MASKS.paged_mask_planner_arguments(selected, custom)
    assert planned is custom
    assert native is False
    assert full is False
    with pytest.raises(ValueError, match="unsupported"):
        MASKS.select_region_dag_mask_backend("causal_prefix_diffusion_suffix_v1")


@pytest.mark.parametrize(
    ("positions", "error", "message"),
    [
        (torch.tensor([2, 3], dtype=torch.int32), TypeError, "int64"),
        (torch.tensor([[2, 3]], dtype=torch.int64), ValueError, "one-dimensional"),
        (torch.tensor([3, 2], dtype=torch.int64), ValueError, "sorted"),
        (torch.tensor([2, 2], dtype=torch.int64), ValueError, "unique"),
        (torch.tensor([-1, 2], dtype=torch.int64), ValueError, "within"),
        (torch.tensor([2, 8], dtype=torch.int64), ValueError, "within"),
        (torch.arange(8, dtype=torch.int64)[::2], ValueError, "contiguous"),
    ],
)
def test_invalid_query_position_contracts_fail_closed(positions, error, message):
    with pytest.raises(error, match=message):
        MASKS.create_region_dag_boolean_mask(abcd_spec(), positions)


@pytest.mark.parametrize(
    ("mask", "query_counts", "kv_counts", "error", "message"),
    [
        (torch.ones(8), [1], [8], TypeError, "dtype bool"),
        (torch.ones((1, 8), dtype=torch.bool), [1], [8], ValueError, "flattened"),
        (torch.ones(16, dtype=torch.bool)[::2], [1], [8], ValueError, "contiguous"),
        (torch.ones(7, dtype=torch.bool), [1], [8], ValueError, "length"),
        (torch.ones(8, dtype=torch.bool), [1, 1], [8], ValueError, "batches"),
    ],
)
def test_invalid_flattened_mask_dimensions_type_and_length_fail_closed(
    mask, query_counts, kv_counts, error, message
):
    with pytest.raises(error, match=message):
        MASKS.validate_region_dag_paged_custom_mask(mask, query_counts, kv_counts)


def test_training_boolean_and_4d_helpers_match_serving_oracle():
    contract = abcd_spec().to_dict()
    positions = torch.tensor([2, 3, 6, 7], dtype=torch.int64)
    serving = MASKS.create_region_dag_boolean_mask(abcd_spec(), positions)
    training = TRAINING["create_region_dag_attention_mask"](contract, positions)
    assert torch.equal(training, serving)

    additive = TRAINING["create_region_dag_4d_mask"](
        contract,
        batch_size=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        query_positions=positions,
    )
    assert additive.shape == (2, 1, 4, 8)
    assert torch.equal(additive[0, 0] == 0, serving)
    assert torch.equal(additive[1], additive[0])


def slow_mask_oracle(spec, query_positions):
    regions = {region.region_id: region for region in spec.regions}
    membership = {}
    for region_value in spec.regions:
        for position in range(region_value.start, region_value.end):
            membership[position] = region_value.region_id

    def ancestors(region_id):
        result = set()
        pending = list(regions[region_id].parent_region_ids)
        while pending:
            parent = pending.pop()
            if parent not in result:
                result.add(parent)
                pending.extend(regions[parent].parent_region_ids)
        return result

    result = torch.zeros((len(query_positions), spec.sequence_length), dtype=torch.bool)
    for row, query_position in enumerate(query_positions):
        query_region = regions[membership[query_position]]
        visible_ancestors = ancestors(query_region.region_id)
        for key_position in range(spec.sequence_length):
            key_region = regions[membership[key_position]]
            if query_region.is_stable:
                visible = (
                    key_region.region_id == query_region.region_id
                    and key_position <= query_position
                ) or (
                    key_region.region_id in visible_ancestors and key_region.is_stable
                )
            else:
                visible = (
                    key_region.region_id == query_region.region_id
                    or key_region.region_id in visible_ancestors
                )
            result[row, key_position] = visible
    return result


def test_random_small_dag_masks_match_independent_python_oracle():
    rng = random.Random(20260825)
    for _ in range(300):
        count = rng.randint(1, 10)
        versions = {f"R{i}": rng.randint(0, 4) for i in range(count)}
        statuses = []
        parents = []
        for index in range(count):
            candidates = list(range(index))
            selected = tuple(
                candidate for candidate in candidates if rng.random() < 0.3
            )
            parent_ids = tuple(f"R{candidate}" for candidate in selected)
            can_be_stable = all(
                statuses[candidate] is Status.STABLE for candidate in selected
            )
            status = (
                Status.STABLE
                if can_be_stable and rng.random() < 0.45
                else Status.ACTIVE
            )
            parents.append(parent_ids)
            statuses.append(status)
        if all(status is Status.STABLE for status in statuses):
            statuses[-1] = Status.ACTIVE
        regions = tuple(
            region(
                f"R{index}",
                index * 2,
                index * 2 + 2,
                statuses[index],
                parents[index],
                versions=versions,
                version=versions[f"R{index}"],
            )
            for index in range(count)
        )
        spec = Contract(
            sequence_length=count * 2,
            regions=regions,
            diffusion_steps=2,
        )
        query_positions = sorted(
            rng.sample(
                range(spec.sequence_length), rng.randint(1, spec.sequence_length)
            )
        )
        query_tensor = torch.tensor(query_positions, dtype=torch.int64)
        expected = slow_mask_oracle(spec, query_positions)
        serving = MASKS.create_region_dag_boolean_mask(spec, query_tensor)
        training = TRAINING["create_region_dag_attention_mask"](
            spec.to_dict(), query_tensor
        )
        assert torch.equal(serving, expected)
        assert torch.equal(training, expected)
