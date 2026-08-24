"""Pure dependency operations for immutable Region-DAG execution specs."""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Mapping
from typing import Union

from sglang.srt.dllm.region.execution_spec import RegionDAGExecutionSpec


RegionSelection = Union[str, Iterable[str]]


class DependencyGraph:
    """Validated graph view that keeps logical invalidation separate from GDN replay."""

    def __init__(self, spec: RegionDAGExecutionSpec):
        if not isinstance(spec, RegionDAGExecutionSpec):
            raise TypeError("DependencyGraph requires a RegionDAGExecutionSpec")
        spec.validate()
        self.spec = spec
        self._regions = {region.region_id: region for region in spec.regions}
        self._parents = {
            region.region_id: tuple(region.parent_region_ids) for region in spec.regions
        }
        self._children = {region_id: [] for region_id in self._regions}
        for region_id, parents in self._parents.items():
            for parent in parents:
                self._children[parent].append(region_id)
        self._text_order = {
            region.region_id: index for index, region in enumerate(spec.regions)
        }
        self._topological = self._compute_topological_order()

    def _require_region(self, region_id: str) -> str:
        region_id = str(region_id)
        if region_id not in self._regions:
            raise ValueError(f"unknown Region-DAG region {region_id!r}")
        return region_id

    def _selection(self, values: RegionSelection) -> tuple[str, ...]:
        values = (values,) if isinstance(values, str) else tuple(values)
        if not values:
            raise ValueError("Region-DAG edit set must be nonempty")
        selected = {self._require_region(value) for value in values}
        return tuple(
            region_id for region_id in self._topological if region_id in selected
        )

    def _compute_topological_order(self) -> tuple[str, ...]:
        indegree = {
            region_id: len(parents) for region_id, parents in self._parents.items()
        }
        ready = [
            (self._text_order[region_id], region_id)
            for region_id, degree in indegree.items()
            if degree == 0
        ]
        heapq.heapify(ready)
        result = []
        while ready:
            _, region_id = heapq.heappop(ready)
            result.append(region_id)
            for child in self._children[region_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(ready, (self._text_order[child], child))
        if len(result) != len(self._regions):
            raise ValueError("Region-DAG contains a cycle")
        return tuple(result)

    def parents(self, region_id: str) -> tuple[str, ...]:
        return self._parents[self._require_region(region_id)]

    def ancestors(self, region_id: str) -> tuple[str, ...]:
        region_id = self._require_region(region_id)
        found = set()
        pending = list(self._parents[region_id])
        while pending:
            ancestor = pending.pop()
            if ancestor in found:
                continue
            found.add(ancestor)
            pending.extend(self._parents[ancestor])
        return tuple(candidate for candidate in self._topological if candidate in found)

    def descendants(self, region_id: str) -> tuple[str, ...]:
        region_id = self._require_region(region_id)
        found = set()
        pending = list(self._children[region_id])
        while pending:
            descendant = pending.pop()
            if descendant in found:
                continue
            found.add(descendant)
            pending.extend(self._children[descendant])
        return tuple(candidate for candidate in self._topological if candidate in found)

    def invalidation_closure(self, edited_regions: RegionSelection) -> tuple[str, ...]:
        edited = self._selection(edited_regions)
        closure = set(edited)
        for region_id in edited:
            closure.update(self.descendants(region_id))
        return tuple(
            region_id for region_id in self._topological if region_id in closure
        )

    def versions_match(
        self,
        region_id: str,
        cached_parent_versions: Union[Mapping[str, int], Iterable[tuple[str, int]]],
    ) -> bool:
        region = self._regions[self._require_region(region_id)]
        expected = region.recorded_parent_versions
        if isinstance(cached_parent_versions, Mapping):
            if set(cached_parent_versions) != set(region.parent_region_ids):
                return False
            observed = tuple(
                (parent, int(cached_parent_versions[parent]))
                for parent in region.parent_region_ids
            )
        else:
            try:
                observed = tuple(
                    (str(parent), int(version))
                    for parent, version in cached_parent_versions
                )
            except (TypeError, ValueError):
                return False
        return observed == expected

    def topological_order(self) -> tuple[str, ...]:
        return self._topological

    def earliest_invalidated_position(self, edited_regions: RegionSelection) -> int:
        closure = self.invalidation_closure(edited_regions)
        return min(self._regions[region_id].start for region_id in closure)

    def conservative_gdn_replay_positions(
        self, edited_regions: RegionSelection
    ) -> tuple[int, ...]:
        replay_start = self.earliest_invalidated_position(edited_regions)
        return tuple(range(replay_start, self.spec.sequence_length))
