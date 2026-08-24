"""Cluster-1 execution contract for one causal prefix and diffusion suffix."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any, Mapping, Tuple


HYBRID_ATTENTION_CONTRACT_V1 = "causal_prefix_diffusion_suffix_v1"


@dataclass(frozen=True)
class HybridBoundaryCommit:
    """Immutable model-produced publication of one stable-prefix advance."""

    request_id: str
    request_pool_idx: int
    request_slot_generation: int
    region_id: str
    previous_boundary: int
    previous_region_version: int
    previous_token_hash: str
    previous_position_hash: str
    committed_advance: int
    committed_token_ids: Tuple[int, ...]
    new_boundary: int
    new_region_version: int
    token_hash: str
    position_hash: str
    model_identity: str
    model_revision: str
    adapter_identity: str
    adapter_revision: str
    attention_contract_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "committed_token_ids",
            tuple(int(token_id) for token_id in self.committed_token_ids),
        )
        if not self.request_id or not self.region_id:
            raise ValueError("hybrid boundary commit identity must be nonempty")
        if self.request_pool_idx < 0 or self.request_slot_generation < 0:
            raise ValueError("hybrid boundary commit slot identity is invalid")
        if self.previous_boundary < 0 or self.previous_region_version < 0:
            raise ValueError("hybrid boundary commit origin is invalid")
        if self.committed_advance <= 0:
            raise ValueError("hybrid boundary commit must advance at least one token")
        if len(self.committed_token_ids) != self.committed_advance:
            raise ValueError("committed token count must equal committed_advance")
        if self.new_boundary != self.previous_boundary + self.committed_advance:
            raise ValueError("hybrid boundary commit has inconsistent boundary")
        if self.new_region_version != self.previous_region_version + 1:
            raise ValueError("hybrid boundary commit has inconsistent version")
        if (
            not self.previous_token_hash
            or not self.previous_position_hash
            or not self.token_hash
            or not self.position_hash
        ):
            raise ValueError("hybrid boundary commit hashes must be nonempty")
        if self.attention_contract_id != HYBRID_ATTENTION_CONTRACT_V1:
            raise ValueError("hybrid boundary commit attention contract is invalid")


def hash_token_ids(token_ids) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def hash_positions(start: int, end: int) -> str:
    digest = hashlib.sha256()
    digest.update(int(start).to_bytes(8, "little", signed=True))
    digest.update(int(end).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def extend_token_hash(previous_hash: str, token_ids) -> str:
    """Deterministically advance a sealed-prefix digest with accepted tokens."""
    digest = hashlib.sha256()
    digest.update(previous_hash.encode("ascii"))
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


class HybridExecutionRoute(str, Enum):
    PREFIX_DIFFUSION = "prefix_diffusion"


@dataclass(frozen=True, order=True)
class PositionInterval:
    """Compact half-open token interval; positions are never materialized."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Invalid half-open interval [{self.start}, {self.end})")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class HybridExecutionSpec:
    """Immutable Stage-1 region layout.

    The tuple-based fields deliberately leave room for multiple regions later,
    while validation rejects every layout except one prefix and one suffix.
    """

    route: HybridExecutionRoute
    ar_boundary: int
    region_ids: Tuple[str, ...]
    region_versions: Tuple[int, ...]
    parent_regions: Tuple[Tuple[str, ...], ...]
    stable_positions: Tuple[PositionInterval, ...]
    active_positions: Tuple[PositionInterval, ...]
    diffusion_steps: int
    attention_contract_id: str = HYBRID_ATTENTION_CONTRACT_V1

    def __post_init__(self) -> None:
        object.__setattr__(self, "route", HybridExecutionRoute(self.route))
        object.__setattr__(self, "region_ids", tuple(self.region_ids))
        object.__setattr__(self, "region_versions", tuple(self.region_versions))
        object.__setattr__(
            self, "parent_regions", tuple(tuple(x) for x in self.parent_regions)
        )
        object.__setattr__(self, "stable_positions", tuple(self.stable_positions))
        object.__setattr__(self, "active_positions", tuple(self.active_positions))
        self.validate()

    @classmethod
    def prefix_diffusion(
        cls,
        *,
        ar_boundary: int,
        sequence_length: int,
        diffusion_steps: int,
        prefix_region_id: str = "causal_prefix",
        suffix_region_id: str = "diffusion_suffix",
        prefix_version: int = 0,
        suffix_version: int = 0,
        attention_contract_id: str = HYBRID_ATTENTION_CONTRACT_V1,
    ) -> "HybridExecutionSpec":
        return cls(
            route=HybridExecutionRoute.PREFIX_DIFFUSION,
            ar_boundary=ar_boundary,
            region_ids=(prefix_region_id, suffix_region_id),
            region_versions=(prefix_version, suffix_version),
            parent_regions=((), (prefix_region_id,)),
            stable_positions=(PositionInterval(0, ar_boundary),),
            active_positions=(PositionInterval(ar_boundary, sequence_length),),
            diffusion_steps=diffusion_steps,
            attention_contract_id=attention_contract_id,
        )

    @property
    def sequence_length(self) -> int:
        return self.active_positions[0].end

    def validate(self) -> None:
        if self.route is not HybridExecutionRoute.PREFIX_DIFFUSION:
            raise ValueError("Cluster 1 only supports route=prefix_diffusion")
        if self.ar_boundary < 0:
            raise ValueError("ar_boundary must be nonnegative")
        if len(self.region_ids) != 2 or len(set(self.region_ids)) != 2:
            raise ValueError("Cluster 1 requires exactly two distinct regions")
        if not all(self.region_ids):
            raise ValueError("region IDs must be nonempty")
        if len(self.region_versions) != 2 or any(v < 0 for v in self.region_versions):
            raise ValueError("Cluster 1 requires two nonnegative region versions")
        if len(self.parent_regions) != 2:
            raise ValueError("parent_regions must have one entry per region")
        if self.parent_regions[0]:
            raise ValueError("the stable prefix cannot depend on another region")
        if self.parent_regions[1] != (self.region_ids[0],):
            raise ValueError("the active suffix must depend only on the stable prefix")
        if len(self.stable_positions) != 1 or len(self.active_positions) != 1:
            raise ValueError("Cluster 1 supports one stable span and one active span")
        stable, active = self.stable_positions[0], self.active_positions[0]
        if stable != PositionInterval(0, self.ar_boundary):
            raise ValueError("stable interval must be [0, ar_boundary)")
        if active.start != self.ar_boundary:
            raise ValueError("active interval must begin at ar_boundary")
        if active.end <= active.start:
            raise ValueError("active diffusion suffix must be nonempty")
        if stable.end != active.start:
            raise ValueError("stable and active regions must have no overlap or gap")
        if self.diffusion_steps <= 0:
            raise ValueError("diffusion_steps must be positive")
        if self.attention_contract_id != HYBRID_ATTENTION_CONTRACT_V1:
            raise ValueError(
                f"Unrecognized attention contract {self.attention_contract_id!r}"
            )

    def advance_boundary(
        self, new_boundary: int, *, sequence_length: int | None = None
    ) -> "HybridExecutionSpec":
        """Publish a new immutable canonical layout after token acceptance."""
        if new_boundary < self.ar_boundary:
            raise ValueError("a canonical boundary cannot move backwards")
        if sequence_length is None:
            sequence_length = max(self.sequence_length, new_boundary)
        if sequence_length < new_boundary:
            raise ValueError("sequence_length cannot precede the new boundary")
        return replace(
            self,
            ar_boundary=new_boundary,
            region_versions=tuple(v + 1 for v in self.region_versions),
            stable_positions=(PositionInterval(0, new_boundary),),
            active_positions=(PositionInterval(new_boundary, sequence_length),),
        )

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["route"] = self.route.value
        return record

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HybridExecutionSpec":
        return cls(
            route=HybridExecutionRoute(value["route"]),
            ar_boundary=int(value["ar_boundary"]),
            region_ids=tuple(value["region_ids"]),
            region_versions=tuple(int(x) for x in value["region_versions"]),
            parent_regions=tuple(tuple(x) for x in value["parent_regions"]),
            stable_positions=tuple(
                (
                    interval
                    if isinstance(interval, PositionInterval)
                    else PositionInterval(**interval)
                )
                for interval in value["stable_positions"]
            ),
            active_positions=tuple(
                (
                    interval
                    if isinstance(interval, PositionInterval)
                    else PositionInterval(**interval)
                )
                for interval in value["active_positions"]
            ),
            diffusion_steps=int(value["diffusion_steps"]),
            attention_contract_id=str(value["attention_contract_id"]),
        )
