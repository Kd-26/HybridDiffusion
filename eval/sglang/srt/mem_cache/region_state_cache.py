"""Versioned exact state cache for Cluster-1 hybrid execution."""

from __future__ import annotations

import copy
from collections import OrderedDict, Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Tuple

import torch


class RegionStateMissReason(str, Enum):
    NOT_FOUND = "not_found"
    TOKEN_MISMATCH = "token_mismatch"
    POSITION_MISMATCH = "position_mismatch"
    MODEL_MISMATCH = "model_mismatch"
    ADAPTER_MISMATCH = "adapter_mismatch"
    CONTRACT_MISMATCH = "contract_mismatch"
    VERSION_MISMATCH = "version_mismatch"
    PARENT_VERSION_MISMATCH = "parent_version_mismatch"
    REQUEST_MISMATCH = "request_mismatch"
    RECYCLED_REQUEST_SLOT = "recycled_request_slot"


@dataclass(frozen=True)
class RegionStateKey:
    request_id: str
    request_pool_idx: int
    request_slot_generation: int
    region_id: str
    region_version: int
    boundary: int
    token_hash: str
    position_hash: str
    model_identity: str
    model_revision: str
    adapter_identity: str
    adapter_revision: str
    attention_contract_id: str
    parent_region_versions: Tuple[Tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "parent_region_versions", tuple(self.parent_region_versions)
        )
        if not self.request_id or not self.region_id:
            raise ValueError("request_id and region_id must be nonempty")
        if self.request_pool_idx < 0 or self.request_slot_generation < 0:
            raise ValueError("request slot identity must be nonnegative")
        if self.region_version < 0 or self.boundary < 0:
            raise ValueError("region_version and boundary must be nonnegative")
        if any(version < 0 for _, version in self.parent_region_versions):
            raise ValueError("parent region versions must be nonnegative")


@dataclass(frozen=True)
class KVPrefixReference:
    """Descriptor for request-owned KV pages; full K/V tensors are not copied."""

    request_id: str
    request_pool_idx: int
    request_slot_generation: int
    pool_identity: int
    locations: Any
    valid_length: int
    retained_by_request: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "locations",
            canonicalize_kv_prefix_locations(self.locations, self.valid_length),
        )


def canonicalize_kv_prefix_locations(
    locations: Any,
    valid_length: int,
    *,
    pool_size: Optional[int] = None,
) -> torch.Tensor:
    """Copy physical KV locations into the canonical int64 descriptor format."""
    if not torch.is_tensor(locations):
        raise TypeError("KV prefix locations must be a tensor")
    if locations.ndim != 1:
        raise ValueError("KV prefix locations must be one-dimensional")
    if int(locations.numel()) != int(valid_length):
        raise ValueError(
            "KV prefix location count must equal valid_length: "
            f"{locations.numel()} != {valid_length}"
        )
    canonical = locations.detach().to(dtype=torch.int64, copy=True).contiguous()
    if canonical.numel() and not bool((canonical >= 0).all().item()):
        raise ValueError("KV prefix locations must be non-negative")
    if (
        pool_size is not None
        and canonical.numel()
        and not bool((canonical < int(pool_size)).all().item())
    ):
        raise ValueError(
            f"KV prefix locations must be below KV pool size {int(pool_size)}"
        )
    return canonical


@dataclass(frozen=True)
class RegionState:
    key: RegionStateKey
    kv_prefix: KVPrefixReference
    kv_valid_length: int
    kv_owner_request_id: str
    gdn_conv_states: Tuple[Any, ...]
    gdn_recurrent_states: Any
    boundary: int


@dataclass(frozen=True)
class RegionStateLookup:
    state: Optional[RegionState]
    miss_reason: Optional[RegionStateMissReason] = None

    @property
    def hit(self) -> bool:
        return self.state is not None


def clone_state_value(value: Any) -> Any:
    """Clone tensors without changing device/dtype/layout; copy other values."""
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(clone_state_value(x) for x in value)
    if isinstance(value, list):
        return [clone_state_value(x) for x in value]
    return copy.deepcopy(value)


class RegionStateCache:
    """Bounded LRU cache with diagnostic compatibility misses."""

    def __init__(
        self,
        max_entries: int = 128,
        *,
        on_evict: Optional[Callable[[RegionState], None]] = None,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = int(max_entries)
        self._entries: OrderedDict[RegionStateKey, RegionState] = OrderedDict()
        self._on_evict = on_evict
        self.hit_count = 0
        self.miss_counts: Counter[str] = Counter()

    def __len__(self) -> int:
        return len(self._entries)

    def put(self, state: RegionState) -> None:
        if state.boundary != state.key.boundary:
            raise ValueError("state boundary must match its key")
        if state.kv_owner_request_id != state.key.request_id:
            raise ValueError("KV owner must match the state request")
        if state.kv_valid_length != state.boundary:
            raise ValueError("KV valid length must equal the sealed boundary")
        frozen = RegionState(
            key=state.key,
            kv_prefix=KVPrefixReference(
                **{
                    **state.kv_prefix.__dict__,
                    "locations": clone_state_value(state.kv_prefix.locations),
                }
            ),
            kv_valid_length=state.kv_valid_length,
            kv_owner_request_id=state.kv_owner_request_id,
            gdn_conv_states=tuple(clone_state_value(state.gdn_conv_states)),
            gdn_recurrent_states=clone_state_value(state.gdn_recurrent_states),
            boundary=state.boundary,
        )
        previous = self._entries.pop(state.key, None)
        if previous is not None:
            self._release(previous)
        self._entries[state.key] = frozen
        while len(self._entries) > self.max_entries:
            _, evicted = self._entries.popitem(last=False)
            self._release(evicted)

    def get(
        self, key: RegionStateKey, *, current_slot_generation: Optional[int] = None
    ) -> RegionStateLookup:
        if (
            current_slot_generation is not None
            and current_slot_generation != key.request_slot_generation
        ):
            return self._miss(RegionStateMissReason.RECYCLED_REQUEST_SLOT)
        state = self._entries.get(key)
        if state is not None:
            self._entries.move_to_end(key)
            self.hit_count += 1
            return RegionStateLookup(state=state)
        return self._miss(self._diagnose_miss(key))

    def contains(self, key: RegionStateKey) -> bool:
        return key in self._entries

    def peek(self, key: RegionStateKey) -> Optional[RegionState]:
        """Inspect an entry for diagnostics without changing LRU/statistics."""
        return self._entries.get(key)

    def invalidate_region(self, request_id: str, region_id: str) -> int:
        return self._invalidate(
            lambda key: key.request_id == request_id and key.region_id == region_id
        )

    def invalidate_key(self, key: RegionStateKey) -> bool:
        state = self._entries.pop(key, None)
        if state is None:
            return False
        self._release(state)
        return True

    def invalidate_request(self, request_id: str) -> int:
        return self._invalidate(lambda key: key.request_id == request_id)

    def clear(self) -> None:
        for state in self._entries.values():
            self._release(state)
        self._entries.clear()

    def _invalidate(self, predicate: Callable[[RegionStateKey], bool]) -> int:
        keys = [key for key in self._entries if predicate(key)]
        for key in keys:
            self._release(self._entries.pop(key))
        return len(keys)

    def _release(self, state: RegionState) -> None:
        if self._on_evict is not None:
            self._on_evict(state)

    def _miss(self, reason: RegionStateMissReason) -> RegionStateLookup:
        self.miss_counts[reason.value] += 1
        return RegionStateLookup(state=None, miss_reason=reason)

    def _diagnose_miss(self, key: RegionStateKey) -> RegionStateMissReason:
        candidates = list(self._entries.keys())
        same_request = [k for k in candidates if k.request_id == key.request_id]
        if not same_request:
            same_slot = [
                k for k in candidates if k.request_pool_idx == key.request_pool_idx
            ]
            return (
                RegionStateMissReason.REQUEST_MISMATCH
                if same_slot
                else RegionStateMissReason.NOT_FOUND
            )
        same_region = [k for k in same_request if k.region_id == key.region_id]
        if not same_region:
            return RegionStateMissReason.NOT_FOUND
        checks = (
            ("token_hash", RegionStateMissReason.TOKEN_MISMATCH),
            ("position_hash", RegionStateMissReason.POSITION_MISMATCH),
            ("model_identity", RegionStateMissReason.MODEL_MISMATCH),
            ("model_revision", RegionStateMissReason.MODEL_MISMATCH),
            ("adapter_identity", RegionStateMissReason.ADAPTER_MISMATCH),
            ("adapter_revision", RegionStateMissReason.ADAPTER_MISMATCH),
            ("attention_contract_id", RegionStateMissReason.CONTRACT_MISMATCH),
            ("region_version", RegionStateMissReason.VERSION_MISMATCH),
            (
                "parent_region_versions",
                RegionStateMissReason.PARENT_VERSION_MISMATCH,
            ),
        )
        for field, reason in checks:
            if all(
                getattr(candidate, field) != getattr(key, field)
                for candidate in same_region
            ):
                return reason
        if all(
            candidate.request_slot_generation != key.request_slot_generation
            for candidate in same_region
        ):
            return RegionStateMissReason.RECYCLED_REQUEST_SLOT
        return RegionStateMissReason.NOT_FOUND
