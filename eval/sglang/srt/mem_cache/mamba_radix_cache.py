from __future__ import annotations

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the hybrid (full and Mamba) KV cache.
"""

import heapq
import json
import os
from collections import defaultdict
from functools import lru_cache
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from numpy import float64

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, mamba_owner_trace
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

import logging

logger = logging.getLogger(__name__)
_EXTRA_BUFFER_TRACE = (
    os.getenv("SGLANG_HYBRID_DIFFUSION_SELF_SPEC_EXTRA_BUFFER_TRACE", "0") == "1"
)
_DISABLE_DLLM_GENERATED_PREFIX_INSERT = (
    os.getenv("SGLANG_HYBRID_DIFFUSION_SELF_SPEC_DISABLE_GENERATED_PREFIX_INSERT", "0")
    == "1"
)


class TreeNode:

    counter = 0
    last_access_time_counter_float = float64(1.0)

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        self.mamba_value: Optional[torch.Tensor] = None
        self.mamba_host_value: Optional[torch.Tensor] = None
        # invariant: for any node, if mamba_lock_ref is locked, full_lock_ref must be locked;
        # if full_lock_ref is locked, mamba_lock_ref doesn't need to be locked. So,
        # full_lock_ref is always >= mamba_lock_ref.
        # for full_lock, once it is locked, its parent must be locked as well
        # for mamba_lock, it only need lock node itself
        self.full_lock_ref = 0
        self.mamba_lock_ref = 0
        # last access time is only used for sanity check. LRU is maintained by the lru list.
        self.last_access_time = get_last_access_time()

        self.hit_count = 0
        self.host_ref_counter = 0
        self.host_mamba_ref_counter = 0
        # store the host indices of KV cache
        self.host_value = None
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None

        # for lru list, invariant:
        # 1. prev has greater last_access_time
        # 2. next has smaller last_access_time
        self.prev = None
        self.next = None
        self.mamba_prev = None
        self.mamba_next = None
        self.host_mamba_prev = None
        self.host_mamba_next = None

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    @property
    def mamba_evicted(self):
        return self.mamba_value is None

    @property
    def backuped(self):
        return self.host_value is not None

    @property
    def mamba_backuped(self):
        return self.mamba_host_value is not None

    def protect_host(self):
        """Protect the host KV value from eviction."""
        self.host_ref_counter += 1

    def release_host(self):
        """Release the host KV value, allowing it to be evicted."""
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")

    def protect_host_mamba(self):
        """Protect the host mamba value from eviction."""
        self.host_mamba_ref_counter += 1

    def release_host_mamba(self):
        """Release the host mamba value, allowing it to be evicted."""
        if self.host_mamba_ref_counter > 0:
            self.host_mamba_ref_counter -= 1
        else:
            raise RuntimeError("Host mamba reference counter is already zero.")

    def get_last_hash_value(self) -> Optional[str]:
        """Returns the hash value of the last page in this node."""
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    @lru_cache(maxsize=1)
    def get_prefix_hash_values(self, node: "TreeNode") -> List[str]:
        if node is None or node.hash_value is None:
            return []
        return node.get_prefix_hash_values(node.parent) + node.hash_value

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def get_last_access_time() -> float64:
    ret = TreeNode.last_access_time_counter_float
    TreeNode.last_access_time_counter_float += 1.0
    return ret


class LRUList:
    def __init__(self, mamba: bool = False):
        self.mamba = mamba
        if self.mamba:
            self.prv = "mamba_prev"
            self.nxt = "mamba_next"
            self.lock_ref = "mamba_lock_ref"
        else:
            self.prv = "prev"
            self.nxt = "next"
            self.lock_ref = "full_lock_ref"
        # Initialize dummy head and tail nodes
        self.head = TreeNode()  # Most recently used side
        self.tail = TreeNode()  # Least recently used side
        setattr(self.head, self.nxt, self.tail)  # self.head.next = self.tail
        setattr(self.tail, self.prv, self.head)  # self.tail.prev = self.head
        self.cache = {}

    def _add_node(self, node):
        """Helper to add node right after head (most recently used)"""
        self._add_node_after(self.head, node)

    def _add_node_after(self, old_node, new_node):
        """Helper to add node right after old_node"""
        setattr(new_node, self.prv, old_node)  # new_node.prev = old_node
        setattr(
            new_node, self.nxt, getattr(old_node, self.nxt)
        )  # new_node.next = old_node.next
        setattr(
            getattr(old_node, self.nxt), self.prv, new_node
        )  # old_node.next.prev = new_node
        setattr(old_node, self.nxt, new_node)  # old_node.next = new_node

    def _remove_node(self, node):
        """Helper to remove node from linked list"""
        setattr(
            getattr(node, self.prv), self.nxt, getattr(node, self.nxt)
        )  # node.prev.next = node.next
        setattr(
            getattr(node, self.nxt), self.prv, getattr(node, self.prv)
        )  # node.next.prev = node.prev

    def _get_lru(self) -> Optional[TreeNode]:
        """
        Get the least recently used node
        """
        if len(self.cache) == 0:
            return None
        return getattr(self.tail, self.prv)

    def reset_node_mru(self, node):
        """
        Move a (existing) node to most recently used position
        """
        assert node.id in self.cache, f"Resetting node {node.id=} not in lru list"
        assert (
            not self.mamba or node.mamba_value is not None
        ), f"Resetting mamba tombstone node in mamba lru list: {node.id=}"
        self._remove_node(node)
        self._add_node(node)

    def reset_node_and_parents_mru(self, node, root_node):
        """
        Move an (existing) node and its parents to most recently used position. Child node is
        more recently used than parent node.
        """
        prev_node = self.head
        while node != root_node:
            if not self.mamba or node.mamba_value is not None:
                assert (
                    node.id in self.cache
                ), f"Resetting node {node.id=} not in lru list when resetting node and parents mru"
                self._remove_node(node)
                self._add_node_after(prev_node, node)
                prev_node = node
            node = node.parent

    def insert_mru(self, node):
        """
        Insert a (new) node as most recently used
        """
        assert (
            not self.mamba or node.mamba_value is not None
        ), f"Inserting mamba tombstone node in mamba lru list: {node.id=}"
        assert (
            node.id not in self.cache
        ), f"Inserting node {node.id=} already in lru list, existing node: {self.cache[node.id].id=}"
        self.cache[node.id] = node
        self._add_node(node)

    def remove_node(self, node: TreeNode):
        """
        Remove node from lru list
        """
        assert node.id in self.cache, f"Removing node {node.id=} not in lru list"
        assert (
            not self.mamba or node.mamba_value is not None
        ), f"Removing mamba tombstone node from mamba lru list: {node.id=}"
        del self.cache[node.id]
        self._remove_node(node)

    def get_lru_no_lock(self) -> Optional[TreeNode]:
        """
        Get the least recently used node that is not locked
        """
        return self.get_prev_no_lock(self.tail, check_id=False)

    def get_leaf_lru_no_lock(self) -> Optional[TreeNode]:
        """
        Get the least recently used leaf node that is not locked
        """
        return self.get_prev_leaf_no_lock(self.tail, check_id=False)

    def get_prev_no_lock(
        self, node: TreeNode, check_id: bool = True
    ) -> Optional[TreeNode]:
        """
        Get the previous (i.e. more recently used) node that is not locked
        """
        if check_id:
            assert (
                node.id in self.cache
            ), f"Getting prev of node {node.id=} not in lru list"
        x = getattr(node, self.prv)  # x = node.prev
        while getattr(x, self.lock_ref) > 0:
            x = getattr(x, self.prv)  # x = x.prev
        # if x is the head, it means there is no node in the lru list without lock
        if x == self.head:
            return None
        return x

    def get_prev_leaf_no_lock(self, node: TreeNode, check_id: bool = True):
        """
        Get the previous (i.e. more recently used) leaf node that is not locked
        """
        if check_id:
            assert (
                node.id in self.cache
            ), f"Getting prev of node {node.id=} not in lru list"
        x = getattr(node, self.prv)  # x = node.prev
        while getattr(x, self.lock_ref) > 0 or len(x.children) > 0:
            x = getattr(x, self.prv)  # x = x.prev
        # if x is the head, it means there is no leaf node in the lru list without lock
        if x == self.head:
            return None
        return x

    def in_list(self, node: Optional[TreeNode]):
        """
        Check if the node is in the lru list
        """
        if not node:
            return False
        return node.id in self.cache

    def pretty_print(self, tree_cache: Optional["MambaRadixCache"] = None):
        """
        Pretty print the lru list
        """
        msg = f"{self.mamba=} LRU list: "
        x_lru = self._get_lru()
        while x_lru is not None and x_lru.id in self.cache:
            msg += f"[{x_lru.id}] {x_lru.last_access_time:f} -> "
            x_lru = getattr(x_lru, self.prv)
        print(msg)

        if not tree_cache:
            return
        msg = f"{self.mamba=} Nodes (sorted by last_access_time): "
        if self.mamba:
            nodes = tree_cache._collect_nontombstone_nodes()
        else:
            nodes = tree_cache._collect_all_nodes()
        heapq.heapify(nodes)
        while len(nodes):
            x = heapq.heappop(nodes)
            msg += f"[{x.id}] {x.last_access_time:f} -> "
        print(msg)

    # Note: this is expensive, only use for debug
    def sanity_check_evictable_size(self, tree_cache: Optional["MambaRadixCache"] = None):
        """
        Check the evictable size (i.e. the size of the nodes that are not locked)
        """
        node = self.get_lru_no_lock()
        evictable_size = 0
        while self.in_list(node):
            evictable_size += (
                tree_cache._kv_value_size(node.value)
                if not self.mamba and tree_cache is not None
                else len(node.value) if not self.mamba else len(node.mamba_value)
            )
            node = self.get_prev_no_lock(node)
        return evictable_size

    # Note: this is expensive, only use for debug or idle check
    def sanity_check(self, tree_cache: "MambaRadixCache"):
        """
        Check if the lru list is valid by rebuilding the lru list from the tree, heapifying it, and
        checking if the lru list is valid.
        """
        try:
            if self.mamba:
                nodes = tree_cache._collect_nontombstone_nodes()
            else:
                nodes = tree_cache._collect_all_nodes()
            total_nodes = len(nodes)
            total_lru = len(self.cache)
            # heapify based on last_access_time
            heapq.heapify(nodes)
            # the root node is not in the lru list
            assert len(nodes) == (
                total_lru + (0 if self.mamba else 1)
            ), f"len(nodes): {len(nodes)}, total_lru: {total_lru}"

            x_lru = self._get_lru()
            while len(nodes):
                x = heapq.heappop(nodes)
                if x == tree_cache.root_node:
                    # root node is not in the lru list
                    continue
                assert (
                    x_lru is not None and x_lru.id in self.cache
                ), f"Incorrect LRU list, x_lru is None or not in cache: {x_lru=}, {x.id=}"

                assert (
                    x == x_lru
                ), f"Incorrect LRU list, {self.mamba=}, x: {x.id=} != x_lru: {x_lru.id=}, {x.last_access_time=}, {x_lru.last_access_time=}"
                assert (
                    x_lru.full_lock_ref == 0
                ), f"x_lru should not be locked when idle, {x_lru.full_lock_ref=}, {x_lru.id=}"
                assert (
                    x_lru.mamba_lock_ref == 0
                ), f"x_lru should not be locked when idle, {x_lru.mamba_lock_ref=}, {x_lru.id=}"
                x_lru = getattr(x, self.prv)

            if self.mamba:
                evictable_size = tree_cache.mamba_evictable_size()
                lru_list_evictable_size = self.sanity_check_evictable_size()
            else:
                evictable_size = tree_cache.full_evictable_size()
                lru_list_evictable_size = self.sanity_check_evictable_size(tree_cache)

            assert (
                evictable_size == lru_list_evictable_size
            ), f"{self.mamba=}, total nodes: {total_nodes}, total lru: {total_lru}, evictable size: {evictable_size} != lru list evictable size: {lru_list_evictable_size}"
        except Exception as e:
            if get_tensor_model_parallel_rank() == 0:
                msg = f"Mamba Radix tree sanity check failed, ping @yizhang2077: {e}"
                logger.error(msg)
                tree_cache.pretty_print()
                tree_cache.full_lru_list.pretty_print(tree_cache)
                tree_cache.mamba_lru_list.pretty_print(tree_cache)
                raise Exception(msg)


class MambaRadixCache(BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        assert isinstance(
            params.token_to_kv_pool_allocator, TokenToKVPoolAllocator
        ) or isinstance(params.token_to_kv_pool_allocator, PagedTokenToKVPoolAllocator)
        self.req_to_token_pool: HybridReqToTokenPool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator

        self.page_size = params.page_size
        self.disable = params.disable
        self.enable_mamba_extra_buffer = params.enable_mamba_extra_buffer

        if not self.enable_mamba_extra_buffer:
            assert (
                self.page_size == 1
            ), f"Page size must be 1 for MambaRadixCache v1, got {self.page_size}"

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if params.enable_metrics:
            self.init_metrics_collector()

        self.reset()

    ##### Public API #####

    def supports_mamba(self) -> bool:
        return True

    def _kv_value_size(self, value: Optional[torch.Tensor]) -> int:
        if value is None:
            return 0
        if self.page_size == 1:
            return len(value)
        return int(torch.unique(value // self.page_size).numel()) * self.page_size

    def reset(self) -> None:
        self.root_node = TreeNode()
        self.root_node.key = RadixKey([], None)
        self.root_node.value = []
        self.root_node.full_lock_ref = 1
        self.root_node.mamba_lock_ref = 1
        self.full_evictable_size_ = 0
        self.mamba_evictable_size_ = 0
        self.full_protected_size_ = 0
        self.mamba_protected_size_ = 0
        # LRU lists are used to maintain the order of eviction of the nodes in the tree
        self.full_lru_list = LRUList(mamba=False)
        self.mamba_lru_list = LRUList(mamba=True)

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Find the matching prefix from the radix tree.
        Args:
            params: MatchPrefixParams containing key and optional Mamba-specific parameters.
        Returns:
            A tuple of a tensor of matching prefix token IDs and
            the last node that contains the prefix values. Note that
            this API can modify the internal state of the Radix tree.
            The last node create a new child if the prefix is shorter
            than the last node's value.
        """
        key = self._match_pre_processor(params)
        if key is None:
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
            )

        value, last_node, best_value_len = self._match_prefix_helper(key)
        return self._match_post_processor(params, value, last_node, best_value_len)

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0, mamba_exist=False)

        key = params.key
        value = params.value
        mamba_value = params.mamba_value
        prev_prefix_len = params.prev_prefix_len

        if value is None:
            value = torch.tensor([x for x in key.token_ids], dtype=torch.int64)
        prefix_len, mamba_exist = self._insert_helper(
            self.root_node, key, value, mamba_value, params.chunked, prev_prefix_len
        )
        mamba_owner_trace(
            "radix_insert",
            mamba_value,
            prefix_len=prefix_len,
            mamba_exist=mamba_exist,
            value_len=len(value) if value is not None else None,
            prev_prefix_len=prev_prefix_len,
        )
        return InsertResult(prefix_len=prefix_len, mamba_exist=mamba_exist)

    def cache_finished_req(self, req: Req, is_insert: bool = True) -> None:
        """Cache request when it finishes."""
        kv_committed_len = req.pop_committed_kv_cache()
        mamba_owner_trace(
            "radix_cache_finished_enter",
            getattr(req, "mamba_pool_idx", None),
            req_pool_idx=req.req_pool_idx,
            rid=getattr(req, "rid", None),
            is_insert=is_insert,
            kv_committed_len=kv_committed_len,
            cache_protected_len=req.cache_protected_len,
            mamba_last_track_seqlen=getattr(req, "mamba_last_track_seqlen", None),
            has_ping_pong=getattr(req, "mamba_ping_pong_track_buffer", None) is not None,
        )
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free_mamba_cache(req)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]

        if is_insert:
            cache_len = (
                req.mamba_last_track_seqlen
                if self.enable_mamba_extra_buffer
                else len(token_ids)
            )
            if cache_len is None:
                cache_len = 0
            if cache_len != len(token_ids):
                cache_end_idx = max(cache_len, req.cache_protected_len)
                self.token_to_kv_pool_allocator.free(kv_indices[cache_end_idx:])
                token_ids = token_ids[:cache_len]
                kv_indices = kv_indices[:cache_len]

            if self.page_size != 1:
                page_aligned_len = len(kv_indices) // self.page_size * self.page_size
                page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
                    dtype=torch.int64, copy=True
                )
            else:
                page_aligned_len = len(kv_indices)
                page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)

            assert (
                cache_len == page_aligned_len
            ), f"It is required {cache_len=}, {page_aligned_len=}, {kv_committed_len=}, {len(req.origin_input_ids)=}, {len(req.output_ids)=} ping @yizhang2077 if you see this"

            # Radix Cache takes one ref in memory pool
            # insert the token_ids and kv_indices into the radix tree
            if self.enable_mamba_extra_buffer:
                mamba_ping_pong_track_buffer_to_keep = (
                    self.req_to_token_pool.get_mamba_ping_pong_other_idx(
                        req.mamba_next_track_idx
                    )
                )
                mamba_value = (
                    req.mamba_ping_pong_track_buffer[
                        mamba_ping_pong_track_buffer_to_keep
                    ]
                    .unsqueeze(-1)
                    .clone()
                )
            else:
                mamba_value = req.mamba_pool_idx.unsqueeze(-1).clone()
                mamba_ping_pong_track_buffer_to_keep = None

            result = self.insert(
                InsertParams(
                    key=RadixKey(token_ids[:page_aligned_len], req.extra_key),
                    value=page_aligned_kv_indices,
                    mamba_value=mamba_value,
                    prev_prefix_len=req.cache_protected_len,
                )
            )
            mamba_exist = result.mamba_exist
            mamba_owner_trace(
                "radix_cache_finished_insert",
                mamba_value,
                req_pool_idx=req.req_pool_idx,
                rid=getattr(req, "rid", None),
                mamba_exist=mamba_exist,
                page_aligned_len=page_aligned_len,
            )
        else:
            kv_indices_to_free = kv_indices[req.cache_protected_len :]
            if self.page_size > 1 and req.cache_protected_len > 0:
                protected_pages = torch.unique(
                    kv_indices[: req.cache_protected_len] // self.page_size
                )
                kv_indices_to_free = kv_indices_to_free[
                    ~torch.isin(kv_indices_to_free // self.page_size, protected_pages)
                ]
            self.token_to_kv_pool_allocator.free(kv_indices_to_free)
            mamba_exist = True
            mamba_owner_trace(
                "radix_cache_finished_skip_insert",
                getattr(req, "mamba_pool_idx", None),
                req_pool_idx=req.req_pool_idx,
                rid=getattr(req, "rid", None),
                cache_protected_len=req.cache_protected_len,
            )

        if mamba_exist:
            mamba_ping_pong_track_buffer_to_keep = None

        free_mamba_cache = True if self.enable_mamba_extra_buffer else mamba_exist

        if free_mamba_cache and (
            req.mamba_pool_idx is not None
            or (
                self.enable_mamba_extra_buffer
                and req.mamba_ping_pong_track_buffer is not None
            )
        ):
            self.req_to_token_pool.free_mamba_cache(
                req,
                mamba_ping_pong_track_buffer_to_keep=mamba_ping_pong_track_buffer_to_keep,
            )

        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False) -> None:
        """Cache request when it is unfinished."""
        mamba_owner_trace(
            "radix_cache_unfinished_enter",
            getattr(req, "mamba_pool_idx", None),
            req_pool_idx=req.req_pool_idx,
            rid=getattr(req, "rid", None),
            cache_protected_len=req.cache_protected_len,
            mamba_last_track_seqlen=getattr(req, "mamba_last_track_seqlen", None),
            has_ping_pong=getattr(req, "mamba_ping_pong_track_buffer", None) is not None,
        )

        def _skip_cache_unfinished_req(req: Req) -> None:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(req.fill_ids)
            ]

            # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
            req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
            return

        if req.is_dllm():
            valid_len = (
                req.dllm_kv_valid_len
                if req.dllm_kv_valid_len is not None
                else req.kv_committed_len
            )
            # Do not insert dLLM-generated tokens into the global radix tree here,
            # but preserve request-local accepted KV/GDN state across mixed
            # scheduling. If the request-local Mamba state has already been
            # detached, fall back to the prompt-only safe prefix.
            has_request_local_mamba = (
                req.mamba_pool_idx is not None
                or req.mamba_ping_pong_track_buffer is not None
            )
            prefix_cap = (
                len(req.fill_ids)
                if has_request_local_mamba
                else max(len(req.origin_input_ids), req.cache_protected_len)
            )
            valid_len = max(valid_len, req.cache_protected_len)
            valid_len = min(valid_len, prefix_cap)
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :valid_len
            ]
            req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
            if _EXTRA_BUFFER_TRACE:
                logger.warning(
                    "[HYBRID_DIFFUSION_SELF_SPEC_EXTRA_BUFFER] event=dllm_unfinished_safety_return "
                    "rid=%s req_pool_idx=%s valid_len=%s origin_len=%s "
                    "kv_committed_len=%s dllm_kv_valid_len=%s fill_len=%s "
                    "cache_protected_len=%s mamba_last_track_seqlen=%s "
                    "extra_buffer=%s request_local_mamba=%s phase=%s",
                    getattr(req, "rid", None),
                    req.req_pool_idx,
                    valid_len,
                    len(req.origin_input_ids),
                    req.kv_committed_len,
                    getattr(req, "dllm_kv_valid_len", None),
                    len(req.fill_ids),
                    req.cache_protected_len,
                    getattr(req, "mamba_last_track_seqlen", None),
                    self.enable_mamba_extra_buffer,
                    has_request_local_mamba,
                    getattr(req, "dllm_phase", None),
                )
            return

        token_ids = req.fill_ids
        cache_len = (
            req.mamba_last_track_seqlen
            if self.enable_mamba_extra_buffer
            else len(token_ids)
        )
        if self.disable or cache_len is None:
            return _skip_cache_unfinished_req(req)

        kv_indices_orig = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]
        # kv_indices is the kv indices to be cached
        kv_indices = kv_indices_orig[:cache_len]
        if self.page_size != 1:
            page_aligned_len = len(kv_indices) // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
                dtype=torch.int64, copy=True
            )
        else:
            page_aligned_len = len(kv_indices)
            page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)

        assert page_aligned_len == len(
            kv_indices
        ), f"page_aligned_len != len(kv_indices), {page_aligned_len=}, {len(kv_indices)=}, {cache_len=}, {self.page_size=}, {FLA_CHUNK_SIZE=}"

        page_aligned_token_ids = token_ids[:page_aligned_len]

        if self.enable_mamba_extra_buffer:
            # copy from the ping pong track buffer
            mamba_ping_pong_track_buffer_to_keep = (
                self.req_to_token_pool.get_mamba_ping_pong_other_idx(
                    req.mamba_next_track_idx
                )
            )
            mamba_value = (
                req.mamba_ping_pong_track_buffer[mamba_ping_pong_track_buffer_to_keep]
                .unsqueeze(-1)
                .clone()
            )
        else:
            mamba_value = self.req_to_token_pool.get_mamba_indices(
                req.req_pool_idx
            ).unsqueeze(-1)
        # radix tree mamba value is forked from req space
        mamba_value_forked = self.req_to_token_pool.mamba_pool.fork_from(mamba_value)
        mamba_owner_trace(
            "radix_cache_unfinished_fork",
            mamba_value_forked,
            src=mamba_value,
            req_pool_idx=req.req_pool_idx,
            rid=getattr(req, "rid", None),
            page_aligned_len=page_aligned_len,
        )

        # if alloc mamba cache failed, do evict and alloc again
        if mamba_value_forked is None:
            self.evict(EvictParams(num_tokens=0, mamba_num=1))
            mamba_value_forked = self.req_to_token_pool.mamba_pool.fork_from(
                mamba_value
            )
            assert mamba_value_forked is not None, "Can not alloc mamba cache"
        result = self.insert(
            InsertParams(
                key=RadixKey(page_aligned_token_ids, req.extra_key),
                value=page_aligned_kv_indices,
                mamba_value=mamba_value_forked,
                prev_prefix_len=req.cache_protected_len,
            )
        )
        new_prefix_len, mamba_exist = result.prefix_len, result.mamba_exist
        mamba_owner_trace(
            "radix_cache_unfinished_insert",
            mamba_value_forked,
            req_pool_idx=req.req_pool_idx,
            rid=getattr(req, "rid", None),
            mamba_exist=mamba_exist,
            new_prefix_len=new_prefix_len,
            last_node_id=getattr(result, "last_device_node", None),
        )
        # there is a mamba cache in radix cache, release it
        if mamba_exist:
            mamba_owner_trace(
                "radix_cache_unfinished_free_duplicate_fork",
                mamba_value_forked,
                req_pool_idx=req.req_pool_idx,
                rid=getattr(req, "rid", None),
            )
            self.req_to_token_pool.mamba_pool.free(mamba_value_forked)

        # The prefix indices could be updated, reuse it
        match_result = self.match_prefix(
            MatchPrefixParams(key=RadixKey(page_aligned_token_ids, req.extra_key))
        )
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )

        if not mamba_exist:
            assert torch.equal(new_last_node.mamba_value, mamba_value_forked)

        assert (
            req.cache_protected_len <= len(new_indices) + self.page_size - 1
        ), f"{req.cache_protected_len=}, {len(new_indices)=}, {len(page_aligned_token_ids)=}, {mamba_exist=}"
        assert new_prefix_len <= len(
            new_indices
        ), f"{new_prefix_len=}, {len(new_indices)=}"

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        # NOTE: this is needed for both page_size == 1 and page_size > 1
        req.prefix_indices = torch.cat(
            [new_indices, kv_indices_orig[len(new_indices) :]]
        )
        req.cache_protected_len = len(new_indices)
        req.mamba_last_track_seqlen = None
        req.last_node = new_last_node

    def _dllm_boundary_token_ids(self, req: Req, boundary_len: int) -> List[int]:
        origin_len = len(req.origin_input_ids)
        if boundary_len <= origin_len:
            return req.origin_input_ids[:boundary_len]
        generated_len = boundary_len - origin_len
        return req.origin_input_ids + req.output_ids[:generated_len]

    def _remove_kv_indices_from_allocator_free(self, kv_indices: torch.Tensor) -> None:
        """Mark KV pages as cache-owned even if a grouped free queued them earlier."""
        if kv_indices.numel() == 0:
            return

        allocator = self.token_to_kv_pool_allocator
        page_size = getattr(allocator, "page_size", 1)
        owned_units = (
            torch.unique(kv_indices // page_size) if page_size > 1 else kv_indices
        )

        def _drop_owned_units(units: torch.Tensor) -> torch.Tensor:
            if units is None or units.numel() == 0:
                return units
            return units[~torch.isin(units, owned_units)]

        allocator.free_pages = _drop_owned_units(allocator.free_pages)
        allocator.release_pages = _drop_owned_units(allocator.release_pages)

        if not allocator.is_not_in_free_group and allocator.free_group:
            new_free_group = []
            for free_indices in allocator.free_group:
                if free_indices.numel() == 0:
                    continue
                free_units = (
                    free_indices // page_size if page_size > 1 else free_indices
                )
                kept = free_indices[~torch.isin(free_units, owned_units)]
                if kept.numel() > 0:
                    new_free_group.append(kept)
            allocator.free_group = new_free_group

    def cache_dllm_generated_boundary(
        self, req: Req, boundary_len: Optional[int], reason: str = ""
    ) -> bool:
        """Insert an aligned dLLM generated-prefix boundary into MambaRadixCache.

        This is intentionally not used in the steady decode loop. It is for
        request exit paths where request-local KV/Mamba state would otherwise be
        released and a later resume would need to re-prefill generated tokens.
        """
        if (
            not req.is_dllm()
            or not self.enable_mamba_extra_buffer
            or self.disable
            or _DISABLE_DLLM_GENERATED_PREFIX_INSERT
            or boundary_len is None
        ):
            return False

        if req.mamba_ping_pong_track_buffer is None or req.mamba_next_track_idx is None:
            if _EXTRA_BUFFER_TRACE:
                logger.warning(
                    "[HYBRID_DIFFUSION_SELF_SPEC_EXTRA_BUFFER] event=boundary_insert_skip "
                    "reason=no_ping_pong rid=%s req_pool_idx=%s boundary=%s",
                    getattr(req, "rid", None),
                    req.req_pool_idx,
                    boundary_len,
                )
            return False

        max_token_len = len(req.origin_input_ids) + len(req.output_ids)
        boundary_len = min(boundary_len, req.kv_committed_len, max_token_len)
        if boundary_len <= req.cache_protected_len:
            return False

        chunk = get_global_server_args().mamba_cache_chunk_size
        if (
            boundary_len <= 0
            or boundary_len % self.page_size != 0
            or boundary_len % chunk != 0
        ):
            if _EXTRA_BUFFER_TRACE:
                logger.warning(
                    "[HYBRID_DIFFUSION_SELF_SPEC_EXTRA_BUFFER] event=boundary_insert_skip "
                    "reason=unaligned rid=%s req_pool_idx=%s boundary=%s "
                    "page_size=%s chunk=%s cache_protected_len=%s",
                    getattr(req, "rid", None),
                    req.req_pool_idx,
                    boundary_len,
                    self.page_size,
                    chunk,
                    req.cache_protected_len,
                )
            return False

        token_ids = self._dllm_boundary_token_ids(req, boundary_len)
        if len(token_ids) != boundary_len:
            if _EXTRA_BUFFER_TRACE:
                logger.warning(
                    "[HYBRID_DIFFUSION_SELF_SPEC_EXTRA_BUFFER] event=boundary_insert_skip "
                    "reason=token_len_mismatch rid=%s req_pool_idx=%s "
                    "boundary=%s token_len=%s origin_len=%s output_len=%s",
                    getattr(req, "rid", None),
                    req.req_pool_idx,
                    boundary_len,
                    len(token_ids),
                    len(req.origin_input_ids),
                    len(req.output_ids),
                )
            return False

        kv_indices_orig = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.kv_committed_len
        ]
        kv_indices = kv_indices_orig[:boundary_len].to(dtype=torch.int64, copy=True)
        mamba_ping_pong_track_buffer_to_keep = (
            self.req_to_token_pool.get_mamba_ping_pong_other_idx(
                req.mamba_next_track_idx
            )
        )
        mamba_value = (
            req.mamba_ping_pong_track_buffer[mamba_ping_pong_track_buffer_to_keep]
            .unsqueeze(-1)
            .clone()
        )
        mamba_value_forked = self.req_to_token_pool.mamba_pool.fork_from(mamba_value)
        if mamba_value_forked is None:
            self.evict(EvictParams(num_tokens=0, mamba_num=1))
            mamba_value_forked = self.req_to_token_pool.mamba_pool.fork_from(
                mamba_value
            )
            assert mamba_value_forked is not None, "Can not alloc mamba cache"

        result = self.insert(
            InsertParams(
                key=RadixKey(token_ids, req.extra_key),
                value=kv_indices,
                mamba_value=mamba_value_forked,
                prev_prefix_len=req.cache_protected_len,
            )
        )
        mamba_exist = result.mamba_exist
        if mamba_exist:
            self.req_to_token_pool.mamba_pool.free(mamba_value_forked)

        match_result = self.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids, req.extra_key))
        )
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )
        if not mamba_exist:
            assert torch.equal(new_last_node.mamba_value, mamba_value_forked)
        assert result.prefix_len <= len(new_indices)

        self._remove_kv_indices_from_allocator_free(new_indices)
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )
        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)
        req.prefix_indices = torch.cat(
            [new_indices, kv_indices_orig[len(new_indices) :]]
        )
        req.cache_protected_len = len(new_indices)
        req.mamba_last_track_seqlen = None
        req.last_node = new_last_node

        mamba_owner_trace(
            "radix_cache_dllm_boundary_insert",
            mamba_value_forked,
            req_pool_idx=req.req_pool_idx,
            rid=getattr(req, "rid", None),
            boundary_len=boundary_len,
            mamba_exist=mamba_exist,
            new_prefix_len=result.prefix_len,
            reason=reason,
        )
        if _EXTRA_BUFFER_TRACE:
            logger.warning(
                "[HYBRID_DIFFUSION_SELF_SPEC_EXTRA_BUFFER] event=boundary_insert "
                "rid=%s req_pool_idx=%s boundary=%s new_prefix_len=%s "
                "cache_protected_len=%s reason=%s mamba_exist=%s",
                getattr(req, "rid", None),
                req.req_pool_idx,
                boundary_len,
                result.prefix_len,
                req.cache_protected_len,
                reason,
                mamba_exist,
            )
        dump_path = os.getenv("SGLANG_HYBRID_DIFFUSION_SELF_SPEC_BOUNDARY_DUMP_PATH")
        if dump_path:
            with open(dump_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "rid": getattr(req, "rid", None),
                            "boundary_len": boundary_len,
                            "token_ids": token_ids,
                            "reason": reason,
                        }
                    )
                    + "\n"
                )
        return True

    def pretty_print(self) -> None:
        self._print_helper(self.root_node, 0)
        total_size, total_mamba_size = self._total_size_helper()
        print(f"#full_tokens: {total_size}, #mamba_num: {total_mamba_size}")

    def total_size(self) -> Tuple[int, int]:
        return self._total_size_helper()

    def _evict_leaf_node(
        self, x: TreeNode, is_evict_mamba: bool
    ) -> Tuple[int, int, TreeNode, TreeNode]:
        assert (
            x.full_lock_ref == 0 and x.mamba_lock_ref == 0
        ), f"evict leaf node invalid with {x.id=} {x.full_lock_ref=} {x.mamba_lock_ref=}"

        assert x.mamba_value is not None, f"leaf node mamba value is not None, {x.id=}"
        # 1. a leaf node, free full tokens and mamba
        self.token_to_kv_pool_allocator.free(x.value)
        full_num_evicted = len(x.value)
        self.req_to_token_pool.mamba_pool.free(x.mamba_value)
        mamba_num_evicted = len(x.mamba_value)

        # 2. get the next node, update the lru lists
        if is_evict_mamba:
            x_next = self.mamba_lru_list.get_prev_no_lock(x)
        else:
            x_next = self.full_lru_list.get_prev_leaf_no_lock(x)
        self.full_lru_list.remove_node(x)
        self.mamba_lru_list.remove_node(x)

        # 3. delete the leaf node
        self._delete_leaf(x)

        # 4. Iteratively delete tombstone leaves to maintain invariant that leaf nodes are not tombstone
        x, leaf_full_num_evicted = self._iteratively_delete_tombstone_leaf(x)
        full_num_evicted += leaf_full_num_evicted
        return full_num_evicted, mamba_num_evicted, x, x_next

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        full_num_evicted = 0
        mamba_num_evicted = 0

        if params.num_tokens > 0:
            full_num_evicted = self.evict_full(params.num_tokens)
        if params.mamba_num > 0:
            mamba_num_evicted = self.evict_mamba(params.mamba_num)

        return EvictResult(
            num_tokens_evicted=full_num_evicted, mamba_num_evicted=mamba_num_evicted
        )

    def evict_mamba(self, mamba_num: int) -> int:
        """Evict mamba states. Returns the number of mamba states evicted."""
        if self.disable or mamba_num <= 0:
            return 0
        # get the least recently used node that is not locked, doesn't have to be a leaf
        x = self.mamba_lru_list.get_lru_no_lock()
        mamba_num_evicted = 0
        # evict lru leaf nodes until mamba_num_tokens is reached
        while mamba_num_evicted < mamba_num and (self.mamba_lru_list.in_list(x)):
            assert x.mamba_value is not None, f"node has no mamba value, {x.id=}"
            assert (
                len(x.mamba_value) == 1
            ), f"node has abnormal mamba length, {x.id=}, {len(x.mamba_value)=}"
            assert x != self.root_node, f"root node is not evictable, {x.id=}"
            assert x.mamba_lock_ref == 0, f"node is in use by mamba kv indices, {x.id=}"

            if len(x.children) > 0:
                # 1. an internal node, free mamba tokens.
                self.req_to_token_pool.mamba_pool.free(x.mamba_value)
                mamba_num_evicted += len(x.mamba_value)

                # 2. get the next node, update the lru lists
                x_next = self.mamba_lru_list.get_prev_no_lock(x)
                self.mamba_lru_list.remove_node(x)

                # 3. tombstone the node
                self._tombstone_internal_node(x)
            else:
                _, mamba_evicted_delta, _, x_next = self._evict_leaf_node(x, True)
                mamba_num_evicted += mamba_evicted_delta

            x = x_next

        return mamba_num_evicted

    def evict_full(self, full_num_tokens: int) -> int:
        """Evict full KV cache. Returns the number of tokens evicted."""
        if self.disable or full_num_tokens <= 0:
            return 0

        full_num_evicted = 0
        # get the least recently used leaf node that is not locked
        x = self.full_lru_list.get_leaf_lru_no_lock()

        while full_num_evicted < full_num_tokens and self.full_lru_list.in_list(x):
            assert (
                x != self.root_node
            ), f"root node should not exist in full lru list, {x.id=}"
            full_num_evicted_delta, _, x, x_next = self._evict_leaf_node(x, False)
            full_num_evicted += full_num_evicted_delta

            # if parent has no more children, it is a leaf. It is possible that this node is lru, so
            # we need to get the first leaf node in the lru list
            if len(x.parent.children) == 0:
                x_next = self.full_lru_list.get_leaf_lru_no_lock()

            x = x_next

        return full_num_evicted

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        """
        Increment the lock reference count for the node.
        It locks the full_lock_ref for nodes between the [last node, root), exclusive.
        It locks the mamba_lock_ref for current node if its mamba_value exists.
        """
        if self.disable:
            return IncLockRefResult()

        # protect mamba value in current node if it exists
        if node.mamba_value is not None:
            mamba_owner_trace(
                "radix_inc_mamba_lock",
                node.mamba_value,
                node_id=getattr(node, "id", None),
                mamba_lock_ref_before=node.mamba_lock_ref,
                full_lock_ref_before=node.full_lock_ref,
            )
            if node.mamba_lock_ref == 0:
                self.mamba_evictable_size_ -= len(node.mamba_value)
                self.mamba_protected_size_ += len(node.mamba_value)
            node.mamba_lock_ref += 1

        while node != self.root_node:
            # lock full from node to root
            assert (
                node.full_lock_ref >= 0
            ), f"inc_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 0:
                full_value_size = self._kv_value_size(node.value)
                self.full_evictable_size_ -= full_value_size
                self.full_protected_size_ += full_value_size
            node.full_lock_ref += 1
            node = node.parent
        return IncLockRefResult()

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        """
        Decrement the lock reference count for the node.
        It unlocks the full_lock_ref for nodes between the [last node, root), exclusive.
        It unlocks the mamba_lock_ref for current node if its mamba_value exists.
        """
        if self.disable:
            return DecLockRefResult()

        if node.mamba_value is not None:
            mamba_owner_trace(
                "radix_dec_mamba_lock",
                node.mamba_value,
                node_id=getattr(node, "id", None),
                mamba_lock_ref_before=node.mamba_lock_ref,
                full_lock_ref_before=node.full_lock_ref,
            )
            assert (
                node.mamba_lock_ref > 0
            ), f"dec_lock_ref on node with {node.mamba_lock_ref=}, {node.id=}"
            if node.mamba_lock_ref == 1:
                self.mamba_evictable_size_ += len(node.mamba_value)
                self.mamba_protected_size_ -= len(node.mamba_value)
            node.mamba_lock_ref -= 1

        while node != self.root_node:
            assert (
                node.full_lock_ref > 0
            ), f"dec_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 1:
                full_value_size = self._kv_value_size(node.value)
                self.full_evictable_size_ += full_value_size
                self.full_protected_size_ -= full_value_size
            node.full_lock_ref -= 1
            node = node.parent

        return DecLockRefResult()

    def sanity_check(self):
        if self.disable:
            return
        self.full_lru_list.sanity_check(self)
        self.mamba_lru_list.sanity_check(self)

    def evictable_size(self) -> Tuple[int, int]:
        # Note: use full_evictable_size() and mamba_evictable_size() instead.
        raise NotImplementedError

    def full_evictable_size(self) -> int:
        return self.full_evictable_size_

    def mamba_evictable_size(self) -> int:
        return self.mamba_evictable_size_

    def protected_size(self) -> Tuple[int, int]:
        # Note: use full_protected_size() and mamba_protected_size() instead.
        raise NotImplementedError

    def full_protected_size(self) -> int:
        # protected size refers to the size of the full cache that is locked
        return self.full_protected_size_

    def mamba_protected_size(self) -> int:
        # protected size refers to the size of the mamba cache that is locked
        return self.mamba_protected_size_

    def all_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values) if len(values) > 0 else torch.tensor([])

    def all_mamba_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs_helper(node: TreeNode):
            if node.mamba_value is not None:
                values.append(node.mamba_value)
            for _, child in node.children.items():
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values) if len(values) > 0 else torch.tensor([])

    def available_and_evictable_str(self) -> str:
        full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable_size = self.full_evictable_size()
        return (
            f"Available full tokens: {full_available_size + full_evictable_size} ({full_available_size=} + {full_evictable_size=})\n"
            f"Full LRU list evictable size: {self.full_lru_list.sanity_check_evictable_size()}\n"
        )

    ##### Internal Helper Functions #####

    def _match_prefix_helper(
        self, key: RadixKey
    ) -> Tuple[List[torch.Tensor], TreeNode, int]:
        """
        Mamba prefix matching helper. It factors in the sliding window size such that
        the matched node is guaranteed to either 1. connected to root without mamba tombstone,
        or 2. the number of matching tokens from the matched node to the last mamba tombstone
        node is greater than or equal to the sliding window size.
        """
        node = self.root_node
        child_key = key.child_key(self.page_size)

        value: List[torch.Tensor] = []
        best_value_len = 0
        best_last_node = node
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            # update best_value_len and best_last_node if needed
            if node.mamba_value is not None:
                best_value_len = len(value)
                best_last_node = node

            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = key.child_key(self.page_size)
        # handle best_value_len and best_last_node, for the case that last node is fully matched
        if node.mamba_value is not None:
            best_value_len = len(value)
            best_last_node = node

        return value, best_last_node, best_value_len

    def _match_pre_processor(self, params: MatchPrefixParams) -> Optional[RadixKey]:
        """Preprocess the key before matching."""
        key = params.key

        if self.disable or len(key) == 0:
            return None

        return key

    def _match_post_processor(
        self,
        params: MatchPrefixParams,
        value: List[torch.Tensor],
        last_node: TreeNode,
        best_value_len: int,
    ) -> MatchResult:
        """Post-process the matched result."""
        cow_mamba = params.cow_mamba
        req = params.req

        # update time for matched nodes, and make nodes closer to root to be least recently used
        # this allows mamba to evict nodes closer to root first
        node_update = last_node
        self.full_lru_list.reset_node_and_parents_mru(node_update, self.root_node)
        self.mamba_lru_list.reset_node_and_parents_mru(node_update, self.root_node)

        # This last_access_time is for sanity check, can be deleted after validation in production
        cur_time = get_last_access_time()
        while node_update:
            node_update.last_access_time = cur_time
            cur_time -= (
                0.00001  # assuming less than 100000 nodes in a branch of the tree
            )
            node_update = node_update.parent

        # Calculate the branching point. It is defined as the last aligned position that
        # does not have a mamba value.
        if len(value) > best_value_len:
            mamba_cache_chunk_size = get_global_server_args().mamba_cache_chunk_size
            mamba_cache_chunk_aligned_seqlen = (
                sum(len(v) for v in value) // mamba_cache_chunk_size
            ) * mamba_cache_chunk_size
            mamba_branching_seqlen = (
                mamba_cache_chunk_aligned_seqlen
                if mamba_cache_chunk_aligned_seqlen > 0
                else None
            )
        else:
            mamba_branching_seqlen = None

        # Copy mamba state to req local space if cow is true
        if cow_mamba and last_node.mamba_value is not None:
            # for reqs without mamba cache
            if req.mamba_pool_idx is None:
                dst_index = self.req_to_token_pool.mamba_pool.alloc(1)
                # try to alloc again, protect last_node from eviction
                if dst_index is None:
                    self.inc_lock_ref(last_node)
                    self.evict(EvictParams(num_tokens=0, mamba_num=1))
                    dst_index = self.req_to_token_pool.mamba_pool.alloc(1)
                    self.dec_lock_ref(last_node)
                    assert dst_index is not None, "Can not alloc mamba cache"
                src_index = last_node.mamba_value
                self.req_to_token_pool.mamba_pool.copy_from(src_index, dst_index)
                req.mamba_pool_idx = dst_index[0]
                mamba_owner_trace(
                    "radix_match_cow_alloc_req_slot",
                    dst_index,
                    src=src_index,
                    req_pool_idx=getattr(req, "req_pool_idx", None),
                    rid=getattr(req, "rid", None),
                    last_node_id=getattr(last_node, "id", None),
                    best_value_len=best_value_len,
                )
            else:
                src_index = last_node.mamba_value
                dst_index = req.mamba_pool_idx.unsqueeze(0)
                self.req_to_token_pool.mamba_pool.copy_from(src_index, dst_index)
                mamba_owner_trace(
                    "radix_match_cow_reuse_req_slot",
                    dst_index,
                    src=src_index,
                    req_pool_idx=getattr(req, "req_pool_idx", None),
                    rid=getattr(req, "rid", None),
                    last_node_id=getattr(last_node, "id", None),
                    best_value_len=best_value_len,
                )

        value = value[:best_value_len]
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
            mamba_branching_seqlen=mamba_branching_seqlen,
        )

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int) -> TreeNode:
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.mamba_value = None  # mamba cache can not be split
        new_node.full_lock_ref = child.full_lock_ref
        new_node.mamba_lock_ref = 0
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len].clone()

        # child time should be later than parent's time for mamba tombstone
        child.last_access_time = get_last_access_time()

        self.full_lru_list.remove_node(child)
        if child.mamba_value is not None:
            self.mamba_lru_list.remove_node(child)
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:].clone()
        new_node.parent.children[key.child_key(self.page_size)] = new_node

        # insert the new node and child into the lru lists, insert
        # parent first so that parent is after child in the lru list
        self.full_lru_list.insert_mru(new_node)
        self.full_lru_list.insert_mru(child)
        if child.mamba_value is not None:
            self.mamba_lru_list.insert_mru(child)
        return new_node

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value,
        mamba_value,
        chunked: bool = False,
        prev_prefix_len: int = 0,
    ) -> Tuple[int, bool]:
        # Update the last access time from root to leaf, so that
        # mamba will tombstone the node closer to root first
        assert mamba_value is not None, "Mamba value should not be None here."
        node.last_access_time = get_last_access_time()
        if node != self.root_node:
            self.full_lru_list.reset_node_mru(node)
            if node.mamba_value is not None:
                self.mamba_lru_list.reset_node_mru(node)
        if len(key) == 0:
            return 0, True

        child_key = key.child_key(self.page_size)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = get_last_access_time()
            self.full_lru_list.reset_node_mru(node)
            if node.mamba_value is not None:
                self.mamba_lru_list.reset_node_mru(node)
            prefix_len = node.key.match(key, page_size=self.page_size)

            if prev_prefix_len < total_prefix_length + prefix_len:
                start = max(0, prev_prefix_len - total_prefix_length)
                self.token_to_kv_pool_allocator.free(value[start:prefix_len])

            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = key.child_key(self.page_size)

        mamba_value_exist = False
        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            new_node.mamba_value = mamba_value
            self.full_lru_list.insert_mru(new_node)
            self.mamba_lru_list.insert_mru(new_node)
            node.children[child_key] = new_node
            self.full_evictable_size_ += self._kv_value_size(value)
            self.mamba_evictable_size_ += len(mamba_value)
        elif node.mamba_value is None:  # add for mamba tombstone
            node.mamba_value = mamba_value
            self.full_lru_list.reset_node_mru(node)
            self.mamba_lru_list.insert_mru(node)
            self.mamba_evictable_size_ += len(mamba_value)
            node.last_access_time = get_last_access_time()
        else:  # mamba value already exists
            mamba_value_exist = True
            self.full_lru_list.reset_node_mru(node)
            self.mamba_lru_list.reset_node_mru(node)
            node.last_access_time = get_last_access_time()

        return total_prefix_length, mamba_value_exist

    def _iteratively_delete_tombstone_leaf(
        self, node: TreeNode
    ) -> Tuple[TreeNode, int]:
        full_num_evicted = 0
        while node.parent.mamba_value is None and len(node.parent.children) == 0:
            # root node is not evictable
            if node.parent == self.root_node:
                break
            # if locked, means node is in use, skip
            if node.parent.full_lock_ref > 0:
                break
            assert (
                node.parent.mamba_lock_ref == 0
            ), f"tombstone mamba_lock_ref should always be 0, {node.parent.full_lock_ref=}, {node.parent.mamba_lock_ref=}, {node.parent.id=}"
            # delete tombstone node evicts full tokens
            self.token_to_kv_pool_allocator.free(node.parent.value)
            full_num_evicted += self._kv_value_size(node.parent.value)
            self.full_lru_list.remove_node(node.parent)
            self._delete_tombstone_leaf(node.parent)
            node = node.parent

        return node, full_num_evicted

    def _delete_leaf(self, node: TreeNode) -> None:
        assert (
            node.mamba_value is not None
        ), f"Invariant violated: leaf node is a tombstone, {node.id=}"
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"
        key = node.key.child_key(self.page_size)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.full_evictable_size_ -= self._kv_value_size(node.value)
        self.mamba_evictable_size_ -= len(node.mamba_value)

    def _tombstone_internal_node(self, node: TreeNode) -> None:
        assert len(node.children) != 0, f"Cannot tombstone a leaf node, {node.id=}"
        self.mamba_evictable_size_ -= len(node.mamba_value)
        node.mamba_value = None

    def _delete_tombstone_leaf(self, node: TreeNode) -> None:
        assert (
            node.mamba_value is None
        ), f"Deleting a unexpected non-tombstone leaf node, {node.id=}"
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"
        key = node.key.child_key(self.page_size)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.full_evictable_size_ -= self._kv_value_size(node.value)

    def _collect_nontombstone_nodes(self) -> List[TreeNode]:
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if cur_node.mamba_value is not None:
                ret_list.append(cur_node)
            stack.extend(cur_node.children.values())

        return ret_list

    def _collect_all_nodes(self) -> List[TreeNode]:
        ret_list = []
        stack = [self.root_node]
        while stack:
            cur_node = stack.pop()
            ret_list.append(cur_node)
            stack.extend(cur_node.children.values())
        return ret_list

    def _print_helper(self, node: TreeNode, indent: int) -> None:
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                f"[{current_node.id}]",
                len(current_node.key),
                f"fr={current_node.full_lock_ref}",
                f"mr={current_node.mamba_lock_ref}",
                f"fll={self.full_lru_list.in_list(current_node)}",
                f"mll={self.mamba_lru_list.in_list(current_node)}",
                f"mv={current_node.mamba_value}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == child.key.child_key(
                    self.page_size
                ), f"{key=}, {child.key.child_key(self.page_size)=}"

    def _total_size_helper(self) -> Tuple[int, int]:
        total_size = 0
        total_mamba_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            if current_node.mamba_value is not None:
                total_mamba_size += len(current_node.mamba_value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size, total_mamba_size
