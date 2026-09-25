# Cluster-3 cache-system comparison: semantic feasibility audit

This audit was performed at base revision
`c57970dca909ac8fa77260ea75da1580797c22ad` before implementation of the
comparison harness.  It distinguishes execution mechanisms, not labels.  A
method is available only when the current runtime can preserve the accepted
attention, position, hidden-state, and recurrent-state contracts exactly.

## Findings

### `controlled_full_replay`: supported

The existing production benchmark already implements the same-runtime oracle.
For every diffusion step it clears request/cache state, constructs a fresh
canonical frontier, executes the complete prefix, discards only the temporary
published snapshots, and continues the suffix from the live recurrent state
without restoration (`eval/scripts/cluster3_production_latency_benchmark.py`,
lines 451-507).  Its work gate requires prefix plus suffix rows to equal the
full sequence (lines 477-505), and its route loop repeats that construction for
every diffusion step (lines 703-741).  This is distinct from the original
public FLARE runtime and is labelled only as the controlled same-runtime
causal oracle.

### `upstream_sglang_radix`: unsupported

Availability reason:
`native_radix_lacks_exact_hybrid_recurrent_state`.

The bundled runtime does not have a single generic “KV-only radix” behavior for
this hybrid model.  Scheduler cache selection chooses `MambaRadixCache` for a
hybrid SSM model (`eval/sglang/srt/managers/scheduler.py`, lines 829-888).
`MambaRadixCache.supports_mamba()` is true and cache insertion publishes both
page-aligned attention-KV indices and a Mamba value
(`eval/sglang/srt/mem_cache/mamba_radix_cache.py`, lines 430-466 and 554-612).
On a hit, `match_prefix` copy-on-writes that Mamba state into request-local
storage (`mamba_radix_cache.py`, lines 1364-1454).  Thus native bundled hybrid
radix is not a KV-only baseline.

It is nevertheless not an exact replacement for the Cluster-3 contract.
Cluster-3 requires a canonical recurrent frontier with complete identity and
explicit conservative replay.  Before that frontier is established, request
initialization deliberately disables ordinary tree matching because a token
match has no equivalent GDN provenance
(`eval/sglang/srt/managers/schedule_batch.py`, lines 974-988).  Region-DAG
replay requires an exact frontier key on every GDN layer and fails closed on a
miss or identity mismatch
(`eval/sglang/srt/layers/attention/linear/gdn_dllm_backend.py`, lines 561-645).
The ordinary dLLM/native-radix path also does not express the explicit
fragmented Region-DAG attention contract.  Disabling the Cluster 1-3 identity,
restore metadata, and active/replay planning therefore removes information
needed to reproduce all three requested cases exactly.  Quietly adding the
Region-DAG GDN mechanism would turn this baseline into the proposed method.

### `safe_kv_only_diffusion`: supported, conservatively

There is no GDN-only layer entry point.  The exact conservative construction
must therefore execute the canonical prefix through the full model on fresh
physical pages to reconstruct every layer's live GDN state.  Only after that
forward may the benchmark replace the request's prefix page-table entries with
the separately retained authoritative attention-KV locations.  The fresh
reconstruction pages must be freed before attachment, and the retained pages
must be hashed before and after the timed route.  The suffix then uses
`restore=False`, so it consumes the live reconstructed GDN state and cannot
consume a Region-DAG snapshot.

The necessary low-level primitives already exist: the request-to-token table
is the authoritative mapping (`eval/sglang/srt/mem_cache/memory_pool.py`, lines
146-204), the allocator can free only the fresh reconstruction locations
(`memory_pool.py`, lines 363-403), and the controlled runtime materializes a
validated contiguous int64 view of a request's physical prefix locations
(`eval/scripts/cluster1_exact_handoff_trace.py`, lines 628-654).  This route is
expected to save little or no latency: it retains full prefix model work and
uses KV reuse only for the subsequent suffix attention lookup.  It is distinct
from `cold_handoff_build`, which publishes and later restores GDN snapshots.

The safe route must fail rather than run if the retained and reconstructed
locations overlap, if the prefix tensor is not contiguous int64 with the exact
boundary length, if any physical location is invalid, if a GDN restore occurs,
or if retained KV bytes change.

### `kv_gdn_handoff_full_rows`: unsupported

Availability reason:
`state_handoff_and_active_rows_structurally_coupled`.

The GDN backend restores a frontier immediately before executing the ordered
replay suffix and rejects nonzero replay without that proven restore
(`eval/sglang/srt/layers/attention/linear/gdn_dllm_backend.py`, lines 695-760).
Its input rows must be exactly the contiguous suffix beginning at the restored
boundary (lines 703-715).  Submitting the stable prefix rows after restoration
would apply their convolutional/recurrent transition a second time.  Omitting
that transition only for GDN while executing full attention/MLP rows would
require cached per-layer stable hidden states or a new layer-selective model
contract.  Neither exists.  Changing masks, hidden states, or recurrence would
change semantics, so a distinct exact “handoff plus full rows” route cannot be
defined in this branch.

### `complete_region_state_execution`: supported

The existing validated warm route first builds the canonical frontier outside
the measured region, then prepares every timed suffix with `restore=True`
(`eval/scripts/cluster3_production_latency_benchmark.py`, lines 652-658 and
742-774).  After the single terminal synchronization it proves that each real
request-pool slot has a contiguous int64 prefix of the correct length, that
the entries equal the authoritative page table, and that every physical
location is valid (lines 545-620).  The GDN backend restores exact layer-local
convolution and recurrent snapshots (lines 561-623 of
`gdn_dllm_backend.py`) and then replays the conservative ordered suffix.
`RegionDAGRuntimePlan` makes logical invalidation distinct from recurrent
replay and requires attention/model rows to equal the complete conservative
suffix (`eval/sglang/srt/dllm/region/runtime.py`, lines 105-162 and 212-259).

This route is the existing `warm_cached_suffix`; it is not relabelled as the
handoff-only ablation.

## Distinctness and setup accounting

Three exact, independently executable routes exist:

1. controlled full replay: no cached KV or GDN state is consumed;
2. safe KV-only diffusion: retained physical KV is consumed, while GDN is
   reconstructed by full prefix execution on fresh pages and never restored;
3. complete region-state execution: retained KV and exact GDN snapshots are
   consumed and only the conservative replay suffix is submitted.

The native-radix and full-row-handoff requests are unsupported, so the
five-way publication matrix is incomplete by construction.

Cold-only work consists of model/checkpoint loading, deterministic token/case
construction, initial frontier execution, snapshot publication, and retained
KV establishment.  For `complete_region_state_execution`, the frontier build
is explicitly outside the measured route (`cluster3_production_latency_benchmark.py`,
lines 421-438 and 652-658).  For safe KV-only execution, only the authoritative
retained-KV build is cold-only; the fresh full-prefix reconstruction is part of
every timed diffusion step.

Every warm complete-method step performs runtime-plan/dependency validation,
canonical page-table materialization, request metadata and frontier-key
construction, schedule/worker/forward-batch preparation, FlashInfer planning,
GDN snapshot lookup/restore, conservative suffix model execution, and state
publication.  The preparation sequence is visible in
`eval/scripts/cluster3_region_dag_validation.py`, lines 1632-1707; the
per-layer restore, forward, and snapshot sequence is in
`gdn_dllm_backend.py`, lines 695-864.  Output transfer, hashing, equality
checks, physical-location inspection, and stable-state hashing belong after
the timed route.

## Runtime impact decision

No serving-runtime modification is required.  The comparison is implemented
entirely in evaluation code using existing request-pool, allocator,
page-table, Region-DAG, and profiling primitives.  Ordinary serving and the
accepted Cluster 1-3 paths remain unchanged.
