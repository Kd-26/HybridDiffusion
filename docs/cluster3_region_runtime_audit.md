# Cluster-3 region runtime audit

Base revision: `aa39883bcc18b342f3b2244ed475bdb2d18a1624`

This audit records the pre-implementation serving and training paths for the
`region_dag_conservative_gdn_v1` contract. Cluster 3 must extend these paths;
it must not create a second diffusion runtime or weaken the exact-prefix
contract.

## Request-to-model path

The production serving path is:

1. `ReqDllmMixin.init_diffusion_llm()` optionally attaches the immutable
   Cluster-1 `HybridExecutionSpec` to a persistent scheduler `Req`.
2. `SchedulerDllmMixin` validates and publishes scheduler-owned exact-handoff
   keys. An unsealed exact request first performs causal prompt prefill.
3. `ScheduleBatch.prepare_for_extend()` or
   `prepare_for_dllm_decode()` allocates a real request-pool slot and physical
   KV slots, writes the request page table, and constructs flattened input
   tokens. The current decode implementation assumes one contiguous extension
   per request.
4. `ScheduleBatch.get_model_worker_batch()` copies request metadata into
   `ModelWorkerBatch`. `ForwardBatch.init_new()` constructs device tensors,
   absolute rotary positions, attention metadata, and cache references.
5. `FlashInferAttnBackend.init_forward_metadata()` builds the request KV page
   list, query offsets, and the selected paged-mask plan. Qwen3.5 full-attention
   layers project exactly the flattened input rows and call `RadixAttention`.
6. Qwen3.5 linear-attention layers call `GDNDllmBackend`, which reads and writes
   the Mamba/GDN slot associated with the request-pool slot.
7. `HybridDiffusionSelfSpec` verifies tokens, commits accepted GDN state and KV
   boundaries, and returns model-authoritative publications to the scheduler.
   The scheduler finalizes or advances the persistent `Req` only after checking
   those publications.

The long-lived objects are the scheduler `Req`, its radix-cache ownership, the
request/Mamba slot mapping while allocated, the algorithm's request-indexed
handoff maps, and entries in `RegionStateCache`. `ModelWorkerBatch` is the
cross-worker batch payload. `ForwardBatch` and FlashInfer `PrefillMetadata` are
per-forward execution objects.

## Existing Cluster-1 and Cluster-2 contracts

`HybridExecutionSpec.prefix_diffusion()` is intentionally restricted to two
regions: stable `[0, ar_boundary)` and active
`[ar_boundary, sequence_length)`. It requires one parent edge from the suffix
to the prefix and the contract ID
`causal_prefix_diffusion_suffix_v1`. Its validator rejects multiple spans,
gaps, overlap, unknown routes, and unknown attention contracts. Cluster 3 must
use an isolated immutable contract and leave this validation unchanged.

The Cluster-2 production path already sends only active suffix rows through
Qwen full-attention projections, MLPs, and GDN layers. Stable full-attention KV
is read through the request page table, while exact GDN state is restored at
the stable-prefix boundary. The Cluster-2 validator observes the existing
modules; it is not a second active-only implementation. In particular,
Qwen3.5 `self_attention` is a bound method, so instrumentation hooks the real
`qkv_proj` module, verified `linear_attn` modules, and verified `mlp` modules.

Cluster-1 sealing and restore are model-authoritative. A restore before an
initial publication is a lifecycle error. `HybridDiffusionSelfSpec` snapshots
complete convolution and recurrent GDN state together with an exact KV page
reference, and the scheduler does not mark the prefix sealed before that
publication. This lifecycle must remain byte-for-byte equivalent for requests
without a Region-DAG contract.

## Positions, page tables, and mask selection

For legacy pure diffusion decode, `ForwardBatch` constructs contiguous absolute
positions from each request's `dllm_block_offset`. Mixed causal prefill uses
`prefix_len`; normal extend uses the standard prefix/extend position builder.
CUDA positions are currently `int32` (HIP/NPU uses `int64`). They are absolute
positions and are never relative to a gathered active tensor.

`Req.prefix_indices` is the canonical detached, contiguous `torch.int64`
descriptor consumed through an int64 pointer by cache-index writing code.
`req_to_token_pool.req_to_token` is intentionally an internal `torch.int32`
page table. Live page-table slices must be explicitly copied to int64 before
being used as a `Req.prefix_indices` or `KVPrefixReference.locations` value.
Allocator-produced `ForwardBatch.out_cache_loc` is `torch.int64`; writes cast
those physical locations to the page table's internal dtype. Cluster 3 must
not change either side of this contract.

FlashInfer's paged prefill wrapper can consume a flattened boolean custom mask
whose rows are described by `qo_indptr` and whose columns are the canonical KV
page-table sequence described by `kv_indptr`. The current serving builder,
however, constructs only `query_len = seq_len - prefix_len` suffix rows and
assumes the custom block mask describes a contiguous active suffix. Backend
selection may choose `native_structured`, `full_paged`, or `custom_paged`;
production custom fallback is currently guarded by an environment flag.
Neither the native fixed mask nor the all-true full-paged fast path can encode
an arbitrary Region-DAG mask. Region-DAG requests therefore require an
explicit `custom_paged` route and exact per-request query/KV counts.

## Request-pool and recurrent-state lifecycle

Fresh `Req` objects start with `req_pool_idx=None` and an empty int64
`prefix_indices`. `alloc_for_extend()` allocates the real request-pool slot
before it writes positive-length page-table state. Inline-prefill absorption
also allocates a real slot before initialization. A positive-length region
lookup without that slot is invalid; an empty-prefix request must not fabricate
slot zero or any other synthetic slot.

The request-pool slot maps to a Mamba/GDN slot. `RegionStateKey` includes the
request identity, request slot and generation, region/version, hashes,
model/adapter identity and revisions, attention contract, and ordered parent
versions. `RegionStateCache` reports structured misses for every incompatible
field. `GDNDllmBackend.snapshot_region_state()` clones every convolution and
recurrent tensor before publication. Restore verifies slot generation, Mamba
mapping, and exact KV ownership before copying state into the live slot.

GDN execution is recurrent in flattened textual order. The existing kernels
accept contiguous per-request sequences and a single initial state; they do not
support sparse execution across textual gaps or independent sibling states.
Consequently, after the earliest logically invalidated position, Cluster 3
must replay every later textual position and recompute every affected
token-layer value. A later stable region can remain outside the graph
invalidation closure while still being in this conservative recurrent replay.

## Missing production primitive

The current scheduler cannot safely represent two non-contiguous active spans
as one request. It derives input rows from
`fill_ids[len(prefix_indices):]`, allocates appended KV positions, writes them
at `prefix_len + offset`, and derives contiguous positions from one block
offset. Treating distant spans as a prefix and suffix would write incorrect KV
locations and renumber positions.

Cluster 3 therefore requires an explicit opt-in primitive carrying, for every
request, sorted absolute query/recompute positions plus their input tokens and
physical output locations. The batch builder must scatter allocated locations
to those exact page-table positions, preserve the full canonical sequence
length for KV planning, and pass exact query offsets and a flattened boolean
mask to FlashInfer. If any of those checks cannot be satisfied, the Region-DAG
request must fail closed; it must never enter the legacy contiguous route.

## Training-only versus serving code

`torchtitan/models/qwen3_5/model/dllm_model.py` is the training/reference model.
Its current mask and forward path implement the doubled `[x0; xt]` layout and
block-diffusion training losses. Cluster 3 may add pure mask-oracle helpers in
that file, but must not alter its training forward, input construction,
sampling, or loss behavior.

Production serving code lives under `eval/sglang/srt`: request/scheduler
mixins, `ScheduleBatch`, `ModelWorkerBatch`, `ForwardBatch`, FlashInfer, the
Qwen3.5 dLLM model adapter, `GDNDllmBackend`, and
`HybridDiffusionSelfSpec`. Region-DAG metadata must be opt-in throughout these
objects. Requests without it must execute the existing Cluster-1/2 control
flow without new branches in their hot path.

## Preserved invariants

- Cluster-1 prefix/suffix validation and sealing remain unchanged.
- Cluster-2 active-only behavior and validator evidence remain unchanged.
- Prefix and external KV-location descriptors remain contiguous int64 values;
  the internal page table remains int32.
- `out_cache_loc` remains allocator-compatible int64.
- Positive state lookup requires a real request-pool slot; empty prefixes never
  fabricate one.
- Arbitrary Region-DAG layouts always select `custom_paged`; native fixed masks
  are forbidden unless a future contract proves exact equivalence.
- GDN replay begins at the earliest logical invalidation and proceeds without
  gaps to sequence end. Logical invalidation and conservative replay are
  recorded separately.
- Training-only `[x0; xt]` construction is not reused as a production serving
  shortcut.
- Instrumentation hooks only actual `torch.nn.Module` objects.

## Post-implementation overlap conclusion

The implementation uses the audited production objects rather than adding a
parallel serving engine. `Req` owns the opt-in immutable contract and runtime
plan; `ScheduleBatch` gathers the conservative replay suffix from the existing
real page table; `ForwardBatch` preserves absolute positions and builds the
contract mask; the existing FlashInfer backend plans `custom_paged`; the
existing Qwen layers execute the rows; and `GDNDllmBackend` restores and
publishes layer-local frontier snapshots. `HybridDiffusionSelfSpec` remains the
model-authoritative publisher consumed by the scheduler.

The new Region-DAG branch is mutually exclusive with Cluster-1 exact-prefix
handoff and is isolated from legacy dLLM batches. Non-Region requests retain
their former lifecycle. The allocator's int32 page table is unchanged;
external selected locations remain detached contiguous int64. No training
forward, loss, checkpoint, precision policy, or dependency was changed.

The controlled exporter reuses the Cluster-1 real-model loader and tensor
hashing and the Cluster-2 verified Qwen row hooks. This is intentional
instrumentation reuse, not runtime duplication. It adds validator-scoped CUDA
event probes to concrete full-attention and GDN backend calls and removes them
on normal and exceptional exits. A missing probe, frontier, request slot,
custom mask, comparison tensor, or timing sample is a failure rather than an
inferred zero.

## Canonical frontier-construction repair

The A30 provenance diagnostic established that cache cloning and restoration
were exact (`S == R == D`), but production initialization was not canonical.
It published boundary snapshots while processing an N-row monolithic prefill.
The strict segmented oracle built the same boundary with a b-row prefix
execution, so BF16 projection/kernel tiling produced a different live GDN
frontier before any cache operation. This was production-visible, not only a
validator labeling error.

Positive-frontier cached requests now use two scheduler-owned prefill rounds.
The first executes exactly `[0:b)` at the original absolute positions using a
projected `region_dag_conservative_gdn_v1` contract and publishes only
frontiers at or before `b`. The scheduler verifies that publication and the KV
boundary before marking the frontier established. The second round attaches
`[b:N)`, requires restoration of the committed boundary, and publishes the
complete frontier set before decode initialization. Entirely active requests
with `b=0` keep the existing full-replay lifecycle.

The projection preserves declared region statuses, parent ordering, versions,
token/position hashes, BF16 dtype, and the source attention contract. It does
not fork recurrent state or merge parents: GDN still replays the conservative
ordered suffix. Batch-specific execution specs, query positions, frontier
maps, restore flags, and reference flags are carried through
`ModelWorkerBatch`; they are never reconstructed from the full request during
the prefix round.
