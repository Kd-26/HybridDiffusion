# Cluster-3 production-efficiency profiling gate

## Scope and current status

This hierarchical profiling repair starts at
`f2ae5bdeb96a59b805b88a7ae3fe40c566ee58c0`. It does not add
generation capability or implement an optimization cluster. Model weights,
BF16 behavior, sampling, attention contracts, absolute positions, Region-DAG
rules, GDN replay, training, and TorchTitan behavior are unchanged.

Local preflight was performed on a non-CUDA macOS host. The requested
`/persistent/hybrid-diffusion-cache` mount is absent, so checkpoint hashes,
checkpoint tensor dtypes, A30 identity, and CUDA timings have not been claimed.
The A30 rerun below is required before CUDA acceptance; local testing cannot
substitute for it.

No optimization may be selected from this document until the retained A30
`efficiency_one` record and Nsight Systems trace identify a meaningful overhead.

## Pre-repair coverage audit

The first request-scoped A30 `efficiency_one` profile passed correctness but
exposed a hierarchy error in the report. The warm medians were 801.972237 ms
for `active_suffix_forward`, 208.891424 ms for `model_forward_total`, and
94.017552 ms for the nested named model leaves. The report subtracted the
model leaves directly from the outer active-suffix envelope, producing 11.72%
coverage and an 88.28% residual. Those phases belong to two hierarchy levels:
the approximately 592.883276 ms active-minus-model gap and approximately
114.803026 ms model-minus-leaves gap must be decomposed separately.

The execution audit found three distinct route lifecycles. Full replay builds
the identity/specification and dependency plan, prepares an uncached batch,
builds the mask and FlashInfer plan, executes the complete segmented reference,
then verifies and scatters outputs. Cold handoff establishes the canonical
prefix and publishes GDN frontiers before executing the suffix. Warm handoff
assembles the existing KV layout, restores the committed GDN frontier, and
executes only the conservative suffix before verification/output processing.

The old preparation and backend measurements are CUDA-event timers. Backend
GDN time is inclusive of nested restore and snapshot calls; the route total is
also an envelope. The report treated the remaining difference as one additive
component and combined independently summarized medians. Host orchestration
was not measured separately from CUDA work.

`prefix_snapshot` in the warm ranking is genuine runtime work. The conservative
GDN implementation republishes the replay-start and final frontier snapshots
for every GDN layer during a warm suffix so later exact frontiers remain
available. This profiling repair preserves that behavior and records the
explicit contract `expected_zero=false`; it does not optimize or remove the
snapshot.

## Instrumentation

`eval/sglang/srt/dllm/region/profiling.py` defines an opt-in request-scoped
schema-v3 profiler with explicit parent and envelope identity. Host work uses
`perf_counter_ns`; CUDA work uses events on the current stream. Event resolution
performs one terminal stream synchronization. Per-phase diagnostic synchronization is
available only through an explicit `debug_sync` setting and is counted with
its source and classification. Request-exclusive and shared-batch timings are
labeled separately. Missing phases remain explicit zero-call records rather
than inferred zero-duration work.

The controlled production-path exporter adds an `efficiency_one` case:

- stable prefix: 2,048 tokens;
- active suffix: 64 tokens;
- diffusion steps: 4;
- batch size: 1;
- warmups: 3;
- measured repetitions: exactly 10;
- native BF16 and TP=1;
- debug synchronization disabled.

Every measured repetition emits one finalized schema-v3 snapshot for each of
`full_replay`, `cold_handoff_build`, and `warm_cached_suffix`. The outer warm
hierarchy is `active_suffix_forward` containing the non-overlapping
`suffix_prepare_total`, `model_forward_total`, and `suffix_finalize_total`
envelopes. Preparation, decoder, attention, and GDN operations have their own
children. CUDA events are resolved with one terminal profiler synchronization.
The JSON report computes raw residuals per repetition before aggregation and
never adds host and CUDA milliseconds.

`request_total` is intentionally a segmented route-wall aggregate, not a
single request envelope. The expected non-overlapping segment counts are 16
for full replay, 24 for cold handoff construction, and 20 for the warm cached
suffix route. Both the semantics and expected count are recorded in every
snapshot and validated by the report.

The record shape is:

```json
{
  "request_phase_profiles": {
    "full_replay": [{"schema_version": 3, "finalized": true, "hierarchy": {}, "phases": {}}],
    "cold_handoff_build": [{"schema_version": 3, "finalized": true, "hierarchy": {}, "phases": {}}],
    "warm_cached_suffix": [{"schema_version": 3, "finalized": true, "hierarchy": {}, "phases": {}}]
  },
  "warm_snapshot_behavior": {
    "calls": 1,
    "expected_zero": false,
    "pass": true,
    "reason": "conservative replay republishes traversed GDN frontiers"
  }
}
```

Each route list contains exactly ten entries for `efficiency_one`; the example
above abbreviates the repeated entries and phase fields.

`eval/scripts/cluster3_efficiency_preflight.py` fails closed unless the exact
five checkpoint files exist, their SHA-256 values match frozen evidence, the
architecture is exactly `Qwen3_5DLLMForConditionalGeneration`, the 2B config
fingerprint matches, and every safetensors tensor entry is BF16. It never
downloads or replaces a checkpoint.

`eval/scripts/cluster3_efficiency_profile_report.py` accepts only finalized
request-scoped evidence from a passing `efficiency_one` record with three
warmups, ten samples, BF16, TP=1, request-exclusive scope, exactly one profiler
resolution synchronization, and debug synchronization disabled. Separate
outer-suffix and model-forward CUDA coverage gates must both reach 90% in
every repetition. The report exports the suffix, decoder, attention, and GDN
sub-hierarchies, raw phase inventory, and unclamped per-repetition residuals.
Host orchestration remains a separate timing domain.

## Initial bottleneck profile

No schema-v3 A30 profile exists in this checkout. The prior A30 evidence above
identifies coverage gaps but cannot satisfy the repaired gates. The following
table is intentionally unpopulated rather than estimated from CPU time.

| Phase | Absolute ms | % warm latency | Difference from full replay | Cold cost | Warm cost |
|---|---:|---:|---:|---:|---:|
| Pending retained A30 run | unavailable | unavailable | unavailable | unavailable | unavailable |

The report generator produces separate host/orchestration and GPU/model
critical-path rankings, route totals, phase call counts, cache/recovery/snapshot
counters, synchronization counts, per-repetition residual distributions, and a
pair of CUDA coverage gates. Missing evidence fails closed.

## A30 execution

Run from a clean checkout of the profiling commit. The provenance discovery
passes every JSON document that records the frozen SHA; preflight rejects
missing or conflicting checkpoint hashes.

```bash
set -euo pipefail

export REPO="$PWD"
export MODEL_PATH=/persistent/hybrid-diffusion-cache/models/HybridDiffusion-2B
export CACHE_ROOT=/persistent/hybrid-diffusion-cache
export RESULT_ROOT=/persistent/hybrid-diffusion-cache/results/cluster3-production-efficiency
export PY=/persistent/hybrid-diffusion-cache/venvs/hybrid-diffusion-eval/bin/python
export PARENT_SHA=f2ae5bdeb96a59b805b88a7ae3fe40c566ee58c0
export FROZEN_SHA=cf5e14c2f5a4e4700fb66b3183dd021bbe722fe4

test "$(git branch --show-current)" = cluster3-hierarchical-profiling-repair
test "$(git rev-parse HEAD^)" = "$PARENT_SHA"

mkdir -p "$RESULT_ROOT/profile-one"
mapfile -t FROZEN_PROVENANCE < <(
  rg -l "$FROZEN_SHA" "$CACHE_ROOT/results" -g '*.json' | sort
)
test "${#FROZEN_PROVENANCE[@]}" -gt 0
PROVENANCE_ARGS=()
for evidence in "${FROZEN_PROVENANCE[@]}"; do
  PROVENANCE_ARGS+=(--frozen-provenance "$evidence")
done

"$PY" "$REPO/eval/scripts/cluster3_efficiency_preflight.py" \
  --repo "$REPO" \
  --model-path "$MODEL_PATH" \
  "${PROVENANCE_ARGS[@]}" \
  --require-a30 \
  --output "$RESULT_ROOT/profile-one/preflight.json"

nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --output="$RESULT_ROOT/profile-one/efficiency-one" \
  "$PY" "$REPO/eval/scripts/cluster3_region_dag_validation.py" \
    --model-path "$MODEL_PATH" \
    --profile efficiency_one \
    --dtype bfloat16 \
    --tp-size 1 \
    --device 0 \
    --max-total-tokens 4096 \
    --timed-repetitions 10 \
    --output-jsonl "$RESULT_ROOT/profile-one/records.jsonl" \
    --summary-json "$RESULT_ROOT/profile-one/summary.json"

"$PY" "$REPO/eval/scripts/cluster3_efficiency_profile_report.py" \
  --record "$RESULT_ROOT/profile-one/records.jsonl" \
  --output-json "$RESULT_ROOT/profile-one/detailed-profiling.json" \
  --output-markdown "$RESULT_ROOT/profile-one/detailed-profiling.md"
```

Do not set `CUDA_LAUNCH_BLOCKING` or pass `--debug-sync-stages` for this run.
Retain `preflight.json`, `records.jsonl`, `summary.json`,
`detailed-profiling.json`, `detailed-profiling.md`, logs, and the optional
`.nsys-rep` file together.

## Correctness invariants

- Cached and full recomputation keep the existing `< 1e-2` unrounded BF16
  logits, hidden-state, and GDN-state thresholds.
- Top-1 IDs, original positions, stable hashes, and invalidation closure must
  match exactly.
- Valid cached execution must report zero fallback/recovery replay.
- The custom paged attention contract remains selected.
- Profiling hooks are removed on normal and exceptional exits.
- Profiling does not change model weights, training code, or TorchTitan code.

## Unsupported claims

- No CUDA or A30 acceptance has been run in this local environment.
- No end-to-end speedup is claimed.
- No dominant overhead has yet been identified.
- No optimization cluster is justified or implemented by this commit.
- Existing historical results are not substituted for the required new,
  matched A30 profile.
