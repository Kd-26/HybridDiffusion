# Cluster-3 production-efficiency profiling gate

## Scope and current status

This branch starts at the frozen Cluster 1–3 reference
`cf5e14c2f5a4e4700fb66b3183dd021bbe722fe4`. The profiling stage does not add
generation capability or implement an optimization cluster. Model weights,
BF16 behavior, sampling, attention contracts, absolute positions, Region-DAG
rules, GDN replay, training, and TorchTitan behavior are unchanged.

Local preflight was performed on a non-CUDA macOS host. The requested
`/persistent/hybrid-diffusion-cache` mount is absent, so checkpoint hashes,
checkpoint tensor dtypes, A30 identity, and CUDA timings have not been claimed.
The existing Cluster 1–3 CPU regression selection passed with 317 tests and 2
expected skips before profiling changes.

No optimization may be selected from this document until the retained A30
`efficiency_one` record and Nsight Systems trace identify a meaningful overhead.

## Instrumentation

`eval/sglang/srt/dllm/region/profiling.py` defines an opt-in request-scoped
profiler with the 20 required phase names. Host work uses `perf_counter_ns`;
CUDA work uses events on the current stream. Event resolution performs one
terminal stream synchronization. Per-phase diagnostic synchronization is
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
- measured repetitions: at least 10;
- native BF16 and TP=1;
- debug synchronization disabled.

Preparation CUDA events are deferred until the measured forward's single
terminal synchronization. NVTX ranges cover request setup, mask construction,
KV restore/batch assembly, FlashInfer planning, attention, GDN, MLP, GDN
restore, prefix snapshots, and total model forward. The exporter retains exact
output, hidden-state, GDN-state, stable-state, position, invalidation, recovery,
work, and peak-memory checks from frozen Cluster 3.

`eval/scripts/cluster3_efficiency_preflight.py` fails closed unless the exact
five checkpoint files exist, their SHA-256 values match frozen evidence, the
architecture is exactly `Qwen3_5DLLMForConditionalGeneration`, the 2B config
fingerprint matches, and every safetensors tensor entry is BF16. It never
downloads or replaces a checkpoint.

`eval/scripts/cluster3_efficiency_profile_report.py` accepts only a passing
`efficiency_one` record with three warmups, ten samples, BF16, TP=1, and debug
synchronization disabled. GDN restore and snapshot time are subtracted from the
inclusive GDN backend interval before ranking, preventing double counting.

## Initial bottleneck profile

No A30 profile exists in this checkout. The following table is intentionally
unpopulated rather than estimated from CPU time or prior runs.

| Phase | Absolute ms | % warm latency | Difference from full replay | Cold cost | Warm cost |
|---|---:|---:|---:|---:|---:|
| Pending retained A30 run | unavailable | unavailable | unavailable | unavailable | unavailable |

The report generator will rank attention, exclusive GDN work, MLP, prefix
snapshot, GDN restore, mask construction, KV restore/batch assembly, and
unattributed scheduler/model work. Cold handoff build, warm cached suffix, and
full replay are reported separately. The frozen exporter does not provide a
non-overlapping per-phase decomposition of cold prefix construction, so that
field remains explicitly unavailable.

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
export BASE_SHA=cf5e14c2f5a4e4700fb66b3183dd021bbe722fe4

mkdir -p "$RESULT_ROOT/profile-one"
mapfile -t FROZEN_PROVENANCE < <(
  rg -l "$BASE_SHA" "$CACHE_ROOT/results" -g '*.json' | sort
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
  --output-json "$RESULT_ROOT/profile-one/bottlenecks.json" \
  --output-markdown "$RESULT_ROOT/profile-one/bottlenecks.md"
```

Do not set `CUDA_LAUNCH_BLOCKING` or pass `--debug-sync-stages` for this run.
Retain `preflight.json`, `records.jsonl`, `summary.json`, `bottlenecks.json`,
`bottlenecks.md`, and the `.nsys-rep` file together.

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
