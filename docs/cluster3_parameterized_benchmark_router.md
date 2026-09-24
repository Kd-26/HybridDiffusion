# Cluster-3 parameterized benchmark and conservative router

This branch extends the validated Cluster 0–3 evaluation harness. It does not
change model weights, attention masks, GDN semantics, cache invalidation,
generation, or default serving behavior. The router is used only by evaluation
scripts.

## Case manifest

`--case-manifest` accepts JSON with `schema_version: 1` and a nonempty `cases`
array. Every case has a unique `case_id`, positive
`total_tokens_per_request`, nonempty half-open `active_spans`, positive
`diffusion_steps`, and `batch_size`. The current production harness supports
only `batch_size: 1`; values 2 and 4 fail before checkpoint or model loading.
Although the Region-DAG position/mask metadata can describe several requests,
the controlled production runtime currently creates one request, one real pool
slot, and one independent GDN state. Serial execution is never labeled batched.

A prefix/suffix case may use `prefix_tokens` as shorthand:

```json
{
  "schema_version": 1,
  "cases": [{
    "case_id": "p2048-a64-s4-b1",
    "total_tokens_per_request": 2112,
    "prefix_tokens": 2048,
    "active_spans": [[2048, 2112]],
    "diffusion_steps": 4,
    "batch_size": 1
  }]
}
```

Internal or disjoint spans require explicit `regions`. Regions must partition
`[0, total_tokens_per_request)` without gaps or overlap. `active_spans` must
exactly equal the non-stable regions. Every parent must exist and the graph must
be acyclic. Normalization constructs the repository's existing
`RegionDAGExecutionSpec`; there is no second execution contract.

The checked-in bounded manifest is
[`cluster3_parameterized_smoke.json`](../eval/manifests/cluster3_parameterized_smoke.json).
The larger starting point is
[`cluster3_parameterized_final_sweep.example.json`](../eval/manifests/cluster3_parameterized_final_sweep.example.json).

## Route and timing definitions

- `full_replay` reconstructs the canonical prefix and live suffix for every
  diffusion step with `restore=False`. It makes no cache-hit claim.
- `cold_handoff_build` constructs the canonical stable frontier once inside the
  measured route, then restores and executes the conservative suffix. Its route
  latency includes build cost.
- `warm_cached_suffix` constructs the compatible frontier before timing. The
  headline interval contains verified restores and conservative suffix work.
  The preconditioning CUDA cost is recorded separately as
  `cache_build_cuda_ms`.

Every retained route has an uninstrumented headline and a minimally profiled
measurement. CUDA events and monotonic host timing are retained. Exactly one
terminal synchronization occurs in each timed route. Output extraction,
hashing, equality checks, page-table checks, host copies, and JSON serialization
occur after timing. The default is three warmups; `--warmups` and
`--timed-repetitions` are configurable for engineering and final runs.

For disjoint active regions, the validated conservative contract replays every
row from the earliest invalidated recurrent boundary. The benchmark records
logical active attention work, actual attention query token-layer work, and
actual GDN replay token-layer work separately. It never reports disjoint-only
GDN work when a contiguous suffix was executed.

Physical KV reuse is evidence, not a route label. For every restored step the
post-timing verifier requires a real request-pool slot, a contiguous `int64`
prefix index tensor, authoritative canonical locations on the same device,
valid allocator bounds, and exact physical page-table equality. Reported
`cache_hits` equals `reusable_prefix_positions × verified_restore_steps`.
The legacy instrumentation counter and its unavailability reason remain
separate.

The omitted `--case-manifest` path retains the validated 2,048-prefix,
64-active, four-step, batch-one regression. Hardware latency is not hardcoded.

## Router policy

`latency_router.py` uses one regularized linear model per route. The default
features are total, stable, and active tokens; stable fraction; diffusion
steps; active-region count; batch size; and four token/step/batch interaction
terms. Each route stores its training means, scales, intercept, coefficients,
ridge strength, and training error in versioned JSON.

`cluster3_fit_latency_router.py` groups all repetitions by normalized
`case_id`, then deterministically splits case IDs into train, validation, and
test sets. No repetition of a case can cross a split. Only the train split is
used for normalization, support ranges, and coefficient fitting. The policy
records all split IDs, source JSONL SHA-256 hashes, feature schema, coefficients,
normalization, seed, fit timestamp, and benchmark revisions.

Cached routes are considered only when the contract, region versions, position
hash, and parent versions are compatible. Warm additionally requires a verified
warm cache. Full replay is always available. The default safety margin is 2%:
a cached route is selected only when it is predicted at least 2% faster than
full replay. Missing/invalid policy files, unsupported schemas, missing or
nonfinite features, nonpositive predictions, incompatible identities, tied
cached predictions, and values outside every calibrated feature range all fail
closed to `full_replay` with a machine-readable reason. The policy does not
read logits or answers.

The checked-in
[`cluster3_latency_router_policy.example.json`](../eval/artifacts/cluster3_latency_router_policy.example.json)
demonstrates serialization only; it is not fitted evidence.

## Evaluation and acceptance

`cluster3_evaluate_latency_router.py` reports exact oracle agreement, aggregate,
median, P95 and worst regret, cases within 2% of oracle, decision overhead,
fallbacks, out-of-domain cases, and aggregate latency for adaptive, always-full,
always-cold-when-valid, always-warm-when-valid, and oracle policies. It exports
the per-case table as paper-ready CSV. It preserves artifacts and returns
nonzero when a gate fails.

Primary gates are 90% exact oracle agreement, aggregate latency within 2% of
oracle, decision overhead below 1% of selected latency (preferably below 0.1
ms), and adaptive latency below fixed full replay. A failed pilot or held-out
gate is reported, never hidden by changing the split.

With `--adaptive-router-policy` and `--adaptive-output-jsonl`, the production
benchmark measures all candidates for a separate oracle and then actually
executes the selected route again. The adaptive record includes the decision,
prediction, selected and actual routes, actual latency, separately measured
oracle, regret, output hash, and fallback/recovery counts. Candidate and
adaptive JSONL are separate so adaptive rows cannot leak into fitting.

## Exact A30 engineering smoke

Use the accepted correctness summary/preflight from the validated environment.
This is bounded engineering evidence, not a paper result.

```bash
set -euo pipefail
export MODEL_PATH=/persistent/hybrid-diffusion-cache/models/HybridDiffusion-2B
export RESULT_DIR=/persistent/hybrid-diffusion-cache/results/cluster3-router-smoke-$(date -u +%Y%m%dT%H%M%SZ)
export CUDA_VISIBLE_DEVICES=0
test ! -e "$RESULT_DIR"
mkdir -p "$RESULT_DIR"

python eval/scripts/cluster3_region_dag_validation.py \
  --model-path "$MODEL_PATH" --profile production_efficiency \
  --case-manifest eval/manifests/cluster3_parameterized_smoke.json \
  --dtype bfloat16 --tp-size 1 --device 0 --max-total-tokens 4096 \
  --warmups 1 --timed-repetitions 2 \
  --routes full_replay cold_handoff_build warm_cached_suffix \
  --correctness-artifact "$CORRECTNESS_ARTIFACT" \
  --preflight-json "$CORRECTNESS_PREFLIGHT" \
  --output-jsonl "$RESULT_DIR/candidates.jsonl" \
  --summary-json "$RESULT_DIR/summary.json"

python eval/scripts/cluster3_fit_latency_router.py \
  --input-jsonl "$RESULT_DIR/candidates.jsonl" \
  --output-policy "$RESULT_DIR/policy.json" --seed 20260924 \
  --ridge-alpha 1.0 --safety-margin 0.02

python eval/scripts/cluster3_evaluate_latency_router.py \
  --policy-json "$RESULT_DIR/policy.json" \
  --input-jsonl "$RESULT_DIR/candidates.jsonl" \
  --summary-json "$RESULT_DIR/router-evaluation.json" \
  --output-csv "$RESULT_DIR/router-evaluation.csv"
```

## Exact final A30 evaluation commands

Run the candidate sweep first, fit once, evaluate the recorded held-out IDs,
then run adaptive execution. Use a new result directory and the validated
checkpoint/correctness paths.

```bash
set -euo pipefail
export MODEL_PATH=/persistent/hybrid-diffusion-cache/models/HybridDiffusion-2B
export RESULT_DIR=/persistent/hybrid-diffusion-cache/results/cluster3-router-final-$(date -u +%Y%m%dT%H%M%SZ)
export CUDA_VISIBLE_DEVICES=0
test ! -e "$RESULT_DIR"
mkdir -p "$RESULT_DIR"

COMMON=(--model-path "$MODEL_PATH" --profile production_efficiency \
  --case-manifest eval/manifests/cluster3_parameterized_final_sweep.example.json \
  --dtype bfloat16 --tp-size 1 --device 0 --max-total-tokens 4096 \
  --warmups 3 --timed-repetitions 10 \
  --routes full_replay cold_handoff_build warm_cached_suffix \
  --correctness-artifact "$CORRECTNESS_ARTIFACT" \
  --preflight-json "$CORRECTNESS_PREFLIGHT")

python eval/scripts/cluster3_region_dag_validation.py "${COMMON[@]}" \
  --output-jsonl "$RESULT_DIR/candidates.jsonl" \
  --summary-json "$RESULT_DIR/candidates-summary.json"
python eval/scripts/cluster3_fit_latency_router.py \
  --input-jsonl "$RESULT_DIR/candidates.jsonl" \
  --output-policy "$RESULT_DIR/policy.json" --seed 20260924 \
  --ridge-alpha 1.0 --safety-margin 0.02
python eval/scripts/cluster3_evaluate_latency_router.py \
  --policy-json "$RESULT_DIR/policy.json" \
  --input-jsonl "$RESULT_DIR/candidates.jsonl" \
  --summary-json "$RESULT_DIR/router-evaluation.json" \
  --output-csv "$RESULT_DIR/router-evaluation.csv"
python eval/scripts/cluster3_region_dag_validation.py "${COMMON[@]}" \
  --adaptive-router-policy "$RESULT_DIR/policy.json" \
  --adaptive-output-jsonl "$RESULT_DIR/adaptive-executed.jsonl" \
  --output-jsonl "$RESULT_DIR/adaptive-candidates.jsonl" \
  --summary-json "$RESULT_DIR/adaptive-summary.json"
```

The companion
[`cluster3_parameterized_router_a30.ipynb`](../eval/notebooks/cluster3_parameterized_router_a30.ipynb)
performs the bounded workflow and creates a downloadable ZIP without
overwriting prior evidence.
