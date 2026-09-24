# Cluster-3 production latency benchmark

The parameterized manifest and conservative routing extension is documented in
[`cluster3_parameterized_benchmark_router.md`](cluster3_parameterized_benchmark_router.md).
Omitting `--case-manifest` retains the fixed regression described below.

`production_efficiency` is a production-only latency path. It preserves the
`efficiency_one` controlled correctness path and does not implement a runtime
optimization. The fixed workload is one HybridDiffusion-2B request with a
2,048-token stable prefix, a 64-token active suffix, four diffusion steps,
batch size one, BF16, and TP=1.

The production latency reference is the accepted Cluster-3 canonical segmented
no-cache execution (`reference_execution=canonical_segmented_no_cache`). For
every diffusion step, `full_replay` clears runtime state, builds and forwards
the canonical `[0:b)` prefix, removes only the temporary committed Region-DAG
layer snapshots, attaches `[b:N)` with `restore=False`, and forwards the live
suffix. The prefix is rebuilt on all four steps. This is Path B from the
accepted B-versus-C handoff oracle. The monolithic 2,112-row BF16 path remains a
numerical diagnostic only; it is not the production latency or cache-handoff
reference because projection/kernel tiling can change with row shape.

The three measured route semantics are:

- `full_replay`: canonical prefix plus live suffix, rebuilt inside timing for
  each of four steps.
- `cold_handoff_build`: canonical prefix built once inside timing, followed by
  four suffixes attached with `restore=True`.
- `warm_cached_suffix`: canonical prefix built before timing, followed by four
  timed suffixes attached with `restore=True`.

Each route runs an uninstrumented control and a minimally profiled variant for
three warmups followed by exactly ten retained measurements. The uninstrumented
CUDA median is the headline latency. The profiled variant adds non-overlapping
preparation and model envelopes; state commit is zero with an explicit reason
because publication occurs inside model forward. Each timed route has one outer
CUDA event pair and exactly one terminal stream synchronization.

Row hooks, backend trace wrappers, hidden/GDN captures, trace snapshots, full
tensor copies, tensor comparisons, finite scans, and JSON/report generation are
absent from the timed path. Compact active-position top-1 values are copied and
hashed only after the route timer is finalized. Warm-prefix construction is
preconditioning and is excluded from the warm route interval. A hash mismatch
still fails closed; its optional outside-timing diagnostic contains only the
paired token IDs, first differing flattened index, route identity, workload
fingerprint, and reference label—never logits or hidden tensors.

Accounting verifies prefix positions are exactly `range(0, b)`, suffix
positions equal the plan's absolute attention-query positions, and prefix plus
suffix equal full-sequence work on every full-replay step. Attention and GDN
token-layer counts include both full-replay forwards. Cold and warm count only
the prefix/suffix work actually executed inside their respective route.

## Physical KV-reuse evidence

The canonical production suffix is assembled through `prepare_for_extend`, not
`prepare_for_region_dag_replay`. Consequently, the legacy
`RegionDAGInstrumentation.kv_cache_hits` counter is not available for this path
and is retained only as a separately labeled raw diagnostic. It is never used
as proof of production reuse.

For every cold and warm restored suffix, the benchmark retains the request-pool
slot and exact `int64` prefix-location tensor without inspecting either tensor
inside the timed route. After the route's existing terminal synchronization, it
requires all four restore steps and compares each retained tensor with the
authoritative canonical page table for the same request slot and 2,048-token
boundary. Shape, dtype, contiguity, device, physical-pool bounds, and exact
tensor identity all fail closed. A passing route reports
`method=canonical_prefix_page_table_identity` and 8,192 observed reused
positions: 2,048 prefix positions across four restored suffix steps.

This post-timing verification adds no synchronization or device-to-host copy to
the measured interval. Output-hash, recovery, fallback, position, work, and
one-terminal-synchronization gates remain mandatory.

The run fails closed unless the accepted `efficiency_one` summary and preflight
belong to `057dcab2261895a0e32357f5d57c65e1eb4b3b8c`, the checkpoint hashes
still match, and both runs use the same one-A30 CUDA/PyTorch environment. It
also fails publication when output hashes differ, a valid warm hit recovers or
falls back, measured work differs between variants, a route uses more than one
timed terminal synchronization, or profiling overhead exceeds 5%. A numerical
speedup is not a supported claim unless the bootstrap improvement interval is
strictly positive.

## Exact A30 command

Run this without Nsight so the uninstrumented control remains the production
headline. The prerequisite files are the retained accepted artifacts generated
at commit `057dcab2261895a0e32357f5d57c65e1eb4b3b8c`.

```bash
set -euo pipefail

export REPO="$PWD"
export MODEL_PATH=/persistent/hybrid-diffusion-cache/models/HybridDiffusion-2B
export CACHE_ROOT=/persistent/hybrid-diffusion-cache
export RESULT_ROOT=/persistent/hybrid-diffusion-cache/results/cluster3-production-efficiency
export PY=/persistent/hybrid-diffusion-cache/venvs/hybrid-diffusion-eval/bin/python
export EXPECTED_PARENT=465044dc0266ace753581ba92fb475e465a721a9
export ACCEPTED_CORRECTNESS_REVISION=057dcab2261895a0e32357f5d57c65e1eb4b3b8c
export CORRECTNESS_DIR="$RESULT_ROOT/$ACCEPTED_CORRECTNESS_REVISION/a30-production-latency-05-correctness"
export CORRECTNESS_ARTIFACT="$CORRECTNESS_DIR/summary.json"
export CORRECTNESS_PREFLIGHT="$CORRECTNESS_DIR/preflight.json"
export PRODUCTION_OUT="$RESULT_ROOT/production-latency/$(git rev-parse HEAD)"
export CUDA_VISIBLE_DEVICES=0
export FLASHINFER_WORKSPACE_BASE="$CACHE_ROOT/flashinfer-workspaces/cluster3-production-$(git rev-parse --short=12 HEAD)"
export TORCH_EXTENSIONS_DIR="$FLASHINFER_WORKSPACE_BASE/torch-extensions"

test "$(git branch --show-current)" = cluster3-production-kv-reuse-evidence-fix
test "$(git rev-parse HEAD^)" = "$EXPECTED_PARENT"
test -f "$CORRECTNESS_ARTIFACT"
test -f "$CORRECTNESS_PREFLIGHT"
test -f "$MODEL_PATH/config.json"
test -f "$MODEL_PATH/model-00001-of-00001.safetensors"
test -f "$MODEL_PATH/tokenizer.json"
test -f "$MODEL_PATH/tokenizer_config.json"
test -f "$MODEL_PATH/chat_template.jinja"
test "$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["cluster3_revision"])' \
  "$CORRECTNESS_ARTIFACT")" = "$ACCEPTED_CORRECTNESS_REVISION"

unset CUDA_LAUNCH_BLOCKING
unset SGLANG_DLLM_REQUEST_METRICS
unset SGLANG_DLLM_REQUEST_METRICS_ALL_MODELS
unset SGLANG_HYBRID_DIFFUSION_SELF_SPEC_EXTRA_BUFFER_TRACE
unset SGLANG_HYBRID_DIFFUSION_SELF_SPEC_DEBUG_STEPS
unset SGLANG_HYBRID_DIFFUSION_SELF_SPEC_TRACE_PATH
unset SGLANG_HYBRID_EXACT_HANDOFF_DEBUG
unset SGLANG_HYBRID_EXACT_HANDOFF_DEBUG_SYNC

mkdir -p "$PRODUCTION_OUT"
"$PY" "$REPO/eval/scripts/cluster3_region_dag_validation.py" \
  --model-path "$MODEL_PATH" \
  --profile production_efficiency \
  --dtype bfloat16 \
  --tp-size 1 \
  --device 0 \
  --max-total-tokens 4096 \
  --timed-repetitions 10 \
  --correctness-artifact "$CORRECTNESS_ARTIFACT" \
  --preflight-json "$CORRECTNESS_PREFLIGHT" \
  --output-jsonl "$PRODUCTION_OUT/raw.jsonl" \
  --summary-json "$PRODUCTION_OUT/summary.json"
```

This local implementation does not claim CUDA or A30 acceptance. The retained
A30 `summary.json` determines which optimization, if any, should be attempted
next.
