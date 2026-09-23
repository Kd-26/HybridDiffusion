# Cluster-3 production latency benchmark

`production_efficiency` is a production-only latency path. It preserves the
`efficiency_one` controlled correctness path and does not implement a runtime
optimization. The fixed workload is one HybridDiffusion-2B request with a
2,048-token stable prefix, a 64-token active suffix, four diffusion steps,
batch size one, BF16, and TP=1.

The benchmark measures `full_replay`, `cold_handoff_build`, and
`warm_cached_suffix`. Each route runs an uninstrumented control and a minimally
profiled variant for three warmups followed by exactly ten retained
measurements. The uninstrumented CUDA median is the headline latency. The
profiled variant adds non-overlapping preparation and model envelopes; state
commit is zero with an explicit reason because publication occurs inside model
forward. Each timed route has one outer CUDA event pair and exactly one
terminal stream synchronization.

Row hooks, backend trace wrappers, hidden/GDN captures, trace snapshots, full
tensor copies, tensor comparisons, finite scans, and JSON/report generation are
absent from the timed path. Compact active-position top-1 values are copied and
hashed only after the route timer is finalized. Warm-prefix construction is
preconditioning and is excluded from the warm route interval.

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
export EXPECTED_PARENT=057dcab2261895a0e32357f5d57c65e1eb4b3b8c
export CORRECTNESS_ARTIFACT="$RESULT_ROOT/profile-one/summary.json"
export CORRECTNESS_PREFLIGHT="$RESULT_ROOT/profile-one/preflight.json"
export PRODUCTION_OUT="$RESULT_ROOT/production-latency/$(git rev-parse HEAD)"
export CUDA_VISIBLE_DEVICES=0

test "$(git branch --show-current)" = cluster3-production-latency-benchmark
test "$(git rev-parse HEAD^)" = "$EXPECTED_PARENT"
test -f "$CORRECTNESS_ARTIFACT"
test -f "$CORRECTNESS_PREFLIGHT"
test -f "$MODEL_PATH/config.json"
test -f "$MODEL_PATH/model-00001-of-00001.safetensors"
test -f "$MODEL_PATH/tokenizer.json"
test -f "$MODEL_PATH/tokenizer_config.json"
test -f "$MODEL_PATH/chat_template.jinja"

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
