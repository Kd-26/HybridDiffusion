# Cluster-3 cache-system comparison

This benchmark compares cache mechanisms inside the same HybridDiffusion-2B
runtime.  It is an evaluation harness, not a serving mode.  The semantic audit
is in `docs/cluster3_cache_system_comparison_audit.md`.

## Methods

The registry always exposes all five requested identifiers.  Three are exact
and executable:

- `controlled_full_replay` reconstructs the full canonical sequence on every
  diffusion step and consumes no prior-step state.
- `safe_kv_only_diffusion` retains physical attention-KV pages, reconstructs
  GDN state by a complete fresh-page prefix forward on every step, frees those
  temporary pages, and attaches the retained pages before the suffix forward.
  It restores no GDN snapshot and is intentionally conservative.
- `complete_region_state_execution` is the existing validated production warm
  route with exact page-table and GDN-frontier restoration.

Two methods are unavailable and produce no fake records:

- `upstream_sglang_radix`:
  `native_radix_lacks_exact_hybrid_recurrent_state`.  This label means bundled
  native SGLang Radix behavior, not another SGLang checkout.
- `kv_gdn_handoff_full_rows`:
  `state_handoff_and_active_rows_structurally_coupled`.

Consequently `publication_matrix_complete` is false.  Passing
`--require-all-methods` fails before checkpoint inspection or model loading.

## Timing and correctness

Headline latency comes only from uninstrumented repetitions.  Separate
minimally profiled repetitions install temporary CUDA-event probes for
attention, GDN, MLP, snapshot, and restore work.  Parent envelopes and nested
shared work are labelled explicitly; snapshot/restore intervals are not added
again to their GDN parent.  Evaluation-only probes also time mask construction,
FlashInfer planning, and physical page-table lookup under their shared prepare
parent.  Components that do not execute remain `null` with a reason.  Both
model and route coverage fail below 90%.

Each route uses one terminal stream synchronization.  Output transfer,
hashing, token equality, physical page-table validation, and stable-KV hashing
occur after timing.  The benchmark rejects fallback, recovery, changed
positions, changed stable tensors, invalid page locations, output mismatch, or
method-specific GDN behavior mismatch.

The checked-in manifest contains exactly the 2,080-token favorable suffix,
1,152-token typical suffix, and 1,536-token fragmented Region-DAG cases.

## A30 command

The correctness and preflight paths below are placeholders for artifacts from
the accepted Cluster-3 A30 validation; replace only those two artifact paths and
the output directory.

```bash
MODEL_PATH=/persistent/hybrid-diffusion-cache/models/HybridDiffusion-2B
CACHE_ROOT=/persistent/hybrid-diffusion-cache
PY=/persistent/hybrid-diffusion-cache/venvs/hybrid-diffusion-eval/bin/python

"$PY" eval/scripts/cluster3_cache_system_comparison.py \
  --model-path "$MODEL_PATH" \
  --case-manifest eval/manifests/cluster3_cache_system_comparison.json \
  --methods controlled_full_replay upstream_sglang_radix safe_kv_only_diffusion kv_gdn_handoff_full_rows complete_region_state_execution \
  --dtype bfloat16 \
  --tp-size 1 \
  --device 0 \
  --max-total-tokens 4096 \
  --warmups 3 \
  --timed-repetitions 10 \
  --correctness-artifact "$CACHE_ROOT/artifacts/cluster3_efficiency_one_correctness.json" \
  --preflight-json "$CACHE_ROOT/artifacts/cluster3_efficiency_preflight.json" \
  --output-jsonl "$CACHE_ROOT/results/cluster3_cache_system_comparison.jsonl" \
  --summary-json "$CACHE_ROOT/results/cluster3_cache_system_comparison_summary.json"
```

Do not add `--require-all-methods` to the publication run while either audited
method is unsupported.  The normal run records those registry entries and
finishes the three valid methods.

## Frozen original FLARE ingestion contract

The controlled oracle is not the original public FLARE implementation.  Run
revision `6ca547aebb72bfe897e80e0a683c776789e5f38c` in a separate clean checkout,
without importing this checkout, using the same checkpoint, one A30, BF16,
TP=1, batch size 1, token inputs, four diffusion steps, three warmups, and ten
timed repetitions.  Use ordinary unmodified FLARE execution.  Only the two
contiguous suffix cases are representable; mark the fragmented case
unsupported.

The external JSONL must label every record
`method: original_flare_external_runtime`, set
`runtime_revision: 6ca547aebb72bfe897e80e0a683c776789e5f38c`, and include the
checkout commit, dirty status, Python/PyTorch/CUDA/SGLang provenance, checkpoint
identity, hardware identity, case ID, token-input hash, diffusion-step count,
warmup flag, repetition index, and latency.  Ingest it with
`--original-flare-jsonl PATH`.  The harness validates the label/revision,
rejects a fragmented-case claim, and compares its token-input hash with the
exact masked token arrays exported under `case_inputs` in the comparison
summary.  The external record uses object-valued `software_provenance`,
`hardware`, and `checkpoint_identity` fields plus `diffusion_steps`, `warmup`,
`repetition_index`, and `total_cuda_ms`.  The summary records that these values
are excluded from same-runtime ablation statistics; it never combines absolute
external-runtime latency with the same-runtime speedups.

No performance claim is valid until the A30 command completes against the
validated checkpoint and all supported-method correctness gates pass.
