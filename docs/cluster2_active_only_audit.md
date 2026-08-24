# Cluster 2: active-only suffix validation audit

## Scope and provenance

Cluster 2 is a validation-only artifact. Active-only suffix execution was
already present in the HybridDiffusion production runtime at revision
`1e21bdec792a665419b290bf89f75c14c730dc94`; this work does not introduce,
replace, or optimize that runtime. It therefore makes no production-speedup or
runtime-novelty claim. Its contribution is a reproducible protocol that proves
the existing tensor shapes and compares their numerical results with a
controlled replay.

The Qwen3.5 Torchtitan `[x0; xt]` construction is a training-only two-stream
layout. It is deliberately outside this artifact and remains unchanged.

## Production call path

The audited server path is:

1. `ScheduleBatch.prepare_for_dllm_decode()` derives one extend length per
   request and allocates `sum(extend_lens)` input/KV rows. The stable prefix is
   represented by `kv_committed_len`, not copied into `input_ids`.
2. `ForwardBatch.init_new()` constructs suffix positions from each request's
   `dllm_block_offset`, preserving the original absolute position IDs.
3. `HybridDiffusionSelfSpec` restores the exact-prefix GDN snapshot before the
   forward and submits the active `ForwardBatch` to the unchanged model.
4. FlashInfer creates `qo_indptr` from `seq_lens - prefix_lens`, so query rows
   are active-only. Its KV indirection spans the stable prefix plus newly
   allocated active suffix.
5. Q/K/V projections, attention output, every MLP, and every GDN layer receive
   only the flattened active hidden rows. GDN starts from restored prefix state.
6. Post-forward verification accepts a model-authoritative advance, trims or
   invalidates unused active KV, and publishes the next exact boundary.

This distinction is central: **query rows are active-only, while readable KV
rows are stable-plus-active**. A KV length larger than the submitted input-row
count is expected and is not stable-prefix recomputation.

## Dynamic protocol

`eval/scripts/cluster2_active_only_validation.py` creates deterministic cases
for `one1`, `smoke16`, and `paper100`. It loads one native-BF16, TP=1 runtime
and installs temporary observation hooks only around proof collection. The
hooks observe attention, MLP, GDN, and transformer row counts and clone bounded
active outputs. `attention_query_rows_per_layer` is observed at
`qkv_projection_input`: every input hidden row to a full-attention layer's real
`qkv_proj` module produces exactly one corresponding query row. Qwen3.5's
bound `self_attention` helper method is not treated as a hookable module. Older
test/model layers remain compatible only when `self_attention` is itself an
actual `torch.nn.Module`. The hooks are removed on both successful and
exceptional exits.

After each synchronized forward, full logits and per-layer hidden/GDN evidence
are copied into independent contiguous CPU tensors and the hook-owned CUDA
references are released. Numerical comparison converts bounded tensor chunks
to FP32, so full-vocabulary logits do not create several full-size CUDA
temporaries. This changes only validation evidence storage: model execution,
cache allocation, BF16 computation, hashes, top-1 checks, and numerical
thresholds remain unchanged.

For each case, the validator performs two executions with identical weights,
tokens, positions, masks, dtype, and diffusion inputs:

- The controlled reference replays causal prefix state before each active
  diffusion step. This is deliberately expensive and exists only to establish
  an independent numerical reference.
- The cached baseline builds the prefix once, commits its KV/GDN state, restores
  that state for subsequent steps, and uses the existing active-only production
  tensor layout.

The reference replay is not presented as an alternative production mode. The
cached path is the optimized production baseline; the experiment only compares
shape, state preservation, active logits, layer hidden states, GDN state, and
top-1 output.

For the prefix-zero reduction there is no stable KV or GDN state to commit or
restore. Both sides therefore execute the entire sequence as active rows. Every
diffusion-step reevaluation clears the controlled-test pools and creates fresh
requests, allowing the real `ScheduleBatch` to allocate request slots while
preventing recurrent GDN state from leaking across steps. The canonical empty
`torch.int64` prefix-location tensor is validated in place; no synthetic request
slot or cache state is created.

The artifact accepts only a loaded TP=1 Qwen3.5-2B architecture fingerprint
(`hidden_size=2048`, 24 layers, `intermediate_size=6144`) and emits
`model_scale: "2B"`. Unknown, ambiguous, and differently sized checkpoints fail
closed rather than receiving a guessed model-scale label.

Every case emits one bounded JSONL record. Position evidence contains count,
first/last values, an eight-value preview, and a SHA-256 digest rather than a
large tensor. Stable K/V and GDN state are hashed before and after cached
execution. Prefix-zero cases explicitly mark stable-state checks inapplicable;
missing evidence for an applicable check is always a failure.

The summary requires exact record cardinality, active-only row counts at every
applicable layer, stable-plus-active KV lengths, exact positions, unchanged
stable state, finite BF16 errors below `1e-2`, identical top-1 tokens, no
fallback/recovery replay, required batch coverage, and an empty production
runtime diff against the fixed base revision.

Text output is preview-only and is never used as correctness evidence. TP sizes
above one are rejected until the collector can distinguish rank-local shards
from complete tensors.

## A30 commands

Run from the repository root with a local Qwen3.5-2B HybridDiffusion model.
Do not add `--enable-deterministic-inference`.

One-case debugging may use blocking launches and stage synchronization:

```bash
CUDA_LAUNCH_BLOCKING=1 python eval/scripts/cluster2_active_only_validation.py \
  --model-path /path/to/model \
  --output-jsonl artifacts/cluster2-one1.jsonl \
  --summary-json artifacts/cluster2-one1-summary.json \
  --profile one1 \
  --dtype bfloat16 \
  --tp-size 1 \
  --device 0 \
  --seed 20260825 \
  --max-total-tokens 8192 \
  --debug-sync-stages
```

The smoke and paper profiles must not use `CUDA_LAUNCH_BLOCKING=1` when timing:

```bash
python eval/scripts/cluster2_active_only_validation.py \
  --model-path /path/to/model \
  --output-jsonl artifacts/cluster2-smoke16.jsonl \
  --summary-json artifacts/cluster2-smoke16-summary.json \
  --profile smoke16 \
  --dtype bfloat16 \
  --tp-size 1 \
  --device 0 \
  --seed 20260825 \
  --max-total-tokens 8192
```

```bash
python eval/scripts/cluster2_active_only_validation.py \
  --model-path /path/to/model \
  --output-jsonl artifacts/cluster2-paper100.jsonl \
  --summary-json artifacts/cluster2-paper100-summary.json \
  --profile paper100 \
  --dtype bfloat16 \
  --tp-size 1 \
  --device 0 \
  --seed 20260825 \
  --max-total-tokens 8192
```

The artifact is valid only when the summary reports `strict_pass: true`. A run
on non-A30 hardware may still be useful diagnostically but is not an A30
acceptance result.
