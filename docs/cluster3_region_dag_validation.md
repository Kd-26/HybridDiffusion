# Cluster-3 Region-DAG validation

Cluster 3 implements the opt-in `region_dag_conservative_gdn_v1` contract. It
keeps graph invalidation and recurrent execution deliberately separate:

- The dependency graph computes the exact logical invalidation closure of the
  edited regions.
- Full-attention KV before the earliest invalidated position can be reused only
  under an exact frontier identity.
- GDN recurrence is ordered, so every position from the earliest invalidated
  position to sequence end is replayed. A later stable region may therefore be
  logically valid but still recomputed.
- Arbitrary Region-DAG requests always use FlashInfer `custom_paged`; the
  legacy prefix/suffix routes are unchanged.

## Controlled exporter

`eval/scripts/cluster3_region_dag_validation.py` loads one real
HybridDiffusion-2B model and runs a complete Region-DAG recomputation and a
production-metadata cached replay for the same inputs. It compares every
captured transformer-layer hidden state, final hidden state, GDN convolution
and recurrent state, logits, and top-1 token. Pre-frontier full-attention KV
and exact GDN state are hashed before and after cached execution and are also
matched to the reference execution.

The exporter fails closed when evidence is absent, the model is not confidently
identified as 2B, TP is not one, dtype is not BF16, a real request slot is
missing, a page-table location is invalid, the custom mask is not selected,
absolute positions change, a GDN frontier cannot be proven, a negative cache
identity hits, numerical error reaches `1e-2`, or fallback/recovery is seen.

Profiles are deterministic:

- `one1`: the exact 256-token A/B/C/D/E diagnostic layout, edit B, two steps.
- `smoke16`: 16 bounded cases covering 256/512/1024 tokens, one/two/four
  active regions, early/middle/late edits, 2/4/8 steps, and an
  entirely-active reduction.
- `paper100`: 100 unique, seeded graph/layout tuples.
- `effectiveness`: nine 1,024-token early/middle/late layouts with one 64-token,
  two 32-token, or four 16-token active regions.

Each completed case produces exactly one JSONL record. Results belong in a new
Cluster-3 directory containing the exact checked-out revision and profile; the
Cluster-2 `paper100` artifacts must never be reused or overwritten.

### Layer-0 three-way equivalence triage

When monolithic BF16 execution first diverges at GDN layer 0, run the dedicated
diagnostic before changing the validation oracle:

```bash
REV=$(git rev-parse HEAD)
OUT="results/cluster3/${REV}/gdn-equivalence"
CUDA_LAUNCH_BLOCKING=1 python \
  eval/scripts/cluster3_gdn_equivalence_diagnostic.py \
  --model-path "$MODEL_DIR" \
  --output-dir "$OUT" \
  --dtype bfloat16 \
  --tp-size 1 \
  --max-total-tokens 4096 \
  --debug-sync-stages
```

The command writes `one1-three-way-diagnostic.json`. Path A is the 256-row
monolithic numerical audit. Path B freshly recomputes rows 0–63 from zero and
then continues the live state for rows 64–255; its `reference_full_ms` includes
both segments and it cannot read or publish a Region-DAG snapshot. Path C uses
the production exact-frontier restore and evaluates the same 192 suffix rows.
Only the strict, shape-matched B-versus-C comparison diagnoses cache handoff;
A-versus-B drift remains visible separately. The diagnostic reports evidence
and a decision case, but never claims A30 acceptance or changes the numerical
oracle automatically.

## A30 protocol

Use one NVIDIA A30, native BF16, TP=1, HybridDiffusion-2B, and at most 4096
total tokens. The notebook `eval/notebooks/cluster3_region_dag_a30.ipynb`
contains the reproducible commands and assertions. It accepts sharded or
single-file safetensor checkpoints and does not assume a particular shard
filename.

The first diagnostic is the only run that enables launch blocking and explicit
stage synchronization:

```bash
REV=$(git rev-parse HEAD)
OUT="results/cluster3/${REV}/one1"
mkdir -p "$OUT"
CUDA_LAUNCH_BLOCKING=1 python eval/scripts/cluster3_region_dag_validation.py \
  --model-path "$MODEL_DIR" \
  --profile one1 \
  --dtype bfloat16 \
  --tp-size 1 \
  --max-total-tokens 4096 \
  --debug-sync-stages \
  --output-jsonl "$OUT/records.jsonl" \
  --summary-json "$OUT/summary.json"
```

Remove `CUDA_LAUNCH_BLOCKING` and `--debug-sync-stages` for `smoke16`,
`paper100`, and `effectiveness`. The effectiveness profile performs one warm-up
and at least ten measured repetitions per case. It uses synchronized CUDA
events and reports independent samples, median, MAD, and deterministic
bootstrap 95% confidence intervals for:

- complete reference time;
- cached total time;
- full-attention time;
- conservative GDN replay time;
- mask construction;
- page-table gather/scatter;
- exact cache lookup and GDN restoration.

Full-attention and GDN token-layer positions come from verified runtime module
row hooks. The full-attention observation point is
`qkv_projection_input`, where one input hidden row produces one query row.
Cache population is excluded from `cached_total_ms`; it represents state that
already exists before the edited replay. The paired host timer is diagnostic
only and is never used to claim speedup.

## Result interpretation and serving gate

Cached execution is correct only when every case has `case_pass=true` and the
summary has `strict_pass=true`. A useful implementation may reduce
full-attention work while showing little or no end-to-end speedup: an early edit
forces a long conservative GDN replay. That result is expected and motivates
Cluster 4; it must not be hidden by a combined paired-case timer.

Run the production HTTP smoke only after controlled `one1` and `smoke16` pass.
The serving check must verify one finalization, request-pool cleanup, the
Region-DAG route, no surviving stale cache entry, and returned generated text.
Text quality is diagnostic rather than the principal numerical criterion.

This repository revision prepares the A30 workflow but does not claim an A30
result. A GPU pass is valid only when the notebook was actually executed on the
specified hardware and its JSON summaries were retained.
