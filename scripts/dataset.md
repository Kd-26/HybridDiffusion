<p align="right">
  <a href="../README.md">HybridDiffusion</a> ·
  <a href="#source-datasets">Sources</a> ·
  <a href="#build-datasets">Build</a> ·
  <a href="#output-schema">Schema</a>
</p>

# Data preparation

This guide prepares the three Arrow datasets consumed by the released HybridDiffusion
Qwen3.5 training configurations. The builders convert heterogeneous public
conversation datasets into one schema, apply source-specific filtering, and
produce the weighted training mixture below.

## Dataset mixture

| Dataset key | Directory below `DATA_ROOT` | Mixture weight | Content |
|---|---|---:|---|
| `nemotron_long_sft_3k` | `Long-SFT-3K` | 0.4 | Long-form reasoning and instruction data |
| `nemotron_math_domain` | `Nemotron-Math-Domain` | 0.4 | Mathematical reasoning data |
| `nemotron_if_domain_cascade` | `Nemotron-IF-Domain-Cascade` | 0.2 | Instruction-following data |

The same mixture is used by [`hybrid_diffusion_2b.toml`](../torchtitan/experiments/qwen3_5/train_configs/hybrid_diffusion_2b.toml),
[`hybrid_diffusion_4b.toml`](../torchtitan/experiments/qwen3_5/train_configs/hybrid_diffusion_4b.toml),
and [`hybrid_diffusion_9b.toml`](../torchtitan/experiments/qwen3_5/train_configs/hybrid_diffusion_9b.toml).
Each source is packed independently to sequence length 4,096; examples from
different mixture sources are not combined in one packed sequence.

## Prerequisites

- Install the training environment with `scripts/setup_envs_models_qwen35.sh`.
- Store source datasets on persistent local storage in Hugging Face Arrow
  format compatible with `datasets.load_from_disk`.
- Set `RAW_DATA_ROOT` to the parent of the downloaded source datasets.
- Set `DATA_ROOT` to the destination for the three processed datasets.
- Review and comply with the licenses and terms of every source dataset.

Hugging Face authentication, when required, is read from `HF_TOKEN`. Do not
put credentials in configuration files or command-line arguments.

## Source datasets

| Source | Required split or configuration | Used by |
|---|---|---|
| [Llama-Nemotron-Post-Training-Dataset](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) | `SFT` | Long-SFT and Math |
| [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) | English SFT | Long-SFT |
| [Nemotron-Instruction-Following-Chat-v1](https://huggingface.co/datasets/nvidia/Nemotron-Instruction-Following-Chat-v1) | `chat_if`, `structured_outputs` | Long-SFT and IF |
| [Nemotron-SFT-Instruction-Following-Chat-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v2) | all normalized SFT splits | Long-SFT |
| [Nemotron-Science-v1](https://huggingface.co/datasets/nvidia/Nemotron-Science-v1) | default SFT data | Long-SFT |
| [Nemotron-Math-Proofs-v1](https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v1) | default SFT data | Long-SFT and Math |
| [Nemotron-SFT-Competitive-Programming-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Competitive-Programming-v2) | default SFT data | Long-SFT |
| [Nemotron Cascade-1 Stage-2](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-SFT-Stage-2) | `instruction-following` | IF |
| [Nemotron Cascade-2](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data) | `instruction_following` | IF |

## Input layout

The default source paths are:

```text
${RAW_DATA_ROOT}/
├── Llama-Nemotron-Post-Training-Dataset-arrow/SFT/
├── Nemotron-Instruction-Following-Chat-v1/
├── Nemotron-Cascade-SFT-Stage-2/
│   └── instruction-following/
├── Nemotron-Cascade-2-SFT-Data/
│   └── instruction_following/
├── Nemotron-SFT-Instruction-Following-Chat-v2/
├── Nemotron-Science-v1/
├── Nemotron-Math-Proofs-v1/
├── Nemotron-SFT-Competitive-Programming-v2/
└── Nemotron-Post-Training-Dataset-v2-arrow/
```

Use `--source_path KEY=/absolute/path` to override any source location. The
option may be repeated; each builder's `--help` output lists its accepted
keys.

## Download source data

The download helpers save Hugging Face datasets as local Arrow directories.

```bash
python scripts/download_nemotron_chat.py \
  --output_dir "${RAW_DATA_ROOT}" \
  --repo_ids \
    nvidia/Nemotron-Instruction-Following-Chat-v1 \
    nvidia/Nemotron-Science-v1 \
    nvidia/Nemotron-Math-Proofs-v1 \
    nvidia/Nemotron-SFT-Competitive-Programming-v2

python scripts/download_nemotron_chat_v2.py \
  --output_dir "${RAW_DATA_ROOT}"

python scripts/download_nemotron_chat.py \
  --output_dir "${RAW_DATA_ROOT}" \
  --repo_ids nvidia/Nemotron-Cascade-SFT-Stage-2 \
  --configs instruction-following

python scripts/download_nemotron_chat.py \
  --output_dir "${RAW_DATA_ROOT}" \
  --repo_ids nvidia/Nemotron-Cascade-2-SFT-Data \
  --configs instruction_following
```

Place the Llama-Nemotron and Nemotron Post-Training-v2 Arrow datasets at the
paths shown above, or pass their locations with `--source_path`.

## Build datasets

Run all builders from the repository root after activating the training
environment.

### Long-SFT

```bash
python scripts/build_long_sft_dataset.py \
  --data_root "${RAW_DATA_ROOT}" \
  --threshold 3000 \
  --output_dir "${DATA_ROOT}/Long-SFT-3K"
```

The builder selects supported Long-SFT source families, normalizes their
conversation schemas, formats reasoning as balanced `<think>...</think>`
blocks, rejects malformed conversations, and emits the common Arrow schema
described below.

### Math

```bash
python scripts/build_domain_dataset.py \
  --data_root "${RAW_DATA_ROOT}" \
  --domain math \
  --output_dir "${DATA_ROOT}/Nemotron-Math-Domain"
```

The Math builder selects mathematical reasoning examples, normalizes message
roles and reasoning fields, rejects malformed or unbalanced think blocks, and
emits the common Arrow schema.

### Instruction following

The released `Nemotron-IF-Domain-Cascade` configuration expects a
constraint-filtered Cascade-2 Arrow input. Provide it through `cascade2_if` and
select `prefiltered` mode:

```bash
python scripts/build_if_dataset.py \
  --data_root "${RAW_DATA_ROOT}" \
  --source_path cascade2_if="${FILTERED_CASCADE2_IF}" \
  --cascade2_mode prefiltered \
  --output_dir "${DATA_ROOT}/Nemotron-IF-Domain-Cascade"
```

To build a separate variant from the complete public `instruction_following`
split, use `raw_open` mode and a distinct output name:

```bash
python scripts/build_if_dataset.py \
  --data_root "${RAW_DATA_ROOT}" \
  --cascade2_mode raw_open \
  --output_dir "${DATA_ROOT}/Nemotron-IF-Domain-Cascade-Open"
```

Both modes combine Cascade-1 instruction-following data, the selected
Cascade-2 input, Chat-v1 rows whose `capability_target` is
`instruction_following`, and the Chat-v1 `structured_outputs` split. The
builder normalizes messages and think blocks, rejects invalid conversations,
and deduplicates by the serialized normalized message sequence.

Only the `prefiltered` output matches the dataset key used by the released
training configurations; `raw_open` produces a separate data variant.

## Output schema

All three builders emit a Hugging Face Arrow dataset with these columns:

| Column | Description |
|---|---|
| `messages_json` | Serialized chat messages as role/content objects |
| `category` | Normalized data category |
| `source` | Source dataset and split identifier |
| `total_tokens` | Estimated serialized token count |
| `assistant_tokens` | Estimated supervised assistant-token count |

Separate assistant `reasoning_content` is inserted into `content` as a
balanced `<think>...</think>` block. The domain and IF builders insert an empty
think block when a supported sample has no separate reasoning field.

Filtering follows the needs of each dataset:

- Long-SFT removes empty normalized records, rejects assistant messages with
  nested, out-of-order, or unclosed think tags, and applies the requested
  estimated-token threshold.
- Math applies the selected domain filters, removes empty normalized records,
  and rejects assistant messages that open a think block without closing it.
- IF requires supported roles, non-empty user and assistant turns, at least one
  user and one assistant turn, a consistent inline/separate reasoning format,
  and balanced non-nested think blocks.

The IF builder additionally performs exact deduplication. It hashes the complete
normalized `messages_json` value with SHA-256 and keeps the first occurrence in
source order. Long-SFT and Math do not perform conversation-level
deduplication.

## Use the mixture in training

After all three directories exist below the same `DATA_ROOT`, launch one of the
released configurations:

```bash
export DATA_ROOT=/persistent/data/hybrid_diffusion
export OUTPUT_DIR=/persistent/outputs/test

bash scripts/run_training.sh \
  torchtitan/experiments/qwen3_5/train_configs/hybrid_diffusion_2b.toml
```

Use `hybrid_diffusion_4b.toml` or `hybrid_diffusion_9b.toml` for the larger model configurations.
The dataset mixer state is included in training checkpoints so resumed runs
preserve deterministic source sampling.
