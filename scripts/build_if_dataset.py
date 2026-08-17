#!/usr/bin/env python3
"""Build the instruction-following (IF) Arrow pool used by the training mix.

The public recipe combines four already-IF-oriented inputs:

1. Nemotron Cascade-1 Stage-2 ``instruction-following``;
2. Nemotron Cascade-2 ``instruction_following``;
3. Chat-v1 ``chat_if`` rows with
   ``capability_target == "instruction_following"``;
4. Chat-v1 ``structured_outputs``.

The original snapshot does not contain NVIDIA's additional Cascade-2
constraint-satisfaction selector (about 820K raw rows to about 362K rows).
``--cascade2_mode raw_open`` therefore builds a documented open approximation
from the complete public IF config.  ``--cascade2_mode prefiltered`` accepts a
locally recovered/precomputed filtered Arrow input and runs the same remaining
pipeline, which is the path for reproducing the paper pool exactly.

All sources are normalized to the five-column training schema:

    messages_json, category, source, total_tokens, assistant_tokens

Separate ``reasoning_content`` is merged into ``content`` using
``<think>...</think>``. Assistant turns without reasoning receive an empty
``<think></think>`` block. Malformed conversations and unbalanced think tags
are rejected, then rows are deduplicated by SHA-256 of the serialized normalized
messages before the final deterministic shuffle.

See ``scripts/dataset.md`` for source download and build commands.
"""

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter
from collections.abc import Callable
from functools import partial

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import numpy as np
from datasets import Dataset, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer


SOURCE_RELATIVE_PATHS = {
    "cascade1_if": "Nemotron-Cascade-SFT-Stage-2/instruction-following",
    "cascade2_if": "Nemotron-Cascade-2-SFT-Data/instruction_following",
    "chat_v1": "Nemotron-Instruction-Following-Chat-v1",
}

OUTPUT_COLUMNS = [
    "messages_json",
    "category",
    "source",
    "total_tokens",
    "assistant_tokens",
]

ALLOWED_ROLES = {"system", "user", "assistant"}
THINK_TAG_RE = re.compile(r"</?think>")


def _load_arrow(path: str) -> Dataset:
    """Load one Arrow Dataset, DatasetDict, or directory of Arrow leaves."""
    if os.path.isfile(os.path.join(path, "dataset_info.json")) or os.path.isfile(
        os.path.join(path, "state.json")
    ):
        return load_from_disk(path)

    if os.path.isfile(os.path.join(path, "dataset_dict.json")):
        dataset_dict = load_from_disk(path)
        return concatenate_datasets(list(dataset_dict.values()))

    subdirs = sorted(
        entry
        for entry in os.listdir(path)
        if os.path.isdir(os.path.join(path, entry))
        and (
            os.path.isfile(os.path.join(path, entry, "dataset_info.json"))
            or os.path.isfile(os.path.join(path, entry, "state.json"))
        )
    )
    if not subdirs:
        raise FileNotFoundError(f"No valid Arrow dataset found at {path}")
    return concatenate_datasets(
        [load_from_disk(os.path.join(path, entry)) for entry in subdirs]
    )


def resolve_source_paths(
    data_root: str, overrides: list[str] | None = None
) -> dict[str, str]:
    """Resolve conventional source paths plus repeatable ``KEY=PATH`` overrides."""
    root = os.path.abspath(os.path.expanduser(data_root))
    paths = {
        key: os.path.join(root, relative_path)
        for key, relative_path in SOURCE_RELATIVE_PATHS.items()
    }
    for item in overrides or []:
        key, separator, value = item.partition("=")
        if not separator or key not in paths or not value:
            valid = ", ".join(sorted(paths))
            raise ValueError(f"Invalid --source_path '{item}'. Expected one of: {valid}")
        paths[key] = os.path.abspath(os.path.expanduser(value))
    return paths


def require_source_path(source_paths: dict[str, str], key: str) -> str:
    path = source_paths[key]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing source '{key}' at {path}. "
            "Download/build it first; see scripts/dataset.md."
        )
    return path


def _think_tags_balanced(content: str) -> bool:
    """Accept zero or more non-nested, correctly ordered think blocks."""
    depth = 0
    for match in THINK_TAG_RE.finditer(content):
        if match.group(0) == "<think>":
            if depth != 0:
                return False
            depth = 1
        else:
            if depth != 1:
                return False
            depth = 0
    return depth == 0


def _invalid_result(reason: str, source: str) -> dict:
    return {
        "messages_json": "",
        "category": "if",
        "source": source,
        "total_chars": 0,
        "assistant_chars": 0,
        "_valid": False,
        "_invalid_reason": reason,
    }


def normalize_if_sample(sample: dict, source: str) -> dict:
    """Normalize one public IF row to role/content messages plus char counts."""
    messages = sample.get("messages")
    if not isinstance(messages, list) or not messages:
        return _invalid_result("missing_messages", source)

    cleaned: list[dict[str, str]] = []
    total_chars = 0
    assistant_chars = 0
    has_user = False
    has_assistant = False

    for message in messages:
        if not isinstance(message, dict):
            return _invalid_result("non_mapping_message", source)

        role = message.get("role")
        if role not in ALLOWED_ROLES:
            return _invalid_result("unsupported_role", source)

        content = message.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            return _invalid_result("non_string_content", source)

        # Empty system prompts are common in Chat-v1 and are valid. Empty user
        # or assistant turns are not useful training targets.
        if role in {"user", "assistant"} and not content.strip():
            return _invalid_result(f"empty_{role}", source)

        if role == "user":
            has_user = True

        if role == "assistant":
            has_assistant = True
            reasoning = message.get("reasoning_content")
            if reasoning is not None and not isinstance(reasoning, str):
                return _invalid_result("non_string_reasoning", source)

            if reasoning and reasoning.strip():
                if "<think>" in content or "</think>" in content:
                    return _invalid_result("reasoning_and_inline_think", source)
                normalized_reasoning = reasoning.strip("\n")
                normalized_answer = content.lstrip("\n")
                content = (
                    f"<think>\n{normalized_reasoning}\n</think>\n\n"
                    f"{normalized_answer}"
                )
            elif "<think>" not in content:
                content = f"<think></think>\n\n{content}"

            if not _think_tags_balanced(content):
                return _invalid_result("unbalanced_think", source)
            assistant_chars += len(content)

        cleaned.append({"role": role, "content": content})
        total_chars += len(content)

    if not has_user:
        return _invalid_result("missing_user", source)
    if not has_assistant:
        return _invalid_result("missing_assistant", source)

    return {
        "messages_json": json.dumps(cleaned, ensure_ascii=False),
        "category": "if",
        "source": source,
        "total_chars": total_chars,
        "assistant_chars": assistant_chars,
        "_valid": True,
        "_invalid_reason": "",
    }


def calibrate_ratios(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    n_samples: int = 3000,
) -> tuple[float, float]:
    """Estimate median characters/token for all and assistant-only content."""
    if len(dataset) == 0:
        raise ValueError("Cannot calibrate token ratios on an empty dataset")

    indices = np.random.default_rng(42).choice(
        len(dataset), size=min(n_samples, len(dataset)), replace=False
    )
    subset = dataset.select(indices.tolist())

    all_contents: list[str] = []
    assistant_contents: list[str] = []
    total_chars: list[int] = []
    assistant_chars: list[int] = []
    for messages_json, total_count, assistant_count in zip(
        subset["messages_json"],
        subset["total_chars"],
        subset["assistant_chars"],
    ):
        messages = json.loads(messages_json)
        all_contents.append("\n".join(message["content"] for message in messages))
        assistant_contents.append(
            "\n".join(
                message["content"]
                for message in messages
                if message["role"] == "assistant"
            )
        )
        total_chars.append(total_count)
        assistant_chars.append(assistant_count)

    template_overhead = 15
    encoded_all = tokenizer(
        all_contents, add_special_tokens=False, padding=False, truncation=False
    )
    encoded_assistant = tokenizer(
        assistant_contents, add_special_tokens=False, padding=False, truncation=False
    )
    total_tokens = np.array(
        [len(ids) + template_overhead for ids in encoded_all.input_ids]
    )
    assistant_tokens = np.array(
        [len(ids) for ids in encoded_assistant.input_ids]
    )

    total_ratio = np.median(
        np.array(total_chars, dtype=np.float64) / np.maximum(total_tokens, 1)
    )
    assistant_ratio = np.median(
        np.array(assistant_chars, dtype=np.float64)
        / np.maximum(assistant_tokens, 1)
    )
    return float(total_ratio), float(assistant_ratio)


def process_source(
    dataset: Dataset,
    source: str,
    tokenizer: AutoTokenizer,
    num_proc: int,
) -> tuple[Dataset, float, float]:
    """Normalize, structurally filter, and calibrate one selected IF source."""
    started = time.time()
    print(f"\n  [{source}] selected input: {len(dataset):,} rows")
    normalize_fn = partial(normalize_if_sample, source=source)
    processed = dataset.map(
        normalize_fn,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
    )

    valid_count = int(sum(processed["_valid"]))
    removed_count = len(processed) - valid_count
    invalid_reasons = Counter(
        reason for reason in processed["_invalid_reason"] if reason
    )
    processed = processed.filter(lambda row: row["_valid"], num_proc=num_proc)
    processed = processed.remove_columns(["_valid", "_invalid_reason"])
    print(
        f"    normalized/valid: {len(processed):,}; "
        f"removed malformed or unbalanced: {removed_count:,}"
    )
    if invalid_reasons:
        print(
            "    rejection reasons: "
            + ", ".join(
                f"{reason}={count:,}"
                for reason, count in sorted(invalid_reasons.items())
            )
        )
    if len(processed) == 0:
        raise ValueError(f"Source '{source}' has no valid rows after normalization")

    total_ratio, assistant_ratio = calibrate_ratios(processed, tokenizer)
    print(
        f"    calibration: total={total_ratio:.3f} chars/token, "
        f"assistant={assistant_ratio:.3f} chars/token; "
        f"elapsed={time.time() - started:.1f}s"
    )
    return processed, total_ratio, assistant_ratio


def _filter_dataset(
    dataset: Dataset,
    predicate: Callable[[dict], bool],
    label: str,
    num_proc: int,
) -> Dataset:
    before = len(dataset)
    filtered = dataset.filter(predicate, num_proc=num_proc)
    print(f"  {label}: {before:,} -> {len(filtered):,}")
    return filtered


def _serialized_message_digest(messages_json: str) -> bytes:
    return hashlib.sha256(messages_json.encode("utf-8")).digest()


def deduplicate_messages(dataset: Dataset) -> tuple[Dataset, int]:
    """Keep the first row for each exact normalized ``messages_json`` hash."""
    seen: set[bytes] = set()
    keep_indices: list[int] = []
    offset = 0
    for batch in dataset.iter(batch_size=2048):
        for messages_json in batch["messages_json"]:
            digest = _serialized_message_digest(messages_json)
            if digest not in seen:
                seen.add(digest)
                keep_indices.append(offset)
            offset += 1
    removed = len(dataset) - len(keep_indices)
    return dataset.select(keep_indices), removed


def _add_estimated_tokens(
    dataset: Dataset,
    total_ratio: float,
    assistant_ratio: float,
) -> Dataset:
    total_chars = np.array(dataset["total_chars"], dtype=np.float64)
    assistant_chars = np.array(dataset["assistant_chars"], dtype=np.float64)
    total_tokens = np.round(total_chars / total_ratio + 15).astype(np.int32)
    assistant_tokens = np.round(assistant_chars / assistant_ratio).astype(np.int32)
    dataset = dataset.add_column("total_tokens", total_tokens.tolist())
    dataset = dataset.add_column("assistant_tokens", assistant_tokens.tolist())
    return dataset


def report_stats(dataset: Dataset) -> None:
    total_tokens = np.array(dataset["total_tokens"])
    assistant_tokens = np.array(dataset["assistant_tokens"])
    sources = np.array(dataset["source"])

    def print_one(label: str, mask: np.ndarray) -> None:
        total = total_tokens[mask]
        assistant = assistant_tokens[mask]
        if len(total) == 0:
            print(f"  {label}: 0 rows")
            return
        capped_total = np.minimum(total, 4096)
        capped_assistant = np.where(
            total > 4096,
            (assistant * (4096 / total)).astype(np.int64),
            assistant,
        )
        raw_rate = assistant.sum() / total.sum()
        capped_rate = capped_assistant.sum() / capped_total.sum()
        print(
            f"  {label:38s}: {len(total):>9,} | "
            f"min={total.min():>6,} mean={total.mean():>8,.0f} "
            f"med={np.median(total):>7,.0f} P95={np.percentile(total, 95):>8,.0f} "
            f"max={total.max():>8,} | raw={total.sum()/1e9:.3f}B/"
            f"{assistant.sum()/1e9:.3f}B ({raw_rate:.1%}) | "
            f"cap@4096={capped_total.sum()/1e9:.3f}B/"
            f"{capped_assistant.sum()/1e9:.3f}B ({capped_rate:.1%})"
        )

    print("\nIF dataset statistics")
    for source in sorted(set(sources.tolist())):
        print_one(source, sources == source)
    print_one("TOTAL", np.ones(len(dataset), dtype=bool))


def build_if_dataset(
    data_root: str,
    output_dir: str,
    model_name: str,
    num_proc: int,
    cascade2_mode: str,
    source_path_overrides: list[str] | None = None,
) -> None:
    started = time.time()
    source_paths = resolve_source_paths(data_root, source_path_overrides)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    cascade1 = _load_arrow(require_source_path(source_paths, "cascade1_if"))
    cascade2 = _load_arrow(require_source_path(source_paths, "cascade2_if"))
    chat_v1 = _load_arrow(require_source_path(source_paths, "chat_v1"))

    if "_split" not in chat_v1.column_names:
        raise ValueError(
            "Chat-v1 Arrow input must contain the '_split' column created by "
            "scripts/download_nemotron_chat.py so chat_if and "
            "structured_outputs remain distinguishable."
        )
    if "capability_target" not in chat_v1.column_names:
        raise ValueError("Chat-v1 Arrow input is missing 'capability_target'")

    print("\nSelecting public IF subsets")
    print(f"  Cascade-1 dedicated IF config: {len(cascade1):,}")
    if cascade2_mode == "raw_open":
        print(
            f"  Cascade-2 raw public IF config: {len(cascade2):,} "
            "(OPEN APPROXIMATION: NVIDIA 362K constraint selector is not applied)"
        )
        cascade2_source = "cascade2_if_raw_open"
    elif cascade2_mode == "prefiltered":
        print(
            f"  Cascade-2 prefiltered IF input: {len(cascade2):,} "
            "(caller asserts constraint selection has already been applied)"
        )
        cascade2_source = "cascade2_if_constraint_filtered"
    else:
        raise ValueError(f"Unsupported Cascade-2 mode: {cascade2_mode}")

    chat_if = _filter_dataset(
        chat_v1,
        lambda row: row["_split"] == "chat_if"
        and row["capability_target"] == "instruction_following",
        "Chat-v1 capability_target=instruction_following",
        num_proc,
    )
    structured_outputs = _filter_dataset(
        chat_v1,
        lambda row: row["_split"] == "structured_outputs",
        "Chat-v1 structured_outputs",
        num_proc,
    )

    selected_sources = [
        (cascade1, "cascade1_stage2_if"),
        (cascade2, cascade2_source),
        (chat_if, "chat_v1_instruction_following"),
        (structured_outputs, "chat_v1_structured_outputs"),
    ]

    normalized_parts: list[Dataset] = []
    for selected, source in selected_sources:
        processed, total_ratio, assistant_ratio = process_source(
            selected, source, tokenizer, num_proc
        )
        normalized_parts.append(
            _add_estimated_tokens(processed, total_ratio, assistant_ratio)
        )

    combined = concatenate_datasets(normalized_parts)
    before_dedup = len(combined)
    combined, removed_duplicates = deduplicate_messages(combined)
    print(
        f"\nExact normalized-message dedup: {before_dedup:,} -> {len(combined):,} "
        f"(removed {removed_duplicates:,})"
    )

    combined = combined.remove_columns(["total_chars", "assistant_chars"])
    combined = combined.select_columns(OUTPUT_COLUMNS)
    combined = combined.shuffle(seed=42)

    if os.path.exists(output_dir):
        raise FileExistsError(
            f"Output already exists: {output_dir}. Choose a new path or move the "
            "existing dataset explicitly; this builder never overwrites data."
        )
    os.makedirs(os.path.dirname(os.path.abspath(output_dir)), exist_ok=True)
    print(f"Saving {len(combined):,} rows to {output_dir}")
    combined.save_to_disk(output_dir)

    report_stats(combined)
    print(f"Completed in {(time.time() - started) / 60:.1f} minutes")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the normalized, deduplicated IF Arrow training pool."
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Root containing the Arrow source directories documented in dataset.md",
    )
    parser.add_argument(
        "--source_path",
        action="append",
        default=[],
        metavar="KEY=PATH",
        help=(
            "Override one source directory; may be repeated. Keys: "
            + ", ".join(sorted(SOURCE_RELATIVE_PATHS))
        ),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument(
        "--cascade2_mode",
        choices=["raw_open", "prefiltered"],
        default="raw_open",
        help=(
            "raw_open uses the complete public 820K IF config and is not an exact "
            "paper reproduction; prefiltered asserts --source_path cascade2_if=... "
            "already contains the NVIDIA constraint-filtered selection"
        ),
    )
    args = parser.parse_args()
    build_if_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_proc=args.num_proc,
        cascade2_mode=args.cascade2_mode,
        source_path_overrides=args.source_path,
    )


if __name__ == "__main__":
    main()
