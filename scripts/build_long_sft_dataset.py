#!/usr/bin/env python3
"""
Build curated long-SFT datasets (>=3K tokens, >=7K tokens) from 7 source datasets.

Strategy:
  1. Preprocess all sources (reasoning merge, format unification) using dataset.map
  2. Compute char counts (total_chars, assistant_chars) -- instant
  3. Apply generous char-based pre-filter
  4. Tokenize a calibration sample per source to get exact char/token ratios
  5. Estimate token counts for all samples using calibrated ratios
  6. Filter by estimated token threshold
  7. Save with estimated token counts

The HybridDiffusion experiments use the 3K variant. The 7K threshold is retained as a
historical/ablation option.

Expected input layout and a complete command are documented in
``scripts/dataset.md``.
"""

import argparse
import json
import os
import re
import time
from functools import partial

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import numpy as np
from datasets import Dataset, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer


THINK_TAG_RE = re.compile(r"</?think>")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def _load_arrow(path: str) -> Dataset:
    if os.path.isfile(os.path.join(path, "dataset_info.json")) or os.path.isfile(
        os.path.join(path, "state.json")
    ):
        return load_from_disk(path)
    if os.path.isfile(os.path.join(path, "dataset_dict.json")):
        dd = load_from_disk(path)
        return concatenate_datasets(list(dd.values()))
    subdirs = sorted(
        d
        for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
        and (
            os.path.isfile(os.path.join(path, d, "dataset_info.json"))
            or os.path.isfile(os.path.join(path, d, "state.json"))
        )
    )
    if not subdirs:
        raise FileNotFoundError(f"No valid Arrow dataset found at {path}")
    return concatenate_datasets(
        [load_from_disk(os.path.join(path, d)) for d in subdirs]
    )


# ---------------------------------------------------------------------------
# Preprocessing: message unification
# ---------------------------------------------------------------------------
def _merge_reasoning_and_clean(sample, category: str, source: str):
    msgs = sample.get("messages", [])
    if not msgs:
        return {"messages_json": "", "category": category, "source": source,
                "total_chars": 0, "assistant_chars": 0}
    cleaned = []
    total_c = 0
    asst_c = 0
    for m in msgs:
        role = m.get("role", "")
        content = m.get("content", "") or ""
        reasoning = m.get("reasoning_content")
        if role == "assistant" and reasoning:
            r = reasoning.strip("\n")
            c = content.lstrip("\n")
            content = f"<think>\n{r}\n</think>\n\n{c}"
        cleaned.append({"role": role, "content": content})
        total_c += len(content)
        if role == "assistant":
            asst_c += len(content)
    return {
        "messages_json": json.dumps(cleaned, ensure_ascii=False),
        "category": category,
        "source": source,
        "total_chars": total_c,
        "assistant_chars": asst_c,
    }


def _convert_llama_nemotron(sample, category: str, source: str):
    inp = sample.get("input", [])
    out = sample.get("output", "")
    msgs = [{"role": m["role"], "content": m["content"]} for m in inp]
    msgs.append({"role": "assistant", "content": out})
    total_c = sum(len(m["content"]) for m in msgs)
    asst_c = len(out)
    return {
        "messages_json": json.dumps(msgs, ensure_ascii=False),
        "category": category,
        "source": source,
        "total_chars": total_c,
        "assistant_chars": asst_c,
    }


def _convert_post_training_v2(sample):
    cat = sample.get("category", "")
    if cat not in {"stem", "chat", "math", "code"}:
        return {"messages_json": "", "category": "", "source": "",
                "total_chars": 0, "assistant_chars": 0}
    msgs = sample.get("messages", [])
    if not msgs:
        return {"messages_json": "", "category": "", "source": "",
                "total_chars": 0, "assistant_chars": 0}
    cleaned = []
    total_c = 0
    asst_c = 0
    for m in msgs:
        content = m.get("content", "") or ""
        role = m.get("role", "")
        cleaned.append({"role": role, "content": content})
        total_c += len(content)
        if role == "assistant":
            asst_c += len(content)
    return {
        "messages_json": json.dumps(cleaned, ensure_ascii=False),
        "category": cat,
        "source": f"post_training_v2_{cat}",
        "total_chars": total_c,
        "assistant_chars": asst_c,
    }


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


def _messages_have_balanced_think(messages_json: str) -> bool:
    """Require balanced think tags in every assistant message."""
    if not messages_json:
        return False
    messages = json.loads(messages_json)
    return all(
        m.get("role") != "assistant"
        or _think_tags_balanced(m.get("content", ""))
        for m in messages
    )


# ---------------------------------------------------------------------------
# Calibration: tokenize a sample to learn char/token ratio per source
# ---------------------------------------------------------------------------
def calibrate_ratios(
    ds: Dataset,
    tok: AutoTokenizer,
    n_samples: int = 3000,
) -> tuple[float, float]:
    """Tokenize a random sample from ds, return (chars_per_total_token, chars_per_asst_token)."""
    indices = np.random.default_rng(42).choice(
        len(ds), size=min(n_samples, len(ds)), replace=False
    )
    subset = ds.select(indices.tolist())

    all_contents = []
    asst_contents = []
    total_chars_list = []
    asst_chars_list = []
    for mj, tc, ac in zip(
        subset["messages_json"], subset["total_chars"], subset["assistant_chars"]
    ):
        msgs = json.loads(mj)
        all_contents.append("\n".join(m["content"] for m in msgs))
        asst_contents.append(
            "\n".join(m["content"] for m in msgs if m["role"] == "assistant")
        )
        total_chars_list.append(tc)
        asst_chars_list.append(ac)

    TEMPLATE_OVERHEAD = 15

    enc_all = tok(all_contents, add_special_tokens=False, padding=False, truncation=False)
    total_tokens = np.array([len(ids) + TEMPLATE_OVERHEAD for ids in enc_all.input_ids])

    enc_asst = tok(asst_contents, add_special_tokens=False, padding=False, truncation=False)
    asst_tokens = np.array([len(ids) for ids in enc_asst.input_ids])

    total_chars = np.array(total_chars_list, dtype=np.float64)
    asst_chars = np.array(asst_chars_list, dtype=np.float64)

    total_ratio = np.median(total_chars / np.maximum(total_tokens, 1))
    asst_ratio = np.median(asst_chars / np.maximum(asst_tokens, 1))

    return float(total_ratio), float(asst_ratio)


# ---------------------------------------------------------------------------
# Per-source pipeline
# ---------------------------------------------------------------------------
def process_single_source(
    ds: Dataset,
    preprocess_fn,
    label: str,
    char_threshold: int,
    tok: AutoTokenizer,
    num_proc: int,
) -> tuple[Dataset | None, float, float]:
    """Preprocess, char-filter, calibrate, return (ds, total_ratio, asst_ratio)."""
    t0 = time.time()
    print(f"\n  [{label}] Starting: {len(ds):,} rows")

    processed = ds.map(
        preprocess_fn, remove_columns=ds.column_names, num_proc=num_proc
    )
    processed = processed.filter(
        lambda x: x["total_chars"] > 0, num_proc=num_proc
    )
    n_pp = len(processed)
    print(f"    After preprocess: {n_pp:,} ({time.time()-t0:.1f}s)")
    if n_pp == 0:
        return None, 3.0, 3.0

    t1 = time.time()
    before_think_filter = len(processed)
    processed = processed.filter(
        lambda row: _messages_have_balanced_think(row["messages_json"]),
        num_proc=num_proc,
    )
    print(
        f"    After think-tag filter: {len(processed):,} "
        f"(removed {before_think_filter - len(processed):,}, "
        f"{time.time()-t1:.1f}s)"
    )
    if len(processed) == 0:
        return None, 3.0, 3.0

    t1 = time.time()
    processed = processed.filter(
        lambda x: x["total_chars"] >= char_threshold, num_proc=num_proc
    )
    n_char = len(processed)
    print(f"    After char filter (>={char_threshold}): {n_char:,} ({time.time()-t1:.1f}s)")
    if n_char == 0:
        return None, 3.0, 3.0

    t2 = time.time()
    total_r, asst_r = calibrate_ratios(processed, tok, n_samples=3000)
    print(
        f"    Calibrated ratios: total={total_r:.2f} chars/tok, asst={asst_r:.2f} chars/tok"
        f" ({time.time()-t2:.1f}s)"
    )
    print(f"    Total: {time.time()-t0:.1f}s")

    return processed, total_r, asst_r


# ---------------------------------------------------------------------------
# Source definitions
# ---------------------------------------------------------------------------
LLAMA_NEMOTRON_SPLITS = {"math": "math", "code": "code", "science": "stem", "chat": "chat"}

DATASETS_WITH_REASONING = {
    "nemotron_chat_v1": {
        "path_key": "chat_v1",
        "category": "chat",
    },
    "nemotron_chat_v2": {
        "path_key": "chat_v2",
        "category": "chat",
    },
    "nemotron_science_v1": {
        "path_key": "science_v1",
        "category": "stem",
    },
    "nemotron_math_proofs_v1": {
        "path_key": "math_proofs_v1",
        "category": "math",
    },
    "nemotron_compprog_v2": {
        "path_key": "compprog_v2",
        "category": "code",
    },
}

SOURCE_RELATIVE_PATHS = {
    "llama_nemotron": "Llama-Nemotron-Post-Training-Dataset-arrow/SFT",
    "chat_v1": "Nemotron-Instruction-Following-Chat-v1",
    "chat_v2": "Nemotron-SFT-Instruction-Following-Chat-v2",
    "science_v1": "Nemotron-Science-v1",
    "math_proofs_v1": "Nemotron-Math-Proofs-v1",
    "compprog_v2": "Nemotron-SFT-Competitive-Programming-v2",
    "post_training_v2": "Nemotron-Post-Training-Dataset-v2-arrow",
}


def resolve_source_paths(
    data_root: str, overrides: list[str] | None = None
) -> dict[str, str]:
    """Resolve sources below data_root, with optional ``KEY=PATH`` overrides."""
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


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def report_stats(dataset: Dataset, threshold: int):
    cap = 4096 if threshold <= 3000 else 8192
    total_arr = np.array(dataset["total_tokens"])
    asst_arr = np.array(dataset["assistant_tokens"])
    cats = dataset["category"]
    sources = dataset["source"]

    def _ps(label, mask):
        t = total_arr[mask]
        a = asst_arr[mask]
        if len(t) == 0:
            print(f"  {label}: 0 samples")
            return
        ct = np.minimum(t, cap)
        ca = np.where(t > cap, (a * (cap / t)).astype(int), a)
        rr = a.sum() / t.sum() if t.sum() > 0 else 0
        cr = ca.sum() / ct.sum() if ct.sum() > 0 else 0
        print(
            f"  {label:30s}: {len(t):>10,} | "
            f"min={t.min():>6,} mean={t.mean():>8,.0f} med={np.median(t):>8,.0f} "
            f"P95={np.percentile(t,95):>8,.0f} max={t.max():>8,} | "
            f"asst_mean={a.mean():>8,.0f} raw_rate={rr:.1%} | "
            f"cap@{cap}: tot={ct.sum()/1e9:.3f}B asst={ca.sum()/1e9:.3f}B rate={cr:.1%}"
        )

    print(f"\n{'='*130}")
    print(f"Statistics for Long-SFT-{threshold // 1000}K (cap={cap})")
    print(f"{'='*130}")
    print("\n--- By Category ---")
    for cat in sorted(set(cats)):
        _ps(cat, np.array([c == cat for c in cats]))
    _ps("TOTAL", np.ones(len(total_arr), dtype=bool))
    print("\n--- By Source ---")
    for src in sorted(set(sources)):
        _ps(src, np.array([s == src for s in sources]))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_dataset(
    threshold: int,
    output_dir: str,
    model_name: str,
    num_proc: int,
    data_root: str,
    source_path_overrides: list[str] | None = None,
):
    char_threshold = int(threshold * 2.0)
    t_start = time.time()
    print(f"\n{'='*80}")
    print(f"Building Long-SFT dataset: threshold >= {threshold} tokens")
    print(f"  char pre-filter: >= {char_threshold} content chars  |  workers: {num_proc}")
    print(f"  output: {output_dir}")
    print(f"{'='*80}")

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    source_paths = resolve_source_paths(data_root, source_path_overrides)

    source_data: list[tuple[Dataset, float, float]] = []

    print(f"\n[1/3] Datasets with reasoning_content")
    for name, cfg in DATASETS_WITH_REASONING.items():
        ds = _load_arrow(source_paths[cfg["path_key"]])
        fn = partial(_merge_reasoning_and_clean, category=cfg["category"], source=name)
        result, tr, ar = process_single_source(ds, fn, name, char_threshold, tok, num_proc)
        if result:
            source_data.append((result, tr, ar))

    print(f"\n[2/3] Post-training v2 (EN)")
    ds = _load_arrow(source_paths["post_training_v2"])
    result, tr, ar = process_single_source(
        ds, _convert_post_training_v2, "post_training_v2", char_threshold, tok, num_proc
    )
    if result:
        source_data.append((result, tr, ar))

    print(f"\n[3/3] Llama-Nemotron")
    dd = load_from_disk(source_paths["llama_nemotron"])
    for split, cat in LLAMA_NEMOTRON_SPLITS.items():
        if split not in dd:
            continue
        src = f"llama_nemotron_{split}"
        fn = partial(_convert_llama_nemotron, category=cat, source=src)
        result, tr, ar = process_single_source(dd[split], fn, src, char_threshold, tok, num_proc)
        if result:
            source_data.append((result, tr, ar))

    if not source_data:
        print("\nNo qualifying data.")
        return

    # --- Estimate token counts and filter ---
    print(f"\nEstimating token counts using calibrated ratios ...")
    final_parts = []
    for part_ds, total_ratio, asst_ratio in source_data:
        tc = np.array(part_ds["total_chars"], dtype=np.float64)
        ac = np.array(part_ds["assistant_chars"], dtype=np.float64)
        est_total = np.round(tc / total_ratio + 15).astype(np.int32)
        est_asst = np.round(ac / asst_ratio).astype(np.int32)

        mask = est_total >= threshold
        keep = np.where(mask)[0].tolist()
        if not keep:
            continue

        filtered = part_ds.select(keep)
        filtered = filtered.add_column("total_tokens", est_total[mask].tolist())
        filtered = filtered.add_column("assistant_tokens", est_asst[mask].tolist())
        final_parts.append(filtered)
        src_name = filtered[0]["source"]
        print(f"  {src_name}: {len(part_ds):,} -> {len(filtered):,} qualify")

    if not final_parts:
        print("No qualifying samples.")
        return

    combined = concatenate_datasets(final_parts)
    print(f"\nTotal qualifying: {len(combined):,}")
    combined = combined.shuffle(seed=42)

    # Drop char columns
    combined = combined.remove_columns(["total_chars", "assistant_chars"])

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving to {output_dir} ...")
    combined.save_to_disk(output_dir)

    report_stats(combined, threshold)
    print(f"Wall time: {(time.time()-t_start)/60:.1f} min")
    print(f"Done. Load with: datasets.load_from_disk('{output_dir}')")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root containing the Arrow source directories listed in scripts/dataset.md",
    )
    p.add_argument(
        "--source_path",
        action="append",
        default=[],
        metavar="KEY=PATH",
        help=(
            "Override one source directory; may be passed multiple times. Keys: "
            + ", ".join(sorted(SOURCE_RELATIVE_PATHS))
        ),
    )
    p.add_argument("--threshold", type=int, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--num_proc", type=int, default=16)
    a = p.parse_args()
    build_dataset(
        a.threshold,
        a.output_dir,
        a.model_name,
        a.num_proc,
        a.data_root,
        a.source_path,
    )


if __name__ == "__main__":
    main()
