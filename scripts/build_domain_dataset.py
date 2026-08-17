#!/usr/bin/env python3
"""
Build per-domain SFT datasets from multiple Nemotron sources.

Produces Arrow datasets in the same format as Long-SFT-3K:
  messages_json, category, source, total_tokens, assistant_tokens

Each message in messages_json has exactly {role, content} — reasoning_content
is inlined as <think>...</think> during preprocessing.

Usage:
    python scripts/build_domain_dataset.py \
        --domain math \
        --data_root /path/to/arrow-sources \
        --output_dir /path/to/Nemotron-Math-Domain \
        --model_name Qwen/Qwen3-1.7B

The HybridDiffusion Mix4 experiment uses the ``math`` output. ``code`` and ``science``
remain available as historical/ablation builders. This snapshot does not
contain the separate builder used for the paper's IF/Cascade pool; see
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
    """For datasets with reasoning_content field (e.g. math_proofs_v1)."""
    msgs = sample.get("messages", [])
    if not msgs:
        return {
            "messages_json": "",
            "category": category,
            "source": source,
            "total_chars": 0,
            "assistant_chars": 0,
        }
    cleaned = []
    total_c = 0
    asst_c = 0
    for m in msgs:
        role = m.get("role", "")
        content = m.get("content", "") or ""
        reasoning = m.get("reasoning_content")
        if role == "assistant":
            if reasoning:
                r = reasoning.strip("\n")
                c = content.lstrip("\n")
                content = f"<think>\n{r}\n</think>\n\n{c}"
            elif "<think>" not in content:
                content = f"<think></think>\n\n{content}"
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
    """For llama_nemotron: input/output → messages. Output already has <think>."""
    inp = sample.get("input", [])
    out = sample.get("output", "")
    if "<think>" not in out:
        out = f"<think></think>\n\n{out}"
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


def _convert_post_training_v2(sample, category: str, source: str):
    """For post_training_v2: messages already have content (may include <think>)."""
    msgs = sample.get("messages", [])
    if not msgs:
        return {
            "messages_json": "",
            "category": category,
            "source": source,
            "total_chars": 0,
            "assistant_chars": 0,
        }
    cleaned = []
    total_c = 0
    asst_c = 0
    for m in msgs:
        role = m.get("role", "")
        content = m.get("content", "") or ""
        if role == "assistant" and "<think>" not in content:
            content = f"<think></think>\n\n{content}"
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


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def _has_unclosed_think(messages_json: str) -> bool:
    """Return True if any assistant message has <think> without </think>."""
    if not messages_json:
        return False
    msgs = json.loads(messages_json)
    for m in msgs:
        if m.get("role") == "assistant":
            c = m.get("content", "")
            if "<think>" in c and "</think>" not in c:
                return True
    return False


# ---------------------------------------------------------------------------
# Calibration: tokenize a sample to learn char/token ratio per source
# ---------------------------------------------------------------------------
def calibrate_ratios(
    ds: Dataset,
    tok: AutoTokenizer,
    n_samples: int = 3000,
) -> tuple[float, float]:
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

    enc_all = tok(
        all_contents, add_special_tokens=False, padding=False, truncation=False
    )
    total_tokens = np.array(
        [len(ids) + TEMPLATE_OVERHEAD for ids in enc_all.input_ids]
    )

    enc_asst = tok(
        asst_contents, add_special_tokens=False, padding=False, truncation=False
    )
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
    tok: AutoTokenizer,
    num_proc: int,
) -> tuple[Dataset | None, float, float]:
    t0 = time.time()
    print(f"\n  [{label}] Starting: {len(ds):,} rows")

    processed = ds.map(
        preprocess_fn, remove_columns=ds.column_names, num_proc=num_proc
    )
    processed = processed.filter(lambda x: x["total_chars"] > 0, num_proc=num_proc)
    n_pp = len(processed)
    print(f"    After preprocess (non-empty): {n_pp:,} ({time.time()-t0:.1f}s)")
    if n_pp == 0:
        return None, 3.0, 3.0

    # Filter unclosed think tags
    t1 = time.time()
    n_before = len(processed)
    processed = processed.filter(
        lambda x: not _has_unclosed_think(x["messages_json"]), num_proc=num_proc
    )
    n_filtered = n_before - len(processed)
    print(
        f"    After unclosed-think filter: {len(processed):,} "
        f"(removed {n_filtered:,}, {time.time()-t1:.1f}s)"
    )
    if len(processed) == 0:
        return None, 3.0, 3.0

    t2 = time.time()
    total_r, asst_r = calibrate_ratios(processed, tok, n_samples=3000)
    print(
        f"    Calibrated ratios: total={total_r:.2f} chars/tok, "
        f"asst={asst_r:.2f} chars/tok ({time.time()-t2:.1f}s)"
    )
    print(f"    Total: {time.time()-t0:.1f}s")

    return processed, total_r, asst_r


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def report_stats(dataset: Dataset, domain: str):
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
        for cap in [4096, 8192, None]:
            if cap:
                ct = np.minimum(t, cap)
                ca = np.where(t > cap, (a * (cap / t)).astype(int), a)
                cap_label = f"cap@{cap}"
            else:
                ct = t
                ca = a
                cap_label = "raw"
            rr = ca.sum() / ct.sum() if ct.sum() > 0 else 0
            if cap is None:
                print(
                    f"  {label:35s}: {len(t):>10,} | "
                    f"min={t.min():>6,} mean={t.mean():>8,.0f} "
                    f"med={np.median(t):>8,.0f} "
                    f"P95={np.percentile(t,95):>8,.0f} "
                    f"max={t.max():>8,} | "
                    f"{cap_label}: tot={ct.sum()/1e9:.3f}B "
                    f"asst={ca.sum()/1e9:.3f}B rate={rr:.1%}"
                )
            else:
                print(
                    f"  {'':35s}  {'':>10s}   "
                    f"{'':>6s} {'':>8s} {'':>8s} {'':>8s} {'':>8s} | "
                    f"{cap_label}: tot={ct.sum()/1e9:.3f}B "
                    f"asst={ca.sum()/1e9:.3f}B rate={rr:.1%}"
                )

    print(f"\n{'='*120}")
    print(f"Statistics for Domain: {domain}")
    print(f"{'='*120}")
    print("\n--- By Source ---")
    for src in sorted(set(sources)):
        _ps(src, np.array([s == src for s in sources]))
    print("\n--- Total ---")
    _ps("ALL", np.ones(len(total_arr), dtype=bool))
    print()


# ---------------------------------------------------------------------------
# Domain definitions
# ---------------------------------------------------------------------------
DOMAIN_SOURCES = {
    "math": {
        "llama_nemotron_math": {
            "path_key": "llama_nemotron",
            "split": "math",
            "filter_fn": lambda sample: sample.get("reasoning") == "on",
            "preprocess": "_convert_llama_nemotron",
            "category": "math",
        },
        "nemotron_math_proofs_v1": {
            "path_key": "math_proofs_v1",
            "preprocess": "_merge_reasoning_and_clean",
            "category": "math",
        },
    },
    "code": {
        "llama_nemotron_code": {
            "path_key": "llama_nemotron",
            "split": "code",
            "filter_fn": lambda sample: sample.get("reasoning") == "on",
            "preprocess": "_convert_llama_nemotron",
            "category": "code",
        },
        "compprog_v2_python": {
            "path_key": "compprog_v2",
            "filter_fn": lambda sample: (sample.get("_split") or "").startswith("python"),
            "preprocess": "_merge_reasoning_and_clean",
            "category": "code",
        },
    },
    "science": {
        "llama_nemotron_science": {
            "path_key": "llama_nemotron",
            "split": "science",
            "preprocess": "_convert_llama_nemotron",
            "category": "stem",
        },
        "nemotron_science_v1": {
            "path_key": "science_v1",
            "preprocess": "_merge_reasoning_and_clean",
            "category": "stem",
        },
    },
}

SOURCE_RELATIVE_PATHS = {
    "llama_nemotron": "Llama-Nemotron-Post-Training-Dataset-arrow/SFT",
    "math_proofs_v1": "Nemotron-Math-Proofs-v1",
    "compprog_v2": "Nemotron-SFT-Competitive-Programming-v2",
    "science_v1": "Nemotron-Science-v1",
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


def require_source_path(source_paths: dict[str, str], path_key: str) -> str:
    """Resolve an Arrow source below an explicit, user-owned data root."""
    path = source_paths[path_key]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing source '{path_key}' at {path}. "
            "Download/build it first; see scripts/dataset.md."
        )
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_domain_dataset(
    domain: str,
    output_dir: str,
    model_name: str,
    num_proc: int,
    data_root: str,
    source_path_overrides: list[str] | None = None,
):
    if domain not in DOMAIN_SOURCES:
        raise ValueError(
            f"Unknown domain '{domain}'. Supported: {list(DOMAIN_SOURCES.keys())}"
        )

    t_start = time.time()
    sources_cfg = DOMAIN_SOURCES[domain]

    print(f"\n{'='*80}")
    print(f"Building Domain Dataset: {domain}")
    print(f"  Sources: {list(sources_cfg.keys())}")
    print(f"  Output:  {output_dir}")
    print(f"  Workers: {num_proc}")
    print(f"{'='*80}")

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    preprocess_fns = {
        "_convert_llama_nemotron": _convert_llama_nemotron,
        "_merge_reasoning_and_clean": _merge_reasoning_and_clean,
        "_convert_post_training_v2": _convert_post_training_v2,
    }
    source_paths = resolve_source_paths(data_root, source_path_overrides)

    source_data: list[tuple[Dataset, float, float]] = []

    for source_name, cfg in sources_cfg.items():
        local_path = require_source_path(source_paths, cfg["path_key"])

        if "split" in cfg:
            dd = load_from_disk(local_path)
            ds = dd[cfg["split"]]
        else:
            ds = _load_arrow(local_path)

        print(f"\n  Loaded {source_name}: {len(ds):,} rows")

        # Pre-filter if specified
        if "filter_fn" in cfg:
            n_before = len(ds)
            ds = ds.filter(cfg["filter_fn"], num_proc=num_proc)
            print(f"    Pre-filter: {n_before:,} -> {len(ds):,}")

        fn_name = cfg["preprocess"]
        fn = partial(
            preprocess_fns[fn_name], category=cfg["category"], source=source_name
        )

        result, tr, ar = process_single_source(ds, fn, source_name, tok, num_proc)
        if result:
            source_data.append((result, tr, ar))

    if not source_data:
        print("\nNo qualifying data.")
        return

    # Estimate token counts
    print(f"\nEstimating token counts ...")
    final_parts = []
    for part_ds, total_ratio, asst_ratio in source_data:
        tc = np.array(part_ds["total_chars"], dtype=np.float64)
        ac = np.array(part_ds["assistant_chars"], dtype=np.float64)
        est_total = np.round(tc / total_ratio + 15).astype(np.int32)
        est_asst = np.round(ac / asst_ratio).astype(np.int32)

        filtered = part_ds.add_column("total_tokens", est_total.tolist())
        filtered = filtered.add_column("assistant_tokens", est_asst.tolist())
        final_parts.append(filtered)
        src_name = filtered[0]["source"]
        print(f"  {src_name}: {len(filtered):,} samples")

    combined = concatenate_datasets(final_parts)
    print(f"\nTotal: {len(combined):,}")
    combined = combined.shuffle(seed=42)

    # Drop char columns
    combined = combined.remove_columns(["total_chars", "assistant_chars"])

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving to {output_dir} ...")
    combined.save_to_disk(output_dir)

    report_stats(combined, domain)
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
    p.add_argument("--domain", type=str, required=True, choices=sorted(DOMAIN_SOURCES.keys()))
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--num_proc", type=int, default=16)
    a = p.parse_args()
    build_domain_dataset(
        a.domain,
        a.output_dir,
        a.model_name,
        a.num_proc,
        a.data_root,
        a.source_path,
    )


if __name__ == "__main__":
    main()
