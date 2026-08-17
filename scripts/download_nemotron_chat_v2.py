#!/usr/bin/env python3
"""
Download nvidia/Nemotron-SFT-Instruction-Following-Chat-v2 and save as Arrow.

Chat v2 has two splits with incompatible schemas:
  - reasoning_off: messages = [{role, content}]  (some rows may also have reasoning_content)
  - reasoning_on:  messages = [{role, content, reasoning_content}]

This script loads both splits with an explicit unified schema (role, content,
reasoning_content), tags each row with a _split column, then concatenates
into a single flat Arrow directory loadable with datasets.load_from_disk().

Usage:
    python scripts/download_nemotron_chat_v2.py --output_dir /path/to/arrow-sources
"""

import argparse
import os

from datasets import Features, Value, concatenate_datasets, load_dataset


REPO_ID = "nvidia/Nemotron-SFT-Instruction-Following-Chat-v2"
# Explicit features so HuggingFace doesn't fail on schema mismatches
# within a split. The reasoning_on split has reasoning_content inside the
# message struct; reasoning_off may or may not have it. We define the
# superset schema for messages so all rows are handled uniformly.
FEATURES = Features(
    {
        "messages": [
            {
                "role": Value("string"),
                "content": Value("string"),
                "reasoning_content": Value("string"),
            }
        ],
        "uuid": Value("string"),
        "license": Value("string"),
        "used_in": [Value("string")],
        "reasoning": Value("string"),
    }
)


def _report_disk_size(path: str):
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total_size += os.path.getsize(os.path.join(dirpath, f))
    print(f"  Size on disk: {total_size / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Download Nemotron Chat v2 with schema normalization."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root directory in which to save the Arrow dataset",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="Number of processes for .map() and save_to_disk()",
    )
    args = parser.parse_args()
    hf_token = os.environ.get("HF_TOKEN") or None

    dataset_name = REPO_ID.split("/")[-1]
    save_path = os.path.join(args.output_dir, dataset_name)

    print(f"Downloading: {REPO_ID}")
    print(f"Saving to:   {save_path}")

    # -- Load both splits with explicit features to avoid schema mismatches --
    # Using the unified FEATURES schema ensures rows with/without
    # reasoning_content in their message struct are all handled.
    print("\nLoading reasoning_off split...")
    ds_off = load_dataset(
        REPO_ID, split="reasoning_off", token=hf_token, features=FEATURES
    )
    print(f"  reasoning_off: {ds_off.num_rows} rows, columns: {ds_off.column_names}")

    print("\nLoading reasoning_on split...")
    ds_on = load_dataset(
        REPO_ID, split="reasoning_on", token=hf_token, features=FEATURES
    )
    print(f"  reasoning_on:  {ds_on.num_rows} rows, columns: {ds_on.column_names}")

    # -- Tag splits and concatenate --
    ds_off = ds_off.add_column("_split", ["reasoning_off"] * len(ds_off))
    ds_on = ds_on.add_column("_split", ["reasoning_on"] * len(ds_on))

    print("\nConcatenating splits...")
    merged = concatenate_datasets([ds_off, ds_on])
    print(f"  Merged: {merged.num_rows} rows, columns: {merged.column_names}")

    # -- Verify --
    print("\nVerification:")
    sample_off = None
    sample_on = None
    for i in range(len(merged)):
        row = merged[i]
        if row["_split"] == "reasoning_off" and sample_off is None:
            sample_off = row
        if row["_split"] == "reasoning_on" and sample_on is None:
            sample_on = row
        if sample_off and sample_on:
            break

    if sample_off:
        msgs = sample_off["messages"]
        asst = [m for m in msgs if m.get("role") == "assistant"]
        if asst:
            rc = asst[0].get("reasoning_content")
            print(f"  reasoning_off sample: assistant reasoning_content = {repr(rc)} (expected None or empty)")
            assert not rc, "reasoning_off should have reasoning_content=None or empty"

    if sample_on:
        msgs = sample_on["messages"]
        asst = [m for m in msgs if m.get("role") == "assistant"]
        if asst:
            rc = asst[0].get("reasoning_content")
            preview = repr(rc[:80]) if rc else repr(rc)
            print(f"  reasoning_on sample:  assistant reasoning_content = {preview}...")
            assert rc and len(rc) > 0, "reasoning_on should have reasoning_content"

    print("  Schema verification passed.")

    # -- Save --
    print(f"\nSaving ({merged.num_rows} rows) to: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    merged.save_to_disk(save_path, num_proc=args.num_proc)
    print("  Saved successfully.")
    _report_disk_size(save_path)

    print(f"\nDone. Load with: datasets.load_from_disk('{save_path}')")


if __name__ == "__main__":
    main()
