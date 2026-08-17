#!/usr/bin/env python3
"""
Download HuggingFace datasets and save to disk in Arrow format.

Handles datasets with or without named configs automatically:
  - Datasets with a single default config (e.g. nvidia/Nemotron-Science-v1)
    → all splits concatenated into one flat Arrow dir
  - Datasets with multiple named configs (e.g. tokyotech-llm/swallow-math-v2)
    → each config saved as its own flat Arrow dir (splits concatenated within)

Every output directory contains dataset_info.json at the top level and is
directly loadable with datasets.load_from_disk().

Usage:
    # Download all default Nemotron datasets at once:
    python scripts/download_nemotron_chat.py \
        --output_dir /path/to/arrow-sources

    # Download a specific dataset:
    python scripts/download_nemotron_chat.py \
        --output_dir /path/to/arrow-sources \
        --repo_ids nvidia/Nemotron-Science-v1

    # Download a multi-config dataset (each config saved separately):
    python scripts/download_nemotron_chat.py \
        --output_dir /path/to/arrow-sources \
        --repo_ids tokyotech-llm/swallow-math-v2

    # Download only specific configs:
    python scripts/download_nemotron_chat.py \
        --output_dir /path/to/arrow-sources \
        --repo_ids tokyotech-llm/swallow-math-v2 \
        --configs swallow-math-v2-qa

    # Download only specific splits:
    python scripts/download_nemotron_chat.py \
        --output_dir /path/to/arrow-sources \
        --repo_ids nvidia/Nemotron-Instruction-Following-Chat-v1 \
        --splits chat_if

Output structure:
    Single-config datasets (splits concatenated into one flat Arrow dir):
        <output_dir>/
            Nemotron-Instruction-Following-Chat-v1/
                dataset_info.json    ← load_from_disk() here
                data-*.arrow
            Nemotron-Science-v1/
                dataset_info.json
                data-*.arrow

    Multi-config datasets (one flat Arrow dir per config):
        <output_dir>/
            swallow-math-v2/
                swallow-math-v2-qa/
                    dataset_info.json    ← load_from_disk() here
                    data-*.arrow
                swallow-math-v2-textbook/
                    dataset_info.json    ← or load_from_disk() here
                    data-*.arrow

Training usage (toml):
    # Single-config dataset:
    dataset_path = "${DATA_ROOT}/Nemotron-Science-v1"

    # Multi-config dataset — pick the config you want:
    dataset_path = "${DATA_ROOT}/swallow-math-v2/swallow-math-v2-qa"
"""

import argparse
import os

from datasets import (
    Dataset,
    concatenate_datasets,
    get_dataset_config_names,
    load_dataset,
)


DEFAULT_REPO_IDS = [
    "nvidia/Nemotron-Instruction-Following-Chat-v1",
    "nvidia/Nemotron-Science-v1",
    "nvidia/Nemotron-Math-Proofs-v1",
]


def _report_disk_size(path: str):
    """Print total size of files under path."""
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total_size += os.path.getsize(os.path.join(dirpath, f))
    print(f"  Size on disk: {total_size / (1024**3):.2f} GB")


def _load_and_concat_splits(
    repo_id: str,
    config_name: str | None,
    splits: list[str] | None,
    hf_token: str | None,
) -> Dataset:
    """
    Load one config of a dataset, concatenate its splits, and return a single
    Dataset. A '_split' column is added so the source split is preserved.
    """
    label = config_name or "default"
    print(f"\n{'-'*60}")
    print(f"Loading config: {label}")
    print(f"{'-'*60}")

    ds = load_dataset(repo_id, name=config_name, token=hf_token)

    # Print dataset info
    print(f"  Structure: {ds}")
    for split_name, split_ds in ds.items():
        print(
            f"    Split '{split_name}': {split_ds.num_rows} rows, "
            f"columns: {split_ds.column_names}"
        )

    # Filter to requested splits
    if splits is not None:
        available_splits = list(ds.keys())
        for s in splits:
            if s not in available_splits:
                raise ValueError(
                    f"Split '{s}' not found in {repo_id} (config={label}). "
                    f"Available: {available_splits}"
                )
        ds = {s: ds[s] for s in splits}
    else:
        ds = dict(ds)

    # Tag each split and concatenate
    tagged: list[Dataset] = []
    for split_name, split_ds in ds.items():
        split_ds = split_ds.add_column("_split", [split_name] * len(split_ds))
        tagged.append(split_ds)

    if len(tagged) == 1:
        return tagged[0]

    merged = concatenate_datasets(tagged)
    print(f"  Concatenated {len(tagged)} splits → {merged.num_rows} total rows")
    return merged


def _save_dataset(ds: Dataset, save_path: str, num_proc: int | None = None):
    """Save a Dataset as Arrow files and report size."""
    os.makedirs(save_path, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Saving ({ds.num_rows} rows, columns: {ds.column_names}) to: {save_path}")
    ds.save_to_disk(save_path, num_proc=num_proc)
    print(f"  Saved successfully.")
    _report_disk_size(save_path)


def download_and_save_one(
    repo_id: str,
    output_dir: str,
    configs: list[str] | None = None,
    splits: list[str] | None = None,
    hf_token: str | None = None,
    num_proc: int | None = None,
):
    """
    Download a single HuggingFace dataset and save as Arrow files on disk.

    - Single-config datasets: all splits concatenated → <dataset_name>/
    - Multi-config datasets: each config gets its own flat dir →
      <dataset_name>/<config_name>/

    Every leaf directory is independently loadable with load_from_disk().

    Args:
        repo_id: HuggingFace dataset repo (e.g. "nvidia/Nemotron-Science-v1").
        output_dir: Root directory for saving.
        configs: Which configs to download. None = all available configs.
        splits: Which splits to download. None = all splits.
        hf_token: Optional HuggingFace token for authentication.
        num_proc: Number of processes for saving (None = single process).
    """
    dataset_name = repo_id.split("/")[-1]
    dataset_dir = os.path.join(output_dir, dataset_name)

    print(f"\n{'#'*60}")
    print(f"# Downloading: {repo_id}")
    print(f"# Saving to:   {dataset_dir}")
    print(f"{'#'*60}")

    # Discover available configs
    available_configs = get_dataset_config_names(repo_id, token=hf_token)
    has_named_configs = available_configs != ["default"]

    if has_named_configs:
        print(f"\nDataset has named configs: {available_configs}")
    else:
        print(f"\nDataset has a single default config.")

    # Determine which configs to iterate over
    if has_named_configs:
        target_configs = configs if configs is not None else available_configs
        if configs is not None:
            for c in configs:
                if c not in available_configs:
                    raise ValueError(
                        f"Config '{c}' not found in {repo_id}. "
                        f"Available: {available_configs}"
                    )
    else:
        target_configs = [None]

    # Process each config
    for config_name in target_configs:
        merged = _load_and_concat_splits(repo_id, config_name, splits, hf_token)

        if has_named_configs:
            # Multi-config: save under <dataset_name>/<config_name>/
            save_path = os.path.join(dataset_dir, config_name)
        else:
            # Single config: save directly under <dataset_name>/
            save_path = dataset_dir

        _save_dataset(merged, save_path, num_proc=num_proc)

    # Print summary for multi-config datasets
    if has_named_configs:
        print(f"\nSaved {len(target_configs)} config(s) under {dataset_dir}/")
        print("Use dataset_path pointing to the config dir you want in your toml:")
        for cfg in target_configs:
            print(f"  dataset_path = \"{os.path.join(dataset_dir, cfg)}\"")

    print(f"\nDone with {repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace datasets and save in Arrow format. "
        "Automatically handles datasets with single or multiple configs."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root directory in which to save Arrow datasets",
    )
    parser.add_argument(
        "--repo_ids",
        type=str,
        nargs="+",
        default=None,
        help="HuggingFace dataset repo IDs to download. "
        f"Default: all three ({', '.join(DEFAULT_REPO_IDS)})",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=None,
        help="Configs to download (default: all). For multi-config datasets only.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        help="Splits to download (default: all).",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="Number of processes for saving (default: single process)",
    )

    args = parser.parse_args()

    repo_ids = args.repo_ids if args.repo_ids is not None else DEFAULT_REPO_IDS

    if (args.splits is not None or args.configs is not None) and len(repo_ids) > 1:
        parser.error(
            "--splits and --configs can only be used when downloading a single repo "
            "(--repo_ids with one ID)"
        )

    for repo_id in repo_ids:
        download_and_save_one(
            repo_id=repo_id,
            output_dir=args.output_dir,
            configs=args.configs,
            splits=args.splits,
            hf_token=os.environ.get("HF_TOKEN") or None,
            num_proc=args.num_proc,
        )

    print(f"\n{'#'*60}")
    print(f"All datasets saved to: {args.output_dir}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
