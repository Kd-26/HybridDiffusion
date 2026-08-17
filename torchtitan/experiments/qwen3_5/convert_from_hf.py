#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Convert a HuggingFace Qwen3.5 checkpoint to TorchTitan DCP format.

The state-dict adapter handles the ``model.language_model.`` prefix so this
works with either:
- a language-only export with ``model.*`` or bare text keys
- a full Qwen3.5 checkpoint where the language backbone lives under
  ``model.language_model.*``

Usage (language):
    python -m torchtitan.experiments.qwen3_5.convert_from_hf \
        --input_dir /path/to/Qwen3.5-2B \
        --output_dir /path/to/dcp/qwen3_5_2b \
        --model_name qwen3_5 \
        --model_flavor 2B

The input_dir should contain HF safetensors files (*.safetensors) and
``config.json``. Download them first with:

    python scripts/download_hf_assets.py \\
        --repo_id Qwen/Qwen3.5-2B --assets safetensors tokenizer config
"""

import argparse
import glob
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

_TORCH_VERSION = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
if _TORCH_VERSION >= (2, 10):
    from torch.distributed.checkpoint import HuggingFaceStorageReader
else:
    HuggingFaceStorageReader = None

TORCHTITAN_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(TORCHTITAN_ROOT))

import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.experiments.qwen3_5.hf_utils import (
    list_hf_checkpoint_keys,
    load_hf_config,
    validate_model_args_against_hf_config,
    validate_state_dict_against_reference,
)


def _load_hf_safetensors(input_dir: Path) -> dict[str, torch.Tensor]:
    """Load all safetensors shards from *input_dir* into a flat state dict.

    Works on any PyTorch version -- does not require HuggingFaceStorageReader.
    """
    from safetensors.torch import load_file

    shard_files = sorted(glob.glob(str(input_dir / "*.safetensors")))
    if not shard_files:
        raise FileNotFoundError(f"No *.safetensors files found in {input_dir}")

    print(f"  Loading {len(shard_files)} safetensors shard(s) ...")
    merged: dict[str, torch.Tensor] = {}
    for path in shard_files:
        print(f"    {Path(path).name}")
        merged.update(load_file(path, device="cpu"))
    return merged


@torch.inference_mode()
def convert_from_hf(
    input_dir: Path,
    output_dir: Path,
    model_name: str,
    model_flavor: str,
    skip_config_validation: bool = False,
    skip_hf_key_audit: bool = False,
    skip_state_dict_validation: bool = False,
):
    """Convert HuggingFace Qwen3.5 checkpoint to TorchTitan DCP format."""
    if model_name != "qwen3_5":
        raise ValueError(f"Unsupported model_name: {model_name}")

    print(f"Converting {model_name}/{model_flavor}")
    print(f"  Input  : {input_dir}")
    print(f"  Output : {output_dir}")

    train_spec = train_spec_module.get_train_spec(model_name)
    model_args = train_spec.model_args[model_flavor]

    print(f"  dim={model_args.dim}, layers={model_args.n_layers}, "
          f"vocab={model_args.vocab_size}")

    with torch.device("cpu"):
        model = train_spec.model_cls(model_args)
    model = ModelWrapper(model)
    reference_state_dict = model._get_state_dict()

    sd_adapter = train_spec.state_dict_adapter(model_args, None)
    assert sd_adapter is not None, "state_dict_adapter is required for conversion"

    if not skip_config_validation:
        print("Validating HF config against TorchTitan model args ...")
        hf_config = load_hf_config(input_dir)
        config_issues = validate_model_args_against_hf_config(model_args, hf_config)
        if config_issues:
            formatted_issues = "\n".join(f"    - {issue}" for issue in config_issues)
            raise ValueError(
                "HF config does not match the selected TorchTitan flavor:\n"
                f"{formatted_issues}"
            )
        print("  HF config matches TorchTitan model args")

    if not skip_hf_key_audit:
        print("Auditing HF checkpoint key coverage ...")
        hf_keys = list_hf_checkpoint_keys(input_dir)
        audit_report = sd_adapter.audit_hf_keys(hf_keys)
        print(
            "  Mapped: {mapped}  Skipped: {skipped}  Unmapped: {unmapped}".format(
                mapped=len(audit_report.mapped_keys),
                skipped=len(audit_report.skipped_keys),
                unmapped=len(audit_report.unmapped_keys),
            )
        )
        if audit_report.unmapped_keys:
            sample_keys = sorted(audit_report.unmapped_keys)[:20]
            raise ValueError(
                "Found unmapped HF checkpoint keys:\n"
                + "\n".join(f"    - {key}" for key in sample_keys)
            )

    # Load HF weights -- prefer HuggingFaceStorageReader when available,
    # fall back to loading safetensors directly.
    if HuggingFaceStorageReader is not None:
        hf_state_dict = sd_adapter.to_hf(reference_state_dict)
        print(f"  HF skeleton keys: {len(hf_state_dict)}")
        print("Loading HuggingFace checkpoint (HuggingFaceStorageReader) ...")
        dcp.load(
            hf_state_dict,
            storage_reader=HuggingFaceStorageReader(path=str(input_dir)),
        )
    else:
        print(f"  PyTorch {torch.__version__} < 2.10 -- using safetensors fallback")
        hf_state_dict = _load_hf_safetensors(input_dir)
        print(f"  Loaded {len(hf_state_dict)} HF keys")

    print("Converting to TorchTitan format ...")
    state_dict = sd_adapter.from_hf(hf_state_dict)
    print(f"  TorchTitan state dict keys: {len(state_dict)}")

    if not skip_state_dict_validation:
        print("Validating converted TorchTitan state dict ...")
        state_dict_issues = validate_state_dict_against_reference(
            reference_state_dict,
            state_dict,
            ignore_keys={"rope_cache", "expert_bias", "tokens_per_expert"},
        )
        if state_dict_issues:
            formatted_issues = "\n".join(f"    - {issue}" for issue in state_dict_issues)
            raise ValueError(
                "Converted TorchTitan state dict failed validation:\n"
                f"{formatted_issues}"
            )
        print("  Converted state dict matches the TorchTitan model schema")

    print(f"Saving DCP checkpoint to {output_dir} ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    dcp.save(state_dict, checkpoint_id=str(output_dir))

    print(f"Done!  Checkpoint saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace Qwen3.5 checkpoint to TorchTitan DCP."
    )
    parser.add_argument(
        "--input_dir", type=Path, required=True,
        help="Directory with HF safetensors (e.g. Qwen/Qwen3.5-2B)",
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Output directory for TorchTitan DCP checkpoint",
    )
    parser.add_argument(
        "--model_name", type=str, default="qwen3_5",
        choices=["qwen3_5"],
        help="Text-only Qwen3.5 model name.",
    )
    parser.add_argument(
        "--model_flavor", type=str, default="2B",
        help="Model flavor/size (default: 2B)",
    )
    parser.add_argument(
        "--skip_config_validation",
        action="store_true",
        help="Skip validating config.json against the selected TorchTitan flavor",
    )
    parser.add_argument(
        "--skip_hf_key_audit",
        action="store_true",
        help="Skip auditing HF checkpoint keys for unmapped patterns",
    )
    parser.add_argument(
        "--skip_state_dict_validation",
        action="store_true",
        help="Skip validating converted TorchTitan keys and tensor shapes",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: input directory does not exist: {args.input_dir}")
        sys.exit(1)

    convert_from_hf(
        args.input_dir,
        args.output_dir,
        args.model_name,
        args.model_flavor,
        skip_config_validation=args.skip_config_validation,
        skip_hf_key_audit=args.skip_hf_key_audit,
        skip_state_dict_validation=args.skip_state_dict_validation,
    )


if __name__ == "__main__":
    main()
