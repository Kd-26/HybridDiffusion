#!/usr/bin/env python3
"""Convert Qwen3.5 DCP checkpoints to a local HF-style safetensors directory.

The output keeps the ordinary Qwen3.5 architecture by default so the same
converted checkpoint can be used for both causal and self-spec serving. The
self-spec server switches to the diffusion-model wrapper when launched with
the internal algorithm selector.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader, TensorStorageMetadata


COPY_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.jinja",
    "added_tokens.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
    "preprocessor_config.json",
    "processor_config.json",
]


def tt_to_hf(name: str) -> str | None:
    if name == "tok_embeddings.weight":
        return "model.language_model.embed_tokens.weight"
    if name == "norm.weight":
        return "model.language_model.norm.weight"
    if name == "output.weight":
        return "lm_head.weight"

    parts = name.split(".", 2)
    if len(parts) != 3 or parts[0] != "layers" or not parts[1].isdigit():
        return None

    layer_id = parts[1]
    suffix = parts[2]
    layer_prefix = f"model.language_model.layers.{layer_id}"

    direct_suffixes = {
        "input_layernorm.weight": "input_layernorm.weight",
        "post_attention_layernorm.weight": "post_attention_layernorm.weight",
        "linear_attn.A_log": "linear_attn.A_log",
        "linear_attn.dt_bias": "linear_attn.dt_bias",
        "linear_attn.in_proj_qkv.weight": "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight": "linear_attn.in_proj_z.weight",
        "linear_attn.in_proj_b.weight": "linear_attn.in_proj_b.weight",
        "linear_attn.in_proj_a.weight": "linear_attn.in_proj_a.weight",
        "linear_attn.conv1d.weight": "linear_attn.conv1d.weight",
        "linear_attn.norm.weight": "linear_attn.norm.weight",
        "linear_attn.out_proj.weight": "linear_attn.out_proj.weight",
        "self_attn.q_norm.weight": "self_attn.q_norm.weight",
        "self_attn.k_norm.weight": "self_attn.k_norm.weight",
    }
    if suffix in direct_suffixes:
        return f"{layer_prefix}.{direct_suffixes[suffix]}"

    renamed_suffixes = {
        "self_attn.wq.weight": "self_attn.q_proj.weight",
        "self_attn.wk.weight": "self_attn.k_proj.weight",
        "self_attn.wv.weight": "self_attn.v_proj.weight",
        "self_attn.wo.weight": "self_attn.o_proj.weight",
        "feed_forward.w1.weight": "mlp.gate_proj.weight",
        "feed_forward.w2.weight": "mlp.down_proj.weight",
        "feed_forward.w3.weight": "mlp.up_proj.weight",
    }
    if suffix in renamed_suffixes:
        return f"{layer_prefix}.{renamed_suffixes[suffix]}"

    return None


def copy_template_files(template: Path, output: Path) -> None:
    for fname in COPY_FILES:
        src = template / fname
        if src.exists():
            shutil.copy2(src, output / fname)
    for src in template.glob("*.py"):
        shutil.copy2(src, output / src.name)


def patch_config(output: Path, mode: str) -> None:
    cfg_path = output / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing template config: {cfg_path}")
    cfg = json.loads(cfg_path.read_text())
    if mode == "dllm":
        cfg["architectures"] = ["Qwen3_5DLLMForConditionalGeneration"]
    else:
        cfg["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")


def convert(dcp_path: Path, hf_template: Path, output: Path, mode: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    copy_template_files(hf_template, output)

    reader = FileSystemReader(str(dcp_path))
    metadata = reader.read_metadata()
    load_plan = {}
    key_map = {}
    skipped = 0

    for key, meta in metadata.state_dict_metadata.items():
        if key.startswith(("optimizer", "dataloader", "train_state", "lr_schedulers")):
            continue
        if not isinstance(meta, TensorStorageMetadata):
            continue
        hf_key = tt_to_hf(key)
        if hf_key is None:
            skipped += 1
            continue
        load_plan[key] = torch.empty(meta.size, dtype=meta.properties.dtype, device="cpu")
        key_map[key] = hf_key

    if not load_plan:
        raise RuntimeError(f"no model tensors found in {dcp_path}")

    print(f"loading {len(load_plan)} tensors from {dcp_path} (skipped={skipped})")
    dcp.load(load_plan, storage_reader=reader)
    state_dict = {
        key_map[key]: tensor.to(torch.bfloat16).contiguous()
        for key, tensor in load_plan.items()
    }
    save_file(state_dict, str(output / "model.safetensors"), metadata={"format": "pt"})
    patch_config(output, mode)
    print(f"saved {len(state_dict)} tensors to {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcp-path", required=True)
    parser.add_argument("--hf-template", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["ar", "dllm"], default="ar")
    args = parser.parse_args()
    convert(
        dcp_path=Path(args.dcp_path),
        hf_template=Path(args.hf_template),
        output=Path(args.output),
        mode=args.mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
