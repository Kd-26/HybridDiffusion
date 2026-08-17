#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Callable

import torch
import torch.nn.functional as F

from torchtitan.experiments.qwen3_5.inference_test import (
    load_model_from_dcp,
    load_tokenizer,
)


def _select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _select_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    return torch.float32


def _prepend_transformers_path(transformers_path: str | None) -> None:
    transformers_path = transformers_path or os.environ.get("QWEN35_TRANSFORMERS_PATH")
    if not transformers_path:
        return
    resolved = str(Path(transformers_path).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _load_hf_model(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    transformers_path: str | None = None,
):
    _prepend_transformers_path(transformers_path)

    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_next.configuration_qwen3_next import (
        Qwen3NextConfig,
    )
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        Qwen3NextForCausalLM,
    )

    load_kwargs: dict[str, object] = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    move_to_device = True
    if device.type == "cuda":
        device_str = f"cuda:{device.index}" if device.index is not None else "cuda:0"
        load_kwargs["device_map"] = {"": device_str}
        move_to_device = False

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            **load_kwargs,
        )
    except ValueError as exc:
        config_path = Path(model_path) / "config.json"
        if "model type `qwen3_5`" not in str(exc) or not config_path.exists():
            raise

        config_dict = json.loads(config_path.read_text())
        text_config = dict(config_dict.get("text_config") or {})
        if not text_config:
            raise ValueError(
                f"{config_path} does not contain a usable text_config for qwen3_5 fallback loading."
            ) from exc

        text_config.setdefault("pad_token_id", config_dict.get("pad_token_id"))
        text_config.setdefault("bos_token_id", config_dict.get("bos_token_id"))
        text_config.setdefault(
            "eos_token_id",
            config_dict.get("eos_token_id", text_config.get("eos_token_id")),
        )

        config = Qwen3NextConfig.from_dict(text_config)
        model = Qwen3NextForCausalLM.from_pretrained(
            model_path,
            config=config,
            **load_kwargs,
        )

    if move_to_device:
        model.to(device)
    model.eval()
    return model


def _compute_logit_metrics(
    hf_logits: torch.Tensor,
    tt_logits: torch.Tensor,
) -> dict[str, float]:
    hf_logits = hf_logits.float()
    tt_logits = tt_logits.float()
    diff = tt_logits - hf_logits
    denom = hf_logits.abs().max().clamp_min(1e-12)
    return {
        "max_abs_diff": diff.abs().max().item(),
        "mean_abs_diff": diff.abs().mean().item(),
        "max_rel_diff": (diff.abs().max() / denom).item(),
        "cosine_similarity": F.cosine_similarity(
            hf_logits.reshape(1, -1), tt_logits.reshape(1, -1), dim=1
        ).item(),
    }


def _greedy_generate(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    input_ids: torch.Tensor,
    max_new_tokens: int,
    max_seq_len: int,
) -> torch.Tensor:
    generated = input_ids.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            current = generated[:, -max_seq_len:]
            logits = forward_fn(current)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=-1)
    return generated


def verify_hf_match(
    hf_model_path: str,
    checkpoint_dir: str,
    tokenizer_path: str,
    model_name: str,
    model_flavor: str,
    prompts: list[str],
    max_new_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    max_abs_tol: float,
    transformers_path: str | None = None,
) -> None:
    if model_name != "qwen3_5":
        raise NotImplementedError(
            "HF parity verification is only implemented for the Qwen3.5 language model."
        )

    _prepend_transformers_path(transformers_path)
    tokenizer = load_tokenizer(tokenizer_path)
    hf_model = _load_hf_model(
        hf_model_path,
        device,
        dtype,
        transformers_path=transformers_path,
    )
    tt_model, tt_model_args = load_model_from_dcp(
        checkpoint_dir, model_name, model_flavor, device
    )
    tt_model = tt_model.to(dtype=dtype)

    if not prompts:
        prompts = ["The capital of France is"]

    for prompt in prompts:
        print(f"\nPrompt: {prompt!r}")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            hf_logits = hf_model(input_ids).logits
            tt_logits = tt_model(input_ids)

        metrics = _compute_logit_metrics(hf_logits, tt_logits)
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.6g}")

        if metrics["max_abs_diff"] > max_abs_tol:
            raise AssertionError(
                f"Max abs diff {metrics['max_abs_diff']:.6g} exceeds tolerance {max_abs_tol:.6g}"
            )

        hf_generated = _greedy_generate(
            lambda ids: hf_model(ids).logits,
            input_ids,
            max_new_tokens=max_new_tokens,
            max_seq_len=tt_model_args.max_seq_len,
        )
        tt_generated = _greedy_generate(
            tt_model,
            input_ids,
            max_new_tokens=max_new_tokens,
            max_seq_len=tt_model_args.max_seq_len,
        )

        sequences_match = torch.equal(hf_generated, tt_generated)
        print(f"  greedy_tokens_match: {sequences_match}")
        if not sequences_match:
            print(f"  hf_tokens: {hf_generated[0].tolist()}")
            print(f"  tt_tokens: {tt_generated[0].tolist()}")
            raise AssertionError("Greedy decoding diverged between HF and TorchTitan")

        generated_text = tokenizer.decode(
            tt_generated[0, input_ids.shape[1] :], skip_special_tokens=True
        )
        print(f"  generated_text: {generated_text!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare HF Qwen3.5 logits and greedy decoding against TorchTitan."
    )
    parser.add_argument(
        "--hf_model_path",
        type=str,
        required=True,
        help="Local HF directory or HF repo id for the source Qwen3.5 model",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the converted TorchTitan DCP checkpoint",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Local tokenizer path or HF repo id",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="qwen3_5",
        choices=["qwen3_5"],
        help="TorchTitan model name",
    )
    parser.add_argument(
        "--model_flavor",
        type=str,
        default="2B",
        help="TorchTitan model flavor",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt to compare. Pass multiple times to run multiple prompts.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=16,
        help="Number of greedy-decoded tokens to compare",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: cpu, cuda, or auto",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="Inference dtype for both models",
    )
    parser.add_argument(
        "--max_abs_tol",
        type=float,
        default=0.5,
        help=(
            "Maximum allowed absolute logit difference. "
            "Use a looser tolerance for bf16/float16 parity checks."
        ),
    )
    parser.add_argument(
        "--transformers_path",
        type=str,
        default=None,
        help=(
            "Optional path prepended to sys.path before importing transformers. "
            "Use this to run against an isolated official transformers build "
            "without modifying the main training environment."
        ),
    )
    args = parser.parse_args()

    verify_hf_match(
        hf_model_path=args.hf_model_path,
        checkpoint_dir=args.checkpoint_dir,
        tokenizer_path=args.tokenizer_path,
        model_name=args.model_name,
        model_flavor=args.model_flavor,
        prompts=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=_select_device(args.device),
        dtype=_select_dtype(args.dtype),
        max_abs_tol=args.max_abs_tol,
        transformers_path=args.transformers_path,
    )


if __name__ == "__main__":
    main()
