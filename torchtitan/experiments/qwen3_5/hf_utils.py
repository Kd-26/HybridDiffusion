from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from torchtitan.models.qwen3_5.model.args import Qwen3_5ModelArgs


def load_hf_config(input_path: str | Path) -> dict[str, Any]:
    path = Path(input_path)
    if path.is_dir():
        path = path / "config.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_text_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("text_config", config)


def list_hf_checkpoint_keys(input_dir: str | Path) -> list[str]:
    input_dir = Path(input_dir)
    index_path = input_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            index_data = json.load(f)
        return sorted(index_data["weight_map"])

    try:
        from safetensors import safe_open
    except ModuleNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not find {index_path} and safetensors is not available."
        ) from exc

    keys: set[str] = set()
    shard_paths = sorted(input_dir.glob("*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No model.safetensors.index.json or *.safetensors found in {input_dir}"
        )
    for shard_path in shard_paths:
        with safe_open(str(shard_path), framework="pt") as shard:
            keys.update(shard.keys())
    return sorted(keys)


def validate_model_args_against_hf_config(
    model_args: Qwen3_5ModelArgs,
    hf_config: dict[str, Any],
) -> list[str]:
    issues: list[str] = []
    text_config = extract_text_config(hf_config)
    rope_config = text_config.get("rope_parameters", {})

    def check(name: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            issues.append(f"{name}: expected {expected!r}, got {actual!r}")

    check("dim", model_args.dim, text_config["hidden_size"])
    check("n_layers", model_args.n_layers, text_config["num_hidden_layers"])
    check("n_heads", model_args.n_heads, text_config["num_attention_heads"])
    check("n_kv_heads", model_args.n_kv_heads, text_config["num_key_value_heads"])
    check("head_dim", model_args.head_dim, text_config["head_dim"])
    check("vocab_size", model_args.vocab_size, text_config["vocab_size"])
    check("norm_eps", model_args.norm_eps, text_config["rms_norm_eps"])
    check(
        "rope_theta",
        model_args.rope_theta,
        rope_config.get("rope_theta", text_config.get("rope_theta")),
    )
    check(
        "partial_rotary_factor",
        model_args.partial_rotary_factor,
        rope_config.get(
            "partial_rotary_factor", text_config.get("partial_rotary_factor")
        ),
    )
    check(
        "linear_conv_kernel_dim",
        model_args.linear_conv_kernel_dim,
        text_config["linear_conv_kernel_dim"],
    )
    check(
        "linear_key_head_dim",
        model_args.linear_key_head_dim,
        text_config["linear_key_head_dim"],
    )
    check(
        "linear_value_head_dim",
        model_args.linear_value_head_dim,
        text_config["linear_value_head_dim"],
    )
    check(
        "linear_num_key_heads",
        model_args.linear_num_key_heads,
        text_config["linear_num_key_heads"],
    )
    check(
        "linear_num_value_heads",
        model_args.linear_num_value_heads,
        text_config["linear_num_value_heads"],
    )
    check(
        "full_attention_interval",
        model_args.full_attention_interval,
        text_config["full_attention_interval"],
    )
    check(
        "enable_weight_tying",
        model_args.enable_weight_tying,
        hf_config.get(
            "tie_word_embeddings", text_config.get("tie_word_embeddings", False)
        ),
    )
    check("max_seq_len", model_args.max_seq_len, text_config["max_position_embeddings"])
    check(
        "eos_id",
        model_args.eos_id,
        text_config.get("eos_token_id", hf_config.get("eos_token_id")),
    )

    if "layer_types" in text_config:
        check("layer_types", model_args.layer_types, text_config["layer_types"])

    if model_args.moe_enabled:
        check(
            "moe_inter_dim",
            model_args.moe_inter_dim,
            text_config["moe_intermediate_size"],
        )
        check(
            "moe_args.num_experts",
            model_args.moe_args.num_experts,
            text_config["num_experts"],
        )
        check(
            "moe_args.top_k",
            model_args.moe_args.top_k,
            text_config["num_experts_per_tok"],
        )
        shared_expert_count = 1 if "shared_expert_intermediate_size" in text_config else 0
        check(
            "moe_args.num_shared_experts",
            model_args.moe_args.num_shared_experts,
            shared_expert_count,
        )
    else:
        check("hidden_dim", model_args.hidden_dim, text_config["intermediate_size"])

    return issues


def validate_state_dict_against_reference(
    reference_state_dict: dict[str, Any],
    candidate_state_dict: dict[str, Any],
    *,
    ignore_keys: Iterable[str] = (),
) -> list[str]:
    ignored = set(ignore_keys)
    reference_keys = {
        k for k in reference_state_dict
        if k not in ignored and not any(ig in k for ig in ignored)
    }
    candidate_keys = set(candidate_state_dict)
    issues: list[str] = []

    missing_keys = sorted(reference_keys - candidate_keys)
    unexpected_keys = sorted(candidate_keys - reference_keys)
    if missing_keys:
        issues.append(
            f"missing keys ({len(missing_keys)}): {', '.join(missing_keys[:10])}"
        )
    if unexpected_keys:
        issues.append(
            f"unexpected keys ({len(unexpected_keys)}): {', '.join(unexpected_keys[:10])}"
        )

    for key in sorted(reference_keys & candidate_keys):
        expected = reference_state_dict[key]
        actual = candidate_state_dict[key]
        if hasattr(expected, "shape") and hasattr(actual, "shape"):
            if tuple(expected.shape) != tuple(actual.shape):
                issues.append(
                    f"shape mismatch for {key}: expected {tuple(expected.shape)}, got {tuple(actual.shape)}"
                )

    return issues
