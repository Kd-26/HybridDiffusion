# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
State dict adapter for converting between HF Qwen3.5 and torchtitan formats.

The HF Qwen3.5 checkpoint uses a VLM layout with prefix
``model.language_model.`` for the text backbone.  We strip / add that prefix
so the adapter works with both the full VLM safetensors and a text-only
export.

Dense models (e.g. 2B, 9B) use a standard MLP per block.
MoE models (e.g. 35B-A3B) use fused 3D expert tensors
(``experts.gate_up_proj``, ``experts.down_proj``) plus a shared expert and
router gate.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
from torchtitan.models.utils import MoEStateDictAdapter

from .args import Qwen3_5ModelArgs

_HF_LM_PREFIX = "model.language_model."


@dataclass
class HFAuditReport:
    mapped_keys: set[str] = field(default_factory=set)
    skipped_keys: set[str] = field(default_factory=set)
    unmapped_keys: set[str] = field(default_factory=set)


class Qwen3_5StateDictAdapter(MoEStateDictAdapter):
    def __init__(self, model_args: Qwen3_5ModelArgs, hf_assets_path: str | None):
        super().__init__(model_args, hf_assets_path)
        self.model_args = model_args
        self.hf_text_prefix = _HF_LM_PREFIX

        self.from_hf_map = {
            # Embeddings
            "embed_tokens.weight": "tok_embeddings.weight",
            # Full attention layers
            "layers.{}.self_attn.q_proj.weight": "layers.{}.self_attn.wq.weight",
            "layers.{}.self_attn.k_proj.weight": "layers.{}.self_attn.wk.weight",
            "layers.{}.self_attn.v_proj.weight": "layers.{}.self_attn.wv.weight",
            "layers.{}.self_attn.o_proj.weight": "layers.{}.self_attn.wo.weight",
            "layers.{}.self_attn.q_norm.weight": "layers.{}.self_attn.q_norm.weight",
            "layers.{}.self_attn.k_norm.weight": "layers.{}.self_attn.k_norm.weight",
            # Linear attention (Gated DeltaNet) layers
            "layers.{}.linear_attn.in_proj_qkv.weight": "layers.{}.linear_attn.in_proj_qkv.weight",
            "layers.{}.linear_attn.in_proj_z.weight": "layers.{}.linear_attn.in_proj_z.weight",
            "layers.{}.linear_attn.in_proj_b.weight": "layers.{}.linear_attn.in_proj_b.weight",
            "layers.{}.linear_attn.in_proj_a.weight": "layers.{}.linear_attn.in_proj_a.weight",
            "layers.{}.linear_attn.conv1d.weight": "layers.{}.linear_attn.conv1d.weight",
            "layers.{}.linear_attn.A_log": "layers.{}.linear_attn.A_log",
            "layers.{}.linear_attn.dt_bias": "layers.{}.linear_attn.dt_bias",
            "layers.{}.linear_attn.norm.weight": "layers.{}.linear_attn.norm.weight",
            "layers.{}.linear_attn.out_proj.weight": "layers.{}.linear_attn.out_proj.weight",
            # Dense MLP (non-MoE)
            "layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            # MoE per-expert (Qwen3 style, kept for compatibility)
            "layers.{}.mlp.experts.{}.gate_proj.weight": "layers.{}.moe.experts.w1",
            "layers.{}.mlp.experts.{}.up_proj.weight": "layers.{}.moe.experts.w3",
            "layers.{}.mlp.experts.{}.down_proj.weight": "layers.{}.moe.experts.w2",
            # MoE router
            "layers.{}.mlp.gate.weight": "layers.{}.moe.router.gate.weight",
            # MoE shared expert
            "layers.{}.mlp.shared_expert.gate_proj.weight": "layers.{}.moe.shared_experts.w1.weight",
            "layers.{}.mlp.shared_expert.up_proj.weight": "layers.{}.moe.shared_experts.w3.weight",
            "layers.{}.mlp.shared_expert.down_proj.weight": "layers.{}.moe.shared_experts.w2.weight",
            # Layer norms
            "layers.{}.input_layernorm.weight": "layers.{}.input_layernorm.weight",
            "layers.{}.post_attention_layernorm.weight": "layers.{}.post_attention_layernorm.weight",
            # Final norm / head
            "norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

    @staticmethod
    def _strip_lm_prefix(key: str) -> str:
        """Strip ``model.language_model.`` or ``model.`` prefix if present."""
        if key.startswith(_HF_LM_PREFIX):
            return key[len(_HF_LM_PREFIX):]
        if key.startswith("model."):
            return key[len("model."):]
        return key

    @staticmethod
    def _detect_hf_text_prefix(hf_keys: Iterable[str]) -> str:
        for key in hf_keys:
            if key.startswith(_HF_LM_PREFIX):
                return _HF_LM_PREFIX
        for key in hf_keys:
            if key.startswith("model."):
                return "model."
        return ""

    def _hf_text_key(self, rel_key: str) -> str:
        return f"{self.hf_text_prefix}{rel_key}"

    def audit_hf_keys(self, hf_keys: Iterable[str]) -> HFAuditReport:
        report = HFAuditReport()
        for key in hf_keys:
            if key == "lm_head.weight":
                report.mapped_keys.add(key)
                continue

            rel_key = self._strip_lm_prefix(key)

            if rel_key.startswith("visual."):
                report.skipped_keys.add(key)
                continue

            if rel_key.startswith("mtp.") or key.startswith("mtp."):
                report.skipped_keys.add(key)
                continue

            if "shared_expert_gate" in rel_key:
                report.skipped_keys.add(key)
                continue

            if self._is_fused_expert_key(rel_key):
                report.mapped_keys.add(key)
                continue

            if "mlp.experts" in rel_key and "shared_expert" not in rel_key:
                abstract_key = re.sub(r"(\d+)", "{}", rel_key, count=2)
                target = self.from_hf_map.get(abstract_key)
                (report.mapped_keys if target is not None else report.unmapped_keys).add(
                    key
                )
                continue

            if "layers" in rel_key:
                abstract_key = re.sub(r"(\d+)", "{}", rel_key, count=1)
                target = self.from_hf_map.get(abstract_key)
                (report.mapped_keys if target is not None else report.unmapped_keys).add(
                    key
                )
                continue

            target = self.from_hf_map.get(rel_key)
            (report.mapped_keys if target is not None else report.unmapped_keys).add(
                key
            )

        return report

    # ------------------------------------------------------------------
    # torchtitan -> HF
    # ------------------------------------------------------------------
    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}
        hf_state_dict: dict[str, Any] = {}
        fused_expert_weights: dict[str, dict[str, Any]] = {}

        for key, value in state_dict.items():
            if "moe.experts" in key and "shared_experts" not in key:
                match = re.search(r"layers\.(\d+)\.moe\.experts\.(w[123])", key)
                if match is None:
                    continue
                layer_num, weight_name = match.groups()
                if layer_num not in fused_expert_weights:
                    fused_expert_weights[layer_num] = {}
                fused_expert_weights[layer_num][weight_name] = value
                continue

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                if abstract_key not in to_hf_map:
                    continue
                layer_num = re.search(r"\d+", key).group(0)
                new_key = self._hf_text_key(to_hf_map[abstract_key].format(layer_num))
                hf_state_dict[new_key] = value
            else:
                if key not in to_hf_map:
                    continue
                if self.model_args.enable_weight_tying and key == "output.weight":
                    continue
                hf_key = to_hf_map[key]
                if hf_key == "lm_head.weight":
                    hf_state_dict[hf_key] = value
                else:
                    hf_state_dict[self._hf_text_key(hf_key)] = value

        for layer_num, expert_weights in fused_expert_weights.items():
            if {"w1", "w2", "w3"} - set(expert_weights):
                missing = sorted({"w1", "w2", "w3"} - set(expert_weights))
                raise ValueError(
                    f"Missing MoE expert weights for layer {layer_num}: {missing}"
                )

            gate_up = torch.cat((expert_weights["w1"], expert_weights["w3"]), dim=1)
            hf_state_dict[
                self._hf_text_key(f"layers.{layer_num}.mlp.experts.gate_up_proj")
            ] = gate_up
            hf_state_dict[
                self._hf_text_key(f"layers.{layer_num}.mlp.experts.down_proj")
            ] = expert_weights["w2"]

        return hf_state_dict

    # ------------------------------------------------------------------
    # HF -> torchtitan
    # ------------------------------------------------------------------
    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict: dict[str, Any] = {}
        expert_weights_by_layer: dict[str, dict] = {}
        self.hf_text_prefix = self._detect_hf_text_prefix(hf_state_dict)

        has_lm_head = any("lm_head" in k for k in hf_state_dict)
        if self.model_args.enable_weight_tying and not has_lm_head:
            embed_key = next(
                (k for k in hf_state_dict if "embed_tokens.weight" in k), None
            )
            if embed_key is not None:
                state_dict["output.weight"] = hf_state_dict[embed_key]

        for key, value in hf_state_dict.items():
            # Handle lm_head directly (no prefix)
            if key == "lm_head.weight":
                state_dict["output.weight"] = value
                continue

            rel_key = self._strip_lm_prefix(key)

            if rel_key.startswith("visual."):
                continue

            if rel_key.startswith("mtp.") or key.startswith("mtp."):
                continue

            if "shared_expert_gate" in rel_key:
                continue

            # ---- Fused 3D expert tensors (Qwen3.5-MoE style) ----
            if self._is_fused_expert_key(rel_key):
                self._handle_fused_experts(rel_key, value, state_dict)
                continue

            # ---- Per-expert tensors (Qwen3 style) ----
            if "mlp.experts" in rel_key and "shared_expert" not in rel_key:
                abstract_key = re.sub(r"(\d+)", "{}", rel_key, count=2)
                if abstract_key not in self.from_hf_map:
                    continue
                layer_num, expert_num = re.findall(r"\d+", rel_key)[:2]
                titan_abstract_key = self.from_hf_map[abstract_key]
                assert titan_abstract_key is not None
                new_key = titan_abstract_key.format(layer_num)

                if layer_num not in expert_weights_by_layer:
                    expert_weights_by_layer[layer_num] = {}
                if titan_abstract_key not in expert_weights_by_layer[layer_num]:
                    expert_weights_by_layer[layer_num][titan_abstract_key] = {}
                expert_weights_by_layer[layer_num][titan_abstract_key][
                    int(expert_num)
                ] = value

                from torch.distributed.tensor import DTensor

                if isinstance(value, DTensor):
                    stacked_value = self._concatenate_expert_weights_dtensor(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                        value.device_mesh,
                    )
                else:
                    stacked_value = self._concatenate_expert_weights(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                        self.model_args.moe_args.num_experts,
                    )

                if stacked_value is not None:
                    state_dict[new_key] = stacked_value
                continue

            # ---- Regular layer keys ----
            if "layers" in rel_key:
                abstract_key = re.sub(r"(\d+)", "{}", rel_key, count=1)
                if abstract_key not in self.from_hf_map:
                    continue
                layer_num = re.search(r"\d+", rel_key).group(0)
                new_key = self.from_hf_map[abstract_key]
                if new_key is None:
                    continue
                state_dict[new_key.format(layer_num)] = value
            else:
                if rel_key not in self.from_hf_map:
                    continue
                new_key = self.from_hf_map[rel_key]
                if new_key is not None:
                    state_dict[new_key] = value

        return state_dict

    # ------------------------------------------------------------------
    # Fused expert helpers (HF Qwen3.5-MoE stores experts as 3D tensors)
    # ------------------------------------------------------------------
    @staticmethod
    def _is_fused_expert_key(rel_key: str) -> bool:
        """Check if key is a fused 3D expert tensor (not per-expert)."""
        return (
            "mlp.experts.gate_up_proj" in rel_key
            or "mlp.experts.down_proj" in rel_key
        ) and not re.search(r"experts\.\d+\.", rel_key)

    def _handle_fused_experts(
        self,
        rel_key: str,
        value: torch.Tensor,
        state_dict: dict[str, Any],
    ) -> None:
        """Convert HF fused expert tensors to torchtitan GroupedExperts format.

        HF stores:
          experts.gate_up_proj: [num_experts, 2*inter, dim]  (gate || up fused)
          experts.down_proj:    [num_experts, dim, inter]

        torchtitan stores:
          moe.experts.w1: [num_experts, inter, dim]  (gate)
          moe.experts.w3: [num_experts, inter, dim]  (up)
          moe.experts.w2: [num_experts, dim, inter]  (down)
        """
        layer_num = re.search(r"\d+", rel_key).group(0)

        if "gate_up_proj" in rel_key:
            inter_2x = value.shape[1]
            inter = inter_2x // 2
            w1 = value[:, :inter, :]
            w3 = value[:, inter:, :]
            state_dict[f"layers.{layer_num}.moe.experts.w1"] = w1.contiguous()
            state_dict[f"layers.{layer_num}.moe.experts.w3"] = w3.contiguous()
        elif "down_proj" in rel_key:
            state_dict[f"layers.{layer_num}.moe.experts.w2"] = value
