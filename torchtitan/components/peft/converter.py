# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch.nn as nn

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import register_model_converter
from torchtitan.tools.logging import logger

from .lora import LoRALinear
from .selector import select_parameter_fqns, select_target_modules
from .utils import build_parameter_alias_maps, summarize_parameter_counts

_PEFT_RUNTIME_STATE_ATTR = "_torchtitan_peft_runtime_state"


@dataclass
class PeftRuntimeState:
    method: str
    adapted_module_fqns: list[str]
    adapted_modules: list[LoRALinear]
    trainable_base_param_fqns: list[str]


class PeftConverter:
    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.job_config = job_config
        self.parallel_dims = parallel_dims
        self.peft_config = job_config.peft
        self._validate_job_config()

    def _validate_job_config(self) -> None:
        if self.job_config.model.converters != ["peft"]:
            raise ValueError(
                "PEFT v1 does not support stacking model converters. "
                f"Expected ['peft'], got {self.job_config.model.converters}."
            )
        if self.peft_config.method != "lora":
            raise ValueError(
                f"Unsupported PEFT method '{self.peft_config.method}'. Only 'lora' is supported in v1."
            )
        if self.parallel_dims.tp_enabled:
            raise ValueError("PEFT v1 does not support Tensor Parallel.")
        if self.parallel_dims.pp_enabled:
            raise ValueError("PEFT v1 does not support Pipeline Parallel.")
        if self.parallel_dims.ep_enabled or self.parallel_dims.etp_enabled:
            raise ValueError("PEFT v1 does not support MoE expert parallelism.")
        if self.job_config.optimizer.name == "Muon":
            raise ValueError("PEFT v1 does not support the Muon optimizer.")
        if self.job_config.optimizer.early_step_in_backward:
            raise ValueError("PEFT v1 does not support optimizer-in-backward.")
        if self.job_config.checkpoint.initial_load_in_hf:
            raise ValueError("PEFT v1 does not support HF initial checkpoint loading.")
        if self.job_config.checkpoint.last_save_in_hf:
            raise ValueError("PEFT v1 does not support HF checkpoint export.")
        if (
            self.job_config.checkpoint.initial_load_path is not None
            and self.job_config.checkpoint.initial_ckpt_load_plan != "ckpt_model:mem_model"
        ):
            raise ValueError(
                "PEFT initial loading requires checkpoint.initial_ckpt_load_plan = "
                "'ckpt_model:mem_model'."
            )

        lora_config = self.peft_config.lora
        if lora_config.rank <= 0:
            raise ValueError("PEFT LoRA rank must be > 0.")
        if not 0.0 <= lora_config.dropout < 1.0:
            raise ValueError("PEFT LoRA dropout must satisfy 0.0 <= dropout < 1.0.")

    def convert(self, model: nn.Module):
        if getattr(getattr(model, "model_args", None), "moe_enabled", False):
            raise ValueError("PEFT v1 does not support MoE-enabled models.")
        if not self.peft_config.target_globs:
            raise ValueError("PEFT requires at least one peft.target_globs entry.")

        selections, unsupported_fqns = select_target_modules(
            model,
            target_globs=self.peft_config.target_globs,
            exclude_globs=self.peft_config.exclude_globs,
        )
        if unsupported_fqns and self.peft_config.strict:
            raise ValueError(
                "PEFT target_globs matched unsupported modules: "
                f"{unsupported_fqns}. Only nn.Linear is supported in v1."
            )
        if unsupported_fqns and not self.peft_config.strict:
            logger.warning(
                "PEFT target_globs matched unsupported modules that will be ignored: %s",
                unsupported_fqns,
            )
        if not selections:
            raise ValueError(
                "PEFT target_globs matched no supported modules after exclusions."
            )

        if (
            getattr(getattr(model, "model_args", None), "enable_weight_tying", False)
            and any(selection.fqn == "output" for selection in selections)
        ):
            raise ValueError(
                "PEFT cannot adapt the output layer when weight tying is enabled."
            )

        trainable_base_param_fqns = select_parameter_fqns(
            model, self.peft_config.trainable_base_globs
        )
        if self.peft_config.trainable_base_globs and not trainable_base_param_fqns and self.peft_config.strict:
            raise ValueError(
                "PEFT trainable_base_globs matched no base-model parameters."
            )

        adapted_modules: list[LoRALinear] = []
        adapted_module_fqns: list[str] = []
        lora_config = self.peft_config.lora
        for selection in selections:
            rank_limit = min(
                selection.module.in_features, selection.module.out_features
            )
            if lora_config.rank > rank_limit:
                raise ValueError(
                    f"PEFT LoRA rank {lora_config.rank} is larger than the supported limit "
                    f"{rank_limit} for module '{selection.fqn}'."
                )
            parent_module = (
                model
                if not selection.parent_fqn
                else model.get_submodule(selection.parent_fqn)
            )
            lora_module = LoRALinear.from_linear(
                selection.module,
                rank=lora_config.rank,
                alpha=lora_config.alpha,
                dropout=lora_config.dropout,
                scaling=lora_config.scaling,
                init=lora_config.init,
            )
            parent_module._modules[selection.child_name] = lora_module
            adapted_modules.append(lora_module)
            adapted_module_fqns.append(selection.fqn)

        runtime_state = PeftRuntimeState(
            method="lora",
            adapted_module_fqns=adapted_module_fqns,
            adapted_modules=adapted_modules,
            trainable_base_param_fqns=trainable_base_param_fqns,
        )
        setattr(model, _PEFT_RUNTIME_STATE_ATTR, runtime_state)

    def post_model_init(self, model: nn.Module | list[nn.Module]):
        models = [model] if isinstance(model, nn.Module) else model
        if len(models) != 1:
            raise ValueError(
                "PEFT v1 does not support pipeline-parallel model partitioning."
            )

        model_part = models[0]
        runtime_state: PeftRuntimeState | None = getattr(
            model_part, _PEFT_RUNTIME_STATE_ATTR, None
        )
        if runtime_state is None:
            return

        for adapted_module in runtime_state.adapted_modules:
            adapted_module.reset_lora_parameters()
            if adapted_module.lora_a.is_meta or adapted_module.lora_b.is_meta:
                raise RuntimeError(
                    "PEFT adapter parameters are still on the meta device during "
                    "post_model_init(). This indicates PEFT initialization ran before "
                    "model materialization or that materialization skipped the adapter "
                    "parameters."
                )

        alias_names_by_id, params_by_canonical_name = build_parameter_alias_maps(
            model_part
        )

        for module_fqn, adapted_module in zip(
            runtime_state.adapted_module_fqns, runtime_state.adapted_modules, strict=True
        ):
            alias_names = alias_names_by_id.get(id(adapted_module.weight))
            if alias_names is None:
                raise ValueError(
                    f"PEFT-adapted module '{module_fqn}' no longer has a reachable base weight after "
                    "parallelization. This commonly happens when the output weight becomes tied."
                )
            if len(alias_names) > 1:
                raise ValueError(
                    f"PEFT-adapted module '{module_fqn}' has a shared base weight with aliases "
                    f"{alias_names}. Shared adapted weights are not supported in v1."
                )

        if self.peft_config.freeze_base_model:
            for param in model_part.parameters():
                param.requires_grad = False

        for adapted_module in runtime_state.adapted_modules:
            adapted_module.lora_a.requires_grad = True
            adapted_module.lora_b.requires_grad = True

        for param_fqn in runtime_state.trainable_base_param_fqns:
            matched_params = params_by_canonical_name.get(param_fqn, [])
            if not matched_params:
                raise ValueError(
                    f"PEFT-selected base parameter '{param_fqn}' is not reachable after parallelization."
                )
            if len(matched_params) > 1:
                raise ValueError(
                    f"PEFT-selected base parameter '{param_fqn}' resolved to multiple parameter "
                    "objects after parallelization."
                )
            matched_params[0].requires_grad = True

        total_params, trainable_params = summarize_parameter_counts(model_part)
        trainable_pct = (
            0.0 if total_params == 0 else 100.0 * trainable_params / total_params
        )
        logger.info(
            "PEFT active with modules=%s trainable_base_params=%s trainable=%d/%d (%.4f%%)",
            runtime_state.adapted_module_fqns,
            runtime_state.trainable_base_param_fqns,
            trainable_params,
            total_params,
            trainable_pct,
        )

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        return None


register_model_converter(PeftConverter, "peft")
