# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from fnmatch import fnmatchcase
from typing import Any, Generic, Iterator, TypeVar

import torch
import torch.distributed.tensor
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import Replicate
from torch.optim import Optimizer

try:
    from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
except ImportError:
    MuonWithAuxAdam = None
    SingleDeviceMuonWithAuxAdam = None

try:
    from flashoptim import FlashAdamW, FlashAdam
except ImportError:
    FlashAdamW = None
    FlashAdam = None

from torchtitan.components.ft import FTManager, has_torchft
from torchtitan.config import Optimizer as OptimizerConfig
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger

__all__ = [
    "OptimizersContainer",
    "build_optimizers",
    "build_optimizers_with_moe_load_balancing",
]


if has_torchft:
    import torchft as ft


T = TypeVar("T", bound=Optimizer)


def _matches_any_glob(fqns: list[str], patterns: list[str]) -> bool:
    return any(fnmatchcase(fqn, pattern) for fqn in fqns for pattern in patterns)


def _collect_unique_trainable_parameters(
    model: nn.Module,
) -> list[tuple[nn.Parameter, list[str]]]:
    params_by_id: dict[int, tuple[nn.Parameter, list[str]]] = {}
    ordered_param_ids: list[int] = []

    for fqn, param in model.named_parameters(remove_duplicate=False):
        if not param.requires_grad:
            continue
        param_id = id(param)
        if param_id not in params_by_id:
            params_by_id[param_id] = (param, [fqn])
            ordered_param_ids.append(param_id)
        else:
            params_by_id[param_id][1].append(fqn)

    return [params_by_id[param_id] for param_id in ordered_param_ids]


def _build_optimizer_param_groups(
    model: nn.Module,
    optimizer_config: OptimizerConfig,
) -> tuple[list[dict[str, Any]], list[str]]:
    overrides = optimizer_config.module_lr_overrides
    if not overrides:
        raise ValueError("Expected optimizer.module_lr_overrides to be configured.")

    for override in overrides:
        if not override.target_globs:
            raise ValueError(
                "Each optimizer.module_lr_overrides entry must set target_globs."
            )
        if (override.lr is None) == (override.lr_multiplier is None):
            raise ValueError(
                "Each optimizer.module_lr_overrides entry must set exactly one of "
                "`lr` or `lr_multiplier`."
            )
        if override.lr is not None and override.lr <= 0:
            raise ValueError("optimizer.module_lr_overrides lr must be positive.")
        if override.lr_multiplier is not None and override.lr_multiplier <= 0:
            raise ValueError(
                "optimizer.module_lr_overrides lr_multiplier must be positive."
            )

    grouped_params: list[list[nn.Parameter]] = [[] for _ in overrides]
    default_params: list[nn.Parameter] = []

    for param, fqns in _collect_unique_trainable_parameters(model):
        matched_group_indices = [
            idx
            for idx, override in enumerate(overrides)
            if _matches_any_glob(fqns, override.target_globs)
        ]
        if len(matched_group_indices) > 1:
            matching_names = [
                overrides[idx].name or ",".join(overrides[idx].target_globs)
                for idx in matched_group_indices
            ]
            raise ValueError(
                "Parameter matched multiple optimizer.module_lr_overrides rules: "
                f"{fqns} -> {matching_names}"
            )
        if matched_group_indices:
            grouped_params[matched_group_indices[0]].append(param)
        else:
            default_params.append(param)

    param_groups: list[dict[str, Any]] = []
    group_layout: list[str] = []
    summary_bits: list[str] = []

    for override, params in zip(overrides, grouped_params, strict=True):
        if not params:
            continue
        lr = (
            override.lr
            if override.lr is not None
            else optimizer_config.lr * override.lr_multiplier
        )
        label = override.name or ",".join(override.target_globs)
        param_groups.append({"params": params, "lr": lr})
        group_layout.append(label)
        summary_bits.append(f"{label}: lr={lr:g}, tensors={len(params)}")

    if default_params:
        param_groups.append({"params": default_params})
        group_layout.append("default")
        summary_bits.append(
            f"default: lr={optimizer_config.lr:g}, tensors={len(default_params)}"
        )

    if not param_groups:
        raise ValueError(
            "optimizer.module_lr_overrides matched no trainable parameters."
        )

    logger.info("Built optimizer param groups: %s", "; ".join(summary_bits))
    return param_groups, group_layout


class OptimizersContainer(Optimizer, Stateful, Generic[T]):
    """A container for multiple optimizers.

    This class is used to wrap multiple optimizers into a single object that can be
    used to reduce the complexity of the training loop. This mimics the behavior of
    ``torch.optim.Optimizer``. This class currently only supports ``Adam`` and ``AdamW``.

    **Note**
    Users who want to customize the optimizer behavior can inherit from this class and
    extend the functionality as needed. The following methods must follow the same signature
    as ``torch.optim.Optimizer`` class: ``step()``, ``zero_grad()``, ``state_dict()``,
    ``load_state_dict()``.

    **Limitations**
    This class assumes that all the optimizers are the same type and have the same
    configurations. With this assumption, TorchTitan can support lr scheduler resharding
    (e.g., loading a checkpoint with a different number of GPUs and/or different
    parallelization strategy). Note that ``get_optimizer_state_dict`` already enables the
    resharding for the optimizer state but not for the lr scheduler state, hence the limitation.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_kwargs (Dict[str, Any]): Keyword arguments for the optimizers.
        name (str): Name of the optimizers.
    """

    optimizers: list[T]
    model_parts: list[nn.Module]

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
        optimizer_inputs: list[list[Any]] | None = None,
    ) -> None:
        all_params = []
        self.optimizers = []
        self.model_parts = model_parts
        if optimizer_inputs is None:
            optimizer_inputs = [
                [p for p in model.parameters() if p.requires_grad]
                for model in self.model_parts
            ]
        assert len(optimizer_inputs) == len(self.model_parts)

        for model, optimizer_input in zip(
            self.model_parts, optimizer_inputs, strict=True
        ):
            if optimizer_input and isinstance(optimizer_input[0], dict):
                params = [
                    param
                    for param_group in optimizer_input
                    for param in param_group["params"]
                ]
            else:
                params = optimizer_input
            self.optimizers.append(optimizer_cls(optimizer_input, **optimizer_kwargs))
            all_params.extend(params)
        self._validate_length(len(self.model_parts))
        self._post_init(all_params, optimizer_kwargs)

    def __iter__(self) -> Iterator[T]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    # pyrefly: ignore [bad-override]
    def step(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {
            k: v
            for sd in map(func, self.model_parts, self.optimizers)
            for k, v in sd.items()
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model_parts, self.optimizers))

    def _validate_length(self, expected_length: int) -> None:
        assert expected_length == len(self.optimizers), (
            "Must pass one optimizer per model part or per param if "
            "using OptimizersInBackwardContainer."
        )

    def _post_init(
        self, all_params: list[nn.Parameter], optimizer_kwargs: dict[str, Any]
    ) -> None:
        # We need to call Optimizer.__init__() to initialize some necessary optimizer
        # functionality such as hooks.
        Optimizer.__init__(self, all_params, optimizer_kwargs)

class MuonOptimizersContainer(OptimizersContainer):
    """Container for Muon + AdamW hybrid optimization, FSDP-compatible.

    Uses SingleDeviceMuonWithAuxAdam (no internal dist.all_gather) because
    FSDP already handles parameter sharding and gradient synchronization via
    DTensor. The original MuonWithAuxAdam does its own dist.all_gather which
    conflicts with FSDP's DTensor (causes DeviceMesh assertion error).

    Parameter splitting via param_groups with use_muon=True/False:
      - 2D weight matrices (excluding embeddings/head) -> Muon (use_muon=True)
      - Other params (embeddings, head, gains, biases) -> AdamW (use_muon=False)
    """

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_kwargs: dict[str, Any],
    ) -> None:
        if SingleDeviceMuonWithAuxAdam is None:
            raise RuntimeError(
                "Cannot use Muon strategy: muon package not installed. "
                "Install via: pip install git+https://github.com/KellerJordan/Muon"
            )

        all_params = []
        self.optimizers = []
        self.model_parts = model_parts

        # Extract config values
        muon_lr = optimizer_kwargs.get("lr", 0.02)
        muon_momentum = optimizer_kwargs.get("momentum", 0.95)
        adamw_lr = 3e-4
        adamw_betas = optimizer_kwargs.get("betas", (0.9, 0.95))
        weight_decay = optimizer_kwargs.get("weight_decay", 0.01)

        for model in self.model_parts:
            muon_params = []
            adamw_params = []

            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                # Muon for 2D weight matrices, excluding embeddings and output head
                if p.ndim >= 2 and "embed" not in name and "output" not in name:
                    muon_params.append(p)
                else:
                    adamw_params.append(p)

            logger.info(
                f"Muon param split: {len(muon_params)} Muon params, "
                f"{len(adamw_params)} AdamW params"
            )

            param_groups = [
                dict(
                    params=muon_params,
                    use_muon=True,
                    lr=muon_lr,
                    momentum=muon_momentum,
                    weight_decay=weight_decay,
                ),
                dict(
                    params=adamw_params,
                    use_muon=False,
                    lr=adamw_lr,
                    betas=adamw_betas,
                    weight_decay=weight_decay,
                ),
            ]

            # Use SingleDeviceMuonWithAuxAdam: no internal dist.all_gather
            # FSDP handles distributed synchronization via DTensor
            self.optimizers.append(SingleDeviceMuonWithAuxAdam(param_groups))
            all_params.extend(muon_params)
            all_params.extend(adamw_params)

        self._post_init(all_params, optimizer_kwargs)

class OptimizersInBackwardContainer(OptimizersContainer):
    """OptimizersContainer for executing ``optim.step()`` in backward pass.

    This class extend ``OptimizersContainer`` to support optimizer step in
    backward pass. ``step()`` and ``zero_grad()`` are no-op in this class.
    Instead, ``register_post_accumulate_grad_hook`` is used to register a hook to
    execute these methods when the gradient is accumulated.
    """

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
    ) -> None:
        all_params = []
        self.model_parts = model_parts

        optim_dict = {}
        for model in self.model_parts:
            for p in model.parameters():
                if p.requires_grad:
                    optim_dict[p] = optimizer_cls([p], **optimizer_kwargs)
                all_params.append(p)

        def optim_hook(param) -> None:
            optim_dict[param].step()
            optim_dict[param].zero_grad()

        for model in self.model_parts:
            for param in model.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(optim_hook)

        self.optimizers = list(optim_dict.values())

        self._validate_length(
            sum(len(list(model.parameters())) for model in self.model_parts)
        )
        self._post_init(all_params, optimizer_kwargs)

    # pyrefly: ignore [bad-override]
    def step(self) -> None:
        pass

    # pyrefly: ignore [bad-override]
    def zero_grad(self) -> None:
        pass


class FTOptimizersContainer(OptimizersContainer):
    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
        ft_manager: "ft.Manager",
        use_ft_optimizer: bool = True,
        optimizer_inputs: list[list[Any]] | None = None,
    ) -> None:
        super().__init__(
            model_parts,
            optimizer_cls,
            optimizer_kwargs,
            optimizer_inputs=optimizer_inputs,
        )

        # Force to initialize the optimizer state so that `optim.step()`
        # won't be called by state_dict() and load_state_dict().
        _ = {
            k: v
            for sd in map(get_optimizer_state_dict, model_parts, self.optimizers)
            for k, v in sd.items()
        }
        self.cache_state_dict: dict[str, Any] = {}
        self._ft_optimizer = ft.Optimizer(ft_manager, self)
        # Whether to determine quorum using FT.optimizer,
        # in semi-sync training we use the synchronization step to start quorum
        self._use_ft_optimizer: bool = use_ft_optimizer

    def init_cache_state_dict(self) -> None:
        self.cache_state_dict = super().state_dict()

    def state_dict(self) -> dict[str, Any]:
        return self.cache_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # We have to invalidate the `cache_state_dict` because optimizer uses
        # assign instead of copy when doing `load_state_dict()`. Without
        # invalidating the `cache_state_dict`, there will be memory leakage.
        self.cache_state_dict = {}
        super().load_state_dict(state_dict)
        self.init_cache_state_dict()

    def step(self, *args, **kwargs) -> None:
        """Calling the correct step() depending on the caller.

        TorchFT's OptimizerWrapper.step() is designed to be called only once
        per train step per ft.Manager regardless how many optimizers are used.
        Hence we will need to appropriately dispatch the call.
        """
        if self._use_ft_optimizer:
            self._use_ft_optimizer = False
            self._ft_optimizer.step(*args, **kwargs)
            self._use_ft_optimizer = True
        else:
            super().step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        """Calling the correct zero_grad() depending on the caller.

        Check the comment in ``step()``.
        """
        if self._use_ft_optimizer:
            self._use_ft_optimizer = False
            self._ft_optimizer.zero_grad(*args, **kwargs)
            self._use_ft_optimizer = True
        else:
            super().zero_grad(*args, **kwargs)


def build_optimizers(
    model_parts: list[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
) -> OptimizersContainer:
    """Create a OptimizersContainer for the given model parts and job config.

    This function creates a ``OptimizersContainer`` for the given model parts.
    ``optimizer_config`` should define the correct optimizer name and parameters.
    This function currently supports creating ``OptimizersContainer`` and
    ``OptimizersInBackwardContainer``.

    **Note**
    Users who want to customize the optimizer behavior can create their own
    ``OptimizersContainer`` subclass and ``build_optimizers``. Passing the
    customized ``build_optimizers`` to ``TrainSpec`` will create the customized
    ``OptimizersContainer``.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_config (OptimizerConfig): Optimizer config containing the optimizer name and parameters.
        parallel_dims (ParallelDims): Parallel dimensions for the model.
        ft_manager (FTManager | None): Fault tolerance manager.
    """
    optim_in_bwd = optimizer_config.early_step_in_backward
    if optimizer_config.module_lr_overrides and optim_in_bwd:
        raise NotImplementedError(
            "optimizer.module_lr_overrides is not supported with "
            "optimizer.early_step_in_backward."
        )
    if optim_in_bwd:
        if optimizer_config.name == "Muon":
             raise NotImplementedError("Muon does not support optimizers in backward.")
        if parallel_dims.ep_enabled:
            raise NotImplementedError(
                "Optimizers in backward is not supported with Expert Parallel."
            )
        if parallel_dims.pp_enabled:
            raise NotImplementedError(
                "Optimizers in backward is not supported with Pipeline Parallel."
            )
        if ft_manager and ft_manager.enabled:
            raise NotImplementedError(
                "TorchFT is not supported with optimizers in backward."
            )

    name = optimizer_config.name
    lr = optimizer_config.lr
    beta1 = optimizer_config.beta1
    beta2 = optimizer_config.beta2
    eps = optimizer_config.eps
    weight_decay = optimizer_config.weight_decay

    optim_implementation = optimizer_config.implementation
    assert optim_implementation in ["fused", "foreach", "for-loop"]

    fused = optim_implementation == "fused"
    foreach = optim_implementation == "foreach"

    optimizer_kwargs = {
        "lr": lr,
        "betas": (beta1, beta2),
        "eps": eps,
        "weight_decay": weight_decay,
        "fused": fused,
        "foreach": foreach,
    }

    optimizer_inputs = None
    if optimizer_config.module_lr_overrides:
        if name == "Muon":
            raise NotImplementedError(
                "optimizer.module_lr_overrides is not supported with Muon."
            )
        optimizer_inputs = []
        group_layouts = []
        for model_part in model_parts:
            param_groups, group_layout = _build_optimizer_param_groups(
                model_part,
                optimizer_config,
            )
            optimizer_inputs.append(param_groups)
            group_layouts.append(group_layout)

        if parallel_dims.pp_enabled:
            expected_layout = group_layouts[0]
            for idx, group_layout in enumerate(group_layouts[1:], start=1):
                if group_layout != expected_layout:
                    raise ValueError(
                        "optimizer.module_lr_overrides must create the same non-empty "
                        "param-group layout for every pipeline stage. "
                        f"Stage 0 has {expected_layout}, stage {idx} has {group_layout}."
                    )

    if name == "Muon":
        # For Muon, return the custom MuonOptimizersContainer directly.
        # Note: ft_manager is not yet compatible with Muon.
        if ft_manager and ft_manager.enabled:
             raise NotImplementedError("Muon is not yet supported with TorchFT.")
             
        return MuonOptimizersContainer(
            model_parts, 
            optimizer_kwargs
        )

    # FlashOptim: memory-efficient optimizers with fused Triton kernels.
    # Uses its own kernels, so fused/foreach are not applicable.
    flash_optimizer_classes = {
        "FlashAdamW": FlashAdamW,
        "FlashAdam": FlashAdam,
    }
    if name in flash_optimizer_classes:
        if flash_optimizer_classes[name] is None:
            raise RuntimeError(
                f"{name} not available. Install: pip install flashoptim"
            )
        if optim_in_bwd:
            raise NotImplementedError(
                "FlashOptim does not support optimizer-in-backward "
                "(gradient release must be configured via flashoptim API)."
            )
        master_weight_bits = optimizer_config.master_weight_bits
        has_fp32_params = any(
            p.dtype == torch.float32
            for m in model_parts for p in m.parameters()
        )
        if has_fp32_params and master_weight_bits is not None:
            logger.warning(
                f"Model has fp32 parameters (FSDP2 mixed precision stores params "
                f"in fp32). Setting master_weight_bits=None (was {master_weight_bits}). "
                f"Optimizer states will still be quantized to 8-bit for memory savings."
            )
            master_weight_bits = None
        flash_kwargs = {
            "lr": lr,
            "betas": (beta1, beta2),
            "eps": eps,
            "weight_decay": weight_decay,
            "master_weight_bits": master_weight_bits,
            "compress_state_dict": optimizer_config.compress_state_dict,
        }
        flash_cls = flash_optimizer_classes[name]
        logger.info(
            f"Using {name} (master_weight_bits={master_weight_bits})"
        )
        return OptimizersContainer(
            model_parts,
            flash_cls,
            flash_kwargs,
            optimizer_inputs=optimizer_inputs,
        )

    optimizer_classes = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
    }
    if name not in optimizer_classes:
        raise NotImplementedError(f"Optimizer {name} not added.")
    optimizer_cls = optimizer_classes[name]

    if optim_in_bwd:
        return OptimizersInBackwardContainer(
            model_parts, optimizer_cls, optimizer_kwargs
        )

    if ft_manager and ft_manager.enabled:
        return FTOptimizersContainer(
            model_parts,
            optimizer_cls,
            optimizer_kwargs,
            ft_manager.manager,
            use_ft_optimizer=ft_manager.use_async_quorum,
            optimizer_inputs=optimizer_inputs,
        )

    return OptimizersContainer(
        model_parts,
        optimizer_cls,
        optimizer_kwargs,
        optimizer_inputs=optimizer_inputs,
    )


def build_optimizers_with_moe_load_balancing(
    model_parts: list[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
) -> OptimizersContainer:
    optimizers = build_optimizers(
        model_parts=model_parts,
        optimizer_config=optimizer_config,
        parallel_dims=parallel_dims,
        ft_manager=ft_manager,
    )

    def _should_register_moe_balancing_hook(model_parts: list[nn.Module]) -> bool:
        for model_part in model_parts:
            # pyrefly: ignore [not-callable]
            for transformer_block in model_part.layers.values():
                # pyrefly: ignore [missing-attribute]
                if transformer_block.moe_enabled:
                    # Assumption: load_balance_coeff is set universally on all moe blocks.
                    # pyrefly: ignore [missing-attribute]
                    return bool(transformer_block.moe.load_balance_coeff)
        return False

    # for MoE auxiliary-loss-free load balancing
    def _is_recomputation_enabled(module):
        return getattr(module, "checkpoint_impl", None) is CheckpointImpl.NO_REENTRANT

    def _update_expert_bias(
        model_parts: list[nn.Module],
        parallel_dims: ParallelDims,
    ):
        loss_mesh = parallel_dims.get_optional_mesh("loss")
        # TODO: Currently this sync is blocking (thus exposed) and happens on the
        # default compute stream. Need to assess if this is OK performance-wise.
        tokens_per_expert_list = []
        for model_part in model_parts:
            # pyrefly: ignore [not-callable]
            for transformer_block in model_part.layers.values():
                # pyrefly: ignore [missing-attribute]
                if not transformer_block.moe_enabled:
                    continue
                # pyrefly: ignore [missing-attribute]
                if transformer_block.moe.load_balance_coeff is None:
                    return
                # pyrefly: ignore [missing-attribute]
                tokens_per_expert = transformer_block.moe.tokens_per_expert
                if _is_recomputation_enabled(transformer_block):
                    # TODO: This is a hack, we assume with full AC, the tokens_per_expert is counted twice.
                    # This does not affect to expert choice, but affects the experts usage metrics.
                    # We divide by 2 to correct for this double-counting due to recomputation
                    # TODO: new API to help determine if AC is enabled https://github.com/pytorch/pytorch/pull/160888
                    tokens_per_expert = tokens_per_expert // 2
                tokens_per_expert_list.append(tokens_per_expert)

        tokens_per_expert_by_layer = torch.vstack(tokens_per_expert_list)

        if loss_mesh is not None:
            if isinstance(tokens_per_expert_by_layer, torch.distributed.tensor.DTensor):
                tokens_per_expert_by_layer = tokens_per_expert_by_layer.redistribute(
                    placements=[Replicate()]
                    * tokens_per_expert_by_layer.device_mesh.ndim
                )
            else:
                # Perform single all-reduce to get global statistics across all processes
                pg = loss_mesh.get_group()
                torch.distributed.all_reduce(
                    tokens_per_expert_by_layer,
                    group=pg,
                    op=torch.distributed.ReduceOp.SUM,
                )

        moe_layer_idx = 0
        with torch.no_grad():
            for model_part in model_parts:
                # pyrefly: ignore [not-callable]
                for transformer_block in model_part.layers.values():
                    # pyrefly: ignore [missing-attribute]
                    if not transformer_block.moe_enabled:
                        continue
                    # pyrefly: ignore [missing-attribute]
                    moe = transformer_block.moe

                    tokens_per_expert = tokens_per_expert_by_layer[
                        moe_layer_idx
                    ].float()
                    moe_layer_idx += 1

                    # update the expert bias
                    # this is not exactly the same as https://arxiv.org/pdf/2408.15664 proposed
                    expert_bias_delta = moe.load_balance_coeff * torch.sign(
                        tokens_per_expert.mean() - tokens_per_expert
                    )
                    expert_bias_delta = expert_bias_delta - expert_bias_delta.mean()
                    moe.expert_bias.add_(expert_bias_delta)
                    moe.tokens_per_expert.zero_()

    if _should_register_moe_balancing_hook(model_parts):
        optimizers.register_step_pre_hook(
            lambda *args, **kwargs: _update_expert_bias(
                model_parts, parallel_dims=parallel_dims
            )
        )

    return optimizers
