# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelization for the text-only Qwen3.5 model (FSDP, TP, AC, compile)."""

import torch
import torch._inductor.config
import torch.nn as nn

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
    register_fsdp_forward_method,
)
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.config.job_config import Compile as CompileConfig
from torchtitan.distributed import NoParallel, ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.dual_pipe_v import get_dual_pipe_v_flag
from torchtitan.models.llama3.infra.parallelize import apply_ddp
from torchtitan.models.llama4.infra.parallelize import (
    apply_fsdp,
    apply_moe_ep_tp,
)
from torchtitan.tools.logging import logger


_TORCH_VERSION = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])

_op_sac_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    torch.ops.aten._scaled_dot_product_attention_math.default,
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    torch.ops.aten.max.default,
    torch._higher_order_ops.flex_attention,
}
if _TORCH_VERSION >= (2, 10):
    _op_sac_save_list.add(torch.ops.torch_attn._varlen_attn.default)
    _op_sac_save_list.add(torch._higher_order_ops.inductor_compiled_code)


def apply_compile(model: nn.Module, compile_config: CompileConfig, ep_enabled: bool):
    """Compile Qwen3.5 blocks with a hybrid policy.

    Full-attention blocks can be compiled as a whole block. Linear-attention
    blocks must leave the gated-delta kernel call in eager because the FLA
    kernel wrapper and backward utility code are not Dynamo-fullgraph safe.
    """
    del ep_enabled  # Qwen3.5 language-only path does not need llama4's MoE patching.

    compiled_full_blocks = 0
    compiled_linear_attn = 0
    compiled_linear_ffn = 0

    for layer_id, transformer_block in model.layers.named_children():
        if isinstance(transformer_block, CheckpointWrapper):
            block = transformer_block._checkpoint_wrapped_module
        else:
            block = transformer_block

        layer_type = getattr(block, "layer_type", None)
        if layer_type == "linear_attention":
            # Skip compiling linear_attn (KDA kernel) — it is not
            # Dynamo-friendly and the compile warmup allocates large
            # temporary buffers that cause OOM without meaningful
            # speed improvement.  Only compile the FFN sub-module.
            if hasattr(block, "feed_forward"):
                block.feed_forward = torch.compile(
                    block.feed_forward,
                    backend=compile_config.backend,
                    fullgraph=True,
                )
                compiled_linear_ffn += 1
        else:
            transformer_block = torch.compile(
                transformer_block,
                backend=compile_config.backend,
                fullgraph=True,
            )
            compiled_full_blocks += 1

        model.layers.register_module(layer_id, transformer_block)

    logger.info(
        "Compiling Qwen3.5 blocks with hybrid policy: "
        f"{compiled_full_blocks} full-attention blocks fullgraph, "
        f"{compiled_linear_attn} linear-attention modules partial graph, "
        f"{compiled_linear_ffn} linear-attention FFNs fullgraph"
    )


def parallelize_qwen3_5(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    assert (
        job_config.training.seq_len % parallel_dims.seq_len_divisor == 0
    ), (
        f"Sequence length {job_config.training.seq_len} must be divisible by "
        f"the product of TP degree ({parallel_dims.tp}) and "
        f"2 * CP degree ({parallel_dims.cp})."
    )

    model_compile_enabled = (
        job_config.compile.enable and "model" in job_config.compile.components
    )

    if parallel_dims.tp_enabled:
        if (
            job_config.parallelism.enable_async_tensor_parallel
            and not model_compile_enabled
        ):
            raise RuntimeError("Async TP requires torch.compile")

        enable_float8_linear = "float8" in job_config.model.converters
        float8_is_rowwise = job_config.quantize.linear.float8.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )
        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise

        tp_mesh = parallel_dims.get_mesh("tp")
        apply_non_moe_tp(
            model,
            tp_mesh,
            loss_parallel=not job_config.parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
            enable_async_tp=job_config.parallelism.enable_async_tensor_parallel,
        )

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        dual_pipe_v = get_dual_pipe_v_flag(job_config, parallel_dims)

        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            etp_mesh=parallel_dims.get_optional_mesh("etp"),
            ep_etp_mesh=parallel_dims.get_optional_mesh(["ep", "etp"]),
            dual_pipe_v=dual_pipe_v,
        )

    if job_config.activation_checkpoint.mode != "none":
        apply_ac(
            model,
            job_config.activation_checkpoint,
            model_compile_enabled=model_compile_enabled,
            op_sac_save_list=_op_sac_save_list,
            base_folder=job_config.job.dump_folder,
        )

    if model_compile_enabled:
        apply_compile(model, job_config.compile, False)

    if parallel_dims.fsdp_enabled:
        dp_mesh_names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=job_config.training.enable_cpu_offload,
            reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
            ep_degree=parallel_dims.ep,
            edp_mesh=edp_mesh,
            gradient_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if parallel_dims.cp_enabled:
            logger.info("Applied Context Parallel to the model")

        if job_config.training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")
    elif parallel_dims.dp_replicate_enabled:
        dp_mesh = parallel_dims.get_mesh("dp_replicate")
        if dp_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(
            model,
            dp_mesh,
            enable_compile=model_compile_enabled,
        )

    if model.model_args.enable_weight_tying:
        model.output.weight = model.tok_embeddings.weight

    return model


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
    enable_async_tp: bool,
):
    """Apply tensor parallelism to Qwen3.5."""
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        },
    )

    if enable_float8_tensorwise_tp:
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            PrepareFloat8ModuleInput,
        )
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
            PrepareFloat8ModuleInput,
        )
    else:
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )

    for transformer_block in model.layers.values():
        layer_type = transformer_block.layer_type

        layer_plan = {
            "input_layernorm": SequenceParallel(),
            "post_attention_layernorm": SequenceParallel(),
        }

        if not transformer_block.moe_enabled:
            layer_plan.update({
                "feed_forward": prepare_module_input(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
                "feed_forward.w1": colwise_parallel(),
                "feed_forward.w2": rowwise_parallel(output_layouts=Shard(1)),
                "feed_forward.w3": colwise_parallel(),
            })

        if layer_type == "full_attention":
            layer_plan.update({
                "self_attn": prepare_module_input(
                    input_layouts=(Shard(1), Replicate(), None, None),
                    desired_input_layouts=(Replicate(), Replicate(), None, None),
                ),
                "self_attn.wq": colwise_parallel(use_local_output=False),
                "self_attn.wk": colwise_parallel(use_local_output=False),
                "self_attn.wv": colwise_parallel(use_local_output=False),
                "self_attn.wo": rowwise_parallel(output_layouts=Shard(1)),
            })
            if hasattr(transformer_block, "self_attn") and transformer_block.self_attn.q_norm is not None:
                layer_plan["self_attn.q_norm"] = SequenceParallel(sequence_dim=2)
                layer_plan["self_attn.k_norm"] = SequenceParallel(sequence_dim=2)
        elif layer_type == "linear_attention":
            layer_plan["linear_attn"] = NoParallel(
                input_layout=Shard(1),
                output_layout=Shard(1),
                use_local_output=True,
            )

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    if enable_async_tp:
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group
        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(tp_mesh.get_group().group_name)

    logger.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}"
        f"{'Async ' if enable_async_tp else ''}Tensor Parallelism to the model"
    )

