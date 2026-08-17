# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import enum
import functools
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType
from torch.distributed.checkpoint.stateful import Stateful

# Feature-detected imports for optional checkpoint features.
try:
    from torch.distributed.checkpoint import HuggingFaceStorageWriter
except ImportError:
    HuggingFaceStorageWriter = None

try:
    from torch.distributed.checkpoint._consolidate_hf_safetensors import (
        consolidate_safetensors_files_on_every_rank,
    )
except ImportError:
    consolidate_safetensors_files_on_every_rank = None  # type: ignore[assignment]

try:
    from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
except ImportError:
    DefaultStager = None  # type: ignore[assignment, misc]
    StagingOptions = None  # type: ignore[assignment, misc]

try:
    from torch.distributed.checkpoint.state_dict_saver import AsyncSaveResponse
except ImportError:
    AsyncSaveResponse = None  # type: ignore[assignment]

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.ft import FTManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Checkpoint as CheckpointConfig, TORCH_DTYPE_MAP
from torchtitan.protocols import BaseStateDictAdapter
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import GarbageCollection, cache_s3_path_if_needed


MODEL = "model"
OPTIMIZER = "optimizer"
LR_SCHEDULER = "lr_scheduler"
DATALOADER = "dataloader"
TRAIN_STATE = "train_state"

# Mapping from loading plan source/target names to internal state keys.
# Includes aliases for Arceus-style plan strings (ckpt_scheduler, ckpt_step, etc.).
_CKPT_SOURCE_TO_STATE_KEY = {
    "ckpt_model": MODEL,
    "ckpt_optimizer": OPTIMIZER,
    "ckpt_lr_scheduler": LR_SCHEDULER,
    "ckpt_scheduler": LR_SCHEDULER,  # alias
    "ckpt_train_state": TRAIN_STATE,
    "ckpt_step": TRAIN_STATE,  # alias
    "ckpt_dataloader": DATALOADER,
}
_MEM_TARGET_TO_STATE_KEY = {
    "mem_model": MODEL,
    "mem_optimizer": OPTIMIZER,
    "mem_lr_scheduler": LR_SCHEDULER,
    "mem_scheduler": LR_SCHEDULER,  # alias
    "mem_train_state": TRAIN_STATE,
    "mem_step": TRAIN_STATE,  # alias
    "mem_dataloader": DATALOADER,
}


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC = "async"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


class ModelWrapper(Stateful):
    def __init__(self, model: nn.Module | list[nn.Module]) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        self.cache_state_dict = self._get_state_dict()

    def _get_state_dict(self) -> dict[str, Any]:
        state_dict = {
            k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()
        }
        return state_dict

    def state_dict(self) -> dict[str, Any]:
        return self.cache_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))
        # `set_model_state_dict()` does change the keys of the input state_dict,
        # we will need to reinitialize the cache_state_dict.
        self.cache_state_dict = self._get_state_dict()


class Terminate:
    pass


class SaveDone:
    pass


def purge_thread(purge_queue: queue.Queue):
    """Thread to purge the old checkpoints.

    This is only used when keep_latest_k > 0.

    Args:
        purge_queue (queue.Queue): The queue to receive the path to purge and Terminate signal.
    """
    try:
        while True:
            path = purge_queue.get()
            if isinstance(path, Terminate):
                return
            assert isinstance(path, str)
            logger.info("Checkpointer is deleting %s.", path)
            begin = time.monotonic()
            shutil.rmtree(path, ignore_errors=True)
            logger.info(
                "Checkpointer deleted %s in %.2f seconds.",
                path,
                time.monotonic() - begin,
            )
    finally:
        logger.info("Destroying the purge thread.")


class CheckpointManager:
    """This class manages the checkpointing logic for the TorchTitan trainer.


    Note: Pipeline Parallelism and Virtual Stages

    1. even for simple PP schedules, there is a separate optimizer each PP rank.
    rank0's optimizer would have a param_group[0] which refers to layers.0 in the original
    model.  rank1's would _also_ have a param_group[0], since it's index based, but
    referring to layers.1.  When saving, these collide and one of them is lost.  Then when
    reloading, only one stage can restore its optimizer states, others will error.

        The solution to this problem is optimizer flattening: it landed in #127071 and is
        enabled in TorchTitan by passing the 'flatten_optimizer_state_dict' kwarg to DCP
        functions called in the OptimizerContainer.
        See PR #127071 (https://github.com/pytorch/pytorch/pull/127071) for the example of
        a flattening state_dict.

    2. With complex PP schedules, we have multiple model chunks per pp rank. This compounds
    challenge (1) by also requiring us to reason about multiple 'optim' objects locally.

        We solve this in the Model and Optimizer wrapper classes by flattening the state dicts
        from each object into one state dict before saving/loading. We rely on the individual
        state_dicts to not collide, which is guaranteed for the model by correct pipeline
        splitting and for the optimizer by the flattening support described in (1).

    3. LR schedulers also index model states like optimizers. Here we flatten the lr_schedulers
    with the assumption that all lr_schedulers have the same state_dict.

    Note: TorchFT checkpointing flow

    There are two types of checkpoints: when TorchFT is enabled: 1) the full persistent
    checkpoint, 2) the per-replica checkpoint.

    The full persistent checkpoint is saved by the replica with
    ``ft_manager.participating_rank() == 0``. It contains everything including the model,
    optimizer, lr_scheduler, dataloader, and train_state. Right now the full persistent
    checkpoint is loaded by all replicas. However, we can optimize it to only load if
    there are no other alive replicas.

    The per-replica checkpoint contains only the dataloader and is saved/loaded by all
    replicas to/from the its own folder. The folder name is prefixed with the ft_replica_id.

    Args:
        dataloader (DataLoader): The dataloader used to load the data.
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizers (OptimizersContainer): The optimizers used to optimize the model.
        lr_schedulers (LRSchedulersContainer): The lr schedulers used to optimize the model.
        states (Dict[str, Any]): The states that need to be saved, other than the
            previous 4 components.
        checkpoint_config (Checkpoint): The config used to configure the checkpointing.
        base_folder (str): The base folder to save the checkpoint. Will be concatenated
            with checkpoint_config.folder
        sd_adapter (Optional[type[BaseStateDictAdapter]]): The adapter used to convert model state
            dicts between native format and other formats.
        ft_manager (Optional[ft.Manager]): The FTManager from TorchFT.

    """

    mp_queue_send: queue.Queue
    pg: dist.ProcessGroup
    purge_thread: threading.Thread | None

    def __init__(
        self,
        dataloader: BaseDataLoader | None,
        model_parts: list[nn.Module],
        optimizers: OptimizersContainer,
        lr_schedulers: LRSchedulersContainer,
        states: dict[str, Any],
        checkpoint_config: CheckpointConfig,
        sd_adapter: BaseStateDictAdapter | None,
        base_folder: str = "",
        ft_manager: FTManager | None = None,
        config_file_path: str | None = None,
        trial_name: str = "",
        run_name: str = "",
    ) -> None:
        self.enable = checkpoint_config.enable
        self.load_only = checkpoint_config.load_only

        self.states = states
        self.states.update(
            {
                MODEL: ModelWrapper(model_parts),
                OPTIMIZER: optimizers,
                DATALOADER: dataloader,
                LR_SCHEDULER: lr_schedulers,
            }
        )

        self.ft_manager = (
            ft_manager.manager if ft_manager and ft_manager.enabled else None
        )

        self.enable_ft_dataloader_checkpoints = (
            self.ft_manager and checkpoint_config.enable_ft_dataloader_checkpoints
        )

        if self.ft_manager and not self.enable_ft_dataloader_checkpoints:
            logger.warning(
                "Fault tolerance is enabled but enable_ft_dataloader_checkpoints is False. "
                "This means replicas can retrain over the same data multiple times, which can result in overfitting."
            )

        if self.ft_manager:
            optimizers.init_cache_state_dict()

            def state_dict():
                ret = {}
                for k, v in self.states.items():
                    if k in {
                        MODEL,
                        OPTIMIZER,
                        LR_SCHEDULER,
                        TRAIN_STATE,
                    }:
                        ret[k] = v.state_dict()
                return ret

            def load_state_dict(state_dict):
                assert state_dict is not None
                for k, v in state_dict.items():
                    self.states[k].load_state_dict(v)

            # pyrefly: ignore [missing-attribute]
            self.ft_manager.set_state_dict_fns(load_state_dict, state_dict)
            assert ft_manager is not None
            self.ft_replica_id = ft_manager.replica_id

        async_mode = checkpoint_config.async_mode.lower()
        self.enable_staging = (
            self.enable and async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM
        ) or self.enable_ft_dataloader_checkpoints

        if not self.enable and not self.enable_ft_dataloader_checkpoints:
            return

        self.ft_states = {DATALOADER: dataloader}

        self.staging = False
        self.sending_to_checkpoint_mp = False
        self.staging_id = None
        self.cpu_offload_state_dict = None
        self.stager = None

        # Build identifiers for path construction
        user_name = os.environ.get("USER", "default")
        user_name = user_name.replace(" ", "_").lower()
        if config_file_path:
            config_basename = os.path.splitext(os.path.basename(config_file_path))[0]
        else:
            config_basename = "default"

        if not run_name:
            if trial_name != "":
                run_name = config_basename + "_" + trial_name
            else:
                run_name = config_basename

        # Path structure: {base_folder}/{user_name}/{run_name}/{checkpoint_folder}
        self.folder = os.path.join(base_folder, user_name, run_name, checkpoint_config.folder)

        # S3 upload path
        if checkpoint_config.s3_upload_path:
            self.s3_upload_path = f"{checkpoint_config.s3_upload_path.rstrip('/')}/{user_name}/{run_name}"
        else:
            self.s3_upload_path = None

        # Checkpoint policy related fields.
        self.initial_load_model_only = checkpoint_config.initial_load_model_only
        self.initial_ckpt_load_plan = checkpoint_config.initial_ckpt_load_plan
        self.initial_load_in_hf = checkpoint_config.initial_load_in_hf
        # Cache S3 path to local if needed, strip s3_upload_path prefix for cleaner local paths
        self.initial_load_path = cache_s3_path_if_needed(
            checkpoint_config.initial_load_path,
            category="checkpoints",
            s3_base_prefix=checkpoint_config.s3_upload_path,
        )
        self.initial_load_in_hf_quantized = (
            checkpoint_config.initial_load_in_hf_quantized
        )
        self.last_save_model_only = checkpoint_config.last_save_model_only
        self.last_save_in_hf = checkpoint_config.last_save_in_hf
        if self.last_save_in_hf:
            assert (
                sd_adapter is not None
            ), "job_config.checkpoint.last_save_in_hf is True, but sd_adapter is not provided."
        self.sd_adapter = sd_adapter
        self.export_dtype = TORCH_DTYPE_MAP[checkpoint_config.export_dtype]
        self.exclude_from_loading = checkpoint_config.exclude_from_loading
        self.interval = checkpoint_config.interval
        self.enable_first_step_checkpoint = (
            checkpoint_config.enable_first_step_checkpoint
        )

        # Async checkpoint related fields.
        async_mode = checkpoint_config.async_mode.lower()
        if (
            async_mode == AsyncMode.ASYNC
            or async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM
            or self.enable_ft_dataloader_checkpoints
        ):
            self.pg = cast(dist.ProcessGroup, dist.new_group(backend="gloo"))

        self.keep_latest_k = checkpoint_config.keep_latest_k
        if self.keep_latest_k > 0:
            if self.keep_latest_k == 1:
                raise ValueError(
                    "We need to maintain at least 2 checkpoint replicas, "
                    "as the last one may be in the process of being saved."
                )
            self.purge_queue = queue.Queue()
            self.purge_thread = threading.Thread(
                target=purge_thread, args=(self.purge_queue,), daemon=True
            )
            self.purge_thread.start()
        else:
            self.purge_thread = None

        self.mp = None
        self.staging_future = None
        self.save_future = None
        if async_mode == AsyncMode.DISABLED:
            self.async_mode = AsyncMode.DISABLED
        elif async_mode == AsyncMode.ASYNC:
            self.async_mode = AsyncMode.ASYNC
        elif async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEM
        else:
            raise ValueError(
                f"Unknown checkpoint async_mode {checkpoint_config.async_mode}"
            )

        logger.info(
            f"Checkpointing active. Checkpoints will be loaded from and saved to {self.folder}"
        )

    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, "enable") and self.enable:
            if hasattr(self, "mp") and self.mp and self.mp.is_alive():
                self.mp_queue_send.put(Terminate())
                self.mp.join()
            if (
                hasattr(self, "purge_thread")
                and self.purge_thread
                and self.purge_thread.is_alive()
            ):
                self.purge_queue.put(Terminate())
                self.purge_thread.join()

            if self.stager is not None:
                self.stager.close()

    @torch.no_grad()
    def dcp_save(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        async_mode: AsyncMode,
        enable_garbage_collection: bool = False,
        to_hf: bool = False,
    ) -> "Future | AsyncSaveResponse | None":
        """Save the checkpoint with dcp.
        Args:
            state_dict (dict): The state dict to save.
            checkpoint_id (str): The checkpoint id to save.
            async_mode (AsyncMode): Whether the checkpoint is async.
            enable_garbage_collection (bool): Whether to enable garbage collection after save.
            to_hf (bool): Whether to save in HF model definition and safetensors format.

        Returns:
            Future: The future object if the checkpoint is async, otherwise None.
        """

        ret = None

        storage_writer = None
        checkpoint_save_id: str | None = None
        fqn_to_index_mapping: dict[Any, int] | None = None
        if to_hf:
            assert (
                self.sd_adapter is not None
            ), "trying to save checkpoint in HF safetensors format, but sd_adapter is not provided."
            if HuggingFaceStorageWriter is None:
                raise ImportError(
                    "Saving in HF format requires torch >= 2.10 (HuggingFaceStorageWriter). "
                    "Please upgrade PyTorch or disable last_save_in_hf."
                )
            state_dict = self.sd_adapter.to_hf(state_dict)

            fqn_to_index_mapping = self.sd_adapter.fqn_to_index_mapping
            if fqn_to_index_mapping:
                storage_writer = HuggingFaceStorageWriter(
                    path=os.path.join(checkpoint_id, "sharded"),
                    save_distributed=True,
                    fqn_to_index_mapping=fqn_to_index_mapping,
                    enable_consolidation=False,
                )
            else:
                # the reason for only enabling consolidation if there is
                # no mapping is because no mapping implies that we save all fqns
                # to one file. This means we only need one rank to consolidate.
                # Otherwise we should use consolidate_safetensors_files_on_every_rank
                storage_writer = HuggingFaceStorageWriter(
                    path=checkpoint_id,
                    save_distributed=True,
                    enable_consolidation=True,
                )

        else:
            checkpoint_save_id = checkpoint_id

        if async_mode == AsyncMode.ASYNC:
            ret = dcp.async_save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
                process_group=self.pg,
            )
        elif async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            ret = dcp.async_save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
                process_group=self.pg,
                async_checkpointer_type=AsyncCheckpointerType.PROCESS,
                async_stager=self.stager,
            )
        else:
            ret = dcp.save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
            )

        if to_hf and fqn_to_index_mapping:
            if consolidate_safetensors_files_on_every_rank is None:
                raise ImportError(
                    "HF safetensors consolidation requires torch >= 2.10."
                )
            consolidate_safetensors_files_on_every_rank(
                input_dir=os.path.join(checkpoint_id, "sharded"),
                output_dir=checkpoint_id,
                fqn_to_index_mapping=fqn_to_index_mapping,
                num_threads=5,
            )

        if enable_garbage_collection:
            GarbageCollection.collect("GC collection invoked by checkpointer.")

        return ret

    def dcp_load(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        from_hf: bool,
        from_quantized: bool,
    ) -> None:
        """Load the checkpoint with dcp.
        Args:
            state_dict (dict): The state dict to load.
            checkpoint_id (str): The checkpoint id to load.
            from_hf (bool): Whether to load from HuggingFace checkpoint with
                its own model definition and safetensors format.
        """

        if from_hf:
            assert (
                self.sd_adapter is not None
            ), "trying to load checkpoint in HF safetensors format, but sd_adapter is not provided."
            hf_state_dict = self.sd_adapter.to_hf(state_dict)
            hf_storage_reader = self.sd_adapter.get_hf_storage_reader(
                checkpoint_id, from_quantized
            )

            dcp.load(
                hf_state_dict,
                storage_reader=hf_storage_reader,
            )

            state_dict = self.sd_adapter.from_hf(hf_state_dict)
            self.states[MODEL].load_state_dict(state_dict)
        else:
            dcp.load(state_dict, checkpoint_id=checkpoint_id)

            # TODO: Since we flatten the model states in state_dict, we need to
            # manually call load_state_dict() for the model. Need to fix this.
            if MODEL in self.states:
                self.states[MODEL].load_state_dict(state_dict)

    @torch.no_grad()
    def save(self, curr_step: int, last_step: bool = False) -> None:
        """Save the checkpoint for the current step.

        This function will save the checkpoint for the current step. If ``last_step`` is
        true, it will save the checkpoint even if the interval has not been reached.
        This only happens when train_state.step == job_config.training.steps, or
        for initial seed checkpoint.

        Args:
            curr_step (int): The current step.
            last_step (bool, optional): Whether this is the last step of training.

        Returns:
            None
        """

        if self.enable_ft_dataloader_checkpoints:
            self._ft_save(curr_step)

        if not self._should_save(curr_step, last_step):
            return

        begin = time.monotonic()
        if not self.enable_ft_dataloader_checkpoints or (
            self.ft_manager
            # pyrefly: ignore [missing-attribute]
            and self.ft_manager.participating_rank() == 0
        ):
            logger.info("Saving the checkpoint (or staging if async is enabled).")
            checkpoint_id = self._create_checkpoint_id(curr_step)
            self._async_wait()
            # This GC is called for async checkpoint as it is useless to do
            # GC right after async_save -- the CPU memory is not able to be
            # freed until _async_wait()
            if last_step:
                self._save_last_step(curr_step)
                return

            states = self._flattened_model_states_sd()
            if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
                GarbageCollection.collect("GC collection invoked by checkpointer.")
                if self.stager is None:
                    if DefaultStager is None or StagingOptions is None:
                        raise ImportError(
                            "async_with_pinned_mem checkpoint mode requires torch >= 2.10 "
                            "(DefaultStager/StagingOptions). Use async or disabled mode instead."
                        )
                    self.stager = DefaultStager(StagingOptions(True, True, True, True))
                result = self.dcp_save(
                    states,
                    checkpoint_id=checkpoint_id,
                    async_mode=self.async_mode,
                )
                assert AsyncSaveResponse is not None and isinstance(result, AsyncSaveResponse)
                self.save_future = result.upload_completion
                self.staging_future = result.staging_completion
                self.staging = True
                if self.s3_upload_path:
                    def upload_after_async():
                        self.save_future.result()
                        self._upload_checkpoint_to_s3(checkpoint_id)
                    threading.Thread(target=upload_after_async, daemon=True).start()
            elif self.async_mode == AsyncMode.ASYNC:
                GarbageCollection.collect("GC collection invoked by checkpointer.")
                self.save_future = self.dcp_save(
                    states, checkpoint_id=checkpoint_id, async_mode=self.async_mode
                )
                GarbageCollection.collect("GC collection invoked by checkpointer.")
                if self.s3_upload_path:
                    def upload_after_async():
                        self.save_future.result()
                        self._upload_checkpoint_to_s3(checkpoint_id)
                    threading.Thread(target=upload_after_async, daemon=True).start()
            else:
                self.dcp_save(
                    states,
                    checkpoint_id=checkpoint_id,
                    async_mode=AsyncMode.DISABLED,
                    enable_garbage_collection=True,
                )
                if self.s3_upload_path:
                    self._upload_checkpoint_to_s3(checkpoint_id)
            self._purge_stale_checkpoints()

            logger.info(
                "Finished saving the checkpoint (or staging if async is enabled)"
                f"in {time.monotonic() - begin:.2f} seconds."
            )
        elif self.enable_ft_dataloader_checkpoints:
            assert self.ft_manager is not None
            logger.info(
                "Replica %d doesn't save checkpoint.",
                # pyrefly: ignore [missing-attribute]
                self.ft_manager.participating_rank(),
            )

    @torch.no_grad()
    def load(self, step: int = -1) -> bool:
        """Load the checkpoint for the given step.

        This function will load the checkpoint for the given step. If ``step`` is -1, it
        will load the latest checkpoint. If the checkpoint does not exist, it will return
        False and load nothing.

        Args:
            step (int, optional): The step to load the checkpoint for. Defaults to -1.

        Returns:
            bool: Whether the checkpoint was loaded successfully.
        """

        if self.enable_ft_dataloader_checkpoints:
            self._ft_load()

        if not self.enable:
            return False

        model_only = False
        from_hf = False
        from_quantized = False
        # Check for valid local checkpoints (step-* dirs with .metadata + shard files).
        # An empty or stale folder (e.g. created by TrainingDataLogger before load()
        # runs) should not prevent loading the seed checkpoint from initial_load_path.
        folder_exists = os.path.isdir(self.folder)
        local_step = self._find_load_step() if (folder_exists and step == -1) else step
        has_valid_checkpoint = folder_exists and local_step != -1

        if not has_valid_checkpoint:
            if folder_exists:
                logger.warning(
                    f"checkpoint.folder {self.folder} exists but contains no valid "
                    "checkpoints (need step-* with .metadata + shard files). "
                    "Falling through to initial_load_path."
                )
            model_only = self.initial_load_model_only
            from_hf = self.initial_load_in_hf
            from_quantized = self.initial_load_in_hf_quantized
            use_load_plan = self.initial_ckpt_load_plan is not None
            if from_hf:
                assert (
                    model_only or use_load_plan
                ), "Only model can be loaded when loading from HF's safetensors checkpoint."

            if from_quantized:
                assert (
                    from_hf
                ), "Quantized checkpoint can only be loaded from HuggingFace format."

            if self.initial_load_path:
                checkpoint_id = self.initial_load_path
                if not os.path.isdir(checkpoint_id):
                    raise ValueError(
                        "checkpoint.initial_load_path is specified but the path is not valid."
                    )
                if from_hf:
                    logger.info(
                        f"loading from HF safetensors from --checkpoint.initial_load_path: {self.initial_load_path}"
                    )
            elif from_hf:
                assert (
                    self.sd_adapter is not None
                    and self.sd_adapter.hf_assets_path is not None
                ), "from_hf is True but sd_adapter or hf_assets_path is not provided."
                hf_assets_path = self.sd_adapter.hf_assets_path
                checkpoint_id = hf_assets_path
                if not os.path.isdir(checkpoint_id):
                    raise ValueError(
                        "model.hf_assets_path is being used to load HF weights but the path is not valid. \
                        Either make sure hf_assets_path is correct or provide a valid checkpoint.initial_load_path"
                    )
                logger.info(
                    f"loading HF safetensors from --model.hf_assets_path: {hf_assets_path}"
                )
            else:
                return False
        else:
            use_load_plan = False  # Only use load plan for initial loading
            step = local_step if step == -1 else step
            model_only = step == 0
            checkpoint_id = self._create_checkpoint_id(step)

            if not os.path.isdir(checkpoint_id):
                raise FileNotFoundError(
                    f"--checkpoint.load_step={step} but checkpoint {checkpoint_id} is not found."
                )

        logger.info(f"Loading the checkpoint from {checkpoint_id}.")
        begin = time.monotonic()

        if use_load_plan:
            # Use flexible loading plan: load specified components, skip missing
            assert self.initial_ckpt_load_plan is not None
            logger.info(
                f"Using initial_ckpt_load_plan: {self.initial_ckpt_load_plan}"
            )
            self._load_with_plan(
                checkpoint_id=checkpoint_id,
                loading_plan=self.initial_ckpt_load_plan,
                from_hf=from_hf,
                from_quantized=from_quantized,
            )
        else:
            # Standard loading path
            states = self._states_to_load(model_only)
            self.dcp_load(
                states,
                checkpoint_id=checkpoint_id,
                from_hf=from_hf,
                from_quantized=from_quantized,
            )

        GarbageCollection.collect("GC collection for checkpoint loading.")
        logger.info(
            f"Finished loading the checkpoint in {time.monotonic() - begin:.2f} seconds."
        )
        return True

    def maybe_wait_for_staging(self) -> None:
        """Wait for the staging to finish if it is enabled.

        This function will wait for staging to finish. The staging is only enabled
        with ``async_checkpoint_with_pinned_memory``.
        """
        if self.enable_staging and self.staging:
            assert self.staging_future is not None
            self.staging_future.result()
            self.staging = False

    def _find_load_step(self, folder: str = "") -> int:
        """Find the step to load the checkpoint for.

        Args:
            folder (str, optional): The folder to find the checkpoint for. If ``folder``
            is "", then ``self.folder`` will be used.

        Returns:
            int: The step to load the checkpoint for.
        """
        folder = folder if folder else self.folder
        pattern = r"step-(\d+)"
        step_counts = []

        if not os.path.isdir(folder):
            return -1

        for filename in os.listdir(folder):
            match = re.search(pattern, filename)
            step_dir = os.path.join(folder, filename)
            dcp_metadata_probe = os.path.join(step_dir, ".metadata")
            safetensors_metadata_probe = os.path.join(
                step_dir, "model.safetensors.index.json"
            )
            if match and os.path.isfile(dcp_metadata_probe):
                # Require at least one DCP shard file alongside .metadata
                if any(f.endswith(".distcp") for f in os.listdir(step_dir)):
                    step_counts.append(int(match.group(1)))
            elif match and os.path.isfile(safetensors_metadata_probe):
                # Require at least one safetensors shard file
                if any(f.endswith(".safetensors") for f in os.listdir(step_dir)):
                    step_counts.append(int(match.group(1)))
        if not step_counts:
            return -1
        return max(step_counts)

    def _ft_folder(self) -> str:
        return os.path.join(self.folder, f"ft-replicat-{self.ft_replica_id}")

    def _create_checkpoint_id(self, step: int, folder: str = "") -> str:
        folder = folder if folder else self.folder
        return os.path.join(folder, f"step-{step}")

    def _ft_save(self, step: int) -> None:
        begin = time.monotonic()
        self._async_wait()
        checkpoint_id = self._create_checkpoint_id(step, folder=self._ft_folder())
        self.save_future = self.dcp_save(
            self.ft_states, checkpoint_id=checkpoint_id, async_mode=AsyncMode.ASYNC
        )
        logger.info(f"Staging ft checkpoint took {time.monotonic() - begin} secs.")

    def _ft_load(self) -> None:
        step = self._find_load_step(folder=self._ft_folder())
        if step == -1:
            return

        begin = time.monotonic()
        logger.info(f"Loading the FT checkpoint at step {step}.")
        checkpoint_id = self._create_checkpoint_id(step, folder=self._ft_folder())
        self.dcp_load(
            self.ft_states,
            checkpoint_id=checkpoint_id,
            # FT checkpoints are always DCP because FT checkpoint currently only save/load dataloader.
            from_hf=False,
            from_quantized=False,
        )
        GarbageCollection.collect("GC collection for checkpoint loading.")
        logger.info(
            f"Finished loading the ft checkpoint in {time.monotonic() - begin:.2f} seconds."
        )

    def _upload_checkpoint_to_s3(self, checkpoint_id: str) -> None:
        """Upload checkpoint to S3. Each rank uploads its own shard files, rank 0 uploads meta files.

        Args:
            checkpoint_id: Local checkpoint path, e.g. /path/to/checkpoints/step-1000
        """
        if not self.s3_upload_path:
            return

        checkpoint_path = Path(checkpoint_id)
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint path does not exist: {checkpoint_path}")
            return

        step_name = checkpoint_path.name
        s3_ckpt_dir = f"{self.s3_upload_path.rstrip('/')}/{step_name}"

        rank = dist.get_rank() if dist.is_initialized() else 0

        try:
            for shard_file in checkpoint_path.glob(f"__{rank}_*.distcp"):
                s3_url = f"{s3_ckpt_dir}/{shard_file.name}"
                self._upload_file_to_s3(str(shard_file), s3_url)

            if rank == 0:
                for meta_file in [".metadata", "shared.pth"]:
                    local_path = checkpoint_path / meta_file
                    if local_path.exists():
                        s3_url = f"{s3_ckpt_dir}/{meta_file}"
                        self._upload_file_to_s3(str(local_path), s3_url)

                for safetensors_file in checkpoint_path.glob("*.safetensors"):
                    s3_url = f"{s3_ckpt_dir}/{safetensors_file.name}"
                    self._upload_file_to_s3(str(safetensors_file), s3_url)

                for model_index_file in checkpoint_path.glob("model*.safetensors.index.json"):
                    s3_url = f"{s3_ckpt_dir}/{model_index_file.name}"
                    self._upload_file_to_s3(str(model_index_file), s3_url)

            if rank == 0:
                logger.info(f"Successfully uploaded checkpoint {step_name} to {s3_ckpt_dir}")
        except Exception as e:
            logger.error(f"Failed to upload checkpoint to S3: {e}")

    def _upload_file_to_s3(self, local_path: str, s3_url: str) -> None:
        """Upload a single file to S3 using aws s3 cp command.

        Args:
            local_path: Local file path
            s3_url: S3 URL (s3://bucket/path/to/file)
        """
        try:
            cmd = ["aws", "s3", "cp", local_path, s3_url]
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True, timeout=3600
            )
            if result.returncode != 0:
                logger.error(f"Failed to upload {local_path} to {s3_url}: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout uploading {local_path} to {s3_url}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to upload {local_path} to {s3_url}: {e.stderr}")
        except Exception as e:
            logger.error(f"Unexpected error uploading {local_path} to {s3_url}: {e}")

    def _flattened_model_states_sd(
        self, state_dict: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Flatten the model states into a single dictionary.

        Note that other states, such as optimizer states, are not flattened.
        """
        states = state_dict if state_dict is not None else self.states
        sd = {k: v for k, v in states.items() if k != MODEL}
        if MODEL in states:
            sd.update(states[MODEL].state_dict())
        return sd

    def _states_to_load(self, model_only: bool) -> dict[str, Any]:
        """Determines which states to load for the given step.

        This API is used to determine which states to load based on the
        configurations.

        Args:
            model_only (bool): Whether to load the model only.

        Returns:
            Dict[str, Any]: The states to load for the given step.
        """
        # For the first step, we will only load the model.
        if model_only:
            return self.states[MODEL].state_dict()

        for exclude_key in self.exclude_from_loading:
            if exclude_key not in self.states:
                raise ValueError(f"{exclude_key} not found in state_dict.")

        states_to_load = {
            k: v for k, v in self.states.items() if k not in self.exclude_from_loading
        }

        states_to_load = self._flattened_model_states_sd(states_to_load)

        if self.enable_ft_dataloader_checkpoints:
            states_to_load.pop(DATALOADER)

        return states_to_load

    @staticmethod
    def _parse_loading_plan(loading_plan: str) -> dict[str, str]:
        """Parse a loading plan string into a mapping of checkpoint sources to memory targets.

        Args:
            loading_plan: Comma-separated 'ckpt_source:mem_target' pairs.
                Example: 'ckpt_model:mem_model,ckpt_optimizer:mem_optimizer'

        Returns:
            Dictionary mapping checkpoint source keys to memory target keys.
        """
        plan_mapping: dict[str, str] = {}
        for mapping in loading_plan.split(","):
            mapping = mapping.strip()
            if not mapping:
                continue
            if ":" not in mapping:
                raise ValueError(
                    f"Invalid loading plan format: '{mapping}'. Expected 'source:target'"
                )
            source, target = mapping.split(":", 1)
            source, target = source.strip(), target.strip()

            if source not in _CKPT_SOURCE_TO_STATE_KEY:
                raise ValueError(
                    f"Invalid source '{source}'. Valid: {list(_CKPT_SOURCE_TO_STATE_KEY)}"
                )
            if target not in _MEM_TARGET_TO_STATE_KEY:
                raise ValueError(
                    f"Invalid target '{target}'. Valid: {list(_MEM_TARGET_TO_STATE_KEY)}"
                )
            plan_mapping[source] = target
        return plan_mapping

    def _load_with_plan(
        self,
        checkpoint_id: str,
        loading_plan: str,
        from_hf: bool = False,
        from_quantized: bool = False,
    ) -> None:
        """Load checkpoint components according to a loading plan, skipping missing ones.

        Parses the loading plan, probes the checkpoint metadata to find which
        components are present, builds a state dict for only those, loads via DCP,
        and applies loaded model states. Non-model Stateful objects (optimizer,
        lr_scheduler, train_state, dataloader) are loaded in-place by DCP.

        Components listed in the plan but missing from the checkpoint are skipped.

        Args:
            checkpoint_id: Path to the checkpoint directory.
            loading_plan: Comma-separated 'ckpt_source:mem_target' pairs.
            from_hf: Whether the checkpoint is in HuggingFace format.
            from_quantized: Whether the checkpoint is quantized HF format.
        """
        plan_mapping = self._parse_loading_plan(loading_plan)
        logger.info(f"Loading plan: {plan_mapping}")

        if from_hf:
            # HF loading only supports model; delegate to standard path
            logger.info("HF checkpoint detected with load plan; loading model only.")
            self.dcp_load(
                self.states[MODEL].state_dict(),
                checkpoint_id=checkpoint_id,
                from_hf=True,
                from_quantized=from_quantized,
            )
            return

        # Probe checkpoint metadata to find which top-level keys exist
        available_keys: set[str] = set()
        try:
            metadata_path = os.path.join(checkpoint_id, ".metadata")
            if os.path.isfile(metadata_path):
                reader = dcp.FileSystemReader(checkpoint_id)
                ckpt_metadata = reader.read_metadata()
                all_fqns = set(ckpt_metadata.state_dict_metadata.keys())
                # Non-model states have their state key as prefix
                for state_key in [OPTIMIZER, LR_SCHEDULER, TRAIN_STATE, DATALOADER]:
                    prefix = f"{state_key}."
                    if any(k.startswith(prefix) or k == state_key for k in all_fqns):
                        available_keys.add(state_key)
                # Model keys are flattened at top level (no "model." prefix)
                other_prefixes = tuple(
                    f"{k}." for k in [OPTIMIZER, LR_SCHEDULER, TRAIN_STATE, DATALOADER]
                )
                if any(not k.startswith(other_prefixes) for k in all_fqns):
                    available_keys.add(MODEL)
                logger.info(f"Checkpoint contains components: {available_keys}")
            else:
                logger.warning(
                    f"No .metadata file at {metadata_path}. "
                    "Will attempt to load all planned components."
                )
                for source in plan_mapping:
                    available_keys.add(_CKPT_SOURCE_TO_STATE_KEY[source])
        except Exception as e:
            logger.warning(
                f"Could not probe checkpoint metadata: {e}. "
                "Will attempt to load all planned components."
            )
            for source in plan_mapping:
                available_keys.add(_CKPT_SOURCE_TO_STATE_KEY[source])

        # Build state dict for components that are both planned and available
        # DCP loads non-model Stateful objects in-place; model is flattened.
        states_to_load: dict[str, Any] = {}
        load_model = False
        loaded_components: list[str] = []

        for source, target in plan_mapping.items():
            source_key = _CKPT_SOURCE_TO_STATE_KEY[source]
            target_key = _MEM_TARGET_TO_STATE_KEY[target]

            if source_key not in available_keys:
                logger.info(
                    f"Skipping '{source}' -> '{target}': "
                    f"'{source_key}' not found in checkpoint."
                )
                continue

            if target_key not in self.states:
                logger.info(
                    f"Skipping '{source}' -> '{target}': "
                    f"target '{target_key}' not in current states."
                )
                continue

            if source_key == MODEL:
                # Model state is flattened at top level in the checkpoint
                model_sd = self.states[MODEL].state_dict()
                states_to_load.update(model_sd)
                load_model = True
            else:
                # Non-model Stateful objects: DCP loads directly into them
                states_to_load[source_key] = self.states[target_key]

            loaded_components.append(f"{source}->{target}")

        if not states_to_load:
            logger.info("No matching components found in checkpoint; nothing to load.")
            return

        # Load from checkpoint via DCP (allow_partial_load=True so missing
        # keys in checkpoint are silently skipped instead of raising errors,
        # e.g. when optimizer state doesn't match current model structure)
        logger.info(f"Loading components: {loaded_components}")
        dcp.load(
            states_to_load,
            checkpoint_id=checkpoint_id,
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )

        # Model needs explicit load_state_dict since it was flattened
        if load_model:
            self.states[MODEL].load_state_dict(states_to_load)
            logger.info("Applied model state dict after DCP load.")

    def _save_last_step(self, curr_step: int) -> None:
        # We only consider saving model only at the end of the training. So this
        # won't affect preemption and training resume. We also only allow dtype
        # conversion when we are checkpointing model only and the current dtype
        # is not the same as the export dtype at the end of the training.

        if self.last_save_model_only:
            states = self.states[MODEL].state_dict()

            if self.export_dtype != torch.float32:
                states = {k: v.to(self.export_dtype) for k, v in states.items()}
            logger.info(
                f"Saving a model only checkpoint in {self.export_dtype} "
                f"at last step, step {curr_step}."
            )
        else:
            logger.info(f"Saving a full checkpoint at last step, step {curr_step}.")
            states = self._flattened_model_states_sd()

        if self.last_save_in_hf:
            assert (
                self.last_save_model_only
            ), "Only model can be saved when saving in HF safetensors format."

        checkpoint_id = self._create_checkpoint_id(curr_step)
        self.dcp_save(
            states,
            checkpoint_id=checkpoint_id,
            async_mode=AsyncMode.DISABLED,
            enable_garbage_collection=True,
            to_hf=self.last_save_in_hf,
        )
        if self.s3_upload_path:
            self._upload_checkpoint_to_s3(checkpoint_id)

    def _should_save(self, curr_step: int, last_step: bool = False) -> bool:
        if not self.enable or self.load_only:
            return False

        if curr_step == 1 and self.enable_first_step_checkpoint:
            return True

        if last_step:
            return True

        if curr_step % self.interval == 0:
            return True

        return False

    def _async_wait(self) -> None:
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            if self.save_future is not None:
                self.save_future.result()
        elif (
            self.async_mode == AsyncMode.ASYNC or self.enable_ft_dataloader_checkpoints
        ):
            if self.save_future is not None:
                self.save_future.result()
                self.save_future = None
        elif self.save_future is not None:
            raise RuntimeError(
                "self.save_future is not None, but self.async_mode is not enabled "
                "and fault tolerance is not active."
            )

    def _purge_stale_checkpoints(self):
        if (
            self.keep_latest_k > 0
            and dist.get_rank() == 0
            and os.path.isdir(self.folder)
            and (
                not self.enable_ft_dataloader_checkpoints
                # pyrefly: ignore [missing-attribute]
                or (self.ft_manager and self.ft_manager.participating_rank() == 0)
            )
        ):
            discovered_checkpoints = []
            for filename in os.listdir(self.folder):
                match = re.search(r"step-(\d+)", filename)
                if match:
                    path = os.path.join(self.folder, filename)
                    discovered_checkpoints.append((int(match.group(1)), path))

            discovered_checkpoints.sort()
            to_delete = discovered_checkpoints[: -1 * self.keep_latest_k]

            for _, path in to_delete:
                assert self.purge_thread is not None
                self.purge_queue.put(path)
