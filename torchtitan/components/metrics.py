# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import html
import os
import time
from collections import namedtuple
from datetime import datetime
from typing import Any, TYPE_CHECKING

import torch
from torch.utils.tensorboard import SummaryWriter
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.tools import utils
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import Color, device_module, device_type

if TYPE_CHECKING:
    from torchtitan.protocols import BaseModelArgs


# named tuple for passing device memory stats for logging
DeviceMemStats = namedtuple(
    "DeviceMemStats",
    [
        "max_active_gib",
        "max_active_pct",
        "max_reserved_gib",
        "max_reserved_pct",
        "num_alloc_retries",
        "num_ooms",
    ],
)


def _build_wandb_training_data_link_card(html_url: str, step: int) -> str:
    escaped_url = html.escape(html_url, quote=True)
    return (
        "<!DOCTYPE html>"
        "<html><head><meta charset='utf-8'></head>"
        f"<body data-step='{step}' style='margin:0;padding:0;font-family:sans-serif;'>"
        f"<a href='{escaped_url}' target='_blank' rel='noopener noreferrer'>"
        "View Batch Visualization"
        "</a>"
        f"<span style='display:none'>step={step}</span>"
        "</body></html>"
    )


def _build_wandb_training_data_payload(
    wandb_module: Any,
    *,
    html_content: str,
    html_url: str | None,
    step_html_url: str | None,
    inject: bool = False,
) -> dict[str, Any]:
    payload = {
        "vis_batch": wandb_module.Html(html_content, inject=inject),
    }
    if isinstance(html_url, str) and html_url:
        payload["training_data/url"] = html_url
    if (
        isinstance(step_html_url, str)
        and step_html_url
        and step_html_url != html_url
    ):
        payload["training_data/step_url"] = step_html_url
    return payload


class DeviceMemoryMonitor:
    def __init__(self, device: str = f"{device_type}:0"):
        # pyrefly: ignore [read-only]
        self.device = torch.device(device)  # device object
        self.device_name = device_module.get_device_name(self.device)
        self.device_index = device_module.current_device()
        self.device_capacity = device_module.get_device_properties(
            self.device
        ).total_memory
        self.device_capacity_gib = self._to_gib(self.device_capacity)

        device_module.reset_peak_memory_stats()
        device_module.empty_cache()

    def _to_gib(self, memory_in_bytes):
        # NOTE: GiB (gibibyte) is 1024, vs GB is 1000
        _gib_in_bytes = 1024 * 1024 * 1024
        memory_in_gib = memory_in_bytes / _gib_in_bytes
        return memory_in_gib

    def _to_pct(self, memory):
        return 100 * memory / self.device_capacity

    def get_peak_stats(self):
        device_info = device_module.memory_stats(self.device)

        max_active = device_info.get("active_bytes.all.peak", -1)
        max_active_gib = self._to_gib(max_active)
        max_active_pct = self._to_pct(max_active)

        max_reserved = device_info.get("reserved_bytes.all.peak", -1)
        max_reserved_gib = self._to_gib(max_reserved)
        max_reserved_pct = self._to_pct(max_reserved)

        num_retries = device_info.get("num_alloc_retries", -1)
        num_ooms = device_info.get("num_ooms", -1)

        if num_retries > 0:
            logger.warning(
                f"{num_retries} {device_type.upper()} memory allocation retries."
            )
        if num_ooms > 0:
            logger.warning(f"{num_ooms} {device_type.upper()} OOM errors thrown.")

        return DeviceMemStats(
            max_active_gib,
            max_active_pct,
            max_reserved_gib,
            max_reserved_pct,
            num_retries,
            num_ooms,
        )

    def reset_peak_stats(self):
        device_module.reset_peak_memory_stats()


def build_device_memory_monitor():
    device_memory_monitor = DeviceMemoryMonitor(device_type)
    logger.info(
        f"{device_type.upper()} capacity: {device_memory_monitor.device_name} "
        f"with {device_memory_monitor.device_capacity_gib:.2f}GiB memory"
    )
    return device_memory_monitor


class BaseLogger:
    """Logger that does nothing, used when logging is disabled."""

    def log(self, metrics: dict[str, Any], step: int) -> None:
        pass

    def close(self) -> None:
        pass


class TensorBoardLogger(BaseLogger):
    """Logger implementation for TensorBoard."""

    def __init__(self, log_dir: str, tag: str | None = None):
        self.tag = tag
        self.writer = SummaryWriter(log_dir, max_queue=1000)
        logger.info(f"TensorBoard logging enabled. Logs will be saved at {log_dir}")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        for k, v in metrics.items():
            tag = k if self.tag is None else f"{self.tag}/{k}"
            self.writer.add_scalar(tag, v, step)

    def close(self) -> None:
        self.writer.close()


class WandBLogger(BaseLogger):
    """Logger implementation for Weights & Biases.

    Supports deferred initialization: wandb.init() is NOT called in __init__
    so that checkpoint loading can provide a run_id for resume before the
    first wandb call.  The actual init happens lazily on the first log() or
    explicitly via ensure_initialized() / resume_with_id().
    """

    def __init__(self, log_dir: str, job_config: JobConfig, tag: str | None = None):
        # Import wandb here to avoid startup import
        import wandb
        import torch.distributed as dist

        self.wandb = wandb
        self.tag = tag

        # Check if we're in distributed training and get rank
        if dist.is_initialized():
            self.is_rank_zero = dist.get_rank() == 0
        else:
            self.is_rank_zero = True

        # Credentials are environment-only. Non-secret metadata may come from
        # either the environment or the TOML config.
        metrics_cfg = job_config.metrics
        host = os.getenv("WANDB_BASE_URL", None) or metrics_cfg.wandb_base_url
        mode = os.getenv("WANDB_MODE", None) or metrics_cfg.wandb_mode
        self.enabled = mode != "disabled"

        # Resolve config fields (toml > env var > default)
        project = os.getenv("WANDB_PROJECT", None) or metrics_cfg.wandb_project
        entity = os.getenv("WANDB_ENTITY", None) or metrics_cfg.wandb_entity
        run_name = os.getenv("WANDB_NAME", None) or metrics_cfg.wandb_name

        if not run_name:
            if job_config.job.run_name:
                run_name = job_config.job.run_name
            elif job_config.job.config_file:
                run_name = os.path.splitext(os.path.basename(job_config.job.config_file))[0]
                if job_config.job.trial_name:
                    run_name = run_name + "_" + job_config.job.trial_name

        # Create logging directory
        os.makedirs(log_dir, exist_ok=True)

        # Store init params for potential re-init on resume
        self._init_kwargs = dict(
            entity=entity, project=project, name=run_name,
            notes=os.getenv("WANDB_RUN_NOTES", None),
            tags=os.getenv("WANDB_RUN_TAGS", None),
            group=os.getenv("WANDB_RUN_GROUP", None),
            job_type=os.getenv("WANDB_RUN_JOB_TYPE", None),
            mode=mode, dir=log_dir,
            config=job_config.to_dict(),
        )
        self._host = host

        # Deferred init: do NOT call wandb.init() here.
        # This avoids creating a throwaway run that gets immediately
        # finished when resume_with_id() is called from checkpoint loading.
        self._initialized = False

        if not self.enabled:
            logger.info("WandB logging disabled (mode=disabled)")
        elif not self.is_rank_zero:
            logger.debug("WandB logging skipped (not rank 0)")
        else:
            logger.info("WandB logger created (deferred init, will start on first log or resume)")

    def _do_init(self, run_id: str | None = None, resume: str | None = None) -> None:
        """Actually call wandb.init(). Safe to call multiple times (no-op if already done)."""
        if self._initialized or not self.enabled or not self.is_rank_zero:
            return

        # Let wandb.init() consume credentials from this process environment.
        # Calling wandb.login(key=...) persists the key to ~/.netrc, which is
        # undesirable for ephemeral compute nodes and public launchers.
        if self._host:
            os.environ["WANDB_BASE_URL"] = self._host
        # Determine run_id and resume mode
        env_run_id = os.getenv("WANDB_RUN_ID", None)
        effective_id = run_id or env_run_id
        effective_resume = resume or ("allow" if effective_id else None)

        try:
            init_extra: dict[str, Any] = {}
            if effective_id:
                init_extra["id"] = effective_id
                init_extra["resume"] = effective_resume
            else:
                # Only pass these for brand-new runs
                init_extra["resume_from"] = os.getenv("WANDB_RESUME_FROM", None)
                init_extra["fork_from"] = os.getenv("WANDB_FORK_FROM", None)

            self.wandb.init(
                **self._init_kwargs,
                **init_extra,
                settings=self.wandb.Settings(init_timeout=5),
            )
            self._initialized = True
        except Exception as e:
            logger.warning(f"WandB init failed: {e}. Will retry without resume args.")
            try:
                self.wandb.init(
                    **self._init_kwargs,
                    settings=self.wandb.Settings(init_timeout=5),
                )
                self._initialized = True
            except Exception as e2:
                logger.warning(f"WandB init retry failed: {e2}. Disabling.")
                self.enabled = False
                return

        # Save wandb URL to file
        if self.enabled and self.wandb.run is not None:
            run_url = self.wandb.run.url
            if run_url:
                log_dir = self._init_kwargs.get("dir", ".")
                wandb_url_file = os.path.join(log_dir, "wandb_url.log")
                with open(wandb_url_file, "w") as f:
                    f.write(run_url)
                logger.info(f"WandB logging enabled. URL saved to {wandb_url_file}")
            else:
                logger.info("WandB logging enabled without a run URL")
        elif self.enabled:
            logger.info("WandB logging enabled")

    def ensure_initialized(self) -> None:
        """Ensure wandb is initialized (called before first log if not yet done)."""
        if not self._initialized:
            self._do_init()

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Log metrics to wandb. Only rank 0 actually logs."""
        if not self.enabled or not self.is_rank_zero:
            return

        # Lazy init: first log triggers wandb.init() if not yet done
        self.ensure_initialized()

        # Convert tensor values to scalars
        processed_metrics = {}
        for k, v in metrics.items():
            tag = k if self.tag is None else f"{self.tag}/{k}"
            if isinstance(v, torch.Tensor):
                processed_metrics[tag] = v.item()
            else:
                processed_metrics[tag] = v

        self.wandb.log(processed_metrics, step=step)

    def log_training_data_samples(
        self, samples: list[dict[str, Any]], step: int
    ) -> None:
        """Log training data samples to wandb, preferring the rich HTML payload."""
        if not self.enabled or not self.is_rank_zero:
            return
        self.ensure_initialized()
        try:
            html_content = samples[0].get("html_content") if samples else None
            html_url = samples[0].get("html_url") if samples else None
            html_path = samples[0].get("html_path") if samples else None
            step_html_url = samples[0].get("step_html_url") if samples else None
            wandb_log_url_only = bool(samples[0].get("wandb_log_url_only")) if samples else False

            if wandb_log_url_only and isinstance(html_url, str) and html_url:
                self.wandb.log(
                    _build_wandb_training_data_payload(
                        self.wandb,
                        html_content=_build_wandb_training_data_link_card(html_url, step),
                        html_url=html_url,
                        step_html_url=step_html_url,
                        inject=False,
                    ),
                    step=step,
                )
                logger.info(
                    "Logged training data visualization link card and URL to wandb at step "
                    f"{step}: {html_url}"
                )
                return

            if isinstance(html_content, str) and html_content:
                payload = _build_wandb_training_data_payload(
                    self.wandb,
                    html_content=html_content,
                    html_url=html_url if isinstance(html_url, str) else None,
                    step_html_url=step_html_url if isinstance(step_html_url, str) else None,
                    inject=False,
                )
                self.wandb.log(payload, step=step)
                logger.info(
                    "Logged embedded training data visualization HTML to wandb at step "
                    f"{step}"
                )
                return

            if isinstance(html_path, str) and html_path and os.path.exists(html_path):
                with open(html_path, "r", encoding="utf-8") as handle:
                    file_html_content = handle.read()
                payload = _build_wandb_training_data_payload(
                    self.wandb,
                    html_content=file_html_content,
                    html_url=html_url if isinstance(html_url, str) else None,
                    step_html_url=step_html_url if isinstance(step_html_url, str) else None,
                    inject=False,
                )
                self.wandb.log(payload, step=step)
                logger.info(
                    "Logged file-backed training data visualization HTML to wandb at step "
                    f"{step}: {html_path}"
                )
                return

            parts = [
                "<div style='font-family:monospace;font-size:13px;'>"
                f"<h2>Training Data — Step {step}</h2>"
            ]
            for sample in samples:
                sid = sample.get("sample_id", 0)
                inp = sample["input_text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                lbl = sample["label_text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                parts.append(
                    f"<details open><summary><b>Sample {sid}</b></summary>"
                    f"<b>Input:</b><pre style='white-space:pre-wrap;background:#f5f5f5;"
                    f"padding:8px;border-radius:4px;max-height:300px;overflow-y:auto;'>"
                    f"{inp}</pre>"
                    f"<b>Label:</b><pre style='white-space:pre-wrap;background:#eef6ee;"
                    f"padding:8px;border-radius:4px;max-height:300px;overflow-y:auto;'>"
                    f"{lbl}</pre></details><hr>"
                )
            parts.append("</div>")
            fallback_payload = _build_wandb_training_data_payload(
                self.wandb,
                html_content="".join(parts),
                html_url=html_url if isinstance(html_url, str) else None,
                step_html_url=step_html_url if isinstance(step_html_url, str) else None,
                inject=False,
            )
            self.wandb.log(fallback_payload, step=step)
            logger.info(
                "Logged fallback training data visualization HTML to wandb at step "
                f"{step}"
            )
        except Exception as e:
            logger.error(f"Error logging samples to wandb: {e}")

    def get_run_id(self) -> str | None:
        """Return the current wandb run ID, or None if not active."""
        if self.enabled and self.is_rank_zero and self._initialized and self.wandb.run is not None:
            return self.wandb.run.id
        return None

    def resume_with_id(self, run_id: str) -> None:
        """Resume (or start) wandb with the given run_id.

        Because init is deferred, this is typically the *first* wandb.init()
        call.  If wandb was already initialized (e.g. by an earlier log()),
        we finish the current run and re-init with the saved run_id.
        """
        if not self.enabled or not self.is_rank_zero:
            return

        if self._initialized and self.wandb.run is not None:
            # Already running — check if it's the same run
            if self.wandb.run.id == run_id:
                logger.info(f"WandB already on run {run_id}, skip re-init.")
                return
            # Different run: finish the current one first
            self.wandb.finish()
            self._initialized = False

        # (Re-)init with the checkpoint's run_id
        self._do_init(run_id=run_id, resume="allow")
        if self._initialized:
            logger.info(f"WandB resumed run {run_id}")

    def close(self) -> None:
        if self.enabled and self.is_rank_zero and self._initialized and self.wandb.run is not None:
            self.wandb.finish()
            self._initialized = False


class LoggerContainer(BaseLogger):
    """Container to call all loggers enabled in the job config."""

    def __init__(self) -> None:
        self._loggers: list[BaseLogger] = []

    def add_logger(self, logger_instance: BaseLogger) -> None:
        self._loggers.append(logger_instance)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        for logger_instance in self._loggers:
            logger_instance.log(metrics, step)

    @property
    def number_of_loggers(self) -> int:
        return len(self._loggers)

    def log_training_data_samples(
        self, samples: list[dict[str, Any]], step: int
    ) -> None:
        for logger_instance in self._loggers:
            if hasattr(logger_instance, "log_training_data_samples"):
                logger_instance.log_training_data_samples(samples, step)

    def get_wandb_run_id(self) -> str | None:
        """Return the wandb run ID from the first WandBLogger, or None."""
        for lg in self._loggers:
            if isinstance(lg, WandBLogger):
                return lg.get_run_id()
        return None

    def resume_wandb(self, run_id: str) -> None:
        """Resume wandb with the given run_id on all WandBLogger instances."""
        for lg in self._loggers:
            if isinstance(lg, WandBLogger):
                lg.resume_with_id(run_id)

    def close(self) -> None:
        for logger_instance in self._loggers:
            logger_instance.close()


def ensure_pp_loss_visible(
    parallel_dims: ParallelDims, job_config: JobConfig, color: Color
) -> None:
    """
    Ensures that the loss is visible on the console for pipeline-parallel training.

    For pipeline-parallel training, the loss is only visible on the last pipeline stage.
    This function checks if the appropriate rank is included in the LOG_RANK environment
    variable and warns if it's not.
    """

    # V Block Schedules return loss on rank 0
    if job_config.parallelism.pipeline_parallel_schedule == "ZBVZeroBubble":
        return

    # Calculate the rank where loss is visible (first rank of the last pipeline stage)
    world_size = parallel_dims.world_size
    pp_size = parallel_dims.pp
    loss_visible_rank = (world_size // pp_size) * (pp_size - 1)

    # Check if the loss-visible rank is included in LOG_RANK environment variable
    env_logged_ranks = os.environ.get("LOG_RANK", "").split(",")
    if env_logged_ranks == [""]:
        env_logged_ranks = []

    if str(loss_visible_rank) not in env_logged_ranks:
        logger.warning(
            f"{color.red}Pipeline Parallel loss is not visible. "
            f"Please add {color.yellow}rank {loss_visible_rank}{color.red} "
            f"to the LOG_RANK environment variable before launching training.{color.reset}"
        )


def _get_metrics_rank(
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> int:
    """
    Determines which rank should log metrics.

    Returns:
       int: The rank responsible for logging metrics:
            - Rank 0 for non-pipeline-parallel configs
            - Rank 0 for pipeline-parallel 'ZBVZeroBubble' schedule
            - The first rank of the last pipeline stage for other pipeline-parallel schedules
    """
    # Early return for non-pipeline-parallel configurations
    if not parallel_dims.pp_enabled:
        return 0

    # V Block Schedules return loss on rank 0
    if job_config.parallelism.pipeline_parallel_schedule == "ZBVZeroBubble":
        return 0

    # Calculate first rank of the last pipeline stage
    world_size = parallel_dims.world_size
    pp_size = parallel_dims.pp
    return (world_size // pp_size) * (pp_size - 1)


def _build_metric_logger(
    job_config: JobConfig, parallel_dims: ParallelDims, tag: str | None = None
) -> BaseLogger:
    """
    Build an appropriate metric logger based on configuration.
    """
    metrics_config = job_config.metrics

    # Log initial config state
    logger.debug(
        f"Building logger with config: wandb={metrics_config.enable_wandb}, "
        f"tensorboard={metrics_config.enable_tensorboard}"
    )

    # Check if any logging backend is enabled
    has_logging_enabled = (
        metrics_config.enable_tensorboard or metrics_config.enable_wandb
    )

    # Determine if this rank should log
    should_log = has_logging_enabled
    if (not metrics_config.save_for_all_ranks) and should_log:
        metrics_rank = _get_metrics_rank(parallel_dims, job_config)
        should_log = torch.distributed.get_rank() == metrics_rank

    logger.debug(
        f"Logging decision: has_logging_enabled={has_logging_enabled}, should_log={should_log}"
    )

    if not should_log:
        logger.debug("Returning BaseLogger due to should_log=False")
        return BaseLogger()

    # Setup logging directory using same path structure as checkpoint:
    # {dump_folder}/{user_name}/{run_name}/{save_tb_folder}/{timestamp}
    dump_dir = job_config.job.dump_folder
    user_name = os.environ.get("USER", "default")
    user_name = user_name.replace(" ", "_").lower()
    if job_config.job.config_file:
        config_basename = os.path.splitext(os.path.basename(job_config.job.config_file))[0]
    else:
        config_basename = "default"
    if job_config.job.run_name:
        run_name = job_config.job.run_name
    else:
        trial_name = job_config.job.trial_name
        run_name = f"{config_basename}_{trial_name}" if trial_name else config_basename

    if os.path.isabs(metrics_config.save_tb_folder):
        # Absolute path: {save_tb_folder}/{user_name}/{run_name}/tb/{timestamp}
        base_log_dir = os.path.join(
            metrics_config.save_tb_folder, user_name, run_name, "tb", datetime.now().strftime("%Y%m%d-%H%M")
        )
    else:
        # Relative path: {dump_folder}/{user_name}/{run_name}/{save_tb_folder}/{timestamp}
        base_log_dir = os.path.join(
            dump_dir, user_name, run_name, metrics_config.save_tb_folder, datetime.now().strftime("%Y%m%d-%H%M")
        )

    if job_config.fault_tolerance.enable:
        base_log_dir = os.path.join(
            base_log_dir,
            f"replica_{job_config.fault_tolerance.replica_id}",
        )

    if metrics_config.save_for_all_ranks:
        base_log_dir = os.path.join(
            base_log_dir, f"rank_{torch.distributed.get_rank()}"
        )

    # Create logger container
    logger_container = LoggerContainer()

    # Create loggers in priority order
    if metrics_config.enable_wandb:
        logger.debug("Attempting to create WandB logger")
        try:
            wandb_logger = WandBLogger(base_log_dir, job_config, tag)
            logger_container.add_logger(wandb_logger)
        except Exception as e:
            if "No module named 'wandb'" in str(e):
                logger.error(
                    "Failed to create WandB logger: No module named 'wandb'. Please install it using 'pip install wandb'."
                )
            else:
                logger.error(f"Failed to create WandB logger: {e}")

    if metrics_config.enable_tensorboard:
        logger.debug("Creating TensorBoard logger")
        tensorboard_logger = TensorBoardLogger(base_log_dir, tag)
        logger_container.add_logger(tensorboard_logger)

    if logger_container.number_of_loggers == 0:
        logger.debug("No loggers enabled, returning an empty LoggerContainer")
    return logger_container


class MetricsProcessor:
    """Metrics processor to processes the metrics and log metrics.

    The current MetricsProcessor log some metrics to STDOUT and some metrics to
    TensorBoard or WandB.

    Args:
        job_config (JobConfig): Job configuration.
        parallel_dims (ParallelDims): Parallel dimensions.
        tag (Optional[str]): Tag to use for TensorBoard or WandB. Defaults to None.
    """

    logger: BaseLogger
    parallel_dims: ParallelDims
    job_config: JobConfig
    device_memory_monitor: DeviceMemoryMonitor
    color: utils.NoColor | utils.Color

    gpu_peak_flops: float
    ntokens_since_last_log: int
    data_loading_times: list[float]
    time_last_log: float

    num_flops_per_token: int
    optimizers: OptimizersContainer | None
    lr_schedulers: LRSchedulersContainer | None
    model_parts: list[torch.nn.Module] | None

    def __init__(
        self,
        job_config: JobConfig,
        parallel_dims: ParallelDims,
        tag: str | None = None,
    ):
        self.logger = _build_metric_logger(job_config, parallel_dims, tag)
        self.parallel_dims = parallel_dims
        self.job_config = job_config
        self.device_memory_monitor = build_device_memory_monitor()
        # used for colorful printing
        self.color = (
            utils.NoColor()
            if job_config.metrics.disable_color_printing
            else utils.Color()
        )

        self.gpu_peak_flops = utils.get_peak_flops(
            self.device_memory_monitor.device_name
        )
        self.ntokens_since_last_log = 0
        self.data_loading_times = []
        self.time_last_log = time.perf_counter()
        self.device_memory_monitor.reset_peak_stats()

        # These variables have to be set later as they depend on other components or model.
        self.num_flops_per_token = -1
        self.optimizers = None
        self.lr_schedulers = None
        self.model_parts = None

    def get_wandb_run_id(self) -> str | None:
        """Return the wandb run ID if available."""
        if isinstance(self.logger, LoggerContainer):
            return self.logger.get_wandb_run_id()
        if isinstance(self.logger, WandBLogger):
            return self.logger.get_run_id()
        return None

    def resume_wandb(self, run_id: str) -> None:
        """Resume wandb with the given run_id."""
        if isinstance(self.logger, LoggerContainer):
            self.logger.resume_wandb(run_id)
        elif isinstance(self.logger, WandBLogger):
            self.logger.resume_with_id(run_id)

    def should_log(self, step: int) -> bool:
        return step == 1 or step % self.job_config.metrics.log_freq == 0

    def log(
        self,
        step: int,
        global_avg_loss: float,
        global_max_loss: float,
        grad_norm: float,
        extra_metrics: dict[str, Any] | None = None,
    ):
        assert self.num_flops_per_token > 0, "num_flops_per_token must be set"

        time_delta = time.perf_counter() - self.time_last_log

        # tokens per second per device, abbreviated as tps
        tps = self.ntokens_since_last_log / (
            time_delta * self.parallel_dims.non_data_parallel_size
        )
        # model FLOPS utilization
        # For its definition and calculation, please refer to the PaLM paper:
        # https://arxiv.org/abs/2204.02311
        mfu = 100 * self.num_flops_per_token * tps / self.gpu_peak_flops
        tflops = self.num_flops_per_token * tps / 1e12

        time_end_to_end = time_delta / self.job_config.metrics.log_freq
        time_data_loading = sum(self.data_loading_times) / len(self.data_loading_times)
        time_data_loading_pct = 100 * sum(self.data_loading_times) / time_delta

        device_mem_stats = self.device_memory_monitor.get_peak_stats()

        metrics = {
            "loss_metrics/global_avg_loss": global_avg_loss,
            "loss_metrics/global_max_loss": global_max_loss,
            "grad_norm": grad_norm,
            "throughput(tps)": tps,
            "tflops": tflops,
            "mfu(%)": mfu,
            "time_metrics/end_to_end(s)": time_end_to_end,
            "time_metrics/data_loading(s)": time_data_loading,
            "time_metrics/data_loading(%)": time_data_loading_pct,
            "memory/max_active(GiB)": device_mem_stats.max_active_gib,
            "memory/max_active(%)": device_mem_stats.max_active_pct,
            "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
            "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
            "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
            "memory/num_ooms": device_mem_stats.num_ooms,
        }

        if extra_metrics:
            metrics.update(extra_metrics)

        self.logger.log(metrics, step)

        color = self.color
        logger.info(
            f"{color.red}step: {step:2}  "
            f"{color.green}loss: {global_avg_loss:7.4f}  "
            f"{color.orange}grad_norm: {grad_norm:7.4f}  "
            f"{color.turquoise}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)  "
            f"{color.blue}tps: {round(tps):,}  "
            f"{color.cyan}tflops: {tflops:,.2f}  "
            f"{color.magenta}mfu: {mfu:.2f}%{color.reset}"
        )

        self.ntokens_since_last_log = 0
        self.data_loading_times.clear()
        self.time_last_log = time.perf_counter()
        self.device_memory_monitor.reset_peak_stats()

    def log_validation(
        self, loss: float, step: int, extra_metrics: dict[str, Any] | None = None
    ):
        time_delta = time.perf_counter() - self.time_last_log

        device_mem_stats = self.device_memory_monitor.get_peak_stats()

        # tokens per second per device, abbreviated as tps
        tps = self.ntokens_since_last_log / (
            time_delta * self.parallel_dims.non_data_parallel_size
        )

        metrics = {
            "validation_metrics/loss": loss,
            "validation_metrics/throughput(tps)": tps,
            "validation_metrics/memory/max_active(GiB)": device_mem_stats.max_active_gib,
            "validation_metrics/memory/max_active(%)": device_mem_stats.max_active_pct,
            "validation_metrics/memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
            "validation_metrics/memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
        }

        if extra_metrics:
            metrics.update(extra_metrics)

        self.logger.log(metrics, step)

        color = self.color
        logger.info(
            f"{color.yellow}validate step: {step:2}  "
            f"{color.green}loss: {loss:7.4f}  "
            f"{color.turquoise}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)  "
            f"{color.blue}tps: {round(tps):,}{color.reset}"
        )

        self.ntokens_since_last_log = 0
        self.time_last_log = time.perf_counter()
        self.device_memory_monitor.reset_peak_stats()

    def log_training_data_samples(
        self,
        samples: list[dict[str, Any]],
        step: int,
    ) -> None:
        """
        Log training data samples to wandb (delegates to logger).

        Args:
            samples: List of dicts with 'input_text' and 'label_text' keys
            step: Current training step
        """
        if hasattr(self.logger, 'log_training_data_samples'):
            self.logger.log_training_data_samples(samples, step)


class _StandaloneWandBLogger(BaseLogger):
    """Standalone WandB logger (not used by MetricsProcessor pipeline).
    
    NOTE: Do NOT name this WandBLogger — it would shadow the main WandBLogger
    class used by _build_metric_logger(), causing it to receive wrong arguments.
    """

    def __init__(
        self,
        project: str,
        name: str | None,
        entity: str | None = None,
        rank: int = 0,
        tag: str | None = None,
        config: dict | None = None,
    ):
        self.is_rank_zero = rank == 0
        self.tag = tag
        self.enabled = False
        self.wandb = None

        if not self.is_rank_zero:
            return

        try:
            import wandb as wandb_module
            self.wandb = wandb_module
            self.enabled = True
        except ImportError:
            logger.error(
                "Failed to create WandB logger: No module named 'wandb'. "
                "Please install it using 'pip install wandb'."
            )
            return

        # Initialize wandb
        if self.enabled and self.is_rank_zero:
            try:
                self.wandb.init(
                    project=project,
                    name=name,
                    entity=entity,
                    config=config,
                    resume="allow",
                )
                logger.info(f"WandB logging enabled for project '{project}', run '{name}'")
            except Exception as e:
                logger.error(f"Failed to initialize wandb: {e}")
                self.enabled = False

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if not self.enabled or not self.is_rank_zero:
            return

        # Process metrics to ensure compatibility with wandb
        processed_metrics = {}
        for k, v in metrics.items():
            tag = k if self.tag is None else f"{self.tag}/{k}"
            if isinstance(v, torch.Tensor):
                processed_metrics[tag] = v.item()
            else:
                processed_metrics[tag] = v

        self.wandb.log(processed_metrics, step=step)

    def close(self) -> None:
        if self.enabled and self.is_rank_zero and self.wandb.run is not None:
            self.wandb.finish()

    def log_training_data_samples(
        self,
        samples: list[dict[str, Any]],
        step: int,
    ) -> None:
        """
        Log training data samples to wandb.

        Args:
            samples: List of dicts with 'input_text' and 'label_text' keys
            step: Current training step
        """
        if not samples:
            return

        try:
            html_content = samples[0].get("html_content") if samples else None
            html_url = samples[0].get("html_url") if samples else None
            html_path = samples[0].get("html_path") if samples else None
            step_html_url = samples[0].get("step_html_url") if samples else None
            wandb_log_url_only = bool(samples[0].get("wandb_log_url_only")) if samples else False

            if wandb_log_url_only and isinstance(html_url, str) and html_url:
                self.wandb.log(
                    _build_wandb_training_data_payload(
                        self.wandb,
                        html_content=_build_wandb_training_data_link_card(html_url, step),
                        html_url=html_url,
                        step_html_url=step_html_url if isinstance(step_html_url, str) else None,
                        inject=False,
                    ),
                    step=step,
                )
                logger.info(
                    "Logged training data visualization link card and URL to wandb at step "
                    f"{step}: {html_url}"
                )
                return

            if isinstance(html_content, str) and html_content:
                payload = _build_wandb_training_data_payload(
                    self.wandb,
                    html_content=html_content,
                    html_url=html_url if isinstance(html_url, str) else None,
                    step_html_url=step_html_url if isinstance(step_html_url, str) else None,
                    inject=False,
                )
                self.wandb.log(payload, step=step)
                logger.info(
                    "Logged embedded training data visualization HTML to wandb at step "
                    f"{step}"
                )
                return

            if isinstance(html_path, str) and html_path and os.path.exists(html_path):
                with open(html_path, "r", encoding="utf-8") as handle:
                    file_html_content = handle.read()
                payload = _build_wandb_training_data_payload(
                    self.wandb,
                    html_content=file_html_content,
                    html_url=html_url if isinstance(html_url, str) else None,
                    step_html_url=step_html_url if isinstance(step_html_url, str) else None,
                    inject=False,
                )
                self.wandb.log(payload, step=step)
                logger.info(
                    "Logged file-backed training data visualization HTML to wandb at step "
                    f"{step}: {html_path}"
                )
                return

            parts = [
                "<div style='font-family:monospace;font-size:13px;'>"
                f"<h2>Training Data — Step {step}</h2>"
            ]
            for sample in samples:
                sid = sample.get("sample_id", 0)
                inp = sample["input_text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                lbl = sample["label_text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                parts.append(
                    f"<details open><summary><b>Sample {sid}</b></summary>"
                    f"<b>Input:</b><pre style='white-space:pre-wrap;background:#f5f5f5;"
                    f"padding:8px;border-radius:4px;max-height:300px;overflow-y:auto;'>"
                    f"{inp}</pre>"
                    f"<b>Label:</b><pre style='white-space:pre-wrap;background:#eef6ee;"
                    f"padding:8px;border-radius:4px;max-height:300px;overflow-y:auto;'>"
                    f"{lbl}</pre></details><hr>"
                )
            parts.append("</div>")
            fallback_payload = _build_wandb_training_data_payload(
                self.wandb,
                html_content="".join(parts),
                html_url=html_url if isinstance(html_url, str) else None,
                step_html_url=step_html_url if isinstance(step_html_url, str) else None,
                inject=False,
            )
            self.wandb.log(fallback_payload, step=step)
            logger.info(
                "Logged fallback training data visualization HTML to wandb at step "
                f"{step}"
            )
        except Exception as e:
            logger.error(f"Error logging samples to wandb: {e}")


def build_metrics_processor(
    job_config: JobConfig,
    parallel_dims: ParallelDims,
    model_args: "BaseModelArgs | None" = None,
    tag: str | None = None,
) -> MetricsProcessor:
    """Create a metrics processor.

    Args:
        job_config (JobConfig): Job configuration.
        parallel_dims (ParallelDims): Parallel dimensions.
        model_args (BaseModelArgs | None): Model-specific arguments. Defaults to None.
        tag (str | None): Tag to use for TensorBoard or WandB. Defaults to None.

    Returns:
        MetricsProcessor: A metrics processor.
    """
    return MetricsProcessor(job_config, parallel_dims, tag)
