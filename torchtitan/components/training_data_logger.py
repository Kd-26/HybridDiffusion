# Copyright (c) HybridDiffusion contributors.
#
# Licensed under the repository License; see LICENSE in the repository root.

"""
Training data visualization module for periodic logging to wandb.
"""

import html
import json
import os
import queue
import threading
import urllib.request
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.components.training_data_rich_html import build_training_step_preview_html
from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger

MAX_HTML_SAMPLES = 10


@dataclass(slots=True)
class _TrainingDataRenderJob:
    step: int
    html_path: str
    html_content: str
    s3_bucket: str | None
    s3_key: str | None


class TrainingDataLogger:
    """Log training data samples to wandb for visualization."""

    def __init__(
        self,
        job_config: JobConfig,
        tokenizer: BaseTokenizer,
    ):
        self.job_config = job_config
        self.tokenizer = tokenizer
        self.viz_config = job_config.training_data_visualization
        self.is_rank_zero = dist.get_rank() == 0 if dist.is_initialized() else True
        self.block_size = getattr(getattr(job_config, "dllm", None), "block_size", 0)
        self.dllm_layout = (
            getattr(getattr(job_config, "dllm", None), "layout", "") or "x0_xt_doubled"
        )
        self.mask_token_id = getattr(getattr(job_config, "dllm", None), "mask_token_id", -1)
        if self.mask_token_id is None or self.mask_token_id < 0:
            self.mask_token_id = tokenizer.get_vocab_size() - 1

        config_pad_token_id = getattr(getattr(job_config, "dllm", None), "pad_token_id", -1)
        tokenizer_pad_token_id = getattr(tokenizer, "pad_id", None)
        self.pad_token_id = (
            tokenizer_pad_token_id
            if tokenizer_pad_token_id is not None
            else (config_pad_token_id if config_pad_token_id >= 0 else None)
        )

        self.user_name, self.run_name, self.ckpt_root = self._build_run_paths()
        self.html_dir = os.path.join(self.ckpt_root, "training_data_viz")
        self.json_dir = os.path.join(self.ckpt_root, "training_data_viz_json")
        os.makedirs(self.html_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)

        self._render_queue: queue.Queue[_TrainingDataRenderJob | None] | None = None
        self._render_thread: threading.Thread | None = None
        self._s3_client = None
        self._closed = False
        self._fixed_html_cache_path = os.path.join(self.html_dir, "fixed_wandb_preview.html")

        if self.is_rank_zero and self.viz_config.enable:
            logger.info(
                f"Training data visualization enabled: logging {self.viz_config.num_samples} "
                f"samples every {self.viz_config.freq} steps"
            )
            if self.viz_config.async_worker:
                self._render_queue = queue.Queue(
                    maxsize=max(1, self.viz_config.max_pending_jobs)
                )
                self._render_thread = threading.Thread(
                    target=self._render_worker_loop,
                    name="training-data-viz-worker",
                    daemon=True,
                )
                self._render_thread.start()
                logger.info(
                    "Training data visualization async worker enabled "
                    f"(max_pending_jobs={self.viz_config.max_pending_jobs})"
                )

    def _num_html_samples(self, batch_size: int) -> int:
        return min(max(self.viz_config.num_samples, 0), MAX_HTML_SAMPLES, batch_size)

    def should_log(self, step: int) -> bool:
        if not self.viz_config.enable or not self.is_rank_zero:
            return False
        return step > 0 and step % self.viz_config.freq == 0

    def _build_run_paths(self) -> tuple[str, str, str]:
        user_name = os.environ.get("USER", "default")
        user_name = user_name.replace(" ", "_").lower()
        if self.job_config.job.config_file:
            config_basename = os.path.splitext(
                os.path.basename(self.job_config.job.config_file)
            )[0]
        else:
            config_basename = "default"

        if self.job_config.job.run_name:
            run_name = self.job_config.job.run_name
        elif self.job_config.job.trial_name:
            run_name = f"{config_basename}_{self.job_config.job.trial_name}"
        else:
            run_name = config_basename

        ckpt_root = os.path.join(
            self.job_config.job.dump_folder,
            user_name,
            run_name,
            self.job_config.checkpoint.folder,
        )
        return user_name, run_name, ckpt_root

    def _build_step_html_path(self, step: int) -> str:
        return os.path.join(self.html_dir, f"step_{step:08d}.html")

    def _build_step_json_path(self, step: int) -> str:
        return os.path.join(self.json_dir, f"step_{step:08d}.json")

    @staticmethod
    def _split_s3_url(s3_url: str) -> tuple[str, str]:
        stripped = s3_url.removeprefix("s3://")
        bucket, _, key = stripped.partition("/")
        return bucket, key

    @staticmethod
    def _build_https_url(bucket: str, key: str, region: str) -> str:
        return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"

    def _get_fixed_wandb_html_url(self) -> str | None:
        fixed_url = (self.viz_config.wandb_fixed_html_url or "").strip()
        if not fixed_url:
            return None

        if fixed_url.startswith("s3://"):
            bucket, key = self._split_s3_url(fixed_url)
            if not bucket or not key:
                logger.warning(
                    "Invalid s3 URL for training_data_visualization.wandb_fixed_html_url: %s",
                    fixed_url,
                )
                return None
            region = (self.viz_config.s3_region or "us-west-2").strip()
            return self._build_https_url(bucket, key, region)

        return fixed_url

    def _build_s3_location(
        self,
        step: int,
        *,
        use_fixed_url: bool = True,
    ) -> tuple[str | None, str | None, str | None]:
        fixed_url = self._get_fixed_wandb_html_url()
        if use_fixed_url and fixed_url:
            return None, None, fixed_url

        if not self.viz_config.upload_to_s3:
            return None, None, None

        region = (self.viz_config.s3_region or "us-west-2").strip()
        checkpoint_s3_base = (self.job_config.checkpoint.s3_upload_path or "").strip()
        if checkpoint_s3_base:
            # Match CheckpointManager S3 path: {s3_base}/{user_name}/{run_name}/
            # (no checkpoint.folder subfolder — S3 paths are flat)
            s3_url = (
                f"{checkpoint_s3_base.rstrip('/')}/"
                f"{self.user_name}/{self.run_name}/"
                f"training_data_viz/step_{step:08d}.html"
            )
            bucket, key = self._split_s3_url(s3_url)
            return bucket, key, self._build_https_url(bucket, key, region)

        bucket = (self.viz_config.s3_bucket or "").strip()
        if not bucket:
            return None, None, None
        prefix = (self.viz_config.s3_prefix or "").strip("/")
        relative_key = "/".join(
            [
                self.user_name,
                self.run_name,
                self.job_config.checkpoint.folder,
                "training_data_viz",
                f"step_{step:08d}.html",
            ]
        )
        key = f"{prefix}/{relative_key}" if prefix else relative_key
        return bucket, key, self._build_https_url(bucket, key, region)

    def _cache_fixed_html_snapshot(self, fixed_url: str) -> str | None:
        if not os.path.exists(self._fixed_html_cache_path):
            try:
                with urllib.request.urlopen(fixed_url, timeout=30) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    html_bytes = response.read()
                with open(self._fixed_html_cache_path, "w", encoding=charset, errors="replace") as handle:
                    handle.write(html_bytes.decode(charset, errors="replace"))
                logger.info(
                    "Cached fixed training data visualization HTML from %s to %s",
                    fixed_url,
                    self._fixed_html_cache_path,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to cache fixed training data visualization HTML from %s: %s",
                    fixed_url,
                    exc,
                )
                return None
        return self._fixed_html_cache_path

    def _decode_tokens(self, token_ids: list[int]) -> str:
        cleaned = [int(token_id) for token_id in token_ids if int(token_id) >= 0]
        if not cleaned:
            return ""
        try:
            return self.tokenizer.decode(cleaned, skip_special_tokens=False)
        except Exception:
            return " ".join(str(token_id) for token_id in cleaned)

    @staticmethod
    def _extract_sample_branch_masks(
        masked_indices: torch.Tensor,
        *,
        sample_id: int,
        batch_size: int,
    ) -> list[tuple[str, torch.Tensor]]:
        if masked_indices.ndim != 2 or masked_indices.shape[0] == 0:
            return []
        if masked_indices.shape[0] == batch_size:
            return [("primary", masked_indices[sample_id])]
        if masked_indices.shape[0] == 2 * batch_size:
            return [
                ("primary", masked_indices[sample_id]),
                ("complementary", masked_indices[sample_id + batch_size]),
            ]
        if sample_id < masked_indices.shape[0]:
            return [(f"branch_{sample_id}", masked_indices[sample_id])]
        return []

    def _serialize_viz_payload(
        self,
        viz_payload: dict[str, Any] | None,
        *,
        num_samples: int,
        batch_size: int,
    ) -> dict[str, Any] | None:
        if viz_payload is None:
            return None

        serialized: dict[str, Any] = {}
        layout = viz_payload.get("layout")
        if isinstance(layout, str):
            serialized["layout"] = layout

        target_slice = viz_payload.get("target_slice")
        if (
            isinstance(target_slice, tuple)
            and len(target_slice) == 2
            and all(isinstance(value, int) for value in target_slice)
        ):
            serialized["target_slice"] = [target_slice[0], target_slice[1]]

        masked_indices = viz_payload.get("masked_indices")
        if isinstance(masked_indices, torch.Tensor):
            sample_branch_masks: dict[str, list[dict[str, Any]]] = {}
            for sample_id in range(num_samples):
                branches = []
                for branch_name, branch_mask in self._extract_sample_branch_masks(
                    masked_indices,
                    sample_id=sample_id,
                    batch_size=batch_size,
                ):
                    branches.append(
                        {
                            "branch": branch_name,
                            "mask": branch_mask.detach().cpu().tolist(),
                        }
                    )
                if branches:
                    sample_branch_masks[str(sample_id)] = branches
            if sample_branch_masks:
                serialized["sample_branch_masks"] = sample_branch_masks

        return serialized or None

    def _build_step_payload(
        self,
        *,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        step: int,
        viz_payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        num_samples = self._num_html_samples(input_ids.size(0))
        payload = {
            "step": int(step),
            "batch_size": int(input_ids.size(0)),
            "block_size": int(self.block_size),
            "layout": self.dllm_layout,
            "samples": [],
        }

        fixed_html_url = (self.viz_config.wandb_fixed_html_url or "").strip()
        if fixed_html_url:
            payload["reference_html_url"] = fixed_html_url

        for sample_id in range(num_samples):
            payload["samples"].append(
                {
                    "sample_id": int(sample_id),
                    "input_ids": input_ids[sample_id].detach().cpu().tolist(),
                    "labels": labels[sample_id].detach().cpu().tolist(),
                }
            )

        serialized_viz = self._serialize_viz_payload(
            viz_payload,
            num_samples=num_samples,
            batch_size=input_ids.size(0),
        )
        if serialized_viz is not None:
            payload["viz_payload"] = serialized_viz
        return payload

    def _build_simple_samples_from_payload(
        self, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for sample in payload.get("samples", []):
            token_ids = [int(token_id) for token_id in sample.get("input_ids", [])]
            label_ids = [int(label_id) for label_id in sample.get("labels", [])]
            input_text = self._decode_tokens(token_ids)
            label_text = self._decode_tokens(
                [token for token, label in zip(token_ids, label_ids) if label != -100]
            )
            samples.append(
                {
                    "step": int(payload.get("step", 0)),
                    "sample_id": int(sample.get("sample_id", 0)),
                    "input_text": input_text,
                    "label_text": label_text,
                    "input_token_count": len(token_ids),
                    "supervised_token_count": sum(1 for label in label_ids if label != -100),
                }
            )
        return samples

    def _compose_fallback_html(
        self, samples: list[dict[str, Any]], step: int
    ) -> str:
        parts = [
            "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1'>",
            f"<title>Training Data Visualization Step {step}</title>",
            "<style>",
            "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; color: #111827; }",
            "h1, h2, h3 { margin: 0.8rem 0 0.5rem; }",
            ".card { border: 1px solid #d1d5db; border-radius: 10px; padding: 16px; margin: 16px 0; }",
            "pre { white-space: pre-wrap; word-break: break-word; background: #f8fafc; padding: 12px; border-radius: 8px; overflow-x: auto; }",
            "</style></head><body>",
            f"<h1>Training Data Visualization Step {step}</h1>",
        ]
        for sample in samples:
            parts.extend(
                [
                    "<div class='card'>",
                    f"<h2>Sample {sample.get('sample_id', 0)}</h2>",
                    (
                        "<p><b>Input tokens:</b> "
                        f"{sample.get('input_token_count', 0)} | "
                        f"<b>Supervised tokens:</b> {sample.get('supervised_token_count', 0)}</p>"
                    ),
                    "<h3>Input</h3>",
                    "<pre>" + html.escape(sample["input_text"]) + "</pre>",
                    "<h3>Label</h3>",
                    "<pre>" + html.escape(sample["label_text"]) + "</pre>",
                    "</div>",
                ]
            )
        parts.append("</body></html>")
        return "".join(parts)

    def _compose_step_html(self, payload: dict[str, Any], step: int) -> str:
        reference_html_url = payload.get("reference_html_url")
        if not isinstance(reference_html_url, str):
            reference_html_url = None
        try:
            return build_training_step_preview_html(
                tokenizer=self.tokenizer,
                payload=payload,
                block_size=self.block_size,
                dllm_layout=self.dllm_layout,
                mask_token_id=self.mask_token_id,
                pad_token_id=self.pad_token_id,
                reference_html_url=reference_html_url,
            )
        except Exception as exc:
            logger.warning(
                "Falling back to simple training data visualization HTML at step %s: %s",
                step,
                exc,
            )
            return self._compose_fallback_html(
                self._build_simple_samples_from_payload(payload),
                step,
            )

    def _write_step_json(self, payload: dict[str, Any], step: int) -> str:
        json_path = self._build_step_json_path(step)
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        return json_path

    @staticmethod
    def _write_html_file(html_path: str, html_content: str) -> None:
        with open(html_path, "w", encoding="utf-8") as handle:
            handle.write(html_content)

    def _get_s3_client(self):
        if self._s3_client is None:
            import boto3

            region_name = self.viz_config.s3_region or None
            self._s3_client = boto3.client("s3", region_name=region_name)
        return self._s3_client

    def _upload_html_to_s3(self, *, html_path: str, bucket: str, key: str) -> None:
        client = self._get_s3_client()
        client.upload_file(
            html_path,
            bucket,
            key,
            ExtraArgs={"ContentType": "text/html; charset=utf-8"},
        )

    def _render_and_maybe_upload(self, job: _TrainingDataRenderJob) -> None:
        self._write_html_file(job.html_path, job.html_content)
        if job.s3_bucket and job.s3_key:
            self._upload_html_to_s3(
                html_path=job.html_path,
                bucket=job.s3_bucket,
                key=job.s3_key,
            )
            logger.info(
                "Uploaded training data visualization HTML to S3 at step %s: s3://%s/%s",
                job.step,
                job.s3_bucket,
                job.s3_key,
            )

    def _render_worker_loop(self) -> None:
        assert self._render_queue is not None
        while True:
            job = self._render_queue.get()
            try:
                if job is None:
                    return
                self._render_and_maybe_upload(job)
            except Exception as exc:
                logger.error(
                    "Async training data visualization render failed at step %s: %s",
                    getattr(job, "step", "unknown"),
                    exc,
                )
            finally:
                self._render_queue.task_done()

    def _enqueue_render_job(self, job: _TrainingDataRenderJob) -> bool:
        if self._render_queue is None:
            self._render_and_maybe_upload(job)
            return True
        try:
            self._render_queue.put_nowait(job)
            return True
        except queue.Full:
            logger.warning(
                "Dropping training data visualization job at step %s because the async queue is full "
                "(max_pending_jobs=%s)",
                job.step,
                self.viz_config.max_pending_jobs,
            )
            return False

    def flush(self) -> None:
        if self._render_queue is not None:
            self._render_queue.join()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._render_queue is not None:
            self._render_queue.join()
            self._render_queue.put(None)
            if self._render_thread is not None:
                self._render_thread.join(timeout=30)

    def log_batch_samples(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        step: int,
        viz_payload: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.should_log(step):
            return []

        try:
            fixed_html_url = self._get_fixed_wandb_html_url()
            wandb_log_url_only = bool(self.viz_config.wandb_log_url_only)
            fixed_html_path = None
            if fixed_html_url and self.viz_config.log_to_wandb:
                fixed_html_path = self._cache_fixed_html_snapshot(fixed_html_url)
                if fixed_html_path is not None:
                    logger.info(
                        "Using fixed training data visualization HTML for wandb at step %s: %s",
                        step,
                        fixed_html_url,
                    )

            use_fixed_snapshot_only = (
                bool(fixed_html_url)
                and self.viz_config.log_to_wandb
                and not self.viz_config.upload_to_s3
                and (wandb_log_url_only or fixed_html_path is not None)
            )
            if use_fixed_snapshot_only and fixed_html_url:
                num_samples = self._num_html_samples(input_ids.size(0))
                return [
                    {
                        "step": step,
                        "sample_id": sample_id,
                        "json_path": None,
                        "html_path": None if wandb_log_url_only else fixed_html_path,
                        "html_url": fixed_html_url,
                        "step_html_url": None,
                        "html_content": None,
                        "wandb_log_url_only": wandb_log_url_only,
                    }
                    for sample_id in range(num_samples)
                ]

            num_samples = self._num_html_samples(input_ids.size(0))
            if num_samples == 0:
                return []

            logger.info(
                "Queueing %s training data samples for visualization at step %s",
                num_samples,
                step,
            )

            payload = self._build_step_payload(
                input_ids=input_ids,
                labels=labels,
                step=step,
                viz_payload=viz_payload,
            )
            if fixed_html_url:
                payload["reference_html_url"] = fixed_html_url

            json_path = self._write_step_json(payload, step)
            html_path = self._build_step_html_path(step)
            html_content = self._compose_step_html(payload, step)
            s3_bucket, s3_key, html_url = self._build_s3_location(
                step,
                use_fixed_url=False,
            )
            step_html_url = html_url
            if not self._enqueue_render_job(
                _TrainingDataRenderJob(
                    step=step,
                    html_path=html_path,
                    html_content=html_content,
                    s3_bucket=s3_bucket,
                    s3_key=s3_key,
                )
            ):
                return []

            use_fixed_wandb_html = bool(fixed_html_url) and (
                wandb_log_url_only or fixed_html_path is not None
            )
            wandb_html_url = fixed_html_url if use_fixed_wandb_html else step_html_url
            wandb_html_path = None if wandb_log_url_only else (fixed_html_path or html_path)
            inline_html_content = None
            if (
                self.viz_config.log_to_wandb
                and not wandb_log_url_only
                and fixed_html_path is None
            ):
                inline_html_content = html_content

            simple_samples = self._build_simple_samples_from_payload(payload)
            for idx, sample in enumerate(simple_samples):
                sample["json_path"] = json_path
                sample["html_path"] = wandb_html_path
                sample["html_url"] = wandb_html_url
                sample["step_html_url"] = (
                    step_html_url if step_html_url != wandb_html_url else None
                )
                sample["html_content"] = inline_html_content if idx == 0 else None
                sample["wandb_log_url_only"] = wandb_log_url_only
            return simple_samples
        except Exception as exc:
            logger.error(f"Error logging training data samples: {exc}")
            return []
