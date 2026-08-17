# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import gc
import os
import subprocess
import time
from dataclasses import dataclass
from types import ModuleType
from typing import Generator, Optional

import torch
import torch.distributed as dist
from torch._utils import _get_available_device_type, _get_device_module

from torchtitan.tools.logging import logger


def has_cuda_capability(major: int, minor: int) -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (
        major,
        minor,
    )


def get_device_info() -> tuple[str, ModuleType]:
    device_type = _get_available_device_type() or "cuda"
    device_module = _get_device_module(device_type)  # default device_module:torch.cuda
    return device_type, device_module


device_type, device_module = get_device_info()


# used to avoid stragglers in garbage collection
class GarbageCollection:
    def __init__(self, gc_freq: int = 1000, debug: bool = False):
        assert gc_freq > 0, "gc_freq must be a positive integer"
        self.gc_freq = gc_freq
        self.debug = debug
        gc.disable()
        self.collect("Initial GC collection")
        if debug:
            from torch.utils.viz._cycles import warn_tensor_cycles

            if torch.distributed.get_rank() == 0:
                warn_tensor_cycles()

    def run(self, step_count: int):
        if self.debug:
            self.collect(
                "Force GC to perform collection to obtain debug information",
                generation=2,
            )
            gc.collect()
        elif step_count > 1 and step_count % self.gc_freq == 0:
            self.collect("Performing periodic GC collection")

    @staticmethod
    def collect(reason: str, generation: int = 1):
        begin = time.monotonic()
        gc.collect(generation)
        logger.info("[GC] %s took %.2f seconds", reason, time.monotonic() - begin)


# hardcoded BF16 type peak flops for NVIDIA A100, H100, H200, B200 GPU and AMD MI250, MI300X, MI325X, MI355X and Intel PVC
def get_peak_flops(device_name: str) -> float:
    try:
        # Run the lspci command and capture the output
        result = subprocess.run(["lspci"], stdout=subprocess.PIPE, text=True)
        # Filter the output for lines containing both "NVIDIA" and "H100"
        filtered_lines = [
            line
            for line in result.stdout.splitlines()
            if "NVIDIA" in line and "H100" in line
        ]
        # Join all filtered lines into a single string
        device_name = " ".join(filtered_lines) or device_name
    except FileNotFoundError as e:
        logger.warning(f"Error running lspci: {e}, fallback to use device_name")
    if "A100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/a100/
        return 312e12
    elif "H100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h100/
        # NOTE: Specifications are one-half lower without sparsity.
        if "NVL" in device_name:
            return 835e12
        elif "PCIe" in device_name:
            return 756e12
        else:  # for H100 SXM and other variants
            return 989e12
    elif "H200" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h200/
        return 989e12
    elif "B200" in device_name:
        # data from https://nvdam.widen.net/s/wwnsxrhm2w/blackwell-datasheet-3384703
        return 2.25e15
    elif "MI355X" in device_name:
        # MI355X data from https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html
        return 2500e12
    elif "MI300X" in device_name or "MI325X" in device_name:
        # MI300X data from https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
        # MI325X data from https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html
        return 1300e12
    elif "MI250X" in device_name:
        # data from https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html (per GCD)
        return 191.5e12
    elif "Data Center GPU Max 1550" in device_name:
        # Also known as Ponte Vecchio (PVC).
        # data from https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/intel-xe-gpu-architecture.html
        # Dot Product Accumulate Systolic (DPAS):
        # - Freq: 1300MHz
        # - #ops: 512
        # Full EU mode (i.e. 512 max compute units): 340.8 TFLOPS (BF16)
        # Standard EU mode (i.e. 448 max compute units): 298.2 TFLOPS (BF16)
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6
    elif "l40s" in device_name:
        # data from: "https://resources.nvidia.com/en-us-l40s/l40s-datasheet-28413"
        return 362e12

    else:  # for other GPU types, assume A100
        logger.warning(f"Peak flops undefined for: {device_name}, fallback to A100")
        return 312e12


@dataclass(frozen=True)
class Color:
    black = "\033[30m"
    red = "\033[31m"
    green = "\033[32m"
    yellow = "\033[33m"
    blue = "\033[34m"
    magenta = "\033[35m"
    cyan = "\033[36m"
    white = "\033[37m"
    reset = "\033[39m"
    orange = "\033[38;2;180;60;0m"
    turquoise = "\033[38;2;54;234;195m"


@dataclass(frozen=True)
class NoColor:
    black = ""
    red = ""
    green = ""
    yellow = ""
    blue = ""
    magenta = ""
    cyan = ""
    white = ""
    reset = ""
    orange = ""
    turquoise = ""


assert set(NoColor.__dataclass_fields__.keys()) == set(
    Color.__dataclass_fields__.keys()
), "NoColor must have the same fields as Color."


def check_if_feature_in_pytorch(
    feature_name: str,
    pull_request: str,
    min_nightly_version: Optional[str] = None,
) -> None:
    if "git" in torch.__version__:  # pytorch is built from source
        # notify users to check if the pull request is included in their pytorch
        logger.warning(
            "Detected that the pytorch is built from source. Please make sure the PR "
            f"({pull_request}) is included in pytorch for correct {feature_name}."
        )
    elif min_nightly_version is not None and torch.__version__ < min_nightly_version:
        logger.warning(
            f"Detected that the pytorch version {torch.__version__} is older than "
            f"{min_nightly_version}. Please upgrade a newer version to include the "
            f"change in ({pull_request}) for correct {feature_name}."
        )


@contextlib.contextmanager
def set_default_dtype(dtype: torch.dtype) -> Generator[None, None, None]:
    """
    Context manager to set torch's default dtype.

    Args:
        dtype (torch.dtype): The desired default dtype inside the context manager.

    Returns:
        ContextManager: context manager for setting default dtype.

    Example:
        >>> with set_default_dtype(torch.bfloat16):
        >>>     x = torch.tensor([1, 2, 3])
        >>>     x.dtype
        torch.bfloat16


    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def _round_up(x: int, y: int) -> int:
    """Round up x to the nearest multiple of y."""
    x_ceil_div_y = (x + y - 1) // y
    return x_ceil_div_y * y


def _is_local_rank_zero() -> bool:
    """Check if current process is local rank 0 (rank 0 on each node)."""
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def _download_s3_file(
    s3_file: str, local_file: str, max_retries: int = 5, use_s5cmd: bool = True
) -> tuple[str, bool]:
    """Download a single file from S3 with retry. Skips if already exists."""
    if os.path.exists(local_file):
        return f"Skipped {os.path.basename(local_file)}", True

    os.makedirs(os.path.dirname(local_file), exist_ok=True)
    cmd = (
        ["s5cmd", "cp", s3_file, local_file]
        if use_s5cmd
        else ["aws", "s3", "cp", s3_file, local_file]
    )

    for attempt in range(max_retries):
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return f"Downloaded {os.path.basename(local_file)}", True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
            return f"Failed {s3_file} after {max_retries} attempts: {e}", False


def _list_s3_files(s3_path: str) -> list[str]:
    """List all files under an S3 prefix. Returns full s3:// paths."""
    result = subprocess.run(
        ["aws", "s3", "ls", s3_path.rstrip("/") + "/", "--recursive"],
        capture_output=True,
        text=True,
        check=True,
    )
    bucket = s3_path.replace("s3://", "").split("/")[0]
    files = []
    for line in result.stdout.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 4:
            files.append(f"s3://{bucket}/{parts[-1]}")
    return files


def _parallel_s3_download(
    s3_path: str,
    local_path: str,
    max_workers: int = 256,
    use_s5cmd: bool = True,
) -> None:
    """Download all files from an S3 prefix to local dir using parallel threads.

    Preserves directory structure. Skips files that already exist locally.
    """
    import concurrent.futures

    files = _list_s3_files(s3_path)
    if not files:
        logger.warning(f"No files found at {s3_path}, falling back to aws s3 sync")
        subprocess.run(
            ["aws", "s3", "sync", s3_path, local_path, "--quiet"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        return

    s3_base = s3_path.rstrip("/") + "/"
    tool = "s5cmd" if use_s5cmd else "aws"
    logger.info(
        f"Downloading {len(files)} files with {max_workers} threads using {tool}"
    )

    downloaded, skipped, failed = 0, 0, 0

    def task(s3_file: str) -> tuple[str, bool]:
        # Compute relative path and map to local
        rel = s3_file.replace(s3_base, "")
        if not rel:
            # File is at the prefix root, use basename
            rel = os.path.basename(s3_file)
        local_file = os.path.join(local_path, rel)
        return _download_s3_file(s3_file, local_file, use_s5cmd=use_s5cmd)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(task, f): f for f in files}
        for future in concurrent.futures.as_completed(futures):
            msg, ok = future.result()
            if ok:
                if "Skipped" in msg:
                    skipped += 1
                else:
                    downloaded += 1
            else:
                failed += 1
                logger.warning(msg)
            total = downloaded + skipped + failed
            if total % 100 == 0 or total == len(files):
                logger.info(
                    f"[{total}/{len(files)}] Downloaded: {downloaded}, "
                    f"Skipped: {skipped}, Failed: {failed}"
                )

    logger.info(
        f"S3 download complete: {downloaded} downloaded, "
        f"{skipped} skipped, {failed} failed"
    )
    if failed > 0:
        raise RuntimeError(f"{failed} files failed to download from {s3_path}")


_S3_CACHE_ROOT = os.environ.get(
    "TORCHTITAN_S3_CACHE_ROOT",
    os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "torchtitan",
        "s3",
    ),
)


def cache_s3_path_if_needed(
    path: str | None,
    category: str = "datasets",
    s3_base_prefix: str | None = None,
    max_workers: int = 256,
    use_s5cmd: bool = True,
) -> str | None:
    """If path is an S3 path, download to local cache and return local path.

    Multi-node safe: only LOCAL_RANK=0 on each node downloads, then all ranks
    wait at a barrier. Uses parallel multi-threaded download (s5cmd/aws).

    Path mapping:
        Without s3_base_prefix: s3://bucket/path -> {cache_root}/{category}/bucket/path
        With s3_base_prefix:    s3://base/prefix/rel/path -> {cache_root}/{category}/rel/path

        ``cache_root`` defaults to ``$XDG_CACHE_HOME/torchtitan/s3`` (or
        ``~/.cache/torchtitan/s3``) and can be overridden with
        ``TORCHTITAN_S3_CACHE_ROOT``.

    Categories:
        - "datasets"    : training/validation datasets
        - "weights"     : HF pretrained assets (model weights, tokenizer, config)
        - "checkpoints" : DCP checkpoints saved/resumed during training

    Args:
        path: Path to file/directory (local or s3://)
        category: Cache subdirectory to separate different asset types
        s3_base_prefix: S3 root prefix to strip from path (e.g. s3_upload_path).
            The relative part after this prefix is used as the local cache subpath.
        max_workers: Number of parallel download threads
        use_s5cmd: Use s5cmd (True) or aws CLI (False) for downloading
    """
    if not path or not path.startswith("s3://"):
        return path

    local_cache_base = os.path.join(_S3_CACHE_ROOT, category)
    if s3_base_prefix and path.startswith(s3_base_prefix.rstrip("/") + "/"):
        # Strip the S3 root prefix, keep only the relative part
        rel_path = path[len(s3_base_prefix.rstrip("/")) + 1:]
        local_path = os.path.join(local_cache_base, rel_path)
    else:
        # Fallback: s3://bucket/path -> {cache_base}/bucket/path
        local_path = path.replace("s3://", f"{local_cache_base}/")

    if _is_local_rank_zero():
        if os.path.isdir(local_path) and os.listdir(local_path):
            logger.info(f"S3 cache hit: {local_path}")
        else:
            logger.info(f"Downloading S3: {path} -> {local_path}")
            os.makedirs(local_path, exist_ok=True)
            _parallel_s3_download(
                path, local_path, max_workers=max_workers, use_s5cmd=use_s5cmd
            )

    # Wait for local rank 0 on all nodes to finish
    if dist.is_initialized():
        dist.barrier()

    return local_path
