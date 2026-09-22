#!/usr/bin/env python3
"""Fail-closed Cluster-3 efficiency checkpoint and A30 preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping


FROZEN_SHA = "cf5e14c2f5a4e4700fb66b3183dd021bbe722fe4"
EXPECTED_ARCHITECTURE = "Qwen3_5DLLMForConditionalGeneration"
REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "model-00001-of-00001.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)
QWEN35_2B_FINGERPRINT = {
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_hidden_layers": 24,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safetensors_header(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as source:
        raw_length = source.read(8)
        if len(raw_length) != 8:
            raise RuntimeError(f"truncated safetensors length header: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length <= 2 or header_length > path.stat().st_size - 8:
            raise RuntimeError(f"invalid safetensors header length: {path}")
        header = source.read(header_length)
    try:
        value = json.loads(header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid safetensors JSON header: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"safetensors header is not an object: {path}")
    return value


def validate_bf16_weights(path: Path) -> dict[str, Any]:
    header = _safetensors_header(path)
    tensors = {
        name: value
        for name, value in header.items()
        if name != "__metadata__" and isinstance(value, dict)
    }
    if not tensors:
        raise RuntimeError("checkpoint contains no safetensors tensor entries")
    dtype_counts: dict[str, int] = {}
    mismatches = []
    for name, value in tensors.items():
        dtype = str(value.get("dtype", ""))
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        if dtype != "BF16":
            mismatches.append({"name": name, "dtype": dtype})
    if mismatches:
        preview = mismatches[:20]
        raise RuntimeError(
            "checkpoint tensors are not uniformly BF16: "
            f"mismatch_count={len(mismatches)} preview={preview}"
        )
    return {"tensor_count": len(tensors), "dtype_counts": dtype_counts}


def _walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def extract_recorded_hashes(value: Any) -> list[dict[str, str]]:
    """Extract common checkpoint-inventory layouts from frozen evidence."""
    candidates: list[dict[str, str]] = []
    required = set(REQUIRED_CHECKPOINT_FILES)
    for node in _walk(value):
        if not isinstance(node, Mapping):
            continue
        direct: dict[str, str] = {}
        for key, child in node.items():
            filename = Path(str(key)).name
            if filename in required and isinstance(child, str) and len(child) == 64:
                direct[filename] = child.lower()
            elif filename in required and isinstance(child, Mapping):
                digest = child.get("sha256")
                if isinstance(digest, str) and len(digest) == 64:
                    direct[filename] = digest.lower()
        if direct:
            candidates.append(direct)

        path_value = node.get("path", node.get("name", node.get("file")))
        digest = node.get("sha256")
        if isinstance(path_value, str) and isinstance(digest, str):
            filename = Path(path_value).name
            if filename in required and len(digest) == 64:
                candidates.append({filename: digest.lower()})
    return candidates


def load_frozen_provenance(paths: Iterable[Path]) -> dict[str, str]:
    merged: dict[str, str] = {}
    used = []
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"frozen provenance file is missing: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        encoded = json.dumps(value, sort_keys=True).lower()
        if FROZEN_SHA not in encoded:
            continue
        for candidate in extract_recorded_hashes(value):
            for filename, digest in candidate.items():
                prior = merged.get(filename)
                if prior is not None and prior != digest:
                    raise RuntimeError(
                        "frozen evidence has conflicting checkpoint hashes: "
                        f"{filename}: {prior} != {digest}"
                    )
                merged[filename] = digest
        used.append(str(path))
    if not used:
        raise RuntimeError(
            f"no supplied provenance JSON records frozen revision {FROZEN_SHA}"
        )
    missing = sorted(set(REQUIRED_CHECKPOINT_FILES) - set(merged))
    if missing:
        raise RuntimeError(
            "frozen Cluster 1-3 evidence does not contain every required "
            f"checkpoint hash: missing={missing} files={used}"
        )
    return merged


def checkpoint_identity(model_path: Path) -> dict[str, Any]:
    missing = [
        name for name in REQUIRED_CHECKPOINT_FILES if not (model_path / name).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"checkpoint is missing required files at {model_path}: {missing}"
        )
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    architectures = config.get("architectures") or []
    if architectures != [EXPECTED_ARCHITECTURE]:
        raise RuntimeError(
            "checkpoint architecture mismatch: "
            f"expected={[EXPECTED_ARCHITECTURE]!r} observed={architectures!r}"
        )
    text_config = config.get("text_config", config)
    if not isinstance(text_config, Mapping):
        raise RuntimeError("checkpoint text_config is not an object")
    observed_scale = {name: text_config.get(name) for name in QWEN35_2B_FINGERPRINT}
    if observed_scale != QWEN35_2B_FINGERPRINT:
        raise RuntimeError(
            "checkpoint scale mismatch; expected HybridDiffusion-2B/Qwen3.5 "
            f"fingerprint={QWEN35_2B_FINGERPRINT}, observed={observed_scale}"
        )
    hashes = {
        name: sha256_file(model_path / name) for name in REQUIRED_CHECKPOINT_FILES
    }
    bf16 = validate_bf16_weights(model_path / "model-00001-of-00001.safetensors")
    return {
        "path": str(model_path.resolve()),
        "architecture": EXPECTED_ARCHITECTURE,
        "model_scale": "2B",
        "config_fingerprint": observed_scale,
        "files": {
            name: {
                "bytes": (model_path / name).stat().st_size,
                "sha256": hashes[name],
            }
            for name in REQUIRED_CHECKPOINT_FILES
        },
        "weights": bf16,
    }


def environment_identity() -> dict[str, Any]:
    torch = __import__("torch")
    cuda_available = bool(torch.cuda.is_available())
    gpus = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpus": gpus,
    }


def git_identity(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(("git", *args), cwd=repo, text=True).strip()

    base_type = run("cat-file", "-t", FROZEN_SHA)
    if base_type != "commit":
        raise RuntimeError(f"frozen object is not a commit: {base_type}")
    status = run("status", "--porcelain")
    if status:
        raise RuntimeError("working tree is not clean")
    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "frozen_base": FROZEN_SHA,
        "working_tree_clean": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--frozen-provenance",
        type=Path,
        action="append",
        required=True,
        help="JSON evidence containing the frozen SHA and checkpoint hashes; repeatable",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-a30", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = checkpoint_identity(args.model_path)
    recorded = load_frozen_provenance(args.frozen_provenance)
    observed = {
        name: details["sha256"] for name, details in checkpoint["files"].items()
    }
    mismatches = {
        name: {"recorded": recorded[name], "observed": observed[name]}
        for name in REQUIRED_CHECKPOINT_FILES
        if recorded[name] != observed[name]
    }
    if mismatches:
        raise RuntimeError(
            "checkpoint hashes differ from frozen Cluster 1-3 evidence: "
            f"{mismatches}"
        )
    environment = environment_identity()
    if args.require_a30:
        if (
            len(environment["gpus"]) != 1
            or environment["gpus"][0]["name"] != "NVIDIA A30"
        ):
            raise RuntimeError(
                "A30 acceptance requires exactly one NVIDIA A30: "
                f"observed={environment['gpus']}"
            )
        if environment["cuda"] is None:
            raise RuntimeError("A30 acceptance requires a CUDA-enabled PyTorch build")
    record = {
        "schema_version": 1,
        "repository": git_identity(args.repo),
        "checkpoint": checkpoint,
        "frozen_checkpoint_hashes": recorded,
        "checkpoint_hashes_match": True,
        "environment": environment,
        "a30_required": bool(args.require_a30),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
