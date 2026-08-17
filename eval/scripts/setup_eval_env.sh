#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -n "${HYBRID_DIFFUSION_CACHE_ROOT:-}" ]]; then
    CACHE_ROOT="$HYBRID_DIFFUSION_CACHE_ROOT"
elif git_root="$(git -C "$PROJECT_ROOT" rev-parse --show-toplevel 2>/dev/null)"; then
    CACHE_ROOT="$git_root/cache"
else
    CACHE_ROOT="$PROJECT_ROOT/.cache"
fi

EVAL_VENV="${EVAL_VENV:-$CACHE_ROOT/venvs/hybrid-diffusion-eval}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.10 || true)}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
FLASHINFER_VENDOR="${FLASHINFER_VENDOR:-$CACHE_ROOT/vendor/flashinfer-python-0.6.7.post3}"
ENV_MANIFEST_DIR="${ENV_MANIFEST_DIR:-$CACHE_ROOT/env_manifests}"
FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$CACHE_ROOT/flashinfer}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$CACHE_ROOT/torch_extensions}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"

export FLASHINFER_WORKSPACE_BASE TORCH_EXTENSIONS_DIR TRITON_CACHE_DIR UV_CACHE_DIR

log() {
    printf '[setup-eval] %s\n' "$*"
}

die() {
    printf '[setup-eval] ERROR: %s\n' "$*" >&2
    exit 1
}

command -v uv >/dev/null 2>&1 || die "uv is required"
[[ -x "$PYTHON_BIN" ]] || die "Python interpreter is not executable: $PYTHON_BIN"

mkdir -p \
    "$UV_CACHE_DIR" \
    "$(dirname "$EVAL_VENV")" \
    "$ENV_MANIFEST_DIR" \
    "$FLASHINFER_WORKSPACE_BASE" \
    "$TORCH_EXTENSIONS_DIR" \
    "$TRITON_CACHE_DIR"

if [[ ! -x "$EVAL_VENV/bin/python" ]]; then
    log "Creating Python environment: $EVAL_VENV"
    uv venv --python "$PYTHON_BIN" "$EVAL_VENV"
else
    log "Reusing Python environment: $EVAL_VENV"
fi

PY="$EVAL_VENV/bin/python"

log "Installing the CUDA 12.8 PyTorch stack"
uv pip install --python "$PY" \
    --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.9.1" \
    "torchvision==0.24.1" \
    "torchaudio==2.9.1"

# The historical archive did not preserve FlashInfer's CUTLASS and spdlog
# submodule revisions. The public 0.6.7.post3 wheel contains the matching data
# tree, so extract that immutable wheel payload instead of cloning an unpinned
# current Git head.
if [[ ! -d "$FLASHINFER_VENDOR/flashinfer/data/cutlass/include" ]] || \
   [[ ! -d "$FLASHINFER_VENDOR/flashinfer/data/spdlog/include" ]]; then
    log "Extracting pinned FlashInfer build data: $FLASHINFER_VENDOR"
    mkdir -p "$FLASHINFER_VENDOR"
    uv pip install --target "$FLASHINFER_VENDOR" --no-deps \
        "flashinfer-python==0.6.7.post3"
else
    log "Reusing pinned FlashInfer build data: $FLASHINFER_VENDOR"
fi

runtime_packages=(
    "torch==2.9.1"
    "torchvision==0.24.1"
    "torchaudio==2.9.1"
    "flashinfer-python==0.6.7.post3"
    "flashinfer-cubin==0.6.7.post3"
    "flash-attn-4==4.0.0b10"
    "flash-linear-attention==0.4.2"
    "sglang-kernel==0.4.1"
    "nvidia-cutlass-dsl==4.4.2"
    "cuda-python==13.0.3"
    "cuda-pathfinder==1.5.3"
    "filelock==3.29.0"
    "quack-kernels==0.3.11"
    "transformers==5.5.4"
    "safetensors==0.7.0"
    "huggingface-hub==1.12.0"
    "mistral-common==1.11.0"
    "modelscope==1.36.2"
    "pillow==12.2.0"
    "python-multipart==0.0.26"
    "typer==0.24.2"
    "smg-grpc-proto==0.4.6"
    "smg-grpc-servicer==0.5.0"
    "protobuf==6.33.6"
    "IPython"
    "aiohttp==3.13.5"
    "apache-tvm-ffi>=0.1.5,<0.2"
    "anthropic>=0.20.0"
    "blobfile==3.0.0"
    "build"
    "compressed-tensors"
    "datasets==4.8.4"
    "einops"
    "fastapi"
    "gguf"
    "interegular"
    "llguidance>=0.7.11,<0.8.0"
    "msgspec"
    "ninja"
    "easydict"
    "numpy"
    "nvidia-ml-py"
    "openai-harmony==0.0.4"
    "openai==2.6.1"
    "orjson"
    "outlines==0.1.11"
    "packaging"
    "partial-json-parser"
    "prometheus-client>=0.20.0"
    "psutil"
    "py-spy"
    "pybase64"
    "pydantic"
    "pyzmq>=25.1.2"
    "requests"
    "scipy"
    "sentencepiece"
    "setproctitle"
    "soundfile==0.13.1"
    "tiktoken"
    "timm==1.0.16"
    "torch-memory-saver==0.0.9"
    "torchao==0.17.0"
    "torchcodec==0.9.1"
    "tqdm"
    "uvicorn"
    "uvloop"
    "watchfiles"
    "xgrammar==0.1.32"
    "kernels==0.12.0"
)

log "Installing the pinned SGLang/Qwen3.5 runtime closure"
uv pip install --python "$PY" \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match \
    "${runtime_packages[@]}"

eval_packages=(
    "lm-eval==0.4.11"
    "math-verify==0.9.0"
    "latex2sympy2-extended==1.11.0"
    "langdetect==1.0.9"
    "immutabledict==4.3.1"
    "human-eval==1.0.3"
)

log "Installing the pinned benchmark/scorer closure"
uv pip install --python "$PY" "${eval_packages[@]}"

FLASHINFER_SOURCE="$PROJECT_ROOT/third_party/flashinfer/flashinfer"
FLASHINFER_DATA="$FLASHINFER_VENDOR/flashinfer/data"
SOURCE_DATA_LINK="$FLASHINFER_SOURCE/data"

[[ -d "$FLASHINFER_DATA/cutlass/include" ]] || \
    die "Pinned FlashInfer CUTLASS data is missing: $FLASHINFER_DATA"
[[ -f "$FLASHINFER_DATA/cutlass/include/cutlass/arch/barrier.h" ]] || \
    die "Pinned FlashInfer CUTLASS headers are incomplete: $FLASHINFER_DATA"
[[ -d "$FLASHINFER_DATA/spdlog/include" ]] || \
    die "Pinned FlashInfer spdlog data is missing: $FLASHINFER_DATA"

if [[ -L "$SOURCE_DATA_LINK" ]]; then
    existing_target="$(readlink -f "$SOURCE_DATA_LINK")"
    expected_target="$(readlink -f "$FLASHINFER_DATA")"
    if [[ "$existing_target" != "$expected_target" ]]; then
        # This directory is generated and gitignored. A worktree copied from a
        # validated installation may retain a link to that installation's
        # external cache, so replace only the link and never its target.
        rm "$SOURCE_DATA_LINK"
    fi
elif [[ -e "$SOURCE_DATA_LINK" ]]; then
    die "Refusing to replace existing FlashInfer source data: $SOURCE_DATA_LINK"
fi
if [[ ! -e "$SOURCE_DATA_LINK" ]]; then
    vendor_rel="$($PY -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' \
        "$FLASHINFER_DATA" "$FLASHINFER_SOURCE")"
    ln -s "$vendor_rel" "$SOURCE_DATA_LINK"
    log "Linked pinned FlashInfer data: $SOURCE_DATA_LINK -> $vendor_rel"
fi

site_packages="$($PY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
pth_file="$site_packages/00-hybrid-diffusion-eval-local.pth"
project_root_repr="$($PY -c 'import json,sys; print(json.dumps(sys.argv[1]))' \
    "$PROJECT_ROOT")"
flashinfer_root_repr="$($PY -c 'import json,sys; print(json.dumps(sys.argv[1]))' \
    "$PROJECT_ROOT/third_party/flashinfer")"
printf 'import sys; sys.path[:0] = [%s, %s]\n' \
    "$project_root_repr" "$flashinfer_root_repr" > "$pth_file"

log "Checking Python dependency consistency"
uv pip check --python "$PY"

log "Checking source-overlay imports on the login host"
PROJECT_ROOT="$PROJECT_ROOT" "$PY" - <<'PY'
import importlib.metadata as md
import os

import flashinfer
import fla
import sglang
import torch
import transformers

project_root = os.path.realpath(os.environ["PROJECT_ROOT"])
expected = {
    "sglang": os.path.join(project_root, "sglang"),
    "flashinfer": os.path.join(project_root, "third_party", "flashinfer"),
}
for name, prefix in expected.items():
    module = {"sglang": sglang, "flashinfer": flashinfer}[name]
    location = os.path.realpath(module.__file__)
    if not location.startswith(prefix + os.sep):
        raise SystemExit(f"{name} is shadowed by a package outside the worktree: {location}")

print("python source-overlay imports: PASS")
print("torch:", torch.__version__, "cuda runtime:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("fla:", fla.__version__)
print("flashinfer:", flashinfer.__version__, flashinfer.__file__)
print("sglang:", sglang.__version__, sglang.__file__)
print("sglang-kernel metadata:", md.version("sglang-kernel"))
print("CUDA available on this host:", torch.cuda.is_available())
PY

manifest_name="$(basename "$EVAL_VENV")"
uv pip freeze --python "$PY" > "$ENV_MANIFEST_DIR/$manifest_name.freeze.txt"
{
    printf 'python=%s\n' "$($PY -c 'import sys; print(sys.version.split()[0])')"
    printf 'python_executable=%s\n' "$PY"
    printf 'project_root=%s\n' "$PROJECT_ROOT"
    printf 'sglang_upstream_commit=%s\n' "7c399f3c82eb5b92f183022b0661a7a334e3059d"
    printf 'flashinfer_public_version=%s\n' "0.6.7.post3"
    printf 'flashinfer_workspace_base=%s\n' "$FLASHINFER_WORKSPACE_BASE"
    printf 'torch_extensions_dir=%s\n' "$TORCH_EXTENSIONS_DIR"
    printf 'triton_cache_dir=%s\n' "$TRITON_CACHE_DIR"
} > "$ENV_MANIFEST_DIR/$manifest_name.context.txt"

log "Environment ready: $EVAL_VENV"
log "Freeze manifest: $ENV_MANIFEST_DIR/$manifest_name.freeze.txt"
log "GPU/native-kernel imports still require an allocated CUDA node"
exit 0
