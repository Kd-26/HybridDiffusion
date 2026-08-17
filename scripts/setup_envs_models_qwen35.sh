#!/usr/bin/env bash
# Create a persistent, reproducible Qwen3.5 TorchTitan environment with uv.
#
# The default installation is sufficient for the text-only Qwen3.5 DLLM
# training path. FlashAttention is optional because the paper configs use
# PyTorch FlexAttention plus the FLA GatedDeltaNet kernels.
#
# Examples:
#   bash scripts/setup_envs_models_qwen35.sh
#   bash scripts/setup_envs_models_qwen35.sh --venv /persistent/venvs/hybrid-diffusion-train
#   bash scripts/setup_envs_models_qwen35.sh --venv /persistent/venvs/hybrid-diffusion-train --with-dev
#   bash scripts/setup_envs_models_qwen35.sh --model-id Qwen/Qwen3.5-2B --model-dir /persistent/models/Qwen3.5-2B

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -n "${HYBRID_DIFFUSION_CACHE_ROOT:-}" ]]; then
    CACHE_ROOT="$HYBRID_DIFFUSION_CACHE_ROOT"
elif git_root="$(git -C "$PROJECT_ROOT" rev-parse --show-toplevel 2>/dev/null)"; then
    CACHE_ROOT="$git_root/cache"
else
    CACHE_ROOT="$PROJECT_ROOT/.cache"
fi

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

# These pinned top-level versions define the reproducible environment validated
# on H100 GPUs. Override them through environment variables when needed.
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.57.1}"
DATASETS_VERSION="${DATASETS_VERSION:-3.6.0}"
FLA_VERSION="${FLA_VERSION:-0.5.1}"
TILELANG_VERSION="${TILELANG_VERSION:-0.1.12}"
LIGER_VERSION="${LIGER_VERSION:-0.8.0}"
WANDB_VERSION="${WANDB_VERSION:-0.28.1}"

VENV_DIR="${VENV_PATH:-${UV_PROJECT_ENVIRONMENT:-${CACHE_ROOT}/venvs/hybrid-diffusion-train}}"
WITH_DEV=0
WITH_FLASH_ATTN=0
RECREATE=0
VERIFY_ONLY=0
MODEL_ID=""
MODEL_DIR=""

usage() {
    cat <<'EOF'
Usage: setup_envs_models_qwen35.sh [options]

Options:
  --venv PATH                  Persistent virtual-environment path.
  --uv-project-environment PATH
                               Backward-compatible alias for --venv.
  --with-dev                   Install the repository's test/development tools.
  --with-flash-attn            Install FlashAttention 2 (not needed by the core
                               Qwen3.5 DLLM configs).
  --model-id ID                Optionally download a public Hugging Face model.
  --model-dir PATH             Destination paired with --model-id.
  --recreate                   Recreate only the selected virtual environment.
  --verify-only                Skip installation and verify an existing venv.
  -h, --help                   Show this help.

Environment overrides:
  HYBRID_DIFFUSION_CACHE_ROOT, PYTHON_VERSION, TORCH_VERSION, TORCH_INDEX_URL, PYPI_INDEX_URL,
  TRANSFORMERS_VERSION, DATASETS_VERSION,
  FLA_VERSION, TILELANG_VERSION, LIGER_VERSION, WANDB_VERSION,
  FLASH_ATTN_VERSION,
  FLASH_ATTN_WHEEL, FLASH_ATTN3_WHEEL, UV_CACHE_DIR, HF_HOME, HF_TOKEN.

No token is read from a file or embedded in this script. If a gated model is
requested, pass HF_TOKEN only through the process environment.
EOF
}

die() {
    echo "[env] ERROR: $*" >&2
    exit 2
}

log() {
    echo "[env] $*"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv|--uv-project-environment|--uv_project_environment)
            [[ $# -ge 2 ]] || die "$1 requires a path"
            VENV_DIR="$2"
            shift 2
            ;;
        --with-dev)
            WITH_DEV=1
            shift
            ;;
        --with-flash-attn)
            WITH_FLASH_ATTN=1
            shift
            ;;
        --model-id)
            [[ $# -ge 2 ]] || die "$1 requires a model ID"
            MODEL_ID="$2"
            shift 2
            ;;
        --model-dir)
            [[ $# -ge 2 ]] || die "$1 requires a path"
            MODEL_DIR="$2"
            shift 2
            ;;
        --recreate)
            RECREATE=1
            shift
            ;;
        --verify-only)
            VERIFY_ONLY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1 (use --help)"
            ;;
    esac
done

if [[ -n "${MODEL_ID}" || -n "${MODEL_DIR}" ]]; then
    [[ -n "${MODEL_ID}" && -n "${MODEL_DIR}" ]] || \
        die "--model-id and --model-dir must be supplied together"
fi

if [[ -n "${FLASH_ATTN_WHEEL:-}" && ! -f "${FLASH_ATTN_WHEEL}" ]]; then
    die "FLASH_ATTN_WHEEL does not exist: ${FLASH_ATTN_WHEEL}"
fi
if [[ -n "${FLASH_ATTN3_WHEEL:-}" && ! -f "${FLASH_ATTN3_WHEEL}" ]]; then
    die "FLASH_ATTN3_WHEEL does not exist: ${FLASH_ATTN3_WHEEL}"
fi

if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
else
    die "uv is required; install it from https://docs.astral.sh/uv/"
fi

create_environment() {
    if [[ "${RECREATE}" == "1" ]]; then
        log "recreating virtual environment: ${VENV_DIR}"
        "${UV_BIN}" venv --clear --python "${PYTHON_VERSION}" "${VENV_DIR}"
    elif [[ -f "${VENV_DIR}/pyvenv.cfg" && -x "${VENV_DIR}/bin/python" ]]; then
        log "reusing virtual environment: ${VENV_DIR}"
    elif [[ -e "${VENV_DIR}" ]]; then
        die "target exists but is not a valid virtual environment: ${VENV_DIR}"
    else
        log "creating virtual environment: ${VENV_DIR}"
        "${UV_BIN}" venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
    fi
}

uv_pip() {
    "${UV_BIN}" pip "$@" --python "${VENV_DIR}/bin/python"
}

install_environment() {
    log "installing PyTorch ${TORCH_VERSION} from ${TORCH_INDEX_URL}"
    uv_pip install \
        "torch==${TORCH_VERSION}" \
        --default-index "${TORCH_INDEX_URL}"

    log "installing TorchTitan and Qwen3.5 training dependencies from public PyPI"
    uv_pip install \
        --default-index "${PYPI_INDEX_URL}" \
        -e "${PROJECT_ROOT}" \
        "datasets==${DATASETS_VERSION}" \
        "transformers==${TRANSFORMERS_VERSION}" \
        "flash-linear-attention[cuda]==${FLA_VERSION}" \
        "tilelang==${TILELANG_VERSION}" \
        "liger-kernel==${LIGER_VERSION}" \
        "wandb==${WANDB_VERSION}"

    if [[ "${WITH_DEV}" == "1" ]]; then
        log "installing development/test dependencies"
        uv_pip install \
            --default-index "${PYPI_INDEX_URL}" \
            -e "${PROJECT_ROOT}[dev]"
    fi

    if [[ -n "${FLASH_ATTN_WHEEL:-}" ]]; then
        log "installing FlashAttention 2 from the explicitly supplied wheel"
        uv_pip install "${FLASH_ATTN_WHEEL}" --no-deps
    elif [[ "${WITH_FLASH_ATTN}" == "1" ]]; then
        log "building optional FlashAttention 2 from public PyPI"
        uv_pip install ninja packaging --default-index "${PYPI_INDEX_URL}"
        uv_pip install \
            "flash-attn==${FLASH_ATTN_VERSION:-2.8.3}" \
            --no-build-isolation \
            --default-index "${PYPI_INDEX_URL}"
    fi

    if [[ -n "${FLASH_ATTN3_WHEEL:-}" ]]; then
        log "installing optional FlashAttention 3 from the supplied wheel"
        uv_pip install "${FLASH_ATTN3_WHEEL}" --no-deps
    fi
}

verify_environment() {
    log "checking dependency consistency"
    uv_pip check

    log "verifying the Qwen3.5 import closure"
    "${VENV_DIR}/bin/python" - <<'PY'
from importlib.metadata import version
import sys

import datasets
import fla
import torch
import transformers
import triton
import tilelang
import wandb
from fla.modules.convolution import causal_conv1d
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy
from torchtitan.models.qwen3_5 import qwen3_5_args, qwen3_5_dllm_args

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}; cuda_runtime={torch.version.cuda}")
print(f"triton={triton.__version__}")
print(f"transformers={transformers.__version__}")
print(f"datasets={datasets.__version__}")
print(f"flash-linear-attention={version('flash-linear-attention')}")
print(f"tilelang={version('tilelang')}")
print(f"liger-kernel={version('liger-kernel')}")
print(f"wandb={wandb.__version__}")
print(f"qwen3_5_flavors={','.join(sorted(qwen3_5_args))}")
assert qwen3_5_args.keys() == qwen3_5_dllm_args.keys()
assert callable(causal_conv1d)
assert callable(chunk_gated_delta_rule)
assert callable(liger_fused_linear_cross_entropy)
print("qwen3_5_import_check=OK")
PY
}

download_model() {
    [[ -n "${MODEL_ID}" ]] || return 0
    mkdir -p "${MODEL_DIR}"
    log "downloading ${MODEL_ID} to ${MODEL_DIR}"
    "${VENV_DIR}/bin/hf" download "${MODEL_ID}" --local-dir "${MODEL_DIR}"
}

main() {
    log "project=${PROJECT_ROOT}"
    log "uv=$(${UV_BIN} --version)"
    log "venv=${VENV_DIR}"

    if [[ "${VERIFY_ONLY}" == "0" ]]; then
        create_environment
        install_environment
    else
        [[ -x "${VENV_DIR}/bin/python" ]] || die "venv not found: ${VENV_DIR}"
    fi

    verify_environment
    download_model

    log "setup complete"
    log "activate with: source ${VENV_DIR}/bin/activate"
}

main
