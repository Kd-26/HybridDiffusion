#!/usr/bin/env bash
# Run the existing TorchTitan training entry point from persistent storage.
#
# Local/single-node usage:
#   GPUS_PER_NODE=8 bash scripts/run_training.sh path/to/config.toml \
#     [extra TorchTitan arguments]
#
# Multi-stage usage uses the same public entry point:
#   bash scripts/run_training.sh path/to/pipeline.stages.conf
#
# Inside a Slurm allocation, the same command supports one or more nodes. One
# srun task is created per node and each task starts a local torchrun worker.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

log() {
    echo "[train] $*"
}

activate_environment() {
    if [[ -n "${VENV_PATH:-}" ]]; then
        if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
            echo "VENV_PATH does not contain bin/activate: ${VENV_PATH}" >&2
            exit 1
        fi
        # shellcheck disable=SC1090
        source "${VENV_PATH}/bin/activate"
    fi
}

run_worker() {
    activate_environment
    cd "${REPO_ROOT}"

    local node_rank="${SLURM_PROCID:-${NODE_RANK:-0}}"
    local metrics_args=(--metrics.no-enable-wandb --metrics.no-enable-tensorboard)
    local optional_args=()

    if [[ "${ENABLE_WANDB}" == "1" ]]; then
        metrics_args[0]=--metrics.enable-wandb
    fi
    if [[ "${ENABLE_TENSORBOARD}" == "1" ]]; then
        metrics_args[1]=--metrics.enable-tensorboard
    fi
    if [[ -n "${MODEL_PATH:-}" ]]; then
        optional_args+=(--model.hf_assets_path="${MODEL_PATH}")
    fi
    if [[ -n "${RUN_NAME:-}" ]]; then
        optional_args+=(--job.run_name="${RUN_NAME}")
    fi
    if [[ -n "${TRAINING_STEPS:-}" ]]; then
        optional_args+=(--training.steps="${TRAINING_STEPS}")
    fi
    if [[ "${DISABLE_CHECKPOINT:-0}" == "1" ]]; then
        optional_args+=(--checkpoint.no-enable)
    fi

    log "node_rank=${node_rank}/${NNODES} gpus_per_node=${GPUS_PER_NODE}"
    log "config=${CONFIG_FILE} output=${OUTPUT_DIR}"

    exec torchrun \
        --nnodes="${NNODES}" \
        --nproc_per_node="${GPUS_PER_NODE}" \
        --node_rank="${node_rank}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        -m torchtitan.train \
        --job.config_file="${CONFIG_FILE}" \
        --job.dump_folder="${OUTPUT_DIR}" \
        "${metrics_args[@]}" \
        "${optional_args[@]}" \
        "$@"
}

if [[ "${1:-}" == "__worker" ]]; then
    shift
    run_worker "$@"
fi

CONFIG_FILE="${CONFIG_FILE:-}"
if [[ -z "${CONFIG_FILE}" && $# -gt 0 && "${1}" != --* ]]; then
    CONFIG_FILE="$1"
    shift
fi
if [[ -z "${CONFIG_FILE}" ]]; then
    echo "Set CONFIG_FILE or pass a config path as the first argument." >&2
    exit 2
fi

cd "${REPO_ROOT}"
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Config file not found: ${CONFIG_FILE}" >&2
    exit 2
fi
CONFIG_FILE="$(readlink -f "${CONFIG_FILE}")"

if [[ "${CONFIG_FILE}" == *.stages.conf ]]; then
    exec bash "${SCRIPT_DIR}/multi_stage_train.sh" "${CONFIG_FILE}" "$@"
fi

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs}"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(readlink -f "${OUTPUT_DIR}")"

CACHE_ROOT="${CACHE_ROOT:-${OUTPUT_DIR}/cache}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
mkdir -p "${TMPDIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRITON_CACHE_DIR}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ENABLE_WANDB="${ENABLE_WANDB:-0}"
ENABLE_TENSORBOARD="${ENABLE_TENSORBOARD:-0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NNODES="${NNODES:-${SLURM_NNODES:-1}}"
TASK_NAME="test"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    if [[ -z "${SLURM_JOB_NODELIST:-}" ]]; then
        echo "SLURM_JOB_ID is set but SLURM_JOB_NODELIST is missing." >&2
        exit 2
    fi
    MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}"
    MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
else
    if [[ "${NNODES}" != "1" ]]; then
        echo "Multi-node execution requires a Slurm allocation." >&2
        exit 2
    fi
    MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    MASTER_PORT="${MASTER_PORT:-29500}"
fi

export CONFIG_FILE OUTPUT_DIR CACHE_ROOT ENABLE_WANDB ENABLE_TENSORBOARD
export GPUS_PER_NODE NNODES MASTER_ADDR MASTER_PORT REPO_ROOT VENV_PATH TASK_NAME
export MODEL_PATH DATA_ROOT RUN_NAME TRAINING_STEPS DISABLE_CHECKPOINT

log "repo=${REPO_ROOT}"
log "nodes=${NNODES} gpus_per_node=${GPUS_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    exec srun \
        --job-name="${TASK_NAME}" \
        --nodes="${NNODES}" \
        --ntasks="${NNODES}" \
        --ntasks-per-node=1 \
        --kill-on-bad-exit=1 \
        bash "${SCRIPT_PATH}" __worker "$@"
else
    run_worker "$@"
fi
