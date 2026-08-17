#!/bin/bash
# Multi-stage sequential training with automatic stage transition and
# crash-safe resume.
#
# This is an implementation helper. Prefer the public launcher:
#   bash scripts/run_training.sh <stages_conf> [extra TorchTitan arguments]
#
# The stages_conf file defines one stage per line:
#   <config_file>  <total_steps>
#
# The script determines which stage to run by inspecting checkpoint folders.
# On crash and re-launch it automatically resumes the correct stage.

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly TORCHTITAN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${TORCHTITAN_ROOT}"
source "${SCRIPT_DIR}/_checkpoint_utils.sh"

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/multi_stage_train.sh <stages_conf>" >&2
    exit 2
fi
readonly STAGES_CONF="$1"
[[ ! -f "${STAGES_CONF}" ]] && { echo "Error: stages config not found: ${STAGES_CONF}" >&2; exit 2; }
shift
readonly PASSTHROUGH_ARGS=("$@")

OUTPUT_DIR="${OUTPUT_DIR:-${TORCHTITAN_ROOT}/outputs}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TRIAL_NAME="${TRIAL_NAME:-}"

# ── Environment ────────────────────────────────────────────────────────
export PYTHONPATH="${TORCHTITAN_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
CACHE_ROOT="${CACHE_ROOT:-${OUTPUT_DIR}/cache}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
mkdir -p "${OUTPUT_DIR}" "${TMPDIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRITON_CACHE_DIR}"

# ── Logging helpers ────────────────────────────────────────────────────
log_info()  { echo "[STAGE] $*"; }
log_warn()  { echo "[STAGE] WARNING: $*"; }

# ── Parse stages.conf ─────────────────────────────────────────────────
# Reads non-empty, non-comment lines into parallel arrays.
STAGE_CONFIGS=()
STAGE_STEPS=()
while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"          # strip comments
    line="$(echo "$line" | xargs)"  # trim whitespace
    [[ -z "$line" ]] && continue
    read -r cfg steps <<< "$line"
    STAGE_CONFIGS+=("$cfg")
    STAGE_STEPS+=("$steps")
done < "${STAGES_CONF}"

NUM_STAGES=${#STAGE_CONFIGS[@]}
[[ $NUM_STAGES -eq 0 ]] && { echo "Error: no stages defined in ${STAGES_CONF}"; exit 1; }

log_info "Loaded ${NUM_STAGES} stages from ${STAGES_CONF}"

# ── Checkpoint helpers (from _checkpoint_utils.sh) ────────────────────
# read_toml_value, get_user_name, compute_local_ckpt_folder,
# compute_s3_ckpt_folder, find_latest_local_step, find_latest_s3_step,
# sync_checkpoint_from_s3, wait_for_s3_checkpoint are all provided
# by the sourced utils.

ckpt_folder_for_stage() {
    local config_file="$1" run_name_override="$2"
    compute_local_ckpt_folder "$config_file" "${OUTPUT_DIR}" "" "$run_name_override"
}

# ── Launch a training stage ───────────────────────────────────────────
run_stage() {
    local stage_idx="$1"
    local config_file="${STAGE_CONFIGS[$stage_idx]}"
    local total_steps="${STAGE_STEPS[$stage_idx]}"
    shift 1
    local extra_overrides=("$@")

    [[ ! -f "$config_file" ]] && { echo "Error: config not found: $config_file"; exit 1; }

    local wandb_name
    wandb_name="$(basename "${STAGES_CONF}" .stages.conf)_stage$((stage_idx+1))_of_${NUM_STAGES}"

    local stage_run_name="${CONF_BASENAME}_stage$((stage_idx+1))"

    log_info "Launching stage $((stage_idx+1))/${NUM_STAGES}: ${config_file} (${total_steps} steps)"
    log_info "  Run name: ${stage_run_name}"

    CONFIG_FILE="${config_file}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    GPUS_PER_NODE="${GPUS_PER_NODE}" \
    RUN_NAME="${stage_run_name}" \
    TRAINING_STEPS="${total_steps}" \
        bash "${SCRIPT_DIR}/run_training.sh" \
            --metrics.wandb_name="${wandb_name}" \
            "${extra_overrides[@]}" \
            "${PASSTHROUGH_ARGS[@]}"
}

# ── Post-stage verification ──────────────────────────────────────────
# If S3 is configured, wait for the checkpoint upload to finish and
# sync it locally before proceeding to the next stage.
verify_stage_completion() {
    local stage_idx="$1" config="$2" stage_run_name="$3" expected_steps="$4"
    local folder s3_folder post_latest

    folder="$(ckpt_folder_for_stage "$config" "$stage_run_name")"
    s3_folder="$(compute_s3_ckpt_folder "$config" "" "$stage_run_name")"

    if [[ -n "$s3_folder" ]]; then
        if ! wait_for_s3_checkpoint "$s3_folder" "$expected_steps" 600; then
            log_warn "Stage $((stage_idx+1)): timed out waiting for S3 checkpoint"
            exit 1
        fi
        sync_checkpoint_from_s3 "$config" "${OUTPUT_DIR}" "" "$stage_run_name"
    fi

    post_latest="$(find_latest_local_step "$folder")"
    if (( post_latest < expected_steps )); then
        log_warn "Stage $((stage_idx+1)) exited but only reached step ${post_latest}/${expected_steps}"
        exit 1
    fi
    log_info "Stage $((stage_idx+1)) finished (step ${post_latest})."
}

# ── Main: determine which stage to run / resume ──────────────────────
CONF_BASENAME="$(basename "${STAGES_CONF}" .stages.conf)"

log_info "=== Multi-Stage Training ==="
log_info "Output dir : ${OUTPUT_DIR}"
log_info "Pipeline   : ${CONF_BASENAME}"
log_info "GPUs/node  : ${GPUS_PER_NODE}"
log_info ""

for (( i=0; i<NUM_STAGES; i++ )); do
    config="${STAGE_CONFIGS[$i]}"
    steps="${STAGE_STEPS[$i]}"
    stage_run_name="${CONF_BASENAME}_stage$((i+1))"

    # Try to sync this stage's checkpoint from S3 if nothing local
    sync_checkpoint_from_s3 "$config" "${OUTPUT_DIR}" "" "$stage_run_name"

    folder="$(ckpt_folder_for_stage "$config" "$stage_run_name")"
    latest="$(find_latest_local_step "$folder")"

    log_info "Stage $((i+1)): config=$(basename "$config"), steps=${steps}, " \
             "ckpt_folder=${folder}, latest_step=${latest}"

    if (( latest >= steps )); then
        log_info "  -> COMPLETE (step ${latest} >= ${steps}), skipping."
        continue
    fi

    if (( latest >= 0 )); then
        log_info "  -> RESUMING from step ${latest}"
        run_stage "$i"
        verify_stage_completion "$i" "$config" "$stage_run_name" "$steps"
        continue
    fi

    # Remove empty checkpoint folder so torchtitan uses initial_load_path
    if [[ -d "$folder" ]] && (( latest < 0 )); then
        log_info "Removing empty checkpoint folder ${folder} so initial_load_path is used"
        rm -rf "$folder"
    fi

    # No checkpoint folder for this stage.
    if (( i == 0 )); then
        log_info "  -> STARTING fresh"
        run_stage "$i"
        verify_stage_completion "$i" "$config" "$stage_run_name" "$steps"
        continue
    fi

    # Later stage: inject initial_load_path from the previous stage's
    # last checkpoint (model weights only -- optimizer/LR/step reset).
    prev_config="${STAGE_CONFIGS[$((i-1))]}"
    prev_run_name="${CONF_BASENAME}_stage${i}"
    prev_folder="$(ckpt_folder_for_stage "$prev_config" "$prev_run_name")"
    prev_latest="$(find_latest_local_step "$prev_folder")"

    if (( prev_latest >= 0 )); then
        prev_ckpt_path="${prev_folder}/step-${prev_latest}"
        log_info "  -> STARTING with initial_load_path=${prev_ckpt_path} (model only)"
    else
        # Previous stage checkpoint not local -- check S3
        prev_s3_folder="$(compute_s3_ckpt_folder "$prev_config" "" "$prev_run_name")"
        if [[ -n "$prev_s3_folder" ]]; then
            prev_s3_latest="$(find_latest_s3_step "$prev_s3_folder")"
            if (( prev_s3_latest >= 0 )); then
                prev_ckpt_path="${prev_s3_folder%/}/step-${prev_s3_latest}"
                log_info "  -> STARTING with S3 initial_load_path=${prev_ckpt_path} (model only)"
            else
                log_warn "Previous stage has no checkpoints locally or on S3. Cannot start stage $((i+1))."
                exit 1
            fi
        else
            log_warn "Previous stage has no local checkpoints and no s3_upload_path configured. Cannot start stage $((i+1))."
            exit 1
        fi
    fi

    run_stage "$i" \
        --checkpoint.initial_load_path="${prev_ckpt_path}" \
        --checkpoint.initial_load_model_only
    verify_stage_completion "$i" "$config" "$stage_run_name" "$steps"
done

log_info ""
log_info "=== All ${NUM_STAGES} stages complete ==="
