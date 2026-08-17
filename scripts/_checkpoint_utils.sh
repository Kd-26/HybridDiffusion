#!/bin/bash
# Shared checkpoint utility functions for S3 auto-resume.
#
# Source this file from any training launch script:
#   source "$(dirname "${BASH_SOURCE[0]}")/_checkpoint_utils.sh"
#
# Provides:
#   read_toml_value           - simple TOML key reader
#   get_user_name             - checkpoint path user component
#   compute_local_ckpt_folder - local checkpoint directory
#   compute_s3_ckpt_folder    - S3 checkpoint directory (flat, no ckpt_subfolder)
#   find_latest_local_step    - latest valid step-N locally
#   find_latest_s3_step       - latest step-N on S3
#   sync_checkpoint_from_s3   - download latest S3 checkpoint if no local one exists

_ckpt_log() { echo -e "\033[35m[CKPT]\033[0m $*" >&2; }

# ── TOML helpers ──────────────────────────────────────────────────────

read_toml_value() {
    local file="$1" section="$2" key="$3" default="$4"
    local val
    val=$(awk -v sec="$section" -v key="$key" '
        /^\[/ { in_sec = ($0 ~ "^\\[" sec "\\]") }
        in_sec && $0 ~ "^[[:space:]]*" key "[[:space:]]*=" {
            sub(/^[^=]*=[[:space:]]*/, ""); sub(/[[:space:]]*#.*$/, "")
            gsub(/^"|"$/, ""); print; exit
        }
    ' "$file" 2>/dev/null)
    echo "${val:-$default}"
}

# ── User / path helpers ──────────────────────────────────────────────

get_user_name() {
    local u="${USER:-default}"
    echo "${u// /_}" | tr '[:upper:]' '[:lower:]'
}

_run_name_for_config() {
    local config_file="$1" trial_name="$2" run_name_override="${3:-}"
    if [[ -n "$run_name_override" ]]; then
        echo "$run_name_override"
        return
    fi
    local basename
    basename="$(basename "$config_file" .toml)"
    if [[ -n "$trial_name" ]]; then
        echo "${basename}_${trial_name}"
    else
        echo "$basename"
    fi
}

compute_local_ckpt_folder() {
    local config_file="$1" dump_folder="$2" trial_name="$3" run_name_override="${4:-}"
    local user_name run_name ckpt_subfolder
    user_name="$(get_user_name)"
    run_name="$(_run_name_for_config "$config_file" "$trial_name" "$run_name_override")"
    ckpt_subfolder="$(read_toml_value "$config_file" "checkpoint" "folder" "checkpoint")"
    echo "${dump_folder}/${user_name}/${run_name}/${ckpt_subfolder}"
}

# S3 upload path is flat: {s3_upload_path}/{user}/{run_name}
# (no ckpt_subfolder -- matches Python CheckpointManager._upload_checkpoint_to_s3)
compute_s3_ckpt_folder() {
    local config_file="$1" trial_name="$2" run_name_override="${3:-}"
    local s3_upload_path user_name run_name
    s3_upload_path="$(read_toml_value "$config_file" "checkpoint" "s3_upload_path" "")"
    [[ -z "$s3_upload_path" ]] && { echo ""; return; }
    user_name="$(get_user_name)"
    run_name="$(_run_name_for_config "$config_file" "$trial_name" "$run_name_override")"
    echo "${s3_upload_path%/}/${user_name}/${run_name}"
}

# ── Step discovery ───────────────────────────────────────────────────

find_latest_local_step() {
    local folder="$1"
    [[ ! -d "$folder" ]] && { echo -1; return; }
    local max=-1
    for d in "${folder}"/step-*; do
        [[ ! -d "$d" ]] && continue
        [[ ! -f "$d/.metadata" ]] && continue
        # Require at least one DCP shard file alongside .metadata
        local has_shards=false
        for f in "$d"/*.distcp; do
            [[ -f "$f" ]] && has_shards=true && break
        done
        [[ "$has_shards" == "false" ]] && continue
        local name
        name="$(basename "$d")"
        [[ "$name" =~ ^step-([0-9]+)$ ]] || continue
        local n="${BASH_REMATCH[1]}"
        (( n > max )) && max=$n
    done
    echo $max
}

find_latest_s3_step() {
    local s3_folder="$1"
    [[ -z "$s3_folder" ]] && { echo -1; return; }
    local max=-1
    local aws_output
    if ! aws_output=$(aws s3 ls "${s3_folder%/}/" 2>&1); then
        _ckpt_log "WARNING: aws s3 ls failed for ${s3_folder}: ${aws_output}"
        echo -1; return
    fi
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        local dirname
        dirname=$(echo "$line" | awk '{print $NF}')
        dirname="${dirname%/}"
        if [[ "$dirname" =~ ^step-([0-9]+)$ ]]; then
            local n="${BASH_REMATCH[1]}"
            (( n > max )) && max=$n
        fi
    done <<< "$aws_output"
    echo $max
}

# ── S3 sync orchestrator ────────────────────────────────────────────

sync_checkpoint_from_s3() {
    local config_file="$1" dump_folder="$2" trial_name="$3" run_name_override="${4:-}"

    local local_folder s3_folder
    local_folder="$(compute_local_ckpt_folder "$config_file" "$dump_folder" "$trial_name" "$run_name_override")"
    s3_folder="$(compute_s3_ckpt_folder "$config_file" "$trial_name" "$run_name_override")"

    local local_latest
    local_latest="$(find_latest_local_step "$local_folder")"

    if (( local_latest >= 0 )); then
        if [[ -n "$s3_folder" ]]; then
            local s3_check
            s3_check="$(find_latest_s3_step "$s3_folder")"
            if (( s3_check > local_latest )); then
                _ckpt_log "S3 has newer checkpoint (step-${s3_check} > local step-${local_latest}), syncing"
                local s3_newer_path="${s3_folder%/}/step-${s3_check}"
                local local_newer_path="${local_folder}/step-${s3_check}"
                mkdir -p "${local_newer_path}"
                if aws s3 sync "${s3_newer_path}/" "${local_newer_path}/" --only-show-errors \
                   && [[ -f "${local_newer_path}/.metadata" ]]; then
                    _ckpt_log "Synced newer checkpoint step-${s3_check} from S3"
                else
                    _ckpt_log "WARNING: failed to sync newer S3 checkpoint, using local step-${local_latest}"
                    rm -rf "${local_newer_path}"
                fi
                return 0
            fi
        fi
        _ckpt_log "Local checkpoint exists at ${local_folder}/step-${local_latest}, skipping S3 sync"
        return 0
    fi

    if [[ -z "$s3_folder" ]]; then
        _ckpt_log "No s3_upload_path configured, skipping S3 sync"
        return 0
    fi

    _ckpt_log "No local checkpoint in ${local_folder}, checking S3: ${s3_folder}"
    local s3_latest
    s3_latest="$(find_latest_s3_step "$s3_folder")"

    if (( s3_latest < 0 )); then
        _ckpt_log "No checkpoint found on S3 at ${s3_folder}"
        return 0
    fi

    local s3_step_path="${s3_folder%/}/step-${s3_latest}"
    local local_step_path="${local_folder}/step-${s3_latest}"

    _ckpt_log "Syncing checkpoint step-${s3_latest} from S3"
    _ckpt_log "  src:  ${s3_step_path}/"
    _ckpt_log "  dest: ${local_step_path}/"
    mkdir -p "${local_step_path}"
    if ! aws s3 sync "${s3_step_path}/" "${local_step_path}/" --only-show-errors; then
        _ckpt_log "WARNING: aws s3 sync failed, training will start fresh"
        rm -rf "${local_step_path}"
        return 0
    fi
    if [[ ! -f "${local_step_path}/.metadata" ]]; then
        _ckpt_log "WARNING: synced checkpoint missing .metadata, discarding"
        rm -rf "${local_step_path}"
        return 0
    fi
    _ckpt_log "S3 checkpoint sync complete (step-${s3_latest})"
    return 0
}

# Poll S3 until a checkpoint at or beyond the expected step appears.
# Used after multi-node training to confirm the S3 upload finished
# before other nodes attempt to sync.
wait_for_s3_checkpoint() {
    local s3_folder="$1" expected_step="$2" timeout="${3:-600}" poll_interval="${4:-15}"
    [[ -z "$s3_folder" ]] && { _ckpt_log "No S3 folder configured, cannot wait"; return 1; }

    local elapsed=0
    _ckpt_log "Waiting for step-${expected_step} on S3: ${s3_folder} (timeout=${timeout}s)"

    while (( elapsed < timeout )); do
        local s3_latest
        s3_latest="$(find_latest_s3_step "$s3_folder")"
        if (( s3_latest >= expected_step )); then
            _ckpt_log "Found step-${s3_latest} on S3 (>= expected step-${expected_step})"
            return 0
        fi
        if (( s3_latest >= 0 )); then
            _ckpt_log "S3 latest: step-${s3_latest}, waiting for step-${expected_step}... (${elapsed}/${timeout}s)"
        else
            _ckpt_log "No checkpoint on S3 yet, waiting... (${elapsed}/${timeout}s)"
        fi
        sleep "$poll_interval"
        elapsed=$((elapsed + poll_interval))
    done

    _ckpt_log "Timed out after ${timeout}s waiting for step-${expected_step} on S3"
    return 1
}
