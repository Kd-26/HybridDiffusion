#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  scripts/benchmarks/run_fixed_output_all.sh [run|summary]

`run` builds a resumable queue of MODEL x C work items and assigns them to the
GPU slots listed in GPU_SLOTS. Every work item is TP1. The default matrix runs
four SDAR checkpoints, LLaDA 2.0/2.1 mini, and HybridDiffusion-2B/4B/9B with causal AR,
and three AR-Trust draft policies. Each case uses one warmup wave followed by
15 measured strict waves of C requests; a new wave starts only after every
request in the previous wave has completed.

Environment:
  HYBRID_DIFFUSION_CACHE_ROOT   persistent cache outside the source tree
  EVAL_VENV          one-shot evaluation environment
  RUN_ID             stable name used to resume a sweep
  RUN_ROOT           explicit result directory
  GPU_SLOTS          space-separated physical GPU ids; default "0"
  PORT_BASE          one local server port per worker; default 32000
  MODEL_KEYS         override the model matrix
  CONCURRENCIES      default "1 4 8"
  TASKS              default "gsm8k humaneval gpqa"
  OUTPUT_LENGTHS     default "2048 16384"
  RANDOM_SEED       SGLang server seed; default 42
  ALLOCATION_END_EPOCH  stop admitting cases before this Unix timestamp
  SHORT_CASE_MIN_REMAINING_SECONDS  default 1800
  LONG_CASE_MIN_REMAINING_SECONDS   default 5400

Example on an allocated 8-GPU node:
  HYBRID_DIFFUSION_CACHE_ROOT=/persistent/cache EVAL_VENV=/path/to/hybrid-diffusion-eval \
  RUN_ID=h100-fixed-output GPU_SLOTS="0 1 2 3 4 5 6 7" \
    scripts/benchmarks/run_fixed_output_all.sh run
EOF
}

case "${1:-run}" in
    -h|--help|help)
        usage
        exit 0
        ;;
    run|summary) action="${1:-run}" ;;
    *)
        usage >&2
        exit 2
        ;;
esac

if [[ -n "${HYBRID_DIFFUSION_CACHE_ROOT:-}" ]]; then
    cache_root="$HYBRID_DIFFUSION_CACHE_ROOT"
elif git_root="$(git -C "$PROJECT_ROOT" rev-parse --show-toplevel 2>/dev/null)"; then
    cache_root="$git_root/cache"
else
    cache_root="$PROJECT_ROOT/.cache"
fi

run_id="${RUN_ID:-$(date -u '+%Y%m%dT%H%M%SZ')-h100-fixed-output}"
run_root="${RUN_ROOT:-$cache_root/eval_results/fixed_output/$run_id}"
summary_script="$PROJECT_ROOT/scripts/benchmarks/summarize_fixed_output.py"
measure_multiplier="${MEASURE_MULTIPLIER:-15}"
model_keys="${MODEL_KEYS:-\
hybrid-diffusion-2b-causal hybrid-diffusion-2b-exact-truncated hybrid-diffusion-2b-softmax-argmax \
hybrid-diffusion-2b-truncated-argmax \
hybrid-diffusion-4b-causal hybrid-diffusion-4b-exact-truncated hybrid-diffusion-4b-softmax-argmax \
hybrid-diffusion-4b-truncated-argmax \
hybrid-diffusion-9b-causal hybrid-diffusion-9b-exact-truncated hybrid-diffusion-9b-softmax-argmax \
hybrid-diffusion-9b-truncated-argmax \
sdar-1.7b sdar-4b sdar-8b sdar-30b-a3b \
llada2.0-mini llada2.1-mini}"
tasks="${TASKS:-gsm8k humaneval gpqa}"
concurrencies="${CONCURRENCIES:-1 4 8}"
output_lengths="${OUTPUT_LENGTHS:-2048 16384}"
mkdir -p "$run_root"

if [[ "$action" == "summary" ]]; then
    exec "${EVAL_VENV:-$cache_root/venvs/hybrid-diffusion-eval}/bin/python" \
        "$summary_script" "$run_root" \
        --measure-multiplier "$measure_multiplier" \
        --models "$model_keys" \
        --tasks "$tasks" \
        --concurrencies "$concurrencies" \
        --lengths "$output_lengths"
fi

gpu_slots="${GPU_SLOTS:-0}"
port_base="${PORT_BASE:-32000}"

queue_file="$run_root/work_queue.tsv"
next_file="$run_root/work_queue.next"
lock_file="$run_root/work_queue.lock"
failure_file="$run_root/work_queue.failures.tsv"
worker_root="$run_root/workers"
pause_file="${PAUSE_FILE:-$run_root/graceful_pause.${SLURM_JOB_ID:-manual}}"
mkdir -p "$worker_root"

# Run C=1 first so HybridDiffusion TPF is available before the higher-concurrency sweep.
: >"$queue_file"
for concurrency in 1 4 8; do
    case " $concurrencies " in
        *" $concurrency "*) ;;
        *) continue ;;
    esac
    for model_key in $model_keys; do
        printf '%s\t%s\n' "$model_key" "$concurrency" >>"$queue_file"
    done
done
printf '0\n' >"$next_file"
: >"$failure_file"

claim_next() {
    local index line
    [[ ! -e "$pause_file" ]] || return 0
    {
        flock -x 9
        read -r index <"$next_file"
        line="$(sed -n "$((index + 1))p" "$queue_file")"
        if [[ -n "$line" ]]; then
            printf '%s\n' "$((index + 1))" >"$next_file"
            printf '%s\n' "$line"
        fi
    } 9>"$lock_file"
}

record_failure() {
    local gpu="$1"
    local model="$2"
    local concurrency="$3"
    local status="$4"
    {
        flock -x 9
        printf '%s\t%s\t%s\t%s\n' \
            "$gpu" "$model" "$concurrency" "$status" >>"$failure_file"
    } 9>"$lock_file"
}

worker() {
    local gpu="$1"
    local worker_index="$2"
    local port=$((port_base + worker_index))
    local item model concurrency status log
    while item="$(claim_next)" && [[ -n "$item" ]]; do
        IFS=$'\t' read -r model concurrency <<<"$item"
        log="$worker_root/gpu${gpu}_${model}_c${concurrency}.log"
        printf '[worker gpu=%s] model=%s C=%s port=%s\n' \
            "$gpu" "$model" "$concurrency" "$port" | tee -a "$log"
        set +e
        CUDA_VISIBLE_DEVICES="$gpu" \
        PORT="$port" \
        RUN_ROOT="$run_root" \
        CONCURRENCIES="$concurrency" \
        TASKS="$tasks" \
        OUTPUT_LENGTHS="$output_lengths" \
        MEASURE_MULTIPLIER="$measure_multiplier" \
        HYBRID_DIFFUSION_CACHE_ROOT="$cache_root" \
        EVAL_VENV="${EVAL_VENV:-$cache_root/venvs/hybrid-diffusion-eval}" \
        ALLOCATION_END_EPOCH="${ALLOCATION_END_EPOCH:-0}" \
        SHORT_CASE_MIN_REMAINING_SECONDS="${SHORT_CASE_MIN_REMAINING_SECONDS:-1800}" \
        LONG_CASE_MIN_REMAINING_SECONDS="${LONG_CASE_MIN_REMAINING_SECONDS:-5400}" \
        PAUSE_FILE="$pause_file" \
            "$SCRIPT_DIR/run_fixed_output_model.sh" "$model" \
            >>"$log" 2>&1
        status=$?
        set -e
        if (( status == 75 )); then
            printf '[worker gpu=%s] paused model=%s C=%s at a safe case boundary\n' \
                "$gpu" "$model" "$concurrency" | tee -a "$log"
            : >"$pause_file"
            break
        elif (( status != 0 )); then
            printf '[worker gpu=%s] FAILED model=%s C=%s status=%s log=%s\n' \
                "$gpu" "$model" "$concurrency" "$status" "$log" >&2
            record_failure "$gpu" "$model" "$concurrency" "$status"
        else
            printf '[worker gpu=%s] complete model=%s C=%s\n' \
                "$gpu" "$model" "$concurrency" | tee -a "$log"
        fi
    done
}

worker_pids=()
worker_index=0
for gpu in $gpu_slots; do
    worker "$gpu" "$worker_index" &
    worker_pids+=("$!")
    worker_index=$((worker_index + 1))
done

status=0
for pid in "${worker_pids[@]}"; do
    wait "$pid" || status=1
done

"${EVAL_VENV:-$cache_root/venvs/hybrid-diffusion-eval}/bin/python" \
    "$summary_script" "$run_root" \
    --measure-multiplier "$measure_multiplier" \
    --models "$model_keys" \
    --tasks "$tasks" \
    --concurrencies "$concurrencies" \
    --lengths "$output_lengths" \
    --allow-partial

if [[ -s "$failure_file" ]]; then
    printf 'One or more work items failed; see %s\n' "$failure_file" >&2
    exit 1
fi
if [[ -e "$pause_file" ]]; then
    printf 'Sweep paused safely; resume with the same RUN_ROOT in a new allocation.\n'
    exit 75
fi
exit "$status"
