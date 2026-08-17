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

usage() {
    cat <<'EOF'
Run paper-task evaluators against one or more already running HybridDiffusion servers.

Usage:
  scripts/evaluate.sh [task ...]

Tasks:
  gsm8k math500 ifeval humaneval mbpp arc_c gpqa
  aime2024 aime2025 mmlu mmlu_pro lcb

Examples:
  NUM_PROBLEMS=8 MAX_TOKENS=4096 scripts/evaluate.sh gsm8k
  PORTS="30000 30001" TASKS="gsm8k ifeval" scripts/evaluate.sh

Environment overrides:
  HYBRID_DIFFUSION_CACHE_ROOT, EVAL_VENV, PORTS, TASKS, MODEL_NAME, RESULT_ROOT, RUN_NAME,
  MAX_TOKENS, REQUEST_TIMEOUT, NUM_PROBLEMS, MAX_WORKERS,
  TEMPERATURE, TOP_P, TOP_K, PRESENCE_PENALTY, ENABLE_THINKING,
  SEED, AIME_SAMPLES, LCB_VERSION, HF_HOME, HF_DATASETS_CACHE.

This script performs evaluation only. Start the desired algorithm with
scripts/serve.sh first. For throughput-only fixed-output tests use the focused
benchmark clients under benchmark_clients; do not mix ignore_eos results with the
natural-stop quality output produced here.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

EVAL_VENV="${EVAL_VENV:-$CACHE_ROOT/venvs/hybrid-diffusion-eval}"
PYTHON="$EVAL_VENV/bin/python"
[[ -x "$PYTHON" ]] || {
    printf 'Environment not found: %s\nRun scripts/setup_eval_env.sh first.\n' "$EVAL_VENV" >&2
    exit 1
}
export PATH="$EVAL_VENV/bin:$PATH"

if [[ $# -gt 0 ]]; then
    TASKS="$*"
else
    TASKS="${TASKS:-gsm8k math500 ifeval humaneval mbpp arc_c gpqa aime2024 aime2025 mmlu mmlu_pro lcb}"
fi

PORTS="${PORTS:-30000}"
MODEL_NAME="${MODEL_NAME:-default}"
RUN_NAME="${RUN_NAME:-$(date -u '+%Y%m%dT%H%M%SZ')}"
RESULT_ROOT="${RESULT_ROOT:-$CACHE_ROOT/eval_results/$RUN_NAME}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1800}"
NUM_PROBLEMS="${NUM_PROBLEMS:-}"
MAX_WORKERS="${MAX_WORKERS:-}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}"
ENABLE_THINKING="${ENABLE_THINKING:-1}"
SEED="${SEED:-}"
AIME_SAMPLES="${AIME_SAMPLES:-1}"
LCB_VERSION="${LCB_VERSION:-6}"

export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$RESULT_ROOT" "$HF_HOME" "$HF_DATASETS_CACHE"

read -r -a port_args <<< "$PORTS"

is_false() {
    case "${1,,}" in
        0|false|no|off) return 0 ;;
        *) return 1 ;;
    esac
}

declare -a completed=()
for task in $TASKS; do
    script=""
    declare -a task_args=()
    supports_num_problems=1
    supports_max_workers=1
    case "$task" in
        gsm8k|math500|ifeval|humaneval|mbpp|arc_c|mmlu|mmlu_pro)
            script="$PROJECT_ROOT/benchmark_clients/quality/eval_${task}.py"
            if [[ "$task" == "gsm8k" && -n "$SEED" ]]; then
                task_args+=(--seed "$SEED")
            fi
            ;;
        gpqa)
            script="$PROJECT_ROOT/benchmark_clients/quality/eval_gpqa.py"
            task_args+=(--subset diamond)
            ;;
        aime2024)
            script="$PROJECT_ROOT/benchmark_clients/quality/eval_aime.py"
            task_args+=(--year 2024 --num-samples "$AIME_SAMPLES")
            supports_max_workers=0
            ;;
        aime2025)
            script="$PROJECT_ROOT/benchmark_clients/quality/eval_aime.py"
            task_args+=(--year 2025 --num-samples "$AIME_SAMPLES")
            supports_max_workers=0
            ;;
        lcb)
            script="$PROJECT_ROOT/benchmark_clients/quality/eval_lcb.py"
            task_args+=(--version "$LCB_VERSION")
            ;;
        *)
            printf 'Unknown task: %s\n' "$task" >&2
            exit 2
            ;;
    esac

    [[ -f "$script" ]] || {
        printf 'Evaluator not found: %s\n' "$script" >&2
        exit 1
    }

    task_dir="$RESULT_ROOT/$task"
    mkdir -p "$task_dir"
    declare -a common=(
        --model "$MODEL_NAME"
        --ports "${port_args[@]}"
        --max-tokens "$MAX_TOKENS"
        --timeout "$REQUEST_TIMEOUT"
        --temperature "$TEMPERATURE"
        --top-p "$TOP_P"
        --top-k "$TOP_K"
        --presence-penalty "$PRESENCE_PENALTY"
        --output-dir "$task_dir"
        --tag "${RUN_NAME}_${task}"
    )
    if is_false "$ENABLE_THINKING"; then
        common+=(--disable-thinking)
    fi
    if [[ -n "$NUM_PROBLEMS" && "$supports_num_problems" -eq 1 ]]; then
        common+=(--num-problems "$NUM_PROBLEMS")
    fi
    if [[ -n "$MAX_WORKERS" && "$supports_max_workers" -eq 1 ]]; then
        common+=(--max-workers "$MAX_WORKERS")
    fi

    command=("$PYTHON" "$script" "${common[@]}" "${task_args[@]}")
    printf '%q ' "${command[@]}" > "$task_dir/command.txt"
    printf '\n' >> "$task_dir/command.txt"
    printf '[evaluate] task=%s output=%s\n' "$task" "$task_dir"
    "${command[@]}" 2>&1 | tee "$task_dir/eval.log"

    summary_file="$(
        find "$task_dir" -maxdepth 1 -type f -name '*summary*.json' \
            -print -quit
    )"
    if [[ -z "$summary_file" ]]; then
        printf 'No summary JSON produced for task %s\n' "$task" >&2
        exit 1
    fi
    if ! "$PYTHON" - "$summary_file" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    summary = json.load(handle)
errors = int(summary.get("errors", 0) or 0)
if errors:
    raise SystemExit(
        f"Evaluator reported {errors} request error(s); inspect {path}"
    )
PY
    then
        exit 1
    fi
    completed+=("$task")
done

COMPLETED="${completed[*]}" \
TASKS_VALUE="$TASKS" \
PORTS_VALUE="$PORTS" \
MODEL_NAME_VALUE="$MODEL_NAME" \
MAX_TOKENS_VALUE="$MAX_TOKENS" \
TEMPERATURE_VALUE="$TEMPERATURE" \
TOP_P_VALUE="$TOP_P" \
TOP_K_VALUE="$TOP_K" \
PRESENCE_PENALTY_VALUE="$PRESENCE_PENALTY" \
ENABLE_THINKING_VALUE="$ENABLE_THINKING" \
SEED_VALUE="$SEED" \
RUN_NAME_VALUE="$RUN_NAME" \
RESULT_ROOT_VALUE="$RESULT_ROOT" \
"$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "run_name": os.environ["RUN_NAME_VALUE"],
    "model_name": os.environ["MODEL_NAME_VALUE"],
    "requested_tasks": os.environ["TASKS_VALUE"].split(),
    "completed_tasks": os.environ["COMPLETED"].split(),
    "ports": [int(value) for value in os.environ["PORTS_VALUE"].split()],
    "sampling": {
        "max_tokens": int(os.environ["MAX_TOKENS_VALUE"]),
        "temperature": float(os.environ["TEMPERATURE_VALUE"]),
        "top_p": float(os.environ["TOP_P_VALUE"]),
        "top_k": int(os.environ["TOP_K_VALUE"]),
        "presence_penalty": float(os.environ["PRESENCE_PENALTY_VALUE"]),
        "enable_thinking": os.environ["ENABLE_THINKING_VALUE"].lower()
        not in {"0", "false", "no", "off"},
        "base_seed": int(os.environ["SEED_VALUE"])
        if os.environ["SEED_VALUE"]
        else None,
    },
}
path = Path(os.environ["RESULT_ROOT_VALUE"]) / "run_manifest.json"
path.write_text(json.dumps(payload, indent=2) + "\n")
print(f"[evaluate] manifest={path}")
PY
