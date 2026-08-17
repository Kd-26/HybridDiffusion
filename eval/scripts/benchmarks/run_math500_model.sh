#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  scripts/benchmarks/run_math500_model.sh MODEL

MODEL:
  sdar-1.7b       SDAR_1_7B_MODEL_PATH
  sdar-4b         SDAR_4B_MODEL_PATH
  sdar-8b         SDAR_8B_MODEL_PATH
  sdar-30b-a3b    SDAR_30B_A3B_MODEL_PATH
  llada2.0-mini    LLADA2_0_MINI_MODEL_PATH
  llada2.1-mini    LLADA2_1_MINI_MODEL_PATH
  llada2.0-flash   LLADA2_0_FLASH_MODEL_PATH
  llada2.1-flash   LLADA2_1_FLASH_MODEL_PATH
  hybrid-diffusion-2b-diffusion | hybrid-diffusion-2b-self-spec    HYBRID_DIFFUSION_2B_MODEL_PATH
  hybrid-diffusion-4b-diffusion | hybrid-diffusion-4b-self-spec    HYBRID_DIFFUSION_4B_MODEL_PATH
  hybrid-diffusion-9b-diffusion | hybrid-diffusion-9b-self-spec    HYBRID_DIFFUSION_9B_MODEL_PATH

Run this script inside an allocated GPU node. Model paths default to their
Hugging Face repository IDs and can be overridden with the corresponding
*_MODEL_PATH environment variable. HYBRID_DIFFUSION_CACHE_ROOT controls all downloads,
server logs, JIT artifacts, and evaluation results.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

[[ $# -eq 1 ]] || {
    usage >&2
    exit 2
}

model_key="$1"
mode=""
default_model=""
model_path_var=""
default_tp=1
default_visible_devices="0"
default_mem_fraction="0.80"
default_max_running=4
default_graph_bs="1 2 4"
default_workers=4
max_tokens=16384
temperature="0.1"
top_p="0.95"
top_k=20
presence_penalty="0"

case "$model_key" in
    sdar-1.7b)
        mode="sdar"
        default_model="JetLM/SDAR-1.7B-Chat"
        model_path_var="SDAR_1_7B_MODEL_PATH"
        default_mem_fraction="0.50"
        default_max_running=8
        default_graph_bs="1 2 4 8"
        default_workers=8
        temperature="0.1"
        top_p="1.0"
        top_k=-1
        ;;
    sdar-4b)
        mode="sdar"
        default_model="JetLM/SDAR-4B-Chat"
        model_path_var="SDAR_4B_MODEL_PATH"
        default_mem_fraction="0.60"
        temperature="0.1"
        top_p="1.0"
        top_k=-1
        ;;
    sdar-8b)
        mode="sdar"
        default_model="JetLM/SDAR-8B-Chat"
        model_path_var="SDAR_8B_MODEL_PATH"
        default_mem_fraction="0.70"
        temperature="0.1"
        top_p="1.0"
        top_k=-1
        ;;
    sdar-30b-a3b)
        mode="sdar"
        default_model="JetLM/SDAR-30B-A3B-Chat"
        model_path_var="SDAR_30B_A3B_MODEL_PATH"
        default_mem_fraction="0.92"
        default_max_running=2
        default_graph_bs="1 2"
        default_workers=2
        temperature="0.1"
        top_p="1.0"
        top_k=-1
        ;;
    llada2.0-mini)
        mode="llada2-0"
        default_model="inclusionAI/LLaDA2.0-mini"
        model_path_var="LLADA2_0_MINI_MODEL_PATH"
        top_p="1.0"
        top_k=-1
        ;;
    llada2.1-mini)
        mode="${LLADA2_1_MODE:-llada2-1-quality}"
        default_model="inclusionAI/LLaDA2.1-mini"
        model_path_var="LLADA2_1_MINI_MODEL_PATH"
        top_p="1.0"
        top_k=-1
        ;;
    llada2.0-flash)
        mode="llada2-0"
        default_model="inclusionAI/LLaDA2.0-flash"
        model_path_var="LLADA2_0_FLASH_MODEL_PATH"
        default_tp=4
        default_visible_devices="0,1,2,3"
        default_mem_fraction="0.85"
        top_p="1.0"
        top_k=-1
        ;;
    llada2.1-flash)
        mode="${LLADA2_1_MODE:-llada2-1-quality}"
        default_model="inclusionAI/LLaDA2.1-flash"
        model_path_var="LLADA2_1_FLASH_MODEL_PATH"
        default_tp=4
        default_visible_devices="0,1,2,3"
        default_max_running=1
        default_graph_bs="1"
        default_workers=1
        top_p="1.0"
        top_k=-1
        ;;
    hybrid-diffusion-2b-diffusion|hybrid-diffusion-2b-self-spec)
        default_model="yuchen-zhu-zyc/HybridDiffusion-2B"
        model_path_var="HYBRID_DIFFUSION_2B_MODEL_PATH"
        mode="${model_key#hybrid-diffusion-2b-}"
        default_mem_fraction="0.55"
        max_tokens=32768
        temperature="1.0"
        presence_penalty="1.5"
        ;;
    hybrid-diffusion-4b-diffusion|hybrid-diffusion-4b-self-spec)
        default_model="yuchen-zhu-zyc/HybridDiffusion-4B"
        model_path_var="HYBRID_DIFFUSION_4B_MODEL_PATH"
        mode="${model_key#hybrid-diffusion-4b-}"
        default_mem_fraction="0.60"
        max_tokens=32768
        temperature="1.0"
        presence_penalty="1.5"
        ;;
    hybrid-diffusion-9b-diffusion|hybrid-diffusion-9b-self-spec)
        default_model="yuchen-zhu-zyc/HybridDiffusion-9B"
        model_path_var="HYBRID_DIFFUSION_9B_MODEL_PATH"
        mode="${model_key#hybrid-diffusion-9b-}"
        default_mem_fraction="0.70"
        max_tokens=32768
        temperature="1.0"
        presence_penalty="1.5"
        ;;
    *)
        printf 'Unknown model: %s\n' "$model_key" >&2
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

model_path="${!model_path_var:-$default_model}"
port="${PORT:-30000}"
tp_size="${TP_SIZE:-$default_tp}"
mem_fraction="${MEM_FRACTION_STATIC:-$default_mem_fraction}"
max_running="${MAX_RUNNING_REQUESTS:-$default_max_running}"
graph_bs="${CUDA_GRAPH_BS:-$default_graph_bs}"
max_workers="${MAX_WORKERS:-$default_workers}"
visible_devices="${CUDA_VISIBLE_DEVICES:-$default_visible_devices}"
run_name="${RUN_NAME:-$(date -u '+%Y%m%dT%H%M%SZ')-$model_key}"
run_root="${RUN_ROOT:-$cache_root/eval_results/math500/$model_key/$run_name}"
server_log="$run_root/server.log"
result_root="$run_root/results"

mkdir -p "$run_root" "$result_root"

server_pid=""
cleanup() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM -- "-$server_pid" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$server_pid" 2>/dev/null; then
            kill -KILL -- "-$server_pid" 2>/dev/null || true
        fi
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

printf '[math500] model=%s mode=%s checkpoint=%s tp=%s port=%s\n' \
    "$model_key" "$mode" "$model_path" "$tp_size" "$port"
printf '[math500] output=%s\n' "$run_root"

CUDA_VISIBLE_DEVICES="$visible_devices" \
PORT="$port" \
TP_SIZE="$tp_size" \
MEM_FRACTION_STATIC="$mem_fraction" \
MAX_RUNNING_REQUESTS="$max_running" \
CUDA_GRAPH_BS="$graph_bs" \
    setsid "$PROJECT_ROOT/scripts/serve.sh" "$mode" "$model_path" \
    >"$server_log" 2>&1 &
server_pid=$!

startup_timeout="${STARTUP_TIMEOUT:-900}"
deadline=$((SECONDS + startup_timeout))
until curl -fsS --max-time 60 "http://127.0.0.1:$port/health" >/dev/null; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        printf 'Server exited before becoming ready; inspect %s\n' "$server_log" >&2
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        printf 'Server did not become ready within %ss; inspect %s\n' \
            "$startup_timeout" "$server_log" >&2
        exit 1
    fi
    sleep 2
done

PORTS="$port" \
RESULT_ROOT="$result_root" \
RUN_NAME="$run_name" \
MAX_TOKENS="${MAX_TOKENS:-$max_tokens}" \
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1800}" \
MAX_WORKERS="$max_workers" \
TEMPERATURE="${TEMPERATURE:-$temperature}" \
TOP_P="${TOP_P:-$top_p}" \
TOP_K="${TOP_K:-$top_k}" \
PRESENCE_PENALTY="${PRESENCE_PENALTY:-$presence_penalty}" \
ENABLE_THINKING="${ENABLE_THINKING:-1}" \
    "$PROJECT_ROOT/scripts/evaluate.sh" math500
