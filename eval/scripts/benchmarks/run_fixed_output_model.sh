#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  scripts/benchmarks/run_fixed_output_model.sh MODEL

MODEL:
  sdar-1.7b | sdar-4b | sdar-8b | sdar-30b-a3b
  llada2.0-mini | llada2.1-mini | llada2.1-mini-speed
  hybrid-diffusion-2b-causal | hybrid-diffusion-4b-causal | hybrid-diffusion-9b-causal
  hybrid-diffusion-2b-causal-no-cuda-graph
  hybrid-diffusion-4b-causal-no-cuda-graph
  hybrid-diffusion-9b-causal-no-cuda-graph
  hybrid-diffusion-2b-diffusion | hybrid-diffusion-4b-diffusion | hybrid-diffusion-9b-diffusion
  hybrid-diffusion-2b-exact-truncated | hybrid-diffusion-4b-exact-truncated
  hybrid-diffusion-9b-exact-truncated
  hybrid-diffusion-2b-softmax-argmax | hybrid-diffusion-2b-truncated-argmax
  hybrid-diffusion-4b-softmax-argmax | hybrid-diffusion-4b-truncated-argmax
  hybrid-diffusion-9b-softmax-argmax | hybrid-diffusion-9b-truncated-argmax

The default protocol is the paper fixed-output protocol extended across output
lengths:

  tasks                 gsm8k, humaneval, gpqa, lcb_v6
  concurrency           1, 4, 8
  measured requests     15 * concurrency
  warmup requests       concurrency
  request schedule      one warmup wave, then 15 strict measured waves
  output lengths        2048, 16384
  sampling              temperature=1.0, top-p=0.95, top-k=50
  stopping              ignore_eos=true, no stop strings
  tensor parallelism    TP1

Run inside an allocated GPU node. HYBRID_DIFFUSION_CACHE_ROOT controls all Hugging Face,
JIT, log, and result storage. RUN_ROOT can select a shared sweep directory.
Existing protocol-valid result JSON files are skipped.

Useful overrides:
  TASKS="gsm8k humaneval gpqa"
  CONCURRENCIES="1 4 8"
  OUTPUT_LENGTHS="2048 16384"
  CUDA_VISIBLE_DEVICES=0 PORT=32000
  EVAL_VENV=/path/to/hybrid-diffusion-eval
  RANDOM_SEED=42
  NCCL_PORT=42000
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
model_path_var=""
default_model=""
dllm_config=""
mem_fraction="0.85"
disable_cuda_graph="false"

case "$model_key" in
    sdar-1.7b)
        mode="sdar"
        model_path_var="SDAR_1_7B_MODEL_PATH"
        default_model="JetLM/SDAR-1.7B-Chat"
        dllm_config="$PROJECT_ROOT/configs/sdar_b4_dynamic.yaml"
        mem_fraction="0.70"
        ;;
    sdar-4b)
        mode="sdar"
        model_path_var="SDAR_4B_MODEL_PATH"
        default_model="JetLM/SDAR-4B-Chat"
        dllm_config="$PROJECT_ROOT/configs/sdar_b4_dynamic.yaml"
        mem_fraction="0.78"
        ;;
    sdar-8b)
        mode="sdar"
        model_path_var="SDAR_8B_MODEL_PATH"
        default_model="JetLM/SDAR-8B-Chat"
        dllm_config="$PROJECT_ROOT/configs/sdar_b4_dynamic.yaml"
        mem_fraction="0.85"
        ;;
    sdar-30b-a3b)
        mode="sdar"
        model_path_var="SDAR_30B_A3B_MODEL_PATH"
        default_model="JetLM/SDAR-30B-A3B-Chat"
        dllm_config="$PROJECT_ROOT/configs/sdar_b4_dynamic.yaml"
        mem_fraction="0.92"
        ;;
    llada2.0-mini)
        mode="llada2-0"
        model_path_var="LLADA2_0_MINI_MODEL_PATH"
        default_model="inclusionAI/LLaDA2.0-mini"
        dllm_config="$PROJECT_ROOT/configs/llada2_0_b32.yaml"
        mem_fraction="0.85"
        ;;
    llada2.1-mini)
        mode="llada2-1-quality"
        model_path_var="LLADA2_1_MINI_MODEL_PATH"
        default_model="inclusionAI/LLaDA2.1-mini"
        dllm_config="$PROJECT_ROOT/configs/llada2_1_b32_quality.yaml"
        mem_fraction="0.85"
        ;;
    llada2.1-mini-speed)
        mode="llada2-1-speed"
        model_path_var="LLADA2_1_MINI_MODEL_PATH"
        default_model="inclusionAI/LLaDA2.1-mini"
        dllm_config="$PROJECT_ROOT/configs/llada2_1_b32_speed.yaml"
        mem_fraction="0.85"
        ;;
    hybrid-diffusion-2b-causal|hybrid-diffusion-2b-causal-no-cuda-graph)
        mode="causal"
        model_path_var="HYBRID_DIFFUSION_2B_MODEL_PATH"
        default_model="yuchen-zhu-zyc/HybridDiffusion-2B"
        mem_fraction="0.70"
        ;;
    hybrid-diffusion-4b-causal|hybrid-diffusion-4b-causal-no-cuda-graph)
        mode="causal"
        model_path_var="HYBRID_DIFFUSION_4B_MODEL_PATH"
        default_model="yuchen-zhu-zyc/HybridDiffusion-4B"
        mem_fraction="0.78"
        ;;
    hybrid-diffusion-9b-causal|hybrid-diffusion-9b-causal-no-cuda-graph)
        mode="causal"
        model_path_var="HYBRID_DIFFUSION_9B_MODEL_PATH"
        default_model="yuchen-zhu-zyc/HybridDiffusion-9B"
        mem_fraction="0.85"
        ;;
    hybrid-diffusion-2b-diffusion)
        mode="diffusion"
        model_path_var="HYBRID_DIFFUSION_2B_MODEL_PATH"
        default_model="yuchen-zhu-zyc/HybridDiffusion-2B"
        dllm_config="$PROJECT_ROOT/configs/hybrid_diffusion_shift_b3_g1.yaml"
        mem_fraction="0.70"
        ;;
    hybrid-diffusion-4b-diffusion)
        mode="diffusion"
        model_path_var="HYBRID_DIFFUSION_4B_MODEL_PATH"
        default_model="yuchen-zhu-zyc/HybridDiffusion-4B"
        dllm_config="$PROJECT_ROOT/configs/hybrid_diffusion_shift_b3_g1.yaml"
        mem_fraction="0.78"
        ;;
    hybrid-diffusion-9b-diffusion)
        mode="diffusion"
        model_path_var="HYBRID_DIFFUSION_9B_MODEL_PATH"
        default_model="yuchen-zhu-zyc/HybridDiffusion-9B"
        dllm_config="$PROJECT_ROOT/configs/hybrid_diffusion_shift_b3_g1.yaml"
        mem_fraction="0.85"
        ;;
    hybrid-diffusion-2b-exact-truncated|hybrid-diffusion-2b-softmax-argmax|hybrid-diffusion-2b-truncated-argmax)
        mode="self-spec"
        model_path_var="HYBRID_DIFFUSION_2B_MODEL_PATH"
        default_model="yuchen-zhu-zyc/HybridDiffusion-2B"
        mem_fraction="0.70"
        ;;
    hybrid-diffusion-4b-exact-truncated|hybrid-diffusion-4b-softmax-argmax|hybrid-diffusion-4b-truncated-argmax)
        mode="self-spec"
        model_path_var="HYBRID_DIFFUSION_4B_MODEL_PATH"
        default_model="yuchen-zhu-zyc/HybridDiffusion-4B"
        mem_fraction="0.78"
        ;;
    hybrid-diffusion-9b-exact-truncated|hybrid-diffusion-9b-softmax-argmax|hybrid-diffusion-9b-truncated-argmax)
        mode="self-spec"
        model_path_var="HYBRID_DIFFUSION_9B_MODEL_PATH"
        default_model="yuchen-zhu-zyc/HybridDiffusion-9B"
        mem_fraction="0.85"
        ;;
    *)
        printf 'Unknown model: %s\n' "$model_key" >&2
        usage >&2
        exit 2
        ;;
esac

if [[ "$model_key" == *-causal-no-cuda-graph ]]; then
    disable_cuda_graph="true"
fi
serve_extra_args=()
if [[ "$disable_cuda_graph" == "true" ]]; then
    serve_extra_args+=(--disable-cuda-graph)
fi

case "$model_key" in
    hybrid-diffusion-*-exact-truncated)
        dllm_config="$PROJECT_ROOT/configs/hybrid_diffusion_self_spec_b7_g4.yaml"
        ;;
    hybrid-diffusion-*-softmax-argmax)
        dllm_config="$PROJECT_ROOT/configs/hybrid_diffusion_self_spec_b7_g4_softmax_argmax.yaml"
        ;;
    hybrid-diffusion-*-truncated-argmax)
        dllm_config="$PROJECT_ROOT/configs/hybrid_diffusion_self_spec_b7_g4_truncated_argmax.yaml"
        ;;
esac

if [[ -n "${HYBRID_DIFFUSION_CACHE_ROOT:-}" ]]; then
    cache_root="$HYBRID_DIFFUSION_CACHE_ROOT"
elif git_root="$(git -C "$PROJECT_ROOT" rev-parse --show-toplevel 2>/dev/null)"; then
    cache_root="$git_root/cache"
else
    cache_root="$PROJECT_ROOT/.cache"
fi

export HF_HOME="${HF_HOME:-$cache_root/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$cache_root/python_cache/hybrid-diffusion-eval}"
mkdir -p "$PYTHONPYCACHEPREFIX"

eval_venv="${EVAL_VENV:-$cache_root/venvs/hybrid-diffusion-eval}"
python="$eval_venv/bin/python"
[[ -x "$python" ]] || {
    printf 'Evaluation environment not found: %s\n' "$eval_venv" >&2
    exit 1
}
[[ -z "$dllm_config" || -f "$dllm_config" ]] || {
    printf 'Algorithm config not found: %s\n' "$dllm_config" >&2
    exit 1
}

model_path="${!model_path_var:-$default_model}"
tasks="${TASKS:-gsm8k humaneval gpqa}"
concurrencies="${CONCURRENCIES:-1 4 8}"
output_lengths="${OUTPUT_LENGTHS:-2048 16384}"
measure_multiplier="${MEASURE_MULTIPLIER:-15}"
temperature="${TEMPERATURE:-1.0}"
top_p="${TOP_P:-0.95}"
top_k="${TOP_K:-50}"
presence_penalty="${PRESENCE_PENALTY:-1.5}"
port="${PORT:-32000}"
nccl_port="${NCCL_PORT:-$((port + 10000))}"
visible_devices="${CUDA_VISIBLE_DEVICES:-0}"
startup_timeout="${STARTUP_TIMEOUT:-1800}"
request_timeout="${REQUEST_TIMEOUT:-14400}"
random_seed="${RANDOM_SEED:-42}"
allocation_end_epoch="${ALLOCATION_END_EPOCH:-0}"
short_case_min_remaining="${SHORT_CASE_MIN_REMAINING_SECONDS:-1800}"
long_case_min_remaining="${LONG_CASE_MIN_REMAINING_SECONDS:-5400}"
pause_file="${PAUSE_FILE:-}"
worker_label="${WORKER_LABEL:-}"
if [[ -n "$worker_label" && ! "$worker_label" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    printf 'WORKER_LABEL contains unsupported characters: %s\n' \
        "$worker_label" >&2
    exit 2
fi
run_root="${RUN_ROOT:-$cache_root/eval_results/fixed_output/$(date -u '+%Y%m%dT%H%M%SZ')}"
model_root="$run_root/$model_key"
mkdir -p "$model_root"

server_pid=""
cleanup_server() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM -- "-$server_pid" 2>/dev/null || true
        for _ in $(seq 1 60); do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$server_pid" 2>/dev/null; then
            kill -KILL -- "-$server_pid" 2>/dev/null || true
        fi
        wait "$server_pid" 2>/dev/null || true
    fi
    server_pid=""
}
trap cleanup_server EXIT INT TERM

can_start_case() {
    local output_length="$1"
    local now remaining required
    if [[ -n "$pause_file" && -e "$pause_file" ]]; then
        return 1
    fi
    if (( allocation_end_epoch <= 0 )); then
        return 0
    fi
    now="$(date +%s)"
    remaining=$((allocation_end_epoch - now))
    if (( output_length >= 16384 )); then
        required="$long_case_min_remaining"
    else
        required="$short_case_min_remaining"
    fi
    (( remaining >= required ))
}

request_graceful_pause() {
    local output_length="$1"
    if [[ -n "$pause_file" ]]; then
        : >"$pause_file"
    fi
    printf '[fixed-output] pause before model=%s C=%s length=%s; allocation safety window reached\n' \
        "$model_key" "${current_concurrency:-unknown}" "$output_length"
}

result_is_valid() {
    local path="$1"
    local concurrency="$2"
    local output_length="$3"
    local task="$4"
    local require_stats="false"
    if [[ "$mode" == "self-spec" || "$mode" == "diffusion" ]]; then
        require_stats="true"
    fi
    [[ -f "$path" ]] || return 1
    jq -e \
        --argjson n "$((measure_multiplier * concurrency))" \
        --argjson warmup "$concurrency" \
        --argjson length "$output_length" \
        --argjson temperature "$temperature" \
        --argjson top_p "$top_p" \
        --argjson top_k "$top_k" \
        --argjson presence_penalty "$presence_penalty" \
        --arg algorithm_config "$dllm_config" \
        --arg task "$task" \
        --arg mode "$mode" \
        --argjson require_stats "$require_stats" \
        '.protocol_valid == true and
         .N == $n and .warmup == $warmup and
         .concurrency == $warmup and .max_new_tokens == $length and
         (($warmup == 1 and
           ((.request_schedule // {}).policy // "strict_waves") ==
             "strict_waves") or
          (((.request_schedule // {}).policy == "strict_waves" and
            (.request_schedule // {}).wave_size == $warmup and
            (.request_schedule // {}).warmup_waves == 1 and
            (.request_schedule // {}).measured_waves == ($n / $warmup)) or
           ((.request_schedule // {}).policy == "strict_waves_sharded" and
            (.request_schedule // {}).wave_size == $warmup and
            (.request_schedule // {}).warmup_waves_per_replica == 1 and
            (.request_schedule // {}).measured_waves == ($n / $warmup) and
            (.request_schedule // {}).replicas >= 2 and
            (((.request_schedule // {}).shard_measured_waves | add) ==
             ($n / $warmup))))) and
         .algorithm_config_path == $algorithm_config and
         .request_sampling_params.temperature == $temperature and
         .request_sampling_params.top_p == $top_p and
         .request_sampling_params.top_k == $top_k and
         .request_sampling_params.max_new_tokens == $length and
         .request_sampling_params.ignore_eos == true and
         .request_sampling_params.stop == null and
         ($task == "gsm8k" or
          .request_sampling_params.presence_penalty == $presence_penalty) and
         .completion_tokens.min == $length and
         .completion_tokens.max == $length and
         ($require_stats == false or
          (.dllm_tpf.decode_forwards > 0 and
           (.dllm_tpf.decode_tpf | type) == "number")) and
         ($mode != "self-spec" or
          (.dllm_tpf.verify_decisions > 0 and
           (.dllm_tpf.verify_tpf | type) == "number"))' \
        "$path" >/dev/null 2>&1
}

flush_server_cache() {
    local response_file="$1"
    if ! curl -fsS -X POST --max-time 120 \
        "http://127.0.0.1:$port/flush_cache" >"$response_file"; then
        printf 'Failed to flush server cache on port %s\n' "$port" >&2
        return 1
    fi
}

run_case() {
    local task="$1"
    local concurrency="$2"
    local output_length="$3"
    local limit=$((measure_multiplier * concurrency))
    local case_root="$model_root/c${concurrency}/results/$task/len${output_length}"
    local result_json="$case_root/result.json"
    local client_log="$case_root/client.log"
    local flush_log="$case_root/flush_cache.json"
    mkdir -p "$case_root"

    if result_is_valid "$result_json" "$concurrency" "$output_length" "$task"; then
        printf '[fixed-output] skip model=%s task=%s C=%s length=%s\n' \
            "$model_key" "$task" "$concurrency" "$output_length"
        return
    fi

    flush_server_cache "$flush_log"
    printf '[fixed-output] run model=%s task=%s C=%s length=%s N=%s warmup=%s\n' \
        "$model_key" "$task" "$concurrency" "$output_length" "$limit" "$concurrency"

    local -a client_cmd
    local -a stats_args=()
    if [[ "$mode" == "self-spec" ]]; then
        stats_args=(--require-dllm-stats --require-verify-stats)
    elif [[ "$mode" == "diffusion" ]]; then
        stats_args=(--require-dllm-stats)
    fi
    if [[ "$task" == "gsm8k" ]]; then
        client_cmd=(
            "$python" "$PROJECT_ROOT/benchmark_clients/bench_gsm8k_fixed_output.py"
            --model_path "$model_path"
            --limit "$limit"
            --warmup "$concurrency"
            --concurrency "$concurrency"
            --max_new_tokens "$output_length"
            --temperature "$temperature"
            --top-p "$top_p"
            --top-k "$top_k"
            --ports "$port"
            --timeout "$request_timeout"
            --progress-every "$concurrency"
            --algorithm-config "$dllm_config"
            "${stats_args[@]}"
            --save "$result_json"
        )
    else
        client_cmd=(
            "$python" "$PROJECT_ROOT/benchmark_clients/bench_tasks_fixed_output.py"
            --task "$task"
            --model_path "$model_path"
            --limit "$limit"
            --warmup "$concurrency"
            --concurrency "$concurrency"
            --max_new_tokens "$output_length"
            --temperature "$temperature"
            --top-p "$top_p"
            --top-k "$top_k"
            --presence-penalty "$presence_penalty"
            --ports "$port"
            --timeout "$request_timeout"
            --progress-every "$concurrency"
            --algorithm-config "$dllm_config"
            "${stats_args[@]}"
            --save "$result_json"
        )
    fi

    "${client_cmd[@]}" >"$client_log" 2>&1
    result_is_valid "$result_json" "$concurrency" "$output_length" "$task" || {
        printf 'Invalid result protocol: %s\n' "$result_json" >&2
        return 1
    }
    jq -r \
        '"[fixed-output] done TPS=\(.aggregate_tps) wall=\(.wall_seconds) " +
         "tokens=\(.total_completion_tokens) finish=\(.finish_reason_counts)"' \
        "$result_json"
}

for concurrency in $concurrencies; do
    current_concurrency="$concurrency"
    case "$concurrency" in
        1) graph_bs="1" ;;
        4) graph_bs="1 2 4" ;;
        8) graph_bs="1 2 4 8" ;;
        *)
            printf 'Unsupported concurrency: %s (expected 1, 4, or 8)\n' \
                "$concurrency" >&2
            exit 2
            ;;
    esac

    concurrency_root="$model_root/c${concurrency}"

    group_complete=true
    for output_length in $output_lengths; do
        for task in $tasks; do
            case_result="$concurrency_root/results/$task/len${output_length}/result.json"
            if ! result_is_valid \
                "$case_result" "$concurrency" "$output_length" "$task"; then
                group_complete=false
                break 2
            fi
        done
    done
    if [[ "$group_complete" == "true" ]]; then
        printf '[fixed-output] skip complete group model=%s C=%s\n' \
            "$model_key" "$concurrency"
        continue
    fi

    mkdir -p "$concurrency_root"
    if [[ -n "$worker_label" ]]; then
        server_log="$concurrency_root/server.${worker_label}.log"
        server_context="$concurrency_root/context.${worker_label}.txt"
    else
        server_log="$concurrency_root/server.log"
        server_context="$concurrency_root/context.txt"
    fi
    {
        printf 'model_key=%s\nmode=%s\nmodel_path=%s\n' \
            "$model_key" "$mode" "$model_path"
        printf 'tensor_parallel_size=1\nconcurrency=%s\n' "$concurrency"
        printf 'algorithm_config=%s\nvisible_devices=%s\nnccl_port=%s\n' \
            "$dllm_config" "$visible_devices" "$nccl_port"
        printf 'random_seed=%s\n' "$random_seed"
        printf 'cuda_graph_enabled=%s\n' \
            "$([[ "$disable_cuda_graph" == "true" ]] && printf false || printf true)"
        printf 'temperature=%s\ntop_p=%s\ntop_k=%s\n' \
            "$temperature" "$top_p" "$top_k"
        printf 'tasks=%s\noutput_lengths=%s\n' "$tasks" "$output_lengths"
        nvidia-smi --query-gpu=name,uuid,driver_version,memory.total \
            --format=csv,noheader 2>/dev/null || true
    } >"$server_context"

    cleanup_server
    printf '[fixed-output] launch model=%s mode=%s C=%s TP=1 port=%s gpu=%s\n' \
        "$model_key" "$mode" "$concurrency" "$port" "$visible_devices"

    CUDA_VISIBLE_DEVICES="$visible_devices" \
    HYBRID_DIFFUSION_CACHE_ROOT="$cache_root" \
    EVAL_VENV="$eval_venv" \
    PORT="$port" \
    TP_SIZE=1 \
    MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-$mem_fraction}" \
    MAX_RUNNING_REQUESTS="$concurrency" \
    CUDA_GRAPH_BS="$graph_bs" \
    DLLM_CONFIG="$dllm_config" \
        setsid "$PROJECT_ROOT/scripts/serve.sh" "$mode" "$model_path" -- \
        --random-seed "$random_seed" \
        --nccl-port "$nccl_port" \
        "${serve_extra_args[@]}" \
        >"$server_log" 2>&1 &
    server_pid=$!

    deadline=$((SECONDS + startup_timeout))
    until curl -fsS --max-time 30 "http://127.0.0.1:$port/health" >/dev/null; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            printf 'Server exited before ready: %s\n' "$server_log" >&2
            exit 1
        fi
        if (( SECONDS >= deadline )); then
            printf 'Server startup timed out after %ss: %s\n' \
                "$startup_timeout" "$server_log" >&2
            exit 1
        fi
        sleep 2
    done

    for output_length in $output_lengths; do
        case "$output_length" in
            2048|4096|8192|16384) ;;
            *)
                printf 'Unsupported output length: %s\n' "$output_length" >&2
                exit 2
                ;;
        esac
        for task in $tasks; do
            case "$task" in
                gsm8k|humaneval|gpqa|gpqa_diamond|lcb_v6) ;;
                *)
                    printf 'Unsupported task: %s\n' "$task" >&2
                    exit 2
                    ;;
            esac
            if ! can_start_case "$output_length"; then
                request_graceful_pause "$output_length"
                exit 75
            fi
            run_case "$task" "$concurrency" "$output_length"
        done
    done
    cleanup_server
done
