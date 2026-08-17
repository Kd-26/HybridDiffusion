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
Usage:
  scripts/serve.sh MODE MODEL_PATH [-- additional sglang arguments]

MODE:
  diffusion   HybridDiffusion shifted diffusion sampling (block 3 / generation 1)
  self-spec   HybridDiffusion self-speculative decoding (block 7 / generation 4)
  causal      Native causal autoregressive decoding
  sdar        SDAR dynamic low-confidence diffusion decoding
  llada2-0    LLaDA 2.0 low-confidence diffusion decoding
  llada2-1-speed
              LLaDA 2.1 editing decoder, speed configuration
  llada2-1-quality
              LLaDA 2.1 editing decoder, quality configuration

Examples:
  scripts/serve.sh diffusion yuchen-zhu-zyc/HybridDiffusion-2B
  PORT=31000 CUDA_GRAPH_BS="1 2 4" scripts/serve.sh self-spec /path/to/model
  FP8_GEMM_BACKEND=flashinfer_deepgemm scripts/serve.sh causal /path/to/fp8-model

Environment overrides:
  HYBRID_DIFFUSION_CACHE_ROOT, EVAL_VENV, HOST, PORT, TP_SIZE, DTYPE, ATTENTION_BACKEND,
  MEM_FRACTION_STATIC, MAX_RUNNING_REQUESTS, CUDA_GRAPH_BS,
  DLLM_CONFIG, FP8_GEMM_BACKEND, HF_HOME, FLASHINFER_WORKSPACE_BASE,
  TORCH_EXTENSIONS_DIR, TRITON_CACHE_DIR, TRTLLM_DG_CACHE_DIR, CUDA_HOME.

Run this command inside an allocated GPU node. The script does not submit,
clone, download a model, or assume a particular scheduler/platform.
EOF
}

[[ $# -ge 2 ]] || {
    usage >&2
    exit 2
}

MODE="$1"
MODEL_PATH="$2"
shift 2
if [[ "${1:-}" == "--" ]]; then
    shift
fi

EVAL_VENV="${EVAL_VENV:-$CACHE_ROOT/venvs/hybrid-diffusion-eval}"
[[ -x "$EVAL_VENV/bin/python" ]] || {
    printf 'Environment not found: %s\nRun scripts/setup_eval_env.sh first.\n' "$EVAL_VENV" >&2
    exit 1
}
export PATH="$EVAL_VENV/bin:$PATH"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-1 2 4 8}"
FP8_GEMM_BACKEND="${FP8_GEMM_BACKEND:-}"

export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$CACHE_ROOT/flashinfer}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$CACHE_ROOT/torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export TRTLLM_DG_CACHE_DIR="${TRTLLM_DG_CACHE_DIR:-$CACHE_ROOT/tensorrt_llm_deep_gemm}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$CACHE_ROOT/python_cache/hybrid-diffusion-eval}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"

# Triton links small helper objects against libcuda. Some driver installations
# expose only libcuda.so.1, while CUDA's stubs directory provides the
# unversioned linker name. Detect the CUDA installation instead of hard-coding
# a cluster path.
cuda_root="${CUDA_HOME:-}"
if [[ -z "$cuda_root" ]] && command -v nvcc >/dev/null 2>&1; then
    cuda_root="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
fi
if [[ -z "$cuda_root" && -d /usr/local/cuda ]]; then
    cuda_root="/usr/local/cuda"
fi
if [[ -n "$cuda_root" ]]; then
    export CUDA_HOME="${CUDA_HOME:-$cuda_root}"
    export PATH="$cuda_root/bin:$PATH"
    for stub_dir in \
        "$cuda_root/targets/x86_64-linux/lib/stubs" \
        "$cuda_root/lib64/stubs"; do
        if [[ -f "$stub_dir/libcuda.so" ]]; then
            export LIBRARY_PATH="$stub_dir${LIBRARY_PATH:+:$LIBRARY_PATH}"
            break
        fi
    done
fi

mkdir -p \
    "$HF_HOME" \
    "$FLASHINFER_WORKSPACE_BASE" \
    "$TORCH_EXTENSIONS_DIR" \
    "$TRITON_CACHE_DIR" \
    "$TRTLLM_DG_CACHE_DIR" \
    "$PYTHONPYCACHEPREFIX"

# The bundled FlashInfer source imports its JIT implementation from this
# checkout, while CUTLASS and spdlog build data are materialized by the
# one-shot installer in the external cache. Fail before allocating a GPU or
# starting a server if that runtime link is missing.
flashinfer_data="$PROJECT_ROOT/third_party/flashinfer/flashinfer/data"
for required_path in \
    "$flashinfer_data/cutlass/include/cutlass/arch/reg_reconfig.h" \
    "$flashinfer_data/cutlass/include/cutlass/arch/barrier.h" \
    "$flashinfer_data/cutlass/include/cute/tensor.hpp" \
    "$flashinfer_data/spdlog/include"; do
    [[ -e "$required_path" ]] || {
        printf 'FlashInfer build data is missing: %s\n' "$required_path" >&2
        printf 'Run scripts/setup_eval_env.sh with the same HYBRID_DIFFUSION_CACHE_ROOT before serving.\n' >&2
        exit 1
    }
done

declare -a mode_args=()
declare -a precision_args=()
if [[ -n "$FP8_GEMM_BACKEND" ]]; then
    precision_args=(--fp8-gemm-backend "$FP8_GEMM_BACKEND")
fi
case "$MODE" in
    diffusion)
        DLLM_CONFIG="${DLLM_CONFIG:-$PROJECT_ROOT/configs/hybrid_diffusion_shift_b3_g1.yaml}"
        mode_args=(
            --dllm-algorithm LowConfidenceShiftHybridDiffusion
            --dllm-algorithm-config "$DLLM_CONFIG"
            --skip-server-warmup
        )
        ;;
    self-spec)
        DLLM_CONFIG="${DLLM_CONFIG:-$PROJECT_ROOT/configs/hybrid_diffusion_self_spec_b7_g4.yaml}"
        mode_args=(
            --dllm-algorithm HybridDiffusionSelfSpec
            --dllm-algorithm-config "$DLLM_CONFIG"
        )
        ;;
    causal)
        ;;
    sdar)
        DLLM_CONFIG="${DLLM_CONFIG:-$PROJECT_ROOT/configs/sdar_b4_dynamic.yaml}"
        mode_args=(
            --dllm-algorithm LowConfidence
            --dllm-algorithm-config "$DLLM_CONFIG"
            --disable-radix-cache
        )
        ;;
    llada2-0)
        DLLM_CONFIG="${DLLM_CONFIG:-$PROJECT_ROOT/configs/llada2_0_b32.yaml}"
        mode_args=(
            --dllm-algorithm LowConfidence
            --dllm-algorithm-config "$DLLM_CONFIG"
            --disable-radix-cache
        )
        ;;
    llada2-1-speed)
        DLLM_CONFIG="${DLLM_CONFIG:-$PROJECT_ROOT/configs/llada2_1_b32_speed.yaml}"
        mode_args=(
            --dllm-algorithm JointThreshold
            --dllm-algorithm-config "$DLLM_CONFIG"
            --disable-radix-cache
        )
        ;;
    llada2-1-quality)
        DLLM_CONFIG="${DLLM_CONFIG:-$PROJECT_ROOT/configs/llada2_1_b32_quality.yaml}"
        mode_args=(
            --dllm-algorithm JointThreshold
            --dllm-algorithm-config "$DLLM_CONFIG"
            --disable-radix-cache
        )
        ;;
    *)
        printf 'Unknown mode: %s\n' "$MODE" >&2
        usage >&2
        exit 2
        ;;
esac

# The validated Qwen3.5 self-spec path requires radix cache plus the Mamba
# extra-buffer strategy. The historical no-buffer path violates the current
# tracking-list contract during the first decode round. Fail before server
# initialization instead of exposing a mode that is known to crash.
if [[ "$MODE" == "self-spec" ]]; then
    previous_arg=""
    for arg in "$@"; do
        if [[ "$arg" == "--disable-radix-cache" ]] || \
           [[ "$arg" == "--mamba-scheduler-strategy=no_buffer" ]] || \
           [[ "$previous_arg" == "--mamba-scheduler-strategy" && "$arg" == "no_buffer" ]]; then
            printf '%s\n' \
                'self-spec does not support radix-off/no_buffer in this release; use the default radix + extra_buffer path.' >&2
            exit 2
        fi
        previous_arg="$arg"
    done
fi

if [[ -n "${DLLM_CONFIG:-}" && ! -f "$DLLM_CONFIG" ]]; then
    printf 'Algorithm config not found: %s\n' "$DLLM_CONFIG" >&2
    exit 1
fi

read -r -a cuda_graph_bs <<< "$CUDA_GRAPH_BS"

exec "$EVAL_VENV/bin/python" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --trust-remote-code \
    --tp-size "$TP_SIZE" \
    --dtype "$DTYPE" \
    --attention-backend "$ATTENTION_BACKEND" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --cuda-graph-bs "${cuda_graph_bs[@]}" \
    --host "$HOST" \
    --port "$PORT" \
    "${precision_args[@]}" \
    "${mode_args[@]}" \
    "$@"
