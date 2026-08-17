#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for script in \
    run_math500_sdar_1_7b.sh \
    run_math500_sdar_4b.sh \
    run_math500_sdar_8b.sh \
    run_math500_sdar_30b_a3b.sh \
    run_math500_llada2_0_mini.sh \
    run_math500_llada2_1_mini.sh \
    run_math500_llada2_0_flash.sh \
    run_math500_llada2_1_flash.sh \
    run_math500_hybrid_diffusion_2b.sh \
    run_math500_hybrid_diffusion_4b.sh \
    run_math500_hybrid_diffusion_9b.sh; do
    "$SCRIPT_DIR/$script"
done
