#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/run_math500_model.sh" hybrid-diffusion-9b-diffusion
"$SCRIPT_DIR/run_math500_model.sh" hybrid-diffusion-9b-self-spec
