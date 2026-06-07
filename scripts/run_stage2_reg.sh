#!/usr/bin/env bash
# Stage-II: reason-conditioned regression training.
#
# Usage:
#   bash scripts/run_stage2_reg.sh
#   bash scripts/run_stage2_reg.sh --cell_line K562 --input_mode oracle_cot
#   bash scripts/run_stage2_reg.sh --input_mode generated_cot --max_epochs 10

set -euo pipefail

LOG_DIR="${LOG_DIR:-./logs/stage2}"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/stage2-$(date +%Y%m%d-%H%M%S).log"

python -m r3lm.train_stage2 "$@" 2>&1 | tee "${LOG_FILE}"
