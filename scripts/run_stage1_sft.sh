#!/usr/bin/env bash
# Stage-I: full-parameter SFT for mechanistic rationale generation.
#
# Prerequisites:
#   pip install llamafactory
#   Register chatml_with_n template (see configs/TEMPLATE.md)
#
# Usage:
#   bash scripts/run_stage1_sft.sh K562
#   bash scripts/run_stage1_sft.sh HepG2
#   bash scripts/run_stage1_sft.sh SKNSH

set -euo pipefail

CELL_LINE="${1:-K562}"
LOG_DIR="${LOG_DIR:-./logs/stage1}"
mkdir -p "$LOG_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export ALLOW_EXTRA_ARGS="${ALLOW_EXTRA_ARGS:-0}"

DATASET="${CELL_LINE}-1000_reason-sharegpt-short-conclu"
OUTPUT_DIR="saves/stage1/${CELL_LINE}"
LOG_FILE="${LOG_DIR}/${CELL_LINE}-$(date +%Y%m%d-%H%M%S).log"

DATA_DIR="${LLAMAFACTORY_DATA_DIR:-./data}"
python -c "from r3lm.hf_assets import prepare_llamafactory_data_dir; prepare_llamafactory_data_dir('${DATA_DIR}')"
export LLAMAFACTORY_DATA_DIR="${DATA_DIR}"

echo "Training Stage-I for ${CELL_LINE} (dataset=${DATASET})"
echo "Logs: ${LOG_FILE}"

FORCE_TORCHRUN=1 llamafactory-cli train configs/stage1_sft.yaml \
  --dataset "${DATASET}" \
  --output_dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_FILE}"
