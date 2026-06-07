#!/usr/bin/env bash
# Build RCC prompts from a sequence table (CSV/JSONL).
#
# Example:
#   bash scripts/build_rcc.sh \
#     sequences.csv \
#     outputs/K562/rcc-sharegpt.jsonl \
#     K562 \
#     /path/to/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt

set -euo pipefail

INPUT_PATH="${1:?input table (csv/jsonl)}"
OUTPUT_PATH="${2:?output jsonl}"
CELL_LINE="${3:-K562}"
MEME_FILE="${4:?path to JASPAR MEME file}"

python -m r3lm.build_rcc \
  --input "${INPUT_PATH}" \
  --output "${OUTPUT_PATH}" \
  --cell_line "${CELL_LINE}" \
  --meme_file "${MEME_FILE}" \
  --format sharegpt
