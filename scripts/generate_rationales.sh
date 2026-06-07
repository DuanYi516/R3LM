#!/usr/bin/env bash
# Offline rationale generation for Stage-II (Generated-CoT data prep).
#
# Prerequisites:
#   pip install llamafactory vllm
#   Stage-I checkpoint
#   Register chatml_with_n (configs/TEMPLATE.md)
#
# Example:
#   bash scripts/generate_rationales.sh \
#     K562 \
#     /path/to/82k-no-reason-sharegpt.jsonl \
#     outputs/K562/self-generated-rationales.jsonl

set -euo pipefail

CELL_LINE="${1:-K562}"
INPUT_JSONL="${2:?RCC-only ShareGPT JSONL}"
OUTPUT_JSONL="${3:?output ShareGPT JSONL with generated rationales}"

EXTRA_ARGS=()
if [[ -n "${MODEL_PATH:-}" ]]; then
  EXTRA_ARGS+=(--model_path "${MODEL_PATH}")
fi

python -m r3lm.generate_rationales \
  --cell_line "${CELL_LINE}" \
  "${EXTRA_ARGS[@]}" \
  --input_jsonl "${INPUT_JSONL}" \
  --output_jsonl "${OUTPUT_JSONL}" \
  --template chatml_with_n \
  --temperature 0.95 \
  --top_p 0.7 \
  --top_k 50 \
  --max_new_tokens 1024 \
  --repetition_penalty 1.0 \
  --batch_size 128
