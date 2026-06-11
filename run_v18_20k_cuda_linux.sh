#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${MODEL_ID:-gbv/mdeberta-ru-prompt-injection}"
DEVICE="${DEVICE:-cuda}"
TOKENIZER_ID="${TOKENIZER_ID:-microsoft/mdeberta-v3-base}"

TARGET_TOTAL_ROWS="${TARGET_TOTAL_ROWS:-20000}"
VALIDATION_ROWS="${VALIDATION_ROWS:-2000}"
SOURCE_DOCUMENT_TARGET="${SOURCE_DOCUMENT_TARGET:-8000}"
MAX_SCAN_PER_SOURCE="${MAX_SCAN_PER_SOURCE:-200000}"
FP_TARGET_DOCUMENTS="${FP_TARGET_DOCUMENTS:-20000}"
ATTACK_BANK_ROWS="${ATTACK_BANK_ROWS:-40000}"
SEED="${SEED:-47}"

OUTPUT_ROOT="${OUTPUT_ROOT:-./v18-run-linux-cuda-20k}"
REVIEWED_MINED_BENIGN_JSONL="${REVIEWED_MINED_BENIGN_JSONL:-./v18-reviewed-near-boundary-benign-source.jsonl}"
HARD_FN_SOURCE_JSONL="${HARD_FN_SOURCE_JSONL:-./v18-hard-fn-reviewed-source.jsonl}"
REVIEWED_ATTACK_BANK_JSONL="${REVIEWED_ATTACK_BANK_JSONL:-./v18-reviewed-attack-bank-source.jsonl}"

FP_CANDIDATE_JSONL="$OUTPUT_ROOT/v18-fp-candidate-corpus.jsonl"
FP_CANDIDATE_REPORT="$OUTPUT_ROOT/v18-fp-candidate-corpus-report.json"
FP_REVIEW_JSONL="$OUTPUT_ROOT/v18-fp-review-threshold-0.82.jsonl"
FP_REVIEW_SUMMARY="$OUTPUT_ROOT/v18-fp-review-threshold-0.82-summary.json"
EXTERNAL_INPUTS_DIR="$OUTPUT_ROOT/v18-external-inputs"
MINED_BENIGN_JSONL="$EXTERNAL_INPUTS_DIR/v18-mined-benign-threshold-0.82.jsonl"
HARD_FN_JSONL="$EXTERNAL_INPUTS_DIR/v18-hard-fn-visible-attacks.jsonl"
ATTACK_BANK_JSONL="$EXTERNAL_INPUTS_DIR/v18-fresh-external-attack-bank.jsonl"
EXTERNAL_VALIDATION_REPORT="$EXTERNAL_INPUTS_DIR/v18-external-inputs-validation-report.json"
DATASET_OUTPUT_DIR="$OUTPUT_ROOT/training-dataset-v18-clean-20k-dry"
DATASET_VALIDATION_DIR="$OUTPUT_ROOT/training-dataset-v18-clean-20k-dry-validation"
DATASET_REPORT_JSON="$OUTPUT_ROOT/training-dataset-v18-clean-20k-dry-report.json"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 2
  fi
}

mkdir -p "$OUTPUT_ROOT" "$EXTERNAL_INPUTS_DIR"

require_file "$REVIEWED_MINED_BENIGN_JSONL"
require_file "$HARD_FN_SOURCE_JSONL"
require_file "$REVIEWED_ATTACK_BANK_JSONL"

"$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch. Check the pod image and torch CUDA wheel.")
print({"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)})
PY

"$PYTHON_BIN" build_false_positive_corpus.py \
  --output-jsonl "$FP_CANDIDATE_JSONL" \
  --report-json "$FP_CANDIDATE_REPORT" \
  --target-documents "$FP_TARGET_DOCUMENTS" \
  --tokenizer-id "$TOKENIZER_ID" \
  --source-pool external_mining_only \
  --allow-source-errors

"$PYTHON_BIN" run_false_positive_review.py \
  --model-id "$MODEL_ID" \
  --input-jsonl "$FP_CANDIDATE_JSONL" \
  --threshold 0.82 \
  --window-batch-size 128 \
  --output-jsonl "$FP_REVIEW_JSONL" \
  --summary-json "$FP_REVIEW_SUMMARY" \
  --device "$DEVICE"

"$PYTHON_BIN" prepare_v18_external_inputs.py \
  --output-dir "$EXTERNAL_INPUTS_DIR" \
  --target-total-rows "$TARGET_TOTAL_ROWS" \
  --mined-review-jsonl "$FP_REVIEW_JSONL" \
  --mined-review-jsonl "$REVIEWED_MINED_BENIGN_JSONL" \
  --hard-fn-source-jsonl "$HARD_FN_SOURCE_JSONL" \
  --reviewed-attack-bank-jsonl "$REVIEWED_ATTACK_BANK_JSONL" \
  --mined-score-threshold 0.82 \
  --attack-bank-rows "$ATTACK_BANK_ROWS" \
  --seed "$SEED" \
  --output-mined-benign-jsonl "$MINED_BENIGN_JSONL" \
  --output-hard-fn-jsonl "$HARD_FN_JSONL" \
  --output-attack-bank-jsonl "$ATTACK_BANK_JSONL"

"$PYTHON_BIN" validate_v18_external_inputs.py \
  --mined-benign-jsonl "$MINED_BENIGN_JSONL" \
  --hard-fn-jsonl "$HARD_FN_JSONL" \
  --attack-bank-jsonl "$ATTACK_BANK_JSONL" \
  --target-total-rows "$TARGET_TOTAL_ROWS" \
  --report-json "$EXTERNAL_VALIDATION_REPORT"

"$PYTHON_BIN" build_v18_clean_500k_dataset.py \
  --tokenizer-id "$TOKENIZER_ID" \
  --mined-benign-jsonl "$MINED_BENIGN_JSONL" \
  --hard-fn-jsonl "$HARD_FN_JSONL" \
  --attack-bank-jsonl "$ATTACK_BANK_JSONL" \
  --output-dir "$DATASET_OUTPUT_DIR" \
  --validation-output-dir "$DATASET_VALIDATION_DIR" \
  --report-json "$DATASET_REPORT_JSON" \
  --target-total-rows "$TARGET_TOTAL_ROWS" \
  --validation-rows "$VALIDATION_ROWS" \
  --source-document-target "$SOURCE_DOCUMENT_TARGET" \
  --max-scan-per-source "$MAX_SCAN_PER_SOURCE" \
  --seed "$SEED" \
  --dry-run \
  --allow-source-errors

echo "V18 CUDA 20K dry run finished."
echo "Dataset report: $DATASET_REPORT_JSON"
echo "External validation report: $EXTERNAL_VALIDATION_REPORT"
