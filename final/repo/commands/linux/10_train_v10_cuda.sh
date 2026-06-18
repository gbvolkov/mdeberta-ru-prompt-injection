#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="$ROOT/.venv/bin/python"; [[ -x "$PYTHON" ]] || PYTHON=python
"$PYTHON" scripts/train_mdeberta_ru_prompt_injection_option_b.py \
  --device cuda --bf16 \
  --student-model microsoft/mdeberta-v3-base \
  --teacher-model protectai/deberta-v3-base-prompt-injection-v2 \
  --teacher-device cuda \
  --prepared-dataset-dir datasets/prepared/training-dataset-v10-benign-coverage \
  --output-dir work/models/mdeberta-ru-prompt-injection-v10-benign-scratch \
  --learning-rate 1e-5 --epochs 3 \
  --distill-weight 0.02 --teacher-distill-mode benign_only \
  --last-n-layers 12 \
  --train-batch-size 16 --eval-batch-size 64 \
  --gradient-accumulation-steps 2 --teacher-batch-size 16 \
  --checkpoint-steps 500 --optim adamw_torch_fused \
  --group-by-length --tf32 --torch-num-threads 16 --preflight-steps 2
