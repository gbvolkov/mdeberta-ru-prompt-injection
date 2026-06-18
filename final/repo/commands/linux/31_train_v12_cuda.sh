#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="$ROOT/.venv/bin/python"; [[ -x "$PYTHON" ]] || PYTHON=python
"$PYTHON" scripts/train_mdeberta_ru_prompt_injection_option_b.py \
  --device cuda --bf16 \
  --student-model work/models/mdeberta-ru-prompt-injection-v11-fp-correction-ft \
  --prepared-dataset-dir work/datasets/training-dataset-v12-russian-critical-correction-windowed \
  --output-dir work/models/mdeberta-ru-prompt-injection-v12-critical-correction-ft \
  --learning-rate 2e-6 --epochs 1 --distill-weight 0.0 --skip-teacher \
  --last-n-layers 4 --train-batch-size 32 --eval-batch-size 128 \
  --gradient-accumulation-steps 1 --checkpoint-steps 250 --save-total-limit 8 \
  --optim adamw_torch_fused --group-by-length --tf32 \
  --torch-num-threads 6 --preflight-steps 2
