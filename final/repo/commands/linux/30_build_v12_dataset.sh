#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="$ROOT/.venv/bin/python"; [[ -x "$PYTHON" ]] || PYTHON=python
"$PYTHON" scripts/build_v12_correction_dataset.py \
  --tokenizer-id work/models/mdeberta-ru-prompt-injection-v11-fp-correction-ft \
  --v11-dataset-dir datasets/prepared/training-dataset-v11-fp-correction-windowed-200k \
  --carrier-jsonl datasets/raw/false-positive-corpus-documents.jsonl \
  --output-dir work/datasets/training-dataset-v12-russian-critical-correction-windowed \
  --validation-output-dir work/datasets/training-dataset-v12-russian-critical-correction-windowed-validation \
  --report-json work/datasets/training-dataset-v12-russian-critical-correction-windowed-report.json \
  --allow-underfilled
