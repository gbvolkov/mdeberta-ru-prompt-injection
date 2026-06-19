#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="$ROOT/.venv/bin/python"; [[ -x "$PYTHON" ]] || PYTHON=python
DEVICE="${1:-cuda}"
"$PYTHON" scripts/run_core_validation.py \
  --v13-model work/models/mdeberta-ru-prompt-injection-v13-critical-correction-ft \
  --v16-model work/models/mdeberta-ru-prompt-injection-v16-critical-recall-restoration-ft \
  --output-dir work/validation-v13-v16-core \
  --device "$DEVICE" --window-batch-size 64
