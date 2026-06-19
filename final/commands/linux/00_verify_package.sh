#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="$ROOT/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON=python
"$PYTHON" scripts/verify_v16_package.py --root "$ROOT"
if [[ -f manifests/SHA256SUMS.txt ]]; then
  "$PYTHON" scripts/verify_sha256_manifest.py --root "$ROOT"
fi
