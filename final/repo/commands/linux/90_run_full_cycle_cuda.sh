#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGES=(
  10_train_v10_cuda.sh
  20_train_v11_cuda.sh
  30_build_v12_dataset.sh
  31_train_v12_cuda.sh
  40_train_v13_cuda.sh
  51_train_v16_cuda.sh
  60_validate_v16.sh
)
for stage in "${STAGES[@]}"; do
  echo "[cycle] starting $stage"
  bash "$DIR/$stage"
  echo "[cycle] completed $stage"
done
