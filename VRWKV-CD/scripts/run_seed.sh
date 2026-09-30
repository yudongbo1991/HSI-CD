#!/usr/bin/env bash
set -euo pipefail
GPU=${1:?Usage: scripts/run_seed.sh GPU DATASET SEED}
DATASET=${2:?Usage: scripts/run_seed.sh GPU DATASET SEED}
SEED=${3:?Usage: scripts/run_seed.sh GPU DATASET SEED}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${PYTHON:-python}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/outputs}
if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
  export CUDA_HOME
fi
export HSICD_DATA_ROOT=${HSICD_DATA_ROOT:-$ROOT/data}
cd "$ROOT"
exec "$PYTHON" -u train.py \
  --dataset "$DATASET" --seed "$SEED" --gpu "$GPU" \
  --epochs 100 --eval-every 10 --batch-size 64 --test-batch-size 128 \
  --strict-radius 6 --lr 0.0006 --weight-decay 0.00001 \
  --output "$OUTPUT_ROOT/$DATASET/seed_$SEED"
