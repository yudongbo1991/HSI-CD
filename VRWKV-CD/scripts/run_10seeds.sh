#!/usr/bin/env bash
set -euo pipefail
GPU=${1:?Usage: scripts/run_10seeds.sh GPU DATASET [PROTOCOL]}
DATASET=${2:?Usage: scripts/run_10seeds.sh GPU DATASET [PROTOCOL]}
PROTOCOL=${3:-val50_of_1pct}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
for SEED in $(seq 0 9); do
  "$ROOT/scripts/run_seed.sh" "$GPU" "$DATASET" "$SEED" "$PROTOCOL"
done
