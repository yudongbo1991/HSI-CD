#!/usr/bin/env bash
set -euo pipefail

GPU=${1:?Usage: scripts/run_seed.sh GPU DATASET SEED [PROTOCOL]}
DATASET=${2:?Usage: scripts/run_seed.sh GPU DATASET SEED [PROTOCOL]}
SEED=${3:?Usage: scripts/run_seed.sh GPU DATASET SEED [PROTOCOL]}
PROTOCOL=${4:-val50_of_1pct}

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${PYTHON:-python}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/outputs}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CUDA_HOME
export VRWKV_LOG_ROOT="$OUTPUT_ROOT/logs"
export VRWKV_RESULT_ROOT="$OUTPUT_ROOT/results"
export VRWKV_SPLIT_ROOT="$OUTPUT_ROOT/splits"
export VRWKV_SELECTION_SPLIT_ROOT="$OUTPUT_ROOT/selection_splits"

case "$DATASET" in
  river) PATCH=7; ALPHA=.5; GROUPS=8; UNSHARED=0 ;;
  yancheng) PATCH=7; ALPHA=.6; GROUPS=32; UNSHARED=1 ;;
  bayarea) PATCH=7; ALPHA=.4; GROUPS=8; UNSHARED=1 ;;
  hermiston390) PATCH=5; ALPHA=.5; GROUPS=32; UNSHARED=1 ;;
  *) echo "Unsupported dataset: $DATASET" >&2; exit 2 ;;
esac
case "$PROTOCOL" in
  val50_of_1pct|test_peak_1pct) ;;
  *) echo "Unsupported protocol: $PROTOCOL" >&2; exit 2 ;;
esac

EXTRA=()
if [[ "$UNSHARED" == 1 ]]; then EXTRA+=(--spatial-unshared-input-projection); fi

cd "$ROOT"
exec "$PYTHON" -u train.py \
  --variant compact_signed_prototype_e2e_reliability \
  --reliability-mode embedded_logistic_learned_adaptive \
  --dataset "$DATASET" --train-fraction .01 --input-normalization train_patches \
  --selection-protocol "$PROTOCOL" --seed "$SEED" --gpu "$GPU" \
  --epochs 100 --test-freq 10 --test-at-round-multiples \
  --patch "$PATCH" --local-patch "$PATCH" --test-exclusion-radius 6 \
  --batch-size 64 --test-batch-size 128 --lr .0006 --lr-min-ratio .01 \
  --lr-scheduler cosine --weight-decay 1e-5 --label-smoothing 0 \
  --spatial-input-dim 128 --spatial-feature-dim 64 \
  --change-residual-mode gated --change-residual-init -1 \
  --spatial-channel-attention none --spatial-classifier-mode flatten \
  --horizontal-vertical-forward-spatial --vrwkv-depth 3 \
  --rwkv-decay-min .15 --rwkv-decay-max 4 --rwkv-lr-multiplier 1 \
  --disable-sobel-modulation --disable-mamba-module \
  --spectral-groups 16 --mamba-state 16 --mamba-depth 1 \
  --classifier-dropout .05 --pixel-spectral-groups "$GROUPS" \
  --pixel-spectral-width 16 --pixel-spectral-dim 64 --pixel-spectral-kernel 3 \
  --pixel-spectral-design symmetric --pixel-spectral-adaptive-multiscale \
  --pixel-spectral-linear-length 32 --disable-pixel-global-component \
  --pixel-fusion-alpha "$ALPHA" --pixel-fused-loss-weight 1 \
  --pixel-main-loss-weight 1 --pixel-branch-loss-weight 1 \
  --change-class-weight 1 --focal-gamma 0 --model-ema-decay 0 \
  --prototype-weight 0 --prototype-memory-weight 0 \
  --supervised-contrastive-weight 0 --boundary-loss-weight 0 \
  --swap-consistency-weight 0 --vat-weight 0 --coarse-aux-weight 0 \
  --context-aux-weight 0 --embedded-physics-aux-weight 0 \
  --learned-difference-aux-weight 0 --coarse-ablation-mode no_refinement \
  --disable-context-modeling --disable-embedded-physics-branch \
  --disable-embedded-physics-routing --disable-learned-difference-routing \
  --disable-physics --single-pass --midpoint-style-prob 0 \
  --spectral-gain-prob 0 --spectral-offset-prob 0 \
  --spectral-group-mask-prob 0 --same-class-mix-prob 0 \
  --experiment-group "final/$PROTOCOL/$DATASET" --run-name "seed_$SEED" \
  "${EXTRA[@]}"
