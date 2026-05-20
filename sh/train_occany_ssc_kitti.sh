#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Reuse shared env setup (PYTHONPATH for vendored deps).
source sh/train_common.sh
occany_prepare_train_env "$REPO_ROOT"

# Default paths -- override via env.
: "${SEMKITTI_ROOT:=$REPO_ROOT/raw_data/OpenDataLab___KITTI_Odometry_2012}"
: "${KITTIODO_ROOT:=$SEMKITTI_ROOT}"
: "${OCCANY_CKPT:=$REPO_ROOT/checkpoints/occany.pth}"
: "${OUTPUT_DIR:=$REPO_ROOT/tb_log_occany/ssc_kitti}"

: "${BATCH_SIZE:=1}"
: "${ACCUM_ITER:=1}"
: "${NUM_WORKERS:=4}"
: "${EPOCHS:=20}"
: "${LR:=1e-4}"
: "${WIDTH:=512}"
: "${HEIGHT:=160}"
: "${N_RENDER:=4}"

# Single-GPU launcher; for multi-GPU run with torchrun and adjust LR accordingly.
LAUNCHER="ft/semantickitti_ft/train.py"
CMD="python $LAUNCHER"
if [ "${NUM_GPU_PER_NODE:-1}" -gt 1 ]; then
    CMD="torchrun --standalone --nproc_per_node=${NUM_GPU_PER_NODE} $LAUNCHER"
fi

echo "RUNNING: $CMD"

$CMD \
    --semkitti_root "$SEMKITTI_ROOT" \
    --kittiodo_root "$KITTIODO_ROOT" \
    --occany_ckpt "$OCCANY_CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --accum_iter "$ACCUM_ITER" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --width "$WIDTH" --height "$HEIGHT" \
    --n_render_views "$N_RENDER" \
    --amp bf16
