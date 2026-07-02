#!/usr/bin/env bash
set -euo pipefail

cd /home/dataset-local/lr/code/OccAny

PYTHONPATH=/home/dataset-local/lr/code/OccAny${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONPATH

PROCESSED_ROOT=/home/dataset-local/lr/code/OccAny/data/kitti_processed
OCCANY_PLUS_CKPT=/home/dataset-local/lr/code/OccAny/checkpoints/occany_plus_recon_1B.pth

echo "[1/1] Training depth_postfusion_only with DA3 / OccAny+ backbone"
torchrun --standalone --nproc_per_node=4 \
  -m ft.kitti_stage1_5f.tools.train \
  --exp depth_postfusion_only \
  --backbone da3 \
  --processed_root "${PROCESSED_ROOT}" \
  --occany_ckpt "${OCCANY_PLUS_CKPT}" \
  --width 518 \
  --height 168 \
  --patch_size 14 \
  --token_dim 3072 \
  --output_dir /home/dataset-local/lr/code/OccAny/output/kitti_stage1_5f_4gpu_depth_postfusion_only_da3 \
  --no-freeze_backbone
