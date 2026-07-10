#!/usr/bin/env bash
set -euo pipefail

cd /home/dataset-local/lr/code/OccAny

PYTHONPATH=/home/dataset-local/lr/code/OccAny${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONPATH

PROCESSED_ROOT=/home/dataset-local/lr/code/OccAny/data/kitti_processed
OCCANY_CKPT=/home/dataset-local/lr/code/OccAny/checkpoints/occany_recon.pth

echo "[1/2] Training depth_original"
torchrun --standalone --nproc_per_node=4 \
  -m ft.kitti_stage1_5f.tools.train \
  --exp depth_original \
  --processed_root "${PROCESSED_ROOT}" \
  --occany_ckpt "${OCCANY_CKPT}" \
  --output_dir /home/dataset-local/lr/code/OccAny/output/kitti_stage1_5f_4gpu_depth_original \
  --no-freeze_backbone

echo "[2/2] Training depth_postfusion_only"
torchrun --standalone --nproc_per_node=4 \
  -m ft.kitti_stage1_5f.tools.train \
  --exp depth_postfusion_only \
  --processed_root "${PROCESSED_ROOT}" \
  --occany_ckpt "${OCCANY_CKPT}" \
  --output_dir /home/dataset-local/lr/code/OccAny/output/kitti_stage1_5f_4gpu_depth_postfusion_only \
  --no-freeze_backbone
