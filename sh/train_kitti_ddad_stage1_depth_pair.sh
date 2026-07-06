#!/usr/bin/env bash
set -euo pipefail

cd /home/dataset-local/lr/code/OccAny

PYTHONPATH=/home/dataset-local/lr/code/OccAny${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONPATH

KITTI_PROCESSED_ROOT=/home/dataset-local/lr/code/OccAny/data/kitti_processed
DDAD_PROCESSED_ROOT=/home/dataset-local/lr/code/OccAny/data/ddad_processed
DDAD_RAW_ROOT=/home/dataset-local/lr/code/OccAny/raw_data/OpenDataLab___DDAD/raw/ddad_train_val
OCCANY_CKPT=/home/dataset-local/lr/code/OccAny/checkpoints/occany_recon.pth

echo "[1/2] Training KITTI+DDAD depth_original"
torchrun --standalone --nproc_per_node=4 \
  -m ft.kitti_stage1_5f.tools.train \
  --multi_dataset \
  --exp depth_original \
  --processed_root "${KITTI_PROCESSED_ROOT}" \
  --ddad_processed_root "${DDAD_PROCESSED_ROOT}" \
  --ddad_raw_root "${DDAD_RAW_ROOT}" \
  --occany_ckpt "${OCCANY_CKPT}" \
  --output_dir /home/dataset-local/lr/code/OccAny/output/kitti_ddad_stage1_5f_4gpu_depth_original \
  --no-freeze_backbone

echo "[2/2] Training KITTI+DDAD depth_postfusion_only"
torchrun --standalone --nproc_per_node=4 \
  -m ft.kitti_stage1_5f.tools.train \
  --multi_dataset \
  --exp depth_postfusion_only \
  --processed_root "${KITTI_PROCESSED_ROOT}" \
  --ddad_processed_root "${DDAD_PROCESSED_ROOT}" \
  --ddad_raw_root "${DDAD_RAW_ROOT}" \
  --occany_ckpt "${OCCANY_CKPT}" \
  --output_dir /home/dataset-local/lr/code/OccAny/output/kitti_ddad_stage1_5f_4gpu_depth_postfusion_only \
  --no-freeze_backbone
