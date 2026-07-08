#!/usr/bin/env bash
set -euo pipefail

cd /home/dataset-local/lr/code/OccAny

PYTHONPATH=/home/dataset-local/lr/code/OccAny${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONPATH

KITTI_DET_ROOT=/home/dataset-local/lr/code/OccAny/raw_data/OpenDataLab___KITTI_Object
OCCANY_CKPT=/home/dataset-local/lr/code/OccAny/checkpoints/occany_recon.pth
TORCHRUN=${TORCHRUN:-/home/dataset-local/envs/occany/bin/torchrun}

echo "[1/1] Training det_postfusion_only"
"${TORCHRUN}" --standalone --nproc_per_node=4 \
  -m ft.kitti_stage1_5f.tools.train \
  --exp det_postfusion_only \
  --kitti_det_root "${KITTI_DET_ROOT}" \
  --occany_ckpt "${OCCANY_CKPT}" \
  --output_dir /home/dataset-local/lr/code/OccAny/output/kitti_stage1_5f_4gpu_det_postfusion_only_fix \
  --no-freeze_backbone
