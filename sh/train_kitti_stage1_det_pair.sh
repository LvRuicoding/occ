#!/usr/bin/env bash
set -euo pipefail

cd /home/dataset-local/lr/code/OccAny

PYTHONPATH=/home/dataset-local/lr/code/OccAny${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONPATH

KITTI_DET_ROOT=/home/dataset-local/lr/code/OccAny/raw_data/OpenDataLab___KITTI_Object
OCCANY_CKPT=/home/dataset-local/lr/code/OccAny/checkpoints/occany_recon.pth
TORCHRUN=${TORCHRUN:-/home/dataset-local/envs/occany/bin/torchrun}

COMMON_ARGS=(
  -m ft.kitti_stage1_5f.tools.train
  --kitti_det_root "${KITTI_DET_ROOT}"
  --occany_ckpt "${OCCANY_CKPT}"
  --det_pc_range 0.0 -40.0 -3.0 70.4 40.0 3.4
  --det_depth_bound 1.0 80.0 0.4
  --depth_supervision
  --no-freeze_backbone
)

echo "[1/2] Training det_postfusion_only"
"${TORCHRUN}" --standalone --nproc_per_node=4 \
  "${COMMON_ARGS[@]}" \
  --exp det_postfusion_only \
  --output_dir /home/dataset-local/lr/code/OccAny/output/kitti_stage1_5f_4gpu_det_postfusion_only_fix

echo "[2/2] Training det_original (no point-cloud fusion branch)"
"${TORCHRUN}" --standalone --nproc_per_node=4 \
  "${COMMON_ARGS[@]}" \
  --exp det_original \
  --output_dir /home/dataset-local/lr/code/OccAny/output/kitti_stage1_5f_4gpu_det_original_fix
