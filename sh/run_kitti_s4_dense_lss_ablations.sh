#!/usr/bin/env bash
set -euo pipefail

cd /home/dataset-local/lr/code/OccAny

TORCHRUN="${TORCHRUN:-/home/dataset-local/envs/occany/bin/torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
EPOCHS="${EPOCHS:-20}"

COMMON_ARGS=(
  -m ft.kitti_stage1_5f.tools.train
  --processed_root /home/dataset-local/lr/code/OccAny/data/kitti_processed
  --kittiodo_root /home/dataset-local/lr/code/OccAny/raw_data/semantickitti
  --velodyne_root /home/dataset-local/lr/code/OccAny/data/kitti
  --occany_ckpt /home/dataset-local/lr/code/OccAny/checkpoints/occany_recon.pth
  --exp bevdetocc_lidar
  --frame_stride 4
  --geometry_channels 256
  --geometry_adapter_gate_init 0.0
  --dense_depth_min 1.0
  --dense_depth_max 80.0
  --depth_supervision
  --depth_loss_weight 0.05
  --batch_size 1
  --num_workers 6
  --amp bf16
  --epochs "${EPOCHS}"
  --lr 1e-4
)

run_exp() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START ${name}"
  "${TORCHRUN}" --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" "$@"
  echo "[$(date '+%F %T')] DONE  ${name}"
}

run_exp "1/3 dense LSS lift supervision only" \
  "${COMMON_ARGS[@]}" \
  --output_dir /home/dataset-local/lr/code/OccAny/output/kitti_stage1_5f_4gpu_bevdetocc_lidar_s4_dense_lss \
  --dense_lss_depth_supervision \
  --dense_lss_depth_loss_weight 0.05 \
  --no-dense_depth_supervision \
  --no-shared_geometry_adapter

run_exp "2/3 dense LSS lift supervision + gated adapter" \
  "${COMMON_ARGS[@]}" \
  --output_dir /home/dataset-local/lr/code/OccAny/output/kitti_stage1_5f_4gpu_bevdetocc_lidar_s4_dense_lss_gated_adapter \
  --dense_lss_depth_supervision \
  --dense_lss_depth_loss_weight 0.05 \
  --no-dense_depth_supervision \
  --shared_geometry_adapter

run_exp "3/3 gated adapter only" \
  "${COMMON_ARGS[@]}" \
  --output_dir /home/dataset-local/lr/code/OccAny/output/kitti_stage1_5f_4gpu_bevdetocc_lidar_s4_gated_adapter \
  --no-dense_lss_depth_supervision \
  --no-dense_depth_supervision \
  --shared_geometry_adapter
