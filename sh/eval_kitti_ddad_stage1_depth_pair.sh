#!/usr/bin/env bash
set -euo pipefail

cd /home/dataset-local/lr/code/OccAny

PYTHONPATH=/home/dataset-local/lr/code/OccAny${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONPATH

PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-auto}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MIN_DEPTH=${MIN_DEPTH:-1e-3}
MAX_DEPTH=${MAX_DEPTH:-80.0}

COMMON_ARGS=(
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --min_depth "${MIN_DEPTH}"
  --max_depth "${MAX_DEPTH}"
)

run_eval() {
  local ckpt=$1
  local dataset=$2
  local out_json=$3

  echo "[eval] ckpt=${ckpt}"
  echo "[eval] dataset=${dataset} output=${out_json}"
  if (( NPROC_PER_NODE > 1 )); then
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
      -m ft.kitti_stage1_5f.tools.eval_pointmap_depth \
      --ckpt "${ckpt}" \
      --eval_dataset "${dataset}" \
      --output_json "${out_json}" \
      "${COMMON_ARGS[@]}"
  else
    "${PYTHON}" -m ft.kitti_stage1_5f.tools.eval_pointmap_depth \
      --nodist \
      --ckpt "${ckpt}" \
      --eval_dataset "${dataset}" \
      --output_json "${out_json}" \
      "${COMMON_ARGS[@]}"
  fi
}

ORIG_DIR=/home/dataset-local/lr/code/OccAny/output/kitti_ddad_stage1_5f_4gpu_depth_original
POST_DIR=/home/dataset-local/lr/code/OccAny/output/kitti_ddad_stage1_5f_4gpu_depth_postfusion_only

run_eval "${ORIG_DIR}/checkpoint-last.pth" kitti "${ORIG_DIR}/depth_metrics_kitti.json"
run_eval "${ORIG_DIR}/checkpoint-last.pth" ddad "${ORIG_DIR}/depth_metrics_ddad.json"
run_eval "${POST_DIR}/checkpoint-last.pth" kitti "${POST_DIR}/depth_metrics_kitti.json"
run_eval "${POST_DIR}/checkpoint-last.pth" ddad "${POST_DIR}/depth_metrics_ddad.json"
