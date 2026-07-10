#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

cd /home/dataset-local/lr/code/OccAny

PYTHONPATH=/home/dataset-local/lr/code/OccAny${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONPATH

PROCESSED_ROOT=/home/dataset-local/lr/code/OccAny/data/kitti_processed
OCCANY_CKPT=/home/dataset-local/lr/code/OccAny/checkpoints/occany_recon.pth
GPU_WAIT_IDS=${GPU_WAIT_IDS:-0,1,2,3}
GPU_WAIT_THRESHOLD_MB=${GPU_WAIT_THRESHOLD_MB:-10240}
GPU_WAIT_INTERVAL_SEC=${GPU_WAIT_INTERVAL_SEC:-60}
GPU_WAIT_DISABLE=${GPU_WAIT_DISABLE:-0}

if [[ ! -f "${OCCANY_CKPT}" ]]; then
  echo "Missing OccAny checkpoint: ${OCCANY_CKPT}" >&2
  exit 1
fi

wait_for_available_gpus() {
  if [[ "${GPU_WAIT_DISABLE}" == "1" || "${GPU_WAIT_DISABLE}" == "true" ]]; then
    echo "[gpu-wait] disabled"
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[gpu-wait] nvidia-smi not found" >&2
    exit 1
  fi

  local -a gpu_ids
  IFS=',' read -r -a gpu_ids <<< "${GPU_WAIT_IDS}"
  if (( ${#gpu_ids[@]} == 0 )); then
    echo "[gpu-wait] GPU_WAIT_IDS is empty" >&2
    exit 1
  fi

  echo "[gpu-wait] waiting for GPUs ${GPU_WAIT_IDS} to use < ${GPU_WAIT_THRESHOLD_MB} MiB each"
  while true; do
    local -a used_mibs
    mapfile -t used_mibs < <(
      nvidia-smi \
        --id="${GPU_WAIT_IDS}" \
        --query-gpu=memory.used \
        --format=csv,noheader,nounits
    )
    if (( ${#used_mibs[@]} != ${#gpu_ids[@]} )); then
      echo "[gpu-wait] expected ${#gpu_ids[@]} readings, got ${#used_mibs[@]}" >&2
      exit 1
    fi

    local ready=1
    local status=""
    local i used
    for i in "${!gpu_ids[@]}"; do
      used="${used_mibs[$i]//[^0-9]/}"
      if [[ -z "${used}" ]]; then
        echo "[gpu-wait] failed to parse memory reading: ${used_mibs[$i]}" >&2
        exit 1
      fi
      status+=" gpu${gpu_ids[$i]}=${used}MiB"
      if (( used >= GPU_WAIT_THRESHOLD_MB )); then
        ready=0
      fi
    done

    echo "[gpu-wait] $(date '+%F %T')${status}"
    if (( ready )); then
      echo "[gpu-wait] all requested GPUs are below threshold; starting training"
      return
    fi
    sleep "${GPU_WAIT_INTERVAL_SEC}"
  done
}

run_exp() {
  local name="$1"
  local layers="$2"
  local output_dir="/home/dataset-local/lr/code/OccAny/output/kitti_stage1_5f_4gpu_depth_postfusion_${name}"

  echo "[multilayer] Training ${name} layers=${layers}"
  torchrun --standalone --nproc_per_node=4 \
    -m ft.kitti_stage1_5f.tools.train \
    --exp depth_postfusion_only \
    --processed_root "${PROCESSED_ROOT}" \
    --occany_ckpt "${OCCANY_CKPT}" \
    --output_dir "${output_dir}" \
    --no-freeze_backbone \
    --encoder_lidar_layers "${layers}" \
    --encoder_lidar_alpha_init 1.0 \
    --encoder_lidar_num_heads 8 \
    --encoder_lidar_window 4 \
    --encoder_lidar_vfe_d_voxel 128 \
    --epochs 20 \
    --lr 1e-4 \
    --eval_freq 4 \
    --save_freq 4
}

wait_for_available_gpus

run_exp "enc_l12" "12"
run_exp "enc_l8_12_16" "8,12,16"
run_exp "enc_l6_12_18_24" "6,12,18,24"
