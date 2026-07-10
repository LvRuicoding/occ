#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-/home/dataset-local/envs/occany}"
ACTIVATE_CONDA="${ACTIVATE_CONDA:-auto}"
if [[ "${ACTIVATE_CONDA}" == "auto" ]]; then
  if [[ "${CONDA_PREFIX:-}" == "${CONDA_ENV}" || "${CONDA_DEFAULT_ENV:-}" == "$(basename "${CONDA_ENV}")" ]]; then
    ACTIVATE_CONDA="0"
  else
    ACTIVATE_CONDA="1"
  fi
fi
if [[ "${ACTIVATE_CONDA}" == "1" ]]; then
  if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
    source /opt/conda/etc/profile.d/conda.sh
    set +u
    if ! conda activate "${CONDA_ENV}"; then
      set -u
      echo "Failed to activate conda env: ${CONDA_ENV}" >&2
      exit 1
    fi
    set -u
  elif [[ -d "${CONDA_ENV}/bin" ]]; then
    export PATH="${CONDA_ENV}/bin:${PATH}"
  else
    echo "Conda env not found: ${CONDA_ENV}" >&2
    exit 1
  fi
fi

GPUS="${GPUS:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${GPUS}}"
DRY_RUN="${DRY_RUN:-0}"

KITTI_PROCESSED_ROOT="${KITTI_PROCESSED_ROOT:-${REPO_ROOT}/data/kitti_processed}"
NUSCENES_PROCESSED_ROOT="${NUSCENES_PROCESSED_ROOT:-${REPO_ROOT}/data/nuscenes_processed}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints}"
OCCANY_CKPT="${OCCANY_CKPT:-${CHECKPOINT_ROOT}/occany_recon.pth}"
INIT_FROM="${INIT_FROM:-${REPO_ROOT}/output/kitti_stage1_5f_4gpu_pointmap_postfusion_only/checkpoint-last.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/three_dataset_pointmap_postfusion_only_experiments}"

WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-160}"
NUM_FRAMES="${NUM_FRAMES:-5}"
FRAME_STRIDE="${FRAME_STRIDE:-4}"
NUSCENES_FRAME_STRIDE="${NUSCENES_FRAME_STRIDE:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-6}"
EPOCHS="${EPOCHS:-20}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
LR="${LR:-1e-4}"
BASE_LR="${BASE_LR:-${LR}}"
HEAD_LR="${HEAD_LR:-${LR}}"
CLASSIFIER_LR="${CLASSIFIER_LR:-${LR}}"
MIN_LR="${MIN_LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
ITERS_PER_EPOCH="${ITERS_PER_EPOCH:-0}"
VAL_ITERS="${VAL_ITERS:-0}"
AMP="${AMP:-bf16}"
SEED="${SEED:-0}"
ACCUM_ITER="${ACCUM_ITER:-1}"
SAVE_FREQ="${SAVE_FREQ:-2}"
KEEP_FREQ="${KEEP_FREQ:-5}"
EVAL_FREQ="${EVAL_FREQ:-1}"
FREEZE_BACKBONE="${FREEZE_BACKBONE:-1}"
FREEZE_BACKBONE_EPOCHS="${FREEZE_BACKBONE_EPOCHS:-0}"
MAX_POINTS_PER_SWEEP="${MAX_POINTS_PER_SWEEP:-0}"
MIXED_DATASET_RATIO="${MIXED_DATASET_RATIO:-1:1}"
CLASS_WEIGHTS_PATH="${CLASS_WEIGHTS_PATH:-}"
POINTMAP_LOSS_WEIGHT="${POINTMAP_LOSS_WEIGHT:-0.1}"
POINTMAP_CONF_ALPHA="${POINTMAP_CONF_ALPHA:-0.2}"

require_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "Required path does not exist: ${path}" >&2
    exit 1
  fi
}

require_path "${KITTI_PROCESSED_ROOT}"
require_path "${NUSCENES_PROCESSED_ROOT}"
require_path "${OCCANY_CKPT}"
if [[ -n "${INIT_FROM}" ]]; then
  require_path "${INIT_FROM}"
fi

FREEZE_ARGS=()
if [[ "${FREEZE_BACKBONE}" == "1" ]]; then
  FREEZE_ARGS+=(--freeze_backbone)
else
  FREEZE_ARGS+=(--no-freeze_backbone)
fi

COMMON_ARGS=(
  --processed_root "${KITTI_PROCESSED_ROOT}"
  --occany_ckpt "${OCCANY_CKPT}"
  --exp pointmap_postfusion_only
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --num_frames "${NUM_FRAMES}"
  --frame_stride "${FRAME_STRIDE}"
  --nuscenes_frame_stride "${NUSCENES_FRAME_STRIDE}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --epochs "${EPOCHS}"
  --warmup_epochs "${WARMUP_EPOCHS}"
  --lr "${LR}"
  --base_lr "${BASE_LR}"
  --head_lr "${HEAD_LR}"
  --classifier_lr "${CLASSIFIER_LR}"
  --min_lr "${MIN_LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --iters_per_epoch "${ITERS_PER_EPOCH}"
  --val_iters "${VAL_ITERS}"
  --amp "${AMP}"
  --seed "${SEED}"
  --accum_iter "${ACCUM_ITER}"
  --save_freq "${SAVE_FREQ}"
  --keep_freq "${KEEP_FREQ}"
  --eval_freq "${EVAL_FREQ}"
  --freeze_backbone_epochs "${FREEZE_BACKBONE_EPOCHS}"
  --max_points_per_sweep "${MAX_POINTS_PER_SWEEP}"
  --pointmap_loss_weight "${POINTMAP_LOSS_WEIGHT}"
  --pointmap_conf_alpha "${POINTMAP_CONF_ALPHA}"
  "${FREEZE_ARGS[@]}"
)

if [[ -n "${INIT_FROM}" ]]; then
  COMMON_ARGS+=(--init_from "${INIT_FROM}")
fi

MULTI_DATASET_ARGS=(
  --multi_dataset
  --nuscenes_processed_root "${NUSCENES_PROCESSED_ROOT}"
)

if [[ -n "${CLASS_WEIGHTS_PATH}" ]]; then
  MULTI_DATASET_ARGS+=(--class_weights_path "${CLASS_WEIGHTS_PATH}")
fi

run_exp() {
  local name="$1"
  shift
  local output_dir="${OUTPUT_ROOT}/${name}"
  local cmd=(
    torchrun
    --standalone
    --nnodes=1
    --nproc_per_node="${NPROC_PER_NODE}"
    -m ft.kitti_stage1_5f.tools.train
    "${COMMON_ARGS[@]}"
    --output_dir "${output_dir}"
    "$@"
  )

  echo
  echo "[${name}]"
  printf '%q ' "${cmd[@]}"
  echo

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  mkdir -p "${output_dir}"
  "${cmd[@]}"
}

run_exp "01_kitti_only_pointmap_postfusion_only" "${MULTI_DATASET_ARGS[@]}" --dataset_ratio "1:0"
run_exp "02_nuscenes_only_pointmap_postfusion_only" "${MULTI_DATASET_ARGS[@]}" --dataset_ratio "0:1"
run_exp "03_kitti_nuscenes_pointmap_postfusion_only" "${MULTI_DATASET_ARGS[@]}" --dataset_ratio "${MIXED_DATASET_RATIO}"
