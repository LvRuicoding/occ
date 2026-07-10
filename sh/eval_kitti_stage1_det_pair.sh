#!/usr/bin/env bash
set -euo pipefail

cd /home/dataset-local/lr/code/OccAny

PYTHONPATH=/home/dataset-local/lr/code/OccAny${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONPATH
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

OCCANY_ENV=${OCCANY_ENV:-/home/dataset-local/envs/occany}
PYTHON=${PYTHON:-${OCCANY_ENV}/bin/python}
TORCHRUN=${TORCHRUN:-${OCCANY_ENV}/bin/torchrun}
DEVICE=${DEVICE:-auto}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
PRINT_FREQ=${PRINT_FREQ:-50}
SAVE_ANNOS=${SAVE_ANNOS:-0}
KITTI_DET_ROOT=${KITTI_DET_ROOT:-/home/dataset-local/lr/code/OccAny/raw_data/OpenDataLab___KITTI_Object}
OCCANY_CKPT=${OCCANY_CKPT:-/home/dataset-local/lr/code/OccAny/checkpoints/occany_recon.pth}

final_ckpt() {
  local exp_dir=$1
  local final
  local latest
  if [[ ! -d "${exp_dir}" ]]; then
    echo "[eval][error] experiment directory not found: ${exp_dir}" >&2
    return 1
  fi

  for final in checkpoint-last.pth checkpoint-final.pth; do
    if [[ -f "${exp_dir}/${final}" ]]; then
      printf '%s\n' "${exp_dir}/${final}"
      return 0
    fi
  done

  latest=$(
    find "${exp_dir}" -maxdepth 1 -type f -name 'checkpoint-[0-9]*.pth' -printf '%f\n' \
      | sed -E 's/^checkpoint-([0-9]+)\.pth$/\1 &/' \
      | sort -n \
      | tail -n 1 \
      | cut -d' ' -f2-
  )
  if [[ -z "${latest}" ]]; then
    echo "[eval][error] no final checkpoint found under ${exp_dir}" >&2
    echo "[eval][hint] Expected checkpoint-last.pth, checkpoint-final.pth, or a file like checkpoint-19.pth. Train this experiment first or override the *_CKPT variable." >&2
    return 1
  fi
  printf '%s\n' "${exp_dir}/${latest}"
}

POSTFUSION_DIR=${POSTFUSION_DIR:-/home/dataset-local/lr/code/OccAny/output/kitti_stage1_5f_4gpu_det_postfusion_only_fix}
ORIGINAL_DIR=${ORIGINAL_DIR:-/home/dataset-local/lr/code/OccAny/output/kitti_stage1_5f_4gpu_det_original_fix}

missing_ckpt=0
if [[ -z "${POSTFUSION_CKPT:-}" ]] && ! POSTFUSION_CKPT=$(final_ckpt "${POSTFUSION_DIR}"); then
  missing_ckpt=1
fi
if [[ -z "${ORIGINAL_CKPT:-}" ]] && ! ORIGINAL_CKPT=$(final_ckpt "${ORIGINAL_DIR}"); then
  missing_ckpt=1
fi
if (( missing_ckpt )); then
  exit 1
fi
if [[ ! -d "${KITTI_DET_ROOT}" ]]; then
  echo "[eval][error] KITTI_DET_ROOT not found: ${KITTI_DET_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${OCCANY_CKPT}" ]]; then
  echo "[eval][error] OCCANY_CKPT not found: ${OCCANY_CKPT}" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ "${SAVE_ANNOS}" == "1" ]]; then
  EXTRA_ARGS+=(--save_annos)
fi

COMMON_ARGS=(
  --kitti_det_root "${KITTI_DET_ROOT}"
  --occany_ckpt "${OCCANY_CKPT}"
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --print_freq "${PRINT_FREQ}"
  --eval_backend official_cpp
)

run_eval() {
  local name=$1
  local ckpt=$2
  local out_json=$3

  echo "[eval] ${name}"
  echo "[eval] ckpt=${ckpt}"
  echo "[eval] output_json=${out_json}"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[eval][error] checkpoint not found: ${ckpt}" >&2
    echo "[eval][hint] Override POSTFUSION_CKPT or ORIGINAL_CKPT with an existing .pth file." >&2
    exit 1
  fi
  if (( NPROC_PER_NODE > 1 )); then
    "${TORCHRUN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" \
      -m ft.kitti_stage1_5f.tools.eval_kitti_object_det \
      --ckpt "${ckpt}" \
      --exp "${name}" \
      --output_json "${out_json}" \
      "${COMMON_ARGS[@]}" \
      "${EXTRA_ARGS[@]}"
  else
    "${PYTHON}" -m ft.kitti_stage1_5f.tools.eval_kitti_object_det \
      --nodist \
      --ckpt "${ckpt}" \
      --exp "${name}" \
      --output_json "${out_json}" \
      "${COMMON_ARGS[@]}" \
      "${EXTRA_ARGS[@]}"
  fi
}

run_eval \
  "det_postfusion_only" \
  "${POSTFUSION_CKPT}" \
  "$(dirname "${POSTFUSION_CKPT}")/kitti_det_metrics_val.json"

run_eval \
  "det_original" \
  "${ORIGINAL_CKPT}" \
  "$(dirname "${ORIGINAL_CKPT}")/kitti_det_metrics_val.json"

echo "Done. JSON metrics are next to each checkpoint; text reports are under eval_kitti_object_val/."
