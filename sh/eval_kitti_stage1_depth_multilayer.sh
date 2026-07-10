#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

cd /home/dataset-local/lr/code/OccAny

PYTHONPATH=/home/dataset-local/lr/code/OccAny${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONPATH

PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-auto}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
EVAL_DATASET=${EVAL_DATASET:-kitti}
MIN_DEPTH=${MIN_DEPTH:-1e-3}
MAX_DEPTH=${MAX_DEPTH:-80.0}
MAX_BATCHES=${MAX_BATCHES:-0}
TARGET_FRAME_ONLY=${TARGET_FRAME_ONLY:-0}

EXP_NAMES=(
  enc_l12
  enc_l8_12_16
  enc_l6_12_18_24
)

EXP_DIRS=(
  /home/dataset-local/lr/code/OccAny/output/kitti_stage1_5f_4gpu_depth_postfusion_enc_l12
  /home/dataset-local/lr/code/OccAny/output/kitti_stage1_5f_4gpu_depth_postfusion_enc_l8_12_16
  /home/dataset-local/lr/code/OccAny/output/kitti_stage1_5f_4gpu_depth_postfusion_enc_l6_12_18_24
)

COMMON_ARGS=(
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --eval_dataset "${EVAL_DATASET}"
  --min_depth "${MIN_DEPTH}"
  --max_depth "${MAX_DEPTH}"
)

if [[ "${MAX_BATCHES}" != "0" ]]; then
  COMMON_ARGS+=(--max_batches "${MAX_BATCHES}")
fi

if [[ "${TARGET_FRAME_ONLY}" == "1" || "${TARGET_FRAME_ONLY}" == "true" ]]; then
  COMMON_ARGS+=(--target_frame_only)
fi

RESULT_JSONS=()

run_eval() {
  local name="$1"
  local exp_dir="$2"
  local ckpt="${exp_dir}/checkpoint-last.pth"
  local out_json="${exp_dir}/depth_metrics_${EVAL_DATASET}_final.json"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[eval:${name}] missing checkpoint: ${ckpt}" >&2
    exit 1
  fi

  echo "[eval:${name}] ckpt=${ckpt}"
  echo "[eval:${name}] output=${out_json}"
  if (( NPROC_PER_NODE > 1 )); then
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
      -m ft.kitti_stage1_5f.tools.eval_pointmap_depth \
      --ckpt "${ckpt}" \
      --output_json "${out_json}" \
      "${COMMON_ARGS[@]}"
  else
    "${PYTHON}" -m ft.kitti_stage1_5f.tools.eval_pointmap_depth \
      --nodist \
      --ckpt "${ckpt}" \
      --output_json "${out_json}" \
      "${COMMON_ARGS[@]}"
  fi

  RESULT_JSONS+=("${name}:${out_json}")
}

for i in "${!EXP_NAMES[@]}"; do
  run_eval "${EXP_NAMES[$i]}" "${EXP_DIRS[$i]}"
done

SUMMARY_PREFIX=/home/dataset-local/lr/code/OccAny/output/kitti_stage1_5f_4gpu_depth_postfusion_multilayer_summary_${EVAL_DATASET}
SUMMARY_JSON="${SUMMARY_PREFIX}.json"
SUMMARY_CSV="${SUMMARY_PREFIX}.csv"

"${PYTHON}" - "${SUMMARY_JSON}" "${SUMMARY_CSV}" "${RESULT_JSONS[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_json = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
items = sys.argv[3:]

rows = []
for item in items:
    name, path = item.split(":", 1)
    data = json.loads(Path(path).read_text())
    overall = data["metrics"]["overall"]
    rows.append(
        {
            "name": name,
            "ckpt": data["ckpt"],
            "json": path,
            "abs_rel": overall.get("abs_rel"),
            "rmse": overall.get("rmse"),
            "mae": overall.get("mae"),
            "delta1": overall.get("delta1"),
            "delta2": overall.get("delta2"),
            "delta3": overall.get("delta3"),
            "silog": overall.get("silog"),
            "valid_pixels": overall.get("valid_pixels"),
            "valid_frames": overall.get("valid_frames"),
            "pred_mean": overall.get("pred_mean"),
            "gt_mean": overall.get("gt_mean"),
        }
    )

summary_json.parent.mkdir(parents=True, exist_ok=True)
summary_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

fieldnames = [
    "name",
    "abs_rel",
    "rmse",
    "mae",
    "delta1",
    "delta2",
    "delta3",
    "silog",
    "valid_pixels",
    "valid_frames",
    "pred_mean",
    "gt_mean",
    "ckpt",
    "json",
]
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

best = min(rows, key=lambda r: float(r["abs_rel"]))
print(f"[summary] wrote {summary_json}")
print(f"[summary] wrote {summary_csv}")
print(
    "[summary] best_abs_rel="
    f"{best['name']} abs_rel={best['abs_rel']:.6f} "
    f"rmse={best['rmse']:.6f} delta1={best['delta1']:.6f}"
)
PY
