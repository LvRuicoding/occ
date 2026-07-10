#!/usr/bin/env bash
set -euo pipefail

cd /home/dataset-local/lr/code/OccAny

PYTHONPATH=/home/dataset-local/lr/code/OccAny${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONPATH
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

PYTHON=${PYTHON:-/home/dataset-local/envs/occany/bin/python}

exec "${PYTHON}" -m ft.kitti_stage1_5f.tools.visualize_kitti_object_det "$@"
