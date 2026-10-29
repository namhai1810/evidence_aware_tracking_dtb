#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
OUTPUT_DIR="${HIVE_HR_REID_OUTPUT:-${SCRIPT_DIR}/outputs/hive_v2_hr_reid_replay/full_run}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing project Python environment: ${PYTHON_BIN}" >&2
  exit 1
fi

export MPLCONFIGDIR="${SCRIPT_DIR}/outputs/.matplotlib"
mkdir -p "${MPLCONFIGDIR}" "${OUTPUT_DIR}"

exec "${PYTHON_BIN}" -m hive_hr_reid.run_experiment \
  --dataset-root "${SCRIPT_DIR}/../QuadTrack/train" \
  --hive-dataset-root "${SCRIPT_DIR}/outputs/hive_history_ranking_v2" \
  --checkpoint "${SCRIPT_DIR}/hr_finding/runs/hive_history_v2_seed42/checkpoints/best.pt" \
  --lr-pred-dir "${SCRIPT_DIR}/outputs/all_sequences_reid_512x120/predictions" \
  --reid-model "${SCRIPT_DIR}/yolo26n-reid.onnx" \
  --output-dir "${OUTPUT_DIR}" \
  --overwrite \
  "$@"

