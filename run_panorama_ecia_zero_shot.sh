#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
CONFIG="${SCRIPT_DIR}/hive_ecia/panorama_config.yaml"
DEVICE="${ECIA_DEVICE:-}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python environment: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ -z "${DEVICE}" ]]; then
  DEVICE="$(${PYTHON_BIN} -c "import torch; print('cuda:0' if torch.cuda.is_available() else 'cpu')")"
fi

export MPLCONFIGDIR="${SCRIPT_DIR}/outputs/.matplotlib"
mkdir -p "${MPLCONFIGDIR}"
cd "${SCRIPT_DIR}"

echo "[1/3] Build frame-aligned, sequence-disjoint panorama switch events"
"${PYTHON_BIN}" -m hive_ecia.panorama_dataset \
  --config "${CONFIG}" \
  --overwrite

echo "[2/3] Verify leakage audit"
"${PYTHON_BIN}" -c "import json,pathlib; p=pathlib.Path('outputs/panorama_ecia_zero_shot/leakage_audit.json'); x=json.loads(p.read_text()); assert x['passed'] and x['sequence_disjoint'] and x['manual_splits_match']; print('Leakage audit passed:', p)"

echo "[3/3] Run frozen QuadTrack HIVE + ECIA Full on held-out panorama test"
"${PYTHON_BIN}" -m hive_ecia.panorama_evaluate \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --overwrite

echo "Done: ${SCRIPT_DIR}/outputs/panorama_ecia_zero_shot/zero_shot_evaluation/REPORT.md"
