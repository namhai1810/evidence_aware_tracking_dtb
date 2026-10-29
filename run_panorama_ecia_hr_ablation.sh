#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
CONFIG="${SCRIPT_DIR}/hive_ecia/panorama_config.yaml"
DEVICE="${ECIA_DEVICE:-}"
EVENT_ROOT="${SCRIPT_DIR}/outputs/panorama_ecia_zero_shot"
FULL_METRICS="${EVENT_ROOT}/zero_shot_evaluation/metrics.json"
BASE_OUTPUT="${EVENT_ROOT}/zero_shot_evaluation_base_only"
COMPARE_OUTPUT="${EVENT_ROOT}/base_only_vs_full"

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

if [[ ! -f "${EVENT_ROOT}/leakage_audit.json" || ! -f "${FULL_METRICS}" ]]; then
  echo "Missing frozen DTB event/full artifacts; running the zero-shot Full protocol first"
  ECIA_DEVICE="${DEVICE}" bash "${SCRIPT_DIR}/run_panorama_ecia_zero_shot.sh"
fi

echo "[1/3] Verify the leakage-safe DTB split and existing Full result"
"${PYTHON_BIN}" -c "import json,pathlib; p=pathlib.Path('${EVENT_ROOT}/leakage_audit.json'); x=json.loads(p.read_text()); assert x['passed'] and x['sequence_disjoint']; print('Leakage audit passed:', p)"

echo "[2/3] Run frozen QuadTrack ECIA Base-only on the identical held-out DTB events"
"${PYTHON_BIN}" -m hive_ecia.panorama_evaluate \
  --config "${CONFIG}" \
  --event-root "${EVENT_ROOT}" \
  --output-dir "${BASE_OUTPUT}" \
  --ablation base_only \
  --ecia-checkpoint "${SCRIPT_DIR}/outputs/hive_ecia_experiment/base_only/training/best.pt" \
  --ecia-thresholds "${SCRIPT_DIR}/outputs/hive_ecia_experiment/base_only/validation_thresholds.json" \
  --device "${DEVICE}" \
  --overwrite

echo "[3/3] Build paired Base-only vs Full metrics, figures, and report"
"${PYTHON_BIN}" -m hive_ecia.panorama_ablation_compare \
  --base-only "${BASE_OUTPUT}/metrics.json" \
  --full "${FULL_METRICS}" \
  --output-dir "${COMPARE_OUTPUT}" \
  --overwrite

echo "Done: ${COMPARE_OUTPUT}/REPORT.md"
