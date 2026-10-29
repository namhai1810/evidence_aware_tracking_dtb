#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
CONFIG="${SCRIPT_DIR}/hive_ecia/panorama_config.yaml"
DEVICE="${ECIA_DEVICE:-}"
EVENT_ROOT="${SCRIPT_DIR}/outputs/panorama_ecia_zero_shot"
BASE_OUTPUT="${EVENT_ROOT}/zero_shot_all_base_only"
FULL_OUTPUT="${EVENT_ROOT}/zero_shot_all_full"
COMPARE_OUTPUT="${EVENT_ROOT}/base_only_vs_full_all"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python environment: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ -z "${DEVICE}" ]]; then
  DEVICE="$(${PYTHON_BIN} -c "import torch; print('cuda:0' if torch.cuda.is_available() else 'cpu')")"
fi
if [[ ! -f "${EVENT_ROOT}/leakage_audit.json" ]]; then
  echo "Missing DTB event dataset. Run run_panorama_ecia_zero_shot.sh first." >&2
  exit 1
fi

export MPLCONFIGDIR="${SCRIPT_DIR}/outputs/.matplotlib"
mkdir -p "${MPLCONFIGDIR}"
cd "${SCRIPT_DIR}"

echo "[1/4] Verify sequence-disjoint source preparation and frozen zero-shot protocol"
"${PYTHON_BIN}" -c "import json,pathlib; p=pathlib.Path('${EVENT_ROOT}/leakage_audit.json'); x=json.loads(p.read_text()); assert x['passed'] and x['sequence_disjoint']; print('Leakage audit passed:', p)"

echo "[2/4] Evaluate frozen ECIA Base-only on all five unseen DTB videos"
"${PYTHON_BIN}" -m hive_ecia.panorama_evaluate \
  --config "${CONFIG}" \
  --event-root "${EVENT_ROOT}" \
  --output-dir "${BASE_OUTPUT}" \
  --all-sequences \
  --ablation base_only \
  --ecia-checkpoint "${SCRIPT_DIR}/outputs/hive_ecia_experiment/base_only/training/best.pt" \
  --ecia-thresholds "${SCRIPT_DIR}/outputs/hive_ecia_experiment/base_only/validation_thresholds.json" \
  --device "${DEVICE}" \
  --overwrite

echo "[3/4] Evaluate frozen ECIA Full on the identical DTB events"
"${PYTHON_BIN}" -m hive_ecia.panorama_evaluate \
  --config "${CONFIG}" \
  --event-root "${EVENT_ROOT}" \
  --output-dir "${FULL_OUTPUT}" \
  --all-sequences \
  --ablation full \
  --ecia-checkpoint "${SCRIPT_DIR}/outputs/hive_ecia_experiment/full/training/best.pt" \
  --ecia-thresholds "${SCRIPT_DIR}/outputs/hive_ecia_experiment/full/validation_thresholds.json" \
  --device "${DEVICE}" \
  --overwrite

echo "[4/4] Create aggregate/per-video comparison, figures, and report"
"${PYTHON_BIN}" -m hive_ecia.panorama_ablation_compare \
  --base-only "${BASE_OUTPUT}/metrics.json" \
  --full "${FULL_OUTPUT}/metrics.json" \
  --output-dir "${COMPARE_OUTPUT}" \
  --overwrite

echo "Done: ${COMPARE_OUTPUT}/REPORT.md"
