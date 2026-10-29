#!/usr/bin/env bash
set -euo pipefail

# One-command, leakage-controlled HIVE + Lightweight ECIA experiment.
# HIVE v2 and ReID are frozen; only the three small ECIA ablations are trained.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
CONFIG="${SCRIPT_DIR}/hive_ecia/config.yaml"
OUTPUT_ROOT="${HIVE_ECIA_OUTPUT:-${SCRIPT_DIR}/outputs/hive_ecia_experiment}"
DEVICE="${ECIA_DEVICE:-}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing project Python environment: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -z "${DEVICE}" ]]; then
  DEVICE="$(${PYTHON_BIN} -c "import torch; print('cuda:0' if torch.cuda.is_available() else 'cpu')")"
fi

export MPLCONFIGDIR="${SCRIPT_DIR}/outputs/.matplotlib"
mkdir -p "${MPLCONFIGDIR}" "${OUTPUT_ROOT}"
cd "${SCRIPT_DIR}"
echo "Using device: ${DEVICE}"

echo "[1/7] Build frozen train/model-validation embedding caches (calibration and test excluded)"
"${PYTHON_BIN}" -m hive_ecia.prepare_data \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --overwrite

echo "[2/7] Verify leakage audit before optimization"
"${PYTHON_BIN}" -c "import json, pathlib; p=pathlib.Path('outputs/hive_ecia_cache/cache_audit.json'); x=json.loads(p.read_text()); groups=[set(x['train']['sequences']),set(x['validation']['sequences']),set(x['configured_calibration_sequences']),set(x['configured_test_sequences'])]; assert x['calibration_sequences_accessed'] is False and x['test_sequences_accessed'] is False; assert all(groups[i].isdisjoint(groups[j]) for i in range(len(groups)) for j in range(i+1,len(groups))); print('Leakage audit passed:', p)"

for MODE in base_only base_hr_gain full; do
  RUN_DIR="${OUTPUT_ROOT}/${MODE}"
  mkdir -p "${RUN_DIR}"

  echo "[3/7] Train ${MODE} using train cache; early-stop on model-validation"
  "${PYTHON_BIN}" -m hive_ecia.train \
    --config "${CONFIG}" \
    --ablation "${MODE}" \
    --device "${DEVICE}" \
    --output-dir "${RUN_DIR}/training"

  echo "[4/7] Calibration inference and calibration-only threshold selection: ${MODE}"
  "${PYTHON_BIN}" -m hive_ecia.inference \
    --config "${CONFIG}" \
    --ecia-checkpoint "${RUN_DIR}/training/best.pt" \
    --split validation \
    --sequences 0030 \
    --device "${DEVICE}" \
    --min-old-probability 0.0 \
    --min-old-margin -1.0 \
    --output "${RUN_DIR}/validation_scores.jsonl" \
    --overwrite

  "${PYTHON_BIN}" -m hive_ecia.calibrate \
    --config "${CONFIG}" \
    --validation-decisions "${RUN_DIR}/validation_scores.jsonl" \
    --validation-events "${SCRIPT_DIR}/outputs/hive_history_ranking_v2/train_data/events.jsonl" \
    --output "${RUN_DIR}/validation_thresholds.json"

done

# No held-out file is read before every ECIA checkpoint and merge threshold has
# been frozen above. The unchanged baseline is reproduced only at this point.
echo "[5/7] Reproduce unchanged HIVE v2 + HR ReID held-out baseline"
HIVE_HR_REID_OUTPUT="${OUTPUT_ROOT}/baseline_hive_hr_reid" \
  bash "${SCRIPT_DIR}/run_hive_v2_hr_reid_experiment.sh" --device "${DEVICE}"

for MODE in base_only base_hr_gain full; do
  RUN_DIR="${OUTPUT_ROOT}/${MODE}"
  echo "[6/7] Frozen checkpoint + threshold test inference/evaluation: ${MODE}"
  "${PYTHON_BIN}" -m hive_ecia.inference \
    --config "${CONFIG}" \
    --ecia-checkpoint "${RUN_DIR}/training/best.pt" \
    --split test \
    --device "${DEVICE}" \
    --thresholds-json "${RUN_DIR}/validation_thresholds.json" \
    --output "${RUN_DIR}/test_decisions.jsonl" \
    --overwrite

  "${PYTHON_BIN}" -m hive_ecia.evaluate \
    --decisions "${RUN_DIR}/test_decisions.jsonl" \
    --evaluation-events "${SCRIPT_DIR}/outputs/hive_history_ranking_v2/test_data/events.jsonl" \
    --output-dir "${RUN_DIR}/test_evaluation"
done

echo "[7/7] Compare unchanged baseline and held-out ECIA ablations"
"${PYTHON_BIN}" -m hive_ecia.compare \
  --baseline-summary "${OUTPUT_ROOT}/baseline_hive_hr_reid/summary.json" \
  --run "base_only=${OUTPUT_ROOT}/base_only/test_evaluation/metrics.json" \
  --run "base_hr_gain=${OUTPUT_ROOT}/base_hr_gain/test_evaluation/metrics.json" \
  --run "full=${OUTPUT_ROOT}/full/test_evaluation/metrics.json" \
  --output-dir "${OUTPUT_ROOT}/comparison"

echo "Done. Final report: ${OUTPUT_ROOT}/comparison/REPORT.md"
