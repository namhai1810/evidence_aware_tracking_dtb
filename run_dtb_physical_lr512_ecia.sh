#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${DTB_PYTHON_BIN:-$(command -v python)}"
CONFIG="${SCRIPT_DIR}/hive_ecia/panorama_config.yaml"
DEVICE="${ECIA_DEVICE:-0}"
RUN_ID="${DTB_LR_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${DTB_LR_OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/dtb_physical_lr512_ecia/${RUN_ID}}"
MANUAL_ROOT="${SCRIPT_DIR}/../cvat_packages_trim300_last100/object_tracking_prepared_manual_dataset"
FLAT_DATASET="${RUN_ROOT}/source_dataset"
TRACKING_ROOT="${RUN_ROOT}/physical_lr_tracking"
EVENT_ROOT="${RUN_ROOT}/event_protocol"
BASE_OUTPUT="${RUN_ROOT}/base_only"
FULL_OUTPUT="${RUN_ROOT}/full"
COMPARE_OUTPUT="${RUN_ROOT}/comparison"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python environment: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Run directory already exists; refusing to overwrite: ${RUN_ROOT}" >&2
  exit 1
fi

mkdir -p "${FLAT_DATASET}" "${TRACKING_ROOT}"
ln -s "${MANUAL_ROOT}/mot_challenge/train/normal_walking" "${FLAT_DATASET}/normal_walking"
ln -s "${MANUAL_ROOT}/mot_challenge/train/crossing" "${FLAT_DATASET}/crossing"
ln -s "${MANUAL_ROOT}/mot_challenge/train/partial_occ" "${FLAT_DATASET}/partial_occ"
ln -s "${MANUAL_ROOT}/mot_challenge/val/seam" "${FLAT_DATASET}/seam"
ln -s "${MANUAL_ROOT}/mot_challenge/test/full_occ" "${FLAT_DATASET}/full_occ"

export MPLCONFIGDIR="${RUN_ROOT}/.matplotlib"
export YOLO_CONFIG_DIR="${RUN_ROOT}/.ultralytics"
mkdir -p "${MPLCONFIGDIR}" "${YOLO_CONFIG_DIR}"
cd "${SCRIPT_DIR}"

echo "Run root: ${RUN_ROOT}"
echo "[1/6] Physical 512x120 YOLO11l + BoT-SORT/ReID tracking"
"${PYTHON_BIN}" track_people.py \
  --dataset-root "${FLAT_DATASET}" \
  --sequences normal_walking crossing partial_occ seam full_occ \
  --tracker botsort_reid \
  --weights "${SCRIPT_DIR}/yolo11l.pt" \
  --output-dir "${TRACKING_ROOT}" \
  --pre-resize 512x120 \
  --imgsz 512 \
  --conf 0.20 \
  --iou 0.70 \
  --fps 30 \
  --device "${DEVICE}" \
  --warmup-frames 3 \
  --run-name pre512x120

echo "[2/6] Build oracle switch events from physical-LR tracker predictions"
"${PYTHON_BIN}" -m hive_ecia.panorama_physical_lr_dataset \
  --config "${CONFIG}" \
  --prediction-dir "${TRACKING_ROOT}" \
  --output-root "${EVENT_ROOT}"

echo "[3/6] Verify physical-LR and leakage audit"
"${PYTHON_BIN}" -c "import json,pathlib; p=pathlib.Path('${EVENT_ROOT}/leakage_audit.json'); x=json.loads(p.read_text()); assert x['passed'] and x['sequence_disjoint'] and x['physical_lr_detection_tracking'] and x['physical_lr_size']==[512,120]; print('Physical-LR audit passed:', p)"

echo "[4/6] Frozen QuadTrack ECIA Base-only on all DTB physical-LR events"
"${PYTHON_BIN}" -m hive_ecia.panorama_evaluate \
  --config "${CONFIG}" \
  --event-root "${EVENT_ROOT}" \
  --output-dir "${BASE_OUTPUT}" \
  --all-sequences \
  --ablation base_only \
  --ecia-checkpoint "${SCRIPT_DIR}/outputs/hive_ecia_experiment/base_only/training/best.pt" \
  --ecia-thresholds "${SCRIPT_DIR}/outputs/hive_ecia_experiment/base_only/validation_thresholds.json" \
  --device "cuda:0"

echo "[5/6] Frozen QuadTrack ECIA Full on identical physical-LR events"
"${PYTHON_BIN}" -m hive_ecia.panorama_evaluate \
  --config "${CONFIG}" \
  --event-root "${EVENT_ROOT}" \
  --output-dir "${FULL_OUTPUT}" \
  --all-sequences \
  --ablation full \
  --ecia-checkpoint "${SCRIPT_DIR}/outputs/hive_ecia_experiment/full/training/best.pt" \
  --ecia-thresholds "${SCRIPT_DIR}/outputs/hive_ecia_experiment/full/validation_thresholds.json" \
  --device "cuda:0"

echo "[6/6] Aggregate metrics, per-video tables, figures, and report"
"${PYTHON_BIN}" -m hive_ecia.panorama_ablation_compare \
  --base-only "${BASE_OUTPUT}/metrics.json" \
  --full "${FULL_OUTPUT}/metrics.json" \
  --output-dir "${COMPARE_OUTPUT}"

echo "Completed without overwriting previous runs."
echo "Report: ${COMPARE_OUTPUT}/REPORT.md"
