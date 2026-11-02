#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ONLINE_ECIA_PYTHON:-$(command -v python)}"
DEVICE="${ONLINE_ECIA_DEVICE:-0}"
PRE_RESIZE="${ONLINE_ECIA_PRE_RESIZE:-512x120}"
IMGSZ="${ONLINE_ECIA_IMGSZ:-${PRE_RESIZE%%x*}}"
RUN_ID="${ONLINE_ECIA_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${ONLINE_ECIA_OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/base_guided_online/${RUN_ID}}"
DATASET_ROOT="${ONLINE_ECIA_DATASET_ROOT:?Set ONLINE_ECIA_DATASET_ROOT}"
DATASET_LABEL="${ONLINE_ECIA_DATASET_LABEL:-dataset}"
read -r -a SEQUENCES <<< "${ONLINE_ECIA_SEQUENCES:?Set ONLINE_ECIA_SEQUENCES}"

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to overwrite: ${OUTPUT_ROOT}" >&2
  exit 1
fi

COMMON=(
  --config "${SCRIPT_DIR}/online_hive_ecia/config.yaml"
  --dataset-root "${DATASET_ROOT}"
  --dataset-label "${DATASET_LABEL}"
  --sequences "${SEQUENCES[@]}"
  --weights "${SCRIPT_DIR}/yolo11l.pt"
  --tracker-config "${SCRIPT_DIR}/botsort_reid.yaml"
  --pre-resize "${PRE_RESIZE}"
  --imgsz "${IMGSZ}"
  --conf 0.20 --iou 0.70 --fps 30
  --device "${DEVICE}" --warmup-frames 3
)
if [[ "${PRE_RESIZE}" != "512x120" ]]; then
  COMMON+=(--allow-resolution-mismatch)
fi

cd "${SCRIPT_DIR}"
for mode in online base_guided_hr_disabled base_guided_online; do
  "${PYTHON_BIN}" -m online_hive_ecia.run "${COMMON[@]}" \
    --mode "${mode}" --output-dir "${OUTPUT_ROOT}/${mode}"
done

"${PYTHON_BIN}" -m online_hive_ecia.compare_base_guided \
  --dataset-root "${DATASET_ROOT}" \
  --sequences "${SEQUENCES[@]}" \
  --rule-run "${OUTPUT_ROOT}/online" \
  --base-hr-disabled-run "${OUTPUT_ROOT}/base_guided_hr_disabled" \
  --base-full-run "${OUTPUT_ROOT}/base_guided_online" \
  --output-dir "${OUTPUT_ROOT}/comparison" --iou 0.50

echo "Report: ${OUTPUT_ROOT}/comparison/REPORT.md"
