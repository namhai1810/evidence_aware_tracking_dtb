#!/usr/bin/env bash
set -euo pipefail

# Paired evaluation of the unchanged Base-Guided Full method and the new
# pre-query safety optimization. This script intentionally performs no train
# or threshold calibration.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${SAFETY_ECIA_PYTHON:-python}"
DEVICE="${SAFETY_ECIA_DEVICE:-0}"
RUN_ID="${SAFETY_ECIA_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${SAFETY_ECIA_OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/safety_aware_online/${RUN_ID}}"

QUAD_ROOT="${SAFETY_ECIA_QUAD_ROOT:-/workspace/evidence_aware_tracking/QuadTrack/train}"
DTB_ROOT="${SAFETY_ECIA_DTB_ROOT:-/workspace/evidence_aware_tracking/cvat_packages_trim300_last100/object_tracking_prepared_manual_dataset/mot_challenge}"
QUAD_SEQUENCES=(0016 0017 0025 0031)
DTB_SEQUENCES=(normal_walking crossing partial_occ seam full_occ)

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to overwrite: ${OUTPUT_ROOT}" >&2
  exit 1
fi

COMMON=(
  --config "${SCRIPT_DIR}/online_hive_ecia/config.yaml"
  --weights "${SCRIPT_DIR}/yolo11l.pt"
  --tracker-config "${SCRIPT_DIR}/botsort_reid.yaml"
  --conf 0.20 --iou 0.70 --fps 30
  --device "${DEVICE}" --warmup-frames 3
)

run_pair() {
  local dataset_name="$1"
  local dataset_root="$2"
  local pre_resize="$3"
  local imgsz="$4"
  shift 4
  local sequences=("$@")
  local dataset_output="${OUTPUT_ROOT}/${dataset_name}"
  local resolution_flag=()
  if [[ "${pre_resize}" != "512x120" ]]; then
    resolution_flag+=(--allow-resolution-mismatch)
  fi

  for mode in base_guided_online base_guided_safety_online; do
    "${PYTHON_BIN}" -m online_hive_ecia.run \
      "${COMMON[@]}" \
      --dataset-root "${dataset_root}" \
      --dataset-label "${dataset_name}" \
      --sequences "${sequences[@]}" \
      --pre-resize "${pre_resize}" --imgsz "${imgsz}" \
      "${resolution_flag[@]}" \
      --mode "${mode}" --output-dir "${dataset_output}/${mode}"
  done

  "${PYTHON_BIN}" -m online_hive_ecia.compare_safety_aware \
    --dataset-root "${dataset_root}" \
    --sequences "${sequences[@]}" \
    --baseline-run "${dataset_output}/base_guided_online" \
    --safety-run "${dataset_output}/base_guided_safety_online" \
    --output-dir "${dataset_output}/comparison" --iou 0.50
}

cd "${SCRIPT_DIR}"
run_pair "quadtrack_512x120" "${QUAD_ROOT}" "512x120" 512 "${QUAD_SEQUENCES[@]}"
run_pair "dtb_640x320" "${DTB_ROOT}" "640x320" 640 "${DTB_SEQUENCES[@]}"

"${PYTHON_BIN}" -m online_hive_ecia.safety_joint_report \
  --quadtrack "${OUTPUT_ROOT}/quadtrack_512x120/comparison/metrics.json" \
  --dtb "${OUTPUT_ROOT}/dtb_640x320/comparison/metrics.json" \
  --output "${OUTPUT_ROOT}/REPORT.md"

echo "Consolidated report: ${OUTPUT_ROOT}/REPORT.md"
