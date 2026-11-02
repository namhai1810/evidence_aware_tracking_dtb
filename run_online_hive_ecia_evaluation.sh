#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ONLINE_ECIA_PYTHON:-$(command -v python)}"
DEVICE="${ONLINE_ECIA_DEVICE:-0}"
PRE_RESIZE="${ONLINE_ECIA_PRE_RESIZE:-512x120}"
IMGSZ="${ONLINE_ECIA_IMGSZ:-${PRE_RESIZE%%x*}}"
RUN_ID="${ONLINE_ECIA_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${ONLINE_ECIA_OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/online_hive_ecia/${RUN_ID}}"
DATASET_ROOT="${ONLINE_ECIA_DATASET_ROOT:-${SCRIPT_DIR}/../cvat_packages_trim300_last100/object_tracking_prepared_manual_dataset/mot_challenge}"
DATASET_LABEL="${ONLINE_ECIA_DATASET_LABEL:-dtb_all}"
RESUME="${ONLINE_ECIA_RESUME:-0}"
read -r -a SEQUENCES <<< "${ONLINE_ECIA_SEQUENCES:-normal_walking crossing partial_occ seam full_occ}"
if [[ "${#SEQUENCES[@]}" -eq 0 ]]; then
  echo "ONLINE_ECIA_SEQUENCES must contain at least one sequence" >&2
  exit 2
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
  --conf 0.20
  --iou 0.70
  --fps 30
  --device "${DEVICE}"
  --warmup-frames 3
)
if [[ "${PRE_RESIZE}" != "512x120" ]]; then
  COMMON+=(--allow-resolution-mismatch)
fi

if [[ -e "${OUTPUT_ROOT}" && "${RESUME}" != "1" ]]; then
  echo "Refusing to overwrite: ${OUTPUT_ROOT}" >&2
  echo "Set ONLINE_ECIA_RESUME=1 to reuse completed method runs." >&2
  exit 1
fi

cd "${SCRIPT_DIR}"
run_method() {
  local mode="$1"
  local method_dir="${OUTPUT_ROOT}/${mode}"
  local complete=1
  [[ -f "${method_dir}/run_manifest.json" ]] || complete=0
  for sequence in "${SEQUENCES[@]}"; do
    [[ -f "${method_dir}/mot/${sequence}.txt" ]] || complete=0
    [[ -f "${method_dir}/raw_mot/${sequence}.txt" ]] || complete=0
    [[ -f "${method_dir}/diagnostics/${sequence}.jsonl" ]] || complete=0
    [[ -f "${method_dir}/timing/${sequence}.json" ]] || complete=0
  done
  if [[ "${RESUME}" == "1" && "${complete}" == "1" ]]; then
    echo "Resume: reuse completed ${mode}: ${method_dir}"
    return
  fi
  if [[ -e "${method_dir}" && "${RESUME}" != "1" ]]; then
    echo "Partial method directory cannot be resumed safely: ${method_dir}" >&2
    echo "Choose a new ONLINE_ECIA_RUN_ID." >&2
    exit 1
  fi
  local resume_args=()
  if [[ "${RESUME}" == "1" ]]; then
    resume_args+=(--resume)
  fi
  "${PYTHON_BIN}" -m online_hive_ecia.run "${COMMON[@]}" \
    --mode "${mode}" --output-dir "${method_dir}" "${resume_args[@]}"
}

run_method baseline
run_method hr_disabled
run_method online

if [[ "${RESUME}" == "1" && -f "${OUTPUT_ROOT}/comparison/metrics.json" ]]; then
  echo "Resume: reuse completed comparison: ${OUTPUT_ROOT}/comparison"
else
  if [[ -e "${OUTPUT_ROOT}/comparison" ]]; then
    echo "Partial comparison directory exists: ${OUTPUT_ROOT}/comparison" >&2
    exit 1
  fi
  "${PYTHON_BIN}" -m online_hive_ecia.compare \
    --dataset-root "${DATASET_ROOT}" \
    --sequences "${SEQUENCES[@]}" \
    --baseline-run "${OUTPUT_ROOT}/baseline" \
    --hr-disabled-run "${OUTPUT_ROOT}/hr_disabled" \
    --online-run "${OUTPUT_ROOT}/online" \
    --output-dir "${OUTPUT_ROOT}/comparison" \
    --iou 0.50
fi

echo "Report: ${OUTPUT_ROOT}/comparison/REPORT.md"
