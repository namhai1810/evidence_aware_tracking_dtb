#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ONLINE_ECIA_PYTHON:-$(command -v python)}"
DEVICE="${ONLINE_ECIA_DEVICE:-0}"
PRE_RESIZE="${ONLINE_ECIA_PRE_RESIZE:-640x320}"
IMGSZ="${ONLINE_ECIA_IMGSZ:-${PRE_RESIZE%%x*}}"
RUN_ID="${ONLINE_ECIA_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="${ONLINE_ECIA_OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/online_hive_ecia_joint/${RUN_ID}}"
RESUME="${ONLINE_ECIA_RESUME:-0}"

if [[ -e "${ROOT}" && "${RESUME}" != "1" ]]; then
  echo "Refusing to overwrite: ${ROOT}" >&2
  echo "Set ONLINE_ECIA_RESUME=1 to reuse completed stages." >&2
  exit 1
fi
mkdir -p "${ROOT}"

echo "[1/3] QuadTrack held-out test: 0016 0017 0025 0031"
ONLINE_ECIA_DEVICE="${DEVICE}" \
ONLINE_ECIA_PRE_RESIZE="${PRE_RESIZE}" \
ONLINE_ECIA_IMGSZ="${IMGSZ}" \
ONLINE_ECIA_RUN_ID="quadtrack" \
ONLINE_ECIA_OUTPUT_ROOT="${ROOT}/quadtrack" \
ONLINE_ECIA_DATASET_ROOT="${SCRIPT_DIR}/../QuadTrack/train" \
ONLINE_ECIA_DATASET_LABEL="quadtrack_heldout_test" \
ONLINE_ECIA_SEQUENCES="0016 0017 0025 0031" \
ONLINE_ECIA_RESUME="${RESUME}" \
bash "${SCRIPT_DIR}/run_online_hive_ecia_evaluation.sh"

echo "[2/3] DTB all five manually annotated sequences"
ONLINE_ECIA_DEVICE="${DEVICE}" \
ONLINE_ECIA_PRE_RESIZE="${PRE_RESIZE}" \
ONLINE_ECIA_IMGSZ="${IMGSZ}" \
ONLINE_ECIA_RUN_ID="dtb" \
ONLINE_ECIA_OUTPUT_ROOT="${ROOT}/dtb" \
ONLINE_ECIA_DATASET_ROOT="${SCRIPT_DIR}/../cvat_packages_trim300_last100/object_tracking_prepared_manual_dataset/mot_challenge" \
ONLINE_ECIA_DATASET_LABEL="dtb_all" \
ONLINE_ECIA_SEQUENCES="normal_walking crossing partial_occ seam full_occ" \
ONLINE_ECIA_RESUME="${RESUME}" \
bash "${SCRIPT_DIR}/run_online_hive_ecia_evaluation.sh"

echo "[3/3] Joint report"
cd "${SCRIPT_DIR}"
"${PYTHON_BIN}" -m online_hive_ecia.joint_report \
  --quadtrack "${ROOT}/quadtrack/comparison/metrics.json" \
  --dtb "${ROOT}/dtb/comparison/metrics.json" \
  --output "${ROOT}/REPORT.md"

echo "Combined report: ${ROOT}/REPORT.md"
