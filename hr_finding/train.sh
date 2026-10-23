#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${HIVE_CONFIG:-${SCRIPT_DIR}/configs/hive.yaml}"
DATASET_ROOT="${HIVE_DATASET_ROOT:-}"
PROJECT_ROOT="${HIVE_PROJECT_ROOT:-}"
DEVICE="${HIVE_DEVICE:-cuda:0}"
PYTHON_BIN="${HIVE_PYTHON:-python}"
RUN_NAME="${HIVE_RUN_NAME:-hive_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${HIVE_RUN_ROOT:-${SCRIPT_DIR}/runs}"
OUTPUT_DIR="${RUN_ROOT}/${RUN_NAME}"

if [[ -n "${HIVE_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${HIVE_CUDA_VISIBLE_DEVICES}"
fi

mkdir -p "${OUTPUT_DIR}"

echo "HIVE config:      ${CONFIG_PATH}"
echo "Dataset root:     ${DATASET_ROOT:-<from config>}"
echo "Project root:     ${PROJECT_ROOT:-<from config>}"
echo "Device:           ${DEVICE}"
echo "Run directory:    ${OUTPUT_DIR}"
echo "TensorBoard:      tensorboard --logdir ${RUN_ROOT} --port 6006"

COMMAND=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/train.py"
  --config "${CONFIG_PATH}"
  --device "${DEVICE}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${DATASET_ROOT}" ]]; then
  COMMAND+=(--dataset-root "${DATASET_ROOT}")
fi
if [[ -n "${PROJECT_ROOT}" ]]; then
  COMMAND+=(--project-root "${PROJECT_ROOT}")
fi

if [[ -n "${HIVE_RESUME:-}" ]]; then
  COMMAND+=(--resume "${HIVE_RESUME}")
fi

exec "${COMMAND[@]}" "$@"
