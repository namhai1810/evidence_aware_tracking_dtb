#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CHECKPOINT="${HIVE_CHECKPOINT:-}"
if [[ $# -gt 0 && "${1}" != --* ]]; then
  CHECKPOINT="$1"
  shift
fi
if [[ -z "${CHECKPOINT}" ]]; then
  echo "Usage: bash test.sh /path/to/run/checkpoints/best.pt [evaluate.py options]" >&2
  echo "Or set HIVE_CHECKPOINT=/path/to/best.pt" >&2
  exit 2
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 2
fi

CHECKPOINT="$(cd "$(dirname "${CHECKPOINT}")" && pwd)/$(basename "${CHECKPOINT}")"
RUN_DIR="$(dirname "$(dirname "${CHECKPOINT}")")"
CONFIG_PATH="${HIVE_CONFIG:-${RUN_DIR}/config.yaml}"
DATASET_ROOT="${HIVE_DATASET_ROOT:-}"
PROJECT_ROOT="${HIVE_PROJECT_ROOT:-}"
DEVICE="${HIVE_DEVICE:-cuda:0}"
PYTHON_BIN="${HIVE_PYTHON:-python}"
OUTPUT_DIR="${HIVE_TEST_OUTPUT:-${RUN_DIR}/test_results}"
NUM_WORKERS="${HIVE_TEST_WORKERS:-0}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Run config not found: ${CONFIG_PATH}" >&2
  exit 2
fi
if [[ -n "${HIVE_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${HIVE_CUDA_VISIBLE_DEVICES}"
fi

echo "HIVE held-out test"
echo "Checkpoint:       ${CHECKPOINT}"
echo "Config:           ${CONFIG_PATH}"
echo "Dataset root:     ${DATASET_ROOT:-<from config>}"
echo "Project root:     ${PROJECT_ROOT:-<from config>}"
echo "Device:           ${DEVICE}"
echo "Data workers:     ${NUM_WORKERS}"
echo "Output directory: ${OUTPUT_DIR}"

COMMAND=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate.py"
  --config "${CONFIG_PATH}"
  --checkpoint "${CHECKPOINT}"
  --split test
  --device "${DEVICE}"
  --num-workers "${NUM_WORKERS}"
  --output-dir "${OUTPUT_DIR}"
)
if [[ -n "${DATASET_ROOT}" ]]; then
  COMMAND+=(--dataset-root "${DATASET_ROOT}")
fi
if [[ -n "${PROJECT_ROOT}" ]]; then
  COMMAND+=(--project-root "${PROJECT_ROOT}")
fi

exec "${COMMAND[@]}" "$@"
