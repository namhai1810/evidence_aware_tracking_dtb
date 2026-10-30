#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TAG="${REID_BENCHMARK_TAG:-20261030_lr_hr_reid_v1}"
DEVICE="${REID_DEVICE:-cuda:0}"

cd "${CODE_ROOT}"

python -m hive_ecia.benchmark_reid_resolution \
  --run "512x120=${CODE_ROOT}/outputs/dtb_physical_lr512_ecia/20261030_physical_lr512_v1" \
  --run "512x256=${CODE_ROOT}/outputs/dtb_physical_lr_experiments/512x256/20261030_resolution_sweep_v1" \
  --run "640x320=${CODE_ROOT}/outputs/dtb_physical_lr_experiments/640x320/20261030_resolution_sweep_v1" \
  --device "${DEVICE}" \
  --batch-size "${REID_BATCH_SIZE:-128}" \
  --output-dir "${CODE_ROOT}/outputs/dtb_reid_resolution_benchmark/${RUN_TAG}"
