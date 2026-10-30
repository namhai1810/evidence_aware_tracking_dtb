#!/usr/bin/env bash
set -euo pipefail

# Retrain ECIA Base-only and Full with two frozen OSNet encoders, then evaluate
# on held-out QuadTrack and strict zero-shot DTB full_occ at three resolutions.

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ECIA_PYTHON_BIN:-$(command -v python)}"
DEVICE="${ECIA_DEVICE:-cuda:0}"
RUN_TAG="${ECIA_REID_RUN_TAG:-20261030_osnet_retrain_v1}"
RUN_ROOT="${ECIA_REID_OUTPUT_ROOT:-${CODE_ROOT}/outputs/hive_ecia_reid_retrain/${RUN_TAG}}"
REUSE_CACHE="${ECIA_REID_REUSE_CACHE:-0}"
BASE_CONFIG="${CODE_ROOT}/hive_ecia/config.yaml"
PANORAMA_CONFIG="${CODE_ROOT}/hive_ecia/panorama_config.yaml"

ENCODERS=(osnet_market osnet_ain_dmc)
ABLATIONS=(base_only full)

declare -A REID_KIND=(
  [osnet_market]="osnet_x1_0"
  [osnet_ain_dmc]="osnet_ain_x1_0"
)
declare -A REID_MODEL=(
  [osnet_market]="${CODE_ROOT}/models/reid_benchmark/osnet_x1_0_market.pth"
  [osnet_ain_dmc]="${CODE_ROOT}/models/reid_benchmark/osnet_ain_x1_0_d_m_c.pth"
)
declare -A DTB_EVENT_ROOT=(
  [512x120]="${CODE_ROOT}/outputs/dtb_physical_lr512_ecia/20261030_physical_lr512_v1/event_protocol"
  [512x256]="${CODE_ROOT}/outputs/dtb_physical_lr_experiments/512x256/20261030_resolution_sweep_v1/event_protocol"
  [640x320]="${CODE_ROOT}/outputs/dtb_physical_lr_experiments/640x320/20261030_resolution_sweep_v1/event_protocol"
)

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ -e "${RUN_ROOT}" && "${REUSE_CACHE}" != "1" ]]; then
  echo "Run root exists; refusing to overwrite: ${RUN_ROOT}" >&2
  exit 1
fi
if [[ ! -e "${RUN_ROOT}" && "${REUSE_CACHE}" == "1" ]]; then
  echo "Cannot reuse missing run root: ${RUN_ROOT}" >&2
  exit 1
fi
for encoder in "${ENCODERS[@]}"; do
  [[ -f "${REID_MODEL[$encoder]}" ]] || { echo "Missing ${REID_MODEL[$encoder]}" >&2; exit 1; }
done
for resolution in "${!DTB_EVENT_ROOT[@]}"; do
  [[ -d "${DTB_EVENT_ROOT[$resolution]}" ]] || { echo "Missing ${DTB_EVENT_ROOT[$resolution]}" >&2; exit 1; }
done

mkdir -p "${RUN_ROOT}/logs"
export MPLCONFIGDIR="${RUN_ROOT}/.matplotlib"
mkdir -p "${MPLCONFIGDIR}"
cd "${CODE_ROOT}"

echo "Run root: ${RUN_ROOT}"
echo "Device: ${DEVICE}"
if [[ "${REUSE_CACHE}" != "1" ]]; then
  "${PYTHON_BIN}" -m hive_ecia.make_reid_retrain_configs \
    --base-config "${BASE_CONFIG}" \
    --output-root "${RUN_ROOT}"
else
  echo "Reusing audited encoder caches and generated configs from ${RUN_ROOT}"
fi

wait_named_jobs() {
  local stage="$1"
  shift
  local failed=0
  for entry in "$@"; do
    local name="${entry%%:*}"
    local pid="${entry##*:}"
    if wait "${pid}"; then
      echo "[${stage}] completed: ${name}"
    else
      echo "[${stage}] FAILED: ${name}" >&2
      failed=1
    fi
  done
  [[ "${failed}" -eq 0 ]]
}

if [[ "${REUSE_CACHE}" != "1" ]]; then
echo "[1/8] Build two encoder-specific frozen caches in parallel"
  cache_jobs=()
  for encoder in "${ENCODERS[@]}"; do
    config="${RUN_ROOT}/${encoder}/config.yaml"
    (
      "${PYTHON_BIN}" -m hive_ecia.prepare_data \
        --config "${config}" \
        --device "${DEVICE}"
    ) >"${RUN_ROOT}/logs/cache_${encoder}.log" 2>&1 &
    cache_jobs+=("${encoder}:$!")
  done
  wait_named_jobs cache "${cache_jobs[@]}"
else
  echo "[1/8] Reuse existing encoder-specific frozen caches"
fi

echo "[2/8] Verify split isolation and checkpoint-specific cache audits"
"${PYTHON_BIN}" - "${RUN_ROOT}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
hashes = set()
for encoder in ("osnet_market", "osnet_ain_dmc"):
    p = root / encoder / "cache" / "cache_audit.json"
    x = json.loads(p.read_text())
    groups = [set(x["train"]["sequences"]), set(x["validation"]["sequences"]), set(x["configured_calibration_sequences"]), set(x["configured_test_sequences"])]
    assert x["calibration_sequences_accessed"] is False
    assert x["test_sequences_accessed"] is False
    assert all(groups[i].isdisjoint(groups[j]) for i in range(len(groups)) for j in range(i + 1, len(groups)))
    assert x["reid_load_audit"]["backbone_keys_loaded"] in {"565/565", "550/550"}
    hashes.add(x["reid_model_sha256"])
assert len(hashes) == 2
print("Leakage and encoder cache audits passed")
PY

echo "[3/8] Train four ECIA models in parallel"
train_jobs=()
for encoder in "${ENCODERS[@]}"; do
  for ablation in "${ABLATIONS[@]}"; do
    run_dir="${RUN_ROOT}/${encoder}/quadtrack/${ablation}"
    mkdir -p "${run_dir}"
    (
      "${PYTHON_BIN}" -m hive_ecia.train \
        --config "${RUN_ROOT}/${encoder}/config.yaml" \
        --ablation "${ablation}" \
        --device "${DEVICE}" \
        --output-dir "${run_dir}/training"
    ) >"${RUN_ROOT}/logs/train_${encoder}_${ablation}.log" 2>&1 &
    train_jobs+=("${encoder}_${ablation}:$!")
  done
done
wait_named_jobs train "${train_jobs[@]}"

echo "[4/8] Calibrate all four frozen checkpoint/threshold pairs on QuadTrack 0030"
calibration_jobs=()
for encoder in "${ENCODERS[@]}"; do
  (
    for ablation in "${ABLATIONS[@]}"; do
      run_dir="${RUN_ROOT}/${encoder}/quadtrack/${ablation}"
      "${PYTHON_BIN}" -m hive_ecia.inference \
        --config "${RUN_ROOT}/${encoder}/config.yaml" \
        --ecia-checkpoint "${run_dir}/training/best.pt" \
        --split validation \
        --sequences 0030 \
        --device "${DEVICE}" \
        --min-old-probability 0.0 \
        --min-old-margin -1.0 \
        --output "${run_dir}/validation_scores.jsonl"
      "${PYTHON_BIN}" -m hive_ecia.calibrate \
        --config "${RUN_ROOT}/${encoder}/config.yaml" \
        --validation-decisions "${run_dir}/validation_scores.jsonl" \
        --validation-events "${CODE_ROOT}/outputs/hive_history_ranking_v2/train_data/events.jsonl" \
        --output "${run_dir}/validation_thresholds.json"
    done
  ) >"${RUN_ROOT}/logs/calibrate_${encoder}.log" 2>&1 &
  calibration_jobs+=("${encoder}:$!")
done
wait_named_jobs calibration "${calibration_jobs[@]}"

echo "[5/8] Open held-out QuadTrack test only after all checkpoints and thresholds are frozen"
quad_jobs=()
for encoder in "${ENCODERS[@]}"; do
  (
    for ablation in "${ABLATIONS[@]}"; do
      run_dir="${RUN_ROOT}/${encoder}/quadtrack/${ablation}"
      "${PYTHON_BIN}" -m hive_ecia.inference \
        --config "${RUN_ROOT}/${encoder}/config.yaml" \
        --ecia-checkpoint "${run_dir}/training/best.pt" \
        --split test \
        --device "${DEVICE}" \
        --thresholds-json "${run_dir}/validation_thresholds.json" \
        --output "${run_dir}/test_decisions.jsonl"
      "${PYTHON_BIN}" -m hive_ecia.evaluate \
        --decisions "${run_dir}/test_decisions.jsonl" \
        --evaluation-events "${CODE_ROOT}/outputs/hive_history_ranking_v2/test_data/events.jsonl" \
        --output-dir "${run_dir}/test_evaluation"
    done
  ) >"${RUN_ROOT}/logs/quad_test_${encoder}.log" 2>&1 &
  quad_jobs+=("${encoder}:$!")
done
wait_named_jobs quad_test "${quad_jobs[@]}"

echo "[6/8] Strict zero-shot DTB full_occ tests at three physical resolutions"
dtb_jobs=()
for encoder in "${ENCODERS[@]}"; do
  (
    for resolution in 512x120 512x256 640x320; do
      for ablation in "${ABLATIONS[@]}"; do
        quad_dir="${RUN_ROOT}/${encoder}/quadtrack/${ablation}"
        output_dir="${RUN_ROOT}/${encoder}/dtb/${resolution}/${ablation}"
        "${PYTHON_BIN}" -m hive_ecia.panorama_evaluate \
          --config "${PANORAMA_CONFIG}" \
          --event-root "${DTB_EVENT_ROOT[$resolution]}" \
          --output-dir "${output_dir}" \
          --ablation "${ablation}" \
          --ecia-checkpoint "${quad_dir}/training/best.pt" \
          --ecia-thresholds "${quad_dir}/validation_thresholds.json" \
          --reid-encoder "${REID_KIND[$encoder]}" \
          --reid-model "${REID_MODEL[$encoder]}" \
          --device "${DEVICE}"
      done
    done
  ) >"${RUN_ROOT}/logs/dtb_test_${encoder}.log" 2>&1 &
  dtb_jobs+=("${encoder}:$!")
done
wait_named_jobs dtb_test "${dtb_jobs[@]}"

echo "[7/8] Re-evaluate current YOLO26n-ReID reference with corrected physical crop sizes"
for resolution in 512x120 512x256 640x320; do
  for ablation in "${ABLATIONS[@]}"; do
    "${PYTHON_BIN}" -m hive_ecia.panorama_evaluate \
      --config "${PANORAMA_CONFIG}" \
      --event-root "${DTB_EVENT_ROOT[$resolution]}" \
      --output-dir "${RUN_ROOT}/reference_yolo26n/dtb/${resolution}/${ablation}" \
      --ablation "${ablation}" \
      --ecia-checkpoint "${CODE_ROOT}/outputs/hive_ecia_experiment/${ablation}/training/best.pt" \
      --ecia-thresholds "${CODE_ROOT}/outputs/hive_ecia_experiment/${ablation}/validation_thresholds.json" \
      --reid-encoder onnx \
      --reid-model "${CODE_ROOT}/yolo26n-reid.onnx" \
      --device "${DEVICE}"
  done
done

echo "[8/8] Create current-vs-OSNet QuadTrack + DTB report and figures"
"${PYTHON_BIN}" -m hive_ecia.compare_reid_retraining \
  --run-root "${RUN_ROOT}" \
  --output-dir "${RUN_ROOT}/final_report_v2"

echo "Completed: ${RUN_ROOT}/final_report_v2/REPORT.md"
