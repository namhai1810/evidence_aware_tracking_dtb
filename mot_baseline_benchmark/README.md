# Fair MOT baseline benchmark

This isolated module compares:

1. traditional SORT;
2. YOLO11l + standard Ultralytics ByteTrack;
3. YOLO11l + the project's unchanged BoT-SORT native-ReID configuration;
4. ByteTrack followed by frozen HIVE–ECIA (zero-shot tracker transfer);
5. the same BoT-SORT predictions followed by frozen Existing Base-Guided Safety.

It never imports an online-aware v1 or balanced-v2 checkpoint. The frozen ECIA
artifact is `outputs/hive_ecia_recovery_aware/20261030_osnet_ain_recovery_v1/training/best.pt`
with validation-selected probability/margin thresholds `0.80/0.40`.

## Fairness design

Raw YOLO detections and native `model=auto` features are captured immediately
before the Ultralytics tracking callback. SORT, ByteTrack and BoT-SORT replay
that immutable cache. A simultaneous live BoT-SORT result is audited against
cached replay: frame/ID/score must be exact and boxes must agree within 0.011
source pixel (the one-rounding-unit allowance covers OpenCV sparse-optical-flow
GMC replay). The untouched live `model.track` MOT is the official BoT-SORT
baseline. Existing Base-Guided Safety runs through the unchanged
`online_hive_ecia.run` entry point; its raw MOT output must be byte-identical to
that official live baseline.

The ByteTrack integration uses the same causal identity registry, Base-guided
policy, HIVE, ECIA, safety guards and frozen thresholds. ByteTrack exposes the
same active/lost lifecycle but no native appearance feature; candidate bounding
uses motion, while ECIA still encodes LR crops with frozen OSNet-AIN. This is
reported as zero-shot tracker transfer because no checkpoint or threshold is
retrained/tuned for ByteTrack.

Direct ByteTrack replay and the public `model.track` path may differ by one
decimal output unit (`0.01` source pixel) while retaining exact frame/ID/score.
The evaluator first verifies association equivalence, then combines baseline
ByteTrack geometry/confidence with HIVE–ECIA canonical IDs. The paired metric
delta therefore measures identity recovery only.

The script stops before HIVE–ECIA unless BoT-SORT reproduces the published
reference metrics. GT is loaded only by `evaluate.py`, after prediction files
are complete.

Detector confidence is fixed at `0.20` for all methods. Standard ByteTrack and
BoT-SORT keep `track_low_thresh=0.10`, but no detections below `0.20` exist in
the shared cache. This established-protocol limitation is reported explicitly;
the detector threshold is never changed for one tracker only.

## Commands

Inside the container:

```bash
cd /workspace/evidence_aware_tracking/code
MOT_BENCH_DEVICE=0 bash benchmark_sort_bytetrack_botsort.sh all
```

The launcher preserves the active `hainn_stitching` Python runtime used by the
historical BoT-SORT reference. A small `sitecustomize.py` bridge appends the
project venv at the end of `sys.path`, so a missing pure-Python `torchreid`
package can be resolved without replacing Conda's Ultralytics/Torch/OpenCV.
To select another complete environment explicitly, set
`MOT_BENCH_PYTHON=/path/to/python`.

Dataset-specific execution is also supported:

```bash
MOT_BENCH_DEVICE=0 MOT_BENCH_RUN_ID=fair_mot_quad_v1 \
  bash benchmark_sort_bytetrack_botsort.sh quadtrack

MOT_BENCH_DEVICE=0 MOT_BENCH_RUN_ID=fair_mot_dtb_v1 \
  bash benchmark_sort_bytetrack_botsort.sh dtb
```

Set `MOT_BENCH_PREFLIGHT_ONLY=1` to validate dependencies, paths and CLI
contracts without inference. Set `MOT_BENCH_RESUME=1` only to continue the same
run directory after interruption.

Per-dataset predictions are written below
`outputs/mot_baseline_benchmark/<run-id>/<dataset>/standard/mot/<method>/` and
`.../hive_ecia/mot/`. Detailed reports are under each dataset's `comparison/`.
An `all` run writes the requested consolidated report to
`outputs/experimental_result.md`.

