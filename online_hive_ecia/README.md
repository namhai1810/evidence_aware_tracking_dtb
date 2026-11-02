# Online HIVE--ECIA identity recovery

This package adds a causal identity layer around the existing DTB
`YOLO11l + BoT-SORT/ReID` loop. The original tracker and all frozen checkpoints
remain unchanged.

## Integration flow

```text
physical 512x120 panorama
  -> unchanged YOLO11l + BoT-SORT
  -> read current active/lost BoT-SORT state
  -> rule trigger: Lost->New + motion + native ReID + association margin
  -> HIVE v2 ranks only previously observed LR candidates
  -> read exactly one selected historical HR crop (online mode only)
  -> frozen OSNet-AIN + Recovery-Aware ECIA Merge/Abstain
  -> collision-safe canonical-ID transfer from this frame onward
  -> MOT prediction
```

No GT, evaluator match, oracle switch event, or future frame is available to
the runner. GT is loaded only by `online_hive_ecia.compare` after prediction
files have been completed.

## ECIA Base-guided HR trigger

The original `online` and `hr_disabled` rule-trigger modes above are preserved.
Two additional modes use the frozen Base branch of the same Recovery-Aware
checkpoint to decide when HR is useful:

```text
unchanged BoT-SORT association/lifecycle signals
  -> candidate pairs: Lost->New + plausible Active->Active conflict
  -> cached LR embeddings (old prototype, target, <=12 competitors)
  -> same frozen ECIA checkpoint: forward_base() exactly once
  -> Base confidence policy
       confident Old       -> merge without HR
       plausible/uncertain -> request HR
       competitor/abstain  -> keep IDs
  -> if requested: HIVE v2 -> exactly one historical HR crop
  -> same ECIA instance: forward_from_base() -> HR Gain + Gate
  -> Merge/Abstain -> existing canonical-ID safety guards
```

`base_guided_hr_disabled` and `base_guided_online` generate candidates from
stable raw-track tokens and log `would_request_hr`, so the evaluator requires
their proposed-event and HR-request schedules to be identical. The disabled
branch uses cached Base logits (equivalent to gate 0); the Full branch reuses
the same Base state and only computes HIVE, one HR embedding, HR Gain and Gate.
No second ECIA object or checkpoint is loaded.

The thresholds under `base_guided_trigger.policy` are explicitly marked for
QuadTrack-validation-only calibration. DTB and held-out test data must never be
used to select them. Trigger precision/recall is calculated post hoc from the
already-written raw MOT files and is never fed back to inference.

BoT-SORT raw IDs are not overwritten: they are internal keys for Kalman and
appearance state. A confirmed recovery updates a generation-safe canonical
identity alias. All later observations of the successor raw track emit the
recovered canonical ID. Safety checks reject active-old-ID conflicts and old
new tracks; duplicate active aliases receive a separate virtual ID. Raw ID
reuse after the configured gap creates a new generation.

The `hr_disabled` ablation uses the identical online trigger and HIVE LR
selection, skips the HR file read, and runs the same Recovery-Aware checkpoint
with `gate_override=0`.

Base-guided modes additionally log Base probability, margin, entropy, policy
action, pair type, cache hits/misses and actual/would-request HR cost in the
same output contract.

## Output contract

Each run writes:

- `mot/<sequence>.txt`: evaluated canonical MOT prediction;
- `raw_mot/<sequence>.txt`: untouched BoT-SORT prediction;
- `diagnostics/<sequence>.jsonl`: trigger, HIVE, ECIA, safety and collision log;
- `timing/<sequence>.json`: FPS, latency and HR retrieval cost;
- `run_manifest.json`: detector, resolution, thresholds and frozen checkpoints.

Comparison aborts unless every method has byte-identical `raw_mot` files, so a
reported difference cannot silently come from a different detector or tracker
run.

## Commands inside `hainn_stitching`

Syntax/path validation only (no model load or frame inference):

```bash
cd /workspace/evidence_aware_tracking/code
python -m online_hive_ecia.run \
  --config /workspace/evidence_aware_tracking/code/online_hive_ecia/config.yaml \
  --dataset-root /workspace/evidence_aware_tracking/cvat_packages_trim300_last100/object_tracking_prepared_manual_dataset/mot_challenge \
  --sequences full_occ --mode online \
  --output-dir /workspace/evidence_aware_tracking/code/outputs/online_hive_ecia/syntax_check \
  --weights /workspace/evidence_aware_tracking/code/yolo11l.pt \
  --tracker-config /workspace/evidence_aware_tracking/code/botsort_reid.yaml \
  --pre-resize 512x120 --imgsz 512 --conf 0.20 --iou 0.70 --fps 30 \
  --device 0 --validate-only
```

Ten-frame smoke test:

```bash
cd /workspace/evidence_aware_tracking/code
python -m online_hive_ecia.run \
  --config online_hive_ecia/config.yaml \
  --dataset-root /workspace/evidence_aware_tracking/cvat_packages_trim300_last100/object_tracking_prepared_manual_dataset/mot_challenge \
  --sequences full_occ --mode online \
  --output-dir outputs/online_hive_ecia/smoke_full_occ \
  --weights yolo11l.pt --tracker-config botsort_reid.yaml \
  --pre-resize 512x120 --imgsz 512 --conf 0.20 --iou 0.70 --fps 30 \
  --device 0 --warmup-frames 3 --max-frames 10
```

Full paired evaluation (baseline, HR-disabled, online, metrics and report):

```bash
cd /workspace/evidence_aware_tracking/code
ONLINE_ECIA_DEVICE=0 \
ONLINE_ECIA_RUN_ID=dtb_online_v1 \
bash run_online_hive_ecia_evaluation.sh
```

The report is written to
`outputs/online_hive_ecia/dtb_online_v1/comparison/REPORT.md`.

### Physical 640x320 zero-shot run

HIVE was trained with a 512x120 physical-LR reference. A 640x320 run is
therefore marked explicitly as resolution zero-shot; the detector, tracker and
all three comparison branches still receive the same 640x320 frames.

```bash
cd /workspace/evidence_aware_tracking/code
ONLINE_ECIA_DEVICE=0 \
ONLINE_ECIA_PRE_RESIZE=640x320 \
ONLINE_ECIA_IMGSZ=640 \
ONLINE_ECIA_RUN_ID=dtb_online_640x320_v1 \
bash run_online_hive_ecia_evaluation.sh
```

`comparison/REPORT.md` additionally reports exact ambiguity-trigger precision,
recall and F1 against post-hoc raw-BoT-SORT ID-switch events. GT is never loaded
by the online runner and cannot affect trigger, HIVE, ECIA or identity updates.

### QuadTrack test + DTB in one command

The joint runner evaluates only the frozen QuadTrack held-out test sequences
`0016, 0017, 0025, 0031`, then all five manually annotated DTB sequences. It
creates two detailed reports and one combined summary.

```bash
cd /workspace/evidence_aware_tracking/code
ONLINE_ECIA_DEVICE=0 \
ONLINE_ECIA_PRE_RESIZE=640x320 \
ONLINE_ECIA_IMGSZ=640 \
ONLINE_ECIA_RUN_ID=quadtrack_dtb_640x320_v1 \
bash run_online_hive_ecia_quadtrack_dtb.sh
```

Combined report:

`outputs/online_hive_ecia_joint/quadtrack_dtb_640x320_v1/REPORT.md`.

## Base-guided commands

Syntax/path check only (no model is loaded and no frame is processed):

```bash
cd /workspace/evidence_aware_tracking/code
python -m online_hive_ecia.run \
  --config online_hive_ecia/config.yaml \
  --dataset-root /workspace/evidence_aware_tracking/cvat_packages_trim300_last100/object_tracking_prepared_manual_dataset/mot_challenge \
  --dataset-label dtb_smoke --sequences full_occ \
  --mode base_guided_online \
  --output-dir outputs/base_guided_online/syntax_check \
  --weights yolo11l.pt --tracker-config botsort_reid.yaml \
  --pre-resize 640x320 --allow-resolution-mismatch \
  --imgsz 640 --device 0 --validate-only
```

Twenty-frame DTB smoke test:

```bash
cd /workspace/evidence_aware_tracking/code
python -m online_hive_ecia.run \
  --config online_hive_ecia/config.yaml \
  --dataset-root /workspace/evidence_aware_tracking/cvat_packages_trim300_last100/object_tracking_prepared_manual_dataset/mot_challenge \
  --dataset-label dtb_smoke --sequences full_occ \
  --mode base_guided_online \
  --output-dir outputs/base_guided_online/smoke_full_occ \
  --weights yolo11l.pt --tracker-config botsort_reid.yaml \
  --pre-resize 640x320 --allow-resolution-mismatch \
  --imgsz 640 --conf 0.20 --iou 0.70 --fps 30 \
  --device 0 --warmup-frames 3 --max-frames 20
```

Full paired QuadTrack held-out test:

```bash
cd /workspace/evidence_aware_tracking/code
ONLINE_ECIA_DEVICE=0 \
ONLINE_ECIA_PRE_RESIZE=512x120 ONLINE_ECIA_IMGSZ=512 \
ONLINE_ECIA_RUN_ID=quadtrack_base_guided_v1 \
ONLINE_ECIA_DATASET_ROOT=/workspace/evidence_aware_tracking/QuadTrack/train \
ONLINE_ECIA_DATASET_LABEL=quadtrack_heldout_test \
ONLINE_ECIA_SEQUENCES="0016 0017 0025 0031" \
bash run_base_guided_online_evaluation.sh
```

Full paired DTB zero-shot evaluation at physical LR 640x320:

```bash
cd /workspace/evidence_aware_tracking/code
ONLINE_ECIA_DEVICE=0 \
ONLINE_ECIA_PRE_RESIZE=640x320 ONLINE_ECIA_IMGSZ=640 \
ONLINE_ECIA_RUN_ID=dtb_base_guided_640x320_v1 \
ONLINE_ECIA_DATASET_ROOT=/workspace/evidence_aware_tracking/cvat_packages_trim300_last100/object_tracking_prepared_manual_dataset/mot_challenge \
ONLINE_ECIA_DATASET_LABEL=dtb_all \
ONLINE_ECIA_SEQUENCES="normal_walking crossing partial_occ seam full_occ" \
bash run_base_guided_online_evaluation.sh
```

Each full run creates `online`, `base_guided_hr_disabled`,
`base_guided_online`, `comparison/metrics.json`, and
`comparison/REPORT.md`. The comparison contains IDSW, IDF1, HOTA, AssA, MOTA,
FPS, latency, trigger precision/recall, HR requests/crops/bytes and per-video
tracking metrics.

If a report/evaluation stage fails after one or more inference branches have
completed, resume the same run without repeating completed methods:

```bash
ONLINE_ECIA_RESUME=1 \
ONLINE_ECIA_DEVICE=0 \
ONLINE_ECIA_PRE_RESIZE=640x320 \
ONLINE_ECIA_IMGSZ=640 \
ONLINE_ECIA_RUN_ID=quadtrack_dtb_640x320_v1 \
bash run_online_hive_ecia_quadtrack_dtb.sh
```
