# HIVE–ECIA tracker-family transfer

This benchmark compares each association backend before and after attaching the frozen Existing Base-Guided Safety controller:

- SORT vs SORT + HIVE–ECIA
- ByteTrack vs ByteTrack + HIVE–ECIA
- OC-SORT vs OC-SORT + HIVE–ECIA
- BoT-SORT vs BoT-SORT + HIVE–ECIA

SORT and OC-SORT replay the exact YOLO11l cache already frozen by the fair MOT benchmark. Their raw MOT predictions and recovered canonical predictions are emitted in the same run, and an evaluator verifies that only identity fields change. ByteTrack and BoT-SORT reuse the already verified fair-benchmark artifacts.

The ECIA/HIVE checkpoints and Base-Guided thresholds remain those selected with the BoT-SORT/QuadTrack validation protocol. Applying them to SORT, OC-SORT and ByteTrack is explicitly a zero-shot tracker transfer; no test or DTB tuning is performed.

Run from the container:

```bash
cd /workspace/evidence_aware_tracking/code
TRACKER_TRANSFER_DEVICE=0 bash benchmark_tracker_family_hive_ecia.sh
```

The consolidated report is written to:

```text
/workspace/evidence_aware_tracking/code/outputs/tracker_family_transfer/sort_ocsort_v1/REPORT.md
```

