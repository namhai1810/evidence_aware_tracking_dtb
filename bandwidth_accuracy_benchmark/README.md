# Bandwidth–Accuracy Benchmark

This isolated benchmark compares three acquisition policies under a common
MOT protocol:

- **Full-HR:** JPEG-transmit and infer on every original panorama.
- **Periodic-HR:** use an HR panorama every `N=5,10,30` frames and the
  established physical LR panorama otherwise.
- **HIVE–ECIA:** continuously transmit LR panoramas and transmit exactly the
  historical HR person crop requested by frozen Existing Base-Guided Safety.

Full/Periodic run YOLO11l and SORT, ByteTrack and BoT-SORT. Detections are
mapped to original panorama/GT coordinates before association, and tracker
state is never reset at an LR/HR transition. On LR steps, GMC sees only the
transmitted LR pixels enlarged to the common coordinate canvas; it never sees
the untransmitted original HR content. HR YOLO uses `imgsz=2048`; LR
uses the established dataset `imgsz`. HIVE–ECIA reuses the frozen predictions
from `fair_mot_conda_v1`, so its MOT metrics must match the established
reference before a report is produced.

Bandwidth is the size of actual in-memory JPEG encodings at quality 90. It is
a transmitted-data model, not measured network throughput. Caching is not
modeled: every repeated HIVE crop request is encoded and counted again.

## Run

Inside the NerdCTL container with `hainn_stitching` activated:

```bash
cd /workspace/evidence_aware_tracking/code
BANDWIDTH_DEVICE=0 BANDWIDTH_RUN_ID=bandwidth_accuracy_v1 \
  bash benchmark_bandwidth_accuracy.sh
```

Resume a partially completed run with `BANDWIDTH_RESUME=1`. Use
`BANDWIDTH_PREFLIGHT_ONLY=1` to validate dependencies, paths and both CLIs
without detector inference.

The joint report is written below the run output, and the same marked section
is inserted into `outputs/experimental_result.md` without removing the
existing fair MOT benchmark.

