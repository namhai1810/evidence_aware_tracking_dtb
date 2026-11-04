# Frozen HIVE–ECIA ablation

This module evaluates four existing inference paths without training or changing a checkpoint:

- **A0**: frozen cached YOLO11l + BoT-SORT reference.
- **A1**: `base_guided_hr_disabled`; ECIA Base acts on LR evidence and physical HR I/O is disabled.
- **A2**: `online`; the preserved rule-based ambiguity trigger invokes HIVE and ECIA Full.
- **A3**: `base_guided_safety_online`; ECIA Base-guided HR gating plus pre-query and final registry safety.

A2 is deliberately the pre-existing rule-trigger Full pipeline requested by the ablation. Its candidate generator is not the Base-guided A1/A3 generator. The report therefore checks A2/A3 output equality but never assumes it. This boundary prevents presenting A1→A2 as an HR-only causal comparison.

The official A0 detector caches and predictions and the official A3 frozen run are reused without modification. A1/A2 must run through native `model.track` because the cache stores detector outputs, not the active/lost lifecycle and association ambiguity consumed by the online controller. Their `raw_mot` is compared against cached A0 with exact frame/identity checks and a 0.011-pixel serialization tolerance. A mismatch blocks a valid report.

Bandwidth is recomputed as in-memory JPEG payload at quality 90: every physical LR frame plus every actually retrieved HIVE HR crop. It is not measured network throughput. Timing excludes decode, resize, JPEG accounting and file writes.

Run from the activated `hainn_stitching` environment:

```bash
cd /workspace/evidence_aware_tracking/code
ABLATION_DEVICE=0 ABLATION_RUN_ID=frozen_hive_ecia_v1 bash run_hive_ecia_ablation.sh
```

Use `ABLATION_PREFLIGHT_ONLY=1` to validate dependencies, paths and CLI contracts without inference.

