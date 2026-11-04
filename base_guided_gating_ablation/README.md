# Controlled Base-Guided Gating Ablation

- **G0 — Gating OFF:** the frozen Base-Guided candidate generator proposes the same pairs, pre-query safety rejects infeasible merges, and every remaining candidate requests one HIVE-selected HR crop before ECIA Full.
- **G1 — Gating ON:** the unchanged frozen Existing Base-Guided Safety reference requests HR only when ECIA Base marks it necessary.

G0 is a new selectable inference mode; all original modes and checkpoints remain unchanged. The evaluator requires identical candidate pools and equivalent raw BoT-SORT predictions, audits that safety-rejected events never request HR, and reports whether final canonical MOT predictions remain identical.

Run inside the activated `hainn_stitching` container:

```bash
cd /workspace/evidence_aware_tracking/code
GATING_DEVICE=0 GATING_RUN_ID=base_guided_gating_v1 bash run_base_guided_gating_ablation.sh
```

The consolidated report is written to:

```text
/workspace/evidence_aware_tracking/code/outputs/base_guided_gating_ablation/base_guided_gating_v1/REPORT.md
```

Set `GATING_PREFLIGHT_ONLY=1` to validate dependencies, frozen artifacts and CLI paths without inference.

