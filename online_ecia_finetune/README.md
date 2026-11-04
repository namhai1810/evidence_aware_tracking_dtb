# Online-Aware ECIA fine-tuning

This package adapts the existing Recovery-Aware ECIA to candidates observed in
the causal BoT-SORT pipeline. HIVE v2 and OSNet-AIN D+M+C remain frozen. The
original checkpoint, online modes, and reports are never overwritten.

## Leakage-safe flow

```text
QuadTrack train + model-validation + calibration videos
  -> causal BoT-SORT candidate collection without GT
  -> frozen OSNet-AIN LR embeddings and at most one HIVE-selected HR embedding
  -> post-hoc GT join into class labels only
  -> video-disjoint cache audit
  -> initialize from Recovery-Aware ECIA
  -> fine-tune ECIA projection/Base/HR Gain/Gate/Abstain
  -> select checkpoint on 0019/0028
  -> calibrate trigger and merge thresholds on 0030
  -> freeze checkpoint/config
  -> full-video QuadTrack test and DTB zero-shot evaluation
```

The collector never opens GT. Labels are created only after raw MOT and frozen
embedding caches are complete. QuadTrack test videos and all DTB videos are
excluded from collection, fitting, checkpoint selection, and calibration.

Candidate types include Lost-to-New positives, competitor/hard-negative
hypotheses, ordinary negatives, and abstain/no-merge cases. Active-to-Active
candidates are collected, but the current registry rejects them while the old
identity is active. They are therefore explicitly labeled non-executable and
never presented as valid corrections. This preserves current online safety
semantics instead of fabricating unsupported positive merges.

## Train and calibrate

Inside the `hainn_stitching` container:

```bash
cd /workspace/evidence_aware_tracking/code
ONLINE_ECIA_DEVICE=0 \
ONLINE_ECIA_RUN_ID=online_aware_v1 \
bash train_online_ecia_quadtrack.sh
```

Outputs:

- checkpoint: `outputs/online_ecia_finetune/online_aware_v1/training/best.pt`;
- calibrated thresholds: `outputs/online_ecia_finetune/online_aware_v1/calibration/thresholds.json`;
- frozen inference config: `outputs/online_ecia_finetune/online_aware_v1/inference_config.yaml`;
- leakage audit: `outputs/online_ecia_finetune/online_aware_v1/leakage_audit.json`.

## Full evaluation

```bash
cd /workspace/evidence_aware_tracking/code
ONLINE_ECIA_DEVICE=0 \
ONLINE_ECIA_TRAIN_ROOT=/workspace/evidence_aware_tracking/code/outputs/online_ecia_finetune/online_aware_v1 \
ONLINE_ECIA_EVAL_RUN_ID=online_aware_v1_eval \
bash evaluate_online_ecia_quadtrack_dtb.sh
```

The consolidated report is written to:

`outputs/online_ecia_evaluation/online_aware_v1_eval/REPORT.md`.

Both scripts support a read-only dependency/path check:

```bash
ONLINE_ECIA_PREFLIGHT_ONLY=1 bash train_online_ecia_quadtrack.sh

ONLINE_ECIA_PREFLIGHT_ONLY=1 \
ONLINE_ECIA_TRAIN_ROOT=/workspace/evidence_aware_tracking/code/outputs/online_ecia_finetune/online_aware_v1 \
bash evaluate_online_ecia_quadtrack_dtb.sh
```

The evaluator runs the unchanged BoT-SORT baseline, old rule-based Full,
existing Base-Guided Full with pre-query safety, fine-tuned HR-disabled, and
fine-tuned Full with pre-query safety. It rejects the comparison unless raw
BoT-SORT predictions are byte-identical and the fine-tuned HR-on/off branches
share the same proposed event and would-request-HR schedules.

## Balanced-v2: Abstain-collapse fix

Balanced-v2 is isolated from the v1 artifacts and keeps the HIVE, OSNet-AIN,
candidate generator, tracker, LR protocols, ECIA architecture, and registry
safety policy unchanged. Its changes are limited to supervision and model
selection:

- safety-infeasible candidates retain an Abstain target but use a distinct
  sample kind instead of being mixed with genuine identity negatives;
- one controlled sampler produces 40% Merge-Old, 20% Competitor, 30% genuine
  Abstain and 10% safety-infeasible examples per epoch;
- misleading-HR Old hard negatives are capped at 15% of the Old quota;
- loss class weights are disabled, so sampling and loss cannot double-weight
  the majority class;
- the wrong-merge loss remains active with a smaller 0.75 coefficient and
  1.25 non-merge multiplier;
- validation logs precision/recall/F1 for Old, Competitor and Abstain, macro-F1,
  Merge Recall, Merge Precision, Wrong Merge and Abstain prediction rate;
- collapsed checkpoints are rejected, and selection uses constrained
  validation macro-F1/Merge Recall instead of overall accuracy;
- checkpoint selection uses videos `0019` and `0028`; former training video
  `0021` is held out from fitting and joined with `0030` for coarse robust
  threshold calibration, with minimum support and a per-video variance penalty.

Train without overwriting v1:

```bash
cd /workspace/evidence_aware_tracking/code
ONLINE_ECIA_DEVICE=0 \
ONLINE_ECIA_RUN_ID=balanced_v2 \
bash train_online_ecia_balanced_quadtrack.sh
```

Evaluate at QuadTrack 512x120 and DTB 640x320:

```bash
cd /workspace/evidence_aware_tracking/code
ONLINE_ECIA_DEVICE=0 \
ONLINE_ECIA_TRAIN_ROOT=/workspace/evidence_aware_tracking/code/outputs/online_ecia_finetune/balanced_v2 \
ONLINE_ECIA_EVAL_RUN_ID=balanced_v2_eval \
bash evaluate_online_ecia_balanced_quadtrack_dtb.sh
```

The selected checkpoint is `.../balanced_v2/training/best.pt`; the consolidated
report is `.../online_ecia_evaluation/balanced_v2_eval/REPORT.md` and includes
tracking metrics, exact-event Merge Recall/Wrong Merge, HR requests and bytes.
