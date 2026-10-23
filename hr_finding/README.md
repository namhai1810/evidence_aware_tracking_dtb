# HIVE — Hypothesis-conditioned Information Value Estimation

HIVE is a lightweight LR-only selector for ranking historical frames that may provide useful
identity evidence after an ambiguous tracking association. It predicts one utility logit per
candidate. HR images and GT are offline supervision only; the network never reads an HR crop,
HR confidence, HR ReID similarity, candidate rank, GT identity, or label at inference.

## Data packages

The original controlled top-5 package is:

```text
../outputs/reid_ranking_dataset_v1
```

It contains 925 LR switch events and 4,606 top-5 candidate pairs. The preserved held-out test
videos are `0016, 0017, 0025, 0031` (92 events). HIVE creates validation only from original
training videos: `0019, 0028, 0030` (76 events). The remaining original training videos form
the learning split. Test data is not loaded by `train.py`.

The recommended LR-proposed history-ranking package is:

```text
../outputs/hive_history_ranking_v2
```

It preserves the same 925 switch events and video split, but expands the pool to at most 20
candidates per event: 18,355 candidate pairs, 5,725 positives and 12,630 negatives. Candidate
proposal uses only LR tracker detections, LR pixels, confidence and time in `T-150 ... T-1`.
GT is joined only after selection to assign labels. The complete construction and leakage audit
are in [DATASET_REPORT.md](../outputs/hive_history_ranking_v2/DATASET_REPORT.md).

Rebuild it without touching v1:

```bash
cd /workspace/evidence_aware_tracking/code
python build_hive_history_ranking_v2.py
```

The builder refuses to overwrite an existing output directory; provide a new `--output-root`
for another version.

Run the full compatibility audit with:

```bash
cd /home/dtb/workspace/evidence_aware_tracking/code/hr_finding
python datasets/audit.py
```

The checked-in [DATA_AUDIT.md](DATA_AUDIT.md) records the current findings.

## Online inputs

Each event contains:

1. up to five old-track LR crops;
2. one LR target crop at the switch;
3. up to five candidate LR crops in v1, or up to twenty in v2;
4. temporal metadata;
5. optional real LR appearance history for the competing `lr_new_pred_id`.

Candidate LR crops are not obtained by shrinking an already-cropped HR person. The loader
first downsamples the complete source panorama to `512x120` with area interpolation, scales
the box, and only then crops the person. Inputs are letterboxed to `128x64` for the shared
tiny CNN.

The v1 offline pool was selected with HR detections and HR ReID. `manifest_bbox` keeps
that pool fixed for controlled reranking experiments. `lr_tracker_iou` restricts inputs to
candidates with a matching LR detection, but changes coverage. Neither mode solves the
missing deployable LR-only proposal generator; see the audit.

The v2 pool removes this proposal dependency: candidate boxes come directly from the LR tracker
and the proposal score is computed only from LR appearance. The HIVE network does not receive
the proposal score, GT match, label, or original proposal rank.

## Architecture

- `TinyAppearanceEncoder`: shared depthwise-separable CNN, configurable 32D/64D output.
- `MaskedAttentionAggregator`: target-conditioned aggregation of old-track history.
- `CandidateRelationEncoder`: old/target/candidate differences, products, and cosine relations.
- `CompetingHypothesisEncoder`: compares candidates with actual competing-track LR history.
- `HIVE` scorer: small MLP that emits one utility logit per candidate in a single batched pass.

Track ID integers are never encoded as appearance. If competing history is unavailable, the
full model uses an explicit mask and a zero contribution rather than fabricating an identity.

## Variants and ablations

Set `model.variant` in YAML or pass `--variant`:

| Variant | Inputs |
|---|---|
| `appearance_only` | Old, target, and candidate LR appearance |
| `appearance_metadata` | Appearance plus temporal metadata |
| `hive_no_hypothesis` | Identity-contrastive old/target/candidate relations plus metadata |
| `hive_full` | Above plus real competing-track appearance when available |

The explicit non-neural temporal baseline is available through `evaluate.py --baseline temporal`.
Geometry/motion baselines are not claimed because association matrices and deployable LR
candidate geometry are incomplete.

## Objectives

```text
L = lambda_cls * BCEWithLogits
  + lambda_rank * event-normalized RankNet
  + lambda_margin * optional teacher regression
```

- Classification labels candidate identity correctness, not counterfactual ID-switch reduction.
- RankNet compares every positive/negative pair only within the same event and averages per event.
- Events with no positive still train classification but do not enter RankNet.
- Margin distillation is implemented and disabled by default. Current `bridge_similarity` is an
  HR-ReID score, not a validated correction-utility target.

## Start training manually

Activate the environment that already contains PyTorch, OpenCV, TensorBoard, PyYAML and tqdm,
then run:

```bash
cd /home/dtb/workspace/evidence_aware_tracking/code/hr_finding
bash train.sh
```

Train the LR-proposed v2 dataset in a separate run:

```bash
HIVE_CONFIG=configs/hive_history_v2.yaml \
HIVE_RUN_NAME=hive_history_v2_seed42 \
bash train.sh
```

This configuration uses 20 candidates, batch size 8 and `num_workers=0` to avoid container
shared-memory failures. It does not overwrite the v1 dataset or earlier checkpoints.

Useful environment overrides:

```bash
HIVE_DEVICE=cuda:0 \
HIVE_CUDA_VISIBLE_DEVICES=0 \
HIVE_RUN_NAME=hive_full_seed42 \
bash train.sh
```

Configuration overrides can be appended directly:

```bash
bash train.sh --variant appearance_only --set training.batch_size=32
```

The script runs in the foreground and propagates errors. It prints the run directory and the
TensorBoard command before launching.

## Monitoring

Open TensorBoard with:

```bash
tensorboard --logdir runs --port 6006
```

Each run saves `train.log`, `metrics.csv`, `metrics.json`, `config.yaml`, TensorBoard events,
`checkpoints/latest.pt`, and `checkpoints/best.pt`. Logged series include all classification
and ranking losses, accuracy, precision/recall/F1, Success@1/3/5, MRR, NDCG@5, oracle coverage,
oracle gap, learning rate, gradient norm, and epoch duration. Batch losses are logged at the
configured interval. Parameter/gradient histograms are optional.

Interpretation guidelines:

- falling train loss with flat/worsening validation Success@1 or MRR indicates overfitting;
- both losses saturating with a large oracle gap indicates model/feature limitations;
- low oracle coverage is a candidate-pool problem and cannot be repaired by reranking;
- high candidate accuracy with poor Success@1 can arise from class imbalance and is not enough.

## Resume

```bash
HIVE_RESUME=/path/to/run/checkpoints/latest.pt \
HIVE_RUN_NAME=the_same_run_name \
bash train.sh
```

The checkpoint restores model, optimizer, scheduler, AMP scaler, epoch, seed, global step,
best validation metric, and early-stopping state.

## Explicit evaluation

Validation:

```bash
python evaluate.py \
  --config runs/<run>/config.yaml \
  --checkpoint runs/<run>/checkpoints/best.pt \
  --split val
```

Final held-out test evaluation must be requested explicitly:

```bash
bash test.sh runs/<run>/checkpoints/best.pt
```

`test.sh` restores the exact run configuration, evaluates only the video-disjoint held-out
test split, and writes results to `runs/<run>/test_results/`. Use `best.pt` for final model
selection because it is chosen only from validation performance; `latest.pt` is also accepted
when explicitly desired. The test artifacts are:

- `metrics.json`: aggregate classification, ranking and latency metrics;
- `predictions.jsonl`: complete predicted candidate ranking for every event;
- `candidate_rankings.csv`: flat candidate-level table with score, rank and GT label;
- `top1_failures.jsonl`: retrievable events where the model chose the wrong top-1 frame;
- `test_report.md`: compact human-readable report and interpretation boundary.

The test loader defaults to `num_workers=0` to avoid shared-memory failures in containers with
a 64 MB `/dev/shm`. Override it only when sufficient shared memory is available:

```bash
HIVE_TEST_WORKERS=1 bash test.sh runs/<run>/checkpoints/best.pt
```

Evaluation reports candidate accuracy/precision/recall/F1; event Success@1/3/5, MRR and
NDCG@5; oracle coverage/gap; trainable parameters; total latency; encoder time; and scoring-head
time per candidate. Events with no positive count as ranking failures in the primary metrics,
and conditional `*_when_retrievable` metrics are reported separately.

The top score can optionally be thresholded into a no-request decision. Calibrate
`evaluation.no_request_threshold` on validation only; request rate and request precision are
reported so an apparently accurate model cannot hide by declining nearly every event.

## Efficiency

- Candidate scoring is batched by event.
- Padded crops are not passed through the CNN.
- Original frames use a per-worker LRU cache.
- Candidate LR evidence is formed once per dataset item; no YOLO inference is repeated.
- AMP, CPU/CUDA, frozen appearance encoder, and configurable embedding size are supported.
- Encoder and scoring-head latency are measured separately.

No compatible cached YOLO ROI feature tensors were found in the current dataset, so the tiny
CNN fallback is used. The modular encoder can later be replaced by cached detector ROI features.

## Scientific limitations

- The candidate pool is HR-ReID-preselected and therefore not yet deployable end to end.
- Only part of the pool matches an actual LR tracker detection.
- Competing LR histories are missing for some events.
- Association matrices and motion residuals are unavailable.
- Candidate correctness is not measured correction utility.
- No tracking replay exists, so IDSW, IDF1 and HOTA improvements must not be inferred from
  Success@K. Those metrics require a real identity-merge/replay evaluation.
