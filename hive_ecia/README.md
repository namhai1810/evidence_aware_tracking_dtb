# HIVE + Lightweight ECIA

Lightweight ECIA is a learnable identity-recovery head placed after the frozen
HIVE v2 historical-frame selector and frozen `yolo26n-reid.onnx` encoder. It
does not add a visual backbone and does not modify the existing
`hive_hr_reid/` baseline.

> Implementation status: code, configuration, documentation, and unit tests
> are present. No ECIA training, inference experiment, evaluation, or benchmark
> has been executed.

## 1. Data flow

```text
LR tracking event
  ├── old history (≤5 LR crops) ─────── frozen ReID ── old prototype 512D
  ├── target LR crop ───────────────── frozen ReID ── target 512D
  ├── competitor histories (≤12 IDs) ─ frozen ReID ── K prototypes 512D
  └── frozen HIVE v2 selects one historical candidate
          └── exactly one HR crop ───── frozen ReID ── HR evidence 512D

normalized embeddings + competitor mask
                  ↓
        Lightweight ECIA (43,684 params)
                  ↓
     Old / Competitor / Abstain softmax
                  ↓
  confidence check → merge proposal or keep IDs
                  ↓
 existing replay collision safety check
```

Only ECIA is trainable. HIVE and ReID are frozen. Inference never reads
`gt_id`, candidate `label`, `matched_gt_id`, or post-replay outcomes.

## 2. Architecture and tensor shapes

Notation: batch `B`, competitor padding `K≤12`, embedding dimension `D=512`,
projection dimension `P=64`. Old occupies hypothesis slot 0; competitors occupy
slots `1..K`; Abstain is appended as the final decision logit.

### Layer 1 — Shared feature projection

```text
Linear(512,64) → LayerNorm(64) → GELU
```

The same module instance projects Old, Target, HR, and all Competitors.

```text
old/target/hr: [B,512] → [B,64]
competitors:   [B,K,512] → [B,K,64]
hypotheses:    [B,1+K,64]
```

Input embeddings are L2-normalized at the model boundary, even when the cache
already contains normalized embeddings.

### Layer 2A — Base identity association

For each hypothesis `q`:

```text
[abs(target-q), target*q, cosine(target,q)]  # 64+64+1 = 129D
Linear(129,32) → GELU → Linear(32,1)
```

Output: `base_scores [B,1+K]`. This branch uses no HR evidence.

### Layer 2B — HR evidence gain

For each hypothesis `q`:

```text
[abs(hr-q), abs(hr-target), hr*q,
 cosine(hr,q), cosine(hr,target), base_score]  # 195D
Linear(195,32) → GELU → Linear(32,1)
```

Output: `delta_scores [B,1+K]`. Delta is signed: HR is allowed to raise or
lower an identity score.

### Layer 3 — Evidence-gated fusion

Five event-level features are used:

1. LR base probability for Old;
2. maximum LR association probability;
3. Old score minus the strongest competitor score;
4. HR–Target cosine similarity;
5. maximum HR–Competitor cosine similarity.

```text
Linear(5,16) → GELU → Linear(16,1) → Sigmoid
g ∈ [0,1]
final_score = base_score + g * delta_score
```

The same scalar gate is shared by all hypotheses in an event. If HR is absent,
the gate is forced to zero and `final_score=base_score`.

### Layer 4 — Recovery decision

```text
Abstain: Linear(5,16) → GELU → Linear(16,1)
Softmax([Old, valid Competitors, Abstain])
```

Padded competitors are masked before softmax. A merge is proposed only if Old
wins and passes learned-probability confidence/margin checks. The four old
cosine thresholds are not used. Replay may still reject a proposal that would
create duplicate IDs in one frame.

## 3. Parameter count

| Block | Trainable parameters |
|---|---:|
| Shared projection + LayerNorm | 32,960 |
| Base association head | 4,193 |
| HR evidence-gain head | 6,305 |
| Evidence gate | 113 |
| Abstain head | 113 |
| **Total** | **43,684** |

The constructor raises an error above the configured 50,000-parameter limit.
Unit tests also assert the exact count.

## 4. Dataset and labels

`dataset.py` uses a non-pickle compressed NPZ cache. Each event stores only:

- normalized old, target, HR, and competitor embeddings;
- competitor mask and tracker IDs;
- target class and sample type;
- event/sequence/split identifiers for auditing.

GT identity is consumed only while materializing the offline class label:

- `positive`: Target belongs to Old and HR supports that identity;
- `negative`: Target belongs to a listed competitor;
- `hard_negative`: confusing competitor or misleading HR evidence;
- `no_merge`: Target belongs to neither Old nor available competitors, so
  Abstain is correct.

The loader audits required sample types and video-disjoint sequence lists.
Adjacent frames must never be randomly split across train/validation/test.

Cache materialization is intentionally an offline, evaluation-aware boundary:
use `inference.extract_event_embeddings()` to obtain frozen model inputs, join
identity metadata in the data-building process, then call
`dataset.build_identity_sample()` and `dataset.save_embedding_cache()`. The
builder assigns Old, Competitor, or Abstain and records one of the four sample
kinds. This label-building step may read GT; `inference.py` cannot.

Configured split:

```text
train:       0015 0018 0020 0021 0022 0023 0024 0026 0027 0029
model-val:   0019 0028
calibration: 0030
test:        0016 0017 0025 0031
```

## 5. Training objectives

`loss.py` combines:

```text
0.50 × base identity classification
1.00 × fused Old/Competitor/Abstain classification
0.25 × margin-ranking loss
0.50 × wrong-merge binary penalty
```

Base classification excludes no-merge examples because Layer 2A has no Abstain
class. Fused classification and ranking include Abstain. The wrong-merge term
explicitly penalizes high Old probability when Old is not the correct class.

`train.py` loads only train and model-validation embedding caches. There is no test
cache argument in the training CLI. It verifies the parameter count, clips
gradients, performs validation early stopping, and records
`test_data_accessed=false` in run metadata.

Training displays nested `tqdm` progress bars for batches and epochs. The epoch
bar reports train/validation/best loss, patience usage, processing speed, and an
ETA to the configured maximum epoch. Every completed epoch is also appended to
`training_log.jsonl` with all loss components, accuracy, wrong-merge rate,
elapsed time, and estimated remaining time; this file remains readable while
training is still running.

Sequence `0030` is reserved for post-training calibration of the operational
Old probability/margin. It is not used for gradient updates or early stopping.

### Training-sample mining

`prepare_data.py` runs frozen HIVE/ReID only on configured train and validation
videos. For each real switch event it keeps the original Old target, mines an
easy and hardest verified competitor, and constructs an Abstain example by
borrowing a target whose GT identity matches neither Old nor any verified
Competitor hypothesis. Borrowing remains inside the same split and prefers the
same sequence. Competitor counterfactuals are admitted only when their tracker history
has a dominant GT identity different from Old at IoU ≥0.5. This avoids treating
another fragment of the same person as a synthetic negative.

For each competitor counterfactual, one real LR crop is held out as Target and
the competitor prototype is recomputed from the remaining history crops. The
Target embedding is therefore never averaged into the hypothesis prototype.

Counterfactuals are training/validation augmentation. Held-out tracking metrics
are always computed on untouched real test events.

## 6. Ablations

The same model supports:

| Mode | Behavior |
|---|---|
| `base_only` | strict LR-only: `g=0`, and HR features are masked before learned Abstain |
| `base_hr_gain` | `g=1` when HR exists; ungated Base + Delta |
| `full` | learned event-level evidence gate |

Set `experiment.ablation` in `config.yaml` or pass `--ablation` to training.

## 7. Inference and caching

`inference.py` reuses the existing frozen HIVE dataset/selector and frozen ONNX
ReID runner. It encodes:

- at most five LR old crops;
- one LR target crop;
- up to twelve LR competitor prototypes;
- exactly one HIVE-selected HR crop.

The content-addressed cache key contains ReID model hash, image/crop source,
bbox, and resolution tag. Cached LR embeddings are loaded from `.npy`; only
cache misses are passed through ReID. No full-frame HR detector is run.

Inference writes a GT-free decision JSONL. `evaluate.py` later joins GT to this
frozen file, runs the existing collision-safe replay, and reports IDSW, IDF1,
HOTA, MOTA, actual fixes, and wrong corrections.

## 8. Commands for future execution

These commands document the intended flow. They have **not** been executed.

### One-command full experiment

The default command trains and evaluates all three ablations:

```bash
cd /home/dtb/workspace/evidence_aware_tracking/code
bash run_hive_ecia_experiment.sh
```

Optional output/device overrides:

```bash
ECIA_DEVICE=cuda:0 \
HIVE_ECIA_OUTPUT=/home/dtb/workspace/evidence_aware_tracking/code/outputs/hive_ecia_experiment \
bash run_hive_ecia_experiment.sh
```

The script enforces this order:

```text
train/model-val frozen embedding caches (calibration and test excluded)
  → train all three checkpoints on train, early-stop on 0019/0028
  → calibration GT-free scores on 0030 for all three checkpoints
  → calibration-only probability/margin selection
  → freeze every checkpoint + threshold pair
  → only now open held-out data
  → reproduce unchanged HIVE v2 + HR ReID baseline
  → held-out ECIA test GT-free decisions
  → post-hoc test GT join, replay, IDSW/IDF1/HOTA/wrong corrections
  → Base / Gain / Full comparison report
```

Thus neither the checkpoint, early-stopping epoch, ablation choice, nor merge
threshold can be selected after observing held-out test performance. `full` is
the predeclared primary ECIA result; Base-only and Base + HR Gain are reported
as ablations rather than candidates for test-set model selection.

The final table contains LR-only, the unchanged HIVE v2 + HR ReID baseline,
ECIA Base-only, ECIA Base + HR Gain, and predeclared-primary ECIA Full.
The same report embeds `tracking_metrics.png` and `correction_outcomes.png` so
the held-out metric changes and the trade-off between actual fixes and wrong
corrections can be inspected visually.
The comparison aborts if any ECIA ablation has a different held-out HR-request
count from the unchanged HIVE v2 + HR ReID baseline; every event is additionally
checked against the one-HR-crop limit.

Final report:

```text
outputs/hive_ecia_experiment/comparison/REPORT.md
```

Train one ablation after caches have been prepared:

```bash
cd /home/dtb/workspace/evidence_aware_tracking/code
.venv/bin/python -m hive_ecia.train \
  --config hive_ecia/config.yaml \
  --ablation full \
  --output-dir outputs/hive_ecia/full
```

Run GT-free inference on held-out test:

```bash
.venv/bin/python -m hive_ecia.inference \
  --config hive_ecia/config.yaml \
  --ecia-checkpoint outputs/hive_ecia/full/best.pt \
  --split test \
  --output outputs/hive_ecia/full/test_decisions.jsonl
```

Evaluate only after decisions are frozen:

```bash
.venv/bin/python -m hive_ecia.evaluate \
  --decisions outputs/hive_ecia/full/test_decisions.jsonl \
  --evaluation-events outputs/hive_history_ranking_v2/test_data/events.jsonl \
  --output-dir outputs/hive_ecia/full/test_evaluation
```

Run unit tests:

```bash
.venv/bin/python -m unittest discover -s hive_ecia/tests -v
```

### Zero-shot panorama generalization

The manually reviewed panorama MOT dataset keeps its existing sequence split:
train = normal walking/crossing/partial occlusion, validation = seam, and test
= full occlusion. For a strict generalization measurement, neither panorama
train nor panorama validation is used to update ECIA or tune thresholds. HIVE
v2, ReID, ECIA Full, and thresholds remain frozen from QuadTrack; only the
full-occlusion test sequence is opened for final recovery replay.

```bash
cd /home/dtb/workspace/evidence_aware_tracking/code
ECIA_DEVICE=cuda:0 bash run_panorama_ecia_zero_shot.sh
```

The builder shifts the original bootstrap predictions by the audited 300-frame
trim offset, creates LR-only candidate pools, freezes them before attaching GT
labels, and fails on any sequence overlap. The evaluator writes GT-free
decisions before joining manual GT. Final artifacts are written under
`outputs/panorama_ecia_zero_shot/zero_shot_evaluation/`.

## 9. Source layout

```text
hive_ecia/
├── __init__.py
├── model.py       # four ECIA layers and ablations
├── dataset.py     # sample labels, variable-K collation, NPZ cache
├── prepare_data.py # frozen train/val cache and counterfactual mining
├── loss.py        # classification, ranking, wrong-merge losses
├── train.py       # ECIA-only optimizer and validation checkpointing
├── inference.py   # frozen HIVE/ReID adapter and GT-free decisions
├── calibrate.py   # validation-only operational threshold selection
├── evaluate.py    # post-hoc replay and MOT metrics
├── compare.py     # held-out comparison across three ablations
├── config.yaml
├── README.md
└── tests/
```
