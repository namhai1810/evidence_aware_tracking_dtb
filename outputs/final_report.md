# Evidence-Aware Tracking: YOLO Model and Resolution Study

## All-sequence tracking and backward-HR-retrieval overview

This section is the current operational summary. It aggregates all 17 QuadTrack train
sequences (`0015`–`0031`, 90,763 GT person instances). The original controlled experiments
on sequence `0015` are retained below for reproducibility.

<p align="center">
  <img src="./report_assets/tracking_retrieval_dashboard.png"
       alt="All-sequence tracking and retrieval dashboard"
       width="100%" />
  <br />
  <em>Figure 1 — Tracking quality at 512x120 and 2048x480, ID-switch counts, and the HR-retrieval dataset funnel.</em>
</p>

### Tracking configuration and aggregate results

- Detector: YOLO11l, person class only, confidence `0.20`, NMS IoU `0.70`.
- Evaluation: Hungarian prediction-to-GT assignment at IoU `>=0.50`, micro-aggregated across
  sequences.
- Low-resolution stream: physical `512x120`, detector `imgsz=512`.
- Original stream: physical `2048x480`, detector `imgsz=960`.
- ReID tracker: BoT-SORT with native YOLO appearance features and sparse optical-flow GMC.

| Physical input / tracker | Precision | Recall | F1 | mIoU | MOTA | IDF1 | IDSW | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512x120 / ByteTrack | **0.833** | 0.325 | 0.468 | **0.734** | 0.246 | 0.303 | 1,243 | 29,507 | **5,912** | 61,256 |
| 512x120 / BoT-SORT+ReID | 0.818 | 0.339 | 0.480 | 0.730 | 0.253 | 0.353 | **925** | 30,794 | 6,861 | 59,969 |
| 2048x480 / ByteTrack | 0.762 | 0.476 | 0.586 | **0.734** | **0.306** | 0.398 | 1,967 | 43,241 | 13,483 | 47,522 |
| 2048x480 / BoT-SORT+ReID | 0.741 | **0.486** | **0.587** | 0.730 | 0.298 | **0.419** | **1,657** | **44,155** | 15,437 | **46,608** |

At `512x120`, BoT-SORT+ReID reduces IDSW from `1,243` to `925` and raises IDF1 from
`0.303` to `0.353`. At `2048x480`, it reduces IDSW from `1,967` to `1,657` and raises IDF1
from `0.398` to `0.419`, but MOTA falls from `0.306` to `0.298` because FP increases.

The lower absolute IDSW count at `512x120` does not mean that low-resolution tracking is
better. Its recall is only `0.339` versus `0.486` for the original stream: many difficult
people become false negatives and therefore cannot produce a counted switch. IDSW must be
read together with recall, IDF1, FN, and the number of predicted tracks.

### Cross-resolution ReID ranking method

<p align="center">
  <img src="./report_assets/reid_ranking_pipeline.png"
       alt="Cross-resolution ReID ranking pipeline"
       width="100%" />
  <br />
  <em>Figure 2 — From a low-resolution ID switch to ranked high-resolution historical candidates using ReID embeddings.</em>
</p>

For each GT-identified ID-switch event in the `512x120` BoT-SORT+ReID output:

1. take up to five observations of the old LR track from the preceding 30 frames and average
   their normalized embeddings into an old-track prototype;
2. extract an embedding from the switched target detection on the physical `512x120` frame;
3. inspect every tracked-person crop from the immediately preceding five seconds of original
   `2048x480` frames (`T-150` through `T-1` at 30 FPS), visiting `T-1` first and then moving
   backward toward `T-150`;
4. extract 512-dimensional L2-normalized embeddings with `yolo26n-reid.onnx`;
5. rank each HR detection by
   `min(cosine(old_track, candidate), cosine(target, candidate))`, retain the best detection
   per frame, and save up to five temporally diverse frames; ReID similarity determines rank,
   while the nearer frame wins only when rounded similarity scores are equal;
6. only after ranking, use the target GT identity and IoU `>=0.50` to label each saved
   candidate positive or negative.

GT/Hungarian matching identifies evaluation switch events and supplies supervision labels;
it is not an input to the embedding ranking score. The historical `hungarian_gt_oracle`
reports are therefore kept separately from this embedding experiment.

### Ranking dataset size and retrieval yield

| Stage or dataset component | Count | Meaning |
|---|---:|---|
| LR ID-switch events | **925** | Starting evaluation events |
| Events with any ranked HR candidate | **922** | At least one historical HR detection has a valid embedding (`99.7%`) |
| Events where the correct HR identity exists anywhere in the window | **920** | Full-pool availability upper bound (`99.5%` of switches) |
| Events with a positive candidate in saved top-5 | **706** | Maximum directly supervised success within the saved top-5 (`76.3%`) |
| Events with the correct identity at top-1 | **514** | Current cosine-ranking result (`55.6%`) |
| Saved LR-event/HR-candidate rows | **4,606** | Ranking samples |
| Positive rows | **1,927** | Candidate box matches the switch identity's GT (`41.8%`) |
| Negative rows | **2,679** | Candidate belongs to another identity (`58.2%`) |
| Unique old-track LR crops | 3,137 | From 3,972 query-crop references |
| Unique ranked HR crops / source images | 3,150 / 2,484 | HR visual data represented by the saved rows |

`922` means that a ranked candidate list exists, not that every list contains the correct
person. A perfect reranker restricted to the saved top-five lists could select a positive for
at most `706/925` events. A perfect reranker over every HR detection in the full window could
reach at most `920/925` events, subject to successful identity merging.

The current top-1 ranking supports 514 switches, giving a retrieval-supported estimate of
`925 -> 411` IDSW if every selected bridge were merged successfully. An oracle choice among
the saved top five supports 706 switches (`925 -> 219`). These are not post-merge tracking
metrics: the system does not yet implement an online uncertainty trigger, identity merge, or
post-merge MOTA/IDF1 evaluation.

Cosine score alone is not yet safe for automatic merging. At threshold `0.30`, 921 top-ranked
candidates are accepted and 514 are correct (`55.8%` precision). At threshold `0.60`,
402 are accepted and 263 are correct (`65.4%` precision). A learned ranker should add crop
quality, blur, occlusion, detector confidence, motion, and HR-track consistency.

### LR-proposed HIVE history-ranking dataset v2

To train historical retrieval before implementing an online switch trigger, a second dataset
was generated separately at `outputs/hive_history_ranking_v2`. It preserves all 925 known
switch events but replaces the HR-ReID-preselected top-five pool with up to 20 candidates
proposed from the LR stream itself.

Construction is label-safe: each event searches `T-150 ... T-1`; the complete panorama is
first downsampled to `512x120`; LR tracker detections are scored using LR-only HSV appearance,
confidence, recency and temporal diversity; the pool is frozen; only then is Hungarian GT
matching at IoU `>=0.50` joined to produce positive/negative labels. HR detections and HR-ReID
embeddings do not participate in candidate proposal.

| Split | Videos | Switch events | Candidate pairs | Positive | Negative | Oracle coverage |
|---|---:|---:|---:|---:|---:|---:|
| Learning train | 10 | 757 | 14,997 | 4,610 | 10,387 | 87.8% |
| Validation | 3 | 76 | 1,518 | 578 | 940 | 89.5% |
| Held-out test | 4 | 92 | 1,840 | 537 | 1,303 | 87.0% |
| **All** | **17** | **925** | **18,355** | **5,725** | **12,630** | **87.9%** |

The negative candidates remain wrong-person or unmatched LR detections inside known switch
events; normal/non-switch events are intentionally deferred until the trigger stage. The
video-level leakage audit passes with zero shared sequences, events, source frames, or exact
LR crop contents between learning train, validation and test. See the
[dataset report](hive_history_ranking_v2/DATASET_REPORT.md) and
[machine-readable audit](hive_history_ranking_v2/leakage_audit.json).

The best HIVE checkpoint was selected at epoch 3 using the video-disjoint validation split
and was then evaluated on the 92 held-out test events:

| Test method | Correct top-1 | Success@1 | Model/ranking latency | Observed evaluation time | Status |
|---|---:|---:|---:|---:|---|
| Nearest historical frame | 25/92 | 27.17% | <0.001 ms/event | 71.95 ms/event | Deployable metadata baseline |
| LR HSV appearance proposal | 48/92 | 52.17% | Not benchmarked | Not benchmarked | Precomputed candidate-proposal order |
| **HIVE, epoch 3** | **58/92** | **63.04%** | **0.815 ms/event** | **79.80 ms/event** | Learned LR-only ranker on CUDA |
| GT oracle candidate pool | 80/92 | 86.96% | N/A | N/A | Non-deployable upper bound |

The HIVE forward-pass timing corresponds to approximately **1,226 ranked events/s**. The
observed end-to-end evaluator throughput is **12.53 events/s** and includes image loading,
crop preparation, batching, metrics, and result serialization; it is not full tracking FPS.
The HSV proposal cost is not reported because its descriptors and candidate order were
materialized during dataset generation rather than timed online. Full test details are in the
[HIVE held-out test report](../hr_finding/runs/hive_history_v2_seed42/test_results/test_report.md).

### Current all-sequence and retrieval artifacts

- [All-sequence ByteTrack/ReID comparison](all_sequences_reid_comparison_report.md)
- [Cross-resolution embedding retrieval report](highres_retrieval_reid_embedding_report.md)
- [Ranking events with top-five candidates](highres_retrieval_reid_embedding/switch_events_top5.json)
- [Compact ranking table](highres_retrieval_reid_embedding/top5_frames.csv)
- [Report-method index](retrieval_reports_index.md)
- [ByteTrack GT/Hungarian oracle reference](highres_retrieval_hungarian_gt_oracle_bytetrack_report.md)
- [BoT-SORT+ReID GT/Hungarian oracle reference](highres_retrieval_hungarian_gt_oracle_botsort_reid_report.md)
- [Reproducible visualization script](../generate_final_report_visuals.py)
- [Cross-resolution embedding experiment source](../retrieve_highres_with_reid_embeddings.py)

## Original single-sequence study summary (sequence 0015)

This report combines five controlled experiments on QuadTrack sequence `0015`:

1. comparison of YOLO11s and YOLO11l with ByteTrack and SORT;
2. a resolution ablation for YOLO11l + ByteTrack;
3. a GPU-controlled comparison of YOLOv10l, YOLO11l, and YOLO12l with ByteTrack at
   `imgsz=640/960/1280/1536`, including cross-model failure analysis;
4. YOLO11l/960 with ByteTrack versus BoT-SORT with native appearance ReID;
5. physical aspect-ratio-preserving downsampling before inference to simulate bandwidth
   reduction, tested with both ByteTrack and BoT-SORT+ReID.

The recommended efficiency/quality baseline remains:

```text
Detector:   YOLO11l
Tracker:    ByteTrack
imgsz:      960
confidence: 0.20
NMS IoU:    0.70
class:      person only
```

At evaluation IoU 0.50, this configuration achieved recall `0.696`, F1 `0.818`, MOTA
`0.689`, IDF1 `0.816`, and 2 identity switches. YOLO12l at `imgsz=1536` has the highest
aggregate quality in the expanded study, but only narrowly: recall `0.700`, F1 `0.820`,
MOTA `0.691`, IDF1 `0.818`, and 3 switches. YOLO11l at 960 therefore remains the more
practical baseline because it is nearly tied while using a substantially smaller input.

The expanded experiment confirms that low recall is not unique to YOLO11. At the best
configuration of each large model, 495 of 1,708 GT instances (`29.0%`) are missed by all
three model/tracker pipelines. Resolution reduction to 640 also degrades identity quality
for every model. The exact GT 9 switch at frame 238 occurs in all three models, directly
showing a shared difficult scene and ByteTrack association limitation.

Enabling BoT-SORT with native YOLO11 appearance ReID does not improve this sequence with the
default tracker thresholds. Relative to ByteTrack, recall changes from `0.696` to `0.697`,
IDF1 remains `0.816`, MOTA changes from `0.689` to `0.687`, and switches increase from 2 to 3.

For bandwidth reduction, YOLO11l + ByteTrack at a physically resized `1024x240` input is the
best tested tradeoff. It uses 25% of the original raw pixels while retaining recall `0.691`,
MOTA `0.684`, IDF1 `0.813`, and only one switch. Reducing further to `768x180` or `512x120`
causes severe identity degradation. BoT-SORT+ReID is especially sensitive at `1024x240`:
recall remains `0.696`, but IDF1 falls to `0.694` and switches rise to 6.

## Experimental environment

- Container: `stitching-dev`
- Conda environment: `hainn_stitching`
- Dataset: QuadTrack train sequence `0015`
- Source frames: 600 frames at 2048x480
- Evaluated GT: 1,708 valid person instances
- Detection class: COCO class 0 (`person`) only
- Evaluation match threshold: IoU >= 0.50
- Experiments 1-2 benchmark device: CPU
- Experiments 3-5 benchmark device: NVIDIA GeForce RTX 5090 (`torch.cuda.is_available() == True`)

CUDA was unavailable when experiments 1-2 were recorded. Therefore, their FPS values remain
CPU measurements and are directly comparable with one another. Experiment 3 was rerun from
scratch on GPU for all three models and all four resolutions. Accuracy metrics may be compared
across the full report, but CPU and GPU throughput must not be compared as if they used the
same hardware path. Experiments 4-5 use the same GPU path as experiment 3.

## Metrics

- **Precision:** fraction of predicted boxes matched to GT.
- **Recall:** fraction of GT person instances matched by the final tracker output.
- **F1:** harmonic mean of precision and recall.
- **mIoU:** mean IoU of matched prediction/GT pairs.
- **MOTA:** combines false negatives, false positives, and identity switches.
- **IDF1:** global identity consistency over the sequence.
- **ID switches:** number of times a GT identity changes its matched predicted identity.

These are end-to-end tracker-output metrics, not raw detector-only metrics. A miss may be
caused by the detector, confidence filtering, tracker confirmation/gating, or failure to meet
the IoU 0.50 matching threshold.

## Experiment 1: model and tracker comparison

### Fixed settings

- `imgsz=960`
- confidence threshold `0.20`
- NMS IoU threshold `0.70`
- same sequence, frame order, decoder, letterbox preprocessing, rendering, and evaluator
- each tracker was initialized independently

### Results

| Model | Tracker | Precision | Recall | F1 | mIoU | MOTA | IDF1 | ID switches | TP | FP | FN | Speed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11s | ByteTrack | 0.988 | 0.584 | 0.734 | 0.882 | 0.564 | 0.460 | 22 | 998 | 12 | 710 | 23.40 FPS |
| YOLO11s | SORT | 0.983 | 0.487 | 0.652 | 0.883 | 0.475 | 0.377 | 7 | 832 | 14 | 876 | 23.50 FPS |
| **YOLO11l** | **ByteTrack** | **0.991** | **0.696** | **0.818** | **0.889** | **0.689** | **0.816** | **2** | **1189** | 11 | **519** | 9.31 FPS |
| YOLO11l | SORT | 0.992 | 0.680 | 0.807 | 0.884 | 0.671 | 0.630 | 6 | 1161 | **9** | 547 | 9.44 FPS |

### Model-size findings

With ByteTrack, replacing YOLO11s with YOLO11l:

- increases recall by 11.2 percentage points (`0.584` to `0.696`);
- increases F1 by `0.084` (`0.734` to `0.818`);
- increases MOTA by `0.125` (`0.564` to `0.689`);
- increases IDF1 by `0.356` (`0.460` to `0.816`);
- reduces identity switches from 22 to 2;
- reduces false negatives by 191 (`710` to `519`);
- reduces CPU speed from 23.40 FPS to 9.31 FPS.

With SORT, YOLO11l also improves recall, F1, MOTA, and IDF1 substantially. This shows that
the larger detector improves both tracking methods rather than benefiting only ByteTrack.

### Tracker findings

For YOLO11l, ByteTrack outperforms SORT in recall, F1, MOTA, and IDF1. ByteTrack produces 28
more true-positive matches and 28 fewer false negatives, while reducing identity switches
from 6 to 2. YOLO11l + ByteTrack is therefore the strongest model/tracker combination in
this experiment.

### Model-comparison artifacts

- [YOLO11s + ByteTrack video](model_comparison/0015_yolo11s_bytetrack.mp4)
- [YOLO11s + SORT video](model_comparison/0015_yolo11s_sort.mp4)
- [YOLO11l + ByteTrack video](model_comparison/0015_yolo11l_bytetrack.mp4)
- [YOLO11l + SORT video](model_comparison/0015_yolo11l_sort.mp4)
- [Machine-readable model metrics](model_comparison/metrics_iou_0.5.json)
- [Original model-comparison report](model_comparison/REPORT.md)

## Experiment 2: resolution comparison

### Controlled settings

- detector fixed to `yolo11l.pt`
- tracker fixed to Ultralytics `bytetrack.yaml`
- tracker reset at the beginning of each run
- confidence threshold fixed at `0.20`
- NMS IoU threshold fixed at `0.70`
- same frame order, image decoding, aspect-ratio-preserving letterbox preprocessing,
  rendering, and evaluation
- only `imgsz` changed: 640, 960, 1280, and 1536

### Results

| imgsz | Precision | Recall | F1 | mIoU | MOTA | IDF1 | ID switches | TP | FP | FN | Speed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 640 | 0.998 | 0.586 | 0.738 | **0.897** | 0.576 | 0.402 | 15 | 1001 | **2** | 707 | **17.35 FPS** |
| **960** | 0.991 | **0.696** | **0.818** | 0.889 | **0.689** | **0.816** | 2 | **1189** | 11 | **519** | 9.04 FPS |
| 1280 | 0.992 | 0.692 | 0.815 | 0.886 | 0.686 | 0.814 | 1 | 1182 | 10 | 526 | 6.18 FPS |
| 1536 | 0.992 | 0.691 | 0.815 | 0.884 | 0.686 | 0.815 | **0** | 1181 | 9 | 527 | 4.50 FPS |

### Effect of reducing resolution

Compared with 960, using 640:

- lowers recall by 11.0 percentage points (`0.696` to `0.586`);
- adds 188 false negatives (`519` to `707`);
- increases identity switches from 2 to 15;
- reduces IDF1 from `0.816` to `0.402`;
- lowers MOTA from `0.689` to `0.576`.

This is strong evidence that resizing the panoramic image to the 640 inference scale removes
person detail required for both continuous detection and stable identity assignment.

The higher mIoU at 640 does not mean it is the better configuration. It describes only the
boxes that were successfully matched; many difficult people were missed entirely, producing
lower recall and much worse identity metrics.

### Effect of increasing resolution beyond 960

Increasing `imgsz` to 1280 or 1536 does not improve recall, F1, MOTA, or IDF1. The raw
identity-switch count decreases from 2 to 1 and 0, but global identity quality remains almost
unchanged (`0.816`, `0.814`, and `0.815` IDF1). At the same time, speed falls from 9.04 FPS at
960 to 6.18 FPS at 1280 and 4.50 FPS at 1536.

Quality therefore saturates at `imgsz=960` for this sequence.

### Resolution-comparison artifacts

- [imgsz 640 video](resolution_comparison/0015_yolo11l_bytetrack_imgsz640.mp4)
- [imgsz 960 video](resolution_comparison/0015_yolo11l_bytetrack_imgsz960.mp4)
- [imgsz 1280 video](resolution_comparison/0015_yolo11l_bytetrack_imgsz1280.mp4)
- [imgsz 1536 video](resolution_comparison/0015_yolo11l_bytetrack_imgsz1536.mp4)
- [Machine-readable resolution metrics](resolution_comparison/metrics_iou_0.5.json)
- [Original resolution report](resolution_comparison/REPORT.md)

## Experiment 3: YOLOv10l, YOLO11l, and YOLO12l across resolution

### Controlled GPU settings

- models: `yolov10l.pt`, `yolo11l.pt`, and `yolo12l.pt`
- tracker: Ultralytics `bytetrack.yaml`, reset independently for every run
- resolutions: `imgsz=640`, `960`, `1280`, and `1536`
- confidence threshold `0.20`, NMS IoU `0.70`, person class only
- same 600 source frames, frame order, decoder, aspect-ratio-preserving preprocessing,
  rendering, and evaluator
- evaluation matching threshold: IoU >= 0.50
- device: NVIDIA GeForce RTX 5090

### Full 3 x 4 results

| Model | imgsz | Precision | Recall | F1 | mIoU | MOTA | IDF1 | ID switches | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv10l | 640 | 0.996 | 0.652 | 0.788 | 0.893 | 0.646 | 0.456 | **5** | 1114 | 5 | 594 |
| **YOLOv10l** | **960** | 0.995 | **0.666** | **0.798** | **0.892** | **0.655** | **0.616** | 13 | **1138** | 6 | **570** |
| YOLOv10l | 1280 | 0.987 | 0.645 | 0.780 | 0.890 | 0.629 | 0.411 | 14 | 1102 | 14 | 606 |
| YOLOv10l | 1536 | 0.994 | 0.627 | 0.769 | 0.889 | 0.617 | 0.589 | 11 | 1071 | 7 | 637 |
| YOLO11l | 640 | 0.998 | 0.587 | 0.739 | **0.896** | 0.578 | 0.402 | 13 | 1002 | **2** | 706 |
| **YOLO11l** | **960** | 0.991 | **0.696** | **0.818** | 0.890 | **0.689** | **0.816** | 2 | **1189** | 11 | **519** |
| YOLO11l | 1280 | 0.992 | 0.692 | 0.815 | 0.886 | 0.686 | 0.814 | 1 | 1182 | 10 | 526 |
| YOLO11l | 1536 | 0.992 | 0.692 | 0.815 | 0.884 | 0.687 | 0.815 | **0** | 1182 | 9 | 526 |
| YOLO12l | 640 | **0.998** | 0.590 | 0.742 | 0.894 | 0.583 | 0.403 | 10 | 1008 | **2** | 700 |
| YOLO12l | 960 | 0.987 | 0.693 | 0.815 | 0.889 | 0.683 | 0.813 | **2** | 1184 | 15 | 524 |
| YOLO12l | 1280 | 0.988 | **0.701** | **0.820** | 0.888 | 0.687 | 0.659 | 8 | **1197** | 15 | **511** |
| **YOLO12l** | **1536** | 0.990 | 0.700 | **0.820** | 0.889 | **0.691** | **0.818** | 3 | 1196 | 12 | 512 |

Bold cells within each model indicate its strongest value for that metric; the preferred row
balances the full set of detection and identity metrics rather than minimizing ID switches in
isolation.

### Model and resolution findings

- At `imgsz=960`, YOLO11l is strongest overall: MOTA `0.689` and IDF1 `0.816`, compared
  with YOLO12l at `0.683/0.813` and YOLOv10l at `0.655/0.616`.
- YOLO12l reaches the best absolute aggregate result at 1536: recall `0.700`, F1 `0.820`,
  MOTA `0.691`, and IDF1 `0.818`. The gain over YOLO11l at 960 is only `0.004` recall,
  `0.002` F1, `0.002` MOTA, and `0.002` IDF1.
- YOLOv10l peaks at 960 and gets worse at 1280/1536. Higher input resolution is therefore
  not universally beneficial; the model architecture and its learned scale behavior matter.
- At 640, IDF1 drops to `0.456`, `0.402`, and `0.403` for YOLOv10l, YOLO11l, and YOLO12l.
  The low-resolution identity problem occurs across every tested model.
- mIoU remains high (`0.884-0.896`) even when recall and IDF1 are poor. This is because mIoU
  is calculated only over successful matches and does not penalize hundreds of missed GT
  instances; it must not be used alone to choose the model.
- YOLO12l at 1280 illustrates tracker sensitivity: it has the highest recall (`0.701`) but
  eight switches and IDF1 `0.659`. More detections do not automatically produce stable IDs.

### Is the failure shared by all models?

Yes, most of the recall problem is shared, while some identity fragmentation is model-specific.
Using each model's best balanced configuration (YOLOv10l/960, YOLO11l/960, YOLO12l/1536):

| Model | FN | Misses also missed by all 3 | Shared fraction of this model's misses | Unique misses | ID switches |
|---|---:|---:|---:|---:|---:|
| YOLOv10l/960 | 570 | 495 | 86.8% | 50 | 13 |
| YOLO11l/960 | 519 | 495 | 95.4% | 4 | 2 |
| YOLO12l/1536 | 512 | 495 | 96.7% | 6 | 3 |

The common 495 misses are `29.0%` of all 1,708 GT instances. This makes the dominant recall
failure systematic across detector generations in this pipeline and sequence. Likely shared
contributors are severe panoramic motion blur, tiny/low-detail people, partial visibility,
the `conf=0.20` filter, ByteTrack output/confirmation behavior, and the evaluator's IoU 0.50
requirement. The result does not mean the three raw detectors miss every one of those instances;
these counts are measured from final tracker output.

Identity failures are less uniform. YOLOv10l produces 13 switches at its best setting versus
2 for YOLO11l and 3 for YOLO12l. However, the exact GT 9 switch at frame 238 occurs in all
three models. YOLO11l and YOLO12l also share the GT 7 switch at frame 211. Visual inspection
shows the same motion-blurred, rapidly changing panoramic region, so these events are not
specific to one YOLO generation and expose a ByteTrack association limitation.

### Representative failure cases by model

- YOLOv10l/960: [GT 9, frame 238, shared by all models](cross_model_resolution_gpu/failure_cases/yolov10l_imgsz960/case_01_timeline.jpg), [GT 2, frame 347](cross_model_resolution_gpu/failure_cases/yolov10l_imgsz960/case_02_timeline.jpg), [metadata](cross_model_resolution_gpu/failure_cases/yolov10l_imgsz960/id_switch_cases.json)
- YOLO11l/960: [GT 7, frame 211](cross_model_resolution_gpu/failure_cases/yolo11l_imgsz960/case_01_timeline.jpg), [GT 9, frame 238, shared by all models](cross_model_resolution_gpu/failure_cases/yolo11l_imgsz960/case_02_timeline.jpg), [metadata](cross_model_resolution_gpu/failure_cases/yolo11l_imgsz960/id_switch_cases.json)
- YOLO12l/1536: [GT 8, frame 210](cross_model_resolution_gpu/failure_cases/yolo12l_imgsz1536/case_01_timeline.jpg), [GT 7, frame 211](cross_model_resolution_gpu/failure_cases/yolo12l_imgsz1536/case_02_timeline.jpg), [GT 9, frame 238, shared by all models](cross_model_resolution_gpu/failure_cases/yolo12l_imgsz1536/case_03_timeline.jpg), [metadata](cross_model_resolution_gpu/failure_cases/yolo12l_imgsz1536/id_switch_cases.json)

### Experiment 3 artifacts

- [All 12 videos and MOT result files](cross_model_resolution_gpu/)
- [Machine-readable 3 x 4 metrics](cross_model_resolution_gpu/metrics_iou_0.5.json)
- [Cross-model miss and switch overlap](cross_model_resolution_gpu/failure_overlap_best_configs.json)
- [Reusable cross-model failure analysis](../compare_failure_overlap.py)

## Experiment 4: YOLO11l/960 with BoT-SORT and ReID

### Controlled settings and ReID verification

This experiment changes only the tracker relative to the GPU YOLO11l/960 ByteTrack result.
The detector weights, frames, preprocessing, `imgsz=960`, confidence `0.20`, NMS IoU `0.70`,
person class, CUDA device, and evaluator IoU `0.50` remain fixed.

The project-local [BoT-SORT ReID configuration](../botsort_reid.yaml) uses:

- `tracker_type: botsort`, `track_buffer: 30`, and `match_thresh: 0.80`;
- sparse optical-flow global motion compensation;
- `with_reid: true`, `model: auto`;
- proximity threshold `0.50` and appearance threshold `0.80`.

The runtime smoke test confirmed `BOTSORT`, `with_reid=True`, a non-null ReID encoder, and an
active native-feature hook. Therefore this is a real appearance-ReID run, not BoT-SORT with
ReID accidentally disabled. With `model: auto`, Ultralytics reuses YOLO11l backbone features
instead of loading a separate classification model.

### Results

| Tracker | Precision | Recall | F1 | mIoU | MOTA | IDF1 | ID switches | TP | FP | FN | Pred tracks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **ByteTrack** | **0.991** | 0.696 | **0.818** | **0.890** | **0.689** | **0.816** | **2** | 1189 | **11** | 519 | **9** |
| BoT-SORT + ReID | 0.988 | **0.697** | 0.818 | 0.859 | 0.687 | 0.816 | 3 | **1191** | 14 | **517** | 10 |

BoT-SORT+ReID recovers two additional GT instances, but adds three false positives and one
identity switch. F1 and IDF1 are unchanged at three-decimal precision, while MOTA decreases
by `0.001` and matched-box mIoU decreases by `0.031`. It ran at 36.55 end-to-end FPS in this
execution; that rate includes decoding, tracking, annotation, and MP4 encoding.

The direct failure comparison reinforces the similarity: 513 misses are common to both
trackers, equal to `30.0%` of all GT instances, `98.8%` of ByteTrack's misses, and `99.2%` of
BoT-SORT+ReID's misses. ReID does not address detector/visibility failures and does not improve
identity continuity under this configuration.

### BoT-SORT+ReID switch cases

| GT ID | Predicted ID | Last match | Switch frame | Gap | Interpretation |
|---:|---:|---:|---:|---:|---|
| 8 | 26 -> 33 | 206 | 211 | 4 | Severe blur and overlap with GT 7; appearance is not distinctive |
| 7 | 26 -> 36 | 203 | 219 | 15 | Long unmatched interval; new low-confidence track (`0.294`) |
| 9 | 44 -> 47 | 234 | 238 | 3 | Same exact failure frame as ByteTrack and all tested YOLO generations |

GT 7 and GT 8 are adjacent and strongly blurred in frames 203-211. Predicted ID 26 is matched
to each GT at different points before both identities fragment, showing that the native
appearance embedding is not sufficiently discriminative in this region. ReID association is
also spatially gated by `proximity_thresh=0.50`; appearance cannot rescue a candidate pair
whose spatial overlap falls outside that gate. The GT 7 gap lasts 15 frames despite the
30-frame lost-track buffer, indicating that keeping a track alive is not enough when spatial
and appearance association both fail.

- [GT 8 switch timeline](botsort_reid/id_switch_cases/case_01_timeline.jpg)
- [GT 7 switch timeline](botsort_reid/id_switch_cases/case_02_timeline.jpg)
- [GT 9 shared switch timeline](botsort_reid/id_switch_cases/case_03_timeline.jpg)
- [All switch metadata](botsort_reid/id_switch_cases/id_switch_cases.json)
- [BoT-SORT+ReID output video](botsort_reid/0015_yolo11l_botsort_reid_imgsz960.mp4)
- [MOT result](botsort_reid/0015_yolo11l_botsort_reid_imgsz960.txt)
- [ByteTrack comparison metrics](botsort_reid/metrics_iou_0.5.json)
- [Failure overlap with ByteTrack](botsort_reid/failure_overlap_with_bytetrack.json)

## Experiment 5: physical downsampling for bandwidth reduction

### Method

This experiment is different from changing YOLO's inference `imgsz` on the original frame.
Each decoded `2048x480` frame is first physically downsampled with OpenCV area interpolation,
preserving its `64:15` aspect ratio:

- `1024x240`: 25.0% of the original raw pixels;
- `768x180`: 14.1% of the original raw pixels;
- `512x120`: 6.25% of the original raw pixels.

YOLO11l and the tracker then receive only that reduced raster. YOLO `imgsz` is set to the
target width (`1024`, `768`, or `512`) so the frame is not enlarged back to the previous 960
scale before inference. Ultralytics may apply small stride-alignment padding internally, but
no lost source detail is restored. Tracker boxes are produced in the reduced coordinate space
and scaled back to `2048x480` only when writing the MOT result for GT evaluation. The saved
videos retain the true reduced dimensions.

All other settings remain controlled: person class only, confidence `0.20`, NMS IoU `0.70`,
evaluation IoU `0.50`, NVIDIA GeForce RTX 5090, and fresh tracker state for every run.
For the end-to-end throughput measurement, every independent run first processes three
samples through the complete model/tracker path without timing them. Tracker state is then
reset, CUDA is synchronized, and the following 600 frames are timed. CUDA is synchronized
again before stopping the clock. Each configuration is run three times and the table reports
the median FPS. A separate controlled compute-only benchmark below removes the substantial
I/O, drawing, and encoding variance from the resolution-speed comparison.

### Results

| Physical input | Raw pixels vs. original | Tracker | Precision | Recall | F1 | mIoU | MOTA | IDF1 | IDSW | TP | FP | FN | Median end-to-end FPS (3 runs) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024x240 | 25.0% | **ByteTrack** | **0.992** | 0.691 | **0.814** | **0.891** | **0.684** | **0.813** | **1** | 1180 | **10** | 528 | 65.88 |
| 1024x240 | 25.0% | BoT-SORT + ReID | 0.982 | **0.696** | **0.814** | 0.868 | 0.679 | 0.694 | 6 | **1188** | 22 | **520** | 70.50 |
| 768x180 | 14.1% | **ByteTrack** | **0.994** | 0.648 | **0.784** | **0.895** | **0.641** | 0.444 | 5 | 1106 | **7** | 602 | 63.87 |
| 768x180 | 14.1% | BoT-SORT + ReID | 0.987 | **0.650** | **0.784** | 0.877 | 0.638 | **0.447** | 5 | **1110** | 15 | **598** | 53.50 |
| 512x120 | 6.25% | **ByteTrack** | **0.986** | **0.529** | **0.689** | **0.894** | **0.516** | **0.338** | **10** | **904** | **13** | **804** | **70.25** |
| 512x120 | 6.25% | BoT-SORT + ReID | 0.981 | 0.523 | 0.683 | 0.881 | 0.505 | 0.319 | 15 | 894 | 17 | 814 | 88.10 |

The individual throughput measurements are:

| Tracker | Physical input | Run 1 FPS | Run 2 FPS | Run 3 FPS | Median FPS |
|---|---:|---:|---:|---:|---:|
| ByteTrack | 1024x240 | 64.74 | 91.20 | 65.88 | **65.88** |
| ByteTrack | 768x180 | 73.00 | 57.78 | 63.87 | **63.87** |
| ByteTrack | 512x120 | 98.94 | 69.27 | 70.25 | **70.25** |
| BoT-SORT + ReID | 1024x240 | 70.50 | 74.05 | 63.42 | **70.50** |
| BoT-SORT + ReID | 768x180 | 53.50 | 46.56 | 58.65 | **53.50** |
| BoT-SORT + ReID | 512x120 | 88.62 | 83.50 | 88.10 | **88.10** |

These FPS values include image decoding, explicit resize, detector inference, tracking,
drawing, MOT writing, and low-resolution MP4 encoding; model loading and the three warm-up
samples are excluded. They are not detector-only latency measurements. ByteTrack at
`1024x240` has a median of `65.88 FPS`, only `2.01 FPS` (`3.1%`) above `768x180` at
`63.87 FPS`. The three-run ranges overlap substantially, so this does not establish that the
larger raster is intrinsically faster. The wide spread between runs also demonstrates shared
system/GPU-load noise. The appropriate conclusion is approximately `64-70 FPS` for this
end-to-end ByteTrack pipeline, while accuracy and bandwidth—not a 2 FPS difference—separate
the resolutions. BoT-SORT + ReID at `768x180` was slower in all three recorded runs, but this
is an observed pipeline/kernel-shape effect and should not be generalized as a monotonic
resolution-speed law.

### Controlled compute-only throughput

The end-to-end measurement above answers how fast the current visualization application
runs, but it is not a clean comparison of input tensor sizes. For the controlled benchmark,
all 600 frames are decoded and physically resized before the clock starts. The timed region
contains only YOLO11l inference/NMS and the tracker update. Model loading, image decoding,
physical resize, drawing, MOT output, video encoding, and progress rendering are excluded.
Every run still uses exactly three untimed warm-up samples, resets tracker state, and
synchronizes CUDA before and after measurement. Five independent repetitions are run with
rotating configuration order, and the median is reported.

| Tracker | Physical input | Median compute FPS | Five-run range |
|---|---:|---:|---:|
| ByteTrack | 1024x240 | 172.77 | 155.33-175.79 |
| ByteTrack | 768x180 | 173.85 | 123.50-177.07 |
| ByteTrack | 512x120 | **178.24** | 131.65-180.61 |
| BoT-SORT + ReID | 1024x240 | 115.49 | 102.59-116.79 |
| BoT-SORT + ReID | 768x180 | 122.50 | 86.97-123.14 |
| BoT-SORT + ReID | 512x120 | **144.60** | 104.87-146.42 |

This resolves the apparent `1024x240` versus `768x180` reversal for ByteTrack: the controlled
result is `172.77` versus `173.85 FPS`, so `768x180` is only `1.08 FPS` (`0.6%`) faster. They
are effectively compute-equivalent on this RTX 5090. At batch size one, both workloads are
small enough that fixed Python/framework, CUDA kernel-launch, NMS, and tracker costs dominate;
the GPU is not saturated enough for pixel count alone to determine throughput. Tensor shapes
also do not scale with equal hardware efficiency. Consequently, reducing raw pixels from
`245,760` to `138,240` does not imply a proportional speedup.

BoT-SORT + ReID has more resolution-sensitive appearance and motion-compensation work and
shows the expected monotonic trend in the medians. Several slow outliers occur across
different configurations and times, which is why the median and rotated order are used. For
the bandwidth decision, `1024x240` remains preferable: it is essentially tied with `768x180`
in ByteTrack compute speed but has much better recall, MOTA, and especially IDF1.

### Change relative to original 2048x480 input

The comparison baselines are the matching GPU runs: YOLO11l/960 ByteTrack and YOLO11l/960
BoT-SORT+ReID on the original decoded `2048x480` frames.

| Tracker | Physical input | Delta recall | Delta MOTA | Delta IDF1 | Delta IDSW | Newly missed vs. original | Previously missed but recovered |
|---|---|---:|---:|---:|---:|---:|---:|
| ByteTrack | 1024x240 | -0.005 | -0.005 | -0.003 | -1 | 15 | 6 |
| ByteTrack | 768x180 | -0.048 | -0.048 | -0.372 | +3 | 91 | 8 |
| ByteTrack | 512x120 | -0.167 | -0.173 | -0.478 | +8 | 294 | 9 |
| BoT-SORT + ReID | 1024x240 | -0.001 | -0.008 | -0.122 | +3 | 18 | 15 |
| BoT-SORT + ReID | 768x180 | -0.047 | -0.049 | -0.369 | +2 | 92 | 11 |
| BoT-SORT + ReID | 512x120 | -0.174 | -0.182 | -0.497 | +12 | 307 | 10 |

Downsampling can occasionally recover a previously missed instance because resized pixels,
confidence, NMS, and association boundaries change slightly. The decisive quantity is the net
metric trend and the much larger set of newly introduced misses below `1024x240`.

### Interpretation and bandwidth recommendation

- `1024x240` + ByteTrack is the only reduced-input configuration that preserves both detection
  and identity quality. It cuts the raw pixel payload by 75% while changing IDF1 by only
  `-0.003` and reducing switches from 2 to 1.
- At `768x180`, recall decreases moderately, but IDF1 collapses from `0.816` to about `0.44`.
  The tracker can still localize many people, but trajectories fragment and global identity
  consistency is no longer reliable.
- At `512x120`, 294 new ByteTrack misses and 307 new BoT-SORT+ReID misses appear relative to
  their original-input baselines. This resolution is unsuitable for identity-aware tracking.
- Native ReID does not compensate for lost spatial detail. At `1024x240`, BoT-SORT+ReID keeps
  recall but produces six switches and IDF1 `0.694`; ByteTrack is substantially more stable.
- Raw pixel fraction is only a bandwidth proxy. Actual network bitrate also depends on codec,
  quality setting, frame rate, scene motion, and transport overhead.

### Downsampling artifacts

- [All six videos, MOT files, and runtime logs](bandwidth_downsample/)
- [Machine-readable metrics](bandwidth_downsample/metrics_iou_0.5.json)
- [Three-run warm-up timing data](bandwidth_downsample/timing_warmup3.json)
- [Controlled five-run compute benchmark](bandwidth_downsample/compute_benchmark_warmup3.json)
- [Controlled benchmark runtime log](bandwidth_downsample/logs/compute_benchmark_warmup3.log)
- [ByteTrack miss-overlap analysis](bandwidth_downsample/bytetrack_failure_overlap.json)
- [BoT-SORT+ReID miss-overlap analysis](bandwidth_downsample/botsort_reid_failure_overlap.json)
- ByteTrack switch examples: [1024x240](bandwidth_downsample/failure_cases/bytetrack_1024x240/case_01_timeline.jpg), [768x180](bandwidth_downsample/failure_cases/bytetrack_768x180/case_01_timeline.jpg), [512x120](bandwidth_downsample/failure_cases/bytetrack_512x120/case_01_timeline.jpg)
- BoT-SORT+ReID switch examples: [1024x240](bandwidth_downsample/failure_cases/botsort_reid_1024x240/case_01_timeline.jpg), [768x180](bandwidth_downsample/failure_cases/botsort_reid_768x180/case_01_timeline.jpg), [512x120](bandwidth_downsample/failure_cases/botsort_reid_512x120/case_01_timeline.jpg)

## Experiment 6: OmniTrack reproducibility audit

The requested local OmniTrack comparison is currently blocked by missing official model
artifacts, not by the QuadTrack images. The official repository snapshot inspected on
2026-10-21 has only the `main` branch, no tags or releases, and contains no trained `.pth`,
`.pt`, or `.ckpt` checkpoint. Its QuadTrack preparation section still says `coming soon`; the
only supplied experiment config is JRDB-specific and expects JRDB annotations, anchors,
transforms, and a `4160x480` raster. The documented ResNet-50 weight is only backbone
initialization and cannot produce trained OmniTrack predictions.

The official dependency stack also cannot safely be installed over the current working
environment without further porting. OmniTrack documents PyTorch `2.1.1`, CUDA `11.8`, MMCV
`1.7.1`, and MMDetection `2.28.2`, whereas `hainn_stitching` currently uses PyTorch
`2.10.0+cu130` on the RTX 5090. Downgrading would risk breaking the validated YOLO experiments
and would still leave the trained checkpoint and QuadTrack config absent. No such downgrade
was performed.

For context only, the paper reports the following on the full QuadTrack **test set**. These
are author-reported percentages under a different evaluator/split and must not be compared
numerically with the local decimal-valued results for train sequence `0015`.

| Official paper result | HOTA | IDF1 | MOTA | QuadTrack FPS |
|---|---:|---:|---:|---:|
| OmniTrack E2E | 19.87% | 19.47% | -5.89% | not reported |
| OmniTrack DA (TBD) | 23.45% | 26.41% | 9.68% | not reported |

The paper's available FPS values are instead measured on JRDB validation at `4160x480` using
one RTX 3090: `11.64 FPS` for OmniTrack E2E and `11.83 FPS` for OmniTrack DA with ByteTrack.
They are not QuadTrack or RTX-5090 measurements. Consequently, this report does not fabricate
local FPS, resolution, MOTA, IDF1, IDSW, or mIoU values for OmniTrack.

To complete Experiment 6, two matching artifacts are required: (1) a QuadTrack-trained
OmniTrack checkpoint, identifying the E2E/Det/DA variant, and (2) its QuadTrack config,
converter, and anchor file. Once supplied, the controlled matrix will be original `2048x480`,
`1024x240`, `768x180`, and `512x120`, using the same three-frame warm-up, CUDA synchronization,
repeated timing, and local evaluator as Experiment 5.

- [Detailed OmniTrack readiness audit](omnitrack/reproducibility_audit.md)
- [Official repository](https://github.com/xifen523/OmniTrack)
- [Official paper](https://arxiv.org/abs/2503.04565)
- [Open QuadTrack preparation issue](https://github.com/xifen523/OmniTrack/issues/7)

## ID-switch case analysis

The recommended YOLO11l + ByteTrack, `imgsz=960` result contains exactly two ID switches at
evaluation IoU 0.50. Both are re-identifications after an unmatched gap rather than direct
one-frame swaps between two simultaneously tracked people.

| Case | GT ID | Predicted ID | Last matched | Unmatched tracker-output gap | New-ID frame | Raw-detector evidence | Primary cause |
|---:|---:|---:|---:|---|---:|---|---|
| 1 | 7 | 27 -> 35 | 203 | 204-210 | 211 | Mostly lost/poor overlap through the blurred interval; usable detections return at 209-210 | Detector loss followed by failed tracker reconnection |
| 2 | 9 | 47 -> 50 | 234 | 235-237 | 238 | Detected in every gap frame, IoU 0.650-0.699 | ByteTrack association/track-lifecycle failure |

### Case 1: GT 7, predicted ID 27 to 35

At frame 203, GT 7 is matched to predicted ID 27 with tracker-output IoU `0.697`. The person
then becomes extremely blurred and low-detail. The raw detector has no valid IoU >= 0.50
match in frames 204-208 (GT 7 is absent from the annotation at frame 207). Raw detections
recover at frames 209 and 210 with IoU `0.518` and `0.501`, but the final tracker output does
not reconnect them to the old trajectory. At frame 211 ByteTrack emits ID 35, producing the
switch. This is therefore a mixed failure: the detector first loses the target, then the
tracker fails to recover the previous identity when evidence returns.

- [Last match, frame 203](id_switch_cases/case_01_last_match_f000203.jpg)
- [Frame immediately before switch, frame 210](id_switch_cases/case_01_previous_f000210.jpg)
- [New identity, frame 211](id_switch_cases/case_01_switch_f000211.jpg)
- [Three-frame case timeline](id_switch_cases/case_01_timeline.jpg)

### Case 2: GT 9, predicted ID 47 to 50

At frame 234, GT 9 is matched to predicted ID 47 with tracker-output IoU `0.695`. The final
tracker output has no match for this GT during frames 235-237, then assigns new ID 50 at
frame 238 with IoU `0.841`. In contrast, raw YOLO11l detects the person throughout the whole
gap: IoU is `0.650`, `0.699`, and `0.682` in frames 235-237, with confidence `0.317`, `0.423`,
and `0.794`. The identity loss is therefore not a detector miss. It is caused by ByteTrack's
association/gating or track lifecycle under motion blur and changing confidence. Improving
the detector alone will not resolve this case.

- [Last match, frame 234](id_switch_cases/case_02_last_match_f000234.jpg)
- [Frame immediately before switch, frame 237](id_switch_cases/case_02_previous_f000237.jpg)
- [New identity, frame 238](id_switch_cases/case_02_switch_f000238.jpg)
- [Three-frame case timeline](id_switch_cases/case_02_timeline.jpg)

### ID-switch artifacts and implications

- [Machine-readable switch metadata](id_switch_cases/id_switch_cases.json)
- [Raw YOLO detector diagnostics](id_switch_cases/raw_detector_diagnostics.json)
- [Reusable extraction script](../extract_id_switch_cases.py)

These cases support keeping `imgsz=960`, while testing ByteTrack association thresholds and
lost-track buffer settings on a validation set. A longer buffer may help bridge short gaps,
but standard ByteTrack has no appearance re-identification. Experiment 4 tested BoT-SORT with
native YOLO11 appearance ReID and did not improve identity metrics. Any further ReID study
should therefore tune the proximity/appearance gates or use a dedicated person-ReID model
trained for this domain, rather than assuming the default native embedding is sufficient.
Case 1 also shows that detector fine-tuning for small, motion-blurred panoramic people remains
useful.

## Final recommendation

Use YOLO11l + ByteTrack at `imgsz=960`, confidence `0.20`, and NMS IoU `0.70` as the current
baseline. It offers the best measured recall and overall identity quality without the severe
identity degradation observed at 640 or the large speed penalty and negligible quality change
at 1280/1536.

If maximum aggregate quality matters more than input size, YOLO12l at `imgsz=1536` is the new
quality winner, but its gain over YOLO11l/960 is only 0.2 percentage points in F1, MOTA, and
IDF1, and it has one additional ID switch. YOLO12l/1280 is not recommended despite its highest
recall because its eight switches reduce IDF1 to `0.659`. YOLOv10l is not competitive on this
sequence: its best configuration has lower recall/MOTA/IDF1 and substantially more switches.

If higher throughput is required, YOLO11s + ByteTrack is the more suitable compromise, but it
reduces recall and identity consistency substantially. Reducing YOLO11l to `imgsz=640` is not
recommended because its IDF1 collapse is much larger than its speed benefit would justify for
identity-sensitive tracking.

Changing YOLO generation alone will not solve the core recall issue: the three best pipelines
share 495 missed GT instances and the same GT 9 switch at frame 238. The next improvement should
combine raw-detector recall analysis/fine-tuning with tracker association tuning. Default
BoT-SORT with native ReID has already been tested and is not better than ByteTrack here; a future
ReID attempt should use tuned gates and preferably a dedicated domain-appropriate embedding.

For bandwidth-constrained deployment, physically resize frames to `1024x240` and use YOLO11l
with ByteTrack. This reduces the raw pixel payload by 75% while preserving the original-input
quality closely. Do not use `768x180` or `512x120` when identity continuity matters, and do not
use the current native-ReID configuration to compensate for aggressive downsampling.

## Reproduction

Enter the required container and activate the required environment:

```bash
sudo nerdctl exec -it stitching-dev bash
source /opt/miniconda/etc/profile.d/conda.sh
conda activate hainn_stitching
cd /workspace/evidence_aware_tracking/code
```

Run the recommended configuration:

```bash
python track_people.py \
  --sequences 0015 \
  --tracker bytetrack \
  --weights yolo11l.pt \
  --imgsz 960 \
  --conf 0.20 \
  --iou 0.70 \
  --device 0 \
  --run-name imgsz960
```

Use `--device cpu` only when reproducing the historical CPU speed measurements in the tables.

Run YOLO11l/960 with BoT-SORT and native ReID:

```bash
python track_people.py \
  --sequences 0015 \
  --tracker botsort_reid \
  --weights yolo11l.pt \
  --output-dir outputs/botsort_reid \
  --imgsz 960 \
  --conf 0.20 \
  --iou 0.70 \
  --device 0 \
  --run-name imgsz960
```

Run the physical bandwidth-downsampling matrix:

```bash
for tracker in bytetrack botsort_reid; do
  for spec in 1024x240:1024 768x180:768 512x120:512; do
    dims="${spec%%:*}"
    size="${spec##*:}"
    python track_people.py \
      --sequences 0015 \
      --tracker "${tracker}" \
      --weights yolo11l.pt \
      --output-dir outputs/bandwidth_downsample \
      --pre-resize "${dims}" \
      --imgsz "${size}" \
      --conf 0.20 \
      --iou 0.70 \
      --device 0 \
      --warmup-frames 3 \
      --run-name "pre${dims}"
  done
done
```

Run the controlled compute-only benchmark separately from video generation:

```bash
python benchmark_tracking_compute.py \
  --trackers bytetrack botsort_reid \
  --sizes 1024x240 768x180 512x120 \
  --weights yolo11l.pt \
  --device 0 \
  --warmup-frames 3 \
  --repeats 5 \
  --output outputs/bandwidth_downsample/compute_benchmark_warmup3.json
```

Run the YOLOv10l and YOLO12l resolution matrices as separate model experiments:

```bash
for size in 640 960 1280 1536; do
  python track_people.py --sequences 0015 --tracker bytetrack \
    --weights yolov10l.pt --output-dir outputs/cross_model_resolution_gpu \
    --run-name "imgsz${size}" --device 0 --imgsz "${size}" --conf 0.20 --iou 0.70
done

for size in 640 960 1280 1536; do
  python track_people.py --sequences 0015 --tracker bytetrack \
    --weights yolo12l.pt --output-dir outputs/cross_model_resolution_gpu \
    --run-name "imgsz${size}" --device 0 --imgsz "${size}" --conf 0.20 --iou 0.70
done
```

Regenerate the ID-switch evidence from an existing MOT result:

```bash
python extract_id_switch_cases.py \
  --gt ../QuadTrack/train/0015/gt/gt.txt \
  --pred outputs/resolution_comparison/0015_yolo11l_bytetrack_imgsz960.txt \
  --images ../QuadTrack/train/0015/img1 \
  --output-dir outputs/id_switch_cases \
  --iou 0.5
```

## Limitations and next steps

- Results currently cover only sequence `0015`; other sequences may have different person
  scale, lighting, motion, and distortion.
- The evaluator does not implement ignored-region handling from the complete MOTChallenge
  protocol, so the numbers should be treated as controlled baseline metrics.
- The raw-detector diagnostic currently covers the two ID-switch windows, not every frame of
  the sequence.
- Experiments 3-5 used CUDA. Their primary tables report tracking accuracy, and the application
  FPS includes image decoding, resize, tracking, annotation, and MP4 output. Experiment 5 now
  additionally reports a controlled YOLO-plus-tracker compute benchmark; it is intentionally
  not an encoded-video or network-throughput measurement.
- Cross-model miss overlap is based on final tracker output. A full-sequence raw-detector
  evaluation is still required to divide the 495 common misses into detector failures versus
  ByteTrack filtering/confirmation and IoU-threshold failures.
- The next validation should repeat the 640/960/1536 comparison on several sequences, run a raw
  YOLO confidence-threshold sweep, and tune BoT-SORT proximity/appearance thresholds or test a
  dedicated person-ReID embedding against the unchanged ByteTrack baseline.
- Pixel-count reduction is not an encoded-bandwidth measurement. A deployment study should
  measure H.264/H.265/AV1 bitrate and end-to-end network latency at the same three dimensions.
