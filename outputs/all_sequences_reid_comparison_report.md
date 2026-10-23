# QuadTrack: ByteTrack vs BoT-SORT + ReID

Generated: 2026-10-22T14:08:15+02:00

## Scope and configuration

- Dataset: all 17 QuadTrack train sequences (`0015`–`0031`).
- Detector: YOLO11l; person class only; confidence `0.20`; NMS IoU `0.70`.
- Evaluation: matching IoU `>=0.50`, micro-aggregated across sequences.
- Low-resolution run: physical input `512x120`, inference `imgsz=512`.
- Original run: physical input `2048x480`, inference `imgsz=960`.
- ReID run: Ultralytics BoT-SORT with `with_reid: true`, `model: auto`, and sparse optical-flow GMC.
- Important: this compares the full trackers. Because ByteTrack and BoT-SORT also differ in matching and GMC, it is not a controlled appearance-ReID-only ablation.

## Aggregate metrics

| Input / tracker | Precision | Recall | F1 | mIoU | MOTA | IDF1 | IDSW | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512x120 / ByteTrack | 0.833 | 0.325 | 0.468 | 0.734 | 0.246 | 0.303 | 1243 | 29507 | 5912 | 61256 |
| 512x120 / BoT-SORT+ReID | 0.818 | 0.339 | 0.480 | 0.730 | 0.253 | 0.353 | 925 | 30794 | 6861 | 59969 |
| 2048x480 / ByteTrack | 0.762 | 0.476 | 0.586 | 0.734 | 0.306 | 0.398 | 1967 | 43241 | 13483 | 47522 |
| 2048x480 / BoT-SORT+ReID | 0.741 | 0.486 | 0.587 | 0.730 | 0.298 | 0.419 | 1657 | 44155 | 15437 | 46608 |

## ReID delta relative to ByteTrack

| Input | Δ Precision | Δ Recall | Δ F1 | Δ mIoU | Δ MOTA | Δ IDF1 | Δ IDSW | Δ TP | Δ FP | Δ FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512x120 | -0.015 | +0.014 | +0.012 | -0.004 | +0.007 | +0.050 | -318 | +1287 | +949 | -1287 |
| 2048x480 | -0.021 | +0.010 | +0.001 | -0.004 | -0.008 | +0.022 | -310 | +914 | +1954 | -914 |

## Main observations

- At `512x120`, ReID reduces ID switches by **318 (25.6%)**, raises IDF1 by **+0.050**, and raises MOTA by **+0.007**.
- At `2048x480`, ReID reduces ID switches by **310 (15.8%)** and raises IDF1 by **+0.022**.
- Original-resolution MOTA changes by **-0.008** because the ReID run adds +1954 FP while recovering +914 FN. This tracker setting improves identity continuity but not overall detection error at that resolution.
- For the planned edge-retrieval system, IDF1 and IDSW are the most directly relevant aggregate signals; the frame-level logs below contain the exact switch/FN/FP evidence for ranking experiments.

## Per-sequence comparison: 512x120

| Seq | ByteTrack MOTA | ReID MOTA | Δ MOTA | ByteTrack IDF1 | ReID IDF1 | Δ IDF1 | ByteTrack IDSW | ReID IDSW | Δ IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0015 | 0.516 | 0.505 | -0.011 | 0.338 | 0.319 | -0.018 | 10 | 15 | +5 |
| 0016 | 0.436 | 0.458 | +0.022 | 0.281 | 0.302 | +0.021 | 23 | 13 | -10 |
| 0017 | 0.280 | 0.291 | +0.011 | 0.194 | 0.338 | +0.143 | 18 | 17 | -1 |
| 0018 | 0.047 | 0.060 | +0.013 | 0.122 | 0.184 | +0.062 | 197 | 96 | -101 |
| 0019 | -0.044 | -0.046 | -0.002 | 0.069 | 0.138 | +0.069 | 109 | 41 | -68 |
| 0020 | -0.307 | -0.298 | +0.009 | 0.030 | 0.073 | +0.043 | 61 | 27 | -34 |
| 0021 | 0.307 | 0.388 | +0.081 | 0.245 | 0.306 | +0.061 | 176 | 109 | -67 |
| 0022 | 0.232 | 0.239 | +0.007 | 0.319 | 0.283 | -0.036 | 70 | 92 | +22 |
| 0023 | 0.155 | 0.154 | -0.001 | 0.232 | 0.253 | +0.021 | 77 | 97 | +20 |
| 0024 | 0.167 | 0.164 | -0.003 | 0.250 | 0.296 | +0.046 | 94 | 105 | +11 |
| 0025 | 0.272 | 0.275 | +0.003 | 0.357 | 0.410 | +0.053 | 49 | 61 | +12 |
| 0026 | 0.322 | 0.309 | -0.013 | 0.417 | 0.410 | -0.007 | 56 | 83 | +27 |
| 0027 | 0.321 | 0.324 | +0.003 | 0.391 | 0.370 | -0.021 | 78 | 133 | +55 |
| 0028 | 0.639 | 0.823 | +0.184 | 0.240 | 0.881 | +0.641 | 157 | 7 | -150 |
| 0029 | 0.914 | 0.952 | +0.037 | 0.428 | 0.975 | +0.548 | 26 | 0 | -26 |
| 0030 | 0.658 | 0.670 | +0.012 | 0.585 | 0.790 | +0.205 | 31 | 28 | -3 |
| 0031 | 0.905 | 0.917 | +0.012 | 0.727 | 0.953 | +0.226 | 11 | 1 | -10 |

## Per-sequence comparison: 2048x480

| Seq | ByteTrack MOTA | ReID MOTA | Δ MOTA | ByteTrack IDF1 | ReID IDF1 | Δ IDF1 | ByteTrack IDSW | ReID IDSW | Δ IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0015 | 0.689 | 0.687 | -0.001 | 0.816 | 0.816 | -0.000 | 2 | 3 | +1 |
| 0016 | 0.597 | 0.587 | -0.010 | 0.280 | 0.196 | -0.084 | 27 | 30 | +3 |
| 0017 | 0.343 | 0.296 | -0.047 | 0.205 | 0.270 | +0.065 | 40 | 62 | +22 |
| 0018 | -0.047 | -0.066 | -0.019 | 0.127 | 0.209 | +0.082 | 287 | 132 | -155 |
| 0019 | -0.017 | -0.006 | +0.011 | 0.100 | 0.133 | +0.033 | 202 | 94 | -108 |
| 0020 | -0.237 | -0.311 | -0.074 | 0.056 | 0.116 | +0.060 | 125 | 66 | -59 |
| 0021 | 0.329 | 0.449 | +0.120 | 0.221 | 0.300 | +0.079 | 217 | 108 | -109 |
| 0022 | 0.335 | 0.339 | +0.005 | 0.391 | 0.382 | -0.009 | 209 | 257 | +48 |
| 0023 | 0.112 | 0.084 | -0.029 | 0.331 | 0.332 | +0.002 | 120 | 146 | +26 |
| 0024 | 0.025 | -0.027 | -0.053 | 0.397 | 0.391 | -0.006 | 97 | 103 | +6 |
| 0025 | 0.407 | 0.399 | -0.007 | 0.519 | 0.500 | -0.019 | 137 | 218 | +81 |
| 0026 | 0.434 | 0.422 | -0.012 | 0.473 | 0.501 | +0.028 | 103 | 135 | +32 |
| 0027 | 0.483 | 0.481 | -0.002 | 0.518 | 0.487 | -0.031 | 190 | 256 | +66 |
| 0028 | 0.763 | 0.887 | +0.123 | 0.336 | 0.786 | +0.450 | 145 | 5 | -140 |
| 0029 | 0.941 | 0.953 | +0.013 | 0.493 | 0.890 | +0.397 | 10 | 2 | -8 |
| 0030 | 0.738 | 0.719 | -0.018 | 0.761 | 0.702 | -0.060 | 35 | 34 | -1 |
| 0031 | 0.896 | 0.911 | +0.014 | 0.623 | 0.949 | +0.327 | 21 | 6 | -15 |

## Historical GT/Hungarian retrieval oracle

This older experiment links the low- and full-resolution streams through GT identities assigned with Hungarian IoU matching. It does not compare cross-resolution appearance embeddings.

- Low-resolution BoT-SORT+ReID ID-switch events: **925**.
- Events recoverable using at least one original-resolution history frame: **111 (12.0%)**.
- Recoverable unique target frames: **108 (12.5%)**.
- Oracle IDSW upper bound: **925 → 814**; MOTA **0.253 → 0.255** if FN and FP stay fixed.
- This is supervision/ranking potential, not a measured online identity merge.

See [the explicitly named GT/Hungarian oracle report](highres_retrieval_hungarian_gt_oracle_botsort_reid_report.md).

## Cross-resolution ReID embedding retrieval

This experiment ranks detections from the immediately preceding 0–5 seconds of the 2048x480 stream (`T-150` through `T-1` at 30 FPS) by cosine similarity to the old-track and target embeddings extracted from the physical 512x120 stream. The nearest frame is visited first; GT is used only to score the ranking.

- Switch events: **925**; correct HR identity available: **920 (99.5%)**.
- Correct top-1 retrieval: **514 (55.6%)**.
- Correct identity in top five: **706 (76.3%)**.
- Retrieval-supported IDSW estimates: **925 → 411** for top-1, or **→ 219** with an oracle choice among top five.

See [the cross-resolution embedding report](highres_retrieval_reid_embedding_report.md).

## ReID artifacts

- [512x120 detailed report](all_sequences_reid_512x120_report.md)
- [512x120 metrics JSON](all_sequences_reid_512x120/metrics_iou_0.5.json)
- [512x120 failure CSV](all_sequences_reid_512x120/failure_frames.csv)
- [512x120 failure JSON](all_sequences_reid_512x120/failure_frames.json)
- [2048x480 detailed report](all_sequences_reid_original_2048x480_report.md)
- [2048x480 metrics JSON](all_sequences_reid_original_2048x480/metrics_iou_0.5.json)
- [2048x480 failure CSV](all_sequences_reid_original_2048x480/failure_frames.csv)
- [2048x480 failure JSON](all_sequences_reid_original_2048x480/failure_frames.json)
- Annotated videos and MOT-format predictions are in each run's `predictions/` directory.
