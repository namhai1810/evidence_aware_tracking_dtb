# QuadTrack all-sequence evaluation: 512x120 + BoT-SORT+ReID

Generated: 2026-10-22T13:10:42+02:00

## Configuration

- Sequences: `0015`–`0031` (17 total)
- Detector/tracker: `YOLO11l` + `BoT-SORT+ReID`
- Physical input: `512x120` (6.25% of 2048x480 raw pixels)
- Inference size: `imgsz=512`
- Detector confidence / NMS IoU: `0.20` / `0.70`
- Evaluation IoU: `>= 0.50`
- Class: person only (COCO class 0; QuadTrack GT class 1)
- Aggregate policy: micro-average; identities from different sequences are treated as disjoint.

## Aggregate result

| Physical input | Raw pixels vs. original | Tracker | Precision | Recall | F1 | mIoU | MOTA | IDF1 | IDSW | TP | FP | FN |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512x120 | 6.25% | **BoT-SORT+ReID** | 0.818 | 0.339 | 0.480 | 0.730 | 0.253 | 0.353 | 925 | 30794 | 6861 | 59969 |

## Per-sequence results

| Sequence | Precision | Recall | F1 | mIoU | MOTA | IDF1 | IDSW | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0015 | 0.981 | 0.523 | 0.683 | 0.881 | 0.505 | 0.319 | 15 | 894 | 17 | 814 |
| 0016 | 0.994 | 0.469 | 0.637 | 0.836 | 0.458 | 0.302 | 13 | 796 | 5 | 903 |
| 0017 | 0.941 | 0.317 | 0.474 | 0.795 | 0.291 | 0.338 | 17 | 998 | 63 | 2155 |
| 0018 | 0.603 | 0.270 | 0.373 | 0.632 | 0.060 | 0.184 | 96 | 797 | 525 | 2159 |
| 0019 | 0.429 | 0.119 | 0.186 | 0.591 | -0.046 | 0.138 | 41 | 744 | 989 | 5517 |
| 0020 | 0.195 | 0.092 | 0.125 | 0.567 | -0.298 | 0.073 | 27 | 240 | 990 | 2367 |
| 0021 | 0.927 | 0.475 | 0.628 | 0.707 | 0.388 | 0.306 | 109 | 1048 | 83 | 1157 |
| 0022 | 0.969 | 0.252 | 0.401 | 0.747 | 0.239 | 0.283 | 92 | 3952 | 125 | 11702 |
| 0023 | 0.754 | 0.240 | 0.364 | 0.717 | 0.154 | 0.253 | 97 | 3151 | 1030 | 10000 |
| 0024 | 0.660 | 0.372 | 0.476 | 0.670 | 0.164 | 0.296 | 105 | 2371 | 1219 | 4011 |
| 0025 | 0.820 | 0.360 | 0.501 | 0.669 | 0.275 | 0.410 | 61 | 3462 | 761 | 6146 |
| 0026 | 0.846 | 0.391 | 0.535 | 0.691 | 0.309 | 0.410 | 83 | 3036 | 552 | 4727 |
| 0027 | 0.915 | 0.369 | 0.526 | 0.742 | 0.324 | 0.370 | 133 | 4334 | 401 | 7402 |
| 0028 | 0.952 | 0.872 | 0.910 | 0.786 | 0.823 | 0.881 | 7 | 1217 | 62 | 178 |
| 0029 | 0.996 | 0.956 | 0.975 | 0.806 | 0.952 | 0.975 | 0 | 1147 | 5 | 53 |
| 0030 | 0.979 | 0.700 | 0.816 | 0.812 | 0.670 | 0.790 | 28 | 1329 | 28 | 570 |
| 0031 | 0.995 | 0.922 | 0.957 | 0.801 | 0.917 | 0.953 | 1 | 1278 | 6 | 108 |

## Incorrect-frame log

A frame is logged when evaluator matching at the configured IoU contains at least one false negative, false positive, or identity switch. A single frame can belong to multiple categories.

- Unique incorrect frames: **7997**
- Frames with at least one FN: **7910**
- Frames with at least one FP: **4003**
- Frames with at least one ID switch: **861**

| Sequence | Incorrect frames | With FN | With FP | With ID switch |
|---|---:|---:|---:|---:|
| 0015 | 359 | 357 | 17 | 13 |
| 0016 | 400 | 398 | 5 | 13 |
| 0017 | 596 | 596 | 58 | 17 |
| 0018 | 600 | 599 | 330 | 89 |
| 0019 | 600 | 600 | 521 | 37 |
| 0020 | 598 | 598 | 570 | 25 |
| 0021 | 407 | 398 | 78 | 102 |
| 0022 | 600 | 600 | 117 | 81 |
| 0023 | 600 | 600 | 530 | 93 |
| 0024 | 600 | 600 | 499 | 99 |
| 0025 | 600 | 600 | 526 | 59 |
| 0026 | 600 | 600 | 350 | 80 |
| 0027 | 600 | 600 | 304 | 118 |
| 0028 | 183 | 141 | 60 | 7 |
| 0029 | 46 | 41 | 5 | 0 |
| 0030 | 501 | 480 | 27 | 27 |
| 0031 | 107 | 102 | 6 | 1 |

The JSON log retains bounding boxes, confidence, matched GT/pred IDs and IoUs, plus the previous matched frame and gap length for every ID switch. The CSV is a compact frame index for filtering.

## Artifacts

- [Machine-readable metrics](all_sequences_reid_512x120/metrics_iou_0.5.json)
- [Detailed incorrect-frame log](all_sequences_reid_512x120/failure_frames.json)
- [Compact incorrect-frame CSV](all_sequences_reid_512x120/failure_frames.csv)
- MOT result files and annotated videos: `all_sequences_reid_512x120/predictions/`
