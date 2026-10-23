# QuadTrack all-sequence evaluation: 2048x480 + ByteTrack

Generated: 2026-10-22T12:19:23+02:00

## Configuration

- Sequences: `0015`–`0031` (17 total)
- Detector/tracker: `YOLO11l` + `ByteTrack`
- Physical input: `2048x480` (100% of 2048x480 raw pixels)
- Inference size: `imgsz=960`
- Detector confidence / NMS IoU: `0.20` / `0.70`
- Evaluation IoU: `>= 0.50`
- Class: person only (COCO class 0; QuadTrack GT class 1)
- Aggregate policy: micro-average; identities from different sequences are treated as disjoint.

## Aggregate result

| Physical input | Raw pixels vs. original | Tracker | Precision | Recall | F1 | mIoU | MOTA | IDF1 | IDSW | TP | FP | FN |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048x480 | 100% | **ByteTrack** | 0.762 | 0.476 | 0.586 | 0.734 | 0.306 | 0.398 | 1967 | 43241 | 13483 | 47522 |

## Per-sequence results

| Sequence | Precision | Recall | F1 | mIoU | MOTA | IDF1 | IDSW | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0015 | 0.991 | 0.696 | 0.818 | 0.890 | 0.689 | 0.816 | 2 | 1189 | 11 | 519 |
| 0016 | 0.982 | 0.624 | 0.764 | 0.851 | 0.597 | 0.280 | 27 | 1061 | 19 | 638 |
| 0017 | 0.946 | 0.377 | 0.539 | 0.780 | 0.343 | 0.205 | 40 | 1189 | 68 | 1964 |
| 0018 | 0.544 | 0.313 | 0.397 | 0.640 | -0.047 | 0.127 | 287 | 924 | 775 | 2032 |
| 0019 | 0.525 | 0.163 | 0.249 | 0.604 | -0.017 | 0.100 | 202 | 1021 | 924 | 5240 |
| 0020 | 0.327 | 0.179 | 0.232 | 0.567 | -0.237 | 0.056 | 125 | 467 | 960 | 2140 |
| 0021 | 0.891 | 0.487 | 0.630 | 0.724 | 0.329 | 0.221 | 217 | 1073 | 131 | 1132 |
| 0022 | 0.914 | 0.384 | 0.541 | 0.758 | 0.335 | 0.391 | 209 | 6011 | 565 | 9643 |
| 0023 | 0.601 | 0.360 | 0.450 | 0.719 | 0.112 | 0.331 | 120 | 4736 | 3139 | 8415 |
| 0024 | 0.519 | 0.541 | 0.530 | 0.678 | 0.025 | 0.397 | 97 | 3452 | 3193 | 2930 |
| 0025 | 0.798 | 0.564 | 0.661 | 0.694 | 0.407 | 0.519 | 137 | 5418 | 1374 | 4190 |
| 0026 | 0.783 | 0.619 | 0.691 | 0.706 | 0.434 | 0.473 | 103 | 4802 | 1332 | 2961 |
| 0027 | 0.881 | 0.577 | 0.697 | 0.740 | 0.483 | 0.518 | 190 | 6773 | 915 | 4963 |
| 0028 | 0.969 | 0.896 | 0.931 | 0.800 | 0.763 | 0.336 | 145 | 1250 | 40 | 145 |
| 0029 | 0.997 | 0.952 | 0.974 | 0.805 | 0.941 | 0.493 | 10 | 1142 | 3 | 58 |
| 0030 | 0.992 | 0.762 | 0.862 | 0.814 | 0.738 | 0.761 | 35 | 1447 | 11 | 452 |
| 0031 | 0.982 | 0.928 | 0.954 | 0.813 | 0.896 | 0.623 | 21 | 1286 | 23 | 100 |

## Incorrect-frame log

A frame is logged when evaluator matching at the configured IoU contains at least one false negative, false positive, or identity switch. A single frame can belong to multiple categories.

- Unique incorrect frames: **7861**
- Frames with at least one FN: **7658**
- Frames with at least one FP: **4869**
- Frames with at least one ID switch: **1664**

| Sequence | Incorrect frames | With FN | With FP | With ID switch |
|---|---:|---:|---:|---:|
| 0015 | 304 | 300 | 9 | 2 |
| 0016 | 355 | 349 | 19 | 23 |
| 0017 | 594 | 594 | 60 | 33 |
| 0018 | 597 | 592 | 457 | 238 |
| 0019 | 599 | 599 | 475 | 166 |
| 0020 | 598 | 595 | 520 | 107 |
| 0021 | 406 | 397 | 120 | 169 |
| 0022 | 600 | 600 | 359 | 176 |
| 0023 | 600 | 600 | 592 | 103 |
| 0024 | 600 | 600 | 600 | 92 |
| 0025 | 600 | 600 | 591 | 120 |
| 0026 | 600 | 599 | 514 | 94 |
| 0027 | 600 | 600 | 476 | 150 |
| 0028 | 250 | 126 | 40 | 134 |
| 0029 | 50 | 43 | 3 | 8 |
| 0030 | 402 | 378 | 11 | 32 |
| 0031 | 106 | 86 | 23 | 17 |

The JSON log retains bounding boxes, confidence, matched GT/pred IDs and IoUs, plus the previous matched frame and gap length for every ID switch. The CSV is a compact frame index for filtering.

## Artifacts

- [Machine-readable metrics](all_sequences_original_2048x480/metrics_iou_0.5.json)
- [Detailed incorrect-frame log](all_sequences_original_2048x480/failure_frames.json)
- [Compact incorrect-frame CSV](all_sequences_original_2048x480/failure_frames.csv)
- MOT result files and annotated videos: `all_sequences_original_2048x480/predictions/`
