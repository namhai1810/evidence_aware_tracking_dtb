# QuadTrack all-sequence evaluation: 512x120 + ByteTrack

Generated: 2026-10-22T12:10:20+02:00

## Configuration

- Sequences: `0015`–`0031` (17 total)
- Detector/tracker: `YOLO11l` + `ByteTrack`
- Physical input: `512x120` (6.25% of 2048x480 raw pixels)
- Inference size: `imgsz=512`
- Detector confidence / NMS IoU: `0.20` / `0.70`
- Evaluation IoU: `>= 0.50`
- Class: person only (COCO class 0; QuadTrack GT class 1)
- Aggregate policy: micro-average; identities from different sequences are treated as disjoint.

## Aggregate result

| Physical input | Raw pixels vs. original | Tracker | Precision | Recall | F1 | mIoU | MOTA | IDF1 | IDSW | TP | FP | FN |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512x120 | 6.25% | **ByteTrack** | 0.833 | 0.325 | 0.468 | 0.734 | 0.246 | 0.303 | 1243 | 29507 | 5912 | 61256 |

## Per-sequence results

| Sequence | Precision | Recall | F1 | mIoU | MOTA | IDF1 | IDSW | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0015 | 0.986 | 0.529 | 0.689 | 0.894 | 0.516 | 0.338 | 10 | 904 | 13 | 804 |
| 0016 | 0.991 | 0.454 | 0.623 | 0.851 | 0.436 | 0.281 | 23 | 771 | 7 | 928 |
| 0017 | 0.955 | 0.300 | 0.457 | 0.805 | 0.280 | 0.194 | 18 | 947 | 45 | 2206 |
| 0018 | 0.650 | 0.246 | 0.357 | 0.627 | 0.047 | 0.122 | 197 | 727 | 392 | 2229 |
| 0019 | 0.446 | 0.110 | 0.176 | 0.593 | -0.044 | 0.069 | 109 | 688 | 853 | 5573 |
| 0020 | 0.167 | 0.071 | 0.100 | 0.561 | -0.307 | 0.030 | 61 | 186 | 926 | 2421 |
| 0021 | 0.939 | 0.414 | 0.575 | 0.709 | 0.307 | 0.245 | 176 | 913 | 59 | 1292 |
| 0022 | 0.965 | 0.245 | 0.391 | 0.753 | 0.232 | 0.319 | 70 | 3838 | 140 | 11816 |
| 0023 | 0.771 | 0.228 | 0.352 | 0.722 | 0.155 | 0.232 | 77 | 3001 | 891 | 10150 |
| 0024 | 0.669 | 0.360 | 0.468 | 0.675 | 0.167 | 0.250 | 94 | 2297 | 1135 | 4085 |
| 0025 | 0.832 | 0.347 | 0.490 | 0.667 | 0.272 | 0.357 | 49 | 3336 | 676 | 6272 |
| 0026 | 0.887 | 0.378 | 0.530 | 0.692 | 0.322 | 0.417 | 56 | 2933 | 375 | 4830 |
| 0027 | 0.927 | 0.356 | 0.514 | 0.748 | 0.321 | 0.391 | 78 | 4176 | 328 | 7560 |
| 0028 | 0.959 | 0.786 | 0.864 | 0.788 | 0.639 | 0.240 | 157 | 1096 | 47 | 299 |
| 0029 | 0.997 | 0.938 | 0.967 | 0.805 | 0.914 | 0.428 | 26 | 1126 | 3 | 74 |
| 0030 | 0.986 | 0.684 | 0.808 | 0.808 | 0.658 | 0.585 | 31 | 1299 | 18 | 600 |
| 0031 | 0.997 | 0.916 | 0.954 | 0.800 | 0.905 | 0.727 | 11 | 1269 | 4 | 117 |

## Incorrect-frame log

A frame is logged when evaluator matching at the configured IoU contains at least one false negative, false positive, or identity switch. A single frame can belong to multiple categories.

- Unique incorrect frames: **8195**
- Frames with at least one FN: **8030**
- Frames with at least one FP: **3708**
- Frames with at least one ID switch: **1130**

| Sequence | Incorrect frames | With FN | With FP | With ID switch |
|---|---:|---:|---:|---:|
| 0015 | 357 | 355 | 13 | 8 |
| 0016 | 409 | 405 | 7 | 22 |
| 0017 | 596 | 596 | 41 | 18 |
| 0018 | 600 | 599 | 303 | 171 |
| 0019 | 600 | 600 | 477 | 101 |
| 0020 | 599 | 599 | 558 | 55 |
| 0021 | 409 | 407 | 55 | 146 |
| 0022 | 600 | 600 | 118 | 66 |
| 0023 | 600 | 600 | 527 | 73 |
| 0024 | 600 | 600 | 497 | 89 |
| 0025 | 600 | 600 | 505 | 48 |
| 0026 | 600 | 600 | 266 | 54 |
| 0027 | 600 | 600 | 269 | 70 |
| 0028 | 338 | 224 | 47 | 148 |
| 0029 | 77 | 53 | 3 | 22 |
| 0030 | 506 | 492 | 18 | 30 |
| 0031 | 104 | 100 | 4 | 9 |

The JSON log retains bounding boxes, confidence, matched GT/pred IDs and IoUs, plus the previous matched frame and gap length for every ID switch. The CSV is a compact frame index for filtering.

## Artifacts

- [Machine-readable metrics](all_sequences_512x120/metrics_iou_0.5.json)
- [Detailed incorrect-frame log](all_sequences_512x120/failure_frames.json)
- [Compact incorrect-frame CSV](all_sequences_512x120/failure_frames.csv)
- MOT result files and annotated videos: `all_sequences_512x120/predictions/`
