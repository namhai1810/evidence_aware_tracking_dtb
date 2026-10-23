# QuadTrack all-sequence evaluation: 2048x480 + BoT-SORT+ReID

Generated: 2026-10-22T13:10:44+02:00

## Configuration

- Sequences: `0015`–`0031` (17 total)
- Detector/tracker: `YOLO11l` + `BoT-SORT+ReID`
- Physical input: `2048x480` (100% of 2048x480 raw pixels)
- Inference size: `imgsz=960`
- Detector confidence / NMS IoU: `0.20` / `0.70`
- Evaluation IoU: `>= 0.50`
- Class: person only (COCO class 0; QuadTrack GT class 1)
- Aggregate policy: micro-average; identities from different sequences are treated as disjoint.

## Aggregate result

| Physical input | Raw pixels vs. original | Tracker | Precision | Recall | F1 | mIoU | MOTA | IDF1 | IDSW | TP | FP | FN |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048x480 | 100% | **BoT-SORT+ReID** | 0.741 | 0.486 | 0.587 | 0.730 | 0.298 | 0.419 | 1657 | 44155 | 15437 | 46608 |

## Per-sequence results

| Sequence | Precision | Recall | F1 | mIoU | MOTA | IDF1 | IDSW | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0015 | 0.988 | 0.697 | 0.818 | 0.859 | 0.687 | 0.816 | 3 | 1191 | 14 | 517 |
| 0016 | 0.985 | 0.614 | 0.757 | 0.824 | 0.587 | 0.196 | 30 | 1044 | 16 | 655 |
| 0017 | 0.940 | 0.337 | 0.497 | 0.771 | 0.296 | 0.270 | 62 | 1064 | 68 | 2089 |
| 0018 | 0.484 | 0.325 | 0.389 | 0.638 | -0.066 | 0.209 | 132 | 960 | 1023 | 1996 |
| 0019 | 0.515 | 0.166 | 0.251 | 0.605 | -0.006 | 0.133 | 94 | 1040 | 981 | 5221 |
| 0020 | 0.280 | 0.182 | 0.221 | 0.578 | -0.311 | 0.116 | 66 | 474 | 1218 | 2133 |
| 0021 | 0.898 | 0.562 | 0.691 | 0.713 | 0.449 | 0.300 | 108 | 1239 | 141 | 966 |
| 0022 | 0.911 | 0.394 | 0.550 | 0.755 | 0.339 | 0.382 | 257 | 6173 | 603 | 9481 |
| 0023 | 0.574 | 0.366 | 0.447 | 0.714 | 0.084 | 0.332 | 146 | 4817 | 3572 | 8334 |
| 0024 | 0.495 | 0.541 | 0.517 | 0.674 | -0.027 | 0.391 | 103 | 3453 | 3524 | 2929 |
| 0025 | 0.780 | 0.588 | 0.670 | 0.690 | 0.399 | 0.500 | 218 | 5651 | 1598 | 3957 |
| 0026 | 0.767 | 0.631 | 0.692 | 0.707 | 0.422 | 0.501 | 135 | 4899 | 1490 | 2864 |
| 0027 | 0.866 | 0.596 | 0.706 | 0.735 | 0.481 | 0.487 | 256 | 6990 | 1085 | 4746 |
| 0028 | 0.956 | 0.933 | 0.945 | 0.811 | 0.887 | 0.786 | 5 | 1302 | 60 | 93 |
| 0029 | 0.991 | 0.964 | 0.977 | 0.813 | 0.953 | 0.890 | 2 | 1157 | 11 | 43 |
| 0030 | 0.990 | 0.745 | 0.850 | 0.817 | 0.719 | 0.702 | 34 | 1414 | 14 | 485 |
| 0031 | 0.985 | 0.929 | 0.956 | 0.813 | 0.911 | 0.949 | 6 | 1287 | 19 | 99 |

## Incorrect-frame log

A frame is logged when evaluator matching at the configured IoU contains at least one false negative, false positive, or identity switch. A single frame can belong to multiple categories.

- Unique incorrect frames: **7766**
- Frames with at least one FN: **7629**
- Frames with at least one FP: **5064**
- Frames with at least one ID switch: **1363**

| Sequence | Incorrect frames | With FN | With FP | With ID switch |
|---|---:|---:|---:|---:|
| 0015 | 305 | 300 | 12 | 3 |
| 0016 | 358 | 353 | 16 | 24 |
| 0017 | 595 | 595 | 65 | 56 |
| 0018 | 598 | 593 | 503 | 110 |
| 0019 | 600 | 600 | 501 | 80 |
| 0020 | 593 | 591 | 554 | 60 |
| 0021 | 398 | 389 | 112 | 90 |
| 0022 | 600 | 600 | 372 | 195 |
| 0023 | 600 | 600 | 595 | 122 |
| 0024 | 600 | 600 | 600 | 98 |
| 0025 | 600 | 597 | 599 | 170 |
| 0026 | 600 | 597 | 519 | 119 |
| 0027 | 600 | 600 | 513 | 195 |
| 0028 | 135 | 83 | 59 | 5 |
| 0029 | 46 | 34 | 11 | 2 |
| 0030 | 427 | 401 | 14 | 28 |
| 0031 | 111 | 96 | 19 | 6 |

The JSON log retains bounding boxes, confidence, matched GT/pred IDs and IoUs, plus the previous matched frame and gap length for every ID switch. The CSV is a compact frame index for filtering.

## Artifacts

- [Machine-readable metrics](all_sequences_reid_original_2048x480/metrics_iou_0.5.json)
- [Detailed incorrect-frame log](all_sequences_reid_original_2048x480/failure_frames.json)
- [Compact incorrect-frame CSV](all_sequences_reid_original_2048x480/failure_frames.csv)
- MOT result files and annotated videos: `all_sequences_reid_original_2048x480/predictions/`
