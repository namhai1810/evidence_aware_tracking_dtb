# High-resolution retrieval using GT/Hungarian oracle labels: 512x120 BoT-SORT+ReID

Generated: 2026-10-22T13:40:41+02:00

## Experiment

For every ID-switch event produced by the 512x120 YOLO11l + BoT-SORT+ReID pipeline, this experiment searches the original-resolution BoT-SORT+ReID tracking result between 5 and 10 seconds before the switch. Predictions are assigned to GT identities with Hungarian matching at IoU >= 0.50; those GT-linked identities create the oracle labels. This report does not perform cross-resolution embedding matching.

- Evaluation IoU: `0.50`
- Frame rate: `30.0` FPS
- Retrieval interval: `5.0`–`10.0` seconds before the switch
- Candidate interval: `150`–`300` frames before the switch
- Retrieval succeeds when at least `1` helpful HR frame is found.
- Up to `5` frames are retained per switch as ranked alternatives; 5 helpful frames are not required.
- When multiple frames are available, ranking prefers at least `15` frames of spacing to reduce near-duplicates.

## Oracle retrieval potential

| Measure | Count | Fraction |
|---|---:|---:|
| Low-resolution ID-switch events | 925 | 100.0% |
| Unique low-resolution switch frames | 861 | 100.0% |
| Events with at least one HR candidate in the window | 387 | 41.8% |
| Events where HR is matched and stable at target | 628 | 67.9% |
| Recoverable events with at least one helpful HR frame | **111** | **12.0%** |
| Recoverable unique target frames with at least one helpful HR frame | **108** | **12.5%** |
| Events with five or more helpful HR alternatives | 109 | 11.8% |

Overlapping diagnostics for non-recoverable cases (these counts are not mutually exclusive):

- `237` switches occur too early to have a complete candidate interval.
- `161` switches have no original-resolution match at the target frame.
- `136` switches are also identity-unstable in the original-resolution stream.
- `517` switches have a stable HR target but no same-track HR candidate in the requested interval.

If every oracle-recoverable switch were removed while keeping FN and FP unchanged, IDSW would fall from `925` to `814`, and MOTA would rise from `0.253` to at most `0.255`. IDF1 is not estimated because it requires actually merging complete predicted track segments, not editing isolated switch counts.

## Comparison with ByteTrack

| Pipeline | Initial IDSW | Oracle-recoverable | Recovery rate | Oracle remaining IDSW | MOTA | Oracle MOTA upper bound |
|---|---:|---:|---:|---:|---:|---:|
| ByteTrack | 1243 | 137 | 11.0% | 1106 | 0.246 | 0.248 |
| BoT-SORT+ReID | 925 | 111 | 12.0% | 814 | 0.253 | 0.255 |

Relative to the original ByteTrack starting point, using BoT-SORT+ReID and then resolving every oracle-recoverable event would reduce IDSW by **429 (34.5%)**, from `1243` to `814`. This is an oracle upper bound across two tracker configurations, not a measured online merge result.

## Per-sequence retrieval yield

| Sequence | LR IDSW events | Recoverable with ≥1 frame | Has ≥5 alternatives | Candidate rows |
|---|---:|---:|---:|---:|
| 0015 | 15 | 14 | 14 | 1935 |
| 0016 | 13 | 0 | 0 | 1306 |
| 0017 | 17 | 0 | 0 | 157 |
| 0018 | 96 | 3 | 3 | 1695 |
| 0019 | 41 | 1 | 1 | 705 |
| 0020 | 27 | 0 | 0 | 678 |
| 0021 | 109 | 1 | 1 | 2878 |
| 0022 | 92 | 11 | 11 | 3477 |
| 0023 | 97 | 22 | 22 | 3045 |
| 0024 | 105 | 22 | 22 | 4656 |
| 0025 | 61 | 15 | 14 | 2525 |
| 0026 | 83 | 6 | 5 | 1601 |
| 0027 | 133 | 12 | 12 | 3180 |
| 0028 | 7 | 4 | 4 | 592 |
| 0029 | 0 | 0 | 0 | 0 |
| 0030 | 28 | 0 | 0 | 1951 |
| 0031 | 1 | 0 | 0 | 0 |

## Example ranked retrievals

| Event | Target | GT ID | LR ID change | Ranked HR frames (seconds before target) |
|---|---:|---:|---|---|
| 0015_0002 | 251 | 2 | 2 → 38 | 83 (5.6s), 98 (5.1s), 52 (6.6s), 12 (8.0s), 31 (7.3s) |
| 0015_0003 | 252 | 2 | 38 → 2 | 83 (5.6s), 98 (5.1s), 52 (6.7s), 12 (8.0s), 31 (7.4s) |
| 0015_0004 | 261 | 2 | 2 → 42 | 109 (5.1s), 83 (5.9s), 52 (7.0s), 12 (8.3s), 31 (7.7s) |
| 0015_0005 | 272 | 2 | 42 → 44 | 115 (5.2s), 83 (6.3s), 98 (5.8s), 52 (7.3s), 12 (8.7s) |
| 0015_0006 | 293 | 1 | 9 → 45 | 135 (5.3s), 108 (6.2s), 48 (8.2s), 64 (7.6s), 86 (6.9s) |
| 0015_0007 | 293 | 2 | 44 → 46 | 115 (5.9s), 135 (5.3s), 83 (7.0s), 98 (6.5s), 52 (8.0s) |
| 0015_0008 | 321 | 1 | 45 → 46 | 166 (5.2s), 147 (5.8s), 108 (7.1s), 126 (6.5s), 48 (9.1s) |
| 0015_0009 | 322 | 2 | 46 → 52 | 151 (5.7s), 115 (6.9s), 135 (6.2s), 166 (5.2s), 83 (8.0s) |
| 0015_0010 | 338 | 1 | 46 → 56 | 177 (5.4s), 162 (5.9s), 147 (6.4s), 108 (7.7s), 126 (7.1s) |
| 0015_0011 | 339 | 2 | 52 → 57 | 151 (6.3s), 115 (7.5s), 177 (5.4s), 135 (6.8s), 83 (8.5s) |

## Ranking data

Candidates are labeled with `oracle_helpful`. A switch is counted as recoverable as soon as at least one candidate has this label. The ranking prior then uses HR match IoU, detector confidence, local same-track support, box area, and temporal proximity. IoU and GT identity are supervision signals and must not be used as online inputs to the future ranking model. Deployable inputs can include confidence, crop size, sharpness, motion/occlusion state, and an appearance embedding.

The output retains up to five candidates as alternatives, but the runtime system may stop after retrieving the first sufficiently strong frame. Multiple candidates are useful only when the first frame is blurred, occluded, ambiguous, or fails the ReID threshold.

## Deployment implication

The current BoT-SORT+ReID pipeline does not automatically consume an isolated frame retrieved from 5–10 seconds ago as long-term identity evidence. A deployable version needs a retrieval-aware association layer: trigger on tracking uncertainty, request ranked HR crops one by one, extract an appearance embedding, and stop as soon as one frame gives a sufficiently confident identity match. Additional ranked frames are fallback evidence before merging or preserving the identity.

This report is therefore an oracle upper-bound and ranking-dataset experiment, not a claim that inserting isolated HR frames into the existing BoT-SORT+ReID loop already produces these gains.

## Artifacts

- [Summary](highres_retrieval_hungarian_gt_oracle_botsort_reid/summary.json)
- [All switch events with top-five frames](highres_retrieval_hungarian_gt_oracle_botsort_reid/switch_events_top5.json)
- [Top-five frame table](highres_retrieval_hungarian_gt_oracle_botsort_reid/top5_frames.csv)
- [All ranking candidates and labels](highres_retrieval_hungarian_gt_oracle_botsort_reid/ranking_candidates.csv)
