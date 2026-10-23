# High-resolution retrieval using GT/Hungarian oracle labels: 512x120 ByteTrack

Generated: 2026-10-22T12:47:50+02:00

## Experiment

For every ID-switch event produced by the 512x120 YOLO11l + ByteTrack pipeline, this experiment searches the original-resolution tracking result between 5 and 10 seconds before the switch. Predictions are assigned to GT identities with Hungarian matching at IoU >= 0.50; those GT-linked identities create the oracle labels. This report does not perform cross-resolution embedding matching.

- Evaluation IoU: `0.50`
- Frame rate: `30.0` FPS
- Retrieval interval: `5.0`–`10.0` seconds before the switch
- Candidate interval: `150`–`300` frames before the switch
- Retrieval succeeds when at least `1` helpful HR frame is found.
- Up to `5` frames are retained per switch as ranked alternatives; five helpful frames are not required.
- When multiple frames are available, the ranking prefers at least `15` frames of spacing to reduce near-duplicates.

## Oracle retrieval potential

| Measure | Count | Fraction |
|---|---:|---:|
| Low-resolution ID-switch events | 1243 | 100.0% |
| Unique low-resolution switch frames | 1130 | 100.0% |
| Events with at least one HR candidate in the window | 692 | 55.7% |
| Events where HR is matched and stable at target | 715 | 57.5% |
| Recoverable events with at least one helpful HR frame | **137** | **11.0%** |
| Recoverable unique target frames with at least one helpful HR frame | **134** | **11.9%** |
| Events with five or more helpful HR alternatives | 132 | 10.6% |

Overlapping diagnostics for non-recoverable cases (these counts are not mutually exclusive):

- `282` switches occur too early to have a complete candidate interval.
- `202` switches have no original-resolution match at the target frame.
- `326` switches are also identity-unstable in the original-resolution stream.
- `578` switches have a stable HR target but no same-track HR candidate in the requested interval.

If every oracle-recoverable switch were removed while keeping FN and FP unchanged, IDSW would fall from `1243` to `1106`, and MOTA would rise from `0.246` to at most `0.248`. IDF1 is not estimated because it requires actually merging complete predicted track segments, not editing isolated switch counts.

## Per-sequence retrieval yield

| Sequence | LR IDSW events | Recoverable with ≥1 frame | Has ≥5 alternatives | Candidate rows |
|---|---:|---:|---:|---:|
| 0015 | 10 | 9 | 9 | 1349 |
| 0016 | 23 | 3 | 3 | 1918 |
| 0017 | 18 | 0 | 0 | 1633 |
| 0018 | 197 | 1 | 0 | 7096 |
| 0019 | 109 | 3 | 3 | 3866 |
| 0020 | 61 | 0 | 0 | 844 |
| 0021 | 176 | 8 | 8 | 4854 |
| 0022 | 70 | 11 | 11 | 1789 |
| 0023 | 77 | 15 | 15 | 2182 |
| 0024 | 94 | 25 | 25 | 4704 |
| 0025 | 49 | 11 | 10 | 1951 |
| 0026 | 56 | 3 | 3 | 793 |
| 0027 | 78 | 5 | 4 | 1219 |
| 0028 | 157 | 29 | 27 | 14032 |
| 0029 | 26 | 4 | 4 | 3136 |
| 0030 | 31 | 10 | 10 | 1185 |
| 0031 | 11 | 0 | 0 | 1484 |

## Example ranked retrievals

| Event | Target | GT ID | LR ID change | Ranked HR frames (seconds before target) |
|---|---:|---:|---|---|
| 0015_0002 | 293 | 2 | 2 → 48 | 143 (5.0s), 115 (5.9s), 95 (6.6s), 51 (8.1s), 71 (7.4s) |
| 0015_0003 | 321 | 2 | 48 → 6 | 143 (5.9s), 160 (5.4s), 115 (6.9s), 95 (7.5s), 51 (9.0s) |
| 0015_0004 | 322 | 1 | 6 → 49 | 169 (5.1s), 140 (6.1s), 106 (7.2s), 124 (6.6s), 48 (9.1s) |
| 0015_0005 | 347 | 1 | 49 → 6 | 169 (5.9s), 140 (6.9s), 195 (5.1s), 106 (8.0s), 124 (7.4s) |
| 0015_0006 | 366 | 1 | 6 → 59 | 207 (5.3s), 169 (6.6s), 140 (7.5s), 187 (6.0s), 106 (8.7s) |
| 0015_0007 | 366 | 2 | 6 → 60 | 211 (5.2s), 143 (7.4s), 175 (6.4s), 160 (6.9s), 115 (8.4s) |
| 0015_0008 | 390 | 2 | 60 → 62 | 240 (5.0s), 211 (6.0s), 143 (8.2s), 175 (7.2s), 160 (7.7s) |
| 0015_0009 | 405 | 1 | 59 → 64 | 237 (5.6s), 207 (6.6s), 255 (5.0s), 169 (7.9s), 140 (8.8s) |
| 0015_0010 | 405 | 2 | 62 → 60 | 240 (5.5s), 255 (5.0s), 211 (6.5s), 143 (8.7s), 175 (7.7s) |
| 0016_0001 | 172 | 2 | 2 → 1 | 0 (5.7s), 19 (5.1s), 5 (5.6s), 2 (5.7s), 11 (5.4s) |

## Ranking data

Candidates are labeled with `oracle_helpful`. A switch is counted as recoverable as soon as at least one candidate has this label. The ranking prior then uses HR match IoU, detector confidence, local same-track support, box area, and temporal proximity. IoU and GT identity are supervision signals and must not be used as online inputs to the future ranking model. Deployable inputs can include confidence, crop size, sharpness, motion/occlusion state, and an appearance embedding.

The output retains up to five candidates as alternatives, but the runtime system may stop after retrieving the first sufficiently strong frame. Multiple candidates are useful only when the first frame is blurred, occluded, ambiguous, or fails the ReID threshold.

## Deployment implication

ByteTrack alone cannot consume a frame from 5–10 seconds ago as identity evidence: its current lost-track buffer is 30 frames and it has no long-term appearance memory. A deployable version needs a retrieval-aware association layer: trigger on tracking uncertainty, request ranked HR crops one by one, extract an appearance embedding, and stop as soon as one frame gives a sufficiently confident identity match. Additional ranked frames are fallback evidence before merging or preserving the identity.

This report is therefore an oracle upper-bound and ranking-dataset experiment, not a claim that inserting isolated HR frames into the existing ByteTrack loop already produces these gains.

## Artifacts

- [Summary](highres_retrieval_hungarian_gt_oracle_bytetrack/summary.json)
- [All switch events with top-five frames](highres_retrieval_hungarian_gt_oracle_bytetrack/switch_events_top5.json)
- [Top-five frame table](highres_retrieval_hungarian_gt_oracle_bytetrack/top5_frames.csv)
- [All ranking candidates and labels](highres_retrieval_hungarian_gt_oracle_bytetrack/ranking_candidates.csv)
