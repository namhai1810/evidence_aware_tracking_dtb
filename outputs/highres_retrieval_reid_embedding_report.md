# Cross-resolution ReID embedding retrieval for 512x120 ID switches

Generated: 2026-10-22T15:08:30+02:00

## What this experiment does

This experiment performs actual appearance-embedding retrieval. For each GT-identified ID-switch event in the 512x120 stream, it builds an embedding prototype from up to five observations of the old low-resolution track, embeds the switched target detection, searches full-resolution tracking detections in the immediately preceding lookback window, and ranks candidates by the minimum cosine similarity to the old-track prototype and target embedding.

Ground truth is not used to select or rank candidates. It is used only to identify the evaluation switch events and to label whether the selected full-resolution box is the correct person.

- ReID extractor: `yolo26n-reid.onnx` (512-dimensional L2-normalized crop embeddings)
- Low-resolution query input: physical `512x120` crops
- Old-track gallery: up to `5` observations from the preceding `30` frames
- Historical search window: the previous `5.0` seconds, excluding the switch frame itself; at `30` FPS this is exactly `T-150` through `T-1`
- Search order: nearest historical frame first, then progressively older frames; ReID bridge similarity remains the ranking score, with recency used only to break equal-score ties
- Exploratory acceptance threshold: cosine bridge similarity `>= 0.30`; threshold-free top-1/top-5 ranking accuracy is also reported
- Correct-person evaluation: selected HR box IoU `>= 0.50` with the switch identity's GT box

## Main result

| Measure | Count | Fraction |
|---|---:|---:|
| Low-resolution ID-switch events | 925 | 100.0% |
| Events where the correct HR identity exists in the search window | 920 | 99.5% |
| Top-1 embedding ranking selects the correct HR identity | **514** | **55.6%** |
| Unique target frames supported by top-1 retrieval | **495** | — |
| Correct identity appears in top-5 ranking | **706** | **76.3%** |

Among events where a correct HR detection is available, top-1 retrieval recall is `55.9%` and top-5 recall is `76.7%`.

If every correct top-1 bridge led to a successful identity merge, IDSW would be estimated to fall from `925` to `411`. With an oracle choice among the top five, the estimate is `219`. These are retrieval-supported estimates, not metrics from rerunning the tracker after track merging.

## Threshold sweep for the top-ranked frame

| Cosine threshold | Accepted | Correct | Wrong | Precision | Correct / all switches |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 922 | 514 | 408 | 55.7% | 55.6% |
| 0.30 | 921 | 514 | 407 | 55.8% | 55.6% |
| 0.40 | 882 | 503 | 379 | 57.0% | 54.4% |
| 0.50 | 715 | 433 | 282 | 60.6% | 46.8% |
| 0.60 | 402 | 263 | 139 | 65.4% | 28.4% |

## Per-sequence retrieval result

| Sequence | IDSW events | Correct HR available | Top-1 improvable | Top-5 improvable |
|---|---:|---:|---:|---:|
| 0015 | 15 | 15 | 15 | 15 |
| 0016 | 13 | 13 | 13 | 13 |
| 0017 | 17 | 17 | 15 | 17 |
| 0018 | 96 | 96 | 55 | 87 |
| 0019 | 41 | 41 | 8 | 22 |
| 0020 | 27 | 27 | 6 | 18 |
| 0021 | 109 | 109 | 80 | 99 |
| 0022 | 92 | 92 | 62 | 77 |
| 0023 | 97 | 97 | 51 | 60 |
| 0024 | 105 | 103 | 56 | 86 |
| 0025 | 61 | 61 | 25 | 39 |
| 0026 | 83 | 81 | 38 | 58 |
| 0027 | 133 | 132 | 58 | 79 |
| 0028 | 7 | 7 | 5 | 7 |
| 0029 | 0 | 0 | 0 | 0 |
| 0030 | 28 | 28 | 26 | 28 |
| 0031 | 1 | 1 | 1 | 1 |

## Example successful top-1 retrievals

| Event | Target | LR ID change | Query frames | HR frame | Similarity old / target / bridge |
|---|---:|---|---|---:|---|
| 0015_0001 | 23 | 1 → 9 | 1, 3, 5, 7, 9 | 12 | 0.665 / 0.515 / 0.515 |
| 0015_0002 | 251 | 2 → 38 | 228, 233, 238, 240, 245 | 248 | 0.608 / 0.502 / 0.502 |
| 0015_0003 | 252 | 38 → 2 | 251 | 248 | 0.502 / 0.511 / 0.502 |
| 0015_0004 | 261 | 2 → 42 | 228, 233, 238, 245, 257 | 260 | 0.574 / 0.578 / 0.574 |
| 0015_0005 | 272 | 42 → 44 | 261, 262, 263, 264, 265 | 268 | 0.709 / 0.666 / 0.666 |
| 0015_0006 | 293 | 9 → 45 | 255, 257, 262, 272, 275 | 284 | 0.562 / 0.624 / 0.562 |
| 0015_0007 | 293 | 44 → 46 | 272, 275, 279, 283, 284 | 265 | 0.672 / 0.559 / 0.559 |
| 0015_0008 | 321 | 45 → 46 | 293, 297, 299, 301, 304 | 314 | 0.548 / 0.562 / 0.548 |
| 0015_0009 | 322 | 46 → 52 | 294, 299, 301, 305, 310 | 313 | 0.650 / 0.598 / 0.598 |
| 0015_0010 | 338 | 46 → 56 | 301, 310, 321, 325, 328 | 285 | 0.547 / 0.579 / 0.547 |

## Interpretation and limitations

- Ranking is based only on appearance embeddings; GT identity and IoU do not enter the ranking score.
- Switch events are still supplied by the evaluator using GT. A deployable system needs its own uncertainty trigger.
- The experiment does not yet inject retrieved embeddings into BoT-SORT or rewrite/merge complete track segments, so post-merge IDF1 and MOTA are not claimed.
- The standalone crop model is an official Ultralytics ReID extractor. It differs from the native YOLO11 feature-map embeddings used by the earlier BoT-SORT runs; this distinction is intentional and recorded for reproducibility.

## Artifacts

- [Summary](highres_retrieval_reid_embedding/summary.json)
- [All switch events and ranked frames](highres_retrieval_reid_embedding/switch_events_top5.json)
- [Top-five retrieval table](highres_retrieval_reid_embedding/top5_frames.csv)
