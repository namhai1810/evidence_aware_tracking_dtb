# QuadTrack cross-resolution ReID ranking dataset v1

## Purpose

Train and evaluate a model that ranks historical 2048x480 person crops for a
512x120 ID-switch query. Each event is one ranking group and must never be split
across datasets.

## Leakage-safe split

The split is grouped by complete video sequence. Train and test have no shared
sequence, event, frame source, or crop file.

- Train sequences: `0015, 0018, 0019, 0020, 0021, 0022, 0023, 0024, 0026, 0027, 0028, 0029, 0030`
- Test sequences: `0016, 0017, 0025, 0031`

| Split | Videos | Events | Candidate pairs | Positive | Negative | Positive in top-5 |
|---|---:|---:|---:|---:|---:|---:|
| Train | 13 | 833 | 4146 | 1734 | 2412 | 636 |
| Test | 4 | 92 | 460 | 193 | 267 | 70 |

## Sample definition

For a switch at frame `T`, the input group contains up to five old-track LR crops,
one switched-target LR crop, and up to five ranked HR person crops from `T-150`
through `T-1`. A candidate label is `1` when its HR box overlaps the target GT
identity by IoU >= 0.50; otherwise it is `0`. GT is applied after ReID ranking.

## Layout

```text
reid_ranking_dataset_v1/
├── split_config.json
├── statistics.json
├── dataset_card.md
├── train_data/
│   ├── sequences.txt
│   ├── events.jsonl
│   ├── events.csv
│   ├── candidates.csv
│   └── crops/<event_id>/{query,target,candidate}_*.jpg
└── test_data/
    └── ... same schema ...
```

All crop paths in manifests are relative to this package root. `events.jsonl` is the
canonical grouped format for learning-to-rank; `candidates.csv` is a convenient flat
view for analysis. Do not randomly split rows from either file.

## Recommended evaluation

Report event-level top-1 accuracy, Recall@5, MRR, and no-match/reject performance.
Events without a positive candidate cannot be solved by reranking alone and should
be retained for reject-head or end-to-end coverage evaluation.

## Reproduction

Run `python build_reid_ranking_dataset.py` inside the project environment. If this
version already exists, pass a new versioned `--output-root`; the script refuses to
overwrite an existing package. Exact source checksums and retrieval configuration
are recorded in `split_config.json`.
