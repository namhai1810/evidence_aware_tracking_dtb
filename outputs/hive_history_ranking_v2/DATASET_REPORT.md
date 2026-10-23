# HIVE history-ranking dataset v2

Generated: `2026-10-23T12:10:41+02:00`

## Scope

This dataset trains **historical candidate retrieval after an ID switch is already known**.
It intentionally contains no normal/non-switch events and does not train a switch trigger.

## Construction

1. Preserve the original 925 GT-confirmed LR switch events and video-level split.
2. Search LR tracker detections only in `T-150 ... T-1`.
3. Downsample the complete panorama to `512x120` before cropping.
4. Build an LR-only HSV appearance descriptor for proposal mining.
5. Select at most 20 candidates using appearance-hard, recent, and temporally diverse routes.
6. Freeze the candidate pool, then use evaluator-identical Hungarian GT matching at IoU 0.50 to assign candidate labels.

Neither HR detections nor HR ReID embeddings are used to form the v2 candidate pool. GT is used only for offline labels after selection.

## Change from v1

| Dataset | Candidate source | Max/event | Candidate pairs | Positive | Negative | Oracle coverage |
|---|---|---:|---:|---:|---:|---:|
| v1 | HR detections + HR ReID top-5 | 5 | 4,606 | 1,927 | 2,679 | 76.3% |
| v2 | LR tracker + LR-only proposal | 20 | 18,355 | 5,725 | 12,630 | 87.9% |

The candidate sources differ, so v2 is not merely v1 with extra rows. It replaces the privileged HR proposal stage with a reproducible LR-only pool and adds 9,951 negative pairs. V2 contains 8,777 wrong-identity negatives and 3,853 unmatched-detection negatives.

## Dataset size

| Split | Videos | Switch events | Candidate pairs | Positive | Negative | Events with ≥1 positive |
|---|---:|---:|---:|---:|---:|---:|
| Train | 10 | 757 | 14,997 | 4,610 | 10,387 | 87.8% |
| Validation | 3 | 76 | 1,518 | 578 | 940 | 89.5% |
| Test | 4 | 92 | 1,840 | 537 | 1,303 | 87.0% |
| All | 17 | 925 | 18,355 | 5,725 | 12,630 | 87.9% |

Negative candidates are LR detections that either match a different GT identity or do not match any GT at IoU 0.50. They are not non-switch frames.

## Leakage audit

- Video-disjoint split: **PASS**.
- Train/validation/test sequence overlap: **0**.
- Cross-split event overlap: **0**.
- Cross-split candidate source-frame overlap: **0**.
- Validation videos: 0019, 0028, 0030.
- Test videos: 0016, 0017, 0025, 0031.

The split is applied at video level before model training. Candidate mining is performed independently inside each video's own history window. Test labels and test statistics must not be used for model selection.

## Files

- `train_data/events.jsonl`, `test_data/events.jsonl`: event-grouped manifests.
- `train_data/candidates.csv`, `test_data/candidates.csv`: flat candidate tables.
- `statistics.json`: full counts by learning-train, validation and test subsets.
- `leakage_audit.json`: machine-readable split audit.
- `split_config.json`: immutable provenance and construction parameters.

Rebuild command from the code directory:

```bash
python build_hive_history_ranking_v2.py
```

## Important limitation

The event trigger remains GT-confirmed offline: this dataset assumes a switch event has already been supplied. It evaluates LR-only historical proposal and ranking, not online switch detection and not measured IDSW correction.
