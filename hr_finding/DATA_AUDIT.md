# HIVE dataset audit

Generated: `2026-10-22T16:42:08+02:00`

## Compatibility summary

- Dataset: `/workspace/evidence_aware_tracking/code/outputs/reid_ranking_dataset_v1`
- Events / candidate pairs: **925 / 4606**.
- Video-disjoint leakage check: **PASS** (`0` shared sequences, `0` shared events).
- Materialized old-query / target LR crops missing: **0 / 0**.
- Materialized candidate LR crops: **0**.
- Candidate LR crops reconstructable by downsampling the complete source panorama first: **4606/4606**.
- MOT-format LR tracking files available: **17/17** (frame, track ID, bbox, confidence).
- Full LR panorama files are not materialized; preprocessing reproduces them from complete source frames with `INTER_AREA` before any crop.
- Candidate boxes matching an actual LR-tracker detection at IoU >= 0.50: **3024/4606 (65.7%)**.
- Events with a real competing `lr_new_pred_id` history in the preceding 30 frames: **563/925 (60.9%)**.

## What HIVE may use online

- Materialized old-track LR crops and target LR crop.
- Candidate LR crops reconstructed by resizing the full panorama to 512x120 before cropping.
- Temporal offsets and genuine LR-tracker histories when present.
- The materialized HR candidate crop, label, GT identity, HR confidence, HR ReID similarities, and candidate rank are never exposed as online features.

## Missing or limited information

- No association matrices or explicit motion-residual features are stored.
- Competing-track history is available only for a subset of events; `hive_full` uses a masked fallback elsewhere.
- Current top-5 candidates were preselected using HR tracking detections and HR ReID similarity.
- Although every candidate LR crop can be reconstructed offline, only the LR-tracker-matched subset has a directly deployable LR proposal box.
- `bridge_similarity` is an offline HR-teacher signal, not a measured identity-margin or counterfactual correction-utility label.
- No real tracking replay outcome is available, so IDSW/IDF1/HOTA improvements cannot be claimed from these labels.

## Supported variants

- `appearance_only`: supported.
- `appearance_metadata`: supported using LR appearance and temporal metadata.
- `hive_no_hypothesis`: supported using old/target/candidate contrastive relations.
- `hive_full`: partially supported; real competing history is masked when unavailable.
- Margin distillation: implemented but disabled because no validated utility margin exists.

## Deployment blocker

A deployable LR-only proposal generator is still required. It must form the candidate pool from LR tracker state without consulting HR detections or HR ReID rankings. Current results evaluate reranking inside an offline HR-preselected pool.
