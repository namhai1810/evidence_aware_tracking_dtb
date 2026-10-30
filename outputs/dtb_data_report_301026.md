# DTB Data Report — Dual-Fisheye Panorama MOT Benchmark

*Evidence snapshot: 30 October 2026 · Paper-oriented summary*

## 1. Data creation and annotation workflow

**Capture → synchronize → stitch → annotate.** Two dual-fisheye camera streams are recorded on a common timeline. Timestamp-matched frames are dewarped and stitched into panoramas; the stitching manifest preserves their mapping to both source cameras. Detection/tracking pseudo-labels are reviewed in CVAT to produce sequence-local MOT ground truth.

<img src="./report_assets/panorama_dataset_pipeline_30102026.svg" width="82%" alt="Panorama dataset pipeline">

*Figure D1 — Dataset construction, from synchronized dual-fisheye streams to manually reviewed MOT annotations (original Figure 4).*

## 2. Recorded sequences and annotated subset

| Scenario | Stitched frames | Manual frames | GT boxes | GT tracks |
|---|---:|---:|---:|---:|
| Normal walking | 1,307 | 907 | 1,782 | 2 |
| Crossing | 1,532 | 1,132 | 2,054 | 2 |
| Seam interaction | 1,248 | 848 | 1,505 | 2 |
| Partial occlusion | 1,053 | 653 | 1,268 | 2 |
| Full occlusion | 1,829 | 1,429 | 1,909 | 2 |
| **Total** | **6,969** | **4,969** | **8,518** | **10** |

The manually reviewed subset excludes the first **300** and last **100** frames of each sequence. IDs are **sequence-local**, not shared across videos. Annotation audit: **0 invalid/out-of-image boxes** and **0 duplicate `(frame, track_id)` entries**. The annotations contain **74 within-track gaps**, including **9 ≥30-frame gaps**; the longest verified reappearance is **220 frames**.

<img src="./report_assets/panorama_scenario_examples_30102026.jpg" width="80%" alt="Five DTB tracking scenarios">

*Figure D2 — Panorama frames from the five recorded tracking scenarios (original Figure 6).*

### Dataset split

| Split | Sequences | Frames | Boxes |
|---|---|---:|---:|
| Train | Normal walking, crossing, partial occlusion | 2,692 | 5,104 |
| Validation | Seam interaction | 848 | 1,505 |
| Test | Full occlusion | 1,429 | 1,909 |

Splits are video-disjoint. Full occlusion contains the longest annotated identity gaps and 184 empty frames. The manual labels distinguish disappearance and reappearance from an ID switch.

<img src="./report_assets/manual_mot_tracking_examples_30102026.jpg" width="78%" alt="GT tracking examples">

*Figure D3 — Manual identity trajectories through crossing, seam interaction and occlusions (original Figure 10).*

## 3. Physical-LR detector/tracker benchmark

In the physical-LR protocol, each original panorama is **resized before detection and tracking**. YOLO11l + BoT-SORT/ReID produces LR track boxes; the corresponding original-resolution coordinates are used only for historical HR evidence retrieval, not for LR detection. Detector/tracker input varies across the following operating points.

| LR input | Precision ↑ | Recall ↑ | TP | FN | Baseline IDSW | Eligible recovery events | HIVE correct candidate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512×120 | **0.9669** | 0.4048 | 3,448 | 5,070 | 51 | 43 | 34/43 |
| 512×256 | 0.9326 | 0.5443 | 4,636 | 3,882 | 73 | 66 | 30/66 |
| 640×320 | 0.9113 | **0.6421** | **5,469** | **3,049** | 94 | 88 | 51/88 |

<img src="./dtb_resolution_sweep/20261030_512x120_512x256_640x320_v1/resolution_detection_and_events.png" width="78%" alt="Detector coverage across LR resolutions">

*Figure D4 — Detection recall/precision and recovery-event coverage at three physical-LR inputs (original Figure 16).*

### DTB five-video recovery at 512×120

| Method | IDSW ↓ | IDF1 ↑ | HOTA ↑ | AssA ↑ | Fixed | Wrong |
|---|---:|---:|---:|---:|---:|---:|
| LR tracker | 51 | 0.2494 | 0.2361 | 0.1757 | 0 | 0 |
| ECIA Base-only | **37** | **0.2683** | **0.2432** | **0.1861** | 13 | 0 |
| ECIA Full + one HR crop | 39 | 0.2645 | 0.2413 | 0.1832 | 11 | 0 |

*Protocol note:* These results are physical-LR evaluations; the input sizes have different aspect ratios and generate different event pools. The previously superseded ECIA recovery results at 512×256 and 640×320 are deliberately omitted here. Recovery is evaluated at oracle-triggered switch events.

---

**Figure paths:** Relative paths are referenced from the source report's `outputs/` directory. Save this Markdown there alongside the original figure folders when exporting PDF.
