# HIVE–ECIA Model Report — Methods and Experimental Results

*Evidence snapshot: 30 October 2026 · Compact paper-preparation version*

## 1. Method: LR tracking, historical HR evidence and identity recovery

**System flow (text only):** Original HR panorama → physical LR resize → YOLO11l + BoT-SORT/ReID tracking → **oracle-supplied** ID-switch event → LR Old/Target/Competitor crops → HIVE v2 historical LR candidate ranking → **one selected HR crop** from the original panorama → frozen ReID encoder → ECIA identity association → Merge/Abstain → replay safety checks → corrected trajectories.

The detector and tracker operate on **physically resized LR frames**, not the HR panorama. HIVE returns a **specific historical person candidate**—its frame index and LR bounding box—not a full HR frame or a final identity. The selected LR box is mapped to the original panorama to obtain **at most one HR person crop per event**. GT is used to construct training labels and score results, not as an inference input. Recovery is currently evaluated at **oracle-provided switch events**, rather than with an online switch detector.

### 1.1 HIVE v2: historical evidence retrieval

HIVE (historical evidence selector) ranks person candidates from earlier LR frames so that the chosen HR crop is likely to provide useful identity evidence. Its inputs include:

- **Old history:** up to five LR crops of the existing identity track;
- **Target:** the current LR crop involved in the suspected switch;
- **Candidate pool:** up to 20 historical LR person crops from the preceding **five seconds**, including competitors;
- **Context:** competing-track history and temporal metadata.

A **shared lightweight crop CNN** encodes LR person crops into **64-dimensional** features. Attention pooling summarizes track/history evidence; relation features and a ranking head assign candidate scores. The highest-ranked candidate identifies the historical **frame + person bounding box** for HR retrieval. The HIVE v2 ranker has **146,251 trainable parameters**. Its held-out retrieval test contains 92 QuadTrack events; a correct top-1 means that the selected crop matches the target's GT identity, **not** that ECIA has repaired the ID switch.

| QuadTrack held-out HIVE retrieval | HIVE v2 | HIVE + HSV ablation |
|---|---:|---:|
| Success@1 | **63.04% (58/92)** | 58.70% (54/92) |
| Success@3 | **78.26%** | 75.00% |
| Oracle candidate coverage | 86.96% (80/92) | 86.96% (80/92) |
| Selection inference | **0.815 ms/event** | 1.939 ms/event |

The controlled HSV variant appended three LR-only color-similarity features and was not retained as the active selector. **HIVE v2 remains frozen** for the subsequent ECIA experiments.

### 1.2 Original ECIA: Base Association, HR Gain and Gate

ECIA consumes **512-D frozen ReID embeddings** for an Old-track prototype, a current Target crop, at most 12 competing LR track prototypes, and the **one HIVE-selected HR crop**. The Old prototype aggregates up to five LR observations; competitor prototypes are built from recent LR history. ECIA itself does not run the ReID backbone.

| Component | Architecture / purpose |
|---|---|
| Shared projection | `Linear(512, 64) → LayerNorm → GELU`, shared across Old, Target, HR and competitors |
| Base Identity Association | Pairwise relation `[abs(t−q), t⊙q, cosine(t,q)]` (**129-D**) → MLP `129→32→1`; scores LR-only hypotheses `q` |
| HR Evidence Gain | Relation `[abs(h−q), abs(h−t), h⊙q, cosine(h,q), cosine(h,t), base_score]` (**195-D**) → MLP `195→32→1`; predicts a signed HR residual |
| Gated Fusion | Original **5-D → 16 → 1 → sigmoid** MLP learns how strongly to apply the HR residual |
| Decision | Compares Old/competitor hypotheses with an Abstain option; valid merges pass collision and missing-segment replay guards |

Here `t`, `q` and `h` denote projected **Target**, identity-hypothesis and **HR** feature vectors. The original Full score for each hypothesis is

\[
S_{\mathrm{Full}}(q)=S_{\mathrm{Base}}(q)+g\,\Delta S_{\mathrm{HR}}(q),\qquad 0\leq g\leq1.
\]

Original ECIA contains **43,684 trainable parameters** with the ReID encoder and HIVE frozen. **Base-only, Base+HR Gain and Full were trained as separate ECIA variants**; comparing them evaluates complete trained systems, not the causal influence of HR on identical weights.

### 1.3 Recovery-Aware ECIA: frozen LR Base and selective HR residual

Recovery-Aware Full uses frozen **OSNet-AIN x1.0 D+M+C** embeddings and frozen HIVE v2. The **Shared Projection, Base Association and Abstain Head** are copied from the best OSNet-AIN Base-only checkpoint and frozen **bit-for-bit**. Only HR Gain and Gate are trained: **6,978 trainable / 44,244 total ECIA parameters**.

| Updated component | Input / behavior |
|---|---|
| HR Evidence Gain | **202-D** hypothesis relation incorporating Base probability, LR confidence, Old-versus-competitor margin, target similarity and HR–LR disagreement |
| Recovery-Aware Gate | **12-D** features including LR probability/margins/entropy, HR–Target/Old/competitor similarities, HR identity margin, Jensen–Shannon LR/HR disagreement and top-hypothesis disagreement |
| Fusion | Signed residual HR correction to frozen Base scores; learned Gate controls its contribution |
| Safety | Same Merge/Abstain replay and collision checks as the original system |

The training objective combines **fused identity classification**, **margin ranking**, **wrong-merge penalty**, **Base-preservation**, **targeted recovery**, and **helpful-versus-harmful gate supervision**. Recovery usefulness and wrong-merge labels are derived from training GT; they do not enter inference. The chosen checkpoint is **epoch 16**; calibrated merge thresholds are **probability 0.80 / margin 0.40**, using QuadTrack calibration sequence `0030` only.

## 2. Training and evaluation protocol

| Role | QuadTrack video IDs |
|---|---|
| Train | 0015, 0018, 0020, 0021, 0022, 0023, 0024, 0026, 0027, 0029 |
| Model validation / early stopping | 0019, 0028 |
| Operating threshold calibration | 0030 |
| Held-out test | 0016, 0017, 0025, 0031 |

HIVE v2, frozen ReID encoders and ECIA heads are kept separate: the original YOLO26n model and the OSNet-based Base/Full variants have **separately trained ECIA checkpoints**. For a given encoder, embedding caches and tracking event pools are held consistent between Base/Full comparisons. **No DTB images or annotations are used to train or calibrate ECIA.**

The **same-checkpoint HR intervention** compares Recovery-Aware Full against the *identical model* with `gate=0`, retaining weights, threshold and event pool. `gate=0` suppresses the HR residual but is **not** numerically equivalent to the separately calibrated Base-only system. **Fixed** denotes a corrected switch-point event; **Wrong** denotes harmful identity contamination elsewhere in the replayed track segment. Global **IDSW, IDF1, HOTA and AssA** are evaluated from replayed trajectories.

## 3. Original ECIA: held-out QuadTrack results

*Frozen HIVE v2 + YOLO26n-ReID, 512-D identity embeddings.*

| Method | IDSW ↓ | IDF1 ↑ | HOTA ↑ | Fixed | Wrong |
|---|---:|---:|---:|---:|---:|
| LR tracker | 92 | 0.4476 | 0.3618 | 0 | 0 |
| HIVE + threshold HR ReID | 88 | 0.4503 | 0.3630 | 4 | 0 |
| ECIA Base-only | 46 | 0.4706 | 0.3768 | 48 | 4 |
| ECIA Base + HR Gain | **42** | 0.4432 | 0.3665 | **53** | 5 |
| **ECIA Full (Gain + Gate)** | 45 | **0.4719** | **0.3783** | 48 | **3** |

<img src="./hive_ecia_experiment/comparison/tracking_metrics.png" width="75%" alt="Original ECIA tracking evaluation">

*Figure M1 — QuadTrack held-out tracking comparison of the original ECIA variants (original Figure 2).*

## 4. Frozen LR↔HR ReID encoder benchmark

On **175 frozen DTB events** with **3,491 LR–HR candidate pairs**, the LR Target was compared to historical HR **same-GT-ID** candidates and **competitors**. GT only labeled pairs for measurement; this benchmark did **not** run HIVE/ECIA decisions or tracking replay.

| Frozen encoder | Same-ID cosine | Competitor cosine | AUC ↑ | Event top-1 ↑ | Best-positive margin ↑ |
|---|---:|---:|---:|---:|---:|
| YOLO26n-ReID | 0.2321 | 0.1653 | 0.6351 | 0.5429 | +0.0170 |
| **OSNet Market1501** | 0.7228 | 0.6662 | 0.6619 | **0.6171** | +0.0211 |
| OSNet-AIN MSMT17 | 0.5881 | 0.5334 | 0.6582 | 0.6000 | +0.0308 |
| **OSNet-AIN D+M+C** | 0.6221 | 0.5608 | **0.7046** | 0.6000 | +0.0306 |
| DIL MLR-CUHK03 | 0.6874 | 0.5359 | 0.6640 | 0.5771 | **+0.0516** |

DIL is trained explicitly for degradation/cross-resolution invariance. OSNet-AIN is a **cross-domain / domain-generalized** encoder, not a dedicated cross-resolution encoder. Absolute cosine scales differ between models; ranking accuracy and AUC are the comparative metrics.

<img src="./dtb_reid_resolution_benchmark/20261030_lr_hr_reid_v2/reid_encoder_summary.png" width="75%" alt="ReID encoder ranking metrics">

*Figure M2 — Pairwise AUC, event top-1 and identity margin for frozen encoders (original Figure 19).*

## 5. ECIA retraining with OSNet encoders

HIVE v2 and each ReID encoder remain frozen; **ECIA is retrained for each encoder** on QuadTrack training data, with checkpoint selection and threshold calibration restricted to the corresponding QuadTrack splits.

| QuadTrack held-out | IDSW ↓ | IDF1 ↑ | HOTA ↑ | AssA ↑ | Wrong |
|---|---:|---:|---:|---:|---:|
| Tracker | 92 | 0.4476 | 0.3618 | 0.4362 | 0 |
| **YOLO26n Full** | 45 | **0.4719** | **0.3783** | **0.4769** | 3 |
| OSNet Market Base | 81 | 0.4546 | 0.3673 | 0.4496 | 2 |
| OSNet Market Full | **40** | 0.4470 | 0.3670 | 0.4504 | 6 |
| OSNet-AIN Base | 66 | 0.4638 | 0.3754 | 0.4700 | 3 |
| OSNet-AIN Full | 58 | 0.4421 | 0.3653 | 0.4461 | 3 |

### DTB `full_occ`: physical LR 640×320 (predefined split)

| Model | IDSW ↓ | IDF1 ↑ | HOTA ↑ | AssA ↑ |
|---|---:|---:|---:|---:|
| Tracker | 21 | 0.2015 | 0.1855 | 0.0935 |
| YOLO26n Full | 17 | 0.2703 | 0.2074 | 0.1171 |
| OSNet Market Full | **14** | 0.2703 | 0.2071 | 0.1168 |
| OSNet-AIN Base | 16 | 0.2050 | 0.1884 | 0.0964 |
| **OSNet-AIN Full** | 16 | **0.2738** | **0.2094** | **0.1193** |

## 6. Recovery-Aware HR Fusion: results and HR intervention

| QuadTrack held-out | IDSW ↓ | IDF1 ↑ | HOTA ↑ | AssA ↑ | Fixed | Wrong |
|---|---:|---:|---:|---:|---:|---:|
| OSNet-AIN Base | 66 | **0.4638** | **0.3754** | **0.4700** | 28 | 3 |
| OSNet-AIN Full (old) | 58 | 0.4421 | 0.3653 | 0.4461 | 35 | 3 |
| **Recovery-Aware Full** | **53** | 0.4389 | 0.3637 | 0.4420 | **41** | 3 |
| Recovery `gate=0` | 72 | 0.4625 | 0.3723 | 0.4619 | 21 | **2** |

| DTB `full_occ` @ 640×320 | IDSW ↓ | IDF1 ↑ | HOTA ↑ | AssA ↑ | Fixed | Wrong |
|---|---:|---:|---:|---:|---:|---:|
| OSNet-AIN Base | 16 | 0.2050 | 0.1884 | 0.0964 | 5 | 0 |
| OSNet-AIN Full (old) | 16 | **0.2738** | 0.2094 | 0.1193 | 5 | 0 |
| **Recovery-Aware Full** | **13** | **0.2738** | **0.2126** | **0.1230** | **8** | 0 |
| Recovery `gate=0` | 19 | 0.2050 | 0.1871 | 0.0951 | 2 | 0 |

<img src="./hive_ecia_recovery_aware_report/20261030_osnet_ain_recovery_v1/dtb_strict_comparison.png" width="76%" alt="Recovery aware strict DTB results">

*Figure M3 — Recovery-Aware comparison on the predefined DTB full-occlusion split (original Figure 28).*

### Five DTB scenarios (exploratory cross-dataset aggregate)

| Model | IDSW ↓ | IDF1 ↑ | HOTA ↑ | AssA ↑ | Fixed | Wrong |
|---|---:|---:|---:|---:|---:|---:|
| Tracker | 94 | 0.3520 | 0.3422 | 0.2273 | 0 | 0 |
| OSNet-AIN Base | 76 | 0.3554 | 0.3501 | 0.2377 | 18 | 0 |
| **OSNet-AIN Full (old)** | 73 | **0.4276** | **0.4049** | **0.3169** | 21 | 0 |
| Recovery-Aware Full | **63** | 0.3816 | 0.3630 | 0.2552 | **31** | 0 |
| Recovery `gate=0` | 86 | 0.3548 | 0.3436 | 0.2291 | 8 | 0 |

**Same-checkpoint HR ablation (Recovery-Aware Full vs. `gate=0`):** on QuadTrack, enabling HR adds **20 immediate fixes** and **one wrong segment correction**; across DTB, HR adds **23 fixes** with **no added wrong correction**. On DTB the same-checkpoint IDSW changes **86→63**, while IDF1 changes **0.3548→0.3816**. On QuadTrack, IDSW changes **72→53** but IDF1 changes **0.4625→0.4389**.

<img src="./hive_ecia_recovery_aware_report/20261030_osnet_ain_recovery_v1/dtb_all_comparison.png" width="76%" alt="Aggregate DTB model comparison">

*Figure M4 — Five-video DTB aggregate: old Full leads IDF1/HOTA/AssA, Recovery-Aware Full leads IDSW (original Figure 29).*

---

**Figure paths:** All image paths are relative to the original report's `outputs/` directory. Place this `.md` file inside `outputs/` alongside the existing figure folders for rendering. The removed section **“1.2 Physical-LR tracking and HR recovery” Mermaid** is **not included**.
