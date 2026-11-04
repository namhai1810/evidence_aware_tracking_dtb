# Fair MOT baseline benchmark

Frozen Existing Base-Guided Safety (HIVE–ECIA) is compared with standard SORT, ByteTrack, OC-SORT and BoT-SORT. Each tracker is evaluated both before and after canonical-ID recovery. No online-aware v1 or balanced-v2 checkpoint is used.

## Paper-ready summary

| Method | QuadTrack IDF1 ↑ | HOTA ↑ | AssA ↑ | IDSW ↓ | DTB IDF1 ↑ | HOTA ↑ | AssA ↑ | IDSW ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SORT | 0.3220 | 0.2678 | 0.2657 | 135 | 0.3508 | 0.3358 | 0.2225 | 103 |
| SORT + HIVE–ECIA† | 0.3601 | 0.2879 | 0.3068 | 113 | <u>0.4182</u> | **0.4027** | <u>0.3189</u> | 88 |
| ByteTrack | 0.3623 | 0.3055 | 0.3193 | 101 | 0.3225 | 0.3404 | 0.2188 | 85 |
| ByteTrack + HIVE–ECIA† | 0.3678 | 0.3089 | 0.3264 | <u>79</u> | 0.3544 | 0.3556 | 0.2385 | <u>76</u> |
| OC-SORT | 0.3299 | 0.2699 | 0.2815 | 95 | 0.4120 | 0.3879 | 0.3060 | 77 |
| OC-SORT + HIVE–ECIA† | 0.3465 | 0.2767 | 0.2956 | **78** | **0.4382** | <u>0.4011</u> | **0.3270** | **59** |
| BoT-SORT | <u>0.4476</u> | <u>0.3618</u> | <u>0.4362</u> | 92 | 0.3520 | 0.3422 | 0.2273 | 94 |
| BoT-SORT + HIVE–ECIA (Ours) | **0.4663** | **0.3725** | **0.4628** | **78** | **0.3722** | <u>0.3525</u> | **0.2408** | 89 |

**Table:** Identity-centric comparison using the same YOLO11l detection cache and evaluation protocol. **Bold** denotes best and <u>underline</u> denotes second-best per dataset/metric; ties are both bolded. †Zero-shot tracker transfer of the frozen recovery module; BoT-SORT + HIVE–ECIA is the primary proposed configuration.

![Tracker-family transfer comparison](figures/tracker_family_transfer_summary.png)

**Figure:** Controlled before/after comparison for four association trackers. Hatched bars use the same frozen HIVE–ECIA checkpoints and thresholds; percentages above the IDSW pairs show relative ID-switch reduction.

## Aggregate results

| Dataset | Method | IDSW ↓ | IDF1 ↑ | HOTA ↑ | AssA ↑ | MOTA ↑ | IDTP | IDFP | IDFN | FPS ↑ | Latency ms ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QuadTrack held-out test (512×120) | SORT | 135 | 0.3220 | 0.2678 | 0.2657 | 0.3140 | 3616 | 2997 | 12230 | **157.84** | **6.34** |
| QuadTrack held-out test (512×120) | SORT + HIVE–ECIA† | 113 | 0.3601 | 0.2879 | 0.3068 | 0.3154 | 4044 | 2569 | 11802 | 120.08 | 8.33 |
| QuadTrack held-out test (512×120) | YOLO11l + ByteTrack | 101 | 0.3623 | 0.3055 | 0.3193 | 0.3465 | 4149 | 2906 | 11697 | 155.09 | 6.45 |
| QuadTrack held-out test (512×120) | YOLO11l + ByteTrack + HIVE–ECIA† | 79 | 0.3678 | 0.3089 | 0.3264 | 0.3478 | 4211 | 2844 | 11635 | 133.45 | 7.49 |
| QuadTrack held-out test (512×120) | YOLO11l + OC-SORT | 95 | 0.3299 | 0.2699 | 0.2815 | 0.3023 | 3645 | 2608 | 12201 | 148.32 | 6.74 |
| QuadTrack held-out test (512×120) | YOLO11l + OC-SORT + HIVE–ECIA† | **78** | 0.3465 | 0.2767 | 0.2956 | 0.3034 | 3829 | 2424 | 12017 | 121.38 | 8.24 |
| QuadTrack held-out test (512×120) | YOLO11l + BoT-SORT | 92 | 0.4476 | 0.3618 | 0.4362 | 0.3538 | 5196 | 2173 | 10650 | 127.96 | 7.82 |
| QuadTrack held-out test (512×120) | YOLO11l + BoT-SORT + HIVE–ECIA | **78** | **0.4663** | **0.3725** | **0.4628** | **0.3547** | **5413** | **1956** | **10433** | 107.53 | 9.30 |
| DTB zero-shot (640×320) | SORT | 103 | 0.3508 | 0.3358 | 0.2225 | 0.5534 | 2500 | 3235 | 6018 | 116.47 | 8.59 |
| DTB zero-shot (640×320) | SORT + HIVE–ECIA† | 88 | <u>0.4182</u> | **0.4027** | <u>0.3189</u> | 0.5552 | 2980 | 2755 | 5538 | 107.44 | 9.31 |
| DTB zero-shot (640×320) | YOLO11l + ByteTrack | 85 | 0.3225 | 0.3404 | 0.2188 | 0.5674 | 2312 | 3508 | 6206 | 114.50 | 8.73 |
| DTB zero-shot (640×320) | YOLO11l + ByteTrack + HIVE–ECIA† | <u>76</u> | 0.3544 | 0.3556 | 0.2385 | 0.5684 | 2541 | 3279 | 5977 | **151.41** | **6.60** |
| DTB zero-shot (640×320) | YOLO11l + OC-SORT | 77 | 0.4120 | 0.3879 | 0.3060 | 0.5447 | 2875 | 2564 | 5643 | 113.91 | 8.78 |
| DTB zero-shot (640×320) | YOLO11l + OC-SORT + HIVE–ECIA† | **59** | **0.4382** | <u>0.4011</u> | **0.3270** | 0.5468 | **3058** | **2381** | **5460** | 106.98 | 9.35 |
| DTB zero-shot (640×320) | YOLO11l + BoT-SORT | 94 | 0.3520 | 0.3422 | 0.2273 | 0.5686 | 2555 | 3446 | 5963 | 95.38 | 10.48 |
| DTB zero-shot (640×320) | YOLO11l + BoT-SORT + HIVE–ECIA | 89 | 0.3722 | 0.3525 | 0.2408 | **0.5691** | 2702 | 3299 | 5816 | 85.44 | 11.70 |

## Direct tracker-family comparisons

| Dataset | Tracker | Baseline IDSW | + HIVE–ECIA IDSW | Reduction | Relative | ΔIDF1 | ΔHOTA | ΔAssA | ΔMOTA |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| QuadTrack held-out test (512×120) | SORT | 135 | 113 | +22 | +16.30% | +0.0381 | +0.0201 | +0.0411 | +0.0014 |
| QuadTrack held-out test (512×120) | ByteTrack | 101 | 79 | +22 | +21.78% | +0.0054 | +0.0034 | +0.0071 | +0.0014 |
| QuadTrack held-out test (512×120) | OC-SORT | 95 | 78 | +17 | +17.89% | +0.0167 | +0.0068 | +0.0141 | +0.0011 |
| QuadTrack held-out test (512×120) | BoT-SORT | 92 | 78 | +14 | +15.22% | +0.0187 | +0.0107 | +0.0266 | +0.0009 |
| DTB zero-shot (640×320) | SORT | 103 | 88 | +15 | +14.56% | +0.0674 | +0.0669 | +0.0964 | +0.0018 |
| DTB zero-shot (640×320) | ByteTrack | 85 | 76 | +9 | +10.59% | +0.0319 | +0.0152 | +0.0197 | +0.0011 |
| DTB zero-shot (640×320) | OC-SORT | 77 | 59 | +18 | +23.38% | +0.0262 | +0.0132 | +0.0210 | +0.0021 |
| DTB zero-shot (640×320) | BoT-SORT | 94 | 89 | +5 | +5.32% | +0.0202 | +0.0102 | +0.0135 | +0.0006 |

BoT-SORT + HIVE–ECIA is the original Existing Base-Guided Safety pipeline. SORT, ByteTrack and OC-SORT use zero-shot tracker-transfer integrations with the same frozen checkpoints, thresholds and safety policy, without tracker-specific, QuadTrack-test or DTB tuning. Aggregate identity metrics improve for all four tracker families on both datasets; this does not imply that every individual sequence improves.

## Fairness checks

- QuadTrack held-out test (512×120): **PASS**.
- DTB zero-shot (640×320): **PASS**.
- Shared detector: YOLO11l, person class only, confidence 0.20 and NMS IoU 0.70.
- QuadTrack sequences: 0016, 0017, 0025, 0031. DTB sequences: normal_walking, crossing, partial_occ, seam, full_occ.
- GT is loaded only after MOT predictions are frozen.
- All baseline/+HIVE pairs use byte-identical raw tracker geometry; recovery changes canonical IDs only.
- Every tracker is freshly reset at sequence boundaries and all methods reuse the same YOLO detection cache.
- SORT, ByteTrack and OC-SORT transfer the frozen BoT-SORT-trained recovery module without tracker-specific threshold tuning.
- OC-SORT uses repository-vendored defaults: detection threshold 0.25, max age 30, minimum hits 3, IoU threshold 0.30, delta_t 3, inertia 0.20 and no Byte second pass.
- Standard tracker timing reports shared detector latency plus tracker replay latency; HIVE–ECIA timing measures detector + the corresponding tracker + recovery controller. Decode/resize/write I/O is excluded.

## Detailed reports

- QuadTrack: `/workspace/evidence_aware_tracking/code/outputs/mot_baseline_benchmark/fair_mot_conda_v1/quadtrack_512x120/comparison/REPORT.md`
- DTB: `/workspace/evidence_aware_tracking/code/outputs/mot_baseline_benchmark/fair_mot_conda_v1/dtb_640x320/comparison/REPORT.md`
- Tracker-family transfer: `/workspace/evidence_aware_tracking/code/outputs/tracker_family_transfer/sort_ocsort_v1/REPORT.md`

<!-- HIVE_ECIA_ABLATION_START -->
## Frozen HIVE–ECIA component ablation

The ablation uses the same frozen YOLO11l, BoT-SORT, HIVE v2, OSNet-AIN and Recovery-Aware ECIA checkpoints. A0/A3 reuse the verified fair-benchmark artifacts; all raw BoT-SORT predictions match A0 and GT is joined only after inference.

- **A0:** LR-only YOLO11l + BoT-SORT.
- **A1:** A0 + ECIA Base-only; no physical HR request.
- **A2:** original rule-trigger HIVE + ECIA Full.
- **A3:** ECIA Base-Guided gating + existing Safety; final method.

![Frozen HIVE–ECIA ablation](figures/hive_ecia_ablation_summary.png)

**Figure:** Component effects on identity quality, ID switches and physical HR requests. A3 is highlighted; arrows report A0→A3 IDSW reduction and A2→A3 HR-request reduction.

### Aggregate ablation

| Dataset | Variant | IDSW ↓ | IDF1 ↑ | HOTA ↑ | AssA ↑ | MOTA ↑ | HR requests ↓ | Encoded HR MB ↓ | Total MB ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| QuadTrack held-out (512×120) | A0 LR-only | 92 | 0.4476 | 0.3618 | 0.4362 | 0.3538 | **0** | **0.000** | **51.308** |
| QuadTrack held-out (512×120) | A1 Base-only | 86 | 0.4485 | 0.3632 | 0.4396 | 0.3542 | **0** | **0.000** | **51.308** |
| QuadTrack held-out (512×120) | A2 Rule Full | 82 | 0.4656 | 0.3722 | 0.4620 | 0.3545 | 120 | 0.593 | 51.901 |
| QuadTrack held-out (512×120) | **A3 Final** | **78** | **0.4663** | **0.3725** | **0.4628** | **0.3547** | 29 | 0.167 | 51.475 |
| DTB zero-shot (640×320) | A0 LR-only | 94 | 0.3520 | 0.3422 | 0.2273 | 0.5686 | **0** | **0.000** | **181.106** |
| DTB zero-shot (640×320) | A1 Base-only | 91 | 0.3520 | 0.3468 | 0.2332 | 0.5689 | **0** | **0.000** | **181.106** |
| DTB zero-shot (640×320) | A2 Rule Full | **85** | 0.3696 | **0.3562** | **0.2458** | **0.5696** | 53 | 0.175 | 181.281 |
| DTB zero-shot (640×320) | **A3 Final** | 89 | **0.3722** | 0.3525 | 0.2408 | 0.5691 | 18 | 0.055 | 181.161 |

### Component contribution

| Dataset | Comparison | ΔIDSW | ΔIDF1 | ΔHOTA | ΔAssA | HR interpretation |
|---|---|---:|---:|---:|---:|---|
| QuadTrack | A0→A1: LR Base | −6 | +0.0009 | +0.0014 | +0.0034 | No HR |
| QuadTrack | A1→A3: HR-enabled final, same Base-guided event pool | −8 | +0.0178 | +0.0093 | +0.0232 | 29 requests |
| QuadTrack | A2→A3: final policy vs rule trigger | −4 | +0.0008 | +0.0003 | +0.0008 | **−75.8% requests** |
| DTB | A0→A1: LR Base | −3 | +0.0000 | +0.0045 | +0.0059 | No HR |
| DTB | A1→A3: HR-enabled final, same Base-guided event pool | −2 | +0.0202 | +0.0057 | +0.0076 | 18 requests |
| DTB | A2→A3: final policy vs rule trigger | +4 | +0.0026 | −0.0037 | −0.0050 | **−66.0% requests** |

### Findings

- On QuadTrack, A3 is best on IDSW, IDF1, HOTA, AssA and MOTA while reducing HR requests from 120 to 29 relative to A2.
- On DTB zero-shot, A2 retains better IDSW/HOTA/AssA, whereas A3 gives the best IDF1 with 66.0% fewer HR requests. Base-Guided gating therefore trades some zero-shot association recall for efficiency.
- A1→A3 shows that HR evidence is useful under the shared Base-guided candidate pool: all identity metrics improve on both datasets.
- A2 and A3 are not prediction-identical because A2 uses the original rule trigger while A3 changes candidate proposal, HR gating and safety. Equality was checked rather than assumed.

Detailed ablation report: [REPORT.md](hive_ecia_ablation/frozen_hive_ecia_v1/REPORT.md).
<!-- HIVE_ECIA_ABLATION_END -->

<!-- BASE_GUIDED_GATING_ABLATION_START -->
## Controlled Base-Guided HR gating ablation

This controlled experiment isolates only the pre-query HR gate. G0 and G1 use the same Base-Guided candidate generator, frozen HIVE/ECIA checkpoints, thresholds, pre-query safety, final identity safety and raw BoT-SORT tracks.

- **G0 — Gating OFF:** request one HIVE-selected HR crop for every pre-query-safe candidate before the final ECIA decision.
- **G1 — Gating ON (Ours):** request HR only when the frozen ECIA Base policy identifies an uncertain but recoverable association.

![Controlled Base-Guided HR gating ablation](figures/base_guided_gating_ablation_summary.png)

**Figure:** With identical candidate pools and raw BoT-SORT tracks, Base-Guided gating preserves all QuadTrack identity metrics while reducing physical HR requests by 71.6%. On DTB zero-shot, it reduces HR requests by 61.7% and prevents the harmful forced-HR merge responsible for two additional ID switches in G0.

### Aggregate gating results

| Dataset | Variant | IDSW ↓ | IDF1 ↑ | HOTA ↑ | AssA ↑ | HR requests ↓ | Encoded HR MB ↓ | Total MB ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| QuadTrack held-out (512×120) | G0 Gating OFF | **78** | **0.4663** | **0.3725** | **0.4628** | 102 | 0.504 | 51.812 |
| QuadTrack held-out (512×120) | **G1 Gating ON (Ours)** | **78** | **0.4663** | **0.3725** | **0.4628** | **29** | **0.167** | **51.475** |
| DTB zero-shot (640×320) | G0 Gating OFF | 91 | 0.3721 | 0.3521 | 0.2403 | 47 | 0.153 | 181.259 |
| DTB zero-shot (640×320) | **G1 Gating ON (Ours)** | **89** | **0.3722** | **0.3525** | **0.2408** | **18** | **0.055** | **181.161** |

### Paired verification and savings

| Dataset | Same candidates | Same raw BoT-SORT | Same final MOT | HR request reduction | HR payload reduction | Total-bandwidth reduction |
|---|---:|---:|---:|---:|---:|---:|
| QuadTrack | Yes | Yes | Yes, all four sequences | **71.57%** (102→29) | **66.87%** | 0.6506% |
| DTB zero-shot | Yes | Yes | No, `seam` differs | **61.70%** (47→18) | **63.91%** | 0.0540% |

G1 exactly reproduces the frozen A3 reference on both datasets. On QuadTrack, gating removes 73 unnecessary HR queries while preserving byte-identical final predictions. On DTB, Gating OFF introduces one harmful HR override at frame 744 of `seam`: Base predicts Abstain (`P(old)=0.1569`, `P(abstain)=0.8431`), but the forced HR query selects historical frame 734 and raises `P(old)` to 0.8426, causing an incorrect merge. This changes 25 MOT rows over frames 744–774 and adds two ID switches. Thus, Base-Guided gating acts as both an acquisition-efficiency mechanism and protection against noisy zero-shot HR evidence.

HR acquisition savings and total transmitted-bandwidth savings are deliberately reported separately. The latter is smaller because continuous LR panorama transmission dominates the byte total. Runtime values are omitted from the paper table because G0 and the reused frozen G1 artifact were measured in different executions and are not suitable for a controlled latency claim.

Detailed gating report: [REPORT.md](base_guided_gating_ablation/base_guided_gating_v1/REPORT.md).
<!-- BASE_GUIDED_GATING_ABLATION_END -->

<!-- BANDWIDTH_ACCURACY_BENCHMARK_START -->
## Bandwidth–accuracy benchmark

Actual transmitted volume is modeled by in-memory JPEG encoding at quality 90. Computed Mbps is encoded bytes divided by video duration; it is not measured network throughput.

![BoT-SORT bandwidth and identity summary](figures/bandwidth_accuracy_botsort_summary.png)

**Figure:** Fair tracker-family comparison. HIVE–ECIA uses the least bandwidth on both datasets while giving the best QuadTrack IDSW; DTB remains a competitive zero-shot trade-off.

### QuadTrack held-out test (512×120)

| Acquisition | Tracker | IDSW ↓ | IDF1 ↑ | HOTA ↑ | AssA ↑ | MOTA ↑ | Total MB ↓ | Mbps ↓ | HR requests | HR frames/crops | Saving vs Full-HR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full-HR | SORT | 261 | 0.4290 | 0.3418 | 0.3120 | 0.3898 | 477.64 | 47.76 | 0 | 2400 | 0.00% |
| Full-HR | ByteTrack | 229 | **0.4877** | 0.3885 | 0.3676 | **0.4296** | 477.64 | 47.76 | 0 | 2400 | 0.00% |
| Full-HR | BoT-SORT | 376 | 0.4830 | **0.3922** | 0.3870 | 0.3963 | 477.64 | 47.76 | 0 | 2400 | 0.00% |
| Periodic-HR N=5 | SORT | 144 | 0.3254 | 0.2665 | 0.2643 | 0.3171 | 136.63 | 13.66 | 0 | 480 | 71.39% |
| Periodic-HR N=5 | ByteTrack | 124 | 0.4187 | 0.3320 | 0.3528 | 0.3685 | 136.63 | 13.66 | 0 | 480 | 71.39% |
| Periodic-HR N=5 | BoT-SORT | 198 | 0.4300 | 0.3483 | 0.3957 | 0.3603 | 136.63 | 13.66 | 0 | 480 | 71.39% |
| Periodic-HR N=10 | SORT | 141 | 0.3197 | 0.2678 | 0.2655 | 0.3133 | 93.86 | 9.39 | 0 | 240 | 80.35% |
| Periodic-HR N=10 | ByteTrack | 121 | 0.3866 | 0.3221 | 0.3442 | 0.3560 | 93.86 | 9.39 | 0 | 240 | 80.35% |
| Periodic-HR N=10 | BoT-SORT | 175 | 0.3880 | 0.3288 | 0.3649 | 0.3487 | 93.86 | 9.39 | 0 | 240 | 80.35% |
| Periodic-HR N=30 | SORT | 140 | 0.3248 | 0.2719 | 0.2727 | 0.3139 | 65.57 | 6.56 | 0 | 80 | 86.27% |
| Periodic-HR N=30 | ByteTrack | 108 | 0.3564 | 0.3084 | 0.3220 | 0.3489 | 65.57 | 6.56 | 0 | 80 | 86.27% |
| Periodic-HR N=30 | BoT-SORT | 171 | 0.4025 | 0.3367 | 0.3873 | 0.3421 | 65.57 | 6.56 | 0 | 80 | 86.27% |
| HIVE–ECIA (Ours) | BoT-SORT | **78** | 0.4663 | 0.3725 | **0.4628** | 0.3547 | **51.47** | 5.15 | 29 | 28 | 89.22% |

![QuadTrack held-out test (512×120) bandwidth](figures/quadtrack_bandwidth_strategies.png)

![QuadTrack held-out test (512×120) accuracy versus bandwidth](figures/quadtrack_identity_tradeoff.png)

Detailed report: [quadtrack](bandwidth_accuracy_benchmark/bandwidth_accuracy_v1/quadtrack/report/REPORT.md)

### DTB zero-shot (640×320)

| Acquisition | Tracker | IDSW ↓ | IDF1 ↑ | HOTA ↑ | AssA ↑ | MOTA ↑ | Total MB ↓ | Mbps ↓ | HR requests | HR frames/crops | Saving vs Full-HR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full-HR | SORT | 178 | 0.3445 | 0.3769 | 0.2232 | 0.6997 | 874.61 | 42.24 | 0 | 4969 | 0.00% |
| Full-HR | ByteTrack | 145 | 0.3176 | **0.3784** | 0.2131 | **0.7215** | 874.61 | 42.24 | 0 | 4969 | 0.00% |
| Full-HR | BoT-SORT | 182 | 0.3038 | 0.3574 | 0.1994 | 0.7118 | 874.61 | 42.24 | 0 | 4969 | 0.00% |
| Periodic-HR N=5 | SORT | 94 | 0.3595 | 0.3466 | 0.2357 | 0.5628 | 320.08 | 15.46 | 0 | 996 | 63.40% |
| Periodic-HR N=5 | ByteTrack | **79** | 0.3753 | 0.3762 | 0.2548 | 0.5976 | 320.08 | 15.46 | 0 | 996 | 63.40% |
| Periodic-HR N=5 | BoT-SORT | 88 | **0.3962** | 0.3709 | **0.2561** | 0.6010 | 320.08 | 15.46 | 0 | 996 | 63.40% |
| Periodic-HR N=10 | SORT | 96 | 0.3595 | 0.3494 | 0.2395 | 0.5602 | 250.71 | 12.11 | 0 | 499 | 71.34% |
| Periodic-HR N=10 | ByteTrack | **79** | 0.3729 | 0.3704 | 0.2534 | 0.5814 | 250.71 | 12.11 | 0 | 499 | 71.34% |
| Periodic-HR N=10 | BoT-SORT | 85 | 0.3716 | 0.3591 | 0.2448 | 0.5859 | 250.71 | 12.11 | 0 | 499 | 71.34% |
| Periodic-HR N=30 | SORT | 106 | 0.3620 | 0.3488 | 0.2404 | 0.5522 | 204.51 | 9.88 | 0 | 168 | 76.62% |
| Periodic-HR N=30 | ByteTrack | 87 | 0.3257 | 0.3427 | 0.2201 | 0.5720 | 204.51 | 9.88 | 0 | 168 | 76.62% |
| Periodic-HR N=30 | BoT-SORT | 97 | 0.3669 | 0.3533 | 0.2404 | 0.5734 | 204.51 | 9.88 | 0 | 168 | 76.62% |
| HIVE–ECIA (Ours) | BoT-SORT | 89 | 0.3722 | 0.3525 | 0.2408 | 0.5691 | **181.16** | 8.75 | 18 | 18 | 79.29% |

![DTB zero-shot (640×320) bandwidth](figures/dtb_acquisition_detail/bandwidth_comparison.png)

**Figure:** Each acquisition strategy appears once because encoded bandwidth is independent of the downstream tracker.

![DTB zero-shot (640×320) accuracy versus bandwidth](figures/dtb_acquisition_detail/accuracy_vs_bandwidth.png)

**Figure:** Tracker-dependent accuracy is shown as separate SORT, ByteTrack and BoT-SORT curves. HIVE–ECIA is the highlighted BoT-SORT recovery point.

Detailed report: [dtb](bandwidth_accuracy_benchmark/bandwidth_accuracy_v1/dtb/report/REPORT.md)

### Interpretation boundary

- Full-HR and Periodic-HR are system-level acquisition baselines, not matched-budget HIVE ablations.
- Full/Periodic methods run real YOLO inference on the transmitted HR or LR frame. Tracker state stays in original panorama coordinates across resolution transitions.
- HIVE–ECIA uses frozen Existing Base-Guided Safety and counts continuous LR JPEG frames plus every requested HR crop transmission. Repeated requests are counted again.
- Ground truth is used only for offline MOT evaluation.

<!-- BANDWIDTH_ACCURACY_BENCHMARK_END -->
