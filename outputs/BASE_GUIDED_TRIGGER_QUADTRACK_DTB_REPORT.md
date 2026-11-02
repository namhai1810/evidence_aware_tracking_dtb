# ECIA Base-Guided Online HR Trigger — QuadTrack and DTB

## Protocol audit

- QuadTrack held-out test uses physical LR `512x120`; DTB uses physical LR `640x320` zero-shot.
- Rule-based Full, Base-guided HR-disabled and Base-guided Full have byte-identical raw BoT-SORT predictions inside each dataset.
- Base-guided HR-disabled and Full share exactly the same proposed-event and would-request-HR schedules.
- GT is joined only after prediction for tracking metrics and trigger precision/recall. It is not available to BoT-SORT, Base, HIVE, ECIA or the identity registry.
- No model was retrained; HIVE v2, OSNet-AIN and Recovery-Aware ECIA remain frozen.

## Aggregate tracking results

### QuadTrack held-out test

| Method | IDSW ↓ | IDF1 ↑ | HOTA ↑ | AssA ↑ | MOTA ↑ | HR requests | Bandwidth |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule-based Full | 82 | 0.4656 | 0.3722 | 0.4620 | 0.3545 | 120 | 32.92 MB |
| Base-guided HR-disabled | 86 | 0.4485 | 0.3632 | 0.4396 | 0.3542 | 0 | 0 |
| **Base-guided Full** | **78** | **0.4663** | **0.3725** | **0.4628** | **0.3547** | **35** | **9.69 MB** |

Base-guided Full versus HR-disabled:

- IDSW: `86 → 78` (`-8`, 9.3% reduction).
- IDF1: `0.4485 → 0.4663` (`+0.0178`).
- HOTA: `0.3632 → 0.3725` (`+0.0093`).
- AssA: `0.4396 → 0.4628` (`+0.0232`).
- IDTP increases by 207; IDFP and IDFN each decrease by 207.

Base-guided Full also slightly exceeds Rule-based Full while using 70.8% fewer HR requests and about 70.6% less source-image bandwidth: IDSW `82 → 78`, IDF1 `+0.0008`, HOTA `+0.0003`, AssA `+0.0008`.

### DTB zero-shot at 640x320

| Method | IDSW ↓ | IDF1 ↑ | HOTA ↑ | AssA ↑ | MOTA ↑ | HR requests | Bandwidth |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Rule-based Full** | **85** | 0.3696 | **0.3562** | **0.2458** | **0.5696** | 53 | 12.67 MB |
| Base-guided HR-disabled | 91 | 0.3520 | 0.3468 | 0.2332 | 0.5689 | 0 | 0 |
| Base-guided Full | 89 | **0.3722** | 0.3525 | 0.2408 | 0.5691 | 33 | 7.91 MB |

Base-guided Full versus HR-disabled:

- IDSW: `91 → 89` (`-2`).
- IDF1: `0.3520 → 0.3722` (`+0.0202`).
- HOTA: `0.3468 → 0.3525` (`+0.0057`).
- AssA: `0.2332 → 0.2408` (`+0.0076`).
- IDTP increases by 147; IDFP and IDFN each decrease by 147.

Compared with Rule-based Full, Base-guided Full has slightly higher IDF1 (`+0.0026`) and 37.7% fewer HR requests, but is worse by 4 ID switches, `-0.0037` HOTA and `-0.0050` AssA. It therefore does not dominate the rule-based trigger on DTB.

## Trigger quality

| Dataset | Trigger | Events | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| QuadTrack | Rule-based | 120 | 22 | 98 | 70 | 0.1833 | 0.2391 | 0.2075 |
| QuadTrack | **Base-guided HR request** | **35** | **17** | **18** | 75 | **0.4857** | 0.1848 | **0.2677** |
| DTB | **Rule-based** | 53 | 20 | 33 | 74 | **0.3774** | **0.2128** | **0.2721** |
| DTB | Base-guided HR request | 33 | 11 | 22 | 83 | 0.3333 | 0.1170 | 0.1732 |

On QuadTrack, Base confidence provides a much cleaner HR trigger: precision rises from 18.3% to 48.6% while F1 also improves. On DTB, however, Base-guided recall falls to 11.7%; the trigger misses too many raw BoT-SORT switch events.

## What HR evidence changes

### QuadTrack

- 145 Base-guided candidates: 103 Lost→New and 42 Active→Active.
- Policy actions: 101 abstain, 35 request HR, 9 direct merge.
- Base-only applies 6 safe merges.
- Full applies 20 safe merges: HR contributes 14 additional applied merges.
- These additional merges coincide with the aggregate reduction of 8 ID switches and improvement in IDF1/HOTA/AssA. This aggregate result does not imply that all 14 individual merges are correct.
- 35 HR requests produce 34 crops; one request has no usable HIVE history/candidate.

### DTB

- 80 Base-guided candidates: 47 Lost→New and 33 Active→Active.
- Policy actions: 32 abstain, 33 request HR, 15 direct merge.
- Base-only applies 4 safe merges.
- Full applies 11 safe merges: HR contributes 7 additional applied merges.
- The 7 extra applied merges improve IDF1 substantially and reduce IDSW by 2, but are insufficient to match the rule trigger's total IDSW/HOTA.

## Active→Active safety bottleneck

This is the clearest verified inefficiency in the current online integration.

- QuadTrack: 6/35 HR requests are Active→Active. None can be applied; one HR merge proposal is rejected because the old identity is still active.
- DTB: 15/33 HR requests are Active→Active. Eight produce an ECIA merge proposal, but all eight are rejected by `old_identity_still_active`; the other seven abstain.
- DTB also has 11 direct Base merges for Active→Active candidates, all rejected by the same safety guard.

Therefore 45.5% of DTB HR requests are spent on candidates that the current identity registry cannot merge by construction. Candidate generation and safety are individually working as designed, but their combination wastes HR budget. Until a collision-safe delayed Active→Active resolution is implemented, these events should not request HR.

## Per-sequence findings

### QuadTrack

- `0016`: Full versus HR-disabled improves IDSW `11→9`, IDF1 `0.3016→0.3392` and HOTA `0.2719→0.2924`.
- `0017`: IDSW stays 17, but IDF1 improves `0.3379→0.3778` and AssA `0.3470→0.4139`; HR improves trajectory consistency without changing the switch count.
- `0025`: Full reduces IDSW `57→51` and improves all identity metrics.
- `0031`: all methods are identical; no measurable contribution from HR.

### DTB

- `crossing`: HR recovers one ID switch and reaches the rule-based result.
- `full_occ`: strongest HR gain; IDSW `20→18`, IDF1 `0.2015→0.2738`, HOTA `0.1858→0.2089`.
- `normal_walking` and `partial_occ`: HR does not change the Base-only result.
- `seam`: IDSW worsens `32→33`, but IDF1 improves `0.2633→0.2984` and HOTA `0.2418→0.2619`; switch count alone does not describe the trajectory-level effect.

## Runtime and HR cost

| Dataset | Method | Recovery-component time | Events | Mean |
|---|---|---:|---:|---:|
| QuadTrack | Rule-based Full | 5384.8 ms | 120 | 44.9 ms/trigger |
| QuadTrack | Base-guided Full HR path | 636.8 ms | 35 | 18.2 ms/request |
| DTB | Rule-based Full | 1722.5 ms | 53 | 32.5 ms/trigger |
| DTB | Base-guided Full HR path | 662.0 ms | 33 | 20.1 ms/request |

The Base-guided path is cheaper because it requests fewer HR crops and reuses cached LR embeddings/Base state. End-to-end FPS was measured in separate executions and contains detector/runtime variability, so small FPS inversions between HR-on and HR-off should not be interpreted as HR making inference faster.

## Conclusion

The ablation verifies that HR evidence is useful: Base-guided Full consistently improves IDSW, IDF1, HOTA and AssA over the identical Base-guided HR-disabled event pool on both datasets. On QuadTrack it is also more accurate and far more HR-efficient than the rule trigger. On DTB zero-shot, the ECIA recovery module still helps, but the Base-guided trigger does not generalize well enough to replace the rule trigger: recall is low and nearly half of its HR budget is consumed by non-actionable Active→Active conflicts.

The primary bottleneck is now candidate/trigger policy and its compatibility with registry safety, not evidence that HR is useless. The highest-priority correction is to suppress unsafe Active→Active HR requests or add a delayed collision-safe resolution mechanism, then calibrate the request thresholds on validation only and keep DTB/test held out.
