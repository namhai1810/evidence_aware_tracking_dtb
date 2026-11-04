"""Combine QuadTrack and DTB online-aware ECIA results into one report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import METHODS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadtrack", type=Path, required=True)
    parser.add_argument("--dtb", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payloads = {
        "QuadTrack 512x120": json.loads(args.quadtrack.read_text(encoding="utf-8")),
        "DTB 640x320 zero-shot": json.loads(args.dtb.read_text(encoding="utf-8")),
    }
    lines = [
        "# Online-Aware ECIA end-to-end evaluation", "",
        "Checkpoint selection used QuadTrack train/model-validation; thresholds used only sequence `0030`. DTB was evaluation-only. All rows are full-video causal runs without oracle triggers.", "",
        "| Dataset | Method | IDSW | IDF1 | HOTA | AssA | MOTA | IDTP | IDFP | IDFN | Merge recall | Wrong merge | HR | Bandwidth bytes | FPS | Latency ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, payload in payloads.items():
        for method in METHODS:
            aggregate = payload["methods"][method]["aggregate"]
            timing = payload["methods"][method]["timing"]
            recovery = payload["methods"][method]["identity_recovery"]["aggregate"]
            lines.append(
                f"| {dataset} | {method} | {aggregate['id_switches']} | "
                f"{aggregate['idf1']:.4f} | {aggregate['hota']:.4f} | "
                f"{aggregate['assa']:.4f} | {aggregate['mota']:.4f} | "
                f"{aggregate['idtp']} | {aggregate['idfp']} | {aggregate['idfn']} | "
                f"{recovery['merge_recall']:.4f} | {recovery['wrong_merges']} | "
                f"{timing['hr_cost']['requests']} | {timing['hr_cost']['bytes_read']} | "
                f"{timing['fps']:.2f} | "
                f"{timing['mean_latency_ms']:.2f} |"
            )
    lines.extend([
        "", "## Controlled fine-tuned HR ablation", "",
        "| Dataset | HR-off IDSW | Full IDSW | HR-off IDF1 | Full IDF1 | HR-off HOTA | Full HOTA | HR requests |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset, payload in payloads.items():
        off = payload["methods"]["fine_tuned_hr_disabled"]
        full = payload["methods"]["fine_tuned_safety_full"]
        lines.append(
            f"| {dataset} | {off['aggregate']['id_switches']} | {full['aggregate']['id_switches']} | "
            f"{off['aggregate']['idf1']:.4f} | {full['aggregate']['idf1']:.4f} | "
            f"{off['aggregate']['hota']:.4f} | {full['aggregate']['hota']:.4f} | "
            f"{full['timing']['hr_cost']['requests']} |"
        )
    lines.extend([
        "", "Artifacts:", "",
        f"- Checkpoint: `{(args.training_root / 'training/best.pt').resolve()}`",
        f"- Calibration: `{(args.training_root / 'calibration/thresholds.json').resolve()}`",
        f"- Leakage audit: `{(args.training_root / 'leakage_audit.json').resolve()}`",
        f"- QuadTrack details: `{(args.quadtrack.parent / 'REPORT.md').resolve()}`",
        f"- DTB details: `{(args.dtb.parent / 'REPORT.md').resolve()}`",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Consolidated report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
