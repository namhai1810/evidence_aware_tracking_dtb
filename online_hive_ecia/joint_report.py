"""Create one concise index report for QuadTrack and DTB online benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadtrack", type=Path, required=True)
    parser.add_argument("--dtb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    quad = load(args.quadtrack.resolve())
    dtb = load(args.dtb.resolve())
    datasets = (("QuadTrack held-out test", quad), ("DTB all sequences", dtb))
    lines = [
        "# Online HIVE--ECIA benchmark: QuadTrack and DTB",
        "",
        "Both datasets use the same frozen HIVE v2, OSNet-AIN and Recovery-Aware ECIA checkpoints. QuadTrack uses its established 512x120 LR protocol; DTB uses the requested 640x320 physical LR input. GT is used only for post-hoc evaluation.",
        "",
        "| Dataset | Physical LR | Method | IDSW | IDF1 | HOTA | AssA | MOTA | FPS | HR crops |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset_name, payload in datasets:
        width, height = payload["protocol"]["paired_run_settings"]["physical_lr_size"]
        for method, result in payload["methods"].items():
            metric = result["aggregate"]
            timing = result["timing"]
            lines.append(
                f"| {dataset_name} | {width}x{height} | {method} | {metric['id_switches']} | "
                f"{metric['idf1']:.4f} | {metric['hota']:.4f} | {metric['assa']:.4f} | "
                f"{metric['mota']:.4f} | {timing['fps']:.2f} | "
                f"{timing['hr_cost']['crops']} |"
            )
    lines.extend(
        [
            "",
            "## Ambiguity-trigger accuracy",
            "",
            "| Dataset | Triggers | True IDSW events | TP | FP | FN | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset_name, payload in datasets:
        row = payload["ambiguity_trigger"]["aggregate"]
        lines.append(
            f"| {dataset_name} | {row['triggers']} | {row['oracle_switches']} | "
            f"{row['tp']} | {row['fp']} | {row['fn']} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"- QuadTrack details: `{args.quadtrack.resolve().parent / 'REPORT.md'}`",
            f"- DTB details: `{args.dtb.resolve().parent / 'REPORT.md'}`",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Joint report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

