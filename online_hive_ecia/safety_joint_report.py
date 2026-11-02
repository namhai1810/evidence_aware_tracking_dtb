"""Create one concise QuadTrack + DTB pre-query safety report."""

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


def main() -> int:
    args = parse_args()
    payloads = {
        "QuadTrack 512x120": json.loads(args.quadtrack.read_text(encoding="utf-8")),
        "DTB 640x320": json.loads(args.dtb.read_text(encoding="utf-8")),
    }
    lines = [
        "# Pre-Query Safety Check — consolidated report", "",
        "The safety-aware method reuses the Base-Guided Full event pool and final "
        "registry guards, but rejects guaranteed-infeasible merges before HIVE/HR.", "",
        "| Dataset | Verification | Baseline HR | Safety HR | Saved | Reduction | Bytes saved |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for label, payload in payloads.items():
        protocol = payload["protocol"]
        savings = payload["savings"]
        lines.append(
            f"| {label} | {'PASS' if protocol['optimization_goal_met'] else 'FAIL'} | "
            f"{savings['baseline_hr_requests']} | {savings['safety_hr_requests']} | "
            f"{savings['saved_hr_requests']} | {savings['hr_request_reduction']:.2%} | "
            f"{savings['bytes_saved']} |"
        )
    lines.extend([
        "", "## Tracking metrics", "",
        "| Dataset | Method | IDSW | IDF1 | HOTA | AssA | MOTA |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for label, payload in payloads.items():
        for method in ("base_guided_full", "safety_aware_full"):
            aggregate = payload["methods"][method]["aggregate"]
            lines.append(
                f"| {label} | {method} | {aggregate['id_switches']} | "
                f"{aggregate['idf1']:.4f} | {aggregate['hota']:.4f} | "
                f"{aggregate['assa']:.4f} | {aggregate['mota']:.4f} |"
            )
    lines.extend([
        "", "Detailed reports:", "",
        f"- QuadTrack: `{args.quadtrack.parent / 'REPORT.md'}`",
        f"- DTB: `{args.dtb.parent / 'REPORT.md'}`",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Consolidated report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
