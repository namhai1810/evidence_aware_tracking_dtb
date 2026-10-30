"""Compare frozen DTB ECIA runs across physical detector/tracker resolutions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=RUN_ROOT",
        help="Repeat for every resolution run",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Invalid --run {value!r}; expected LABEL=RUN_ROOT")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path).resolve()
    if not label or not path.is_dir():
        raise ValueError(f"Invalid resolution run: {value!r}")
    return label, path


def load_run(label: str, root: Path) -> dict:
    audit = load_json(root / "event_protocol" / "leakage_audit.json")
    base = load_json(root / "base_only" / "metrics.json")
    full = load_json(root / "full" / "metrics.json")
    comparison = load_json(root / "final_report" / "comparison.json")
    if not (
        audit.get("passed")
        and audit.get("sequence_disjoint")
        and audit.get("physical_lr_detection_tracking")
    ):
        raise RuntimeError(f"Physical-LR/leakage audit failed for {root}")
    expected_size = "x".join(str(int(value)) for value in audit["physical_lr_size"])
    if label != expected_size:
        raise ValueError(f"Label {label!r} does not match audited size {expected_size!r}")
    if base["test_baseline"] != full["test_baseline"]:
        raise RuntimeError(f"Unpaired tracker baseline for {label}")
    baseline = full["test_baseline"]["aggregate"]
    base_metrics = base["test_ecia_base_only"]["aggregate"]
    full_metrics = full["test_ecia_full"]["aggregate"]
    heldout = "full_occ"
    return {
        "label": label,
        "width": int(audit["physical_lr_size"][0]),
        "height": int(audit["physical_lr_size"][1]),
        "pixels": int(audit["physical_lr_size"][0]) * int(audit["physical_lr_size"][1]),
        "run_root": str(root),
        "baseline": baseline,
        "base_only": base_metrics,
        "full": full_metrics,
        "correction_base_only": base["correction"],
        "correction_full": full["correction"],
        "eligible_events": int(full["correction"]["events"]),
        "hive_selection_posthoc": comparison.get("hive_selection_posthoc"),
        "heldout": {
            "baseline": full["test_baseline"]["sequences"][heldout],
            "base_only": base["test_ecia_base_only"]["sequences"][heldout],
            "full": full["test_ecia_full"]["sequences"][heldout],
        },
        "fingerprints": {
            "hive": full["protocol"]["hive_checkpoint_sha256"],
            "reid": full["protocol"]["reid_model_sha256"],
            "base_ecia": base["protocol"]["ecia_checkpoint_sha256"],
            "full_ecia": full["protocol"]["ecia_checkpoint_sha256"],
            "base_thresholds": base["protocol"]["frozen_quadtrack_thresholds"],
            "full_thresholds": full["protocol"]["frozen_quadtrack_thresholds"],
        },
    }


def verify_frozen_protocol(rows: list[dict]) -> None:
    reference = rows[0]["fingerprints"]
    for row in rows[1:]:
        if row["fingerprints"] != reference:
            raise RuntimeError(f"Frozen model/threshold mismatch for {row['label']}")


def plot_detection(output: Path, rows: list[dict]) -> None:
    labels = [row["label"] for row in rows]
    x = np.arange(len(rows))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    precision = [row["baseline"]["precision"] for row in rows]
    recall = [row["baseline"]["recall"] for row in rows]
    width = 0.34
    p = axes[0].bar(x - width / 2, precision, width, label="Precision", color="#2563eb")
    r = axes[0].bar(x + width / 2, recall, width, label="Recall", color="#f59e0b")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Detection quality versus physical input size")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()
    axes[0].bar_label(p, labels=[f"{value:.3f}" for value in precision], padding=3)
    axes[0].bar_label(r, labels=[f"{value:.3f}" for value in recall], padding=3)

    switches = [row["baseline"]["id_switches"] for row in rows]
    events = [row["eligible_events"] for row in rows]
    s = axes[1].bar(x - width / 2, switches, width, label="Baseline IDSW", color="#64748b")
    e = axes[1].bar(x + width / 2, events, width, label="Eligible events", color="#8b5cf6")
    axes[1].set_xticks(x, labels)
    axes[1].set_title("Identity-switch opportunity versus input size")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    axes[1].bar_label(s, padding=3)
    axes[1].bar_label(e, padding=3)
    figure.savefig(output / "resolution_detection_and_events.png", dpi=180)
    plt.close(figure)


def plot_recovery(output: Path, rows: list[dict]) -> None:
    labels = [row["label"] for row in rows]
    x = np.arange(len(rows))
    width = 0.25
    methods = (
        ("baseline", "Tracker baseline", "#64748b"),
        ("base_only", "ECIA Base-only", "#f59e0b"),
        ("full", "ECIA Full + HR", "#10b981"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for axis, (metric, title, fmt) in zip(
        axes,
        (("id_switches", "IDSW ↓", "{:.0f}"), ("idf1", "IDF1 ↑", "{:.4f}"), ("hota", "HOTA ↑", "{:.4f}")),
    ):
        for index, (key, name, color) in enumerate(methods):
            values = [float(row[key][metric]) for row in rows]
            bars = axis.bar(x + (index - 1) * width, values, width, label=name, color=color)
            axis.bar_label(bars, labels=[fmt.format(value) for value in values], padding=2, fontsize=8)
        axis.set_xticks(x, labels)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    figure.suptitle("Frozen ECIA recovery across physical DTB resolutions")
    figure.savefig(output / "resolution_recovery_metrics.png", dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Output exists; refusing to overwrite: {output}")
    output.mkdir(parents=True)
    rows = [load_run(*parse_run(value)) for value in args.run]
    rows.sort(key=lambda row: row["pixels"])
    verify_frozen_protocol(rows)

    payload = {
        "protocol": {
            "only_physical_input_resolution_varies": True,
            "same_manual_gt": True,
            "same_detector_tracker_family": True,
            "same_hive_reid_ecia_and_thresholds": True,
            "all_dtb_aggregate_is_exploratory_zero_shot": True,
            "strict_heldout_sequence": "full_occ",
        },
        "resolutions": rows,
    }
    (output / "resolution_comparison.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    plot_detection(output, rows)
    plot_recovery(output, rows)

    report = [
        "# DTB physical-resolution sweep with frozen HIVE + ECIA",
        "",
        "Only the physical detector/tracker input resolution changes. YOLO11l, BoT-SORT/ReID, confidence/IoU thresholds, manual GT, frozen HIVE/ReID/ECIA checkpoints, and QuadTrack-calibrated merge thresholds are identical.",
        "",
        "## Detection and recovery opportunity",
        "",
        "| Size | Precision | Recall | TP | FN | Baseline IDSW | Eligible events | HIVE correct frame |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        hive = row["hive_selection_posthoc"]
        hive_text = f"{hive['correct']}/{hive['total']} ({100*hive['accuracy']:.1f}%)" if hive else "n/a"
        report.append(
            f"| {row['label']} | {row['baseline']['precision']:.4f} | {row['baseline']['recall']:.4f} | "
            f"{row['baseline']['tp']} | {row['baseline']['fn']} | {row['baseline']['id_switches']} | "
            f"{row['eligible_events']} | {hive_text} |"
        )
    report.extend(
        [
            "",
            "## All-video exploratory zero-shot recovery",
            "",
            "| Size | Method | IDSW ↓ | ΔIDSW | IDF1 ↑ | ΔIDF1 | HOTA ↑ | Fixed | Wrong |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        baseline = row["baseline"]
        for key, name, correction in (
            ("baseline", "Tracker baseline", None),
            ("base_only", "ECIA Base-only", row["correction_base_only"]),
            ("full", "ECIA Full + HR", row["correction_full"]),
        ):
            metrics = row[key]
            report.append(
                f"| {row['label']} | {name} | {metrics['id_switches']} | "
                f"{metrics['id_switches'] - baseline['id_switches']:+d} | {metrics['idf1']:.4f} | "
                f"{metrics['idf1'] - baseline['idf1']:+.4f} | {metrics['hota']:.4f} | "
                f"{0 if correction is None else correction['actually_fixed']} | "
                f"{0 if correction is None else correction['wrong_corrections']} |"
            )
    report.extend(
        [
            "",
            "## Strict held-out `full_occ`",
            "",
            "| Size | Baseline IDSW | Base-only IDSW | Full IDSW | Baseline IDF1 | Base-only IDF1 | Full IDF1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        held = row["heldout"]
        report.append(
            f"| {row['label']} | {held['baseline']['id_switches']} | {held['base_only']['id_switches']} | "
            f"{held['full']['id_switches']} | {held['baseline']['idf1']:.4f} | "
            f"{held['base_only']['idf1']:.4f} | {held['full']['idf1']:.4f} |"
        )
    report.extend(
        [
            "",
            "The five-video aggregate is exploratory because every DTB video has now been observed. The `full_occ` row is the predeclared strict held-out result. More ID switches at a higher resolution provide more recovery opportunities, but do not alone imply a better recovery model; IDF1/HOTA and wrong corrections remain authoritative.",
            "",
            "![Detection and event coverage](resolution_detection_and_events.png)",
            "",
            "![Recovery metrics](resolution_recovery_metrics.png)",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Resolution sweep report: {output / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
