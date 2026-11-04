"""Evaluate acquisition baselines, account HIVE JPEG bytes and create reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from evaluate_mot import load_mot
from hive_hr_reid.run_experiment import basic_metrics
from mot_baseline_benchmark.protocol import resolve_sequence_dir, write_json

from .hive_accounting import account_hive_sequence, aggregate as aggregate_hive


TRACKERS = ("sort", "bytetrack", "botsort")
LABELS = {"sort": "SORT", "bytetrack": "ByteTrack", "botsort": "BoT-SORT"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--dataset-key", choices=("quadtrack", "dtb"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--acquisition-run", type=Path, required=True)
    parser.add_argument("--hive-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_bandwidth(run: Path, strategy: str, sequences: list[str], fps: float) -> tuple[dict, dict]:
    rows = {
        sequence: json.loads(
            (run / "bandwidth" / strategy / f"{sequence}.json").read_text(encoding="utf-8")
        )
        for sequence in sequences
    }
    frames = sum(int(row["lr_frames"] + row["hr_frames"]) for row in rows.values())
    lr_bytes = sum(int(row["lr_bytes"]) for row in rows.values())
    hr_bytes = sum(int(row["hr_bytes"]) for row in rows.values())
    total = lr_bytes + hr_bytes
    duration = frames / fps
    aggregate = {
        "lr_bytes": lr_bytes,
        "hr_bytes": hr_bytes,
        "total_bytes": total,
        "total_mb": total / 1e6,
        "duration_seconds": duration,
        "megabits_per_second": 8 * total / 1e6 / duration if duration else 0.0,
        "bytes_per_frame": total / frames if frames else 0.0,
        "lr_frames": sum(int(row["lr_frames"]) for row in rows.values()),
        "hr_frames": sum(int(row["hr_frames"]) for row in rows.values()),
        "hr_crops": 0,
        "hr_requests": 0,
        "jpeg_encoding_ms_total": sum(float(row["jpeg_encoding_ms_total"]) for row in rows.values()),
        "jpeg_encoding_ms_per_frame": (
            sum(float(row["jpeg_encoding_ms_total"]) for row in rows.values()) / frames
            if frames else 0.0
        ),
    }
    return aggregate, rows


def load_timing(run: Path, strategy: str, tracker: str, sequences: list[str]) -> dict:
    rows = [
        json.loads((run / "timing" / strategy / tracker / f"{sequence}.json").read_text())
        for sequence in sequences
    ]
    frames = sum(int(row["frames"]) for row in rows)
    detector_ms = sum(float(row["detector_ms_total"]) for row in rows)
    tracker_ms = sum(float(row["tracker_ms_total"]) for row in rows)
    jpeg_ms = sum(float(row["jpeg_encoding_ms_total"]) for row in rows)
    pipeline_ms = detector_ms + tracker_ms
    return {
        "frames": frames,
        "detector_ms_total": detector_ms,
        "detector_ms_per_frame": detector_ms / frames if frames else 0.0,
        "tracker_ms_total": tracker_ms,
        "tracker_ms_per_frame": tracker_ms / frames if frames else 0.0,
        "jpeg_encoding_ms_total": jpeg_ms,
        "jpeg_encoding_ms_per_frame": jpeg_ms / frames if frames else 0.0,
        "pipeline_latency_ms_excluding_jpeg": pipeline_ms / frames if frames else 0.0,
        "pipeline_fps_excluding_jpeg": frames / (pipeline_ms / 1000.0) if pipeline_ms else 0.0,
        "timing_scope": "detector + tracker; excludes decode, resize, JPEG encoding and writes",
    }


def load_hive_timing(run: Path, sequences: list[str]) -> dict:
    rows = [json.loads((run / "timing" / f"{sequence}.json").read_text()) for sequence in sequences]
    frames = sum(int(row["frames"]) for row in rows)
    wall = sum(float(row["wall_seconds"]) for row in rows)
    engine = [row["engine"] for row in rows]
    hr_ms = sum(float(row["latency"].get("hr_ms", 0.0)) for row in engine)
    return {
        "frames": frames,
        "detector_ms_total": None,
        "detector_ms_per_frame": None,
        "tracker_ms_total": None,
        "tracker_ms_per_frame": None,
        "hr_retrieval_ms_total": hr_ms,
        "hr_retrieval_ms_per_frame": hr_ms / frames if frames else 0.0,
        "pipeline_latency_ms_excluding_jpeg": 1000.0 * wall / frames if frames else 0.0,
        "pipeline_fps_excluding_jpeg": frames / wall if wall else 0.0,
        "timing_scope": "detector + BoT-SORT + online recovery controller; excludes decode, resize, JPEG encoding and writes",
    }


def metrics_for(mot_root: Path, sequences: list[str], gt: dict, iou: float) -> dict:
    predictions = {
        sequence: load_mot(mot_root / f"{sequence}.txt", ground_truth=False)
        for sequence in sequences
    }
    return basic_metrics(
        argparse.Namespace(iou=iou), sequences, predictions, gt, include_hota=True
    )


def expected_difference(actual: dict, expected: dict) -> dict:
    differences = {}
    for key, value in expected.items():
        observed = actual[key]
        if isinstance(value, int):
            if int(observed) != value:
                differences[key] = {"expected": value, "actual": observed}
        elif round(float(observed), 4) != round(float(value), 4):
            differences[key] = {"expected": value, "actual": observed}
    return differences


def bold(value, best, spec="") -> str:
    text = format(value, spec)
    return f"**{text}**" if abs(float(value) - float(best)) <= 1e-12 else text


def plot_results(methods: dict, output: Path, dataset_label: str) -> None:
    """Plot acquisition payload once and tracker-specific accuracy separately."""

    output.mkdir(parents=True, exist_ok=True)
    strategies = ("full_hr", "periodic_5", "periodic_10", "periodic_30")
    strategy_labels = ("Full-HR", "Periodic-HR N=5", "Periodic-HR N=10", "Periodic-HR N=30")
    representative = [methods[f"{strategy}_botsort"] for strategy in strategies]
    bandwidth = [row["bandwidth"]["total_mb"] for row in representative]
    bandwidth.append(methods["hive_ecia"]["bandwidth"]["total_mb"])
    acquisition_labels = [*strategy_labels, "HIVE–ECIA (Ours)"]
    acquisition_colors = ["#9CA3AF", "#78A6C8", "#5F91B5", "#467A9F", "#E76F51"]

    # Bandwidth is an acquisition property, not a tracker property. Each
    # strategy therefore appears exactly once instead of being repeated for
    # SORT, ByteTrack and BoT-SORT.
    fig, axis = plt.subplots(figsize=(10.8, 4.8))
    positions = np.arange(len(acquisition_labels))
    bars = axis.barh(positions, bandwidth, color=acquisition_colors, height=0.68)
    bars[-1].set_edgecolor("#9E321D")
    bars[-1].set_linewidth(2.0)
    bars[-1].set_hatch("//")
    axis.bar_label(bars, labels=[f"{value:.2f}" for value in bandwidth], padding=4, fontsize=8.5)
    axis.set_yticks(positions, acquisition_labels, fontsize=9)
    axis.invert_yaxis()
    axis.get_yticklabels()[-1].set_color("#B33A20")
    axis.get_yticklabels()[-1].set_fontweight("bold")
    axis.set_xlabel("Transmitted JPEG data (MB) ↓")
    axis.set_title(f"Unique acquisition payload — {dataset_label}", loc="left", fontweight="bold")
    axis.grid(axis="x", alpha=.2)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "bandwidth_comparison.png", dpi=180)
    plt.close(fig)

    # Accuracy remains tracker-specific. Lines prevent equal-bandwidth points
    # from looking like duplicate acquisition measurements.
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    x = np.arange(5)
    tracker_colors = {"sort": "#7C8798", "bytetrack": "#4E8FC1", "botsort": "#387F68"}
    tracker_markers = {"sort": "o", "bytetrack": "s", "botsort": "D"}
    metric_specs = ((axes[0], "idf1", "IDF1 ↑"), (axes[1], "id_switches", "ID switches ↓"))
    for axis, metric, ylabel in metric_specs:
        for tracker in TRACKERS:
            values = [
                methods[f"{strategy}_{tracker}"]["metrics"]["aggregate"][metric]
                for strategy in strategies
            ]
            axis.plot(
                x[:4], values, marker=tracker_markers[tracker], color=tracker_colors[tracker],
                linewidth=1.8, markersize=6, label=LABELS[tracker],
            )
            if tracker == "botsort":
                hive_value = methods["hive_ecia"]["metrics"]["aggregate"][metric]
                axis.plot(x[3:5], [values[-1], hive_value], "--", color="#B33A20", linewidth=2.0)
                axis.scatter(x[-1], hive_value, marker="*", s=230, color="#E76F51", edgecolor="#8C2F1B", linewidth=1.4, zorder=5, label="HIVE–ECIA")
                if metric == "idf1":
                    relative = (hive_value - values[-1]) / values[-1] if values[-1] else 0.0
                    change = f"+{relative:.1%} vs BoT-SORT N=30"
                else:
                    relative = (values[-1] - hive_value) / values[-1] if values[-1] else 0.0
                    change = f"−{relative:.1%} vs BoT-SORT N=30"
                axis.annotate(
                    change, (x[-1], hive_value), xytext=(-8, 12), textcoords="offset points",
                    ha="right", color="#B33A20", fontsize=8.5, fontweight="bold",
                )
        axis.axvspan(3.65, 4.35, color="#FCE8E1", alpha=0.75, zorder=0)
        axis.set_xticks(
            x,
            [
                "Full-HR", "N=5", "N=10", "N=30",
                f"HIVE–ECIA\n{bandwidth[-1]:.1f} MB",
            ],
            fontsize=8.5,
        )
        axis.get_xticklabels()[-1].set_color("#B33A20")
        axis.get_xticklabels()[-1].set_fontweight("bold")
        axis.set_xlabel("Acquisition strategy (encoded payload shown for HIVE–ECIA)")
        axis.set_ylabel(ylabel)
        axis.set_title("Identity accuracy by tracker" if metric == "idf1" else "Identity errors by tracker", loc="left", fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, ncol=2, fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle(f"Tracker-specific accuracy across acquisition strategies — {dataset_label}", x=0.055, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.93))
    fig.savefig(output / "accuracy_vs_bandwidth.png", dpi=180)
    plt.close(fig)


def report(payload: dict, output: Path) -> None:
    methods = payload["methods"]
    best = {
        "idsw": min(row["metrics"]["aggregate"]["id_switches"] for row in methods.values()),
        "idf1": max(row["metrics"]["aggregate"]["idf1"] for row in methods.values()),
        "hota": max(row["metrics"]["aggregate"]["hota"] for row in methods.values()),
        "assa": max(row["metrics"]["aggregate"]["assa"] for row in methods.values()),
        "mota": max(row["metrics"]["aggregate"]["mota"] for row in methods.values()),
        "mb": min(row["bandwidth"]["total_mb"] for row in methods.values()),
        "mbps": min(row["bandwidth"]["megabits_per_second"] for row in methods.values()),
    }
    full_bytes = payload["strategy_bandwidth"]["full_hr"]["total_bytes"]
    lines = [
        f"# Bandwidth–accuracy benchmark — {payload['dataset_label']}", "",
        "All byte counts are actual in-memory JPEG encodings at quality "
        f"`{payload['jpeg_quality']}`; Mbps is computed from bytes and video duration and is not measured network throughput.", "",
        "## Aggregate comparison", "",
        "| Method | Tracker | IDSW ↓ | IDF1 ↑ | HOTA ↑ | AssA ↑ | MOTA ↑ | Total MB ↓ | Mbps ↓ | LR MB | HR MB | HR requests | HR frames/crops | Saving vs Full-HR | FPS ↑ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in methods.items():
        metric = row["metrics"]["aggregate"]
        bw = row["bandwidth"]
        saving = 1.0 - bw["total_bytes"] / full_bytes if full_bytes else 0.0
        hr_count = bw["hr_frames"] + bw["hr_crops"]
        lines.append(
            f"| {row['display']} | {LABELS[row['tracker']]} | "
            f"{bold(metric['id_switches'], best['idsw'])} | {bold(metric['idf1'], best['idf1'], '.4f')} | "
            f"{bold(metric['hota'], best['hota'], '.4f')} | {bold(metric['assa'], best['assa'], '.4f')} | "
            f"{bold(metric['mota'], best['mota'], '.4f')} | {bold(bw['total_mb'], best['mb'], '.2f')} | "
            f"{bold(bw['megabits_per_second'], best['mbps'], '.2f')} | {bw['lr_bytes']/1e6:.2f} | "
            f"{bw['hr_bytes']/1e6:.2f} | {bw.get('hr_requests', 0)} | "
            f"{hr_count} | {saving:.2%} | "
            f"{row['timing']['pipeline_fps_excluding_jpeg']:.2f} |"
        )
    lines.extend([
        "", "## HIVE bandwidth reduction", "",
        "| Reference acquisition | Reference MB | HIVE–ECIA MB | Reduction |",
        "|---|---:|---:|---:|",
    ])
    hive_bw = methods["hive_ecia"]["bandwidth"]
    for strategy, reference in payload["strategy_bandwidth"].items():
        label = (
            "Full-HR" if strategy == "full_hr"
            else f"Periodic-HR N={strategy.split('_')[-1]}"
        )
        reduction = 1.0 - hive_bw["total_bytes"] / reference["total_bytes"]
        lines.append(
            f"| {label} | {reference['total_mb']:.2f} | "
            f"{hive_bw['total_mb']:.2f} | {reduction:.2%} |"
        )
    lines.extend([
        "", "## Efficiency breakdown", "",
        "| Method | Tracker | Detector ms/frame | Tracker ms/frame | HR retrieval ms/frame | JPEG ms/frame | Pipeline ms/frame | FPS ↑ |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in methods.values():
        timing = row["timing"]
        detector = timing.get("detector_ms_per_frame")
        tracker = timing.get("tracker_ms_per_frame")
        hr = timing.get("hr_retrieval_ms_per_frame", 0.0)
        jpeg = row["bandwidth"].get("jpeg_encoding_ms_per_frame", 0.0)
        latency = timing["pipeline_latency_ms_excluding_jpeg"]
        lines.append(
            f"| {row['display']} | {LABELS[row['tracker']]} | "
            f"{'n/a' if detector is None else f'{detector:.2f}'} | "
            f"{'n/a' if tracker is None else f'{tracker:.2f}'} | {hr:.3f} | "
            f"{jpeg:.2f} | {latency:.2f} | "
            f"{timing['pipeline_fps_excluding_jpeg']:.2f} |"
        )
    lines.extend([
        "", "## Visualizations", "",
        "![Bandwidth comparison](plots/bandwidth_comparison.png)", "",
        "![Accuracy versus bandwidth](plots/accuracy_vs_bandwidth.png)", "",
        "## Per-sequence results", "",
    ])
    for sequence in payload["sequences"]:
        lines.extend([
            f"### {sequence}", "",
            "| Method | Tracker | IDSW ↓ | IDF1 ↑ | HOTA ↑ | AssA ↑ | MOTA ↑ | Total MB ↓ | Mbps ↓ | HR requests | HR frames/crops |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in methods.values():
            metric = row["metrics"]["sequences"][sequence]
            bw = row["bandwidth_sequences"][sequence]
            lines.append(
                f"| {row['display']} | {LABELS[row['tracker']]} | "
                f"{metric['id_switches']} | {metric['idf1']:.4f} | "
                f"{metric['hota']:.4f} | {metric['assa']:.4f} | "
                f"{metric['mota']:.4f} | {bw['total_mb']:.2f} | "
                f"{bw['megabits_per_second']:.2f} | {bw.get('hr_requests', 0)} | "
                f"{bw['hr_frames'] + bw['hr_crops']} |"
            )
        lines.append("")
    lines.extend([
        "The machine-readable `metrics.json` preserves IDTP, IDFP, IDFN, byte counts and all timing fields.", "",
        "## Scientific interpretation boundary", "",
        "- Full-HR and Periodic-HR are system-level acquisition baselines, not matched-budget ablations of HIVE crop selection.",
        "- Full/Periodic HR detection uses real original panoramas with the configured HR detector `imgsz`; LR frames use the established physical LR size and detector `imgsz`.",
        "- All tracker states remain in original panorama/GT coordinates and are not reset at LR↔HR transitions.",
        "- HIVE–ECIA uses the frozen Existing Base-Guided Safety checkpoint and thresholds. GT is used only after prediction for evaluation.",
        "- Repeated HIVE requests are encoded and counted again; cache hits are disabled.",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    protocol = config["protocol"]
    dataset = config[args.dataset_key]
    sequences = list(map(str, dataset["sequences"]))
    gt = {
        sequence: load_mot(
            resolve_sequence_dir(args.dataset_root.resolve(), sequence) / "gt/gt.txt",
            ground_truth=True,
        )
        for sequence in sequences
    }
    strategy_bandwidth = {}
    strategy_per_sequence = {}
    methods = {}
    strategy_names = ["full_hr", *[f"periodic_{int(n)}" for n in protocol["periodic_intervals"]]]
    for strategy in strategy_names:
        aggregate_bw, sequence_bw = load_bandwidth(
            args.acquisition_run.resolve(), strategy, sequences, float(protocol["fps"])
        )
        strategy_bandwidth[strategy] = aggregate_bw
        strategy_per_sequence[strategy] = sequence_bw
        for tracker in TRACKERS:
            key = f"{strategy}_{tracker}"
            metrics = metrics_for(
                args.acquisition_run.resolve() / "mot" / strategy / tracker,
                sequences, gt, float(protocol["evaluation_iou"]),
            )
            methods[key] = {
                "display": (
                    "Full-HR" if strategy == "full_hr"
                    else f"Periodic-HR N={strategy.split('_')[-1]}"
                ),
                "strategy": strategy,
                "tracker": tracker,
                "metrics": metrics,
                "bandwidth": aggregate_bw,
                "bandwidth_sequences": sequence_bw,
                "timing": load_timing(args.acquisition_run.resolve(), strategy, tracker, sequences),
            }

    hive_sequence_bw = {
        sequence: account_hive_sequence(
            sequence=sequence, dataset_root=args.dataset_root.resolve(),
            hive_run=args.hive_run.resolve(), lr_size=tuple(map(int, dataset["lr_size"])),
            quality=int(protocol["jpeg_quality"]), fps=float(protocol["fps"]),
        )
        for sequence in sequences
    }
    hive_bw = aggregate_hive(hive_sequence_bw, float(protocol["fps"]))
    hive_metrics = metrics_for(
        args.hive_run.resolve() / "mot", sequences, gt, float(protocol["evaluation_iou"])
    )
    differences = expected_difference(hive_metrics["aggregate"], dataset["expected_hive"])
    if differences:
        raise RuntimeError(f"Frozen HIVE–ECIA reference mismatch: {differences}")
    methods["hive_ecia"] = {
        "display": "HIVE–ECIA (Ours)", "strategy": "hive_ecia", "tracker": "botsort",
        "metrics": hive_metrics, "bandwidth": hive_bw,
        "bandwidth_sequences": hive_sequence_bw,
        "timing": load_hive_timing(args.hive_run.resolve(), sequences),
    }
    full_bytes = strategy_bandwidth["full_hr"]["total_bytes"]
    periodic_savings = {
        strategy: 1.0 - hive_bw["total_bytes"] / row["total_bytes"]
        for strategy, row in strategy_bandwidth.items()
        if strategy.startswith("periodic_")
    }
    payload = {
        "dataset_key": args.dataset_key,
        "dataset_label": dataset["label"],
        "sequences": sequences,
        "jpeg_quality": int(protocol["jpeg_quality"]),
        "protocol": protocol,
        "strategy_bandwidth": strategy_bandwidth,
        "strategy_bandwidth_sequences": strategy_per_sequence,
        "hive_reference_passed": True,
        "hive_savings_vs_full_hr": 1.0 - hive_bw["total_bytes"] / full_bytes,
        "hive_savings_vs_periodic": periodic_savings,
        "methods": methods,
        "gt_used_only_for_offline_evaluation": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "metrics.json", payload)
    plot_results(methods, args.output_dir / "plots", dataset["label"])
    report(payload, args.output_dir / "REPORT.md")
    print(f"Bandwidth dataset report: {(args.output_dir / 'REPORT.md').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

