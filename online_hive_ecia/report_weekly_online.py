"""Create paper-oriented figures for the Base-guided online experiment."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


METHODS = [
    "tracker_baseline",
    "rule_based_full",
    "base_guided_hr_disabled",
    "base_guided_full",
]
METHOD_LABELS = [
    "YOLO + BoT-SORT",
    "Rule-based Full",
    "Base-guided HR-off",
    "Base-guided Full",
]
METHOD_COLORS = ["#1E293B", "#64748B", "#F59E0B", "#0F766E"]
DATASETS = ["QuadTrack", "DTB 640×320"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadtrack", type=Path, required=True)
    parser.add_argument("--dtb", type=Path, required=True)
    parser.add_argument("--quadtrack-baseline", type=Path, required=True)
    parser.add_argument("--dtb-baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#FAFAFA",
            "figure.facecolor": "white",
            "grid.color": "#E2E8F0",
            "grid.linewidth": 0.8,
        }
    )


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def label_bars(axis: plt.Axes, bars, *, decimals: int) -> None:
    for bar in bars:
        value = bar.get_height()
        text = f"{value:.{decimals}f}" if decimals else f"{int(round(value))}"
        axis.annotate(
            text,
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#334155",
        )


def tracking_figure(payloads: list[dict], output_dir: Path) -> None:
    specifications = [
        ("id_switches", "ID switches ↓", 0),
        ("idf1", "IDF1 ↑", 3),
        ("hota", "HOTA ↑", 3),
        ("assa", "AssA ↑", 3),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    x = np.arange(len(DATASETS))
    width = 0.19
    for axis, (metric, title, decimals) in zip(axes.flat, specifications):
        all_values = []
        for method_index, (method, label, color) in enumerate(
            zip(METHODS, METHOD_LABELS, METHOD_COLORS)
        ):
            values = [payload["methods"][method]["aggregate"][metric] for payload in payloads]
            all_values.extend(values)
            bars = axis.bar(
                x + (method_index - 1.5) * width,
                values,
                width,
                label=label,
                color=color,
                edgecolor="white",
                linewidth=0.8,
            )
            label_bars(axis, bars, decimals=decimals)
        axis.set_title(title, loc="left")
        axis.set_xticks(x, DATASETS)
        axis.grid(axis="y")
        axis.set_axisbelow(True)
        upper = max(all_values) * (1.20 if metric == "id_switches" else 1.16)
        axis.set_ylim(0, upper)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
    figure.suptitle("Online identity recovery: tracking quality", fontsize=16, fontweight="bold", y=1.06)
    figure.text(
        0.5,
        1.015,
        "Paired raw BoT-SORT predictions; Base-guided HR-on/off share the same proposed events",
        ha="center",
        color="#475569",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.4, w_pad=1.8)
    save_figure(figure, output_dir, "online_base_guided_tracking_02112026")


def trigger_cost_figure(payloads: list[dict], output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    group_labels = ["Quad\nRule", "Quad\nBase", "DTB\nRule", "DTB\nBase"]
    trigger_rows = []
    for payload in payloads:
        trigger_rows.extend(
            [
                payload["trigger_accuracy"]["rule_based"]["aggregate"],
                payload["trigger_accuracy"]["base_guided_hr_request"]["aggregate"],
            ]
        )
    x = np.arange(4)
    width = 0.22
    metric_specs = [("precision", "Precision", "#0F766E"), ("recall", "Recall", "#2563EB"), ("f1", "F1", "#F59E0B")]
    for offset, (metric, label, color) in enumerate(metric_specs):
        values = [row[metric] for row in trigger_rows]
        bars = axes[0].bar(x + (offset - 1) * width, values, width, label=label, color=color)
        label_bars(axes[0], bars, decimals=2)
    axes[0].set_title("Post-hoc trigger quality", loc="left")
    axes[0].set_xticks(x, group_labels)
    axes[0].set_ylim(0, 0.62)
    axes[0].grid(axis="y")
    axes[0].legend(frameon=False, ncol=3, fontsize=8, loc="upper right")

    x = np.arange(2)
    width = 0.23
    cost_methods = ("tracker_baseline", "rule_based_full", "base_guided_full")
    cost_labels = ("YOLO + BoT-SORT", "Rule", "Base-guided")
    cost_colors = (METHOD_COLORS[0], METHOD_COLORS[1], METHOD_COLORS[3])
    for offset, method in enumerate(cost_methods):
        values = [payload["methods"][method]["timing"]["hr_cost"]["requests"] for payload in payloads]
        bars = axes[1].bar(x + (offset - 1) * width, values, width, color=cost_colors[offset], label=cost_labels[offset])
        label_bars(axes[1], bars, decimals=0)
    axes[1].set_title("HR requests ↓", loc="left")
    axes[1].set_xticks(x, DATASETS)
    axes[1].set_ylim(0, 140)
    axes[1].grid(axis="y")
    axes[1].legend(frameon=False, fontsize=8)

    for offset, method in enumerate(cost_methods):
        values = [payload["methods"][method]["timing"]["hr_cost"]["bytes_read"] / 1e6 for payload in payloads]
        bars = axes[2].bar(x + (offset - 1) * width, values, width, color=cost_colors[offset], label=cost_labels[offset])
        label_bars(axes[2], bars, decimals=1)
    axes[2].set_title("Source-image bandwidth ↓", loc="left")
    axes[2].set_ylabel("MB read")
    axes[2].set_xticks(x, DATASETS)
    axes[2].set_ylim(0, 38)
    axes[2].grid(axis="y")
    axes[2].legend(frameon=False, fontsize=8)

    figure.suptitle("Trigger selectivity and HR acquisition cost", fontsize=16, fontweight="bold", y=1.03)
    figure.tight_layout(w_pad=2.0)
    save_figure(figure, output_dir, "online_base_guided_trigger_cost_02112026")


def sequence_gain_figure(payloads: list[dict], output_dir: Path) -> None:
    rows = []
    labels = []
    for prefix, payload in zip(("Q", "D"), payloads):
        for sequence, base in payload["methods"]["base_guided_hr_disabled"]["sequences"].items():
            full = payload["methods"]["base_guided_full"]["sequences"][sequence]
            rows.append(
                [
                    base["id_switches"] - full["id_switches"],
                    full["idf1"] - base["idf1"],
                    full["hota"] - base["hota"],
                    full["assa"] - base["assa"],
                ]
            )
            labels.append(f"{prefix} · {sequence}")
    values = np.asarray(rows, dtype=float)
    scaled = np.zeros_like(values)
    for column in range(values.shape[1]):
        scale = np.abs(values[:, column]).max()
        scaled[:, column] = values[:, column] / scale if scale else values[:, column]

    figure, axis = plt.subplots(figsize=(10.5, 7.2))
    image = axis.imshow(scaled, cmap="RdYlGn", norm=colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), aspect="auto")
    axis.set_xticks(range(4), ["IDSW reduction", "Δ IDF1", "Δ HOTA", "Δ AssA"])
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title("Contribution of HR: Full minus HR-disabled by sequence", loc="left", pad=16)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            text = f"{values[row, column]:+.0f}" if column == 0 else f"{values[row, column]:+.4f}"
            axis.text(column, row, text, ha="center", va="center", fontsize=9, fontweight="bold", color="#0F172A")
    axis.axhline(3.5, color="white", linewidth=4)
    axis.text(3.65, 1.5, "QuadTrack", rotation=90, va="center", color="#475569", fontsize=9)
    axis.text(3.65, 6.5, "DTB", rotation=90, va="center", color="#475569", fontsize=9)
    axis.tick_params(length=0)
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.08, label="Column-normalized gain (green = better)")
    figure.tight_layout()
    save_figure(figure, output_dir, "online_base_guided_per_sequence_02112026")


def diagnostic_rows(run_root: Path) -> list[dict]:
    rows = []
    for path in sorted((run_root / "base_guided_online" / "diagnostics").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("type") == "base_guided_event":
                rows.append(row)
    return rows


def outcome_figure(run_roots: list[Path], output_dir: Path) -> None:
    rows_by_dataset = [diagnostic_rows(path) for path in run_roots]
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 8.0))
    x = np.arange(2)

    pair_counts = [Counter(row["pair_kind"] for row in rows) for rows in rows_by_dataset]
    bottom = np.zeros(2)
    for key, label, color in (("lost_to_new", "Lost→New", "#0F766E"), ("active_to_active", "Active→Active", "#F59E0B")):
        values = np.asarray([counter[key] for counter in pair_counts])
        bars = axes[0, 0].bar(x, values, bottom=bottom, label=label, color=color)
        for bar, value, base in zip(bars, values, bottom):
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2, base + value / 2, str(int(value)), ha="center", va="center", color="white", fontweight="bold")
        bottom += values
    axes[0, 0].set_title("Proposed candidate pairs", loc="left")
    axes[0, 0].legend(frameon=False, fontsize=8)

    action_counts = [Counter(row["action"] for row in rows) for rows in rows_by_dataset]
    bottom = np.zeros(2)
    for key, label, color in (("abstain", "Abstain", "#94A3B8"), ("request_hr", "Request HR", "#2563EB"), ("direct_merge", "Direct merge", "#0F766E")):
        values = np.asarray([counter[key] for counter in action_counts])
        bars = axes[0, 1].bar(x, values, bottom=bottom, label=label, color=color)
        for bar, value, base in zip(bars, values, bottom):
            if value:
                axes[0, 1].text(bar.get_x() + bar.get_width() / 2, base + value / 2, str(int(value)), ha="center", va="center", color="white", fontweight="bold")
        bottom += values
    axes[0, 1].set_title("Base policy actions", loc="left")
    axes[0, 1].legend(frameon=False, fontsize=8)

    width = 0.34
    actionable = []
    blocked = []
    for rows in rows_by_dataset:
        requests = [row for row in rows if row.get("would_request_hr")]
        actionable.append(sum(row["pair_kind"] == "lost_to_new" for row in requests))
        blocked.append(sum(row["pair_kind"] == "active_to_active" for row in requests))
    bars1 = axes[1, 0].bar(x - width / 2, actionable, width, label="Lost→New", color="#0F766E")
    bars2 = axes[1, 0].bar(x + width / 2, blocked, width, label="Active→Active", color="#DC2626")
    label_bars(axes[1, 0], bars1, decimals=0)
    label_bars(axes[1, 0], bars2, decimals=0)
    axes[1, 0].set_title("HR requests by pair safety", loc="left")
    axes[1, 0].legend(frameon=False, fontsize=8)

    base_applied = []
    full_applied = []
    for root in run_roots:
        base_rows = []
        for path in sorted((root / "base_guided_hr_disabled" / "diagnostics").glob("*.jsonl")):
            base_rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if json.loads(line).get("type") == "base_guided_event")
        base_applied.append(sum(bool(row.get("safety", {}).get("applied")) for row in base_rows))
    for rows in rows_by_dataset:
        full_applied.append(sum(bool(row.get("safety", {}).get("applied")) for row in rows))
    bars1 = axes[1, 1].bar(x - width / 2, base_applied, width, label="HR-off", color="#F59E0B")
    bars2 = axes[1, 1].bar(x + width / 2, full_applied, width, label="Full", color="#0F766E")
    label_bars(axes[1, 1], bars1, decimals=0)
    label_bars(axes[1, 1], bars2, decimals=0)
    axes[1, 1].set_title("Safety-approved identity merges", loc="left")
    axes[1, 1].legend(frameon=False, fontsize=8)

    for axis in axes.flat:
        axis.set_xticks(x, DATASETS)
        axis.grid(axis="y")
        axis.set_axisbelow(True)
    figure.suptitle("Online event flow and the Active→Active bottleneck", fontsize=16, fontweight="bold", y=1.01)
    figure.tight_layout(h_pad=2.2, w_pad=1.8)
    save_figure(figure, output_dir, "online_base_guided_event_flow_02112026")


def main() -> int:
    args = parse_args()
    configure_style()
    payloads = [load_payload(args.quadtrack), load_payload(args.dtb)]
    baseline_payloads = [
        load_payload(args.quadtrack_baseline),
        load_payload(args.dtb_baseline),
    ]
    for payload, baseline in zip(payloads, baseline_payloads):
        # The baseline artifacts were accepted only after raw-MOT hashes were
        # checked byte-for-byte against the Base-guided runs.
        payload["methods"]["tracker_baseline"] = baseline["methods"]["baseline"]
    tracking_figure(payloads, args.output_dir)
    trigger_cost_figure(payloads, args.output_dir)
    sequence_gain_figure(payloads, args.output_dir)
    outcome_figure([args.quadtrack.parents[1], args.dtb.parents[1]], args.output_dir)
    print(f"Figures written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
