"""Compare two retrained ReID+ECIA systems on QuadTrack and strict DTB test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D


ENCODERS = {
    "yolo26n": "YOLO26n-ReID (current)",
    "osnet_market": "OSNet Market",
    "osnet_ain_dmc": "OSNet-AIN D+M+C",
}
ABLATIONS = {"base_only": "Base-only", "full": "Full + HR"}
RESOLUTIONS = ("512x120", "512x256", "640x320")
ENCODER_COLORS = {
    "tracker": "#64748B",
    "yolo26n": "#0F766E",
    "osnet_market": "#2563EB",
    "osnet_ain_dmc": "#7C3AED",
}
SHORT_ENCODERS = {
    "yolo26n": "YOLO26n",
    "osnet_market": "OSNet Market",
    "osnet_ain_dmc": "OSNet-AIN",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--refresh-plots-only",
        action="store_true",
        help="Overwrite only the four generated figures, preserving JSON and Markdown.",
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def same_dict(rows: list[dict], name: str) -> dict:
    first = rows[0]
    if any(row != first for row in rows[1:]):
        raise RuntimeError(f"Unpaired {name} across systems")
    return first


def system_label(encoder: str, ablation: str) -> str:
    return f"{ENCODERS[encoder]} + {ABLATIONS[ablation]}"


def short_system_label(system: dict) -> str:
    suffix = "Base" if system["ablation"] == "base_only" else "Full + HR"
    return f"{SHORT_ENCODERS[system['encoder']]} · {suffix}"


def apply_plot_style() -> None:
    """Use a restrained, paper-friendly style shared by every report figure."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#334155",
            "axes.titlecolor": "#0F172A",
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "font.size": 10,
            "grid.color": "#E2E8F0",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def collect(root: Path) -> dict:
    systems = {}
    code_root = Path(__file__).resolve().parents[1]

    # Frozen current-system reference. QuadTrack artifacts predate this run;
    # DTB is re-evaluated here with the corrected physical-resolution crop size.
    reference_audit = load(code_root / "outputs" / "hive_ecia_cache" / "cache_audit.json")
    reference_config = yaml.safe_load((code_root / "hive_ecia" / "config.yaml").read_text())
    for ablation in ABLATIONS:
        base = code_root / "outputs" / "hive_ecia_experiment" / ablation
        history = load(base / "training" / "history.json")
        dtb = {
            resolution: load(
                root / "reference_yolo26n" / "dtb" / resolution / ablation / "metrics.json"
            )
            for resolution in RESOLUTIONS
        }
        systems[f"yolo26n:{ablation}"] = {
            "encoder": "yolo26n",
            "ablation": ablation,
            "display": system_label("yolo26n", ablation),
            "quad": load(base / "test_evaluation" / "metrics.json"),
            "dtb": dtb,
            "history": history,
            "checkpoint_sha256": sha256(base / "training" / "best.pt"),
            "thresholds_sha256": sha256(base / "validation_thresholds.json"),
            "reid_encoder": "onnx",
            "reid_model_sha256": reference_audit["reid_model_sha256"],
            "reid_backend": "onnxruntime",
            "reid_load_audit": {"backbone_keys_loaded": "ONNX graph"},
            "hive_checkpoint_sha256": reference_audit["hive_checkpoint_sha256"],
            "best_epoch": min(
                history, key=lambda row: float(row["validation"]["loss"])
            )["epoch"],
            "config": reference_config,
        }

    for encoder in ("osnet_market", "osnet_ain_dmc"):
        config = yaml.safe_load((root / encoder / "config.yaml").read_text())
        cache_audit = load(root / encoder / "cache" / "cache_audit.json")
        if cache_audit.get("calibration_sequences_accessed") is not False:
            raise RuntimeError(f"Calibration leakage in {encoder} cache")
        if cache_audit.get("test_sequences_accessed") is not False:
            raise RuntimeError(f"Test leakage in {encoder} cache")
        for ablation in ABLATIONS:
            base = root / encoder / "quadtrack" / ablation
            quad = load(base / "test_evaluation" / "metrics.json")
            checkpoint = base / "training" / "best.pt"
            thresholds = base / "validation_thresholds.json"
            dtb = {}
            for resolution in RESOLUTIONS:
                payload = load(root / encoder / "dtb" / resolution / ablation / "metrics.json")
                if payload["protocol"]["evaluation_scope"] != "held_out_test":
                    raise RuntimeError(f"DTB result is not strict test: {encoder}/{resolution}/{ablation}")
                if payload["protocol"]["test_sequences"] != ["full_occ"]:
                    raise RuntimeError(f"Unexpected DTB test sequence: {encoder}/{resolution}/{ablation}")
                dtb[resolution] = payload
            history = load(base / "training" / "history.json")
            systems[f"{encoder}:{ablation}"] = {
                "encoder": encoder,
                "ablation": ablation,
                "display": system_label(encoder, ablation),
                "quad": quad,
                "dtb": dtb,
                "history": history,
                "checkpoint_sha256": sha256(checkpoint),
                "thresholds_sha256": sha256(thresholds),
                "reid_encoder": cache_audit["reid_encoder"],
                "reid_model_sha256": cache_audit["reid_model_sha256"],
                "reid_backend": cache_audit["reid_backend"],
                "reid_load_audit": cache_audit["reid_load_audit"],
                "hive_checkpoint_sha256": cache_audit["hive_checkpoint_sha256"],
                "best_epoch": min(
                    history, key=lambda row: float(row["validation"]["loss"])
                )["epoch"],
                "config": config,
            }
    values = list(systems.values())
    same_dict([row["quad"]["baseline"]["aggregate"] for row in values], "QuadTrack baseline")
    for resolution in RESOLUTIONS:
        same_dict(
            [row["dtb"][resolution]["test_baseline"]["aggregate"] for row in values],
            f"DTB {resolution} baseline",
        )
    if len({row["hive_checkpoint_sha256"] for row in values}) != 1:
        raise RuntimeError("HIVE checkpoint differs across systems")
    for encoder in ENCODERS:
        pair = [row for row in values if row["encoder"] == encoder]
        if len({row["reid_model_sha256"] for row in pair}) != 1:
            raise RuntimeError(f"ReID checkpoint differs within {encoder}")
        if pair[0]["quad"]["protocol"]["hr_requests"] != pair[1]["quad"]["protocol"]["hr_requests"]:
            raise RuntimeError(f"QuadTrack HR budget differs within {encoder}")
        for resolution in RESOLUTIONS:
            budgets = [row["dtb"][resolution]["correction"]["hr_requests"] for row in pair]
            if len(set(budgets)) != 1:
                raise RuntimeError(f"DTB HR budget differs: {encoder}/{resolution}")
    if len({row["reid_model_sha256"] for row in values}) != 3:
        raise RuntimeError("Expected exactly three distinct ReID checkpoints")
    return systems


def metric_row(system: dict, dataset: str, resolution: str | None = None) -> tuple[dict, dict, dict]:
    if dataset == "quad":
        payload = system["quad"]
        return payload["baseline"]["aggregate"], payload["ecia"]["aggregate"], payload["correction"]
    assert resolution is not None
    payload = system["dtb"][resolution]
    key = f"test_ecia_{system['ablation']}"
    return payload["test_baseline"]["aggregate"], payload[key]["aggregate"], payload["correction"]


def plot_metrics(output: Path, systems: list[dict], dataset: str, resolution: str | None, filename: str, title: str) -> None:
    baseline, _, _ = metric_row(systems[0], dataset, resolution)
    labels = ["Tracker baseline"] + [short_system_label(row) for row in systems]
    rows = [baseline] + [metric_row(row, dataset, resolution)[1] for row in systems]
    colors = [ENCODER_COLORS["tracker"]] + [ENCODER_COLORS[row["encoder"]] for row in systems]
    markers = ["D"] + ["o" for _ in systems]
    filled = [True] + [row["ablation"] == "full" for row in systems]
    y = np.arange(len(rows))
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), sharey=True, constrained_layout=True)
    for axis, (metric, heading, fmt) in zip(
        axes.flat,
        (("id_switches", "IDSW ↓", "{:.0f}"), ("idf1", "IDF1 ↑", "{:.4f}"), ("hota", "HOTA ↑", "{:.4f}"), ("assa", "AssA ↑", "{:.4f}")),
    ):
        values = [float(row[metric]) for row in rows]
        baseline_value = values[0]
        axis.axvline(baseline_value, color=ENCODER_COLORS["tracker"], linestyle="--", linewidth=1.2, alpha=0.65)
        for index, (value, color, marker, is_filled) in enumerate(zip(values, colors, markers, filled)):
            if index:
                axis.plot(
                    [baseline_value, value],
                    [index, index],
                    color=color,
                    linewidth=2.2,
                    alpha=0.32,
                    solid_capstyle="round",
                )
            axis.scatter(
                value,
                index,
                s=78 if index else 64,
                marker=marker,
                facecolor=color if is_filled else "white",
                edgecolor=color,
                linewidth=2.0 if not is_filled else 0.8,
                zorder=3,
            )
        span = max(values) - min(values)
        padding = max(span * 0.16, 0.008 if metric != "id_switches" else 7.0)
        axis.set_xlim(min(values) - padding * 0.35, max(values) + padding)
        for index, value in enumerate(values):
            axis.annotate(
                fmt.format(value),
                (value, index),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8.5,
                color="#334155",
            )
        axis.set_yticks(y, labels, fontsize=9)
        axis.set_title(heading, loc="left")
        axis.grid(axis="x")
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    axes[0, 0].invert_yaxis()
    figure.suptitle(title, y=1.055, fontsize=16, fontweight="bold", color="#0F172A")
    figure.text(
        0.5,
        1.015,
        "Dots show absolute scores; segments are changes from the tracker baseline. Hollow = Base-only, filled = Full + HR.",
        ha="center",
        color="#64748B",
        fontsize=9.5,
    )
    figure.savefig(output / filename, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_dtb_resolutions(output: Path, systems: list[dict]) -> None:
    x = np.arange(len(RESOLUTIONS))
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
    baseline_values = {
        metric: [metric_row(systems[0], "dtb", resolution)[0][metric] for resolution in RESOLUTIONS]
        for metric in ("id_switches", "idf1")
    }
    series = [("Tracker", baseline_values, ENCODER_COLORS["tracker"], ":", "D", True)]
    for system in systems:
        values = {
            metric: [metric_row(system, "dtb", resolution)[1][metric] for resolution in RESOLUTIONS]
            for metric in ("id_switches", "idf1")
        }
        full = system["ablation"] == "full"
        series.append(
            (
                short_system_label(system),
                values,
                ENCODER_COLORS[system["encoder"]],
                "-" if full else "--",
                "o",
                full,
            )
        )
    for name, values, color, linestyle, marker, full in series:
        for axis, metric in zip(axes, ("id_switches", "idf1")):
            axis.plot(
                x,
                values[metric],
                label=name,
                color=color,
                linestyle=linestyle,
                linewidth=2.4 if full else 1.8,
                marker=marker,
                markersize=7,
                markerfacecolor=color if full else "white",
                markeredgecolor=color,
                markeredgewidth=1.6,
            )
    for axis, title in zip(axes, ("Strict DTB test IDSW ↓", "Strict DTB test IDF1 ↑")):
        axis.set_xticks(x, RESOLUTIONS)
        axis.set_title(title, loc="left")
        axis.grid(axis="y")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Resolution-stratified DTB generalization", fontsize=16, fontweight="bold", color="#0F172A")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=9)
    figure.savefig(output / "dtb_resolution_generalization.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_training(output: Path, systems: list[dict]) -> None:
    encoders = ("yolo26n", "osnet_market", "osnet_ain_dmc")
    figure, axes = plt.subplots(2, 3, figsize=(14, 7.7), sharex="col", constrained_layout=True)
    by_key = {(system["encoder"], system["ablation"]): system for system in systems}
    variant_styles = {
        "base_only": {"label": "Base-only", "linestyle": "--", "face": "white"},
        "full": {"label": "Full + HR", "linestyle": "-", "face": None},
    }
    for column, encoder in enumerate(encoders):
        color = ENCODER_COLORS[encoder]
        for ablation in ("base_only", "full"):
            system = by_key[(encoder, ablation)]
            history = system["history"]
            epochs = [row["epoch"] for row in history]
            style = variant_styles[ablation]
            for row_index, metric in enumerate(("loss", "accuracy")):
                axis = axes[row_index, column]
                values = [row["validation"][metric] for row in history]
                axis.plot(
                    epochs,
                    values,
                    color=color,
                    linestyle=style["linestyle"],
                    linewidth=2.2,
                    alpha=0.9,
                )
                best_index = min(range(len(history)), key=lambda index: history[index]["validation"]["loss"])
                axis.scatter(
                    epochs[best_index],
                    values[best_index],
                    s=62,
                    facecolor=color if style["face"] is None else style["face"],
                    edgecolor=color,
                    linewidth=1.8,
                    zorder=3,
                )
        axes[0, column].set_title(SHORT_ENCODERS[encoder])
        axes[1, column].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Validation loss ↓")
    axes[1, 0].set_ylabel("Validation accuracy ↑")
    for axis in axes.flat:
        axis.grid(alpha=0.85)
        axis.spines[["top", "right"]].set_visible(False)
    legend = [
        Line2D([0], [0], color="#334155", linestyle="--", linewidth=2.2, marker="o", markerfacecolor="white", label="Base-only"),
        Line2D([0], [0], color="#334155", linestyle="-", linewidth=2.2, marker="o", label="Full + HR"),
    ]
    figure.suptitle("QuadTrack model-validation diagnostics", fontsize=16, fontweight="bold", color="#0F172A")
    figure.legend(handles=legend, loc="outside lower center", ncol=2)
    figure.savefig(output / "training_curves.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def table_rows(systems: list[dict], dataset: str, resolution: str | None = None) -> list[str]:
    baseline, _, _ = metric_row(systems[0], dataset, resolution)
    rows = [
        f"| Tracker baseline | {baseline['id_switches']} | — | {baseline['idf1']:.4f} | — | {baseline['hota']:.4f} | {baseline['assa']:.4f} | 0 | 0 |"
    ]
    for system in systems:
        _, metric, correction = metric_row(system, dataset, resolution)
        rows.append(
            f"| {system['display']} | {metric['id_switches']} | {metric['id_switches'] - baseline['id_switches']:+d} | "
            f"{metric['idf1']:.4f} | {metric['idf1'] - baseline['idf1']:+.4f} | {metric['hota']:.4f} | "
            f"{metric['assa']:.4f} | {correction['actually_fixed']} | {correction['wrong_corrections']} |"
        )
    return rows


def main() -> int:
    args = parse_args()
    root = args.run_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.refresh_plots_only:
        raise FileExistsError(f"Refusing to overwrite report: {output}")
    output.mkdir(parents=True, exist_ok=True)
    apply_plot_style()
    systems_map = collect(root)
    systems = [systems_map[f"{encoder}:{ablation}"] for encoder in ENCODERS for ablation in ABLATIONS]

    plot_metrics(output, systems, "quad", None, "quadtrack_test_metrics.png", "Held-out QuadTrack test")
    plot_metrics(output, systems, "dtb", "640x320", "dtb_640x320_test_metrics.png", "Strict zero-shot DTB full_occ test @ 640×320")
    plot_dtb_resolutions(output, systems)
    plot_training(output, systems)
    if args.refresh_plots_only:
        print(f"Refreshed figures in: {output}")
        return 0

    payload = {
        "protocol": {
            "quadtrack_train": ["0015", "0018", "0020", "0021", "0022", "0023", "0024", "0026", "0027", "0029"],
            "quadtrack_model_validation": ["0019", "0028"],
            "quadtrack_threshold_calibration": ["0030"],
            "quadtrack_test": ["0016", "0017", "0025", "0031"],
            "dtb_fit_or_calibration": False,
            "dtb_test": ["full_occ"],
            "dtb_resolutions": list(RESOLUTIONS),
            "primary_dtb_resolution": "640x320",
            "oracle_switch_trigger": True,
        },
        "systems": systems,
    }
    (output / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    quad_best_idsw = min(systems, key=lambda row: metric_row(row, "quad")[1]["id_switches"])
    quad_best_idf1 = max(systems, key=lambda row: metric_row(row, "quad")[1]["idf1"])
    dtb_best_idsw = min(
        systems, key=lambda row: metric_row(row, "dtb", "640x320")[1]["id_switches"]
    )
    dtb_best_idf1 = max(
        systems, key=lambda row: metric_row(row, "dtb", "640x320")[1]["idf1"]
    )

    report = [
        "# ReID-retrained ECIA on QuadTrack and zero-shot DTB",
        "",
        "HIVE v2 and each ReID encoder are frozen. The two OSNet ECIA Base-only and Full variants are retrained from scratch on the same encoder-specific QuadTrack train/model-validation embedding cache. The existing YOLO26n-ReID ECIA checkpoints are included as the frozen current-system reference. Thresholds are calibrated only on QuadTrack sequence `0030`; QuadTrack test and DTB labels are not used for fitting or threshold selection.",
        "",
        "## Held-out QuadTrack test",
        "",
        "| Method | IDSW ↓ | ΔIDSW | IDF1 ↑ | ΔIDF1 | HOTA ↑ | AssA ↑ | Fixed | Wrong |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *table_rows(systems, "quad"),
        "",
        "![QuadTrack test](quadtrack_test_metrics.png)",
        "",
        "## Strict zero-shot DTB test: full_occ @ 640×320",
        "",
        "The DTB detector/tracker event pool is frozen and no DTB sequence is used for optimization or calibration. `640×320` is the primary operating point because it has the highest detector recall among the three completed physical-resolution runs; it also uses a distorted 2:1 direct resize and must not be described as an aspect-preserving panorama setting.",
        "",
        "| Method | IDSW ↓ | ΔIDSW | IDF1 ↑ | ΔIDF1 | HOTA ↑ | AssA ↑ | Fixed | Wrong |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *table_rows(systems, "dtb", "640x320"),
        "",
        "![DTB primary test](dtb_640x320_test_metrics.png)",
        "",
        "## DTB resolution robustness",
        "",
    ]
    for resolution in RESOLUTIONS:
        report.extend(
            [
                f"### {resolution}",
                "",
                "| Method | IDSW ↓ | ΔIDSW | IDF1 ↑ | ΔIDF1 | HOTA ↑ | AssA ↑ | Fixed | Wrong |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                *table_rows(systems, "dtb", resolution),
                "",
            ]
        )
    report.extend(
        [
            "![DTB resolution robustness](dtb_resolution_generalization.png)",
            "",
            "## Training diagnostics",
            "",
            "| System | Best epoch | ReID backbone load | ECIA checkpoint SHA256 |",
            "|---|---:|---:|---|",
            *[
                f"| {row['display']} | {row['best_epoch']} | {row['reid_load_audit']['backbone_keys_loaded']} | `{row['checkpoint_sha256'][:16]}…` |"
                for row in systems
            ],
            "",
            "![Training curves](training_curves.png)",
            "",
            "## Main findings",
            "",
            f"- Lowest QuadTrack IDSW: **{quad_best_idsw['display']}** with {metric_row(quad_best_idsw, 'quad')[1]['id_switches']} switches.",
            f"- Highest QuadTrack IDF1: **{quad_best_idf1['display']}** with {metric_row(quad_best_idf1, 'quad')[1]['idf1']:.4f}.",
            f"- Lowest strict DTB 640×320 IDSW: **{dtb_best_idsw['display']}** with {metric_row(dtb_best_idsw, 'dtb', '640x320')[1]['id_switches']} switches.",
            f"- Highest strict DTB 640×320 IDF1: **{dtb_best_idf1['display']}** with {metric_row(dtb_best_idf1, 'dtb', '640x320')[1]['idf1']:.4f}.",
            "- A lower IDSW count is not sufficient on its own: aggressive merges can reduce switches while lowering IDF1. IDSW, IDF1, HOTA, AssA, fixed events, and wrong corrections must be interpreted together.",
            "",
            "## Interpretation constraints",
            "",
            "- Base-only and Full are separately trained checkpoints. Their difference is a complete-system ablation, not a same-weight causal proof of HR utility.",
            "- Every recovery experiment uses an oracle GT-confirmed switch trigger. Online switch detection is not evaluated.",
            "- DTB is cross-dataset zero-shot. Results may be used to assess generalization, but not to retune these checkpoints while preserving a held-out claim.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Final report: {output / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
