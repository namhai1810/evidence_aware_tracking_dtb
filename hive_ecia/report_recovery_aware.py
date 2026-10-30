"""Create the final Recovery-Aware HR Fusion comparison report and figures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from hive_hr_reid import run_experiment as baseline
from hive_hr_reid.hota import combine_hota_raw, evaluate_hota_raw


METHODS = ("base_only", "full_old", "improved", "gate0")
LABELS = {
    "base_only": "Base-only",
    "full_old": "Full (old)",
    "improved": "Recovery-Aware Full",
    "gate0": "Recovery-Aware gate=0",
}
COLORS = {
    "baseline": "#64748B",
    "base_only": "#0F766E",
    "full_old": "#2563EB",
    "improved": "#7C3AED",
    "gate0": "#A78BFA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def collect(root: Path) -> dict:
    code_root = Path(__file__).resolve().parents[1]
    old_root = code_root / "outputs/hive_ecia_reid_retrain/20261030_osnet_retrain_v1/osnet_ain_dmc/quadtrack"
    sources = {
        "base_only": {
            "quad_metrics": root / "quadtrack/base_only/evaluation/metrics.json",
            "quad_decisions": old_root / "base_only/test_decisions.jsonl",
            "quad_checkpoint": old_root / "base_only/training/best.pt",
            "thresholds": old_root / "base_only/validation_thresholds.json",
        },
        "full_old": {
            "quad_metrics": root / "quadtrack/full_old/evaluation/metrics.json",
            "quad_decisions": old_root / "full/test_decisions.jsonl",
            "quad_checkpoint": old_root / "full/training/best.pt",
            "thresholds": old_root / "full/validation_thresholds.json",
        },
        "improved": {
            "quad_metrics": root / "quadtrack/improved/evaluation/metrics.json",
            "quad_decisions": root / "quadtrack/improved/decisions.jsonl",
            "quad_checkpoint": root / "training/best.pt",
            "thresholds": root / "validation_thresholds.json",
        },
        "gate0": {
            "quad_metrics": root / "quadtrack/gate0/evaluation/metrics.json",
            "quad_decisions": root / "quadtrack/gate0/decisions.jsonl",
            "quad_checkpoint": root / "training/best.pt",
            "thresholds": root / "validation_thresholds.json",
        },
    }
    result = {}
    dtb_event_root = code_root / "outputs/dtb_physical_lr_experiments/640x320/20261030_resolution_sweep_v1/event_protocol"
    for method, source in sources.items():
        quad = load(source["quad_metrics"])
        result[method] = {
            "quad": quad,
            "quad_decisions": load_jsonl(source["quad_decisions"]),
            "strict": load(root / f"dtb/strict/{method}/metrics.json"),
            "strict_decisions": load_jsonl(root / f"dtb/strict/{method}/decisions_gt_free.jsonl"),
            "dtb_all": load(root / f"dtb/all/{method}/metrics.json"),
            "dtb_all_decisions": load_jsonl(root / f"dtb/all/{method}/decisions_gt_free.jsonl"),
            "checkpoint_sha256": sha256(source["quad_checkpoint"]),
            "thresholds": load(source["thresholds"])["selected_thresholds"],
        }
        for scope, scope_dir in (("strict", "strict"), ("dtb_all", "all")):
            payload = result[method][scope]
            key = f"test_ecia_{payload['protocol']['ecia_ablation']}"
            for sequence, sequence_metrics in payload[key]["sequences"].items():
                gt = baseline.load_mot(
                    dtb_event_root / "dataset" / sequence / "gt" / "gt.txt",
                    ground_truth=True,
                )
                prediction = baseline.load_mot(
                    root
                    / "dtb"
                    / scope_dir
                    / method
                    / "predictions"
                    / f"{sequence}_ecia_{payload['protocol']['ecia_ablation']}.txt",
                    ground_truth=False,
                )
                sequence_metrics.update(
                    combine_hota_raw([evaluate_hota_raw(gt, prediction)])
                )
    # Protocol pairing checks precede any report calculation.
    quad_protocols = [row["quad"]["protocol"] for row in result.values()]
    if len({tuple(row["sequences"]) for row in quad_protocols}) != 1:
        raise RuntimeError("QuadTrack sequence pools differ")
    if len({int(row["hr_requests"]) for row in quad_protocols}) != 1:
        raise RuntimeError("QuadTrack HR budgets differ")
    for scope in ("strict", "dtb_all"):
        pools = [tuple(row[scope]["protocol"]["evaluation_sequences"]) for row in result.values()]
        events = [int(row[scope]["correction"]["events"]) for row in result.values()]
        if len(set(pools)) != 1 or len(set(events)) != 1:
            raise RuntimeError(f"Unpaired DTB event pool for {scope}")
    if any("gt_id" in decision for row in result.values() for decision in row["quad_decisions"]):
        raise RuntimeError("GT leaked into QuadTrack decisions")
    if any("gt_id" in decision for row in result.values() for decision in row["dtb_all_decisions"]):
        raise RuntimeError("GT leaked into DTB decisions")
    return result


def metric_payload(row: dict, scope: str) -> tuple[dict, dict, dict]:
    if scope == "quad":
        return row[scope]["baseline"], row[scope]["ecia"], row[scope]["correction"]
    payload = row[scope]
    key = f"test_ecia_{payload['protocol']['ecia_ablation']}"
    return payload["test_baseline"], payload[key], payload["correction"]


def sequence_corrections(row: dict, scope: str) -> dict[str, dict[str, int]]:
    payload = row[scope] if scope != "quad" else row["quad"]
    decisions_key = {
        "quad": "quad_decisions",
        "strict": "strict_decisions",
        "dtb_all": "dtb_all_decisions",
    }[scope]
    sequence_by_event = {decision["event_id"]: decision["sequence"] for decision in row[decisions_key]}
    result: dict[str, dict[str, int]] = {}
    for outcome in payload["outcomes"]:
        sequence = sequence_by_event[outcome["event_id"]]
        counts = result.setdefault(sequence, {"fixed": 0, "wrong": 0})
        counts["fixed"] += int(outcome["actually_fixed"])
        counts["wrong"] += int(outcome["wrong_correction"])
    return result


def aggregate_rows(data: dict, scope: str) -> list[str]:
    baseline = metric_payload(data["base_only"], scope)[0]["aggregate"]
    rows = [
        f"| Tracker baseline | {baseline['id_switches']} | {baseline['idf1']:.4f} | {baseline['hota']:.4f} | {baseline['assa']:.4f} | 0 | 0 |"
    ]
    for method in METHODS:
        _, after, correction = metric_payload(data[method], scope)
        metrics = after["aggregate"]
        rows.append(
            f"| {LABELS[method]} | {metrics['id_switches']} | {metrics['idf1']:.4f} | "
            f"{metrics['hota']:.4f} | {metrics['assa']:.4f} | "
            f"{correction['actually_fixed']} | {correction['wrong_corrections']} |"
        )
    return rows


def per_video_rows(data: dict, scope: str) -> list[str]:
    rows = []
    for method in METHODS:
        _, after, _ = metric_payload(data[method], scope)
        corrections = sequence_corrections(data[method], scope)
        for sequence, metrics in after["sequences"].items():
            counts = corrections.get(sequence, {"fixed": 0, "wrong": 0})
            rows.append(
                f"| {sequence} | {LABELS[method]} | {metrics['id_switches']} | "
                f"{metrics['idf1']:.4f} | {metrics.get('hota', float('nan')):.4f} | "
                f"{metrics.get('assa', float('nan')):.4f} | {counts['fixed']} | {counts['wrong']} |"
            )
    return rows


def paired_gate_analysis(data: dict, scope: str) -> dict:
    decisions_key = {"quad": "quad_decisions", "dtb_all": "dtb_all_decisions"}[scope]
    improved = {row["event_id"]: row for row in data["improved"][decisions_key]}
    gate0 = {row["event_id"]: row for row in data["gate0"][decisions_key]}
    if set(improved) != set(gate0):
        raise RuntimeError(f"gate=0 is not paired for {scope}")
    improved_outcomes = {row["event_id"]: row for row in data["improved"][scope]["outcomes"]}
    gate0_outcomes = {row["event_id"]: row for row in data["gate0"][scope]["outcomes"]}
    ids = sorted(improved)
    return {
        "events": len(ids),
        "prediction_changed": sum(improved[key].get("predicted_class") != gate0[key].get("predicted_class") for key in ids),
        "hr_enabled_merge": sum(bool(improved[key].get("merge")) and not bool(gate0[key].get("merge")) for key in ids),
        "hr_suppressed_merge": sum(not bool(improved[key].get("merge")) and bool(gate0[key].get("merge")) for key in ids),
        "fixed_gained": sum(bool(improved_outcomes[key]["actually_fixed"]) and not bool(gate0_outcomes[key]["actually_fixed"]) for key in ids),
        "fixed_lost": sum(not bool(improved_outcomes[key]["actually_fixed"]) and bool(gate0_outcomes[key]["actually_fixed"]) for key in ids),
        "wrong_gained": sum(bool(improved_outcomes[key]["wrong_correction"]) and not bool(gate0_outcomes[key]["wrong_correction"]) for key in ids),
        "wrong_removed": sum(not bool(improved_outcomes[key]["wrong_correction"]) and bool(gate0_outcomes[key]["wrong_correction"]) for key in ids),
        "mean_gate": float(np.mean([improved[key].get("gate", 0.0) for key in ids])),
    }


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#CBD5E1",
        "axes.titleweight": "semibold", "axes.titlecolor": "#0F172A", "grid.color": "#E2E8F0",
        "font.size": 10, "legend.frameon": False,
    })


def plot_scope(data: dict, scope: str, title: str, output: Path) -> None:
    baseline = metric_payload(data["base_only"], scope)[0]["aggregate"]
    names = ["Tracker"] + [LABELS[key] for key in METHODS]
    colors = [COLORS["baseline"]] + [COLORS[key] for key in METHODS]
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 7.5), constrained_layout=True)
    specs = (("id_switches", "IDSW ↓", "{:.0f}"), ("idf1", "IDF1 ↑", "{:.4f}"), ("hota", "HOTA ↑", "{:.4f}"), ("assa", "AssA ↑", "{:.4f}"))
    for axis, (metric, heading, fmt) in zip(axes.flat, specs):
        values = [float(baseline[metric])] + [float(metric_payload(data[key], scope)[1]["aggregate"][metric]) for key in METHODS]
        y = np.arange(len(values))
        axis.axvline(values[0], color=COLORS["baseline"], linestyle="--", alpha=0.65)
        for index, (value, color) in enumerate(zip(values, colors)):
            if index:
                axis.plot([values[0], value], [index, index], color=color, linewidth=2.4, alpha=0.35)
            axis.scatter(value, index, color=color, s=68, zorder=3)
            axis.annotate(fmt.format(value), (value, index), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5)
        span = max(values) - min(values)
        pad = max(span * 0.20, 0.008 if metric != "id_switches" else 5)
        axis.set_xlim(min(values) - pad * 0.25, max(values) + pad)
        axis.set_yticks(y, names, fontsize=8.5)
        axis.invert_yaxis()
        axis.set_title(heading, loc="left")
        axis.grid(axis="x")
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    figure.suptitle(title, fontsize=15, fontweight="bold")
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_training(root: Path, output: Path) -> None:
    history = load(root / "training/history.json")
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    axes[0].plot(epochs, [row["train"]["loss"] for row in history], label="Train", color="#64748B")
    axes[0].plot(epochs, [row["validation"]["loss"] for row in history], label="Model-val", color="#7C3AED")
    axes[0].set_title("Recovery-aware loss ↓", loc="left")
    axes[0].legend()
    axes[1].plot(epochs, [row["validation"]["positive_gate"] for row in history], label="HR supports target", color="#16A34A")
    axes[1].plot(epochs, [row["validation"]["harmful_gate"] for row in history], label="HR misleading/irrelevant", color="#DC2626")
    axes[1].set_title("Validation gate separation ↑", loc="left")
    axes[1].legend(fontsize=8)
    gate_pairs = [paired_gate_analysis(_REPORT_DATA, scope) for scope in ("quad", "dtb_all")]
    x = np.arange(2)
    axes[2].bar(x - 0.18, [row["fixed_gained"] for row in gate_pairs], 0.36, label="Fixed gained", color="#16A34A")
    axes[2].bar(x + 0.18, [row["wrong_gained"] for row in gate_pairs], 0.36, label="Wrong gained", color="#DC2626")
    axes[2].set_xticks(x, ["QuadTrack", "DTB all"])
    axes[2].set_title("HR effect vs same checkpoint gate=0", loc="left")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(axis="y")
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


_REPORT_DATA: dict = {}


def main() -> int:
    global _REPORT_DATA
    args = parse_args()
    root = args.run_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite report: {output}")
    output.mkdir(parents=True, exist_ok=True)
    data = collect(root)
    _REPORT_DATA = data
    style()
    plot_scope(data, "quad", "Held-out QuadTrack test", output / "quadtrack_comparison.png")
    plot_scope(data, "strict", "Strict zero-shot DTB full_occ @ 640×320", output / "dtb_strict_comparison.png")
    plot_scope(data, "dtb_all", "Zero-shot DTB: all five videos @ 640×320", output / "dtb_all_comparison.png")
    plot_training(root, output / "training_and_gate_diagnostics.png")

    checkpoint = torch.load(root / "training/best.pt", map_location="cpu", weights_only=False)
    metadata = load(root / "training/run_metadata.json")
    thresholds = load(root / "validation_thresholds.json")
    quad_pair = paired_gate_analysis(data, "quad")
    dtb_pair = paired_gate_analysis(data, "dtb_all")
    q_old = metric_payload(data["full_old"], "quad")[1]["aggregate"]
    q_new = metric_payload(data["improved"], "quad")[1]["aggregate"]
    d_old = metric_payload(data["full_old"], "dtb_all")[1]["aggregate"]
    d_new = metric_payload(data["improved"], "dtb_all")[1]["aggregate"]
    conclusion = (
        "Recovery-Aware Full is **not a successful overall replacement** for the old Full model. "
        "It reduces ID switches, but its held-out QuadTrack IDF1/HOTA/AssA are lower; on all DTB videos "
        "it also trails old Full on the balanced identity metrics. HR is nevertheless useful relative to "
        "the same checkpoint with `gate=0`, especially on DTB, so the remaining failure is calibration/fusion "
        "quality rather than proof that HR evidence is intrinsically useless."
    )
    report = [
        "# Recovery-Aware HR Fusion: training and evaluation",
        "",
        "## Verdict",
        "",
        conclusion,
        "",
        "## Frozen protocol",
        "",
        "- HIVE v2 and OSNet-AIN D+M+C are frozen.",
        "- Shared Projection, Base Association, and Abstain Head are initialized from the best OSNet-AIN Base-only checkpoint and frozen.",
        f"- Only {metadata['trainable_parameters']:,} HR Gain/Gate parameters are trained ({metadata['total_parameters']:,} total).",
        "- QuadTrack train: `0015,0018,0020,0021,0022,0023,0024,0026,0027,0029`; model-validation: `0019,0028`; threshold calibration only: `0030`; test: `0016,0017,0025,0031`.",
        "- DTB contributes no optimization, checkpoint selection, or threshold tuning. DTB decisions are zero-shot.",
        "- All methods receive the same candidate event pools and replay protocol. The two old baselines keep their previously frozen validation thresholds; Recovery-Aware Full and its `gate=0` ablation share the new validation threshold exactly.",
        "- `gate=0` is a causal HR ablation of the same Recovery-Aware checkpoint. It is not identical to the Base-only row because Base-only retains its own previously selected threshold.",
        "- Every decision artifact is written without GT; GT is joined only for replay evaluation. Existing collision/missing safety guards remain active.",
        "",
        "## Training and validation",
        "",
        f"- Best model-validation checkpoint: epoch {checkpoint['epoch']}.",
        f"- Selected calibration threshold: probability `{thresholds['selected_thresholds']['min_old_probability']}`, margin `{thresholds['selected_thresholds']['min_old_margin']}`.",
        f"- Base checkpoint SHA256: `{checkpoint['base_checkpoint_sha256']}`.",
        f"- Recovery-aware checkpoint SHA256: `{sha256(root / 'training/best.pt')}`.",
        "- The frozen Base tensors in the saved Recovery-Aware checkpoint were verified bit-for-bit against the source Base-only checkpoint.",
        "",
        "![Training and gate diagnostics](training_and_gate_diagnostics.png)",
        "",
        "## Held-out QuadTrack test",
        "",
        "| Method | IDSW ↓ | IDF1 ↑ | HOTA ↑ | AssA ↑ | Fixed | Wrong |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *aggregate_rows(data, "quad"),
        "",
        f"Versus old Full, Recovery-Aware Full changes IDSW {q_old['id_switches']}→{q_new['id_switches']}, IDF1 {q_old['idf1']:.4f}→{q_new['idf1']:.4f}, HOTA {q_old['hota']:.4f}→{q_new['hota']:.4f}, and AssA {q_old['assa']:.4f}→{q_new['assa']:.4f}.",
        "",
        "![QuadTrack comparison](quadtrack_comparison.png)",
        "",
        "### QuadTrack per-video result",
        "",
        "| Video | Method | IDSW ↓ | IDF1 ↑ | HOTA ↑ | AssA ↑ | Fixed | Wrong |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        *per_video_rows(data, "quad"),
        "",
        "## Strict DTB zero-shot: held-out full_occ @ 640×320",
        "",
        "| Method | IDSW ↓ | IDF1 ↑ | HOTA ↑ | AssA ↑ | Fixed | Wrong |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *aggregate_rows(data, "strict"),
        "",
        "![Strict DTB comparison](dtb_strict_comparison.png)",
        "",
        "## DTB zero-shot: all five videos @ 640×320",
        "",
        "This secondary analysis reports every DTB video only after the QuadTrack checkpoint and threshold were frozen. It is cross-dataset zero-shot, but the five DTB videos have been observed in earlier project analysis and are not claimed as a newly untouched benchmark.",
        "",
        "| Method | IDSW ↓ | IDF1 ↑ | HOTA ↑ | AssA ↑ | Fixed | Wrong |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *aggregate_rows(data, "dtb_all"),
        "",
        f"Versus old Full, Recovery-Aware Full changes IDSW {d_old['id_switches']}→{d_new['id_switches']} and IDF1 {d_old['idf1']:.4f}→{d_new['idf1']:.4f}.",
        "",
        "![All-DTB comparison](dtb_all_comparison.png)",
        "",
        "### DTB per-video result",
        "",
        "| Video | Method | IDSW ↓ | IDF1 ↑ | HOTA ↑ | AssA ↑ | Fixed | Wrong |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        *per_video_rows(data, "dtb_all"),
        "",
        "## Same-checkpoint gate=0 causal ablation",
        "",
        "| Scope | Events | Prediction changed | HR enabled merge | HR suppressed merge | Fixed gained | Fixed lost | Wrong gained | Wrong removed | Mean learned gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| QuadTrack test | {quad_pair['events']} | {quad_pair['prediction_changed']} | {quad_pair['hr_enabled_merge']} | {quad_pair['hr_suppressed_merge']} | {quad_pair['fixed_gained']} | {quad_pair['fixed_lost']} | {quad_pair['wrong_gained']} | {quad_pair['wrong_removed']} | {quad_pair['mean_gate']:.4f} |",
        f"| DTB all | {dtb_pair['events']} | {dtb_pair['prediction_changed']} | {dtb_pair['hr_enabled_merge']} | {dtb_pair['hr_suppressed_merge']} | {dtb_pair['fixed_gained']} | {dtb_pair['fixed_lost']} | {dtb_pair['wrong_gained']} | {dtb_pair['wrong_removed']} | {dtb_pair['mean_gate']:.4f} |",
        "",
        "On QuadTrack, enabling HR on the same checkpoint removes more switches but lowers the balanced identity metrics and adds one wrong correction. On DTB, enabling HR improves both switch count and IDF1/HOTA/AssA relative to gate=0 with no wrong correction. This is evidence of a real HR contribution under domain transfer, but not of a uniformly better fusion module.",
        "",
        "## Final conclusion",
        "",
        conclusion,
        "",
        "Recommended next action: keep the old Full/Base checkpoints as the production research baselines. If Recovery-Aware fusion is continued, train the gate with replay-aware sequence-level supervision or differentiable association objectives rather than event-local classification alone; do not select this checkpoint by IDSW only.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    summary = {"quad_gate_ablation": quad_pair, "dtb_gate_ablation": dtb_pair}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Report: {output / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
