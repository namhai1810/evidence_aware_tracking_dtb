#!/usr/bin/env python3
"""Benchmark LR-to-HR identity separability for frozen ReID encoders.

For every frozen DTB ID-switch event, this script compares one LR target crop
against all HR candidates that were proposed *before* GT was joined.  GT is
used only after candidate construction to mark same-identity and competitor
pairs.  The benchmark therefore measures representation quality; it does not
change HIVE, replay tracking, or ECIA decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


CODE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CODE_ROOT.parents[1]
DEFAULT_MODELS = CODE_ROOT / "models" / "reid_benchmark"
DEFAULT_CURRENT_REID = CODE_ROOT / "yolo26n-reid.onnx"
DEFAULT_DIL_SOURCE = CODE_ROOT / "third_party" / "Degradation-Invariant-ReID-pytorch"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display: str
    family: str
    architecture: str
    checkpoint: Path
    source_url: str
    input_size: tuple[int, int]
    embedding_dim: int


MODEL_SPECS = (
    ModelSpec(
        key="yolo26n_reid",
        display="YOLO26n-ReID (current)",
        family="current lightweight ReID",
        architecture="Ultralytics YOLO26n-ReID ONNX",
        checkpoint=DEFAULT_CURRENT_REID,
        source_url="local project baseline",
        input_size=(224, 224),
        embedding_dim=512,
    ),
    ModelSpec(
        key="osnet_market",
        display="OSNet x1.0 (Market1501)",
        family="conventional ReID",
        architecture="osnet_x1_0",
        checkpoint=DEFAULT_MODELS / "osnet_x1_0_market.pth",
        source_url="https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO",
        input_size=(128, 256),
        embedding_dim=512,
    ),
    ModelSpec(
        key="osnet_ain_msmt17",
        display="OSNet-AIN x1.0 (MSMT17)",
        family="cross-domain ReID",
        architecture="osnet_ain_x1_0",
        checkpoint=DEFAULT_MODELS / "osnet_ain_x1_0_msmt17.pth",
        source_url="https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO",
        input_size=(128, 256),
        embedding_dim=512,
    ),
    ModelSpec(
        key="osnet_ain_dmc",
        display="OSNet-AIN x1.0 (D+M+C)",
        family="multi-source domain-generalized ReID",
        architecture="osnet_ain_x1_0",
        checkpoint=DEFAULT_MODELS / "osnet_ain_x1_0_d_m_c.pth",
        source_url="https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO",
        input_size=(128, 256),
        embedding_dim=512,
    ),
    ModelSpec(
        key="dil_cross_resolution",
        display="DIL (MLR-CUHK03)",
        family="cross-resolution degradation-invariant ReID",
        architecture="DIL ContentEncoder / invariant branch",
        checkpoint=DEFAULT_MODELS / "dil_mlr_cuhk03_E_con_00078480.pt",
        source_url="https://github.com/Yukun-Huang/Degradation-Invariant-ReID-pytorch",
        input_size=(128, 256),
        embedding_dim=512,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=RUN_ROOT",
        help="Frozen physical-resolution DTB run; repeat for every resolution.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pad-fraction", type=float, default=0.05)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[spec.key for spec in MODEL_SPECS],
        choices=[spec.key for spec in MODEL_SPECS],
    )
    parser.add_argument("--dil-source", type=Path, default=DEFAULT_DIL_SOURCE)
    parser.add_argument("--visual-examples", type=int, default=8)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Invalid --run {value!r}; expected LABEL=RUN_ROOT")
    label, raw_root = value.split("=", 1)
    root = Path(raw_root).resolve()
    if not label or not root.is_dir():
        raise ValueError(f"Invalid run: {value!r}")
    audit = json.loads((root / "event_protocol" / "leakage_audit.json").read_text())
    expected = "x".join(str(int(value)) for value in audit["physical_lr_size"])
    if label != expected:
        raise ValueError(f"Run label {label!r} does not match audited size {expected!r}")
    if not audit.get("passed") or not audit.get("physical_lr_detection_tracking"):
        raise RuntimeError(f"Frozen event audit failed: {root}")
    return label, root


def portable_path(value: str | Path) -> Path:
    """Resolve paths written inside either /workspace or /home/dtb/workspace."""
    path = Path(value)
    if path.exists():
        return path
    raw = str(path)
    mappings = (
        ("/workspace/", "/home/dtb/workspace/"),
        ("/home/dtb/workspace/", "/workspace/"),
    )
    for old, new in mappings:
        if raw.startswith(old):
            alternate = Path(new + raw[len(old) :])
            if alternate.exists():
                return alternate
    return path


def clipped_crop(image: np.ndarray, bbox: Iterable[float], pad_fraction: float) -> np.ndarray:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    width, height = x2 - x1, y2 - y1
    x1 -= width * pad_fraction
    x2 += width * pad_fraction
    y1 -= height * pad_fraction
    y2 += height * pad_fraction
    h, w = image.shape[:2]
    left = max(0, min(w, int(math.floor(x1))))
    top = max(0, min(h, int(math.floor(y1))))
    right = max(0, min(w, int(math.ceil(x2))))
    bottom = max(0, min(h, int(math.ceil(y2))))
    if right <= left or bottom <= top:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return image[top:bottom, left:right].copy()


class ImageCache:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.image: np.ndarray | None = None

    def read(self, path: Path) -> np.ndarray:
        path = portable_path(path)
        if path != self.path:
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(f"Could not decode image: {path}")
            self.path, self.image = path, image
        assert self.image is not None
        return self.image


def load_events(label: str, root: Path) -> list[dict]:
    rows: list[dict] = []
    for split in ("train", "val", "test"):
        event_file = root / "event_protocol" / f"{split}_data" / "events.jsonl"
        for line in event_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            positives = sum(int(candidate["label"]) == 1 for candidate in event["candidates"])
            negatives = sum(int(candidate["label"]) == 0 for candidate in event["candidates"])
            if positives and negatives:
                event["resolution"] = label
                event["event_root"] = str(root / "event_protocol")
                event["positive_count"] = positives
                event["negative_count"] = negatives
                rows.append(event)
    return rows


def collect_crops(events: list[dict], pad_fraction: float) -> tuple[list[np.ndarray], list[dict]]:
    """Return a flat crop list plus indices required to reconstruct events."""
    crops: list[np.ndarray] = []
    manifest: list[dict] = []
    image_cache = ImageCache()
    for event in tqdm(events, desc="materialize LR/HR crops", unit="event"):
        event_root = Path(event["event_root"])
        target_path = event_root / event["target_crop"]["crop_path"]
        target = cv2.imread(str(target_path))
        if target is None:
            raise FileNotFoundError(f"Could not decode LR target: {target_path}")
        target_index = len(crops)
        crops.append(target)
        candidates = []
        for candidate in event["candidates"]:
            image = image_cache.read(Path(candidate["source_image"]))
            crop = clipped_crop(image, candidate["candidate_bbox_xyxy"], pad_fraction)
            if not crop.size:
                continue
            crop_index = len(crops)
            crops.append(crop)
            candidates.append(
                {
                    "crop_index": crop_index,
                    "candidate_id": candidate["candidate_id"],
                    "candidate_frame": int(candidate["candidate_frame"]),
                    "label": int(candidate["label"]),
                    "matched_gt_id": candidate.get("matched_gt_id"),
                    "source_image": candidate["source_image"],
                    "bbox": candidate["candidate_bbox_xyxy"],
                }
            )
        if any(row["label"] == 1 for row in candidates) and any(
            row["label"] == 0 for row in candidates
        ):
            manifest.append(
                {
                    "resolution": event["resolution"],
                    "split": event["split"],
                    "sequence": event["sequence"],
                    "event_id": event["event_id"],
                    "target_frame": int(event["target_frame"]),
                    "gt_id": int(event["gt_id"]),
                    "target_crop_index": target_index,
                    "target_crop_path": str(target_path),
                    "target_source_image": event["target_crop"]["source_image"],
                    "target_bbox_original": event["target_crop"]["bbox_original_xyxy"],
                    "candidates": candidates,
                }
            )
    return crops, manifest


class Encoder:
    def encode(self, crops: list[np.ndarray], batch_size: int) -> np.ndarray:
        raise NotImplementedError


class ONNXEncoder(Encoder):
    def __init__(self, model_path: Path, device: torch.device) -> None:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
        providers = ["CPUExecutionProvider"]
        if device.type == "cuda" and "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.provider = self.session.get_providers()[0]

    def encode(self, crops: list[np.ndarray], batch_size: int) -> np.ndarray:
        output: list[np.ndarray] = []
        for start in tqdm(range(0, len(crops), batch_size), desc="ONNX embedding", unit="batch"):
            group = crops[start : start + batch_size]
            tensors = []
            for crop in group:
                resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                tensors.append(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
            features = self.session.run(
                self.output_names, {self.input_name: np.stack(tensors)}
            )[0]
            features = np.asarray(features, dtype=np.float32).reshape(len(group), -1)
            features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
            output.append(features)
        return np.concatenate(output)


class TorchEncoder(Encoder):
    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model = model.eval().to(device)
        self.device = device
        self.mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)

    def preprocess(self, crop: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().div_(255.0)
        return (tensor - self.mean) / self.std

    @torch.inference_mode()
    def encode(self, crops: list[np.ndarray], batch_size: int) -> np.ndarray:
        output: list[np.ndarray] = []
        for start in tqdm(range(0, len(crops), batch_size), desc="PyTorch embedding", unit="batch"):
            group = torch.stack([self.preprocess(crop) for crop in crops[start : start + batch_size]])
            features = self.model(group.to(self.device, non_blocking=True))
            if isinstance(features, (tuple, list)):
                features = features[0]
            features = F.normalize(features.float().flatten(1), dim=1)
            output.append(features.cpu().numpy())
        return np.concatenate(output)


def matching_state_dict(model: torch.nn.Module, checkpoint: Path) -> tuple[dict, int, int]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    current = model.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    skipped = 0
    for raw_key, value in state.items():
        key = raw_key.removeprefix("module.")
        if key in current and current[key].shape == value.shape:
            loaded[key] = value
        else:
            skipped += 1
    return loaded, len(loaded), skipped


def build_osnet(spec: ModelSpec, device: torch.device) -> tuple[Encoder, dict]:
    try:
        from torchreid.reid.models import build_model
    except ImportError:
        from torchreid.models import build_model

    model = build_model(spec.architecture, num_classes=1000, pretrained=False, use_gpu=device.type == "cuda")
    state, loaded, skipped = matching_state_dict(model, spec.checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    backbone_keys = [key for key in model.state_dict() if not key.startswith("classifier")]
    backbone_loaded = sum(key in state for key in backbone_keys)
    if backbone_loaded != len(backbone_keys):
        raise RuntimeError(
            f"Incomplete {spec.key} backbone: {backbone_loaded}/{len(backbone_keys)} keys"
        )
    return TorchEncoder(model, device), {
        "loaded_keys": loaded,
        "skipped_keys": skipped,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "backbone_keys_loaded": f"{backbone_loaded}/{len(backbone_keys)}",
    }


class DILInvariantBranch(torch.nn.Module):
    def __init__(self, content_encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = content_encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        invariant = self.encoder.identity(x)
        return self.encoder.classifier.add_block(invariant)


def build_dil(spec: ModelSpec, source: Path, device: torch.device) -> tuple[Encoder, dict]:
    source = source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(
            f"DIL source missing: {source}. Clone the official repository listed in README."
        )
    sys.path.insert(0, str(source))
    try:
        from core.network.reid.encoder import ContentEncoder
        import core.network.reid.encoder as encoder_module

        # The full checkpoint contains every ResNet parameter. Avoid an unrelated
        # ImageNet download requested by the 2022 constructor.
        original_resnet50 = encoder_module.models.resnet50
        encoder_module.models.resnet50 = lambda pretrained=True: original_resnet50(weights=None)
        try:
            model = ContentEncoder(
                {"num_class": 1367, "feat_inv_dim": 2048},
                pool_type="max",
                last_stride=2,
                classifier_type="base",
            )
        finally:
            encoder_module.models.resnet50 = original_resnet50
        state = torch.load(spec.checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(state, strict=True)
    finally:
        sys.path.pop(0)
    return TorchEncoder(DILInvariantBranch(model), device), {
        "loaded_keys": len(state),
        "skipped_keys": 0,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "feature_branch": "degradation-invariant content feature after learned 512D add_block",
    }


def build_encoder(spec: ModelSpec, args: argparse.Namespace, device: torch.device) -> tuple[Encoder, dict]:
    if not spec.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {spec.checkpoint}")
    if spec.key == "yolo26n_reid":
        encoder = ONNXEncoder(spec.checkpoint, device)
        return encoder, {"provider": encoder.provider}
    if spec.key == "dil_cross_resolution":
        return build_dil(spec, args.dil_source, device)
    return build_osnet(spec, device)


def pairwise_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    if not len(positive) or not len(negative):
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    return float((np.count_nonzero(comparisons > 0) + 0.5 * np.count_nonzero(comparisons == 0)) / comparisons.size)


def summarize(rows: list[dict]) -> dict:
    positives = np.asarray([score for row in rows for score in row["positive_scores"]])
    negatives = np.asarray([score for row in rows for score in row["negative_scores"]])
    margins = np.asarray([row["best_positive_margin"] for row in rows])
    return {
        "events": len(rows),
        "positive_pairs": int(len(positives)),
        "negative_pairs": int(len(negatives)),
        "positive_cosine_mean": float(positives.mean()),
        "positive_cosine_std": float(positives.std()),
        "negative_cosine_mean": float(negatives.mean()),
        "negative_cosine_std": float(negatives.std()),
        "cosine_gap": float(positives.mean() - negatives.mean()),
        "pairwise_auc": pairwise_auc(positives, negatives),
        "top1_accuracy": float(np.mean([row["top1_correct"] for row in rows])),
        "positive_beats_best_competitor": float(np.mean(margins > 0)),
        "best_positive_margin_mean": float(margins.mean()),
        "best_positive_margin_median": float(np.median(margins)),
        "best_positive_margin_p10": float(np.quantile(margins, 0.10)),
    }


def score_model(features: np.ndarray, manifest: list[dict]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    for event in manifest:
        target = features[event["target_crop_index"]]
        candidates = []
        for candidate in event["candidates"]:
            score = float(np.dot(target, features[candidate["crop_index"]]))
            candidates.append({**candidate, "cosine": score})
        positives = [row for row in candidates if row["label"] == 1]
        negatives = [row for row in candidates if row["label"] == 0]
        best_positive = max(positives, key=lambda row: row["cosine"])
        best_negative = max(negatives, key=lambda row: row["cosine"])
        top1 = max(candidates, key=lambda row: row["cosine"])
        rows.append(
            {
                **{key: event[key] for key in ("resolution", "split", "sequence", "event_id", "target_frame", "gt_id")},
                "target_crop_path": event["target_crop_path"],
                "target_source_image": event["target_source_image"],
                "target_bbox_original": event["target_bbox_original"],
                "positive_scores": [row["cosine"] for row in positives],
                "negative_scores": [row["cosine"] for row in negatives],
                "best_positive": best_positive,
                "best_competitor": best_negative,
                "best_positive_margin": best_positive["cosine"] - best_negative["cosine"],
                "top1_candidate_id": top1["candidate_id"],
                "top1_correct": bool(top1["label"]),
                "candidates": candidates,
            }
        )
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["resolution"]].append(row)
    metrics = {
        "overall_descriptive": summarize(rows),
        "by_resolution": {label: summarize(group) for label, group in sorted(groups.items())},
    }
    return rows, metrics


def paired_top1_transitions(results: dict[str, dict], reference_key: str) -> dict[str, dict]:
    """Count event-level wins and regressions versus the current ReID encoder."""
    reference = {
        row["event_id"] + "@" + row["resolution"]: bool(row["top1_correct"])
        for row in results[reference_key]["events"]
    }
    comparisons: dict[str, dict] = {}
    for model_key, result in results.items():
        if model_key == reference_key:
            continue
        counts = {
            "both_correct": 0,
            "fixed_current_error": 0,
            "broke_current_success": 0,
            "both_wrong": 0,
        }
        for row in result["events"]:
            key = row["event_id"] + "@" + row["resolution"]
            old, new = reference[key], bool(row["top1_correct"])
            if old and new:
                counts["both_correct"] += 1
            elif not old and new:
                counts["fixed_current_error"] += 1
            elif old and not new:
                counts["broke_current_success"] += 1
            else:
                counts["both_wrong"] += 1
        counts["net_top1_events"] = (
            counts["fixed_current_error"] - counts["broke_current_success"]
        )
        counts["events"] = len(reference)
        comparisons[model_key] = counts
    return comparisons


def plot_distributions(output: Path, results: dict[str, dict], specs: list[ModelSpec]) -> None:
    figure, axes = plt.subplots(1, len(specs), figsize=(4.2 * len(specs), 4.8), sharey=True, constrained_layout=True)
    if len(specs) == 1:
        axes = [axes]
    for axis, spec in zip(axes, specs):
        rows = results[spec.key]["events"]
        positive = [value for row in rows for value in row["positive_scores"]]
        negative = [value for row in rows for value in row["negative_scores"]]
        parts = axis.violinplot([positive, negative], [1, 2], showmeans=True, showextrema=True)
        for body, color in zip(parts["bodies"], ("#10b981", "#ef4444")):
            body.set_facecolor(color)
            body.set_alpha(0.55)
        axis.set_xticks([1, 2], ["Same ID", "Competitor"])
        axis.set_title(spec.display, fontsize=10)
        axis.grid(axis="y", alpha=0.25)
        metric = results[spec.key]["metrics"]["overall_descriptive"]
        axis.text(
            0.04,
            0.04,
            f"gap={metric['cosine_gap']:.3f}\nAUC={metric['pairwise_auc']:.3f}",
            transform=axis.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cbd5e1"},
        )
    axes[0].set_ylabel("Cosine similarity: LR target vs HR crop")
    figure.suptitle("LR↔HR identity separation on the same frozen candidate pools")
    figure.savefig(output / "cosine_same_id_vs_competitor.png", dpi=180)
    plt.close(figure)


def plot_summary(output: Path, results: dict[str, dict], specs: list[ModelSpec]) -> None:
    labels = [spec.display.replace(" (", "\n(") for spec in specs]
    x = np.arange(len(specs))
    metrics = [results[spec.key]["metrics"]["overall_descriptive"] for spec in specs]
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    values_list = (
        ([row["pairwise_auc"] for row in metrics], "Pairwise AUC ↑", "{:.3f}"),
        ([row["top1_accuracy"] for row in metrics], "Event top-1 accuracy ↑", "{:.3f}"),
        ([row["best_positive_margin_mean"] for row in metrics], "Mean best-positive margin ↑", "{:+.3f}"),
    )
    colors = ["#64748b", "#2563eb", "#8b5cf6", "#10b981", "#f59e0b"]
    for axis, (values, title, fmt) in zip(axes, values_list):
        bars = axis.bar(x, values, color=colors[: len(specs)])
        axis.set_xticks(x, labels, rotation=12, ha="right", fontsize=8)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, labels=[fmt.format(value) for value in values], padding=3, fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(0, 1.05)
    figure.suptitle("Frozen ReID encoder comparison (GT used only for scoring)")
    figure.savefig(output / "reid_encoder_summary.png", dpi=180)
    plt.close(figure)


def plot_by_resolution(output: Path, results: dict[str, dict], specs: list[ModelSpec]) -> None:
    resolutions = sorted(
        results[specs[0].key]["metrics"]["by_resolution"],
        key=lambda value: math.prod(int(part) for part in value.split("x")),
    )
    x = np.arange(len(resolutions))
    width = 0.8 / len(specs)
    figure, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for model_index, spec in enumerate(specs):
        metrics = results[spec.key]["metrics"]["by_resolution"]
        offset = (model_index - (len(specs) - 1) / 2) * width
        axes[0].bar(
            x + offset,
            [metrics[label]["pairwise_auc"] for label in resolutions],
            width,
            label=spec.display,
        )
        axes[1].bar(
            x + offset,
            [metrics[label]["top1_accuracy"] for label in resolutions],
            width,
            label=spec.display,
        )
    for axis, title in zip(axes, ("Pairwise AUC ↑", "Event top-1 accuracy ↑")):
        axis.set_xticks(x, resolutions)
        axis.set_ylim(0, 1.05)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.suptitle("Resolution-stratified LR↔HR ReID separability")
    figure.savefig(output / "reid_by_lr_resolution.png", dpi=180)
    plt.close(figure)


def fit_panel(image: np.ndarray, size: tuple[int, int] = (260, 360)) -> np.ndarray:
    width, height = size
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    if image is None or not image.size:
        return canvas
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def candidate_crop(record: dict, pad_fraction: float) -> np.ndarray:
    image = cv2.imread(str(portable_path(record["source_image"])))
    if image is None:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return clipped_crop(image, record["bbox"], pad_fraction)


def plot_qualitative(
    output: Path,
    results: dict[str, dict],
    specs: list[ModelSpec],
    count: int,
    pad_fraction: float,
) -> None:
    if len(specs) < 2 or count <= 0:
        return
    current = {row["event_id"] + "@" + row["resolution"]: row for row in results[specs[0].key]["events"]}
    cross = {row["event_id"] + "@" + row["resolution"]: row for row in results[specs[-1].key]["events"]}
    ranked = sorted(
        current,
        key=lambda key: cross[key]["best_positive_margin"] - current[key]["best_positive_margin"],
        reverse=True,
    )
    chosen = ranked[: max(1, count // 2)] + ranked[-max(1, count // 2) :]
    figure, axes = plt.subplots(len(chosen), 3, figsize=(10, 3.2 * len(chosen)), constrained_layout=True)
    if len(chosen) == 1:
        axes = np.asarray([axes])
    for row_axes, key in zip(axes, chosen):
        row = cross[key]
        images = (
            cv2.imread(row["target_crop_path"]),
            candidate_crop(row["best_positive"], pad_fraction),
            candidate_crop(row["best_competitor"], pad_fraction),
        )
        current_margin = current[key]["best_positive_margin"]
        cross_margin = row["best_positive_margin"]
        titles = (
            f"LR target\n{row['event_id']} · {row['resolution']}",
            f"HR same identity\ncos={row['best_positive']['cosine']:.3f}",
            f"HR competitor\ncos={row['best_competitor']['cosine']:.3f}",
        )
        for axis, image, title in zip(row_axes, images, titles):
            axis.imshow(cv2.cvtColor(fit_panel(image), cv2.COLOR_BGR2RGB))
            axis.set_title(title, fontsize=9)
            axis.axis("off")
        row_axes[0].set_ylabel(
            f"Δ margin cross-current\n{cross_margin - current_margin:+.3f}",
            fontsize=9,
        )
    figure.suptitle(f"Largest gains/regressions: {specs[-1].display} versus {specs[0].display}")
    figure.savefig(output / "qualitative_lr_hr_pairs.png", dpi=160)
    plt.close(figure)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def report_text(
    results: dict[str, dict],
    specs: list[ModelSpec],
    manifest: list[dict],
    transitions: dict[str, dict],
) -> str:
    resolutions = sorted(
        {row["resolution"] for row in manifest},
        key=lambda value: math.prod(int(part) for part in value.split("x")),
    )
    lines = [
        "# DTB LR↔HR ReID cosine benchmark",
        "",
        "## Question",
        "",
        "Does a frozen ReID encoder assign a higher cosine to the HR crop of the same identity than to HR competitor crops when the query is the LR target at an ID-switch event?",
        "",
        "## Protocol and leakage control",
        "",
        "- The LR target, candidate pool, and train/validation/test sequence split come from the already-frozen physical-resolution event protocol.",
        "- Candidate proposals were created before GT labels were joined. GT is used here only to mark same-identity versus competitor pairs and compute metrics; it is never input to an encoder.",
        "- No encoder is fine-tuned on DTB. No threshold is tuned, and ECIA/HIVE/replay decisions are not involved in this representation-only benchmark.",
        "- Each event contributes one LR target query. Every valid HR candidate is scored. Event top-1 is correct when the highest-cosine candidate has the target GT identity.",
        "- `overall descriptive` pools the three repeated resolution protocols and is not an independent statistical test. Resolution-stratified rows are the main comparison.",
        "",
        "## Encoders",
        "",
        "| Encoder | Category | Input | Embedding | Checkpoint SHA256 |",
        "|---|---|---:|---:|---|",
    ]
    for spec in specs:
        lines.append(
            f"| {spec.display} | {spec.family} | {spec.input_size[0]}×{spec.input_size[1]} | "
            f"{spec.embedding_dim}D | `{sha256(spec.checkpoint)[:16]}…` |"
        )
    lines.extend(
        [
            "",
            "DIL is the only evaluated checkpoint trained explicitly for degradation/cross-resolution invariance (MLR-CUHK03). OSNet-AIN improves domain robustness but is not mislabeled as a dedicated cross-resolution method.",
            "",
            "## Overall descriptive result",
            "",
            "| Encoder | Events | Pos/Neg pairs | Same-ID cosine | Competitor cosine | Gap ↑ | AUC ↑ | Top-1 ↑ | Best-pos margin ↑ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for spec in specs:
        row = results[spec.key]["metrics"]["overall_descriptive"]
        lines.append(
            f"| {spec.display} | {row['events']} | {row['positive_pairs']}/{row['negative_pairs']} | "
            f"{row['positive_cosine_mean']:.4f} | {row['negative_cosine_mean']:.4f} | "
            f"{row['cosine_gap']:+.4f} | {row['pairwise_auc']:.4f} | "
            f"{row['top1_accuracy']:.4f} | {row['best_positive_margin_mean']:+.4f} |"
        )
    lines.extend(["", "## By physical LR detector/tracker resolution", ""])
    for resolution in resolutions:
        lines.extend(
            [
                f"### {resolution}",
                "",
                "| Encoder | Events | AUC ↑ | Top-1 ↑ | Same-ID wins ↑ | Cosine gap ↑ | Mean best-pos margin ↑ |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for spec in specs:
            row = results[spec.key]["metrics"]["by_resolution"][resolution]
            lines.append(
                f"| {spec.display} | {row['events']} | {row['pairwise_auc']:.4f} | "
                f"{row['top1_accuracy']:.4f} | {row['positive_beats_best_competitor']:.4f} | "
                f"{row['cosine_gap']:+.4f} | {row['best_positive_margin_mean']:+.4f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Paired event changes versus current ReID",
            "",
            "This table uses exactly the same event/candidate pool and compares top-1 outcomes event by event. `Fixed` means the current encoder chose a competitor but the alternative chose a same-ID crop; `Broken` is the reverse.",
            "",
            "| Alternative encoder | Both correct | Fixed current error | Broke current success | Both wrong | Net top-1 events ↑ |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for spec in specs[1:]:
        row = transitions[spec.key]
        lines.append(
            f"| {spec.display} | {row['both_correct']} | {row['fixed_current_error']} | "
            f"{row['broke_current_success']} | {row['both_wrong']} | {row['net_top1_events']:+d} |"
        )
    lines.append("")
    best_auc = max(specs, key=lambda spec: results[spec.key]["metrics"]["overall_descriptive"]["pairwise_auc"])
    best_top1 = max(specs, key=lambda spec: results[spec.key]["metrics"]["overall_descriptive"]["top1_accuracy"])
    current = results[specs[0].key]["metrics"]["overall_descriptive"]
    cross_spec = next((spec for spec in specs if spec.key == "dil_cross_resolution"), None)
    lines.extend(
        [
            "## Interpretation",
            "",
            f"- Best pooled pairwise separation: **{best_auc.display}** (AUC {results[best_auc.key]['metrics']['overall_descriptive']['pairwise_auc']:.4f}).",
            f"- Best pooled event top-1: **{best_top1.display}** ({results[best_top1.key]['metrics']['overall_descriptive']['top1_accuracy']:.4f}).",
        ]
    )
    if cross_spec is not None:
        cross = results[cross_spec.key]["metrics"]["overall_descriptive"]
        lines.append(
            f"- Dedicated cross-resolution DIL versus current ReID: ΔAUC {cross['pairwise_auc'] - current['pairwise_auc']:+.4f}, "
            f"Δtop-1 {cross['top1_accuracy'] - current['top1_accuracy']:+.4f}, and Δmean margin "
            f"{cross['best_positive_margin_mean'] - current['best_positive_margin_mean']:+.4f}."
        )
    lines.extend(
        [
            "- High absolute cosine alone is not evidence of good identity recovery. The useful quantities are separation from competitors, AUC, top-1, and positive margin.",
            "- This benchmark can show whether a larger or cross-resolution embedding is more discriminative on the frozen DTB candidate pool. It does **not** by itself prove fewer tracking ID switches; that requires substituting the encoder, calibrating only on validation, and replaying the held-out tracking evaluation.",
            "",
            "## Visualizations",
            "",
            "![Same identity versus competitor cosine](cosine_same_id_vs_competitor.png)",
            "",
            "![Encoder summary](reid_encoder_summary.png)",
            "",
            "![Per-resolution results](reid_by_lr_resolution.png)",
            "",
            "![Qualitative LR-HR pairs](qualitative_lr_hr_pairs.png)",
            "",
            "## Reproducibility artifacts",
            "",
            "- `metrics.json`: aggregate and per-resolution metrics, checkpoint fingerprints, and load audits.",
            "- `events_<model>.jsonl`: per-event same-ID/competitor cosine scores and margins.",
            "- `pair_manifest.jsonl`: frozen crop/candidate audit without embeddings.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Output exists; refusing to overwrite: {output}")
    output.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    selected_specs = [spec for spec in MODEL_SPECS if spec.key in args.models]
    runs = [parse_run(value) for value in args.run]
    events = [event for label, root in runs for event in load_events(label, root)]
    crops, manifest = collect_crops(events, args.pad_fraction)
    if not manifest:
        raise RuntimeError("No event contains both same-identity and competitor HR candidates")
    write_jsonl(output / "pair_manifest.jsonl", manifest)

    results: dict[str, dict] = {}
    model_audit = {}
    for spec in selected_specs:
        print(f"\n=== {spec.display} ===", flush=True)
        encoder, load_audit = build_encoder(spec, args, device)
        features = encoder.encode(crops, args.batch_size)
        if features.shape != (len(crops), spec.embedding_dim):
            raise RuntimeError(
                f"Unexpected {spec.key} embedding shape {features.shape}; expected {(len(crops), spec.embedding_dim)}"
            )
        rows, metrics = score_model(features, manifest)
        results[spec.key] = {"events": rows, "metrics": metrics}
        model_audit[spec.key] = {
            "display": spec.display,
            "family": spec.family,
            "architecture": spec.architecture,
            "checkpoint": str(spec.checkpoint.resolve()),
            "checkpoint_sha256": sha256(spec.checkpoint),
            "source_url": spec.source_url,
            "input_size": spec.input_size,
            "embedding_dim": spec.embedding_dim,
            "load_audit": load_audit,
        }
        write_jsonl(output / f"events_{spec.key}.jsonl", rows)
        del encoder, features
        if device.type == "cuda":
            torch.cuda.empty_cache()

    transitions = paired_top1_transitions(results, selected_specs[0].key)

    payload = {
        "protocol": {
            "question": "cosine(LR target, HR same identity) versus cosine(LR target, HR competitor)",
            "gt_use": "evaluation labels only; not encoder input, candidate proposal, or decision",
            "candidate_pool_frozen_before_gt_join": True,
            "dtb_finetuning": False,
            "threshold_tuning": False,
            "hive_or_ecia_decision_involved": False,
            "runs": {label: str(root) for label, root in runs},
            "events": len(manifest),
            "crops": len(crops),
            "pad_fraction": args.pad_fraction,
        },
        "models": model_audit,
        "metrics": {key: value["metrics"] for key, value in results.items()},
        "paired_top1_vs_current": transitions,
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot_distributions(output, results, selected_specs)
    plot_summary(output, results, selected_specs)
    plot_by_resolution(output, results, selected_specs)
    plot_qualitative(output, results, selected_specs, args.visual_examples, args.pad_fraction)
    (output / "REPORT.md").write_text(
        report_text(results, selected_specs, manifest, transitions), encoding="utf-8"
    )
    print(f"\nReport: {output / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
