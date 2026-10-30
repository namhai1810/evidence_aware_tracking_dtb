"""Frozen ReID encoder adapters shared by ECIA training and inference."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from hive_hr_reid import run_experiment as baseline


VALID_ENCODERS = ("onnx", "osnet_x1_0", "osnet_ain_x1_0")
CODE_ROOT = Path(__file__).resolve().parents[1]


class ReIDEncoder(Protocol):
    backend: str

    def encode(
        self, crops: list[np.ndarray], batch_size: int = 128
    ) -> list[np.ndarray | None]: ...


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def encoder_fingerprint(kind: str, checkpoint: Path) -> str:
    """Separate caches by architecture, weights, and preprocessing contract."""
    return hashlib.sha256(
        f"{kind}|{checkpoint_sha256(checkpoint)}|preprocess-v1".encode("utf-8")
    ).hexdigest()


def resolve_checkpoint(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = CODE_ROOT / path
    return path.resolve()


def resolve_reid_settings(
    args,
    config: dict,
    *,
    default_checkpoint: Path = baseline.DEFAULT_REID,
) -> tuple[str, Path]:
    configured = config.get("reid", {})
    kind = getattr(args, "reid_encoder", None) or configured.get("encoder", "onnx")
    checkpoint_value = (
        getattr(args, "reid_model", None)
        or configured.get("checkpoint")
        or default_checkpoint
    )
    if kind not in VALID_ENCODERS:
        raise ValueError(f"Unknown ReID encoder {kind!r}; choose from {VALID_ENCODERS}")
    checkpoint = resolve_checkpoint(checkpoint_value)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing frozen ReID checkpoint: {checkpoint}")
    return kind, checkpoint


def _matching_state_dict(
    model: torch.nn.Module, checkpoint: Path
) -> tuple[dict[str, torch.Tensor], int, int]:
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


class TorchreidEncoder:
    """Official Torchreid OSNet/OSNet-AIN inference adapter."""

    def __init__(self, architecture: str, checkpoint: Path, device: torch.device) -> None:
        try:
            from torchreid.reid.models import build_model
        except ImportError:
            from torchreid.models import build_model

        model = build_model(
            architecture,
            num_classes=1000,
            pretrained=False,
            use_gpu=device.type == "cuda",
        )
        state, loaded_count, skipped_count = _matching_state_dict(model, checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)
        backbone_keys = [key for key in model.state_dict() if not key.startswith("classifier")]
        backbone_loaded = sum(key in state for key in backbone_keys)
        if backbone_loaded != len(backbone_keys):
            raise RuntimeError(
                f"Incomplete {architecture} backbone: "
                f"{backbone_loaded}/{len(backbone_keys)} keys"
            )
        self.model = model.eval().to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
        self.backend = f"torchreid_{architecture}_{device.type}"
        self.embedding_dim = 512
        self.load_audit = {
            "loaded_keys": loaded_count,
            "skipped_keys": skipped_count,
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "backbone_keys_loaded": f"{backbone_loaded}/{len(backbone_keys)}",
        }

    def _preprocess(self, crop: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().div_(255.0)
        return (tensor - self.mean) / self.std

    @torch.inference_mode()
    def encode(
        self, crops: list[np.ndarray], batch_size: int = 128
    ) -> list[np.ndarray | None]:
        valid = [crop is not None and bool(crop.size) for crop in crops]
        valid_crops = [crop for crop, keep in zip(crops, valid) if keep]
        encoded: list[np.ndarray] = []
        for start in range(0, len(valid_crops), batch_size):
            group = torch.stack(
                [self._preprocess(crop) for crop in valid_crops[start : start + batch_size]]
            ).to(self.device, non_blocking=True)
            features = self.model(group)
            if isinstance(features, (tuple, list)):
                features = features[0]
            features = F.normalize(features.float().flatten(1), dim=1)
            encoded.extend(features.cpu().numpy())
        iterator = iter(encoded)
        return [next(iterator) if keep else None for keep in valid]


def build_reid_encoder(
    kind: str, checkpoint: Path, device: torch.device
) -> ReIDEncoder:
    if kind == "onnx":
        return baseline.OpenCVONNXReID(checkpoint, device)
    return TorchreidEncoder(kind, checkpoint, device)
