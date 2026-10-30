"""Train only Recovery-Aware HR Gain + Gate on frozen OSNet-AIN caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import yaml

from .dataset import SAMPLE_KINDS, assert_sample_coverage, audit_named_splits, collate_ecia, load_embedding_cache
from .recovery_aware import (
    ARCHITECTURE_NAME,
    RecoveryAwareECIA,
    total_parameter_count,
    trainable_parameter_count,
)
from .recovery_aware_loss import RecoveryAwareLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def apply_kind_weights(batch: dict, configured: dict[str, float]) -> None:
    lookup = torch.tensor(
        [float(configured.get(name, 1.0)) for name in SAMPLE_KINDS],
        device=batch["sample_kind"].device,
    )
    batch["sample_weight"] = lookup[batch["sample_kind"]]


def run_epoch(
    model: RecoveryAwareECIA,
    loader: DataLoader,
    criterion: RecoveryAwareLoss,
    device: torch.device,
    kind_weights: dict[str, float],
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
    description: str,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = defaultdict(float)
    examples = 0
    iterator = tqdm(loader, desc=description, unit="batch", leave=False, dynamic_ncols=True)
    for raw_batch in iterator:
        batch = move_batch(raw_batch, device)
        apply_kind_weights(batch, kind_weights)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(
                old=batch["old"],
                target=batch["target"],
                hr=batch["hr"],
                competitors=batch["competitors"],
                competitor_mask=batch["competitor_mask"],
                hr_available=batch["hr_available"],
                ablation="full",
            )
            losses = criterion(
                output,
                batch["target_class"],
                batch["merge_target"],
                batch["sample_kind"],
                batch["sample_weight"],
            )
            if training:
                losses.total.backward()
                nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    gradient_clip_norm,
                )
                optimizer.step()
        batch_size = int(batch["target_class"].shape[0])
        examples += batch_size
        for key, value in losses.detached().items():
            totals[key] += value * batch_size
        predictions = output.decision_logits.argmax(dim=-1)
        base_predictions = output.base_decision_logits.argmax(dim=-1)
        totals["accuracy"] += float((predictions == batch["target_class"]).sum())
        totals["base_accuracy"] += float((base_predictions == batch["target_class"]).sum())
        totals["mean_gate"] += float(output.gate.sum().detach())
        positive = batch["sample_kind"] == 0
        if positive.any():
            totals["positive_gate_sum"] += float(output.gate[positive].sum().detach())
            totals["positive_count"] += int(positive.sum())
        harmful = ~positive
        if harmful.any():
            totals["harmful_gate_sum"] += float(output.gate[harmful].sum().detach())
            totals["harmful_count"] += int(harmful.sum())
        iterator.set_postfix(
            loss=f"{totals['loss'] / examples:.4f}",
            acc=f"{totals['accuracy'] / examples:.3f}",
            gate=f"{totals['mean_gate'] / examples:.3f}",
            refresh=False,
        )
    result = {key: value / max(1, examples) for key, value in totals.items()}
    result["positive_gate"] = totals["positive_gate_sum"] / max(1.0, totals["positive_count"])
    result["harmful_gate"] = totals["harmful_gate_sum"] / max(1.0, totals["harmful_count"])
    return result


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    data = config["data"]
    audit_named_splits(
        train=data["train_sequences"],
        validation=data["validation_sequences"],
        calibration=data["calibration_sequences"],
        test=data["test_sequences"],
    )
    train_dataset = load_embedding_cache(Path(data["train_cache"]))
    validation_dataset = load_embedding_cache(Path(data["validation_cache"]))
    assert_sample_coverage(train_dataset.samples, data["required_sample_kinds"])
    if {sample.sequence for sample in train_dataset.samples} - set(map(str, data["train_sequences"])):
        raise ValueError("Train cache contains a sequence outside the train split")
    if {sample.sequence for sample in validation_dataset.samples} - set(map(str, data["validation_sequences"])):
        raise ValueError("Validation cache contains a sequence outside model-validation")

    training = config["training"]
    device = resolve_device(str(args.device or training["device"]))
    model_config = config["model"]
    recovery = config["recovery_aware"]
    model = RecoveryAwareECIA(
        embedding_dim=int(model_config["embedding_dim"]),
        projection_dim=int(model_config["projection_dim"]),
        relation_hidden_dim=int(model_config["relation_hidden_dim"]),
        event_hidden_dim=int(recovery["event_hidden_dim"]),
        parameter_limit=int(model_config["parameter_limit"]),
        hr_distribution_temperature=float(recovery["hr_distribution_temperature"]),
    )
    base_checkpoint = Path(recovery["base_checkpoint"])
    transfer_audit = model.load_frozen_base(base_checkpoint)
    model = model.to(device)
    frozen_names = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if any(not name.startswith(("hr_evidence_gain", "evidence_gate")) for name in trainable_names):
        raise RuntimeError(f"Unexpected trainable parameter outside HR branches: {trainable_names}")
    if not all(any(name.startswith(prefix) for name in frozen_names) for prefix in model.FROZEN_PREFIXES):
        raise RuntimeError("One or more Base branches were not frozen")

    criterion = RecoveryAwareLoss(**config["loss"]).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    loader_kwargs = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training["num_workers"]),
        "collate_fn": collate_ecia,
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite training output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best_loss = float("inf")
    stale = 0
    start = time.perf_counter()
    maximum_epochs = int(training["epochs"])
    for epoch in tqdm(range(1, maximum_epochs + 1), desc="Recovery-Aware ECIA", unit="epoch"):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(
            model, train_loader, criterion, device, data["sample_weights"], optimizer,
            float(training["gradient_clip_norm"]), f"train {epoch}/{maximum_epochs}",
        )
        validation_metrics = run_epoch(
            model, validation_loader, criterion, device, data["sample_weights"], None,
            float(training["gradient_clip_norm"]), f"val {epoch}/{maximum_epochs}",
        )
        improved = validation_metrics["loss"] < best_loss
        if improved:
            best_loss = validation_metrics["loss"]
            stale = 0
            torch.save(
                {
                    "architecture": ARCHITECTURE_NAME,
                    "model": model.state_dict(),
                    "config": config,
                    "ablation": "full",
                    "epoch": epoch,
                    "validation": validation_metrics,
                    "trainable_parameters": trainable_parameter_count(model),
                    "total_parameters": total_parameter_count(model),
                    "base_checkpoint_sha256": sha256(base_checkpoint),
                    "base_transfer_audit": transfer_audit,
                },
                output / "best.pt",
            )
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "epoch_seconds": time.perf_counter() - epoch_start,
            "elapsed_seconds": time.perf_counter() - start,
            "improved": improved,
            "best_validation_loss": best_loss,
            "stale_epochs": stale,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(row)
        print(
            f"epoch={epoch} train={train_metrics['loss']:.5f} val={validation_metrics['loss']:.5f} "
            f"gate_pos={validation_metrics['positive_gate']:.3f} "
            f"gate_harm={validation_metrics['harmful_gate']:.3f} stale={stale}",
            flush=True,
        )
        if stale >= int(training["patience"]):
            break
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    metadata = {
        "architecture": ARCHITECTURE_NAME,
        "device": str(device),
        "trainable_parameters": trainable_parameter_count(model),
        "total_parameters": total_parameter_count(model),
        "frozen_parameter_names": frozen_names,
        "trainable_parameter_names": trainable_names,
        "base_checkpoint": str(base_checkpoint.resolve()),
        "base_checkpoint_sha256": sha256(base_checkpoint),
        "best_validation_loss": best_loss,
        "calibration_data_accessed": False,
        "test_data_accessed": False,
        "dtb_data_accessed": False,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Best checkpoint: {output / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
