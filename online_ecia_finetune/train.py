"""Fine-tune Recovery-Aware ECIA on GT-labeled causal online candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
import yaml

from hive_ecia.dataset import (
    SAMPLE_KINDS, assert_sample_coverage, audit_named_splits, collate_ecia,
    load_embedding_cache,
)
from hive_ecia.recovery_aware import RecoveryAwareECIA, total_parameter_count, trainable_parameter_count
from .loss import OnlineAwareLoss
from .training_utils import (
    ControlledIdentitySampler,
    checkpoint_score,
    confusion_update,
    metrics_from_confusion,
)


ARCHITECTURE = "online_aware_recovery_ecia_v1"
BALANCED_ARCHITECTURE = "online_aware_recovery_ecia_balanced_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def build_model(config: dict, checkpoint: dict) -> RecoveryAwareECIA:
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
    model.load_state_dict(checkpoint["model"], strict=True)
    trainable_prefixes = tuple(config["fine_tuning"]["trainable_prefixes"])
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(trainable_prefixes))
    return model


def run_epoch(model, loader, criterion, device, kind_weights=None, optimizer=None) -> dict:
    training = optimizer is not None
    model.train(training)
    totals = defaultdict(float)
    examples = 0
    confusion = torch.zeros((3, 3), dtype=torch.long)
    base_confusion = torch.zeros((3, 3), dtype=torch.long)
    for raw in tqdm(loader, leave=False, unit="batch", desc="train" if training else "val"):
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in raw.items()}
        weights = None
        if kind_weights is not None:
            lookup = torch.tensor(
                [float(kind_weights.get(name, 1.0)) for name in SAMPLE_KINDS], device=device
            )
            weights = lookup[batch["sample_kind"]]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(
                old=batch["old"], target=batch["target"], hr=batch["hr"],
                competitors=batch["competitors"], competitor_mask=batch["competitor_mask"],
                hr_available=batch["hr_available"],
            )
            losses = criterion(output, batch["target_class"], batch["merge_target"], weights)
            if training:
                losses.total.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 5.0
                )
                optimizer.step()
        count = int(batch["target_class"].shape[0])
        examples += count
        for key, value in losses.detached().items():
            totals[key] += value * count
        totals["accuracy"] += int((output.decision_logits.argmax(1) == batch["target_class"]).sum())
        totals["base_accuracy"] += int((output.base_decision_logits.argmax(1) == batch["target_class"]).sum())
        totals["wrong_old"] += int(((output.decision_logits.argmax(1) == 0) & ~batch["merge_target"].bool()).sum())
        totals["gate"] += float(output.gate.sum().detach())
        abstain_index = int(output.decision_logits.shape[1] - 1)
        confusion_update(
            confusion, batch["target_class"], output.decision_logits.argmax(1), abstain_index
        )
        confusion_update(
            base_confusion,
            batch["target_class"],
            output.base_decision_logits.argmax(1),
            abstain_index,
        )
    result = {key: value / max(1, examples) for key, value in totals.items()}
    result["decision_metrics"] = metrics_from_confusion(confusion)
    result["base_decision_metrics"] = metrics_from_confusion(base_confusion)
    return result


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = config["data"]
    audit_named_splits(
        train=data["train_sequences"], validation=data["validation_sequences"],
        calibration=data["calibration_sequences"], test=data["test_sequences"],
    )
    required = [Path(data["train_cache"]), Path(data["validation_cache"]), Path(config["fine_tuning"]["initial_checkpoint"])]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing fine-tuning inputs: {missing}")
    if args.validate_only:
        print("Fine-tuning CLI, split audit and dependencies valid; training not started.")
        return 0

    seed = int(config["experiment"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train_set = load_embedding_cache(required[0])
    val_set = load_embedding_cache(required[1])
    if {s.sequence for s in train_set.samples} != set(map(str, data["train_sequences"])):
        raise ValueError("Train cache sequence set does not match declared split")
    if {s.sequence for s in val_set.samples} != set(map(str, data["validation_sequences"])):
        raise ValueError("Validation cache sequence set does not match declared split")
    assert_sample_coverage(train_set.samples, data["required_sample_kinds"])

    device = torch.device(args.device or config["training"]["device"])
    initial = torch.load(required[2], map_location="cpu", weights_only=False)
    model = build_model(config, initial).to(device)
    criterion = OnlineAwareLoss(**config["loss"]).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    sampling = config.get("sampling")
    if sampling and sampling.get("strategy") == "controlled_identity":
        sampler = ControlledIdentitySampler(
            train_set.samples,
            fractions=sampling["fractions"],
            epoch_size=int(sampling.get("epoch_size") or len(train_set)),
            old_hard_negative_fraction=float(
                sampling["old_hard_negative_fraction"]
            ),
            seed=seed,
        )
        kind_weights = None
        sampling_audit = sampler.audit()
    else:
        # Backward-compatible v1 path. Balanced-v2 never enters this branch.
        kind_weights = config["data"]["sample_weights"]
        sampler_weights = [
            float(kind_weights.get(sample.sample_kind, 1.0))
            for sample in train_set.samples
        ]
        sampler = WeightedRandomSampler(
            sampler_weights, len(sampler_weights), replacement=True
        )
        sampling_audit = {
            "strategy": "legacy_weighted_sampler_plus_weighted_loss",
            "warning": "double weighting retained only for v1 reproducibility",
        }
    common = dict(
        batch_size=int(config["training"]["batch_size"]), num_workers=int(config["training"]["num_workers"]),
        collate_fn=collate_ecia,
    )
    train_loader = DataLoader(train_set, sampler=sampler, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)

    balanced_selection = config.get("validation_selection")
    architecture = BALANCED_ARCHITECTURE if balanced_selection else ARCHITECTURE
    best = -float("inf") if balanced_selection else float("inf")
    best_epoch = None
    stale = 0; history = []; started = time.perf_counter()
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        if isinstance(sampler, ControlledIdentitySampler):
            sampler.set_epoch(epoch - 1)
        train_metrics = run_epoch(model, train_loader, criterion, device, kind_weights, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device, kind_weights)
        if balanced_selection:
            eligible, selection_score, rejection_reasons = checkpoint_score(
                val_metrics["decision_metrics"], balanced_selection
            )
            improved = eligible and selection_score > best
        else:
            eligible = True
            selection_score = -float(val_metrics["total"])
            rejection_reasons = []
            improved = val_metrics["total"] < best
        if improved:
            best = selection_score if balanced_selection else val_metrics["total"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "architecture": architecture, "model": model.state_dict(), "config": config,
                    "ablation": "full", "epoch": epoch, "validation": val_metrics,
                    "trainable_parameters": trainable_parameter_count(model),
                    "total_parameters": total_parameter_count(model),
                    "initial_checkpoint": str(required[2].resolve()),
                    "initial_checkpoint_sha256": sha256(required[2]),
                    "selection_metric": (
                        "validation_macro_f1_merge_recall_constrained"
                        if balanced_selection else "validation_loss"
                    ),
                    "selection_score": selection_score,
                    "sampling_audit": sampling_audit,
                }, output / "best.pt"
            )
        else:
            # Do not early-stop before the first non-collapsed checkpoint has
            # appeared; otherwise a conservative initialization could abort
            # the run before Merge Recall recovers.
            if best_epoch is not None:
                stale += 1
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
            "checkpoint_eligible": eligible,
            "checkpoint_rejection_reasons": rejection_reasons,
            "selection_score": selection_score,
            "best": best if best_epoch is not None else None,
            "stale": stale,
        }
        history.append(row)
        validation_decision = val_metrics["decision_metrics"]
        print(
            f"epoch={epoch} train={train_metrics['total']:.5f} "
            f"val={val_metrics['total']:.5f} macro_f1={validation_decision['macro_f1']:.4f} "
            f"merge_p={validation_decision['merge_precision']:.4f} "
            f"merge_r={validation_decision['merge_recall']:.4f} "
            f"wrong_merge={validation_decision['wrong_merge']} "
            f"wrong_rate={validation_decision['wrong_merge_rate']:.4f} "
            f"abstain={validation_decision['abstain_prediction_fraction']:.4f} "
            f"eligible={eligible}",
            flush=True,
        )
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if stale >= int(config["training"]["patience"]):
            break
    if best_epoch is None:
        raise RuntimeError(
            "Every checkpoint failed anti-collapse validation constraints; "
            "no checkpoint was selected. Inspect history.json."
        )
    metadata = {
        "architecture": architecture, "trainable_parameters": trainable_parameter_count(model),
        "trainable_prefixes": config["fine_tuning"]["trainable_prefixes"],
        "train_sequences": data["train_sequences"], "validation_sequences": data["validation_sequences"],
        "calibration_data_accessed": False, "test_data_accessed": False, "dtb_data_accessed": False,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_selection": (
            "validation_macro_f1_merge_recall_constrained"
            if balanced_selection else "validation_loss"
        ),
        "best_epoch": best_epoch,
        "sampling_audit": sampling_audit,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Best checkpoint: {output / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
