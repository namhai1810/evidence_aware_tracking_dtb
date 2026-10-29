"""Train only the 43.7K-parameter ECIA module on cached frozen embeddings.

This entry point never loads or updates HIVE v2 or the ReID encoder.  Test data
is intentionally absent from the training CLI to enforce the split boundary.
"""

from __future__ import annotations

import argparse
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

from .dataset import (
    SAMPLE_KINDS,
    assert_sample_coverage,
    audit_named_splits,
    collate_ecia,
    load_embedding_cache,
)
from .loss import ECIALoss
from .model import ECIA, count_trainable_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ablation",
        choices=sorted(ECIA.VALID_ABLATIONS),
        default=None,
        help="Override experiment.ablation from YAML.",
    )
    parser.add_argument("--device", default=None, help="Override training.device from YAML")
    return parser.parse_args()


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
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return device


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def apply_kind_weights(batch: dict, configured: dict[str, float]) -> None:
    lookup = torch.tensor(
        [float(configured.get(name, 1.0)) for name in SAMPLE_KINDS],
        device=batch["sample_kind"].device,
    )
    batch["sample_weight"] = lookup[batch["sample_kind"]]


def forward_batch(model: ECIA, batch: dict, ablation: str):
    return model(
        old=batch["old"],
        target=batch["target"],
        hr=batch["hr"],
        competitors=batch["competitors"],
        competitor_mask=batch["competitor_mask"],
        hr_available=batch["hr_available"],
        ablation=ablation,
    )


def run_epoch(
    *,
    model: ECIA,
    loader: DataLoader,
    criterion: ECIALoss,
    device: torch.device,
    ablation: str,
    kind_weights: dict[str, float],
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
    progress_desc: str | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = defaultdict(float)
    examples = 0

    iterator = tqdm(
        loader,
        desc=progress_desc,
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    ) if progress_desc else loader
    for raw_batch in iterator:
        batch = move_batch(raw_batch, device)
        apply_kind_weights(batch, kind_weights)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            output = forward_batch(model, batch, ablation)
            losses = criterion(
                output,
                batch["target_class"],
                batch["merge_target"],
                batch["sample_weight"],
            )
            if training:
                losses.total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()

        batch_size = int(batch["target_class"].shape[0])
        examples += batch_size
        for key, value in losses.detached().items():
            totals[key] += value * batch_size
        predictions = output.decision_logits.argmax(dim=-1)
        totals["accuracy"] += float(
            (predictions == batch["target_class"]).float().sum().detach()
        )
        non_merge = ~batch["merge_target"].bool()
        totals["wrong_merge_predictions"] += float(
            ((predictions == 0) & non_merge).sum().detach()
        )
        totals["non_merge_examples"] += int(non_merge.sum())
        if progress_desc:
            iterator.set_postfix(
                loss=f"{totals['loss'] / max(1, examples):.4f}",
                acc=f"{totals['accuracy'] / max(1, examples):.3f}",
                wrong=f"{totals['wrong_merge_predictions'] / max(1.0, totals['non_merge_examples']):.3f}",
                refresh=False,
            )

    metrics = {key: value / max(1, examples) for key, value in totals.items()}
    metrics["wrong_merge_rate"] = totals["wrong_merge_predictions"] / max(
        1.0, totals["non_merge_examples"]
    )
    return metrics


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)

    # Split audit is performed before any dataset is loaded or optimizer built.
    data_config = config["data"]
    audit_named_splits(
        train=data_config["train_sequences"],
        validation=data_config["validation_sequences"],
        calibration=data_config["calibration_sequences"],
        test=data_config["test_sequences"],
    )
    train_dataset = load_embedding_cache(Path(data_config["train_cache"]))
    validation_dataset = load_embedding_cache(Path(data_config["validation_cache"]))
    assert_sample_coverage(train_dataset.samples, data_config["required_sample_kinds"])
    if {sample.sequence for sample in train_dataset.samples} - set(data_config["train_sequences"]):
        raise ValueError("Train cache contains a sequence outside configured train split")
    if {sample.sequence for sample in validation_dataset.samples} - set(
        data_config["validation_sequences"]
    ):
        raise ValueError("Validation cache contains a sequence outside validation split")

    training_config = config["training"]
    device = resolve_device(str(args.device or training_config["device"]))
    model = ECIA(**{
        key: config["model"][key]
        for key in (
            "embedding_dim",
            "projection_dim",
            "relation_hidden_dim",
            "event_hidden_dim",
            "parameter_limit",
        )
    }).to(device)
    parameter_count = count_trainable_parameters(model)
    expected = int(config["model"]["expected_trainable_parameters"])
    if parameter_count != expected:
        raise RuntimeError(f"Parameter count changed: expected {expected}, got {parameter_count}")

    loss_config = config["loss"]
    criterion = ECIALoss(**loss_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "batch_size": int(training_config["batch_size"]),
        "num_workers": int(training_config["num_workers"]),
        "collate_fn": collate_ecia,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_kwargs
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ablation = args.ablation or str(config["experiment"]["ablation"])
    history = []
    best_loss = float("inf")
    stale_epochs = 0
    maximum_epochs = int(training_config["epochs"])
    log_path = output_dir / "training_log.jsonl"
    # A new one-command experiment is deterministic and starts a fresh live
    # log. history.json is still written at the end as a convenient array.
    log_path.write_text("", encoding="utf-8")
    training_started = time.perf_counter()

    # Only ECIA parameters participate in optimization. Input embeddings are
    # cached constants, so HIVE and ReID cannot accidentally receive gradients.
    with tqdm(
        range(1, maximum_epochs + 1),
        desc=f"ECIA {ablation}",
        unit="epoch",
        dynamic_ncols=True,
    ) as epoch_progress:
        for epoch in epoch_progress:
            epoch_started = time.perf_counter()
            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                ablation=ablation,
                kind_weights=data_config["sample_weights"],
                optimizer=optimizer,
                gradient_clip_norm=float(training_config["gradient_clip_norm"]),
                progress_desc=f"{ablation} train {epoch}/{maximum_epochs}",
            )
            validation_metrics = run_epoch(
                model=model,
                loader=validation_loader,
                criterion=criterion,
                device=device,
                ablation=ablation,
                kind_weights=data_config["sample_weights"],
                optimizer=None,
                gradient_clip_norm=float(training_config["gradient_clip_norm"]),
                progress_desc=f"{ablation} val {epoch}/{maximum_epochs}",
            )

            improved = validation_metrics["loss"] < best_loss
            if improved:
                best_loss = validation_metrics["loss"]
                stale_epochs = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": config,
                        "ablation": ablation,
                        "epoch": epoch,
                        "validation": validation_metrics,
                        "trainable_parameters": parameter_count,
                    },
                    output_dir / "best.pt",
                )
            else:
                stale_epochs += 1

            elapsed = time.perf_counter() - training_started
            epoch_seconds = time.perf_counter() - epoch_started
            mean_epoch_seconds = elapsed / epoch
            # This is an upper-bound ETA to the configured maximum. Early
            # stopping can finish sooner when patience is exhausted.
            eta_seconds = mean_epoch_seconds * (maximum_epochs - epoch)
            row = {
                "epoch": epoch,
                "maximum_epochs": maximum_epochs,
                "ablation": ablation,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_seconds": epoch_seconds,
                "elapsed_seconds": elapsed,
                "eta_seconds_to_max_epochs": eta_seconds,
                "improved": improved,
                "best_validation_loss": best_loss,
                "early_stopping_stale_epochs": stale_epochs,
                "early_stopping_patience": int(training_config["patience"]),
                "train": train_metrics,
                "validation": validation_metrics,
            }
            history.append(row)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

            epoch_progress.set_postfix(
                train=f"{train_metrics['loss']:.4f}",
                val=f"{validation_metrics['loss']:.4f}",
                best=f"{best_loss:.4f}",
                stale=f"{stale_epochs}/{training_config['patience']}",
            )
            tqdm.write(
                f"[{ablation}] epoch {epoch}/{maximum_epochs} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"val_loss={validation_metrics['loss']:.6f} "
                f"best={best_loss:.6f} epoch_time={epoch_seconds:.1f}s "
                f"eta_max={eta_seconds / 60.0:.1f}m"
            )

            if stale_epochs >= int(training_config["patience"]):
                tqdm.write(
                    f"[{ablation}] early stopping at epoch {epoch}; "
                    f"validation did not improve for {stale_epochs} epochs."
                )
                break

    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "ablation": ablation,
                "device": str(device),
                "trainable_parameters": parameter_count,
                "best_validation_loss": best_loss,
                "test_data_accessed": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
