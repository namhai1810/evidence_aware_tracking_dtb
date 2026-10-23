#!/usr/bin/env python3
"""Train HIVE on video-disjoint train/validation splits; never touches test data."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import HIVERankingDataset, collate_event_batch
from losses import HIVEMultiTaskLoss
from models import HIVE, count_trainable_parameters
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import apply_overrides, load_config, resolve_path, save_config
from utils.logger import MetricHistory, create_logger, create_summary_writer
from utils.metrics import MetricAccumulator, move_batch_to_device
from utils.seed import seed_everything, seed_worker


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "configs" / "hive.yaml")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--variant", choices=sorted(HIVE.SUPPORTED_VARIANTS), default=None)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a YAML value, for example --set training.batch_size=32",
    )
    return parser.parse_args()


def make_datasets(config: dict, dataset_root: Path, project_root: Path) -> tuple:
    data_config = dict(config["data"])
    data_config["teacher_margin_field"] = config["loss"].get("margin_teacher_field")
    data_config["include_competitor_history"] = bool(
        data_config.get("include_competitor_history", True)
        and config["model"].get("variant") == "hive_full"
    )
    validation_sequences = set(str(value) for value in data_config["validation_sequences"])
    package_split = json.loads(
        (dataset_root / "split_config.json").read_text(encoding="utf-8")
    )
    original_train = set(package_split["train_sequences"])
    original_test = set(package_split["test_sequences"])
    if not validation_sequences <= original_train:
        raise ValueError("Validation sequences must be drawn only from original train videos")
    if validation_sequences & original_test:
        raise ValueError("Test videos cannot be used for model selection")
    learning_sequences = sorted(original_train - validation_sequences)
    train_dataset = HIVERankingDataset(
        dataset_root,
        data_config["train_manifest"],
        project_root,
        data_config,
        include_sequences=learning_sequences,
    )
    validation_dataset = HIVERankingDataset(
        dataset_root,
        data_config["train_manifest"],
        project_root,
        data_config,
        include_sequences=sorted(validation_sequences),
    )
    if not train_dataset or not validation_dataset:
        raise ValueError("Train and validation datasets must both be non-empty")
    return train_dataset, validation_dataset, learning_sequences, sorted(validation_sequences)


def make_loader(
    dataset: HIVERankingDataset,
    config: dict,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    workers = int(config["data"].get("num_workers", 4))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(config["data"].get("pin_memory", True)),
        persistent_workers=workers > 0,
        collate_fn=collate_event_batch,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


def resolve_positive_weight(config: dict, dataset: HIVERankingDataset) -> float | None:
    setting = config["loss"].get("positive_weight", "auto")
    if setting is None or str(setting).lower() == "none":
        return None
    if str(setting).lower() == "auto":
        positives, negatives = dataset.class_counts()
        return negatives / positives if positives else None
    return float(setting)


def run_epoch(
    *,
    model: HIVE,
    loader: DataLoader,
    criterion: HIVEMultiTaskLoss,
    device: torch.device,
    epoch: int,
    epochs: int,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    config: dict,
    writer,
    global_step: int,
) -> tuple[dict, int]:
    training = optimizer is not None
    model.train(training)
    accumulator = MetricAccumulator(
        config["evaluation"]["classification_threshold"],
        config["evaluation"].get("no_request_threshold"),
    )
    totals = {"total": 0.0, "classification": 0.0, "ranking": 0.0, "margin": 0.0}
    processed_batches = 0
    grad_norm_sum = 0.0
    description = f"{'train' if training else 'val'} {epoch:03d}/{epochs:03d}"
    progress = tqdm(loader, desc=description, dynamic_ncols=True, leave=False)
    log_interval = int(config["training"].get("log_interval", 10))
    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"
    grad_clip = float(config["training"].get("grad_clip_norm", 0.0))

    for batch_index, batch in enumerate(progress, start=1):
        batch = move_batch_to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(batch)
                losses = criterion(logits, batch)
            if training:
                assert scaler is not None
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    grad_norm = float(clip_grad_norm_(model.parameters(), grad_clip))
                else:
                    norms = [
                        parameter.grad.detach().norm(2)
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    ]
                    grad_norm = float(torch.stack(norms).norm(2)) if norms else 0.0
                scaler.step(optimizer)
                scaler.update()
                grad_norm_sum += grad_norm

        processed_batches += 1
        for key in totals:
            totals[key] += float(losses[key].detach())
        accumulator.update(logits, batch["labels"], batch["candidate_mask"], batch["event_id"])
        learning_rate = optimizer.param_groups[0]["lr"] if optimizer else 0.0
        progress.set_postfix(
            loss=f"{totals['total'] / processed_batches:.4f}",
            cls=f"{totals['classification'] / processed_batches:.4f}",
            rank=f"{totals['ranking'] / processed_batches:.4f}",
            lr=f"{learning_rate:.2e}" if training else "-",
        )
        if training and batch_index % log_interval == 0:
            writer.add_scalar("train_batch/loss_total", float(losses["total"]), global_step)
            writer.add_scalar(
                "train_batch/loss_cls", float(losses["classification"]), global_step
            )
            writer.add_scalar("train_batch/loss_rank", float(losses["ranking"]), global_step)
            writer.add_scalar("train_batch/learning_rate", learning_rate, global_step)
        if training:
            global_step += 1

    if not processed_batches:
        raise RuntimeError(f"No usable batches in {description}")
    metrics = accumulator.compute()
    metrics.update(
        {
            "loss_total": totals["total"] / processed_batches,
            "loss_cls": totals["classification"] / processed_batches,
            "loss_rank": totals["ranking"] / processed_batches,
            "loss_margin": totals["margin"] / processed_batches,
            "grad_norm": grad_norm_sum / processed_batches if training else 0.0,
        }
    )
    return metrics, global_step


def checkpoint_state(
    *,
    model: HIVE,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    config: dict,
    epoch: int,
    global_step: int,
    best_metric: float,
    bad_epochs: int,
) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,
        "epoch": epoch,
        "global_step": global_step,
        "seed": int(config["experiment"]["seed"]),
        "best_metric": best_metric,
        "bad_epochs": bad_epochs,
    }


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    if args.variant:
        config["model"]["variant"] = args.variant
    if args.dataset_root:
        config["data"]["dataset_root"] = str(args.dataset_root)
    if args.project_root:
        config["data"]["project_root"] = str(args.project_root)

    seed = int(config["experiment"]["seed"])
    seed_everything(seed, bool(config["experiment"].get("deterministic", True)))
    dataset_root = resolve_path(config["data"]["dataset_root"], SCRIPT_DIR)
    project_root = resolve_path(config["data"]["project_root"], SCRIPT_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else resolve_path(config["output"]["root"], SCRIPT_DIR)
        / f"{config['experiment']['name']}_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(run_dir)
    writer = create_summary_writer(run_dir)
    history = MetricHistory(run_dir)
    save_config(config, run_dir / "config.yaml")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    train_dataset, val_dataset, train_sequences, val_sequences = make_datasets(
        config, dataset_root, project_root
    )
    train_loader = make_loader(train_dataset, config, True, seed)
    val_loader = make_loader(val_dataset, config, False, seed + 1)
    model = HIVE(config["model"], metadata_dim=train_dataset.metadata_dim).to(device)
    positive_weight = resolve_positive_weight(config, train_dataset)
    criterion = HIVEMultiTaskLoss(config["loss"], positive_weight).to(device)
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(config["training"].get("min_learning_rate", 1e-5)),
    )
    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    logger.info("Dataset: %s", dataset_root)
    logger.info("Train sequences (%d): %s", len(train_sequences), train_sequences)
    logger.info("Validation sequences (%d): %s", len(val_sequences), val_sequences)
    logger.info("Test manifest is not loaded by train.py")
    logger.info("Train / validation events: %d / %d", len(train_dataset), len(val_dataset))
    logger.info("Model variant: %s; trainable parameters: %d", model.variant, count_trainable_parameters(model))
    logger.info("Positive BCE weight: %s", positive_weight)

    start_epoch = 1
    global_step = 0
    mode = str(config["training"].get("monitor_mode", "max"))
    best_metric = float("-inf") if mode == "max" else float("inf")
    bad_epochs = 0
    if args.resume:
        state = load_checkpoint(args.resume, device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state.get("global_step", 0))
        best_metric = float(state.get("best_metric", best_metric))
        bad_epochs = int(state.get("bad_epochs", 0))
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    monitor = str(config["training"].get("monitor", "success_at_1"))
    patience = int(config["training"].get("early_stopping_patience", 15))
    histogram_interval = int(config["training"].get("histogram_interval", 0))
    try:
        for epoch in range(start_epoch, epochs + 1):
            epoch_start = time.perf_counter()
            train_metrics, global_step = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
                epochs=epochs,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
                writer=writer,
                global_step=global_step,
            )
            with torch.no_grad():
                val_metrics, global_step = run_epoch(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=device,
                    epoch=epoch,
                    epochs=epochs,
                    optimizer=None,
                    scaler=None,
                    config=config,
                    writer=writer,
                    global_step=global_step,
                )
            learning_rate = optimizer.param_groups[0]["lr"]
            scheduler.step()
            elapsed = time.perf_counter() - epoch_start

            for key in ("loss_total", "loss_cls", "loss_rank", "loss_margin", "accuracy"):
                writer.add_scalar(f"train/{key}", train_metrics[key], epoch)
            for key in (
                "loss_total", "loss_cls", "loss_rank", "loss_margin", "accuracy",
                "success_at_1", "success_at_3", "success_at_5", "mrr", "ndcg_at_5",
                "precision", "recall", "f1", "oracle_coverage", "oracle_gap_at_1",
                "request_rate", "request_precision",
            ):
                writer.add_scalar(f"val/{key}", val_metrics[key], epoch)
            # Preserve the exact requested TensorBoard tag spelling as aliases.
            writer.add_scalar("val/MRR", val_metrics["mrr"], epoch)
            writer.add_scalar("val/NDCG_at_5", val_metrics["ndcg_at_5"], epoch)
            writer.add_scalar("val/F1", val_metrics["f1"], epoch)
            writer.add_scalar("train/learning_rate", learning_rate, epoch)
            writer.add_scalar("train/grad_norm", train_metrics["grad_norm"], epoch)
            writer.add_scalar("epoch/time_seconds", elapsed, epoch)
            if histogram_interval > 0 and epoch % histogram_interval == 0:
                for name, parameter in model.named_parameters():
                    writer.add_histogram(f"parameters/{name}", parameter.detach().cpu(), epoch)
                    if parameter.grad is not None:
                        writer.add_histogram(f"gradients/{name}", parameter.grad.detach().cpu(), epoch)

            row = {"epoch": epoch, "time_seconds": elapsed, "learning_rate": learning_rate}
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            history.append(row)
            current = float(val_metrics[monitor])
            improved = current > best_metric if mode == "max" else current < best_metric
            if improved:
                best_metric = current
                bad_epochs = 0
            else:
                bad_epochs += 1
            state = checkpoint_state(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                bad_epochs=bad_epochs,
            )
            save_checkpoint(run_dir / "checkpoints" / "latest.pt", state)
            if improved:
                save_checkpoint(run_dir / "checkpoints" / "best.pt", state)
            logger.info(
                "epoch=%d train_loss=%.4f val_loss=%.4f val_%s=%.4f oracle=%.4f gap=%.4f time=%.1fs",
                epoch,
                train_metrics["loss_total"],
                val_metrics["loss_total"],
                monitor,
                current,
                val_metrics["oracle_coverage"],
                val_metrics["oracle_gap_at_1"],
                elapsed,
            )
            if bad_epochs >= patience:
                logger.info("Early stopping after %d epochs without improvement", bad_epochs)
                break
    finally:
        writer.close()


if __name__ == "__main__":
    main()
