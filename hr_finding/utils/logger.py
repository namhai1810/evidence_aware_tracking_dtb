"""File/console logging, TensorBoard, and metric history persistence."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def create_logger(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"hive.{run_dir.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


class MetricHistory:
    def __init__(self, run_dir: Path) -> None:
        self.csv_path = run_dir / "metrics.csv"
        self.json_path = run_dir / "metrics.json"
        self.rows: list[dict] = (
            json.loads(self.json_path.read_text(encoding="utf-8"))
            if self.json_path.is_file()
            else []
        )

    def append(self, row: dict) -> None:
        self.rows.append(row)
        fields = []
        for item in self.rows:
            for key in item:
                if key not in fields:
                    fields.append(key)
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)
        self.json_path.write_text(json.dumps(self.rows, indent=2) + "\n", encoding="utf-8")


def create_summary_writer(run_dir: Path) -> SummaryWriter:
    return SummaryWriter(log_dir=str(run_dir / "tensorboard"))
