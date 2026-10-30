"""Materialize isolated ECIA configs for frozen ReID encoder retraining."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ENCODERS = {
    "osnet_market": {
        "encoder": "osnet_x1_0",
        "checkpoint": "models/reid_benchmark/osnet_x1_0_market.pth",
        "display": "OSNet x1.0 (Market1501)",
    },
    "osnet_ain_dmc": {
        "encoder": "osnet_ain_x1_0",
        "checkpoint": "models/reid_benchmark/osnet_ain_x1_0_d_m_c.pth",
        "display": "OSNet-AIN x1.0 (D+M+C)",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for key, encoder in ENCODERS.items():
        encoder_root = root / key
        encoder_root.mkdir(parents=True, exist_ok=True)
        config = yaml.safe_load(yaml.safe_dump(base))
        config["experiment"]["name"] = f"hive_v2_ecia_{key}"
        config["reid"] = dict(encoder)
        config["data"]["train_cache"] = str(encoder_root / "cache" / "train.npz")
        config["data"]["validation_cache"] = str(
            encoder_root / "cache" / "validation.npz"
        )
        config["inference"]["embedding_cache_dir"] = str(
            encoder_root / "embedding_cache"
        )
        config_path = encoder_root / "config.yaml"
        if config_path.exists():
            raise FileExistsError(f"Refusing to overwrite generated config: {config_path}")
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        print(f"Generated {key}: {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
