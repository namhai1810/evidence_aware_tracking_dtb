"""Read-only artifact, dependency and G1-reference audit."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from mot_baseline_benchmark.preflight import EXPECTED
from mot_baseline_benchmark.protocol import DATASETS, resolve_sequence_dir, sha256


CODE_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadtrack-root", type=Path, required=True)
    parser.add_argument("--dtb-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    args = parser.parse_args()

    missing = []
    for module in ("cv2", "numpy", "torch", "torchreid", "ultralytics", "yaml"):
        try:
            importlib.import_module(module)
        except Exception as error:  # pragma: no cover - runtime dependent
            missing.append(f"{module}: {error}")
    if missing:
        raise RuntimeError(f"Missing Python dependencies: {missing}")

    artifacts = {}
    for name, (relative, expected_hash) in EXPECTED.items():
        path = CODE_ROOT / relative
        if not path.is_file() or sha256(path) != expected_hash:
            raise RuntimeError(f"Frozen artifact missing or changed: {path}")
        artifacts[name] = {"path": str(path), "sha256": expected_hash}

    roots = {"quadtrack": args.quadtrack_root.resolve(), "dtb": args.dtb_root.resolve()}
    references = {}
    for key, spec in DATASETS.items():
        run = args.reference_root.resolve() / spec.name / "hive_ecia"
        manifest_path = run / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("mode") != "base_guided_safety_online":
            raise RuntimeError(f"G1 reference has unexpected mode: {manifest.get('mode')}")
        if "online_ecia_finetune" in json.dumps(manifest.get("frozen_models", {})):
            raise RuntimeError("Online-aware checkpoints are forbidden in this ablation")
        for sequence in spec.sequences:
            sequence_dir = resolve_sequence_dir(roots[key], sequence)
            required = (
                sequence_dir / "gt" / "gt.txt",
                run / "mot" / f"{sequence}.txt",
                run / "raw_mot" / f"{sequence}.txt",
                run / "diagnostics" / f"{sequence}.jsonl",
                run / "timing" / f"{sequence}.json",
            )
            absent = [str(path) for path in required if not path.is_file()]
            if absent:
                raise FileNotFoundError(f"Missing G1 reference artifacts: {absent}")
        references[key] = {"run": str(run), "sequences": list(spec.sequences)}

    print(json.dumps({"status": "passed", "artifacts": artifacts, "g1_references": references}, indent=2))
    print("Gating ablation preflight passed; no inference or benchmark was executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

