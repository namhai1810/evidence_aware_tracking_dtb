"""Read-only dependency, cache, checkpoint and reference-artifact audit."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from mot_baseline_benchmark.preflight import EXPECTED
from mot_baseline_benchmark.protocol import DATASETS, resolve_sequence_dir, sha256

from .protocol import EXPECTED_MODES, normalized_protocol


CODE_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadtrack-root", type=Path, required=True)
    parser.add_argument("--dtb-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    args = parser.parse_args()
    missing = []
    for name in ("cv2", "numpy", "scipy", "torch", "torchreid", "ultralytics", "yaml"):
        try:
            importlib.import_module(name)
        except Exception as error:  # pragma: no cover - environment specific
            missing.append(f"{name}: {error}")
    if missing:
        raise RuntimeError(f"Missing Python dependencies: {missing}")

    roots = {"quadtrack": args.quadtrack_root.resolve(), "dtb": args.dtb_root.resolve()}
    audit = {"artifacts": {}, "datasets": {}}
    for name, (relative, expected_hash) in EXPECTED.items():
        path = CODE_ROOT / relative
        if not path.is_file() or sha256(path) != expected_hash:
            raise RuntimeError(f"Frozen artifact missing or changed: {path}")
        audit["artifacts"][name] = {"path": str(path), "sha256": expected_hash}

    for key, spec in DATASETS.items():
        label = spec.name
        standard = args.reference_root.resolve() / label / "standard"
        a3 = args.reference_root.resolve() / label / "hive_ecia"
        for sequence in spec.sequences:
            sequence_dir = resolve_sequence_dir(roots[key], sequence)
            required = (
                sequence_dir / "gt" / "gt.txt",
                standard / "detections" / f"{sequence}.npz",
                standard / "mot" / "botsort" / f"{sequence}.txt",
                standard / "timing" / "botsort" / f"{sequence}.json",
                a3 / "mot" / f"{sequence}.txt",
                a3 / "raw_mot" / f"{sequence}.txt",
                a3 / "diagnostics" / f"{sequence}.jsonl",
                a3 / "timing" / f"{sequence}.json",
            )
            absent = [str(path) for path in required if not path.is_file()]
            if absent:
                raise FileNotFoundError(f"Missing frozen {key} artifacts: {absent}")
        a3_manifest = json.loads((a3 / "run_manifest.json").read_text(encoding="utf-8"))
        if a3_manifest.get("mode") != EXPECTED_MODES["a3"]:
            raise RuntimeError(f"Frozen A3 has unexpected mode: {a3_manifest.get('mode')}")
        standard_protocol = normalized_protocol("a0", standard)
        a3_protocol = normalized_protocol("a3", a3)
        if standard_protocol != a3_protocol:
            raise RuntimeError(f"Frozen A0/A3 protocol mismatch for {key}: {standard_protocol} != {a3_protocol}")
        audit["datasets"][key] = {
            "a0": str(standard), "a3": str(a3),
            "detection_caches": len(spec.sequences),
            "protocol": standard_protocol,
        }
    print(json.dumps(audit, indent=2))
    print("Ablation preflight passed; no model inference or benchmark was executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

