"""Expose only missing project-local packages after the active environment.

The historical tracking reference was produced in the active
``hainn_stitching`` Conda environment.  That environment may not contain the
pure-Python ``torchreid`` package, while the project-local venv does.  Appending
the venv site-packages (rather than prepending it) preserves Conda's
Ultralytics/Torch/OpenCV versions and uses the venv only as a final fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[2]
VENV_SITE = (
    CODE_ROOT
    / ".venv"
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)
if (VENV_SITE / "torchreid").is_dir() and str(VENV_SITE) not in sys.path:
    sys.path.append(str(VENV_SITE))

