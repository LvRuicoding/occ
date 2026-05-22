"""Module-level sys.path setup for vendored OccAny dependencies.

Importing this once (and as early as possible) ensures `dust3r`, `croco`,
`sam2`, `sam3`, and the DA3 source tree resolve when this package is used
as a script (`python -m ft.kitti_stage1_5f.tools.train`) or imported from a
parent script. Mirrors `REPO_ROOT/inference.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VENDORED_PATHS = [
    _REPO_ROOT,
    _REPO_ROOT / "third_party",
    _REPO_ROOT / "third_party" / "dust3r",
    _REPO_ROOT / "third_party" / "croco" / "models" / "curope",
    _REPO_ROOT / "third_party" / "Grounded-SAM-2",
    _REPO_ROOT / "third_party" / "Grounded-SAM-2" / "grounding_dino",
    _REPO_ROOT / "third_party" / "sam3",
    _REPO_ROOT / "third_party" / "Depth-Anything-3" / "src",
]
for _p in reversed(_VENDORED_PATHS):
    _s = str(_p)
    if _p.exists() and _s not in sys.path:
        sys.path.insert(0, _s)
