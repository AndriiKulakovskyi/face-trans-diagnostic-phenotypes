"""Pytest path bootstrap for the clean src/ layout.

`src/` holds our development packages (face_common); `archive/` holds the
vendored sister engine (face_stratification, face_rlvr). Putting both on
sys.path lets pytest run without installing the project.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "archive"):
    p = str(REPO_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
