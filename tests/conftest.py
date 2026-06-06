"""Pytest path bootstrap for the ``src/`` layout.

Only ``src/`` needs to be importable to run the V3 tests without installing the
project (``import v3`` resolves from there).
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_src = str(REPO_ROOT / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
