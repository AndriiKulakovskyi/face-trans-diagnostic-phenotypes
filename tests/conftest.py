"""Pytest path bootstrap for the ``src/`` layout.

``face_common`` is self-contained (the stratification engine is internalized in
``face_common.engine``), so only ``src/`` needs to be importable to run the
tests without installing the project.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_src = str(REPO_ROOT / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
