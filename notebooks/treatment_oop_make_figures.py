#!/usr/bin/env python
"""Render the M5 treatment-OOP figures from the cached stages (no recompute).

    PYTHONPATH=$PWD/src python notebooks/treatment_oop_make_figures.py

Reads ``results/face/treatment_oop/<stage>/`` and writes ``docs/figures/treatment_oop/``.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src" / "face" / "treatment").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not locate FACE repository root from {start}")


REPO = _find_repo_root(Path(__file__).resolve())
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from face.treatment.treatment_model_oop import TreatmentConfig, TreatmentRunner, TreatmentVisualizer  # noqa: E402


def main() -> None:
    config = TreatmentConfig()
    state = TreatmentRunner(config).load_state()
    made = {}
    if "moderation" in state:
        made["moderation"] = str(TreatmentVisualizer(config).moderation_bars(state["moderation"]))
    print(made)
    print("[treatment-oop] figures done.")


if __name__ == "__main__":
    main()
