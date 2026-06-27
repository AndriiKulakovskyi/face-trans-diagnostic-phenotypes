#!/usr/bin/env python
"""Render the M4 prognosis-OOP figures from the cached stages (no recompute).

    PYTHONPATH=$PWD/src python notebooks/prognosis_oop_make_figures.py

Reads ``results/face/prognosis_oop/<stage>/`` and writes ``docs/figures/prognosis_oop/``.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src" / "face" / "prognosis").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not locate FACE repository root from {start}")


REPO = _find_repo_root(Path(__file__).resolve())
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from face.prognosis.prognosis_model_oop import (  # noqa: E402
    PrognosisConfig,
    PrognosisRunner,
    PrognosisVisualizer,
)


def main() -> None:
    config = PrognosisConfig()
    state = PrognosisRunner(config).load_state()
    viz = PrognosisVisualizer(config)
    made = {}
    if "incremental" in state:
        made["incremental_added_value"] = str(viz.incremental_bars(state["incremental"]))
    if "operative_k" in state:
        made["operative_k"] = state["operative_k"]
    print(made)
    print("[prognosis-oop] figures done.")


if __name__ == "__main__":
    main()
