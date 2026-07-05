#!/usr/bin/env python
"""Render the M3 temporal-OOP figures from the cached stages (no recompute).

    PYTHONPATH=$PWD/src python notebooks/temporal_oop_make_figures.py

Reads ``results/m3_temporal/<stage>/`` and writes ``docs/figures/temporal_oop/``.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src" / "face" / "temporal").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not locate FACE repository root from {start}")


REPO = _find_repo_root(Path(__file__).resolve())
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from face.temporal.engine import (  # noqa: E402
    TemporalConfig,
    TemporalRunner,
    TemporalVisualizer,
)


def main() -> None:
    config = TemporalConfig()
    state = TemporalRunner(config).load_state()
    viz = TemporalVisualizer(config)
    made = {}
    if "trait_state" in state:
        made["trait_state_icc"] = str(viz.trait_state(state["trait_state"]))
    if "persistence" in state:
        made["g3_g4_synthesis"] = state["persistence"].get("g3_g4_synthesis")
    print(made)
    print("[temporal-oop] figures done.")


if __name__ == "__main__":
    main()
