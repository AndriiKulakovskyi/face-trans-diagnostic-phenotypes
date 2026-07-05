"""Canonical repository paths — the single place every module resolves locations from.

Replaces the per-file ``REPO = Path(__file__).resolve().parents[N]`` constants (N drifted 3/4
across the old engines, a bug magnet) and the scattered ``results/face/<engine>_oop`` /
``docs/figures/<engine>_oop`` string literals. Milestone artifacts live under a clean,
milestone-keyed tree; sensitivity arms under ``results/analyses/`` and ``docs/figures/sensitivity/``.
"""
from __future__ import annotations

import os
from pathlib import Path

# src/face/config/paths.py -> parents[0]=config, [1]=face, [2]=src, [3]=repo root
REPO = Path(__file__).resolve().parents[3]

DATA = REPO / "data"                       # raw cohort CSVs + dictionary live at the data/ root
DATA_PROCESSED = Path(os.environ.get("FACE_DATA_DIR", str(DATA / "processed")))
CONFIGS = REPO / "configs"
RESULTS = REPO / "results"
REPORTS = REPO / "reports"
DOCS = REPO / "docs"
DOCS_FIGURES = DOCS / "figures"

# milestone key -> clean directory stem (no `_oop`, no stage/factor markers)
MILESTONES = {
    "m1": "m1_measurement",
    "m2": "m2_strata",
    "m3": "m3_temporal",
    "m4": "m4_prognosis",
    "m5": "m5_treatment",
}


def results(milestone: str) -> Path:
    """``results/<m1_measurement|m2_strata|...>`` for a milestone key or its full stem."""
    return RESULTS / MILESTONES.get(milestone, milestone)


def figures(milestone: str) -> Path:
    """``docs/figures/<milestone stem>``."""
    return DOCS_FIGURES / MILESTONES.get(milestone, milestone)


def analysis_results(name: str) -> Path:
    """``results/analyses/<name>`` — sensitivity/exploration arms, physically separated from the core tree."""
    return RESULTS / "analyses" / name


def analysis_figures(name: str) -> Path:
    """``docs/figures/sensitivity/<name>``."""
    return DOCS_FIGURES / "sensitivity" / name
