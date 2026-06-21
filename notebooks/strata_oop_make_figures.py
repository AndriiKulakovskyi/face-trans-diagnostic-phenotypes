#!/usr/bin/env python
"""Generate the soft-region strata figures from cached stage artifacts (never recomputes).

Run AFTER ``run_strata_model_oop.py`` has populated ``results/face/strata_oop/<stage>/``.

    PYTHONPATH=$PWD/src python notebooks/strata_oop_make_figures.py
    PYTHONPATH=$PWD/src python notebooks/strata_oop_make_figures.py --full-si

Writes into ``docs/figures/strata_oop/``: the structure-discovery panel, the soft-region centroid +
archetype profile heatmaps, the soft-transition-boundary map, the per-region confidence bars, and a
PCA embedding (visualization-only) colored by region / cohort / DSM-5.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src" / "face" / "strata").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not locate FACE repository root from {start}")


REPO = _find_repo_root(Path(__file__).resolve())
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from face.strata.strata_model_oop import StrataConfig, StrataRunner, StrataVisualizer  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full-si", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--figure-dir", type=Path, default=None)
    args = p.parse_args()

    config = StrataConfig()
    if args.full_si:
        config = replace(config.with_full_si(), output_dir=config.output_dir / "full_si",
                         figure_dir=config.figure_dir / "full_si")
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    if args.figure_dir is not None:
        config = replace(config, figure_dir=args.figure_dir)

    state = StrataRunner(config).load_state()
    if "region_A" not in state:
        raise SystemExit("no cached fits found — run `python notebooks/run_strata_model_oop.py --mode full` first")

    viz = StrataVisualizer(config)
    out = {}
    if "structure" in state:
        out["structure_panel"] = str(viz.structure_panel(state["structure"]))
    out["region_profiles"] = str(viz.region_profiles(state["region_A"]))
    out["archetype_profiles"] = str(viz.archetype_profiles(state["arch_A"]))
    out["boundary_map"] = str(viz.boundary_map(state["coords"], state["region_A"]))
    out["confidence_bars"] = str(viz.confidence_bars(state["region_A"]))
    out["embedding"] = str(viz.embedding(state["coords"], state["region_A"]))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
