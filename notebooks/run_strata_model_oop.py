#!/usr/bin/env python
"""Run the OOP soft-region stratification engine (M2 reworked on the Gaussian-copula map) from a terminal.

Mirrors ``run_measurement_model_oop.py``: it walks the deterministic strata plan
(coordinates -> structure -> regions -> archetypes -> usefulness -> consolidate) on the certified
cohort-weighted copula map, caching each stage under ``results/face/strata_oop/<stage>/``.

    PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_strata_model_oop.py --mode smoke
    PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_strata_model_oop.py --mode full --stop-after structure
    PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_strata_model_oop.py --mode full --overwrite

Modes:
* smoke: tiny sweeps / few draws — a wiring check, not science.
* full:  the reported plan (data-driven operational K + archetype selection).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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

# drop any stale `face` import from a different checkout
loaded = sys.modules.get("face")
loaded_file = getattr(loaded, "__file__", None) if loaded is not None else None
if loaded is not None and (loaded_file is None or SRC not in Path(loaded_file).resolve().parents):
    for name in [n for n in list(sys.modules) if n == "face" or n.startswith("face.")]:
        del sys.modules[name]

from face.strata.engine import StrataConfig, StrataRunner, StrataVisualizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--stop-after", default=None,
                   help="halt after this stage (e.g. 'structure' for the continuum discussion gate)")
    p.add_argument("--full-si", action="store_true", help="use the full per-patient covariance S_i [N,9,9].")
    p.add_argument("--n-perm", type=int, default=30, help="permutations for the coverage-artefact test.")
    p.add_argument("--overwrite", action="store_true", help="recompute stages even when a cache exists.")
    p.add_argument("--no-plots", action="store_true", help="skip figures.")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--figure-dir", type=Path, default=None)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> StrataConfig:
    config = StrataConfig()
    if args.mode == "smoke":
        config = config.with_smoke_defaults()
    if args.full_si:
        config = config.with_full_si()
        config = replace(config, output_dir=config.output_dir / "full_si",
                         figure_dir=config.figure_dir / "full_si")
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    if args.figure_dir is not None:
        config = replace(config, figure_dir=args.figure_dir)
    return config


def make_figures(config: StrataConfig, state: dict) -> dict:
    viz = StrataVisualizer(config)
    out = {}
    if "structure" in state:
        out["structure"] = str(viz.structure_panel(state["structure"]))
    if "region_A" in state:
        out["region_profiles"] = str(viz.region_profiles(state["region_A"]))
        out["boundary_map"] = str(viz.boundary_map(state["coords"], state["region_A"]))
        out["confidence_bars"] = str(viz.confidence_bars(state["region_A"]))
        out["embedding"] = str(viz.embedding(state["coords"], state["region_A"]))
    if "arch_A" in state:
        out["archetype_profiles"] = str(viz.archetype_profiles(state["arch_A"]))
    return out


def main() -> None:
    args = parse_args()
    config = build_config(args)
    runner = StrataRunner(config)
    print(f"[strata-oop] mode={args.mode} output={config.output_dir}", flush=True)

    t0 = time.time()
    state = runner.run_plan(stop_after=args.stop_after, overwrite=args.overwrite, n_perm=args.n_perm)
    dt = time.time() - t0

    summary: dict = {"elapsed_min": round(dt / 60, 1)}
    if "structure" in state:
        summary["structure_verdict_A"] = state["structure"]["verdict_A"]["label"]
    if "choose_K" in state:
        summary["chosen_K"] = state["choose_K"]["chosen_K"]
        summary["region_names"] = state["region_A"].names
    if "arch_A" in state:
        summary["archetypes_A"] = state["arch_A"].A
        summary["archetype_stability"] = round(state["arch_A"].stability["min_tucker_congruence"], 3)
    if "usefulness" in state:
        summary["usefulness"] = state["usefulness"]["summary"]
    if "patient_strata" in state:
        summary["patient_strata_rows"] = len(state["patient_strata"])
    if "k_family_menu" in state:
        m = state["k_family_menu"]
        summary["k_family"] = {"Ks": [int(k) for k in m["K"]],
                               "operative_K": "deferred to M4/M5 incremental validity"}

    if not args.no_plots and "region_A" in state:
        summary["figures"] = make_figures(config, state)

    print(json.dumps(summary, indent=2, default=str))
    print("\n[strata-oop] done.")


if __name__ == "__main__":
    main()
