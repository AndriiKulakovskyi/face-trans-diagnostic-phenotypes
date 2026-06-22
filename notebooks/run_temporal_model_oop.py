#!/usr/bin/env python
"""Run the OOP temporal-coherence engine (M3 reworked on the Gaussian-copula objects) from a terminal.

Mirrors ``run_prognosis_model_oop.py``: walks the plan (invariance -> panel -> attrition -> trait_state ->
persistence -> consolidate) on the copula M1/M2 objects, caching each stage under
``results/face/temporal_oop/<stage>/``. Wraps the proven temporal kernels; never touches native M3.

    PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_temporal_model_oop.py --mode smoke --stop-after panel
    PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_temporal_model_oop.py --mode full

Modes: smoke = tiny MCMC draws (wiring check); full = the reported plan (run detached — heavy).
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
        if (candidate / "src" / "face" / "temporal").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not locate FACE repository root from {start}")


REPO = _find_repo_root(Path(__file__).resolve())
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

loaded = sys.modules.get("face")
loaded_file = getattr(loaded, "__file__", None) if loaded is not None else None
if loaded is not None and (loaded_file is None or SRC not in Path(loaded_file).resolve().parents):
    for name in [n for n in list(sys.modules) if n == "face" or n.startswith("face.")]:
        del sys.modules[name]

from face.temporal.temporal_model_oop import TemporalConfig, TemporalRunner, TemporalVisualizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--stop-after", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> TemporalConfig:
    config = TemporalConfig()
    if args.mode == "smoke":
        config = config.with_smoke_defaults()
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    return config


def main() -> None:
    args = parse_args()
    config = build_config(args)
    runner = TemporalRunner(config)
    print(f"[temporal-oop] mode={args.mode} output={config.output_dir}", flush=True)
    t0 = time.time()
    state = runner.run_plan(stop_after=args.stop_after, overwrite=args.overwrite)
    dt = time.time() - t0

    summary: dict = {"elapsed_min": round(dt / 60, 1)}
    if "invariance" in state:
        summary["g1_license"] = state["invariance"]["license"].to_dict("records")
    if "panel" in state:
        summary["panel_rows"] = int(len(state["panel"]))
    if "trait_state" in state:
        summary["g3_trait_state"] = state["trait_state"][["axis", "icc", "verdict"]].to_dict("records")
    if "persistence" in state:
        summary["g4"] = {"spine_corner": state["persistence"].get("spine_corner"),
                         "synthesis": state["persistence"].get("g3_g4_synthesis")}
    if not args.no_plots and "trait_state" in state:
        summary["figure"] = str(TemporalVisualizer(config).trait_state(state["trait_state"]))
    print(json.dumps(summary, indent=2, default=str))
    print("\n[temporal-oop] done.")


if __name__ == "__main__":
    main()
