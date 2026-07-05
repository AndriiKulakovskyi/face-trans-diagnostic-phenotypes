#!/usr/bin/env python
"""Run the OOP treatment-moderation engine (M5 reworked on the Gaussian-copula objects) from a terminal.

Mirrors the other OOP drivers: walks the plan (exposures -> frame -> propensity -> moderation -> confounder ->
tolerability -> consolidate) on the copula objects, caching each stage under
``results/m5_treatment/<stage>/``. Wraps the proven M5 + M4 kernels; native M5 untouched.

    PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_treatment_model_oop.py --mode smoke --stop-after propensity
    PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_treatment_model_oop.py --mode full

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
        if (candidate / "src" / "face" / "treatment").exists() and (candidate / "pyproject.toml").exists():
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

from face.treatment.engine import (  # noqa: E402
    TreatmentConfig,
    TreatmentRunner,
    TreatmentVisualizer,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--stop-after", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> TreatmentConfig:
    config = TreatmentConfig()
    if args.mode == "smoke":
        config = config.with_smoke_defaults()
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    return config


def main() -> None:
    args = parse_args()
    config = build_config(args)
    runner = TreatmentRunner(config)
    print(f"[treatment-oop] mode={args.mode} output={config.output_dir}", flush=True)
    t0 = time.time()
    state = runner.run_plan(stop_after=args.stop_after, overwrite=args.overwrite)
    dt = time.time() - t0

    summary: dict = {"elapsed_min": round(dt / 60, 1)}
    if "frame" in state:
        summary["frame_rows"] = int(len(state["frame"]))
    if "propensity_summary" in state:
        summary["overlap"] = state["propensity_summary"].to_dict("records")
    if "moderation" in state and len(state["moderation"]):
        summary["moderation"] = state["moderation"][
            ["question", "outcome", "representation", "moderation_d_elpd", "moderation_any_axis", "e_value"]
        ].to_dict("records")
    if "confounder" in state and len(state["confounder"]):
        summary["confounder"] = state["confounder"][["representation", "axis", "survives", "attenuation_pct"]].to_dict("records")
    if not args.no_plots and "moderation" in state:
        summary["figure"] = str(TreatmentVisualizer(config).moderation_bars(state["moderation"]))
    print(json.dumps(summary, indent=2, default=str))
    print("\n[treatment-oop] done.")


if __name__ == "__main__":
    main()
