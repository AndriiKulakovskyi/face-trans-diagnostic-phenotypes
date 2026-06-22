#!/usr/bin/env python
"""Run the OOP prognosis engine (M4 reworked on the Gaussian-copula M2 object) from a terminal.

Mirrors ``run_strata_model_oop.py``: it walks the deterministic plan
(frame -> reference -> incremental -> transdiagnostic -> endpoints -> clinical_value -> robustness ->
consolidate) on the copula M2 hand-off (``results/face/strata_oop/``), caching each stage under
``results/face/prognosis_oop/<stage>/``. Wraps the proven kernels (``glm``/``compare``/``reference``); never
touches the native ``results/face/m4`` or ``scripts/40-48``.

    PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_prognosis_model_oop.py --mode smoke
    PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_prognosis_model_oop.py --mode full --stop-after incremental
    PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_prognosis_model_oop.py --mode full

Modes:
* smoke: tiny draws / 2 chains — a wiring check, not science.
* full:  the reported plan (4-chain NUTS; run detached via scripts/run_job.py for the full plan).
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
        if (candidate / "src" / "face" / "prognosis").exists() and (candidate / "pyproject.toml").exists():
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

from face.prognosis.prognosis_model_oop import (  # noqa: E402
    PrognosisConfig,
    PrognosisRunner,
    PrognosisVisualizer,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--stop-after", default=None, help="halt after this stage (e.g. 'incremental').")
    p.add_argument("--overwrite", action="store_true", help="recompute stages even when a cache exists.")
    p.add_argument("--no-plots", action="store_true", help="skip figures.")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--figure-dir", type=Path, default=None)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> PrognosisConfig:
    config = PrognosisConfig()
    if args.mode == "smoke":
        config = config.with_smoke_defaults()
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    if args.figure_dir is not None:
        config = replace(config, figure_dir=args.figure_dir)
    return config


def main() -> None:
    args = parse_args()
    config = build_config(args)
    runner = PrognosisRunner(config)
    print(f"[prognosis-oop] mode={args.mode} output={config.output_dir}", flush=True)

    t0 = time.time()
    state = runner.run_plan(stop_after=args.stop_after, overwrite=args.overwrite)
    dt = time.time() - t0

    summary: dict = {"elapsed_min": round(dt / 60, 1)}
    if "frame" in state:
        summary["frame_rows"] = int(len(state["frame"]))
    if "operative_k" in state:
        summary["operative_k"] = state["operative_k"]
    if "prognosis_summary" in state:
        summary["summary_rows"] = int(len(state["prognosis_summary"]))

    if not args.no_plots and "incremental" in state:
        summary["figure"] = str(PrognosisVisualizer(config).incremental_bars(state["incremental"]))

    print(json.dumps(summary, indent=2, default=str))
    print("\n[prognosis-oop] done.")


if __name__ == "__main__":
    main()
