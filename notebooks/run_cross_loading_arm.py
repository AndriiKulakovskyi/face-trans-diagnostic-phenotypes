#!/usr/bin/env python
"""Cross-loading arm (sensitivity): fit the certified S5 copula map PLUS the theory-motivated
``plausible_cross`` specific cells (the immunometabolic metabolic<->inflammatory bridge), freed at
Normal(0, 0.25), warm-started from the certified ``s5_9dim_mixed`` so it starts in the hard-zero basin.

This NEVER re-runs or overwrites the certified fit: it only reads the cached
``results/face/oop_measurement/copula/s5_9dim_mixed/idata.nc`` for the warm start, and writes the new
fit to a separate stage dir ``.../copula/s5_xcross/``.

    # quick wiring check (tiny draws, verifies the 37 cells are freed + warm-start loads):
    HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_cross_loading_arm.py --smoke
    # full fit (long; run detached via scripts/run_job.py):
    python3 scripts/run_job.py xcross -- env HDF5_USE_FILE_LOCKING=FALSE \
        python notebooks/run_cross_loading_arm.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src" / "face" / "models").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not locate FACE repository root from {start}")


REPO = _find_repo_root(Path(__file__).resolve())
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

loaded_face = sys.modules.get("face")
loaded_face_file = getattr(loaded_face, "__file__", None) if loaded_face is not None else None
if loaded_face is not None and (
    loaded_face_file is None or SRC not in Path(loaded_face_file).resolve().parents
):
    for module_name in [name for name in sys.modules if name == "face" or name.startswith("face.")]:
        del sys.modules[module_name]

from face.models.bayesian.measurement_model_oop import (  # noqa: E402
    MeasurementConfig,
    MeasurementDataset,
    StageRunner,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true",
                   help="Fast wiring check (tiny draws/tune, 2 chains) — verifies the cross cells are "
                        "freed and the warm-start loads; NOT convergence evidence.")
    p.add_argument("--overwrite", action="store_true", help="Refit even if s5_xcross is cached.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = MeasurementConfig().with_gaussian_copula()
    config = replace(config, output_dir=config.output_dir / "copula", figure_dir=config.figure_dir / "copula")

    source = config.stage_plan[-1]                      # certified s5_9dim_mixed (warm-start source; cached)
    stage = config.cross_loading_stage_plan()[0]        # s5_xcross
    if args.smoke:
        stage = replace(stage, name="smoke_s5_xcross", draws=40, tune=40, chains=2)

    src_idata = config.output_dir / source.name / "idata.nc"
    if not src_idata.exists():
        raise FileNotFoundError(f"warm-start source missing: {src_idata} (run the certified copula S5 first)")

    # Report what the arm frees, before the long fit, by building the spec once.
    ds = MeasurementDataset(config)
    base = ds.mixed(stage.factors, explicit_factors=stage.explicit_factors, min_cohorts=stage.min_cohorts,
                    balanced=stage.balanced, n_subsample=stage.n_subsample, seed=stage.seed).base
    spec_hard = ds.loading_spec(base, windows=True, specific_cross=False)
    spec_x = ds.loading_spec(base, windows=True, specific_cross=True, cross_sd_scale=stage.cross_sd_scale)
    cross_cells = sorted(
        (base.items[j], base.factor_cols[c]) for (j, c), k in spec_x.kind.items() if k == "cross"
    )
    print(f"[arm] stage={stage.name}  warm-start={source.name}", flush=True)
    print(f"[arm] free loading cells: hard-zero {spec_hard.n_free} -> cross-loading {spec_x.n_free} "
          f"(+{spec_x.n_free - spec_hard.n_free} specific cross cells)", flush=True)
    print(f"[arm] freed cross cells ({len(cross_cells)}): {cross_cells}", flush=True)
    print(f"[arm] prior on cross cells: Normal(0, {0.25 * stage.cross_sd_scale:.3g})", flush=True)

    runner = StageRunner(config)
    t0 = time.time()
    _idata, manifest = runner.run_stage(stage, overwrite=args.overwrite, prev_stage=source)
    elapsed = time.time() - t0
    diag = manifest.get("diagnostics", {})
    print(json.dumps({"stage": stage.name, "diagnostics": diag, "elapsed_sec": round(elapsed, 1),
                      "out": str(config.output_dir / stage.name)}, indent=2), flush=True)
    print(f"[arm] DONE -> {config.output_dir / stage.name}/idata.nc", flush=True)
    print("[arm] next: export CI-aware loadings with "
          f"`python notebooks/run_export_loadings.py --idata {config.output_dir / stage.name}/idata.nc`",
          flush=True)


if __name__ == "__main__":
    main()
