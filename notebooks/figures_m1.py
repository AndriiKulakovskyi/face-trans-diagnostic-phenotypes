#!/usr/bin/env python
"""Generate core measurement-model figures + patient-projection demos from cached OOP fits.

Run AFTER a staged fit has populated ``results/m1_measurement/<stage>/idata.nc``
(e.g. ``python notebooks/run_measurement_model_oop.py --mode medium``).  This reuses the
cached posteriors; it never refits.

    PYTHONPATH=$PWD/src python notebooks/oop_make_figures.py            # medium-scale stages
    PYTHONPATH=$PWD/src python notebooks/oop_make_figures.py --balanced --n-subsample 2000

It writes, into ``docs/figures/oop_measurement/``:
  * the full 9-dim map: loading atlas + factor-correlation (Phi) heatmap from ``s5_9dim_mixed``
    (the headline core property — biology should sit ~orthogonal to G),
  * the continuous map + patient projections from the richest continuous rung
    (``s3_continuous``): reliability tiers, per-patient 94% HDI forests for one
    well / partial / prior-dominated patient, and a 2-D map scatter with those
    patients drawn as 94% HDI crosses.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src" / "face" / "models").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not locate FACE repository root from {start}")


REPO = _find_repo_root(Path(__file__).resolve())
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import arviz as az  # noqa: E402

from face.measurement.engine import (  # noqa: E402
    MeasurementConfig,
    MeasurementDataset,
    MeasurementVisualizer,
    PatientProjector,
    StageRunner,
)


def _load(config: MeasurementConfig, stage_name: str):
    nc = config.output_dir / stage_name / "idata.nc"
    if not nc.exists():
        return None
    return az.from_netcdf(str(nc))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous-stage", default="s3_continuous")
    parser.add_argument("--mixed-stage", default="s5_9dim_mixed")
    parser.add_argument("--balanced", action="store_true", help="Match a balanced-subsample fit.")
    parser.add_argument("--n-subsample", type=int, default=2000, help="Subsample N the stages were fit on (None for full-N).")
    parser.add_argument("--full-n", action="store_true", help="Stages were fit at full N (ignore --n-subsample).")
    parser.add_argument("--results-dir", type=Path, default=None, help="Override results dir (e.g. a copula/ subdir).")
    parser.add_argument("--figure-dir", type=Path, default=None, help="Override figure output dir.")
    parser.add_argument("--likelihood-mode", choices=["native", "gaussian_copula"], default="native",
                        help="Must match the fit so the reconstructed core uses the same encoding.")
    args = parser.parse_args()

    from dataclasses import replace  # noqa: PLC0415
    config = MeasurementConfig(likelihood_mode=args.likelihood_mode)
    if args.results_dir is not None:
        config = replace(config, output_dir=args.results_dir)
    if args.figure_dir is not None:
        config = replace(config, figure_dir=args.figure_dir)
    dataset = MeasurementDataset(config)
    projector = PatientProjector(config)
    visualizer = MeasurementVisualizer(config)
    runner = StageRunner(config)
    n_sub = None if args.full_n else args.n_subsample
    balanced = args.balanced or n_sub is not None
    out: dict[str, str] = {}

    # --- full 9-dim map: the headline core property -------------------------------------
    mixed_idata = _load(config, args.mixed_stage)
    if mixed_idata is not None:
        stage = next((s for s in config.stage_plan if s.name == args.mixed_stage), None)
        factors = stage.factors if stage else None
        base = dataset.mixed(
            factors,
            explicit_factors=(stage.explicit_factors if stage else None),
            min_cohorts=(stage.min_cohorts if stage else 2),
            balanced=True,
            n_subsample=(stage.n_subsample if stage else n_sub),
        ).base
        spec = dataset.loading_spec(base, windows=True, bifactor_g_sd={f: 0.05 for f in (stage.explicit_factors if stage else []) if f != "overall_severity"})
        out["mixed_loading_atlas"] = str(visualizer.loading_atlas(spec, mixed_idata.posterior, filename=f"{args.mixed_stage}_loading_atlas.png"))
        out["mixed_phi"] = str(visualizer.phi_heatmap(mixed_idata.posterior, base.factor_cols, filename=f"{args.mixed_stage}_phi.png"))

    # --- continuous map + patient-projection demos --------------------------------------
    cont_idata = _load(config, args.continuous_stage)
    if cont_idata is not None:
        stage = next((s for s in config.stage_plan if s.name == args.continuous_stage), None)
        core = dataset.core(
            stage.factors if stage else None,
            correlated=(stage.correlated if stage else True),
            windows=(stage.windows if stage else True),
            balanced=balanced,
            n_subsample=None if args.full_n else n_sub,
            seed=20260605,
        )
        spec = dataset.loading_spec(core, windows=(stage.windows if stage else True))
        out["cont_loading_atlas"] = str(visualizer.loading_atlas(spec, cont_idata.posterior, filename=f"{args.continuous_stage}_loading_atlas.png"))
        out["cont_phi"] = str(visualizer.phi_heatmap(cont_idata.posterior, core.factor_cols, filename=f"{args.continuous_stage}_phi.png"))

        projection = projector.projection_frame(core, cont_idata.posterior)
        out["reliability"] = str(visualizer.reliability_bar(projection, core.factor_cols, filename=f"{args.continuous_stage}_reliability.png"))

        examples: dict[str, object] = {}
        rel_col = "overall_severity__reliability"
        for tier in ["well", "partial", "prior-dominated"]:
            matches = projection.index[projection[rel_col] == tier]
            if len(matches):
                examples[tier] = matches[0]
                out[f"patient_{tier}"] = str(visualizer.patient_uncertainty(projection, matches[0], core.factor_cols, filename=f"{args.continuous_stage}_patient_{tier}.png"))
        if {"overall_severity__mean", "metabolic__mean"} <= set(projection.columns):
            out["map_scatter"] = str(visualizer.map_scatter(projection, "overall_severity", "metabolic", highlight=examples, filename=f"{args.continuous_stage}_map_scatter.png"))

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
