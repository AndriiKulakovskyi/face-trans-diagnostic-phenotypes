#!/usr/bin/env python
"""Run the OOP FACE M1 measurement model outside Jupyter.

The script mirrors the notebook workflow but is easier to run/retry from a terminal:

    python notebooks/run_measurement_model_oop.py --mode smoke
    python notebooks/run_measurement_model_oop.py --mode mixed-smoke --overwrite
    python notebooks/run_measurement_model_oop.py --mode medium

Modes:
* smoke: fast S1 wiring check only.
* mixed-smoke: fast S1 + mixed-likelihood wiring checks.
* medium: cohort-balanced diagnostic development runs.
* production: full continuous stages and largest-N mixed stage from MeasurementConfig.

Production can be long. Start with mixed-smoke before escalating.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[reportMissingImports]


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

from face.measurement.engine import (  # noqa: E402
    MeasurementConfig,
    MeasurementDataset,
    MeasurementVisualizer,
    PatientProjector,
    StageDefinition,
    StageRunner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["smoke", "mixed-smoke", "medium", "production"],
        default="smoke",
        help="Run size. Start with smoke or mixed-smoke before medium/production.",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Speed-oriented variant (hard-zero, fast-mode flag).",
    )
    parser.add_argument(
        "--soft",
        action="store_true",
        help="Soft-unlikely SENSITIVITY arm (free unlikely_cross / near-zero g-anchor cells). "
        "The default is the hard-zero primary; --soft writes to an 'soft/' subdir to keep results separate.",
    )
    parser.add_argument(
        "--likelihood-mode",
        choices=["native", "gaussian_copula"],
        default="native",
        help="native = certified tiered mixed likelihood (default); gaussian_copula = rank-INT "
        "Gaussianize the continuous + high-cardinality ordinal/count block (acceleration vertical). "
        "Copula runs write to a 'copula/' subdir to keep results separate.",
    )
    parser.add_argument(
        "--cohort-weighted",
        action="store_true",
        help="§3.6 cohort-weighted FULL-N fit: use all patients with weights that equalize each "
        "cohort's influence (transdiagnostic estimand, single coherent posterior). Forces full-N "
        "stages; writes to a 'weighted/' subdir. The mixed stage at full-N is heavy (run detached).",
    )
    parser.add_argument(
        "--substance-orthogonal",
        action="store_true",
        help="Pin the substance factor orthogonal to the other specifics (recommended substance "
        "handling: its cross-factor correlations are non-identifiable/unstable). Writes to a 'subortho/' subdir.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Refit stages even when cache exists.")
    parser.add_argument("--no-plots", action="store_true", help="Skip projection and visualization outputs.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override results directory.")
    parser.add_argument("--figure-dir", type=Path, default=None, help="Override figure directory.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> MeasurementConfig:
    config = MeasurementConfig()  # hard-zero primary by default
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    if args.figure_dir is not None:
        config = replace(config, figure_dir=args.figure_dir)
    if args.mode in {"smoke", "mixed-smoke"}:
        config = config.with_smoke_defaults()
    elif args.fast_mode:
        config = config.with_fast_mode()
    if args.soft:
        # Sensitivity arm: keep results separate from the hard-zero primary.
        config = config.with_soft_unlikely()
        config = replace(config, output_dir=config.output_dir / "soft", figure_dir=config.figure_dir / "soft")
    if args.likelihood_mode == "gaussian_copula":
        # Acceleration vertical: keep results separate from the native primary.
        config = config.with_gaussian_copula()
        config = replace(config, output_dir=config.output_dir / "copula", figure_dir=config.figure_dir / "copula")
    if args.cohort_weighted:
        # §3.6 all-data balanced-influence fit; separate subdir.
        config = config.with_cohort_weighted()
        config = replace(config, output_dir=config.output_dir / "weighted", figure_dir=config.figure_dir / "weighted")
    if args.substance_orthogonal:
        # Recommended substance handling: model it as an independent axis (non-identifiable couplings).
        config = config.with_substance_orthogonal()
        config = replace(config, output_dir=config.output_dir / "subortho", figure_dir=config.figure_dir / "subortho")
    return config


def build_stages(config: MeasurementConfig, args: argparse.Namespace) -> list[StageDefinition]:
    if args.cohort_weighted:
        # Cohort-weighted is a FULL-N method: use all patients (n_subsample=None, not balanced --
        # the weights handle cohort balance). Keep each rung's native draws/tune (continuous
        # 1000/1000/4, mixed 1500/2000/4 @ ta 0.95). The mixed full-N rung is the heavy long pole.
        return [replace(stage, n_subsample=None, balanced=False) for stage in config.stage_plan]
    if args.mode == "smoke":
        return list(config.smoke_stage_plan)
    if args.mode == "mixed-smoke":
        return list(config.smoke_stage_plan) + list(config.mixed_smoke_stage_plan)
    if args.mode == "medium":
        # Diagnostic scale: N=2000 balanced subsample with the certified sensitivity-arm
        # sampling budget (2 chains; 600/800 continuous, 800/1200 mixed).  Two chains
        # still give a meaningful R-hat/ESS read, and this is the regime the certified
        # 10b/s5_corrg fits use.  Production (full draws/chains/N) is the default plan.
        stages: list[StageDefinition] = []
        for stage in config.stage_plan:
            if stage.mixed and args.likelihood_mode == "gaussian_copula":
                # The copula transform converts the mixed residual from multimodal (budget-proof)
                # to slow-mixing-unimodal, so the fuller budget (4 chains / tune 2000 / 1500 draws)
                # takes the lone weak correlation (dev<->suicidality) sub-1.05 -- max R-hat 1.04, 0 div.
                stages.append(replace(stage, n_subsample=stage.n_subsample or 2000, balanced=True, draws=1500, tune=2000, chains=4))
            elif stage.mixed:
                stages.append(replace(stage, n_subsample=stage.n_subsample or 2000, balanced=True, draws=800, tune=1200, chains=2))
            else:
                stages.append(replace(stage, n_subsample=2000, balanced=True, draws=600, tune=800, chains=2))
        return stages
    return list(config.stage_plan)


def input_summary(config: MeasurementConfig, dataset: MeasurementDataset, stages: list[StageDefinition]) -> dict[str, Any]:
    first_stage = stages[0]
    baseline = pd.read_parquet(config.processed_dir / "baseline_v0.parquet")
    prior = pd.read_csv(config.prior_matrix)
    core = dataset.core(
        first_stage.factors,
        correlated=first_stage.correlated,
        windows=first_stage.windows,
        n_subsample=first_stage.n_subsample,
        seed=first_stage.seed,
    )
    spec = dataset.loading_spec(core, windows=first_stage.windows)
    return {
        "repo": str(REPO),
        "baseline_shape": list(baseline.shape),
        "baseline_missing_fraction": round(float(baseline.isna().mean().mean()), 4),
        "prior_cells": int(len(prior)),
        "first_stage_core_shape": list(core.M.shape),
        "first_stage_free_loading_cells": int(spec.n_free),
        "covariate_count": int(core.covariates.shape[1]),
    }


def run_stages(
    runner: StageRunner,
    stages: list[StageDefinition],
    *,
    overwrite: bool,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    prev_stage: StageDefinition | None = None
    for stage in stages:
        print(f"\n=== Running {stage.name} ===", flush=True)
        t0 = time.time()
        # Thread the previous rung so each stage warm-starts from its posterior.
        idata, manifest = runner.run_stage(stage, overwrite=overwrite, prev_stage=prev_stage)
        elapsed = time.time() - t0
        diagnostics = manifest.get("diagnostics", {})
        print(json.dumps({"stage": stage.name, "diagnostics": diagnostics, "elapsed_sec": round(elapsed, 1)}, indent=2))
        results[stage.name] = {"idata": idata, "manifest": manifest, "elapsed_sec": elapsed, "stage": stage}
        prev_stage = stage
    return results


def make_outputs(
    config: MeasurementConfig,
    dataset: MeasurementDataset,
    results: dict[str, dict[str, Any]],
) -> dict[str, str]:
    continuous = next((result for result in reversed(list(results.values())) if not result["stage"].mixed), None)
    if continuous is None:
        return {}

    stage: StageDefinition = continuous["stage"]
    idata = continuous["idata"]
    core = dataset.core(
        stage.factors,
        correlated=stage.correlated,
        windows=stage.windows,
        n_subsample=stage.n_subsample,
        seed=stage.seed,
    )
    spec = dataset.loading_spec(core, windows=stage.windows)
    projector = PatientProjector(config)
    visualizer = MeasurementVisualizer(config)
    projection = projector.projection_frame(core, idata.posterior)

    paths = {
        "loading_atlas": str(visualizer.loading_atlas(spec, idata.posterior, filename=f"{stage.name}_loading_atlas.png")),
        "phi_heatmap": str(visualizer.phi_heatmap(idata.posterior, core.factor_cols, filename=f"{stage.name}_phi.png")),
        "reliability": str(visualizer.reliability_bar(projection, core.factor_cols, filename=f"{stage.name}_reliability.png")),
    }

    rel_col = "overall_severity__reliability"
    examples: dict[str, Any] = {}
    for tier in ["well", "partial", "prior-dominated"]:
        matches = projection.index[projection[rel_col] == tier]
        if len(matches):
            examples[tier] = matches[0]
            paths[f"patient_{tier}"] = str(
                visualizer.patient_uncertainty(
                    projection,
                    matches[0],
                    core.factor_cols,
                    filename=f"{stage.name}_patient_{tier}.png",
                )
            )
    # Patient-projection map demo: biology (metabolic) vs general severity G, with
    # the example patients drawn as 94% HDI crosses (position + uncertainty).
    if {"overall_severity__mean", "metabolic__mean"} <= set(projection.columns):
        paths["map_scatter"] = str(
            visualizer.map_scatter(
                projection,
                "overall_severity",
                "metabolic",
                highlight=examples,
                filename=f"{stage.name}_map_scatter.png",
            )
        )
    return paths


def main() -> None:
    args = parse_args()
    config = build_config(args)
    dataset = MeasurementDataset(config)
    stages = build_stages(config, args)
    runner = StageRunner(config)

    print("Input summary:")
    print(json.dumps(input_summary(config, dataset, stages), indent=2))
    print("\nStages:")
    print(json.dumps([stage.__dict__ for stage in stages], indent=2, default=str))

    results = run_stages(runner, stages, overwrite=args.overwrite)
    if not args.no_plots:
        print("\nGenerated outputs:")
        print(json.dumps(make_outputs(config, dataset, results), indent=2))

    print("\nRun complete.")


if __name__ == "__main__":
    main()
