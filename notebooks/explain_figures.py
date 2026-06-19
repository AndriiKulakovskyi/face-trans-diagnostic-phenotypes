#!/usr/bin/env python
"""Generate the key figures of the converged hard-zero 9-dim mixed-likelihood ESEM model.

Uses the cleanest multi-seed mixed fit (s5_seed20260607, R-hat 1.06) for the 9-factor
structure, and the hard-zero S3 continuous fit for patient projection. Writes to
docs/figures/oop_measurement/hardzero/.

    PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE python notebooks/explain_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import arviz as az  # noqa: E402

from face.models.bayesian.measurement_model_oop import (  # noqa: E402
    DEFAULT_EXPLICIT_FACTORS,
    S3_CONT_FACTORS,
    S5_FACTORS,
    MeasurementConfig,
    MeasurementDataset,
    MeasurementVisualizer,
    PatientProjector,
)

HARD = MeasurementConfig().output_dir / "hardzero"
config = MeasurementConfig(figure_dir=MeasurementConfig().figure_dir / "hardzero")  # hard-zero is the default
dataset = MeasurementDataset(config)
viz = MeasurementVisualizer(config)
projector = PatientProjector(config)
out = {}

# --- 9-dim mixed model: factor-correlation Phi + loading atlas (cleanest seed) ----------
mixed = az.from_netcdf(str(HARD / "s5_seed20260607" / "idata.nc"))
base = dataset.mixed(S5_FACTORS, explicit_factors=DEFAULT_EXPLICIT_FACTORS, min_cohorts=2,
                     balanced=True, n_subsample=2000, seed=20260607).base
spec = dataset.loading_spec(base, windows=True,
                            bifactor_g_sd={f: 0.05 for f in DEFAULT_EXPLICIT_FACTORS if f != "overall_severity"})
out["phi_9dim"] = str(viz.phi_heatmap(mixed.posterior, base.factor_cols, filename="s5_9dim_phi.png"))
out["atlas_9dim"] = str(viz.loading_atlas(spec, mixed.posterior, filename="s5_9dim_loading_atlas.png"))

# --- patient projection on the hard-zero continuous map (S3) -----------------------------
cont = az.from_netcdf(str(HARD / "s3_continuous" / "idata.nc"))
core = dataset.core(S3_CONT_FACTORS, correlated=True, windows=True, balanced=True, n_subsample=2000, seed=20260605)
proj = projector.projection_frame(core, cont.posterior)
out["reliability"] = str(viz.reliability_bar(proj, core.factor_cols, filename="s3_reliability.png"))
examples = {}
rel = "overall_severity__reliability"
for tier in ["well", "partial", "prior-dominated"]:
    m = proj.index[proj[rel] == tier]
    if len(m):
        examples[tier] = m[0]
        out[f"patient_{tier}"] = str(viz.patient_uncertainty(proj, m[0], core.factor_cols, filename=f"s3_patient_{tier}.png"))
out["map_scatter"] = str(viz.map_scatter(proj, "overall_severity", "metabolic", highlight=examples, filename="s3_map_scatter.png"))

import json  # noqa: E402
print(json.dumps(out, indent=2))
