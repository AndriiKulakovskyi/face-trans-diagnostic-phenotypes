#!/usr/bin/env python
"""Re-fit ONLY the 9-dim mixed stage at the certified production budget.

The medium ladder converges cleanly through the continuous rungs (S1-S3) but the
mixed S5's binary/ordinal/count `substance` factor (BP/SZ-only; DR has no substance
data) mixes slowly at the lean 2-chain/800-draw diagnostic budget.  This re-runs just
that stage at the certified protocol (4 chains, tune=2000, draws=1500), warm-started
from the cached medium S3, and reports a per-group diagnostic so we can see whether
the structural backbone certifies even where the substance ESS stays provisional
(as in the certified s5_certify9 multi-seed protocol).

    PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE python notebooks/refit_mixed_prod.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import arviz as az  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from face.models.bayesian.measurement_model_oop import MeasurementConfig, StageRunner  # noqa: E402

config = MeasurementConfig()
runner = StageRunner(config)
plan = config.stage_plan
s3_def = next(s for s in plan if s.name == "s3_continuous")
s5_def = next(s for s in plan if s.name == "s5_9dim_mixed")

# Reconstruct the medium S3 stage exactly so its cache is reused for the warm-start
# (must match the medium stage_spec: N=2000 balanced, 600/800/2).
s3_medium = replace(s3_def, n_subsample=2000, balanced=True, draws=600, tune=800, chains=2)
# Production-budget mixed stage (overwrites the medium s5 cache).
s5_prod = replace(s5_def, draws=1500, tune=2000, chains=4)

print(f"Re-fitting {s5_prod.name} at production budget "
      f"(draws={s5_prod.draws}, tune={s5_prod.tune}, chains={s5_prod.chains}), "
      f"warm-started from cached {s3_medium.name}.", flush=True)
idata, manifest = runner.run_stage(s5_prod, overwrite=True, prev_stage=s3_medium)
print("diagnostics (max over all structural + per-item params):", manifest.get("diagnostics"), flush=True)

# Per-group breakdown: does the structural backbone certify even where substance is thin?
post = idata.posterior
groups = {v: [v] for v in ["lam_pos", "lam_cross", "sigma", "Phi_spec"] if v in post}
groups["lh_(home)"] = [v for v in post.data_vars if str(v).startswith("lh_")]
groups["lg_(G)"] = [v for v in post.data_vars if str(v).startswith("lg_")]
rows = []
for g, vs in groups.items():
    if not vs:
        continue
    s = az.summary(idata, var_names=vs)
    rc = "r_hat" if "r_hat" in s.columns else "rhat"
    ec = next((c for c in s.columns if c.startswith("ess")), None)
    rh = pd.to_numeric(s[rc], errors="coerce")
    es = pd.to_numeric(s[ec], errors="coerce")
    rows.append((g, len(s), round(float(rh.max()), 2), int((rh > 1.05).sum()),
                 round(float(es.min()), 0), round(float(es.median()), 0)))
print(pd.DataFrame(rows, columns=["group", "n", "max_rhat", "n_rhat>1.05", "min_ess", "median_ess"]).to_string(index=False), flush=True)
print("REFIT COMPLETE.", flush=True)
