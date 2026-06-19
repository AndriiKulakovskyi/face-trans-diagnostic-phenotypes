#!/usr/bin/env python
"""Re-run the full staged ladder with HARD-ZERO unlikely cells and confirm the 9-dim mixed S5 converges.

The smoking-factor test showed soft_unlikely=True starves thin factors (their column is flooded by
~70-980 weak cross-loadings). This runs a fully hard-zero ladder (soft_unlikely=False,
soft_g_anchor_specific=False) — the certified engine's primary — at N=2000 balanced, S5 at 4 chains so
any residual multimodality shows. Writes to results/face/oop_measurement/hardzero/ (soft results kept).

    PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE python notebooks/refit_s5_hardzero.py
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

config = MeasurementConfig(
    soft_unlikely=False,
    soft_g_anchor_specific=False,
    output_dir=MeasurementConfig().output_dir / "hardzero",
)
runner = StageRunner(config)
plan = config.stage_plan


def med(stage, **kw):
    return replace(stage, n_subsample=2000, balanced=True, **kw)


stages = [
    med(plan[0], draws=600, tune=800, chains=4),     # s1_core
    med(plan[1], draws=600, tune=800, chains=4),     # s2_esem
    med(plan[2], draws=600, tune=800, chains=4),     # s3_continuous
    med(plan[3], draws=1000, tune=1500, chains=4),   # s5_9dim_mixed (4 chains: exposes multimodality)
]

prev = None
for st in stages:
    print(f"\n=== HARD-ZERO {st.name} (N=2000 balanced, {st.draws}+{st.tune}x{st.chains}) ===", flush=True)
    idata, manifest = runner.run_stage(st, overwrite=True, prev_stage=prev)
    print(f"{st.name}: {manifest.get('diagnostics')}", flush=True)
    prev = st

# --- S5 per-group breakdown + substance loadings (the decisive comparison vs the soft fit) -------
idata = az.from_netcdf(str(config.output_dir / "s5_9dim_mixed" / "idata.nc"))
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
print("\n=== S5 HARD-ZERO per-group diagnostics ===", flush=True)
print(pd.DataFrame(rows, columns=["group", "n", "max_rhat", "n_rhat>1.05", "min_ess", "median_ess"]).to_string(index=False), flush=True)

print("\n=== substance explicit item loadings (rare SUD expected provisional) ===", flush=True)
subitems = ["sudose_cigarettes_lt", "suoccur_alcool", "suoccur_cannabis"]
names = [f"lh_{it}" for it in subitems if f"lh_{it}" in post.data_vars]
s = az.summary(idata, var_names=names)
rc = "r_hat" if "r_hat" in s.columns else "rhat"
cols = [c for c in ["mean", "sd", rc] if c in s.columns] + [c for c in s.columns if c.startswith("ess")][:1]
print(s[cols].to_string(), flush=True)
print("\nHARD-ZERO LADDER COMPLETE.", flush=True)
