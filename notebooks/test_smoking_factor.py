#!/usr/bin/env python
"""Hypothesis test: does a SMOKING factor (Fagerstrom + pack-years) identify cleanly?

Builds a temporary prior matrix that relabels the two smoking indicators (`fagers`,
`sudose_cigarettes_lt`) as home items of a new `smoking` factor (pack-years treated as
continuous/log so it joins the fast marginalized block), then fits the S1 continuous
backbone + smoking (correlated Phi, bifactor G) at full N and reports whether smoking's
two home loadings are large + well-mixed and how smoking correlates with the other axes.

    PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE python notebooks/test_smoking_factor.py
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
import pymc as pm  # noqa: E402

from face.models.bayesian.measurement_model_oop import (  # noqa: E402
    BayesianBifactorESEM,
    MeasurementConfig,
    MeasurementDataset,
)

# --- build a patched prior matrix: smoking as its own factor -------------------------------
m = pd.read_csv(REPO / "configs" / "prior_loading_matrix_v3.csv")
home_rows = m["prior_type"].isin(["primary", "g_anchor"]) & m["item"].isin(["fagers", "sudose_cigarettes_lt"])
m.loc[home_rows, "factor"] = "smoking"
# pack-years: treat as continuous log-count so it is marginalizable alongside Fagerstrom.
m.loc[m["item"] == "sudose_cigarettes_lt", "likelihood_family"] = "lognormal"
m.loc[m["item"] == "sudose_cigarettes_lt", "modeling_block"] = "continuous"
tmp_matrix = REPO / "configs" / "_tmp_smoking_matrix.csv"
m.to_csv(tmp_matrix, index=False)

# covariate_mode="none": we want the raw coherence of the smoking signal (residualizing on
# age would strip the age-driven shared variance that is part of cumulative smoking).
# soft_unlikely controlled by env: SOFT=1 floods the column with ~70 weak cross-loadings,
# SOFT=0 (hard-zero) leaves only the 2 home loadings + bifactor G.
import os
soft = os.environ.get("SOFT", "1") == "1"
config = MeasurementConfig(prior_matrix=tmp_matrix, covariate_mode="none",
                           soft_unlikely=soft, soft_g_anchor_specific=soft)
print(f"soft_unlikely={soft}", flush=True)
dataset = MeasurementDataset(config)
builder = BayesianBifactorESEM(config)

factors = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep", "smoking"]
# N=2000 balanced is ample to pin 2 smoking loadings (and fast); the full-N empirical
# correlation already established cross-cohort coherence (r~0.43-0.55).
core = dataset.core(factors, correlated=True, windows=False, balanced=True, n_subsample=2000, seed=20260605)
spec = dataset.loading_spec(core, windows=False)
sm_items = [it for it, h in zip(core.items, core.home) if h == "smoking"]
print(f"N={core.M.shape[0]} J={core.M.shape[1]} factors={core.factor_cols}", flush=True)
print(f"smoking home items in fit: {sm_items}", flush=True)
n_sm = {it: int(np.isfinite(core.M[:, core.items.index(it)]).sum()) for it in sm_items}
print(f"smoking items observed in subsample: {n_sm}", flush=True)

model = builder.build_marginalized(core, spec, correlated=True)
with model:
    idata = pm.sample(draws=600, tune=800, chains=4, target_accept=0.9, random_seed=20260605,
                      nuts_sampler="numpyro", nuts={"max_tree_depth": 8},
                      idata_kwargs={"log_likelihood": False}, progressbar=True)

post = idata.posterior
Lam = post["Lam"]
sc = core.factor_cols.index("smoking")
gc = core.factor_cols.index("overall_severity")
print("\n=== SMOKING factor home loadings (should be large + positive if it identifies) ===", flush=True)
for it in sm_items:
    j = core.items.index(it)
    load = Lam.isel({Lam.dims[-1]: sc, Lam.dims[-2]: j})
    print(f"  {it:24s} smoking-loading mean={float(load.mean()):+.3f} sd={float(load.std()):.3f}", flush=True)

# convergence on the structural loadings
summ = az.summary(idata, var_names=[v for v in ["lam_pos", "lam_cross", "sigma", "Phi_spec"] if v in post])
rc = "r_hat" if "r_hat" in summ.columns else "rhat"
ec = next((c for c in summ.columns if c.startswith("ess")), None)
print(f"\nstructural diagnostics: max_rhat={pd.to_numeric(summ[rc],errors='coerce').max():.3f} "
      f"min_ess={pd.to_numeric(summ[ec],errors='coerce').min():.0f} "
      f"divergences={int(np.asarray(idata.sample_stats['diverging']).sum())}", flush=True)

Phi = np.asarray(post["Phi"].mean(("chain", "draw")).values)
print("\n=== smoking factor correlations with the other axes (Phi row) ===", flush=True)
for f in core.factor_cols:
    if f == "smoking":
        continue
    print(f"  Phi(smoking, {f:16s}) = {Phi[sc, core.factor_cols.index(f)]:+.3f}", flush=True)
# per-chain smoking loadings (is it multimodal across chains?)
print("\n=== per-chain smoking loadings (multimodality check) ===", flush=True)
for it in sm_items:
    j = core.items.index(it)
    for ch in range(Lam.sizes["chain"]):
        v = float(Lam.isel({Lam.dims[-1]: sc, Lam.dims[-2]: j}).isel(chain=ch).mean().values)
        print(f"  {it:24s} chain {ch}: {v:+.3f}", flush=True)
# top rhat param
alls = az.summary(idata)
rc2 = "r_hat" if "r_hat" in alls.columns else "rhat"
top = alls.sort_values(rc2, ascending=False).head(6)
print("\n=== top-6 rhat params ===", flush=True)
print(top[[rc2] + [c for c in alls.columns if c.startswith("ess")][:1]].to_string(), flush=True)
# model-implied vs empirical correlation of the two smoking items
jf, jp = core.items.index("fagers"), core.items.index("sudose_cigarettes_lt")
Lm = np.asarray(Lam.mean(("chain", "draw")).values)
Ph = np.asarray(post["Phi"].mean(("chain", "draw")).values)
sg = config.psi_floor + np.asarray(post["sigma"].mean(("chain", "draw")).values)
Sig = Lm @ Ph @ Lm.T + np.diag(sg ** 2)
implied = Sig[jf, jp] / np.sqrt(Sig[jf, jf] * Sig[jp, jp])
M = core.M
ok = np.isfinite(M[:, jf]) & np.isfinite(M[:, jp])
emp = float(np.corrcoef(M[ok, jf], M[ok, jp])[0, 1])
print(f"\nfagers~packyears  model-implied r={implied:+.3f}  empirical r={emp:+.3f}  (n={int(ok.sum())})", flush=True)
idata.to_netcdf(str(config.output_dir / "_smoking_test_idata.nc"))
print("\nSMOKING TEST COMPLETE.", flush=True)
tmp_matrix.unlink(missing_ok=True)
