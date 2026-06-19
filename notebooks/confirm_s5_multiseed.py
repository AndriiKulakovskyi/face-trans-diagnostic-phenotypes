#!/usr/bin/env python
"""Multi-seed confirmation that the hard-zero 9-dim mixed S5 converges, incl. developmental_risk.

The single hard-zero S5 fit converged except for one stuck chain on the recall-noisy
developmental_risk factor (3/4 chains agreed at R-hat ~1.02; one collapsed). A stuck chain
is stochastic, so this runs several seeds with stronger warmup (tune=2000, target_accept=0.95)
and reports, per seed: full-chain R-hat, R-hat after dropping the single worst chain, and the
developmental item loadings -- plus cross-seed stability of those loadings (the certified
s5_certify "resample-stability" criterion). Hard-zero, N=2000 balanced, warm-started from the
cached hard-zero S3.

    PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE python notebooks/confirm_s5_multiseed.py
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

SEEDS = [20260606, 20260607, 20260608]
config = MeasurementConfig(output_dir=MeasurementConfig().output_dir / "hardzero")  # hard-zero is now the default
runner = StageRunner(config)
plan = config.stage_plan
s3_def = next(s for s in plan if s.name == "s3_continuous")
s5_def = next(s for s in plan if s.name == "s5_9dim_mixed")
# the cached hard-zero S3 used for the warm-start (must match its stage_spec)
s3_warm = replace(s3_def, n_subsample=2000, balanced=True, draws=600, tune=800, chains=4)

# developmental_risk explicit items (those that pass coverage will be in the fit; the
# stability report uses whichever have an lh_ loading in the posterior).
DEV_ITEMS = ["ctq40", "ctq41", "honeonat", "mere_structure", "naisstyp",
             "pere_structure", "prembrth", "traumacra_mhoccur", "autneuro_mhoccur", "epilepsie_mhoccur"]


def worst_chain_dropped_rhat(idata, var_names):
    """Max R-hat over var_names after dropping the single chain that most inflates it."""
    nch = idata.posterior.sizes["chain"]
    best = np.inf
    for drop in range(nch):
        keep = [c for c in range(nch) if c != drop]
        sub = idata.sel(chain=keep)
        s = az.summary(sub, var_names=[v for v in var_names if v in sub.posterior])
        rc = "r_hat" if "r_hat" in s.columns else "rhat"
        best = min(best, float(pd.to_numeric(s[rc], errors="coerce").max()))
    return best


rows = []
dev_loadings = {}
for seed in SEEDS:
    stage = replace(s5_def, name=f"s5_seed{seed}", n_subsample=2000, balanced=True,
                    draws=1000, tune=2000, chains=4, target_accept=0.95, seed=seed)
    print(f"\n=== S5 hard-zero seed {seed} (4 chains, tune 2000, ta 0.95) ===", flush=True)
    idata, manifest = runner.run_stage(stage, overwrite=True, prev_stage=s3_warm)
    diag = manifest.get("diagnostics", {})
    struct = [v for v in ["lam_pos", "lam_cross", "sigma", "Phi_spec"] if v in idata.posterior]
    dropped = worst_chain_dropped_rhat(idata, struct)
    # developmental item loadings (home lh_)
    post = idata.posterior
    dl = {}
    for it in DEV_ITEMS:
        if f"lh_{it}" in post.data_vars:
            dl[it] = float(post[f"lh_{it}"].mean().values)
    dev_loadings[seed] = dl
    rows.append({"seed": seed, "rhat_all": diag.get("rhat"), "ess": diag.get("ess"),
                 "div": diag.get("divergences"), "rhat_drop_worst_chain": round(dropped, 3)})
    print(f"seed {seed}: rhat_all={diag.get('rhat')} rhat_drop_worst={dropped:.3f} ess={diag.get('ess')} div={diag.get('divergences')}", flush=True)

print("\n=== per-seed convergence ===", flush=True)
print(pd.DataFrame(rows).to_string(index=False), flush=True)

print("\n=== cross-seed developmental loading stability (home lh_) ===", flush=True)
allk = sorted({k for d in dev_loadings.values() for k in d})
for it in allk:
    vals = [dev_loadings[s].get(it) for s in SEEDS if it in dev_loadings[s]]
    if vals:
        print(f"  lh_{it:10s} per-seed={[round(v, 3) for v in vals]}  range={max(vals) - min(vals):.3f}", flush=True)
print("\nMULTISEED CONFIRMATION COMPLETE.", flush=True)
