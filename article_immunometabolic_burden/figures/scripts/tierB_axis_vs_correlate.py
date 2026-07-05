#!/usr/bin/env python3
"""
Tier B — biology-as-axis vs biology-as-correlate model comparison.

Question the manuscript makes (Fig 3 / the hinge): immunometabolic biology is a
CO-EQUAL latent axis, not a downstream correlate of general severity. This script
turns that assertion into a held-out model comparison.

Two nested marginalized (Woodbury) models on the SAME N=9,013 continuous block:

  AXIS  (current):  5 factors  = severity + cognition + metabolic + inflammatory + sleep
                    the 37 biology indicators (29 metabolic + 8 inflammatory) load on
                    their own dedicated latent axes.
  CORR  (correlate): 3 factors = severity + cognition + sleep
                    the SAME 37 biology indicators are reassigned home=overall_severity,
                    so biology loads on general severity ONLY (no dedicated axis).

Both fit the identical patients/indicators; they differ only in whether biology has
its own axis. Compared by LOO (expected log pointwise predictive density) via ArviZ.
If AXIS wins (elpd_diff >> its SE), the biology axis carries predictive variance that
general severity cannot absorb — i.e. it is co-equal, not a correlate.

Run (repo venv):  cd article_v2 && ../.venv/bin/python figures/scripts/tierB_axis_vs_correlate.py
Smoke:            ... tierB_axis_vs_correlate.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parents[2]                       # repo root
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

OUT = ROOT / "results" / "face" / "tierB_axis_vs_correlate"
OUT.mkdir(parents=True, exist_ok=True)

from face.models.bayesian import continuous_core as cc


def build_correlate_prep(base):
    """Return a CorePrep where the metabolic+inflammatory indicators load on
    overall_severity (G) only, and the metabolic/inflammatory factors are removed.
    Reuses `base`'s data matrix and reassigns loading cells."""
    import copy
    # Which items are biology (home metabolic/inflammatory) in the AXIS prep?
    bio_items = {it for it, h in zip(base.items, base.home) if h in ("metabolic", "inflammatory")}
    # AXIS factor order: [overall_severity, cognition, metabolic, inflammatory, sleep]
    fc_axis = list(base.factor_cols)
    keep = [f for f in fc_axis if f not in ("metabolic", "inflammatory")]  # -> [sev, cog, sleep]
    old2new = {f: keep.index(f) for f in keep}
    g_new = keep.index("overall_severity")
    # Rebuild pos/sgn cells: biology items -> G column; others -> their kept column (dropped if home removed)
    def remap(cells, biology_to_g):
        out = []
        for (j, c, mu, sd) in cells:
            fold = fc_axis[c]
            if fc_axis[c] in ("metabolic", "inflammatory"):
                # this cell is a biology-axis loading; redirect the biology ITEM onto G
                if biology_to_g and base.items[j] in bio_items:
                    out.append((j, g_new, mu, sd))
                # else: drop (a cross-loading onto a removed factor)
            else:
                out.append((j, old2new[fold], mu, sd))
        return out
    prep = copy.copy(base)
    prep.factor_cols = keep
    prep.home = ["overall_severity" if h in ("metabolic", "inflammatory") else h for h in base.home]
    prep.pos_cells = remap(base.pos_cells, True)
    prep.sgn_cells = remap(base.sgn_cells, False)
    prep.g_col = g_new
    return prep


def fit(prep, *, draws, tune, chains, seed, label, sampler="pymc"):
    import pymc as pm
    model = cc.build_marginalized(prep)
    # attach per-patient pointwise log-lik as a deterministic so az.loo can read it
    t0 = time.time()
    kw = dict(draws=draws, tune=tune, chains=chains, target_accept=0.92, random_seed=seed,
              idata_kwargs={"log_likelihood": False}, progressbar=False)
    if sampler == "numpyro":
        kw["nuts_sampler"] = "numpyro"     # JAX path — fast but crashes at full N on this stack
    # default: PyMC C-backend NUTS — stable at full N, cores=chains
    else:
        kw["cores"] = chains
    with model:
        idata = pm.sample(**kw)
    dt = time.time() - t0
    div = int(np.asarray(idata.sample_stats["diverging"]).sum())
    print(f"  [{label}] {prep.M.shape[0]}x{prep.M.shape[1]}, F={len(prep.factor_cols)} | "
          f"{dt:.0f}s | divergences={div}", flush=True)
    return idata, model


def pointwise_ll(idata, model, prep):
    """Compute per-patient log-lik posterior array [chain,draw,N] from the marginalized
    Woodbury potential, evaluated at posterior draws — the input az.loo needs."""
    import numpy as np
    # rebuild the per-patient ll expression as a compiled function of the free RVs
    # (simpler: recompute from posterior of Lam, Phi, sigma via the model's cached tensors)
    # We recompute Sigma per draw in numpy for robustness.
    post = idata.posterior
    Lam = post["Lam"].values      # [c,d,J,F]
    Phi = post["Phi"].values      # [c,d,F,F]
    sig = post["sigma"].values    # [c,d,J]
    M = prep.M
    mask = (~np.isnan(M)).astype(float)   # [N,J]
    x = np.nan_to_num(M, nan=0.0)
    C, D, J, F = Lam.shape
    N = M.shape[0]
    log2pi = np.log(2*np.pi)
    ll = np.empty((C, D, N), dtype=np.float32)
    for c in range(C):
        for d in range(D):
            L = Lam[c,d]; P = Phi[c,d]; s2 = sig[c,d]**2
            Sig = L @ P @ L.T + np.diag(s2)          # [J,J]
            for i in range(N):
                o = mask[i] > 0
                xi = x[i][o]
                So = Sig[np.ix_(o,o)]
                sgn, logdet = np.linalg.slogdet(So)
                sol = np.linalg.solve(So, xi)
                ll[c,d,i] = -0.5*(o.sum()*log2pi + logdet + xi @ sol)
    return ll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=0, help="subsample N (0 = full 9013)")
    ap.add_argument("--draws", type=int, default=600)
    ap.add_argument("--tune", type=int, default=600)
    ap.add_argument("--chains", type=int, default=2)
    a = ap.parse_args()
    if a.smoke:
        a.n, a.draws, a.tune = 500, 100, 150
    seed = 20260703
    nsub = a.n if a.n > 0 else None

    print(f"Tier B: axis vs correlate | N={'full 9013' if nsub is None else nsub} "
          f"| draws={a.draws} tune={a.tune} chains={a.chains}\n", flush=True)

    # Both models fit at Phi=I (simple structure): the Tier B question is whether biology
    # needs its OWN loading axis, not whether specifics inter-correlate. Phi=I makes this a
    # clean nested comparison (correlate = axis with the 2 biology factors' loadings folded
    # onto G) and both models get identical Phi treatment.
    base = cc.prepare(cc.S1_FACTORS, correlated=False, windows=False, g_correlated=False,
                      balanced=(nsub is not None), n_subsample=nsub, seed=seed)
    print(f"AXIS prep: {base.M.shape[0]} patients x {base.M.shape[1]} indicators, "
          f"factors={base.factor_cols}", flush=True)
    corr = build_correlate_prep(base)
    print(f"CORR prep: factors={corr.factor_cols} "
          f"(biology indicators reassigned to G)\n", flush=True)

    import arviz as az
    res = {}
    idatas = {}
    for name, prep in [("axis", base), ("correlate", corr)]:
        idata, model = fit(prep, draws=a.draws, tune=a.tune, chains=a.chains, seed=seed, label=name)
        ll = pointwise_ll(idata, model, prep)          # [chain, draw, N] float32
        import xarray as xr
        C_, D_, N_ = ll.shape
        ll_ds = xr.Dataset(
            {"obs": (("chain", "draw", "obs_id"), ll)},
            coords={"chain": np.arange(C_), "draw": np.arange(D_), "obs_id": np.arange(N_)},
        )
        idata.add_groups(log_likelihood=ll_ds)
        idatas[name] = idata
        loo = az.loo(idata)
        res[name] = {"elpd_loo": float(loo.elpd_loo), "p_loo": float(loo.p_loo),
                     "se": float(loo.se), "n_factors": len(prep.factor_cols)}
        print(f"  [{name}] elpd_loo={loo.elpd_loo:.1f} (SE {loo.se:.1f}) p_loo={loo.p_loo:.1f}", flush=True)
        idata.to_netcdf(str(OUT / f"{name}_n{prep.M.shape[0]}.nc"))

    cmp = az.compare({"axis": idatas["axis"], "correlate": idatas["correlate"]}, ic="loo")
    cmp.to_csv(OUT / "compare.csv")
    diff = res["axis"]["elpd_loo"] - res["correlate"]["elpd_loo"]
    res["elpd_diff_axis_minus_correlate"] = diff
    (OUT / "summary.json").write_text(json.dumps(res, indent=2))
    print("\n=== az.compare ===\n", cmp[["rank","elpd_loo","elpd_diff","dse","weight"]].to_string(), flush=True)
    print(f"\nΔELPD (axis - correlate) = {diff:+.1f}", flush=True)
    print(f"written -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
