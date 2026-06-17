#!/usr/bin/env python3
"""10 — covariate-adjusted biology⊥G sensitivity arm (issue P0-04).

The published measurement equation adjusts each item mean by ``β_jᵀ c_i`` (age, sex, education, site),
but the primary engine implements no covariates — so the headline "metabolic/inflammatory burden is the
least severity-entangled domain" is, as fitted, UNADJUSTED for the obvious confounders of a
biology-vs-severity contrast (age, sex, site). This arm re-derives biology⊥G with the covariates
partialled out of every continuous item (age natural-spline + age×sex + sex + edulevel + site dummies;
Frisch–Waugh–Lovell-equivalent to the published term for Gaussian items), on the same correlated-G
marginalized model as ``scripts/s5_corrg.py``. It reports Φ(G,·) adjusted vs unadjusted: if metabolic
and inflammatory stay the least G-correlated, the headline survives covariate adjustment.

    python3 scripts/10_covariate_sensitivity.py                 # N=2000 balanced, 2 seeds, both arms
    python3 scripts/10_covariate_sensitivity.py --smoke         # fast end-to-end check
    # run detached:  python3 scripts/run_job.py covar_sens -- python -u scripts/10_covariate_sensitivity.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

from face.confirm import corr_no_g_prep  # noqa: E402
from face.io import manifest, progress  # noqa: E402
from face.models.bayesian.continuous_core import S1_FACTORS, prepare  # noqa: E402
from face.runner import quick_diag, sample_marginalized  # noqa: E402

REPORTS = REPO / "reports"
ARMS = [("unadjusted", False), ("adjusted", True)]


def _fit_arm(arm: str, adjust: bool, n: int, seeds: int, draws: int, tune: int, chains: int):
    import arviz as az
    cache = REPO / "results" / "face" / f"covar_{arm}"
    cache.mkdir(parents=True, exist_ok=True)
    phis, diags = {}, []
    for i in range(seeds):
        seed = 20260605 + i
        nc = cache / f"s{i + 1}_n{n}.nc"          # N-aware cache key (smoke N≠ real N must not collide)
        progress.heartbeat(stage=f"{arm} seed {i + 1}/{seeds}", frac=None,
                           msg=f"{arm} arm, seed {i + 1}")
        if nc.exists():
            idata = az.from_netcdf(str(nc))
            print(f"  [cached] {arm} seed {i + 1}", flush=True)
        else:
            base = prepare(S1_FACTORS, correlated=True, windows=False, g_correlated=True,
                           balanced=True, n_subsample=n, seed=seed, covariate_adjust=adjust)
            prep = corr_no_g_prep(base)
            idata = sample_marginalized(prep, draws=draws, tune=tune, chains=chains, seed=seed,
                                        target_accept=0.92, label=f"covar-{arm} s{i + 1}",
                                        step=f"[{arm} {i + 1}/{seeds}] ")
            try:
                idata.to_netcdf(str(nc))
            except Exception:
                pass
            d = quick_diag(idata)
            manifest.write_manifest(f"covar_{arm}_s{i + 1}", out_dir=cache, N=base.M.shape[0],
                                    index=base.index, cohort=base.cohort, seed=seed,
                                    diagnostics={k: float(v) for k, v in d.items()},
                                    extra={"arm": arm, "covariate_adjust": adjust})
        fcols = ["overall_severity"] + [f for f in S1_FACTORS if f != "overall_severity"]
        Phi = idata.posterior["Phi"].mean(("chain", "draw")).values
        g = fcols.index("overall_severity")
        phis[i] = {fcols[c]: float(Phi[g, c]) for c in range(len(fcols)) if c != g}
        d = quick_diag(idata)
        diags.append({"arm": arm, "seed": f"s{i + 1}",
                      **{k: round(v, 3) if k == "rhat" else int(v) for k, v in d.items()}})
        print(f"    → {diags[-1]} · Φ(G,·) {({k: round(v, 3) for k, v in phis[i].items()})}", flush=True)
    return phis, diags


def main(n=2000, seeds=2, draws=600, tune=800, chains=2, smoke=False):
    if smoke:
        n, seeds, draws, tune, chains = 600, 1, 150, 200, 2
    print(f"covariate-adjusted biology⊥G sensitivity: N≈{n} balanced · {seeds} seed(s) · both arms\n",
          flush=True)
    specs = [f for f in S1_FACTORS if f != "overall_severity"]
    arm_phis, all_diags = {}, []
    for arm, adjust in ARMS:
        phis, diags = _fit_arm(arm, adjust, n, seeds, draws, tune, chains)
        arm_phis[arm] = {f: float(np.mean([phis[i][f] for i in range(seeds)])) for f in specs}
        all_diags += diags

    rows = []
    for f in specs:
        u, a = arm_phis["unadjusted"][f], arm_phis["adjusted"][f]
        rows.append(dict(domain=f, phi_G_unadjusted=round(u, 3), phi_G_adjusted=round(a, 3),
                         delta=round(a - u, 3)))
    tab = pd.DataFrame(rows).sort_values("phi_G_adjusted")
    REPORTS.mkdir(parents=True, exist_ok=True)
    tab.to_csv(REPORTS / "10_covariate_sensitivity.csv", index=False)

    bio = tab[tab.domain.isin(["metabolic", "inflammatory"])]
    survives = bool((bio.phi_G_adjusted.abs() < 0.30).all()
                    and bio.phi_G_adjusted.max() < tab[tab.domain.isin(["cognition", "sleep"])]
                    .phi_G_adjusted.min())
    md = ["# 10 — covariate-adjusted biology⊥G sensitivity (P0-04)", "",
          f"Correlated-G marginalized model, N≈{n} balanced, {seeds} seed(s). Each continuous item is "
          "partialled on age(spline)+age×sex+sex+edulevel+site before the factor model (FWL-equivalent to "
          "the published β_jᵀc_i). Φ(G,·) compares the unadjusted vs covariate-adjusted G-correlations.", "",
          "## Convergence", pd.DataFrame(all_diags).to_markdown(index=False), "",
          "## Biology⊥G — unadjusted vs covariate-adjusted", tab.to_markdown(index=False), "",
          ("- **Survives covariate adjustment:** metabolic and inflammatory remain the least "
           "severity-entangled domains after adjusting for age/sex/education/site." if survives else
           "- **⚠ Headline shifts under adjustment** — see the table; revise the biology⊥G claim accordingly."),
          ""]
    (REPORTS / "10_covariate_sensitivity_report.md").write_text("\n".join(md))
    print("\n".join(md))
    print("\nwrote reports/10_covariate_sensitivity_report.md (+ .csv)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--draws", type=int, default=600)
    ap.add_argument("--tune", type=int, default=800)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(n=a.n, seeds=a.seeds, draws=a.draws, tune=a.tune, chains=a.chains, smoke=a.smoke)
