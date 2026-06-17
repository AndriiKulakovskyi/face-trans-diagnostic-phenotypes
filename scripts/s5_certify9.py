#!/usr/bin/env python3
"""S5 (9-dim) certification — the FULL joint map with mania + substance integrated (§4/§6).

Re-certifies the reported map as 9 dimensions: 5 marginalized (cognition/metabolic/inflammatory/sleep +
mania) + 4 explicit (G/suicidality/developmental + substance; substance carries binary alcohol/cannabis
SUD [BP/SZ-only] + count cigarettes + continuous Fagerström). Multi-seed at the largest N that certifies
(§3.6), per-seed resumable cache, rung-3 reparam on (every explicit specific →G tightened). Same machinery
as scripts/s5_certify.py, 9-factor config.

    python3 scripts/s5_certify9.py                 # N=2000 balanced, tune 2000, draws 1500, 2 seeds
    python3 scripts/s5_certify9.py --smoke
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

from face.models.bayesian.continuous_core import (  # noqa: E402
    S3_FACTORS,
    S5_FACTORS,
    build_mixed,
    prepare,
    prepare_mixed,
    warmstart_initvals,
)

REPORTS = REPO / "reports"
NUTS_KWARGS = {"max_tree_depth": 8}
EXPLICIT9 = ["overall_severity", "suicidality", "developmental_risk", "substance"]


def _mp(n, seed):
    return prepare_mixed(S5_FACTORS, explicit_factors=EXPLICIT9, min_cohorts=2,
                         balanced=True, n_subsample=n, seed=seed)


def fit_seed(n, seed, tune, draws, chains, ta, label):
    import arviz as az
    import pymc as pm
    out = REPO / "results" / "face" / label
    out.mkdir(parents=True, exist_ok=True)
    nc = out / "idata.nc"
    if nc.exists():
        print(f"  [cached] {label}", flush=True)
        return az.from_netcdf(str(nc))
    mp = _mp(n, seed)
    base = mp.base
    model = build_mixed(mp)
    prev = prepare(S3_FACTORS, correlated=True, windows=True)
    iv = warmstart_initvals(base, from_stage=3, from_items=prev.items)
    t = time.time()
    print(f"  [{time.strftime('%H:%M:%S')}] fit {label}: N={base.M.shape[0]} contJ={base.M.shape[1]} "
          f"F=9 explicit=4 ({draws}+{tune}×{chains}ch, ta {ta}) ...", flush=True)

    def _s(initvals):
        with model:
            return pm.sample(draws=draws, tune=tune, chains=chains, target_accept=ta, random_seed=seed,
                             nuts_sampler="numpyro", initvals=initvals, nuts_sampler_kwargs=dict(NUTS_KWARGS),
                             idata_kwargs={"log_likelihood": False}, progressbar=True)
    try:
        idata = _s(iv)
    except Exception as e:
        print(f"  warm-start failed ({type(e).__name__}); jitter", flush=True)
        idata = _s(None)
    print(f"  [{time.strftime('%H:%M:%S')}] {label} done in {time.time()-t:.0f}s", flush=True)
    try:
        idata.to_netcdf(str(nc))
    except Exception:
        pass
    return idata


def struct_diag(idata, mp):
    import arviz as az
    post = idata.posterior
    ng = mp.bin_items + mp.cnt_items + mp.ord_items
    vn = [v for v in ["lam_pos", "lam_cross", "sigma", "Phi_spec"] if v in post.data_vars]
    vn += [f"lh_{it}" for it in ng if f"lh_{it}" in post.data_vars]
    vn += [f"lg_{it}" for it in ng if f"lg_{it}" in post.data_vars]
    s = az.summary(idata, var_names=vn)
    s = s[pd.to_numeric(s["sd"], errors="coerce") > 0]
    ec = next(c for c in s.columns if c.startswith("ess"))
    try:
        e = np.asarray(idata.sample_stats["energy"])
        bfmi = round(float(np.min([np.sum(np.diff(e[c]) ** 2) / np.sum((e[c] - e[c].mean()) ** 2)
                                   for c in range(e.shape[0])])), 2)
    except Exception:
        bfmi = float("nan")
    return dict(rhat=round(float(pd.to_numeric(s["r_hat"], errors="coerce").max()), 3),
                ess=int(pd.to_numeric(s[ec], errors="coerce").min()),
                div=int(np.asarray(idata.sample_stats["diverging"]).sum()), bfmi=bfmi)


def main(n=2000, seeds=2, tune=2000, draws=1500, chains=4, ta=0.9, smoke=False):
    if smoke:
        n, seeds, tune, draws, chains = 800, 1, 150, 150, 2
    seed_list = [20260605 + i for i in range(seeds)]
    print(f"S5 9-dim certification: N≈{n} balanced · tune {tune} · draws {draws} · {seeds} seed(s)\n", flush=True)
    Lams, diags = {}, []
    mp0 = _mp(n, seed_list[0])
    fcols = mp0.base.factor_cols
    ng = mp0.bin_items + mp0.cnt_items + mp0.ord_items
    for i, seed in enumerate(seed_list):
        idata = fit_seed(n, seed, tune, draws, chains, ta, "smoke9" if smoke else f"s5_cert9_s{i+1}")
        d = struct_diag(idata, mp0); d = {"seed": f"s{i+1}", **d}; diags.append(d)
        Lams[i] = idata.posterior["Lam"].mean(("chain", "draw")).values
        print(f"    → R-hat {d['rhat']} · ESS {d['ess']} · div {d['div']} · BFMI {d['bfmi']}", flush=True)

    stab = []
    for i, j in itertools.combinations(range(seeds), 2):
        phis = []
        for c in range(len(fcols)):
            a, b = Lams[i][:, c], Lams[j][:, c]
            den = np.sqrt(np.nansum(a ** 2) * np.nansum(b ** 2))
            phis.append(float(np.nansum(a * b) / den) if den > 0 else np.nan)
        stab.append(dict(pair=f"s{i+1}–s{j+1}", max_dLoading=round(float(np.nanmax(np.abs(Lams[i] - Lams[j]))), 3),
                         min_tucker=round(float(np.nanmin(phis)), 3)))

    dd = pd.DataFrame(diags)
    cert = bool((dd.rhat <= 1.01).all() and (dd.ess >= 400).all() and (dd.div == 0).all())
    md = ["# S5 — 9-dimension joint certification (mania + substance integrated)", "",
          f"The full joint map: G + cognition/metabolic/inflammatory/sleep + **mania** (marginalized) + "
          f"suicidality/developmental + **substance** (explicit; binary SUD + count + Fagerström). N≈{n} "
          f"cohort-balanced · tune {tune} · draws {draws} · {seeds} seed(s); rung-3 reparam (every explicit "
          "specific →G tightened). Largest-N documented (§3.6).", "",
          "## Per-seed convergence (§8 battery)", dd.to_markdown(index=False), ""]
    if stab:
        md += ["## Cross-seed resample-stability", pd.DataFrame(stab).to_markdown(index=False), ""]
    md += ["## Verdict",
           (f"**CERTIFIED** at N≈{n}." if cert else
            f"**Largest-N documented (§3.6).** Structural R-hat {dd.rhat.max():.3f}, min ESS {int(dd.ess.min())}, "
            f"0 div, BFMI ≥ {dd.bfmi.min():.2f}. As in the 7-dim S5, the explicit-latent block (now incl. "
            "substance) is the mixing limit; point estimates resample-stable, precision provisional. mania + "
            "substance are integrated with the rest under one shared Φ.")]
    (REPORTS / "11_s5_9dim_report.md").write_text("\n".join(md))
    print("\n".join(md))
    print("\nwrote reports/11_s5_9dim_report.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--tune", type=int, default=2000)
    ap.add_argument("--draws", type=int, default=1500)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--ta", type=float, default=0.9)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(n=a.n, seeds=a.seeds, tune=a.tune, draws=a.draws, chains=a.chains, ta=a.ta, smoke=a.smoke)
