"""Shared sampling runner — one canonical, progress-tracked fit helper for the post-S2 scripts
(05 confirm · 06 invariance · 07 score · 08 robustness). Consolidates the NUTS settings and adds
per-fit timestamps + live progress + a quick diagnostic read-out, so every sub-task is observable.

Working pattern (methods §3.6): fit on a random, **cohort-balanced** subsample (N≈2,000), repeat over
2–3 seeds, and report cross-seed stability — fast and observable, with the full-N reported map reserved
for the certified stages. The marginalized (Woodbury) fits are quick at this scale (seconds to ~1 min).
"""
from __future__ import annotations

import time

import numpy as np

NUTS_KWARGS = {"max_tree_depth": 8}


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def sample_marginalized(prep, *, draws: int = 600, tune: int = 600, chains: int = 2,
                        seed: int = 20260605, target_accept: float = 0.9, label: str = "",
                        step: str = "", progressbar: bool = True, weights=None):
    """Fit the marginalized continuous-core model for `prep`, with progress + timing. `step` is an
    optional "[k/N]" prefix for multi-sub-task runs. `weights` → per-patient likelihood weights
    (§3.6 cohort-weighted fit). Returns the InferenceData."""
    import pymc as pm

    from face.measurement.kernel import build_marginalized
    N, J = prep.M.shape
    F = len(prep.factor_cols)
    t = time.time()
    print(f"  {step}[{_ts()}] fit {label}: N={N} J={J} F={F} ({draws}+{tune}×{chains}ch) ...", flush=True)
    model = build_marginalized(prep, weights=weights)
    with model:
        # dict(NUTS_KWARGS): the numpyro path mutates nuts_sampler_kwargs in place (injects
        # target_accept) → a shared dict trips "target_accept defined twice" on the next call.
        idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept,
                          random_seed=seed, nuts_sampler="numpyro", nuts_sampler_kwargs=dict(NUTS_KWARGS),
                          idata_kwargs={"log_likelihood": False}, progressbar=progressbar)
    d = quick_diag(idata)
    print(f"  {step}[{_ts()}] {label} done in {time.time()-t:.0f}s "
          f"(R-hat {d['rhat']:.3f} · ESS {d['ess']:.0f} · div {d['div']})", flush=True)
    return idata


def quick_diag(idata) -> dict:
    """Max R-hat / min ESS / divergences over the structural loading+Φ+residual params."""
    import arviz as az
    post = idata.posterior
    vnames = [v for v in ["lam_pos", "lam_cross", "sigma", "Phi_spec"] if v in post.data_vars]
    summ = az.summary(idata, var_names=vnames)
    if "sd" in summ.columns:
        import pandas as pd
        summ = summ[pd.to_numeric(summ["sd"], errors="coerce") > 0]
    rc = "r_hat" if "r_hat" in summ.columns else "rhat"
    ec = next((c for c in summ.columns if c.startswith("ess")), None)
    import pandas as pd
    rhat = float(pd.to_numeric(summ[rc], errors="coerce").max())
    ess = float(pd.to_numeric(summ[ec], errors="coerce").min()) if ec else float("nan")
    div = int(np.asarray(idata.sample_stats["diverging"]).sum())
    return dict(rhat=rhat, ess=ess, div=div)
