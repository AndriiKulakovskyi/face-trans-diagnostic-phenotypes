#!/usr/bin/env python3
"""04 — fit a measurement-model stage (S1: continuous-core bifactor, full-N V0).

    python3 scripts/04_fit.py --stage 1 --gpu            # full sample, NumPyro/JAX (RTX 4090)
    python3 scripts/04_fit.py --stage 1                  # full sample, CPU NUTS (slow)
    python3 scripts/04_fit.py --stage 1 --subsample 800  # quick smoke

Reads data/processed/baseline_v0.parquet + configs/prior_loading_matrix_v3.csv (via the engine
src/face/models/bayesian/continuous_core), fits the explicit-latent bifactor continuous core,
certifies (R-hat / ESS / divergences / Heywood), and writes:
  reports/04_stage{S}_report.md      certification + loadings summary (shareable aggregate)
  reports/04_stage{S}_loadings.csv   per-(item, factor) posterior loading (aggregate)
  results/face/stage{S}/{factor_scores.csv, diagnostics.json, idata.nc}   (per-patient -> gitignored)
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

from face.models.bayesian.continuous_core import (  # noqa: E402
    S1_FACTORS, build_marginalized, build_model, prepare)

REPORTS = REPO / "reports"
GATES = dict(rhat=1.01, ess=400.0, div=0, heywood=2.5)


def run(stage: int = 1, subsample: int | None = None, marginalized: bool = True,
        gpu: bool = False, draws: int = 1000, tune: int = 1000, chains: int = 4) -> dict:
    import arviz as az
    import pymc as pm

    prep = prepare(S1_FACTORS, n_subsample=subsample)
    factors = ["overall_severity"] + prep.spec_factors
    N, J = prep.M.shape
    mode = "marginalized (Woodbury)" if marginalized else "explicit-latent"
    print(f"Stage {stage} [{mode}]: N={N} J={J} factors={factors} | "
          f"G-anchors {len(prep.g_anchor_items)} · specific {len(prep.spec_items)} · "
          f"obs cells {int((~np.isnan(prep.M)).sum()):,}")
    model = build_marginalized(prep) if marginalized else build_model(prep)
    use_numpyro = marginalized or gpu          # marginalized needs the JAX backend (batched k×k linalg)
    with model:
        if use_numpyro:
            idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=0.95,
                              random_seed=20260605, nuts_sampler="numpyro",
                              idata_kwargs={"log_likelihood": False})
        else:
            idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1, target_accept=0.95,
                              random_seed=20260605, progressbar=False, init="jitter+adapt_diag",
                              idata_kwargs={"log_likelihood": False})
    return _diagnose_write(stage, prep, idata, az, marginalized)


def _diagnose_write(stage, prep, idata, az, marginalized=True) -> dict:
    out = REPO / "results" / "face" / f"stage{stage}"
    out.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    post = idata.posterior
    lamG = post["lamG_full"].mean(("chain", "draw")).values
    lamS = post["lamS_full"].mean(("chain", "draw")).values
    items, spec = prep.items, prep.spec_factors

    rows = [dict(item=items[j], factor="overall_severity", loading=round(float(lamG[j]), 3),
                 kind="g_anchor") for j in prep.g_anchor_items]
    for j in prep.spec_items:
        k = prep.spec_home[j]
        rows.append(dict(item=items[j], factor=spec[k], loading=round(float(lamS[j, k]), 3),
                         kind="primary"))
        rows.append(dict(item=items[j], factor="overall_severity",
                         loading=round(float(lamG[j]), 3), kind="bifactor_G"))
    load = pd.DataFrame(rows)
    load.to_csv(REPORTS / f"04_stage{stage}_loadings.csv", index=False)

    summ = az.summary(idata, var_names=[v for v in ["lamG_anchor", "lamG_spec", "lamS_home", "sigma"]
                                        if v in post])
    rc = "r_hat" if "r_hat" in summ.columns else "rhat"
    ec = next((c for c in summ.columns if c.startswith("ess")), None)
    rhat = float(pd.to_numeric(summ[rc], errors="coerce").max())
    ess = float(pd.to_numeric(summ[ec], errors="coerce").min()) if ec else float("nan")
    div = int(np.asarray(idata.sample_stats["diverging"]).sum())
    heywood = bool((load.loading.abs() > GATES["heywood"]).any())
    cert = (round(rhat, 3) <= GATES["rhat"] and (np.isnan(ess) or ess >= GATES["ess"])
            and div <= GATES["div"] and not heywood)
    diag = dict(stage=stage, N=int(prep.M.shape[0]), J=int(prep.M.shape[1]),
                factors=["overall_severity"] + spec, rhat_max=round(rhat, 4),
                ess_min=round(ess, 1), divergences=div, heywood=heywood, certified=cert)
    (out / "diagnostics.json").write_text(json.dumps(diag, indent=2))

    # per-patient factor scores: explicit-latent only. The marginalized model integrates the
    # latents out, so scores are computed post-hoc later (regression/Thomson) for stratification.
    if "G" in post:
        sc = pd.DataFrame({"cohort": prep.cohort, "G": post["G"].mean(("chain", "draw")).values})
        Dm = post["D"].mean(("chain", "draw")).values
        for i, f in enumerate(spec):
            sc[f] = Dm[:, i]
        sc.to_csv(out / "factor_scores.csv", index=False)
    try:
        idata.to_netcdf(str(out / "idata.nc"))
    except Exception:
        pass

    gA = load[load.kind == "g_anchor"].sort_values("loading", ascending=False)
    md = [f"# Stage {stage} — continuous-core bifactor (full-N V0)", "",
          f"N={diag['N']:,} · J={diag['J']} · factors={diag['factors']}. "
          f"{'Marginalized (Woodbury)' if marginalized else 'Explicit-latent'}, "
          "observed-cell Gaussian likelihood, no imputation.", "",
          f"## Certification — {'**CERTIFIED**' if cert else '**NOT certified — provisional**'}",
          f"- max R-hat **{rhat:.3f}** · min ESS {ess:.0f} · divergences {div} · Heywood {heywood} "
          f"(gates: R-hat≤{GATES['rhat']}, ESS≥{int(GATES['ess'])}, div=0)", "",
          "## G (functional burden) — anchor loadings", gA[["item", "loading"]].to_markdown(index=False), "",
          "## Specific factors — mean primary home loading",
          load[load.kind == "primary"].groupby("factor")["loading"].mean().round(2).to_markdown(), "",
          "## Bifactor — mean |loading on G| of each specific factor's items (G ⊥ biology check)",
          load[load.kind == "bifactor_G"].assign(
              f=load[load.kind == "primary"].set_index("item").reindex(
                  load[load.kind == "bifactor_G"].item)["factor"].values)
          .groupby("f")["loading"].apply(lambda s: round(float(s.abs().mean()), 2)).to_markdown(), "",
          "Artifacts: `reports/04_stage{S}_loadings.csv` · `results/face/stage{S}/` (per-patient, gitignored)."]
    (REPORTS / f"04_stage{stage}_report.md").write_text("\n".join(md))
    print("\n".join(md[:7]))
    print(f"\nwrote reports/04_stage{stage}_report.md + loadings.csv ; results/face/stage{stage}/")
    return diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--explicit", action="store_true",
                    help="explicit-latent engine (default: marginalized Woodbury via NumPyro)")
    ap.add_argument("--gpu", action="store_true", help="route NUTS through NumPyro/JAX (CUDA box)")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    a = ap.parse_args()
    run(a.stage, a.subsample, marginalized=not a.explicit, gpu=a.gpu,
        draws=a.draws, tune=a.tune, chains=a.chains)


if __name__ == "__main__":
    main()
