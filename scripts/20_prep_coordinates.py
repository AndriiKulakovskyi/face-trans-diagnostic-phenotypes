#!/usr/bin/env python3
"""20 — M2.0 prep: full-N 9-dim coordinates (+ uncertainty) + validation table.

The M2 stratification layer needs **all nine** M1 dimensions for **all 9,013** patients, with
uncertainty. M1 left three explicit (non-Gaussian) axes — suicidality / developmental_risk /
substance — scored only on the ~1,884-patient fit subsample. This step closes that gap and assembles
the M2 input (methods of record: docs/STRATIFICATION_MODEL.md §6 M2.0):

  1. continuous-anchored axes (overall_severity / cognition / metabolic / inflammatory / sleep /
     mania) — full-N analytic conditional-Gaussian scores from the certified 9-dim loadings, with
     draw-wise samples (face.strata.scoring.conditional_gaussian_draws).
  2. explicit axes (suicidality / developmental_risk / substance) — full-N PROJECTION: hold the
     certified measurement parameters fixed and sample each patient's f_e from their observed
     non-Gaussian cells (face.strata.scoring.project_explicit_full_n). No re-fit, no imputation.
  3. a per-patient posterior-DRAWS export (both blocks) — the uncertainty-faithfulness arm for the
     measurement-error mixture (full S_i) and archetypes-over-draws.
  4. a validation table (cohort / arm / age / sex / education / site) — validation-only, never a
     clustering input.

QC: the projection must reproduce the certified f_e on the fit subsample (Pearson r per axis).

    python3 scripts/20_prep_coordinates.py

Writes results/face/m2/{coordinates_full.parquet, coordinates_draws.npz, validation_table.parquet,
proj.npz, DONE} + reports/20_prep_coordinates.md + docs/figures/20_coverage.png. (results/face is gitignored.)
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

REPORTS = REPO / "reports"
FIGS = REPO / "docs" / "figures"
OUT = REPO / "results" / "face" / "m2"
OUT.mkdir(parents=True, exist_ok=True)
HDI = 0.94

CANON = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep", "mania_activation",
         "suicidality", "developmental_risk", "substance"]
CONT6 = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep", "mania_activation"]
EXPL3 = ["suicidality", "developmental_risk", "substance"]
EXPLICIT9 = ["overall_severity", "suicidality", "developmental_risk", "substance"]

NDRAW_CONT = 200          # continuous conditional-Gaussian samples per patient
DRAWS, TUNE, CHAINS = 400, 500, 2   # explicit projection NUTS
NKEEP = 200               # draws kept in the export
SEED = 20260609


def _tier(n):
    return np.where(n >= 3, "well", np.where(n >= 1, "partial", "prior-dominated"))


def main():
    import arviz as az
    from scipy.stats import norm

    from face.models.bayesian.continuous_core import PROC, S5_FACTORS, prepare, prepare_mixed
    from face.scoring import reliability_flags
    from face.strata.scoring import (
        align_ordinals_to_fit,
        coherent_joint_coords,
        explicit_nobs,
        project_explicit_full_n,
    )

    t0 = time.time()
    idata = az.from_netcdf(REPO / "results/face/s5_cert9_s1/idata.nc")
    post = idata.posterior

    # ---------- 1) reliability tiers (observed home-indicator counts; coords scored coherently in 3) ----------
    print("[1/4] reliability tiers (observed home-indicator counts)...", flush=True)
    prep = prepare(S5_FACTORS, correlated=True, windows=True)
    fcols = prep.factor_cols                                   # 9 factors (prep order)
    cidx = {f: i for i, f in enumerate(fcols)}
    base_index = prep.index                                    # MultiIndex (cohort, patient_id)
    nobs_c, tier_c = reliability_flags(prep.M, prep.items, prep.home, fcols)
    N = prep.M.shape[0]
    assert list(base_index) == list(prep.index)

    # ---------- 2) explicit block: 3 dims, full-N projection ----------
    print("[2/4] explicit full-N projection (fixed certified params)...", flush=True)
    mp = prepare_mixed(S5_FACTORS, explicit_factors=EXPLICIT9, min_cohorts=2,
                       balanced=False, n_subsample=None)
    mpC = prepare_mixed(S5_FACTORS, explicit_factors=EXPLICIT9, min_cohorts=2,
                        balanced=True, n_subsample=2000, seed=20260605)   # the certified subsample
    B = pd.read_parquet(PROC / "baseline_v0.parquet")
    clipped = align_ordinals_to_fit(mp, mpC.base.index, B)     # match certified ordinal coding
    cert_K = {it: int(post[f"c_{it}"].shape[-1]) + 1 for it in mp.ord_items}
    for it, K in zip(mp.ord_items, mp.ord_K, strict=False):
        assert K == cert_K[it], f"ord K mismatch {it}: {K} vs certified {cert_K[it]}"

    proj_cache = OUT / "proj.npz"
    if proj_cache.exists():                                    # resume: skip the expensive re-sample
        d = np.load(proj_cache, allow_pickle=True)
        res = {"mean": d["mean"], "sd": d["sd"], "draws": d["draws"], "fcols": list(d["fcols"]),
               "diag": {"max_rhat": float(d["max_rhat"]), "divergences": int(d["divergences"]),
                        "n_draws": int(d["n_draws"])}}
        print(f"[2/4] loaded cached projection {res['mean'].shape}", flush=True)
    else:
        res = project_explicit_full_n(mp, idata, draws=DRAWS, tune=TUNE, chains=CHAINS, seed=SEED)
        step = max(1, res["draws"].shape[0] // NKEEP)
        res["draws"] = res["draws"][::step][:NKEEP]            # thin once; downstream uses as-is
        np.savez_compressed(proj_cache, mean=res["mean"], sd=res["sd"], draws=res["draws"],
                            fcols=np.array(res["fcols"]), index=mp.base.index.to_frame().to_numpy(),
                            max_rhat=res["diag"]["max_rhat"], divergences=res["diag"]["divergences"],
                            n_draws=res["diag"]["n_draws"])
    ecols = res["fcols"]                                       # [overall_severity, suic, dev, substance]
    en = explicit_nobs(mp)
    ne = pd.DataFrame(en["n_obs"], index=mp.base.index, columns=en["fcols"]).reindex(base_index)

    # QC: projection reproduces the certified f_e on the fit subsample
    fe_cert = np.asarray(post["f_e"].mean(("chain", "draw")).values)      # [1884, 4] over mpC index
    pos = pd.Index(mp.base.index).get_indexer(pd.Index(mpC.base.index))
    repro = {name: float(np.corrcoef(res["mean"][pos, k], fe_cert[:, k])[0, 1])
             for k, name in enumerate(ecols) if name in EXPL3 or name == "overall_severity"}

    # ---------- 3) coherent joint coordinates (one model state per draw) + full S_i (P2-01/02/04) ----------
    print("[3/4] coherent joint scoring (explicit-block G; f_m|f_e under shared Phi)...", flush=True)
    ch = coherent_joint_coords(mp, idata, projection=res, n_draws=NKEEP)
    cmap = {f: ch["cols"].index(f) for f in CANON}            # CANON dim -> coherent column
    pos = pd.Index(mp.base.index).get_indexer(pd.Index(base_index))
    assert (pos >= 0).all(), "coherent coords missing some base_index rows"

    z = float(norm.ppf(1 - (1 - HDI) / 2))
    df = pd.DataFrame(index=base_index)
    nkeep = ch["draws"].shape[0]
    draws = np.full((nkeep, N, len(CANON)), np.nan, dtype="float32")
    cov_full = np.full((N, len(CANON), len(CANON)), np.nan, dtype="float32")
    for di, f in enumerate(CANON):
        ci = cmap[f]
        m, s = ch["mean"][pos, ci], ch["sd"][pos, ci]
        n = nobs_c[:, cidx[f]] if f in CONT6 else ne[f].to_numpy()
        rel = tier_c[:, cidx[f]] if f in CONT6 else _tier(n)
        df[f"{f}__mean"] = np.round(m, 3); df[f"{f}__sd"] = np.round(s, 3)
        df[f"{f}__hdi_lo"] = np.round(m - z * s, 3); df[f"{f}__hdi_hi"] = np.round(m + z * s, 3)
        df[f"{f}__n_obs"] = n; df[f"{f}__reliability"] = rel
        draws[:, :, di] = ch["draws"][:, pos, ci]
    order = [cmap[f] for f in CANON]
    cov_full[:] = ch["cov"][np.ix_(pos, order, order)]        # per-patient S_i in CANON order (the P2-04 arm)
    df.reset_index().to_parquet(OUT / "coordinates_full.parquet")
    coh = np.asarray(base_index.get_level_values("cohort"))
    pid = np.asarray(base_index.get_level_values("patient_id"))
    np.savez_compressed(OUT / "coordinates_draws.npz", draws=draws, dims=np.array(CANON),
                        cohort=coh, patient_id=pid)
    np.savez_compressed(OUT / "coordinates_cov.npz", cov=cov_full, dims=np.array(CANON),
                        cohort=coh, patient_id=pid)

    # ---------- 4) validation table (validation-only) ----------
    print("[4/4] validation table...", flush=True)
    from face.data import build_unified_dataframe
    w = build_unified_dataframe("data", "data/face-common-vars.xlsx", ["READY", "PARTIAL"], format="wide")
    w = w.assign(cohort=w["cohort"].astype(str).str.lower(),
                 patient_id=w["usubjid_patients"].astype(str))
    keep = {"age_V0": "age", "sex_V0": "sex", "education_years_V0": "education_years",
            "siteid_city_V0": "siteid_city", "arm": "arm"}
    vt = (w.set_index(["cohort", "patient_id"])[list(keep)].rename(columns=keep).reindex(base_index))
    vt.reset_index().to_parquet(OUT / "validation_table.parquet")

    # ---------- report + figure ----------
    rel = pd.DataFrame({f: pd.Series(df[f"{f}__reliability"]).value_counts() for f in CANON}).T
    rel = rel.reindex(columns=["well", "partial", "prior-dominated"]).fillna(0).astype(int)
    summ = pd.DataFrame({f: {"mean": round(float(np.nanmean(df[f"{f}__mean"])), 2),
                             "sd_across_patients": round(float(np.nanstd(df[f"{f}__mean"])), 2),
                             "mean_posterior_SD": round(float(np.nanmean(df[f"{f}__sd"])), 2)}
                         for f in CANON}).T

    _figure(rel, summ)
    dt = time.time() - t0
    md = ["# 20 — M2.0 prep: full-N 9-dim coordinates + validation table", "",
          f"All **{N:,}** patients now carry **all 9** dimensions with uncertainty (M1 left "
          "suicidality/developmental/substance on the ~1,884 fit subsample). **Coherent joint scoring** "
          "(P2-01/02/04): every 9D draw comes from ONE model state — the explicit-block latents f_e (incl "
          "the explicit-block G) plus the marginalized specifics f_m conditioned on that same f_e under the "
          "shared Phi; full-N projection under fixed certified parameters (no re-fit, no imputation). "
          "Exports the joint draws AND the full per-patient covariance S_i.", "",
          f"**Projection sampler:** R-hat(z_e) max **{res['diag']['max_rhat']:.3f}** · divergences "
          f"**{res['diag']['divergences']}** · draws {res['diag']['n_draws']}. Runtime {dt/60:.1f} min.", "",
          "## QC — projection reproduces the certified f_e on the fit subsample (Pearson r)",
          pd.Series(repro).round(3).to_frame("r").to_markdown(), "",
          "Ordinal re-coding to the certified categories (top-category absorption): "
          + ", ".join(f"{k}={v}" for k, v in clipped.items()) + " patients re-mapped.", "",
          "## Reliability — patients per tier, by dimension",
          "(well = ≥3 observed home indicators · partial = 1–2 · prior-dominated = 0)",
          rel.to_markdown(), "",
          "## Dimension summary (posterior-mean coordinate, z-scored, higher = more burden)",
          summ.to_markdown(), "",
          "## Validation table (validation-only; never a clustering input)",
          f"- columns: {list(vt.columns)} · rows {len(vt):,}",
          "- coverage: " + ", ".join(f"{c} {int(vt[c].notna().sum())}" for c in vt.columns), "",
          "## Artifacts (results/face/m2/, gitignored)",
          "- `coordinates_full.parquet` — per-patient 9-dim mean/SD/HDI/n_obs/reliability (the M2 input).",
          f"- `coordinates_draws.npz` — [{NKEEP}, {N}, 9] coherent joint posterior draws (archetypes-over-draws / structure gate).",
          f"- `coordinates_cov.npz` — [{N}, 9, 9] full per-patient covariance S_i (coherent; the full-S_i XD arm, P2-04).",
          "- `validation_table.parquet` — cohort/arm/age/sex/education/site.",
          "- `proj.npz` — raw explicit projection (mean/sd/draws).", "",
          "Figure: `docs/figures/20_coverage.png`."]
    (REPORTS / "20_prep_coordinates.md").write_text("\n".join(md))
    (OUT / "DONE").write_text(f"ok {dt/60:.1f}min\n")
    print("\n".join(md))
    print(f"\n[done] {dt/60:.1f} min — wrote coordinates_full.parquet {df.shape}, draws {draws.shape}")


def _figure(rel, summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    dims = list(rel.index)
    btm = np.zeros(len(dims))
    colors = {"well": "#2c7fb8", "partial": "#7fcdbb", "prior-dominated": "#edf8b1"}
    for tier in ["well", "partial", "prior-dominated"]:
        ax[0].bar(dims, rel[tier].values, bottom=btm, label=tier, color=colors[tier])
        btm += rel[tier].values
    ax[0].set_title("Coverage / reliability per dimension (M2.0, N=9,013)")
    ax[0].set_ylabel("patients"); ax[0].legend(fontsize=8)
    ax[0].tick_params(axis="x", rotation=60)
    ax[1].bar(dims, summ["mean_posterior_SD"].values, color="#756bb1")
    ax[1].set_title("Mean per-patient posterior SD per dimension")
    ax[1].set_ylabel("posterior SD (z units)")
    ax[1].tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(FIGS / "20_coverage.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
