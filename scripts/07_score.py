#!/usr/bin/env python3
"""07 — per-patient dimension scoring at scale (§7).

Projects every patient's OBSERVED cells onto the fitted map (fit-once-score-all, §3.6), with full
uncertainty + reliability flags. Two sources:

  · continuous core (G, cognition, metabolic, inflammatory, sleep) — draw-wise analytic
    conditional-Gaussian factor scores from the certified full-N **S2** posterior, for ALL 9,013
    patients (src/face/scoring.conditional_gaussian_scores).
  · suicidality, developmental — explicit-latent f_e draws from the certified **S5** mixed fit
    (scored on the S5 subsample; full-N projection of the non-Gaussian block is a documented follow-on).

Each (patient, dimension): posterior mean · SD · HDI · #observed home indicators · reliability tier
(well / partial / prior-dominated) · orientation higher = more burden (sign-anchored loadings).

    python3 scripts/07_score.py

Writes results/face/patient_scores.parquet (per-patient -> gitignored) + reports/07_scoring_report.md.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

from face.models.bayesian.continuous_core import S5_FACTORS, prepare, prepare_mixed  # noqa: E402
from face.scoring import conditional_gaussian_scores, reliability_flags  # noqa: E402

REPORTS = REPO / "reports"
PROC = REPO / "data" / "processed"
HDI = 0.94


def main():
    import arviz as az
    # ---- continuous core: 5 factors, ALL patients, from certified S2 ----
    # continuous-anchored dimensions (6) scored from the certified 9-dim joint loadings; the explicit
    # non-Gaussian dimensions (suicidality/developmental/substance) come from f_e below.
    CONT = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep", "mania_activation"]
    EXPLICIT9 = ["overall_severity", "suicidality", "developmental_risk", "substance"]
    prep = prepare(S5_FACTORS, correlated=True, windows=True)
    post = az.from_netcdf(REPO / "results/face/s5_cert9_s1/idata.nc").posterior
    fcols = prep.factor_cols
    print(f"scoring continuous-anchored ({len(CONT)}): N={prep.M.shape[0]} · {CONT}", flush=True)
    sc = conditional_gaussian_scores(prep.M, post, fcols, hdi_prob=HDI)
    nobs, tier = reliability_flags(prep.M, prep.items, prep.home, fcols)
    col = {f: i for i, f in enumerate(fcols)}

    df = pd.DataFrame(index=prep.index)
    for f in CONT:
        c = col[f]
        df[f"{f}__mean"] = sc["mean"][:, c].round(3)
        df[f"{f}__sd"] = sc["sd"][:, c].round(3)
        df[f"{f}__hdi_lo"] = sc["hdi_low"][:, c].round(3)
        df[f"{f}__hdi_hi"] = sc["hdi_high"][:, c].round(3)
        df[f"{f}__n_obs"] = nobs[:, c]
        df[f"{f}__reliability"] = tier[:, c]

    # ---- suicidality + developmental + substance: explicit f_e from the certified 9-dim S5 (subsample) ----
    s5 = REPO / "results/face/s5_cert9_s1/idata.nc"
    if s5.exists():
        mp = prepare_mixed(S5_FACTORS, explicit_factors=EXPLICIT9, min_cohorts=2,
                           balanced=True, n_subsample=2000, seed=20260605)
        fe = np.asarray(az.from_netcdf(s5).posterior["f_e"].values)            # [c,d,Nsub,4]
        fe = fe.reshape((-1,) + fe.shape[2:])                                  # [S,Nsub,4]
        sub = pd.DataFrame(index=mp.base.index)
        for k, name in [(1, "suicidality"), (2, "developmental_risk"), (3, "substance")]:
            sub[f"{name}__mean"] = fe[:, :, k].mean(0).round(3)
            sub[f"{name}__sd"] = fe[:, :, k].std(0).round(3)
            lo, hi = np.quantile(fe[:, :, k], [(1 - HDI) / 2, 1 - (1 - HDI) / 2], axis=0)
            sub[f"{name}__hdi_lo"] = lo.round(3); sub[f"{name}__hdi_hi"] = hi.round(3)
        df = df.join(sub, how="left")
        n_sub = len(sub)
    else:
        n_sub = 0

    out = REPO / "results" / "face" / "patient_scores.parquet"
    df.reset_index().to_parquet(out)

    # ---- aggregate report (no per-patient values) ----
    rel = pd.DataFrame({f: pd.Series(tier[:, col[f]]).value_counts() for f in CONT}).T.fillna(0).astype(int)
    md = ["# 07 — per-patient dimension scoring (§7), 9-dim joint map", "",
          f"Per-patient coordinates with uncertainty for **{len(df):,} patients**, fit-once-score-all "
          "(§3.6). Six continuous-anchored dimensions (G + cognition/metabolic/inflammatory/sleep/**mania**) "
          "via draw-wise analytic conditional-Gaussian scores from the **certified 9-dim joint** loadings; "
          f"three explicit (suicidality/developmental/**substance**) via f_e from the same fit (subsample "
          f"n={n_sub:,}). Orientation: higher = more burden. Each dimension carries mean · SD · HDI · "
          "#observed home indicators · reliability tier.", "",
          "## Reliability — patients per tier, by continuous-anchored dimension",
          "(well = ≥3 observed home indicators · partial = 1–2 · prior-dominated = 0)",
          rel.to_markdown(), "",
          "## Dimension summary (posterior-mean scores, z-scored, higher = more burden)",
          pd.DataFrame({f: {"mean": round(float(np.nanmean(sc['mean'][:, col[f]])), 2),
                            "sd_across_patients": round(float(np.nanstd(sc['mean'][:, col[f]])), 2),
                            "mean_posterior_SD": round(float(np.nanmean(sc['sd'][:, col[f]])), 2)}
                        for f in CONT}).T.to_markdown(), "",
          "## Notes",
          "- A patient with few observed indicators for a dimension gets a **prior-dominated** flag and a "
          "wider posterior SD — downstream strata (M2) must propagate this uncertainty, not treat all "
          "coordinates as equally characterised.",
          "- **Suicidality/developmental/substance are scored on the S5 subsample** (their explicit f_e); "
          "full-N projection of the non-Gaussian block (a logistic/count projection, not Gaussian) is a "
          "documented follow-on for M2.",
          "", "Artifacts: `results/face/patient_scores.parquet` (per-patient, gitignored)."]
    (REPORTS / "07_scoring_report.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote results/face/patient_scores.parquet ({df.shape}) + reports/07_scoring_report.md")


if __name__ == "__main__":
    main()
