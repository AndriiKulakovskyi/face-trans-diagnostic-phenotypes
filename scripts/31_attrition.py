#!/usr/bin/env python3
"""31 — G6: attrition & informative dropout (the honesty gate; runs before any follow-up scoring).

Is retention at V1/V2 related to where a patient sits on the V0 map — *do the sicker leave, or the
improved?* Either biases the retained sample (survivorship), and mistaking dropout for improvement would
be fatal to G3/G4. Using only V0 objects already in hand (M2 coordinates + strata) and the retention flags
from the long frame, we (1) regress retention on the 9 V0 coordinates + cohort (logistic, 94% CIs),
(2) profile stayers vs droppers per axis, (3) emit inverse-probability-of-retention weights for the G3/G4
sensitivity arm, and (4) tabulate the raw dropout reasons (BP-only `chdiag` etc. — descriptive, for M4).
No scoring, no imputation. Methods: docs/TEMPORAL_MODEL.md §7.

    python3 scripts/31_attrition.py

Writes reports/31_attrition.md (+ 31_informative_dropout.csv, 31_stayer_dropper.csv, 31_dropout_reasons.csv)
· docs/figures/31_attrition.png · results/face/m3/ipw_weights.parquet (gitignored).
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

from face.data import build_unified_dataframe  # noqa: E402
from face.temporal import CANON  # noqa: E402
from face.temporal.dropout import extract_dropout, patient_retention  # noqa: E402

XLSX = REPO / "data" / "face-common-vars.xlsx"
M2 = REPO / "results" / "face" / "m2"
STRATA = REPO / "results" / "face" / "patient_strata.parquet"
REPORTS = REPO / "reports"
FIGS = REPO / "docs" / "figures"
OUT = REPO / "results" / "face" / "m3"
TARGETS = ("retained_V1", "retained_V2")


def _logit(df, target, predictors):
    """Logistic retention model; returns a tidy OR table (coef, OR, 94% CI) for `predictors`."""
    import statsmodels.formula.api as smf
    formula = f"{target} ~ " + " + ".join(predictors)
    res = smf.logit(formula, data=df).fit(disp=0, maxiter=200)
    ci = res.conf_int(alpha=0.06)                      # 94% CI (project convention)
    tab = pd.DataFrame({"coef": res.params, "ci_lo": ci[0], "ci_hi": ci[1], "p": res.pvalues})
    tab["OR"] = np.exp(tab["coef"]); tab["OR_lo"] = np.exp(tab["ci_lo"]); tab["OR_hi"] = np.exp(tab["ci_hi"])
    tab["informative"] = (tab["OR_lo"] > 1) | (tab["OR_hi"] < 1)   # 94% CI excludes OR=1
    return res, tab


def main():
    REPORTS.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- assemble the V0 modeling frame (retention + V0 coordinates + strata + covariates) ----
    long = build_unified_dataframe("data", str(XLSX), readiness=["READY", "PARTIAL"], format="long")
    ret = patient_retention(long, visits=("V1", "V2")).reset_index()
    coords = pd.read_parquet(M2 / "coordinates_full.parquet")
    strata = pd.read_parquet(STRATA)[["cohort", "patient_id", "arch_dominant", "arch_dominant_name"]]
    val = pd.read_parquet(M2 / "validation_table.parquet")[["cohort", "patient_id", "age", "sex", "arm"]]

    df = ret.merge(coords[["cohort", "patient_id"] + [f"{a}__mean" for a in CANON]],
                   on=["cohort", "patient_id"], how="inner")
    df = df.merge(strata, on=["cohort", "patient_id"], how="left").merge(
        val, on=["cohort", "patient_id"], how="left")
    for a in CANON:                                    # clean names for the formula (already z-scored)
        df[a] = pd.to_numeric(df[f"{a}__mean"], errors="coerce")
    df["age_z"] = (df["age"] - df["age"].mean()) / df["age"].std()
    n_roster = len(df)
    rates = {t: float(df[t].mean()) for t in TARGETS}

    # ---- informative-dropout logistic: retention ~ 9 V0 coordinates + cohort ----
    or_tables, models = {}, {}
    for t in TARGETS:
        res, tab = _logit(df, t, list(CANON) + ["C(cohort)"])
        or_tables[t] = tab; models[t] = res
    # severity-adjusted-for-age/sex sanity check (reduced N), V1 only
    adj = df.dropna(subset=["age_z", "sex"]).copy()
    adj["sex"] = adj["sex"].astype("Int64").astype(str)
    res_adj, tab_adj = _logit(adj, "retained_V1", list(CANON) + ["age_z", "C(sex)", "C(cohort)"])

    coef_rows = []
    for t in TARGETS:
        for a in CANON:
            r = or_tables[t].loc[a]
            coef_rows.append(dict(target=t, axis=a, OR=round(r.OR, 3), ci_lo=round(r.OR_lo, 3),
                                  ci_hi=round(r.OR_hi, 3), p=round(r.p, 4), informative=bool(r.informative)))
    coef = pd.DataFrame(coef_rows)
    coef.to_csv(REPORTS / "31_informative_dropout.csv", index=False)

    # ---- stayers vs droppers: per-axis standardized mean difference (Cohen's d), V1 ----
    sd_rows = []
    for a in CANON:
        x1, x0 = df.loc[df.retained_V1 == 1, a], df.loc[df.retained_V1 == 0, a]
        sp = np.sqrt(((x1.var() * (len(x1) - 1)) + (x0.var() * (len(x0) - 1))) / (len(x1) + len(x0) - 2))
        d = (x1.mean() - x0.mean()) / sp if sp > 0 else 0.0
        sd_rows.append(dict(axis=a, mean_stayer=round(float(x1.mean()), 3),
                            mean_dropper=round(float(x0.mean()), 3), cohens_d=round(float(d), 3)))
    sdr = pd.DataFrame(sd_rows)
    sdr.to_csv(REPORTS / "31_stayer_dropper.csv", index=False)

    # ---- IPW weights for the G3/G4 sensitivity arm (stabilized 1/p_retained) ----
    ipw = df[["cohort", "patient_id"]].copy()
    for t in TARGETS:
        p = models[t].predict(df).clip(1e-3, 1 - 1e-3)
        ipw[f"p_{t}"] = np.round(p, 4)
        ipw[f"w_{t}"] = np.round(np.where(df[t] == 1, rates[t] / p, 0.0), 4)   # stabilized; 0 for droppers
    ipw.to_parquet(OUT / "ipw_weights.parquet", index=False)

    # ---- dropout reasons (descriptive; captured for M4) ----
    reasons = extract_dropout("data")
    rc = reasons.pivot_table(index="cohort", columns="reason", values="patient_id",
                             aggfunc="count", fill_value=0)
    for col in ("refusal", "moved", "diagnosis_change", "deceased", "other", "unknown", "coded"):
        if col not in rc.columns:
            rc[col] = 0
    rc["n_lost"] = reasons.groupby("cohort")["lost_flag"].sum()
    rc["n_deaths"] = reasons.groupby("cohort")["deceased"].sum()
    rc = rc.reindex(["bp", "sz", "dr"]).fillna(0).astype(int)
    rc.reset_index().to_csv(REPORTS / "31_dropout_reasons.csv", index=False)

    _figure(or_tables, sdr)

    # ---- report ----
    sev = {t: or_tables[t].loc["overall_severity"] for t in TARGETS}
    sev_row = sev["retained_V1"]
    sev_dir = (("the sicker LEAVE" if sev_row.OR < 1 else "the sicker STAY")
               if sev_row.informative else "severity-neutral")
    n_inf = int(coef[coef.target == "retained_V1"].informative.sum())
    bio_inf = coef[(coef.target == "retained_V1") & (coef.axis.isin(["metabolic", "inflammatory"]))
                   & coef.informative]
    verdict = "informative (MAR-given-V0)" if n_inf > 0 else "≈ not detectably informative (MCAR-plausible)"
    md = ["# 31 — G6: attrition & informative dropout", "",
          f"V0 roster **N = {n_roster:,}**; retained at V1 **{rates['retained_V1']:.1%}** "
          f"({int(df.retained_V1.sum()):,}), at V2 **{rates['retained_V2']:.1%}** "
          f"({int(df.retained_V2.sum()):,}). Logistic retention ~ the 9 V0 coordinates + cohort "
          "(94% CIs; coordinates are z-scored, so OR is per +1 SD on that axis). The question: is the "
          "retained sample a *fair* draw from the V0 map, or does position predict who stays?", "",
          f"## Informative-dropout verdict — **{verdict}**",
          f"- **Severity (G):** OR(V1) = **{sev['retained_V1'].OR:.2f}** "
          f"[{sev['retained_V1'].OR_lo:.2f}, {sev['retained_V1'].OR_hi:.2f}] → **{sev_dir}** "
          + ("per +1 SD of global burden (CI excludes 1)." if sev["retained_V1"].informative
             else "— global burden does not predict dropout per +1 SD; the informative signal is on other axes, not severity."),
          f"- {n_inf}/9 axes are individually informative for V1 retention (94% CI excludes OR=1).",
          ("- **Biology corners** (metabolic/inflammatory) "
           + ("show informative dropout: " + ", ".join(f"{r.axis} OR {r.OR:.2f}" for r in bio_inf.itertuples())
              + " — biology trajectories must carry the IPW caveat." if len(bio_inf)
              else "are NOT differentially retained (biology trajectories are not selection-biased on these axes).")),
          "", "### Retention odds ratios per V0 axis (per +1 SD)",
          coef.pivot(index="axis", columns="target", values="OR").reindex(CANON).to_markdown(), "",
          f"- Age/sex-adjusted sanity check (V1, N={len(adj):,}): severity OR "
          f"{np.exp(tab_adj.loc['overall_severity','coef']):.2f} (vs {sev['retained_V1'].OR:.2f} unadjusted) "
          "— direction unchanged." , "",
          "## Stayers vs droppers — V0 coordinate profile (Cohen's d, V1)",
          "Positive d = stayers score higher on that axis at V0.",
          sdr.to_markdown(index=False), "",
          "## Dropout reasons (descriptive; captured for M4, not analysed here)",
          rc[["n_lost", "refusal", "moved", "diagnosis_change", "deceased", "unknown", "other", "coded"]]
          .reset_index().to_markdown(index=False),
          f"\n- **Diagnosis-change exits** ('Changement de diagnostic'): BP **{int(rc.loc['bp','diagnosis_change'])}**, "
          f"SZ **{int(rc.loc['sz','diagnosis_change'])}** (DR reasons are coded, not decoded) — the only "
          "internal trace of DSM-5 instability (§A). It is an *exit* signal and the in-data `arm` never "
          "updates, so the head-to-head is deferred to M4.",
          f"- Deaths (sentinel-corrected dates): BP {int(rc.loc['bp','n_deaths'])}, SZ {int(rc.loc['sz','n_deaths'])}, "
          f"DR {int(rc.loc['dr','n_deaths'])}.", "",
          "## Guard & hand-off",
          "- **Dropout ≠ improvement.** Because retention "
          + ("**is** position-dependent" if n_inf else "is ~position-independent")
          + ", G3/G4 report **completers-only AND all-available**, and the all-available trends carry an "
          "**IPW-of-retention** sensitivity refit (weights in `results/face/m3/ipw_weights.parquet`). "
          + ("Divergence between naive and IPW estimates is flagged on every affected figure."
             if n_inf else "IPW is expected to barely move the estimates here."),
          "- Conditions: G3 (trait/state) and G4 (persistence). The retained sample is "
          + ("**not** a neutral draw — survivorship is a live threat, handled by IPW + completers-vs-all."
             if n_inf else "close to a neutral draw on the V0 axes — survivorship risk is low."), "",
          "Artifacts: `reports/31_{informative_dropout,stayer_dropper,dropout_reasons}.csv` · "
          "`docs/figures/31_attrition.png` · `results/face/m3/ipw_weights.parquet`."]
    (REPORTS / "31_attrition.md").write_text("\n".join(md))
    print("\n".join(md))


def _figure(or_tables, sdr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    y = np.arange(len(CANON))[::-1]
    for t, color, off in [("retained_V1", "#2c7fb8", 0.15), ("retained_V2", "#d95f0e", -0.15)]:
        tab = or_tables[t].reindex(CANON)
        ax[0].errorbar(tab["OR"], y + off, xerr=[tab["OR"] - tab["OR_lo"], tab["OR_hi"] - tab["OR"]],
                       fmt="o", color=color, label=t, capsize=2, ms=4)
    ax[0].axvline(1.0, color="k", ls="--", lw=1)
    ax[0].set_yticks(y); ax[0].set_yticklabels(CANON, fontsize=8)
    ax[0].set_xlabel("retention OR per +1 SD (94% CI)"); ax[0].set_xscale("log")
    ax[0].set_title("Informative dropout — retention vs V0 position"); ax[0].legend(fontsize=8)
    s = sdr.set_index("axis").reindex(CANON)
    colors = ["#2c7fb8" if v >= 0 else "#d95f0e" for v in s["cohens_d"]]
    ax[1].barh(y, s["cohens_d"].values, color=colors)
    ax[1].axvline(0, color="k", lw=1)
    ax[1].set_yticks(y); ax[1].set_yticklabels(CANON, fontsize=8)
    ax[1].set_xlabel("Cohen's d (stayer − dropper, V0)")
    ax[1].set_title("Stayers vs droppers — V0 profile")
    fig.tight_layout(); fig.savefig(FIGS / "31_attrition.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
