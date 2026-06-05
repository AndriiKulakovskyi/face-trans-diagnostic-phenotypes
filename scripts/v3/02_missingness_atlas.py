"""V3 Phase B — missingness atlas + observation-probability models.

In FACE, missingness is not noise: it reflects cohort design, site practice, questionnaire routing,
and sometimes severity. This script makes it an explicit object (docs/V3_PLAN.md Phase B) so we can
choose between an observed-likelihood MAR model and an informative-missingness sensitivity model
(docs/PIPELINE.md §1) with evidence rather than assumption.

  B1  observation matrix  R_ij = 1[observed]  summarized by cohort / site / age / sex / severity / dimension
  B2  per-variable mechanism class  (structural / clinical-skip / design-or-informative / sporadic)
  B3  observation-probability models  observed_j ~ cohort + age + sex (+ severity), within designed cohorts
        -> which variables have missingness driven by SEVERITY (informative / MNAR red flag) vs design

Severity proxy = row-mean of z(CGI-severity) and z(-GAF/EGF) on observed support (no imputation of the
latent model's cells; this is a diagnostic regression, listwise on its own predictors).

Aggregate outputs only:
  results/v3/missingness/missingness_atlas.md
  results/v3/missingness/missingness_by_variable.csv
  results/v3/missingness/observation_models.csv

Run:  python3 scripts/v3/02_missingness_atlas.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from v3.data import build_unified_dataframe, load_variables, to_harmonized_dataset  # noqa: E402

COH = ["bp", "sz", "dr"]
DICT = ROOT / "data" / "face-common-vars.xlsx"
CFG = ROOT / "configs" / "candidate_dimensions_v3.yaml"
OUT = ROOT / "results" / "v3" / "missingness"
PRED_COVARS = ["age", "sex", "education_years", "siteid_city"]  # not "indicators" of any dimension


def z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0


def main() -> None:
    cfg = yaml.safe_load(CFG.read_text())
    variables = load_variables(str(DICT))
    by_name = {v.canonical_name: v for v in variables}
    df = build_unified_dataframe("data", str(DICT), readiness=["READY", "PARTIAL"], format="long")
    ds = to_harmonized_dataset(df, variables, visit="V0", normalize=False, apply_skip_logic=True)
    X = ds.X
    cohort = X.index.get_level_values("cohort")
    feats = list(X.columns)

    # ---- B1: observation matrix -------------------------------------------------
    R = X.notna().astype(int)
    miss_by_cohort = (1 - R.groupby(cohort).mean()).T.reindex(columns=COH)  # feature x cohort missing-rate

    # predictors for B3
    age = z(X["age"]) if "age" in X else pd.Series(0.0, index=X.index)
    sex = pd.to_numeric(X.get("sex"), errors="coerce")
    edu = z(X["education_years"]) if "education_years" in X else pd.Series(np.nan, index=X.index)
    sev = pd.concat([
        z(X["cgi01"]) if "cgi01" in X else pd.Series(np.nan, index=X.index),
        -z(X["egf"]) if "egf" in X else pd.Series(np.nan, index=X.index),
    ], axis=1).mean(axis=1, skipna=True)   # higher = more severe
    coh_d = pd.get_dummies(pd.Series(cohort, index=X.index), prefix="coh", drop_first=True).astype(float)
    cohort_s = pd.Series(cohort, index=X.index)   # cohort label per patient (for 1/n_cohort weighting)

    designcoh = {f: {c: bool(getattr(by_name[f], f"{c}_csv_col")) if f in by_name else False for c in COH} for f in feats}
    sec_of = {f: (by_name[f].section if f in by_name else None) for f in feats}

    # ---- B3: per-variable observation-probability models ------------------------
    recs = []
    for f in feats:
        if f in PRED_COVARS:
            continue
        present = [c for c in COH if designcoh[f][c]]
        mask = pd.Series(cohort.isin(present), index=X.index)
        y = R[f][mask]
        if y.nunique() < 2:                      # fully observed or fully missing in design cohorts
            recs.append({"variable": f, "model": "constant", "n": int(mask.sum()),
                         "pseudo_r2_base": np.nan, "d_pseudo_r2_severity": np.nan,
                         "sev_coef": np.nan, "sev_p": np.nan, "driver": "none(const)"})
            continue
        base = pd.DataFrame({"age": age, "sex": sex}, index=X.index)
        if len(present) > 1:
            base = base.join(coh_d[[c for c in coh_d.columns if c.split("_", 1)[1] in present]])
        base = base[mask]
        # cohort-balancing weights (BP>>SZ>>DR): 1/n_cohort, rescaled to preserve sample size so each
        # present cohort contributes equally to the pooled observation model (the 1/n_cohort correction).
        present_n, k = int(mask.sum()), len(present)
        ncoh = cohort_s[mask].value_counts()
        wt = (present_n / (k * cohort_s.map(ncoh)))[mask]

        def fit(Xp, yv):
            d = pd.concat([yv.rename("y"), Xp, wt.rename("w")], axis=1).dropna()
            if d["y"].nunique() < 2 or len(d) < 50:
                return None
            try:
                Xd = sm.add_constant(d.drop(columns=["y", "w"]), has_constant="add")
                res = sm.GLM(d["y"], Xd, family=sm.families.Binomial(),
                             freq_weights=d["w"].to_numpy()).fit()
                prsq = (1.0 - res.deviance / res.null_deviance) if res.null_deviance else np.nan
                return {"params": res.params, "pvalues": res.pvalues, "prsq": float(prsq)}
            except Exception:
                return None

        m0 = fit(base, y)
        m1 = fit(base.join(sev.rename("severity")), y)
        pr2 = m0["prsq"] if m0 else np.nan
        pr2s = m1["prsq"] if m1 else np.nan
        sev_c = float(m1["params"].get("severity", np.nan)) if m1 else np.nan
        sev_p = float(m1["pvalues"].get("severity", np.nan)) if m1 else np.nan
        # dominant driver
        driver = "sporadic"
        if m0 is not None and not np.isnan(pr2):
            if len(present) < 3:
                driver = "design(cohort)"
            if not np.isnan(sev_p) and sev_p < 0.01 and abs(sev_c) >= 0.25:
                driver = "informative(severity↓obs)" if sev_c < 0 else "informative(severity↑obs)"
            elif not np.isnan(pr2) and pr2 < 0.02 and (np.isnan(sev_p) or sev_p > 0.05):
                driver = "sporadic" if len(present) == 3 else "design(cohort)"
        recs.append({"variable": f, "model": "logit", "n": int(y.shape[0]),
                     "pseudo_r2_base": round(pr2, 3) if not np.isnan(pr2) else np.nan,
                     "d_pseudo_r2_severity": round(pr2s - pr2, 3) if not (np.isnan(pr2) or np.isnan(pr2s)) else np.nan,
                     "sev_coef": round(sev_c, 3) if not np.isnan(sev_c) else np.nan,
                     "sev_p": round(sev_p, 4) if not np.isnan(sev_p) else np.nan,
                     "driver": driver})
    obs = pd.DataFrame(recs).set_index("variable")
    obs["section"] = [sec_of.get(f) for f in obs.index]

    # site spread: std of per-site observed-rate (B3 site question, kept out of the per-var logit)
    if "siteid_city" in X:
        site = X["siteid_city"].astype("category")
        site_rate = R.groupby(site.values).mean()
        site_spread = site_rate.std(axis=0)  # per feature
        obs["site_spread"] = [round(float(site_spread.get(f, np.nan)), 3) for f in obs.index]

    # ---- per-variable table -----------------------------------------------------
    vt = pd.DataFrame({
        "section": [sec_of.get(f) for f in feats],
        "miss_bp": miss_by_cohort["bp"].round(3), "miss_sz": miss_by_cohort["sz"].round(3),
        "miss_dr": miss_by_cohort["dr"].round(3),
    }, index=feats)
    vt = vt.join(obs[["driver", "sev_coef", "sev_p", "d_pseudo_r2_severity", "site_spread"]])
    vt.index.name = "variable"

    # ---- summaries --------------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    vt.sort_values("section").to_csv(OUT / "missingness_by_variable.csv")
    obs.to_csv(OUT / "observation_models.csv")

    overall = (1 - R.groupby(cohort).mean().mean(axis=1)).reindex(COH).round(3)
    by_sec = (1 - R.T.groupby([sec_of.get(f) for f in feats]).mean().T.groupby(cohort).mean()).T.reindex(columns=COH).round(2)
    drivers = obs["driver"].value_counts()
    informative = obs[obs["driver"].str.startswith("informative", na=False)].sort_values("sev_coef")
    # age-band coverage for the cognition block (classic informative case)
    cogfeats = [f for f in feats if sec_of.get(f) == "NEUROPSYCHOLOGIE" and f != "education_years"]

    md = ["# V3 missingness atlas (Phase B)",
          "",
          f"V0 N={len(X):,}. R_ij = 1[observed]; **no imputation**. Observation models fit within each "
          "variable's designed cohorts; severity = mean z(CGI-S, −GAF). Driver flags are heuristic.",
          "",
          "## Overall missingness rate by cohort (mean over features)",
          overall.to_frame("missing_rate").to_markdown(),
          "",
          "## Missingness rate by section × cohort",
          by_sec.to_markdown(),
          "",
          "## B3 — observation-mechanism drivers (per-variable)",
          drivers.to_frame("n_variables").to_markdown(),
          "",
          f"**Informative-missingness variables (severity predicts observation), n={len(informative)}** — "
          "these are the MNAR red flags for the sensitivity model:",
          (informative[["section", "sev_coef", "sev_p", "d_pseudo_r2_severity"]].head(25).to_markdown()
           if len(informative) else "_none flagged at sev p<0.01 & |coef|≥0.25_"),
          "",
          "## Candidate-dimension coverage by cohort (1 − missing, median over indicators)",
          ]
    for d in cfg["dimensions"]:
        ms = [m for m in (d.get("indicators", {}) or {}).get("names", []) if m in feats]
        ms += [f for f in feats if sec_of.get(f) in ((d.get("indicators", {}) or {}).get("sections", []) or [])]
        ms = sorted(set(ms) - set((d.get("indicators", {}) or {}).get("exclude", []) or []))
        if not ms:
            continue
        cov = {c: round(float((1 - miss_by_cohort.loc[ms, c]).median()), 2) for c in COH}
        md.append(f"- **{d['key']}** ({d['role']}): BP {cov['bp']} · SZ {cov['sz']} · DR {cov['dr']}  (n_ind={len(ms)})")
    md += ["",
           f"Cognition block ({len(cogfeats)} tests) severity-driver check: "
           + ", ".join(f"{f}:{obs.loc[f, 'driver']}" for f in cogfeats if f in obs.index) + ".",
           "",
           "Artifacts: `missingness_by_variable.csv`, `observation_models.csv`.",
           "",
           "### Read-out for modeling",
           "- **Structural/design** missingness (cohort-absent 2-cohort variables) → handle by "
           "eligibility tier (core/extension/module), **not** by imputation.",
           "- **Informative (severity-related)** variables → include in the Phase-F missingness "
           "**sensitivity** model (model `R_ij` jointly), not just MAR.",
           "- **Sporadic** (low pseudo-R², no severity signal) → safe under the observed-likelihood MAR model.",
           ]
    (OUT / "missingness_atlas.md").write_text("\n".join(md))

    # ---- console ---------------------------------------------------------------
    print(f"V0 N={len(X):,}\n\noverall missing by cohort:\n{overall.to_string()}\n")
    print("driver counts:\n", drivers.to_string(), "\n")
    print(f"informative (severity-related) variables: {len(informative)}")
    if len(informative):
        print(informative[["section", "sev_coef", "sev_p"]].head(20).to_string())
    print("\ncognition-block drivers:", {f: obs.loc[f, "driver"] for f in cogfeats if f in obs.index})
    print("\nwrote:", OUT.relative_to(ROOT), "/ {missingness_atlas.md, missingness_by_variable.csv, observation_models.csv}")


if __name__ == "__main__":
    main()
