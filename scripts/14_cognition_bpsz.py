"""Cognition (NEUROPSYCHOLOGIE) — a BP/SZ complementary analysis.

Cognition is absent in the depression cohort by design (DR = 0% coverage; BP 71%,
SZ 86%), so it CANNOT enter the 3-cohort dimensional model without re-injecting a
cohort/availability confound. We therefore analyse it within the BP/SZ subset, where
it is measured, as a complementary psychosis-spectrum dimension.

Three steps:
  1. Derive cognitive factor(s) — NEUROPSYCHOLOGIE items → instrument-domain scores
     (CVLT memory, TMT executive/speed, WAIS working memory, …) → age/sex-residualized
     factor analysis (parallel analysis for K; a general-cognition factor expected).
  2. Relate cognition to the 6 trans-diagnostic symptom axes (both age/sex-residualized)
     in BP/SZ — correlation matrix.
  3. Outcome prediction within BP/SZ — does cognition add to the symptom axes for
     predicting V1 functioning (EGF)? Nested 5-fold CV ΔR².

Artifacts: results/cognition_bpsz_{loadings.csv,scores.parquet,corr.csv,meta.json},
reports/cognition_bpsz.html.
Run:  python3 scripts/14_cognition_bpsz.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.decomposition import FactorAnalysis  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from trans_diag import (  # noqa: E402
    ADMINISTRATIVE_FEATURES,
    AXIS_NAMES,
    build_domain_scores,
    build_unified_dataframe,
    load_variables,
    residualize_features,
    to_harmonized_dataset,
)
from trans_diag.outcomes import cv_metric  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
REPORTS_DIR = REPO_ROOT / "reports"
SECTION = "NEUROPSYCHOLOGIE"
MIN_DOMAINS, COV_FLOOR, RANDOM = 3, 0.30, 0
SYMPTOM_AXES = AXIS_NAMES   # the 6 symptom axes (shared constant, trans_diag.axes)

# Curated standard cognitive constructs (so WAIS sub-items don't dominate by count).
# sign = +1 higher-is-better; TMT is time/error-based (higher-is-worse, sign −1,
# confirmed: TMT-B correlates −0.16 with CVLT). The sign orients each member so every
# domain reads "higher = better cognition".
COGNITIVE_DOMAINS = {
    "memory_cvlt":       [("cvlt", +1)],
    "executive_tmt":     [("tmtb", -1), ("tmtba", -1)],
    "proc_speed":        [("tmta", -1), ("code01_wais", +1), ("code02_wais", +1),
                          ("code03_wais", +1), ("code04_wais", +1), ("code05_wais", +1),
                          ("ivt01_wais", +1), ("ivt02_wais", +1), ("ivt04_wais", +1)],
    "working_memory":    [("nbrut_w", +1), ("nstand_w", +1), ("vstand_w", +1),
                          ("mcod_w", +1), ("mcoi_w", +1), ("mcoc_w", +1), ("mcodemp_w", +1),
                          ("mcoiemp_w", +1), ("empdid_w", +1), ("wais_mc_end_std_wais", +1),
                          ("wais_mc_env_std_wais", +1), ("wais_mc_cro_wais", +1),
                          ("wais_mc_cro_std_wais", +1)],
    "verbal_reasoning":  [("similtot_wais", +1), ("similstd_wais", +1), ("similcr_wais", +1)],
    "percept_reasoning": [("mat_tot_w", +1), ("mat_std_w", +1), ("mat_cr_w", +1)],
    "fluency":           [("fv", +1)],
}


def parallel_analysis(X, n_iter=30, seed=0, cap=4):
    rng = np.random.default_rng(seed)
    real = np.sort(np.linalg.eigvalsh(np.corrcoef(X.T)))[::-1]
    null = np.empty((n_iter, X.shape[1]))
    for i in range(n_iter):
        Xp = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
        null[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(Xp.T)))[::-1]
    K = int((real > np.percentile(null, 95, axis=0)).sum())
    return min(max(K, 1), cap)


def main() -> int:
    REPORTS_DIR.mkdir(exist_ok=True)
    variables = load_variables(REPO_ROOT / "face-common-vars.xlsx")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(REPO_ROOT / "data", REPO_ROOT / "face-common-vars.xlsx",
                                     readiness=["READY", "PARTIAL"], format="long")
        cog_ds = to_harmonized_dataset(df, variables, visit="V0",
                                       exclude=ADMINISTRATIVE_FEATURES, sections={SECTION})
        full = to_harmonized_dataset(df, variables, visit="V0", exclude=ADMINISTRATIVE_FEATURES)

    # 1. cognitive domains — two-level: raw items → instrument stem-domains → curated
    #    standard constructs (so WAIS sub-items don't dominate by count).
    cog_stems, _ = build_domain_scores(cog_ds.X, variables, symptom_sections={SECTION}, biology={})
    cog, cmeta = build_domain_scores(cog_stems, variables, symptom_sections=set(),
                                     biology=COGNITIVE_DOMAINS)
    # keep BP/SZ patients with cognition; drop DR (all-NaN) and sparse rows
    have = cog.notna().sum(axis=1) >= MIN_DOMAINS
    cog = cog[have]
    cov = cog.notna().mean()
    cog = cog[cov[cov >= COV_FLOOR].index]
    cohort = pd.Series([c for c, _ in cog.index], index=cog.index)
    print(f"cognition subset: {len(cog):,} patients "
          f"{cohort.value_counts().to_dict()}, {cog.shape[1]} cognitive domains: {list(cog.columns)}")

    age = full.X.reindex(cog.index)["age"]
    sex = full.X.reindex(cog.index)["sex"]
    covars = pd.DataFrame({"age": age, "sex": sex}, index=cog.index)
    cog_r = residualize_features(cog, covars, spline_df=4, cross_fit=5, random_state=RANDOM)
    Z = StandardScaler().fit_transform(((cog_r - cog_r.mean()) / cog_r.std(ddof=0)).fillna(0.0))

    K = parallel_analysis(Z)
    fa = FactorAnalysis(n_components=K, rotation="varimax", random_state=RANDOM).fit(Z)
    load = fa.components_.T
    order = np.argsort(-(load ** 2).sum(0)); load = load[:, order]
    scores = fa.transform(Z)[:, order]
    # orient each factor so its top-|loading| domain is positive
    for a in range(K):
        if load[np.argmax(np.abs(load[:, a])), a] < 0:
            load[:, a] *= -1; scores[:, a] *= -1
    names = [f"cog{a+1}" for a in range(K)]
    cogf = pd.DataFrame(scores, index=cog.index, columns=names)
    cogf.to_parquet(RESULTS_DIR / "cognition_bpsz_scores.parquet")
    print(f"\ncognitive factors (K={K}, parallel analysis):")
    for a in range(K):
        s = pd.Series(load[:, a], index=cog.columns).sort_values(key=abs, ascending=False)
        print(f"  cog{a+1}: " + "; ".join(f"{d}({v:+.2f})" for d, v in s.head(5).items()))
    pd.DataFrame([{"factor": names[a], "domain": d, "loading": float(load[i, a])}
                  for a in range(K) for i, d in enumerate(cog.columns)]).to_csv(
        RESULTS_DIR / "cognition_bpsz_loadings.csv", index=False)

    # 2. relate cognition to the 6 symptom axes (both age/sex-residualized)
    sym = pd.read_parquet(RESULTS_DIR / "dimensional_final_scores.parquet")
    sym.index = pd.MultiIndex.from_arrays(
        [sym.index.get_level_values("cohort").astype(str),
         sym.index.get_level_values("patient_id").astype(str)], names=("cohort", "patient_id"))
    sym.columns = SYMPTOM_AXES
    common = cogf.index.intersection(sym.index)
    corr = pd.DataFrame(index=names, columns=SYMPTOM_AXES, dtype=float)
    pvals = pd.DataFrame(index=names, columns=SYMPTOM_AXES, dtype=float)
    for c in names:
        for s in SYMPTOM_AXES:
            d = pd.concat([cogf.loc[common, c], sym.loc[common, s]], axis=1).dropna()
            r, p = pearsonr(d.iloc[:, 0], d.iloc[:, 1])
            corr.loc[c, s], pvals.loc[c, s] = r, p
    corr.to_csv(RESULTS_DIR / "cognition_bpsz_corr.csv")
    print(f"\ncognition ↔ symptom-axis correlations (n={len(common):,} BP/SZ):")
    print(corr.round(3).to_string())

    # 3. does cognition add to the symptom axes for predicting V1 functioning?
    df["pid"] = df["cohort"].str.lower() + "::" + df["usubjid_patients"].astype(str)
    cogf2 = cogf.copy(); cogf2["pid"] = [f"{c}::{p}" for c, p in cogf.index]
    sym2 = sym.copy(); sym2["pid"] = [f"{c}::{p}" for c, p in sym.index]
    v0 = df[df.visit == "V0"].drop_duplicates("pid").set_index("pid")
    v1 = df[df.visit == "V1"].drop_duplicates("pid").set_index("pid")
    base = pd.DataFrame({"baseline": pd.to_numeric(v0["egf"], errors="coerce"),
                         "age": pd.to_numeric(v0["age"], errors="coerce"),
                         "sex": pd.to_numeric(v0["sex"], errors="coerce"),
                         "y": pd.to_numeric(v1["egf"], errors="coerce")})
    d = (base.join(sym2.set_index("pid")[SYMPTOM_AXES]).join(cogf2.set_index("pid")[names])
         .dropna(subset=["y", "baseline", "age", "sex"] + SYMPTOM_AXES + names))
    bc = ["baseline", "age", "sex"]
    r_sym = cv_metric(d[bc + SYMPTOM_AXES].to_numpy(float), d["y"].to_numpy(float), "continuous")
    r_symcog = cv_metric(d[bc + SYMPTOM_AXES + names].to_numpy(float), d["y"].to_numpy(float), "continuous")
    print(f"\nV1 functioning (EGF), n={len(d):,} BP/SZ:  symptom-axes R²={r_sym:.3f}  "
          f"+cognition R²={r_symcog:.3f}  (Δ {r_symcog-r_sym:+.3f})")

    meta = {"n_cognition": int(len(cog)), "cohort": cohort.value_counts().to_dict(),
            "K": int(K), "factors": names,
            "outcome_egf": {"n": int(len(d)), "symptom_axes_R2": r_sym,
                            "symptom_plus_cognition_R2": r_symcog, "delta": r_symcog - r_sym}}
    (RESULTS_DIR / "cognition_bpsz_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    _report(load, list(cog.columns), names, corr, r_sym, r_symcog, len(common))
    print("\nWrote results/cognition_bpsz_* + reports/cognition_bpsz.html. Done.")
    return 0


def _report(load, domains, names, corr, r_sym, r_symcog, n):
    f1 = go.Figure(go.Heatmap(z=load.T, x=domains, y=names, colorscale="RdBu", zmid=0,
                              colorbar=dict(title="loading", thickness=12)))
    f1.update_layout(title="Cognitive factor loadings (instrument-domains × factors)",
                     height=120 + 50 * len(names), margin=dict(t=46, l=80, b=120), xaxis_tickangle=-45)
    f2 = go.Figure(go.Heatmap(z=corr.to_numpy(float), x=list(corr.columns), y=list(corr.index),
                              text=corr.round(2).to_numpy(), texttemplate="%{text}",
                              colorscale="RdBu", zmid=0, zmin=-0.4, zmax=0.4,
                              colorbar=dict(title="Pearson r", thickness=12)))
    f2.update_layout(title=f"Cognition ↔ symptom axes (BP/SZ, n={n:,}; both age/sex-residualized)",
                     height=120 + 50 * len(names), margin=dict(t=46, l=80, b=140), xaxis_tickangle=-30)
    css = "body{font-family:-apple-system,Segoe UI,sans-serif;margin:0;padding:0 24px 60px}h1{color:#2b3a55}.c{background:#f2fbf6;border-left:4px solid #16a085;padding:10px 14px;margin:12px 0}"
    html = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{css}</style></head><body>",
            "<h1>Cognition (BP/SZ complementary analysis)</h1>",
            f"<div class='c'>Cognition is absent in DR by design → analysed within BP/SZ only. "
            f"V1 functioning (EGF): symptom-axes R²={r_sym:.3f}; +cognition R²={r_symcog:.3f} "
            f"(Δ {r_symcog-r_sym:+.3f}).</div>",
            pio.to_html(f1, include_plotlyjs="cdn", full_html=False),
            pio.to_html(f2, include_plotlyjs=False, full_html=False), "</body></html>"]
    (REPORTS_DIR / "cognition_bpsz.html").write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
