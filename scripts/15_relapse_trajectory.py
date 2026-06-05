"""Relapse improvement #2 (v2) — EARLY-COURSE prognosis: predict V1->V2 relapse from V0 + V1.

A different (legitimate) question from baseline-only prognosis: among patients remitted at V1
(CGI-S 1-3), predict deterioration to CGI-S>=4 by V2, using the V0->V1 EARLY TRAJECTORY (early
response/course is among the strongest relapse predictors). All predictors are pre-V2 (no leakage);
CGI_V1 (the interval's baseline severity) is controlled in every model (de-confounds regression-to-
mean, as in scripts/46). Reuses the validated Study-C machinery (scripts/44) to compute V1 dimension
scores by projecting the V0 measurement model onto V1.

Predictor sets (outcome = CGI_V2 >= 4; one row per patient; StratifiedKFold on cohort x outcome):
  base   = age + sex + CGI_V1
  +V0dims= base + V0 axes (baseline-only prognosis — the comparator)
  +DSM   = base + dsm_diagnosis
  +traj  = base + dCGI(V0->V1) + V1 axes + d-axes(V1-V0)   [the early-trajectory model]
  +full  = base + DSM + traj features
Cognition is baseline-anchored (V1 ~5%) so its V1/d terms are mostly NaN -> gboost (NaN-native)
handles them; logistic shown on the well-measured features. Writes studyD4_trajectory.json.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trans_diag import build_unified_dataframe, load_variables, to_harmonized_dataset
from trans_diag.axes import AXIS_NAMES

warnings.simplefilter("ignore")
OUT = ROOT / "results" / "hfa"
SEED = 0
DIMS = [f"dim{i+1}" for i in range(len(AXIS_NAMES))]       # the data-locked K axes (trans_diag.axes)


def _load(stem):
    spec = importlib.util.spec_from_file_location(stem.replace(".py", ""), ROOT / "scripts" / stem)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


S44 = _load("11_longitudinal_coherence.py")


def gb():
    return HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3, max_iter=300,
                                          l2_regularization=1.0, early_stopping=True,
                                          validation_fraction=0.15, random_state=SEED)


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))


def cv_oof(X, y, strat, factory, allow_nan):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    ok = np.ones(len(y), bool) if allow_nan else np.isfinite(X).all(1)
    for tr, te in skf.split(X, strat):
        tr2, te2 = tr[ok[tr]], te[ok[te]]
        m = factory(); m.fit(X[tr2], y[tr2]); oof[te2] = m.predict_proba(X[te2])[:, 1]
    return oof, ok


def boot_ci(y, oa, ob, mask, n=1000):
    rng = np.random.default_rng(SEED); idx = np.where(mask)[0]; d = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        d.append(roc_auc_score(y[s], ob[s]) - roc_auc_score(y[s], oa[s]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def to_uid(idx):
    return [f"{c.upper()}::{p}" for c, p in idx]


def main() -> None:
    df = build_unified_dataframe(str(ROOT / "data"), str(ROOT / "data" / "face-common-vars.xlsx"),
                                 readiness=["READY", "PARTIAL", "NOT USABLE", "ID"], format="long")
    vs = load_variables(str(ROOT / "data" / "face-common-vars.xlsx"))
    by = {v.canonical_name: v for v in vs}
    ds = to_harmonized_dataset(df, vs, visit="V0", sections=None, residualize_on=None, normalize=False)

    # V0 & V1 construct scores (validated Study-C path), then project the committed V0 axis loadings
    S0 = S44.construct_scores_at(df, vs, by, "V0")
    S1 = S44.construct_scores_at(df, vs, by, "V1")
    L0 = pd.read_csv(OUT / "stage3_loadings.csv", index_col=0)[DIMS]
    A0 = S44.axis_scores(S0, L0); A0.columns = DIMS; A0.index = to_uid(A0.index)
    A1 = S44.axis_scores(S1, L0); A1.columns = DIMS; A1.index = to_uid(A1.index)
    S0.index = to_uid(S0.index); S1.index = to_uid(S1.index)

    M = pd.DataFrame(index=A0.index)
    for d in DIMS:
        M[f"{d}_v0"] = A0[d]; M[f"{d}_v1"] = A1[d].reindex(M.index); M[f"{d}_dl"] = M[f"{d}_v1"] - M[f"{d}_v0"]
    for c in ["mania_activation", "suicidal_ideation"]:
        M[f"{c}_v0"] = S0[c].reindex(M.index); M[f"{c}_v1"] = S1[c].reindex(M.index)
        M[f"{c}_dl"] = M[f"{c}_v1"] - M[f"{c}_v0"]
    cov = ds.X[["age", "sex"]].copy(); cov.index = to_uid(cov.index)
    M[["age", "sex"]] = cov.reindex(M.index)
    M["dsm"] = ds.metadata["dsm_diagnosis"].set_axis(to_uid(ds.metadata.index)).reindex(M.index)

    d2 = df[df.visit.isin(["V0", "V1", "V2"])].drop_duplicates(["patient_uid", "visit"])
    cg = pd.to_numeric(d2["cgi01"], errors="coerce")
    cgi = d2.assign(cgi=cg.where(cg.between(1, 7))).pivot_table(index="patient_uid", columns="visit", values="cgi", aggfunc="first")
    M["cgi_v0"] = cgi["V0"].reindex(M.index); M["cgi_v1"] = cgi["V1"].reindex(M.index); M["cgi_v2"] = cgi["V2"].reindex(M.index)
    M["dcgi"] = M["cgi_v1"] - M["cgi_v0"]
    M["cohort"] = [u.split("::")[0] for u in M.index]

    # at-risk: remitted at V1 (CGI 1-3), V2 observed, base predictors present
    M = M[M.cgi_v1.between(1, 3) & M.cgi_v2.notna() & M[["age", "sex"]].notna().all(1)].copy()
    y = (M["cgi_v2"] >= 4).astype(int).to_numpy()
    print(f"early-course at-risk (remitted@V1): n={len(M)}  relapse(by V2)={y.mean()*100:.0f}%  "
          f"cohorts={dict(M.cohort.value_counts())}")

    v0d = [f"{d}_v0" for d in DIMS] + ["mania_activation_v0", "suicidal_ideation_v0"]
    trajd = ["dcgi"] + [f"{d}_v1" for d in DIMS] + [f"{d}_dl" for d in DIMS] + \
            ["mania_activation_v1", "suicidal_ideation_v1", "mania_activation_dl", "suicidal_ideation_dl"]
    base = M[["age", "sex", "cgi_v1"]].to_numpy(float)
    dsm_oh = pd.get_dummies(M["dsm"].astype("object").fillna("NA"), prefix="dsm").astype(float).to_numpy(float)
    sets = {"base": base, "+V0dims": np.hstack([base, M[v0d].to_numpy(float)]),
            "+DSM": np.hstack([base, dsm_oh]), "+traj": np.hstack([base, M[trajd].to_numpy(float)]),
            "+full(DSM+traj)": np.hstack([base, dsm_oh, M[trajd].to_numpy(float)])}
    strat = M["cohort"] + "_" + pd.Series(y, index=M.index).astype(str)

    results = {"n": len(M), "relapse_rate": round(float(y.mean()), 3)}
    for label, factory, allow_nan in [("gboost", gb, True), ("logistic", lr, False)]:
        print(f"\n=== {label} — early-course (V1->V2) relapse AUC ===")
        oof = {k: cv_oof(X, y, strat.to_numpy(), factory, allow_nan) for k, X in sets.items()}
        for k in sets:
            o, ok = oof[k]; print(f"    {k:18s} AUC={roc_auc_score(y[ok], o[ok]):.3f}  (n={int(ok.sum())})")
        def delta(a, b):
            ma = oof[a][1] & oof[b][1]
            return roc_auc_score(y[ma], oof[b][0][ma]) - roc_auc_score(y[ma], oof[a][0][ma]), boot_ci(y, oof[a][0], oof[b][0], ma)
        dtv, ctv = delta("+V0dims", "+traj"); dtd, ctd = delta("+DSM", "+traj"); dtb, ctb = delta("base", "+traj")
        print(f"    Δ(traj vs V0-only)  = {dtv:+.3f}  CI[{ctv[0]:+.3f},{ctv[1]:+.3f}]")
        print(f"    Δ(traj vs DSM)      = {dtd:+.3f}  CI[{ctd[0]:+.3f},{ctd[1]:+.3f}]")
        print(f"    Δ(traj vs base)     = {dtb:+.3f}  CI[{ctb[0]:+.3f},{ctb[1]:+.3f}]")
        results[label] = {"AUC": {k: round(roc_auc_score(y[oof[k][1]], oof[k][0][oof[k][1]]), 4) for k in sets},
                          "d_traj_vs_v0": [round(dtv, 4), ctv], "d_traj_vs_dsm": [round(dtd, 4), ctd]}
    json.dump(results, open(OUT / "studyD4_trajectory.json", "w"), indent=2, default=str)
    print(f"\nsaved -> {OUT}/studyD4_trajectory.json")


if __name__ == "__main__":
    main()
