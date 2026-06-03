"""Relapse improvement #1 (v2) — does a RICHER baseline feature set beat the 6 compressed axes?

Same remission-based discrete-time-survival outcome as scripts/46 (at-risk = V0-remitted CGI 1-3;
relapse = deterioration to CGI>=4; person-intervals V0->V1, V1->V2). Question: did compressing the
88 construct scores into 6 axes cost relapse-prediction signal? Predictor sets:
  base = age + sex + V0 CGI + interval
  +DSM = base + dsm_diagnosis
  +axes(6) = base + the 4 axes + mania + suicidality
  +rich(constructs) = base + ALL well-covered construct scores
Leakage-safe: GroupKFold BY PATIENT; bootstrap CIs by patient; OOF AUC only; no feature selection
outside folds. HistGradientBoosting handles NaN natively (no imputation, full at-risk set); logistic
shown on the ~complete axes/DSM as the linear reference. Writes studyD3_richbaseline.json.
"""
from __future__ import annotations

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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trans_diag import build_unified_dataframe, load_variables, to_harmonized_dataset

warnings.simplefilter("ignore")
OUT = ROOT / "results" / "hfa"
AX = ["dim1", "dim2", "dim3", "dim4", "mania_activation", "suicidal_ideation"]
SEED = 0


def gb():
    return HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3, max_iter=300,
                                          l2_regularization=1.0, early_stopping=True,
                                          validation_fraction=0.15, random_state=SEED)


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))


def cv_oof(X, y, groups, factory, allow_nan):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    ok = np.isfinite(X).all(1) if not allow_nan else np.ones(len(y), bool)
    for tr, te in sgkf.split(X, y, groups):
        tr2, te2 = tr[ok[tr]], te[ok[te]]            # predict only on complete rows (logistic needs it)
        m = factory(); m.fit(X[tr2], y[tr2]); oof[te2] = m.predict_proba(X[te2])[:, 1]
    return oof, ok


def auc_ci(y, oa, ob, groups, mask, n=1000):
    rng = np.random.default_rng(SEED)
    uniq = np.unique(groups[mask])
    idxg = {g: np.where((groups == g) & mask)[0] for g in uniq}
    d = []
    for _ in range(n):
        rows = np.concatenate([idxg[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[rows])) < 2:
            continue
        d.append(roc_auc_score(y[rows], ob[rows]) - roc_auc_score(y[rows], oa[rows]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    df = build_unified_dataframe(str(ROOT / "data"), str(ROOT / "data" / "face-common-vars.xlsx"),
                                 readiness=["READY", "PARTIAL", "NOT USABLE", "ID"], format="long")
    vs = load_variables(str(ROOT / "data" / "face-common-vars.xlsx"))
    ds = to_harmonized_dataset(df, vs, visit="V0", sections=None, residualize_on=None, normalize=False)

    F = pd.read_pickle(OUT / "stage3_scores.pkl").set_index(["cohort", "patient_id"])[["dim1", "dim2", "dim3", "dim4"]]
    Sc = pd.read_pickle(OUT / "stage2_scores.pkl").set_index(["cohort", "patient_id"])
    constructs = [c for c in Sc.columns if Sc[c].notna().mean() >= 0.30]      # well-covered constructs
    P = F.join(Sc)
    P.index = [f"{c.upper()}::{p}" for c, p in P.index]
    cov = ds.X[["age", "sex"]].copy(); cov.index = [f"{c.upper()}::{p}" for c, p in cov.index]
    dsm = ds.metadata["dsm_diagnosis"].copy(); dsm.index = [f"{c.upper()}::{p}" for c, p in dsm.index]

    d = df[df.visit.isin(["V0", "V1", "V2"])].drop_duplicates(["patient_uid", "visit"])
    cg = pd.to_numeric(d["cgi01"], errors="coerce")
    cgi = d.assign(cgi=cg.where(cg.between(1, 7))).pivot_table(index="patient_uid", columns="visit", values="cgi", aggfunc="first")
    rows = []
    for uid in cgi.index[cgi["V0"].between(1, 3).fillna(False)]:
        v0, v1, v2 = cgi.loc[uid, "V0"], cgi.loc[uid, "V1"], cgi.loc[uid, "V2"]
        if pd.notna(v1):
            rows.append((uid, 1, int(v1 >= 4), v0))
            if v1 <= 3 and pd.notna(v2):
                rows.append((uid, 2, int(v2 >= 4), v0))
    R = pd.DataFrame(rows, columns=["uid", "interval", "event", "cgi0"])
    R = R.merge(P, left_on="uid", right_index=True, how="left").merge(cov, left_on="uid", right_index=True, how="left")
    R["dsm"] = R["uid"].map(dsm); R["is_int2"] = (R.interval == 2).astype(float)
    R = R[R[["age", "sex", "cgi0"]].notna().all(1)].reset_index(drop=True)
    dsm_oh = pd.get_dummies(R["dsm"].astype("object").fillna("NA"), prefix="dsm").astype(float)
    groups, y = R["uid"].to_numpy(), R["event"].to_numpy()
    print(f"person-intervals n={len(R)} (patients={len(np.unique(groups))}), events={int(y.sum())} ({y.mean()*100:.0f}%)")
    print(f"rich construct features: {len(constructs)}")

    base = R[["age", "sex", "cgi0", "is_int2"]].to_numpy(float)
    sets = {"base": base, "+DSM": np.hstack([base, dsm_oh.to_numpy(float)]),
            "+axes(6)": np.hstack([base, R[AX].to_numpy(float)]),
            "+rich(constructs)": np.hstack([base, R[constructs].to_numpy(float)])}
    results = {"n_intervals": len(R), "n_events": int(y.sum()), "n_constructs": len(constructs)}
    for label, factory, allow_nan in [("gboost", gb, True), ("logistic", lr, False)]:
        print(f"\n=== {label} — grouped-CV AUC ===")
        oof = {}
        for k, X in sets.items():
            if label == "logistic" and k == "+rich(constructs)":
                continue                                    # 81 feats need complete-case -> skip for linear
            o, ok = cv_oof(X, y, groups, factory, allow_nan)
            oof[k] = (o, ok)
            print(f"    {k:20s} AUC={roc_auc_score(y[ok], o[ok]):.3f}  (n={int(ok.sum())})")
        res = {}
        if "+rich(constructs)" in oof and "+axes(6)" in oof:
            mask = oof["+rich(constructs)"][1] & oof["+axes(6)"][1]
            d_ra = roc_auc_score(y[mask], oof["+rich(constructs)"][0][mask]) - roc_auc_score(y[mask], oof["+axes(6)"][0][mask])
            ci = auc_ci(y, oof["+axes(6)"][0], oof["+rich(constructs)"][0], groups, mask)
            print(f"    Δ(rich vs axes) = {d_ra:+.3f}  CI[{ci[0]:+.3f},{ci[1]:+.3f}]")
            res["rich_vs_axes"] = [round(d_ra, 4), ci]
        results[label] = res
    json.dump(results, open(OUT / "studyD3_richbaseline.json", "w"), indent=2, default=str)
    print(f"\nsaved -> {OUT}/studyD3_richbaseline.json")


if __name__ == "__main__":
    main()
