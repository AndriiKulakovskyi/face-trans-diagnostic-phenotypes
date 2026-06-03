"""Study D-refined (v2) — relapse prediction done right: remission-based + discrete-time survival.

Fixes the regression-to-mean confound in the original change-based relapse (Study D, scripts/45):
  - REMISSION-BASED: at-risk = patients remitted/stable at V0 (CGI-S in 1-3); relapse = deterioration
    to CGI-S >= 4 ("moderately ill"+). Everyone starts in the same low band -> baseline CGI can no
    longer mechanically predict "worsening" (the confound that inflated the old AUC and unfairly
    suppressed the dimensions).
  - DISCRETE-TIME SURVIVAL: person-interval format (V0->V1, V1->V2); a patient leaves the risk set on
    relapse and is censored on dropout -> uses partial information (handles the steep attrition that
    the binary-by-V2 outcome discarded). Discrete-time hazard = pooled model over person-intervals
    with an interval (baseline-hazard) term.

FAIR comparison vs DSM (so we don't jeopardize the model):
  - identical model class + hyperparameters for every predictor set (M0..M3);
  - DSM = the finest dsm_diagnosis (one-hot) — its best shot; same shared baseline M0 for all;
  - GroupKFold BY PATIENT (a patient's intervals never split train/test -> no leakage);
  - bootstrap CIs resample PATIENTS, not rows (respect clustering).
Two methods: regularized logistic (standard discrete-time hazard, interpretable) and HistGradient-
Boosting (modern, nonlinear; a deep net would overfit at this n). Writes studyD2_survival_v2.json.
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
from trans_diag.axes import AXIS_NAMES

warnings.simplefilter("ignore")
OUT = ROOT / "results" / "hfa"
AXES = ["dim1", "dim2", "dim3", "dim4", "mania", "suicide"]
AXES_X = ["dim2", "dim3", "dim4", "mania", "suicide"]      # cross-domain (drop internalizing)
SEED = 0


def models():
    return {"logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)),
            "gboost": HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3, max_iter=300,
                                                     l2_regularization=1.0, early_stopping=True,
                                                     validation_fraction=0.15, random_state=SEED)}


def cv_oof(X, y, groups, make_model):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    for tr, te in sgkf.split(X, y, groups):
        m = make_model(); m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def boot_ci(y, oa, ob, groups, n=1000):
    rng = np.random.default_rng(SEED)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    d = []
    for _ in range(n):
        gs = rng.choice(uniq, len(uniq), replace=True)
        rows = np.concatenate([idx_by_g[g] for g in gs])
        if len(np.unique(y[rows])) < 2:
            continue
        d.append(roc_auc_score(y[rows], ob[rows]) - roc_auc_score(y[rows], oa[rows]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    df = build_unified_dataframe(str(ROOT / "data"), str(ROOT / "data" / "face-common-vars.xlsx"),
                                 readiness=["READY", "PARTIAL", "NOT USABLE", "ID"], format="long")
    vs = load_variables(str(ROOT / "data" / "face-common-vars.xlsx"))
    ds = to_harmonized_dataset(df, vs, visit="V0", sections=None, residualize_on=None, normalize=False)

    # predictors keyed by patient_uid
    F = pd.read_pickle(OUT / "stage3_scores_v2.pkl").set_index(["cohort", "patient_id"])[["dim1", "dim2", "dim3", "dim4"]]
    Sc = pd.read_pickle(OUT / "stage2_scores_v2.pkl").set_index(["cohort", "patient_id"])[["mania_activation", "suicidal_ideation"]]
    P = F.join(Sc).rename(columns={"mania_activation": "mania", "suicidal_ideation": "suicide"})
    P.index = [f"{c.upper()}::{p}" for c, p in P.index]
    cov = ds.X[["age", "sex"]].copy(); cov.index = [f"{c.upper()}::{p}" for c, p in cov.index]
    dsm = ds.metadata["dsm_diagnosis"].copy(); dsm.index = [f"{c.upper()}::{p}" for c, p in dsm.index]

    # CGI wide (valid 1-7; 0 -> NaN), build remission-based discrete-time intervals
    d = df[df.visit.isin(["V0", "V1", "V2"])].drop_duplicates(["patient_uid", "visit"])
    cg = pd.to_numeric(d["cgi01"], errors="coerce")
    d = d.assign(cgi=cg.where((cg >= 1) & (cg <= 7)))
    cgi = d.pivot_table(index="patient_uid", columns="visit", values="cgi", aggfunc="first")
    rem = cgi["V0"].between(1, 3)
    rows = []
    for uid in cgi.index[rem.fillna(False)]:
        v0, v1, v2 = cgi.loc[uid, "V0"], cgi.loc[uid, "V1"], cgi.loc[uid, "V2"]
        if pd.notna(v1):                                   # interval 1 observed
            rows.append((uid, 1, int(v1 >= 4), v0))
            if v1 <= 3 and pd.notna(v2):                   # survived int1, interval 2 observed
                rows.append((uid, 2, int(v2 >= 4), v0))
    # uid as a COLUMN (patients have 2 interval-rows -> duplicate keys); map predictors, no reindex
    R = pd.DataFrame(rows, columns=["uid", "interval", "event", "cgi0"])
    R = R.merge(P, left_on="uid", right_index=True, how="left")
    R = R.merge(cov, left_on="uid", right_index=True, how="left")
    R["dsm"] = R["uid"].map(dsm)
    R["is_int2"] = (R["interval"] == 2).astype(float)
    need = AXES + ["age", "sex", "cgi0"]
    R = R[R[need].notna().all(1)].reset_index(drop=True)
    dsm_oh = pd.get_dummies(R["dsm"].astype("object").fillna("NA"), prefix="dsm").astype(float)
    groups = R["uid"].to_numpy()
    y = R["event"].to_numpy()
    print(f"person-intervals n={len(R)} (patients={int(len(np.unique(groups)))}), events={int(y.sum())} ({y.mean()*100:.0f}%)")

    base = R[["age", "sex", "cgi0", "is_int2"]].to_numpy(float)
    axX = R[AXES].to_numpy(float); axXcd = R[AXES_X].to_numpy(float); dsmX = dsm_oh.to_numpy(float)
    designs = {"M0_base": base, "M1_+DSM": np.hstack([base, dsmX]), "M2_+axes": np.hstack([base, axX]),
               "M3_+DSM+axes": np.hstack([base, dsmX, axX]), "M2x_crossdomain": np.hstack([base, axXcd])}

    results = {"n_intervals": len(R), "n_patients": int(int(len(np.unique(groups)))), "n_events": int(y.sum())}
    for method, _ in models().items():
        oof = {k: cv_oof(X, y, groups, lambda m=method: models()[m]) for k, X in designs.items()}
        pts = {k: roc_auc_score(y, o) for k, o in oof.items()}
        print(f"\n=== {method} — discrete-time relapse hazard (remission-based) — grouped-CV AUC ===")
        for k in designs:
            print(f"    {k:18s} AUC={pts[k]:.3f}")
        c21 = boot_ci(y, oof["M1_+DSM"], oof["M2_+axes"], groups)
        c31 = boot_ci(y, oof["M1_+DSM"], oof["M3_+DSM+axes"], groups)
        cx0 = boot_ci(y, oof["M0_base"], oof["M2x_crossdomain"], groups)
        print(f"    Δ(axes vs DSM)       = {pts['M2_+axes']-pts['M1_+DSM']:+.3f}  CI[{c21[0]:+.3f},{c21[1]:+.3f}]")
        print(f"    Δ(axes add over DSM) = {pts['M3_+DSM+axes']-pts['M1_+DSM']:+.3f}  CI[{c31[0]:+.3f},{c31[1]:+.3f}]")
        print(f"    Δ(cross-domain vs base, non-circular) = {pts['M2x_crossdomain']-pts['M0_base']:+.3f}  CI[{cx0[0]:+.3f},{cx0[1]:+.3f}]")
        results[method] = {"AUC": {k: round(v, 4) for k, v in pts.items()},
                           "d_axes_vs_dsm": [round(pts['M2_+axes']-pts['M1_+DSM'], 4), c21],
                           "d_axes_add_dsm": [round(pts['M3_+DSM+axes']-pts['M1_+DSM'], 4), c31],
                           "d_crossdomain_vs_base": [round(pts['M2x_crossdomain']-pts['M0_base'], 4), cx0]}

    json.dump(results, open(OUT / "studyD2_survival_v2.json", "w"), indent=2, default=str)
    print(f"\nsaved -> {OUT}/studyD2_survival_v2.json")


if __name__ == "__main__":
    main()
