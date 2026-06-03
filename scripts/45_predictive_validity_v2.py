"""Study D (v2) — predictive validity: do the V0 axes beat DSM at predicting V1-V2 outcomes?

THE make-or-break. Out-of-sample (cohort-stratified CV) only. Nested model comparison:
  M0 = age + sex + V0 baseline-of-the-outcome   (controls circularity/autocorrelation)
  M1 = M0 + DSM (dsm_diagnosis)                 (the comparator)
  M2 = M0 + 4 axes + mania + suicidality        (the dimensions)
  M3 = M0 + DSM + axes                          (do axes ADD over DSM?)
  M2x = M0 + CROSS-DOMAIN axes only (drop internalizing) — the non-circular test for CGI/functioning.

Outcomes (V2 horizon): relapse-by-V2 (CGI-S, binary, locked) + GAF (egf) + FAST functioning.
Guards: no outcome ever enters the FA (axes are outcome-blind features -> no leakage); V0 baseline in
every model; internalizing is BP+DR/proxy (Study A) so the cross-domain model is the clean test;
attrition = completers + a dropout-predictor check. Decision: axes earn their keep iff they add
cross-validated incremental value over DSM (CI excludes 0) on a HARD outcome. Masked / no-imputation
(complete-case on predictors+outcome; predictors are the existing masked scores).
Writes results/hfa/studyD_predictive_v2.json.
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trans_diag import build_unified_dataframe, load_variables, to_harmonized_dataset
from trans_diag.axes import AXIS_NAMES

warnings.simplefilter("ignore")
OUT = ROOT / "results" / "hfa"
AXES = ["dim1", "dim2", "dim3", "dim4"]            # internalizing, cognition, illness_course, cardiometab
AXES_X = ["dim2", "dim3", "dim4"]                  # cross-domain (drop internalizing = circular w/ CGI)
SEED = 0


def _load(stem):
    spec = importlib.util.spec_from_file_location(stem.replace(".py", ""), ROOT / "scripts" / stem)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cv_oof(X, y, task, strat, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    for tr, te in skf.split(X, strat):
        if task == "binary":
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
            m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
        else:
            m = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            m.fit(X[tr], y[tr]); oof[te] = m.predict(X[te])
    return oof


def metric(y, oof, task):
    return roc_auc_score(y, oof) if task == "binary" else r2_score(y, oof)


def boot_diff(y, oa, ob, task, n=1000):
    rng = np.random.default_rng(SEED)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if task == "binary" and len(np.unique(y[i])) < 2:
            continue
        d.append(metric(y[i], ob[i], task) - metric(y[i], oa[i], task))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    df = build_unified_dataframe(str(ROOT / "data"), str(ROOT / "data" / "face-common-vars.xlsx"),
                                 readiness=["READY", "PARTIAL", "NOT USABLE", "ID"], format="long")
    vs = load_variables(str(ROOT / "data" / "face-common-vars.xlsx"))
    ds = to_harmonized_dataset(df, vs, visit="V0", sections=None, residualize_on=None, normalize=False)

    # predictors: V0 axes + mania + suicidality, keyed by patient_uid (COHORT::patient_id)
    F = pd.read_pickle(OUT / "stage3_scores_v2.pkl").set_index(["cohort", "patient_id"])[AXES]
    S = pd.read_pickle(OUT / "stage2_scores_v2.pkl").set_index(["cohort", "patient_id"])[["mania_activation", "suicidal_ideation"]]
    P = F.join(S)
    P.index = [f"{c.upper()}::{p}" for c, p in P.index]          # -> patient_uid (the join gotcha)
    P = P.rename(columns={"mania_activation": "mania", "suicidal_ideation": "suicide"})

    meta = ds.metadata.copy(); meta.index = [f"{c.upper()}::{p}" for c, p in meta.index]
    dsm = meta["dsm_diagnosis"]
    cov = ds.X[["age", "sex"]].copy(); cov.index = [f"{c.upper()}::{p}" for c, p in cov.index]

    # outcomes: relapse-by-V2 (recompute, reproducible) + V0/V2 GAF & FAST
    relapse = _load("41_v1v4_inventory_v2.py").derive_relapse(df)
    def visit_wide(col):
        w = df[df.visit.isin(["V0", "V2"])].drop_duplicates(["patient_uid", "visit"]) \
            .pivot_table(index="patient_uid", columns="visit", values=col, aggfunc="first")
        return pd.to_numeric(w.get("V0"), errors="coerce"), pd.to_numeric(w.get("V2"), errors="coerce")
    egf0, egf2 = visit_wide("egf"); fast0, fast2 = visit_wide("fast"); cgi0, _ = visit_wide("cgi01")

    M = pd.DataFrame(index=P.index)
    M = M.join(P).join(cov)
    M["dsm"] = dsm
    M["cohort"] = [u.split("::")[0] for u in M.index]
    M["relapse"] = relapse["relapse_cgi_byV2"].reindex(M.index)
    M["relapse_ev"] = relapse["cgi_evaluable"].reindex(M.index)
    for nm, s in [("egf0", egf0), ("egf2", egf2), ("fast0", fast0), ("fast2", fast2), ("cgi0", cgi0)]:
        M[nm] = s.reindex(M.index)
    dsm_oh = pd.get_dummies(M["dsm"].astype("object").fillna("NA"), prefix="dsm").astype(float)

    print(f"master n={len(M)} | axes coverage {M[AXES].notna().all(1).mean():.2f}")

    results = {}
    OUTCOMES = [
        ("relapse-by-V2", "relapse", "binary", "cgi0", None),    # hard outcome; baseline = V0 CGI-S
        ("GAF@V2", "egf2", "cont", "egf0", None),                # functioning (3-cohort)
        ("FAST@V2", "fast2", "cont", "fast0", ["BP", "DR"]),     # functioning (BP+DR only)
    ]
    for name, ycol, task, base, cohorts in OUTCOMES:
        sub = M.copy()
        if cohorts:
            sub = sub[sub.cohort.isin(cohorts)]
        if ycol == "relapse":
            sub = sub[sub.relapse_ev == True]  # noqa: E712 (preserve NaN-safe filter behavior)
        need = AXES + ["mania", "suicide", "age", "sex", base, ycol]
        sub = sub[sub[need].notna().all(1)]
        y = sub[ycol].to_numpy(float)
        if len(sub) < 200:
            print(f"\n{name}: n={len(sub)} too small — skip"); continue
        strat = (sub["cohort"] + "_" + (sub[ycol] > np.median(y)).astype(str)) if task == "cont" \
            else (sub["cohort"] + "_" + sub[ycol].astype(int).astype(str))
        base_cols = sub[["age", "sex", base]].to_numpy(float)
        dsmX = dsm_oh.reindex(sub.index).fillna(0).to_numpy(float)
        axX = sub[AXES + ["mania", "suicide"]].to_numpy(float)
        axXcd = sub[AXES_X + ["mania", "suicide"]].to_numpy(float)
        designs = {"M0_base": base_cols, "M1_+DSM": np.hstack([base_cols, dsmX]),
                   "M2_+axes": np.hstack([base_cols, axX]),
                   "M3_+DSM+axes": np.hstack([base_cols, dsmX, axX]),
                   "M2x_crossdomain": np.hstack([base_cols, axXcd])}
        oof = {k: cv_oof(X, y, task, strat) for k, X in designs.items()}
        pts = {k: metric(y, o, task) for k, o in oof.items()}
        mname = "AUC" if task == "binary" else "R2"
        print(f"\n=== {name}  (n={len(sub)}, {'relapse %.0f%%' % (y.mean()*100) if task=='binary' else 'continuous'}) ===")
        for k in designs:
            print(f"    {k:18s} {mname}={pts[k]:.3f}")
        ci21 = boot_diff(y, oof["M1_+DSM"], oof["M2_+axes"], task)
        ci31 = boot_diff(y, oof["M1_+DSM"], oof["M3_+DSM+axes"], task)
        cix0 = boot_diff(y, oof["M0_base"], oof["M2x_crossdomain"], task)
        print(f"    Δ(axes vs DSM)        = {pts['M2_+axes']-pts['M1_+DSM']:+.3f}  CI[{ci21[0]:+.3f},{ci21[1]:+.3f}]")
        print(f"    Δ(axes add over DSM)  = {pts['M3_+DSM+axes']-pts['M1_+DSM']:+.3f}  CI[{ci31[0]:+.3f},{ci31[1]:+.3f}]")
        print(f"    Δ(cross-domain axes vs base, non-circular) = {pts['M2x_crossdomain']-pts['M0_base']:+.3f}  CI[{cix0[0]:+.3f},{cix0[1]:+.3f}]")
        results[name] = {"n": len(sub), "metric": mname, "points": {k: round(v, 4) for k, v in pts.items()},
                         "delta_axes_vs_dsm": [round(pts['M2_+axes']-pts['M1_+DSM'], 4), ci21],
                         "delta_axes_add_dsm": [round(pts['M3_+DSM+axes']-pts['M1_+DSM'], 4), ci31],
                         "delta_crossdomain_vs_base": [round(pts['M2x_crossdomain']-pts['M0_base'], 4), cix0]}
        if task == "cont":   # which axis carries the functional prediction? (incremental over baseline)
            attr = {AXIS_NAMES[j]: round(metric(y, cv_oof(np.hstack([base_cols, sub[[ax]].to_numpy(float)]),
                                                          y, task, strat), task) - pts["M0_base"], 4)
                    for j, ax in enumerate(AXES)}
            print(f"    per-axis ΔR² over baseline: {attr}")
            results[name]["per_axis_dR2"] = attr

    # dropout analysis: do V0 axes predict having a V2 relapse evaluation? (informative attrition)
    M["has_followup"] = M["relapse_ev"].fillna(False).astype(int)
    sub = M[M[AXES + ["age", "sex"]].notna().all(1)]
    oof = cv_oof(sub[AXES].to_numpy(float), sub["has_followup"].to_numpy(),
                 "binary", sub["cohort"] + "_" + sub["has_followup"].astype(str))
    auc_drop = roc_auc_score(sub["has_followup"], oof)
    print(f"\n=== attrition check: V0 axes -> has-V2-followup AUC = {auc_drop:.3f} "
          f"({'informative dropout — caveat' if auc_drop > 0.60 else 'near-chance — dropout ~ignorable by axes'}) ===")
    results["dropout_auc_from_axes"] = round(float(auc_drop), 3)

    json.dump(results, open(OUT / "studyD_predictive_v2.json", "w"), indent=2, default=str)
    print(f"\nsaved -> {OUT}/studyD_predictive_v2.json")


if __name__ == "__main__":
    main()
