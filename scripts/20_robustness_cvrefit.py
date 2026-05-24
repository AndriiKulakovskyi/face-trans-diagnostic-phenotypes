"""Robustness #4 — re-fit the dimensional axes INSIDE each CV fold (remove optimism).

The deferred check (MANUSCRIPT Limitation 10). The Phase-5 head-to-head (10_phase5_outcomes.py)
scores patients with axis loadings that were estimated once on the FULL sample (07), and then
uses those scores as predictors in cross-validation. The held-out fold therefore helped fit the
very loadings used to score it — a mild optimism. Does the "axes add over DSM" result survive
when the factor model is re-derived using only each training fold?

For every V1 outcome and every CV fold we compute three quantities under *identical* folds and
*train-only* preprocessing (scaling fit on train), so the only thing that changes is where the
masked-FA loadings come from:

    DSM        : baseline + age + sex + DSM(arm dummies)                 (no axes)
    axes_alldata : baseline + age + sex + axes scored from FULL-sample loadings (07)
    axes_refit   : baseline + age + sex + axes RE-FIT on the training fold only

and the combined (DSM + axes) analogues. The gap (axes_refit − axes_alldata) is exactly the
optimism that fold-honest refitting removes. Factor sign/order may differ across folds, which is
irrelevant: Ridge/logistic are invariant to a sign flip or a column permutation.

Repeated over N_REPEAT shuffled 5-fold splits (continuous → Ridge R²; binary → logistic AUC).

Artifacts: results/robustness_cvrefit.json, reports/robustness_cvrefit.html.
Run:  python3 scripts/20_robustness_cvrefit.py
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

from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import r2_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import KFold, StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from trans_diag import build_unified_dataframe  # noqa: E402
from trans_diag.masked_fa import masked_loadings, masked_scores  # noqa: E402
from trans_diag.outcomes import OUTCOMES  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
REPORTS_DIR = REPO_ROOT / "reports"
DOMAINS_PATH = RESULTS_DIR / "cluster_domains_scores.parquet"
AXES_PATH = RESULTS_DIR / "dimensional_final_scores.parquet"
K = 7                 # locked number of axes (matches 07)
N_REPEAT = 5          # shuffled 5-fold repeats (different seeds) for a stable estimate
RANDOM = 0


def orient_order(load: np.ndarray) -> np.ndarray:
    """Orient each factor so its defining domain is positive, order by sum-of-squares (as in 07).

    Cosmetic for prediction (Ridge/logistic are sign/permutation invariant) but keeps the per-fold
    axes interpretable and parallel to the locked model."""
    load = load.copy()
    for a in range(load.shape[1]):
        j = int(np.argmax(np.abs(load[:, a])))
        if load[j, a] < 0:
            load[:, a] = -load[:, a]
    return load[:, np.argsort(-(load ** 2).sum(0))]


def _fit_eval(Xtr, ytr, Xte, yte, kind) -> float:
    """Train-only standardize, fit, score one fold (R² for continuous, AUC for binary)."""
    ss = StandardScaler().fit(Xtr)
    Xtr, Xte = ss.transform(Xtr), ss.transform(Xte)
    if kind == "continuous":
        return float(r2_score(yte, Ridge(alpha=1.0).fit(Xtr, ytr).predict(Xte)))
    mdl = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    return float(roc_auc_score(yte, mdl.predict_proba(Xte)[:, 1]))


def _refit_axes(D: pd.DataFrame, tr: np.ndarray, te: np.ndarray):
    """Masked-FA loadings on the TRAIN fold only → posterior-mean scores for train + test.

    Standardization (per-domain mean/SD) is also estimated on train only. No cell is ever filled
    (masked correlation + observed-support scoring throughout)."""
    Dtr = D.iloc[tr]
    mu, sd = Dtr.mean(), Dtr.std(ddof=0)
    sd = sd.where(sd > 0, 1.0)
    load = orient_order(masked_loadings(Dtr, K))
    ztr = ((Dtr - mu) / sd).to_numpy(float)
    zte = ((D.iloc[te] - mu) / sd).to_numpy(float)
    return masked_scores(ztr, load), masked_scores(zte, load)


def main() -> int:
    REPORTS_DIR.mkdir(exist_ok=True)

    # domain scores (the masked-FA input) + the locked full-sample axis scores
    dom = pd.read_parquet(DOMAINS_PATH)
    dom.index = pd.MultiIndex.from_arrays(
        [dom.index.get_level_values("cohort").astype(str),
         dom.index.get_level_values("patient_id").astype(str)], names=("cohort", "patient_id"))
    domains = list(dom.columns)
    dom = dom.reset_index(); dom["pid"] = dom["cohort"] + "::" + dom["patient_id"]
    D_all = dom.set_index("pid")[domains]

    axes = pd.read_parquet(AXES_PATH)
    axis_cols = list(axes.columns)
    axes.index = pd.MultiIndex.from_arrays(
        [axes.index.get_level_values("cohort").astype(str),
         axes.index.get_level_values("patient_id").astype(str)], names=("cohort", "patient_id"))
    axes = axes.reset_index(); axes["pid"] = axes["cohort"] + "::" + axes["patient_id"]
    A_all = axes.set_index("pid")[axis_cols]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(REPO_ROOT / "data", REPO_ROOT / "face-common-vars.xlsx",
                                     readiness=["READY", "PARTIAL"], format="long")
    df["pid"] = df["cohort"].str.lower() + "::" + df["usubjid_patients"].astype(str)
    v0 = df[df["visit"] == "V0"].set_index("pid")
    vk = df[df["visit"] == "V1"].set_index("pid")

    base = pd.DataFrame(index=v0.index)
    base["age"] = pd.to_numeric(v0["age"], errors="coerce")
    base["sex"] = pd.to_numeric(v0["sex"], errors="coerce")
    dsm = pd.get_dummies(v0["arm"].astype(str), drop_first=True).astype(float)
    dsm_cols = list(dsm.columns)
    base = base.join(dsm)

    orig = pd.read_csv(RESULTS_DIR / "phase5_headtohead_V1.csv").set_index("outcome")

    print(f"CV-refit robustness: K={K} axes, {N_REPEAT}× shuffled 5-fold, refit per training fold\n")
    rows = []
    for name, kind, col, tf in OUTCOMES:
        if col not in df.columns:
            continue
        y0 = pd.to_numeric(v0[col], errors="coerce").rename("baseline")
        yk = pd.to_numeric(vk[col], errors="coerce")
        if tf is not None:
            yk = tf(yk)
        d = base.join(y0).join(yk.rename("y")).dropna(subset=["y", "baseline", "age", "sex"])
        D = D_all.reindex(d.index)
        A = A_all.reindex(d.index)
        keep = (D.notna().sum(axis=1) >= K) & A.notna().all(axis=1)   # need ≥K domains + a locked score
        d, D, A = d[keep], D[keep], A[keep]
        if kind == "binary" and (d["y"].nunique() < 2 or d["y"].mean() < 0.02):
            continue
        if len(d) < 200:
            continue

        yv = d["y"].to_numpy(float)
        bm = d[["baseline", "age", "sex"]].to_numpy(float)
        dm = d[dsm_cols].to_numpy(float)
        Am = A.to_numpy(float)
        idx = np.arange(len(d))

        acc = {kk: [] for kk in ("dsm", "ax_all", "ax_re", "comb_all", "comb_re")}
        for r in range(N_REPEAT):
            if kind == "continuous":
                splits = list(KFold(5, shuffle=True, random_state=RANDOM + r).split(idx))
            else:
                splits = list(StratifiedKFold(5, shuffle=True, random_state=RANDOM + r).split(idx, yv))
            fold = {kk: [] for kk in acc}
            for tr, te in splits:
                f_tr, f_te = _refit_axes(D, tr, te)
                m = {
                    "dsm": (np.hstack([bm[tr], dm[tr]]), np.hstack([bm[te], dm[te]])),
                    "ax_all": (np.hstack([bm[tr], Am[tr]]), np.hstack([bm[te], Am[te]])),
                    "ax_re": (np.hstack([bm[tr], f_tr]), np.hstack([bm[te], f_te])),
                    "comb_all": (np.hstack([bm[tr], dm[tr], Am[tr]]), np.hstack([bm[te], dm[te], Am[te]])),
                    "comb_re": (np.hstack([bm[tr], dm[tr], f_tr]), np.hstack([bm[te], dm[te], f_te])),
                }
                for kk, (Xtr, Xte) in m.items():
                    fold[kk].append(_fit_eval(Xtr, yv[tr], Xte, yv[te], kind))
            for kk in acc:
                acc[kk].append(float(np.mean(fold[kk])))

        def mm(a):
            return float(np.mean(a)), float(np.min(a)), float(np.max(a))
        dsm_m = mm(acc["dsm"]); axa = mm(acc["ax_all"]); axr = mm(acc["ax_re"])
        cba = mm(acc["comb_all"]); cbr = mm(acc["comb_re"])
        ax_re_minus_dsm = mm([a - b for a, b in zip(acc["ax_re"], acc["dsm"], strict=True)])
        cb_re_minus_dsm = mm([a - b for a, b in zip(acc["comb_re"], acc["dsm"], strict=True)])
        optimism = mm([a - b for a, b in zip(acc["ax_all"], acc["ax_re"], strict=True)])
        metric = "R2" if kind == "continuous" else "AUC"
        o_axd = float(orig.loc[name, "axes_minus_DSM"]) if name in orig.index else np.nan

        rows.append({
            "outcome": name, "n": int(len(d)), "metric": metric,
            "DSM": round(dsm_m[0], 3),
            "axes_alldata": round(axa[0], 3), "axes_refit": round(axr[0], 3),
            "combined_alldata": round(cba[0], 3), "combined_refit": round(cbr[0], 3),
            "axes_refit_minus_DSM": round(ax_re_minus_dsm[0], 3),
            "axes_refit_minus_DSM_range": [round(ax_re_minus_dsm[1], 3), round(ax_re_minus_dsm[2], 3)],
            "combined_refit_minus_DSM": round(cb_re_minus_dsm[0], 3),
            "optimism_alldata_minus_refit": round(optimism[0], 3),
            "orig_axes_minus_DSM": round(o_axd, 3) if np.isfinite(o_axd) else None,
        })
        print(f"{name}: n={len(d)} {metric}\n"
              f"  DSM={dsm_m[0]:.3f}  axes(all-data)={axa[0]:.3f}  axes(refit)={axr[0]:.3f}  "
              f"combined(refit)={cbr[0]:.3f}\n"
              f"  axes(refit)−DSM = {ax_re_minus_dsm[0]:+.3f} [{ax_re_minus_dsm[1]:+.3f},{ax_re_minus_dsm[2]:+.3f}]"
              f"  (orig all-data {o_axd:+.3f}); optimism removed = {optimism[0]:+.3f}\n")

    head = pd.DataFrame(rows)
    meta = {"K": K, "n_repeat": N_REPEAT,
            "design": "axes re-fit (masked FA) within each training fold; train-only scaling; "
                      "identical folds across DSM/axes_alldata/axes_refit",
            "headtohead_cvrefit": head.to_dict(orient="records")}
    (RESULTS_DIR / "robustness_cvrefit.json").write_text(json.dumps(meta, indent=2, default=str))
    _report(head)
    print("Wrote results/robustness_cvrefit.json + reports/robustness_cvrefit.html. Done.")
    return 0


def _report(head: pd.DataFrame):
    rows = "".join(
        f"<tr><td>{r.outcome}</td><td>{r.n}</td><td>{r.metric}</td><td>{r.DSM}</td>"
        f"<td>{r.axes_alldata}</td><td>{r.axes_refit}</td><td>{r.combined_refit}</td>"
        f"<td><b>{r.axes_refit_minus_DSM:+}</b></td><td>{r.orig_axes_minus_DSM:+}</td>"
        f"<td>{r.optimism_alldata_minus_refit:+}</td></tr>"
        for r in head.itertuples())
    css = ("body{font-family:-apple-system,Segoe UI,sans-serif;margin:0;padding:0 24px 60px}"
           "h1{color:#2b3a55}table{border-collapse:collapse;font-size:13px;margin:12px 0}"
           "th,td{border:1px solid #e5e7eb;padding:5px 10px}th{background:#eef2f7}"
           ".c{background:#f2fbf6;border-left:4px solid #16a085;padding:10px 14px;margin:12px 0}")
    html = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{css}</style></head><body>",
            "<h1>Robustness — re-fit axes inside CV folds (Limitation 10)</h1>",
            "<div class='c'>The masked factor model is re-derived on each training fold and used to "
            "score the held-out fold (train-only loadings <i>and</i> scaling). 'axes(refit)−DSM' is "
            "the fold-honest incremental value; compare to the original all-data 'orig axes−DSM'. "
            "'optimism' = all-data minus refit (how much the full-sample loadings inflated the "
            "estimate).</div>",
            "<table><tr><th>outcome</th><th>n</th><th>metric</th><th>DSM</th>"
            "<th>axes (all-data)</th><th>axes (refit)</th><th>combined (refit)</th>"
            "<th>axes(refit)−DSM</th><th>orig axes−DSM</th><th>optimism</th></tr>",
            rows, "</table></body></html>"]
    (REPORTS_DIR / "robustness_cvrefit.html").write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
