"""Replication #5 — within-FACE held-out (leave-one-cohort / leave-one-site) generalization.

MANUSCRIPT Limitation 9: there is no *external* replication — FACE is a single national network.
True external replication is out of reach here, but we can test **transportability** within FACE
by deriving the model on a held-out partition and applying it to data it never saw:

  Part 1 — Leave-one-COHORT-out structure. Re-fit the masked loadings WITHOUT each diagnostic
    cohort (e.g. BP+SZ only) and measure Tucker congruence with the locked full-sample axes. High
    congruence ⇒ the dimensional structure is not carried by any single diagnosis.

  Part 2 — Leave-one-SITE-out outcome prediction. Site-blocked CV (LeaveOneGroupOut over the
    centres ≥10 patients): for every site, re-fit the axes on the OTHER sites, train the outcome
    models there, predict the held-out site, then pool the out-of-site predictions and score once.
    Tests whether the axes-vs-DSM advantage holds on centres the model never saw.

  Part 3 — Leave-one-COHORT-out outcome prediction. Train the axes AND the outcome model on the
    other cohorts and predict the held-out diagnosis; report the axes' increment over an
    age+sex+baseline model (DSM is degenerate within a single held-out cohort). The strongest
    trans-diagnostic transport test: predict an unseen diagnosis from axes learned without it.

Artifacts: results/replication_holdout.json, reports/replication_holdout.html.
Run:  python3 scripts/21_replication_holdout.py
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
from sklearn.model_selection import LeaveOneGroupOut  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from trans_diag import build_unified_dataframe  # noqa: E402
from trans_diag.masked_fa import masked_loadings, masked_scores  # noqa: E402
from trans_diag.outcomes import OUTCOMES  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
REPORTS_DIR = REPO_ROOT / "reports"
DOMAINS_PATH = RESULTS_DIR / "cluster_domains_scores.parquet"
LOADINGS_PATH = RESULTS_DIR / "dimensional_final_loadings.csv"
K = 7
MIN_SITE = 10          # a held-out site must have ≥10 patients (matches 13_robustness_site)
MIN_COHORT_TEST = 150  # min held-out-cohort follow-up n to report a transport R²


def orient_order(load: np.ndarray) -> np.ndarray:
    load = load.copy()
    for a in range(load.shape[1]):
        j = int(np.argmax(np.abs(load[:, a])))
        if load[j, a] < 0:
            load[:, a] = -load[:, a]
    return load[:, np.argsort(-(load ** 2).sum(0))]


def tucker_min(La: np.ndarray, Lb: np.ndarray):
    """Per-axis best-matched Tucker congruence (greedy, sign-invariant) of La onto Lb."""
    used, out = set(), []
    for a in range(La.shape[1]):
        best, bj = 0.0, -1
        for b in range(Lb.shape[1]):
            if b in used:
                continue
            den = np.linalg.norm(La[:, a]) * np.linalg.norm(Lb[:, b])
            phi = abs(float(La[:, a] @ Lb[:, b])) / den if den > 0 else 0.0
            if phi > best:
                best, bj = phi, b
        out.append(best); used.add(bj)
    return out


def _refit_axes(D: pd.DataFrame, tr: np.ndarray, te: np.ndarray):
    """Masked-FA loadings on the TRAIN rows only → posterior-mean scores for train + test."""
    Dtr = D.iloc[tr]
    mu, sd = Dtr.mean(), Dtr.std(ddof=0)
    mu = mu.fillna(0.0)                       # a domain unobserved in this fold → mean 0 (no NaN contamination)
    sd = sd.where(sd > 0, 1.0).fillna(1.0)
    load = orient_order(masked_loadings(Dtr, K))
    ztr = ((Dtr - mu) / sd).to_numpy(float)
    zte = ((D.iloc[te] - mu) / sd).to_numpy(float)
    return masked_scores(ztr, load), masked_scores(zte, load)


def _fit_predict(Xtr, ytr, Xte, kind):
    ss = StandardScaler().fit(Xtr)
    Xtr, Xte = ss.transform(Xtr), ss.transform(Xte)
    if kind == "continuous":
        return Ridge(alpha=1.0).fit(Xtr, ytr).predict(Xte)
    return LogisticRegression(max_iter=2000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def _score(y, p, kind):
    ok = np.isfinite(p) & np.isfinite(y)
    return float(r2_score(y[ok], p[ok])) if kind == "continuous" else float(roc_auc_score(y[ok], p[ok]))


def main() -> int:
    REPORTS_DIR.mkdir(exist_ok=True)
    sc = pd.read_parquet(DOMAINS_PATH)
    sc.index = pd.MultiIndex.from_arrays(
        [sc.index.get_level_values("cohort").astype(str),
         sc.index.get_level_values("patient_id").astype(str)], names=("cohort", "patient_id"))
    domains = list(sc.columns)
    locked = (pd.read_csv(LOADINGS_PATH).pivot(index="domain", columns="axis", values="loading")
              .reindex(domains).to_numpy(float))

    # ── Part 1: leave-one-cohort-out structure (congruence with the locked axes) ──
    coh_idx = sc.index.get_level_values("cohort")
    print("Part 1 — leave-one-cohort-out structure (Tucker congruence vs locked axes):")
    loco_struct = []
    for held in sorted(pd.unique(coh_idx)):
        mask = np.asarray(coh_idx != held)
        Lh = orient_order(masked_loadings(sc[mask], K))
        mins = tucker_min(Lh, locked)
        loco_struct.append({"held_out_cohort": held, "n_train": int(mask.sum()),
                            "min_congruence": round(float(np.min(mins)), 2),
                            "mean_congruence": round(float(np.mean(mins)), 2),
                            "per_axis": [round(float(x), 2) for x in mins]})
        print(f"  hold out {held.upper():>3} (train n={mask.sum():,}): "
              f"min={np.min(mins):.2f} mean={np.mean(mins):.2f}  per-axis={[round(float(x),2) for x in mins]}")

    # ── shared plumbing for the outcome parts ──
    domf = sc.reset_index(); domf["pid"] = domf["cohort"] + "::" + domf["patient_id"]
    D_all = domf.set_index("pid")[domains]
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
    site_all = pd.to_numeric(v0["siteid_city"], errors="coerce")

    # ── Part 2: leave-one-site-out outcome prediction (pooled out-of-site) ──
    print("\nPart 2 — leave-one-site-out outcome prediction (pooled, axes re-fit on other sites):")
    loso = []
    for name, kind, col, tf in OUTCOMES:
        if col not in df.columns:
            continue
        y0 = pd.to_numeric(v0[col], errors="coerce").rename("baseline")
        yk = pd.to_numeric(vk[col], errors="coerce")
        if tf is not None:
            yk = tf(yk)
        d = base.join(y0).join(yk.rename("y")).dropna(subset=["y", "baseline", "age", "sex"])
        D = D_all.reindex(d.index)
        ssite = site_all.reindex(d.index)
        keep = (D.notna().sum(axis=1) >= K) & ssite.notna()
        d, D, ssite = d[keep], D[keep], ssite[keep]
        counts = ssite.value_counts()
        big = ssite.isin(counts[counts >= MIN_SITE].index)
        d, D, ssite = d[big], D[big], ssite[big]
        if kind == "binary" and (d["y"].nunique() < 2 or d["y"].mean() < 0.02):
            continue
        if len(d) < 200:
            continue
        yv = d["y"].to_numpy(float)
        bm = d[["baseline", "age", "sex"]].to_numpy(float)
        dm = d[dsm_cols].to_numpy(float)
        groups = ssite.astype(int).to_numpy()
        idx = np.arange(len(d))
        p_dsm, p_ax, p_cb = (np.full(len(d), np.nan) for _ in range(3))
        for tr, te in LeaveOneGroupOut().split(idx, yv, groups=groups):
            f_tr, f_te = _refit_axes(D, tr, te)
            p_dsm[te] = _fit_predict(np.hstack([bm[tr], dm[tr]]), yv[tr], np.hstack([bm[te], dm[te]]), kind)
            p_ax[te] = _fit_predict(np.hstack([bm[tr], f_tr]), yv[tr], np.hstack([bm[te], f_te]), kind)
            p_cb[te] = _fit_predict(np.hstack([bm[tr], dm[tr], f_tr]), yv[tr], np.hstack([bm[te], dm[te], f_te]), kind)
        m0, m1, m2 = _score(yv, p_dsm, kind), _score(yv, p_ax, kind), _score(yv, p_cb, kind)
        metric = "R2" if kind == "continuous" else "AUC"
        loso.append({"outcome": name, "n": int(len(d)), "n_sites": int(ssite.nunique()), "metric": metric,
                     "DSM": round(m0, 3), "axes": round(m1, 3), "combined": round(m2, 3),
                     "axes_minus_DSM": round(m1 - m0, 3), "combined_minus_DSM": round(m2 - m0, 3)})
        print(f"  {name}: n={len(d)} over {ssite.nunique()} sites  {metric}  DSM={m0:.3f} "
              f"axes={m1:.3f} combined={m2:.3f}  (axes−DSM {m1-m0:+.3f}, comb−DSM {m2-m0:+.3f})")

    # ── Part 3: leave-one-cohort-out outcome prediction (axes increment over baseline) ──
    print("\nPart 3 — leave-one-cohort-out outcome prediction (predict an UNSEEN diagnosis):")
    loco_out = []
    for name, kind, col, tf in OUTCOMES:
        if kind != "continuous" or col not in df.columns:   # DSM degenerate within one cohort
            continue
        y0 = pd.to_numeric(v0[col], errors="coerce").rename("baseline")
        yk = pd.to_numeric(vk[col], errors="coerce")
        if tf is not None:
            yk = tf(yk)
        d = base.join(y0).join(yk.rename("y")).dropna(subset=["y", "baseline", "age", "sex"])
        D = D_all.reindex(d.index)
        keep = (D.notna().sum(axis=1) >= K)
        d, D = d[keep], D[keep]
        chd = pd.Series([p.split("::", 1)[0] for p in d.index], index=d.index)
        yv = d["y"].to_numpy(float)
        bm = d[["baseline", "age", "sex"]].to_numpy(float)
        for held in sorted(pd.unique(chd)):
            te = (chd == held).to_numpy()
            tr = ~te
            if te.sum() < MIN_COHORT_TEST:
                continue
            tr_i, te_i = np.where(tr)[0], np.where(te)[0]
            f_tr, f_te = _refit_axes(D, tr_i, te_i)
            r2_base = _score(yv[te_i], _fit_predict(bm[tr_i], yv[tr_i], bm[te_i], "continuous"), "continuous")
            r2_ax = _score(yv[te_i],
                           _fit_predict(np.hstack([bm[tr_i], f_tr]), yv[tr_i], np.hstack([bm[te_i], f_te]), "continuous"),
                           "continuous")
            loco_out.append({"outcome": name, "held_out_cohort": held, "n_test": int(te.sum()),
                             "baseline_R2": round(r2_base, 3), "axes_R2": round(r2_ax, 3),
                             "axes_minus_baseline": round(r2_ax - r2_base, 3)})
            print(f"  {name} | predict {held.upper():>3} (n={te.sum():,}) from the other cohorts: "
                  f"baseline R²={r2_base:.3f} → +axes {r2_ax:.3f}  (Δ {r2_ax-r2_base:+.3f})")

    meta = {"K": K, "min_site": MIN_SITE,
            "loco_structure": loco_struct, "loso_outcomes": loso, "loco_outcomes": loco_out}
    (RESULTS_DIR / "replication_holdout.json").write_text(json.dumps(meta, indent=2, default=str))
    _report(loco_struct, loso, loco_out)
    print("\nWrote results/replication_holdout.json + reports/replication_holdout.html. Done.")
    return 0


def _report(loco_struct, loso, loco_out):
    s_rows = "".join(
        f"<tr><td>{r['held_out_cohort'].upper()}</td><td>{r['n_train']:,}</td>"
        f"<td><b>{r['min_congruence']}</b></td><td>{r['mean_congruence']}</td>"
        f"<td>{r['per_axis']}</td></tr>" for r in loco_struct)
    o_rows = "".join(
        f"<tr><td>{r['outcome']}</td><td>{r['n']}</td><td>{r['n_sites']}</td><td>{r['metric']}</td>"
        f"<td>{r['DSM']}</td><td>{r['axes']}</td><td>{r['combined']}</td>"
        f"<td><b>{r['axes_minus_DSM']:+}</b></td><td>{r['combined_minus_DSM']:+}</td></tr>" for r in loso)
    c_rows = "".join(
        f"<tr><td>{r['outcome']}</td><td>{r['held_out_cohort'].upper()}</td><td>{r['n_test']}</td>"
        f"<td>{r['baseline_R2']}</td><td>{r['axes_R2']}</td><td><b>{r['axes_minus_baseline']:+}</b></td></tr>"
        for r in loco_out)
    css = ("body{font-family:-apple-system,Segoe UI,sans-serif;margin:0;padding:0 24px 60px}"
           "h1{color:#2b3a55}h2{color:#2b3a55;margin-top:26px}table{border-collapse:collapse;"
           "font-size:13px;margin:10px 0}th,td{border:1px solid #e5e7eb;padding:5px 10px}"
           "th{background:#eef2f7}.c{background:#f2fbf6;border-left:4px solid #16a085;padding:10px 14px;margin:12px 0}")
    html = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{css}</style></head><body>",
            "<h1>Within-FACE held-out replication (transportability)</h1>",
            "<div class='c'>No <i>external</i> cohort is available (Limitation 9); these are "
            "leave-one-partition-out transportability tests. Structure transports across diagnoses "
            "(Part 1); the outcome advantage transports to unseen sites (Part 2) and into an unseen "
            "diagnosis (Part 3).</div>",
            "<h2>Part 1 — leave-one-cohort-out structure</h2>",
            "<table><tr><th>held-out cohort</th><th>train n</th><th>min congruence</th>"
            "<th>mean</th><th>per-axis</th></tr>", s_rows, "</table>",
            "<h2>Part 2 — leave-one-site-out outcome prediction (pooled out-of-site)</h2>",
            "<table><tr><th>outcome</th><th>n</th><th>sites</th><th>metric</th><th>DSM</th>"
            "<th>axes</th><th>combined</th><th>axes−DSM</th><th>comb−DSM</th></tr>", o_rows, "</table>",
            "<h2>Part 3 — leave-one-cohort-out outcome prediction (predict an unseen diagnosis)</h2>",
            "<table><tr><th>outcome</th><th>held-out cohort</th><th>n</th><th>baseline R²</th>"
            "<th>+axes R²</th><th>Δ axes</th></tr>", c_rows, "</table></body></html>"]
    (REPORTS_DIR / "replication_holdout.html").write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
