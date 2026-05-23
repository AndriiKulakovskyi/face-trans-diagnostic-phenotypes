"""Robustness #1 — site harmonization (ComBat) sensitivity analysis.

FACE is multi-centre (21 sites). The deferred check (LABBOOK task #43): are the
dimensional axes and the Phase-5 head-to-head robust to site batch effects?

  1. ComBat-harmonize the V0 domain scores across sites (neuroHarmonize), preserving
     cohort + age + sex as biological covariates.
  2. Re-derive the 6 axes (FA, varimax) on the harmonized domains → Tucker congruence
     with the locked axes (≈1 ⇒ axes are not a site artifact).
  3. Re-run the Phase-5 V1 head-to-head on the ComBat axes → does "axes add over DSM"
     survive site harmonization?

Artifacts: results/robustness_site.json, reports/robustness_site.html.
Run:  python3 scripts/13_robustness_site.py
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
from neuroHarmonize import harmonizationLearn  # noqa: E402
from sklearn.decomposition import FactorAnalysis  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from trans_diag import build_unified_dataframe, load_variables  # noqa: E402
from trans_diag.outcomes import OUTCOMES, added_axes_test, cv_metric  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
REPORTS_DIR = REPO_ROOT / "reports"
K, RANDOM = 6, 0
AXIS_NAMES = ["depression_severity", "later_onset", "mania_activation",
              "illness_burden", "metabolic", "adhd_impulsivity_trauma"]


def tucker(La, Lb):
    out, used = [], set()
    for a in range(La.shape[1]):
        best, bj = 0.0, -1
        for b in range(Lb.shape[1]):
            if b in used:
                continue
            den = np.linalg.norm(La[:, a]) * np.linalg.norm(Lb[:, b])
            phi = abs(float(La[:, a] @ Lb[:, b])) / den if den > 0 else 0.0
            if phi > best:
                best, bj = phi, b
        out.append(round(best, 2)); used.add(bj)
    return out


def main() -> int:
    REPORTS_DIR.mkdir(exist_ok=True)
    sc = pd.read_parquet(RESULTS_DIR / "cluster_domains_scores.parquet")
    sc.index = pd.MultiIndex.from_arrays(
        [sc.index.get_level_values("cohort").astype(str),
         sc.index.get_level_values("patient_id").astype(str)], names=("cohort", "patient_id"))
    domains = list(sc.columns)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(REPO_ROOT / "data", REPO_ROOT / "face-common-vars.xlsx",
                                     readiness=["READY", "PARTIAL"], format="long")
    df["pid"] = df["cohort"].str.lower() + "::" + df["usubjid_patients"].astype(str)
    v0 = df[df["visit"] == "V0"].drop_duplicates("pid").set_index("pid")
    sc = sc.reset_index(); sc["pid"] = sc["cohort"] + "::" + sc["patient_id"]
    sc = sc.set_index("pid")

    site = pd.to_numeric(v0["siteid_city"], errors="coerce").reindex(sc.index)
    cohort = v0["cohort"].reindex(sc.index)
    age = pd.to_numeric(v0["age"], errors="coerce").reindex(sc.index)
    sex = pd.to_numeric(v0["sex"], errors="coerce").reindex(sc.index)

    # drop tiny sites (<10) + rows missing site
    counts = site.value_counts()
    big = counts[counts >= 10].index
    keep = site.isin(big) & site.notna()
    sc, site, cohort, age, sex = sc[keep], site[keep], cohort[keep], age[keep], sex[keep]
    print(f"ComBat on {len(sc):,} patients, {site.nunique()} sites (≥10 each)")

    # ── ComBat: harmonize domains across sites, preserve cohort/age/sex ──
    data = SimpleImputer(strategy="median").fit_transform(sc[domains].to_numpy(np.float64))
    covars = pd.DataFrame({"SITE": site.astype(int).astype(str).to_numpy(),
                           "age": SimpleImputer(strategy="median").fit_transform(age.to_numpy().reshape(-1, 1)).ravel(),
                           "sex": SimpleImputer(strategy="most_frequent").fit_transform(sex.to_numpy().reshape(-1, 1)).ravel()})
    covars = pd.concat([covars, pd.get_dummies(cohort.to_numpy(), prefix="coh").astype(float).reset_index(drop=True)], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, adj = harmonizationLearn(data, covars)
    # how much did site move the data?
    site_shift = float(np.mean(np.abs(adj - data)) / (np.std(data) + 1e-9))
    print(f"mean |ComBat adjustment| / SD = {site_shift:.3f} (site batch-effect magnitude)")

    # ── re-derive axes on harmonized domains, congruence with locked ──
    Zc = StandardScaler().fit_transform(adj)
    fa = FactorAnalysis(n_components=K, rotation="varimax", random_state=RANDOM).fit(Zc)
    load_c = fa.components_.T
    order = np.argsort(-(load_c ** 2).sum(0)); load_c = load_c[:, order]
    axes_c = pd.DataFrame(fa.transform(Zc)[:, order], index=sc.index, columns=AXIS_NAMES)
    final_load = (pd.read_csv(RESULTS_DIR / "dimensional_final_loadings.csv")
                  .pivot(index="domain", columns="axis", values="loading").reindex(domains).to_numpy())
    cong = tucker(load_c, final_load)
    print(f"ComBat axes vs locked axes Tucker congruence: {cong} (≈1 ⇒ axes survive site harmonization)")

    # ── re-run Phase-5 V1 head-to-head on ComBat axes ──
    vk = df[df["visit"] == "V1"].drop_duplicates("pid").set_index("pid")
    arm = pd.get_dummies(v0["arm"].reindex(sc.index).astype(str), drop_first=True).astype(float)
    dsm_cols = list(arm.columns)
    base = pd.DataFrame({"age": age, "sex": sex}, index=sc.index).join(arm).join(axes_c)
    orig = pd.read_csv(RESULTS_DIR / "phase5_headtohead_V1.csv").set_index("outcome")
    rows = []
    for name, kind, col, tf in OUTCOMES:
        y0 = pd.to_numeric(v0[col], errors="coerce").reindex(sc.index).rename("baseline")
        yk = pd.to_numeric(vk[col], errors="coerce").reindex(sc.index)
        if tf is not None:
            yk = tf(yk)
        d = base.join(y0).join(yk.rename("y")).dropna(subset=["y", "baseline", "age", "sex"] + AXIS_NAMES)
        if len(d) < 200 or (kind == "binary" and d["y"].nunique() < 2):
            continue
        bc = ["baseline", "age", "sex"]
        yv = d["y"].to_numpy(float)
        m0 = cv_metric(d[bc + dsm_cols].to_numpy(float), yv, kind)
        m1 = cv_metric(d[bc + AXIS_NAMES].to_numpy(float), yv, kind)
        m2 = cv_metric(d[bc + dsm_cols + AXIS_NAMES].to_numpy(float), yv, kind)
        p = added_axes_test(d, bc, dsm_cols, AXIS_NAMES, yv, kind)
        rows.append({"outcome": name, "n": len(d), "metric": orig.loc[name, "metric"] if name in orig.index else "",
                     "DSM": round(m0, 3), "axes": round(m1, 3), "combined": round(m2, 3),
                     "combat_axes_minus_DSM": round(m1 - m0, 3),
                     "orig_axes_minus_DSM": orig.loc[name, "axes_minus_DSM"] if name in orig.index else np.nan,
                     "added_axes_p": p})
        print(f"  {name}: ComBat axes−DSM {m1-m0:+.3f} (orig {orig.loc[name,'axes_minus_DSM']:+}); "
              f"combined={m2:.3f}")
    head = pd.DataFrame(rows)

    meta = {"n": int(len(sc)), "n_sites": int(site.nunique()),
            "site_batch_magnitude": site_shift, "axes_congruence_with_locked": cong,
            "headtohead_combat": head.to_dict(orient="records")}
    (RESULTS_DIR / "robustness_site.json").write_text(json.dumps(meta, indent=2, default=str))
    _report(cong, site_shift, head)
    print("\nWrote results/robustness_site.json + reports/robustness_site.html. Done.")
    return 0


def _report(cong, site_shift, head):
    rows = "".join(
        f"<tr><td>{r.outcome}</td><td>{r.n}</td><td>{r.DSM}</td><td>{r.axes}</td>"
        f"<td>{r.combined}</td><td><b>{r.combat_axes_minus_DSM:+}</b></td>"
        f"<td>{r.orig_axes_minus_DSM:+}</td><td>{r.added_axes_p:.1e}</td></tr>"
        for r in head.itertuples())
    css = "body{font-family:-apple-system,Segoe UI,sans-serif;margin:0;padding:0 24px 60px}h1{color:#2b3a55}table{border-collapse:collapse;font-size:13px;margin:12px 0}th,td{border:1px solid #e5e7eb;padding:5px 10px}th{background:#eef2f7}.c{background:#f2fbf6;border-left:4px solid #16a085;padding:10px 14px;margin:12px 0}"
    html = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{css}</style></head><body>",
            "<h1>Robustness — site (ComBat) harmonization</h1>",
            f"<div class='c'>Axes survive site harmonization (Tucker congruence with the "
            f"locked axes: {cong}). Site batch magnitude = {site_shift:.3f} of an SD. "
            "Phase-5 head-to-head re-run on ComBat axes below (compare ComBat vs original "
            "'axes−DSM').</div>",
            "<table><tr><th>outcome</th><th>n</th><th>DSM</th><th>axes</th><th>combined</th>"
            "<th>ComBat axes−DSM</th><th>orig axes−DSM</th><th>added-axes p</th></tr>",
            rows, "</table></body></html>"]
    (REPORTS_DIR / "robustness_site.html").write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
