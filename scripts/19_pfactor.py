"""§4.6 (exploratory) — the trans-diagnostic general factor ('p'-factor) as a severity index.

Our primary model uses ORTHOGONAL varimax axes — uncorrelated by construction, so there is no
general factor *among the axis scores*. But the axes can still share variance at the domain level,
and that shared variance is the general psychopathology factor ('p'). We test and extract it
imputation-free:

  1. **Does a general factor exist?** Apply an OBLIQUE (promax) rotation to the K=7 masked-FA
     solution and inspect the factor-correlation matrix Phi: broadly positive inter-factor
     correlations ⇒ a general factor. Quantify with the mean off-diagonal of Phi.
  2. **Extract it.** The first *unrotated* principal-axis factor of the masked correlation matrix
     is the dominant shared dimension (the classic general factor that varimax then splits into
     group factors); orient it so higher = more severe.
  3. **Score it** per patient (masked posterior mean → p-score), and validate: correlation with the
     seven orthogonal axes (a general factor should load on all), and a *one-number* outcome
     head-to-head vs the 7-subtype DSM (can a single severity score rival the diagnosis?).

Artifacts: results/pfactor.json, results/reports/pfactor.html.
Run:  python3 scripts/19_pfactor.py
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

from trans_diag import AXIS_NAMES, build_unified_dataframe  # noqa: E402
from trans_diag.masked_fa import (  # noqa: E402
    masked_correlation,
    masked_scores,  # noqa: E402
    paf_loadings,
    varimax,
)
from trans_diag.outcomes import (  # noqa: E402
    OUTCOMES,
    apply_outcome_tf,
    cv_metric,
)

RESULTS_DIR = REPO_ROOT / "results"
REPORTS_DIR = REPO_ROOT / "results" / "reports"
K = json.loads((RESULTS_DIR / "dimensional_final_meta.json").read_text())["K"]  # locked by 07


def promax_phi(L: np.ndarray, power: int = 4) -> np.ndarray:
    """Factor-correlation matrix Phi from a promax (oblique) rotation of loadings L, normalized to
    a correlation matrix. Off-diagonals > 0 ⇒ the factors share a general dimension."""
    Lv = varimax(L)
    H = Lv * np.abs(Lv) ** (power - 1)            # promax target (sign-preserving power)
    U = np.linalg.lstsq(Lv, H, rcond=None)[0]
    U = U * np.sqrt(np.maximum(np.diag(np.linalg.inv(U.T @ U)), 1e-12))
    Phi = np.linalg.inv(U.T @ U)
    d = np.sqrt(np.clip(np.diag(Phi), 1e-12, None))
    return Phi / np.outer(d, d)


def main() -> int:
    REPORTS_DIR.mkdir(exist_ok=True)
    sc = pd.read_parquet(RESULTS_DIR / "cluster_domains_scores.parquet")
    sc.index = pd.MultiIndex.from_arrays(
        [sc.index.get_level_values("cohort").astype(str),
         sc.index.get_level_values("patient_id").astype(str)], names=("cohort", "patient_id"))
    domains = list(sc.columns)
    z = (sc - sc.mean()) / sc.std(ddof=0)

    R = masked_correlation(sc)

    # 1. does a general factor exist? — oblique factor correlations
    L6 = paf_loadings(R, K)
    Phi = promax_phi(L6)
    off = Phi[~np.eye(K, dtype=bool)]
    mean_off, pos_frac = float(off.mean()), float((off > 0).mean())
    print(f"oblique (promax) inter-factor correlations: mean off-diagonal Phi = {mean_off:.2f} "
          f"({pos_frac*100:.0f}% positive) → {'a general factor is present' if mean_off > 0.1 else 'weak/no general factor'}")

    # 2. extract the general factor = first unrotated PAF factor, oriented severity-positive
    g = paf_loadings(R, 1)[:, 0]
    if np.nansum(g) < 0:
        g = -g
    gpos = int((g > 0.2).sum())
    var_general = float(np.sum(g ** 2) / np.sum(np.diag(R)))   # share of total variance
    top = pd.Series(g, index=domains).sort_values(key=abs, ascending=False).head(10)
    print(f"general factor: loads positively on {gpos}/{len(domains)} domains; "
          f"explains {var_general*100:.0f}% of total domain variance")
    print("  top domains: " + "; ".join(f"{d}({v:+.2f})" for d, v in top.items()))

    # 3. masked per-patient p-score severity score
    p = masked_scores(z.to_numpy(float), g.reshape(-1, 1))[:, 0]
    pf = pd.Series(p, index=sc.index, name="p_score")

    # validation A: correlation with the seven orthogonal axes (a general factor loads on all)
    axes = pd.read_parquet(RESULTS_DIR / "dimensional_final_scores.parquet")
    axes.index = pf.index
    axes.columns = AXIS_NAMES
    corr_axes = {}
    print("\ncorrelation of p-score with the seven orthogonal axes:")
    for a in AXIS_NAMES:
        m = pf.notna() & axes[a].notna()
        r = float(np.corrcoef(pf[m], axes[a][m])[0, 1])
        corr_axes[a] = round(r, 2)
        print(f"  {a:20s} r={r:+.2f}")

    # validation B: one-number outcome head-to-head vs the 7-subtype DSM
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(REPO_ROOT / "data", REPO_ROOT / "data" / "face-common-vars.xlsx",
                                     readiness=["READY", "PARTIAL"], format="long")
    df["pid"] = df["cohort"].str.lower() + "::" + df["usubjid_patients"].astype(str)
    pf_pid = pd.Series(pf.to_numpy(), index=(pf.index.get_level_values("cohort") + "::" +
                                             pf.index.get_level_values("patient_id")), name="p_score")
    v0 = df[df["visit"] == "V0"].set_index("pid")
    vk = df[df["visit"] == "V1"].set_index("pid")
    base = pd.DataFrame(index=v0.index)
    base["age"] = pd.to_numeric(v0["age"], errors="coerce")
    base["sex"] = pd.to_numeric(v0["sex"], errors="coerce")
    dsm = pd.get_dummies(v0["arm"].astype(str), drop_first=True).astype(float)
    dsm_cols = list(dsm.columns)
    base = base.join(dsm).join(pf_pid)

    print("\none-number head-to-head (p-score vs the 7-subtype DSM):")
    hh = []
    for name, kind, col, tf in OUTCOMES:
        if col not in df.columns:
            continue
        y0 = pd.to_numeric(v0[col], errors="coerce").rename("baseline")
        yk = pd.to_numeric(vk[col], errors="coerce").reindex(y0.index)
        if tf is not None:
            yk = apply_outcome_tf(y0, yk, tf)
        d = base.join(y0).join(yk.rename("y")).dropna(subset=["y", "baseline", "age", "sex", "p_score"])
        if kind == "binary" and (d["y"].nunique() < 2 or d["y"].mean() < 0.02):
            continue
        if len(d) < 200:
            continue
        bc = ["baseline", "age", "sex"]
        yv = d["y"].to_numpy(float)
        m_dsm = cv_metric(d[bc + dsm_cols].to_numpy(float), yv, kind)
        m_p = cv_metric(d[bc + ["p_score"]].to_numpy(float), yv, kind)
        m_both = cv_metric(d[bc + dsm_cols + ["p_score"]].to_numpy(float), yv, kind)
        metric = "R2" if kind == "continuous" else "AUC"
        hh.append({"outcome": name, "n": len(d), "metric": metric, "DSM": round(m_dsm, 3),
                   "p_score": round(m_p, 3), "DSM+p_score": round(m_both, 3),
                   "p_score_minus_DSM": round(m_p - m_dsm, 3)})
        print(f"  {name}: n={len(d)} {metric}  DSM(7 subtypes)={m_dsm:.3f}  p-score(1 number)={m_p:.3f}  "
              f"both={m_both:.3f}  (p-score−DSM {m_p-m_dsm:+.3f})")

    meta = {"K": K, "oblique_phi_mean_offdiag": round(mean_off, 3), "phi_positive_frac": round(pos_frac, 3),
            "general_factor_var_share": round(var_general, 3),
            "general_factor_top_domains": {d: round(float(v), 3) for d, v in top.items()},
            "corr_with_axes": corr_axes, "n_scored": int(pf.notna().sum()),
            "headtohead_p_score": hh,
            "note": "Exploratory general ('p') factor. Phi mean off-diagonal quantifies inter-axis "
                    "correlation (g-saturation); the index is the first unrotated masked PAF factor, "
                    "scored masked. One-number severity vs the 7-subtype DSM."}
    (RESULTS_DIR / "pfactor.json").write_text(json.dumps(meta, indent=2, default=str))
    _report(mean_off, pos_frac, var_general, gpos, len(domains), top, corr_axes, hh)
    print("\nWrote results/pfactor.json + results/reports/pfactor.html. Done.")
    return 0


def _report(mean_off, pos_frac, var_general, gpos, ndom, top, corr_axes, hh):
    tr = "".join(f"<tr><td>{d}</td><td>{v:+.2f}</td></tr>" for d, v in top.items())
    ca = "".join(f"<tr><td>{a}</td><td>{r:+}</td></tr>" for a, r in corr_axes.items())
    hr = "".join(
        f"<tr><td>{r['outcome']}</td><td>{r['n']}</td><td>{r['metric']}</td><td>{r['DSM']}</td>"
        f"<td>{r['p_score']}</td><td>{r['DSM+p_score']}</td><td><b>{r['p_score_minus_DSM']:+}</b></td></tr>"
        for r in hh)
    css = ("body{font-family:-apple-system,Segoe UI,sans-serif;margin:0;padding:0 24px 60px}"
           "h1{color:#2b3a55}h2{color:#2b3a55;margin-top:22px}table{border-collapse:collapse;"
           "font-size:13px;margin:10px 0}th,td{border:1px solid #e5e7eb;padding:5px 10px}"
           "th{background:#eef2f7}.c{background:#eef6fb;border-left:4px solid #2b8cbe;padding:10px 14px;margin:12px 0}")
    html = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{css}</style></head><body>",
            "<h1>Trans-diagnostic general ('p') factor — is there one?</h1>",
            f"<div class='c'>Oblique inter-factor correlation: mean off-diagonal Phi = "
            f"<b>{mean_off:.2f}</b> ({pos_frac*100:.0f}% positive) → "
            f"{'a general factor is present' if mean_off>0.1 else 'weak general factor'}. The general "
            f"factor (first unrotated masked PAF factor) loads positively on {gpos}/{ndom} domains and "
            f"explains {var_general*100:.0f}% of total domain variance.</div>",
            "<h2>Top domains of the general factor</h2>",
            "<table><tr><th>domain</th><th>loading</th></tr>", tr, "</table>",
            "<h2>Correlation with the seven orthogonal axes</h2>",
            "<table><tr><th>axis</th><th>r with p-score</th></tr>", ca, "</table>",
            "<h2>One-number head-to-head: p-score vs the 7-subtype DSM</h2>",
            "<table><tr><th>outcome</th><th>n</th><th>metric</th><th>DSM (7 subtypes)</th>"
            "<th>p-score (1 number)</th><th>DSM+p-score</th><th>p-score−DSM</th></tr>", hr, "</table>",
            "</body></html>"]
    (REPORTS_DIR / "pfactor.html").write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
