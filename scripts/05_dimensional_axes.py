"""Dimensional trans-diagnostic axis model (classical / sklearn).

Step 3 (LABBOOK E12): the structure is dimensional, not discrete, so we describe
trans-diagnostic variation by a few INTERPRETABLE CONTINUOUS AXES instead of
clusters. Classical pipeline:

  residualized domain scores  (results/cluster_domains_scores.parquet)
    → standardize (z), mean-impute the gaps (light; the PyTorch AE companion does
      the no-imputation version)
    → parallel analysis (Horn) to choose the number of factors K
    → Factor Analysis with varimax rotation → K named axes (domain loadings)
    → validate: variance per axis, split-half reproducibility (Tucker congruence),
      confound independence (age/sex/cohort), and the DSM mood↔psychosis continuum.

Each patient gets a score on each axis — the trans-diagnostic representation we
carry into Phase 4 (do axes persist?) and Phase 5 (do axes beat DSM on outcomes?).

Artifacts: results/dimensional_axes_{loadings.csv,scores.parquet,meta.json},
reports/dimensional_axes.html.
Run:  python3 scripts/05_dimensional_axes.py [--max-k 8]
"""
from __future__ import annotations

import argparse
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
from scipy.stats import spearmanr  # noqa: E402
from sklearn.decomposition import FactorAnalysis  # noqa: E402

from trans_diag import (  # noqa: E402
    ADMINISTRATIVE_FEATURES,
    build_unified_dataframe,
    load_variables,
    to_harmonized_dataset,
)

RESULTS_DIR = REPO_ROOT / "results"
REPORTS_DIR = REPO_ROOT / "reports"
SCORES_PATH = RESULTS_DIR / "cluster_domains_scores.parquet"
RANDOM = 0
SPECTRUM = {"Trouble dépressif majeur": 0, "Bipolaire de type 2": 1,
            "Bipolaire de type 1": 2, "Bipolaire non spécifié": 3,
            "Trouble schizo-affectif": 4, "Trouble schizophréniforme": 5,
            "Schizophrénie": 6}


def parallel_analysis(X, n_iter=30, seed=0):
    """Horn's parallel analysis: keep factors whose eigenvalue beats the 95th pct
    of eigenvalues from column-permuted (independent) data."""
    rng = np.random.default_rng(seed)
    real = np.sort(np.linalg.eigvalsh(np.corrcoef(X.T)))[::-1]
    null = np.empty((n_iter, X.shape[1]))
    for i in range(n_iter):
        Xp = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
        null[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(Xp.T)))[::-1]
    thresh = np.percentile(null, 95, axis=0)
    return int((real > thresh).sum()), real, thresh


def varimax_loadings(X, k):
    fa = FactorAnalysis(n_components=k, rotation="varimax", random_state=RANDOM).fit(X)
    return fa, fa.components_.T          # loadings: features × k


def tucker_congruence(La, Lb):
    """Max Tucker's φ matching of two loading matrices (factor reproducibility)."""
    cong = []
    used = set()
    for a in range(La.shape[1]):
        best = 0.0
        for b in range(Lb.shape[1]):
            if b in used:
                continue
            num = abs(float(La[:, a] @ Lb[:, b]))
            den = np.linalg.norm(La[:, a]) * np.linalg.norm(Lb[:, b])
            phi = num / den if den > 0 else 0.0
            if phi > best:
                best, bj = phi, b
        cong.append(best); used.add(bj)
    return cong


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--max-k", type=int, default=8, help="cap on #factors for interpretability")
    args = ap.parse_args()
    REPORTS_DIR.mkdir(exist_ok=True)

    sc = pd.read_parquet(SCORES_PATH)
    sc.index = pd.MultiIndex.from_arrays(
        [sc.index.get_level_values("cohort").astype(str),
         sc.index.get_level_values("patient_id").astype(str)], names=("cohort", "patient_id"))
    domains = list(sc.columns)

    # standardize per domain (NaN-aware), then mean-impute the gaps (→ 0 in z-space)
    z = (sc - sc.mean()) / sc.std(ddof=0)
    X = z.fillna(0.0).to_numpy(np.float64)
    print(f"domain matrix: {X.shape[0]:,} patients × {X.shape[1]} domains")

    # subtype anchor
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(REPO_ROOT / "data", REPO_ROOT / "face-common-vars.xlsx",
                                     readiness=["READY", "PARTIAL"], format="long")
        full = to_harmonized_dataset(df, load_variables(REPO_ROOT / "face-common-vars.xlsx"),
                                     visit="V0", exclude=ADMINISTRATIVE_FEATURES)
    arm = full.metadata.reindex(sc.index)["dsm_diagnosis"]
    rank = arm.map(SPECTRUM).to_numpy()
    age = full.X.reindex(sc.index)["age"].to_numpy(float)
    sex = full.X.reindex(sc.index)["sex"].to_numpy(float)
    cohort = np.array([c for c, _ in sc.index])

    # K via parallel analysis (capped)
    k_pa, real_ev, thresh = parallel_analysis(X)
    k = min(k_pa, args.max_k)
    print(f"parallel analysis suggests K={k_pa}; using K={k} (cap {args.max_k})")

    # factor analysis + varimax
    fa, load = varimax_loadings(X, k)
    scores = fa.transform(X)                       # patients × k
    ss = (load ** 2).sum(0)                        # SS loadings per factor
    var_prop = ss / X.shape[1]
    order = np.argsort(-ss)                        # order axes by variance
    load, scores, var_prop = load[:, order], scores[:, order], var_prop[order]

    # name each axis by its top-loading domains
    print("\nAxes (top domains by |loading|):")
    names, loadings_rows = [], []
    for a in range(k):
        s = pd.Series(load[:, a], index=domains).sort_values(key=abs, ascending=False)
        top = s.head(6)
        names.append(f"axis{a+1}")
        sign = "↑" if top.iloc[0] > 0 else "↓"
        desc = "; ".join(f"{d}({'+' if v>0 else '−'}{abs(v):.2f})" for d, v in top.items())
        print(f"  axis{a+1} (var {var_prop[a]:.1%}) {sign}: {desc}")
        for d, v in zip(domains, load[:, a]):
            loadings_rows.append({"axis": f"axis{a+1}", "domain": d, "loading": float(v)})
    pd.DataFrame(loadings_rows).to_csv(RESULTS_DIR / "dimensional_axes_loadings.csv", index=False)

    # validation -----------------------------------------------------------
    # (1) mood↔psychosis continuum: which axis best orders the 7 DSM subtypes?
    cont = []
    sub = pd.DataFrame(scores, columns=names); sub["rank"] = rank
    cent = sub.dropna(subset=["rank"]).groupby("rank").mean()
    for a in range(k):
        rho = spearmanr(cent.index, cent[names[a]]).statistic
        cont.append(float(rho))
    mood_axis = int(np.argmax(np.abs(cont)))
    print(f"\nmood↔psychosis continuum: best axis = axis{mood_axis+1} "
          f"(|Spearman| {abs(cont[mood_axis]):.2f})")

    # (2) confound independence (|corr| with age/sex; eta² with cohort)
    def corr(a_idx, y):
        m = np.isfinite(y)
        return float(abs(np.corrcoef(scores[m, a_idx], y[m])[0, 1]))
    conf = {names[a]: {"age": corr(a, age), "sex": corr(a, sex)} for a in range(k)}
    max_conf = max(max(v.values()) for v in conf.values())
    print(f"confound independence: max |corr| with age/sex across axes = {max_conf:.3f}")

    # (3) split-half reproducibility (Tucker congruence)
    rng = np.random.default_rng(RANDOM)
    perm = rng.permutation(len(X)); half = len(X) // 2
    _, La = varimax_loadings(X[perm[:half]], k)
    _, Lb = varimax_loadings(X[perm[half:]], k)
    cong = tucker_congruence(La, Lb)
    print(f"split-half Tucker congruence (per axis): {[round(c,2) for c in cong]} "
          f"(>0.85 good)")

    # persist scores + meta
    pd.DataFrame(scores, columns=names, index=sc.index).to_parquet(
        RESULTS_DIR / "dimensional_axes_scores.parquet")
    meta = {"k_parallel_analysis": k_pa, "k_used": k,
            "variance_proportion": var_prop.tolist(),
            "continuum_spearman_per_axis": cont, "mood_axis": f"axis{mood_axis+1}",
            "confound": conf, "max_confound_corr": max_conf,
            "split_half_congruence": cong,
            "note": "classical FA (varimax); PyTorch masked AE companion gives the "
                    "no-imputation nonlinear version (next)."}
    (RESULTS_DIR / "dimensional_axes_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    _report(load, domains, names, var_prop, real_ev, thresh, k_pa, cent, cont, mood_axis, cong)
    print("\nWrote results/dimensional_axes_* + reports/dimensional_axes.html. Done.")
    return 0


def _report(load, domains, names, var_prop, real_ev, thresh, k_pa, cent, cont, mood_axis, cong):
    spec_lbl = {0: "MDD", 1: "BP-II", 2: "BP-I", 3: "BP-NOS", 4: "schizoaff",
                5: "schizophrenif", 6: "schizophr"}
    # loadings heatmap
    f1 = go.Figure(go.Heatmap(z=load.T, x=domains, y=[f"{n} ({v:.0%})" for n, v in zip(names, var_prop)],
                              colorscale="RdBu", zmid=0, colorbar=dict(title="loading", thickness=12)))
    f1.update_layout(title="Factor loadings (domains × axes)", height=80+60*len(names),
                     margin=dict(t=46, l=120, b=140), xaxis_tickangle=-60)
    # parallel analysis scree
    f2 = go.Figure()
    f2.add_scatter(y=real_ev[:20], mode="lines+markers", name="data")
    f2.add_scatter(y=thresh[:20], mode="lines", name="null 95th pct")
    f2.update_layout(title=f"Parallel analysis (K={k_pa} factors above null)", height=320,
                     xaxis_title="factor", yaxis_title="eigenvalue", margin=dict(t=46))
    # subtype centroids on the mood axis
    f3 = go.Figure(go.Scatter(x=[spec_lbl.get(int(r), str(r)) for r in cent.index],
                              y=cent[names[mood_axis]], mode="lines+markers"))
    f3.update_layout(title=f"DSM subtypes on {names[mood_axis]} (mood↔psychosis continuum, "
                     f"|ρ|={abs(cont[mood_axis]):.2f})", height=320,
                     xaxis_title="subtype (mood→psychosis)", yaxis_title="axis score", margin=dict(t=46))
    css = "body{font-family:-apple-system,Segoe UI,sans-serif;margin:0;padding:0 24px 60px}h1{color:#2b3a55}.c{background:#f2fbf6;border-left:4px solid #16a085;padding:10px 14px;margin:12px 0}"
    html = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{css}</style></head><body>",
            "<h1>Trans-diagnostic dimensional axes (classical / factor analysis)</h1>",
            f"<div class='c'>{len(names)} interpretable axes; split-half Tucker congruence "
            f"{[round(c,2) for c in cong]} (>0.85 = reproducible). The mood↔psychosis "
            f"continuum loads on {names[mood_axis]}.</div>",
            pio.to_html(f1, include_plotlyjs="cdn", full_html=False),
            pio.to_html(f2, include_plotlyjs=False, full_html=False),
            pio.to_html(f3, include_plotlyjs=False, full_html=False), "</body></html>"]
    (REPORTS_DIR / "dimensional_axes.html").write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
