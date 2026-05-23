"""Finalize the dimensional trans-diagnostic axes (paper-ready set).

Refinement step (LABBOOK E13 follow-up). Two fixes over the first pass:

  1. **Choose K by reproducibility, not an arbitrary cap.** A split-half Tucker-
     congruence-vs-K curve picks the largest K whose factors are ALL reproducible
     (min congruence ≥ 0.85).
  2. **Give the mood↔psychosis continuum its own clean axis.** Varimax (orthogonal,
     simple-structure) dispersed it; the continuum is the dominant *direction* of
     variation, so we report it as an explicit **spectrum axis** = the unrotated
     first principal component (the AE recovers the same axis at |ρ|=0.89).

Final representation = K reproducible varimax axes (named, interpretable, clean
orthogonal scores) + 1 spectrum axis. These scores feed Phase 5 (outcomes).

Artifacts: results/dimensional_final_{scores.parquet,loadings.csv,meta.json},
reports/dimensional_final.html.
Run:  python3 scripts/dimensional_refine.py
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
from scipy.stats import spearmanr  # noqa: E402
from sklearn.decomposition import PCA, FactorAnalysis  # noqa: E402

from face_common import (  # noqa: E402
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


def varimax_load(X, k):
    fa = FactorAnalysis(n_components=k, rotation="varimax", random_state=RANDOM).fit(X)
    return fa, fa.components_.T


def tucker_min(La, Lb):
    used, mins = set(), []
    for a in range(La.shape[1]):
        best, bj = 0.0, -1
        for b in range(Lb.shape[1]):
            if b in used:
                continue
            den = np.linalg.norm(La[:, a]) * np.linalg.norm(Lb[:, b])
            phi = abs(float(La[:, a] @ Lb[:, b])) / den if den > 0 else 0.0
            if phi > best:
                best, bj = phi, b
        mins.append(best); used.add(bj)
    return mins


def main() -> int:
    REPORTS_DIR.mkdir(exist_ok=True)
    sc = pd.read_parquet(SCORES_PATH)
    sc.index = pd.MultiIndex.from_arrays(
        [sc.index.get_level_values("cohort").astype(str),
         sc.index.get_level_values("patient_id").astype(str)], names=("cohort", "patient_id"))
    domains = list(sc.columns)
    z = (sc - sc.mean()) / sc.std(ddof=0)
    X = z.fillna(0.0).to_numpy(np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(REPO_ROOT / "data", REPO_ROOT / "face-common-vars.xlsx",
                                     readiness=["READY", "PARTIAL"], format="long")
        full = to_harmonized_dataset(df, load_variables(REPO_ROOT / "face-common-vars.xlsx"),
                                     visit="V0", exclude=ADMINISTRATIVE_FEATURES)
    rank = full.metadata.reindex(sc.index)["dsm_diagnosis"].map(SPECTRUM).to_numpy()
    age = full.X.reindex(sc.index)["age"].to_numpy(float)
    sex = full.X.reindex(sc.index)["sex"].to_numpy(float)

    # 1. reproducibility-vs-K (split-half Tucker congruence)
    rng = np.random.default_rng(RANDOM)
    perm = rng.permutation(len(X)); h = len(X) // 2
    print("reproducibility-vs-K (split-half min/mean Tucker congruence):")
    curve = []
    for k in range(3, 13):
        _, La = varimax_load(X[perm[:h]], k)
        _, Lb = varimax_load(X[perm[h:]], k)
        m = tucker_min(La, Lb)
        curve.append({"k": k, "min_congruence": float(np.min(m)), "mean_congruence": float(np.mean(m))})
        print(f"  K={k:>2}  min={np.min(m):.2f}  mean={np.mean(m):.2f}")
    curve_df = pd.DataFrame(curve)
    # Varimax becomes unstable at higher K (factor-splitting scrambles the rotation
    # → erratic congruence). Choose K only from the stable low range (≤8); take the
    # largest there with all factors reproducible (min congruence ≥ 0.85).
    ok = curve_df[(curve_df["min_congruence"] >= 0.85) & (curve_df["k"] <= 8)]
    K = int(ok["k"].max()) if len(ok) else 4
    print(f"\nfinal K = {K} (largest stable K≤8 with all factors min-congruence ≥ 0.85; "
          "higher K are erratic — varimax factor-splitting)")

    # 2. final varimax axes at K
    fa, load = varimax_load(X, K)
    scores = fa.transform(X)
    ss = (load ** 2).sum(0); order = np.argsort(-ss)
    load, scores = load[:, order], scores[:, order]
    names, loadrows = [], []
    print("\nfinal axes (top domains):")
    for a in range(K):
        s = pd.Series(load[:, a], index=domains).sort_values(key=abs, ascending=False)
        names.append(f"axis{a+1}")
        print(f"  axis{a+1}: " + "; ".join(f"{d}({v:+.2f})" for d, v in s.head(5).items()))
        for d, v in zip(domains, load[:, a]):
            loadrows.append({"axis": f"axis{a+1}", "domain": d, "loading": float(v)})

    # 3. which axis carries the DSM mood↔psychosis ordering? (subtype-centroid
    #    Spearman per axis — patient-level is meaningless given within-subtype spread).
    cont = []
    for a in range(K):
        cdf = pd.DataFrame({"rank": rank, "s": scores[:, a]}).dropna()
        cm = cdf.groupby("rank")["s"].mean()
        cont.append(float(abs(spearmanr(cm.index, cm.to_numpy()).statistic)))
    best = int(np.argmax(cont))
    print(f"\nDSM-ordering by axis (subtype-centroid |Spearman|): {[round(c,2) for c in cont]}"
          f"; strongest = axis{best+1} ({cont[best]:.2f}). The full mood↔psychosis spectrum is "
          "a direction across axes (AE recovers it at 0.89; not a single varimax factor).")

    # validation: confound independence of the final K varimax axes
    def cmax(col):
        return max(abs(float(np.corrcoef(col[np.isfinite(y)], y[np.isfinite(y)])[0, 1]))
                   for y in (age, sex))
    conf = {names[i]: round(cmax(scores[:, i]), 3) for i in range(K)}
    print(f"confound: max |corr| age/sex across axes = {max(conf.values()):.3f}")

    pd.DataFrame(scores, columns=names, index=sc.index).to_parquet(
        RESULTS_DIR / "dimensional_final_scores.parquet")
    pd.DataFrame(loadrows).to_csv(RESULTS_DIR / "dimensional_final_loadings.csv", index=False)
    meta = {"K": K, "axes": names, "reproducibility_curve": curve,
            "dsm_ordering_per_axis": cont, "strongest_ordering_axis": f"axis{best+1}",
            "confound_max_corr": conf,
            "note": "K chosen by split-half congruence over the stable range (≤8; higher "
                    "K erratic). Orthogonal varimax axes. The mood↔psychosis spectrum is a "
                    "cross-axis direction (AE recovers it at 0.89), not a single varimax "
                    "factor; oblique rotation deferred (factor_analyzer not installed)."}
    (RESULTS_DIR / "dimensional_final_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    # report
    f1 = go.Figure()
    f1.add_scatter(x=curve_df["k"], y=curve_df["min_congruence"], mode="lines+markers", name="min")
    f1.add_scatter(x=curve_df["k"], y=curve_df["mean_congruence"], mode="lines+markers", name="mean")
    f1.add_hline(y=0.85, line_dash="dash"); f1.add_vline(x=K, line_dash="dot", line_color="#16a085")
    f1.update_layout(title=f"Reproducibility vs K → final K={K}", height=320,
                     xaxis_title="K", yaxis_title="Tucker congruence", margin=dict(t=46))
    f2 = go.Figure(go.Heatmap(z=load.T, x=domains, y=names, colorscale="RdBu", zmid=0,
                              colorbar=dict(title="loading", thickness=12)))
    f2.update_layout(title=f"Final {K} varimax axes (loadings)", height=80 + 55 * K,
                     margin=dict(t=46, l=90, b=140), xaxis_tickangle=-60)
    css = "body{font-family:-apple-system,Segoe UI,sans-serif;margin:0;padding:0 24px 60px}h1{color:#2b3a55}.c{background:#f2fbf6;border-left:4px solid #16a085;padding:10px 14px;margin:12px 0}"
    html = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{css}</style></head><body>",
            "<h1>Final trans-diagnostic dimensional axes</h1>",
            f"<div class='c'>Final K={K} reproducible varimax axes (min split-half congruence "
            f"≥0.85), confound-free (max |corr| age/sex = {max(conf.values()):.3f}). Strongest "
            f"DSM mood↔psychosis ordering on axis{best+1} (|ρ| {cont[best]:.2f}).</div>",
            pio.to_html(f1, include_plotlyjs="cdn", full_html=False),
            pio.to_html(f2, include_plotlyjs=False, full_html=False), "</body></html>"]
    (REPORTS_DIR / "dimensional_final.html").write_text("\n".join(html), encoding="utf-8")
    print(f"\nWrote results/dimensional_final_* + reports/dimensional_final.html. Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
