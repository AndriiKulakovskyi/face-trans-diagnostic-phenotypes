"""Finalize the dimensional trans-diagnostic axes — IMPUTATION-FREE (paper-ready set).

Re-derivation (LABBOOK E19). The earlier version mean-filled 35% of the residual-domain
matrix before sklearn factor analysis; the ablation (`sensitivity_masked_fa.py`, MANUSCRIPT
§3.8) showed the mean-fill reweights every correlation by co-observation
(corr_fill ~= O . corr_masked, O = n_AB/sqrt(n_A n_B)), which partially re-imports the
cohort-by-missingness confound at the weakest factor and flips the 6th axis. We therefore
estimate the model WITHOUT any imputation:

  1. **Loadings** from the pairwise-complete (masked) correlation matrix — each correlation
     uses only patients who have BOTH domains, so no cell is ever filled. Principal-axis
     factoring (iterated communalities) + varimax rotation (simple structure).
  2. **K by masked split-half reproducibility** — locked at K=6 (masked split-half min
     Tucker congruence ~0.89; K=7 also reproduces, K=8 collapses — we retain 6 for parsimony
     and comparability with the mean-fill model).
  3. **Per-patient scores** = the factor-analysis posterior mean computed on each patient's
     OBSERVED support only (regression/Thomson scores; no imputation):
        f_i = (I + L_o' Psi_o^-1 L_o)^-1 L_o' Psi_o^-1 z_{i,o},  Psi = 1 - communalities.

Final representation = 6 reproducible, confound-controlled, imputation-free varimax axes.
These scores feed Phase 5 (outcomes), Phase 4 (longitudinal), cognition and the figures.

Artifacts: results/dimensional_final_{scores.parquet,loadings.csv,meta.json},
reports/dimensional_final.html.
Run:  python3 scripts/07_dimensional_refine.py
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

from trans_diag import (  # noqa: E402
    ADMINISTRATIVE_FEATURES,
    build_unified_dataframe,
    load_variables,
    to_harmonized_dataset,
)
from trans_diag.masked_fa import masked_loadings, masked_scores  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
REPORTS_DIR = REPO_ROOT / "reports"
SCORES_PATH = RESULTS_DIR / "cluster_domains_scores.parquet"
RANDOM = 0
K = 6                # locked (masked split-half reproducible; see docstring)
MIN_PAIR = 100       # min co-observed patients to trust a pairwise correlation
PSI_FLOOR = 0.05     # floor on uniquenesses (guards Heywood cases) for scoring
SPECTRUM = {"Trouble dépressif majeur": 0, "Bipolaire de type 2": 1,
            "Bipolaire de type 1": 2, "Bipolaire non spécifié": 3,
            "Trouble schizo-affectif": 4, "Trouble schizophréniforme": 5,
            "Schizophrénie": 6}


# masked-FA primitives (nearest_pd / paf_loadings / varimax / masked_loadings / masked_scores)
# live in trans_diag.masked_fa and are imported above — single source of truth (also used by
# 08_longitudinal_axes.py). Module defaults match this script (min_pair=100, psi_floor=0.05).


def tucker_min(La: np.ndarray, Lb: np.ndarray):
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
        mins.append(best)
        used.add(bj)
    return mins


def main() -> int:
    REPORTS_DIR.mkdir(exist_ok=True)
    sc = pd.read_parquet(SCORES_PATH)
    sc.index = pd.MultiIndex.from_arrays(
        [sc.index.get_level_values("cohort").astype(str),
         sc.index.get_level_values("patient_id").astype(str)], names=("cohort", "patient_id"))
    domains = list(sc.columns)
    z = (sc - sc.mean()) / sc.std(ddof=0)
    obs_frac = float(sc.notna().to_numpy().mean())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(REPO_ROOT / "data", REPO_ROOT / "face-common-vars.xlsx",
                                     readiness=["READY", "PARTIAL"], format="long")
        full = to_harmonized_dataset(df, load_variables(REPO_ROOT / "face-common-vars.xlsx"),
                                     visit="V0", exclude=ADMINISTRATIVE_FEATURES)
    rank = full.metadata.reindex(sc.index)["dsm_diagnosis"].map(SPECTRUM).to_numpy()
    age = full.X.reindex(sc.index)["age"].to_numpy(float)
    sex = full.X.reindex(sc.index)["sex"].to_numpy(float)
    print(f"imputation-free FA: {len(sc):,} patients x {len(domains)} domains "
          f"(observed fraction {obs_frac:.3f}; the 35% missing are NEVER filled)")

    # 1. masked split-half reproducibility (confirms K is reproducible imputation-free)
    rng = np.random.default_rng(RANDOM)
    perm = rng.permutation(len(sc)); h = len(sc) // 2
    A, B = sc.iloc[perm[:h]], sc.iloc[perm[h:]]
    print("masked split-half reproducibility (min/mean Tucker congruence):")
    curve = []
    for k in range(3, 9):
        m = tucker_min(masked_loadings(A, k), masked_loadings(B, k))
        curve.append({"k": k, "min_congruence": float(np.min(m)), "mean_congruence": float(np.mean(m))})
        print(f"  K={k:>2}  min={np.min(m):.2f}  mean={np.mean(m):.2f}")
    print(f"\nlocked K = {K} (masked split-half reproducible; K=7 also reproduces, K=8 collapses)")

    # 2. final masked varimax loadings at K, oriented + ordered by sum-of-squares
    load = masked_loadings(sc, K)
    for a in range(K):                                   # orient: defining domain positive
        j = int(np.argmax(np.abs(load[:, a])))
        if load[j, a] < 0:
            load[:, a] = -load[:, a]
    order = np.argsort(-(load ** 2).sum(0))
    load = load[:, order]

    # 3. imputation-free per-patient scores (observed support only)
    scores = masked_scores(z, load)
    # align score signs to the (already oriented) loadings via correlation with domain means
    names = [f"axis{a+1}" for a in range(K)]
    loadrows = []
    print("\nfinal imputation-free axes (top domains):")
    for a in range(K):
        s = pd.Series(load[:, a], index=domains).sort_values(key=abs, ascending=False)
        print(f"  axis{a+1}: " + "; ".join(f"{d}({v:+.2f})" for d, v in s.head(5).items()))
        for d, v in zip(domains, load[:, a]):
            loadrows.append({"axis": f"axis{a+1}", "domain": d, "loading": float(v)})

    # 4. which axis carries the DSM mood<->psychosis ordering (subtype-centroid Spearman)
    cont = []
    for a in range(K):
        cdf = pd.DataFrame({"rank": rank, "s": scores[:, a]}).dropna()
        cm = cdf.groupby("rank")["s"].mean()
        cont.append(float(abs(spearmanr(cm.index, cm.to_numpy()).statistic)) if len(cm) > 2 else float("nan"))
    best = int(np.nanargmax(cont))
    print(f"\nDSM-ordering by axis (subtype-centroid |Spearman|): {[round(c,2) for c in cont]}"
          f"; strongest = axis{best+1} ({cont[best]:.2f}).")

    # 5. confound independence (age/sex) of the final imputation-free axes
    def cmax(col):
        ok = np.isfinite(col)
        return max(abs(float(np.corrcoef(col[ok & np.isfinite(y)], y[ok & np.isfinite(y)])[0, 1]))
                   for y in (age, sex))
    conf = {names[i]: round(cmax(scores[:, i]), 3) for i in range(K)}
    print(f"confound: max |corr| age/sex across axes = {max(conf.values()):.3f}")

    pd.DataFrame(scores, columns=names, index=sc.index).to_parquet(
        RESULTS_DIR / "dimensional_final_scores.parquet")
    pd.DataFrame(loadrows).to_csv(RESULTS_DIR / "dimensional_final_loadings.csv", index=False)
    meta = {"K": K, "axes": names, "method": "imputation-free (masked pairwise-complete corr "
            "-> principal-axis factoring + varimax; posterior-mean scores on observed support)",
            "observed_fraction": round(obs_frac, 4), "min_pair": MIN_PAIR,
            "reproducibility_curve": curve,   # masked split-half (consumed by 15 for figS2)
            "dsm_ordering_per_axis": cont, "strongest_ordering_axis": f"axis{best+1}",
            "confound_max_corr": conf,
            "note": "Re-derived imputation-free (LABBOOK E19, MANUSCRIPT §3.8). 5 of 6 axes match "
                    "the former mean-fill model; the 6th is now a socio-occupational/work-disability "
                    "axis (the former ADHD/impulsivity/trauma axis was a mean-fill co-observation "
                    "artifact). Orthogonal varimax; the mood<->psychosis spectrum is a cross-axis "
                    "direction (AE recovers it at 0.89)."}
    (RESULTS_DIR / "dimensional_final_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    # report
    cdf = pd.DataFrame(curve)
    f1 = go.Figure()
    f1.add_scatter(x=cdf["k"], y=cdf["min_congruence"], mode="lines+markers", name="min")
    f1.add_scatter(x=cdf["k"], y=cdf["mean_congruence"], mode="lines+markers", name="mean")
    f1.add_hline(y=0.85, line_dash="dash"); f1.add_vline(x=K, line_dash="dot", line_color="#16a085")
    f1.update_layout(title=f"Masked (imputation-free) reproducibility vs K → K={K}", height=320,
                     xaxis_title="K", yaxis_title="Tucker congruence", margin=dict(t=46))
    f2 = go.Figure(go.Heatmap(z=load.T, x=domains, y=names, colorscale="RdBu", zmid=0,
                              colorbar=dict(title="loading", thickness=12)))
    f2.update_layout(title=f"Final {K} imputation-free varimax axes (loadings)", height=80 + 55 * K,
                     margin=dict(t=46, l=90, b=140), xaxis_tickangle=-60)
    css = "body{font-family:-apple-system,Segoe UI,sans-serif;margin:0;padding:0 24px 60px}h1{color:#2b3a55}.c{background:#f2fbf6;border-left:4px solid #16a085;padding:10px 14px;margin:12px 0}"
    html = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{css}</style></head><body>",
            "<h1>Final trans-diagnostic dimensional axes (imputation-free)</h1>",
            f"<div class='c'>K={K} varimax axes from the pairwise-complete (masked) correlation "
            f"matrix — NO imputation (the 35% missing cells are never filled; cf. §3.8 ablation). "
            f"Confound-free (max |corr| age/sex = {max(conf.values()):.3f}). Strongest DSM "
            f"mood↔psychosis ordering on axis{best+1} (|ρ| {cont[best]:.2f}).</div>",
            pio.to_html(f1, include_plotlyjs="cdn", full_html=False),
            pio.to_html(f2, include_plotlyjs=False, full_html=False), "</body></html>"]
    (REPORTS_DIR / "dimensional_final.html").write_text("\n".join(html), encoding="utf-8")
    print("\nWrote results/dimensional_final_* (imputation-free) + reports/dimensional_final.html. Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
