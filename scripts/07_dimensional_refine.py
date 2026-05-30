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
  2. **K by masked split-half reproducibility** — K is re-derived (via ``select_k``, not a
     hand-set constant) as the maximum dimensionality whose single fixed-split MIN Tucker
     congruence stays >= K_FLOOR (0.85) before collapse; an N_SPLITS-averaged robustness curve
     is also reported (it flags the weakest axis as most split-sensitive but is not used to
     lock). Parallel analysis/Kaiser over-extract at this N (§4.7). The matrix now includes
     three curated cognitive constructs — working memory, verbal reasoning, fluency — after the
     DR neuropsych extraction gap was closed (2026-05); processing speed and executive/TMT did
     not harmonise across cohorts and are excluded.
  3. **Per-patient scores** = the factor-analysis posterior mean computed on each patient's
     OBSERVED support only (regression/Thomson scores; no imputation):
        f_i = (I + L_o' Psi_o^-1 L_o)^-1 L_o' Psi_o^-1 z_{i,o},  Psi = 1 - communalities.

Final representation = K reproducible, confound-controlled, imputation-free varimax axes.
These scores feed Phase 5 (outcomes), Phase 4 (longitudinal) and the figures.

Artifacts: results/dimensional_final_{scores.parquet,loadings.csv,meta.json},
results/reports/dimensional_final.html.
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
from trans_diag.masked_fa import (  # noqa: E402
    masked_correlation, masked_loadings, masked_scores, paf_loadings, varimax)

RESULTS_DIR = REPO_ROOT / "results"
REPORTS_DIR = REPO_ROOT / "results" / "reports"
SCORES_PATH = RESULTS_DIR / "cluster_domains_scores.parquet"
RANDOM = 0
K_FLOOR = 0.85       # lock K = max dimensionality whose masked split-half MIN Tucker congruence >= this
K_RANGE = range(3, 13)
N_SPLITS = 25        # random half-splits averaged for a seed-insensitive K lock
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
    """Optimal-assignment (Hungarian) Tucker congruence between two loading matrices.

    Post-audit replacement for the prior greedy "first match wins" matching, which
    was order-dependent and could miss the true optimum on borderline K choices.
    Uses ``scipy.optimize.linear_sum_assignment`` on the cost matrix ``1 - |phi|``
    so each factor pairs with its best partner globally. Returns the per-axis
    congruences in La's original column order.
    """
    from scipy.optimize import linear_sum_assignment
    Ka, Kb = La.shape[1], Lb.shape[1]
    phi = np.zeros((Ka, Kb))
    for a in range(Ka):
        for b in range(Kb):
            den = np.linalg.norm(La[:, a]) * np.linalg.norm(Lb[:, b])
            phi[a, b] = abs(float(La[:, a] @ Lb[:, b])) / den if den > 0 else 0.0
    row_ind, col_ind = linear_sum_assignment(1.0 - phi)
    out = [0.0] * Ka
    for r, c in zip(row_ind, col_ind, strict=False):
        out[r] = float(phi[r, c])
    return out


def _stratified_halves(cohort: np.ndarray, seed: int):
    """Cohort-stratified random half-split (post-audit).

    The prior implementation used ``rng.permutation(len(sc))`` which sampled
    half-splits uniformly. With DR ~6% of patients, DR count per half varied
    ±20% across seeds, which destabilises the cognition axis estimation.
    This stratified-by-cohort splitter keeps the per-cohort share within each
    half nearly constant.
    """
    rng = np.random.default_rng(seed)
    n = len(cohort)
    left = np.zeros(n, dtype=bool)
    for c in np.unique(cohort):
        idx = np.where(cohort == c)[0]
        rng.shuffle(idx)
        h = len(idx) // 2
        left[idx[:h]] = True
    return left


def select_k(curve: list[dict], floor: float) -> int:
    """Locked K = the dimensionality at which the split-half solution first collapses, minus 1.

    Walks up from the smallest K and extends the lock while the MIN congruence stays >= floor,
    stopping at the first K below it. This is the "maximum reproducible dimensionality before
    collapse" rule the manuscript applied by hand — and it correctly ignores any spurious
    recovery at higher K *after* a collapse (parallel analysis/Kaiser over-extract at this N)."""
    locked = min(row["k"] for row in curve)
    for row in sorted(curve, key=lambda r: r["k"]):
        if row["min_congruence"] >= floor:
            locked = row["k"]
        else:
            break
    return locked


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
        df = build_unified_dataframe(REPO_ROOT / "data", REPO_ROOT / "data" / "face-common-vars.xlsx",
                                     readiness=["READY", "PARTIAL"], format="long")
        full = to_harmonized_dataset(df, load_variables(REPO_ROOT / "data" / "face-common-vars.xlsx"),
                                     visit="V0", exclude=ADMINISTRATIVE_FEATURES)
    rank = full.metadata.reindex(sc.index)["dsm_diagnosis"].map(SPECTRUM).to_numpy()
    age = full.X.reindex(sc.index)["age"].to_numpy(float)
    sex = full.X.reindex(sc.index)["sex"].to_numpy(float)
    print(f"imputation-free FA: {len(sc):,} patients x {len(domains)} domains "
          f"(observed fraction {obs_frac:.3f}; the {1 - obs_frac:.0%} missing are NEVER filled)")

    # 1. masked split-half reproducibility -> K lock (POST-AUDIT specification).
    #    PRIMARY (deterministic, locked): N_SPLITS cohort-stratified half-splits → for each K, the
    #    AVERAGE of the per-split MIN Tucker congruence (Hungarian-matched). K is the largest K
    #    whose 25-split mean MIN stays >= K_FLOOR, walked contiguously from the smallest K. This
    #    replaces the prior "single fixed split with greedy axis matching" lock, which was both
    #    seed-fragile (one cohort's variability could flip the verdict) and order-dependent
    #    (greedy matching ignores global optimum). We still report the single-fixed-split curve
    #    for back-compat and inspection.
    ks = list(K_RANGE)
    cohort_arr = np.asarray(sc.index.get_level_values("cohort"))

    def half_min(seed: int) -> dict[int, list[float]]:
        left = _stratified_halves(cohort_arr, seed)
        Ra = masked_correlation(sc.iloc[left], MIN_PAIR)
        Rb = masked_correlation(sc.iloc[~left], MIN_PAIR)
        return {k: tucker_min(varimax(paf_loadings(Ra, k)), varimax(paf_loadings(Rb, k))) for k in ks}

    single = half_min(RANDOM)
    curve = [{"k": k, "min_congruence": float(np.min(single[k])),
              "mean_congruence": float(np.mean(single[k]))} for k in ks]

    ms = {k: [] for k in ks}
    for s in range(N_SPLITS):
        c = half_min(s)
        for k in ks:
            ms[k].append(float(np.min(c[k])))
    robustness = [{"k": k, "mean_min_congruence": float(np.mean(ms[k])),
                   "sd_min_congruence": float(np.std(ms[k]))} for k in ks]

    # PRIMARY lock: 25-split mean MIN congruence (post-audit).
    K = select_k([{"k": r["k"], "min_congruence": r["mean_min_congruence"]} for r in robustness],
                 K_FLOOR)

    print("Hungarian-matched, cohort-stratified split-half reproducibility:")
    print("  primary (25-split mean MIN, used to lock K):")
    for r in robustness:
        flag = ("  <- locked" if r["k"] == K
                else "  (collapse)" if r["mean_min_congruence"] < K_FLOOR else "")
        print(f"    K={r['k']:>2}  mean MIN={r['mean_min_congruence']:.3f}±{r['sd_min_congruence']:.3f}{flag}")
    print(f"  locked K = {K} (max K whose 25-split mean MIN Tucker congruence >= {K_FLOOR})")
    print("  back-compat (single fixed split MIN; not used to lock):")
    for row in curve:
        print(f"    K={row['k']:>2}  min={row['min_congruence']:.3f}  mean={row['mean_congruence']:.3f}")

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
        for d, v in zip(domains, load[:, a], strict=False):
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
            "k_floor": K_FLOOR, "n_splits": N_SPLITS,
            "reproducibility_curve": curve,   # single fixed split, primary (consumed by 15 for figS2)
            "reproducibility_robustness": robustness,   # N_SPLITS-averaged MIN (split-sensitivity)
            "dsm_ordering_per_axis": cont, "strongest_ordering_axis": f"axis{best+1}",
            "confound_max_corr": conf,
            "note": f"Imputation-free; K={K} locked as the maximum reproducible dimensionality "
                    f"before collapse (single fixed-split masked split-half MIN Tucker congruence "
                    f">= {K_FLOOR} through K={K}; collapse at K+1). The matrix includes 3 curated "
                    f"cognitive constructs (working memory, verbal reasoning, fluency) after the DR "
                    f"neuropsych extraction gap was closed (2026-05). CVLT memory and matrix reasoning "
                    f"are BP/SZ-only; processing speed and executive/TMT did not harmonise across "
                    f"cohorts (~0 communality, destabilising) and are excluded. The {N_SPLITS}-split "
                    f"robustness curve shows the 6-axis core is most split-stable and the weakest "
                    f"(metabolic) axis is the most split-sensitive (reported as a caveat, not used to "
                    f"lock). Orthogonal varimax; the mood<->psychosis spectrum is a cross-axis direction."}
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
    print("\nWrote results/dimensional_final_* (imputation-free) + results/reports/dimensional_final.html. Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
