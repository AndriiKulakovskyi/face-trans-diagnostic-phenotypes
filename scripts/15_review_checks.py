"""Clear the remaining review items with quantitative checks.

#5  K-selection curve → static figure figS2 (show the jagged split-half congruence honestly).
#6  "confound-free" is near-tautological for age/sex (residualized out). Compute the
    meaningful independence: η² of each locked axis explained by COHORT and by SITE.
#7  AE↔FA canonical correlations are maximized by construction → permutation null.
#8  HDBSCAN≈cohort may be missingness: can cohort be predicted from the observation MASK alone?
#9  the mood↔psychosis ρ rests on 7 subtype centroids → bootstrap 95% CI.

Writes results/review_checks.json + results/reports/figures/figS2_kcurve.{png,svg}.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.cross_decomposition import CCA  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402

from trans_diag import (  # noqa: E402
    AXIS_INDEX_TO_NAME,
    COGNITIVE_COMPOSITES,
    build_unified_dataframe,
)
from trans_diag.masked_fa import masked_loadings  # noqa: E402

RES = REPO / "results"
FIG = REPO / "results" / "reports" / "figures"
SPECTRUM = {"Trouble dépressif majeur": 0, "Bipolaire de type 2": 1, "Bipolaire de type 1": 2,
            "Bipolaire non spécifié": 3, "Trouble schizo-affectif": 4,
            "Trouble schizophréniforme": 5, "Schizophrénie": 6}


def eta_squared(values, groups):
    m = np.isfinite(values)
    v, g = values[m], np.asarray(groups)[m]
    grand = v.mean(); ss_tot = float(((v - grand) ** 2).sum())
    ss_bet = float(sum(((v[g == lv].mean() - grand) ** 2) * (g == lv).sum() for lv in pd.unique(g)))
    return ss_bet / ss_tot if ss_tot > 0 else float("nan")


def _mi(p):
    p = p.copy()
    p.index = pd.MultiIndex.from_arrays(
        [p.index.get_level_values("cohort").astype(str),
         p.index.get_level_values("patient_id").astype(str)], names=("cohort", "patient_id"))
    return p


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    out = {}
    final = _mi(pd.read_parquet(RES / "dimensional_final_scores.parquet"))
    scores = _mi(pd.read_parquet(RES / "cluster_domains_scores.parquet"))
    emb = _mi(pd.read_parquet(RES / "cluster_domains_embedding.parquet"))
    ae = _mi(pd.read_parquet(RES / "dimensional_ae_scores.parquet"))
    axcols = list(final.columns)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(REPO / "data", REPO / "data" / "face-common-vars.xlsx",
                                     readiness=["READY", "PARTIAL"], format="long")
    v0 = df[df["visit"] == "V0"].copy()
    v0["key"] = list(zip(v0["cohort"].str.lower(), v0["usubjid_patients"].astype(str), strict=False))
    site = pd.Series(v0["siteid_city"].astype(str).to_numpy(), index=v0["key"])
    dsm = pd.Series(v0["arm"].astype(str).to_numpy(), index=v0["key"])
    site = site[~site.index.duplicated()]; dsm = dsm[~dsm.index.duplicated()]
    keys = list(final.index)
    cohort = np.array([c for c, _ in keys])
    site_a = site.reindex(keys).to_numpy()
    dsm_a = dsm.reindex(keys).to_numpy()

    # ── #6: η²(cohort) and η²(site) per locked axis (the meaningful confound check) ──
    eta_c = {a: round(eta_squared(final[a].to_numpy(float), cohort), 3) for a in axcols}
    sv = pd.notna(site_a)
    eta_s = {a: round(eta_squared(final[a].to_numpy(float)[sv], site_a[sv]), 3) for a in axcols}
    out["eta_cohort"] = eta_c
    out["eta_site"] = eta_s
    out["eta_cohort_max"] = max(eta_c.values())
    out["eta_site_max"] = max(eta_s.values())
    print(f"#6 η²(cohort) max={out['eta_cohort_max']:.3f}  η²(site) max={out['eta_site_max']:.3f}")

    # ── #7: AE↔FA canonical correlations + permutation null ──
    # FA side = the FINAL imputation-free model (07), not the superseded mean-fill 05.
    # Masked scoring leaves sparsely-observed patients (<K observed domains) as NaN; drop them.
    common = ae.index.intersection(final.index)
    A = ae.reindex(common).to_numpy(float)
    F = final.reindex(common).to_numpy(float)
    ok = np.isfinite(A).all(1) & np.isfinite(F).all(1)
    out["cca_n_dropped_unscored"] = int((~ok).sum())
    A, F = A[ok], F[ok]
    kc = min(A.shape[1], F.shape[1])
    cca = CCA(n_components=kc).fit(A, F)
    U, V = cca.transform(A, F)
    obs = [float(np.corrcoef(U[:, i], V[:, i])[0, 1]) for i in range(kc)]
    rng = np.random.default_rng(0); Bc = 200; null_lead = np.empty(Bc)
    for b in range(Bc):
        Fp = F[rng.permutation(len(F))]
        cc = CCA(n_components=kc).fit(A, Fp)
        Up, Vp = cc.transform(A, Fp)
        null_lead[b] = np.corrcoef(Up[:, 0], Vp[:, 0])[0, 1]
    out["cca_observed"] = [round(x, 3) for x in obs]
    out["cca_null_leading_p95"] = round(float(np.percentile(null_lead, 95)), 3)
    out["cca_null_leading_mean"] = round(float(null_lead.mean()), 3)
    print(f"#7 CCA leading obs={obs[0]:.2f} vs permutation null mean={null_lead.mean():.2f} "
          f"(95th pct {np.percentile(null_lead,95):.2f})")

    # ── #8: predict cohort from the observation MASK alone (missingness confound) ──
    mask = scores.reindex(final.index).notna().astype(int).to_numpy()
    y = pd.Series(cohort).astype("category").cat.codes.to_numpy()
    clf = HistGradientBoostingClassifier(random_state=0)
    cvk = StratifiedKFold(5, shuffle=True, random_state=0)   # cohort-ordered ⇒ must shuffle
    bacc_mask = float(np.mean(cross_val_score(clf, mask, y, cv=cvk, scoring="balanced_accuracy")))
    bacc_emb = float(np.mean(cross_val_score(HistGradientBoostingClassifier(random_state=0),
                                             emb.reindex(final.index).to_numpy(float), y, cv=cvk,
                                             scoring="balanced_accuracy")))
    out["cohort_from_mask_bacc"] = round(bacc_mask, 3)
    out["cohort_from_embedding_bacc"] = round(bacc_emb, 3)
    out["chance_bacc"] = round(1 / 3, 3)
    print(f"#8 balanced-acc cohort from MASK={bacc_mask:.3f}, from embedding={bacc_emb:.3f} "
          f"(chance 0.333) ⇒ missingness {'DOES' if bacc_mask>0.6 else 'does not strongly'} "
          f"encode cohort")

    # ── #9: bootstrap 95% CI on the 7-subtype-centroid mood↔psychosis Spearman ──
    pc1 = PCA(n_components=1, random_state=0).fit_transform(
        emb.reindex(final.index).to_numpy(float))[:, 0]
    rank = pd.Series(dsm_a).map(SPECTRUM).to_numpy(float)
    valid = np.isfinite(rank)
    def cent_rho(p, r):
        d = pd.DataFrame({"r": r, "p": p}).dropna()
        cm = d.groupby("r")["p"].mean()
        return abs(float(spearmanr(cm.index, cm.to_numpy()).statistic)) if len(cm) > 2 else np.nan
    obs_rho = cent_rho(pc1[valid], rank[valid])
    idx = np.where(valid)[0]; Bb = 2000; boot = np.empty(Bb)
    for b in range(Bb):
        s = rng.integers(0, len(idx), len(idx))
        boot[b] = cent_rho(pc1[idx[s]], rank[idx[s]])
    boot = boot[np.isfinite(boot)]
    out["continuum_rho"] = round(obs_rho, 3)
    out["continuum_rho_ci"] = [round(float(np.percentile(boot, 2.5)), 3),
                               round(float(np.percentile(boot, 97.5)), 3)]
    print(f"#9 mood↔psychosis |ρ|={obs_rho:.2f} (n=7 centroids) 95% CI {out['continuum_rho_ci']}")

    # ── #10: cognition availability/confound battery (DR neuropsych recovered 2026-05) ──
    # Decide whether the cognitive axes are genuine trans-diagnostic dimensions or merely
    # track "was cognition tested" (a cohort/availability proxy — the original exclusion reason).
    cog_domains = [d for d in COGNITIVE_COMPOSITES if d in scores.columns]
    Lpiv = (pd.read_csv(RES / "dimensional_final_loadings.csv")
            .pivot(index="domain", columns="axis", values="loading"))
    cog_share = {}
    for ax in axcols:
        col = Lpiv[ax]
        ss_tot = float((col ** 2).sum())
        cog_share[ax] = round(float((col.reindex(cog_domains) ** 2).sum()) / ss_tot, 3) if ss_tot > 0 else 0.0
    cog_axes = [ax for ax in axcols if cog_share[ax] >= 0.30]   # cognition-loaded axes
    nm = lambda ax: AXIS_INDEX_TO_NAME.get(ax, ax)
    out["cognition_loading_share"] = {nm(a): cog_share[a] for a in axcols}
    out["cognition_axes"] = [nm(a) for a in cog_axes]
    out["cognition_axis_eta_cohort"] = {nm(a): eta_c[a] for a in cog_axes}
    out["cognition_axis_eta_site"] = {nm(a): eta_s[a] for a in cog_axes}

    # (b) availability test — does the NUMBER of cognition tests done (0..len) predict the axis?
    #     A high R² would mean the axis tracks availability, not ability.
    n_obs = scores.reindex(final.index)[cog_domains].notna().sum(axis=1).to_numpy(float)
    avail = {}
    for ax in cog_axes:
        s = final[ax].to_numpy(float); ok = np.isfinite(s)
        r = float(np.corrcoef(s[ok], n_obs[ok])[0, 1]) if ok.sum() > 2 else float("nan")
        avail[nm(ax)] = round(r ** 2, 3)
    out["cognition_axis_r2_from_availability"] = avail

    # (c) permutation placebo — row-permute the cognition domains WITHIN cohort, refit masked
    #     loadings, and measure how well each real cognition axis reproduces. If the cognitive
    #     structure is genuine it should COLLAPSE (best congruence with any permuted factor ≪ 1).
    sc_perm = scores.copy()
    coh_lv = sc_perm.index.get_level_values("cohort")
    rngp = np.random.default_rng(0)
    for d in cog_domains:
        for cl in pd.unique(coh_lv):
            m = (np.asarray(coh_lv) == cl)
            vals = sc_perm.loc[m, d].to_numpy().copy()
            sc_perm.loc[m, d] = vals[rngp.permutation(len(vals))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Lperm = masked_loadings(sc_perm, len(axcols))
    placebo = {}
    for ax in cog_axes:
        rc = Lpiv[ax].reindex(scores.columns).to_numpy(float)
        na = np.linalg.norm(rc)
        placebo[nm(ax)] = round(max(abs(float(rc @ Lperm[:, b])) / (na * np.linalg.norm(Lperm[:, b]) + 1e-12)
                                    for b in range(Lperm.shape[1])), 3)
    out["cognition_axis_permutation_congruence"] = placebo
    print(f"#10 cognition axes {out['cognition_axes']} (loading share {[cog_share[a] for a in cog_axes]})")
    print(f"    η²(cohort)={out['cognition_axis_eta_cohort']}  η²(site)={out['cognition_axis_eta_site']}")
    print(f"    R²(axis~#tests-done)={avail}  permutation-collapse-congruence={placebo}")

    (RES / "review_checks.json").write_text(json.dumps(out, indent=2))

    # ── #5: K-selection curve figure (honest, jagged) ──
    meta = json.loads((RES / "dimensional_final_meta.json").read_text())
    Kloc = int(meta["K"])
    cur = pd.DataFrame(meta["reproducibility_curve"])
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(cur["k"], cur["min_congruence"], "o-", label="min congruence (single split)", color="#1f77b4")
    ax.plot(cur["k"], cur["mean_congruence"], "s--", label="mean congruence", color="#999")
    if "reproducibility_robustness" in meta:
        rob = pd.DataFrame(meta["reproducibility_robustness"])
        ax.plot(rob["k"], rob["mean_min_congruence"], "^:", color="#ff7f0e",
                label="min congruence (25-split mean)")
    ax.axhline(0.85, ls=":", color="#d62728", label="0.85 threshold")
    ax.axvline(Kloc, ls="-", color="#2ca02c", alpha=0.4, lw=6, label=f"selected K={Kloc}")
    ax.set(xlabel="number of factors K", ylabel="split-half Tucker congruence",
           ylim=(0, 1.02), title=f"Figure S2. Reproducibility-vs-K is non-monotone (varimax\n"
           f"factor-splitting); K={Kloc} is the maximum reproducible solution (K>={Kloc+1} collapse)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"figS2_kcurve.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("#5 wrote results/reports/figures/figS2_kcurve.png/.svg")
    print("\nWrote results/review_checks.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
