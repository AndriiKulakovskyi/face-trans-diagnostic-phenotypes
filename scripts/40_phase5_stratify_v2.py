"""Phase 5 (v2) — patient stratification: discrete subtypes or a continuum?

Complementary to the dimensional arm. Tests whether patients form DISCRETE clusters or lie on a
CONTINUUM across the validated structure, with the same battery as the v1 structure test (04):
  1. graph eigengap (natural #clusters from a kNN-Laplacian);
  2. gap statistic vs a matched-Gaussian null (real vs null silhouette);
  3. HDBSCAN density + ARI(dense, cohort) — is any 'cluster' just DSM diagnosis?;
  4. bimodality (Sarle) of the axes;
  5. DSM-subtype anchor: ARI + whether subtype centroids order on a mood<->psychosis continuum;
  6. bootstrap cluster stability (ARI of resampled k-means vs full) — unstable => continuum.

Inputs:  A (primary) = 4 dimension scores + mania + suicidal_ideation (the validated 6 axes);
         B (sensitivity) = ~75 construct scores via the masked MultipartiteSpectral embedding (engine).
Masked / no-imputation. Writes results/hfa/phase5_structure_v2.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import kurtosis, skew, spearmanr
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import kneighbors_graph

from trans_diag import build_unified_dataframe, load_variables, to_harmonized_dataset
from trans_diag.engine import FeatureSchema, HarmonizedDataset, MultipartiteSpectralEmbedding

warnings.simplefilter("ignore")
OUT = ROOT / "results" / "hfa"
RANDOM = 0
SPECTRUM = {  # clinical mood<->psychosis order for the continuum test
    "Trouble dépressif majeur": 0, "Trouble Dépressif Majeur": 0,
    "Bipolaire de type 2": 1, "Bipolaire de type 1": 2, "Bipolaire non spécifié": 3,
    "Trouble schizo-affectif": 4, "Trouble schizophréniforme": 5, "Schizophrénie": 6,
}
EMBED_CONFIG = dict(min_coverage=0.30, min_features_per_partition=3, n_components_per_partition=8,
                    k_neighbours=10, include_4way=True, include_mask_columns=True,
                    l2_normalize=True, feature_mode="cumulative", partition_weighting="sqrt_info")


def eigengap(arr, n_neighbors=15, subsample=3500, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arr), size=min(subsample, len(arr)), replace=False)
    A = kneighbors_graph(arr[idx], n_neighbors=n_neighbors, mode="connectivity", include_self=False)
    A = A.maximum(A.T)
    deg = np.asarray(A.sum(1)).ravel()
    dis = sp.diags(1.0 / np.sqrt(np.where(deg > 0, deg, 1.0)))
    L = (sp.eye(A.shape[0]) - dis @ A @ dis).toarray()
    vals = np.linalg.eigvalsh(L)[:20]
    return vals.tolist(), int(np.argmax(np.diff(vals)[1:10]) + 2)


def gap_vs_gaussian(arr, ks, n_ref=5, seed=0):
    rng = np.random.default_rng(seed)
    mean, cov = arr.mean(0), np.cov(arr.T)
    rows = []
    for k in ks:
        km = KMeans(k, random_state=RANDOM, n_init=10).fit(arr)
        sil = silhouette_score(arr, km.labels_, sample_size=4000, random_state=RANDOM)
        nsil = []
        for _ in range(n_ref):
            null = rng.multivariate_normal(mean, cov, size=len(arr))
            kmn = KMeans(k, random_state=RANDOM, n_init=5).fit(null)
            nsil.append(silhouette_score(null, kmn.labels_, sample_size=3000, random_state=RANDOM))
        rows.append({"k": k, "sil_real": float(sil), "sil_null": float(np.mean(nsil))})
    return pd.DataFrame(rows)


def bimodality(x):
    n = len(x)
    g, k = skew(x), kurtosis(x, fisher=True)
    return float((g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def bootstrap_stability(arr, ks, n_boot=20, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for k in ks:
        ref = KMeans(k, random_state=RANDOM, n_init=10).fit(arr)
        aris = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(arr), len(arr))
            bl = KMeans(k, random_state=RANDOM, n_init=5).fit_predict(arr[idx])
            aris.append(adjusted_rand_score(ref.labels_[idx], bl))
        out[k] = float(np.mean(aris))
    return out


def structure_test(arr, arm, cohort, label):
    print(f"\n########## STRUCTURE TEST — {label}  ({arr.shape[0]:,} × {arr.shape[1]}) ##########")
    arr = (arr - arr.mean(0)) / (arr.std(0) + 1e-12)
    vals, k_star = eigengap(arr)
    print(f"[eigengap] smallest Laplacian eigenvalues {[round(v,3) for v in vals[:7]]} -> natural k={k_star}")
    gv = gap_vs_gaussian(arr, list(range(2, 9)))
    gv["sil_gap"] = gv.sil_real - gv.sil_null
    print("[gap vs Gaussian null]  (real ≈ null silhouette => no discrete structure):")
    print(gv.round(3).to_string(index=False))
    hdb = HDBSCAN(min_cluster_size=150, min_samples=10).fit(arr)
    n_clu = int(len(set(hdb.labels_)) - (1 if -1 in hdb.labels_ else 0))
    noise = float((hdb.labels_ == -1).mean())
    dm = hdb.labels_ >= 0
    hdb_coh = float(adjusted_rand_score(cohort[dm], hdb.labels_[dm])) if dm.any() and n_clu else float("nan")
    print(f"[HDBSCAN] dense clusters={n_clu} noise={noise:.0%} ARI(dense,cohort)={hdb_coh:.2f}")
    pcs = PCA(n_components=min(5, arr.shape[1]), random_state=RANDOM).fit_transform(arr)
    bc = [bimodality(pcs[:, i]) for i in range(pcs.shape[1])]
    print(f"[bimodality] Sarle BC of top PCs {[round(b,3) for b in bc]}  (>0.555 ≈ multimodal)")
    stab = bootstrap_stability(arr, list(range(2, 8)))
    print(f"[bootstrap stability] ARI per k: {{{', '.join(f'{k}:{v:.2f}' for k,v in stab.items())}}}  "
          f"(>0.8 => stable/discrete)")
    ari = {k: float(adjusted_rand_score(arm.fillna('NA'),
            KMeans(k, random_state=RANDOM, n_init=10).fit_predict(arr))) for k in (5, 6, 7)}
    rank = arm.map(SPECTRUM)
    cent = pd.DataFrame({"pc1": pcs[:, 0], "rank": rank.to_numpy()}).dropna().groupby("rank")["pc1"].mean()
    rho = float(spearmanr(cent.index, cent.values).statistic) if len(cent) > 2 else np.nan
    print(f"[DSM anchor] ARI(clusters,subtypes)={ari} | subtype-centroid continuum |Spearman PC1|={abs(rho):.2f}")

    gap_monotonic = bool(gv.sil_gap.is_monotonic_increasing or (gv.sil_gap.abs() < 0.02).all())
    stable = max(stab.values()) > 0.80
    multimodal = max(bc) > 0.56
    if stable and (not gap_monotonic) and multimodal and (np.isnan(hdb_coh) or hdb_coh < 0.30):
        verdict = "DISCRETE subtypes supported"
    elif not np.isnan(hdb_coh) and hdb_coh >= 0.40:
        verdict = "Only discrete structure = DSM diagnosis; trans-diagnostic variation is DIMENSIONAL"
    else:
        verdict = "DIMENSIONAL / continuum (no stable clusters, real≈null silhouette, ~unimodal)"
    print(f"=== VERDICT ({label}): {verdict} ===")
    return {"label": label, "eigengap_k": k_star, "gap_vs_gaussian": gv.round(4).to_dict("records"),
            "hdbscan": {"n": n_clu, "noise": noise, "cohort_ari": hdb_coh}, "bimodality": bc,
            "bootstrap_stability": stab, "dsm_ari": ari, "continuum_spearman": rho, "verdict": verdict}


def main() -> None:
    F = pd.read_pickle(OUT / "stage3_scores_v2.pkl").set_index(["cohort", "patient_id"])
    S = pd.read_pickle(OUT / "stage2_scores_v2.pkl").set_index(["cohort", "patient_id"])
    df = build_unified_dataframe(str(ROOT / "data"), str(ROOT / "data" / "face-common-vars.xlsx"),
                                 readiness=["READY", "PARTIAL", "NOT USABLE", "ID"], format="long")
    vs = load_variables(str(ROOT / "data" / "face-common-vars.xlsx"))
    ds = to_harmonized_dataset(df, vs, visit="V0", sections=None, residualize_on=None, normalize=False)

    # A (primary): 4 dims + mania + suicidal_ideation
    A = F[["dim1", "dim2", "dim3", "dim4"]].join(S[["mania_activation", "suicidal_ideation"]]).dropna()
    arm_A = ds.metadata.reindex(A.index)["dsm_diagnosis"]
    coh_A = np.array(A.index.get_level_values("cohort"))
    resA = structure_test(A.to_numpy(float), arm_A, coh_A, "A: 6 axes (dims+mania+suicide)")

    # B (sensitivity): construct scores -> masked spectral embedding (engine, no imputation)
    cov = S.notna().mean()
    keep = [c for c in S.columns if cov[c] >= 0.30 and S[c].var() > 1e-9]
    meta = ds.metadata.reindex(S.index)[["cohort", "patient_id", "dsm_diagnosis"]]
    feats = [{"id": d, "label_fr": d, "block": "construct", "type": "continuous",
              "temporal_scope": "current", "cohorts": ("bp", "sz", "dr")} for d in keep]
    schema = FeatureSchema.model_validate({"version": "v2", "blocks": [
        {"id": "construct", "label_fr": "construct", "description": "construct scores"}], "features": feats})
    fm = pd.DataFrame([{"feature_id": d, "label_fr": d, "block": "construct", "type": "continuous",
                        "temporal_scope": "current", "unit": None, "direction": "none",
                        "cohorts": "bp,sz,dr"} for d in keep]).set_index("feature_id")
    hd = HarmonizedDataset(X=S[keep], metadata=meta, feature_metadata=fm, schema=schema)
    emb = MultipartiteSpectralEmbedding(**EMBED_CONFIG).fit(hd).transform().values
    arm_B = ds.metadata.reindex(emb.index)["dsm_diagnosis"]
    coh_B = np.array(emb.index.get_level_values("cohort"))
    resB = structure_test(emb.to_numpy(np.float64), arm_B, coh_B, "B: 75 construct scores (engine embedding)")

    json.dump({"A": resA, "B": resB}, open(OUT / "phase5_structure_v2.json", "w"), indent=2, default=str)
    print(f"\nsaved -> {OUT}/phase5_structure_v2.json")


if __name__ == "__main__":
    main()
