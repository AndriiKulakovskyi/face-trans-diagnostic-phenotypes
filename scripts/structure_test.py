"""Step 1 — does the V0 data contain DISCRETE clusters, or is it DIMENSIONAL?

We have been assuming discrete phenotypes. The internal indices (flat silhouette,
no CH/DB optimum) suggest weak separation. This script runs the decisive tests to
tell discrete from dimensional structure, on the domain embedding:

  1. Graph eigengap — the natural #clusters for a spectral view. A clear gap after
     the k-th smallest Laplacian eigenvalue ⇒ k discrete clusters; no gap ⇒ continuum.
  2. Gap statistic vs a matched Gaussian null — is there cluster structure beyond a
     single (possibly elongated) blob? Real silhouette vs null silhouette per k.
  3. HDBSCAN density — if most patients are "noise"/one component ⇒ no dense islands.
  4. Bimodality of the top principal axes (Sarle's coefficient; >0.555 ≈ multimodal).
  5. DSM-subtype anchor — ARI(clusters, 7 DSM subtypes) AND whether subtype centroids
     order along a mood↔psychosis continuum (the dimensional signature).

Writes results/structure_test.json + reports/structure_test.html.
Run:  python3 scripts/structure_test.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402
from scipy.stats import kurtosis, skew  # noqa: E402
from sklearn.cluster import HDBSCAN, KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    adjusted_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
)
from sklearn.neighbors import kneighbors_graph  # noqa: E402

from face_common import (  # noqa: E402
    ADMINISTRATIVE_FEATURES,
    build_unified_dataframe,
    load_variables,
    to_harmonized_dataset,
)

RESULTS_DIR = REPO_ROOT / "results"
REPORTS_DIR = REPO_ROOT / "reports"
EMB_PATH = RESULTS_DIR / "cluster_domains_embedding.parquet"
RANDOM = 0
# clinical mood↔psychosis spectrum order for the continuum test
SPECTRUM = {
    "Trouble dépressif majeur": 0, "Trouble Dépressif Majeur": 0,
    "Bipolaire de type 2": 1, "Bipolaire de type 1": 2, "Bipolaire non spécifié": 3,
    "Trouble schizo-affectif": 4, "Trouble schizophréniforme": 5, "Schizophrénie": 6,
}


def eigengap(arr, n_neighbors=15, subsample=3500, seed=0):
    """Smallest Laplacian eigenvalues of a kNN graph (subsampled) + the largest gap."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arr), size=min(subsample, len(arr)), replace=False)
    A = kneighbors_graph(arr[idx], n_neighbors=n_neighbors, mode="connectivity",
                         include_self=False)
    A = A.maximum(A.T)
    deg = np.asarray(A.sum(1)).ravel()
    d_isqrt = sp.diags(1.0 / np.sqrt(np.where(deg > 0, deg, 1.0)))
    L = (sp.eye(A.shape[0]) - d_isqrt @ A @ d_isqrt).toarray()
    vals = np.linalg.eigvalsh(L)[:20]
    gaps = np.diff(vals)
    # natural k = position of the largest gap among the first eigenvalues (>=2)
    k_star = int(np.argmax(gaps[1:10]) + 2)
    return vals.tolist(), gaps.tolist(), k_star


def gap_vs_gaussian(arr, ks, n_ref=5, seed=0):
    """Real vs matched-Gaussian-null: silhouette + gap (log W_null - log W_real)."""
    rng = np.random.default_rng(seed)
    mean, cov = arr.mean(0), np.cov(arr.T)
    rows = []
    for k in ks:
        km = KMeans(k, random_state=RANDOM, n_init=10).fit(arr)
        sil = silhouette_score(arr, km.labels_, sample_size=4000, random_state=RANDOM)
        logw = np.log(km.inertia_ + 1e-12)
        null_logw, null_sil = [], []
        for _ in range(n_ref):
            null = rng.multivariate_normal(mean, cov, size=len(arr))
            kmn = KMeans(k, random_state=RANDOM, n_init=5).fit(null)
            null_logw.append(np.log(kmn.inertia_ + 1e-12))
            null_sil.append(silhouette_score(null, kmn.labels_, sample_size=3000,
                                              random_state=RANDOM))
        rows.append({"k": k, "silhouette_real": float(sil),
                     "silhouette_null": float(np.mean(null_sil)),
                     "gap": float(np.mean(null_logw) - logw)})
    return pd.DataFrame(rows)


def bimodality(x):
    """Sarle's bimodality coefficient; > ~0.555 suggests bi/multimodality."""
    n = len(x)
    g = skew(x)
    k = kurtosis(x, fisher=True)
    denom = k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float((g ** 2 + 1) / denom)


def main() -> int:
    REPORTS_DIR.mkdir(exist_ok=True)
    emb = pd.read_parquet(EMB_PATH)
    arr = emb.to_numpy(np.float64)
    n = len(arr)
    print(f"embedding: {n:,} × {arr.shape[1]}")

    # subtype anchor (arm) via metadata MultiIndex (robust to id formatting)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(REPO_ROOT / "data", REPO_ROOT / "face-common-vars.xlsx",
                                     readiness=["READY", "PARTIAL"], format="long")
        full = to_harmonized_dataset(df, load_variables(REPO_ROOT / "face-common-vars.xlsx"),
                                     visit="V0", exclude=ADMINISTRATIVE_FEATURES)
    arm = full.metadata.reindex(emb.index)["dsm_diagnosis"]
    rank = arm.map(SPECTRUM)
    print(f"DSM subtypes (arm): {arm.value_counts().to_dict()}")

    # 1. eigengap
    vals, gaps, k_star = eigengap(arr)
    print(f"\n[eigengap] smallest Laplacian eigenvalues: {[round(v,3) for v in vals[:8]]}")
    print(f"           largest early gap at k = {k_star} (gap={max(gaps[1:10]):.3f})")

    # 2. gap vs Gaussian null
    ks = list(range(2, 13))
    gv = gap_vs_gaussian(arr, ks)
    print("\n[gap vs Gaussian null] (real≈null silhouette ⇒ no discrete structure):")
    print(gv.round(3).to_string(index=False))

    # 3. HDBSCAN density — and is the dense structure just diagnosis?
    hdb = HDBSCAN(min_cluster_size=150, min_samples=10).fit(arr)
    n_clu = int(len(set(hdb.labels_)) - (1 if -1 in hdb.labels_ else 0))
    noise = float((hdb.labels_ == -1).mean())
    cohort = np.array([c for c, _ in emb.index])
    dm = hdb.labels_ >= 0
    hdb_cohort_ari = float(adjusted_rand_score(cohort[dm], hdb.labels_[dm])) if dm.any() else float("nan")
    print(f"\n[HDBSCAN] dense clusters={n_clu}  noise={noise:.0%}  "
          f"ARI(dense,cohort)={hdb_cohort_ari:.2f}  (high ARI ⇒ the 'clusters' are just diagnosis)")

    # 4. bimodality of top PCs
    pcs = PCA(n_components=5, random_state=RANDOM).fit_transform(arr)
    bc = [bimodality(pcs[:, i]) for i in range(5)]
    print(f"\n[bimodality] Sarle BC of PC1..5: {[round(b,3) for b in bc]} "
          f"(>0.555 ≈ multimodal)")

    # 5. subtype anchor + continuum
    ari = {k: float(adjusted_rand_score(arm.fillna("NA"),
            KMeans(k, random_state=RANDOM, n_init=10).fit_predict(arr))) for k in (5, 6, 7)}
    ami7 = float(adjusted_mutual_info_score(arm.fillna("NA"),
            KMeans(7, random_state=RANDOM, n_init=10).fit_predict(arr)))
    sub = pd.DataFrame({"pc1": pcs[:, 0], "pc2": pcs[:, 1], "rank": rank.to_numpy()})
    cent = sub.dropna().groupby("rank")[["pc1", "pc2"]].mean()
    from scipy.stats import spearmanr
    rho_pc1 = float(spearmanr(cent.index, cent["pc1"]).statistic) if len(cent) > 2 else float("nan")
    rho_pc2 = float(spearmanr(cent.index, cent["pc2"]).statistic) if len(cent) > 2 else float("nan")
    rho_best = max(abs(rho_pc1), abs(rho_pc2))
    print(f"\n[subtype anchor] ARI(clusters,7 subtypes): {ari}  AMI(k=7)={ami7:.3f}")
    print(f"[continuum] subtype centroids vs mood↔psychosis rank: "
          f"|Spearman| PC1={abs(rho_pc1):.2f} PC2={abs(rho_pc2):.2f} (≥~0.8 ⇒ continuum)")

    # ── verdict: discrete TRANS-DIAGNOSTIC clusters vs dimensional ──
    # A discrete trans-diagnostic finding needs (a) a natural k (gap statistic that
    # PEAKS, not monotone), (b) dense clusters NOT explained by diagnosis, and (c)
    # multimodal axes. A high HDBSCAN-vs-cohort ARI means the only discrete structure
    # is DSM diagnosis itself.
    gap_monotonic = bool(gv["gap"].is_monotonic_increasing)
    dense_not_diagnosis = bool(noise < 0.30 and n_clu >= 3 and hdb_cohort_ari < 0.30)
    multimodal = bool(max(bc) > 0.56)
    if (not gap_monotonic) and dense_not_diagnosis and multimodal:
        verdict = "DISCRETE trans-diagnostic clusters supported"
    elif hdb_cohort_ari >= 0.40:
        verdict = ("No discrete trans-diagnostic clusters — the only discrete structure is "
                   "DSM diagnosis (HDBSCAN≈cohort); trans-diagnostic variation is DIMENSIONAL")
    else:
        verdict = "Weak/diffuse — leans DIMENSIONAL (no eigengap, monotone gap, ~unimodal axes)"
    print(f"\n=== VERDICT: {verdict} ===")
    print(f"    natural k (gap peaks): {not gap_monotonic} | HDBSCAN dense=diagnosis ARI={hdb_cohort_ari:.2f} | "
          f"max bimodality={max(bc):.2f} | subtype continuum |rho|PC1={abs(rho_pc1):.2f}")

    meta = {
        "eigengap": {"eigenvalues": vals[:10], "k_star": k_star},
        "gap_vs_gaussian": gv.to_dict(orient="records"), "gap_monotonic": gap_monotonic,
        "hdbscan": {"n_clusters": n_clu, "noise_frac": noise, "cohort_ari": hdb_cohort_ari},
        "bimodality_pc": bc,
        "subtype_ari": ari, "subtype_ami_k7": ami7,
        "continuum_spearman": {"pc1": rho_pc1, "pc2": rho_pc2},
        "verdict": verdict,
    }
    (RESULTS_DIR / "structure_test.json").write_text(json.dumps(meta, indent=2, default=str))
    _report(vals, gv, cent, arm, meta)
    print(f"\nWrote results/structure_test.json + reports/structure_test.html")
    return 0


def _report(vals, gv, cent, arm, meta):
    spec_lbl = {0: "MDD", 1: "BP-II", 2: "BP-I", 3: "BP-NOS", 4: "schizoaff",
                5: "schizophrenif", 6: "schizophr"}
    f1 = go.Figure(go.Scatter(y=vals[:15], mode="lines+markers"))
    f1.update_layout(title="Laplacian eigenvalue scree (gap ⇒ #clusters)", height=320,
                     xaxis_title="index", yaxis_title="eigenvalue", margin=dict(t=40))
    f2 = go.Figure()
    f2.add_scatter(x=gv["k"], y=gv["silhouette_real"], mode="lines+markers", name="real")
    f2.add_scatter(x=gv["k"], y=gv["silhouette_null"], mode="lines+markers", name="Gaussian null")
    f2.update_layout(title="Silhouette: real vs matched-Gaussian null (overlap ⇒ no clusters)",
                     height=320, xaxis_title="k", yaxis_title="silhouette", margin=dict(t=40))
    f3 = go.Figure()
    for r in cent.index:
        f3.add_scatter(x=[cent.loc[r, "pc1"]], y=[cent.loc[r, "pc2"]], mode="markers+text",
                       text=[spec_lbl.get(int(r), str(r))], textposition="top center",
                       marker=dict(size=12), name=spec_lbl.get(int(r), str(r)))
    f3.update_layout(title="DSM-subtype centroids in PC space (ordered line ⇒ continuum)",
                     height=360, xaxis_title="PC1", yaxis_title="PC2", margin=dict(t=40))
    css = "body{font-family:-apple-system,Segoe UI,sans-serif;margin:0;padding:0 24px 60px}h1{color:#2b3a55}.v{font-size:18px;padding:12px;background:#eef2f7;border-radius:8px}"
    html = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{css}</style></head><body>",
            "<h1>Step 1 — discrete vs dimensional structure</h1>",
            f"<div class='v'><b>Verdict:</b> {meta['verdict']}</div>",
            pio.to_html(f1, include_plotlyjs="cdn", full_html=False),
            pio.to_html(f2, include_plotlyjs=False, full_html=False),
            pio.to_html(f3, include_plotlyjs=False, full_html=False), "</body></html>"]
    (REPORTS_DIR / "structure_test.html").write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
