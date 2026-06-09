"""M2.1 — structure-discovery gate: cluster vs continuum vs branched (§3.1).

Before fitting any mixture, characterize the SHAPE of the 9-D coordinate cloud — is there discrete
cluster structure at all, a graded continuum, or a branched manifold? Reported as a primary result; it
decides which view (mixture vs archetypes) leads, and the honest null (continuum) is permitted.

Run uncertainty-aware (over M1 posterior draws) so the verdict is not an artefact of treating posterior
blobs as points. Lean stack — sklearn / scipy / umap / hdbscan / diptest / networkx. Coordinates are used
on their native latent z-scale (no re-standardization: that would inflate the noisy low-variance axes like
substance to equal weight; the cross-patient SDs 0.58–1.07 are within ~2×, so native scale gives the
well-measured axes appropriately more influence)."""
from __future__ import annotations

import numpy as np


def hopkins(X: np.ndarray, m: int | None = None, n_rep: int = 5, seed: int = 0) -> float:
    """Hopkins statistic of cluster tendency. Sample m real points (NN distance to other real points,
    w) and m uniform-random points in the data's bounding box (NN distance to real points, u);
    H = Σu / (Σu + Σw). H ≈ 0.5 ⇒ no cluster tendency (uniform/continuum-like); H → 1 ⇒ clustered.
    Averaged over n_rep resamples. (Dimensionality/n-sensitive — read alongside the other diagnostics.)"""
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    N, D = X.shape
    m = m or min(150, max(20, N // 20))
    lo, hi = X.min(0), X.max(0)
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    out = []
    for _ in range(n_rep):
        ridx = rng.choice(N, size=m, replace=False)
        w = nn.kneighbors(X[ridx], n_neighbors=2)[0][:, 1]          # NN to another real point
        U = rng.uniform(lo, hi, size=(m, D))
        u = nn.kneighbors(U, n_neighbors=1)[0][:, 0]               # NN (real) to a uniform point
        out.append(u.sum() / (u.sum() + w.sum() + 1e-12))
    return float(np.mean(out))


def dip_test(X: np.ndarray, axes: list[str]) -> dict:
    """Hartigan's dip test (unimodal vs multimodal) per axis + on PC1. Low p ⇒ multimodal (clusters)."""
    import diptest
    from sklearn.decomposition import PCA
    res = {}
    for j, name in enumerate(axes):
        d, p = diptest.diptest(np.asarray(X[:, j], dtype="float64"))
        res[name] = {"dip": float(d), "p": float(p)}
    pc1 = PCA(2, random_state=0).fit_transform(X)[:, 0]
    d, p = diptest.diptest(pc1.astype("float64"))
    res["PC1"] = {"dip": float(d), "p": float(p)}
    return res


def gmm_bic_sweep(X: np.ndarray, Ks=range(1, 13), seed: int = 0) -> dict:
    """GaussianMixture (full cov) BIC over K. Returns BIC per K, the BIC-optimal K, the gain of the
    best K>1 over K=1 (ΔBIC>0 ⇒ a mixture beats a single Gaussian), and whether BIC is still strictly
    decreasing at Kmax (monotone ⇒ no interior optimum ⇒ continuum-like over-segmentation)."""
    from sklearn.mixture import GaussianMixture
    Ks = list(Ks)
    bic = {}
    for k in Ks:
        gm = GaussianMixture(k, covariance_type="full", n_init=3, random_state=seed,
                             reg_covar=1e-4, max_iter=300).fit(X)
        bic[k] = float(gm.bic(X))
    kbest = min(bic, key=bic.get)
    gain1 = (bic[1] - min(v for k, v in bic.items() if k > 1)) if 1 in bic and len(Ks) > 1 else float("nan")
    monotone = kbest == max(Ks)
    return {"bic": bic, "k_best": int(kbest), "gain_over_k1": float(gain1), "monotone_decreasing": bool(monotone)}


def silhouette_sweep(X: np.ndarray, Ks=range(2, 13), seed: int = 0, sample: int = 3000) -> dict:
    """KMeans silhouette over K (on a subsample for the O(n²) score). Peak < ~0.15 ⇒ no clear clusters."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    Xs = X[idx]
    sil = {}
    for k in Ks:
        lab = KMeans(k, n_init=5, random_state=seed).fit_predict(Xs)
        sil[k] = float(silhouette_score(Xs, lab)) if len(set(lab)) > 1 else float("nan")
    kbest = max(sil, key=sil.get)
    return {"silhouette": sil, "k_best": int(kbest), "peak": float(sil[kbest])}


def gap_statistic(X: np.ndarray, Ks=range(1, 13), B: int = 10, seed: int = 0, sample: int = 3000) -> dict:
    """Tibshirani gap statistic. Optimal K = smallest k with gap(k) ≥ gap(k+1) − s(k+1)."""
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    Xs = X[idx]
    lo, hi = Xs.min(0), Xs.max(0)
    Ks = list(Ks)

    def Wk(Xd, k):
        km = KMeans(k, n_init=3, random_state=seed).fit(Xd)
        return sum(float(((Xd[km.labels_ == c] - km.cluster_centers_[c]) ** 2).sum())
                   for c in range(k)) + 1e-12

    logW = np.array([np.log(Wk(Xs, k)) for k in Ks])
    ref = np.zeros((B, len(Ks)))
    for b in range(B):
        Xb = rng.uniform(lo, hi, size=Xs.shape)
        ref[b] = [np.log(Wk(Xb, k)) for k in Ks]
    gap = ref.mean(0) - logW
    sk = ref.std(0) * np.sqrt(1 + 1.0 / B)
    kopt = Ks[-1]
    for i in range(len(Ks) - 1):
        if gap[i] >= gap[i + 1] - sk[i + 1]:
            kopt = Ks[i]; break
    return {"gap": {k: float(g) for k, g in zip(Ks, gap)}, "k_opt": int(kopt)}


def hdbscan_summary(X: np.ndarray, min_cluster_size: int | None = None) -> dict:
    """Density clustering. Many points as 'noise' or a single dominant cluster ⇒ continuum signal;
    several persistent well-separated clusters ⇒ discrete structure."""
    import hdbscan
    mcs = min_cluster_size or max(25, len(X) // 100)
    cl = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=10).fit(X)
    lab = cl.labels_
    n_clusters = int(len(set(lab)) - (1 if -1 in lab else 0))
    noise = float((lab == -1).mean())
    sizes = {int(c): int((lab == c).sum()) for c in sorted(set(lab)) if c != -1}
    return {"n_clusters": n_clusters, "noise_fraction": noise, "cluster_sizes": sizes}


def mapper_graph(X: np.ndarray, lens: np.ndarray, n_cubes: int = 12, overlap: float = 0.4,
                 min_node: int = 20, seed: int = 0):
    """Minimal Mapper: cover the 1-D lens in overlapping intervals; per interval DBSCAN the points
    (in full space); nodes = (interval, local cluster); edges = shared members. Returns a networkx
    graph with node attrs (size, lens_mean, member idx). A single chain ⇒ continuum; flares/branches
    ⇒ branched; disconnected dense blobs ⇒ clusters."""
    import networkx as nx
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    lo, hi = float(lens.min()), float(lens.max())
    step = (hi - lo) / n_cubes
    width = step * (1 + overlap)
    G = nx.Graph()
    node_members = {}
    nid = 0
    for c in range(n_cubes):
        center = lo + (c + 0.5) * step
        sel = np.flatnonzero((lens >= center - width / 2) & (lens <= center + width / 2))
        if len(sel) < min_node:
            continue
        Xs = X[sel]
        # eps heuristic: median distance to the 5th NN within the bin
        k = min(6, len(sel) - 1)
        d5 = NearestNeighbors(n_neighbors=k).fit(Xs).kneighbors(Xs)[0][:, -1]
        eps = float(np.median(d5)) * 1.3
        lab = DBSCAN(eps=eps, min_samples=5).fit_predict(Xs)
        for lc in sorted(set(lab)):
            if lc == -1:
                continue
            mem = set(sel[lab == lc].tolist())
            if len(mem) < min_node:
                continue
            G.add_node(nid, size=len(mem), lens_mean=float(lens[list(mem)].mean()))
            node_members[nid] = mem
            nid += 1
    ids = list(node_members)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            shared = len(node_members[ids[i]] & node_members[ids[j]])
            if shared:
                G.add_edge(ids[i], ids[j], weight=shared)
    return G, node_members


def uncertainty_sweep(draws: np.ndarray, cols: list[int], n_draw: int = 20, Kmax: int = 12,
                      seed: int = 0) -> dict:
    """Re-run Hopkins + GMM-BIC-optimal-K over n_draw posterior draws (measurement-error-aware):
    is the verdict stable when each patient's coordinate is sampled from its posterior, not fixed?"""
    rng = np.random.default_rng(seed)
    S = draws.shape[0]
    pick = rng.choice(S, size=min(n_draw, S), replace=False)
    Hs, Ks = [], []
    for s in pick:
        Xs = draws[s][:, cols]
        Hs.append(hopkins(Xs, seed=int(s) % 1000))
        Ks.append(gmm_bic_sweep(Xs, range(1, Kmax + 1), seed=0)["k_best"])
    vals, cnts = np.unique(Ks, return_counts=True)
    return {"hopkins_mean": float(np.mean(Hs)), "hopkins_sd": float(np.std(Hs)),
            "k_best_distribution": {int(k): int(c) for k, c in zip(vals, cnts)},
            "k_best_mode": int(vals[cnts.argmax()])}


def verdict(diag: dict) -> dict:
    """Synthesize the battery into clustered / continuum / branched, with the evidence and a confidence.
    Conservative: declare 'clustered' only on converging evidence (tendency + modality + separation +
    an interior K). Default to 'continuum' when the signals are weak — the honest null (§1.2)."""
    H = diag["hopkins"]
    gmm = diag["gmm_bic"]
    sil = diag["silhouette"]["peak"]
    gap_k = diag["gap"]["k_opt"]
    dip_pc1 = diag["dip"]["PC1"]["p"]
    dip_multi = sum(1 for a, v in diag["dip"].items() if v["p"] < 0.05)
    hdb_n = diag["hdbscan"]["n_clusters"]
    hdb_noise = diag["hdbscan"]["noise_fraction"]

    clustered_signals = [
        H > 0.75,                                   # cluster tendency
        sil > 0.25,                                 # clear separation
        not gmm["monotone_decreasing"],             # an interior BIC optimum (not over-segmenting)
        gap_k > 1,                                  # gap prefers K>1
        dip_pc1 < 0.05,                             # multimodal along PC1
        hdb_n >= 2 and hdb_noise < 0.5,             # density finds separated clusters
    ]
    score = sum(clustered_signals)
    if score >= 4 and sil > 0.2:
        label = "clustered"
    elif score <= 2 or sil < 0.15:
        label = "continuum"
    else:
        label = "weak/ambiguous (lean continuum)"
    return {"label": label, "clustered_score": int(score), "n_signals": len(clustered_signals),
            "evidence": {"hopkins": H, "silhouette_peak": sil, "gmm_k_best": gmm["k_best"],
                         "gmm_monotone": gmm["monotone_decreasing"], "gap_k_opt": gap_k,
                         "dip_pc1_p": dip_pc1, "n_axes_multimodal": dip_multi,
                         "hdbscan_n": hdb_n, "hdbscan_noise": hdb_noise}}
