"""V3 Bayesian engine — proper initialization (EFA warm-start, Procrustes-aligned).

The user's emphasis: don't cold-start the bifactor. We run an EFA on a TEMPORARY
mean-filled copy of the continuous matrix (used for `initvals` ONLY — never a likelihood
input), align each empirical factor to the prior's home-item pattern (a discrete
Procrustes match), and seed the positive loading cells with the aligned magnitudes. The
general factor G is seeded from its prior anchors (EFA specifics don't contain it). Cross
cells start at 0, residual variances at 1 - communality, intercepts at 0 (z-scored data).

This is a partial warm-start: it seeds the loadings (the hard part for identifying a
bifactor G) and leaves the correlation/Z parameters to the sampler's jitter init. It is
wrapped defensively by the caller; on any failure the fit falls back to pure jitter.
"""
from __future__ import annotations

import numpy as np


def efa_initvals(data, meta, spec: dict, cell_priors: dict, n_specific: int) -> dict:
    """Return an initvals dict for {lam_pos, lam_cross, psi_raw, nu}. Empty on failure."""
    try:
        from sklearn.decomposition import FactorAnalysis
    except Exception:
        return {}

    fc = meta["factor_cols"]
    g_key = spec.get("general_factor") if spec.get("include_g") else None
    items = data.cont_items
    J = len(items)

    # EFA on the mean-filled COPY (init only)
    Z = data.mean_fill
    k = max(1, min(n_specific, J - 1))
    try:
        fa = FactorAnalysis(n_components=k, random_state=0).fit(Z)
        L = fa.components_.T                       # [J, k] empirical loadings
        comm = np.clip((L ** 2).sum(axis=1), 0.0, 0.95)
    except Exception:
        L, comm = np.zeros((J, k)), np.full(J, 0.5)

    # map each specific factor -> the EFA component best covering its primary items
    item_idx = {it: j for j, it in enumerate(items)}
    home = {it: data.cont_home[j] for j, it in enumerate(items)}
    aligned: dict[tuple, float] = {}
    used = set()
    for f in fc:
        if f == g_key:
            continue
        prim = [item_idx[it] for it in items if home[it] == f]
        if not prim:
            continue
        # best component by summed |loading| over this factor's primary items
        scores = [np.abs(L[prim, e]).sum() if e not in used else -1 for e in range(k)]
        e = int(np.argmax(scores))
        used.add(e)
        s = np.sign(L[prim, e].sum()) or 1.0       # orient so primaries are positive
        for it in items:
            aligned[(it, f)] = float(s * L[item_idx[it], e])

    # seed positive cells (primary + g_anchor)
    lam_pos = np.zeros(len(meta["pos_cells"]))
    for kk, (j, c) in enumerate(meta["pos_cells"]):
        it, f = items[j], fc[c]
        if f == g_key:
            _, mean, _ = cell_priors.get((it, f), ("g_anchor", 0.6, 0.3))
            lam_pos[kk] = max(0.3, float(mean))    # G anchors: start at the prior mean
        else:
            v = abs(aligned.get((it, f), 0.4))
            lam_pos[kk] = float(np.clip(v, 0.1, 1.5))

    init = {
        "lam_pos": lam_pos,
        "psi_raw": np.clip(1.0 - comm, 0.05, 1.5).astype(float),
        "nu": np.zeros(J),
    }
    if meta["sgn_cells"]:
        init["lam_cross"] = np.zeros(len(meta["sgn_cells"]))
    return init
