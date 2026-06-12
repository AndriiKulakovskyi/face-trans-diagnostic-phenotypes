"""M2.3 — archetypal analysis: soft archetype membership (the LEAD view; §3.3).

The M2.1 continuum verdict ⇒ no natural-kind biotypes; the honest representation is a few **extreme
phenotypes** (archetypes), each patient a convex blend of them (soft simplex weights = continuum-honest
probabilistic decision regions). Wraps the `archetypes` library (PCHA-style AA). Uncertainty-aware by
projecting M1 posterior draws onto the fixed anchor archetypes. Coordinates on native latent z-scale (as
M2.1) — no re-standardization. Out-of-sample membership = project a new patient onto the fixed archetypes.
"""
from __future__ import annotations

import numpy as np


def fit_aa(X, A, seed=0, n_init=3, max_iter=120):
    """Fit AA with A archetypes. Returns (model, Z [A,D] archetype profiles, W [N,A] simplex weights, rss).
    max_iter=120: explained variance plateaus by ~100 iters on these coordinates (benchmarked)."""
    from archetypes import AA
    m = AA(n_archetypes=A, random_state=seed, n_init=n_init, max_iter=max_iter)
    W = np.asarray(m.fit_transform(np.asarray(X, dtype="float64")))
    Z = np.asarray(m.archetypes_)
    rss = float(((np.asarray(X) - W @ Z) ** 2).sum())
    return m, Z, W, rss


def explained_variance(X, rss):
    tss = float(((np.asarray(X) - np.asarray(X).mean(0)) ** 2).sum())
    return 1.0 - rss / tss


def select_A(X, As=range(2, 9), seed=0, knee_gain=0.02):
    """Explained-variance scree over A; knee = largest A whose marginal gain ≥ knee_gain."""
    ev = {}
    for A in As:
        _, _, _, rss = fit_aa(X, A, seed=seed, n_init=1)
        ev[A] = explained_variance(X, rss)
    As_ = sorted(ev)
    gains = {As_[i]: ev[As_[i]] - ev[As_[i - 1]] for i in range(1, len(As_))}
    knee = As_[0]
    for A in As_[1:]:
        if gains[A] >= knee_gain:
            knee = A
        else:
            break
    return {"explained_variance": ev, "gains": gains, "A_knee": int(knee)}


def _align(Zref, Z):
    from scipy.optimize import linear_sum_assignment
    cost = ((Zref[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    _, c = linear_sum_assignment(cost)
    return c


def stability(X, A, seeds=(0, 1, 2, 3), n_init=2):
    """Refit AA across seeds, align archetypes (Hungarian), report mean across-seed profile SD + min
    congruence — are the extreme phenotypes reproducible?"""
    fits = [fit_aa(X, A, seed=s, n_init=n_init) for s in seeds]
    Zref = fits[0][1]
    aligned = [Zref] + [fits[i][1][_align(Zref, fits[i][1])] for i in range(1, len(fits))]
    stack = np.stack(aligned)                                  # [n_seed, A, D]
    # Tucker congruence of each archetype profile vs the reference, min over archetypes
    def tucker(a, b):
        return float(abs((a * b).sum()) / (np.sqrt((a ** 2).sum()) * np.sqrt((b ** 2).sum()) + 1e-12))
    congr = [min(tucker(stack[0, a], stack[s, a]) for a in range(A)) for s in range(1, len(seeds))]
    return {"profile_across_seed_sd": float(stack.std(0).mean()),
            "min_tucker_congruence": float(min(congr)) if congr else 1.0, "n_seeds": len(seeds)}


def project_to_Z(X, Z):
    """Simplex-constrained least squares: weights w≥0, Σw=1 minimizing ‖x − wZ‖² (the standard AA
    NNLS-with-sum-constraint trick). Used to project M1 draws / new patients onto FIXED archetypes."""
    from scipy.optimize import nnls
    X = np.asarray(X, dtype="float64")
    A = Z.shape[0]
    M = 50.0 * (abs(Z).max() + 1.0)                            # large penalty enforces Σw≈1
    Za = np.vstack([Z.T, M * np.ones((1, A))])                 # [(D+1), A]
    out = np.zeros((X.shape[0], A))
    for i in range(X.shape[0]):
        xa = np.concatenate([X[i], [M]])
        w, _ = nnls(Za, xa)
        s = w.sum()
        out[i] = w / s if s > 0 else np.ones(A) / A
    return out


def project_draws(Z, draws, cols, n_draw=40, seed=0):
    """Per-patient weight uncertainty: project a subset of posterior draws onto fixed anchor archetypes."""
    rng = np.random.default_rng(seed)
    S = draws.shape[0]
    pick = rng.choice(S, size=min(n_draw, S), replace=False)
    Ws = np.stack([project_to_Z(draws[s][:, cols], Z) for s in pick])   # [n, N, A]
    return {"mean": Ws.mean(0), "sd": Ws.std(0), "n_draw": len(pick)}


def name_archetypes(Z, dims, thr=0.6):
    """Data-driven label per archetype from its extreme axes (|z| ≥ thr), top 3."""
    names = []
    for a in range(Z.shape[0]):
        z = Z[a]
        order = np.argsort(-np.abs(z))
        parts = [("↑" if z[d] > 0 else "↓") + dims[d] for d in order[:3] if abs(z[d]) >= thr]
        names.append(" ".join(parts) if parts else "near-average")
    return names
