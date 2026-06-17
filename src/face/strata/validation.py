"""M2.4 — validation battery (§7): Q1 existence · Q2 not-just-severity · Q3 transdiagnostic · Q4 stable /
not-an-artefact · head-to-head vs DSM-5 (the "better description" test, §1.7). Applied to both soft views
(archetypes = lead; tessellation). Diagnosis/covariates are validation-only, never inputs."""
from __future__ import annotations

import numpy as np


def eta_squared(labels, X):
    """Per-axis η² (between-group SS / total SS) — how much of each coordinate a partition explains."""
    labels = np.asarray(labels)
    X = np.asarray(X, dtype="float64")
    grand = X.mean(0)
    tot = ((X - grand) ** 2).sum(0)
    bet = np.zeros(X.shape[1])
    for g in np.unique(labels):
        m = labels == g
        bet += m.sum() * (X[m].mean(0) - grand) ** 2
    return bet / np.where(tot > 0, tot, 1.0)


def cramers_v(a, b):
    import pandas as pd
    from scipy.stats import chi2_contingency
    ct = pd.crosstab(pd.Series(np.asarray(a)), pd.Series(np.asarray(b))).to_numpy()
    if min(ct.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(ct)[0]
    n = ct.sum()
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))


def ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(np.asarray(a), np.asarray(b)))


def coverage_artifact(nobs, labels, seed=0, n_perm=30):
    """Can the COVERAGE pattern (observed-indicator counts per axis) predict the partition beyond chance?

    Predictive skill ⇒ membership is driven by missingness, not values (an artefact). Plain accuracy is
    weak under class imbalance (P3-06), so we add imbalance-robust metrics — **balanced accuracy**,
    **macro-F1**, **log-loss** — and a **permutation test** (does balanced accuracy beat label-permuted
    nulls?). ``lift`` is retained for backward compatibility."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    rng = np.random.default_rng(seed)
    nobs, labels = np.asarray(nobs), np.asarray(labels)
    n_classes = len(np.unique(labels))

    def _clf():
        return RandomForestClassifier(120, random_state=seed, n_jobs=-1)

    def _bal(y):
        return float(cross_val_score(_clf(), nobs, y, cv=4, scoring="balanced_accuracy").mean())

    acc = float(cross_val_score(_clf(), nobs, labels, cv=4).mean())
    base = float(np.bincount(labels).max() / len(labels))
    bal = _bal(labels)
    f1m = float(cross_val_score(_clf(), nobs, labels, cv=4, scoring="f1_macro").mean())
    ll = float(-cross_val_score(_clf(), nobs, labels, cv=4, scoring="neg_log_loss").mean())
    null = np.array([_bal(rng.permutation(labels)) for _ in range(n_perm)])
    p_perm = float((null >= bal).mean())
    return {"classifier_acc": acc, "majority_baseline": base, "lift": acc - base,
            "balanced_acc": bal, "balanced_chance": 1.0 / n_classes, "macro_f1": f1m, "log_loss": ll,
            "perm_p_value": p_perm}


def tess_seed_stability(X, S, K, seeds=(1, 2, 3)):
    """Re-fit the XD tessellation across seeds; ARI of the MAP partition vs seed 0 (reproducibility)."""
    from face.strata.mixture import xd_em
    base = xd_em(X, S, K, seed=0)["resp"].argmax(1)
    aris = [ari(base, xd_em(X, S, K, seed=s)["resp"].argmax(1)) for s in seeds]
    return {"mean_ari": float(np.mean(aris)), "min_ari": float(np.min(aris))}
