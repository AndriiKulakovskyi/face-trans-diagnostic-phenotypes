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


def coverage_artifact(nobs, labels, seed=0):
    """Can the COVERAGE pattern (observed-indicator counts per axis) predict the partition? High accuracy
    over the majority baseline ⇒ membership is driven by missingness, not values (an artefact)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    labels = np.asarray(labels)
    acc = float(cross_val_score(RandomForestClassifier(120, random_state=seed, n_jobs=-1),
                                np.asarray(nobs), labels, cv=4).mean())
    base = float(np.bincount(labels).max() / len(labels))
    return {"classifier_acc": acc, "majority_baseline": base, "lift": acc - base}


def tess_seed_stability(X, S, K, seeds=(1, 2, 3)):
    """Re-fit the XD tessellation across seeds; ARI of the MAP partition vs seed 0 (reproducibility)."""
    from face.strata.mixture import xd_em
    base = xd_em(X, S, K, seed=0)["resp"].argmax(1)
    aris = [ari(base, xd_em(X, S, K, seed=s)["resp"].argmax(1)) for s in seeds]
    return {"mean_ari": float(np.mean(aris)), "min_ari": float(np.min(aris))}
