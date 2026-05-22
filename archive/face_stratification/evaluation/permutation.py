"""Permutation testing for clustering metric significance.

Two null distribution types:

- **Label permutation null** (for silhouette, CH, DB): shuffle cluster
  labels while keeping the embedding fixed. Tests whether the observed
  cluster structure is better than random partitioning.
- **Reference permutation null** (for ARI, NMI, Cramer's V): shuffle the
  DSM cohort labels. Tests whether cluster–cohort alignment exceeds chance.

Uses mini-batch k-means for speed in the null (1000+ runs).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def permutation_test_silhouette(
    embedding: np.ndarray,
    labels: np.ndarray,
    *,
    n_permutations: int = 1000,
    seed: int = 42,
    sample_size: int | None = 5000,
) -> dict[str, float]:
    """Permutation test for silhouette: shuffle cluster labels, recompute."""
    from sklearn.metrics import silhouette_score

    labels = np.asarray(labels)
    valid = labels >= 0
    emb_v = embedding[valid]
    lab_v = labels[valid]
    n = len(lab_v)

    if n < 2 or len(np.unique(lab_v)) < 2:
        return {"observed": float("nan"), "p_value": float("nan"),
                "null_mean": float("nan"), "null_std": float("nan")}

    # Observed
    if sample_size and n > sample_size:
        rng_s = np.random.default_rng(seed)
        idx = rng_s.choice(n, sample_size, replace=False)
        observed = float(silhouette_score(emb_v[idx], lab_v[idx], metric="cosine"))
    else:
        observed = float(silhouette_score(emb_v, lab_v, metric="cosine"))
        idx = np.arange(n)

    # Null distribution
    rng = np.random.default_rng(seed)
    null_vals = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm_labels = rng.permutation(lab_v)
        if sample_size and n > sample_size:
            null_vals[i] = silhouette_score(emb_v[idx], perm_labels[idx], metric="cosine")
        else:
            null_vals[i] = silhouette_score(emb_v, perm_labels, metric="cosine")

    p_value = float(np.mean(null_vals >= observed))

    return {
        "observed": observed,
        "p_value": p_value,
        "null_mean": float(null_vals.mean()),
        "null_std": float(null_vals.std()),
        "ci_lower": float(np.percentile(null_vals, 2.5)),
        "ci_upper": float(np.percentile(null_vals, 97.5)),
    }


def permutation_test_external(
    labels: np.ndarray,
    reference: np.ndarray,
    *,
    n_permutations: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Permutation tests for ARI, NMI, Cramer's V: shuffle reference labels."""
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
    )

    labels = np.asarray(labels)
    reference = np.asarray(reference)
    valid = labels >= 0
    lab_v = labels[valid]
    ref_v = reference[valid]

    if len(lab_v) < 2 or len(np.unique(lab_v)) < 2:
        return pd.DataFrame()

    # Observed values
    obs_ari = adjusted_rand_score(ref_v, lab_v)
    obs_nmi = normalized_mutual_info_score(ref_v, lab_v)
    obs_cramers = _cramers_v(lab_v, ref_v)

    metrics = {
        "ari": {"observed": obs_ari, "fn": adjusted_rand_score},
        "nmi": {"observed": obs_nmi, "fn": normalized_mutual_info_score},
        "cramers_v": {"observed": obs_cramers, "fn": None},
    }

    rng = np.random.default_rng(seed)
    null_dists: dict[str, np.ndarray] = {k: np.zeros(n_permutations) for k in metrics}

    for i in range(n_permutations):
        perm_ref = rng.permutation(ref_v)
        null_dists["ari"][i] = adjusted_rand_score(perm_ref, lab_v)
        null_dists["nmi"][i] = normalized_mutual_info_score(perm_ref, lab_v)
        null_dists["cramers_v"][i] = _cramers_v(lab_v, perm_ref)

    rows = []
    for name, info in metrics.items():
        obs = info["observed"]
        null = null_dists[name]
        p_value = float(np.mean(null >= obs))
        rows.append({
            "metric": name,
            "observed": float(obs),
            "p_value": p_value,
            "null_mean": float(null.mean()),
            "null_std": float(null.std()),
            "ci_lower": float(np.percentile(null, 2.5)),
            "ci_upper": float(np.percentile(null, 97.5)),
        })

    return pd.DataFrame(rows)


def permutation_test_all(
    embedding: np.ndarray,
    labels: np.ndarray,
    reference: np.ndarray,
    *,
    n_permutations: int = 1000,
    seed: int = 42,
    silhouette_sample_size: int | None = 5000,
) -> pd.DataFrame:
    """Run all permutation tests and return a combined tidy DataFrame."""
    # Silhouette
    sil = permutation_test_silhouette(
        embedding, labels,
        n_permutations=n_permutations, seed=seed,
        sample_size=silhouette_sample_size,
    )
    sil_row = {"metric": "silhouette", **sil}

    # External metrics
    ext_df = permutation_test_external(
        labels, reference,
        n_permutations=n_permutations, seed=seed,
    )

    rows = [sil_row]
    if not ext_df.empty:
        rows.extend(ext_df.to_dict("records"))

    return pd.DataFrame(rows)


def _cramers_v(labels: np.ndarray, reference: np.ndarray) -> float:
    """Cramer's V between two categorical arrays."""
    from scipy.stats import chi2_contingency

    ct = pd.crosstab(labels, reference)
    n = ct.sum().sum()
    if n == 0:
        return 0.0
    try:
        chi2, _, _, _ = chi2_contingency(ct)
        k = min(ct.shape) - 1
        return float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0
    except ValueError:
        return 0.0
