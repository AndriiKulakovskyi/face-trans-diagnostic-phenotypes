"""Dual-criterion k selection: data science metrics + clinical utility.

Combines internal clustering validity indices with clinical outcome metrics
to select a scientifically justified k.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class KSelectionResult:
    """Result of dual-criterion k selection."""
    k_range: list[int]
    # Data science metrics per k
    silhouette: dict[int, float] = field(default_factory=dict)
    davies_bouldin: dict[int, float] = field(default_factory=dict)
    calinski_harabasz: dict[int, float] = field(default_factory=dict)
    gap_statistic: dict[int, float] = field(default_factory=dict)
    bootstrap_ari: dict[int, float] = field(default_factory=dict)
    # Clinical utility metrics per k
    treatment_entropy: dict[int, float] = field(default_factory=dict)
    functioning_variance: dict[int, float] = field(default_factory=dict)
    suicide_risk_chi2_p: dict[int, float] = field(default_factory=dict)
    dsm_subtype_entropy: dict[int, float] = field(default_factory=dict)
    # Selected k and rationale
    selected_k: int | None = None
    selection_rationale: str = ""


def compute_gap_statistic(
    X: np.ndarray,
    labels: np.ndarray,
    n_references: int = 10,
    random_state: int = 42,
) -> float:
    """Compute the gap statistic for a clustering."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances

    rng = np.random.RandomState(random_state)

    def _wk(data: np.ndarray, cluster_labels: np.ndarray) -> float:
        total = 0.0
        for c in np.unique(cluster_labels):
            mask = cluster_labels == c
            if mask.sum() < 2:
                continue
            dists = pairwise_distances(data[mask])
            total += np.sum(dists) / (2 * mask.sum())
        return total

    wk = _wk(X, labels)
    log_wk = np.log(wk + 1e-10)

    log_wk_refs = []
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    k = len(np.unique(labels))

    for _ in range(n_references):
        ref_data = rng.uniform(mins, maxs, size=X.shape)
        ref_labels = KMeans(
            n_clusters=k, random_state=rng.randint(10000), n_init=3,
        ).fit_predict(ref_data)
        ref_wk = _wk(ref_data, ref_labels)
        log_wk_refs.append(np.log(ref_wk + 1e-10))

    gap = np.mean(log_wk_refs) - log_wk
    return float(gap)


def compute_clinical_metrics(
    labels: np.ndarray,
    harmonized_X: pd.DataFrame,
    metadata: pd.DataFrame,
    dsm_subtypes: pd.Series | None = None,
) -> dict[str, float]:
    """Compute clinical utility metrics for a clustering.

    Returns dict with:
    - treatment_entropy: mean Shannon entropy of treatment class distribution per cluster
    - functioning_variance: mean within-cluster variance of functioning scores
    - suicide_risk_chi2_p: p-value from chi-squared test of attempt rates vs cluster
    - dsm_subtype_entropy: mean Shannon entropy of DSM subtypes per cluster
    """
    from scipy.stats import chi2_contingency

    results: dict[str, float] = {}

    # Treatment entropy
    tx_cols = [c for c in harmonized_X.columns if c.startswith('tx_on_')]
    if tx_cols:
        entropies = []
        for c in np.unique(labels):
            mask = labels == c
            for col in tx_cols:
                vals = harmonized_X.loc[mask, col].dropna()
                if len(vals) > 0:
                    p = vals.mean()
                    if 0 < p < 1:
                        entropies.append(-p * np.log2(p) - (1 - p) * np.log2(1 - p))
        results['treatment_entropy'] = float(np.mean(entropies)) if entropies else np.nan
    else:
        results['treatment_entropy'] = np.nan

    # Functioning variance
    func_cols = [
        c for c in harmonized_X.columns
        if c.startswith('inst_fast') or c.startswith('inst_psp')
        or c.startswith('inst_egf') or c.startswith('inst_eq5d')
    ]
    if func_cols:
        variances = []
        for c in np.unique(labels):
            mask = labels == c
            for col in func_cols:
                vals = harmonized_X.loc[mask, col].dropna()
                if len(vals) > 1:
                    variances.append(float(vals.var()))
        results['functioning_variance'] = float(np.mean(variances)) if variances else np.nan
    else:
        results['functioning_variance'] = np.nan

    # Suicide risk concentration
    sui_col = 'sui_ever_attempt'
    if sui_col in harmonized_X.columns:
        attempt_data = harmonized_X[sui_col].dropna()
        valid_idx = attempt_data.index
        valid_labels = labels[harmonized_X.index.isin(valid_idx)]
        if len(np.unique(valid_labels)) > 1:
            try:
                ct = pd.crosstab(valid_labels, attempt_data.values)
                if ct.shape[1] == 2:
                    chi2, p, _, _ = chi2_contingency(ct)
                    results['suicide_risk_chi2_p'] = float(p)
                else:
                    results['suicide_risk_chi2_p'] = np.nan
            except Exception:
                results['suicide_risk_chi2_p'] = np.nan
        else:
            results['suicide_risk_chi2_p'] = np.nan
    else:
        results['suicide_risk_chi2_p'] = np.nan

    # DSM subtype entropy
    if dsm_subtypes is not None:
        entropies = []
        for c in np.unique(labels):
            mask = labels == c
            sub_counts = dsm_subtypes[mask].value_counts(normalize=True)
            sub_counts = sub_counts[sub_counts > 0]
            ent = -np.sum(sub_counts * np.log2(sub_counts))
            entropies.append(float(ent))
        results['dsm_subtype_entropy'] = float(np.mean(entropies))
    else:
        results['dsm_subtype_entropy'] = np.nan

    return results


def run_dual_criterion_k_selection(
    embeddings: np.ndarray,
    harmonized_X: pd.DataFrame,
    metadata: pd.DataFrame,
    k_range: range = range(4, 13),
    dsm_subtypes: pd.Series | None = None,
    n_bootstrap: int = 10,
    random_state: int = 42,
) -> KSelectionResult:
    """Run dual-criterion k selection over a range of k values.

    For each k:
    1. Run k-means
    2. Compute data science metrics (silhouette, DB, CH, gap, bootstrap ARI)
    3. Compute clinical metrics (treatment entropy, functioning var, suicide chi2, DSM entropy)

    Selects k at the intersection of acceptable data science metrics and optimal clinical utility.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        adjusted_rand_score,
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    result = KSelectionResult(k_range=list(k_range))

    for k in k_range:
        logger.info(f"Evaluating k={k}")
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(embeddings)

        # Data science metrics
        if len(np.unique(labels)) > 1:
            result.silhouette[k] = float(silhouette_score(
                embeddings, labels, sample_size=min(5000, len(embeddings)),
            ))
            result.davies_bouldin[k] = float(davies_bouldin_score(embeddings, labels))
            result.calinski_harabasz[k] = float(calinski_harabasz_score(embeddings, labels))
            result.gap_statistic[k] = compute_gap_statistic(
                embeddings, labels, random_state=random_state,
            )

        # Bootstrap stability
        rng = np.random.RandomState(random_state)
        aris = []
        for _ in range(n_bootstrap):
            idx = rng.choice(len(embeddings), size=int(0.8 * len(embeddings)), replace=False)
            sub_labels = KMeans(
                n_clusters=k, random_state=rng.randint(10000), n_init=5,
            ).fit_predict(embeddings[idx])
            full_sub = labels[idx]
            aris.append(adjusted_rand_score(full_sub, sub_labels))
        result.bootstrap_ari[k] = float(np.mean(aris))

        # Clinical metrics
        clinical = compute_clinical_metrics(labels, harmonized_X, metadata, dsm_subtypes)
        result.treatment_entropy[k] = clinical['treatment_entropy']
        result.functioning_variance[k] = clinical['functioning_variance']
        result.suicide_risk_chi2_p[k] = clinical['suicide_risk_chi2_p']
        result.dsm_subtype_entropy[k] = clinical['dsm_subtype_entropy']

    # Select k: find acceptable range from data science, then optimize clinical
    acceptable_ks = [
        k for k in k_range
        if result.silhouette.get(k, 0) > 0.2
        and result.bootstrap_ari.get(k, 0) > 0.6
    ]

    if acceptable_ks:
        best_k = min(
            acceptable_ks,
            key=lambda k: result.functioning_variance.get(k, float('inf')),
        )
        result.selected_k = best_k
        result.selection_rationale = (
            f"k={best_k} selected from acceptable range {acceptable_ks} "
            f"(silhouette>{0.2}, bootstrap ARI>{0.6}) by minimizing "
            f"within-cluster functioning variance ({result.functioning_variance[best_k]:.4f})"
        )
    else:
        best_k = max(result.silhouette, key=result.silhouette.get) if result.silhouette else list(k_range)[0]
        result.selected_k = best_k
        result.selection_rationale = (
            f"k={best_k} selected by best silhouette (no k met both thresholds)"
        )

    logger.info(f"Selected k={result.selected_k}: {result.selection_rationale}")
    return result
