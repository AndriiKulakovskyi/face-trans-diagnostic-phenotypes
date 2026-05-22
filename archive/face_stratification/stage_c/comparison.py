"""Formal cluster-vs-DSM comparison — chi², Cramér's V, purity, entropy.

Stage B reported a handful of agreement metrics (ARI, NMI, V-measure).
Stage C promotes this to a formal statistical comparison suitable for a
scientific article:

- **Pearson chi-square test of independence** on the cluster × cohort
  contingency table. Reports the statistic, degrees of freedom, p-value,
  and expected frequencies.
- **Cramér's V effect size** — the chi-square counterpart of correlation,
  normalized to ``[0, 1]``.
- **Per-cohort purity**: the fraction of each cohort concentrated in its
  single most-dominant cluster.
- **Per-cluster Shannon entropy** of the cohort distribution (in bits),
  reported in absolute terms and as a fraction of ``log₂(n_cohorts)``.
- **Transdiagnostic score** per cluster: ``entropy / log₂(n_cohorts)``,
  ranging from 0 (single-cohort) to 1 (perfectly mixed).
- **Full sklearn suite** (ARI, AMI, NMI, V-measure, homogeneity, completeness)
  via :func:`face_stratification.clustering.metrics.compute_cluster_metrics`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Public types ─────────────────────────────────────────────────────────────


@dataclass
class FullDSMComparison:
    """Scalar + table outputs of the formal cluster-vs-DSM comparison."""

    n_patients: int
    n_clusters: int
    n_cohorts: int

    # Contingency tables
    contingency: pd.DataFrame
    row_normalized: pd.DataFrame
    col_normalized: pd.DataFrame

    # Chi-square test
    chi2_statistic: float
    chi2_dof: int
    chi2_p_value: float
    cramers_v: float

    # Per-cohort
    per_cohort_purity: dict[str, float]
    per_cohort_top_cluster: dict[str, int]

    # Per-cluster
    per_cluster_entropy_bits: dict[int, float]
    per_cluster_transdiagnostic_score: dict[int, float]

    # sklearn metrics
    ari: float
    ami: float
    nmi: float
    v_measure: float
    homogeneity: float
    completeness: float

    # Headline summary numbers
    mean_cluster_entropy_bits: float
    max_possible_entropy_bits: float
    mean_transdiagnostic_score: float

    def summary_dict(self) -> dict[str, Any]:
        """Flat dict suitable for JSON export / tables."""
        return {
            "n_patients": self.n_patients,
            "n_clusters": self.n_clusters,
            "n_cohorts": self.n_cohorts,
            "chi2_statistic": self.chi2_statistic,
            "chi2_dof": self.chi2_dof,
            "chi2_p_value": self.chi2_p_value,
            "cramers_v": self.cramers_v,
            "ari": self.ari,
            "ami": self.ami,
            "nmi": self.nmi,
            "v_measure": self.v_measure,
            "homogeneity": self.homogeneity,
            "completeness": self.completeness,
            "mean_cluster_entropy_bits": self.mean_cluster_entropy_bits,
            "max_possible_entropy_bits": self.max_possible_entropy_bits,
            "mean_transdiagnostic_score": self.mean_transdiagnostic_score,
            "per_cohort_purity": self.per_cohort_purity,
            "per_cohort_top_cluster": self.per_cohort_top_cluster,
            "per_cluster_entropy_bits": {
                str(k): v for k, v in self.per_cluster_entropy_bits.items()
            },
            "per_cluster_transdiagnostic_score": {
                str(k): v for k, v in self.per_cluster_transdiagnostic_score.items()
            },
        }


# ─── Individual statistics ────────────────────────────────────────────────────


def chi_square_independence(
    cluster_labels: np.ndarray,
    cohort_labels: np.ndarray,
    *,
    correction: bool = False,
) -> tuple[float, int, float, np.ndarray]:
    """Pearson chi-square test of independence on cluster × cohort.

    Yates' continuity correction is **disabled by default** because we use
    the chi² statistic both for the hypothesis test and as the basis for
    Cramér's V effect size — and Cramér's V should be computed on the
    uncorrected statistic. The correction is a hypothesis-test artifact
    that biases the effect size downward on small 2 × 2 tables.

    Returns ``(chi2_statistic, dof, p_value, expected_frequencies)``.
    """
    try:
        from scipy.stats import chi2_contingency
    except ImportError as exc:
        raise ImportError("chi_square_independence requires scipy.") from exc

    table = pd.crosstab(cluster_labels, cohort_labels)
    chi2, p, dof, expected = chi2_contingency(table.values, correction=correction)
    return float(chi2), int(dof), float(p), expected


def cramers_v(
    cluster_labels: np.ndarray,
    cohort_labels: np.ndarray,
) -> float:
    """Cramér's V effect size for a contingency table.

    Defined as ``V = sqrt(chi2 / (n * (min(r, c) - 1)))`` with a
    bias-corrected denominator. Range ``[0, 1]``; 0 means independence,
    1 means perfect association.
    """
    chi2, _dof, _p, _exp = chi_square_independence(cluster_labels, cohort_labels)
    table = pd.crosstab(cluster_labels, cohort_labels)
    n = int(table.values.sum())
    r, c = table.shape
    min_dim = min(r, c) - 1
    if n <= 0 or min_dim <= 0:
        return float("nan")
    v = np.sqrt(chi2 / (n * min_dim))
    return float(v)


def per_cohort_purity(
    cluster_labels: np.ndarray,
    cohort_labels: np.ndarray,
) -> tuple[dict[str, float], dict[str, int]]:
    """For each cohort, the maximum fraction in a single cluster.

    Returns two dicts:
        - purity: {cohort: fraction in dominant cluster}
        - top cluster: {cohort: id of that cluster}
    """
    cluster_labels = np.asarray(cluster_labels)
    cohort_labels = np.asarray(cohort_labels)
    purity: dict[str, float] = {}
    top: dict[str, int] = {}
    for cohort in sorted(np.unique(cohort_labels)):
        mask = cohort_labels == cohort
        if not mask.any():
            purity[str(cohort)] = 0.0
            top[str(cohort)] = -1
            continue
        vals, counts = np.unique(cluster_labels[mask], return_counts=True)
        idx = int(np.argmax(counts))
        purity[str(cohort)] = float(counts[idx] / mask.sum())
        top[str(cohort)] = int(vals[idx])
    return purity, top


def per_cluster_cohort_entropy(
    cluster_labels: np.ndarray,
    cohort_labels: np.ndarray,
) -> tuple[dict[int, float], dict[int, float]]:
    """Per-cluster Shannon entropy of the cohort distribution (in bits).

    Also returns the transdiagnostic score = entropy / log₂(n_cohorts),
    which is 0 for a single-cohort cluster and 1 for perfectly mixed.
    """
    cluster_labels = np.asarray(cluster_labels)
    cohort_labels = np.asarray(cohort_labels)
    n_cohorts = int(pd.Series(cohort_labels).nunique())
    max_entropy = float(np.log2(n_cohorts)) if n_cohorts > 1 else 0.0

    entropy: dict[int, float] = {}
    td_score: dict[int, float] = {}
    for cluster in sorted(np.unique(cluster_labels)):
        if cluster < 0:
            continue
        mask = cluster_labels == cluster
        if not mask.any():
            entropy[int(cluster)] = 0.0
            td_score[int(cluster)] = 0.0
            continue
        vals, counts = np.unique(cohort_labels[mask], return_counts=True)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        ent = float(-(probs * np.log2(probs)).sum())
        entropy[int(cluster)] = ent
        td_score[int(cluster)] = ent / max_entropy if max_entropy > 0 else 0.0
    return entropy, td_score


# ─── Full comparison ─────────────────────────────────────────────────────────


def full_dsm_comparison(
    cluster_labels: pd.Series | np.ndarray,
    cohort_labels: pd.Series | np.ndarray,
) -> FullDSMComparison:
    """One-stop comparison: chi², Cramér's V, purity, entropy, sklearn suite."""
    try:
        from sklearn.metrics import (
            adjusted_mutual_info_score,
            adjusted_rand_score,
            completeness_score,
            homogeneity_score,
            normalized_mutual_info_score,
            v_measure_score,
        )
    except ImportError as exc:
        raise ImportError("full_dsm_comparison requires scikit-learn.") from exc

    clu = np.asarray(cluster_labels)
    coh = np.asarray(cohort_labels)
    valid = clu >= 0
    clu = clu[valid]
    coh = coh[valid]

    # Contingency
    contingency = pd.crosstab(
        pd.Series(clu, name="cluster"),
        pd.Series(coh, name="cohort"),
    )
    row_norm = contingency.div(contingency.sum(axis=1), axis=0)
    col_norm = contingency.div(contingency.sum(axis=0), axis=1)

    # Chi-square
    chi2, dof, p, _expected = chi_square_independence(clu, coh)
    v = cramers_v(clu, coh)

    # Purity + entropy
    cohort_purity_dict, top_cluster_dict = per_cohort_purity(clu, coh)
    entropy_dict, td_score_dict = per_cluster_cohort_entropy(clu, coh)

    # sklearn agreement suite
    ari = float(adjusted_rand_score(coh, clu))
    ami = float(adjusted_mutual_info_score(coh, clu))
    nmi = float(normalized_mutual_info_score(coh, clu))
    v_m = float(v_measure_score(coh, clu))
    hom = float(homogeneity_score(coh, clu))
    com = float(completeness_score(coh, clu))

    n_cohorts = int(pd.Series(coh).nunique())
    max_entropy = float(np.log2(n_cohorts)) if n_cohorts > 1 else 0.0

    return FullDSMComparison(
        n_patients=int(clu.size),
        n_clusters=int(contingency.shape[0]),
        n_cohorts=n_cohorts,
        contingency=contingency,
        row_normalized=row_norm,
        col_normalized=col_norm,
        chi2_statistic=chi2,
        chi2_dof=dof,
        chi2_p_value=p,
        cramers_v=v,
        per_cohort_purity=cohort_purity_dict,
        per_cohort_top_cluster=top_cluster_dict,
        per_cluster_entropy_bits=entropy_dict,
        per_cluster_transdiagnostic_score=td_score_dict,
        ari=ari,
        ami=ami,
        nmi=nmi,
        v_measure=v_m,
        homogeneity=hom,
        completeness=com,
        mean_cluster_entropy_bits=float(np.mean(list(entropy_dict.values()))) if entropy_dict else 0.0,
        max_possible_entropy_bits=max_entropy,
        mean_transdiagnostic_score=float(np.mean(list(td_score_dict.values()))) if td_score_dict else 0.0,
    )
