"""Treatment response proxy computation and cluster validation.

Defines treatment response proxies from available FACE data and tests
whether clusters have differential treatment profiles and outcomes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TreatmentValidationResult:
    """Results of treatment response validation per cluster."""
    n_clusters: int = 0
    # Per-cluster treatment profiles
    treatment_profiles: pd.DataFrame | None = None
    # Per-cluster functioning scores (mean, std)
    functioning_by_cluster: pd.DataFrame | None = None
    # ANOVA / Kruskal-Wallis results
    functioning_tests: dict[str, dict[str, float]] = field(default_factory=dict)
    # Adherence by cluster
    adherence_by_cluster: pd.DataFrame | None = None
    # Summary
    has_differential_treatment: bool = False
    has_differential_functioning: bool = False


def compute_treatment_profiles(
    labels: np.ndarray,
    harmonized_X: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-cluster treatment class proportions.
    
    Returns DataFrame with clusters as rows, treatment features as columns,
    values are fraction of patients on each treatment.
    """
    tx_cols = [c for c in harmonized_X.columns if c.startswith('tx_on_')]
    if not tx_cols:
        return pd.DataFrame()
    
    df = harmonized_X[tx_cols].copy()
    df['cluster'] = labels
    return df.groupby('cluster').mean()


def compute_functioning_by_cluster(
    labels: np.ndarray,
    harmonized_X: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Compute per-cluster functioning score statistics and tests.
    
    Returns
    -------
    stats_df : DataFrame with mean/std per cluster per functioning measure
    tests : dict mapping feature -> {'statistic': H, 'p_value': p}
    """
    from scipy.stats import kruskal
    
    func_cols = [c for c in harmonized_X.columns 
                 if any(c.startswith(p) for p in 
                        ['inst_fast', 'inst_psp', 'inst_egf', 'inst_eq5d', 'inst_leaps'])]
    
    if not func_cols:
        return pd.DataFrame(), {}
    
    df = harmonized_X[func_cols].copy()
    df['cluster'] = labels
    
    stats = df.groupby('cluster').agg(['mean', 'std', 'count'])
    
    tests = {}
    for col in func_cols:
        groups = [
            df.loc[df['cluster'] == c, col].dropna().values
            for c in np.unique(labels)
        ]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) >= 2:
            try:
                h, p = kruskal(*groups)
                tests[col] = {'statistic': float(h), 'p_value': float(p)}
            except Exception:
                tests[col] = {'statistic': np.nan, 'p_value': np.nan}
    
    return stats, tests


def compute_adherence_by_cluster(
    labels: np.ndarray,
    harmonized_X: pd.DataFrame,
) -> pd.DataFrame:
    """Compute MARS adherence statistics per cluster."""
    mars_col = 'inst_mars_total'
    if mars_col not in harmonized_X.columns:
        return pd.DataFrame()
    
    df = pd.DataFrame({
        'mars': harmonized_X[mars_col],
        'cluster': labels,
    })
    return df.groupby('cluster')['mars'].agg(['mean', 'std', 'count'])


def run_treatment_validation(
    labels: np.ndarray,
    harmonized_X: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
) -> TreatmentValidationResult:
    """Run full treatment response validation for a clustering.
    
    Tests whether clusters have differential treatment profiles,
    functioning outcomes, and adherence patterns.
    """
    result = TreatmentValidationResult(n_clusters=len(np.unique(labels)))
    
    # Treatment profiles
    result.treatment_profiles = compute_treatment_profiles(labels, harmonized_X)
    
    # Functioning
    func_stats, func_tests = compute_functioning_by_cluster(labels, harmonized_X)
    result.functioning_by_cluster = func_stats
    result.functioning_tests = func_tests
    
    # Adherence
    result.adherence_by_cluster = compute_adherence_by_cluster(labels, harmonized_X)
    
    # Determine if differential
    significant_func = sum(
        1 for t in func_tests.values() 
        if t.get('p_value', 1) < 0.05
    )
    result.has_differential_functioning = significant_func > 0
    
    if result.treatment_profiles is not None and not result.treatment_profiles.empty:
        tx_range = result.treatment_profiles.max() - result.treatment_profiles.min()
        result.has_differential_treatment = bool((tx_range > 0.15).any())
    
    return result
