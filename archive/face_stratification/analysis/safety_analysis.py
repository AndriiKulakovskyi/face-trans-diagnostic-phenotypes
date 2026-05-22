"""Safety-weighted analysis: suicide risk concentration per cluster.

Computes per-cluster suicide attempt and ideation rates, tests whether
any cluster significantly concentrates suicide risk across cohorts.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SafetyAnalysisResult:
    """Results of safety-weighted suicide risk analysis."""
    # Per-cluster rates
    attempt_rates: dict[int, float] = field(default_factory=dict)
    ideation_rates: dict[int, float] = field(default_factory=dict)
    # Statistical tests
    attempt_chi2: float | None = None
    attempt_p_value: float | None = None
    ideation_chi2: float | None = None
    ideation_p_value: float | None = None
    # High-risk clusters
    high_risk_clusters: list[int] = field(default_factory=list)
    # Cross-cohort risk
    cross_cohort_risk: pd.DataFrame | None = None


def run_safety_analysis(
    labels: np.ndarray,
    harmonized_X: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
) -> SafetyAnalysisResult:
    """Analyze suicide risk concentration across clusters.
    
    Tests:
    1. Per-cluster attempt and ideation rates
    2. Chi-squared test: are attempt rates independent of cluster?
    3. Identify high-risk clusters (rate > 1.5x population rate)
    4. Cross-cohort risk: within each cluster, is risk consistent across cohorts?
    """
    from scipy.stats import chi2_contingency
    
    result = SafetyAnalysisResult()
    
    # Attempt rates
    attempt_col = 'sui_ever_attempt'
    if attempt_col in harmonized_X.columns:
        for c in np.unique(labels):
            mask = labels == c
            vals = harmonized_X.loc[mask, attempt_col].dropna()
            if len(vals) > 0:
                result.attempt_rates[int(c)] = float(vals.mean())
        
        # Chi-squared test
        valid = harmonized_X[attempt_col].notna()
        if valid.sum() > 10:
            try:
                ct = pd.crosstab(labels[valid], harmonized_X.loc[valid, attempt_col])
                if ct.shape[1] == 2:
                    chi2, p, _, _ = chi2_contingency(ct)
                    result.attempt_chi2 = float(chi2)
                    result.attempt_p_value = float(p)
            except Exception:
                pass
    
    # Ideation rates
    ideation_col = 'sui_ever_ideation'
    if ideation_col in harmonized_X.columns:
        for c in np.unique(labels):
            mask = labels == c
            vals = harmonized_X.loc[mask, ideation_col].dropna()
            if len(vals) > 0:
                result.ideation_rates[int(c)] = float(vals.mean())
        
        valid = harmonized_X[ideation_col].notna()
        if valid.sum() > 10:
            try:
                ct = pd.crosstab(labels[valid], harmonized_X.loc[valid, ideation_col])
                if ct.shape[1] == 2:
                    chi2, p, _, _ = chi2_contingency(ct)
                    result.ideation_chi2 = float(chi2)
                    result.ideation_p_value = float(p)
            except Exception:
                pass
    
    # Identify high-risk clusters
    if result.attempt_rates:
        pop_rate = np.mean(list(result.attempt_rates.values()))
        result.high_risk_clusters = [
            c for c, r in result.attempt_rates.items()
            if r > 1.5 * pop_rate and pop_rate > 0
        ]
    
    # Cross-cohort risk analysis
    if metadata is not None and 'cohort' in metadata.columns:
        rows = []
        for c in np.unique(labels):
            mask = labels == c
            for cohort in metadata['cohort'].unique():
                cohort_mask = mask & (metadata['cohort'].values == cohort)
                if attempt_col in harmonized_X.columns:
                    vals = harmonized_X.loc[cohort_mask, attempt_col].dropna()
                    if len(vals) > 0:
                        rows.append({
                            'cluster': int(c),
                            'cohort': cohort,
                            'n_patients': int(cohort_mask.sum()),
                            'attempt_rate': float(vals.mean()),
                        })
        if rows:
            result.cross_cohort_risk = pd.DataFrame(rows)
    
    return result
