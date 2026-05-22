"""Clinical validation: treatment response, suicide risk, and biomarker analysis.

These go beyond statistical metrics to assess whether the discovered clusters
have meaningful clinical differences — the ultimate test for precision psychiatry.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def treatment_response_validation(
    labels: pd.Series,
    harmonized_X: pd.DataFrame,
    *,
    treatment_features: list[str] | None = None,
    functioning_features: list[str] | None = None,
) -> dict[str, Any]:
    """Per-cluster treatment profiles and functioning differences.

    Returns
    -------
    dict with:
        - treatment_profiles: DataFrame of medication class proportions per cluster
        - functioning_tests: Kruskal-Wallis H-test per functioning measure
        - functioning_stats: mean/std per cluster per functioning measure
    """
    from scipy.stats import kruskal

    labs = np.asarray(labels)
    valid = labs >= 0
    labs_v = labs[valid]
    X_v = harmonized_X.iloc[valid] if isinstance(harmonized_X, pd.DataFrame) else harmonized_X[valid]

    # Auto-detect treatment and functioning features from column names
    if treatment_features is None:
        treatment_features = [
            c for c in X_v.columns
            if any(kw in c for kw in ["mood_stabilizer", "antipsychotic", "antidepressant",
                                       "benzodiazepine", "lithium", "polypharmacy",
                                       "treatment_", "adherence"])
        ]
    if functioning_features is None:
        functioning_features = [
            c for c in X_v.columns
            if any(kw in c for kw in ["fast_total", "psp_total", "gaf_score",
                                       "eq5d", "functioning", "disability"])
        ]

    # Treatment profiles: proportion of each treatment per cluster
    tx_profiles: dict[str, dict] = {}
    unique_clusters = np.unique(labs_v)
    for feat in treatment_features:
        if feat not in X_v.columns:
            continue
        col = X_v[feat].astype(float)
        for c in unique_clusters:
            mask = labs_v == c
            vals = col.iloc[mask] if isinstance(col, pd.Series) else col[mask]
            mean_val = float(vals.mean()) if vals.notna().any() else float("nan")
            tx_profiles.setdefault(f"cluster_{c}", {})[feat] = mean_val

    # Functioning: Kruskal-Wallis test per feature
    func_tests: list[dict] = []
    func_stats: list[dict] = []
    for feat in functioning_features:
        if feat not in X_v.columns:
            continue
        col = X_v[feat].astype(float)
        groups = []
        for c in unique_clusters:
            mask = labs_v == c
            vals = col.iloc[mask].dropna() if isinstance(col, pd.Series) else col[mask]
            vals = vals[np.isfinite(vals)] if isinstance(vals, np.ndarray) else vals.dropna()
            if len(vals) >= 3:
                groups.append(np.asarray(vals))

        if len(groups) >= 2:
            try:
                h_stat, p_val = kruskal(*groups)
                func_tests.append({
                    "feature": feat,
                    "h_statistic": float(h_stat),
                    "p_value": float(p_val),
                    "significant_005": p_val < 0.05,
                })
            except Exception:
                pass

        # Per-cluster stats
        for c in unique_clusters:
            mask = labs_v == c
            vals = col.iloc[mask] if isinstance(col, pd.Series) else col[mask]
            vals_clean = vals.dropna() if hasattr(vals, "dropna") else vals[np.isfinite(vals)]
            func_stats.append({
                "feature": feat,
                "cluster": int(c),
                "mean": float(vals_clean.mean()) if len(vals_clean) > 0 else float("nan"),
                "std": float(vals_clean.std()) if len(vals_clean) > 1 else float("nan"),
                "n": len(vals_clean),
            })

    return {
        "treatment_profiles": pd.DataFrame(tx_profiles).T if tx_profiles else pd.DataFrame(),
        "functioning_tests": pd.DataFrame(func_tests) if func_tests else pd.DataFrame(),
        "functioning_stats": pd.DataFrame(func_stats) if func_stats else pd.DataFrame(),
        "n_treatment_features": len(treatment_features),
        "n_functioning_features": len(functioning_features),
    }


def suicide_risk_validation(
    labels: pd.Series,
    harmonized_X: pd.DataFrame,
    *,
    risk_features: list[str] | None = None,
) -> dict[str, Any]:
    """Per-cluster suicide risk distribution and concentration test.

    Safety check: no cluster should disproportionately concentrate
    high-risk patients without clinical rationale.
    """
    from scipy.stats import chi2_contingency

    labs = np.asarray(labels)
    valid = labs >= 0
    labs_v = labs[valid]
    X_v = harmonized_X.iloc[valid] if isinstance(harmonized_X, pd.DataFrame) else harmonized_X[valid]

    if risk_features is None:
        risk_features = [
            c for c in X_v.columns
            if any(kw in c for kw in ["suicide", "attempt", "ideation",
                                       "self_harm", "isf_", "cssrs_"])
        ]

    results: dict[str, Any] = {"per_feature": {}}
    unique_clusters = np.unique(labs_v)

    for feat in risk_features:
        if feat not in X_v.columns:
            continue
        col = X_v[feat].astype(float)

        # Per-cluster prevalence
        cluster_rates: dict[int, float] = {}
        for c in unique_clusters:
            mask = labs_v == c
            vals = col.iloc[mask] if isinstance(col, pd.Series) else col[mask]
            vals_clean = vals.dropna() if hasattr(vals, "dropna") else vals[np.isfinite(vals)]
            # For binary risk features: proportion > 0
            if len(vals_clean) > 0:
                cluster_rates[int(c)] = float((vals_clean > 0).mean())
            else:
                cluster_rates[int(c)] = float("nan")

        # Chi-squared concentration test
        try:
            # Binarize: risk present (>0) vs absent
            risk_binary = (col > 0).astype(int)
            ct = pd.crosstab(labs_v, risk_binary.iloc[valid] if isinstance(risk_binary, pd.Series) else risk_binary[valid])
            if ct.shape[1] >= 2:
                chi2, p_val, _, _ = chi2_contingency(ct)
            else:
                chi2, p_val = float("nan"), float("nan")
        except Exception:
            chi2, p_val = float("nan"), float("nan")

        results["per_feature"][feat] = {
            "cluster_rates": cluster_rates,
            "chi2": float(chi2) if np.isfinite(chi2) else float("nan"),
            "p_value": float(p_val) if np.isfinite(p_val) else float("nan"),
            "max_rate": max(cluster_rates.values()) if cluster_rates else float("nan"),
            "min_rate": min(cluster_rates.values()) if cluster_rates else float("nan"),
        }

    # Identify high-risk clusters (rate > 2x overall rate for any risk feature)
    high_risk_clusters: set[int] = set()
    for feat, info in results["per_feature"].items():
        rates = info["cluster_rates"]
        overall = np.mean([r for r in rates.values() if np.isfinite(r)])
        for c, rate in rates.items():
            if np.isfinite(rate) and np.isfinite(overall) and overall > 0 and rate > 2 * overall:
                high_risk_clusters.add(c)

    results["high_risk_clusters"] = sorted(high_risk_clusters)
    results["n_risk_features"] = len(risk_features)

    return results


def biomarker_validation(
    labels: pd.Series,
    harmonized_X: pd.DataFrame,
    *,
    biomarker_features: list[str] | None = None,
) -> dict[str, Any]:
    """Per-cluster biological marker distributions.

    Tests whether clusters have distinct biological signatures (BMI,
    metabolic syndrome, lipids, inflammatory markers).
    """
    from scipy.stats import kruskal

    labs = np.asarray(labels)
    valid = labs >= 0
    labs_v = labs[valid]
    X_v = harmonized_X.iloc[valid] if isinstance(harmonized_X, pd.DataFrame) else harmonized_X[valid]

    if biomarker_features is None:
        biomarker_features = [
            c for c in X_v.columns
            if any(kw in c for kw in ["bmi", "metabolic", "waist", "systolic",
                                       "diastolic", "heart_rate", "cholesterol",
                                       "triglycerides", "glucose", "hba1c",
                                       "crp", "tsh", "weight"])
        ]

    results: list[dict] = []
    unique_clusters = np.unique(labs_v)

    for feat in biomarker_features:
        if feat not in X_v.columns:
            continue
        col = X_v[feat].astype(float)

        groups = []
        stats_per_cluster: dict[int, dict] = {}
        for c in unique_clusters:
            mask = labs_v == c
            vals = col.iloc[mask].dropna() if isinstance(col, pd.Series) else col[mask]
            if hasattr(vals, "dropna"):
                vals = vals.dropna()
            vals_arr = np.asarray(vals, dtype=float)
            vals_arr = vals_arr[np.isfinite(vals_arr)]

            if len(vals_arr) >= 3:
                groups.append(vals_arr)
            stats_per_cluster[int(c)] = {
                "mean": float(vals_arr.mean()) if len(vals_arr) > 0 else float("nan"),
                "std": float(vals_arr.std()) if len(vals_arr) > 1 else float("nan"),
                "n": len(vals_arr),
            }

        h_stat, p_val = (float("nan"), float("nan"))
        if len(groups) >= 2:
            try:
                h_stat, p_val = kruskal(*groups)
                h_stat, p_val = float(h_stat), float(p_val)
            except Exception:
                pass

        results.append({
            "feature": feat,
            "h_statistic": h_stat,
            "p_value": p_val,
            "significant_005": p_val < 0.05 if np.isfinite(p_val) else False,
            "cluster_stats": stats_per_cluster,
        })

    # Sort by significance
    results.sort(key=lambda x: x["p_value"] if np.isfinite(x["p_value"]) else 999)

    return {
        "biomarker_tests": results,
        "n_significant": sum(1 for r in results if r["significant_005"]),
        "n_tested": len(results),
    }
