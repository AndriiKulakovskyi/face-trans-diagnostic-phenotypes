"""Per-cluster feature enrichment with Benjamini-Hochberg FDR.

For each cluster, we compare the distribution of every unified feature
inside that cluster against its distribution outside, using the
non-parametric Mann-Whitney U test (robust to non-normality and heavy
tails, which is the norm for psychiatric scales). P-values are adjusted
across all (n_clusters × n_features) tests with Benjamini-Hochberg FDR
(default q=0.05).

The output is a tidy DataFrame with one row per (cluster, feature),
sorted by cluster then by absolute effect size. Stage D / clinical
interpretation can filter it to the top N enriched features per cluster.

No imputation: missing values are dropped pairwise before the test, and
every enrichment row reports the size of the inside / outside sample
actually used. Features that are too sparse inside a cluster (< 5
observed values) are silently skipped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureEnrichmentResult:
    """Full enrichment table + summary statistics."""

    table: pd.DataFrame
    q_threshold: float
    n_tests: int
    n_significant: int

    def top_per_cluster(self, top_n: int = 10) -> pd.DataFrame:
        """Return the top-``top_n`` features per cluster by abs effect size."""
        return (
            self.table
            .loc[self.table["significant"]]
            .sort_values(["cluster", "abs_effect"], ascending=[True, False])
            .groupby("cluster", as_index=False)
            .head(top_n)
        )

    def cluster_summary(self) -> pd.DataFrame:
        """Per-cluster count + mean absolute effect size."""
        g = self.table.loc[self.table["significant"]].groupby("cluster")
        return pd.DataFrame(
            {
                "n_significant_features": g.size(),
                "mean_abs_effect": g["abs_effect"].mean(),
            }
        )


def _rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation from a Mann-Whitney U1 statistic.

    Given :func:`scipy.stats.mannwhitneyu` with arguments ``(inside, outside)``,
    the returned ``U1`` is the number of pairs where ``inside > outside``. The
    rank-biserial correlation in the standard Wendt (1972) convention is

        rb  =  2 * U1 / (n1 * n2)  -  1

    This maps:

    - inside strictly ``>`` outside (``U1 = n1 * n2``) → rb = +1
    - inside strictly ``<`` outside (``U1 = 0``)       → rb = −1
    - balanced ranks (``U1 = n1 * n2 / 2``)            → rb =  0

    **Positive** therefore means the feature median is **higher inside** the
    cluster than outside, which is the standard psychological effect-size
    interpretation.

    (An earlier version of this module used the inverse convention
    ``rb = 1 − 2 U1 / n1 n2``, which flipped the sign of every effect; that has
    since been corrected to the form computed below.)
    """
    if n1 == 0 or n2 == 0:
        return float("nan")
    return (2.0 * u_stat) / (n1 * n2) - 1.0


def _benjamini_hochberg(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """Classic BH step-up procedure. Returns a boolean reject mask."""
    n = pvals.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    below = ranked <= thresholds
    if not below.any():
        return np.zeros(n, dtype=bool)
    cutoff = np.max(np.where(below)[0])
    reject = np.zeros(n, dtype=bool)
    reject[order[: cutoff + 1]] = True
    return reject


def compute_cluster_feature_enrichment(
    X: pd.DataFrame,
    cluster_labels: pd.Series,
    *,
    q_threshold: float = 0.05,
    min_samples_per_side: int = 5,
    feature_subset: list[str] | None = None,
) -> FeatureEnrichmentResult:
    """Enrich every (cluster × feature) pair via Mann-Whitney + BH FDR.

    Parameters
    ----------
    X:
        Unified harmonized matrix (raw or normalized — the test is rank-based
        so normalization does not change the outcome). Must be indexed
        identically to ``cluster_labels``.
    cluster_labels:
        Series indexed identically to ``X`` with integer cluster ids. ``-1``
        is treated as noise and excluded.
    q_threshold:
        Benjamini-Hochberg significance threshold.
    min_samples_per_side:
        A (cluster, feature) test is skipped if either inside or outside
        has fewer than this many observed (non-NaN) values.
    feature_subset:
        Optional list of feature ids to restrict the test to. ``None``
        uses every column of ``X``.
    """
    try:
        from scipy.stats import mannwhitneyu
    except ImportError as exc:
        raise ImportError(
            "Feature enrichment requires scipy. Install the 'stratification' extra."
        ) from exc

    if not X.index.equals(cluster_labels.index):
        raise ValueError(
            "X and cluster_labels must share the same index. Align them first."
        )

    clusters = sorted(c for c in cluster_labels.unique() if c >= 0)
    features = feature_subset if feature_subset is not None else list(X.columns)

    records = []
    for cluster in clusters:
        mask = (cluster_labels == cluster).to_numpy()
        inside_idx = np.where(mask)[0]
        outside_idx = np.where(~mask)[0]

        for feat in features:
            col = X[feat].to_numpy()
            inside = col[inside_idx]
            outside = col[outside_idx]
            inside = inside[np.isfinite(inside)]
            outside = outside[np.isfinite(outside)]

            if inside.size < min_samples_per_side or outside.size < min_samples_per_side:
                continue

            # Mann-Whitney U (two-sided) with tie correction
            try:
                u, p = mannwhitneyu(inside, outside, alternative="two-sided")
            except ValueError:
                continue
            effect = _rank_biserial(float(u), int(inside.size), int(outside.size))
            records.append(
                {
                    "cluster": int(cluster),
                    "feature_id": feat,
                    "n_inside": int(inside.size),
                    "n_outside": int(outside.size),
                    "median_inside": float(np.median(inside)),
                    "median_outside": float(np.median(outside)),
                    "u_statistic": float(u),
                    "p_value": float(p),
                    "effect_rank_biserial": float(effect),
                    "abs_effect": float(abs(effect)),
                }
            )

    if not records:
        logger.warning("No cluster × feature enrichments computed.")
        empty = pd.DataFrame(
            columns=[
                "cluster", "feature_id", "n_inside", "n_outside",
                "median_inside", "median_outside", "u_statistic", "p_value",
                "effect_rank_biserial", "abs_effect", "p_value_bh", "significant",
            ]
        )
        return FeatureEnrichmentResult(
            table=empty, q_threshold=q_threshold, n_tests=0, n_significant=0
        )

    table = pd.DataFrame(records)
    pvals = table["p_value"].to_numpy()
    reject = _benjamini_hochberg(pvals, q_threshold)

    # Compute BH-adjusted p-values (for reporting) via the step-up procedure
    n = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    table["p_value_bh"] = np.nan
    table.loc[table.index[order], "p_value_bh"] = adj
    table["significant"] = reject

    table = table.sort_values(
        ["cluster", "abs_effect"], ascending=[True, False]
    ).reset_index(drop=True)

    return FeatureEnrichmentResult(
        table=table,
        q_threshold=q_threshold,
        n_tests=int(n),
        n_significant=int(reject.sum()),
    )
