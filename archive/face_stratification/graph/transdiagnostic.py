"""Data-driven transdiagnostic feature selection + pure transdiagnostic graph.

The "transdiagnostic graph" is a parallel, methodologically conservative view
of the cohort: it uses only features whose **observed-value coverage** is
above a threshold in *every* cohort. This guarantees that every edge in this
graph is supported by real, comparable measurements — no feature is "partially
transdiagnostic" and no imputation is ever used.

Construction steps
------------------
1. For every feature in the harmonized matrix, compute per-cohort observed
   coverage (fraction of non-NaN values within each cohort).
2. Admit a feature to the transdiagnostic set if its minimum across-cohort
   coverage is ``>= min_cohort_coverage`` (from the schema config).
3. Build a single masked kNN graph over the selected feature set using the
   configured metric, with the semantic overlap constraint
   ``ceil(min_shared_features_fraction * |selected|)``.

The result is returned as a :class:`TransdiagnosticGraphResult` containing
both the selected set (for auditing) and the edge list. The graph builder in
:mod:`face_stratification.graph.patient_similarity` wires this into the
multiplex graph as a distinct edge type called ``transdiagnostic``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from face_stratification.graph.masked_similarity import (
    Metric,
    masked_knn_edges,
)
from face_stratification.harmonization.feature_schema import (
    FeatureSchema,
)

logger = logging.getLogger(__name__)


# ─── Public result types ──────────────────────────────────────────────────────


@dataclass
class TransdiagnosticFeatureSet:
    """Audit record describing which features made it into the transdiagnostic set."""

    feature_ids: tuple[str, ...]
    min_cohort_coverage: float
    per_cohort_coverage: pd.DataFrame  # rows = feature_id, columns = cohort
    excluded_by_coverage: tuple[str, ...]
    excluded_by_config: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_ids", tuple(self.feature_ids))
        object.__setattr__(self, "excluded_by_coverage", tuple(self.excluded_by_coverage))
        object.__setattr__(self, "excluded_by_config", tuple(self.excluded_by_config))

    @property
    def n_selected(self) -> int:
        return len(self.feature_ids)


@dataclass
class TransdiagnosticGraphResult:
    """Output of :func:`build_transdiagnostic_graph`."""

    feature_set: TransdiagnosticFeatureSet
    edges: list[tuple[int, int, float, int, float]]  # (src, dst, sim, overlap, dist)
    metric: Metric
    min_shared_features: int
    n_nodes: int = 0
    bandwidth: float = 1.0

    @property
    def n_edges(self) -> int:
        return len(self.edges)


# ─── Selection ────────────────────────────────────────────────────────────────


def compute_per_cohort_coverage(
    X: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Return a ``(n_features × n_cohorts)`` DataFrame of observed-value coverage.

    Cell ``(f, c)`` is the fraction of patients in cohort ``c`` for which
    feature ``f`` has a non-NaN value. Works whether ``cohort`` lives as a
    MultiIndex level on ``metadata`` or as a plain column.
    """
    # Resolve the cohort labels without triggering the "both level and column"
    # ambiguity that pandas raises when a name shadows an index level.
    if "cohort" in metadata.index.names:
        cohort_labels = metadata.index.get_level_values("cohort")
    elif "cohort" in metadata.columns:
        cohort_labels = metadata["cohort"]
    else:
        raise ValueError("metadata must expose a 'cohort' index level or column")

    rows = []
    for cohort in pd.unique(cohort_labels):
        mask = np.asarray(cohort_labels == cohort)
        sub = X.loc[mask]
        coverage = sub.notna().mean(axis=0)
        coverage.name = str(cohort)
        rows.append(coverage)
    out = pd.concat(rows, axis=1)
    out.index.name = "feature_id"
    return out


def select_transdiagnostic_features(
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    schema: FeatureSchema,
) -> TransdiagnosticFeatureSet:
    """Select the feature subset that is genuinely transdiagnostic in the data.

    A feature is selected iff its per-cohort observed coverage is
    ``>= schema.transdiagnostic_selection.min_cohort_coverage`` in **every**
    cohort. Features listed in the config's ``excluded_feature_ids`` are never
    admitted regardless of coverage.
    """
    cfg = schema.transdiagnostic_selection
    coverage = compute_per_cohort_coverage(X, metadata)

    # Per-feature minimum across cohorts
    min_across = coverage.min(axis=1)
    passes_coverage = min_across >= cfg.min_cohort_coverage
    selected = set(coverage.index[passes_coverage])

    excluded_by_coverage = tuple(
        sorted(set(coverage.index) - selected)
    )
    # Apply hard exclusion list
    explicitly_excluded = set(cfg.excluded_feature_ids)
    final = selected - explicitly_excluded
    excluded_by_config = tuple(
        sorted(selected & explicitly_excluded)
    )

    # Preserve the schema order for reproducibility
    final_ids = tuple(f.id for f in schema.features if f.id in final)

    logger.info(
        "Transdiagnostic selection: %d / %d features passed (%.1f%%)",
        len(final_ids),
        len(coverage),
        100.0 * len(final_ids) / max(1, len(coverage)),
    )

    return TransdiagnosticFeatureSet(
        feature_ids=final_ids,
        min_cohort_coverage=cfg.min_cohort_coverage,
        per_cohort_coverage=coverage,
        excluded_by_coverage=excluded_by_coverage,
        excluded_by_config=excluded_by_config,
    )


# ─── Graph construction ──────────────────────────────────────────────────────


def _compute_feature_ranges(X_sub: pd.DataFrame) -> np.ndarray:
    """Empirical range (max − min) per column, safely handling all-NaN columns."""
    with np.errstate(invalid="ignore"):
        max_vals = np.nanmax(X_sub.to_numpy(dtype=np.float32), axis=0)
        min_vals = np.nanmin(X_sub.to_numpy(dtype=np.float32), axis=0)
    rng = max_vals - min_vals
    rng = np.where(np.isfinite(rng) & (rng > 0), rng, 1.0)
    return rng.astype(np.float32)


def build_transdiagnostic_graph(
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    schema: FeatureSchema,
    *,
    k: int = 10,
    feature_set: TransdiagnosticFeatureSet | None = None,
) -> TransdiagnosticGraphResult:
    """Build the pure transdiagnostic masked-kNN graph (no imputation).

    Parameters
    ----------
    X:
        Unified harmonized matrix (raw or normalized — normalization improves
        cosine behaviour on continuous features).
    metadata:
        Side metadata DataFrame indexed identically to ``X``.
    schema:
        Loaded feature schema. Its ``transdiagnostic_selection`` config drives
        both feature selection and the overlap constraint.
    k:
        Target number of neighbours per node. Nodes may end up with fewer
        neighbours if the overlap constraint disqualifies candidates — this is
        the desired honest behaviour.
    feature_set:
        Optional pre-computed selection (useful for audit notebooks that want
        to inspect the selection before building the graph).
    """
    if feature_set is None:
        feature_set = select_transdiagnostic_features(X, metadata, schema)

    if feature_set.n_selected == 0:
        logger.warning("Transdiagnostic feature set is empty; no graph built.")
        return TransdiagnosticGraphResult(
            feature_set=feature_set,
            edges=[],
            metric=schema.transdiagnostic_selection.metric,
            min_shared_features=0,
        )

    cfg = schema.transdiagnostic_selection
    selected_cols = list(feature_set.feature_ids)
    X_sub = X[selected_cols]

    min_shared = max(
        1,
        math.ceil(cfg.min_shared_features_fraction * feature_set.n_selected),
    )
    metric: Metric = cfg.metric

    # Gower needs per-feature ranges computed on the *observed* subset.
    feature_ranges = (
        _compute_feature_ranges(X_sub) if metric == "gower" else None
    )

    edges = masked_knn_edges(
        X_sub.to_numpy(dtype=np.float32),
        metric=metric,
        k=k,
        min_shared_features=min_shared,
        feature_ranges=feature_ranges,
    )

    # Bandwidth is the median of observed distances; used by the multiplex
    # builder for Gaussian edge weighting.
    if edges:
        dist_arr = np.array([e[4] for e in edges], dtype=np.float32)
        bw = float(np.median(dist_arr[np.isfinite(dist_arr)]))
        if not np.isfinite(bw) or bw <= 0:
            bw = 1.0
    else:
        bw = 1.0

    return TransdiagnosticGraphResult(
        feature_set=feature_set,
        edges=edges,
        metric=metric,
        min_shared_features=min_shared,
        n_nodes=X.shape[0],
        bandwidth=bw,
    )


# ─── Tiered transdiagnostic graphs ───────────────────────────────────────────


def _remove_redundant_features(
    X: pd.DataFrame,
    feature_ids: list[str],
    threshold: float = 0.85,
) -> list[str]:
    """Remove redundant features by correlation threshold.

    For each pair with |r| > threshold, drop the one with higher missingness.
    """
    if len(feature_ids) < 2:
        return list(feature_ids)

    sub = X[feature_ids]
    corr = sub.corr(method="pearson", min_periods=5).to_numpy(dtype=np.float64)
    np.fill_diagonal(corr, 0.0)
    corr = np.where(np.isfinite(corr), corr, 0.0)

    missingness = sub.isna().mean(axis=0).to_numpy()

    dropped: set[int] = set()
    n = len(feature_ids)
    for i in range(n):
        if i in dropped:
            continue
        for j in range(i + 1, n):
            if j in dropped:
                continue
            if abs(corr[i, j]) > threshold:
                victim = i if missingness[i] > missingness[j] else j
                dropped.add(victim)

    kept = [f for idx, f in enumerate(feature_ids) if idx not in dropped]

    if dropped:
        logger.info(
            "Redundancy guard: dropped %d / %d features (|r| > %.2f)",
            len(dropped),
            n,
            threshold,
        )

    return kept



def build_tiered_transdiagnostic_graphs(
    X: pd.DataFrame,
    patient_ids: np.ndarray,
    metadata: pd.DataFrame,
    schema: Any,
    *,
    k: int = 10,
    redundancy_threshold: float = 0.85,
) -> dict[str, nx.Graph]:
    """Build tiered transdiagnostic graphs with redundancy guard.

    Tiers:
    - tier1_strict: features present in all 4 cohorts with >=50% coverage
    - tier2_relaxed: features present in >=3 cohorts with >=30% coverage
    - tier3_pairwise: features shared by 2+ cohorts with >=40% coverage in each

    Within each tier, features with Pearson |r| > redundancy_threshold are
    deduplicated (keep the one with lower overall missingness).

    Parameters
    ----------
    X : harmonized feature matrix
    patient_ids : patient identifier array
    metadata : DataFrame with 'cohort' column
    schema : loaded feature schema
    k : neighbors per patient
    redundancy_threshold : correlation threshold for deduplication

    Returns
    -------
    dict mapping tier name -> nx.Graph
    """
    patient_ids = np.asarray(patient_ids)
    coverage = compute_per_cohort_coverage(X, metadata)

    # --- Tier 1: all 4 cohorts, >=50% coverage in each ---
    tier1_mask = coverage.min(axis=1) >= 0.50
    tier1_ids = set(coverage.index[tier1_mask])

    # --- Tier 2: >=3 cohorts with >=30% coverage (excluding tier 1) ---
    passes_30 = coverage >= 0.30
    n_passing_30 = passes_30.sum(axis=1)
    tier2_mask = (n_passing_30 >= 3) & (~coverage.index.isin(tier1_ids))
    tier2_ids = set(coverage.index[tier2_mask])

    # --- Tier 3: >=2 cohorts with >=40% coverage (excluding tiers 1 & 2) ---
    passes_40 = coverage >= 0.40
    n_passing_40 = passes_40.sum(axis=1)
    tier3_mask = (
        (n_passing_40 >= 2)
        & (~coverage.index.isin(tier1_ids))
        & (~coverage.index.isin(tier2_ids))
    )
    tier3_ids = set(coverage.index[tier3_mask])

    raw_tiers: dict[str, list[str]] = {
        "tier1_strict": sorted(tier1_ids),
        "tier2_relaxed": sorted(tier2_ids),
        "tier3_pairwise": sorted(tier3_ids),
    }

    for name, fids in raw_tiers.items():
        logger.info("Tier %s: %d features before redundancy guard", name, len(fids))

    # --- Deduplicate within each tier ---
    deduped_tiers: dict[str, list[str]] = {}
    for name, fids in raw_tiers.items():
        deduped_tiers[name] = _remove_redundant_features(
            X, fids, threshold=redundancy_threshold,
        )

    # --- Build a kNN graph per tier ---
    metric: Metric = "cosine"
    if hasattr(schema, "transdiagnostic_selection"):
        metric = schema.transdiagnostic_selection.metric

    graphs: dict[str, nx.Graph] = {}
    for name, fids in deduped_tiers.items():
        G = nx.Graph()
        for pos in range(len(patient_ids)):
            G.add_node(pos, patient_id=patient_ids[pos])

        if len(fids) < 2:
            logger.warning("Tier %s: fewer than 2 features; empty graph.", name)
            graphs[name] = G
            continue

        X_sub = X[fids]
        arr = X_sub.to_numpy(dtype=np.float32)
        n_features = arr.shape[1]
        min_shared = max(1, math.ceil(0.5 * n_features))

        feature_ranges = (
            _compute_feature_ranges(X_sub) if metric == "gower" else None
        )

        edges = masked_knn_edges(
            arr,
            metric=metric,
            k=k,
            min_shared_features=min_shared,
            feature_ranges=feature_ranges,
        )

        if edges:
            dist_arr = np.array([e[4] for e in edges], dtype=np.float32)
            finite = dist_arr[np.isfinite(dist_arr)]
            bandwidth = float(np.median(finite)) if finite.size else 1.0
            if not np.isfinite(bandwidth) or bandwidth <= 0:
                bandwidth = 1.0
        else:
            bandwidth = 1.0

        for src, dst, _sim, overlap_count, dist in edges:
            gauss = float(np.exp(-(dist * dist) / (2.0 * bandwidth * bandwidth)))
            confidence = overlap_count / max(1, n_features)
            weight = gauss * confidence
            G.add_edge(int(src), int(dst), weight=weight)

        logger.info(
            "Tier %s: %d features, %d nodes, %d edges, bandwidth=%.4f",
            name,
            len(fids),
            G.number_of_nodes(),
            G.number_of_edges(),
            bandwidth,
        )
        graphs[name] = G

    return graphs
