"""Normalization-scope ablation: global vs per-cohort.

The default Stage A pipeline uses **global** robust normalization: median
and MAD are computed across the entire 11 k cohort, so a BP patient with
MADRS = 30 and a DR patient with MADRS = 30 come out with the same
normalized value. This is the right choice for transdiagnostic analysis
because it preserves raw-score comparability.

The alternative is **per-cohort** normalization: fit the robust stats
separately for each cohort and apply them only to rows in that cohort.
This removes all per-cohort scale differences mathematically, which
sounds attractive but would be destructive — it would guarantee that
cohort labels are invisible to the similarity computation by construction,
destroying any transdiagnostic signal that depends on absolute score
magnitudes.

This module runs both variants end-to-end, fits the same Stage B
composite embedding to each, clusters both with the same k-means
configuration, and reports the following for each variant:

- silhouette, ARI, NMI, V-measure, homogeneity, completeness, cohort
  entropy (from :mod:`face_stratification.clustering.metrics`),
- bootstrap ARI stability at the chosen ``k``,
- per-cohort purity (how much each DSM cohort is split across clusters),
- the ARI of the two variants' clusterings against each other.

The comparison gives a quantitative answer to *"does the choice of
normalization scope change the stratification?"*. If both variants
produce similar clusterings, the global-normalization choice is robust.
If they diverge dramatically, that divergence is itself a signal about
how much of the clustering depends on cohort-level distributional
differences.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from face_stratification.clustering.algorithms import (
    bootstrap_stability,
    run_kmeans,
)
from face_stratification.clustering.metrics import ClusterMetrics
from face_stratification.harmonization.harmonizer import HarmonizedDataset
from face_stratification.harmonization.normalization import (
    fit_normalization,
    fit_per_cohort_normalization,
    transform_normalization,
    transform_per_cohort_normalization,
)
from face_stratification.graph.patient_similarity import build_multiplex_graph
from face_stratification.models.composite import ConcatenatedEmbedding

logger = logging.getLogger(__name__)


@dataclass
class AblationVariantResult:
    """Result of a single normalization-scope variant."""

    scope: str  # "global" or "per_cohort"
    embedding_shape: tuple[int, int]
    cluster_labels: pd.Series
    metrics: ClusterMetrics
    bootstrap_mean_ari: float
    bootstrap_std_ari: float
    n_edges: int
    n_isolated_nodes: int


@dataclass
class AblationResult:
    """Full ablation output comparing global vs per-cohort normalization."""

    global_result: AblationVariantResult
    per_cohort_result: AblationVariantResult
    variant_vs_variant_ari: float
    variant_vs_variant_nmi: float
    k_clusters: int
    config: dict[str, Any] = field(default_factory=dict)

    def summary_table(self) -> pd.DataFrame:
        """Side-by-side comparison table of the two variants' metrics."""
        rows = []
        for variant in (self.global_result, self.per_cohort_result):
            m = variant.metrics
            rows.append(
                {
                    "normalization_scope": variant.scope,
                    "n_patients": m.n_patients,
                    "n_clusters": m.n_clusters,
                    "silhouette": m.silhouette,
                    "ari_vs_cohort": m.ari_vs_reference,
                    "nmi_vs_cohort": m.nmi_vs_reference,
                    "v_measure_vs_cohort": m.v_measure,
                    "homogeneity": m.homogeneity,
                    "completeness": m.completeness,
                    "cohort_entropy_mean": m.cohort_entropy_mean,
                    "bootstrap_mean_ari": variant.bootstrap_mean_ari,
                    "bootstrap_std_ari": variant.bootstrap_std_ari,
                }
            )
        return pd.DataFrame(rows).set_index("normalization_scope")


# ─── Main entry point ────────────────────────────────────────────────────────


def run_normalization_ablation(
    dataset: HarmonizedDataset,
    *,
    k_clusters: int = 5,
    k_neighbours_graph: int = 10,
    pca_dim: int = 8,
    td_spectral_dim: int = 16,
    multiplex_spectral_dim: int = 32,
    n_bootstraps: int = 20,
    random_state: int = 0,
) -> AblationResult:
    """Run the global vs per-cohort normalization ablation end-to-end.

    For each scope:

    1. Fit the appropriate normalization and transform ``dataset.X``.
    2. Build the masked multiplex graph on the normalized matrix.
    3. Fit the default 3-view composite embedding.
    4. Cluster with k-means at ``k_clusters``.
    5. Compute cluster metrics against the DSM cohort label.
    6. Compute bootstrap ARI stability.

    Finally, report the ARI/NMI between the two variants' clusterings so
    we can see whether the choice of normalization changes the
    stratification at all.
    """
    cohort_reference = dataset.metadata["cohort"].values

    results: dict[str, AblationVariantResult] = {}
    cluster_series: dict[str, pd.Series] = {}

    for scope in ("global", "per_cohort"):
        logger.info("Running ablation variant: %s", scope)

        if scope == "global":
            stats = fit_normalization(dataset.X, dataset.schema)
            Xn = transform_normalization(dataset.X, stats)
        else:
            per_stats = fit_per_cohort_normalization(dataset.X, dataset.schema)
            Xn = transform_per_cohort_normalization(dataset.X, per_stats)

        graph, _block_graphs, _td = build_multiplex_graph(
            Xn, dataset.schema, k=k_neighbours_graph, metadata=dataset.metadata
        )

        model = ConcatenatedEmbedding.build_default(
            pca_dim=pca_dim,
            td_spectral_dim=td_spectral_dim,
            multiplex_spectral_dim=multiplex_spectral_dim,
        )
        embedding = model.fit_transform(dataset, graph=graph)

        assignment = run_kmeans(
            embedding.values,
            n_clusters=k_clusters,
            random_state=random_state,
            reference_labels=cohort_reference,
        )
        bs = bootstrap_stability(
            embedding.values,
            n_clusters=k_clusters,
            n_bootstraps=n_bootstraps,
            random_state=random_state,
        )

        results[scope] = AblationVariantResult(
            scope=scope,
            embedding_shape=tuple(embedding.values.shape),
            cluster_labels=assignment.labels,
            metrics=assignment.metrics,  # type: ignore[arg-type]
            bootstrap_mean_ari=float(bs["mean_ari"]),
            bootstrap_std_ari=float(bs["std_ari"]),
            n_edges=graph.number_of_edges(),
            n_isolated_nodes=embedding.n_isolated_nodes,
        )
        cluster_series[scope] = assignment.labels

    # Cross-variant comparison
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
    )
    common_index = cluster_series["global"].index.intersection(
        cluster_series["per_cohort"].index
    )
    g_labels = cluster_series["global"].loc[common_index].to_numpy()
    p_labels = cluster_series["per_cohort"].loc[common_index].to_numpy()
    cross_ari = float(adjusted_rand_score(g_labels, p_labels))
    cross_nmi = float(normalized_mutual_info_score(g_labels, p_labels))

    return AblationResult(
        global_result=results["global"],
        per_cohort_result=results["per_cohort"],
        variant_vs_variant_ari=cross_ari,
        variant_vs_variant_nmi=cross_nmi,
        k_clusters=k_clusters,
        config={
            "k_clusters": k_clusters,
            "k_neighbours_graph": k_neighbours_graph,
            "pca_dim": pca_dim,
            "td_spectral_dim": td_spectral_dim,
            "multiplex_spectral_dim": multiplex_spectral_dim,
            "n_bootstraps": n_bootstraps,
            "random_state": random_state,
        },
    )
