"""End-to-end Stage C orchestrator.

Given a Stage A :class:`HarmonizedDataset` and a Stage B
:class:`PatientEmbedding`, the Stage C pipeline:

1. Runs the **algorithm × k ablation grid** to identify the (algorithm, k)
   combination that maximizes the optimization score.
2. Runs the **embedding view ablation** at the chosen k to quantify which
   sub-views carry the signal.
3. Runs **multi-algorithm + multi-seed base clusterings** at the chosen k
   to populate the consensus pool (16 base clusterings by default:
   KMeans × 5 + GMM × 5 + Ward × 1 + Spectral × 5).
4. Builds the **co-association matrix** and the **consensus partition**
   via Ward hierarchical clustering on ``1 − M``.
5. Computes **per-patient confidence scores** from the co-association
   matrix.
6. Runs the **formal cluster-vs-DSM comparison** (chi², Cramér's V,
   purity, entropy, sklearn metrics) on the final consensus partition.
7. Computes **per-cluster feature enrichment** (Mann-Whitney U + BH FDR).
8. Assembles **per-cluster narrative cards** with French vignettes.

Returns a :class:`StageCResult` with all artifacts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from face_stratification.analysis.enrichment import (
    FeatureEnrichmentResult,
    compute_cluster_feature_enrichment,
)
from face_stratification.harmonization.harmonizer import HarmonizedDataset
from face_stratification.models.base import PatientEmbedding
from face_stratification.stage_c.ablation import (
    pick_best_configuration,
    run_algorithm_k_grid,
    run_embedding_view_ablation,
)
from face_stratification.stage_c.algorithms import ALGORITHMS, run_algorithm
from face_stratification.stage_c.comparison import (
    FullDSMComparison,
    full_dsm_comparison,
)
from face_stratification.stage_c.consensus import (
    ConsensusResult,
    run_consensus_clustering,
)
from face_stratification.stage_c.narrative import (
    ClusterCard,
    build_cluster_cards,
)

logger = logging.getLogger(__name__)


# ─── Result type ──────────────────────────────────────────────────────────────


@dataclass
class StageCResult:
    """Everything Stage C produces, ready to be persisted."""

    # Optimization
    algorithm_k_grid: pd.DataFrame
    best_configuration: dict[str, Any]
    view_ablation: pd.DataFrame

    # Final clustering
    consensus: ConsensusResult
    final_labels: pd.Series

    # Comparison + enrichment
    dsm_comparison: FullDSMComparison
    enrichment: FeatureEnrichmentResult
    cluster_cards: dict[int, ClusterCard]

    # Stage B baseline (kmeans @ k=8) for direct comparison
    stage_b_baseline_metrics: dict[str, Any]
    stage_c_vs_stage_b_ari: float

    # Provenance
    config: dict[str, Any] = field(default_factory=dict)


# ─── Main pipeline ───────────────────────────────────────────────────────────


def run_stage_c(
    dataset: HarmonizedDataset,
    embedding: PatientEmbedding,
    *,
    k_grid_values: tuple[int, ...] = (4, 5, 6, 7, 8, 9, 10),
    consensus_k: int | None = None,
    base_algorithms: tuple[str, ...] = ("kmeans", "gmm", "ward", "spectral"),
    n_seeds_per_algorithm: int = 5,
    keep_consensus_matrix: bool = False,
) -> StageCResult:
    """Run the full Stage C pipeline.

    Parameters
    ----------
    dataset:
        Stage A :class:`HarmonizedDataset`.
    embedding:
        Stage B :class:`PatientEmbedding` (the 56-dim composite by default).
    k_grid_values:
        ``k`` values to sweep in the algorithm × k ablation.
    consensus_k:
        Number of consensus clusters to extract. If ``None``, picked from
        the best configuration of the algorithm × k grid.
    base_algorithms:
        Algorithms to include in the consensus pool.
    n_seeds_per_algorithm:
        Number of random seeds per stochastic algorithm. Ward is run once
        regardless because it is deterministic.
    keep_consensus_matrix:
        If True, the (N, N) co-association matrix is kept in the result
        (≈ 485 MB on the full cohort). Default is False — drop after
        confidence is computed.
    """
    cohort_reference = dataset.metadata.loc[embedding.values.index, "cohort"].values

    # ─── 1. Algorithm × k grid ─────────────────────────────────────────────
    logger.info("Stage C step 1/8: algorithm × k ablation grid")
    grid = run_algorithm_k_grid(
        embedding.values,
        reference_labels=cohort_reference,
        k_values=k_grid_values,
        algorithms=base_algorithms,
        random_states=(0,),
    )

    best = pick_best_configuration(grid)
    logger.info(
        "Best configuration: algorithm=%s k=%d (silhouette=%.3f, ari=%.3f, score=%.3f)",
        best["algorithm"], best["k"], best["silhouette"], best["ari"], best["optimization_score"]
    )

    final_k = consensus_k if consensus_k is not None else int(best["k"])

    # ─── 2. View ablation at the chosen k ──────────────────────────────────
    logger.info("Stage C step 2/8: embedding view ablation at k=%d", final_k)
    view_ablation = run_embedding_view_ablation(
        embedding.values,
        view_dims=embedding.view_dims,
        reference_labels=cohort_reference,
        n_clusters=final_k,
        algorithm="kmeans",
    )

    # ─── 3. Build base clusterings ─────────────────────────────────────────
    logger.info(
        "Stage C step 3/8: building base clusterings (%d algorithms × seeds)",
        len(base_algorithms),
    )
    base_labels: dict[str, np.ndarray] = {}
    for algo in base_algorithms:
        is_stochastic = ALGORITHMS[algo]["stochastic"]
        seeds = tuple(range(n_seeds_per_algorithm)) if is_stochastic else (0,)
        for seed in seeds:
            try:
                a = run_algorithm(
                    algo,
                    embedding.values,
                    n_clusters=final_k,
                    random_state=seed,
                    reference_labels=cohort_reference,
                    silhouette_sample_size=2000,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("base clustering %s seed=%d failed: %s", algo, seed, exc)
                continue
            base_labels[f"{algo}_s{seed}"] = a.labels.to_numpy()
            logger.info(
                "  %s seed=%d: silhouette=%.3f ari=%.3f",
                algo, seed,
                a.metrics.silhouette if a.metrics else float("nan"),
                a.metrics.ari_vs_reference if a.metrics else float("nan"),
            )

    # ─── 4 + 5. Consensus + per-patient confidence ─────────────────────────
    logger.info("Stage C step 4-5/8: consensus matrix + per-patient confidence")
    consensus = run_consensus_clustering(
        base_labels,
        n_clusters=final_k,
        embedding_index=embedding.values.index,
        linkage_method="average",
        keep_matrix=keep_consensus_matrix,
    )
    final_labels = consensus.labels

    # ─── 6. Formal DSM comparison ──────────────────────────────────────────
    logger.info("Stage C step 6/8: formal cluster-vs-DSM comparison")
    dsm = full_dsm_comparison(final_labels, cohort_reference)
    logger.info(
        "Final clustering: chi²=%.0f (dof=%d, p=%.2e), Cramér's V=%.3f",
        dsm.chi2_statistic, dsm.chi2_dof, dsm.chi2_p_value, dsm.cramers_v
    )
    logger.info(
        "ARI=%.3f NMI=%.3f V=%.3f mean cohort entropy=%.3f bits (max=%.3f)",
        dsm.ari, dsm.nmi, dsm.v_measure,
        dsm.mean_cluster_entropy_bits, dsm.max_possible_entropy_bits,
    )

    # ─── 7. Feature enrichment ─────────────────────────────────────────────
    logger.info("Stage C step 7/8: per-cluster feature enrichment")
    enrichment = compute_cluster_feature_enrichment(
        dataset.X.loc[embedding.values.index],
        final_labels,
        q_threshold=0.05,
    )
    logger.info(
        "Enrichment: %d / %d significant",
        enrichment.n_significant, enrichment.n_tests,
    )

    # ─── 8. Cluster narrative cards ────────────────────────────────────────
    logger.info("Stage C step 8/8: per-cluster narrative cards")
    csv_paths = {
        "bp": "data/BP.csv",
        "sz": "data/SZ.csv",
        "dr": "data/DR.csv",
        "asp": "data/ASP.csv",
    }
    cards = build_cluster_cards(
        cluster_labels=final_labels,
        confidence=consensus.confidence,
        embedding=embedding.values,
        metadata=dataset.metadata.loc[embedding.values.index],
        enrichment_table=enrichment.table,
        csv_paths=csv_paths,
    )

    # ─── Stage B baseline comparison ───────────────────────────────────────
    from face_stratification.clustering.algorithms import run_kmeans
    stage_b_assignment = run_kmeans(
        embedding.values,
        n_clusters=8,
        random_state=0,
        reference_labels=cohort_reference,
    )
    from sklearn.metrics import adjusted_rand_score
    cross_ari = float(
        adjusted_rand_score(
            stage_b_assignment.labels.to_numpy(),
            final_labels.to_numpy(),
        )
    )
    stage_b_metrics = {
        "k": 8,
        "silhouette": stage_b_assignment.metrics.silhouette,
        "ari_vs_cohort": stage_b_assignment.metrics.ari_vs_reference,
        "nmi_vs_cohort": stage_b_assignment.metrics.nmi_vs_reference,
    }

    return StageCResult(
        algorithm_k_grid=grid,
        best_configuration=best,
        view_ablation=view_ablation,
        consensus=consensus,
        final_labels=final_labels,
        dsm_comparison=dsm,
        enrichment=enrichment,
        cluster_cards=cards,
        stage_b_baseline_metrics=stage_b_metrics,
        stage_c_vs_stage_b_ari=cross_ari,
        config={
            "k_grid_values": list(k_grid_values),
            "final_k": final_k,
            "base_algorithms": list(base_algorithms),
            "n_seeds_per_algorithm": n_seeds_per_algorithm,
            "n_base_clusterings": len(base_labels),
            "consensus_linkage": "average",
        },
    )
