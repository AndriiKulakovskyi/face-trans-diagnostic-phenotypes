"""Patient-only similarity graph builders for FACE V1 stratification.

All edges are computed with **pairwise-complete masked similarity** — no
imputation is ever performed in the default path. Two patients can only be
connected by a block-specific edge if they share at least
``min_shared_features`` observed features in that block (the semantic overlap
edge constraint). A separate ``transdiagnostic`` edge type is also available,
built from the data-driven transdiagnostic feature set.
"""

from face_stratification.graph.masked_similarity import (
    MaskedSimilarityResult,
    Metric,
    masked_cosine,
    masked_euclidean,
    masked_gower,
    masked_knn_edges,
    masked_manhattan,
    masked_similarity,
)
from face_stratification.graph.transdiagnostic import (
    TransdiagnosticFeatureSet,
    TransdiagnosticGraphResult,
    build_tiered_transdiagnostic_graphs,
    build_transdiagnostic_graph,
    compute_per_cohort_coverage,
    select_transdiagnostic_features,
)
from face_stratification.graph.patient_similarity import (
    BlockGraph,
    GraphSummary,
    build_balanced_knn_graph,
    build_block_knn_graph,
    build_multiplex_graph,
    build_mutual_knn_graph,
    summarize_graph,
)

__all__ = [
    # masked similarity kernels
    "MaskedSimilarityResult",
    "Metric",
    "masked_cosine",
    "masked_euclidean",
    "masked_gower",
    "masked_manhattan",
    "masked_knn_edges",
    "masked_similarity",
    # transdiagnostic selection + graph
    "TransdiagnosticFeatureSet",
    "TransdiagnosticGraphResult",
    "build_tiered_transdiagnostic_graphs",
    "build_transdiagnostic_graph",
    "compute_per_cohort_coverage",
    "select_transdiagnostic_features",
    # block + multiplex builders
    "BlockGraph",
    "GraphSummary",
    "build_balanced_knn_graph",
    "build_block_knn_graph",
    "build_multiplex_graph",
    "build_mutual_knn_graph",
    "summarize_graph",
]
