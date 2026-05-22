"""Multipartite graph construction using features with partial cohort coverage.

The 4-way "transdiagnostic" subset (~9 features) is too thin to build a
meaningful cross-cohort phenotype. But features with coverage in 2-3 cohorts
(~46 additional features) carry substantial clinical signal:

- **BP+DR mood partition** (17 features: MADRS, QIDS, STAI, Mathys) —
  unified depression/anxiety severity axis across mood disorders.
- **BP+SZ cognitive partition** (5 features: CVLT, fluency, WAIS) —
  shared neurocognition dimension across mood and psychotic disorders.
- **DR+SZ metabolic partition** (4 features: lipids, glucose) — shared
  antipsychotic/antidepressant-induced metabolic syndrome.
- **BP+DR+SZ affective-severity partition** (20 features: YMRS, CGI-S,
  suicide items) — broader severity spectrum.

Each partition contributes an embedding that is meaningful only for its
participating cohorts. Patients outside the cohort subset receive zero
vectors in that partition's columns, with a binary mask column indicating
participation. Downstream clustering operates on the concatenated,
per-patient L2-normalized vector.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from face_stratification.harmonization.harmonizer import HarmonizedDataset
from face_stratification.models.base import BaseEmbeddingModel, PatientEmbedding

logger = logging.getLogger(__name__)


# ─── Partition identification ────────────────────────────────────────────────


@dataclass
class CoveragePartition:
    """A feature partition defined by its cohort coverage."""

    cohorts: frozenset[str]      # e.g. {'bp', 'dr'}
    features: list[str]          # features with ≥threshold coverage in ALL cohorts of subset
    name: str                    # auto-generated: 'bp+dr' or 'bp+dr+sz'
    n_features: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_features = len(self.features)


def identify_coverage_partitions(
    dataset: HarmonizedDataset,
    *,
    min_coverage_in_subset: float = 0.30,
    min_features_per_partition: int = 3,
    max_cohort_subset_size: int | None = None,
    exclude_single_cohort: bool = True,
    feature_mode: str = "cumulative",
) -> list[CoveragePartition]:
    """Find feature partitions based on cross-cohort coverage patterns.

    Two modes for assigning features to partitions:

    - ``"cumulative"`` (default, recommended): For each candidate cohort
      subset S, include every feature whose coverage-qualified cohorts
      are a **superset** of S. This is the scientifically correct mode for
      bipartite/tripartite graphs — a BP+DR graph uses every feature
      available in both BP and DR, regardless of whether that feature is
      also available in SZ or ASP. Produces overlapping partitions.

    - ``"pattern_exact"``: Group features by their exact coverage pattern.
      Each feature is assigned to exactly one partition. Produces disjoint
      partitions useful for reporting unique contributions per coverage
      pattern, but under-counts features available for each graph.

    Example — feature counts at 30% coverage in FACE data (4 cohorts):

    | Partition | pattern_exact | cumulative |
    |-----------|--------------:|-----------:|
    | 4-way     | 9             | 9          |
    | bp+dr+sz  | 20            | 29         |
    | bp+dr     | 17            | 46         |
    | bp+sz     | 5             | 37         |
    | dr+sz     | 4             | 33         |

    Parameters
    ----------
    dataset:
        The harmonized dataset.
    min_coverage_in_subset:
        Minimum per-cohort coverage for a cohort to count as covered.
    min_features_per_partition:
        Partitions with fewer features are dropped.
    max_cohort_subset_size:
        If set, drop partitions with more cohorts than this.
    exclude_single_cohort:
        Drop single-cohort partitions (they are intra-cohort blocks).
    feature_mode:
        ``"cumulative"`` or ``"pattern_exact"``. See above.
    """
    from itertools import combinations

    if feature_mode not in {"cumulative", "pattern_exact"}:
        raise ValueError(f"feature_mode must be 'cumulative' or 'pattern_exact', got {feature_mode!r}")

    cov = dataset.feature_availability()
    cov_cols = [c for c in cov.columns if c.startswith("coverage_") and c != "coverage_total"]
    all_cohorts = sorted(c.replace("coverage_", "") for c in cov_cols)

    # Per feature: which cohorts have ≥threshold coverage
    feat_to_cohorts: dict[str, frozenset[str]] = {}
    for feat_id in cov.index:
        covered = frozenset(
            c.replace("coverage_", "")
            for c in cov_cols
            if cov.loc[feat_id, c] >= min_coverage_in_subset
        )
        if covered:
            feat_to_cohorts[feat_id] = covered

    # Group features by coverage pattern
    pattern_to_features: dict[frozenset[str], list[str]] = defaultdict(list)
    for feat, coh_set in feat_to_cohorts.items():
        pattern_to_features[coh_set].append(feat)

    partitions: list[CoveragePartition] = []

    if feature_mode == "pattern_exact":
        # Each feature assigned to exactly one partition (its coverage pattern)
        candidate_subsets = list(pattern_to_features.keys())
    else:
        # Cumulative: enumerate all possible non-empty cohort subsets
        candidate_subsets = []
        for size in range(1, len(all_cohorts) + 1):
            for combo in combinations(all_cohorts, size):
                candidate_subsets.append(frozenset(combo))

    seen_subsets: set[frozenset[str]] = set()
    for subset in candidate_subsets:
        if subset in seen_subsets:
            continue
        seen_subsets.add(subset)

        if exclude_single_cohort and len(subset) < 2:
            continue
        if max_cohort_subset_size is not None and len(subset) > max_cohort_subset_size:
            continue

        # Collect features for this subset based on mode
        if feature_mode == "pattern_exact":
            features = pattern_to_features.get(subset, [])
        else:
            # Cumulative: features whose coverage pattern is a superset of subset
            features = sorted({
                f
                for pattern, feats in pattern_to_features.items()
                if pattern.issuperset(subset)
                for f in feats
            })

        if len(features) < min_features_per_partition:
            continue

        name = "+".join(sorted(subset))
        partitions.append(CoveragePartition(
            cohorts=subset,
            features=features,
            name=name,
        ))

    # Sort by subset size descending, then by feature count descending
    partitions.sort(key=lambda p: (-len(p.cohorts), -p.n_features))

    logger.info(
        "Identified %d partitions (mode=%s, coverage>=%.0f%%, min_features=%d)",
        len(partitions),
        feature_mode,
        100 * min_coverage_in_subset,
        min_features_per_partition,
    )
    for p in partitions:
        logger.info("  %s (%d cohorts, %d features)",
                    p.name.upper(), len(p.cohorts), p.n_features)

    return partitions


# ─── Partition graph + spectral embedding ────────────────────────────────────


def _build_partition_knn_graph(
    X_sub: pd.DataFrame,
    *,
    k: int = 10,
    metric: str = "cosine",
    min_shared: int = 2,
) -> sp.csr_matrix:
    """Build a symmetric kNN graph for patients in a partition.

    Uses pairwise-complete masked similarity on the partition's feature set.
    """
    from face_stratification.graph.masked_similarity import masked_cosine

    n = X_sub.shape[0]
    if n <= k + 1:
        logger.warning("Partition has only %d patients, cannot build kNN(k=%d)", n, k)
        return sp.csr_matrix((n, n), dtype=np.float64)

    values = X_sub.to_numpy(dtype=np.float64)

    # For each patient, find top-k most similar OTHER patients
    if metric == "cosine":
        sim_result = masked_cosine(values)
        sim = sim_result.similarity.copy()
        overlap = sim_result.overlap
    else:
        raise NotImplementedError(f"metric={metric} not supported yet")

    # Mask pairs with too little overlap
    sim[overlap < min_shared] = -np.inf
    # Self-similarity to -inf
    np.fill_diagonal(sim, -np.inf)
    sim = np.nan_to_num(sim, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

    # Build kNN edges
    rows, cols, weights = [], [], []
    for i in range(n):
        finite_count = int(np.isfinite(sim[i]).sum())
        if finite_count == 0:
            continue
        k_eff = min(k, finite_count)
        top_k_idx = np.argpartition(-sim[i], kth=k_eff - 1)[:k_eff]
        for j in top_k_idx:
            s = sim[i, j]
            if np.isfinite(s) and s > 0:
                rows.append(i)
                cols.append(j)
                weights.append(float(s))

    if not rows:
        return sp.csr_matrix((n, n), dtype=np.float64)

    # Symmetrize via max
    all_rows = np.array(rows + cols, dtype=np.int64)
    all_cols = np.array(cols + rows, dtype=np.int64)
    all_weights = np.array(weights + weights, dtype=np.float64)
    A = sp.coo_matrix((all_weights, (all_rows, all_cols)), shape=(n, n))
    A = A.maximum(A.T)
    return A.tocsr()


def _spectral_embed_sparse(
    A: sp.csr_matrix,
    n_components: int,
) -> np.ndarray:
    """Laplacian eigenmap embedding of a sparse adjacency."""
    from scipy.sparse.linalg import eigsh

    n = A.shape[0]
    if n < n_components + 2:
        return np.zeros((n, n_components))

    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg = np.where(deg > 0, deg, 1.0)
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg))
    L_sym = sp.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

    try:
        eigenvalues, eigenvectors = eigsh(
            L_sym.tocsc(), k=n_components + 1, which="SM", maxiter=5000,
        )
    except Exception as exc:
        logger.warning("Sparse eigsh failed (%s); falling back to dense eigh", exc)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L_sym.toarray())
        except Exception:
            return np.zeros((n, n_components))

    # Sort ascending, skip trivial first eigenvector (eigenvalue ≈ 0)
    order = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, order]
    Z = eigenvectors[:, 1:n_components + 1]  # skip first

    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


# ─── Multipartite embedding model ────────────────────────────────────────────


class MultipartiteSpectralEmbedding(BaseEmbeddingModel):
    """Per-partition spectral embeddings with masked concatenation.

    For each coverage partition (e.g., BP+DR mood, BP+SZ cognition, etc.):
        1. Restrict to patients in the partition's cohort subset
        2. Build masked kNN graph on the partition's feature set
        3. Spectral-embed to ``n_components_per_partition``
        4. Pad back to full N with zeros for non-participating patients
        5. Add a binary mask column indicating participation

    The final embedding is a per-patient L2-normalized concatenation.
    This produces a patient vector where each block of dimensions
    corresponds to a different cross-cohort bridge.

    Parameters
    ----------
    min_coverage:
        Minimum per-cohort coverage for a feature to be included in a
        partition.
    min_features_per_partition:
        Partitions with fewer features are dropped.
    n_components_per_partition:
        Spectral components per partition. Total embedding dim will be
        roughly ``n_partitions * n_components_per_partition + n_partitions``
        (the extra is the mask columns).
    k_neighbours:
        k for the kNN graph within each partition.
    include_4way:
        Whether to include the full 4-cohort partition.
    include_mask_columns:
        Whether to append binary mask columns indicating participation.
    feature_mode:
        ``"cumulative"`` (default, recommended) includes every feature
        available in all cohorts of the subset — so a BP+DR graph uses
        features covered in BP+DR, BP+DR+SZ, BP+DR+ASP, and all-4.
        ``"pattern_exact"`` restricts each partition to features present
        in exactly that cohort subset (disjoint partitions).
    partition_weighting:
        How to weight each partition's contribution before concatenation:

        - ``"sqrt_info"`` (default): weight by ``sqrt(n_features * n_patients)``
          normalized by the max. Upweights partitions that are both
          feature-rich and patient-rich (e.g., BP+DR with 46 features and
          ~5400 patients). Pushes clustering toward clinically strong bridges.
        - ``"sqrt_features"``: weight by sqrt of feature count only.
        - ``"none"``: equal weighting (all partitions contribute equally).
        - ``"mask_only"``: each partition weighted by the fraction of patients
          it actually contains.
    """

    name = "multipartite_spectral"

    def __init__(
        self,
        *,
        min_coverage: float = 0.30,
        min_features_per_partition: int = 3,
        n_components_per_partition: int = 8,
        k_neighbours: int = 10,
        include_4way: bool = True,
        include_mask_columns: bool = True,
        l2_normalize: bool = True,
        feature_mode: str = "cumulative",
        partition_weighting: str = "sqrt_info",
    ) -> None:
        super().__init__(
            min_coverage=min_coverage,
            min_features_per_partition=min_features_per_partition,
            n_components_per_partition=n_components_per_partition,
            k_neighbours=k_neighbours,
            include_4way=include_4way,
            include_mask_columns=include_mask_columns,
            l2_normalize=l2_normalize,
            feature_mode=feature_mode,
            partition_weighting=partition_weighting,
        )
        self.min_coverage = min_coverage
        self.min_features_per_partition = min_features_per_partition
        self.n_components_per_partition = n_components_per_partition
        self.k_neighbours = k_neighbours
        self.include_4way = include_4way
        self.include_mask_columns = include_mask_columns
        self.l2_normalize = l2_normalize
        self.feature_mode = feature_mode
        self.partition_weighting = partition_weighting

        self._embedding: PatientEmbedding | None = None
        self._partitions: list[CoveragePartition] = []
        self._view_dims: dict[str, int] = {}
        self._partition_weights: dict[str, float] = {}

    def fit(
        self,
        dataset: HarmonizedDataset,
        *,
        graph: Any | None = None,
    ) -> MultipartiteSpectralEmbedding:
        # 1. Identify partitions
        exclude_4way = not self.include_4way
        partitions = identify_coverage_partitions(
            dataset,
            min_coverage_in_subset=self.min_coverage,
            min_features_per_partition=self.min_features_per_partition,
            feature_mode=self.feature_mode,
        )
        if exclude_4way:
            partitions = [p for p in partitions if len(p.cohorts) < 4]
        self._partitions = partitions

        if not partitions:
            raise RuntimeError("No partitions met the minimum feature threshold")

        cohort_labels = dataset.metadata["cohort"].values
        full_index = dataset.X.index
        n_full = len(full_index)

        # 2. Compute per-partition weights (raw values before normalization)
        raw_weights: dict[str, float] = {}
        for p in partitions:
            n_patients_p = int(np.isin(cohort_labels, list(p.cohorts)).sum())
            if self.partition_weighting == "sqrt_info":
                raw_weights[p.name] = float(np.sqrt(p.n_features * n_patients_p))
            elif self.partition_weighting == "sqrt_features":
                raw_weights[p.name] = float(np.sqrt(p.n_features))
            elif self.partition_weighting == "mask_only":
                raw_weights[p.name] = float(n_patients_p / n_full)
            elif self.partition_weighting == "none":
                raw_weights[p.name] = 1.0
            else:
                raise ValueError(f"Unknown partition_weighting={self.partition_weighting!r}")

        # Normalize so max weight = 1
        max_w = max(raw_weights.values()) if raw_weights else 1.0
        self._partition_weights = {k: v / max_w for k, v in raw_weights.items()}
        logger.info("Partition weights (normalized):")
        for name, w in sorted(self._partition_weights.items(), key=lambda kv: -kv[1]):
            logger.info("  %-25s weight=%.3f", name, w)

        # 3. Per-partition spectral embedding
        blocks = []
        column_names = []
        view_dims = {}

        for p in partitions:
            logger.info("Fitting partition %s (%d features, cohorts=%s)",
                        p.name, p.n_features, sorted(p.cohorts))

            # Restrict patients to partition's cohort subset
            patient_mask = np.isin(cohort_labels, list(p.cohorts))
            n_sub = patient_mask.sum()

            if n_sub < self.k_neighbours + 2:
                logger.warning("Skipping %s: only %d patients", p.name, n_sub)
                continue

            X_sub = dataset.X.loc[patient_mask, p.features]

            # Build partition kNN graph
            A_sub = _build_partition_knn_graph(
                X_sub, k=self.k_neighbours, metric="cosine",
            )

            if A_sub.nnz == 0:
                logger.warning("Skipping %s: empty graph", p.name)
                continue

            # Spectral embed
            Z_sub = _spectral_embed_sparse(A_sub, self.n_components_per_partition)

            # Per-partition L2 normalize (only on the participating patients)
            if self.l2_normalize:
                norms = np.linalg.norm(Z_sub, axis=1, keepdims=True)
                norms = np.where(norms > 1e-12, norms, 1.0)
                Z_sub = Z_sub / norms

            # Apply partition weight
            w = self._partition_weights[p.name]
            Z_sub = Z_sub * w

            # Pad to full N
            Z_full = np.zeros((n_full, self.n_components_per_partition))
            Z_full[patient_mask, :] = Z_sub

            blocks.append(Z_full)
            part_cols = [f"{p.name}::dim_{i}" for i in range(self.n_components_per_partition)]
            column_names.extend(part_cols)
            view_dims[p.name] = self.n_components_per_partition

            # Mask column (also weighted so it doesn't dominate)
            if self.include_mask_columns:
                mask_col = (patient_mask.astype(np.float64) * w).reshape(-1, 1)
                blocks.append(mask_col)
                column_names.append(f"{p.name}::mask")
                view_dims[f"{p.name}_mask"] = 1

        if not blocks:
            raise RuntimeError("No partitions produced valid embeddings")

        # 4. Concatenate + final L2 normalize per patient
        Z = np.hstack(blocks)

        if self.l2_normalize:
            norms = np.linalg.norm(Z, axis=1, keepdims=True)
            norms = np.where(norms > 1e-12, norms, 1.0)
            Z = Z / norms

        self._view_dims = view_dims
        self._embedding = PatientEmbedding(
            values=pd.DataFrame(Z, index=full_index, columns=column_names),
            model_name=self.name,
            model_config={
                **self.config,
                "partitions": [
                    {"name": p.name, "cohorts": sorted(p.cohorts), "n_features": p.n_features}
                    for p in partitions
                ],
                "total_dim": Z.shape[1],
            },
            view_dims=view_dims,
            schema_version=getattr(dataset.schema, "version", "unknown"),
        )
        self._fitted = True
        logger.info("MultipartiteSpectral: %d patients × %d dims from %d partitions",
                    n_full, Z.shape[1], len([p for p in partitions]))
        return self

    def transform(self) -> PatientEmbedding:
        self._ensure_fitted()
        assert self._embedding is not None
        return self._embedding

    def partitions_summary(self) -> pd.DataFrame:
        """Return a DataFrame describing the identified partitions + weights."""
        rows = []
        for p in self._partitions:
            rows.append({
                "name": p.name,
                "n_cohorts": len(p.cohorts),
                "cohorts": "+".join(sorted(p.cohorts)),
                "n_features": p.n_features,
                "weight": self._partition_weights.get(p.name, float("nan")),
                "features_sample": ", ".join(p.features[:3]) + ("..." if p.n_features > 3 else ""),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("weight", ascending=False).reset_index(drop=True)
        return df
