"""Per-block masked kNN graphs + multiplex aggregation (no imputation).

Every similarity is computed **pairwise-complete**: only the features observed
by *both* patients contribute. Edges are created only for pairs that share at
least the block's ``min_shared_features`` measurements (the *semantic overlap*
edge constraint). Together, these two guarantees make the graph honest about
what the data actually supports — a patient with no cognition assessments
receives no cognition-block edges, period.

A multiplex patient graph has one edge type per clinical block **plus** one
edge type called ``transdiagnostic`` built from the data-driven transdiagnostic
feature set (see :mod:`face_stratification.graph.transdiagnostic`).

Edge weights combine the similarity (Gaussian kernel on the masked distance)
with an overlap confidence factor — a pair that shares 100 % of a block's
features gets full weight, a pair that only just barely crosses the overlap
threshold gets proportionally less.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

import networkx as nx

from face_stratification.graph.masked_similarity import (
    Metric,
    masked_knn_edges,
    masked_similarity,
)
from face_stratification.graph.transdiagnostic import (
    TransdiagnosticFeatureSet,
    TransdiagnosticGraphResult,
    build_transdiagnostic_graph,
    select_transdiagnostic_features,
)
from face_stratification.harmonization.feature_schema import (
    FeatureBlock,
    FeatureSchema,
    FeatureType,
)

logger = logging.getLogger(__name__)


# ─── Public types ─────────────────────────────────────────────────────────────


@dataclass
class BlockGraph:
    """Edge list for a single clinical block's masked kNN graph."""

    block_id: str
    label_fr: str
    metric: str
    n_candidate_nodes: int  # patients passing min_fraction_present
    n_edges_raw: int
    edges: list[tuple[int, int, float]]  # (src_global, dst_global, weight)
    bandwidth: float
    min_shared_features: int
    feature_ids: tuple[str, ...]
    candidate_indices: np.ndarray  # global indices of participating patients


@dataclass
class GraphSummary:
    """Summary of a multiplex patient graph."""

    n_nodes: int
    n_edge_types: int
    edges_per_type: dict[str, int] = field(default_factory=dict)
    mean_degree_per_type: dict[str, float] = field(default_factory=dict)
    cohort_assortativity: dict[str, float] = field(default_factory=dict)
    candidate_nodes_per_type: dict[str, int] = field(default_factory=dict)
    min_shared_features_per_type: dict[str, int] = field(default_factory=dict)
    blocks_with_zero_edges: list[str] = field(default_factory=list)
    transdiagnostic_feature_set: TransdiagnosticFeatureSet | None = None


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _default_min_shared(block: FeatureBlock, n_features: int) -> int:
    """Compute the default overlap threshold for a block."""
    if block.min_shared_features is not None:
        return min(block.min_shared_features, n_features)
    return max(1, math.ceil(n_features / 2))


def _empirical_feature_ranges(sub: pd.DataFrame) -> np.ndarray:
    """Per-column range computed on observed values only (NaN-aware)."""
    arr = sub.to_numpy(dtype=np.float32)
    with np.errstate(invalid="ignore"):
        mx = np.nanmax(arr, axis=0)
        mn = np.nanmin(arr, axis=0)
    rng = mx - mn
    rng = np.where(np.isfinite(rng) & (rng > 0), rng, 1.0)
    return rng.astype(np.float32)


# ─── Block graph builder ──────────────────────────────────────────────────────


def build_block_knn_graph(
    block_df: pd.DataFrame,
    block: FeatureBlock,
    *,
    k: int = 10,
    min_fraction_present: float | None = None,
    min_shared_features: int | None = None,
) -> BlockGraph:
    """Build a masked kNN graph for a single clinical block.

    Two-stage filtering:

    1. **Candidate filter** — patients with less than ``min_fraction_present``
       of the block's features observed are dropped from this block's graph.
       This corresponds to "the patient was not substantively assessed on this
       block", and corresponds directly to the block's
       ``min_fraction_present`` config value.

    2. **Semantic overlap constraint** — among the remaining candidates, an
       edge ``(i, j)`` is only created if the pair's *shared observed* feature
       count is at least ``min_shared_features``.

    No values are ever imputed. The returned :class:`BlockGraph` uses **global
    indices** (positional in the original input frame), so the multiplex
    builder can stitch multiple blocks together without index remapping.
    """
    n_features = block_df.shape[1]
    n_rows = block_df.shape[0]

    fraction_threshold = (
        min_fraction_present
        if min_fraction_present is not None
        else block.min_fraction_present
    )
    shared_threshold = (
        min_shared_features
        if min_shared_features is not None
        else _default_min_shared(block, n_features)
    )

    if n_rows < 2 or n_features == 0:
        return BlockGraph(
            block_id=block.id,
            label_fr=block.label_fr,
            metric=block.metric,
            n_candidate_nodes=0,
            n_edges_raw=0,
            edges=[],
            bandwidth=1.0,
            min_shared_features=shared_threshold,
            feature_ids=tuple(block_df.columns),
            candidate_indices=np.array([], dtype=np.int64),
        )

    # Candidate filter
    fraction_present = block_df.notna().mean(axis=1).to_numpy()
    keep_mask = fraction_present >= fraction_threshold
    candidate_idx = np.where(keep_mask)[0]

    if candidate_idx.size < 2:
        logger.warning(
            "Block %s: only %d patient(s) pass min_fraction_present=%.2f; no edges built.",
            block.id,
            candidate_idx.size,
            fraction_threshold,
        )
        return BlockGraph(
            block_id=block.id,
            label_fr=block.label_fr,
            metric=block.metric,
            n_candidate_nodes=int(candidate_idx.size),
            n_edges_raw=0,
            edges=[],
            bandwidth=1.0,
            min_shared_features=shared_threshold,
            feature_ids=tuple(block_df.columns),
            candidate_indices=candidate_idx,
        )

    sub = block_df.iloc[candidate_idx]
    feature_ranges = (
        _empirical_feature_ranges(sub) if block.metric == "gower" else None
    )

    raw_edges = masked_knn_edges(
        sub.to_numpy(dtype=np.float32),
        metric=block.metric,  # type: ignore[arg-type]
        k=k,
        min_shared_features=shared_threshold,
        feature_ranges=feature_ranges,
    )

    # Bandwidth for the Gaussian weighting kernel.
    if raw_edges:
        dists = np.array([e[4] for e in raw_edges], dtype=np.float32)
        finite = dists[np.isfinite(dists)]
        bandwidth = float(np.median(finite)) if finite.size else 1.0
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            bandwidth = 1.0
    else:
        bandwidth = 1.0

    # Remap local candidate indices back to global row indices.
    weighted_edges: list[tuple[int, int, float]] = []
    for src_local, dst_local, _sim, overlap_count, dist in raw_edges:
        src_global = int(candidate_idx[src_local])
        dst_global = int(candidate_idx[dst_local])
        gauss = float(np.exp(-(dist * dist) / (2.0 * bandwidth * bandwidth)))
        confidence = overlap_count / max(1, n_features)
        weight = gauss * confidence
        weighted_edges.append((src_global, dst_global, weight))

    return BlockGraph(
        block_id=block.id,
        label_fr=block.label_fr,
        metric=block.metric,
        n_candidate_nodes=int(candidate_idx.size),
        n_edges_raw=len(raw_edges),
        edges=weighted_edges,
        bandwidth=bandwidth,
        min_shared_features=shared_threshold,
        feature_ids=tuple(block_df.columns),
        candidate_indices=candidate_idx,
    )


# ─── Balanced kNN ─────────────────────────────────────────────────────────────


def build_balanced_knn_graph(
    X: np.ndarray,
    patient_ids: np.ndarray,
    cohorts: np.ndarray,
    *,
    k: int = 10,
    metric: str = "cosine",
    min_shared_features: int = 1,
    balance_mode: str = "equal",
    mutual: bool = False,
) -> nx.Graph:
    """Build a kNN graph with cohort-balanced neighbor selection.

    Parameters
    ----------
    X : (n, d) feature matrix (may contain NaN for masked similarity)
    patient_ids : (n,) patient identifier array
    cohorts : (n,) cohort label array (e.g., 'bp', 'sz', 'dr', 'asp')
    k : number of neighbors per patient
    metric : distance metric
    min_shared_features : minimum shared non-NaN features for an edge
    balance_mode :
        "equal" - allocate k/n_cohorts neighbors per cohort (for transdiagnostic)
        "proportional" - allocate neighbors proportional to cohort size in candidates
    mutual : if True, only keep edges where both nodes selected each other

    Returns
    -------
    nx.Graph with weighted edges (attributes: weight, cohort_src, cohort_dst)
    """
    if balance_mode not in ("equal", "proportional"):
        raise ValueError(
            f"balance_mode must be 'equal' or 'proportional', got {balance_mode!r}"
        )

    X = np.asarray(X, dtype=np.float32)
    patient_ids = np.asarray(patient_ids)
    cohorts = np.asarray(cohorts)
    n = X.shape[0]

    if n < 2:
        G = nx.Graph()
        return G

    unique_cohorts = np.unique(cohorts)
    cohort_to_indices: dict[str, np.ndarray] = {
        c: np.where(cohorts == c)[0] for c in unique_cohorts
    }

    feature_ranges: np.ndarray | None = None
    if metric == "gower":
        feature_ranges = _empirical_feature_ranges(
            pd.DataFrame(X)
        )

    # Full pairwise similarity in batches.
    batch_size = 512
    sim_full = np.full((n, n), -np.inf, dtype=np.float32)
    ovl_full = np.zeros((n, n), dtype=np.int32)
    dist_full = np.full((n, n), np.inf, dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        q_idx = np.arange(start, end)
        res = masked_similarity(
            X, metric, query_indices=q_idx, feature_ranges=feature_ranges  # type: ignore[arg-type]
        )
        sim_full[start:end] = res.similarity
        ovl_full[start:end] = res.overlap
        dist_full[start:end] = res.distance

    # Zero out self-loops and pairs failing overlap constraint.
    np.fill_diagonal(sim_full, -np.inf)
    np.fill_diagonal(dist_full, np.inf)
    sim_full[ovl_full < min_shared_features] = -np.inf
    dist_full[ovl_full < min_shared_features] = np.inf

    # Per-patient balanced neighbor selection.
    # neighbor_sets[i] stores the set of global indices selected as i's neighbors.
    neighbor_sets: list[set[int]] = [set() for _ in range(n)]

    for i in range(n):
        src_cohort = cohorts[i]
        sims_i = sim_full[i]

        # Partition candidate indices (finite similarity) by cohort.
        finite_mask = np.isfinite(sims_i)
        candidates_by_cohort: dict[str, np.ndarray] = {}
        for c, c_indices in cohort_to_indices.items():
            valid = c_indices[finite_mask[c_indices]]
            # Exclude self.
            valid = valid[valid != i]
            if valid.size > 0:
                candidates_by_cohort[c] = valid

        n_available_cohorts = len(candidates_by_cohort)
        if n_available_cohorts == 0:
            continue

        # Compute per-cohort quotas.
        quotas: dict[str, int] = {}
        if balance_mode == "equal":
            per_cohort = math.ceil(k / n_available_cohorts)
            for c in candidates_by_cohort:
                quotas[c] = per_cohort
        else:
            total_candidates = sum(v.size for v in candidates_by_cohort.values())
            if total_candidates == 0:
                continue
            for c, c_cands in candidates_by_cohort.items():
                raw = k * (c_cands.size / total_candidates)
                quotas[c] = max(1, math.ceil(raw))

        # Pick closest within each cohort up to quota.
        selected: list[int] = []
        for c, c_cands in candidates_by_cohort.items():
            c_sims = sims_i[c_cands]
            quota = quotas[c]
            n_pick = min(quota, c_cands.size)
            top_local = np.argpartition(-c_sims, kth=min(n_pick, c_cands.size) - 1)[:n_pick]
            # Sort those by descending similarity for deterministic truncation.
            top_local = top_local[np.argsort(-c_sims[top_local])]
            selected.extend(c_cands[top_local].tolist())

        # Truncate to k total, keeping highest similarities.
        if len(selected) > k:
            selected_arr = np.array(selected, dtype=np.int64)
            selected_sims = sims_i[selected_arr]
            keep = np.argsort(-selected_sims)[:k]
            selected = selected_arr[keep].tolist()

        neighbor_sets[i] = set(selected)

    # Compute bandwidth from all selected distances.
    all_dists: list[float] = []
    for i, nbrs in enumerate(neighbor_sets):
        for j in nbrs:
            d = float(dist_full[i, j])
            if np.isfinite(d):
                all_dists.append(d)
    bandwidth = float(np.median(all_dists)) if all_dists else 1.0
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = 1.0

    # Build the graph.
    n_features = X.shape[1]
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, patient_id=patient_ids[i], cohort=cohorts[i])

    for i, nbrs in enumerate(neighbor_sets):
        for j in nbrs:
            if mutual and i not in neighbor_sets[j]:
                continue
            if G.has_edge(i, j):
                continue
            d = float(dist_full[i, j])
            ov = int(ovl_full[i, j])
            gauss = float(np.exp(-(d * d) / (2.0 * bandwidth * bandwidth)))
            confidence = ov / max(1, n_features)
            weight = gauss * confidence
            G.add_edge(
                i,
                j,
                weight=weight,
                cohort_src=str(cohorts[i]),
                cohort_dst=str(cohorts[j]),
            )

    logger.info(
        "Balanced kNN (mode=%s, k=%d, mutual=%s): %d nodes, %d edges, bandwidth=%.4f",
        balance_mode,
        k,
        mutual,
        G.number_of_nodes(),
        G.number_of_edges(),
        bandwidth,
    )
    return G


# ─── Mutual kNN filter ───────────────────────────────────────────────────────


def build_mutual_knn_graph(G: nx.Graph) -> nx.Graph:
    """Filter an existing kNN graph to keep only mutual edges.

    An edge (i, j) is "mutual" if j was selected as a neighbor of i AND
    i was selected as a neighbor of j in the original directed kNN selection.
    Since the input graph is undirected (each edge was added when at least one
    direction existed), this function re-checks mutual support via a degree
    heuristic: it keeps only edges where both endpoints have degree >= 1 in the
    subgraph formed by the other endpoint's neighbors.

    For a proper mutual filter from scratch, prefer the ``mutual=True`` flag in
    :func:`build_balanced_knn_graph`.

    For pre-built graphs where the original directed neighbor sets are lost,
    this applies a **symmetric-kNN** approximation: keep edge (i, j) only if
    both nodes appear in each other's adjacency lists in ``G`` (always true for
    an undirected graph — so this is a no-op on simple Graphs). For MultiGraphs
    where an edge type may connect i→j but not j→i, it filters per edge type.
    """
    if isinstance(G, nx.MultiGraph):
        edges_by_key: dict[str, list[tuple[int, int, dict]]] = {}
        for u, v, key, data in G.edges(keys=True, data=True):
            edges_by_key.setdefault(key, []).append((u, v, data))

        H: nx.MultiGraph = nx.MultiGraph()
        H.add_nodes_from(G.nodes(data=True))
        for key, edge_list in edges_by_key.items():
            adj: dict[int, set[int]] = {}
            for u, v, _ in edge_list:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            directed: dict[int, set[int]] = {}
            for u, v, _ in edge_list:
                directed.setdefault(u, set()).add(v)
            for u, v, data in edge_list:
                if v in directed.get(u, set()) and u in directed.get(v, set()):
                    if not H.has_edge(u, v, key=key):
                        H.add_edge(u, v, key=key, **data)
        return H  # type: ignore[return-value]

    H_simple: nx.Graph = nx.Graph()
    H_simple.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        if G.has_edge(u, v) and G.has_edge(v, u):
            H_simple.add_edge(u, v, **data)
    return H_simple


# ─── Multiplex builder ────────────────────────────────────────────────────────


def build_multiplex_graph(
    X: pd.DataFrame,
    schema: FeatureSchema,
    *,
    k: int = 10,
    skip_blocks: Iterable[str] = (),
    include_transdiagnostic: bool = True,
    metadata: pd.DataFrame | None = None,
    balanced: bool = False,
    cohorts: np.ndarray | None = None,
    balance_mode: str = "equal",
    normalize_block_weights: bool = True,
):
    """Build a NetworkX ``MultiGraph`` aggregating every block's masked kNN.

    Parameters
    ----------
    X:
        Unified harmonized matrix (with or without normalization). **Must not
        be imputed** — the masked similarity kernels rely on NaNs marking
        unobserved values.
    schema:
        The feature schema driving block config and transdiagnostic selection.
    k:
        Target neighbours per node within each block.
    skip_blocks:
        Block ids to exclude from the graph.
    include_transdiagnostic:
        If True, adds a ``transdiagnostic`` edge type built from the
        data-driven transdiagnostic feature set. Requires ``metadata`` so the
        per-cohort coverage can be measured.
    metadata:
        Side metadata DataFrame indexed identically to ``X``. Only required
        when ``include_transdiagnostic=True``.
    balanced:
        If True and ``cohorts`` is provided, uses
        :func:`build_balanced_knn_graph` for each block instead of the
        standard :func:`build_block_knn_graph`.
    cohorts:
        ``(n,)`` array of cohort labels aligned with ``X``. Required when
        ``balanced=True``.
    balance_mode:
        Passed to :func:`build_balanced_knn_graph` when ``balanced=True``.
    normalize_block_weights:
        If True (default), rescale each block's edge weights so that every
        block's total weight equals the median total weight across active
        blocks. This prevents low-dimensional blocks (e.g. comorbidities
        with 2 features) from dominating the spectral embedding simply
        because Gower on few features produces many ties.
    """
    if balanced and cohorts is None:
        raise ValueError("balanced=True requires the `cohorts` array.")

    from face_stratification.harmonization.missingness import split_blocks

    G: "nx.MultiGraph" = nx.MultiGraph()
    for pos, (cohort, pid) in enumerate(X.index):
        G.add_node(pos, cohort=cohort, patient_id=pid)

    if cohorts is None:
        cohorts_arr = np.array([c for c, _ in X.index])
    else:
        cohorts_arr = np.asarray(cohorts)

    patient_ids_arr = np.array([pid for _, pid in X.index])

    skip = set(skip_blocks)
    per_block = split_blocks(X, schema)
    built: dict[str, BlockGraph] = {}

    # Collect per-block edge lists before adding to the graph, so we can
    # optionally normalize weights across blocks.
    block_edge_lists: dict[str, list[tuple[int, int, float]]] = {}

    for block_id, block_df in per_block.items():
        if block_id in skip:
            continue
        try:
            block = schema.block(block_id)
        except KeyError:
            logger.warning("Unknown block id %s; skipping", block_id)
            continue

        if balanced:
            block_arr = block_df.to_numpy(dtype=np.float32)
            n_features = block_arr.shape[1]
            shared_threshold = (
                block.min_shared_features
                if block.min_shared_features is not None
                else _default_min_shared(block, n_features)
            )
            balanced_g = build_balanced_knn_graph(
                block_arr,
                patient_ids_arr,
                cohorts_arr,
                k=k,
                metric=block.metric,
                min_shared_features=shared_threshold,
                balance_mode=balance_mode,
            )
            edges = [(u, v, data["weight"]) for u, v, data in balanced_g.edges(data=True)]
            block_edge_lists[block_id] = edges
        else:
            bg = build_block_knn_graph(block_df, block, k=k)
            built[block_id] = bg
            block_edge_lists[block_id] = list(bg.edges)

    # Transdiagnostic edge list (computed before normalization).
    td_result: TransdiagnosticGraphResult | None = None
    td_edges: list[tuple[int, int, float]] = []
    if include_transdiagnostic:
        if metadata is None:
            raise ValueError(
                "include_transdiagnostic=True requires `metadata` so that "
                "per-cohort coverage can be measured."
            )
        td_result = build_transdiagnostic_graph(X, metadata, schema, k=k)
        for src, dst, sim, overlap_count, dist in td_result.edges:
            gauss = float(
                np.exp(-(dist * dist) / (2.0 * td_result.bandwidth * td_result.bandwidth))
                if td_result.bandwidth > 0
                else 1.0
            )
            confidence = overlap_count / max(1, td_result.feature_set.n_selected)
            weight = gauss * confidence
            td_edges.append((int(src), int(dst), weight))
        block_edge_lists["transdiagnostic"] = td_edges

    # ── Per-block weight normalization ──────────────────────────────────────
    if normalize_block_weights:
        total_weights: dict[str, float] = {}
        for bid, edges in block_edge_lists.items():
            tw = sum(w for _, _, w in edges)
            if tw > 0:
                total_weights[bid] = tw

        if total_weights:
            median_tw = float(np.median(list(total_weights.values())))
            for bid in list(block_edge_lists.keys()):
                tw = total_weights.get(bid, 0.0)
                if tw > 0:
                    scale = median_tw / tw
                    block_edge_lists[bid] = [
                        (s, d, w * scale) for s, d, w in block_edge_lists[bid]
                    ]
            logger.info(
                "Block weight normalization: median total weight=%.2f, "
                "%d blocks rescaled (range %.4f–%.4f → all ≈%.2f)",
                median_tw,
                len(total_weights),
                min(total_weights.values()),
                max(total_weights.values()),
                median_tw,
            )

    # ── Add all edges to the MultiGraph ─────────────────────────────────────
    for bid, edges in block_edge_lists.items():
        for src, dst, w in edges:
            G.add_edge(src, dst, key=bid, block=bid, weight=w)

    return G, built, td_result


# ─── Summary ──────────────────────────────────────────────────────────────────


def summarize_graph(
    G,
    schema: FeatureSchema,
    built: dict[str, BlockGraph] | None = None,
    transdiagnostic: TransdiagnosticGraphResult | None = None,
) -> GraphSummary:
    """Compute summary statistics for a multiplex patient graph."""

    summary = GraphSummary(n_nodes=G.number_of_nodes(), n_edge_types=0)

    edges_by_type: dict[str, list[tuple[int, int]]] = {}
    for u, v, data in G.edges(data=True):
        bid = data.get("block", "unknown")
        edges_by_type.setdefault(bid, []).append((u, v))

    summary.n_edge_types = len(edges_by_type)
    summary.edges_per_type = {k: len(v) for k, v in edges_by_type.items()}

    # Zero-edge blocks — anything in the schema that produced nothing
    for block_id, feats in schema.features_by_block().items():
        if feats and block_id not in edges_by_type:
            summary.blocks_with_zero_edges.append(block_id)

    # Mean degree per edge type
    for bid, edge_list in edges_by_type.items():
        sub = nx.MultiGraph()
        sub.add_nodes_from(G.nodes())
        sub.add_edges_from(edge_list)
        degrees = [d for _, d in sub.degree()]
        summary.mean_degree_per_type[bid] = float(np.mean(degrees)) if degrees else 0.0

    # Cohort assortativity per edge type. NetworkX returns NaN when the
    # induced sub-graph has zero variance on the attribute (e.g. all edges
    # live inside a single cohort). Treat that as +1.0 since it means the
    # block's edges are perfectly cohort-assortative — they exist only within
    # one cohort — which is the clinically correct interpretation for
    # single-cohort blocks like psychosis (SZ-only) or cohort_specific.
    for bid, edge_list in edges_by_type.items():
        sub = nx.Graph()
        sub.add_nodes_from(G.nodes(data=True))
        sub.add_edges_from(edge_list)
        try:
            val = float(nx.attribute_assortativity_coefficient(sub, "cohort"))
        except Exception:  # noqa: BLE001
            val = float("nan")
        if not math.isfinite(val):
            # Check whether the sub-graph is mono-cohort (→ +1) or empty (→ 0)
            cohorts_in_sub = {sub.nodes[n].get("cohort") for _, n in enumerate(sub) if sub.degree(n) > 0}
            val = 1.0 if len(cohorts_in_sub) == 1 else 0.0
        summary.cohort_assortativity[bid] = val

    # Candidate node counts per block (for audit)
    if built is not None:
        for bid, bg in built.items():
            summary.candidate_nodes_per_type[bid] = bg.n_candidate_nodes
            summary.min_shared_features_per_type[bid] = bg.min_shared_features
    if transdiagnostic is not None:
        summary.candidate_nodes_per_type["transdiagnostic"] = transdiagnostic.n_nodes
        summary.min_shared_features_per_type["transdiagnostic"] = (
            transdiagnostic.min_shared_features
        )
        summary.transdiagnostic_feature_set = transdiagnostic.feature_set

    return summary
