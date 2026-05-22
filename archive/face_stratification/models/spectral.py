"""Spectral embedding models for Stage B (graph-respecting, no imputation).

Two concrete spectral models:

- :class:`TransdiagnosticSpectral` — normalized Laplacian eigenvectors of
  the transdiagnostic edge type only. Uses exclusively the trusted
  transdiagnostic backbone built by Stage A.

- :class:`MultiplexSpectral` — normalized Laplacian eigenvectors of the
  **weighted union** of every edge type in the Stage A multiplex graph.
  Edge weights are summed (or optionally max-pooled) across the 17
  parallel edge types per node pair, yielding one aggregated weighted
  graph that is then spectrally embedded.

Mathematical background
-----------------------
For a weighted undirected graph with adjacency ``W ∈ ℝ^(N × N)`` and
degree ``D = diag(W · 1)``, the symmetric normalized Laplacian is

    L_sym  =  I  -  D^(-1/2) W D^(-1/2)

The bottom ``k`` non-trivial eigenvectors of ``L_sym`` provide a smooth
``k``-dimensional coordinate system that minimizes

    Σ_{i,j} w_{ij} · ‖y_i - y_j‖²   s.t.   Y^T D Y = I,  Y ⊥ 𝟙

i.e. "nodes connected by strong edges should be close". This is the
classic spectral embedding / Laplacian eigenmaps construction; see
Belkin & Niyogi (2003).

Crucially, spectral embedding operates **only** on the edge structure.
Stage A built the edges with pairwise-complete masked similarity and the
semantic overlap constraint — so the no-imputation guarantee propagates
transparently into the embedding without any extra machinery.

Implementation notes
--------------------
- We use ``scipy.sparse.linalg.eigsh`` on the symmetric normalized
  Laplacian via ``sigma=0`` shift-invert mode for the bottom-``k``
  eigenvectors. This is orders of magnitude faster than dense
  eigendecomposition on 11 k × 11 k Laplacians.
- Isolated nodes (degree 0) are handled by excluding them from the
  Laplacian decomposition and assigning them the zero vector. They are
  counted in ``n_isolated_nodes`` so the audit is transparent.
- The first eigenvector is always the trivial constant one and is
  dropped before returning ``k`` components.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp

from face_stratification.harmonization.harmonizer import HarmonizedDataset
from face_stratification.models.base import (
    BaseEmbeddingModel,
    PatientEmbedding,
)

logger = logging.getLogger(__name__)


# ─── Adjacency builders ──────────────────────────────────────────────────────


def _nx_multiplex_to_adjacency(
    G,
    *,
    n_nodes: int,
    edge_types: Iterable[str] | None = None,
    combine: str = "sum",
) -> sp.csr_matrix:
    """Collapse a NetworkX MultiGraph into one weighted sparse adjacency.

    Parameters
    ----------
    G:
        NetworkX MultiGraph where each edge has a ``block`` attribute and a
        numeric ``weight``.
    n_nodes:
        Full node count (the harmonized matrix row count). Any patient
        without edges still gets a row in the adjacency.
    edge_types:
        If given, restrict aggregation to these edge types (``block``
        attribute values). ``None`` uses every edge.
    combine:
        - ``"sum"`` (default): sum weights across parallel edges.
        - ``"max"``: take the maximum weight across parallel edges.
    """
    if combine not in {"sum", "max"}:
        raise ValueError(f"Unknown combine={combine!r}")
    selected = set(edge_types) if edge_types is not None else None

    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []

    for u, v, data in G.edges(data=True):
        bid = data.get("block")
        if selected is not None and bid not in selected:
            continue
        w = float(data.get("weight", 1.0))
        if w <= 0:
            continue
        rows.append(u)
        cols.append(v)
        weights.append(w)

    if not rows:
        return sp.csr_matrix((n_nodes, n_nodes), dtype=np.float64)

    # Undirected → symmetric
    all_rows = np.array(rows + cols, dtype=np.int64)
    all_cols = np.array(cols + rows, dtype=np.int64)
    all_weights = np.array(weights + weights, dtype=np.float64)

    coo = sp.coo_matrix(
        (all_weights, (all_rows, all_cols)), shape=(n_nodes, n_nodes)
    )
    if combine == "sum":
        return coo.tocsr()
    # max: convert to CSR and reduce duplicates with maximum
    csr = coo.tocsr()
    csr.sum_duplicates()  # actually sum, but we don't care for max path
    # For "max", we re-build: deduplicate by (i, j) keeping the maximum
    coo = coo.tocoo()
    pair_to_max: dict[tuple[int, int], float] = {}
    for r, c, w in zip(coo.row, coo.col, coo.data, strict=True):
        key = (int(r), int(c))
        prev = pair_to_max.get(key)
        if prev is None or w > prev:
            pair_to_max[key] = float(w)
    rr = np.fromiter((k[0] for k in pair_to_max), dtype=np.int64)
    cc = np.fromiter((k[1] for k in pair_to_max), dtype=np.int64)
    ww = np.fromiter(pair_to_max.values(), dtype=np.float64)
    return sp.coo_matrix((ww, (rr, cc)), shape=(n_nodes, n_nodes)).tocsr()


def _symmetric_normalized_laplacian(W: sp.csr_matrix) -> tuple[sp.csr_matrix, np.ndarray]:
    """Return (L_sym, degree) for a symmetric weighted adjacency ``W``.

    L_sym = I - D^{-1/2} W D^{-1/2}
    """
    deg = np.asarray(W.sum(axis=1)).ravel()
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_W = D_inv_sqrt @ W @ D_inv_sqrt
    L = sp.eye(W.shape[0], format="csr") - norm_W
    return L.tocsr(), deg


def _spectral_embedding(
    L: sp.csr_matrix,
    degree: np.ndarray,
    *,
    n_components: int,
    drop_first: bool = True,
    dense_cutoff: int = 1500,
) -> tuple[np.ndarray, int]:
    """Compute the bottom eigenvectors of the symmetric normalized Laplacian.

    Isolated nodes (``degree == 0``) are excluded from the decomposition
    and assigned the zero embedding. Returns the embedding + the number
    of isolated nodes.

    Solver strategy
    ---------------
    - If the active sub-Laplacian has ≤ ``dense_cutoff`` nodes, use dense
      ``numpy.linalg.eigh``. This is O(n³) but robust and always
      converges — the right choice for small / medium graphs.
    - Otherwise, use ``scipy.sparse.linalg.eigsh`` with shift-invert,
      falling back to ``which="SM"`` and finally to a dense solve if
      ARPACK fails to converge (which can happen on graphs with many
      near-zero eigenvalues, e.g. when the graph has many disconnected
      components).
    """
    from scipy.sparse.linalg import eigsh, ArpackNoConvergence

    n = L.shape[0]
    active_mask = degree > 0
    n_isolated = int((~active_mask).sum())
    active_idx = np.where(active_mask)[0]
    n_active = active_idx.size

    if n_active < 2:
        return np.zeros((n, n_components), dtype=np.float64), n_isolated

    L_active = L[active_idx][:, active_idx]
    k_requested = n_components + (1 if drop_first else 0)
    k_requested = min(k_requested, n_active - 1)
    k_requested = max(k_requested, 1)

    vals: np.ndarray | None = None
    vecs: np.ndarray | None = None

    if n_active <= dense_cutoff:
        # Dense path: always robust.
        dense = L_active.toarray()
        dense = 0.5 * (dense + dense.T)  # enforce symmetry (numerical)
        all_vals, all_vecs = np.linalg.eigh(dense)
        vals = all_vals[:k_requested]
        vecs = all_vecs[:, :k_requested]
    else:
        # Sparse path with multiple fallbacks.
        try:
            vals, vecs = eigsh(L_active, k=k_requested, sigma=0.0, which="LM")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "eigsh shift-invert failed (%s); retrying with which='SM'", exc
            )
            try:
                vals, vecs = eigsh(L_active, k=k_requested, which="SM")
            except (ArpackNoConvergence, Exception) as exc2:  # noqa: BLE001
                logger.warning(
                    "eigsh which='SM' failed (%s); falling back to dense eigh", exc2
                )
                dense = L_active.toarray()
                dense = 0.5 * (dense + dense.T)
                all_vals, all_vecs = np.linalg.eigh(dense)
                vals = all_vals[:k_requested]
                vecs = all_vecs[:, :k_requested]

    assert vals is not None and vecs is not None
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]

    if drop_first and vecs.shape[1] > n_components:
        vecs = vecs[:, 1 : 1 + n_components]
    else:
        vecs = vecs[:, :n_components]

    # Re-expand to full N
    out = np.zeros((n, vecs.shape[1]), dtype=np.float64)
    out[active_idx] = vecs
    return out, n_isolated


# ─── Models ───────────────────────────────────────────────────────────────────


class _SpectralBase(BaseEmbeddingModel):
    """Shared spectral machinery. Subclasses choose the edge-type filter."""

    _edge_types: tuple[str, ...] | None = None  # None → all edges

    def __init__(
        self,
        *,
        n_components: int = 16,
        combine: str = "sum",
        l2_normalize: bool = True,
    ) -> None:
        super().__init__(
            n_components=n_components, combine=combine, l2_normalize=l2_normalize
        )
        self._n_components = n_components
        self._combine = combine
        self._l2 = l2_normalize
        self._embedding: pd.DataFrame | None = None
        self._schema_version: str = "unknown"
        self._n_isolated: int = 0

    def fit(
        self,
        dataset: HarmonizedDataset,
        *,
        graph: Any | None = None,
    ) -> _SpectralBase:
        if graph is None:
            raise ValueError(
                f"{type(self).__name__}.fit requires the Stage A multiplex graph "
                "via the `graph=` keyword."
            )

        n_nodes = dataset.n_patients
        W = _nx_multiplex_to_adjacency(
            graph,
            n_nodes=n_nodes,
            edge_types=self._edge_types,
            combine=self._combine,
        )
        if W.nnz == 0:
            raise RuntimeError(
                f"{type(self).__name__}: selected edge types produced an empty "
                "adjacency. Check the multiplex graph and edge_types filter."
            )

        L, deg = _symmetric_normalized_laplacian(W)
        vecs, n_iso = _spectral_embedding(
            L, deg, n_components=self._n_components, drop_first=True
        )

        if self._l2:
            vecs = self._l2_normalize_rows(vecs)

        col_names = [f"{self.name}_{i}" for i in range(vecs.shape[1])]
        self._embedding = pd.DataFrame(
            vecs, index=dataset.X.index, columns=col_names, dtype=np.float64
        )
        self._n_isolated = n_iso
        self._schema_version = dataset.schema.version
        self._config.update(
            {
                "n_components_requested": self._n_components,
                "n_components_effective": vecs.shape[1],
                "combine": self._combine,
                "edge_types": list(self._edge_types) if self._edge_types else "all",
                "n_edges_used": int(W.nnz // 2),
                "n_isolated_nodes": n_iso,
            }
        )
        self._fitted = True
        return self

    def transform(self) -> PatientEmbedding:
        self._ensure_fitted()
        assert self._embedding is not None
        return PatientEmbedding(
            values=self._embedding,
            model_name=self.name,
            model_config=self.config,
            view_dims={self.name: self._embedding.shape[1]},
            n_isolated_nodes=self._n_isolated,
            schema_version=self._schema_version,
        )


class TransdiagnosticSpectral(_SpectralBase):
    """Spectral embedding using only the ``transdiagnostic`` edge type.

    The most conservative graph view: uses only the trusted backbone
    built from data-driven transdiagnostic features. Every edge was built
    on at least 75% (default) of the selected transdiagnostic feature set.
    """

    name = "transdiagnostic_spectral"
    _edge_types = ("transdiagnostic",)


class MultiplexSpectral(_SpectralBase):
    """Spectral embedding on the weighted union of every edge type.

    Aggregates all 17 parallel edge types of the Stage A multiplex graph
    into a single weighted adjacency (sum of weights per node pair), then
    computes the symmetric normalized Laplacian spectral embedding. Uses
    the full multi-relational similarity structure — including the
    within-cohort blocks like ``cognition`` and ``biology`` that provide
    rich intra-cohort geometry.
    """

    name = "multiplex_spectral"
    _edge_types = None  # all edges
