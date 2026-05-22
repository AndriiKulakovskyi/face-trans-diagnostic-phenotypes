"""Sparse GCN primitives built on plain PyTorch (no torch-geometric).

Two layers and an encoder are enough to produce high-quality 11 k-patient
embeddings; keeping the code to a single file makes it auditable.

Math
----
A standard Kipf-Welling GCN layer computes

    H' = σ( D̂^(-1/2) · Â · D̂^(-1/2) · H · W )

where ``Â = A + I`` is the self-looped adjacency and ``D̂`` is its degree
matrix. We precompute the normalized adjacency once as a sparse COO tensor
and reuse it for every forward pass.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Auto-detect best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_adjacency(
    A: sp.csr_matrix,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build the symmetric normalized adjacency with self-loops as a sparse tensor.

    Returns a ``torch.sparse_coo_tensor`` of shape ``(N, N)`` suitable for
    ``torch.sparse.mm`` multiplications.  When *device* is given the tensor
    is created directly on that device.
    """
    n = A.shape[0]
    A_tilde = A + sp.eye(n, format="csr")
    deg = np.asarray(A_tilde.sum(axis=1)).ravel()
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_A = D_inv_sqrt @ A_tilde @ D_inv_sqrt

    coo = norm_A.tocoo()
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    result = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    if device is not None:
        result = result.to(device)
    return result


class SparseGCNLayer(nn.Module):
    """Single Kipf-Welling GCN layer using sparse matmul.

    Parameters
    ----------
    in_dim, out_dim:
        Channel sizes.
    bias:
        Whether to include a bias term.
    dropout:
        Dropout applied to the input ``H`` before propagation (standard
        GCN regularization, applied at train time only).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.dropout = dropout
        # Glorot init
        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, H: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        if self.training and self.dropout > 0:
            H = F.dropout(H, p=self.dropout, training=True)
        # Apply linear transform, then propagate over sparse adjacency.
        HW = self.linear(H)
        return torch.sparse.mm(norm_adj, HW)


class GCNEncoder(nn.Module):
    """Variable-depth GCN encoder.

    Layer stack pattern:

        [GCN → ReLU → dropout] × (n_layers - 1)  →  GCN

    The final layer is left linear so the produced embedding lives in
    ``ℝ^out_dim`` with no activation restriction. All intermediate layers
    use ``hidden_dim`` channels.

    ``n_layers = 1`` collapses to a single linear propagation
    (``in_dim → out_dim``); ``n_layers = 2`` matches the Stage B2 default;
    larger values add depth without skip connections (watch for
    over-smoothing past ``n_layers = 3``).

    The output can be optionally L2-normalized to make downstream cosine
    clustering well-behaved.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        n_layers: int = 2,
        dropout: float = 0.1,
        l2_normalize: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be ≥ 1, got {n_layers}")
        self.n_layers = n_layers
        self.dropout = dropout
        self.l2_normalize = l2_normalize

        self.layers = nn.ModuleList()
        if n_layers == 1:
            self.layers.append(SparseGCNLayer(in_dim, out_dim, dropout=dropout))
        else:
            self.layers.append(SparseGCNLayer(in_dim, hidden_dim, dropout=dropout))
            for _ in range(n_layers - 2):
                self.layers.append(SparseGCNLayer(hidden_dim, hidden_dim, dropout=dropout))
            self.layers.append(SparseGCNLayer(hidden_dim, out_dim, dropout=dropout))

        if device is not None:
            self.to(device)

    def forward(self, H: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        h = H
        for i, layer in enumerate(self.layers):
            h = layer(h, norm_adj)
            # ReLU + dropout after every layer except the last
            if i < self.n_layers - 1:
                h = F.relu(h)
                if self.training and self.dropout > 0:
                    h = F.dropout(h, p=self.dropout, training=True)
        if self.l2_normalize:
            h = F.normalize(h, dim=1, eps=1e-8)
        return h


# ─── Adjacency helpers (public API) ─────────────────────────────────────────


def build_multiplex_adjacency_from_nx(
    G: Any,
    *,
    n_nodes: int,
    combine: str = "sum",
    include_edge_types: Iterable[str] | None = None,
    exclude_edge_types: Iterable[str] | None = None,
    device: torch.device | str | None = None,
) -> sp.csr_matrix:
    """Collapse a NetworkX MultiGraph's edges into one weighted sparse adjacency.

    Each edge carries a ``block`` attribute naming its clinical edge type
    (e.g. ``"mood"``, ``"psychosis"``, ``"transdiagnostic"``). This helper
    supports filtering:

    - ``include_edge_types``: keep only these edge types.
    - ``exclude_edge_types``: drop these edge types.

    If both are given, ``include`` is applied first.

    Returns a scipy CSR matrix.
    """
    if combine not in {"sum", "max"}:
        raise ValueError(f"Unknown combine={combine!r}")

    include_set = set(include_edge_types) if include_edge_types is not None else None
    exclude_set = set(exclude_edge_types) if exclude_edge_types is not None else set()

    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []
    for u, v, data in G.edges(data=True):
        block = data.get("block", "unknown")
        if include_set is not None and block not in include_set:
            continue
        if block in exclude_set:
            continue
        w = float(data.get("weight", 1.0))
        if w <= 0:
            continue
        rows.append(u)
        cols.append(v)
        weights.append(w)

    if not rows:
        return sp.csr_matrix((n_nodes, n_nodes), dtype=np.float64)

    all_rows = np.array(rows + cols, dtype=np.int64)
    all_cols = np.array(cols + rows, dtype=np.int64)
    all_weights = np.array(weights + weights, dtype=np.float64)

    coo = sp.coo_matrix((all_weights, (all_rows, all_cols)), shape=(n_nodes, n_nodes))
    if combine == "sum":
        return coo.tocsr()
    # max: dedup by taking the max of parallel edges
    coo = coo.tocoo()
    pair_to_max: dict[tuple[int, int], float] = {}
    for r, c, w in zip(coo.row, coo.col, coo.data, strict=True):
        key = (int(r), int(c))
        prev = pair_to_max.get(key)
        if prev is None or w > prev:
            pair_to_max[key] = float(w)
    rr = np.fromiter((k[0] for k in pair_to_max), dtype=np.int64, count=len(pair_to_max))
    cc = np.fromiter((k[1] for k in pair_to_max), dtype=np.int64, count=len(pair_to_max))
    ww = np.fromiter(pair_to_max.values(), dtype=np.float64, count=len(pair_to_max))
    return sp.coo_matrix((ww, (rr, cc)), shape=(n_nodes, n_nodes)).tocsr()
