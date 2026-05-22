"""R-GCN (Relational Graph Convolutional Network) with sparse operations.

Preserves per-block edge type information using relation-specific weight
matrices with basis decomposition. Rewritten with sparse adjacency matrices
to keep memory within 16 GB MPS budget (the original dense implementation
required ~5.7 GB for 17 relation matrices at N=9,250).

Implements :class:`BaseEmbeddingModel` for seamless integration with the
Stage B pipeline.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from face_stratification.harmonization.harmonizer import HarmonizedDataset
from face_stratification.models.base import BaseEmbeddingModel, PatientEmbedding
from face_stratification.stage_b2.gcn import get_device

logger = logging.getLogger(__name__)


def _scipy_to_sparse_tensor(
    A: sp.csr_matrix,
    device: torch.device,
) -> torch.Tensor:
    """Convert scipy sparse matrix to torch sparse COO tensor on device."""
    coo = A.tocoo()
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    t = torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()
    return t.to(device)


def _row_normalize_sparse(A: sp.csr_matrix) -> sp.csr_matrix:
    """Row-normalize a sparse adjacency matrix: D^{-1} A."""
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg_inv = np.where(deg > 0, 1.0 / deg, 0.0)
    return sp.diags(deg_inv) @ A


class SparseRGCNLayer(nn.Module):
    """Single R-GCN layer with basis decomposition and sparse message-passing.

    Instead of dense N×N adjacency per relation, this layer uses sparse
    torch.mm for each relation — memory is O(E) not O(R × N²).

    Parameters
    ----------
    in_features, out_features:
        Channel sizes.
    n_relations:
        Number of distinct edge types (clinical blocks).
    n_bases:
        Basis decomposition rank. Limits parameters to
        ``n_bases × in × out + n_relations × n_bases`` instead of
        ``n_relations × in × out``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_relations: int,
        n_bases: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_relations = n_relations
        self.n_bases = n_bases or min(n_relations, 8)
        self.dropout = dropout

        # Basis decomposition: W_r = sum_b c_rb * V_b
        self.bases = nn.Parameter(torch.empty(self.n_bases, in_features, out_features))
        self.coefficients = nn.Parameter(torch.empty(n_relations, self.n_bases))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Self-loop weight
        self.self_weight = nn.Linear(in_features, out_features, bias=False)

        nn.init.xavier_uniform_(self.bases)
        nn.init.xavier_uniform_(self.coefficients)

    def forward(
        self,
        x: torch.Tensor,
        sparse_adjs: list[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass with sparse adjacency matrices.

        Parameters
        ----------
        x : (N, in_features)
        sparse_adjs : list of sparse (N, N) tensors, one per relation
        """
        # Compose relation-specific weights from bases
        # W: (n_relations, in_features, out_features)
        W = torch.einsum("rb,bio->rio", self.coefficients, self.bases)

        out = self.self_weight(x)

        for r, adj in enumerate(sparse_adjs):
            if r >= self.n_relations:
                break
            # Message passing: A_r @ (X @ W_r) — sparse @ dense
            h = torch.matmul(x, W[r])  # (N, out)
            h = torch.sparse.mm(adj, h)  # sparse (N,N) @ (N, out)
            out = out + h

        return out + self.bias


class SparseRGCNEncoder(nn.Module):
    """Multi-layer R-GCN encoder with residual connections and sparse ops."""

    def __init__(
        self,
        n_features: int,
        n_relations: int,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        n_layers: int = 2,
        n_bases: int | None = None,
        dropout: float = 0.1,
        l2_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.l2_normalize = l2_normalize

        dims = [n_features] + [hidden_dim] * (n_layers - 1) + [embed_dim]
        self.layers = nn.ModuleList([
            SparseRGCNLayer(dims[i], dims[i + 1], n_relations, n_bases, dropout)
            for i in range(n_layers)
        ])
        self.residuals = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1], bias=False)
            if dims[i] != dims[i + 1] else nn.Identity()
            for i in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        sparse_adjs: list[torch.Tensor],
    ) -> torch.Tensor:
        h = x
        for i, (layer, residual) in enumerate(zip(self.layers, self.residuals)):
            h_new = layer(h, sparse_adjs)
            if i < self.n_layers - 1:
                h_new = F.elu(h_new)
                h_new = self.dropout(h_new)
            h = h_new + residual(h)
        if self.l2_normalize:
            h = F.normalize(h, dim=1, eps=1e-8)
        return h


class StageB2RGCN(BaseEmbeddingModel):
    """R-GCN embedding model with sparse operations.

    Preserves per-block edge types as separate sparse relation matrices.
    Uses basis decomposition to keep parameter count manageable.
    Memory: O(E_total) instead of O(R × N²).

    Parameters
    ----------
    hidden_dim:
        Hidden channel size.
    out_dim:
        Embedding dimension.
    n_layers:
        Encoder depth.
    n_bases:
        Basis decomposition rank.
    n_epochs:
        Training epochs.
    lr:
        Learning rate.
    """

    name = "rgcn"

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        out_dim: int = 32,
        n_layers: int = 2,
        n_bases: int | None = 8,
        dropout: float = 0.1,
        n_epochs: int = 200,
        lr: float = 0.01,
        l2_normalize: bool = True,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=n_layers,
            n_bases=n_bases,
            dropout=dropout,
            n_epochs=n_epochs,
            lr=lr,
            l2_normalize=l2_normalize,
        )
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.n_bases = n_bases
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.l2_normalize = l2_normalize
        self._embedding: PatientEmbedding | None = None

    def _build_sparse_adjacencies(
        self,
        graph: Any,
        n: int,
        device: torch.device,
    ) -> tuple[list[torch.Tensor], list[str]]:
        """Extract per-block sparse adjacency matrices from a NetworkX MultiGraph."""
        # Collect edges by block type
        block_edges: dict[str, list[tuple[int, int, float]]] = {}
        for u, v, data in graph.edges(data=True):
            block = data.get("block", "unknown")
            w = float(data.get("weight", 1.0))
            if w <= 0:
                continue
            block_edges.setdefault(block, []).append((u, v, w))

        block_names = sorted(block_edges.keys())
        sparse_adjs = []

        for block in block_names:
            edges = block_edges[block]
            rows = np.array([e[0] for e in edges] + [e[1] for e in edges], dtype=np.int64)
            cols = np.array([e[1] for e in edges] + [e[0] for e in edges], dtype=np.int64)
            weights = np.array([e[2] for e in edges] * 2, dtype=np.float64)

            A = sp.csr_matrix((weights, (rows, cols)), shape=(n, n))
            A_norm = _row_normalize_sparse(A)
            sparse_adjs.append(_scipy_to_sparse_tensor(A_norm, device))

        return sparse_adjs, block_names

    def fit(
        self,
        dataset: HarmonizedDataset,
        *,
        graph: Any | None = None,
    ) -> StageB2RGCN:
        if graph is None:
            raise ValueError("R-GCN requires a graph (NetworkX MultiGraph)")

        device = get_device()
        n = dataset.X.shape[0]
        t0 = time.time()

        sparse_adjs, block_names = self._build_sparse_adjacencies(graph, n, device)
        n_relations = len(sparse_adjs)

        # Node features
        X_np = dataset.X.fillna(0.0).to_numpy(dtype=np.float32)
        X_t = torch.from_numpy(X_np).to(device)

        # Combined adjacency for reconstruction target (sparse)
        from face_stratification.stage_b2.gcn import build_multiplex_adjacency_from_nx
        A_combined = build_multiplex_adjacency_from_nx(graph, n_nodes=n, combine="sum")

        # Encoder
        encoder = SparseRGCNEncoder(
            n_features=X_np.shape[1],
            n_relations=n_relations,
            hidden_dim=self.hidden_dim,
            embed_dim=self.out_dim,
            n_layers=self.n_layers,
            n_bases=self.n_bases,
            dropout=self.dropout,
            l2_normalize=self.l2_normalize,
        ).to(device)

        optimizer = torch.optim.Adam(encoder.parameters(), lr=self.lr)

        # Sample edges for link prediction (instead of dense N×N target)
        coo = sp.triu(A_combined).tocoo()
        pos_edges = np.column_stack([coo.row, coo.col])
        edge_set = {(min(u, v), max(u, v)) for u, v in pos_edges}
        rng = np.random.default_rng(42)

        # Training via link prediction (memory-efficient)
        encoder.train()
        for epoch in range(self.n_epochs):
            Z = encoder(X_t, sparse_adjs)

            # Sample negatives
            from face_stratification.stage_b2.gae import _sample_negative_edges
            neg_edges = _sample_negative_edges(n, len(pos_edges), edge_set, rng=rng)

            pos_src = torch.tensor(pos_edges[:, 0], device=device)
            pos_dst = torch.tensor(pos_edges[:, 1], device=device)
            neg_src = torch.tensor(neg_edges[:, 0], device=device)
            neg_dst = torch.tensor(neg_edges[:, 1], device=device)

            pos_logits = (Z[pos_src] * Z[pos_dst]).sum(dim=1)
            neg_logits = (Z[neg_src] * Z[neg_dst]).sum(dim=1)

            logits = torch.cat([pos_logits, neg_logits])
            targets = torch.cat([
                torch.ones(len(pos_logits), device=device),
                torch.zeros(len(neg_logits), device=device),
            ])
            loss = F.binary_cross_entropy_with_logits(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                logger.debug(
                    "R-GCN epoch %d/%d loss=%.4f",
                    epoch + 1, self.n_epochs, loss.item(),
                )

        # Extract embeddings
        encoder.eval()
        with torch.no_grad():
            Z = encoder(X_t, sparse_adjs).cpu().numpy()

        elapsed = time.time() - t0
        logger.info(
            "R-GCN: %d patients → %d dims, %d relations in %.1fs",
            n, Z.shape[1], n_relations, elapsed,
        )

        columns = [f"rgcn_{i}" for i in range(Z.shape[1])]
        self._embedding = PatientEmbedding(
            values=pd.DataFrame(Z, index=dataset.X.index, columns=columns),
            model_name=self.name,
            model_config={**self.config, "n_relations": n_relations, "block_names": block_names},
            view_dims={self.name: Z.shape[1]},
            schema_version=getattr(dataset.schema, "version", "unknown"),
        )
        self._fitted = True
        return self

    def transform(self) -> PatientEmbedding:
        self._ensure_fitted()
        assert self._embedding is not None
        return self._embedding
