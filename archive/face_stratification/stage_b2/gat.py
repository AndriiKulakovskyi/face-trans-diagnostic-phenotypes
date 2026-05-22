"""Graph Attention Network (GAT) for patient embedding.

Veličković et al. (2018). Learns per-edge attention weights that discover
which block-similarity edges carry more clinical information per patient.
This recovers edge heterogeneity that the standard GCN loses when summing
all edge types into a single adjacency.

Uses sparse attention (masked softmax over neighbors only) to keep memory
within 16 GB MPS budget at N=9,250.
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
from face_stratification.stage_b2.gae import _sample_negative_edges
from face_stratification.stage_b2.gcn import (
    build_multiplex_adjacency_from_nx,
    get_device,
)

logger = logging.getLogger(__name__)


class SparseGATLayer(nn.Module):
    """Single GAT layer with sparse attention over the adjacency structure.

    Attention: ``alpha_ij = softmax_j(LeakyReLU(a^T [Wh_i || Wh_j]))``

    Multi-head attention is implemented by concatenating (intermediate layers)
    or averaging (final layer) the outputs of ``n_heads`` independent attention heads.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        n_heads: int = 4,
        concat: bool = True,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.concat = concat
        self.dropout = dropout
        self.attention_dropout = attention_dropout

        # Per-head linear transforms
        self.W = nn.Parameter(torch.empty(n_heads, in_dim, out_dim))
        # Attention vectors: [a_left || a_right] per head
        self.a_left = nn.Parameter(torch.empty(n_heads, out_dim))
        self.a_right = nn.Parameter(torch.empty(n_heads, out_dim))

        self.leaky_relu = nn.LeakyReLU(negative_slope)

        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_left.unsqueeze(-1))
        nn.init.xavier_uniform_(self.a_right.unsqueeze(-1))

    def forward(
        self,
        H: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        H : (N, in_dim)
        edge_index : (2, E) source, target indices
        """
        n = H.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Project: (n_heads, N, out_dim)
        Wh = torch.einsum("nd,hdo->hno", H, self.W)  # (H, N, out_dim)

        # Attention logits per head
        e_left = torch.einsum("hnd,hd->hn", Wh, self.a_left)   # (H, N)
        e_right = torch.einsum("hnd,hd->hn", Wh, self.a_right)  # (H, N)

        # Per-edge attention: e_ij = LeakyReLU(e_left_i + e_right_j)
        e_ij = self.leaky_relu(e_left[:, src] + e_right[:, dst])  # (H, E)

        # Sparse softmax per destination node
        alpha = self._sparse_softmax(e_ij, dst, n)  # (H, E)

        if self.training and self.attention_dropout > 0:
            alpha = F.dropout(alpha, p=self.attention_dropout, training=True)

        # Aggregate: for each head, sum alpha_ij * Wh_j over neighbors
        # Wh_src: (H, E, out_dim)
        Wh_src = Wh[:, src, :]  # source node features
        weighted = alpha.unsqueeze(-1) * Wh_src  # (H, E, out_dim)

        # Scatter-add to destination nodes
        out = torch.zeros(self.n_heads, n, self.out_dim, device=H.device)
        dst_expanded = dst.unsqueeze(0).unsqueeze(-1).expand(
            self.n_heads, -1, self.out_dim
        )
        out.scatter_add_(1, dst_expanded, weighted)

        if self.concat:
            # Concatenate all heads: (N, n_heads * out_dim)
            return out.permute(1, 0, 2).reshape(n, self.n_heads * self.out_dim)
        else:
            # Average heads: (N, out_dim)
            return out.mean(dim=0)

    @staticmethod
    def _sparse_softmax(
        values: torch.Tensor,
        indices: torch.Tensor,
        n: int,
    ) -> torch.Tensor:
        """Softmax over sparse groups defined by ``indices``.

        Parameters
        ----------
        values : (H, E) attention logits
        indices : (E,) destination node indices
        n : number of nodes
        """
        # Numerical stability: subtract max per group
        max_vals = torch.full((values.size(0), n), -1e9, device=values.device)
        max_vals.scatter_reduce_(
            1,
            indices.unsqueeze(0).expand(values.size(0), -1),
            values,
            reduce="amax",
            include_self=True,
        )
        values = values - max_vals[:, indices]

        exp_vals = torch.exp(values)
        sum_exp = torch.zeros(values.size(0), n, device=values.device)
        sum_exp.scatter_add_(
            1,
            indices.unsqueeze(0).expand(values.size(0), -1),
            exp_vals,
        )
        return exp_vals / (sum_exp[:, indices] + 1e-12)


class GATEncoder(nn.Module):
    """Multi-layer GAT encoder.

    Intermediate layers concatenate heads; final layer averages them.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        l2_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout
        self.l2_normalize = l2_normalize
        self.layers = nn.ModuleList()

        if n_layers == 1:
            self.layers.append(SparseGATLayer(
                in_dim, out_dim, n_heads=1, concat=False,
                dropout=dropout, attention_dropout=attention_dropout,
            ))
        else:
            # First layer: concat heads
            self.layers.append(SparseGATLayer(
                in_dim, hidden_dim, n_heads=n_heads, concat=True,
                dropout=dropout, attention_dropout=attention_dropout,
            ))
            # Middle layers
            for _ in range(n_layers - 2):
                self.layers.append(SparseGATLayer(
                    hidden_dim * n_heads, hidden_dim, n_heads=n_heads, concat=True,
                    dropout=dropout, attention_dropout=attention_dropout,
                ))
            # Final layer: average heads
            self.layers.append(SparseGATLayer(
                hidden_dim * n_heads, out_dim, n_heads=1, concat=False,
                dropout=dropout, attention_dropout=attention_dropout,
            ))

    def forward(self, H: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = H
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index)
            if i < self.n_layers - 1:
                h = F.elu(h)
                if self.training and self.dropout > 0:
                    h = F.dropout(h, p=self.dropout, training=True)
        if self.l2_normalize:
            h = F.normalize(h, dim=1, eps=1e-8)
        return h


class StageB2GAT(BaseEmbeddingModel):
    """Graph Attention Network for patient embedding.

    Trains via link prediction (same as GAE) but uses attention to
    learn per-edge importance weights.

    Parameters
    ----------
    hidden_dim, out_dim:
        Encoder channel sizes.
    n_heads:
        Number of attention heads per layer.
    n_layers:
        Encoder depth.
    n_epochs:
        Training epochs.
    lr:
        Learning rate.
    """

    name = "gat"

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        out_dim: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        n_epochs: int = 200,
        lr: float = 0.005,
        edge_holdout: float = 0.2,
        l2_normalize: bool = True,
        include_edge_types: list[str] | None = None,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            attention_dropout=attention_dropout,
            n_epochs=n_epochs,
            lr=lr,
            l2_normalize=l2_normalize,
        )
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.edge_holdout = edge_holdout
        self.l2_normalize = l2_normalize
        self.include_edge_types = include_edge_types
        self._embedding: PatientEmbedding | None = None

    def fit(
        self,
        dataset: HarmonizedDataset,
        *,
        graph: Any | None = None,
    ) -> StageB2GAT:
        if graph is None:
            raise ValueError("GAT requires a graph (NetworkX MultiGraph)")

        device = get_device()
        n = dataset.X.shape[0]
        t0 = time.time()

        # Build adjacency
        A = build_multiplex_adjacency_from_nx(
            graph, n_nodes=n, combine="sum",
            include_edge_types=self.include_edge_types,
        )

        # Extract edge index
        A_sym = A + A.T  # ensure symmetric
        A_sym = A_sym.tocoo()
        edge_index = torch.tensor(
            np.vstack([A_sym.row, A_sym.col]), dtype=torch.long, device=device
        )

        # Node features
        X_np = dataset.X.fillna(0.0).to_numpy(dtype=np.float32)
        X_t = torch.from_numpy(X_np).to(device)

        # Edge split for evaluation
        coo_upper = sp.triu(A).tocoo()
        edges = np.column_stack([coo_upper.row, coo_upper.col])
        rng = np.random.default_rng(42)
        n_test = max(1, int(len(edges) * self.edge_holdout))
        perm = rng.permutation(len(edges))
        test_edges = edges[perm[:n_test]]
        train_edges = edges[perm[n_test:]]

        # Train edge index (symmetric)
        tr = np.concatenate([train_edges, train_edges[:, ::-1]], axis=0)
        train_edge_index = torch.tensor(tr.T, dtype=torch.long, device=device)

        # Model
        encoder = GATEncoder(
            X_np.shape[1],
            self.hidden_dim,
            self.out_dim,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            l2_normalize=self.l2_normalize,
        ).to(device)

        optimizer = torch.optim.Adam(encoder.parameters(), lr=self.lr)
        edge_set = {(min(u, v), max(u, v)) for u, v in edges}

        # Training via link prediction
        encoder.train()
        for epoch in range(self.n_epochs):
            z = encoder(X_t, train_edge_index)

            # Positive + negative edge logits
            neg = _sample_negative_edges(n, len(train_edges), edge_set, rng=rng)
            pos_src = torch.tensor(train_edges[:, 0], device=device)
            pos_dst = torch.tensor(train_edges[:, 1], device=device)
            neg_src = torch.tensor(neg[:, 0], device=device)
            neg_dst = torch.tensor(neg[:, 1], device=device)

            pos_logits = (z[pos_src] * z[pos_dst]).sum(dim=1)
            neg_logits = (z[neg_src] * z[neg_dst]).sum(dim=1)

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
                logger.debug("GAT epoch %d/%d loss=%.4f", epoch + 1, self.n_epochs, loss.item())

        # Extract embeddings
        encoder.eval()
        with torch.no_grad():
            Z = encoder(X_t, edge_index).cpu().numpy()

        elapsed = time.time() - t0
        logger.info("GAT: %d patients → %d dims in %.1fs", n, Z.shape[1], elapsed)

        columns = [f"gat_{i}" for i in range(Z.shape[1])]
        self._embedding = PatientEmbedding(
            values=pd.DataFrame(Z, index=dataset.X.index, columns=columns),
            model_name=self.name,
            model_config=self.config,
            view_dims={self.name: Z.shape[1]},
            schema_version=getattr(dataset.schema, "version", "unknown"),
        )
        self._fitted = True
        return self

    def transform(self) -> PatientEmbedding:
        self._ensure_fitted()
        assert self._embedding is not None
        return self._embedding
