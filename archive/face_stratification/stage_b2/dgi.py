"""Deep Graph Infomax (DGI) — mutual information maximization.

Velickovic et al. (2019). Maximizes mutual information between node-level
and graph-level representations. The negative samples come from a
corrupted graph where node features are row-shuffled.

Different inductive bias from GraphCL: DGI asks "what makes this patient
similar to the overall population structure" rather than "what is invariant
under perturbation."
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
from face_stratification.stage_b2.gcn import (
    GCNEncoder,
    build_multiplex_adjacency_from_nx,
    get_device,
    normalize_adjacency,
)

logger = logging.getLogger(__name__)


class BilinearDiscriminator(nn.Module):
    """Bilinear scoring function: D(h, s) = sigmoid(h^T W s).

    Scores whether a node embedding ``h`` and a graph summary ``s``
    belong to the same (real) graph or to a corrupted version.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.empty(dim, dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Score (N, d) node embeddings against (d,) graph summary."""
        # h: (N, d), s: (d,)
        return torch.sigmoid((h @ self.W) @ s)


class _Readout(nn.Module):
    """Graph-level readout: mean pool + sigmoid."""

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(h.mean(dim=0))


class StageB2DGI(BaseEmbeddingModel):
    """Deep Graph Infomax for patient embedding.

    Parameters
    ----------
    hidden_dim:
        GCN hidden channels.
    out_dim:
        Embedding dimension.
    n_layers:
        GCN encoder depth.
    n_epochs:
        Training epochs.
    lr:
        Learning rate.
    """

    name = "dgi"

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        out_dim: int = 32,
        n_layers: int = 2,
        dropout: float = 0.1,
        n_epochs: int = 300,
        lr: float = 0.001,
        l2_normalize: bool = True,
        include_edge_types: list[str] | None = None,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=n_layers,
            dropout=dropout,
            n_epochs=n_epochs,
            lr=lr,
            l2_normalize=l2_normalize,
        )
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.l2_normalize = l2_normalize
        self.include_edge_types = include_edge_types
        self._embedding: PatientEmbedding | None = None

    def fit(
        self,
        dataset: HarmonizedDataset,
        *,
        graph: Any | None = None,
    ) -> StageB2DGI:
        if graph is None:
            raise ValueError("DGI requires a graph (NetworkX MultiGraph)")

        device = get_device()
        n = dataset.X.shape[0]
        t0 = time.time()

        # Build adjacency
        A = build_multiplex_adjacency_from_nx(
            graph, n_nodes=n, combine="sum",
            include_edge_types=self.include_edge_types,
        )
        norm_adj = normalize_adjacency(A, device=device)

        # Node features
        X_np = dataset.X.fillna(0.0).to_numpy(dtype=np.float32)
        X_t = torch.from_numpy(X_np).to(device)

        # Model components
        encoder = GCNEncoder(
            X_np.shape[1],
            self.hidden_dim,
            self.out_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
            l2_normalize=False,  # DGI handles its own normalization
        ).to(device)

        discriminator = BilinearDiscriminator(self.out_dim).to(device)
        readout = _Readout().to(device)

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(discriminator.parameters()),
            lr=self.lr,
        )

        rng = np.random.default_rng(42)

        # Training
        encoder.train()
        discriminator.train()
        for epoch in range(self.n_epochs):
            # Positive: real graph
            h_pos = encoder(X_t, norm_adj)
            s = readout(h_pos)

            # Negative: corrupted graph (row-shuffle features)
            perm = torch.from_numpy(rng.permutation(n)).long().to(device)
            X_corrupted = X_t[perm]
            h_neg = encoder(X_corrupted, norm_adj)

            # Discriminator scores
            scores_pos = discriminator(h_pos, s)  # (N,) — should be high
            scores_neg = discriminator(h_neg, s)  # (N,) — should be low

            # Binary cross-entropy
            loss = -(
                torch.log(scores_pos + 1e-8).mean()
                + torch.log(1 - scores_neg + 1e-8).mean()
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                logger.debug(
                    "DGI epoch %d/%d loss=%.4f pos=%.3f neg=%.3f",
                    epoch + 1, self.n_epochs, loss.item(),
                    scores_pos.mean().item(), scores_neg.mean().item(),
                )

        # Extract embeddings
        encoder.eval()
        with torch.no_grad():
            Z = encoder(X_t, norm_adj).cpu().numpy()

        if self.l2_normalize:
            Z = self._l2_normalize_rows(Z)

        elapsed = time.time() - t0
        logger.info("DGI: %d patients → %d dims in %.1fs", n, Z.shape[1], elapsed)

        columns = [f"dgi_{i}" for i in range(Z.shape[1])]
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
