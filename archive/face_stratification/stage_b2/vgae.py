"""Variational Graph Autoencoder (VGAE) — GAE with KL regularization.

Extends :class:`StageB2GAE` with a variational latent space: the encoder
outputs ``(mu, log_sigma)`` and the loss is
``BCE_link_prediction + beta * KL(q(Z|X,A) || N(0,I))``.

The KL term produces smoother embeddings than GAE because it prevents
the latent space from collapsing to a few isolated points.  This is
particularly relevant with ~9K patients and 17 edge types where GAE
can overfit to specific edge patterns.
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
    GCNEncoder,
    SparseGCNLayer,
    build_multiplex_adjacency_from_nx,
    get_device,
    normalize_adjacency,
)

logger = logging.getLogger(__name__)


class VGCNEncoder(nn.Module):
    """GCN encoder that outputs (mu, log_sigma) for the reparameterization trick.

    Shares all layers except the last, which is split into two parallel
    projections for the mean and log-variance.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        # Shared layers (all but last)
        self.shared = nn.ModuleList()
        if n_layers <= 1:
            prev = in_dim
        else:
            self.shared.append(SparseGCNLayer(in_dim, hidden_dim, dropout=dropout))
            for _ in range(n_layers - 2):
                self.shared.append(SparseGCNLayer(hidden_dim, hidden_dim, dropout=dropout))
            prev = hidden_dim

        # Two parallel output heads
        self.mu_layer = SparseGCNLayer(prev, out_dim, dropout=0.0)
        self.logvar_layer = SparseGCNLayer(prev, out_dim, dropout=0.0)

    def forward(
        self, H: torch.Tensor, norm_adj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = H
        for layer in self.shared:
            h = layer(h, norm_adj)
            h = F.relu(h)
            if self.training and self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=True)

        mu = self.mu_layer(h, norm_adj)
        logvar = self.logvar_layer(h, norm_adj)
        return mu, logvar


class StageB2VGAE(BaseEmbeddingModel):
    """Variational Graph Autoencoder on the Stage A multiplex graph.

    Parameters
    ----------
    hidden_dim:
        Hidden channel size.
    out_dim:
        Latent embedding dimension.
    n_layers:
        Depth of the shared GCN encoder.
    beta:
        KL divergence weight. ``beta=1.0`` is the standard VGAE;
        ``beta<1`` reduces regularization pressure.
    n_epochs:
        Number of training epochs.
    lr:
        Learning rate.
    edge_holdout:
        Fraction of edges held out for link prediction evaluation.
    """

    name = "vgae"

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        out_dim: int = 32,
        n_layers: int = 2,
        dropout: float = 0.1,
        beta: float = 1.0,
        n_epochs: int = 200,
        lr: float = 0.01,
        edge_holdout: float = 0.2,
        l2_normalize: bool = True,
        include_edge_types: list[str] | None = None,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=n_layers,
            dropout=dropout,
            beta=beta,
            n_epochs=n_epochs,
            lr=lr,
            edge_holdout=edge_holdout,
            l2_normalize=l2_normalize,
        )
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.beta = beta
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
    ) -> StageB2VGAE:
        if graph is None:
            raise ValueError("VGAE requires a graph (NetworkX MultiGraph)")

        device = get_device()
        n = dataset.X.shape[0]
        t0 = time.time()

        # Build adjacency
        A = build_multiplex_adjacency_from_nx(
            graph,
            n_nodes=n,
            combine="sum",
            include_edge_types=self.include_edge_types,
        )

        # Edge split: train / test
        coo = sp.triu(A).tocoo()
        edges = np.column_stack([coo.row, coo.col])
        rng = np.random.default_rng(42)
        n_test = max(1, int(len(edges) * self.edge_holdout))
        perm = rng.permutation(len(edges))
        test_edges = edges[perm[:n_test]]
        train_edges = edges[perm[n_test:]]

        # Build train adjacency
        train_rows = np.concatenate([train_edges[:, 0], train_edges[:, 1]])
        train_cols = np.concatenate([train_edges[:, 1], train_edges[:, 0]])
        train_weights = np.ones(len(train_rows), dtype=np.float64)
        A_train = sp.csr_matrix((train_weights, (train_rows, train_cols)), shape=(n, n))
        norm_adj = normalize_adjacency(A_train, device=device)

        # Node features
        X_np = dataset.X.fillna(0.0).to_numpy(dtype=np.float32)
        X_t = torch.from_numpy(X_np).to(device)

        # Negative sampling
        edge_set = {(min(u, v), max(u, v)) for u, v in edges}
        neg_test = _sample_negative_edges(n, len(test_edges), edge_set, rng=rng)

        # Model
        encoder = VGCNEncoder(
            X_np.shape[1],
            self.hidden_dim,
            self.out_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(encoder.parameters(), lr=self.lr)

        # Training
        encoder.train()
        for epoch in range(self.n_epochs):
            mu, logvar = encoder(X_t, norm_adj)

            # Reparameterization
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std

            # Link prediction loss (BCE on train edges + negatives)
            neg_train = _sample_negative_edges(n, len(train_edges), edge_set, rng=rng)

            pos_src = torch.tensor(train_edges[:, 0], device=device)
            pos_dst = torch.tensor(train_edges[:, 1], device=device)
            neg_src = torch.tensor(neg_train[:, 0], device=device)
            neg_dst = torch.tensor(neg_train[:, 1], device=device)

            pos_logits = (z[pos_src] * z[pos_dst]).sum(dim=1)
            neg_logits = (z[neg_src] * z[neg_dst]).sum(dim=1)

            logits = torch.cat([pos_logits, neg_logits])
            targets = torch.cat([
                torch.ones(len(pos_logits), device=device),
                torch.zeros(len(neg_logits), device=device),
            ])
            recon_loss = F.binary_cross_entropy_with_logits(logits, targets)

            # KL divergence
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            loss = recon_loss + self.beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                logger.debug(
                    "VGAE epoch %d/%d recon=%.4f kl=%.4f",
                    epoch + 1, self.n_epochs, recon_loss.item(), kl_loss.item(),
                )

        # Extract embeddings using the mean (no sampling)
        encoder.eval()
        with torch.no_grad():
            mu_final, _ = encoder(X_t, norm_adj)
            Z = mu_final.cpu().numpy()

        if self.l2_normalize:
            Z = self._l2_normalize_rows(Z)

        elapsed = time.time() - t0
        logger.info("VGAE: %d patients → %d dims in %.1fs", n, Z.shape[1], elapsed)

        columns = [f"vgae_{i}" for i in range(Z.shape[1])]
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
