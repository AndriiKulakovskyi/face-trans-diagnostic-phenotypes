"""GraphCL-style contrastive self-supervised GNN.

Inspired by GraphCL (You et al. 2020) and GRACE (Zhu et al. 2020). Two
augmented views of the Stage A multiplex graph are encoded with a shared
GCN; an NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss pulls
the two views of the same patient together and pushes different patients
apart.

The augmentations used here are the two standard ones:

- **Edge drop** — each edge in the sparse adjacency is kept with
  probability ``1 - p_edge``.
- **Feature masking** — each feature column in the node-feature matrix
  is masked (set to 0) with probability ``p_feat``.

No labels required; the objective is fully self-supervised.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from face_stratification.harmonization.harmonizer import HarmonizedDataset
from face_stratification.models.base import BaseEmbeddingModel, PatientEmbedding
from face_stratification.stage_b2.gae import StageB2GAE
from face_stratification.stage_b2.gcn import (
    GCNEncoder,
    build_multiplex_adjacency_from_nx,
    get_device,
    normalize_adjacency,
)

logger = logging.getLogger(__name__)


# ─── Augmentations ─────────────────────────────────────────────────────────


def _drop_edges(A: sp.csr_matrix, p_edge: float, rng: np.random.Generator) -> sp.csr_matrix:
    """Symmetric edge-drop augmentation.

    Operates on the upper-triangular edges (so both (i,j) and (j,i) are
    dropped together) and returns a new symmetric csr matrix.
    """
    coo = A.tocoo()
    mask_ut = coo.row < coo.col
    ut_rows = coo.row[mask_ut]
    ut_cols = coo.col[mask_ut]
    ut_data = coo.data[mask_ut]

    keep = rng.random(ut_rows.size) > p_edge
    kept_rows = ut_rows[keep]
    kept_cols = ut_cols[keep]
    kept_data = ut_data[keep]

    sym_rows = np.concatenate([kept_rows, kept_cols])
    sym_cols = np.concatenate([kept_cols, kept_rows])
    sym_data = np.concatenate([kept_data, kept_data])

    return sp.coo_matrix((sym_data, (sym_rows, sym_cols)), shape=A.shape).tocsr()


def _mask_features(X: torch.Tensor, p_feat: float, rng: np.random.Generator) -> torch.Tensor:
    """Column-wise feature masking: zero out a fraction of columns."""
    if p_feat <= 0:
        return X
    d = X.shape[1]
    mask = rng.random(d) > p_feat
    col_keep = torch.tensor(mask, dtype=torch.float32, device=X.device)
    return X * col_keep.unsqueeze(0)


# ─── NT-Xent loss ──────────────────────────────────────────────────────────


def _nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    *,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Normalized temperature-scaled cross-entropy loss on two views.

    ``z1`` and ``z2`` are two L2-normalized ``(N, d)`` embeddings of the
    same ``N`` patients under different augmentations. The loss treats
    ``(z1[i], z2[i])`` as a positive pair and all other pairs as negatives.
    """
    N, _ = z1.shape
    # Concatenate and build a (2N, 2N) similarity matrix.
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)
    sim = (z @ z.T) / temperature

    # Mask out self-similarity
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))

    # For each row i in [0, N), the positive is at column i + N, and
    # for each row i in [N, 2N), the positive is at column i - N.
    positives = torch.cat([
        torch.arange(N, 2 * N, device=z.device),
        torch.arange(0, N, device=z.device),
    ])
    loss = F.cross_entropy(sim, positives)
    return loss


# ─── Model ─────────────────────────────────────────────────────────────────


class StageB2GraphContrastive(BaseEmbeddingModel):
    """GraphCL-style contrastive self-supervised GCN.

    Parameters
    ----------
    p_edge, p_feat:
        Augmentation probabilities for edge drop and feature masking.
    temperature:
        NT-Xent temperature.
    All other parameters: same as :class:`StageB2GAE`.
    """

    name = "stage_b2_contrastive"

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        out_dim: int = 32,
        n_layers: int = 2,
        n_epochs: int = 150,
        learning_rate: float = 5e-3,
        weight_decay: float = 5e-4,
        dropout: float = 0.1,
        l2_normalize: bool = True,
        seed: int = 0,
        device: torch.device | str | None = None,
        feature_source: str = "composite",
        p_edge: float = 0.2,
        p_feat: float = 0.1,
        temperature: float = 0.5,
        include_edge_types: tuple[str, ...] | None = None,
        exclude_edge_types: tuple[str, ...] = (),
    ) -> None:
        resolved_device = torch.device(device) if device is not None else get_device()
        super().__init__(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=n_layers,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout=dropout,
            l2_normalize=l2_normalize,
            seed=seed,
            device=str(resolved_device),
            feature_source=feature_source,
            p_edge=p_edge,
            p_feat=p_feat,
            temperature=temperature,
            include_edge_types=include_edge_types,
            exclude_edge_types=exclude_edge_types,
        )
        self._hidden_dim = hidden_dim
        self._out_dim = out_dim
        self._n_layers = n_layers
        self._n_epochs = n_epochs
        self._lr = learning_rate
        self._wd = weight_decay
        self._dropout = dropout
        self._l2 = l2_normalize
        self._seed = seed
        self._device = resolved_device
        self._feature_source = feature_source
        self._p_edge = p_edge
        self._p_feat = p_feat
        self._temperature = temperature
        self._include_edge_types = include_edge_types
        self._exclude_edge_types = exclude_edge_types

        self._encoder: GCNEncoder | None = None
        self._embedding: pd.DataFrame | None = None
        self._training_history: list[dict[str, float]] = []
        self._schema_version: str = "unknown"

    def fit(
        self,
        dataset: HarmonizedDataset,
        *,
        graph: Any | None = None,
    ) -> StageB2GraphContrastive:
        if graph is None:
            raise ValueError("StageB2GraphContrastive requires the Stage A multiplex graph")

        device = self._device
        rng = np.random.default_rng(self._seed)
        torch.manual_seed(self._seed)

        helper = StageB2GAE(feature_source=self._feature_source)
        helper._feature_source = self._feature_source
        features = helper._resolve_features(dataset)
        n_nodes, in_dim = features.shape
        logger.info(
            "StageB2GraphContrastive: features %d × %d, p_edge=%.2f, p_feat=%.2f, device=%s",
            n_nodes, in_dim, self._p_edge, self._p_feat, device,
        )

        A_full = build_multiplex_adjacency_from_nx(
            graph,
            n_nodes=n_nodes,
            combine="sum",
            include_edge_types=self._include_edge_types,
            exclude_edge_types=self._exclude_edge_types or (),
        )
        logger.info(
            "  adjacency: %d nodes, %d edges (include=%s, exclude=%s)",
            n_nodes, A_full.nnz // 2,
            self._include_edge_types, self._exclude_edge_types,
        )

        X_full = torch.tensor(features, dtype=torch.float32, device=device)

        self._encoder = GCNEncoder(
            in_dim=in_dim,
            hidden_dim=self._hidden_dim,
            out_dim=self._out_dim,
            n_layers=self._n_layers,
            dropout=self._dropout,
            l2_normalize=self._l2,
            device=device,
        )

        optimizer = torch.optim.Adam(
            self._encoder.parameters(),
            lr=self._lr,
            weight_decay=self._wd,
        )

        self._encoder.train()
        self._training_history = []
        t_start = time.time()

        for epoch in range(self._n_epochs):
            optimizer.zero_grad()

            A1 = _drop_edges(A_full, self._p_edge, rng)
            A2 = _drop_edges(A_full, self._p_edge, rng)
            norm_adj_1 = normalize_adjacency(A1, device=device)
            norm_adj_2 = normalize_adjacency(A2, device=device)
            X1 = _mask_features(X_full, self._p_feat, rng)
            X2 = _mask_features(X_full, self._p_feat, rng)

            z1 = self._encoder(X1, norm_adj_1)
            z2 = self._encoder(X2, norm_adj_2)

            loss = _nt_xent_loss(z1, z2, temperature=self._temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._encoder.parameters(), max_norm=5.0)
            optimizer.step()

            if epoch % max(self._n_epochs // 10, 1) == 0 or epoch == self._n_epochs - 1:
                self._training_history.append({
                    "epoch": epoch,
                    "loss": float(loss.item()),
                })
                logger.info("  epoch %3d  loss=%.4f", epoch, loss.item())

        # Final embedding — unaugmented graph, back to CPU for numpy
        self._encoder.eval()
        norm_adj_full = normalize_adjacency(A_full, device=device)
        with torch.no_grad():
            Z = self._encoder(X_full, norm_adj_full).cpu().numpy()

        col_names = [f"{self.name}_{i}" for i in range(Z.shape[1])]
        self._embedding = pd.DataFrame(
            Z, index=dataset.X.index, columns=col_names, dtype=np.float64
        )
        self._schema_version = dataset.schema.version
        self._config.update({
            "n_nodes": n_nodes,
            "in_dim": in_dim,
            "out_dim": Z.shape[1],
            "device": str(device),
            "training_time_seconds": float(time.time() - t_start),
            "final_loss": self._training_history[-1]["loss"] if self._training_history else None,
        })
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
            n_isolated_nodes=0,
            schema_version=self._schema_version,
        )
