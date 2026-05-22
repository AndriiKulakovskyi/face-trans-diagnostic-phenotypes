"""Graph Autoencoder (GAE) — unsupervised GCN trained via link prediction.

Kipf & Welling (2016) Graph Autoencoder. The encoder is the 2-layer
:class:`~face_stratification.stage_b2.gcn.GCNEncoder`; the decoder is the
inner product of the encoded representations plus a sigmoid. Training
minimizes the binary cross-entropy over a mix of positive edges (from the
Stage A multiplex graph) and randomly sampled negatives.

The result is a ``(N × d)`` embedding matrix suitable for clustering,
concatenation with other Stage B views, or any downstream task that
consumes dense patient representations.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from face_stratification.harmonization.harmonizer import HarmonizedDataset
from face_stratification.models.base import BaseEmbeddingModel, PatientEmbedding
from face_stratification.stage_b2.gcn import (
    GCNEncoder,
    build_multiplex_adjacency_from_nx,
    get_device,
    normalize_adjacency,
)

logger = logging.getLogger(__name__)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _sample_negative_edges(
    n_nodes: int,
    n_samples: int,
    existing_edges_set: set[tuple[int, int]],
    *,
    rng: np.random.Generator,
    max_attempts: int = 10,
) -> np.ndarray:
    """Uniformly sample ``n_samples`` non-edge pairs (negative sampling)."""
    sampled: set[tuple[int, int]] = set()
    attempts = 0
    while len(sampled) < n_samples and attempts < max_attempts:
        batch = rng.integers(0, n_nodes, size=(n_samples * 2, 2))
        for u, v in batch:
            u, v = int(u), int(v)
            if u == v:
                continue
            key = (min(u, v), max(u, v))
            if key in existing_edges_set or key in sampled:
                continue
            sampled.add(key)
            if len(sampled) >= n_samples:
                break
        attempts += 1
    return np.array(list(sampled), dtype=np.int64)


# ─── Model ───────────────────────────────────────────────────────────────────


class StageB2GAE(BaseEmbeddingModel):
    """Graph Autoencoder trained on the Stage A multiplex graph.

    Parameters
    ----------
    hidden_dim:
        Hidden channel size of the first GCN layer.
    out_dim:
        Output embedding dimension (the second GCN layer output).
    n_epochs:
        Number of training iterations (full-batch; the encoder is cheap
        enough to run the whole graph per step).
    learning_rate:
        Adam learning rate. 1e-2 is a standard GCN default.
    weight_decay:
        L2 regularization on the weight matrices.
    dropout:
        Dropout applied inside each GCN layer at train time.
    l2_normalize:
        If True, L2-normalize the final embedding row-wise so cosine
        clustering downstream is well-behaved.
    seed:
        Torch + numpy seed for reproducibility.
    device:
        ``"cpu"`` (default) or ``"cuda"``.
    feature_source:
        ``"composite"`` (default) — use the Stage B composite embedding
        as node features. ``"raw"`` — use the normalized Stage A matrix
        after filling NaN with 0 (less well-conditioned).
    """

    name = "stage_b2_gae"

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        out_dim: int = 32,
        n_layers: int = 2,
        n_epochs: int = 150,
        learning_rate: float = 1e-2,
        weight_decay: float = 5e-4,
        dropout: float = 0.1,
        l2_normalize: bool = True,
        seed: int = 0,
        device: torch.device | str | None = None,
        feature_source: str = "composite",
        negative_sample_ratio: float = 1.0,
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
            negative_sample_ratio=negative_sample_ratio,
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
        self._neg_ratio = negative_sample_ratio
        self._include_edge_types = include_edge_types
        self._exclude_edge_types = exclude_edge_types

        self._encoder: GCNEncoder | None = None
        self._embedding: pd.DataFrame | None = None
        self._training_history: list[dict[str, float]] = []
        self._schema_version: str = "unknown"

    # ─── Edge splitting ────────────────────────────────────────────────

    @staticmethod
    def _split_edges(
        pos_edges: np.ndarray,
        test_fraction: float = 0.2,
        *,
        edge_types: np.ndarray | None = None,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split positive edges into train / test sets.

        Parameters
        ----------
        pos_edges:
            ``(E, 2)`` array of undirected edges.
        test_fraction:
            Fraction of edges held out for evaluation.
        edge_types:
            Optional ``(E,)`` string array. When provided the split is
            stratified so every edge type is represented proportionally
            in both train and test sets.
        rng:
            Numpy random generator for reproducibility.

        Returns
        -------
        (train_edges, test_edges): each ``(E_split, 2)`` int64 arrays.
        """
        n_edges = pos_edges.shape[0]
        n_test = max(1, int(round(test_fraction * n_edges)))

        if edge_types is not None and len(np.unique(edge_types)) > 1:
            train_idx_list: list[int] = []
            test_idx_list: list[int] = []
            for etype in np.unique(edge_types):
                type_idx = np.where(edge_types == etype)[0]
                rng.shuffle(type_idx)
                n_type_test = max(1, int(round(test_fraction * len(type_idx))))
                test_idx_list.extend(type_idx[:n_type_test].tolist())
                train_idx_list.extend(type_idx[n_type_test:].tolist())
            train_idx = np.array(train_idx_list, dtype=np.int64)
            test_idx = np.array(test_idx_list, dtype=np.int64)
        else:
            perm = rng.permutation(n_edges)
            test_idx = perm[:n_test]
            train_idx = perm[n_test:]

        return pos_edges[train_idx], pos_edges[test_idx]

    # ─── Fit / transform ────────────────────────────────────────────────

    def fit(
        self,
        dataset: HarmonizedDataset,
        *,
        graph: Any | None = None,
    ) -> StageB2GAE:
        if graph is None:
            raise ValueError("StageB2GAE requires the Stage A multiplex graph (pass via `graph=`).")

        device = self._device
        rng = np.random.default_rng(self._seed)
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)

        # ─── Build features ─────────────────────────────────────────────
        features = self._resolve_features(dataset)
        n_nodes, in_dim = features.shape
        logger.info(
            "StageB2GAE: features %d × %d from '%s' on %s",
            n_nodes, in_dim, self._feature_source, device,
        )

        # ─── Build adjacency ────────────────────────────────────────────
        A = build_multiplex_adjacency_from_nx(
            graph,
            n_nodes=n_nodes,
            combine="sum",
            include_edge_types=self._include_edge_types,
            exclude_edge_types=self._exclude_edge_types or (),
        )
        logger.info(
            "  adjacency: %d nodes, %d edges (include=%s, exclude=%s)",
            n_nodes, A.nnz // 2,
            self._include_edge_types, self._exclude_edge_types,
        )
        norm_adj = normalize_adjacency(A, device=device)

        # ─── Positive edges for link prediction ─────────────────────────
        coo_A = A.tocoo()
        mask_ut = coo_A.row < coo_A.col
        pos_rows = coo_A.row[mask_ut]
        pos_cols = coo_A.col[mask_ut]
        pos_edges = np.vstack([pos_rows, pos_cols]).T.astype(np.int64)
        n_pos = pos_edges.shape[0]
        if n_pos == 0:
            raise RuntimeError("No positive edges in the multiplex graph")

        # Extract dominant edge type per (u,v) pair for stratified splitting
        edge_type_map: dict[tuple[int, int], str] = {}
        for u, v, data in graph.edges(data=True):
            key = (min(u, v), max(u, v))
            block = data.get("block", "unknown")
            w = float(data.get("weight", 1.0))
            prev_block, prev_w = edge_type_map.get(key, (None, 0.0))
            if prev_block is None or w > prev_w:
                edge_type_map[key] = (block, w)  # type: ignore[assignment]
        edge_types_arr: np.ndarray | None = None
        if edge_type_map:
            labels = [
                edge_type_map.get((int(r), int(c)), ("unknown", 0.0))[0]
                for r, c in pos_edges
            ]
            edge_types_arr = np.array(labels)

        # ─── Train / test split ──────────────────────────────────────────
        train_edges, test_edges = self._split_edges(
            pos_edges, test_fraction=0.2,
            edge_types=edge_types_arr, rng=rng,
        )
        existing_set = {(int(u), int(v)) for u, v in pos_edges}
        logger.info(
            "  positive edges: %d (train=%d, test=%d)",
            n_pos, train_edges.shape[0], test_edges.shape[0],
        )

        # ─── Torch tensors (on device) ──────────────────────────────────
        X = torch.tensor(features, dtype=torch.float32, device=device)
        train_edges_t = torch.tensor(train_edges.T, dtype=torch.long, device=device)
        test_edges_t = torch.tensor(test_edges.T, dtype=torch.long, device=device)

        self._encoder = GCNEncoder(
            in_dim=in_dim,
            hidden_dim=self._hidden_dim,
            out_dim=self._out_dim,
            dropout=self._dropout,
            l2_normalize=self._l2,
            device=device,
        )

        optimizer = torch.optim.Adam(
            self._encoder.parameters(),
            lr=self._lr,
            weight_decay=self._wd,
        )

        # ─── Training loop ──────────────────────────────────────────────
        self._encoder.train()
        self._training_history = []
        t_start = time.time()

        n_train = train_edges.shape[0]
        n_neg = int(round(self._neg_ratio * n_train))

        for epoch in range(self._n_epochs):
            optimizer.zero_grad()

            Z = self._encoder(X, norm_adj)

            # Positive link scores (train edges only)
            pos_src = Z[train_edges_t[0]]
            pos_dst = Z[train_edges_t[1]]
            pos_logits = (pos_src * pos_dst).sum(dim=1)

            # Sample negatives freshly every epoch
            neg_edges = _sample_negative_edges(
                n_nodes=n_nodes,
                n_samples=n_neg,
                existing_edges_set=existing_set,
                rng=rng,
            )
            if neg_edges.size == 0:
                neg_logits = torch.zeros(0, device=device)
            else:
                neg_edges_t = torch.tensor(neg_edges.T, dtype=torch.long, device=device)
                neg_src = Z[neg_edges_t[0]]
                neg_dst = Z[neg_edges_t[1]]
                neg_logits = (neg_src * neg_dst).sum(dim=1)

            labels = torch.cat([
                torch.ones(pos_logits.shape[0], device=device),
                torch.zeros(neg_logits.shape[0], device=device),
            ])
            logits = torch.cat([pos_logits, neg_logits])
            loss = F.binary_cross_entropy_with_logits(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._encoder.parameters(), max_norm=5.0)
            optimizer.step()

            if epoch % max(self._n_epochs // 10, 1) == 0 or epoch == self._n_epochs - 1:
                with torch.no_grad():
                    pos_prob = torch.sigmoid(pos_logits).mean().item()
                    neg_prob = (
                        torch.sigmoid(neg_logits).mean().item()
                        if neg_logits.numel() > 0
                        else 0.0
                    )
                    gap = pos_prob - neg_prob

                    # ── Compute train AUC ────────────────────────────
                    train_scores = torch.sigmoid(logits).cpu().numpy()
                    train_labels = labels.cpu().numpy()
                    try:
                        train_auc = float(roc_auc_score(train_labels, train_scores))
                    except ValueError:
                        train_auc = float("nan")

                    # ── Compute test AUC ─────────────────────────────
                    test_pos_src = Z[test_edges_t[0]]
                    test_pos_dst = Z[test_edges_t[1]]
                    test_pos_logits = (test_pos_src * test_pos_dst).sum(dim=1)
                    n_test_neg = max(test_edges.shape[0], 1)
                    test_neg = _sample_negative_edges(
                        n_nodes=n_nodes,
                        n_samples=n_test_neg,
                        existing_edges_set=existing_set,
                        rng=rng,
                    )
                    if test_neg.size == 0:
                        test_auc = float("nan")
                    else:
                        test_neg_t = torch.tensor(
                            test_neg.T, dtype=torch.long, device=device,
                        )
                        test_neg_logits = (
                            Z[test_neg_t[0]] * Z[test_neg_t[1]]
                        ).sum(dim=1)
                        test_all_logits = torch.cat([test_pos_logits, test_neg_logits])
                        test_all_labels = torch.cat([
                            torch.ones(test_pos_logits.shape[0], device=device),
                            torch.zeros(test_neg_logits.shape[0], device=device),
                        ])
                        test_scores = torch.sigmoid(test_all_logits).cpu().numpy()
                        test_lbl = test_all_labels.cpu().numpy()
                        try:
                            test_auc = float(roc_auc_score(test_lbl, test_scores))
                        except ValueError:
                            test_auc = float("nan")

                self._training_history.append({
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "pos_prob_mean": float(pos_prob),
                    "neg_prob_mean": float(neg_prob),
                    "gap": float(gap),
                    "train_auc": train_auc,
                    "test_auc": test_auc,
                })
                logger.info(
                    "  epoch %3d  loss=%.4f  pos=%.3f  neg=%.3f  gap=%.3f  "
                    "train_auc=%.3f  test_auc=%.3f",
                    epoch, loss.item(), pos_prob, neg_prob, gap,
                    train_auc, test_auc,
                )

        # ─── Extract final embedding (back to CPU for numpy) ─────────────
        self._encoder.eval()
        with torch.no_grad():
            Z_final = self._encoder(X, norm_adj).cpu().numpy()

        col_names = [f"{self.name}_{i}" for i in range(Z_final.shape[1])]
        self._embedding = pd.DataFrame(
            Z_final, index=dataset.X.index, columns=col_names, dtype=np.float64
        )
        self._schema_version = dataset.schema.version

        last_hist = self._training_history[-1] if self._training_history else {}
        self._config.update(
            {
                "n_nodes": n_nodes,
                "in_dim": in_dim,
                "out_dim": Z_final.shape[1],
                "n_positive_edges": n_pos,
                "n_train_edges": train_edges.shape[0],
                "n_test_edges": test_edges.shape[0],
                "device": str(device),
                "training_time_seconds": float(time.time() - t_start),
                "final_loss": last_hist.get("loss"),
                "final_gap": last_hist.get("gap"),
                "final_train_auc": last_hist.get("train_auc"),
                "final_test_auc": last_hist.get("test_auc"),
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
            n_isolated_nodes=0,
            schema_version=self._schema_version,
        )

    # ─── Feature resolution ────────────────────────────────────────────

    def _resolve_features(self, dataset: HarmonizedDataset) -> np.ndarray:
        """Return the ``(N, d_in)`` feature matrix used as GCN node features."""
        if self._feature_source == "composite":
            # Prefer the cached Stage B composite embedding if available.
            # This is the cleanest input: dense, L2-normalized, 56-dim.
            return self._load_stage_b_composite(dataset)
        if self._feature_source == "raw":
            # Normalized Stage A matrix with NaN→0 fallback.
            from face_stratification.harmonization.normalization import (
                fit_normalization, transform_normalization,
            )
            stats = fit_normalization(dataset.X, dataset.schema)
            Xn = transform_normalization(dataset.X, stats)
            arr = Xn.to_numpy(dtype=np.float32)
            arr = np.where(np.isfinite(arr), arr, 0.0)
            return arr
        raise ValueError(f"Unknown feature_source: {self._feature_source!r}")

    def _load_stage_b_composite(self, dataset: HarmonizedDataset) -> np.ndarray:
        """Load the cached Stage B composite embedding from disk."""
        from pathlib import Path
        from face_stratification.models.base import PatientEmbedding

        repo = Path(__file__).resolve().parents[3]
        cache = repo / "output" / "stratification" / "stage_b_review" / "embedding_cache"
        if (cache / "embedding.parquet").is_file():
            emb = PatientEmbedding.load(cache)
            # Align to the dataset index
            aligned = emb.values.reindex(dataset.X.index)
            return aligned.to_numpy(dtype=np.float32)

        logger.warning(
            "No cached Stage B embedding found at %s; using identity features (one-hot)",
            cache,
        )
        n = dataset.n_patients
        return np.eye(n, dtype=np.float32)
