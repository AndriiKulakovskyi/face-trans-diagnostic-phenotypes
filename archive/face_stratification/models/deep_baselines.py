"""Feature-based deep learning baselines: Vanilla AE and VAE.

These do NOT use the graph — they are pure feature-based deep embeddings.
Critical for a fair comparison: if GNN methods cannot beat these, the
graph is not contributing meaningful signal.

Both use PyTorch with MPS support for Apple Silicon.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from face_stratification.harmonization.harmonizer import HarmonizedDataset
from face_stratification.models.base import BaseEmbeddingModel, PatientEmbedding
from face_stratification.stage_b2.gcn import get_device

logger = logging.getLogger(__name__)


# ─── Encoder / Decoder architectures ────────────────────────────────────────


class _Encoder(nn.Module):
    """Symmetric encoder: input → hidden layers → bottleneck."""

    def __init__(self, in_dim: int, hidden_dims: list[int], bottleneck_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            prev = h
        layers.append(nn.Linear(prev, bottleneck_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Decoder(nn.Module):
    """Symmetric decoder: bottleneck → hidden layers (reversed) → output."""

    def __init__(self, bottleneck_dim: int, hidden_dims: list[int], out_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = bottleneck_dim
        for h in reversed(hidden_dims):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _VAEEncoder(nn.Module):
    """VAE encoder: outputs (mu, log_sigma) for the reparameterization trick."""

    def __init__(self, in_dim: int, hidden_dims: list[int], bottleneck_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            prev = h
        self.shared = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev, bottleneck_dim)
        self.fc_logvar = nn.Linear(prev, bottleneck_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.fc_mu(h), self.fc_logvar(h)


# ─── Training loop ──────────────────────────────────────────────────────────


def _prepare_data(
    dataset: HarmonizedDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[DataLoader, torch.Tensor, np.ndarray]:
    """Fill NaN with 0, build a DataLoader, and return the full tensor."""
    X = dataset.X.fillna(0.0).to_numpy(dtype=np.float32)
    # Also create a missingness mask as additional input channels
    mask = (~dataset.X.isna()).to_numpy(dtype=np.float32)
    X_aug = np.concatenate([X, mask], axis=1)

    tensor = torch.from_numpy(X_aug).to(device)
    ds = TensorDataset(tensor)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    return loader, tensor, X


# ─── Vanilla Autoencoder ────────────────────────────────────────────────────


class VanillaAE(BaseEmbeddingModel):
    """Standard autoencoder on normalized features.

    NaN values are filled with 0 and a binary missingness mask is
    concatenated as additional input channels, doubling the input
    dimension. The bottleneck produces the patient embedding.

    Parameters
    ----------
    bottleneck_dim:
        Embedding dimension.
    hidden_dims:
        Hidden layer sizes (encoder mirrors decoder).
    n_epochs:
        Training epochs.
    lr:
        Learning rate.
    weight_decay:
        L2 regularization.
    batch_size:
        Mini-batch size.
    """

    name = "vanilla_ae"

    def __init__(
        self,
        *,
        bottleneck_dim: int = 32,
        hidden_dims: list[int] | None = None,
        n_epochs: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        l2_normalize: bool = True,
    ) -> None:
        super().__init__(
            bottleneck_dim=bottleneck_dim,
            hidden_dims=hidden_dims or [128, 64],
            n_epochs=n_epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            l2_normalize=l2_normalize,
        )
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dims = hidden_dims or [128, 64]
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.l2_normalize = l2_normalize
        self._embedding: PatientEmbedding | None = None

    def fit(
        self,
        dataset: HarmonizedDataset,
        *,
        graph: Any | None = None,
    ) -> VanillaAE:
        device = get_device()
        loader, X_full, _ = _prepare_data(dataset, self.batch_size, device)
        in_dim = X_full.shape[1]  # features + mask

        encoder = _Encoder(in_dim, self.hidden_dims, self.bottleneck_dim).to(device)
        # Decoder reconstructs only the original features, not the mask
        out_dim = in_dim // 2
        decoder = _Decoder(self.bottleneck_dim, self.hidden_dims, out_dim).to(device)

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        encoder.train()
        decoder.train()
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            for (batch,) in loader:
                z = encoder(batch)
                x_hat = decoder(z)
                # MSE only on the feature half (not the mask)
                x_orig = batch[:, :out_dim]
                loss = F.mse_loss(x_hat, x_orig)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.size(0)
            if (epoch + 1) % 50 == 0:
                logger.debug("AE epoch %d/%d loss=%.4f", epoch + 1, self.n_epochs,
                             total_loss / len(X_full))

        # Extract embeddings
        encoder.eval()
        with torch.no_grad():
            Z = encoder(X_full).cpu().numpy()

        if self.l2_normalize:
            Z = self._l2_normalize_rows(Z)

        columns = [f"ae_{i}" for i in range(Z.shape[1])]
        self._embedding = PatientEmbedding(
            values=pd.DataFrame(Z, index=dataset.X.index, columns=columns),
            model_name=self.name,
            model_config=self.config,
            view_dims={self.name: Z.shape[1]},
            schema_version=getattr(dataset.schema, "version", "unknown"),
        )
        self._fitted = True
        logger.info("VanillaAE: %d patients → %d dims", Z.shape[0], Z.shape[1])
        return self

    def transform(self) -> PatientEmbedding:
        self._ensure_fitted()
        assert self._embedding is not None
        return self._embedding


# ─── Variational Autoencoder ────────────────────────────────────────────────


class VAE(BaseEmbeddingModel):
    """Variational autoencoder with KL regularization (beta-VAE).

    The KL term produces a smoother, more regularized latent space than
    the vanilla AE. With heavy missingness, the regularization helps
    prevent overfitting to observed-feature patterns.

    Parameters
    ----------
    bottleneck_dim:
        Latent dimension.
    beta:
        KL divergence weight. ``beta=1.0`` is the standard VAE;
        ``beta>1`` encourages disentanglement.
    """

    name = "vae"

    def __init__(
        self,
        *,
        bottleneck_dim: int = 32,
        hidden_dims: list[int] | None = None,
        n_epochs: int = 200,
        lr: float = 1e-3,
        beta: float = 1.0,
        batch_size: int = 256,
        l2_normalize: bool = True,
    ) -> None:
        super().__init__(
            bottleneck_dim=bottleneck_dim,
            hidden_dims=hidden_dims or [128, 64],
            n_epochs=n_epochs,
            lr=lr,
            beta=beta,
            batch_size=batch_size,
            l2_normalize=l2_normalize,
        )
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dims = hidden_dims or [128, 64]
        self.n_epochs = n_epochs
        self.lr = lr
        self.beta = beta
        self.batch_size = batch_size
        self.l2_normalize = l2_normalize
        self._embedding: PatientEmbedding | None = None

    def fit(
        self,
        dataset: HarmonizedDataset,
        *,
        graph: Any | None = None,
    ) -> VAE:
        device = get_device()
        loader, X_full, _ = _prepare_data(dataset, self.batch_size, device)
        in_dim = X_full.shape[1]
        out_dim = in_dim // 2

        encoder = _VAEEncoder(in_dim, self.hidden_dims, self.bottleneck_dim).to(device)
        decoder = _Decoder(self.bottleneck_dim, self.hidden_dims, out_dim).to(device)

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=self.lr,
        )

        encoder.train()
        decoder.train()
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            for (batch,) in loader:
                mu, logvar = encoder(batch)
                # Reparameterization trick
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std

                x_hat = decoder(z)
                x_orig = batch[:, :out_dim]

                # Reconstruction + KL
                recon_loss = F.mse_loss(x_hat, x_orig, reduction="sum")
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + self.beta * kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 50 == 0:
                logger.debug("VAE epoch %d/%d loss=%.4f", epoch + 1, self.n_epochs,
                             total_loss / len(X_full))

        # Extract embeddings using the mean (no sampling)
        encoder.eval()
        with torch.no_grad():
            mu, _ = encoder(X_full)
            Z = mu.cpu().numpy()

        if self.l2_normalize:
            Z = self._l2_normalize_rows(Z)

        columns = [f"vae_{i}" for i in range(Z.shape[1])]
        self._embedding = PatientEmbedding(
            values=pd.DataFrame(Z, index=dataset.X.index, columns=columns),
            model_name=self.name,
            model_config=self.config,
            view_dims={self.name: Z.shape[1]},
            schema_version=getattr(dataset.schema, "version", "unknown"),
        )
        self._fitted = True
        logger.info("VAE: %d patients → %d dims (beta=%.2f)", Z.shape[0], Z.shape[1], self.beta)
        return self

    def transform(self) -> PatientEmbedding:
        self._ensure_fitted()
        assert self._embedding is not None
        return self._embedding
