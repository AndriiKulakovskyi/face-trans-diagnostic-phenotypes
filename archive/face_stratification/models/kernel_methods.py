"""Non-linear embedding baselines: Kernel PCA and Diffusion Maps.

These complement the linear PCA baseline with non-linear structure capture:

- **KernelPCA** operates on features (no graph), using an RBF kernel to
  capture non-linear feature interactions that linear PCA misses.
- **DiffusionMap** operates on the multiplex graph adjacency, capturing
  multi-scale community structure via the diffusion operator.

Both implement :class:`BaseEmbeddingModel` and produce :class:`PatientEmbedding`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from face_stratification.harmonization.harmonizer import HarmonizedDataset
from face_stratification.models.base import BaseEmbeddingModel, PatientEmbedding

logger = logging.getLogger(__name__)


class KernelPCAEmbedding(BaseEmbeddingModel):
    """Kernel PCA on transdiagnostic features with RBF kernel.

    This is the key non-graph, non-linear baseline. Captures non-linear
    feature interactions that linear PCA misses, without requiring a graph.

    Parameters
    ----------
    n_components:
        Number of embedding dimensions.
    kernel:
        Kernel type: ``"rbf"``, ``"poly"``, ``"cosine"``.
    gamma:
        RBF kernel bandwidth. ``"auto"`` uses ``1 / (n_features * X.var())``.
    """

    name = "kernel_pca"

    def __init__(
        self,
        *,
        n_components: int = 16,
        kernel: str = "rbf",
        gamma: float | str | None = "auto",
        l2_normalize: bool = True,
    ) -> None:
        super().__init__(
            n_components=n_components,
            kernel=kernel,
            gamma=gamma,
            l2_normalize=l2_normalize,
        )
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.l2_normalize = l2_normalize
        self._embedding: PatientEmbedding | None = None

    def fit(
        self,
        dataset: HarmonizedDataset,
        *,
        graph: Any | None = None,
    ) -> KernelPCAEmbedding:
        from sklearn.decomposition import KernelPCA

        X = dataset.X.copy()
        # Fill NaN with 0 for transdiagnostic features (rare after selection)
        X = X.fillna(0.0)
        arr = X.to_numpy(dtype=np.float64)

        gamma = None
        if self.gamma == "auto":
            var = np.var(arr)
            gamma = 1.0 / (arr.shape[1] * var) if var > 0 else 1.0
        elif isinstance(self.gamma, (int, float)):
            gamma = float(self.gamma)

        n_comp = min(self.n_components, arr.shape[0] - 1, arr.shape[1])

        kpca = KernelPCA(
            n_components=n_comp,
            kernel=self.kernel,
            gamma=gamma,
            random_state=0,
        )
        Z = kpca.fit_transform(arr)

        if self.l2_normalize:
            Z = self._l2_normalize_rows(Z)

        columns = [f"kpca_{i}" for i in range(Z.shape[1])]
        self._embedding = PatientEmbedding(
            values=pd.DataFrame(Z, index=dataset.X.index, columns=columns),
            model_name=self.name,
            model_config=self.config,
            view_dims={self.name: Z.shape[1]},
            schema_version=getattr(dataset.schema, "version", "unknown"),
        )
        self._fitted = True
        logger.info(
            "KernelPCA: %d patients → %d dims (kernel=%s)",
            Z.shape[0],
            Z.shape[1],
            self.kernel,
        )
        return self

    def transform(self) -> PatientEmbedding:
        self._ensure_fitted()
        assert self._embedding is not None
        return self._embedding


class DiffusionMapEmbedding(BaseEmbeddingModel):
    """Diffusion maps on the multiplex graph adjacency.

    Captures multi-scale community structure by building a diffusion
    operator (row-normalized adjacency raised to power t) and extracting
    the top eigenvectors weighted by eigenvalues.

    Parameters
    ----------
    n_components:
        Number of embedding dimensions (excluding the trivial constant eigenvector).
    diffusion_time:
        Power ``t`` of the diffusion operator. ``t=1`` is a single-step
        random walk; higher values capture coarser structure.
    alpha:
        Coifman-Lafon density normalization parameter (0 to 1). ``alpha=0``
        is the classical graph Laplacian; ``alpha=1`` is the Laplace-Beltrami
        operator; ``alpha=0.5`` is the recommended default.
    """

    name = "diffusion_map"

    def __init__(
        self,
        *,
        n_components: int = 16,
        diffusion_time: int = 2,
        alpha: float = 0.5,
        l2_normalize: bool = True,
    ) -> None:
        super().__init__(
            n_components=n_components,
            diffusion_time=diffusion_time,
            alpha=alpha,
            l2_normalize=l2_normalize,
        )
        self.n_components = n_components
        self.diffusion_time = diffusion_time
        self.alpha = alpha
        self.l2_normalize = l2_normalize
        self._embedding: PatientEmbedding | None = None

    def fit(
        self,
        dataset: HarmonizedDataset,
        *,
        graph: Any | None = None,
    ) -> DiffusionMapEmbedding:
        if graph is None:
            raise ValueError("DiffusionMapEmbedding requires a graph (NetworkX MultiGraph)")

        from face_stratification.stage_b2.gcn import build_multiplex_adjacency_from_nx

        n = dataset.X.shape[0]
        A = build_multiplex_adjacency_from_nx(graph, n_nodes=n, combine="sum")

        # Coifman-Lafon density normalization
        if self.alpha > 0:
            A = self._density_normalize(A, self.alpha)

        # Build row-stochastic transition matrix
        P = self._row_normalize(A)

        # Raise to diffusion power t
        if self.diffusion_time > 1:
            P_t = P
            for _ in range(self.diffusion_time - 1):
                P_t = P_t @ P
        else:
            P_t = P

        # Eigen-decomposition of P^t (sparse)
        k = min(self.n_components + 1, n - 2)
        from scipy.sparse.linalg import eigsh

        # P is not symmetric, but P_sym = D^{1/2} P D^{-1/2} is
        # Use the symmetric form for stable eigendecomposition
        deg = np.asarray(A.sum(axis=1)).ravel()
        deg = np.where(deg > 0, deg, 1.0)
        D_sqrt = sp.diags(np.sqrt(deg))
        D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg))
        P_sym = D_sqrt @ P_t @ D_inv_sqrt

        # Ensure symmetric (numerical precision)
        P_sym = 0.5 * (P_sym + P_sym.T)

        try:
            eigenvalues, eigenvectors = eigsh(P_sym.tocsc(), k=k, which="LM")
        except Exception:
            logger.warning("Sparse eigsh failed; falling back to dense eigendecomposition")
            eigenvalues, eigenvectors = np.linalg.eigh(P_sym.toarray())
            idx = np.argsort(-eigenvalues)[:k]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

        # Sort by descending eigenvalue, skip the trivial constant eigenvector
        order = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Transform back to diffusion coordinates: psi_i = lambda_i^t * phi_i
        # Skip first (trivial) eigenvector
        n_use = min(self.n_components, len(eigenvalues) - 1)
        lam = eigenvalues[1 : n_use + 1]
        phi = eigenvectors[:, 1 : n_use + 1]

        # Diffusion map coordinates
        Z = phi * (np.abs(lam) ** self.diffusion_time)[np.newaxis, :]

        # Transform back from symmetric representation
        Z = D_inv_sqrt.toarray() @ Z if sp.issparse(D_inv_sqrt) else D_inv_sqrt @ Z

        if self.l2_normalize:
            Z = self._l2_normalize_rows(Z)

        # Replace any NaN/Inf from isolated nodes with zeros
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

        columns = [f"diffmap_{i}" for i in range(Z.shape[1])]
        self._embedding = PatientEmbedding(
            values=pd.DataFrame(Z, index=dataset.X.index, columns=columns),
            model_name=self.name,
            model_config=self.config,
            view_dims={self.name: Z.shape[1]},
            schema_version=getattr(dataset.schema, "version", "unknown"),
        )
        self._fitted = True
        logger.info(
            "DiffusionMap: %d patients → %d dims (t=%d, alpha=%.2f)",
            Z.shape[0],
            Z.shape[1],
            self.diffusion_time,
            self.alpha,
        )
        return self

    def transform(self) -> PatientEmbedding:
        self._ensure_fitted()
        assert self._embedding is not None
        return self._embedding

    @staticmethod
    def _density_normalize(A: sp.csr_matrix, alpha: float) -> sp.csr_matrix:
        """Coifman-Lafon density normalization: K_alpha = D^{-alpha} K D^{-alpha}."""
        deg = np.asarray(A.sum(axis=1)).ravel()
        deg = np.where(deg > 0, deg, 1.0)
        D_neg_alpha = sp.diags(deg ** (-alpha))
        return D_neg_alpha @ A @ D_neg_alpha

    @staticmethod
    def _row_normalize(A: sp.csr_matrix) -> sp.csr_matrix:
        """Row-stochastic normalization: P = D^{-1} A."""
        deg = np.asarray(A.sum(axis=1)).ravel()
        deg = np.where(deg > 0, deg, 1.0)
        D_inv = sp.diags(1.0 / deg)
        return D_inv @ A
