"""Abstract base class + output type for Stage B embedding models.

Stage B produces patient embeddings from the Stage A harmonized dataset
and multi-relational graph. All models share a narrow common interface:

    model.fit(dataset, graph=..., ...)
    model.transform()   ->  PatientEmbedding

Concrete implementations in this first PR are baselines that do not use
torch — they run on numpy / scipy / sklearn and produce deterministic
results that Stage C can cluster directly. A later PR will add GNN models
(R-GCN, GraphSAGE, contrastive SSL) under the same interface.

The output :class:`PatientEmbedding` is the stable contract that every
downstream stage consumes. It is index-aligned with the harmonized matrix
(``MultiIndex[cohort, patient_id]``) and carries enough metadata to be
re-joinable with the Stage A diagnosis labels for comparison.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .harmonized_dataset import HarmonizedDataset

logger = logging.getLogger(__name__)


# ─── Output type ──────────────────────────────────────────────────────────────


@dataclass
class PatientEmbedding:
    """A trained Stage B embedding ready to be consumed by Stage C clustering.

    Attributes
    ----------
    values:
        ``(N, d)`` float DataFrame with the same ``MultiIndex[cohort,
        patient_id]`` as the harmonized matrix. Every row is one patient's
        learned embedding.
    model_name:
        Short machine id of the model that produced the embedding
        (``"transdiagnostic_pca"``, ``"concat_v1"``, ...).
    model_config:
        JSON-serializable dict of the model's hyperparameters. Used by run-id
        hashing and reproducibility audits.
    view_dims:
        For composite embeddings, a map from sub-view name to the number of
        columns contributed. For single-view models it is
        ``{model_name: d}``.
    n_isolated_nodes:
        Number of patients that ended up with zero edges at fit time (and
        therefore a degenerate — typically zero — embedding in graph-based
        views). Always ``0`` for PCA-style feature-only models.
    schema_version:
        Version of the Stage A feature schema that produced the dataset.
    """

    values: pd.DataFrame
    model_name: str
    model_config: dict[str, Any] = field(default_factory=dict)
    view_dims: dict[str, int] = field(default_factory=dict)
    n_isolated_nodes: int = 0
    schema_version: str = "unknown"

    def __post_init__(self) -> None:
        if not isinstance(self.values, pd.DataFrame):
            raise TypeError("PatientEmbedding.values must be a pandas DataFrame")
        if self.values.index.nlevels != 2 or tuple(self.values.index.names) != (
            "cohort",
            "patient_id",
        ):
            raise ValueError(
                "PatientEmbedding.values must be indexed by MultiIndex['cohort', 'patient_id']"
            )
        if not self.values.index.is_unique:
            raise ValueError("PatientEmbedding has duplicate (cohort, patient_id) rows")
        if np.isnan(self.values.to_numpy()).any():
            raise ValueError(
                "PatientEmbedding contains NaN — embeddings must be dense. "
                "Models should replace degenerate rows with zero vectors explicitly."
            )
        if not np.isfinite(self.values.to_numpy()).all():
            raise ValueError("PatientEmbedding contains non-finite values")

    @property
    def n_patients(self) -> int:
        return len(self.values)

    @property
    def dim(self) -> int:
        return self.values.shape[1]

    # ─── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        """Save the embedding to a ``.parquet`` plus a sidecar ``.json`` manifest.

        ``path`` should be a directory; two files are written:

            {path}/embedding.parquet
            {path}/manifest.json
        """
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.values.to_parquet(out / "embedding.parquet")
        manifest = {
            "model_name": self.model_name,
            "model_config": self.model_config,
            "view_dims": self.view_dims,
            "n_isolated_nodes": self.n_isolated_nodes,
            "schema_version": self.schema_version,
            "n_patients": self.n_patients,
            "dim": self.dim,
        }
        with open(out / "manifest.json", "w") as fh:
            json.dump(manifest, fh, indent=2, default=str)
        return out

    @classmethod
    def load(cls, path: str | Path) -> PatientEmbedding:
        """Re-load a previously saved embedding."""
        src = Path(path)
        values = pd.read_parquet(src / "embedding.parquet")
        with open(src / "manifest.json") as fh:
            manifest = json.load(fh)
        return cls(
            values=values,
            model_name=manifest["model_name"],
            model_config=manifest["model_config"],
            view_dims=manifest.get("view_dims", {}),
            n_isolated_nodes=int(manifest.get("n_isolated_nodes", 0)),
            schema_version=manifest.get("schema_version", "unknown"),
        )


# ─── Abstract interface ──────────────────────────────────────────────────────


class BaseEmbeddingModel(ABC):
    """Narrow interface every Stage B model implements.

    Models are stateful: ``fit`` learns parameters from the harmonized
    dataset (and optionally the multiplex graph), ``transform`` produces the
    :class:`PatientEmbedding`. Single-call convenience is provided by
    :meth:`fit_transform`.

    The interface deliberately does not expose torch — it works for any
    numpy-based implementation. A future torch-geometric subclass can
    inherit this and add its own ``.encode()`` hook.
    """

    name: str = "base"

    def __init__(self, **kwargs: Any) -> None:
        self._fitted: bool = False
        self._config: dict[str, Any] = dict(kwargs)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def config(self) -> dict[str, Any]:
        return dict(self._config)

    @abstractmethod
    def fit(
        self,
        dataset: HarmonizedDataset,
        *,
        graph: Any | None = None,
    ) -> BaseEmbeddingModel:
        """Fit the model on the given harmonized dataset (+ optional graph)."""
        ...

    @abstractmethod
    def transform(self) -> PatientEmbedding:
        """Return the learned :class:`PatientEmbedding`."""
        ...

    def fit_transform(
        self,
        dataset: HarmonizedDataset,
        *,
        graph: Any | None = None,
    ) -> PatientEmbedding:
        self.fit(dataset, graph=graph)
        return self.transform()

    # ─── Utilities shared across models ──────────────────────────────────

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} has not been fit yet")

    @staticmethod
    def _l2_normalize_rows(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Row-wise L2 normalization, safe for zero rows."""
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms > eps, norms, 1.0)
        return mat / norms
