"""Hyperparameter sweep infrastructure for Stage B2.5.

Trains many GCN configurations with cheap evaluation metrics to find the
architecture that best improves the **transdiagnostic signal** of the
Stage C stratification, rather than the DSM alignment.

Design
------
For each :class:`SweepConfig`:

1. Train a :class:`StageB2GraphContrastive` (or GAE) on the Stage A
   multiplex graph with the specified hyperparameters.
2. Combine the resulting GNN embedding with the Stage B composite by
   concatenation + row-wise L2 normalization.
3. Run a quick k-means at a small k grid (5, 6, 7, 8) on the combined
   representation.
4. For each k, compute:

   - **Internal**: silhouette, Davies-Bouldin, Calinski-Harabasz
   - **DSM alignment**: ARI / NMI / Cramér's V against the cohort labels
   - **Transdiagnostic content**: mean cluster Shannon entropy over
     cohort, normalized to ``[0, 1]`` by dividing by ``log₂(n_cohorts)``

5. Pick the best k per config by the **transdiagnostic optimization
   score**

   .. math::

       s = w_{\text{sil}}\,\text{silhouette}
         + w_{\text{db}}\,\frac{1}{1+\text{DB}}
         + w_{\text{trans}}\,\frac{H}{\log_2 n_c}
         + w_{\text{non\_dsm}}\,(1 - V)

   with defaults :math:`w = (1, 1, 2, 1)` so the transdiagnostic entropy
   term is weighted twice the others. This explicitly rewards
   configurations that produce sharper clusters *without* gaining DSM
   alignment.

6. Return a tidy :class:`SweepResult` table containing one row per
   ``(config, k)`` pair with all metrics + the optimization score.

Run a sweep via :func:`run_sweep` from
``scripts/run_stage_b2_sweep.py``. Each config takes ~1.5 min on CPU
for 50 training epochs, so a 12-config primary sweep finishes in ~20 min.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch

from face_stratification.harmonization.harmonizer import HarmonizedDataset
from face_stratification.models.base import PatientEmbedding
from face_stratification.stage_b2.contrastive import StageB2GraphContrastive
from face_stratification.stage_b2.gae import StageB2GAE

logger = logging.getLogger(__name__)


# ─── Configuration + result types ────────────────────────────────────────────


@dataclass
class SweepConfig:
    """One point in the Stage B2.5 hyperparameter grid.

    ``model`` selects which Stage B2 model to train. Each field below has
    a sensible default matching the Stage B2 production configuration.
    """

    model: str = "contrastive"  # or "gae"
    n_layers: int = 2
    hidden_dim: int = 64
    out_dim: int = 32
    n_epochs: int = 50
    learning_rate: float = 5e-3
    weight_decay: float = 5e-4
    dropout: float = 0.1
    # Contrastive-specific
    temperature: float = 0.5
    p_edge: float = 0.2
    p_feat: float = 0.1
    # Graph filtering
    include_edge_types: tuple[str, ...] | None = None
    exclude_edge_types: tuple[str, ...] = ()
    # Bookkeeping
    name: str = ""
    seed: int = 0

    def config_id(self) -> str:
        """Short human-readable id for this config, used in result tables."""
        if self.name:
            return self.name
        parts = [
            f"m={self.model}",
            f"L={self.n_layers}",
            f"h={self.hidden_dim}",
            f"d={self.out_dim}",
        ]
        if self.model == "contrastive":
            parts.append(f"T={self.temperature}")
            parts.append(f"pE={self.p_edge}")
            parts.append(f"pF={self.p_feat}")
        if self.include_edge_types is not None:
            parts.append(f"inc={','.join(self.include_edge_types)}")
        if self.exclude_edge_types:
            parts.append(f"exc={','.join(self.exclude_edge_types)}")
        return "|".join(parts)


@dataclass
class SweepResult:
    """One row per (config, k) from a sweep run."""

    config: SweepConfig
    k: int
    n_clusters_actual: int
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    ari_vs_cohort: float
    nmi_vs_cohort: float
    cramers_v: float
    mean_cluster_entropy_bits: float
    max_possible_entropy_bits: float
    transdiagnostic_score: float  # mean_cluster_entropy / max_possible
    dsm_score: float              # Cramér's V (higher = more DSM-aligned)
    optimization_score: float
    training_time_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config.config_id(),
            "model": self.config.model,
            "n_layers": self.config.n_layers,
            "hidden_dim": self.config.hidden_dim,
            "out_dim": self.config.out_dim,
            "temperature": self.config.temperature,
            "p_edge": self.config.p_edge,
            "include_edge_types": (
                ",".join(self.config.include_edge_types)
                if self.config.include_edge_types
                else "all"
            ),
            "k": self.k,
            "n_clusters_actual": self.n_clusters_actual,
            "silhouette": self.silhouette,
            "davies_bouldin": self.davies_bouldin,
            "calinski_harabasz": self.calinski_harabasz,
            "ari_vs_cohort": self.ari_vs_cohort,
            "nmi_vs_cohort": self.nmi_vs_cohort,
            "cramers_v": self.cramers_v,
            "mean_cluster_entropy_bits": self.mean_cluster_entropy_bits,
            "max_possible_entropy_bits": self.max_possible_entropy_bits,
            "transdiagnostic_score": self.transdiagnostic_score,
            "dsm_score": self.dsm_score,
            "optimization_score": self.optimization_score,
            "training_time_s": self.training_time_s,
        }


# ─── Optimization score ──────────────────────────────────────────────────────


def compute_transdiagnostic_score(
    silhouette: float,
    davies_bouldin: float,
    transdiagnostic_score: float,
    cramers_v: float,
    *,
    w_silhouette: float = 1.0,
    w_davies_bouldin_inv: float = 1.0,
    w_transdiagnostic: float = 2.0,
    w_non_dsm: float = 1.0,
) -> float:
    """Transdiagnostic-weighted optimization score.

    Range: unbounded above, but typical values are in ``[0, 5]``. Higher
    is better. The transdiagnostic term is doubled by default so this
    score explicitly rewards configurations whose clusters mix DSM
    cohorts, unlike the standard Stage C score which treats DSM
    redundancy as an equal sibling of cluster quality.
    """
    sil = float(silhouette) if np.isfinite(silhouette) else 0.0
    db = float(davies_bouldin) if np.isfinite(davies_bouldin) else np.inf
    db_component = 1.0 / (1.0 + max(db, 0.0))
    trans = float(transdiagnostic_score) if np.isfinite(transdiagnostic_score) else 0.0
    cv = float(cramers_v) if np.isfinite(cramers_v) else 0.0
    non_dsm = 1.0 - cv

    return (
        w_silhouette * sil
        + w_davies_bouldin_inv * db_component
        + w_transdiagnostic * trans
        + w_non_dsm * non_dsm
    )


# ─── Per-config evaluation ──────────────────────────────────────────────────


def _combine_with_stage_b(
    stage_b_emb: PatientEmbedding,
    gnn_values: pd.DataFrame,
) -> np.ndarray:
    """Concat Stage B composite with GNN embedding + row-L2-normalize."""
    stage_b_arr = stage_b_emb.values.loc[gnn_values.index].to_numpy(dtype=np.float64)
    gnn_arr = gnn_values.to_numpy(dtype=np.float64)
    combined = np.concatenate([stage_b_arr, gnn_arr], axis=1)
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return combined / norms


def _evaluate_one_k(
    combined: np.ndarray,
    cohort_labels: np.ndarray,
    *,
    k: int,
    random_state: int = 0,
) -> dict[str, float]:
    """Run k-means at one k value and compute all metrics."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        adjusted_rand_score,
        calinski_harabasz_score,
        davies_bouldin_score,
        normalized_mutual_info_score,
        silhouette_score,
    )

    from face_stratification.stage_c.comparison import (
        cramers_v,
        per_cluster_cohort_entropy,
    )

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(combined)
    n_actual = int(np.unique(labels).size)

    # Subsample silhouette to keep it tractable on 11 k points
    if combined.shape[0] > 5000:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(combined.shape[0], 5000, replace=False)
        sil = float(silhouette_score(combined[idx], labels[idx], metric="cosine"))
    else:
        sil = float(silhouette_score(combined, labels, metric="cosine"))

    db = float(davies_bouldin_score(combined, labels))
    ch = float(calinski_harabasz_score(combined, labels))
    ari = float(adjusted_rand_score(cohort_labels, labels))
    nmi = float(normalized_mutual_info_score(cohort_labels, labels))

    try:
        cv = float(cramers_v(labels, cohort_labels))
    except Exception:  # noqa: BLE001
        cv = float("nan")

    entropy_dict, td_dict = per_cluster_cohort_entropy(labels, cohort_labels)
    mean_entropy = float(np.mean(list(entropy_dict.values()))) if entropy_dict else 0.0
    mean_td = float(np.mean(list(td_dict.values()))) if td_dict else 0.0
    max_entropy = float(np.log2(pd.Series(cohort_labels).nunique()))

    return {
        "n_clusters_actual": n_actual,
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "ari_vs_cohort": ari,
        "nmi_vs_cohort": nmi,
        "cramers_v": cv,
        "mean_cluster_entropy_bits": mean_entropy,
        "max_possible_entropy_bits": max_entropy,
        "transdiagnostic_score": mean_td,
    }


def evaluate_config(
    dataset: HarmonizedDataset,
    graph: Any,
    stage_b_emb: PatientEmbedding,
    cohort_labels: np.ndarray,
    config: SweepConfig,
    *,
    k_grid: tuple[int, ...] = (5, 6, 7, 8),
) -> list[SweepResult]:
    """Train one config and evaluate it across ``k_grid``.

    Returns one :class:`SweepResult` per k value.
    """
    t0 = time.time()
    torch.manual_seed(config.seed)

    if config.model == "contrastive":
        model = StageB2GraphContrastive(
            hidden_dim=config.hidden_dim,
            out_dim=config.out_dim,
            n_layers=config.n_layers,
            n_epochs=config.n_epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            dropout=config.dropout,
            seed=config.seed,
            feature_source="composite",
            temperature=config.temperature,
            p_edge=config.p_edge,
            p_feat=config.p_feat,
            include_edge_types=config.include_edge_types,
            exclude_edge_types=config.exclude_edge_types,
        )
    elif config.model == "gae":
        model = StageB2GAE(
            hidden_dim=config.hidden_dim,
            out_dim=config.out_dim,
            n_layers=config.n_layers,
            n_epochs=config.n_epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            dropout=config.dropout,
            seed=config.seed,
            feature_source="composite",
            include_edge_types=config.include_edge_types,
            exclude_edge_types=config.exclude_edge_types,
        )
    else:
        raise ValueError(f"Unknown model: {config.model!r}")

    model.fit(dataset, graph=graph)
    gnn_emb = model.transform()
    training_time = float(time.time() - t0)

    combined = _combine_with_stage_b(stage_b_emb, gnn_emb.values)

    results: list[SweepResult] = []
    for k in k_grid:
        metrics = _evaluate_one_k(combined, cohort_labels, k=k)
        score = compute_transdiagnostic_score(
            silhouette=metrics["silhouette"],
            davies_bouldin=metrics["davies_bouldin"],
            transdiagnostic_score=metrics["transdiagnostic_score"],
            cramers_v=metrics["cramers_v"],
        )
        dsm_score = metrics["cramers_v"]
        results.append(SweepResult(
            config=config,
            k=k,
            n_clusters_actual=metrics["n_clusters_actual"],
            silhouette=metrics["silhouette"],
            davies_bouldin=metrics["davies_bouldin"],
            calinski_harabasz=metrics["calinski_harabasz"],
            ari_vs_cohort=metrics["ari_vs_cohort"],
            nmi_vs_cohort=metrics["nmi_vs_cohort"],
            cramers_v=metrics["cramers_v"],
            mean_cluster_entropy_bits=metrics["mean_cluster_entropy_bits"],
            max_possible_entropy_bits=metrics["max_possible_entropy_bits"],
            transdiagnostic_score=metrics["transdiagnostic_score"],
            dsm_score=dsm_score,
            optimization_score=score,
            training_time_s=training_time,
        ))
    return results


# ─── Full sweep driver ──────────────────────────────────────────────────────


def run_sweep(
    dataset: HarmonizedDataset,
    graph: Any,
    stage_b_emb: PatientEmbedding,
    cohort_labels: np.ndarray,
    configs: list[SweepConfig],
    *,
    k_grid: tuple[int, ...] = (5, 6, 7, 8),
) -> pd.DataFrame:
    """Run a list of :class:`SweepConfig` and return a tidy DataFrame.

    Each row corresponds to one (config, k) pair with every metric and
    the optimization score.
    """
    rows: list[dict[str, Any]] = []
    for i, config in enumerate(configs):
        logger.info(
            "Sweep %d/%d: %s", i + 1, len(configs), config.config_id()
        )
        try:
            per_k_results = evaluate_config(
                dataset, graph, stage_b_emb, cohort_labels, config, k_grid=k_grid,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Config %s failed: %s", config.config_id(), exc)
            continue
        for r in per_k_results:
            rows.append(r.as_dict())
            logger.info(
                "  k=%d  sil=%.3f  db=%.3f  ari=%.3f  td_score=%.3f  "
                "CramérV=%.3f  opt=%.3f",
                r.k, r.silhouette, r.davies_bouldin, r.ari_vs_cohort,
                r.transdiagnostic_score, r.cramers_v, r.optimization_score,
            )
    return pd.DataFrame(rows)


def pick_best_transdiagnostic_config(
    sweep_df: pd.DataFrame,
    *,
    score_column: str = "optimization_score",
) -> dict[str, Any]:
    """Return the row with the highest optimization score."""
    sorted_df = sweep_df.sort_values(score_column, ascending=False)
    return sorted_df.iloc[0].to_dict()
