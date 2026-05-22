"""Stage C ablation studies — algorithm × k × view × metric grids.

Three ablations are implemented:

1. :func:`run_algorithm_k_grid` — full grid over (algorithm, k) for a
   single embedding. Reports internal metrics (silhouette, Davies-Bouldin,
   Calinski-Harabasz) and external metrics vs DSM cohort (ARI, NMI,
   V-measure, mean cluster entropy, Cramér's V). Lets the user identify
   the (algorithm, k) combination that maximizes a scalar quality score.

2. :func:`run_embedding_view_ablation` — for a fixed algorithm and k,
   cluster each sub-view of a composite embedding alone and compare against
   the full composite. Answers "which view is carrying the signal?"

3. :func:`compute_optimization_score` — computes the scientifically
   justified weighted score used to pick the best configuration. Defaults
   balance clustering quality (silhouette, lower is worse; Davies-Bouldin,
   lower is better) with transdiagnostic content (mean cohort entropy, the
   whole point of the sub-project).

The results of each ablation are a tidy DataFrame that can be directly
exported to CSV or rendered in a notebook.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from face_stratification.stage_c.algorithms import ALGORITHMS, run_algorithm
from face_stratification.stage_c.comparison import (
    cramers_v,
    per_cluster_cohort_entropy,
)

logger = logging.getLogger(__name__)


# ─── Internal metric computation ──────────────────────────────────────────────


def _internal_metrics(embedding: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Davies-Bouldin + Calinski-Harabasz on the full embedding."""
    try:
        from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
    except ImportError as exc:
        raise ImportError("internal metrics require scikit-learn.") from exc

    valid = labels >= 0
    if valid.sum() < 2 or np.unique(labels[valid]).size < 2:
        return {"davies_bouldin": float("nan"), "calinski_harabasz": float("nan")}
    return {
        "davies_bouldin": float(davies_bouldin_score(embedding[valid], labels[valid])),
        "calinski_harabasz": float(calinski_harabasz_score(embedding[valid], labels[valid])),
    }


def _summarize_row(
    *,
    embedding_name: str,
    algorithm: str,
    k: int,
    random_state: int,
    runtime: float,
    labels: np.ndarray,
    arr: np.ndarray,
    reference_labels: np.ndarray,
) -> dict[str, Any]:
    """Compose one row of an ablation table."""
    from face_stratification.clustering.metrics import compute_cluster_metrics

    ext = compute_cluster_metrics(arr, labels, reference_labels, silhouette_sample_size=5000)
    internal = _internal_metrics(arr, labels)

    # Mean cluster entropy (transdiagnostic content)
    entropy_dict, td_dict = per_cluster_cohort_entropy(labels, reference_labels)
    mean_entropy = float(np.mean(list(entropy_dict.values()))) if entropy_dict else 0.0
    mean_td_score = float(np.mean(list(td_dict.values()))) if td_dict else 0.0

    # Cramér's V
    try:
        cv = cramers_v(labels, reference_labels)
    except Exception:  # noqa: BLE001
        cv = float("nan")

    return {
        "embedding": embedding_name,
        "algorithm": algorithm,
        "k": k,
        "random_state": random_state,
        "runtime_s": runtime,
        "silhouette": ext.silhouette,
        "davies_bouldin": internal["davies_bouldin"],
        "calinski_harabasz": internal["calinski_harabasz"],
        "ari": ext.ari_vs_reference,
        "ami": ext.ami_vs_reference,
        "nmi": ext.nmi_vs_reference,
        "v_measure": ext.v_measure,
        "homogeneity": ext.homogeneity,
        "completeness": ext.completeness,
        "cramers_v": cv,
        "mean_cluster_entropy_bits": mean_entropy,
        "mean_transdiagnostic_score": mean_td_score,
        "n_clusters_actual": ext.n_clusters,
    }


# ─── Algorithm × k grid ──────────────────────────────────────────────────────


def run_algorithm_k_grid(
    embedding: pd.DataFrame,
    *,
    reference_labels: np.ndarray,
    k_values: Iterable[int] = (4, 5, 6, 7, 8, 9, 10),
    algorithms: Iterable[str] = ("kmeans", "gmm", "ward", "spectral"),
    random_states: Iterable[int] = (0,),
    embedding_name: str = "composite",
) -> pd.DataFrame:
    """Full grid over (algorithm, k, random_state) on one embedding.

    For each cell in the grid, we run the algorithm once and record:

    - Internal metrics (silhouette, Davies-Bouldin, Calinski-Harabasz)
    - External metrics against ``reference_labels`` (ARI, NMI, AMI,
      V-measure, homogeneity, completeness, Cramér's V)
    - Transdiagnostic content (mean cluster entropy + mean transdiagnostic
      score)
    - Runtime in seconds

    Returns a long-format DataFrame with one row per configuration.
    """
    arr = embedding.to_numpy(dtype=np.float64)
    rows = []
    for algo in algorithms:
        for k in k_values:
            for seed in random_states:
                t0 = time.time()
                try:
                    assignment = run_algorithm(
                        algo,
                        embedding,
                        n_clusters=k,
                        random_state=seed,
                        reference_labels=reference_labels,
                        silhouette_sample_size=5000,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("%s k=%d seed=%d failed: %s", algo, k, seed, exc)
                    continue
                labels = assignment.labels.to_numpy()
                runtime = time.time() - t0
                rows.append(
                    _summarize_row(
                        embedding_name=embedding_name,
                        algorithm=algo,
                        k=k,
                        random_state=seed,
                        runtime=runtime,
                        labels=labels,
                        arr=arr,
                        reference_labels=reference_labels,
                    )
                )
                logger.info(
                    "%s k=%d seed=%d: sil=%.3f db=%.3f ari=%.3f ent=%.3f (%.1fs)",
                    algo,
                    k,
                    seed,
                    rows[-1]["silhouette"],
                    rows[-1]["davies_bouldin"],
                    rows[-1]["ari"],
                    rows[-1]["mean_cluster_entropy_bits"],
                    runtime,
                )
    return pd.DataFrame(rows)


# ─── Embedding view ablation ──────────────────────────────────────────────────


def run_embedding_view_ablation(
    embedding: pd.DataFrame,
    view_dims: dict[str, int],
    *,
    reference_labels: np.ndarray,
    n_clusters: int = 8,
    algorithm: str = "kmeans",
    random_state: int = 0,
) -> pd.DataFrame:
    """Cluster each sub-view of a composite embedding alone and compare.

    The embedding columns are assumed to be ordered such that the first
    ``view_dims[v0]`` columns correspond to view ``v0``, the next
    ``view_dims[v1]`` to ``v1``, etc. (This matches the layout produced
    by :class:`ConcatenatedEmbedding`.)
    """
    rows = []
    cursor = 0

    # First, the full composite as the reference row
    full = _summarize_row(
        embedding_name="composite (full)",
        algorithm=algorithm,
        k=n_clusters,
        random_state=random_state,
        runtime=0.0,
        labels=run_algorithm(
            algorithm,
            embedding,
            n_clusters=n_clusters,
            random_state=random_state,
            reference_labels=reference_labels,
        ).labels.to_numpy(),
        arr=embedding.to_numpy(dtype=np.float64),
        reference_labels=reference_labels,
    )
    rows.append(full)

    for view_name, dim in view_dims.items():
        sub_cols = embedding.columns[cursor : cursor + dim]
        cursor += dim
        sub = embedding[sub_cols]
        t0 = time.time()
        try:
            assignment = run_algorithm(
                algorithm,
                sub,
                n_clusters=n_clusters,
                random_state=random_state,
                reference_labels=reference_labels,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("view %s failed: %s", view_name, exc)
            continue
        rows.append(
            _summarize_row(
                embedding_name=view_name,
                algorithm=algorithm,
                k=n_clusters,
                random_state=random_state,
                runtime=time.time() - t0,
                labels=assignment.labels.to_numpy(),
                arr=sub.to_numpy(dtype=np.float64),
                reference_labels=reference_labels,
            )
        )
    return pd.DataFrame(rows)


# ─── Best-configuration scoring ───────────────────────────────────────────────


def compute_optimization_score(
    row: pd.Series | dict,
    *,
    w_silhouette: float = 1.0,
    w_davies_bouldin_inv: float = 1.0,
    w_transdiagnostic: float = 1.0,
    w_internal_agreement: float = 0.5,
) -> float:
    """Weighted composite score for ranking ablation configurations.

    The score combines four normalized components:

    - ``silhouette`` (in ``[-1, 1]``; higher is better)
    - ``1 / (1 + davies_bouldin)`` (maps DB in ``[0, ∞)`` to ``(0, 1]``,
      higher is better)
    - ``mean_transdiagnostic_score`` (in ``[0, 1]``; higher is better
      — more transdiagnostic clusters)
    - ``1 - ari_vs_cohort`` as a "non-DSM-redundancy" bonus — rewards
      clusters that do NOT simply re-label the DSM cohort (weighted lower
      because the primary criterion is cluster quality).

    Default weights give roughly equal weight to internal quality and
    transdiagnostic content, with a small bonus for non-DSM-redundancy.
    """
    sil = float(row.get("silhouette", 0.0)) if hasattr(row, "get") else 0.0
    db = float(row.get("davies_bouldin", np.inf)) if hasattr(row, "get") else np.inf
    td = float(row.get("mean_transdiagnostic_score", 0.0)) if hasattr(row, "get") else 0.0
    ari = float(row.get("ari", 0.0)) if hasattr(row, "get") else 0.0

    db_component = 1.0 / (1.0 + max(db, 0.0))
    internal_agreement = 1.0 - abs(ari)

    return (
        w_silhouette * sil
        + w_davies_bouldin_inv * db_component
        + w_transdiagnostic * td
        + w_internal_agreement * internal_agreement
    )


def pick_best_configuration(
    grid: pd.DataFrame,
    **score_kwargs: float,
) -> dict[str, Any]:
    """Add an ``optimization_score`` column and return the best row as dict."""
    scored = grid.copy()
    scored["optimization_score"] = scored.apply(
        lambda r: compute_optimization_score(r, **score_kwargs), axis=1
    )
    best = scored.sort_values("optimization_score", ascending=False).iloc[0]
    return best.to_dict()
