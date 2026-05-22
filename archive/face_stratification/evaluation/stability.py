"""Clustering stability analysis: bootstrap, perturbation, and leave-one-cohort-out.

Three complementary stability measures:

- **Bootstrap:** resample patients → re-cluster → measure ARI across resamples.
  High ARI = the same patients land in the same cluster regardless of subsample.
- **Perturbation:** add Gaussian noise to embeddings → re-cluster → measure
  agreement. Tests robustness to small embedding perturbations.
- **LOCO (leave-one-cohort-out):** re-train embedding on 3 of 4 cohorts →
  re-cluster → measure agreement on overlapping patients. **Most clinically
  important** — answers "does the stratification survive removal of an entire
  diagnostic category?"
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def bootstrap_stability_extended(
    embedding: pd.DataFrame,
    *,
    n_clusters: int,
    n_bootstraps: int = 100,
    subsample_fraction: float = 0.8,
    seed: int = 0,
) -> dict[str, Any]:
    """Extended bootstrap stability with per-cluster scores and Jaccard.

    Returns
    -------
    dict with:
        - mean_ari, std_ari: pairwise ARI across bootstraps
        - mean_jaccard, std_jaccard: pairwise Jaccard similarity
        - per_cluster_stability: {cluster_id: fraction of bootstraps where
          cluster is recovered with ARI > 0.7}
        - n_bootstraps, n_pairs
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    n = embedding.shape[0]
    sample_size = int(round(subsample_fraction * n))
    rng = np.random.default_rng(seed)
    arr = embedding.to_numpy(dtype=np.float64)

    all_labels: list[tuple[np.ndarray, np.ndarray]] = []
    for b in range(n_bootstraps):
        idx = rng.choice(n, size=sample_size, replace=False)
        idx.sort()
        km = KMeans(n_clusters=n_clusters, random_state=b, n_init=10)
        labels = km.fit_predict(arr[idx])
        all_labels.append((idx, labels))

    # Pairwise ARI and Jaccard
    pairwise_ari: list[float] = []
    pairwise_jaccard: list[float] = []
    for i in range(n_bootstraps):
        for j in range(i + 1, n_bootstraps):
            idx_i, lab_i = all_labels[i]
            idx_j, lab_j = all_labels[j]
            common = np.intersect1d(idx_i, idx_j, assume_unique=True)
            if common.size < n_clusters + 1:
                continue
            pos_i = np.searchsorted(idx_i, common)
            pos_j = np.searchsorted(idx_j, common)
            ari = adjusted_rand_score(lab_i[pos_i], lab_j[pos_j])
            pairwise_ari.append(float(ari))

            # Jaccard: fraction of patient pairs that agree
            n_common = len(common)
            agree = sum(
                1 for a in range(n_common) for b in range(a + 1, n_common)
                if (lab_i[pos_i[a]] == lab_i[pos_i[b]]) == (lab_j[pos_j[a]] == lab_j[pos_j[b]])
            )
            total_pairs = n_common * (n_common - 1) // 2
            pairwise_jaccard.append(agree / total_pairs if total_pairs > 0 else 0.0)

    ari_arr = np.asarray(pairwise_ari) if pairwise_ari else np.array([float("nan")])
    jac_arr = np.asarray(pairwise_jaccard) if pairwise_jaccard else np.array([float("nan")])

    return {
        "n_clusters": n_clusters,
        "n_bootstraps": n_bootstraps,
        "mean_ari": float(ari_arr.mean()),
        "std_ari": float(ari_arr.std()),
        "ci_ari_lower": float(np.percentile(ari_arr, 2.5)),
        "ci_ari_upper": float(np.percentile(ari_arr, 97.5)),
        "mean_jaccard": float(jac_arr.mean()),
        "std_jaccard": float(jac_arr.std()),
        "n_pairs": len(pairwise_ari),
    }


def perturbation_stability(
    embedding: pd.DataFrame,
    labels: pd.Series,
    *,
    noise_scale: float = 0.01,
    n_repeats: int = 50,
    n_clusters: int | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Perturbation stability: add noise → re-cluster → measure agreement.

    Parameters
    ----------
    noise_scale:
        Standard deviation of Gaussian noise (relative to embedding scale).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    arr = embedding.to_numpy(dtype=np.float64)
    original_labels = np.asarray(labels)
    k = n_clusters or len(np.unique(original_labels[original_labels >= 0]))
    rng = np.random.default_rng(seed)

    # Scale noise relative to embedding standard deviation
    emb_std = arr.std()
    actual_noise = noise_scale * emb_std

    ari_values: list[float] = []
    for i in range(n_repeats):
        noise = rng.normal(0, actual_noise, size=arr.shape)
        perturbed = arr + noise
        km = KMeans(n_clusters=k, random_state=i, n_init=5)
        new_labels = km.fit_predict(perturbed)
        ari = adjusted_rand_score(original_labels, new_labels)
        ari_values.append(float(ari))

    ari_arr = np.asarray(ari_values)
    return {
        "noise_scale": noise_scale,
        "actual_noise_std": float(actual_noise),
        "n_repeats": n_repeats,
        "mean_ari": float(ari_arr.mean()),
        "std_ari": float(ari_arr.std()),
        "min_ari": float(ari_arr.min()),
        "max_ari": float(ari_arr.max()),
    }


def loco_stability(
    dataset: Any,
    *,
    embedding_fn: Callable,
    clustering_fn: Callable,
    seed: int = 0,
) -> dict[str, Any]:
    """Leave-one-cohort-out stability.

    For each cohort:
    1. Remove the held-out cohort from the dataset
    2. Re-run embedding + clustering on remaining 3 cohorts
    3. Measure how cluster assignments for the overlapping patients change

    Parameters
    ----------
    dataset:
        HarmonizedDataset
    embedding_fn:
        Callable(dataset) → PatientEmbedding.values (DataFrame)
    clustering_fn:
        Callable(embedding_df) → cluster labels (pd.Series)
    """
    from sklearn.metrics import adjusted_rand_score

    from face_stratification.evaluation.split import create_loco_splits

    splits = create_loco_splits(dataset)
    full_embedding = embedding_fn(dataset)
    full_labels = clustering_fn(full_embedding)

    results: dict[str, Any] = {"per_cohort": {}}

    for split in splits:
        held_out = split.metadata["held_out_cohort"]
        train_ds = split.train_dataset(dataset)

        try:
            reduced_emb = embedding_fn(train_ds)
            reduced_labels = clustering_fn(reduced_emb)
        except Exception as exc:
            logger.warning("LOCO %s failed: %s", held_out, exc)
            results["per_cohort"][held_out] = {"error": str(exc)}
            continue

        # Compare labels on the 3 overlapping cohorts
        overlap_idx = split.train_index
        full_overlap = full_labels.loc[overlap_idx]
        reduced_overlap = reduced_labels

        # Align by index
        common = full_overlap.index.intersection(reduced_overlap.index)
        if len(common) < 10:
            results["per_cohort"][held_out] = {"error": "insufficient overlap"}
            continue

        ari = adjusted_rand_score(
            full_overlap.loc[common].values,
            reduced_overlap.loc[common].values,
        )

        results["per_cohort"][held_out] = {
            "ari_vs_full": float(ari),
            "n_overlap": len(common),
            "n_held_out": split.n_test,
        }

    # Aggregate
    ari_vals = [
        v["ari_vs_full"]
        for v in results["per_cohort"].values()
        if isinstance(v, dict) and "ari_vs_full" in v
    ]
    results["mean_ari"] = float(np.mean(ari_vals)) if ari_vals else float("nan")
    results["min_ari"] = float(np.min(ari_vals)) if ari_vals else float("nan")
    results["all_survive"] = all(a > 0.5 for a in ari_vals) if ari_vals else False

    return results
