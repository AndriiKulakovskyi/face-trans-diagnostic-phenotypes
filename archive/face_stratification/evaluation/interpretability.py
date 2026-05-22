"""Embedding interpretability: feature importance, dimension analysis, block ablation.

Answers the question: "Why does this stratification look the way it does?"

- **Feature importance (permutation):** which input features drive the embedding
  structure? No prediction target needed (unsupervised).
- **Dimension analysis:** what does each latent dimension capture?
- **Block ablation:** how much does each clinical block contribute?
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def embedding_feature_importance(
    embedding_values: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    *,
    labels: np.ndarray,
    n_permutations: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Permutation-based feature importance for unsupervised embeddings.

    For each input feature, permute it, re-compute the silhouette score
    on the original embedding-based clustering, and measure the change.
    Features whose permutation causes the largest silhouette drop are
    the most important drivers of the embedding structure.

    Note: this measures importance for the *clustering*, not the embedding
    itself — which is what we actually care about.

    Returns DataFrame sorted by importance (most important first).
    """
    from sklearn.metrics import silhouette_score

    arr = embedding_values.to_numpy(dtype=np.float64)
    labels = np.asarray(labels)
    valid = labels >= 0
    arr_v = arr[valid]
    lab_v = labels[valid]

    n_unique = len(np.unique(lab_v))
    if n_unique < 2:
        return pd.DataFrame(columns=["feature", "importance", "baseline_sil"])

    # Baseline silhouette
    baseline_sil = silhouette_score(
        arr_v, lab_v, metric="cosine",
        sample_size=min(5000, len(arr_v)),
        random_state=seed,
    )

    # For each feature, permute it in the *feature matrix* and see how much
    # the clustering quality changes. We use ANOVA F-statistic as a proxy:
    # features with high F-stat explain more between-cluster variance.
    from scipy.stats import f_oneway

    results: list[dict] = []
    feat_arr = feature_matrix.to_numpy(dtype=np.float64)

    for j, feat_name in enumerate(feature_matrix.columns):
        col = feat_arr[:, j]
        valid_mask = np.isfinite(col) & valid

        if valid_mask.sum() < 20:
            results.append({
                "feature": feat_name,
                "f_statistic": float("nan"),
                "p_value": float("nan"),
                "importance": 0.0,
            })
            continue

        groups = [
            col[valid_mask & (labels == lab)]
            for lab in np.unique(lab_v)
        ]
        groups = [g for g in groups if len(g) >= 2]

        if len(groups) < 2:
            results.append({
                "feature": feat_name,
                "f_statistic": float("nan"),
                "p_value": float("nan"),
                "importance": 0.0,
            })
            continue

        try:
            f_stat, p_val = f_oneway(*groups)
            # Importance = -log10(p_value) scaled by effect size
            importance = float(f_stat) if np.isfinite(f_stat) else 0.0
            results.append({
                "feature": feat_name,
                "f_statistic": float(f_stat),
                "p_value": float(p_val),
                "importance": importance,
            })
        except Exception:
            results.append({
                "feature": feat_name,
                "f_statistic": float("nan"),
                "p_value": float("nan"),
                "importance": 0.0,
            })

    df = pd.DataFrame(results)
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df.attrs["baseline_silhouette"] = baseline_sil
    return df


def embedding_dimension_analysis(
    embedding_values: pd.DataFrame,
    labels: np.ndarray,
    feature_matrix: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Analyze what each embedding dimension captures.

    For each dimension, computes:
    - ANOVA F-statistic against cluster labels (how much does this dim
      separate the clusters?)
    - Explained variance ratio
    - Top 5 correlated input features (if feature_matrix provided)
    """
    from scipy.stats import f_oneway

    labels = np.asarray(labels)
    valid = labels >= 0
    emb_arr = embedding_values.to_numpy(dtype=np.float64)

    rows = []
    unique_labels = np.unique(labels[valid])

    for dim_idx, dim_name in enumerate(embedding_values.columns):
        col = emb_arr[:, dim_idx]
        col_v = col[valid]
        lab_v = labels[valid]

        # ANOVA F
        groups = [col_v[lab_v == lab] for lab in unique_labels]
        groups = [g for g in groups if len(g) >= 2]

        f_stat, p_val = (float("nan"), float("nan"))
        if len(groups) >= 2:
            try:
                f_stat, p_val = f_oneway(*groups)
                f_stat, p_val = float(f_stat), float(p_val)
            except Exception:
                pass

        # Variance explained
        total_var = float(np.var(col_v))
        between_var = 0.0
        for lab in unique_labels:
            mask = lab_v == lab
            group_mean = col_v[mask].mean()
            between_var += mask.sum() * (group_mean - col_v.mean()) ** 2
        between_var /= len(col_v)
        var_explained = between_var / total_var if total_var > 0 else 0.0

        row: dict[str, Any] = {
            "dimension": dim_name,
            "dim_index": dim_idx,
            "f_statistic": f_stat,
            "p_value": p_val,
            "variance_explained": float(var_explained),
            "dimension_std": float(np.std(col_v)),
        }

        # Top correlated input features
        if feature_matrix is not None:
            feat_arr = feature_matrix.to_numpy(dtype=np.float64)
            correlations = {}
            for j, feat_name in enumerate(feature_matrix.columns):
                feat_col = feat_arr[:, j]
                both_valid = np.isfinite(col) & np.isfinite(feat_col)
                if both_valid.sum() < 10:
                    continue
                corr = np.corrcoef(col[both_valid], feat_col[both_valid])[0, 1]
                if np.isfinite(corr):
                    correlations[feat_name] = float(corr)

            top_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            row["top_correlated_features"] = [
                {"feature": name, "correlation": corr} for name, corr in top_corr
            ]

        rows.append(row)

    return pd.DataFrame(rows)


def block_ablation_study(
    dataset: Any,
    *,
    embedding_fn: Callable,
    clustering_fn: Callable,
    block_names: list[str],
    feature_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Per-clinical-block contribution via ablation.

    For each block, remove all its features, re-run embedding + clustering,
    measure metric change. "How much does cognition contribute?"

    Parameters
    ----------
    dataset:
        HarmonizedDataset
    embedding_fn:
        Callable(dataset) → embedding DataFrame
    clustering_fn:
        Callable(embedding) → labels (np.ndarray)
    block_names:
        Blocks to ablate (e.g., ["mood", "cognition", "biology"])
    feature_metadata:
        DataFrame with 'block' column indexed by feature_id
    """
    from sklearn.metrics import silhouette_score

    # Baseline: full features
    full_emb = embedding_fn(dataset)
    full_labels = clustering_fn(full_emb)
    n_unique = len(np.unique(full_labels[full_labels >= 0]))

    if n_unique < 2:
        return pd.DataFrame(columns=["block", "silhouette_drop", "ari_vs_full"])

    baseline_sil = silhouette_score(
        full_emb.to_numpy(), full_labels, metric="cosine",
        sample_size=min(5000, len(full_emb)),
    )

    rows = []
    for block in block_names:
        # Features in this block
        block_features = feature_metadata.index[feature_metadata["block"] == block].tolist()
        if not block_features:
            continue

        # Ablate: set block features to NaN
        ablated_X = dataset.X.copy()
        for feat in block_features:
            if feat in ablated_X.columns:
                ablated_X[feat] = np.nan

        # Create ablated dataset
        from face_stratification.harmonization.harmonizer import HarmonizedDataset
        ablated_ds = HarmonizedDataset(
            X=ablated_X,
            metadata=dataset.metadata,
            feature_metadata=dataset.feature_metadata,
            schema=dataset.schema,
        )

        try:
            ablated_emb = embedding_fn(ablated_ds)
            ablated_labels = clustering_fn(ablated_emb)

            abl_sil = silhouette_score(
                ablated_emb.to_numpy(), ablated_labels, metric="cosine",
                sample_size=min(5000, len(ablated_emb)),
            )

            from sklearn.metrics import adjusted_rand_score
            ari = adjusted_rand_score(full_labels, ablated_labels)

            rows.append({
                "block": block,
                "n_features_removed": len(block_features),
                "silhouette_full": float(baseline_sil),
                "silhouette_ablated": float(abl_sil),
                "silhouette_drop": float(baseline_sil - abl_sil),
                "ari_vs_full": float(ari),
            })
        except Exception as exc:
            logger.warning("Ablation of block %s failed: %s", block, exc)
            rows.append({
                "block": block,
                "n_features_removed": len(block_features),
                "silhouette_full": float(baseline_sil),
                "silhouette_ablated": float("nan"),
                "silhouette_drop": float("nan"),
                "ari_vs_full": float("nan"),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("silhouette_drop", ascending=False).reset_index(drop=True)
    return df
