"""Matplotlib helpers for Stage B review visualizations.

All helpers take data-level arguments, create a ``matplotlib.figure.Figure``,
and return it so callers can either ``plt.show()`` or save it to disk.
UMAP is imported lazily — if it's not installed, the 2D projection falls
back to a plain PCA.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


_COHORT_COLORS = {
    "bp": "#1f77b4",   # blue
    "sz": "#d62728",   # red
    "dr": "#2ca02c",   # green
    "asp": "#ff7f0e",  # orange
}


def umap_project(
    embedding: np.ndarray | pd.DataFrame,
    *,
    n_neighbors: int = 30,
    min_dist: float = 0.2,
    metric: str = "cosine",
    random_state: int = 0,
) -> tuple[np.ndarray, str]:
    """Project a high-dim embedding into 2D.

    Uses UMAP if installed, otherwise PCA. Returns ``(coords, method_used)``
    where ``coords`` is an ``(N, 2)`` numpy array.
    """
    arr = (
        embedding.to_numpy(dtype=np.float64)
        if isinstance(embedding, pd.DataFrame)
        else np.asarray(embedding, dtype=np.float64)
    )
    try:
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        coords = reducer.fit_transform(arr)
        return coords, "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=random_state).fit_transform(arr)
        logger.info("UMAP not installed; falling back to PCA for 2D projection.")
        return coords, "PCA"


def tsne_project(
    embedding: np.ndarray | pd.DataFrame,
    *,
    perplexity: float = 30.0,
    learning_rate: str | float = "auto",
    n_iter: int = 1000,
    metric: str = "cosine",
    random_state: int = 0,
    init: str = "pca",
) -> tuple[np.ndarray, str]:
    """Project a high-dim embedding into 2D with t-SNE.

    t-SNE is a non-linear manifold-learning technique particularly well
    suited for visualizing high-dim clusters because it preserves local
    neighbourhood structure. For patient embeddings on the unit sphere
    (cosine-compatible), ``metric="cosine"`` is the right choice.

    Parameters
    ----------
    embedding:
        ``(N, d)`` numpy array or DataFrame.
    perplexity:
        The "effective number of neighbours" t-SNE tries to preserve
        around each point. 5-50 is the typical range; 30 is a good
        default for large (N > 5000) embeddings.
    learning_rate:
        ``"auto"`` (scikit-learn 1.2+) picks a sensible default for the
        sample size. Passing a numeric override is only needed for
        debugging.
    n_iter:
        Number of gradient-descent steps. 1000 is the default and
        converges reliably on well-behaved embeddings.
    metric:
        Distance metric. ``"cosine"`` matches our L2-normalized
        composite embedding.
    random_state:
        Seed for reproducibility.
    init:
        Initial embedding. ``"pca"`` gives stable, reproducible layouts
        and is almost always preferred over the default random init.

    Returns
    -------
    ``(coords, "t-SNE")`` where ``coords`` is ``(N, 2)``.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError("tsne_project requires scikit-learn") from exc

    arr = (
        embedding.to_numpy(dtype=np.float64)
        if isinstance(embedding, pd.DataFrame)
        else np.asarray(embedding, dtype=np.float64)
    )
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=n_iter,
        metric=metric,
        init=init,
        random_state=random_state,
        verbose=0,
    )
    coords = reducer.fit_transform(arr)
    return coords, "t-SNE"


def plot_embedding_projection(
    coords: np.ndarray,
    *,
    cohort: pd.Series | np.ndarray | None = None,
    cluster: pd.Series | np.ndarray | None = None,
    title: str = "Stage B embedding projection",
    method_used: str = "UMAP",
    figsize: tuple[float, float] = (12, 5),
):
    """Side-by-side 2D scatter plots colored by cohort and by cluster.

    Passes the same coordinates through twice: the left panel colors by
    the DSM cohort (reference / ground truth), the right panel colors by
    the learned cluster label. Visual comparison of the two panels gives
    an immediate sense of how much the clustering aligns with DSM.
    """
    import matplotlib.pyplot as plt

    n_panels = int(cohort is not None) + int(cluster is not None)
    if n_panels == 0:
        raise ValueError("Provide at least `cohort` or `cluster`.")

    fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    panel = 0

    if cohort is not None:
        ax = axes[panel]
        coh = np.asarray(cohort)
        for c, color in _COHORT_COLORS.items():
            mask = coh == c
            if mask.any():
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    s=6, alpha=0.5, c=color, label=c.upper(),
                )
        ax.set_title(f"{title}\ncolored by DSM cohort")
        ax.set_xlabel(f"{method_used} 1")
        ax.set_ylabel(f"{method_used} 2")
        ax.legend(loc="best", fontsize=8)
        panel += 1

    if cluster is not None:
        ax = axes[panel]
        clu = np.asarray(cluster)
        unique_clusters = sorted(np.unique(clu))
        cmap = plt.get_cmap("tab10")
        for i, c in enumerate(unique_clusters):
            mask = clu == c
            color = cmap(i % 10)
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                s=6, alpha=0.5, c=[color], label=f"cluster {c}",
            )
        ax.set_title(f"{title}\ncolored by cluster")
        ax.set_xlabel(f"{method_used} 1")
        ax.set_ylabel(f"{method_used} 2")
        ax.legend(loc="best", fontsize=8, ncol=2)

    fig.tight_layout()
    return fig


def plot_cluster_cohort_heatmap(
    cluster_labels: pd.Series | np.ndarray,
    cohort_labels: pd.Series | np.ndarray,
    *,
    normalize: str = "index",  # "index" (rows) / "columns" / "all"
    title: str = "Cluster × cohort",
    figsize: tuple[float, float] = (8, 5),
    cmap: str = "Blues",
):
    """Heatmap of the cluster × cohort contingency table.

    ``normalize="index"`` normalizes each row to 1 — i.e. reads as
    "what fraction of each cluster is cohort X". Use ``"columns"`` to
    read as "what fraction of each cohort is in cluster Y".
    """
    import matplotlib.pyplot as plt

    ct = pd.crosstab(
        pd.Series(cluster_labels, name="cluster"),
        pd.Series(cohort_labels, name="cohort"),
        normalize=normalize,
    )
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(ct.values, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(ct.columns)))
    ax.set_xticklabels(ct.columns)
    ax.set_yticks(np.arange(len(ct.index)))
    ax.set_yticklabels([f"C{c}" for c in ct.index])
    ax.set_xlabel("cohort")
    ax.set_ylabel("cluster")
    ax.set_title(title)

    # Annotate cells
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            val = ct.values[i, j]
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                color="white" if val > 0.5 else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    return fig, ct


def plot_enrichment_bars(
    enrichment_table: pd.DataFrame,
    *,
    top_n: int = 10,
    figsize: tuple[float, float] = (11, 8),
    title: str = "Top enriched features per cluster",
):
    """Horizontal bar chart of the top-N enriched features per cluster.

    Expects the DataFrame returned by
    :class:`FeatureEnrichmentResult.table` — must contain columns
    ``cluster``, ``feature_id``, ``effect_rank_biserial``, and
    ``significant``.
    """
    import matplotlib.pyplot as plt

    sig = enrichment_table[enrichment_table["significant"]].copy()
    if sig.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5, "No significant features at q < threshold",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig

    clusters = sorted(sig["cluster"].unique())
    n = len(clusters)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, cluster in enumerate(clusters):
        ax = axes[i]
        sub = (
            sig[sig["cluster"] == cluster]
            .sort_values("abs_effect", ascending=False)
            .head(top_n)
            .sort_values("effect_rank_biserial")
        )
        colors = [
            "#d62728" if e > 0 else "#1f77b4"
            for e in sub["effect_rank_biserial"]
        ]
        ax.barh(sub["feature_id"], sub["effect_rank_biserial"], color=colors)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_title(f"Cluster {cluster}")
        ax.set_xlabel("effect (rank-biserial)")
        ax.tick_params(axis="y", labelsize=7)

    for j in range(len(clusters), len(axes)):
        axes[j].set_axis_off()

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_kmeans_sweep(
    sweep_df: pd.DataFrame,
    *,
    title: str = "k-means sweep",
    figsize: tuple[float, float] = (11, 4),
):
    """Two-panel plot: silhouette and ARI vs k."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    if "silhouette" in sweep_df.columns:
        ax1.plot(sweep_df["k"], sweep_df["silhouette"], "o-", color="#1f77b4")
        ax1.set_xlabel("k")
        ax1.set_ylabel("silhouette (cosine)")
        ax1.set_title("Silhouette vs k")
        ax1.grid(alpha=0.3)

    for col, color, label in [
        ("ari", "#d62728", "ARI"),
        ("nmi", "#2ca02c", "NMI"),
        ("v_measure", "#ff7f0e", "V"),
    ]:
        if col in sweep_df.columns:
            ax2.plot(sweep_df["k"], sweep_df[col], "o-", color=color, label=label)
    ax2.set_xlabel("k")
    ax2.set_ylabel("metric vs DSM cohort")
    ax2.set_title("Cluster vs cohort agreement")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    return fig
