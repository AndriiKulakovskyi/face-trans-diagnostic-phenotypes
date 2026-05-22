"""Generate a t-SNE projection of the Stage B composite embedding.

Produces three new figures under ``output/stratification/stage_b_review/figures/``:

- ``06_projection_tsne.png``           — t-SNE scatter colored by cohort + cluster
- ``07_projection_comparison.png``     — 2×2 grid: PCA vs t-SNE, each × cohort + cluster
- ``08_projection_tsne_cluster_only.png`` — single t-SNE panel colored by cluster
  (the most legible view for cluster interpretation)

Also caches:

- ``output/stratification/stage_b_review/embedding.parquet``  — the 56-dim composite
  embedding, so re-running the projection doesn't need to rebuild it.
- ``output/stratification/stage_b_review/projection_tsne.npy`` — t-SNE coordinates.

Run:

    python scripts/run_tsne_projection.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from face_stratification import (
    ConcatenatedEmbedding,
    PatientEmbedding,
    build_harmonized_dataset,
    fit_embedding,
)
from face_stratification.analysis.visualization import (
    plot_embedding_projection,
    tsne_project,
    umap_project,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tsne_projection")


CSV_PATHS = {
    "bp": REPO / "data" / "BP.csv",
    "sz": REPO / "data" / "SZ.csv",
    "dr": REPO / "data" / "DR.csv",
    "asp": REPO / "data" / "ASP.csv",
}

OUT_DIR = REPO / "output" / "stratification" / "stage_b_review"
FIG_DIR = OUT_DIR / "figures"
EMBED_DIR = OUT_DIR / "embedding_cache"
EMBED_DIR.mkdir(parents=True, exist_ok=True)


def _load_or_build_embedding() -> tuple[PatientEmbedding, pd.DataFrame]:
    """Load a cached ``PatientEmbedding`` if present; otherwise rebuild and cache."""
    if (EMBED_DIR / "embedding.parquet").is_file():
        logger.info("Loading cached embedding from %s", EMBED_DIR)
        emb = PatientEmbedding.load(EMBED_DIR)
        # We still need the metadata (for cohort labels) — rebuild the harmonized
        # dataset but only to grab the metadata table (cheap, ~7 s).
        ds = build_harmonized_dataset(csv_paths=CSV_PATHS)
        return emb, ds.metadata

    logger.info("No cached embedding found — rebuilding end-to-end")
    t0 = time.time()
    ds = build_harmonized_dataset(csv_paths=CSV_PATHS)
    logger.info("Harmonized %d × %d in %.1fs", ds.n_patients, ds.n_features, time.time() - t0)

    model = ConcatenatedEmbedding.build_default(
        pca_dim=8, td_spectral_dim=16, multiplex_spectral_dim=32
    )
    t1 = time.time()
    emb, _graph = fit_embedding(ds, model=model)
    logger.info("Built embedding in %.1fs", time.time() - t1)

    emb.save(EMBED_DIR)
    logger.info("Cached embedding to %s", EMBED_DIR)
    return emb, ds.metadata


def main() -> None:
    emb, metadata = _load_or_build_embedding()
    labels_path = OUT_DIR / "cluster_labels.parquet"
    if not labels_path.is_file():
        raise FileNotFoundError(
            f"{labels_path} not found — run scripts/run_stage_b_review.py first."
        )
    cluster_labels = pd.read_parquet(labels_path)["cluster"].astype(int)

    # Align metadata + labels to the embedding's index.
    aligned_index = emb.values.index
    cohort_series = metadata.loc[aligned_index, "cohort"].values
    cluster_series = cluster_labels.loc[aligned_index].values

    # Load the previously-saved PCA projection for comparison.
    pca_path = OUT_DIR / "projection_pca.npy"
    if pca_path.is_file():
        logger.info("Loading cached PCA projection from %s", pca_path)
        pca_coords = np.load(pca_path)
        pca_method = "PCA"
    else:
        logger.info("Computing PCA projection")
        pca_coords, pca_method = umap_project(emb.values, random_state=0)
        np.save(pca_path, pca_coords)

    # ─── Compute t-SNE ────────────────────────────────────────────────────
    t_tsne = time.time()
    logger.info(
        "Computing t-SNE (perplexity=30, metric=cosine, init=pca, n_iter=1000) — "
        "this will take 60-120 s on 11 k points"
    )
    tsne_coords, _ = tsne_project(
        emb.values,
        perplexity=30.0,
        metric="cosine",
        init="pca",
        n_iter=1000,
        random_state=0,
    )
    logger.info("t-SNE done in %.1f s", time.time() - t_tsne)
    np.save(OUT_DIR / "projection_tsne.npy", tsne_coords)

    # ─── Figure 06: t-SNE only ─────────────────────────────────────────────
    fig = plot_embedding_projection(
        tsne_coords,
        cohort=cohort_series,
        cluster=cluster_series,
        title="Stage B composite embedding",
        method_used="t-SNE",
        figsize=(12, 5),
    )
    fig.savefig(FIG_DIR / "06_projection_tsne.png", dpi=120)
    plt.close(fig)
    logger.info("saved 06_projection_tsne.png")

    # ─── Figure 07: 2 × 2 comparison of PCA and t-SNE ───────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    cohort_colors = {
        "bp": "#1f77b4",
        "sz": "#d62728",
        "dr": "#2ca02c",
        "asp": "#ff7f0e",
    }
    unique_clusters = sorted(np.unique(cluster_series))
    cmap = plt.get_cmap("tab10")

    # Top row: colored by cohort
    for col_idx, (coords, title) in enumerate(
        [(pca_coords, pca_method), (tsne_coords, "t-SNE")]
    ):
        ax = axes[0, col_idx]
        for c, color in cohort_colors.items():
            mask = cohort_series == c
            if mask.any():
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=4,
                    alpha=0.45,
                    c=color,
                    label=c.upper(),
                )
        ax.set_title(f"{title} — colored by DSM cohort")
        ax.set_xlabel(f"{title} 1")
        ax.set_ylabel(f"{title} 2")
        ax.legend(loc="best", fontsize=8)

    # Bottom row: colored by cluster
    for col_idx, (coords, title) in enumerate(
        [(pca_coords, pca_method), (tsne_coords, "t-SNE")]
    ):
        ax = axes[1, col_idx]
        for i, c in enumerate(unique_clusters):
            mask = cluster_series == c
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=4,
                alpha=0.45,
                c=[cmap(i % 10)],
                label=f"cluster {c}",
            )
        ax.set_title(f"{title} — colored by cluster")
        ax.set_xlabel(f"{title} 1")
        ax.set_ylabel(f"{title} 2")
        ax.legend(loc="best", fontsize=7, ncol=2)

    fig.suptitle(
        "PCA vs t-SNE projection of the 56-dim Stage B composite embedding "
        "(top: DSM cohort; bottom: learned cluster)",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_projection_comparison.png", dpi=120)
    plt.close(fig)
    logger.info("saved 07_projection_comparison.png")

    # ─── Figure 08: t-SNE cluster only (large, legible) ───────────────────
    fig, ax = plt.subplots(figsize=(9, 8))
    for i, c in enumerate(unique_clusters):
        mask = cluster_series == c
        ax.scatter(
            tsne_coords[mask, 0],
            tsne_coords[mask, 1],
            s=8,
            alpha=0.55,
            c=[cmap(i % 10)],
            label=f"cluster {c}  (n = {int(mask.sum())})",
        )
    ax.set_title("t-SNE of Stage B composite embedding — 8 discovered clusters")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_projection_tsne_cluster_only.png", dpi=120)
    plt.close(fig)
    logger.info("saved 08_projection_tsne_cluster_only.png")


if __name__ == "__main__":
    main()
