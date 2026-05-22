"""End-to-end Stage C pipeline driver for the FACE V1 cohort.

Reuses the cached Stage B embedding when present
(``output/stratification/stage_b_review/embedding_cache/``) — otherwise
rebuilds it from scratch (~2 min).

Produces under ``output/stratification/stage_c/``:

- ``algorithm_k_grid.csv``                 — full ablation grid
- ``view_ablation.csv``                    — embedding view ablation
- ``base_clusterings.parquet``             — aligned base clusterings (N × B)
- ``algorithm_pairwise_ari.csv``           — agreement between base clusterings
- ``consensus_labels.parquet``             — final consensus partition
- ``per_patient_confidence.parquet``       — confidence scores
- ``dsm_comparison.json``                  — chi², Cramér's V, ARI/NMI/V
- ``cluster_enrichment_top.csv``           — significant features per cluster
- ``cluster_cards/cluster_NN.md``          — narrative cards
- ``stage_c_summary.json``                 — top-level summary
- ``figures/*.png``                        — t-SNE + heatmap + confidence + ablation

Run:

    python scripts/run_stage_c.py
"""

from __future__ import annotations

import json
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
    plot_cluster_cohort_heatmap,
    plot_embedding_projection,
    tsne_project,
)
from face_stratification.stage_c import (
    run_stage_c,
    write_cluster_cards,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("stage_c")


CSV_PATHS = {
    "bp": REPO / "data" / "BP.csv",
    "sz": REPO / "data" / "SZ.csv",
    "dr": REPO / "data" / "DR.csv",
    "asp": REPO / "data" / "ASP.csv",
}

OUT_DIR = REPO / "output" / "stratification" / "stage_c"
FIG_DIR = OUT_DIR / "figures"
CARD_DIR = OUT_DIR / "cluster_cards"
EMBED_CACHE = REPO / "output" / "stratification" / "stage_b_review" / "embedding_cache"
TSNE_CACHE = REPO / "output" / "stratification" / "stage_b_review" / "projection_tsne.npy"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _load_or_build_embedding() -> tuple[PatientEmbedding, "HarmonizedDataset"]:  # noqa: F821
    if (EMBED_CACHE / "embedding.parquet").is_file():
        logger.info("Loading cached embedding from %s", EMBED_CACHE)
        emb = PatientEmbedding.load(EMBED_CACHE)
        logger.info("Re-loading harmonized dataset for metadata + raw matrix...")
        ds = build_harmonized_dataset(csv_paths=CSV_PATHS)
        return emb, ds

    logger.info("No cached embedding — rebuilding")
    ds = build_harmonized_dataset(csv_paths=CSV_PATHS)
    model = ConcatenatedEmbedding.build_default(
        pca_dim=8, td_spectral_dim=16, multiplex_spectral_dim=32
    )
    emb, _ = fit_embedding(ds, model=model)
    emb.save(EMBED_CACHE)
    return emb, ds


def main() -> None:
    t0 = time.time()
    emb, ds = _load_or_build_embedding()
    logger.info("Embedding ready: %d × %d", emb.n_patients, emb.dim)

    # ─── Run Stage C ───────────────────────────────────────────────────────
    logger.info("Launching Stage C pipeline...")
    result = run_stage_c(
        ds,
        emb,
        k_grid_values=(4, 5, 6, 7, 8, 9, 10),
        base_algorithms=("kmeans", "gmm", "ward", "spectral"),
        n_seeds_per_algorithm=5,
        keep_consensus_matrix=True,
    )
    logger.info("Stage C pipeline done in %.1f s", time.time() - t0)

    # ─── Persist tables ────────────────────────────────────────────────────
    result.algorithm_k_grid.to_csv(OUT_DIR / "algorithm_k_grid.csv", index=False)
    result.view_ablation.to_csv(OUT_DIR / "view_ablation.csv", index=False)
    result.consensus.aligned_base_labels.to_parquet(OUT_DIR / "base_clusterings.parquet")
    result.consensus.algorithm_pairwise_ari.to_csv(OUT_DIR / "algorithm_pairwise_ari.csv")
    result.final_labels.to_frame("cluster").to_parquet(OUT_DIR / "consensus_labels.parquet")
    result.consensus.confidence.to_frame("confidence").to_parquet(
        OUT_DIR / "per_patient_confidence.parquet"
    )
    result.enrichment.table.to_csv(OUT_DIR / "cluster_enrichment_full.csv", index=False)
    result.enrichment.top_per_cluster(top_n=15).to_csv(
        OUT_DIR / "cluster_enrichment_top.csv", index=False
    )
    logger.info("wrote 7 parquet/csv tables")

    # ─── DSM comparison JSON ───────────────────────────────────────────────
    with open(OUT_DIR / "dsm_comparison.json", "w") as fh:
        json.dump(result.dsm_comparison.summary_dict(), fh, indent=2, default=str)
    result.dsm_comparison.contingency.to_csv(OUT_DIR / "dsm_contingency.csv")
    result.dsm_comparison.row_normalized.to_csv(OUT_DIR / "dsm_contingency_rows.csv")
    result.dsm_comparison.col_normalized.to_csv(OUT_DIR / "dsm_contingency_cols.csv")

    # ─── Cluster cards ─────────────────────────────────────────────────────
    write_cluster_cards(result.cluster_cards, CARD_DIR)
    logger.info("wrote %d cluster cards to %s", len(result.cluster_cards), CARD_DIR)

    # ─── Figures ───────────────────────────────────────────────────────────
    cohort_labels = ds.metadata.loc[emb.values.index, "cohort"].values
    final_labels = result.final_labels.values

    # Figure 1 — algorithm × k grid heatmap (silhouette)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    grid = result.algorithm_k_grid
    pivot_sil = grid.pivot_table(index="algorithm", columns="k", values="silhouette")
    pivot_ari = grid.pivot_table(index="algorithm", columns="k", values="ari")
    for ax, pv, title, cmap in [
        (axes[0], pivot_sil, "Silhouette (higher = better)", "viridis"),
        (axes[1], pivot_ari, "ARI vs DSM cohort (lower = more transdiagnostic)", "magma"),
    ]:
        im = ax.imshow(pv.values, cmap=cmap, aspect="auto")
        ax.set_xticks(np.arange(pv.shape[1]))
        ax.set_xticklabels(pv.columns)
        ax.set_yticks(np.arange(pv.shape[0]))
        ax.set_yticklabels(pv.index)
        ax.set_xlabel("k")
        ax.set_title(title)
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                ax.text(j, i, f"{pv.values[i, j]:.2f}", ha="center", va="center",
                        color="white" if pv.values[i, j] < pv.values.mean() else "black",
                        fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle("Stage C ablation grid: algorithm × k")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_algorithm_k_grid.png", dpi=120)
    plt.close(fig)
    logger.info("saved 01_algorithm_k_grid.png")

    # Figure 2 — algorithm pairwise ARI heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    pa = result.consensus.algorithm_pairwise_ari
    im = ax.imshow(pa.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(pa.columns)))
    ax.set_xticklabels(pa.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(pa.index)))
    ax.set_yticklabels(pa.index, fontsize=7)
    ax.set_title("Pairwise ARI between base clusterings (consensus pool)")
    for i in range(pa.shape[0]):
        for j in range(pa.shape[1]):
            ax.text(j, i, f"{pa.values[i, j]:.2f}", ha="center", va="center",
                    color="white" if pa.values[i, j] > 0.5 else "black", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_pairwise_ari.png", dpi=120)
    plt.close(fig)
    logger.info("saved 02_pairwise_ari.png")

    # Figure 3 — per-patient confidence histogram + per-cluster
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    conf = result.consensus.confidence.values
    axes[0].hist(conf, bins=50, color="#1f77b4", alpha=0.85)
    axes[0].axvline(0, color="black", linestyle="--")
    axes[0].set_xlabel("per-patient confidence")
    axes[0].set_ylabel("number of patients")
    axes[0].set_title(f"Confidence distribution\n(mean = {conf.mean():.3f}, median = {np.median(conf):.3f})")
    # Per-cluster boxplot
    cluster_ids = sorted(np.unique(final_labels))
    box_data = [conf[final_labels == c] for c in cluster_ids]
    axes[1].boxplot(box_data, tick_labels=[f"C{c}" for c in cluster_ids])
    axes[1].set_xlabel("cluster")
    axes[1].set_ylabel("confidence")
    axes[1].axhline(0, color="black", linestyle="--", alpha=0.5)
    axes[1].set_title("Per-cluster confidence distribution")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_confidence.png", dpi=120)
    plt.close(fig)
    logger.info("saved 03_confidence.png")

    # Figure 4 — cluster × cohort heatmap (final consensus)
    fig, _ = plot_cluster_cohort_heatmap(
        final_labels,
        cohort_labels,
        normalize="index",
        title=f"Stage C consensus — fraction of each cluster by cohort (k={result.config['final_k']})",
    )
    fig.savefig(FIG_DIR / "04_cluster_cohort_rows.png", dpi=120)
    plt.close(fig)
    fig, _ = plot_cluster_cohort_heatmap(
        final_labels,
        cohort_labels,
        normalize="columns",
        title=f"Stage C consensus — fraction of each cohort by cluster (k={result.config['final_k']})",
    )
    fig.savefig(FIG_DIR / "05_cluster_cohort_cols.png", dpi=120)
    plt.close(fig)
    logger.info("saved 04 + 05 (cluster x cohort heatmaps)")

    # Figure 6 — t-SNE colored by Stage C consensus clusters
    if TSNE_CACHE.is_file():
        logger.info("Loading cached t-SNE coordinates from %s", TSNE_CACHE)
        coords = np.load(TSNE_CACHE)
    else:
        logger.info("Computing t-SNE for Stage C visualization...")
        coords, _ = tsne_project(emb.values, perplexity=30.0, metric="cosine", init="pca")
        np.save(TSNE_CACHE, coords)
    fig = plot_embedding_projection(
        coords,
        cohort=cohort_labels,
        cluster=final_labels,
        title=f"Stage C consensus clusters (k={result.config['final_k']})",
        method_used="t-SNE",
        figsize=(13, 5),
    )
    fig.savefig(FIG_DIR / "06_tsne_consensus.png", dpi=120)
    plt.close(fig)
    logger.info("saved 06_tsne_consensus.png")

    # Figure 7 — t-SNE colored by per-patient confidence
    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=conf, cmap="RdBu", s=5, alpha=0.6,
        vmin=-max(abs(conf.min()), abs(conf.max())),
        vmax=max(abs(conf.min()), abs(conf.max())),
    )
    ax.set_title("t-SNE colored by per-patient confidence (red = boundary, blue = central)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="confidence")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_tsne_confidence.png", dpi=120)
    plt.close(fig)
    logger.info("saved 07_tsne_confidence.png")

    # Figure 8 — Stage B vs Stage C cluster comparison
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    stage_b_path = REPO / "output" / "stratification" / "stage_b_review" / "cluster_labels.parquet"
    if stage_b_path.is_file():
        stage_b_labels = pd.read_parquet(stage_b_path)["cluster"].astype(int).loc[emb.values.index].values
        for ax, labels, title in [
            (axes[0], stage_b_labels, "Stage B (kmeans, k=8)"),
            (axes[1], final_labels, f"Stage C consensus (k={result.config['final_k']})"),
        ]:
            cmap = plt.get_cmap("tab10")
            unique = sorted(np.unique(labels))
            for i, c in enumerate(unique):
                m = labels == c
                ax.scatter(coords[m, 0], coords[m, 1], s=4, alpha=0.5, c=[cmap(i % 10)], label=f"C{c}")
            ax.set_title(title)
            ax.legend(fontsize=7, loc="best", ncol=2)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
        fig.suptitle(f"Stage B vs Stage C — cross-stage ARI = {result.stage_c_vs_stage_b_ari:.3f}")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "08_stage_b_vs_c.png", dpi=120)
        plt.close(fig)
        logger.info("saved 08_stage_b_vs_c.png")

    # ─── Final summary JSON ────────────────────────────────────────────────
    summary = {
        "n_patients": int(emb.n_patients),
        "embedding_dim": int(emb.dim),
        "k_final": int(result.config["final_k"]),
        "best_configuration": {k: v for k, v in result.best_configuration.items() if not isinstance(v, (np.ndarray,))},
        "n_base_clusterings": int(result.consensus.n_base_clusterings),
        "consensus_mean_confidence": float(result.consensus.confidence.mean()),
        "consensus_median_confidence": float(np.median(result.consensus.confidence.values)),
        "consensus_min_confidence": float(result.consensus.confidence.min()),
        "consensus_max_confidence": float(result.consensus.confidence.max()),
        "stage_b_baseline": result.stage_b_baseline_metrics,
        "stage_c_vs_stage_b_ari": float(result.stage_c_vs_stage_b_ari),
        "dsm_comparison": result.dsm_comparison.summary_dict(),
        "n_significant_enrichments": int(result.enrichment.n_significant),
        "n_enrichment_tests": int(result.enrichment.n_tests),
        "config": result.config,
        "total_runtime_seconds": float(time.time() - t0),
    }
    with open(OUT_DIR / "stage_c_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.info("Stage C complete in %.1f s — results in %s", time.time() - t0, OUT_DIR)


if __name__ == "__main__":
    main()
