"""Evaluate Stage B2 embeddings by comparing Stage C results.

Loads the pre-trained Stage B2 embeddings + the cached Stage B embedding,
runs Stage C on the combined composite, and reports whether the deep
embeddings improve the stratification vs the Stage B baseline.

Also produces a side-by-side t-SNE comparison (baseline vs combined) and
a boundary-patient count comparison to test whether Stage B2 resolves
the C5 ↔ C0 boundary identified in the deep analysis.
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

from face_stratification import PatientEmbedding, build_harmonized_dataset
from face_stratification.analysis.visualization import tsne_project
from face_stratification.stage_c import run_stage_c
from sklearn.metrics import adjusted_rand_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("stage_b2_eval")


CSV_PATHS = {
    "bp": REPO / "data" / "BP.csv",
    "sz": REPO / "data" / "SZ.csv",
    "dr": REPO / "data" / "DR.csv",
    "asp": REPO / "data" / "ASP.csv",
}

OUT_DIR = REPO / "output" / "stratification" / "stage_b2"
FIG_DIR = OUT_DIR / "figures"
STAGE_C_DIR = OUT_DIR / "stage_c_on_combined"
STAGE_C_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.time()

    logger.info("Loading cached dataset + embeddings")
    ds = build_harmonized_dataset(csv_paths=CSV_PATHS)
    combined_emb = PatientEmbedding.load(OUT_DIR / "embedding_combined")
    logger.info("  combined composite: %d × %d", combined_emb.n_patients, combined_emb.dim)

    # ─── Stage C on the combined composite ─────────────────────────────
    logger.info("Running Stage C consensus on the combined composite (light config)")
    stage_c_combined = run_stage_c(
        ds, combined_emb,
        k_grid_values=(5, 6, 7, 8),
        base_algorithms=("kmeans", "gmm", "ward"),
        n_seeds_per_algorithm=3,
        keep_consensus_matrix=False,
    )

    # Save outputs
    stage_c_combined.algorithm_k_grid.to_csv(STAGE_C_DIR / "algorithm_k_grid.csv", index=False)
    stage_c_combined.final_labels.to_frame("cluster").to_parquet(
        STAGE_C_DIR / "consensus_labels.parquet"
    )
    stage_c_combined.consensus.confidence.to_frame("confidence").to_parquet(
        STAGE_C_DIR / "per_patient_confidence.parquet"
    )
    stage_c_combined.dsm_comparison.contingency.to_csv(STAGE_C_DIR / "contingency.csv")
    stage_c_combined.dsm_comparison.row_normalized.to_csv(STAGE_C_DIR / "contingency_rows.csv")
    stage_c_combined.dsm_comparison.col_normalized.to_csv(STAGE_C_DIR / "contingency_cols.csv")
    with open(STAGE_C_DIR / "dsm_comparison.json", "w") as fh:
        json.dump(stage_c_combined.dsm_comparison.summary_dict(), fh, indent=2, default=str)

    # ─── Load Stage C baseline results (from Stage C full run) ─────────
    baseline_labels = pd.read_parquet(
        REPO / "output" / "stratification" / "stage_c" / "consensus_labels.parquet"
    )["cluster"].astype(int).loc[combined_emb.values.index]
    baseline_conf = pd.read_parquet(
        REPO / "output" / "stratification" / "stage_c" / "per_patient_confidence.parquet"
    )["confidence"].loc[combined_emb.values.index]
    with open(REPO / "output" / "stratification" / "stage_c" / "dsm_comparison.json") as fh:
        baseline_dsm = json.load(fh)

    # Cross-ARI
    cross_ari = float(adjusted_rand_score(
        baseline_labels.to_numpy(),
        stage_c_combined.final_labels.to_numpy(),
    ))

    # ─── Figures ───────────────────────────────────────────────────────
    # Figure 2: t-SNE comparison
    logger.info("Computing t-SNE on the combined composite")
    coords_combined, _ = tsne_project(
        combined_emb.values, perplexity=30.0, metric="cosine", init="pca"
    )
    np.save(STAGE_C_DIR / "projection_tsne.npy", coords_combined)

    tsne_baseline_path = (
        REPO / "output" / "stratification" / "stage_b_review" / "projection_tsne.npy"
    )
    if tsne_baseline_path.is_file():
        coords_baseline = np.load(tsne_baseline_path)
    else:
        coords_baseline = coords_combined

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    cmap = plt.get_cmap("tab10")

    # Baseline
    ax = axes[0]
    for i, c in enumerate(sorted(baseline_labels.unique())):
        m = baseline_labels.values == c
        ax.scatter(coords_baseline[m, 0], coords_baseline[m, 1],
                   s=3, alpha=0.4, c=[cmap(i % 10)], label=f"C{c}")
    ax.set_title(f"Stage C on Stage B only (k={baseline_dsm['n_clusters']})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=7, loc="best", ncol=2)

    # Combined
    ax = axes[1]
    for i, c in enumerate(sorted(stage_c_combined.final_labels.unique())):
        m = stage_c_combined.final_labels.values == c
        ax.scatter(coords_combined[m, 0], coords_combined[m, 1],
                   s=3, alpha=0.4, c=[cmap(i % 10)], label=f"C{c}")
    ax.set_title(f"Stage C on Stage B + B2 combined (k={stage_c_combined.config['final_k']})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=7, loc="best", ncol=2)

    fig.suptitle(f"Stage C consensus: Stage B baseline vs Stage B + B2 combined "
                 f"(cross-ARI = {cross_ari:.3f})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_baseline_vs_combined_tsne.png", dpi=120)
    plt.close(fig)
    logger.info("  saved 02_baseline_vs_combined_tsne.png")

    # Figure 3: confidence comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    conf_combined = stage_c_combined.consensus.confidence.to_numpy()
    axes[0].hist(baseline_conf.to_numpy(), bins=50, alpha=0.6, label="Stage B only", color="#1f77b4")
    axes[0].hist(conf_combined, bins=50, alpha=0.6, label="Stage B + B2", color="#d62728")
    axes[0].axvline(0, color="black", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("per-patient confidence")
    axes[0].set_ylabel("number of patients")
    axes[0].set_title("Confidence distributions")
    axes[0].legend()

    baseline_neg = int((baseline_conf < 0).sum())
    combined_neg = int((conf_combined < 0).sum())
    axes[1].bar(
        ["Stage B only", "Stage B + B2"],
        [baseline_neg, combined_neg],
        color=["#1f77b4", "#d62728"],
    )
    axes[1].set_ylabel("number of negative-confidence (boundary) patients")
    axes[1].set_title(f"Boundary patients: {baseline_neg} → {combined_neg}")
    for i, v in enumerate([baseline_neg, combined_neg]):
        axes[1].text(i, v + 5, str(v), ha="center", fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_boundary_comparison.png", dpi=120)
    plt.close(fig)
    logger.info("  saved 03_boundary_comparison.png")

    # ─── Summary JSON ──────────────────────────────────────────────────
    summary = {
        "n_patients": int(combined_emb.n_patients),
        "combined_dim": int(combined_emb.dim),
        "view_dims": combined_emb.view_dims,
        "stage_c_baseline": {
            "k": int(baseline_dsm["n_clusters"]),
            "ari_vs_dsm": baseline_dsm["ari"],
            "nmi_vs_dsm": baseline_dsm["nmi"],
            "cramers_v": baseline_dsm["cramers_v"],
            "mean_cohort_entropy": baseline_dsm["mean_cluster_entropy_bits"],
            "consensus_mean_confidence": float(baseline_conf.mean()),
            "n_negative_confidence": baseline_neg,
        },
        "stage_c_combined": {
            "k": stage_c_combined.config["final_k"],
            "silhouette": stage_c_combined.best_configuration["silhouette"],
            "davies_bouldin": stage_c_combined.best_configuration["davies_bouldin"],
            "ari_vs_dsm": stage_c_combined.dsm_comparison.ari,
            "nmi_vs_dsm": stage_c_combined.dsm_comparison.nmi,
            "cramers_v": stage_c_combined.dsm_comparison.cramers_v,
            "mean_cohort_entropy": stage_c_combined.dsm_comparison.mean_cluster_entropy_bits,
            "consensus_mean_confidence": float(conf_combined.mean()),
            "n_negative_confidence": combined_neg,
        },
        "baseline_vs_combined_ari": cross_ari,
        "boundary_reduction": baseline_neg - combined_neg,
        "boundary_reduction_fraction": (baseline_neg - combined_neg) / max(baseline_neg, 1),
        "total_runtime_seconds": float(time.time() - t0),
    }
    with open(OUT_DIR / "stage_b2_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("Stage B baseline:       k=%d  ARI=%.3f  NMI=%.3f  entropy=%.3f  n_boundary=%d",
                summary["stage_c_baseline"]["k"],
                summary["stage_c_baseline"]["ari_vs_dsm"],
                summary["stage_c_baseline"]["nmi_vs_dsm"],
                summary["stage_c_baseline"]["mean_cohort_entropy"],
                baseline_neg)
    logger.info("Stage B + B2 combined:  k=%d  ARI=%.3f  NMI=%.3f  entropy=%.3f  n_boundary=%d",
                summary["stage_c_combined"]["k"],
                summary["stage_c_combined"]["ari_vs_dsm"],
                summary["stage_c_combined"]["nmi_vs_dsm"],
                summary["stage_c_combined"]["mean_cohort_entropy"],
                combined_neg)
    logger.info("Cross-ARI (baseline vs combined): %.3f", cross_ari)
    logger.info("Boundary reduction: %d patients (%.1f%%)",
                baseline_neg - combined_neg,
                100 * (baseline_neg - combined_neg) / max(baseline_neg, 1))
    logger.info("Total time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
