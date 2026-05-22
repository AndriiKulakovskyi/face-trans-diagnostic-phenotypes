"""Run the full Stage B review pass end-to-end.

Produces:

- ``output/stratification/stage_b_review/ablation_summary.csv`` — global vs
  per-cohort normalization comparison.
- ``output/stratification/stage_b_review/kmeans_sweep.csv`` — k sweep with
  silhouette + ARI + NMI + V + bootstrap stability.
- ``output/stratification/stage_b_review/cluster_enrichment.csv`` — per-cluster
  feature enrichment (Mann-Whitney + BH FDR).
- ``output/stratification/stage_b_review/cluster_medoids.json`` — medoid
  patient ids + their French vignettes.
- ``output/stratification/stage_b_review/figures/*.png`` — all figures.

Run via:

    python scripts/run_stage_b_review.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from face_stratification import (
    ConcatenatedEmbedding,
    bootstrap_stability,
    build_harmonized_dataset,
    fit_embedding,
    kmeans_sweep,
    run_kmeans,
)
from face_stratification.analysis.ablation import run_normalization_ablation
from face_stratification.analysis.enrichment import (
    compute_cluster_feature_enrichment,
)
from face_stratification.analysis.medoids import (
    extract_cluster_medoids,
    fetch_medoid_vignettes,
)
from face_stratification.analysis.visualization import (
    plot_cluster_cohort_heatmap,
    plot_embedding_projection,
    plot_enrichment_bars,
    plot_kmeans_sweep,
    umap_project,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("stage_b_review")


CSV_PATHS = {
    "bp": REPO / "data" / "BP.csv",
    "sz": REPO / "data" / "SZ.csv",
    "dr": REPO / "data" / "DR.csv",
    "asp": REPO / "data" / "ASP.csv",
}

OUT_DIR = REPO / "output" / "stratification" / "stage_b_review"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.time()
    logger.info("Step 1: harmonize full cohort")
    ds = build_harmonized_dataset(csv_paths=CSV_PATHS)
    logger.info("  %d × %d in %.1fs", ds.n_patients, ds.n_features, time.time() - t0)

    # ─── k sweep on the global-normalization embedding (default Stage B) ────
    logger.info("Step 2: fit default Stage B embedding (global normalization)")
    t1 = time.time()
    model = ConcatenatedEmbedding.build_default(
        pca_dim=8, td_spectral_dim=16, multiplex_spectral_dim=32
    )
    embedding, graph = fit_embedding(ds, model=model)
    logger.info(
        "  embedding %d × %d in %.1fs (isolated=%d)",
        embedding.n_patients,
        embedding.dim,
        time.time() - t1,
        embedding.n_isolated_nodes,
    )

    logger.info("Step 3: k sweep (k=2..10) with silhouette + ARI/NMI/V")
    sweep = kmeans_sweep(
        embedding.values,
        k_values=list(range(2, 11)),
        reference_labels=ds.metadata["cohort"].values,
        silhouette_sample_size=5000,
    )
    sweep.to_csv(OUT_DIR / "kmeans_sweep.csv", index=False)
    logger.info("  wrote kmeans_sweep.csv")

    # Pick best k by silhouette
    best_k = int(sweep.iloc[sweep["silhouette"].idxmax()]["k"])
    logger.info("  best k by silhouette: %d", best_k)

    # Bootstrap stability at best k
    logger.info("Step 4: bootstrap stability at k=%d", best_k)
    bs = bootstrap_stability(
        embedding.values,
        n_clusters=best_k,
        n_bootstraps=25,
        subsample_fraction=0.8,
    )
    logger.info(
        "  bootstrap mean ARI=%.3f (std=%.3f, %d pairs)",
        bs["mean_ari"], bs["std_ari"], bs["n_pairs"],
    )

    # Final k-means with best k
    final_assignment = run_kmeans(
        embedding.values,
        n_clusters=best_k,
        reference_labels=ds.metadata["cohort"].values,
    )
    labels = final_assignment.labels
    labels.to_frame("cluster").to_parquet(OUT_DIR / "cluster_labels.parquet")
    logger.info("  wrote cluster_labels.parquet")

    # ─── Figure 1: k-sweep ─────────────────────────────────────────────────
    fig = plot_kmeans_sweep(sweep, title=f"Global normalization — k sweep (silhouette peaks at k={best_k})")
    fig.savefig(FIG_DIR / "01_kmeans_sweep.png", dpi=120)
    plt.close(fig)
    logger.info("  saved 01_kmeans_sweep.png")

    # ─── Normalization ablation ───────────────────────────────────────────
    logger.info("Step 5: normalization ablation (global vs per-cohort) at k=%d", best_k)
    t5 = time.time()
    ablation = run_normalization_ablation(
        ds,
        k_clusters=best_k,
        pca_dim=8,
        td_spectral_dim=16,
        multiplex_spectral_dim=32,
        n_bootstraps=20,
    )
    logger.info("  ablation done in %.1fs", time.time() - t5)
    ablation_table = ablation.summary_table()
    ablation_table.to_csv(OUT_DIR / "ablation_summary.csv")
    logger.info("  wrote ablation_summary.csv")
    logger.info("  variant-vs-variant ARI: %.3f", ablation.variant_vs_variant_ari)

    # Write a JSON manifest with the scalar findings
    with open(OUT_DIR / "ablation_manifest.json", "w") as fh:
        json.dump(
            {
                "k_clusters": best_k,
                "variant_vs_variant_ari": ablation.variant_vs_variant_ari,
                "variant_vs_variant_nmi": ablation.variant_vs_variant_nmi,
                "global": {
                    "silhouette": ablation.global_result.metrics.silhouette,
                    "ari_vs_cohort": ablation.global_result.metrics.ari_vs_reference,
                    "nmi_vs_cohort": ablation.global_result.metrics.nmi_vs_reference,
                    "bootstrap_mean_ari": ablation.global_result.bootstrap_mean_ari,
                    "bootstrap_std_ari": ablation.global_result.bootstrap_std_ari,
                },
                "per_cohort": {
                    "silhouette": ablation.per_cohort_result.metrics.silhouette,
                    "ari_vs_cohort": ablation.per_cohort_result.metrics.ari_vs_reference,
                    "nmi_vs_cohort": ablation.per_cohort_result.metrics.nmi_vs_reference,
                    "bootstrap_mean_ari": ablation.per_cohort_result.bootstrap_mean_ari,
                    "bootstrap_std_ari": ablation.per_cohort_result.bootstrap_std_ari,
                },
            },
            fh,
            indent=2,
            default=str,
        )
    logger.info("  wrote ablation_manifest.json")

    # ─── Feature enrichment ───────────────────────────────────────────────
    logger.info("Step 6: per-cluster feature enrichment (BH FDR)")
    enrichment = compute_cluster_feature_enrichment(
        ds.X, labels, q_threshold=0.05
    )
    enrichment.table.to_csv(OUT_DIR / "cluster_enrichment.csv", index=False)
    top = enrichment.top_per_cluster(top_n=15)
    top.to_csv(OUT_DIR / "cluster_enrichment_top.csv", index=False)
    logger.info(
        "  %d tests, %d significant → wrote cluster_enrichment.csv + top",
        enrichment.n_tests,
        enrichment.n_significant,
    )

    # Figure 2: enrichment bars
    fig = plot_enrichment_bars(
        enrichment.table,
        top_n=8,
        title=f"Top enriched features per cluster (k={best_k}, BH q < 0.05)",
    )
    fig.savefig(FIG_DIR / "02_enrichment_bars.png", dpi=120)
    plt.close(fig)
    logger.info("  saved 02_enrichment_bars.png")

    # ─── Medoids + vignettes ──────────────────────────────────────────────
    logger.info("Step 7: extract cluster medoids and retrieve French vignettes")
    medoids = extract_cluster_medoids(embedding.values, labels, n_per_cluster=1)
    vr = fetch_medoid_vignettes(medoids, csv_paths=CSV_PATHS)

    medoid_records = []
    for m in medoids:
        vignette = vr.vignettes.get(m.cluster, "")
        medoid_records.append(
            {
                "cluster": m.cluster,
                "cohort": m.cohort,
                "patient_id": m.patient_id,
                "distance_to_centroid": m.distance_to_centroid,
                "cluster_size": m.cluster_size,
                "cluster_cohort_mix": m.cluster_cohort_mix,
                "vignette_full": vignette,
                "vignette_preview": vignette[:600],
            }
        )
    with open(OUT_DIR / "cluster_medoids.json", "w") as fh:
        json.dump(medoid_records, fh, indent=2, ensure_ascii=False, default=str)
    logger.info("  wrote cluster_medoids.json")

    # ─── 2D projection ────────────────────────────────────────────────────
    logger.info("Step 8: 2D projection + cohort/cluster scatter plots")
    t8 = time.time()
    coords, method = umap_project(embedding.values, random_state=0)
    logger.info("  %s done in %.1fs", method, time.time() - t8)

    np.save(OUT_DIR / f"projection_{method.lower()}.npy", coords)

    fig = plot_embedding_projection(
        coords,
        cohort=ds.metadata["cohort"].values,
        cluster=labels.values,
        title=f"Stage B composite embedding (k={best_k})",
        method_used=method,
    )
    fig.savefig(FIG_DIR / "03_projection.png", dpi=120)
    plt.close(fig)
    logger.info("  saved 03_projection.png")

    # ─── Cluster × cohort heatmap ────────────────────────────────────────
    logger.info("Step 9: cluster × cohort heatmap")
    fig, ct_row = plot_cluster_cohort_heatmap(
        labels.values,
        ds.metadata["cohort"].values,
        normalize="index",
        title=f"Row-normalized cluster × cohort (k={best_k}) — fraction of each cluster by cohort",
    )
    fig.savefig(FIG_DIR / "04_cluster_cohort_heatmap_rows.png", dpi=120)
    plt.close(fig)
    ct_row.to_csv(OUT_DIR / "cluster_cohort_contingency_rows.csv")

    fig, ct_col = plot_cluster_cohort_heatmap(
        labels.values,
        ds.metadata["cohort"].values,
        normalize="columns",
        title=f"Column-normalized cluster × cohort (k={best_k}) — fraction of each cohort by cluster",
    )
    fig.savefig(FIG_DIR / "05_cluster_cohort_heatmap_cols.png", dpi=120)
    plt.close(fig)
    ct_col.to_csv(OUT_DIR / "cluster_cohort_contingency_cols.csv")
    logger.info("  saved 04 + 05")

    # Final report
    summary = {
        "n_patients": ds.n_patients,
        "n_features": ds.n_features,
        "embedding_dim": embedding.dim,
        "best_k": best_k,
        "silhouette_at_best_k": float(final_assignment.metrics.silhouette),
        "ari_at_best_k": float(final_assignment.metrics.ari_vs_reference),
        "nmi_at_best_k": float(final_assignment.metrics.nmi_vs_reference),
        "bootstrap_mean_ari": bs["mean_ari"],
        "bootstrap_std_ari": bs["std_ari"],
        "n_significant_enrichments": enrichment.n_significant,
        "ablation_variant_vs_variant_ari": ablation.variant_vs_variant_ari,
        "total_runtime_seconds": float(time.time() - t0),
    }
    with open(OUT_DIR / "review_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("Done — wrote review_summary.json in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
