"""Evaluate the Stage B2.5 best configuration via the full Stage C pipeline.

Loads:
- the Stage B composite (56d)
- the Stage B2.5 canonical embedding (32d, L=3 transdiagnostic_only T=0.5)

Builds a **Stage B + Stage B2.5** combined composite (88d) and runs the full
Stage C consensus clustering (KMeans + GMM + Ward + Spectral) for a fair
comparison to:

- the original Stage C on Stage B alone (k=6)
- the Stage B2 combined on Stage B + GAE + Contrastive (k=7)

Saves results under ``output/stratification/stage_b2/sweep/stage_c_on_best/``.
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
    PatientEmbedding,
    build_harmonized_dataset,
)
from face_stratification.analysis.visualization import plot_cluster_cohort_heatmap
from face_stratification.stage_c import run_stage_c

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("stage_b2_5_eval")


CSV_PATHS = {
    "bp": REPO / "data" / "BP.csv",
    "sz": REPO / "data" / "SZ.csv",
    "dr": REPO / "data" / "DR.csv",
    "asp": REPO / "data" / "ASP.csv",
}

SWEEP_DIR = REPO / "output" / "stratification" / "stage_b2" / "sweep"
OUT_DIR = SWEEP_DIR / "stage_c_on_best"
FIG_DIR = SWEEP_DIR / "figures"
EMBED_CACHE = REPO / "output" / "stratification" / "stage_b_review" / "embedding_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.time()

    logger.info("Loading harmonized dataset + both embeddings")
    ds = build_harmonized_dataset(csv_paths=CSV_PATHS)
    stage_b = PatientEmbedding.load(EMBED_CACHE)
    stage_b2_5 = PatientEmbedding.load(SWEEP_DIR / "embedding_best")
    logger.info("  Stage B:   %d × %d", stage_b.n_patients, stage_b.dim)
    logger.info("  Stage B2.5: %d × %d", stage_b2_5.n_patients, stage_b2_5.dim)

    # ─── Build combined composite (Stage B + B2.5) ──────────────────────
    stage_b_arr = stage_b.values.loc[stage_b2_5.values.index].to_numpy(dtype=np.float64)
    b2_5_arr = stage_b2_5.values.to_numpy(dtype=np.float64)
    combined = np.concatenate([stage_b_arr, b2_5_arr], axis=1)
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    combined = combined / np.where(norms > 0, norms, 1.0)

    col_names = (
        [f"stage_b::{c}" for c in stage_b.values.columns]
        + [f"stage_b2_5::{c}" for c in stage_b2_5.values.columns]
    )
    combined_df = pd.DataFrame(
        combined,
        index=stage_b2_5.values.index,
        columns=col_names,
        dtype=np.float64,
    )
    combined_emb = PatientEmbedding(
        values=combined_df,
        model_name="stage_b_plus_b2_5",
        model_config={
            "stage_b_dim": stage_b.dim,
            "stage_b2_5_dim": stage_b2_5.dim,
            "stage_b2_5_model": "L=3, transdiagnostic_only, T=0.5",
            "total_dim": combined_df.shape[1],
        },
        view_dims={"stage_b": stage_b.dim, "stage_b2_5": stage_b2_5.dim},
        n_isolated_nodes=0,
        schema_version=ds.schema.version,
    )
    combined_emb.save(SWEEP_DIR / "embedding_b2_5_combined")
    logger.info("  combined composite: %d × %d", combined_emb.n_patients, combined_emb.dim)

    # ─── Run Stage C on combined (light config) ─────────────────────────
    logger.info("Running Stage C consensus on combined composite")
    result = run_stage_c(
        ds, combined_emb,
        k_grid_values=(5, 6, 7, 8),
        base_algorithms=("kmeans", "gmm", "ward"),
        n_seeds_per_algorithm=3,
        keep_consensus_matrix=False,
    )

    # Save outputs
    result.algorithm_k_grid.to_csv(OUT_DIR / "algorithm_k_grid.csv", index=False)
    result.final_labels.to_frame("cluster").to_parquet(OUT_DIR / "consensus_labels.parquet")
    result.consensus.confidence.to_frame("confidence").to_parquet(OUT_DIR / "per_patient_confidence.parquet")
    result.dsm_comparison.contingency.to_csv(OUT_DIR / "contingency.csv")
    result.dsm_comparison.row_normalized.to_csv(OUT_DIR / "contingency_rows.csv")
    result.dsm_comparison.col_normalized.to_csv(OUT_DIR / "contingency_cols.csv")
    with open(OUT_DIR / "dsm_comparison.json", "w") as fh:
        json.dump(result.dsm_comparison.summary_dict(), fh, indent=2, default=str)

    # ─── Load baselines for comparison ───────────────────────────────────
    stage_b_only = {}
    stage_c_root = REPO / "output" / "stratification" / "stage_c"
    if (stage_c_root / "dsm_comparison.json").is_file():
        with open(stage_c_root / "dsm_comparison.json") as fh:
            stage_b_only = json.load(fh)

    stage_b2_combined = {}
    stage_b2_c = REPO / "output" / "stratification" / "stage_b2" / "stage_c_on_combined"
    if (stage_b2_c / "dsm_comparison.json").is_file():
        with open(stage_b2_c / "dsm_comparison.json") as fh:
            stage_b2_combined = json.load(fh)

    # ─── Cross-partition ARI ─────────────────────────────────────────────
    from sklearn.metrics import adjusted_rand_score

    baseline_labels_path = stage_c_root / "consensus_labels.parquet"
    b2_labels_path = stage_b2_c / "consensus_labels.parquet"
    baseline_labels = pd.read_parquet(baseline_labels_path)["cluster"].astype(int).loc[combined_emb.values.index] if baseline_labels_path.is_file() else None
    b2_labels = pd.read_parquet(b2_labels_path)["cluster"].astype(int).loc[combined_emb.values.index] if b2_labels_path.is_file() else None

    cross_ari_vs_b = float(adjusted_rand_score(baseline_labels.to_numpy(), result.final_labels.to_numpy())) if baseline_labels is not None else None
    cross_ari_vs_b2 = float(adjusted_rand_score(b2_labels.to_numpy(), result.final_labels.to_numpy())) if b2_labels is not None else None

    # ─── Per-cohort entropy of the new final partition ───────────────────
    conf = result.consensus.confidence.to_numpy()
    n_boundary = int((conf < 0).sum())
    mean_conf = float(conf.mean())

    # ─── Summary ─────────────────────────────────────────────────────────
    summary = {
        "n_patients": int(combined_emb.n_patients),
        "stage_b_dim": int(stage_b.dim),
        "stage_b2_5_dim": int(stage_b2_5.dim),
        "combined_dim": int(combined_emb.dim),
        "stage_b2_5_config": "L=3, transdiagnostic_only, T=0.5, h=64, d=32",
        "final_k": int(result.config["final_k"]),
        "silhouette": float(result.best_configuration["silhouette"]),
        "davies_bouldin": float(result.best_configuration["davies_bouldin"]),
        "ari_vs_dsm": float(result.dsm_comparison.ari),
        "nmi_vs_dsm": float(result.dsm_comparison.nmi),
        "cramers_v": float(result.dsm_comparison.cramers_v),
        "mean_cluster_entropy_bits": float(result.dsm_comparison.mean_cluster_entropy_bits),
        "mean_transdiagnostic_score": float(result.dsm_comparison.mean_transdiagnostic_score),
        "consensus_mean_confidence": mean_conf,
        "n_negative_confidence": n_boundary,
        "cross_ari_vs_stage_b_only": cross_ari_vs_b,
        "cross_ari_vs_stage_b_plus_b2": cross_ari_vs_b2,
        "total_runtime_seconds": float(time.time() - t0),
    }
    with open(OUT_DIR / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    # ─── Cluster × cohort heatmaps ───────────────────────────────────────
    cohort_labels = ds.metadata.loc[combined_emb.values.index, "cohort"].to_numpy()
    fig, _ = plot_cluster_cohort_heatmap(
        result.final_labels.to_numpy(),
        cohort_labels,
        normalize="index",
        title=f"Stage B + B2.5 consensus — rows normalized (k={result.config['final_k']})",
    )
    fig.savefig(FIG_DIR / "04_b2_5_cluster_cohort_rows.png", dpi=120)
    plt.close(fig)

    fig, _ = plot_cluster_cohort_heatmap(
        result.final_labels.to_numpy(),
        cohort_labels,
        normalize="columns",
        title=f"Stage B + B2.5 consensus — cols normalized (k={result.config['final_k']})",
    )
    fig.savefig(FIG_DIR / "05_b2_5_cluster_cohort_cols.png", dpi=120)
    plt.close(fig)

    logger.info("=" * 60)
    logger.info("Stage C on Stage B only:  k=%s  ARI=%.3f  V=%.3f  entropy=%.3f",
                stage_b_only.get("n_clusters", "?"),
                float(stage_b_only.get("ari", float("nan"))),
                float(stage_b_only.get("cramers_v", float("nan"))),
                float(stage_b_only.get("mean_cluster_entropy_bits", float("nan"))))
    logger.info("Stage C on Stage B+B2:    k=%s  ARI=%.3f  V=%.3f  entropy=%.3f",
                stage_b2_combined.get("n_clusters", "?"),
                float(stage_b2_combined.get("ari", float("nan"))),
                float(stage_b2_combined.get("cramers_v", float("nan"))),
                float(stage_b2_combined.get("mean_cluster_entropy_bits", float("nan"))))
    logger.info("Stage C on Stage B+B2.5:  k=%d  ARI=%.3f  V=%.3f  entropy=%.3f",
                result.config["final_k"],
                summary["ari_vs_dsm"],
                summary["cramers_v"],
                summary["mean_cluster_entropy_bits"])
    logger.info("Cross-ARI vs Stage B only:     %s",
                f"{cross_ari_vs_b:.3f}" if cross_ari_vs_b else "n/a")
    logger.info("Cross-ARI vs Stage B + B2:     %s",
                f"{cross_ari_vs_b2:.3f}" if cross_ari_vs_b2 else "n/a")
    logger.info("Mean confidence: %.3f (boundary patients: %d)", mean_conf, n_boundary)
    logger.info("Total: %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()
