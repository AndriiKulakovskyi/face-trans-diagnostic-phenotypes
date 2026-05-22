"""End-to-end Stage B2 training driver.

Loads the cached Stage B composite embedding + Stage A multiplex graph,
trains both GNN models (GAE + GraphCL contrastive), saves their
embeddings, builds a combined composite (Stage B + Stage B2 views), and
re-runs Stage C consensus on the combined composite.

Outputs under ``output/stratification/stage_b2/``:

- ``embedding_gae/``           — StageB2GAE PatientEmbedding (parquet + manifest)
- ``embedding_contrastive/``   — StageB2GraphContrastive PatientEmbedding
- ``embedding_combined/``      — concatenated Stage B + Stage B2 composite
- ``training_history.json``    — loss curves for both models
- ``stage_c_on_combined/``     — Stage C results on the combined composite
- ``figures/*.png``            — training curves + t-SNE comparisons

Run:

    python scripts/train_stage_b2.py
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
    plot_embedding_projection,
    tsne_project,
)
from face_stratification.graph.patient_similarity import build_multiplex_graph
from face_stratification.harmonization.normalization import (
    fit_normalization,
    transform_normalization,
)
from face_stratification.stage_b2 import (
    StageB2GAE,
    StageB2GraphContrastive,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("stage_b2")


CSV_PATHS = {
    "bp": REPO / "data" / "BP.csv",
    "sz": REPO / "data" / "SZ.csv",
    "dr": REPO / "data" / "DR.csv",
    "asp": REPO / "data" / "ASP.csv",
}

OUT_DIR = REPO / "output" / "stratification" / "stage_b2"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.time()

    # ─── 1. Load dataset + rebuild multiplex graph ───────────────────────
    logger.info("Loading harmonized dataset")
    ds = build_harmonized_dataset(csv_paths=CSV_PATHS)

    logger.info("Building multiplex graph")
    stats = fit_normalization(ds.X, ds.schema)
    Xn = transform_normalization(ds.X, stats)
    G, _blocks, _td = build_multiplex_graph(Xn, ds.schema, k=10, metadata=ds.metadata)
    logger.info("  multiplex graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    # ─── 2. Train GAE ────────────────────────────────────────────────────
    logger.info("Training Stage B2 GAE (150 epochs, feature_source=composite)")
    gae = StageB2GAE(
        hidden_dim=64,
        out_dim=32,
        n_epochs=150,
        learning_rate=1e-2,
        dropout=0.1,
        seed=0,
        feature_source="composite",
    )
    gae.fit(ds, graph=G)
    gae_emb = gae.transform()
    gae_emb.save(OUT_DIR / "embedding_gae")
    logger.info("  GAE trained in %.1fs  →  %d × %d embedding",
                gae.config["training_time_seconds"], gae_emb.n_patients, gae_emb.dim)

    # ─── 3. Train contrastive ────────────────────────────────────────────
    logger.info("Training Stage B2 GraphContrastive (150 epochs)")
    contrastive = StageB2GraphContrastive(
        hidden_dim=64,
        out_dim=32,
        n_epochs=150,
        learning_rate=5e-3,
        dropout=0.1,
        seed=0,
        feature_source="composite",
        p_edge=0.2,
        p_feat=0.1,
        temperature=0.5,
    )
    contrastive.fit(ds, graph=G)
    contrastive_emb = contrastive.transform()
    contrastive_emb.save(OUT_DIR / "embedding_contrastive")
    logger.info("  Contrastive trained in %.1fs  →  %d × %d embedding",
                contrastive.config["training_time_seconds"], contrastive_emb.n_patients, contrastive_emb.dim)

    # ─── 4. Build combined composite (Stage B + Stage B2) ───────────────
    logger.info("Building combined composite (Stage B 56d + GAE 32d + Contrastive 32d)")
    stage_b = PatientEmbedding.load(
        REPO / "output" / "stratification" / "stage_b_review" / "embedding_cache"
    )
    stage_b_arr = stage_b.values.loc[gae_emb.values.index].to_numpy(dtype=np.float64)
    gae_arr = gae_emb.values.to_numpy(dtype=np.float64)
    contrastive_arr = contrastive_emb.values.to_numpy(dtype=np.float64)

    # Concatenate + row-L2-normalize so all views contribute equally
    combined = np.concatenate([stage_b_arr, gae_arr, contrastive_arr], axis=1)
    row_norms = np.linalg.norm(combined, axis=1, keepdims=True)
    row_norms = np.where(row_norms > 0, row_norms, 1.0)
    combined = combined / row_norms

    col_names = (
        [f"stage_b::{c}" for c in stage_b.values.columns]
        + [f"gae::{c}" for c in gae_emb.values.columns]
        + [f"contrastive::{c}" for c in contrastive_emb.values.columns]
    )
    combined_df = pd.DataFrame(combined, index=gae_emb.values.index, columns=col_names, dtype=np.float64)
    combined_emb = PatientEmbedding(
        values=combined_df,
        model_name="stage_b_plus_b2_combined",
        model_config={
            "stage_b_dim": stage_b.dim,
            "gae_dim": gae_emb.dim,
            "contrastive_dim": contrastive_emb.dim,
            "total_dim": combined_df.shape[1],
        },
        view_dims={
            "stage_b": stage_b.dim,
            "stage_b2_gae": gae_emb.dim,
            "stage_b2_contrastive": contrastive_emb.dim,
        },
        n_isolated_nodes=0,
        schema_version=ds.schema.version,
    )
    combined_emb.save(OUT_DIR / "embedding_combined")
    logger.info("  combined composite: %d × %d", combined_emb.n_patients, combined_emb.dim)

    # ─── 5. Save training history ────────────────────────────────────────
    with open(OUT_DIR / "training_history.json", "w") as fh:
        json.dump({
            "gae": gae._training_history,  # noqa: SLF001
            "contrastive": contrastive._training_history,  # noqa: SLF001
        }, fh, indent=2, default=str)

    # ─── 6. Training loss curves ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    gae_hist = pd.DataFrame(gae._training_history)  # noqa: SLF001
    cont_hist = pd.DataFrame(contrastive._training_history)  # noqa: SLF001
    axes[0].plot(gae_hist["epoch"], gae_hist["loss"], "o-", color="#1f77b4", label="loss")
    if "gap" in gae_hist.columns:
        ax2 = axes[0].twinx()
        ax2.plot(gae_hist["epoch"], gae_hist["gap"], "o-", color="#d62728", label="pos-neg gap")
        ax2.set_ylabel("pos−neg link probability gap", color="#d62728")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("BCE loss", color="#1f77b4")
    axes[0].set_title("Stage B2 GAE — training curve")
    axes[1].plot(cont_hist["epoch"], cont_hist["loss"], "o-", color="#2ca02c")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("NT-Xent loss")
    axes[1].set_title("Stage B2 GraphContrastive — training curve")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_training_curves.png", dpi=120)
    plt.close(fig)
    logger.info("  saved 01_training_curves.png")

    # ─── 7. Run Stage C on the combined composite ───────────────────────
    logger.info("Running Stage C consensus on the combined composite")
    from face_stratification.stage_c import run_stage_c
    stage_c_combined = run_stage_c(
        ds, combined_emb,
        k_grid_values=(4, 5, 6, 7, 8),
        base_algorithms=("kmeans", "gmm", "ward"),  # skip spectral to save time
        n_seeds_per_algorithm=3,
        keep_consensus_matrix=False,
    )

    # Save summary
    stage_c_dir = OUT_DIR / "stage_c_on_combined"
    stage_c_dir.mkdir(parents=True, exist_ok=True)
    stage_c_combined.algorithm_k_grid.to_csv(stage_c_dir / "algorithm_k_grid.csv", index=False)
    stage_c_combined.final_labels.to_frame("cluster").to_parquet(
        stage_c_dir / "consensus_labels.parquet"
    )
    stage_c_combined.consensus.confidence.to_frame("confidence").to_parquet(
        stage_c_dir / "per_patient_confidence.parquet"
    )
    stage_c_combined.dsm_comparison.contingency.to_csv(stage_c_dir / "contingency.csv")
    stage_c_combined.dsm_comparison.row_normalized.to_csv(stage_c_dir / "contingency_rows.csv")
    stage_c_combined.dsm_comparison.col_normalized.to_csv(stage_c_dir / "contingency_cols.csv")
    with open(stage_c_dir / "dsm_comparison.json", "w") as fh:
        json.dump(stage_c_combined.dsm_comparison.summary_dict(), fh, indent=2, default=str)

    # ─── 8. Compare to vanilla Stage C (Stage B only) ───────────────────
    logger.info("Comparing Stage C on Stage B vs Stage B + B2")
    from face_stratification.stage_c import run_stage_c as run_stage_c_again
    stage_c_baseline = run_stage_c_again(
        ds, stage_b,
        k_grid_values=(4, 5, 6, 7, 8),
        base_algorithms=("kmeans", "gmm", "ward"),
        n_seeds_per_algorithm=3,
        keep_consensus_matrix=False,
    )

    from sklearn.metrics import adjusted_rand_score
    baseline_vs_combined_ari = float(adjusted_rand_score(
        stage_c_baseline.final_labels.to_numpy(),
        stage_c_combined.final_labels.to_numpy(),
    ))

    # ─── 9. Summary JSON ─────────────────────────────────────────────────
    summary = {
        "n_patients": int(combined_emb.n_patients),
        "stage_b_dim": int(stage_b.dim),
        "gae_dim": int(gae_emb.dim),
        "contrastive_dim": int(contrastive_emb.dim),
        "combined_dim": int(combined_emb.dim),
        "gae_training": {
            "time_s": gae.config["training_time_seconds"],
            "final_loss": gae.config["final_loss"],
            "final_pos_neg_gap": gae.config["final_gap"],
        },
        "contrastive_training": {
            "time_s": contrastive.config["training_time_seconds"],
            "final_loss": contrastive.config["final_loss"],
        },
        "stage_c_baseline": {
            "k": stage_c_baseline.config["final_k"],
            "silhouette": stage_c_baseline.best_configuration["silhouette"],
            "ari_vs_dsm": stage_c_baseline.dsm_comparison.ari,
            "nmi_vs_dsm": stage_c_baseline.dsm_comparison.nmi,
            "cramers_v": stage_c_baseline.dsm_comparison.cramers_v,
            "mean_cohort_entropy": stage_c_baseline.dsm_comparison.mean_cluster_entropy_bits,
            "consensus_mean_confidence": float(stage_c_baseline.consensus.confidence.mean()),
            "n_negative_confidence": int((stage_c_baseline.consensus.confidence < 0).sum()),
        },
        "stage_c_combined": {
            "k": stage_c_combined.config["final_k"],
            "silhouette": stage_c_combined.best_configuration["silhouette"],
            "ari_vs_dsm": stage_c_combined.dsm_comparison.ari,
            "nmi_vs_dsm": stage_c_combined.dsm_comparison.nmi,
            "cramers_v": stage_c_combined.dsm_comparison.cramers_v,
            "mean_cohort_entropy": stage_c_combined.dsm_comparison.mean_cluster_entropy_bits,
            "consensus_mean_confidence": float(stage_c_combined.consensus.confidence.mean()),
            "n_negative_confidence": int((stage_c_combined.consensus.confidence < 0).sum()),
        },
        "baseline_vs_combined_ari": baseline_vs_combined_ari,
        "total_runtime_seconds": float(time.time() - t0),
    }
    with open(OUT_DIR / "stage_b2_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.info("Stage B2 complete in %.1f s  →  %s", time.time() - t0, OUT_DIR)
    logger.info("Stage B baseline:  silhouette=%.3f  ari=%.3f  n_neg_conf=%d",
                summary["stage_c_baseline"]["silhouette"],
                summary["stage_c_baseline"]["ari_vs_dsm"],
                summary["stage_c_baseline"]["n_negative_confidence"])
    logger.info("Stage B+B2:        silhouette=%.3f  ari=%.3f  n_neg_conf=%d",
                summary["stage_c_combined"]["silhouette"],
                summary["stage_c_combined"]["ari_vs_dsm"],
                summary["stage_c_combined"]["n_negative_confidence"])
    logger.info("Cross-ARI (baseline vs combined clustering): %.3f", baseline_vs_combined_ari)


if __name__ == "__main__":
    main()
