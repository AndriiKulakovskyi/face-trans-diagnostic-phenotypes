"""Stage B2.5 — hyperparameter sweep for transdiagnostic-optimized GNN embeddings.

Runs a focused grid over (depth × temperature × edge-type filter) on the
GraphContrastive model and picks the configuration that maximizes a
transdiagnostic-weighted optimization score.

Primary grid (12 configs):
    depth ∈ {1, 2, 3}
    edge_filter ∈ {all, transdiagnostic_only}
    temperature ∈ {0.1, 0.5}

Supplementary zoom-in at the best primary config: vary hidden_dim and
edge_drop_prob to find a refined optimum.

Outputs under ``output/stratification/stage_b2/sweep/``:

- ``sweep_primary.csv``             — primary 12 × k_grid results
- ``sweep_supplementary.csv``       — supplementary zoom-in results
- ``best_config.json``              — the winning configuration + metrics
- ``figures/01_sweep_heatmap.png``  — depth × filter × temperature ×
                                      optimization score heatmap
- ``figures/02_transdiagnostic_vs_dsm.png`` — scatter of every config
                                       showing transdiagnostic score vs Cramér's V
- ``figures/03_best_loss_curve.png`` — training curve of the best config

After the sweep, trains the winning configuration for the full 150 epochs
and saves it as the canonical Stage B2.5 embedding.
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
from face_stratification.graph.patient_similarity import build_multiplex_graph
from face_stratification.harmonization.normalization import (
    fit_normalization,
    transform_normalization,
)
from face_stratification.stage_b2 import (
    StageB2GraphContrastive,
    SweepConfig,
    pick_best_transdiagnostic_config,
    run_sweep,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("stage_b2_sweep")


CSV_PATHS = {
    "bp": REPO / "data" / "BP.csv",
    "sz": REPO / "data" / "SZ.csv",
    "dr": REPO / "data" / "DR.csv",
    "asp": REPO / "data" / "ASP.csv",
}

OUT_DIR = REPO / "output" / "stratification" / "stage_b2" / "sweep"
FIG_DIR = OUT_DIR / "figures"
EMBED_CACHE = REPO / "output" / "stratification" / "stage_b_review" / "embedding_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ─── Sweep grids ─────────────────────────────────────────────────────────────


def primary_grid() -> list[SweepConfig]:
    """12 configs: 3 × 2 × 2 (depth × filter × temperature)."""
    configs = []
    for depth in (1, 2, 3):
        for edge_filter_name, include in [
            ("all", None),
            ("transdiagnostic_only", ("transdiagnostic",)),
        ]:
            for temperature in (0.1, 0.5):
                configs.append(SweepConfig(
                    model="contrastive",
                    n_layers=depth,
                    hidden_dim=64,
                    out_dim=32,
                    n_epochs=30,
                    learning_rate=5e-3,
                    dropout=0.1,
                    temperature=temperature,
                    p_edge=0.2,
                    p_feat=0.1,
                    include_edge_types=include,
                    name=f"primary_L{depth}_{edge_filter_name}_T{temperature}",
                ))
    return configs


def supplementary_grid(best: dict) -> list[SweepConfig]:
    """Zoom in around the best primary config."""
    base_include = None
    if "transdiagnostic" in str(best.get("include_edge_types", "")):
        base_include = ("transdiagnostic",)
    elif str(best.get("include_edge_types", "all")) != "all":
        base_include = tuple(str(best["include_edge_types"]).split(","))

    configs = []
    for hidden_dim in (32, 64, 128):
        for p_edge in (0.1, 0.3):
            configs.append(SweepConfig(
                model="contrastive",
                n_layers=int(best["n_layers"]),
                hidden_dim=hidden_dim,
                out_dim=32,
                n_epochs=50,
                learning_rate=5e-3,
                dropout=0.1,
                temperature=float(best["temperature"]),
                p_edge=p_edge,
                p_feat=0.1,
                include_edge_types=base_include,
                name=f"supp_h{hidden_dim}_pE{p_edge}",
            ))
    return configs


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    t0 = time.time()

    logger.info("Loading harmonized dataset + Stage B cached embedding")
    ds = build_harmonized_dataset(csv_paths=CSV_PATHS)
    stage_b = PatientEmbedding.load(EMBED_CACHE)
    cohort_labels = ds.metadata.loc[stage_b.values.index, "cohort"].to_numpy()

    logger.info("Building multiplex graph")
    stats = fit_normalization(ds.X, ds.schema)
    Xn = transform_normalization(ds.X, stats)
    G, _blocks, _td = build_multiplex_graph(Xn, ds.schema, k=10, metadata=ds.metadata)
    logger.info("  multiplex graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    # ─── Primary sweep ───────────────────────────────────────────────────
    primary_configs = primary_grid()
    logger.info("=" * 70)
    logger.info("Primary sweep: %d configurations × k ∈ [5,6,7,8]", len(primary_configs))
    logger.info("=" * 70)
    t_primary = time.time()
    primary_df = run_sweep(
        ds, G, stage_b, cohort_labels,
        primary_configs,
        k_grid=(5, 6, 7, 8),
    )
    logger.info("Primary sweep done in %.1f min", (time.time() - t_primary) / 60)
    primary_df.to_csv(OUT_DIR / "sweep_primary.csv", index=False)

    # Pick best primary config
    best_primary = pick_best_transdiagnostic_config(primary_df)
    logger.info("=" * 70)
    logger.info("Best primary config: %s @ k=%d", best_primary["config_id"], best_primary["k"])
    logger.info("  silhouette=%.3f  DB=%.3f  ARI=%.3f  td_score=%.3f  CramérV=%.3f  opt=%.3f",
                best_primary["silhouette"], best_primary["davies_bouldin"],
                best_primary["ari_vs_cohort"], best_primary["transdiagnostic_score"],
                best_primary["cramers_v"], best_primary["optimization_score"])
    logger.info("=" * 70)

    # ─── Supplementary sweep ─────────────────────────────────────────────
    supp_configs = supplementary_grid(best_primary)
    logger.info("Supplementary sweep: %d configurations × k ∈ [best ± 1]",
                len(supp_configs))
    k_grid_supp = tuple(
        k for k in (int(best_primary["k"]) - 1, int(best_primary["k"]), int(best_primary["k"]) + 1)
        if 3 <= k <= 10
    )
    t_supp = time.time()
    supp_df = run_sweep(
        ds, G, stage_b, cohort_labels,
        supp_configs,
        k_grid=k_grid_supp,
    )
    logger.info("Supplementary sweep done in %.1f min", (time.time() - t_supp) / 60)
    supp_df.to_csv(OUT_DIR / "sweep_supplementary.csv", index=False)

    # Combined results for "best of all"
    all_df = pd.concat([primary_df, supp_df], ignore_index=True)
    all_df.to_csv(OUT_DIR / "sweep_all.csv", index=False)
    best_overall = pick_best_transdiagnostic_config(all_df)
    logger.info("=" * 70)
    logger.info("Best overall config: %s @ k=%d", best_overall["config_id"], best_overall["k"])
    logger.info("  silhouette=%.3f  DB=%.3f  ARI=%.3f  td_score=%.3f  CramérV=%.3f  opt=%.3f",
                best_overall["silhouette"], best_overall["davies_bouldin"],
                best_overall["ari_vs_cohort"], best_overall["transdiagnostic_score"],
                best_overall["cramers_v"], best_overall["optimization_score"])

    # Compare to Stage B2 default baseline (which was also evaluated in the primary sweep
    # as depth=2, temperature=0.5, filter=all)
    baseline_row = primary_df[
        (primary_df["n_layers"] == 2)
        & (primary_df["include_edge_types"] == "all")
        & (primary_df["temperature"] == 0.5)
    ].sort_values("optimization_score", ascending=False).iloc[0].to_dict()

    with open(OUT_DIR / "best_config.json", "w") as fh:
        json.dump({
            "best_overall": best_overall,
            "best_primary": best_primary,
            "stage_b2_default_baseline": baseline_row,
            "primary_configs": len(primary_configs),
            "supplementary_configs": len(supp_configs),
        }, fh, indent=2, default=str)

    # ─── Figures ─────────────────────────────────────────────────────────
    # Figure 1: Primary sweep heatmap — depth × (filter, temperature) @ k of best score
    try:
        primary_best_k = (
            primary_df.sort_values("optimization_score", ascending=False)
            .drop_duplicates("config_id")
            .copy()
        )
        primary_best_k["edge_filter"] = primary_best_k["include_edge_types"]
        primary_best_k["temp_label"] = primary_best_k["temperature"].astype(str)
        primary_best_k["col_label"] = primary_best_k["edge_filter"] + " | T=" + primary_best_k["temp_label"]
        pivot = primary_best_k.pivot_table(
            index="n_layers",
            columns="col_label",
            values="optimization_score",
            aggfunc="max",
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels([f"L{int(i)}" for i in pivot.index])
        ax.set_xlabel("edge filter | temperature")
        ax.set_ylabel("GCN depth")
        ax.set_title("Stage B2.5 primary sweep — transdiagnostic optimization score")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v < pivot.values.mean() else "black",
                            fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "01_sweep_heatmap.png", dpi=120)
        plt.close(fig)
        logger.info("  saved 01_sweep_heatmap.png")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Figure 1 failed: %s", exc)

    # Figure 2: Transdiagnostic score vs Cramér's V scatter
    try:
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(
            all_df["cramers_v"], all_df["transdiagnostic_score"],
            s=60, alpha=0.6, c="#1f77b4", edgecolor="black",
        )
        # Highlight best
        best_row_df = all_df[all_df["optimization_score"] == best_overall["optimization_score"]].iloc[0]
        ax.scatter(
            best_row_df["cramers_v"], best_row_df["transdiagnostic_score"],
            s=200, facecolor="red", edgecolor="black", marker="*",
            label=f"best: {best_overall['config_id']}",
            zorder=5,
        )
        # Highlight Stage B2 default
        ax.scatter(
            baseline_row["cramers_v"], baseline_row["transdiagnostic_score"],
            s=200, facecolor="orange", edgecolor="black", marker="s",
            label=f"Stage B2 default (L=2, T=0.5, all)",
            zorder=5,
        )
        ax.set_xlabel("Cramér's V vs DSM cohort (lower = less DSM-aligned)")
        ax.set_ylabel("Mean transdiagnostic score (higher = more cohort-mixing)")
        ax.set_title(
            "Stage B2.5 — transdiagnostic score vs DSM alignment across configurations\n"
            "(upper-left is best: high transdiagnostic content + low DSM redundancy)"
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "02_transdiagnostic_vs_dsm.png", dpi=120)
        plt.close(fig)
        logger.info("  saved 02_transdiagnostic_vs_dsm.png")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Figure 2 failed: %s", exc)

    # ─── Train the best config for the full 150 epochs and save ─────────
    logger.info("Training the best overall config for 150 epochs (canonical B2.5)")
    best_include = None
    if str(best_overall["include_edge_types"]) != "all":
        parts = str(best_overall["include_edge_types"]).split(",")
        best_include = tuple(parts)

    final_model = StageB2GraphContrastive(
        hidden_dim=int(best_overall["hidden_dim"]),
        out_dim=int(best_overall["out_dim"]),
        n_layers=int(best_overall["n_layers"]),
        n_epochs=150,
        learning_rate=5e-3,
        dropout=0.1,
        seed=0,
        feature_source="composite",
        temperature=float(best_overall["temperature"]),
        p_edge=float(best_overall["p_edge"]),
        p_feat=0.1,
        include_edge_types=best_include,
        exclude_edge_types=(),
    )
    final_model.fit(ds, graph=G)
    final_emb = final_model.transform()
    final_emb.save(OUT_DIR / "embedding_best")
    logger.info("  saved final embedding: %d × %d", final_emb.n_patients, final_emb.dim)

    # Figure 3: training curve of the best config
    hist = pd.DataFrame(final_model._training_history)  # noqa: SLF001
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hist["epoch"], hist["loss"], "o-", color="#1f77b4")
    ax.set_xlabel("epoch")
    ax.set_ylabel("NT-Xent loss")
    ax.set_title(f"Stage B2.5 best config — training curve\n{best_overall['config_id']}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_best_loss_curve.png", dpi=120)
    plt.close(fig)
    logger.info("  saved 03_best_loss_curve.png")

    # ─── Final summary ───────────────────────────────────────────────────
    summary = {
        "n_patients": int(final_emb.n_patients),
        "n_primary_configs": len(primary_configs),
        "n_supplementary_configs": len(supp_configs),
        "k_grid_primary": [5, 6, 7, 8],
        "k_grid_supplementary": list(k_grid_supp),
        "best_overall": best_overall,
        "stage_b2_default_baseline": baseline_row,
        "improvement_over_baseline": {
            "optimization_score": best_overall["optimization_score"] - baseline_row["optimization_score"],
            "transdiagnostic_score": best_overall["transdiagnostic_score"] - baseline_row["transdiagnostic_score"],
            "cramers_v_delta": best_overall["cramers_v"] - baseline_row["cramers_v"],
        },
        "final_embedding_path": str(OUT_DIR / "embedding_best"),
        "total_runtime_seconds": float(time.time() - t0),
    }
    with open(OUT_DIR / "stage_b2_5_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.info("=" * 70)
    logger.info("Stage B2.5 sweep complete in %.1f min", (time.time() - t0) / 60)
    logger.info("Stage B2 default:    td=%.3f  V=%.3f  opt=%.3f",
                baseline_row["transdiagnostic_score"], baseline_row["cramers_v"],
                baseline_row["optimization_score"])
    logger.info("Best B2.5 config:    td=%.3f  V=%.3f  opt=%.3f",
                best_overall["transdiagnostic_score"], best_overall["cramers_v"],
                best_overall["optimization_score"])
    logger.info("Δ optimization_score: %+.3f",
                best_overall["optimization_score"] - baseline_row["optimization_score"])


if __name__ == "__main__":
    main()
