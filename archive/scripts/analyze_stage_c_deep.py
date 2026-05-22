"""Deep analytical pass over the Stage C consensus clustering.

Loads everything Stage C saved + the cached Stage B embedding and produces:

- **boundary_analysis.csv** — the 583 negative-confidence patients with
  their second-best cluster, confidence gap, and cohort.
- **boundary_migration_matrix.csv** — assigned × second-best count matrix.
- **cluster_feature_profile.csv** — per-cluster standardized profile for
  every Stage A feature.
- **cluster_compactness.csv** — per-cluster mean/median radius + density.
- **cohort_stratified_c{N}.csv** — cross-cohort feature spread inside the
  3 transdiagnostic clusters (3, 4, 5).
- **sub_cluster_c5.parquet** + similar for clusters 3 and 4 — hidden
  sub-structure analysis.
- **clinical_panel_c{N}.json** — minimum clinical-feature panel for each
  cluster (the leakage-safe successor to the legacy ``biomarker_panel_c{N}.json``;
  the default whitelist excludes the eight features that seed the Stage A
  transdiagnostic graph).
- **cluster_card_c{N}.md** (supplement) — deep narrative for each cluster.
- **figures/09_..** through **figures/14_..** — new figures.
- **deep_analysis_summary.json** — headline scalars.

Run:

    python scripts/analyze_stage_c_deep.py
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
from face_stratification.stage_c.clinical_panels import (
    default_clinical_feature_whitelist,
    discover_all_clinical_feature_panels,
)
from face_stratification.stage_c.deep_analysis import (
    analyze_boundary_patients,
    cluster_feature_profile,
    cohort_stratified_profile,
    compute_cluster_compactness,
    rebuild_coassociation_from_base,
    sub_cluster,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("stage_c_deep")


CSV_PATHS = {
    "bp": REPO / "data" / "BP.csv",
    "sz": REPO / "data" / "SZ.csv",
    "dr": REPO / "data" / "DR.csv",
    "asp": REPO / "data" / "ASP.csv",
}

OUT_DIR = REPO / "output" / "stratification" / "stage_c"
DEEP_DIR = OUT_DIR / "deep_analysis"
FIG_DIR = OUT_DIR / "figures"
EMBED_CACHE = REPO / "output" / "stratification" / "stage_b_review" / "embedding_cache"
TSNE_CACHE = REPO / "output" / "stratification" / "stage_b_review" / "projection_tsne.npy"

DEEP_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _load_stage_c_artifacts():
    logger.info("Loading Stage C artifacts + Stage B embedding + harmonized dataset")
    emb = PatientEmbedding.load(EMBED_CACHE)
    ds = build_harmonized_dataset(csv_paths=CSV_PATHS)
    cluster_labels = pd.read_parquet(OUT_DIR / "consensus_labels.parquet")["cluster"].astype(int)
    confidence = pd.read_parquet(OUT_DIR / "per_patient_confidence.parquet")["confidence"].astype(float)
    base_labels = pd.read_parquet(OUT_DIR / "base_clusterings.parquet")
    enrichment = pd.read_csv(OUT_DIR / "cluster_enrichment_top.csv")
    return emb, ds, cluster_labels, confidence, base_labels, enrichment


def main() -> None:
    t0 = time.time()
    emb, ds, cluster_labels, confidence, base_labels, enrichment = _load_stage_c_artifacts()

    # Align everything to the embedding index
    idx = emb.values.index
    cluster_labels = cluster_labels.loc[idx]
    confidence = confidence.loc[idx]
    # The harmonized metadata already has 'cohort' and 'patient_id' as columns
    # AND on the MultiIndex — drop the index level duplication to get a flat frame.
    metadata = ds.metadata.loc[idx].reset_index(drop=True)
    X = ds.X.loc[idx]
    cohort_labels = ds.metadata.loc[idx, "cohort"]

    logger.info("n_patients=%d  n_features=%d", len(idx), X.shape[1])

    # ─── Boundary analysis ─────────────────────────────────────────────────
    logger.info("Step 1/6: rebuilding co-association matrix from base clusterings")
    t = time.time()
    M = rebuild_coassociation_from_base(base_labels)
    logger.info("  matrix shape %s in %.1fs", M.shape, time.time() - t)

    logger.info("Step 2/6: boundary analysis")
    ba = analyze_boundary_patients(M, cluster_labels.reset_index(drop=True), confidence.reset_index(drop=True), metadata)
    logger.info(
        "  %d negative-confidence patients (%.1f%% of cohort)",
        ba.n_negative_conf, ba.negative_fraction * 100,
    )
    ba.assigned_to_second_best.to_csv(DEEP_DIR / "boundary_patients.csv", index=False)
    if not ba.migration_matrix.empty:
        ba.migration_matrix.to_csv(DEEP_DIR / "boundary_migration_matrix.csv")
    if not ba.by_cohort.empty:
        ba.by_cohort.to_csv(DEEP_DIR / "boundary_by_cohort.csv")
    # Free the big matrix before the heavier steps
    del M

    # Boundary figure — migration matrix as a heatmap
    if not ba.migration_matrix.empty:
        fig, ax = plt.subplots(figsize=(9, 7))
        mat = ba.migration_matrix
        # Ensure square even if some clusters are missing
        all_clusters = sorted(set(mat.index) | set(mat.columns))
        mat = mat.reindex(index=all_clusters, columns=all_clusters, fill_value=0)
        im = ax.imshow(mat.values, cmap="Reds")
        ax.set_xticks(np.arange(len(all_clusters)))
        ax.set_xticklabels([f"C{c}" for c in all_clusters])
        ax.set_yticks(np.arange(len(all_clusters)))
        ax.set_yticklabels([f"C{c}" for c in all_clusters])
        ax.set_xlabel("Second-best cluster (where the patient co-clusters more)")
        ax.set_ylabel("Assigned cluster")
        ax.set_title(f"Boundary patient migration ({ba.n_negative_conf} negative-confidence patients)")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat.values[i, j]
                if val > 0:
                    ax.text(j, i, int(val), ha="center", va="center",
                            color="white" if val > mat.values.max() / 2 else "black",
                            fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "09_boundary_migration.png", dpi=120)
        plt.close(fig)
        logger.info("  saved 09_boundary_migration.png")

    # ─── Cluster compactness ───────────────────────────────────────────────
    logger.info("Step 3/6: cluster compactness (embedding-space)")
    compactness = compute_cluster_compactness(emb.values, cluster_labels)
    compactness_df = pd.DataFrame([
        {
            "cluster": c.cluster_id,
            "n_patients": c.n_patients,
            "mean_radius": c.mean_radius,
            "median_radius": c.median_radius,
            "std_radius": c.std_radius,
            "max_radius": c.max_radius,
            "density": c.density,
        }
        for c in compactness.values()
    ])
    compactness_df.to_csv(DEEP_DIR / "cluster_compactness.csv", index=False)
    logger.info("  wrote cluster_compactness.csv")

    # ─── Per-cluster feature profile ───────────────────────────────────────
    logger.info("Step 4/6: per-cluster feature profile (z-score of cluster means)")
    profile = cluster_feature_profile(X, cluster_labels)
    profile.to_csv(DEEP_DIR / "cluster_feature_profile.csv", index=False)
    logger.info("  wrote cluster_feature_profile.csv (%d rows)", len(profile))

    # Figure 10: z-score heatmap of top features per cluster
    # Pick the 25 features with the largest max |z| across any cluster
    abs_z_max = profile.groupby("feature")["z_cluster_mean"].apply(lambda s: s.abs().max())
    top_features = abs_z_max.sort_values(ascending=False).head(25).index.tolist()
    pivot = (
        profile[profile["feature"].isin(top_features)]
        .pivot(index="feature", columns="cluster", values="z_cluster_mean")
        .loc[top_features]
    )
    fig, ax = plt.subplots(figsize=(9, 9))
    v = max(abs(pivot.values.min()), abs(pivot.values.max()))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([f"C{c}" for c in pivot.columns])
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel("cluster")
    ax.set_title(
        "Per-cluster feature profile — z-score of cluster mean\n"
        "(red = above global mean, blue = below)"
    )
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:+.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(pivot.values[i, j]) > v * 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_cluster_feature_zscores.png", dpi=120)
    plt.close(fig)
    logger.info("  saved 10_cluster_feature_zscores.png")

    # ─── Cohort-stratified profiles for transdiagnostic clusters ─────────
    logger.info("Step 5/6: cohort-stratified feature profiles (transdiagnostic clusters 3, 4, 5)")
    for target in (3, 4, 5):
        strat = cohort_stratified_profile(X, cluster_labels, cohort_labels, target)
        strat.to_csv(DEEP_DIR / f"cohort_stratified_c{target}.csv", index=False)
    # Pick the 10 features with the LARGEST cross-cohort spread in cluster 5
    # (those would argue for a forced grouping) and the 10 with smallest
    # (those argue for a genuine transdiagnostic phenotype)
    strat_c5 = pd.read_csv(DEEP_DIR / "cohort_stratified_c5.csv")
    strat_c5_clean = strat_c5[strat_c5["cross_cohort_std_z"].notna()].copy()
    strat_c5_clean = strat_c5_clean.sort_values("cross_cohort_std_z")
    tight = strat_c5_clean.head(10)[["feature", "cross_cohort_std_z", "bp_mean", "sz_mean", "dr_mean", "asp_mean"]]
    wide = strat_c5_clean.tail(10)[["feature", "cross_cohort_std_z", "bp_mean", "sz_mean", "dr_mean", "asp_mean"]]
    tight.to_csv(DEEP_DIR / "c5_features_tightest_across_cohorts.csv", index=False)
    wide.to_csv(DEEP_DIR / "c5_features_widest_across_cohorts.csv", index=False)
    logger.info("  wrote tight + wide feature lists for cluster 5")

    # Figure 11: cross-cohort spread histogram for cluster 5
    fig, ax = plt.subplots(figsize=(9, 5))
    spreads = strat_c5_clean["cross_cohort_std_z"].values
    ax.hist(spreads, bins=30, color="#1f77b4", alpha=0.85)
    ax.axvline(np.median(spreads), color="black", linestyle="--", label=f"median = {np.median(spreads):.3f}")
    ax.set_xlabel("Cross-cohort std of cluster mean (global-z units)")
    ax.set_ylabel("Number of features")
    ax.set_title(
        "Cluster 5 (4-cohort transdiagnostic) — cross-cohort feature spread\n"
        "Small = genuine transdiagnostic phenotype, large = forced grouping"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_c5_cross_cohort_spread.png", dpi=120)
    plt.close(fig)
    logger.info("  saved 11_c5_cross_cohort_spread.png")

    # ─── Sub-clustering analysis ───────────────────────────────────────────
    logger.info("Step 6/6: sub-clustering inside transdiagnostic clusters")
    sub_results = {}
    for target in (3, 4, 5):
        try:
            sc = sub_cluster(
                emb.values, cluster_labels, target_cluster=target, n_sub_clusters=3, method="kmeans"
            )
            sub_results[target] = sc
            sc.sub_labels.to_frame("sub_cluster").to_parquet(
                DEEP_DIR / f"sub_cluster_c{target}.parquet"
            )
            logger.info("  cluster %d sub-clusters: %s", target, sc.sub_sizes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("  sub-clustering of C%d failed: %s", target, exc)

    # Figure 12: sub-cluster cohort composition within cluster 5
    if 5 in sub_results:
        sc5 = sub_results[5]
        sub_labels = sc5.sub_labels
        sub_meta = metadata.set_index(["cohort", "patient_id"]).loc[sub_labels.index]
        sub_meta["sub_cluster"] = sub_labels.values
        sub_ct = pd.crosstab(sub_meta["sub_cluster"], sub_meta.index.get_level_values("cohort"), normalize="index")
        sub_sizes = sub_meta["sub_cluster"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(9, 5))
        sub_ct.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
        ax.set_xlabel("Sub-cluster of Cluster 5")
        ax.set_ylabel("Cohort fraction")
        ax.set_title(
            f"Cluster 5 sub-clustering (k=3)\n"
            f"sizes: " + ", ".join(f"S{i}={n}" for i, n in sub_sizes.items())
        )
        ax.legend(title="cohort", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "12_c5_sub_clusters.png", dpi=120)
        plt.close(fig)
        logger.info("  saved 12_c5_sub_clusters.png")

    # ─── Minimum clinical-feature panels ──────────────────────────────────
    # Sanitised default: exclude the eight embedding-input features that
    # seed the Stage A transdiagnostic graph (otherwise the panel trivially
    # recovers the cluster labels — see clinical_panels.EMBEDDING_INPUT_FEATURES).
    logger.info(
        "Clinical-feature panel discovery (sanitised whitelist, "
        "embedding inputs excluded)"
    )
    panels = discover_all_clinical_feature_panels(
        X, cluster_labels, cohort_labels,
        max_panel_size=6,
        feature_whitelist=default_clinical_feature_whitelist(
            exclude_embedding_inputs=True
        ),
    )
    for cid, panel in panels.items():
        with open(DEEP_DIR / f"clinical_panel_c{cid}.json", "w") as fh:
            json.dump(panel.as_dict(), fh, indent=2, default=str)
    logger.info("  wrote %d clinical-feature panels", len(panels))
    # Flag any cluster that was silently dropped for being too small.
    missing = sorted(set(int(c) for c in cluster_labels.unique() if c >= 0) - set(panels))
    if missing:
        logger.warning(
            "No clinical-feature panel for clusters %s — "
            "likely too small (< MIN_PANEL_POSITIVES).", missing,
        )

    # Figure 13: clinical-feature panel AUC bar chart
    if panels:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        cids = sorted(panels.keys())
        overall_aucs = [panels[c].overall_auc for c in cids]
        axes[0].bar([f"C{c}" for c in cids], overall_aucs, color="#1f77b4")
        axes[0].set_ylim(0.5, 1.0)
        axes[0].axhline(0.7, color="gray", linestyle="--", alpha=0.5, label="0.7 = fair")
        axes[0].axhline(0.8, color="black", linestyle="--", alpha=0.5, label="0.8 = good")
        axes[0].set_ylabel("Overall AUC (panel vs rest)")
        axes[0].set_title("Clinical-feature panel AUC per cluster\n(sanitised whitelist)")
        axes[0].legend()

        # Right panel: panel size
        sizes = [len(panels[c].panel_features) for c in cids]
        axes[1].bar([f"C{c}" for c in cids], sizes, color="#d62728")
        axes[1].set_ylabel("Panel size (features)")
        axes[1].set_title("Clinical-feature panel size per cluster")

        fig.tight_layout()
        fig.savefig(FIG_DIR / "13_clinical_panels.png", dpi=120)
        plt.close(fig)
        logger.info("  saved 13_clinical_panels.png")

    # Figure 14: t-SNE + per-cluster centroids labelled
    if TSNE_CACHE.is_file():
        coords = np.load(TSNE_CACHE)
        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.get_cmap("tab10")
        for i, c in enumerate(sorted(cluster_labels.unique())):
            m = cluster_labels.values == c
            ax.scatter(
                coords[m, 0], coords[m, 1],
                s=3, alpha=0.35, c=[cmap(i % 10)],
            )
            # Centroid label
            cx = coords[m, 0].mean()
            cy = coords[m, 1].mean()
            ax.annotate(
                f"C{c}\nn={int(m.sum())}",
                (cx, cy),
                fontsize=11, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor=cmap(i % 10)),
            )
        # Overlay boundary patients in red crosses
        neg_mask = confidence.values < 0
        ax.scatter(
            coords[neg_mask, 0], coords[neg_mask, 1],
            s=8, c="red", marker="x", alpha=0.6,
            label=f"boundary (conf < 0, n={int(neg_mask.sum())})",
        )
        ax.legend(loc="lower left", fontsize=9)
        ax.set_title("Stage C consensus clusters with centroids + boundary patients")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "14_tsne_with_boundaries.png", dpi=120)
        plt.close(fig)
        logger.info("  saved 14_tsne_with_boundaries.png")

    # Top migration flows — convert tuple keys to string for JSON
    top_flows = {}
    if not ba.migration_matrix.empty:
        stacked = ba.migration_matrix.stack().sort_values(ascending=False).head(5)
        for (src, dst), count in stacked.items():
            top_flows[f"C{int(src)} → C{int(dst)}"] = int(count)

    # ─── Summary JSON ──────────────────────────────────────────────────────
    summary = {
        "n_patients": int(len(idx)),
        "n_clusters": int(cluster_labels.nunique()),
        "boundary_analysis": {
            "n_negative_confidence": int(ba.n_negative_conf),
            "negative_fraction": float(ba.negative_fraction),
            "all_negative_from_cluster": (
                int(ba.assigned_to_second_best["assigned_cluster"].mode()[0])
                if not ba.assigned_to_second_best.empty
                else None
            ),
            "top_migration_flows": top_flows,
        },
        "cluster_compactness": {
            int(c.cluster_id): {
                "n_patients": c.n_patients,
                "mean_radius": c.mean_radius,
                "median_radius": c.median_radius,
                "std_radius": c.std_radius,
                "density": c.density,
            }
            for c in compactness.values()
        },
        "sub_clustering": {
            int(target): {
                "n_parent": sc.n_parent,
                "n_sub_clusters": sc.n_sub_clusters,
                "sub_sizes": sc.sub_sizes,
            }
            for target, sc in sub_results.items()
        },
        "clinical_feature_panels": {
            int(cid): {
                "size": len(p.panel_features),
                "overall_auc": p.overall_auc,
                "cohort_stratified_auc": p.cohort_stratified_auc,
                "panel_features": p.panel_features,
                "whitelist_excludes_embedding_inputs": (
                    p.whitelist_excludes_embedding_inputs
                ),
            }
            for cid, p in panels.items()
        },
        "clinical_feature_panels_skipped_clusters": missing,
        "runtime_seconds": float(time.time() - t0),
    }
    with open(DEEP_DIR / "deep_analysis_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.info("Deep analysis complete in %.1fs — results in %s", time.time() - t0, DEEP_DIR)


if __name__ == "__main__":
    main()
