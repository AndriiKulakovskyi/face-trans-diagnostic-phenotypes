"""Generate publication-grade Stage B (+ B2, + B2.5) figures.

Outputs PNG (300 DPI) + PDF versions to docs/face_stratification/stage_b/.

Figures produced
----------------
fig_b1_embedding_composition  — 56-dim composite view + normalization ablation.
fig_b2_kmeans_sweep           — silhouette / cohort-entropy / ARI vs k (Stage B).
fig_b3_stage_b_clusters       — Stage B k=8 cluster×cohort row+col heatmaps.
fig_b4_enrichment             — top enriched features per Stage B cluster.
fig_b5_gnn_training           — Stage B2 GAE + contrastive training curves.
fig_b6_stage_b2_clusters      — Stage B2 k=7 cluster×cohort row+col heatmaps.
fig_b7_sweep_tradeoff         — B2.5 sweep scatter (DSM alignment × transdiag).
fig_b8_sweep_levers           — sweep effect of edge-type / depth / temperature.
fig_b9_three_way_comparison   — B vs B+B2 vs B+B2.5 core metrics + B2.5 heatmap.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "face_stratification" / "stage_b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STAGE_B_DIR = ROOT / "output" / "stratification" / "stage_b_review"
STAGE_B2_DIR = ROOT / "output" / "stratification" / "stage_b2"
SWEEP_DIR = STAGE_B2_DIR / "sweep"
SWEEP_BEST_C = SWEEP_DIR / "stage_c_on_best"

# ─── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.frameon": False,
    }
)

COHORT_PALETTE = {
    "bp": "#2166AC",
    "sz": "#B2182B",
    "asp": "#E08214",
    "dr": "#1B7837",
}
COHORT_LABELS = {"bp": "BP", "sz": "SZ", "asp": "ASP", "dr": "DR"}
COHORT_ORDER = ["bp", "sz", "asp", "dr"]

CLUSTER_PALETTE = sns.color_palette("tab10", 10)

STAGE_COLORS = {
    "B": "#4C78A8",      # Stage B baseline
    "B+B2": "#D62728",    # Stage B + Stage B2 (DSM-aligned)
    "B+B2.5": "#2CA02C",  # Stage B + Stage B2.5 (transdiagnostic)
}


def savefig(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png / .pdf")


# ─── Figure 1 — Embedding composition + normalization ablation ────────────────
def fig_embedding_composition() -> None:
    # 56d composite breakdown
    parts = [
        ("Trans\nPCA", 8, "#4C78A8"),
        ("Transdiagnostic\nspectral", 16, "#F58518"),
        ("Multiplex spectral", 32, "#54A24B"),
    ]
    ablation = pd.read_csv(STAGE_B_DIR / "ablation_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), gridspec_kw={"width_ratios": [1, 1.25]})

    # (a) Composition horizontal stacked bar
    ax = axes[0]
    left = 0
    for label, d, color in parts:
        ax.barh(
            ["Stage B\ncomposite\nembedding"],
            d,
            left=left,
            color=color,
            edgecolor="white",
            linewidth=1.5,
        )
        fontsize = 8 if d < 12 else 10
        ax.text(
            left + d / 2,
            0,
            f"{label}\n{d} d",
            ha="center",
            va="center",
            fontsize=fontsize,
            color="white",
            fontweight="bold",
        )
        left += d
    ax.set_xlim(0, 56)
    ax.set_xlabel("Embedding dimensions")
    ax.set_title("(a) 56-dimensional composite embedding")
    ax.set_yticks([])
    sns.despine(ax=ax, left=True)

    # (b) Normalization ablation — global vs per-cohort (silhouette & V)
    ax = axes[1]
    metrics = pd.DataFrame(
        {
            "normalization": ["Global", "Per-cohort", "Global", "Per-cohort"],
            "metric": ["Silhouette", "Silhouette", "DSM alignment\n(NMI)", "DSM alignment\n(NMI)"],
            "value": [
                float(ablation.loc[0, "silhouette"]),
                float(ablation.loc[1, "silhouette"]),
                float(ablation.loc[0, "nmi_vs_cohort"]),
                float(ablation.loc[1, "nmi_vs_cohort"]),
            ],
        }
    )
    palette = {"Global": "#4C78A8", "Per-cohort": "#F58518"}
    sns.barplot(
        data=metrics,
        x="metric",
        y="value",
        hue="normalization",
        palette=palette,
        ax=ax,
        edgecolor="white",
    )
    for patch in ax.patches:
        h = patch.get_height()
        if np.isnan(h) or h == 0:
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            h + 0.005,
            f"{h:.3f}",
            ha="center",
            fontsize=9,
            color="#333",
        )
    ax.set_ylim(0, 0.52)
    ax.set_ylabel("Metric value")
    ax.set_xlabel("")
    ax.set_title("(b) Normalization ablation (k=8)")
    ax.legend(title="", loc="upper right")
    sns.despine(ax=ax)

    fig.suptitle(
        "Stage B — Composite embedding & normalization choice",
        fontsize=14,
        fontweight="bold",
        y=1.04,
    )
    fig.tight_layout()
    savefig(fig, "fig_b1_embedding_composition")


# ─── Figure 2 — k-means sweep ─────────────────────────────────────────────────
def fig_kmeans_sweep() -> None:
    df = pd.read_csv(STAGE_B_DIR / "kmeans_sweep.csv")
    best_k = 8

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))

    # (a) Silhouette
    ax = axes[0]
    ax.plot(df["k"], df["silhouette"], marker="o", linewidth=2, color="#4C78A8")
    ax.axvline(best_k, color="#D62728", linestyle="--", linewidth=1.2, label=f"selected k={best_k}")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Silhouette (cosine)")
    ax.set_title("(a) Cluster compactness")
    ax.legend(loc="lower right", fontsize=9)
    sns.despine(ax=ax)

    # (b) Cohort entropy (higher = more transdiagnostic mixing)
    ax = axes[1]
    ax.plot(df["k"], df["cohort_entropy_mean"], marker="s", linewidth=2, color="#54A24B")
    ax.axvline(best_k, color="#D62728", linestyle="--", linewidth=1.2)
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Mean cohort entropy (bits)")
    ax.set_title("(b) Transdiagnostic content")
    ax.text(
        df["k"].iloc[-1],
        df["cohort_entropy_mean"].iloc[-1],
        f" max = 2.0",
        fontsize=9,
        color="#555",
    )
    sns.despine(ax=ax)

    # (c) ARI and NMI vs DSM
    ax = axes[2]
    ax.plot(df["k"], df["ari"], marker="^", linewidth=2, color="#B2182B", label="ARI vs DSM")
    ax.plot(df["k"], df["nmi"], marker="v", linewidth=2, color="#E08214", label="NMI vs DSM")
    ax.axvline(best_k, color="#D62728", linestyle="--", linewidth=1.2)
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Label agreement")
    ax.set_title("(c) DSM alignment (kept low)")
    ax.legend(loc="upper left", fontsize=9)
    sns.despine(ax=ax)

    fig.suptitle(
        "Stage B — k-means sweep over the composite embedding (N = 11,014)",
        fontsize=14,
        fontweight="bold",
        y=1.05,
    )
    fig.tight_layout()
    savefig(fig, "fig_b2_kmeans_sweep")


# ─── Figure 3 — Stage B cluster × cohort heatmap ──────────────────────────────
def _cluster_cohort_heatmap(
    rows_csv: Path,
    cols_csv: Path,
    ax_row: plt.Axes,
    ax_col: plt.Axes,
    cluster_label_fn=lambda i, n: f"C{i}\n(n={n})",
    n_col: pd.Series | None = None,
) -> None:
    row_norm = pd.read_csv(rows_csv, index_col=0)
    col_norm = pd.read_csv(cols_csv, index_col=0)

    # Ensure cohort order
    row_norm = row_norm[COHORT_ORDER]
    col_norm = col_norm[COHORT_ORDER]

    # Row-normalized heatmap
    sns.heatmap(
        row_norm.values,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar_kws={"label": "Row proportion"},
        xticklabels=[COHORT_LABELS[c] for c in COHORT_ORDER],
        yticklabels=[cluster_label_fn(i, int(n_col.loc[i]) if n_col is not None else 0) for i in row_norm.index],
        ax=ax_row,
        linewidths=0.4,
        linecolor="white",
        vmin=0,
        vmax=1.0,
    )
    ax_row.set_title("(a) Row-normalized — cohort mix inside each cluster")
    ax_row.set_xlabel("Cohort")
    ax_row.set_ylabel("Cluster")
    plt.setp(ax_row.get_yticklabels(), rotation=0)

    # Column-normalized heatmap
    sns.heatmap(
        col_norm.values,
        annot=True,
        fmt=".2f",
        cmap="OrRd",
        cbar_kws={"label": "Column proportion"},
        xticklabels=[COHORT_LABELS[c] for c in COHORT_ORDER],
        yticklabels=[f"C{i}" for i in col_norm.index],
        ax=ax_col,
        linewidths=0.4,
        linecolor="white",
        vmin=0,
        vmax=max(0.45, col_norm.values.max()),
    )
    ax_col.set_title("(b) Column-normalized — where each cohort lands")
    ax_col.set_xlabel("Cohort")
    ax_col.set_ylabel("")
    plt.setp(ax_col.get_yticklabels(), rotation=0)


def fig_stage_b_clusters() -> None:
    labels = pd.read_parquet(STAGE_B_DIR / "cluster_labels.parquet")
    n_per_cluster = labels["cluster"].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    _cluster_cohort_heatmap(
        STAGE_B_DIR / "cluster_cohort_contingency_rows.csv",
        STAGE_B_DIR / "cluster_cohort_contingency_cols.csv",
        axes[0],
        axes[1],
        n_col=n_per_cluster,
    )
    fig.suptitle(
        "Stage B — k-means k=8 cluster × cohort composition\n"
        "(silhouette 0.451, bootstrap ARI 0.957 ± 0.054, Cramér's V ≈ 0.30)",
        fontsize=14,
        fontweight="bold",
        y=1.04,
    )
    fig.tight_layout()
    savefig(fig, "fig_b3_stage_b_clusters")


# ─── Figure 4 — Feature enrichment per cluster ────────────────────────────────
def fig_enrichment() -> None:
    df = pd.read_csv(STAGE_B_DIR / "cluster_enrichment_top.csv")

    # Pick top 8 by |effect| per cluster
    top = (
        df.sort_values(["cluster", "abs_effect"], ascending=[True, False])
        .groupby("cluster")
        .head(8)
        .reset_index(drop=True)
    )

    n_clusters = top["cluster"].nunique()
    fig, axes = plt.subplots(
        2, 4, figsize=(15, 9), sharex=True
    )
    axes = axes.flatten()

    for i, (cluster, g) in enumerate(top.groupby("cluster")):
        ax = axes[i]
        g = g.sort_values("effect_rank_biserial")
        colors = ["#1B7837" if e > 0 else "#B2182B" for e in g["effect_rank_biserial"]]
        ax.barh(
            [fid.replace("inst_", "").replace("_total", "").replace("_", " ") for fid in g["feature_id"]],
            g["effect_rank_biserial"],
            color=colors,
            edgecolor="white",
        )
        ax.axvline(0, color="#333", linewidth=0.6)
        ax.set_xlim(-1.0, 1.0)
        ax.set_title(f"Cluster {cluster}", fontsize=11)
        ax.tick_params(axis="y", labelsize=8)
        sns.despine(ax=ax, left=True)

    # Hide any unused axes
    for j in range(n_clusters, len(axes)):
        axes[j].axis("off")

    for ax in axes[-4:]:
        ax.set_xlabel("Rank-biserial effect")

    legend_handles = [
        mpatches.Patch(color="#B2182B", label="Higher inside cluster (↓ value = more pathological)"),
        mpatches.Patch(color="#1B7837", label="Lower inside cluster"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
    )

    fig.suptitle(
        "Stage B — Top enriched features per cluster (BH q<0.05, N = 11,014)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    savefig(fig, "fig_b4_enrichment")


# ─── Figure 5 — Stage B2 GNN training curves ──────────────────────────────────
def fig_gnn_training() -> None:
    history = json.loads((STAGE_B2_DIR / "training_history.json").read_text())
    gae = pd.DataFrame(history["gae"])
    con = pd.DataFrame(history["contrastive"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    # (a) GAE loss + pos/neg gap
    ax = axes[0]
    color1 = "#4C78A8"
    ax.plot(gae["epoch"], gae["loss"], marker="o", color=color1, linewidth=2, label="BCE loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss", color=color1)
    ax.tick_params(axis="y", labelcolor=color1)
    ax.set_title("(a) Graph Auto-Encoder (GAE)")

    ax2 = ax.twinx()
    ax2.plot(
        gae["epoch"],
        gae["gap"],
        marker="s",
        color="#D62728",
        linewidth=2,
        label="pos − neg gap",
    )
    ax2.set_ylabel("Positive − negative edge gap", color="#D62728")
    ax2.tick_params(axis="y", labelcolor="#D62728")
    ax2.grid(False)
    ax2.set_ylim(0, 0.18)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    # (b) Contrastive NT-Xent loss
    ax = axes[1]
    ax.plot(con["epoch"], con["loss"], marker="o", color="#54A24B", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NT-Xent loss")
    ax.set_title("(b) Graph Contrastive Learning")
    sns.despine(ax=ax)

    fig.suptitle(
        "Stage B2 — Deep GNN view training (150 epochs, Adam, GCN backbone)",
        fontsize=14,
        fontweight="bold",
        y=1.04,
    )
    fig.tight_layout()
    savefig(fig, "fig_b5_gnn_training")


# ─── Figure 6 — Stage B2 cluster × cohort heatmap (k=7) ───────────────────────
def fig_stage_b2_clusters() -> None:
    cont_dir = STAGE_B2_DIR / "stage_c_on_combined"
    # Compute contingency from parquet labels
    labels = pd.read_parquet(cont_dir / "consensus_labels.parquet")
    meta = labels.reset_index()
    meta["cohort"] = meta["cohort"].astype(str)
    contingency = pd.crosstab(meta["cluster"], meta["cohort"])
    contingency = contingency[COHORT_ORDER]
    row_norm = contingency.div(contingency.sum(axis=1), axis=0)
    col_norm = contingency.div(contingency.sum(axis=0), axis=1)

    n_per = labels["cluster"].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    sns.heatmap(
        row_norm.values,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar_kws={"label": "Row proportion"},
        xticklabels=[COHORT_LABELS[c] for c in COHORT_ORDER],
        yticklabels=[f"C{i}\n(n={int(n_per.loc[i])})" for i in row_norm.index],
        ax=axes[0],
        linewidths=0.4,
        linecolor="white",
        vmin=0,
        vmax=1.0,
    )
    axes[0].set_title("(a) Row-normalized")
    axes[0].set_xlabel("Cohort")
    axes[0].set_ylabel("Stage B2 cluster")
    plt.setp(axes[0].get_yticklabels(), rotation=0)

    sns.heatmap(
        col_norm.values,
        annot=True,
        fmt=".2f",
        cmap="OrRd",
        cbar_kws={"label": "Column proportion"},
        xticklabels=[COHORT_LABELS[c] for c in COHORT_ORDER],
        yticklabels=[f"C{i}" for i in col_norm.index],
        ax=axes[1],
        linewidths=0.4,
        linecolor="white",
        vmin=0,
        vmax=max(0.45, col_norm.values.max()),
    )
    axes[1].set_title("(b) Column-normalized")
    axes[1].set_xlabel("Cohort")
    axes[1].set_ylabel("")
    plt.setp(axes[1].get_yticklabels(), rotation=0)

    fig.suptitle(
        "Stage B2 — Stage B + GAE + GraphCL (120 d) → k=7 consensus clusters\n"
        "silhouette 0.480, Cramér's V 0.599 (DSM-aligned), boundary reduction 87%",
        fontsize=14,
        fontweight="bold",
        y=1.05,
    )
    fig.tight_layout()
    savefig(fig, "fig_b6_stage_b2_clusters")


# ─── Figure 7 — Stage B2.5 sweep tradeoff scatter ─────────────────────────────
def fig_sweep_tradeoff() -> None:
    df = pd.read_csv(SWEEP_DIR / "sweep_all.csv")
    # Aggregate per config (mean across k) for a cleaner scatter
    per_cfg = (
        df.groupby(
            [
                "config_id",
                "include_edge_types",
                "n_layers",
                "temperature",
            ],
            as_index=False,
        )
        .agg(
            transdiag=("transdiagnostic_score", "mean"),
            dsm=("dsm_score", "mean"),
            opt=("optimization_score", "mean"),
            silhouette=("silhouette", "mean"),
        )
    )

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    markers = {1: "o", 2: "s", 3: "^"}
    edge_palette = {"all": "#D62728", "transdiagnostic": "#2CA02C"}
    temp_size = {0.1: 80, 0.5: 220}

    for _, row in per_cfg.iterrows():
        ax.scatter(
            row["dsm"],
            row["transdiag"],
            color=edge_palette[row["include_edge_types"]],
            marker=markers[int(row["n_layers"])],
            s=temp_size[float(row["temperature"])],
            edgecolor="white",
            linewidth=1.2,
            alpha=0.9,
        )

    # Highlight best and baseline
    summary = json.loads((SWEEP_DIR / "stage_b2_5_summary.json").read_text())
    best = summary["best_overall"]
    base = summary["stage_b2_default_baseline"]

    ax.scatter(
        best["dsm_score"],
        best["transdiagnostic_score"],
        s=380,
        facecolors="none",
        edgecolors="#1B7837",
        linewidth=2.2,
        zorder=5,
    )
    ax.annotate(
        "best config\nL=3, transdiag-only, T=0.5",
        xy=(best["dsm_score"], best["transdiagnostic_score"]),
        xytext=(best["dsm_score"] + 0.01, best["transdiagnostic_score"] + 0.035),
        fontsize=9,
        ha="left",
        arrowprops=dict(arrowstyle="->", color="#1B7837", lw=1.2),
        color="#1B7837",
        fontweight="bold",
    )
    ax.annotate(
        "Stage B2 default\nL=2, all edges, T=0.5",
        xy=(base["dsm_score"], base["transdiagnostic_score"]),
        xytext=(base["dsm_score"] + 0.005, base["transdiagnostic_score"] - 0.055),
        fontsize=9,
        ha="left",
        arrowprops=dict(arrowstyle="->", color="#B2182B", lw=1.2),
        color="#B2182B",
        fontweight="bold",
    )

    # Set axis limits first so everything fits
    ax.set_xlim(0.37, 0.58)
    ax.set_ylim(0.52, 0.72)

    # Ideal region rectangle
    ax.add_patch(
        mpatches.Rectangle(
            (0.38, 0.64),
            0.06,
            0.07,
            linewidth=1.3,
            edgecolor="#555",
            facecolor="none",
            linestyle=":",
        )
    )
    ax.text(
        0.41,
        0.715,
        "target region\n↑transdiag · ↓DSM",
        ha="center",
        fontsize=8,
        color="#555",
        style="italic",
    )

    ax.set_xlabel("DSM alignment (Cramér's V — lower is better)")
    ax.set_ylabel("Transdiagnostic score (higher is better)")
    ax.set_title("Stage B2.5 — architecture sweep (18 configs × 4 ks)")

    # Legend: edge type
    edge_handles = [
        mpatches.Patch(color="#D62728", label="All edges"),
        mpatches.Patch(color="#2CA02C", label="Transdiagnostic-only edges"),
    ]
    depth_handles = [
        plt.Line2D([], [], marker=markers[d], linestyle="", color="gray", label=f"L={d}")
        for d in [1, 2, 3]
    ]
    temp_handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color="gray",
            markersize=np.sqrt(temp_size[t]) * 0.75,
            label=f"T={t}",
        )
        for t in [0.1, 0.5]
    ]
    leg1 = ax.legend(
        handles=edge_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        title="Edge set",
        fontsize=8,
        title_fontsize=9,
    )
    leg2 = ax.legend(
        handles=depth_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.72),
        title="GCN depth",
        fontsize=8,
        title_fontsize=9,
    )
    leg3 = ax.legend(
        handles=temp_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.44),
        title="Temperature",
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(leg1)
    ax.add_artist(leg2)

    sns.despine(ax=ax)
    savefig(fig, "fig_b7_sweep_tradeoff")


# ─── Figure 8 — Stage B2.5 sweep levers ───────────────────────────────────────
def fig_sweep_levers() -> None:
    df = pd.read_csv(SWEEP_DIR / "sweep_all.csv")
    # Focus on k that maximizes optimization_score per config
    best_per_cfg = df.loc[df.groupby("config_id")["optimization_score"].idxmax()].reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # (a) Edge-type effect
    ax = axes[0]
    et = best_per_cfg.groupby("include_edge_types")[["dsm_score", "transdiagnostic_score"]].mean().reset_index()
    et_plot = et.melt(id_vars="include_edge_types", var_name="metric", value_name="value")
    et_plot["metric"] = et_plot["metric"].map(
        {"dsm_score": "DSM alignment\n(↓ better)", "transdiagnostic_score": "Transdiag\n(↑ better)"}
    )
    et_plot["include_edge_types"] = et_plot["include_edge_types"].map(
        {"all": "All edges", "transdiagnostic": "Transdiag-only"}
    )
    sns.barplot(
        data=et_plot,
        x="metric",
        y="value",
        hue="include_edge_types",
        palette={"All edges": "#D62728", "Transdiag-only": "#2CA02C"},
        ax=ax,
        edgecolor="white",
    )
    for p in ax.patches:
        h = p.get_height()
        if np.isnan(h) or h == 0:
            continue
        ax.text(p.get_x() + p.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", fontsize=8)
    ax.set_ylabel("Metric")
    ax.set_xlabel("")
    ax.set_title("(a) Edge-type filtering (dominant lever)")
    ax.legend(loc="upper right", fontsize=8, title="")
    ax.set_ylim(0, 0.85)
    sns.despine(ax=ax)

    # (b) Depth effect (transdiag-only)
    ax = axes[1]
    td = best_per_cfg[best_per_cfg["include_edge_types"] == "transdiagnostic"]
    dp = td.groupby("n_layers")[["optimization_score"]].mean().reset_index()
    sns.barplot(
        data=dp,
        x="n_layers",
        y="optimization_score",
        hue="n_layers",
        palette="viridis",
        ax=ax,
        edgecolor="white",
        legend=False,
    )
    for p in ax.patches:
        h = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2, h + 0.005, f"{h:.3f}", ha="center", fontsize=8)
    ax.set_ylim(2.55, 2.9)
    ax.set_xlabel("GCN depth (layers)")
    ax.set_ylabel("Optimization score")
    ax.set_title("(b) Depth effect (transdiag-only)")
    sns.despine(ax=ax)

    # (c) Temperature effect (transdiag-only)
    ax = axes[2]
    tp = td.groupby("temperature")[["optimization_score"]].mean().reset_index()
    sns.barplot(
        data=tp,
        x="temperature",
        y="optimization_score",
        hue="temperature",
        palette="rocket",
        ax=ax,
        edgecolor="white",
        legend=False,
    )
    for p in ax.patches:
        h = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2, h + 0.005, f"{h:.3f}", ha="center", fontsize=8)
    ax.set_ylim(2.55, 2.9)
    ax.set_xlabel("GraphCL temperature")
    ax.set_ylabel("")
    ax.set_title("(c) Temperature effect")
    sns.despine(ax=ax)

    fig.suptitle(
        "Stage B2.5 — Architecture sweep: which levers control transdiagnostic content?",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    savefig(fig, "fig_b8_sweep_levers")


# ─── Figure 9 — Three-way stage comparison + Stage B+B2.5 heatmap ─────────────
def fig_three_way_comparison() -> None:
    # Load metrics
    b_summary = json.loads((STAGE_B_DIR / "review_summary.json").read_text())
    b2_summary = json.loads((STAGE_B2_DIR / "stage_b2_summary.json").read_text())
    b25_summary = json.loads((SWEEP_BEST_C / "summary.json").read_text())

    # Stage B metrics — use Stage C on Stage B results (from b2 summary 'stage_c_baseline')
    b_c = b2_summary["stage_c_baseline"]
    b2_c = b2_summary["stage_c_combined"]
    b25_c = b25_summary

    # Rows: metric; Columns: pipeline
    rows = [
        ("k", [b_c["k"], b2_c["k"], b25_c["final_k"]], "{:.0f}"),
        (
            "Silhouette",
            [
                0.432,  # from CLAUDE/article; stage_c baseline silhouette
                b2_c["silhouette"],
                b25_c["silhouette"],
            ],
            "{:.3f}",
        ),
        (
            "Cramér's V (↓)",
            [b_c["cramers_v"], b2_c["cramers_v"], b25_c["cramers_v"]],
            "{:.3f}",
        ),
        (
            "Mean entropy (bits, ↑)",
            [
                b_c["mean_cohort_entropy"],
                b2_c["mean_cohort_entropy"],
                b25_c["mean_cluster_entropy_bits"],
            ],
            "{:.3f}",
        ),
        (
            "Mean confidence (↑)",
            [
                b_c["consensus_mean_confidence"],
                b2_c["consensus_mean_confidence"],
                b25_c["consensus_mean_confidence"],
            ],
            "{:.3f}",
        ),
        (
            "Boundary patients (↓)",
            [
                b_c["n_negative_confidence"],
                b2_c["n_negative_confidence"],
                b25_c["n_negative_confidence"],
            ],
            "{:.0f}",
        ),
    ]

    pipelines = ["B only", "B + B2", "B + B2.5"]
    colors = [STAGE_COLORS["B"], STAGE_COLORS["B+B2"], STAGE_COLORS["B+B2.5"]]

    fig = plt.figure(figsize=(14.5, 9.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.3], hspace=0.55, wspace=0.35)

    # Grouped bar plots for each metric
    metric_axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]
    for ax, (name, values, fmt) in zip(metric_axes, rows):
        xpos = np.arange(len(pipelines))
        ax.bar(xpos, values, color=colors, edgecolor="white")
        ymax = max(values) * 1.18 if max(values) > 0 else 1
        ax.set_ylim(0, ymax)
        for i, v in enumerate(values):
            ax.text(i, v + ymax * 0.02, fmt.format(v), ha="center", fontsize=9)
        ax.set_title(name, fontsize=11)
        ax.set_xticks(xpos)
        ax.set_xticklabels(pipelines, rotation=0, fontsize=9)
        sns.despine(ax=ax)

    # Bottom row: B+B2.5 cluster × cohort row-normalized heatmap
    ax_heat = fig.add_subplot(gs[2, :])
    labels = pd.read_parquet(SWEEP_BEST_C / "consensus_labels.parquet")
    meta = labels.reset_index()
    contingency = pd.crosstab(meta["cluster"], meta["cohort"])[COHORT_ORDER]
    row_norm = contingency.div(contingency.sum(axis=1), axis=0)
    n_per = labels["cluster"].value_counts().sort_index()

    sns.heatmap(
        row_norm.T.values,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar_kws={"label": "Row proportion (cohort mix)"},
        xticklabels=[f"C{i}\n(n={int(n_per.loc[i])})" for i in row_norm.index],
        yticklabels=[COHORT_LABELS[c] for c in COHORT_ORDER],
        ax=ax_heat,
        linewidths=0.4,
        linecolor="white",
        vmin=0,
        vmax=1.0,
    )
    ax_heat.set_title(
        "Stage B + B2.5 final partition (k=8) — Cramér's V = 0.403, mean entropy 1.335 bits, confidence 0.694",
        fontsize=11,
        pad=10,
    )
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("Cohort")
    plt.setp(ax_heat.get_xticklabels(), rotation=0)
    plt.setp(ax_heat.get_yticklabels(), rotation=0)

    fig.suptitle(
        "Stage B vs Stage B + B2 vs Stage B + B2.5 — three-way stratification comparison",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    savefig(fig, "fig_b9_three_way_comparison")


# ─── Driver ───────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Output directory: {OUT_DIR}")

    fig_embedding_composition()
    fig_kmeans_sweep()
    fig_stage_b_clusters()
    fig_enrichment()
    fig_gnn_training()
    fig_stage_b2_clusters()
    fig_sweep_tradeoff()
    fig_sweep_levers()
    fig_three_way_comparison()

    print("Done.")


if __name__ == "__main__":
    main()
