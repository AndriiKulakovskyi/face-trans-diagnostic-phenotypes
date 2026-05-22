"""Generate a single composite figure summarising the Patient Embedding section.

Produces fig_patient_embedding_overview.{png,pdf} under
docs/face_stratification/stage_b/.

All user-visible labels use clinical/mathematical terminology rather than
the internal pipeline names ("Stage B / B2 / B2.5"). Panel semantics:

Layout (2 rows × 3 columns)
---------------------------
(a) Consensus k=6 partition × DSM cohort (row-normalised heatmap) —
    the canonical six-cluster partition narrated by the article.
(b) k selection on the 56-D multi-view composite embedding:
    raw k-means silhouette vs the transdiagnostic-weighted consensus
    score s = sil + (1+DB)^-1 + 2·H/log₂4 + (1−V), across k ∈ [4, 10].
    k*=6 is the consensus optimum; the raw silhouette peak at k=8 is
    annotated for context.
(c) Graph neural network training curves: link-prediction auto-encoder
    (BCE loss) and contrastive GCN (NT-Xent loss), dual axes.
(d) Sharpness vs DSM alignment: three embeddings in the
    (Cramér's V ↓, silhouette ↑) plane, with an arrow visualising the
    effect of restricting GCN message passing to transdiagnostic edges.
(e) Three-way metric comparison (Cramér's V, mean cohort entropy,
    mean consensus confidence) across the three embeddings.
(f) Contrastive GCN (transdiagnostic-only edges) partition × cohort,
    row-normalised — an ablation against the canonical partition of
    panel (a).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "face_stratification" / "stage_b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STAGE_B_DIR = ROOT / "output" / "stratification" / "stage_b_review"
STAGE_C_DIR = ROOT / "output" / "stratification" / "stage_c"
STAGE_B2_DIR = ROOT / "output" / "stratification" / "stage_b2"
SWEEP_DIR = STAGE_B2_DIR / "sweep"
SWEEP_BEST_C = SWEEP_DIR / "stage_c_on_best"

# Short clinical labels for the k=6 Stage C consensus clusters (see §6.2).
CLUSTER_SHORT_LABELS = {
    0: "C0 Pediatric ASP",
    1: "C1 High-burden mood",
    2: "C2 Cardiometab. SMI",
    3: "C3 Metabolic-impuls.",
    4: "C4 Early-onset neurodev.",
    5: "C5 Chronic stabilised",
}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlesize": 11.5,
        "axes.labelsize": 10,
        "legend.frameon": False,
    }
)

COHORT_ORDER = ["bp", "sz", "asp", "dr"]
COHORT_LABELS = {"bp": "BP", "sz": "SZ", "asp": "ASP", "dr": "DR"}

# Colours for the three embeddings used in panels (d), (e) and (f).
# Keys are clinical/mathematical short names, not the internal pipeline
# labels, so they render consistently across legends.
EMBEDDING_COLORS = {
    "composite": "#4C78A8",   # 56-D spectral multi-view composite
    "gcn_full": "#D62728",    # GCN on the full 17-relation multiplex
    "gcn_trans": "#2CA02C",   # GCN restricted to transdiagnostic edges
}


def _row_normalized_contingency(labels_parquet: Path) -> tuple[pd.DataFrame, pd.Series]:
    labels = pd.read_parquet(labels_parquet)
    meta = labels.reset_index()
    meta["cohort"] = meta["cohort"].astype(str)
    contingency = pd.crosstab(meta["cluster"], meta["cohort"])[COHORT_ORDER]
    row_norm = contingency.div(contingency.sum(axis=1), axis=0)
    n_per = labels["cluster"].value_counts().sort_index()
    return row_norm, n_per


# ─── Panels ───────────────────────────────────────────────────────────────────
def panel_a_stage_c_consensus_clusters(ax: plt.Axes) -> None:
    """Stage C k=6 consensus partition × cohort contingency (row-normalised).

    This panel shows the *canonical* partition narrated by the article
    (§5, §6) — the 6-cluster consensus on the 56-D Stage B composite.
    Each row sums to 1 and gives the cohort composition of the cluster.
    """
    labels = pd.read_parquet(STAGE_C_DIR / "consensus_labels.parquet")
    meta = labels.reset_index()
    meta["cohort"] = meta["cohort"].astype(str).str.lower()
    contingency = pd.crosstab(meta["cluster"], meta["cohort"])
    contingency = contingency.reindex(columns=COHORT_ORDER, fill_value=0)
    row_norm = contingency.div(contingency.sum(axis=1), axis=0)
    n_per = labels["cluster"].value_counts().sort_index()

    yticklabels = [
        f"{CLUSTER_SHORT_LABELS.get(int(i), f'C{i}')}  (n={int(n_per.loc[i])})"
        for i in row_norm.index
    ]

    sns.heatmap(
        row_norm.values,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar_kws={"label": "Row prop.", "shrink": 0.85},
        xticklabels=[COHORT_LABELS[c] for c in COHORT_ORDER],
        yticklabels=yticklabels,
        ax=ax,
        linewidths=0.3,
        linecolor="white",
        vmin=0,
        vmax=1.0,
        annot_kws={"fontsize": 8},
    )
    ax.set_title(
        "(a)  Consensus k=6 partition × DSM cohort",
        fontsize=10.5,
        pad=8,
    )
    ax.set_xlabel("DSM cohort", fontsize=9)
    ax.set_ylabel("Consensus cluster", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    plt.setp(ax.get_xticklabels(), fontsize=9)


def panel_b_k_selection(ax: plt.Axes) -> None:
    """k-selection on the 56-D composite: raw silhouette vs transdiagnostic
    consensus score across k ∈ [4, 10].

    Left y-axis  : silhouette from k-means on the composite (peaks at k=8).
    Right y-axis : transdiagnostic-weighted consensus score s_trans
                   = sil + 1/(1+DB) + 2·H/log2(4) + (1−V), which peaks at
                   k=6 — the consensus optimum narrated in §5.3.

    The two competing optima illustrate the article's central
    methodological point: pure geometric sharpness (silhouette) alone
    would pick k=8, but the transdiagnostic objective (which rewards
    cohort entropy and penalises DSM alignment) picks k=6.
    """
    df = pd.read_csv(STAGE_C_DIR / "algorithm_k_grid.csv")
    km = df[(df["embedding"] == "composite") & (df["algorithm"] == "kmeans")].sort_values("k")

    # Transdiagnostic-weighted score, weights (w_sil, w_db, w_trans, w_ndsm) = (1,1,2,1)
    s_trans = (
        km["silhouette"]
        + 1.0 / (1.0 + km["davies_bouldin"])
        + 2.0 * km["mean_cluster_entropy_bits"] / np.log2(4)
        + (1.0 - km["cramers_v"])
    )

    consensus_k = int(km["k"].iloc[int(np.argmax(s_trans.values))])
    silhouette_peak_k = int(km["k"].iloc[int(np.argmax(km["silhouette"].values))])

    color_sil = "#4C78A8"
    color_trans = "#2CA02C"

    ax.plot(
        km["k"],
        km["silhouette"],
        marker="o",
        color=color_sil,
        linewidth=2,
        markersize=6,
        label="Silhouette",
    )
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Silhouette", color=color_sil)
    ax.tick_params(axis="y", labelcolor=color_sil)
    sil_min = float(km["silhouette"].min())
    sil_max = float(km["silhouette"].max())
    pad_s = 0.015
    ax.set_ylim(sil_min - pad_s, sil_max + 0.035)

    ax2 = ax.twinx()
    ax2.plot(
        km["k"],
        s_trans.values,
        marker="s",
        color=color_trans,
        linewidth=2,
        markersize=6,
        label="Transdiag. consensus score",
    )
    ax2.set_ylabel("Transdiag. consensus score", color=color_trans)
    ax2.tick_params(axis="y", labelcolor=color_trans)
    tmin = float(s_trans.min())
    tmax = float(s_trans.max())
    pad_t = 0.03
    ax2.set_ylim(tmin - pad_t, tmax + 0.06)
    ax2.grid(False)

    # Consensus pick at k=6 — primary optimum
    ax.axvline(consensus_k, color="#D62728", linestyle="--", linewidth=1.3, zorder=1)
    ax.text(
        consensus_k - 0.08,
        sil_max + 0.018,
        f"k*={consensus_k}\n(consensus)",
        fontsize=8.5,
        color="#D62728",
        fontweight="bold",
        ha="right",
        va="top",
    )

    # Raw silhouette peak — secondary annotation (dotted grey)
    if silhouette_peak_k != consensus_k:
        ax.axvline(
            silhouette_peak_k,
            color="#888",
            linestyle=":",
            linewidth=1.0,
            zorder=1,
        )
        ax.text(
            silhouette_peak_k + 0.12,
            sil_min + 0.005,
            f"raw sil. peak\nk={silhouette_peak_k}",
            fontsize=7.5,
            color="#555",
            ha="left",
            va="bottom",
        )

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        fontsize=8,
        framealpha=0.92,
        frameon=True,
        edgecolor="#bbb",
    )
    ax.set_title(
        "(b)  k selection: silhouette vs transdiagnostic score",
        fontsize=10.5,
        pad=8,
    )


def panel_c_gnn_training(ax: plt.Axes) -> None:
    history = json.loads((STAGE_B2_DIR / "training_history.json").read_text())
    gae = pd.DataFrame(history["gae"])
    con = pd.DataFrame(history["contrastive"])

    color_gae = "#4C78A8"
    color_con = "#54A24B"

    # Normalize GraphCL loss to share axis with GAE
    ax.plot(
        gae["epoch"],
        gae["loss"],
        marker="o",
        color=color_gae,
        linewidth=2,
        markersize=5,
        label="GAE (BCE)",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("GAE loss", color=color_gae)
    ax.tick_params(axis="y", labelcolor=color_gae)
    ax.set_ylim(0.59, 0.76)

    ax2 = ax.twinx()
    ax2.plot(
        con["epoch"],
        con["loss"],
        marker="s",
        color=color_con,
        linewidth=2,
        markersize=5,
        label="GraphCL (NT-Xent)",
    )
    ax2.set_ylabel("GraphCL loss", color=color_con)
    ax2.tick_params(axis="y", labelcolor=color_con)
    ax2.set_ylim(8.3, 9.8)
    ax2.grid(False)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax.set_title(
        "(c)  Graph neural network training curves",
        fontsize=10.5,
        pad=8,
    )


def panel_d_sweep_tradeoff(ax: plt.Axes) -> None:
    """Three-point Pareto plot in the (Cramér's V, silhouette) plane.

    Every coordinate on this panel is computed by running the consensus
    clustering on the corresponding embedding — exactly the coordinate
    system the article's §8 narrates. This fixes an earlier version of
    this panel which had plotted sweep-internal Cramér's V values that
    were almost identical for the two GCN reference configurations
    (0.421 vs 0.425) and therefore hid the actual Pareto advantage of
    the transdiagnostic-only GCN reported in the text (0.403 vs 0.599).
    """
    b2_summary = json.loads((STAGE_B2_DIR / "stage_b2_summary.json").read_text())
    b25_summary = json.loads((SWEEP_BEST_C / "summary.json").read_text())

    # --- Three reference embeddings -----------------------------------
    # (1) The closed-form spectral multi-view composite, silhouette taken
    #     at the consensus pick k = 6 (see panel (b)).
    # (2) The contrastive GCN trained on the full 17-relation multiplex.
    # (3) The contrastive GCN trained on transdiagnostic edges only.
    composite_sweep = pd.read_csv(STAGE_B_DIR / "kmeans_sweep.csv")
    composite_sil = float(composite_sweep.loc[composite_sweep["k"] == 6, "silhouette"].iloc[0])
    composite_v = float(b2_summary["stage_c_baseline"]["cramers_v"])

    gcn_full_sil = float(b2_summary["stage_c_combined"]["silhouette"])
    gcn_full_v = float(b2_summary["stage_c_combined"]["cramers_v"])

    gcn_trans_sil = float(b25_summary["silhouette"])
    gcn_trans_v = float(b25_summary["cramers_v"])

    points = [
        {
            "name": "Spectral composite\n(closed-form, 56 d)",
            "v": composite_v,
            "sil": composite_sil,
            "color": EMBEDDING_COLORS["composite"],
            "marker": "o",
            "offset": (0.012, -0.005),
            "ha": "left",
        },
        {
            "name": "Contrastive GCN\n— all 17 relations",
            "v": gcn_full_v,
            "sil": gcn_full_sil,
            "color": EMBEDDING_COLORS["gcn_full"],
            "marker": "s",
            "offset": (-0.012, 0.008),
            "ha": "right",
        },
        {
            "name": "Contrastive GCN\n— transdiagnostic edges only",
            "v": gcn_trans_v,
            "sil": gcn_trans_sil,
            "color": EMBEDDING_COLORS["gcn_trans"],
            "marker": "^",
            "offset": (-0.012, 0.006),
            "ha": "right",
        },
    ]

    # Pareto-frontier hint: dashed grey line connecting the two Pareto-optimal
    # embeddings (spectral composite and transdiagnostic-only GCN). The
    # full-multiplex GCN lies strictly above the frontier (slightly more
    # sharpness but much higher DSM alignment), so the visual "front" runs
    # composite → transdiagnostic-only GCN.
    front = sorted(
        [(p["v"], p["sil"]) for p in points], key=lambda t: t[0]
    )
    fx, fy = zip(*front)
    ax.plot(
        fx,
        fy,
        linestyle=":",
        linewidth=1.1,
        color="#999",
        zorder=1,
    )

    # Draw each reference point as a large filled marker with a white edge.
    for p in points:
        ax.scatter(
            p["v"],
            p["sil"],
            s=260,
            marker=p["marker"],
            color=p["color"],
            edgecolor="white",
            linewidth=1.8,
            zorder=4,
        )
        dx, dy = p["offset"]
        ax.annotate(
            p["name"],
            xy=(p["v"], p["sil"]),
            xytext=(p["v"] + dx, p["sil"] + dy),
            fontsize=7.8,
            color="#222",
            ha=p["ha"],
            va="center",
            zorder=5,
        )

    # Arrow from the full-multiplex GCN to the transdiagnostic-only GCN,
    # illustrating the key intervention of §8.2: restricting message
    # passing to transdiagnostic edges drops Cramér's V by 0.196 points
    # (0.599 → 0.403) at essentially unchanged silhouette.
    ax.annotate(
        "",
        xy=(gcn_trans_v, gcn_trans_sil),
        xytext=(gcn_full_v, gcn_full_sil),
        arrowprops=dict(
            arrowstyle="->",
            color="#444",
            lw=1.3,
            shrinkA=10,
            shrinkB=10,
        ),
        zorder=3,
    )
    mid_v = 0.5 * (gcn_full_v + gcn_trans_v)
    mid_sil = 0.5 * (gcn_full_sil + gcn_trans_sil)
    ax.text(
        mid_v,
        mid_sil + 0.006,
        "restrict to\ntransdiag. edges",
        fontsize=7.5,
        color="#444",
        ha="center",
        va="bottom",
        style="italic",
    )

    # Quadrant label
    ax.text(
        0.32,
        0.495,
        "← more transdiagnostic",
        fontsize=7.5,
        color="#555",
        ha="left",
        va="bottom",
        style="italic",
    )

    ax.set_xlim(0.30, 0.66)
    ax.set_ylim(0.40, 0.52)
    ax.set_xlabel("Cramér's V against DSM cohort  (↓ better)")
    ax.set_ylabel("Silhouette on composite  (↑ better)")
    ax.set_title(
        "(d)  Sharpness vs DSM alignment — three pipelines",
        fontsize=10.5,
        pad=8,
    )


def panel_e_three_way(ax: plt.Axes) -> None:
    b2_summary = json.loads((STAGE_B2_DIR / "stage_b2_summary.json").read_text())
    b25_summary = json.loads((SWEEP_BEST_C / "summary.json").read_text())

    b_c = b2_summary["stage_c_baseline"]
    b2_c = b2_summary["stage_c_combined"]

    # Five key metrics (normalized to [0, 1] where possible for a grouped bar)
    metrics = [
        ("Cramér's V\n(↓ better)", [b_c["cramers_v"], b2_c["cramers_v"], b25_summary["cramers_v"]]),
        (
            "Entropy\n(bits, ↑)",
            [
                b_c["mean_cohort_entropy"] / 2,  # normalize to max 2 bits
                b2_c["mean_cohort_entropy"] / 2,
                b25_summary["mean_cluster_entropy_bits"] / 2,
            ],
        ),
        (
            "Confidence\n(↑)",
            [
                b_c["consensus_mean_confidence"],
                b2_c["consensus_mean_confidence"],
                b25_summary["consensus_mean_confidence"],
            ],
        ),
    ]

    # Raw values for annotations
    raw_values = [
        [b_c["cramers_v"], b2_c["cramers_v"], b25_summary["cramers_v"]],
        [b_c["mean_cohort_entropy"], b2_c["mean_cohort_entropy"], b25_summary["mean_cluster_entropy_bits"]],
        [b_c["consensus_mean_confidence"], b2_c["consensus_mean_confidence"], b25_summary["consensus_mean_confidence"]],
    ]

    x = np.arange(len(metrics))
    width = 0.26
    offsets = [-width, 0, width]
    pipelines = [
        "Spectral composite",
        "Contrastive GCN (all edges)",
        "Contrastive GCN (trans. only)",
    ]
    colors = [
        EMBEDDING_COLORS["composite"],
        EMBEDDING_COLORS["gcn_full"],
        EMBEDDING_COLORS["gcn_trans"],
    ]

    for i, (pipeline, offset, color) in enumerate(zip(pipelines, offsets, colors)):
        vals = [m[1][i] for m in metrics]
        bars = ax.bar(
            x + offset,
            vals,
            width,
            color=color,
            edgecolor="white",
            label=pipeline,
        )
        for j, (bar, rv) in enumerate(zip(bars, [raw_values[jj][i] for jj in range(len(metrics))])):
            fmt = "{:.2f}" if j != 0 else "{:.2f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                fmt.format(rv),
                ha="center",
                fontsize=7.5,
                color="#333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], fontsize=9)
    ax.set_ylabel("Metric (entropy scaled to max = 1)")
    ax.set_ylim(0, 0.95)
    ax.set_title(
        "(e)  Three embeddings compared on three quality metrics",
        fontsize=10.5,
        pad=8,
    )
    ax.legend(loc="upper right", fontsize=7.5)
    sns.despine(ax=ax)


def panel_f_b25_clusters(ax: plt.Axes) -> None:
    row_norm, n_per = _row_normalized_contingency(
        SWEEP_BEST_C / "consensus_labels.parquet"
    )
    sns.heatmap(
        row_norm.values,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        cbar_kws={"label": "Row prop.", "shrink": 0.85},
        xticklabels=[COHORT_LABELS[c] for c in COHORT_ORDER],
        yticklabels=[f"C{i} (n={int(n_per.loc[i])})" for i in row_norm.index],
        ax=ax,
        linewidths=0.3,
        linecolor="white",
        vmin=0,
        vmax=1.0,
        annot_kws={"fontsize": 8},
    )
    k_actual = int(row_norm.shape[0])
    ax.set_title(
        f"(f)  Contrastive GCN (trans. edges) partition × cohort  (k={k_actual}, ablation)",
        fontsize=10.5,
        pad=8,
    )
    ax.set_xlabel("DSM cohort", fontsize=9)
    ax.set_ylabel("Cluster", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    plt.setp(ax.get_xticklabels(), fontsize=9)


# ─── Driver ───────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Output directory: {OUT_DIR}")

    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(
        2,
        3,
        hspace=0.55,
        wspace=0.40,
        left=0.06,
        right=0.98,
        top=0.93,
        bottom=0.07,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    panel_a_stage_c_consensus_clusters(ax_a)
    panel_b_k_selection(ax_b)
    panel_c_gnn_training(ax_c)
    panel_d_sweep_tradeoff(ax_d)
    panel_e_three_way(ax_e)
    panel_f_b25_clusters(ax_f)

    fig.suptitle(
        "Composite embedding, k selection, and comparison of three "
        "candidate embeddings on the 11,014-patient multiplex graph",
        fontsize=13.5,
        fontweight="bold",
        y=0.995,
    )

    out_png = OUT_DIR / "fig_patient_embedding_overview.png"
    out_pdf = OUT_DIR / "fig_patient_embedding_overview.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_png.name} / {out_pdf.name}")


if __name__ == "__main__":
    main()
