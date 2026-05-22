"""Render the two-panel headline figure: DSM cohort vs. 6-cluster consensus partition.

This replaces the earlier 8-cluster t-SNE figure that was generated from the
Stage B2.5 GCN embedding. The article's central narrative is the k=6 Stage C
consensus partition on the deterministic 56-dimensional Stage B composite
embedding, so we regenerate the side-by-side t-SNE using exactly those two
artefacts:

    embedding : output/stratification/stage_b_review/embedding_cache/embedding.parquet
    labels    : output/stratification/stage_c/consensus_labels.parquet   (k = 6)

Produces
    docs/face_stratification/results/fig_r0_dsm_vs_6clusters.{png,pdf}

The figure is intentionally minimal: two scatter plots, same coordinate
system, same points, different colour mapping. Left = DSM cohort label,
right = 6-cluster consensus. This is the qualitative evidence that the
cohort label and the data-driven partition are not the same structure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "face_stratification" / "results"
CACHE_DIR = ROOT / "output" / "stratification" / "figures_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

EMB_PATH = (
    ROOT
    / "output"
    / "stratification"
    / "stage_b_review"
    / "embedding_cache"
    / "embedding.parquet"
)
LABELS_PATH = (
    ROOT / "output" / "stratification" / "stage_c" / "consensus_labels.parquet"
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
sns.set_theme(style="white", context="paper", font_scale=1.15)
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COHORT_ORDER = ["BP", "SZ", "ASP", "DR"]
COHORT_COLORS = {
    "BP": "#1f77b4",
    "SZ": "#d62728",
    "ASP": "#2ca02c",
    "DR": "#9467bd",
}

# Six clusters, with clinically interpretable short labels. Colours chosen from
# a colourblind-friendly qualitative palette.
CLUSTER_LABELS = {
    0: "C0 · Pediatric ASP",
    1: "C1 · High-burden mood",
    2: "C2 · Cardiometabolic SMI",
    3: "C3 · Metabolic-impulsive",
    4: "C4 · Early-onset neurodev.",
    5: "C5 · Chronic stabilised",
}
CLUSTER_COLORS = {
    0: "#F0B323",  # gold
    1: "#1F77B4",  # blue
    2: "#D62728",  # red
    3: "#17BECF",  # cyan
    4: "#2CA02C",  # green
    5: "#9467BD",  # purple
}


def _load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (embedding, cohort_labels, cluster_labels) aligned row-wise."""
    emb_df = pd.read_parquet(EMB_PATH)
    labels_df = pd.read_parquet(LABELS_PATH)
    # Align on the MultiIndex (cohort, patient_id)
    joined = emb_df.join(labels_df, how="inner")
    assert len(joined) == len(emb_df) == len(labels_df), (
        f"row-count mismatch: emb={len(emb_df)} labels={len(labels_df)} "
        f"joined={len(joined)}"
    )
    cluster_labels = joined["cluster"].to_numpy()
    cohort_labels = (
        joined.index.get_level_values("cohort").astype(str).str.upper().to_numpy()
    )
    X = joined.drop(columns=["cluster"]).to_numpy(dtype=np.float32)
    X = normalize(X, norm="l2", axis=1)
    return X, cohort_labels, cluster_labels


def _tsne_projection(X: np.ndarray, perplexity: int = 30, seed: int = 0) -> np.ndarray:
    cache = CACHE_DIR / f"tsne_stage_b_composite_p{perplexity}_seed{seed}.npy"
    if cache.exists():
        arr = np.load(cache)
        if arr.shape[0] == X.shape[0]:
            return arr
    print(f"[tsne] fitting p={perplexity} on {X.shape} …")
    tsne = TSNE(
        n_components=2,
        metric="cosine",
        perplexity=perplexity,
        init="pca",
        random_state=seed,
        learning_rate="auto",
        max_iter=1000,
    )
    Y = tsne.fit_transform(X).astype(np.float32)
    np.save(cache, Y)
    return Y


def _scatter(
    ax: plt.Axes,
    Y: np.ndarray,
    labels: np.ndarray,
    palette: dict,
    order: list,
    title: str,
) -> None:
    rng = np.random.default_rng(0)
    order_idx = rng.permutation(len(Y))
    Yp = Y[order_idx]
    lp = labels[order_idx]
    for key in order:
        mask = lp == key
        if mask.sum() == 0:
            continue
        ax.scatter(
            Yp[mask, 0],
            Yp[mask, 1],
            s=5,
            c=palette[key],
            alpha=0.55,
            linewidths=0,
            rasterized=True,
        )
    ax.set_title(title, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_aspect("equal", adjustable="datalim")
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#b0b0b0")
        ax.spines[spine].set_linewidth(0.6)


def _legend_below(fig, labels: list[str], colors: list[str], y: float, ncol: int) -> None:
    handles = [
        mpatches.Patch(color=c, label=lab) for lab, c in zip(labels, colors)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
        fontsize=10,
        handlelength=1.4,
        columnspacing=1.2,
        handletextpad=0.5,
    )


def main() -> None:
    X, cohort_labels, cluster_labels = _load_data()
    print(
        f"[data] n={len(X)}  embedding_dim={X.shape[1]}  "
        f"n_cohorts={len(np.unique(cohort_labels))}  "
        f"n_clusters={len(np.unique(cluster_labels))}"
    )
    Y = _tsne_projection(X, perplexity=30)
    # Keep the same axes for both panels
    xlim = (Y[:, 0].min() - 3, Y[:, 0].max() + 3)
    ylim = (Y[:, 1].min() - 3, Y[:, 1].max() + 3)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.0))

    _scatter(
        axes[0],
        Y,
        cohort_labels,
        COHORT_COLORS,
        COHORT_ORDER,
        title="(a)  DSM-5 cohort label",
    )
    _scatter(
        axes[1],
        Y,
        cluster_labels,
        CLUSTER_COLORS,
        sorted(CLUSTER_COLORS.keys()),
        title="(b)  Data-driven 6-cluster consensus partition",
    )
    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    _legend_below(
        fig,
        labels=COHORT_ORDER,
        colors=[COHORT_COLORS[k] for k in COHORT_ORDER],
        y=-0.02,
        ncol=4,
    )
    # Build a second, lower legend row for the clusters.
    cluster_order = sorted(CLUSTER_LABELS.keys())
    cluster_handles = [
        mpatches.Patch(color=CLUSTER_COLORS[k], label=CLUSTER_LABELS[k])
        for k in cluster_order
    ]
    fig.legend(
        handles=cluster_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
        frameon=False,
        fontsize=10,
        handlelength=1.4,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    plt.subplots_adjust(wspace=0.05, bottom=0.22, top=0.93)

    out_png = OUT_DIR / "fig_r0_dsm_vs_6clusters.png"
    out_pdf = OUT_DIR / "fig_r0_dsm_vs_6clusters.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[write] {out_png}")
    print(f"[write] {out_pdf}")


if __name__ == "__main__":
    main()
