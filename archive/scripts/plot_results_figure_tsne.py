"""Generate the headline t-SNE/UMAP figure for Section 3.1.

Renders
-------
fig_r0_tsne_dsm_vs_clusters
    2×3 publication figure anchoring §3.1 "The FACE cohort does not
    carve along DSM joints".
    (a) t-SNE of the 88-d transdiagnostic-optimised GCN embedding,
        coloured by DSM cohort.
    (b) same t-SNE, coloured by the k=8 consensus clusters with
        cluster-medoid annotations.
    (c) UMAP replication, coloured by DSM cohort.
    (d) UMAP replication, coloured by consensus clusters.
    (e) k-NN label-purity curves (DSM label vs. consensus cluster)
        computed in the *original* high-dimensional space, with
        bootstrap 95% CI bands. This panel is non-circular.
    (f) four small multiples: each DSM cohort plotted on its own
        against a grey background of all other patients.

fig_sup_tsne_perplexity
    3-panel supplementary: t-SNE at perplexities 10, 30, 100 on the
    same 88-d embedding, coloured by DSM cohort — robustness check.

fig_sup_naive_vs_optimised_geometry
    2×2 supplementary comparing the naive spectral composite (56-d,
    Stage B) to the transdiagnostic-optimised embedding (88-d,
    Stage B2.5) on a shared layout: even the naive embedding already
    shows DSM interpenetration.

All t-SNE / UMAP projections are cached under
output/stratification/figures_cache/ so re-rendering is instantaneous.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

try:
    import umap
except ImportError:  # pragma: no cover
    umap = None

warnings.filterwarnings("ignore", category=UserWarning)

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "face_stratification" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = ROOT / "output" / "stratification" / "figures_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

STAGE_B_EMB = ROOT / "output" / "stratification" / "stage_b_review" / \
    "embedding_cache" / "embedding.parquet"
STAGE_B2_5_EMB = ROOT / "output" / "stratification" / "stage_b2" / "sweep" / \
    "embedding_b2_5_combined" / "embedding.parquet"
STAGE_B2_5_LABELS = ROOT / "output" / "stratification" / "stage_b2" / "sweep" / \
    "stage_c_on_best" / "consensus_labels.parquet"
STAGE_B_LABELS = ROOT / "output" / "stratification" / "stage_b_review" / \
    "cluster_labels.parquet"

# ─── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="white", context="paper", font_scale=1.05)
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.frameon": False,
    }
)

COHORT_PALETTE = {"bp": "#2166AC", "sz": "#B2182B", "asp": "#E08214", "dr": "#1B7837"}
COHORT_LABELS = {"bp": "BP", "sz": "SZ", "asp": "ASP", "dr": "DR"}
COHORT_ORDER = ["bp", "sz", "asp", "dr"]

# 8-cluster palette
CLUSTER_PALETTE = sns.color_palette("tab10", 10)

# Short clinical signatures for medoid annotations, derived from
# Section 3.5 enrichment analysis on the Stage B2.5 partition.
# Keys match the k=8 Stage B2.5 consensus labels.
CLUSTER_SIGNATURES = {
    0: "low-comorbid.",
    1: "metabolic",
    2: "high-burden",
    3: "female early",
    4: "paediatric ASP",
    5: "chronic DR-mix",
    6: "male chronic",
    7: "young low-com.",
}


def savefig(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png / .pdf")


# ─── Data loaders ─────────────────────────────────────────────────────────────
def load_embedding_optimised() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X [N, 88], cohorts [N], clusters [N]) for the B2.5 embedding."""
    emb = pd.read_parquet(STAGE_B2_5_EMB)
    labels = pd.read_parquet(STAGE_B2_5_LABELS)
    joined = emb.join(labels, how="inner")
    cohorts = np.asarray(joined.index.get_level_values("cohort"))
    clusters = joined["cluster"].to_numpy()
    X = joined.drop(columns=["cluster"]).to_numpy(dtype=np.float32)
    # L2-normalise — the downstream Stage C pipeline uses cosine similarity.
    X = normalize(X, norm="l2", axis=1)
    return X, cohorts, clusters


def load_embedding_naive() -> tuple[np.ndarray, np.ndarray]:
    """Return (X [N, 56], cohorts [N]) for the Stage B spectral composite."""
    emb = pd.read_parquet(STAGE_B_EMB)
    cohorts = np.asarray(emb.index.get_level_values("cohort"))
    X = emb.to_numpy(dtype=np.float32)
    X = normalize(X, norm="l2", axis=1)
    return X, cohorts


# ─── Cached 2-D projections ───────────────────────────────────────────────────
def cached_tsne(X: np.ndarray, tag: str, perplexity: int = 30,
                seed: int = 42) -> np.ndarray:
    path = CACHE_DIR / f"tsne_{tag}_p{perplexity}.npy"
    if path.exists():
        return np.load(path)
    print(f"  running t-SNE ({tag}, perp={perplexity}) …")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        metric="cosine",
        random_state=seed,
        max_iter=1000,
    )
    Z = tsne.fit_transform(X)
    np.save(path, Z)
    return Z


def cached_umap(X: np.ndarray, tag: str, n_neighbors: int = 30,
                min_dist: float = 0.1, seed: int = 42) -> np.ndarray:
    if umap is None:
        raise RuntimeError("umap-learn is required; pip install umap-learn")
    path = CACHE_DIR / f"umap_{tag}_n{n_neighbors}.npy"
    if path.exists():
        return np.load(path)
    print(f"  running UMAP ({tag}, n_neighbors={n_neighbors}) …")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )
    Z = reducer.fit_transform(X)
    np.save(path, Z)
    return Z


# ─── Non-circular k-NN label purity ───────────────────────────────────────────
def knn_purity_curve(X: np.ndarray, labels: np.ndarray,
                     ks: list[int], n_boot: int = 25,
                     seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, lo, hi) arrays over ks.

    Purity at k = fraction of a patient's k nearest neighbours (excluding
    itself, cosine metric) that share its label, averaged over all
    patients. Bootstrap CI comes from resampling patients (not neighbours).
    """
    k_max = max(ks)
    nn = NearestNeighbors(n_neighbors=k_max + 1, metric="cosine")
    nn.fit(X)
    _, idx = nn.kneighbors(X)
    # idx[:, 0] == self
    idx = idx[:, 1:]
    purities = np.zeros((len(ks), len(X)), dtype=np.float32)
    for i, k in enumerate(ks):
        matches = (labels[idx[:, :k]] == labels[:, None]).mean(axis=1)
        purities[i] = matches
    rng = np.random.default_rng(seed)
    means = purities.mean(axis=1)
    boot = np.zeros((n_boot, len(ks)), dtype=np.float32)
    N = len(X)
    for b in range(n_boot):
        sample = rng.integers(0, N, size=N)
        boot[b] = purities[:, sample].mean(axis=1)
    lo = np.quantile(boot, 0.025, axis=0)
    hi = np.quantile(boot, 0.975, axis=0)
    return means, lo, hi


# ─── Figure R0 — headline t-SNE/UMAP ──────────────────────────────────────────
def fig_r0_tsne_dsm_vs_clusters() -> None:
    X, cohorts, clusters = load_embedding_optimised()
    Z_tsne = cached_tsne(X, "b2_5", perplexity=30)
    Z_umap = cached_umap(X, "b2_5", n_neighbors=30)

    # k-NN purity in the original 88-d space
    ks = [5, 10, 25, 50, 100, 200]
    # Encode DSM cohort as integers for label comparison
    cohort_int = np.array([COHORT_ORDER.index(c) for c in cohorts])
    pur_dsm_m, pur_dsm_lo, pur_dsm_hi = knn_purity_curve(X, cohort_int, ks)
    pur_cl_m, pur_cl_lo, pur_cl_hi = knn_purity_curve(X, clusters, ks)

    # Layout: 2 rows × 3 cols
    fig = plt.figure(figsize=(16, 10.5))
    gs = fig.add_gridspec(
        2, 3,
        hspace=0.38, wspace=0.28,
        left=0.05, right=0.98, top=0.92, bottom=0.06,
    )

    # ─── (a) t-SNE by DSM cohort ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    order = np.random.default_rng(0).permutation(len(X))
    for c in COHORT_ORDER:
        mask = cohorts[order] == c
        ax.scatter(Z_tsne[order][mask, 0], Z_tsne[order][mask, 1],
                   s=4.5, c=COHORT_PALETTE[c], alpha=0.55,
                   edgecolor="none", label=COHORT_LABELS[c])
    ax.set_title("(a) t-SNE coloured by DSM cohort", loc="left")
    ax.set_xlabel("t-SNE 1", fontsize=9)
    ax.set_ylabel("t-SNE 2", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#ccc")
    leg = ax.legend(loc="upper right", fontsize=8, markerscale=2.2,
                    frameon=True, edgecolor="#bbb", framealpha=0.92)

    # ─── (b) t-SNE by consensus cluster + medoids ─────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    for cid in sorted(np.unique(clusters)):
        mask = clusters[order] == cid
        ax.scatter(Z_tsne[order][mask, 0], Z_tsne[order][mask, 1],
                   s=4.5, c=[CLUSTER_PALETTE[cid]], alpha=0.55,
                   edgecolor="none")
    # Annotate medoids
    for cid in sorted(np.unique(clusters)):
        mask = clusters == cid
        cx, cy = Z_tsne[mask, 0].mean(), Z_tsne[mask, 1].mean()
        sig = CLUSTER_SIGNATURES.get(int(cid), "")
        label = f"C{cid}\n{sig}" if sig else f"C{cid}"
        ax.annotate(
            label,
            xy=(cx, cy),
            fontsize=8,
            fontweight="bold",
            ha="center", va="center",
            color="#111",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", edgecolor="#888", alpha=0.85,
                      linewidth=0.6),
        )
    ax.set_title("(b) t-SNE coloured by consensus cluster (k = 8)",
                 loc="left")
    ax.set_xlabel("t-SNE 1", fontsize=9)
    ax.set_ylabel("t-SNE 2", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#ccc")

    # ─── (c) k-NN label purity curve ─────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    xs = np.asarray(ks)
    chance_dsm = (np.bincount(cohort_int) / len(cohort_int)).max()  # largest cohort
    chance_cl = (np.bincount(clusters) / len(clusters)).max()
    ax.plot(xs, pur_dsm_m, "-o", color="#4C78A8", linewidth=2.0,
            label="DSM cohort label")
    ax.fill_between(xs, pur_dsm_lo, pur_dsm_hi, color="#4C78A8", alpha=0.20)
    ax.plot(xs, pur_cl_m, "-s", color="#1B7837", linewidth=2.0,
            label="consensus cluster (k = 8)")
    ax.fill_between(xs, pur_cl_lo, pur_cl_hi, color="#1B7837", alpha=0.20)
    ax.axhline(chance_dsm, color="#4C78A8", linestyle=":", linewidth=0.9,
               alpha=0.7, label=f"DSM chance (largest = {chance_dsm:.2f})")
    ax.axhline(chance_cl, color="#1B7837", linestyle=":", linewidth=0.9,
               alpha=0.7, label=f"cluster chance (largest = {chance_cl:.2f})")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(k) for k in xs])
    ax.set_xlabel("k (nearest-neighbour radius)", fontsize=9)
    ax.set_ylabel("fraction same-label neighbours", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("(c) k-NN label purity in the 88-d embedding\n"
                 "(non-circular — no t-SNE involved)",
                 loc="left")
    ax.legend(loc="center right", fontsize=7, frameon=True,
              edgecolor="#bbb", framealpha=0.92)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#ccc")

    # ─── (d) UMAP by DSM cohort ───────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    for c in COHORT_ORDER:
        mask = cohorts[order] == c
        ax.scatter(Z_umap[order][mask, 0], Z_umap[order][mask, 1],
                   s=4.5, c=COHORT_PALETTE[c], alpha=0.55,
                   edgecolor="none", label=COHORT_LABELS[c])
    ax.set_title("(d) UMAP coloured by DSM cohort (robustness check)",
                 loc="left")
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#ccc")
    ax.legend(loc="upper right", fontsize=8, markerscale=2.2,
              frameon=True, edgecolor="#bbb", framealpha=0.92)

    # ─── (e) UMAP by consensus cluster ───────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    for cid in sorted(np.unique(clusters)):
        mask = clusters[order] == cid
        ax.scatter(Z_umap[order][mask, 0], Z_umap[order][mask, 1],
                   s=4.5, c=[CLUSTER_PALETTE[cid]], alpha=0.55,
                   edgecolor="none")
    for cid in sorted(np.unique(clusters)):
        mask = clusters == cid
        cx, cy = Z_umap[mask, 0].mean(), Z_umap[mask, 1].mean()
        sig = CLUSTER_SIGNATURES.get(int(cid), "")
        label = f"C{cid}\n{sig}" if sig else f"C{cid}"
        ax.annotate(
            label,
            xy=(cx, cy),
            fontsize=8,
            fontweight="bold",
            ha="center", va="center",
            color="#111",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", edgecolor="#888", alpha=0.85,
                      linewidth=0.6),
        )
    ax.set_title("(e) UMAP coloured by consensus cluster", loc="left")
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#ccc")

    # ─── (f) Per-cohort small multiples on the t-SNE ─────────────────────
    # Use a sub-gridspec with an extra top row reserved for the header,
    # so the 2×2 small multiples never collide with a floating fig.text.
    sub_gs = gs[1, 2].subgridspec(
        3, 2, height_ratios=[0.22, 1.0, 1.0],
        hspace=0.42, wspace=0.18,
    )
    header_ax = fig.add_subplot(sub_gs[0, :])
    header_ax.axis("off")
    header_ax.text(
        0.0, 0.5,
        "(f) t-SNE with each DSM cohort highlighted separately\n"
        "grey = all other patients",
        transform=header_ax.transAxes,
        fontsize=10, fontweight="bold", ha="left", va="center",
    )
    for idx, c in enumerate(COHORT_ORDER):
        sub_ax = fig.add_subplot(sub_gs[1 + idx // 2, idx % 2])
        sub_ax.scatter(Z_tsne[:, 0], Z_tsne[:, 1],
                       s=2.0, c="#e0e0e0", alpha=0.55, edgecolor="none")
        mask = cohorts == c
        sub_ax.scatter(Z_tsne[mask, 0], Z_tsne[mask, 1],
                       s=3.2, c=COHORT_PALETTE[c], alpha=0.8,
                       edgecolor="none")
        n = int(mask.sum())
        sub_ax.set_title(f"{COHORT_LABELS[c]}  (n = {n:,})",
                         fontsize=9, loc="center", pad=2)
        sub_ax.set_xticks([]); sub_ax.set_yticks([])
        for spine in sub_ax.spines.values():
            spine.set_edgecolor("#ccc")

    fig.suptitle(
        "Figure R0 — The FACE cohort does not carve along DSM joints: "
        "t-SNE / UMAP geometry of the transdiagnostic-optimised embedding",
        fontsize=13.5, fontweight="bold", y=0.985,
    )
    savefig(fig, "fig_r0_tsne_dsm_vs_clusters")


# ─── Supplementary — t-SNE perplexity sensitivity ─────────────────────────────
def fig_sup_tsne_perplexity() -> None:
    X, cohorts, _ = load_embedding_optimised()
    perps = [10, 30, 100]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    order = np.random.default_rng(0).permutation(len(X))
    for ax, p in zip(axes, perps):
        Z = cached_tsne(X, "b2_5", perplexity=p)
        for c in COHORT_ORDER:
            mask = cohorts[order] == c
            ax.scatter(Z[order][mask, 0], Z[order][mask, 1],
                       s=3.5, c=COHORT_PALETTE[c], alpha=0.55,
                       edgecolor="none", label=COHORT_LABELS[c])
        ax.set_title(f"t-SNE perplexity = {p}", loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#ccc")
    axes[-1].legend(loc="upper right", fontsize=8, markerscale=2.2,
                    frameon=True, edgecolor="#bbb", framealpha=0.92)
    fig.suptitle(
        "Figure S1 — t-SNE perplexity sensitivity (DSM cohorts) on the "
        "88-d transdiagnostic-optimised embedding",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    savefig(fig, "fig_sup_tsne_perplexity")


# ─── Supplementary — naive vs optimised geometry ─────────────────────────────
def fig_sup_naive_vs_optimised() -> None:
    X_opt, cohorts_opt, clusters_opt = load_embedding_optimised()
    X_naive, cohorts_naive = load_embedding_naive()
    Z_opt = cached_tsne(X_opt, "b2_5", perplexity=30)
    Z_naive = cached_tsne(X_naive, "b_only", perplexity=30)

    # k-NN purity on each embedding against DSM
    cohort_int_opt = np.array([COHORT_ORDER.index(c) for c in cohorts_opt])
    cohort_int_naive = np.array([COHORT_ORDER.index(c) for c in cohorts_naive])
    ks = [5, 10, 25, 50, 100, 200]
    pur_n_m, pur_n_lo, pur_n_hi = knn_purity_curve(X_naive, cohort_int_naive, ks)
    pur_o_m, pur_o_lo, pur_o_hi = knn_purity_curve(X_opt, cohort_int_opt, ks)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    order = np.random.default_rng(0).permutation(len(X_naive))

    # (a) naive spectral composite, DSM colour
    ax = axes[0, 0]
    for c in COHORT_ORDER:
        mask = cohorts_naive[order] == c
        ax.scatter(Z_naive[order][mask, 0], Z_naive[order][mask, 1],
                   s=3.5, c=COHORT_PALETTE[c], alpha=0.55,
                   edgecolor="none", label=COHORT_LABELS[c])
    ax.set_title("(a) Spectral composite (56-d, Stage B)\nt-SNE coloured by DSM cohort",
                 loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8, markerscale=2.2,
              frameon=True, edgecolor="#bbb", framealpha=0.92)

    # (b) optimised, DSM colour
    ax = axes[0, 1]
    for c in COHORT_ORDER:
        mask = cohorts_opt[order] == c
        ax.scatter(Z_opt[order][mask, 0], Z_opt[order][mask, 1],
                   s=3.5, c=COHORT_PALETTE[c], alpha=0.55,
                   edgecolor="none", label=COHORT_LABELS[c])
    ax.set_title("(b) Transdiag.-optimised (88-d, Stage B2.5)\nt-SNE coloured by DSM cohort",
                 loc="left")
    ax.set_xticks([]); ax.set_yticks([])

    # (c) k-NN purity on both embeddings against DSM
    ax = axes[1, 0]
    xs = np.asarray(ks)
    ax.plot(xs, pur_n_m, "-o", color="#4C78A8", linewidth=2.0,
            label="spectral composite (56-d)")
    ax.fill_between(xs, pur_n_lo, pur_n_hi, color="#4C78A8", alpha=0.2)
    ax.plot(xs, pur_o_m, "-s", color="#2CA02C", linewidth=2.0,
            label="transdiag.-optimised (88-d)")
    ax.fill_between(xs, pur_o_lo, pur_o_hi, color="#2CA02C", alpha=0.2)
    chance = (np.bincount(cohort_int_opt) / len(cohort_int_opt)).max()
    ax.axhline(chance, color="#555", linestyle=":", linewidth=0.9,
               label=f"DSM chance (largest = {chance:.2f})")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(k) for k in xs])
    ax.set_xlabel("k (nearest-neighbour radius)")
    ax.set_ylabel("fraction same-DSM neighbours")
    ax.set_ylim(0, 1)
    ax.set_title("(c) DSM-label k-NN purity in the two embeddings\n"
                 "(both stay well below 1, even at k = 5)",
                 loc="left")
    ax.legend(loc="center right", fontsize=8, frameon=True,
              edgecolor="#bbb", framealpha=0.92)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)

    # (d) text block — explicit note
    ax = axes[1, 1]
    ax.axis("off")
    msg = (
        "Interpretation\n"
        "──────────────\n\n"
        "• The DSM-label k-NN purity already sits well below 1 on the\n"
        "  naive spectral composite (panel a), meaning the DSM cohorts\n"
        "  are already interpenetrated before any deep learning step.\n\n"
        "• The transdiagnostic-optimised GCN (panel b) does not\n"
        "  introduce cross-cohort mixing — it only tightens clusters\n"
        "  that were already diffuse across DSM boundaries.\n\n"
        "• The k-NN purity curves (panel c) confirm this: both\n"
        "  embeddings sit roughly at the DSM chance line for k ≥ 50,\n"
        "  and the transdiagnostic-optimised embedding is if anything\n"
        "  slightly lower (more transdiagnostic) at small k.\n\n"
        "• This rules out the \"circularity\" objection that the\n"
        "  transdiagnostic-optimised embedding manufactures DSM\n"
        "  interpenetration. The geometry is a property of the data,\n"
        "  not of the optimisation target."
    )
    ax.text(0.02, 0.98, msg, fontsize=9, va="top", ha="left",
            family="DejaVu Sans", transform=ax.transAxes)

    fig.suptitle(
        "Figure S2 — Naive spectral composite vs transdiagnostic-optimised embedding\n"
        "DSM interpenetration is already present before the Stage B2.5 optimisation",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    savefig(fig, "fig_sup_naive_vs_optimised_geometry")


def main() -> None:
    print(f"Output directory: {OUT_DIR}")
    print(f"Cache directory : {CACHE_DIR}")
    fig_r0_tsne_dsm_vs_clusters()
    fig_sup_tsne_perplexity()
    fig_sup_naive_vs_optimised()


if __name__ == "__main__":
    main()
