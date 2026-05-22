"""Generate publication-grade Stage A figures.

Outputs PNG (300 DPI) + PDF versions to docs/face_stratification/stage_a/.

Figures produced
----------------
fig_a1_cohort_sizes        — FACE cohort sample sizes + total stack.
fig_a2_block_coverage      — Cohort x clinical-block feature-count heatmap.
fig_a3_block_thresholds    — Per-block semantic-overlap constraints & metric.
fig_a4_graph_structure     — Candidate nodes & edges per block type.
fig_a5_cohort_assortativity — Per-block cohort assortativity (diverging).
fig_a6_feature_composition — Feature types, directions, and cohort coverage.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "config" / "face_stratification" / "feature_schema.yaml"
MASKED_SUMMARY = ROOT / "output" / "stratification" / "stage_a_masked_summary.json"
OUT_DIR = ROOT / "docs" / "face_stratification" / "stage_a"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

# Real full-cohort sample sizes (from docs/face_stratification/stage_a.md §6.4).
COHORT_SIZES_FULL = {"bp": 6252, "sz": 2209, "asp": 2001, "dr": 552}

BLOCK_LABELS = {
    "demographics": "Demographics",
    "mood": "Mood",
    "psychosis": "Psychosis",
    "anxiety_impulsivity": "Anxiety / impulsivity",
    "functioning": "Functioning",
    "sleep_circadian": "Sleep / circadian",
    "cognition": "Cognition",
    "biology": "Biology",
    "treatment": "Treatment",
    "substance": "Substance use",
    "trauma": "Trauma",
    "family_history": "Family history",
    "comorbidities": "Comorbidities",
    "suicide_history": "Suicide history",
    "psychiatric_history": "Psychiatric history",
    "cohort_specific": "Cohort-specific",
    "transdiagnostic": "Transdiagnostic",
}

METRIC_COLORS = {
    "cosine": "#3B6BA5",
    "gower": "#D95F02",
    "euclidean": "#7570B3",
}


def savefig(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png / .pdf")


# ─── Load ─────────────────────────────────────────────────────────────────────
def load_schema() -> dict:
    with SCHEMA_PATH.open() as fh:
        return yaml.safe_load(fh)


def load_masked_summary() -> dict:
    with MASKED_SUMMARY.open() as fh:
        return json.load(fh)


# ─── Figure 1 — Cohort sizes ──────────────────────────────────────────────────
def fig_cohort_sizes(schema: dict) -> None:
    sizes = COHORT_SIZES_FULL
    total = sum(sizes.values())
    df = pd.DataFrame(
        {
            "cohort": [COHORT_LABELS[c] for c in COHORT_ORDER],
            "n": [sizes[c] for c in COHORT_ORDER],
            "pct": [sizes[c] / total * 100 for c in COHORT_ORDER],
            "color": [COHORT_PALETTE[c] for c in COHORT_ORDER],
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.35, 1]})

    # (a) horizontal bar
    ax = axes[0]
    bars = ax.barh(
        df["cohort"], df["n"], color=df["color"], edgecolor="white", linewidth=1.2
    )
    for bar, n, pct in zip(bars, df["n"], df["pct"]):
        ax.text(
            bar.get_width() + 70,
            bar.get_y() + bar.get_height() / 2,
            f"{n:,} ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=10,
            color="#333",
        )
    ax.set_xlabel("Number of patients")
    ax.set_title("(a) FACE cohort sample sizes")
    ax.set_xlim(0, max(df["n"]) * 1.22)
    ax.invert_yaxis()
    sns.despine(ax=ax, left=True)

    # (b) summary donut with totals
    ax = axes[1]
    wedges, _ = ax.pie(
        df["n"],
        colors=df["color"],
        startangle=90,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
    )
    ax.text(
        0, 0.06, f"{total:,}", ha="center", va="center", fontsize=22, fontweight="bold"
    )
    ax.text(0, -0.18, "patients", ha="center", va="center", fontsize=11, color="#555")
    ax.text(
        0, -0.38, f"{len(schema['features'])} features", ha="center", va="center", fontsize=10, color="#888"
    )
    ax.set_title("(b) Harmonized dataset")
    ax.legend(
        wedges,
        [f"{c}: {n:,}" for c, n in zip(df["cohort"], df["n"])],
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=9,
    )

    fig.suptitle(
        "Stage A — FACE multi-cohort harmonized dataset",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    savefig(fig, "fig_a1_cohort_sizes")


# ─── Figure 2 — Block × cohort coverage heatmap ───────────────────────────────
def fig_block_coverage(schema: dict) -> None:
    block_cohort_counts: dict[str, Counter] = defaultdict(Counter)
    block_totals: Counter = Counter()
    for feat in schema["features"]:
        block = feat["block"]
        block_totals[block] += 1
        for cohort in feat["cohorts"]:
            block_cohort_counts[block][cohort] += 1

    block_order = [b["id"] for b in schema["blocks"]]
    data = np.zeros((len(block_order), len(COHORT_ORDER)), dtype=int)
    for i, block in enumerate(block_order):
        for j, cohort in enumerate(COHORT_ORDER):
            data[i, j] = block_cohort_counts[block].get(cohort, 0)

    labels = np.array(
        [
            [f"{data[i, j]}/{block_totals[block_order[i]]}" if data[i, j] else "—"
             for j in range(len(COHORT_ORDER))]
            for i in range(len(block_order))
        ]
    )
    row_labels = [f"{BLOCK_LABELS[b]}  ({block_totals[b]})" for b in block_order]

    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    sns.heatmap(
        data,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar_kws={"label": "# features provided"},
        linewidths=0.6,
        linecolor="white",
        xticklabels=[COHORT_LABELS[c] for c in COHORT_ORDER],
        yticklabels=row_labels,
        ax=ax,
        annot_kws={"fontsize": 9, "color": "#222"},
    )
    ax.set_title(
        "Stage A — Feature coverage by clinical block × cohort\n"
        f"(89 unified features over 16 blocks; row total in parentheses)",
        pad=14,
    )
    ax.set_xlabel("Cohort")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.setp(ax.get_yticklabels(), rotation=0)
    savefig(fig, "fig_a2_block_coverage")


# ─── Figure 3 — Per-block thresholds & metrics ────────────────────────────────
def fig_block_thresholds(schema: dict) -> None:
    rows = []
    for b in schema["blocks"]:
        rows.append(
            {
                "block": BLOCK_LABELS[b["id"]],
                "metric": b["metric"],
                "min_fraction_present": b["min_fraction_present"],
                "min_shared_features": b.get("min_shared_features", 0),
            }
        )
    df = pd.DataFrame(rows).sort_values("min_fraction_present", ascending=True).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 7), sharey=True)

    # (a) min_fraction_present
    ax = axes[0]
    bar_colors = [METRIC_COLORS[m] for m in df["metric"]]
    ax.barh(df["block"], df["min_fraction_present"], color=bar_colors, edgecolor="white")
    for i, (val, metric) in enumerate(zip(df["min_fraction_present"], df["metric"])):
        ax.text(val + 0.01, i, f"{val:.2f}", va="center", fontsize=9, color="#333")
    ax.set_xlim(0, 0.78)
    ax.set_xlabel("min_fraction_present")
    ax.set_title("(a) Node-inclusion threshold per block")
    sns.despine(ax=ax, left=True)

    # (b) min_shared_features
    ax = axes[1]
    ax.barh(df["block"], df["min_shared_features"], color=bar_colors, edgecolor="white")
    for i, val in enumerate(df["min_shared_features"]):
        ax.text(val + 0.08, i, f"{val}", va="center", fontsize=9, color="#333")
    ax.set_xlim(0, max(df["min_shared_features"]) + 1.2)
    ax.set_xlabel("min_shared_features")
    ax.set_title("(b) Edge semantic-overlap threshold")
    sns.despine(ax=ax, left=True)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=METRIC_COLORS[m]) for m in METRIC_COLORS
    ]
    fig.legend(
        legend_handles,
        [m.capitalize() for m in METRIC_COLORS],
        title="Similarity metric",
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.03),
        frameon=False,
    )

    fig.suptitle(
        "Stage A — Semantic-overlap edge constraints per clinical block",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    savefig(fig, "fig_a3_block_thresholds")


# ─── Figure 4 — Graph edges + candidate nodes per block ───────────────────────
def fig_graph_structure(summary: dict) -> None:
    edges = summary["edges_per_type"]
    nodes = summary["candidate_nodes_per_type"]
    n_patients = summary["n_patients"]

    df = pd.DataFrame(
        [
            {
                "block": BLOCK_LABELS.get(b, b),
                "block_id": b,
                "edges": edges.get(b, 0),
                "nodes": nodes.get(b, 0),
                "is_transdiag": b == "transdiagnostic",
            }
            for b in edges.keys()
        ]
    ).sort_values("edges", ascending=True).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 7.5), sharey=True)

    # (a) edges per type
    ax = axes[0]
    colors = [
        "#D62728" if t else "#4C78A8" for t in df["is_transdiag"]
    ]
    ax.barh(df["block"], df["edges"], color=colors, edgecolor="white")
    for i, val in enumerate(df["edges"]):
        ax.text(val + 120, i, f"{val:,}", va="center", fontsize=9, color="#333")
    ax.set_xlabel("# edges in kNN similarity graph")
    ax.set_title("(a) Edges per block")
    ax.set_xlim(0, max(df["edges"]) * 1.18)
    sns.despine(ax=ax, left=True)

    # (b) candidate nodes
    ax = axes[1]
    ax.barh(df["block"], df["nodes"], color=colors, edgecolor="white")
    ax.axvline(n_patients, linestyle="--", color="#555", linewidth=1)
    ax.text(
        n_patients,
        len(df) - 0.2,
        f" N = {n_patients:,}",
        fontsize=9,
        va="top",
        color="#444",
    )
    for i, val in enumerate(df["nodes"]):
        ax.text(val + 14, i, f"{val:,}", va="center", fontsize=9, color="#333")
    ax.set_xlabel("# candidate patient nodes")
    ax.set_title("(b) Patients meeting node threshold")
    ax.set_xlim(0, n_patients * 1.13)
    sns.despine(ax=ax, left=True)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#4C78A8"),
        plt.Rectangle((0, 0), 1, 1, color="#D62728"),
    ]
    fig.legend(
        legend_handles,
        ["Clinical block", "Transdiagnostic edge type"],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )

    fig.suptitle(
        "Stage A — Multi-relational similarity graph\n"
        f"(masked 1,200-patient subset; {len(df)} edge types total)",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    savefig(fig, "fig_a4_graph_structure")


# ─── Figure 5 — Cohort assortativity per block ────────────────────────────────
def fig_cohort_assortativity(summary: dict) -> None:
    asst = summary["cohort_assortativity"]
    df = pd.DataFrame(
        [
            {
                "block": BLOCK_LABELS.get(b, b),
                "assortativity": v,
                "is_transdiag": b == "transdiagnostic",
                "is_singletype": v >= 0.999,
            }
            for b, v in asst.items()
        ]
    ).sort_values("assortativity", ascending=True).reset_index(drop=True)

    def bar_color(row):
        if row["is_transdiag"]:
            return "#D62728"
        if row["is_singletype"]:
            return "#BDBDBD"
        if row["assortativity"] >= 0:
            return "#4C78A8"
        return "#1B7837"

    colors = df.apply(bar_color, axis=1).tolist()

    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.barh(df["block"], df["assortativity"], color=colors, edgecolor="white")
    ax.axvline(0, color="#333", linewidth=0.9)
    for i, (val, single) in enumerate(zip(df["assortativity"], df["is_singletype"])):
        label = "≈ 1.00 (single-cohort)" if single else f"{val:+.2f}"
        offset = 0.015 if val >= 0 else -0.015
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, i, label, va="center", ha=ha, fontsize=9, color="#333")
    ax.set_xlim(-0.38, 1.22)
    ax.set_xlabel("Newman cohort assortativity coefficient")
    ax.set_title(
        "Stage A — Cohort assortativity by edge type\n"
        "(1 = edges stay within a single cohort; 0 = cohort-blind)",
        pad=12,
    )
    sns.despine(ax=ax, left=True)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#4C78A8"),
        plt.Rectangle((0, 0), 1, 1, color="#1B7837"),
        plt.Rectangle((0, 0), 1, 1, color="#BDBDBD"),
        plt.Rectangle((0, 0), 1, 1, color="#D62728"),
    ]
    ax.legend(
        legend_handles,
        [
            "Cross-cohort (positive)",
            "Cross-cohort (negative)",
            "Single-cohort block",
            "Transdiagnostic edge",
        ],
        loc="lower right",
        fontsize=9,
    )
    savefig(fig, "fig_a5_cohort_assortativity")


# ─── Figure 6 — Feature composition ───────────────────────────────────────────
def fig_feature_composition(schema: dict) -> None:
    feats = schema["features"]
    type_counts = Counter(f["type"] for f in feats)
    dir_counts = Counter(f["direction"] for f in feats)

    # per-cohort feature availability
    cohort_counts = Counter()
    for f in feats:
        for c in f["cohorts"]:
            cohort_counts[c] += 1

    # n cohorts per feature (1,2,3,4 — shared breadth)
    breadth = Counter(len(f["cohorts"]) for f in feats)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a) Feature type
    ax = axes[0, 0]
    types_df = (
        pd.DataFrame({"type": list(type_counts.keys()), "n": list(type_counts.values())})
        .sort_values("n", ascending=False)
    )
    palette_a = sns.color_palette("Set2", len(types_df))
    ax.bar(types_df["type"], types_df["n"], color=palette_a, edgecolor="white")
    for i, v in enumerate(types_df["n"]):
        ax.text(i, v + 0.8, str(v), ha="center", fontsize=10)
    ax.set_ylim(0, max(types_df["n"]) * 1.18)
    ax.set_ylabel("# features")
    ax.set_title("(a) Feature type")
    sns.despine(ax=ax)

    # (b) Clinical direction
    ax = axes[0, 1]
    dir_df = (
        pd.DataFrame({"direction": list(dir_counts.keys()), "n": list(dir_counts.values())})
        .sort_values("n", ascending=False)
    )
    dir_map = {
        "higher_is_worse": "Higher = worse",
        "higher_is_better": "Higher = better",
        "none": "Neutral",
    }
    dir_df["label"] = dir_df["direction"].map(dir_map)
    palette_b = ["#B2182B", "#1B7837", "#BDBDBD"]
    ax.bar(dir_df["label"], dir_df["n"], color=palette_b[: len(dir_df)], edgecolor="white")
    for i, v in enumerate(dir_df["n"]):
        ax.text(i, v + 0.8, str(v), ha="center", fontsize=10)
    ax.set_ylim(0, max(dir_df["n"]) * 1.18)
    ax.set_ylabel("# features")
    ax.set_title("(b) Clinical direction")
    sns.despine(ax=ax)

    # (c) Features provided per cohort
    ax = axes[1, 0]
    cdf = pd.DataFrame(
        {
            "cohort": [COHORT_LABELS[c] for c in COHORT_ORDER],
            "n": [cohort_counts[c] for c in COHORT_ORDER],
        }
    )
    ax.bar(
        cdf["cohort"],
        cdf["n"],
        color=[COHORT_PALETTE[c] for c in COHORT_ORDER],
        edgecolor="white",
    )
    for i, v in enumerate(cdf["n"]):
        ax.text(i, v + 0.8, str(v), ha="center", fontsize=10)
    ax.axhline(len(feats), linestyle="--", color="#555", linewidth=1)
    ax.text(
        3.35,
        len(feats),
        f" all = {len(feats)}",
        fontsize=9,
        va="center",
        color="#444",
    )
    ax.set_ylim(0, len(feats) * 1.12)
    ax.set_ylabel("# features available")
    ax.set_title("(c) Feature availability per cohort")
    sns.despine(ax=ax)

    # (d) Cross-cohort breadth
    ax = axes[1, 1]
    bdf = pd.DataFrame(
        {
            "shared_in": sorted(breadth.keys()),
            "n": [breadth[k] for k in sorted(breadth.keys())],
        }
    )
    bdf["label"] = bdf["shared_in"].map(
        {1: "1 cohort\n(single)", 2: "2 cohorts", 3: "3 cohorts", 4: "4 cohorts\n(all)"}
    )
    palette_d = sns.color_palette("YlGnBu", len(bdf))
    ax.bar(bdf["label"], bdf["n"], color=palette_d, edgecolor="white")
    for i, v in enumerate(bdf["n"]):
        ax.text(i, v + 0.6, str(v), ha="center", fontsize=10)
    ax.set_ylim(0, max(bdf["n"]) * 1.2)
    ax.set_ylabel("# features")
    ax.set_title("(d) Cross-cohort breadth")
    sns.despine(ax=ax)

    fig.suptitle(
        "Stage A — Unified feature schema composition (89 features × 16 blocks)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    savefig(fig, "fig_a6_feature_composition")


# ─── Driver ───────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Output directory: {OUT_DIR}")
    schema = load_schema()
    summary = load_masked_summary()

    fig_cohort_sizes(schema)
    fig_block_coverage(schema)
    fig_block_thresholds(schema)
    fig_graph_structure(summary)
    fig_cohort_assortativity(summary)
    fig_feature_composition(schema)

    print("Done.")


if __name__ == "__main__":
    main()
