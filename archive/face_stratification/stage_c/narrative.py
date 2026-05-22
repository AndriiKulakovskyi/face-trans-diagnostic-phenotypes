"""Per-cluster clinical narrative cards.

For each final consensus cluster, this module assembles a markdown
"narrative card" combining:

- Cluster size and cohort mix
- Shannon entropy of the cohort distribution + transdiagnostic score
- Median and mean per-patient confidence (from the consensus matrix)
- Top-15 enriched features (Mann-Whitney U + BH-FDR), with effect size
  and direction
- 3 medoid patients selected by **highest confidence** within the
  cluster, with their full French ``face_rlvr`` vignettes
- Auto-generated clinical signature paragraph

The cards are written one-per-file under
``output/stratification/stage_c/cluster_cards/`` and are also returned
as an in-memory dict so the notebook can render them inline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from face_stratification.analysis.medoids import (
    ClusterMedoid,
    extract_cluster_medoids,
    fetch_medoid_vignettes,
)

logger = logging.getLogger(__name__)


# ─── Public types ─────────────────────────────────────────────────────────────


@dataclass
class ClusterCard:
    """Structured contents of a per-cluster narrative card."""

    cluster_id: int
    n_patients: int
    cohort_mix: dict[str, int]
    cohort_mix_pct: dict[str, float]
    cohort_entropy_bits: float
    transdiagnostic_score: float
    confidence_mean: float
    confidence_median: float
    confidence_min: float
    confidence_max: float
    top_features: list[dict[str, Any]]  # one row per significant feature
    medoid_patients: list[ClusterMedoid]
    medoid_vignettes: dict[int, str]
    auto_signature: str
    markdown: str


# ─── Auto-signature generation ───────────────────────────────────────────────


def _format_effect(effect: float) -> str:
    sign = "↑" if effect > 0 else "↓"
    return f"{sign}{abs(effect):.2f}"


def _generate_auto_signature(
    cluster_id: int,
    cohort_mix_pct: dict[str, float],
    cohort_entropy_bits: float,
    top_features: list[dict[str, Any]],
    confidence_median: float,
) -> str:
    """A short auto-generated clinical signature paragraph."""
    # Cohort line
    sorted_cohorts = sorted(cohort_mix_pct.items(), key=lambda kv: -kv[1])
    cohort_str = ", ".join(f"{c.upper()} {pct * 100:.0f}%" for c, pct in sorted_cohorts if pct > 0)
    dominant = sorted_cohorts[0]
    is_transdiagnostic = cohort_entropy_bits >= 1.4

    if is_transdiagnostic:
        cluster_type = "**transdiagnostic**"
    elif dominant[1] >= 0.85:
        cluster_type = "DSM-aligned (single-cohort)"
    else:
        cluster_type = "DSM-adjacent (mostly single-cohort)"

    # Top 3 enriched features by absolute effect
    feats = sorted(top_features, key=lambda r: -abs(r.get("effect_rank_biserial", 0)))[:3]
    feat_summary = "; ".join(
        f"{r['feature_id']} ({_format_effect(r['effect_rank_biserial'])})"
        for r in feats
    )

    return (
        f"Cluster {cluster_id} is a {cluster_type} cluster ({cohort_str}; "
        f"Shannon entropy {cohort_entropy_bits:.2f} bits). "
        f"Top distinguishing features: {feat_summary}. "
        f"Median patient-level confidence: {confidence_median:.3f}."
    )


# ─── Markdown rendering ──────────────────────────────────────────────────────


def _render_cluster_card_markdown(card_dict: dict[str, Any]) -> str:
    """Render a structured card dict to markdown."""
    cid = card_dict["cluster_id"]
    n = card_dict["n_patients"]
    cohort_mix = card_dict["cohort_mix"]
    cohort_pct = card_dict["cohort_mix_pct"]
    entropy = card_dict["cohort_entropy_bits"]
    td_score = card_dict["transdiagnostic_score"]
    conf_mean = card_dict["confidence_mean"]
    conf_median = card_dict["confidence_median"]
    conf_min = card_dict["confidence_min"]
    conf_max = card_dict["confidence_max"]
    top = card_dict["top_features"]
    medoids = card_dict["medoid_patients"]
    vignettes = card_dict["medoid_vignettes"]
    sig = card_dict["auto_signature"]

    lines: list[str] = []
    lines.append(f"# Cluster {cid} — narrative card\n")
    lines.append(f"**Auto-signature.** {sig}\n")

    lines.append("## 1. Composition\n")
    lines.append(f"- **n patients:** {n:,}")
    cohort_lines = ", ".join(
        f"{c.upper()} {cohort_mix.get(c, 0):,} ({cohort_pct.get(c, 0) * 100:.1f}%)"
        for c in sorted(cohort_pct, key=lambda x: -cohort_pct[x])
    )
    lines.append(f"- **Cohort mix:** {cohort_lines}")
    lines.append(f"- **Cohort Shannon entropy:** {entropy:.3f} bits "
                 f"(transdiagnostic score: {td_score:.2f} / 1.0)")
    lines.append("")

    lines.append("## 2. Per-patient confidence (consensus matrix)\n")
    lines.append(f"- Mean confidence:   {conf_mean:+.3f}")
    lines.append(f"- Median confidence: {conf_median:+.3f}")
    lines.append(f"- Range:             [{conf_min:+.3f}, {conf_max:+.3f}]")
    lines.append("")
    lines.append(
        "*Confidence = (intra-cluster co-association) − (max extra-cluster co-association). "
        "Positive values mean the patient co-clusters with this cluster more than with any other.*\n"
    )

    lines.append("## 3. Top enriched features (BH q < 0.05)\n")
    if not top:
        lines.append("*(No features survived BH-FDR correction.)*\n")
    else:
        lines.append("| Feature | Effect (rank-biserial) | Median inside | Median outside | BH p-value |")
        lines.append("|---|---:|---:|---:|---:|")
        for f in top[:15]:
            eff = f["effect_rank_biserial"]
            arrow = "↑" if eff > 0 else "↓"
            lines.append(
                f"| `{f['feature_id']}` | {arrow} {abs(eff):.2f} | "
                f"{f['median_inside']:.2f} | {f['median_outside']:.2f} | "
                f"{f['p_value_bh']:.2e} |"
            )
        lines.append("")

    lines.append("## 4. Medoid patients (top 3 by confidence)\n")
    for i, m in enumerate(medoids, 1):
        lines.append(f"### Medoid {i} — {m.cohort.upper()}:{m.patient_id}\n")
        lines.append(f"- Distance to centroid: {m.distance_to_centroid:.3f}")
        lines.append(f"- Cluster size: {m.cluster_size}")
        lines.append("")
        vig = vignettes.get(cid, "") if i == 1 else ""
        # The current MedoidVignetteResult only stores one vignette per cluster.
        # We render it under medoid 1 only.
        if i == 1 and vig:
            preview = vig.strip().split("\n\n")[0]  # synthesis paragraph
            lines.append("**Synthèse clinique:**")
            lines.append("")
            lines.append(f"> {preview}")
            lines.append("")

    return "\n".join(lines)


# ─── Main entry point ────────────────────────────────────────────────────────


def build_cluster_cards(
    *,
    cluster_labels: pd.Series,
    confidence: pd.Series,
    embedding: pd.DataFrame,
    metadata: pd.DataFrame,
    enrichment_table: pd.DataFrame,
    csv_paths: dict[str, Any],
    n_medoids_per_cluster: int = 3,
) -> dict[int, ClusterCard]:
    """Assemble narrative cards for every cluster.

    Parameters
    ----------
    cluster_labels:
        Series indexed by ``MultiIndex[cohort, patient_id]``.
    confidence:
        Per-patient confidence (same index).
    embedding:
        Stage B composite embedding (same index).
    metadata:
        Harmonized dataset metadata DataFrame (same index).
    enrichment_table:
        DataFrame from
        :func:`face_stratification.analysis.enrichment.compute_cluster_feature_enrichment`,
        already filtered to ``significant=True`` rows is fine.
    csv_paths:
        Mapping ``{cohort: csv path}`` for vignette retrieval.
    n_medoids_per_cluster:
        How many medoid patients to extract per cluster.
    """
    cards: dict[int, ClusterCard] = {}

    # Extract embedding-space medoids (centroid-closest)
    emb_medoids = extract_cluster_medoids(
        embedding, cluster_labels, n_per_cluster=n_medoids_per_cluster
    )

    # Group medoids by cluster
    medoids_by_cluster: dict[int, list[ClusterMedoid]] = {}
    for m in emb_medoids:
        medoids_by_cluster.setdefault(m.cluster, []).append(m)

    # Pull vignettes for the *top* medoid (lowest distance) per cluster.
    # The MedoidVignetteResult dict stores one vignette per cluster id, so
    # we need exactly one medoid per cluster to be fetched here.
    top_medoids: list[ClusterMedoid] = []
    for cid, medoids in medoids_by_cluster.items():
        if medoids:
            top_medoids.append(min(medoids, key=lambda m: m.distance_to_centroid))
    vignette_result = fetch_medoid_vignettes(top_medoids, csv_paths=csv_paths)

    cluster_ids = sorted(c for c in cluster_labels.unique() if c >= 0)
    for cid in cluster_ids:
        mask = cluster_labels == cid
        n = int(mask.sum())
        cohort_series = metadata.loc[mask, "cohort"]
        cohort_mix = cohort_series.value_counts().to_dict()
        cohort_pct = {c: int(v) / n for c, v in cohort_mix.items()}

        # Shannon entropy in bits
        probs = np.asarray(list(cohort_pct.values()))
        probs = probs[probs > 0]
        entropy = float(-(probs * np.log2(probs)).sum())
        n_cohorts = int(metadata["cohort"].nunique())
        td_score = entropy / float(np.log2(n_cohorts)) if n_cohorts > 1 else 0.0

        # Confidence stats
        conf_values = confidence.loc[mask].to_numpy()
        conf_mean = float(np.mean(conf_values))
        conf_median = float(np.median(conf_values))
        conf_min = float(np.min(conf_values))
        conf_max = float(np.max(conf_values))

        # Top features for this cluster
        sub = enrichment_table[enrichment_table["cluster"] == cid].copy()
        sub = sub[sub["significant"]]
        sub = sub.sort_values("abs_effect", ascending=False)
        top_features = sub.to_dict("records")

        # Medoids
        medoids_here = medoids_by_cluster.get(cid, [])

        sig = _generate_auto_signature(
            cid, cohort_pct, entropy, top_features, conf_median
        )

        card_dict = {
            "cluster_id": cid,
            "n_patients": n,
            "cohort_mix": {str(k): int(v) for k, v in cohort_mix.items()},
            "cohort_mix_pct": {str(k): float(v) for k, v in cohort_pct.items()},
            "cohort_entropy_bits": entropy,
            "transdiagnostic_score": td_score,
            "confidence_mean": conf_mean,
            "confidence_median": conf_median,
            "confidence_min": conf_min,
            "confidence_max": conf_max,
            "top_features": top_features[:15],
            "medoid_patients": medoids_here,
            "medoid_vignettes": vignette_result.vignettes,
            "auto_signature": sig,
        }
        markdown = _render_cluster_card_markdown(card_dict)
        cards[cid] = ClusterCard(**card_dict, markdown=markdown)
    return cards


def write_cluster_cards(
    cards: dict[int, ClusterCard],
    out_dir: str | Path,
) -> Path:
    """Write each card to ``{out_dir}/cluster_NN.md``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for cid, card in cards.items():
        with open(out / f"cluster_{cid:02d}.md", "w") as fh:
            fh.write(card.markdown)
    return out
