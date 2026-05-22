"""Cluster medoid extraction + French vignette retrieval.

A cluster's **medoid** is the patient whose embedding is closest to the
cluster centroid. Medoids are the most representative examples of each
cluster and are the natural anchors for clinical interpretation — pulling
the existing ``face_rlvr`` vignette for a medoid gives reviewers a
concrete, clinically realistic "portrait" of the cluster.

This module ties the Stage B embedding back to the Stage 0
``face_rlvr.profiles`` pipeline: given a cluster assignment and the
original CSV paths, it re-runs the cohort-appropriate extractor + vignette
builder on each medoid's row and returns the full French text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ClusterMedoid:
    """The single representative patient chosen for a cluster."""

    cluster: int
    cohort: str
    patient_id: str
    distance_to_centroid: float
    cluster_size: int
    cluster_cohort_mix: dict[str, int]


@dataclass
class MedoidVignetteResult:
    """Result of :func:`fetch_medoid_vignettes`."""

    medoids: list[ClusterMedoid]
    vignettes: dict[int, str] = field(default_factory=dict)  # cluster → vignette
    synthesis: dict[int, str] = field(default_factory=dict)  # cluster → 1-liner synthesis


# ─── Medoid extraction ───────────────────────────────────────────────────────


def extract_cluster_medoids(
    embedding: pd.DataFrame,
    cluster_labels: pd.Series,
    *,
    n_per_cluster: int = 1,
) -> list[ClusterMedoid]:
    """Find the patient(s) closest to each cluster's centroid.

    Parameters
    ----------
    embedding:
        ``(N, d)`` DataFrame indexed by ``MultiIndex[cohort, patient_id]``.
    cluster_labels:
        Series indexed identically to ``embedding`` with integer cluster
        ids. Noise (``-1``) is ignored.
    n_per_cluster:
        Number of medoids to return per cluster. Defaults to 1 (the
        single closest patient). Higher values return the closest
        ``n`` patients sorted by distance.
    """
    if not embedding.index.equals(cluster_labels.index):
        raise ValueError("embedding and cluster_labels must share the same index")

    arr = embedding.to_numpy(dtype=np.float64)
    clusters = sorted(c for c in cluster_labels.unique() if c >= 0)
    out: list[ClusterMedoid] = []

    for cluster in clusters:
        mask = (cluster_labels == cluster).to_numpy()
        cluster_idx = np.where(mask)[0]
        if cluster_idx.size == 0:
            continue
        sub = arr[cluster_idx]
        centroid = sub.mean(axis=0)
        # Cosine distance on normalized embeddings ≡ 0.5 * ||x − y||²
        dists = np.linalg.norm(sub - centroid[None, :], axis=1)
        order = np.argsort(dists)[:n_per_cluster]
        cohort_mix = (
            cluster_labels.index[cluster_idx]
            .to_frame(index=False)["cohort"]
            .value_counts()
            .to_dict()
        )
        for rank_i, pos in enumerate(order):
            global_idx = int(cluster_idx[pos])
            cohort_label, pid = embedding.index[global_idx]
            out.append(
                ClusterMedoid(
                    cluster=int(cluster),
                    cohort=str(cohort_label),
                    patient_id=str(pid),
                    distance_to_centroid=float(dists[pos]),
                    cluster_size=int(cluster_idx.size),
                    cluster_cohort_mix={str(k): int(v) for k, v in cohort_mix.items()},
                )
            )
    return out


# ─── Vignette retrieval ──────────────────────────────────────────────────────


def _extractor_and_builder_for(cohort: str):
    """Return (extract_fn, build_fn) for the given cohort."""
    from face_rlvr.profiles import (
        extract_asp_patient,
        extract_bp_patient,
        extract_dr_patient,
        extract_sz_patient,
        build_asp_profile,
        build_bp_profile,
        build_dr_profile,
        build_sz_profile,
    )

    table = {
        "bp": (extract_bp_patient, build_bp_profile),
        "sz": (extract_sz_patient, build_sz_profile),
        "dr": (extract_dr_patient, build_dr_profile),
        "asp": (extract_asp_patient, build_asp_profile),
    }
    if cohort not in table:
        raise ValueError(f"Unknown cohort: {cohort!r}")
    return table[cohort]


def fetch_medoid_vignettes(
    medoids: list[ClusterMedoid],
    csv_paths: dict[str, str | Path],
) -> MedoidVignetteResult:
    """Build the ``face_rlvr`` vignette for each medoid patient.

    Parameters
    ----------
    medoids:
        List produced by :func:`extract_cluster_medoids`.
    csv_paths:
        Mapping ``{cohort: Path}`` — same format as
        :func:`face_stratification.build_harmonized_dataset`.
    """
    from face_rlvr.profiles.glossary_loader import get_cohort_column_map

    # Group medoids by cohort so we only open each CSV once.
    by_cohort: dict[str, list[ClusterMedoid]] = {}
    for m in medoids:
        by_cohort.setdefault(m.cohort, []).append(m)

    vignettes: dict[int, str] = {}
    synthesis: dict[int, str] = {}

    for cohort, m_list in by_cohort.items():
        path = Path(csv_paths[cohort])
        if not path.is_file():
            logger.warning("CSV for cohort %s not found at %s", cohort, path)
            continue

        extract_fn, build_fn = _extractor_and_builder_for(cohort)
        cm = get_cohort_column_map(cohort)
        id_col = cm.patient_id_column

        wanted_ids = {m.patient_id for m in m_list}
        # Read only what we need — still load the full CSV because id_col
        # is usually the first column but we don't assume that.
        df = pd.read_csv(path, low_memory=False)
        df[id_col] = df[id_col].astype(str)
        sub = df[df[id_col].isin(wanted_ids)]

        for m in m_list:
            rows = sub[sub[id_col] == m.patient_id]
            if rows.empty:
                logger.warning(
                    "Patient %s (cohort %s) not found in CSV", m.patient_id, cohort
                )
                continue
            row = rows.iloc[0]
            try:
                data = extract_fn(row)
                profile = build_fn(data)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Failed to build vignette for %s:%s — %s",
                    cohort, m.patient_id, exc,
                )
                continue
            vignettes[m.cluster] = profile.full_vignette
            # Try to extract a synthesis one-liner if present
            synth = getattr(profile, "synthesis_section", "") or ""
            synthesis[m.cluster] = synth.strip()[:500]

    return MedoidVignetteResult(
        medoids=medoids,
        vignettes=vignettes,
        synthesis=synthesis,
    )
