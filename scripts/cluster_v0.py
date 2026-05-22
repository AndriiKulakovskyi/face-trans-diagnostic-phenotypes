"""Phase-3 deliverable: cluster the 3-cohort V0 data with the vendored engine.

Goal (ROADMAP §9–10): drive the sister ``face_stratification`` clustering engine
from OUR common-variables features and ask whether we recover the sister's
clinically-meaningful clusters — minus the Asperger (ASP) cluster, which has no
longitudinal data and is dropped here.

Pipeline (all engine code is reused, not reimplemented):

    build_unified_dataframe(READY+PARTIAL, long)         our pipeline
        → to_harmonized_dataset(visit="V0")              our adapter
        → MultipartiteSpectralEmbedding.fit/.transform   engine (no imputation)
        → kmeans_sweep / run_kmeans / bootstrap_stability engine

We then compare our k=6 solution to ``results/v0_clusters_anchor.csv`` (the
sister's 4-cohort clusters projected onto OUR BP/SZ/DR ids; the ASP-dominated
cluster is empty there, leaving 6 populated reference clusters).

Artifacts (results/):
    cluster_v0_embedding.parquet   our multipartite-spectral embedding
    cluster_v0_assignments.csv     per-patient cluster + reference cluster
    cluster_v0_sweep.csv           k vs silhouette / ARI / NMI / cohort-entropy
    cluster_v0_contingency.csv     k=6 cluster × cohort counts
    cluster_v0_meta.json           config, partitions, stability, metrics

Run:  python3 scripts/cluster_v0.py            # k=6 headline, k-sweep 2..10
      python3 scripts/cluster_v0.py --k 7 --readiness READY
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "archive"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from sklearn.metrics import (  # noqa: E402
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from face_common import (  # noqa: E402
    ADMINISTRATIVE_FEATURES,
    CLINICAL_SECTIONS,
    build_unified_dataframe,
    load_variables,
    to_harmonized_dataset,
)
from face_stratification.clustering.algorithms import (  # noqa: E402
    bootstrap_stability,
    run_kmeans,
)
from face_stratification.graph.multipartite import (  # noqa: E402
    MultipartiteSpectralEmbedding,
)

DATA_DIR = REPO_ROOT / "data"
DICT_PATH = REPO_ROOT / "face-common-vars.xlsx"
RESULTS_DIR = REPO_ROOT / "results"
REFERENCE_PATH = RESULTS_DIR / "v0_clusters_anchor.csv"

# Embedding config — the sister's reference run (multipartite_manifest.json),
# applied to our 3 cohorts. include_4way is a harmless no-op here (there is no
# 4-way subset without ASP); bp+dr+sz is the top partition.
EMBED_CONFIG = dict(
    min_coverage=0.30,
    min_features_per_partition=3,
    n_components_per_partition=8,
    k_neighbours=10,
    include_4way=True,
    include_mask_columns=True,
    l2_normalize=True,
    feature_mode="cumulative",
    partition_weighting="sqrt_info",
)
RANDOM_STATE = 0
SILHOUETTE_SAMPLE = 5000


def _git_rev() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--k", type=int, default=6,
                   help="headline cluster count (default 6 = sister's non-ASP clusters)")
    p.add_argument("--k-sweep", type=int, nargs=2, default=(2, 10),
                   metavar=("LO", "HI"), help="inclusive k range for the sweep")
    p.add_argument("--readiness", nargs="+", default=["READY", "PARTIAL"],
                   help="cluster_readiness prefixes (default: READY PARTIAL)")
    p.add_argument("--n-bootstraps", type=int, default=25,
                   help="bootstrap resamples for stability (default 25)")
    return p.parse_args()


def cohort_entropy_mean(labels: pd.Series, cohorts: pd.Series) -> float:
    """Mean (size-weighted) normalized cohort entropy across clusters.

    1.0 = clusters are maximally mixed across cohorts (transdiagnostic);
    0.0 = every cluster is single-cohort (DSM-aligned).
    """
    n_cohorts = cohorts.nunique()
    if n_cohorts < 2:
        return float("nan")
    log_k = np.log(n_cohorts)
    total, weight = 0.0, 0
    for _, idx in labels.groupby(labels).groups.items():
        dist = cohorts.loc[idx].value_counts(normalize=True).to_numpy()
        ent = -(dist * np.log(dist + 1e-12)).sum() / log_k
        total += ent * len(idx)
        weight += len(idx)
    return float(total / weight) if weight else float("nan")


def load_reference(embedding_index: pd.MultiIndex) -> pd.Series:
    """Sister cluster labels aligned to our embedding index (NaN where absent)."""
    if not REFERENCE_PATH.exists():
        return pd.Series(index=embedding_index, dtype="float64", name="their_cluster")
    ref = pd.read_csv(REFERENCE_PATH)
    ref_idx = pd.MultiIndex.from_arrays(
        [ref["cohort"].str.lower().to_numpy(), ref["usubjid_patients"].astype(str).to_numpy()],
        names=("cohort", "patient_id"),
    )
    ref_s = pd.Series(ref["cluster"].to_numpy(), index=ref_idx, name="their_cluster")
    ref_s = ref_s[~ref_s.index.duplicated(keep="first")]
    return ref_s.reindex(embedding_index)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING)
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── 1. our features → harmonized V0 dataset ─────────────────────────────
    print(f"Loading unified frame (readiness={args.readiness})...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(
            DATA_DIR, DICT_PATH, readiness=args.readiness, format="long"
        )
        variables = load_variables(DICT_PATH)
        # Cluster on psychiatric phenotype, net of demographics:
        #   sections=CLINICAL_SECTIONS  → drop physiology / cognition / demographics
        #   residualize_on=(age, sex)   → regress out the dominant demographic axes
        #   normalize=True              → robust per-feature scaling for cosine
        #   exclude: recruitment site + *_mhoccur physical-comorbidity flags
        #            (HIV/MI/lupus/asthma — physical health, not psychiatric phenotype)
        exclude = set(ADMINISTRATIVE_FEATURES) | {
            v.canonical_name for v in variables if v.canonical_name.endswith("_mhoccur")
        }
        dataset = to_harmonized_dataset(
            df, variables, visit="V0",
            sections=CLINICAL_SECTIONS, residualize_on=("age", "sex"),
            normalize=True, exclude=exclude,
        )
    print(f"  HarmonizedDataset: {dataset.n_patients:,} patients × "
          f"{dataset.n_features} clinical features "
          f"(age/sex-residualized, normalized)  {dataset.cohort_counts().to_dict()}")

    # ── 2. multipartite-spectral embedding (engine, no imputation) ──────────
    print("\nFitting MultipartiteSpectralEmbedding (engine)...")
    model = MultipartiteSpectralEmbedding(**EMBED_CONFIG)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embedding = model.fit(dataset).transform()
    emb = embedding.values
    parts = model.partitions_summary()
    print(f"  embedding: {emb.shape[0]:,} patients × {emb.shape[1]} dims "
          f"from {len(model._partitions)} partitions")
    print(parts.to_string(index=False))
    emb.to_parquet(RESULTS_DIR / "cluster_v0_embedding.parquet")

    cohorts = pd.Series(emb.index.get_level_values("cohort"), index=emb.index)
    their = load_reference(emb.index)
    shared = their.notna()
    print(f"\n  reference overlap: {int(shared.sum()):,} / {len(emb):,} patients "
          f"carry a sister cluster label")

    # ── 3. k sweep ──────────────────────────────────────────────────────────
    lo, hi = args.k_sweep
    print(f"\nk-sweep over k={lo}..{hi}:")
    sweep_rows = []
    for k in range(lo, hi + 1):
        asn = run_kmeans(emb, n_clusters=k, random_state=RANDOM_STATE)
        labels = asn.labels
        sil = float(silhouette_score(
            emb.to_numpy(np.float64), labels.to_numpy(),
            sample_size=min(SILHOUETTE_SAMPLE, len(emb)), random_state=RANDOM_STATE,
        ))
        ari = nmi = float("nan")
        if shared.any():
            ari = float(adjusted_rand_score(their[shared], labels[shared]))
            nmi = float(normalized_mutual_info_score(their[shared], labels[shared]))
        sweep_rows.append({
            "k": k,
            "silhouette": sil,
            "ari_vs_reference": ari,
            "nmi_vs_reference": nmi,
            "cohort_entropy_mean": cohort_entropy_mean(labels, cohorts),
            "inertia": asn.config["inertia"],
            "n_nonempty": int(labels.nunique()),
        })
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(RESULTS_DIR / "cluster_v0_sweep.csv", index=False)
    print(sweep.round(4).to_string(index=False))

    # ── 4. headline k ────────────────────────────────────────────────────────
    k = args.k
    print(f"\nHeadline clustering at k={k}:")
    asn = run_kmeans(emb, n_clusters=k, random_state=RANDOM_STATE)
    labels = asn.labels

    contingency = pd.crosstab(labels, cohorts)
    contingency.index.name = "cluster"
    contingency.to_csv(RESULTS_DIR / "cluster_v0_contingency.csv")
    print("  cluster × cohort (counts):")
    print(contingency.to_string())

    ari = nmi = float("nan")
    cross = None
    if shared.any():
        ari = float(adjusted_rand_score(their[shared], labels[shared]))
        nmi = float(normalized_mutual_info_score(their[shared], labels[shared]))
        cross = pd.crosstab(labels[shared], their[shared].astype(int))
        cross.index.name = "our_cluster"
        cross.columns.name = "their_cluster"
        print(f"\n  vs sister reference (on {int(shared.sum()):,} shared patients): "
              f"ARI={ari:.3f}  NMI={nmi:.3f}")
        print("  our cluster × their cluster (counts):")
        print(cross.to_string())

    # ── 5. bootstrap stability ───────────────────────────────────────────────
    print(f"\nBootstrap stability (k={k}, {args.n_bootstraps} resamples)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stab = bootstrap_stability(
            emb, n_clusters=k, n_bootstraps=args.n_bootstraps, random_state=RANDOM_STATE,
        )
    print(f"  mean pairwise ARI = {stab['mean_ari']:.3f} ± {stab['std_ari']:.3f}")

    # ── 6. persist assignments + meta ────────────────────────────────────────
    assignments = pd.DataFrame({
        "cohort": [c.upper() for c in emb.index.get_level_values("cohort")],
        "usubjid_patients": emb.index.get_level_values("patient_id"),
        "cluster": labels.to_numpy(),
        "their_cluster": their.to_numpy(),
    })
    assignments.insert(
        0, "patient_uid",
        assignments["cohort"] + "::" + assignments["usubjid_patients"].astype(str),
    )
    assignments.to_csv(RESULTS_DIR / "cluster_v0_assignments.csv", index=False)

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_rev": _git_rev(),
        "readiness": args.readiness,
        "normalized": True,
        "residualized_on": ["age", "sex"],
        "clinical_sections": sorted(CLINICAL_SECTIONS),
        "excluded_features": sorted(exclude),
        "embed_config": EMBED_CONFIG,
        "n_patients": int(dataset.n_patients),
        "n_features": int(dataset.n_features),
        "cohort_counts": {k_: int(v) for k_, v in dataset.cohort_counts().items()},
        "embedding_dim": int(emb.shape[1]),
        "partitions": parts.to_dict(orient="records"),
        "headline_k": k,
        "cluster_sizes": labels.value_counts().sort_index().to_dict(),
        "reference_overlap": int(shared.sum()),
        "ari_vs_reference": ari,
        "nmi_vs_reference": nmi,
        "bootstrap_stability": stab,
        "schema_version": dataset.schema.version,
    }
    (RESULTS_DIR / "cluster_v0_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str)
    )

    print("\nWrote:")
    for name in ("cluster_v0_embedding.parquet", "cluster_v0_assignments.csv",
                 "cluster_v0_sweep.csv", "cluster_v0_contingency.csv",
                 "cluster_v0_meta.json"):
        print(f"  {RESULTS_DIR / name}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
