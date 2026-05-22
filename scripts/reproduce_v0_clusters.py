"""Reproduce the parallel face_stratification V0 clusters and emit a verified anchor.

Step b2 of the merge. We adopt their clustering engine; before wiring up the
per-visit longitudinal bridge we confirm we can drive their clustering from the
saved multipartite-spectral embedding and that we exactly reproduce their
published cluster x cohort contingency.

Their recipe (face_stratification/clustering/algorithms.run_kmeans, with k
selected by their composite transdiagnostic score): plain
    sklearn.cluster.KMeans(n_clusters=7, random_state=0, n_init=10)
on the already-L2-normalized 99-dim embedding. k=7 is their composite optimum
(silhouette peaks at k=11; composite peaks at k=7 -> the 7-cluster solution on
disk). No extra preprocessing.

Inputs  (copied into data/external/face_stratification/):
    multipartite_embedding.parquet   MultiIndex[cohort, patient_id] x 99 dims
    multipartite_contingency.csv     their row-normalized cluster x cohort table
Output  (results/):
    v0_clusters_anchor.csv           per-patient V0 cluster, keyed to OUR ids
                                     (patient_uid, cohort, usubjid_patients, cluster)
                                     BP/SZ/DR only (ASP dropped: no longitudinal data)

Run:  python3 scripts/reproduce_v0_clusters.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_DIR = REPO_ROOT / "data" / "external" / "face_stratification"
EMB_PATH = ENGINE_DIR / "multipartite_embedding.parquet"
REF_CONTINGENCY = ENGINE_DIR / "multipartite_contingency.csv"
OUT_PATH = REPO_ROOT / "results" / "v0_clusters_anchor.csv"

BEST_K = 7          # their composite-score optimum
RANDOM_STATE = 0    # run_kmeans default
N_INIT = 10         # run_kmeans default
COHORTS = ["asp", "bp", "dr", "sz"]   # contingency column order
OUR_COHORTS = {"BP", "SZ", "DR"}      # cohorts we carry forward (have longitudinal data)


def row_normalized_contingency(labels: pd.Series, cohorts: pd.Series) -> pd.DataFrame:
    """cluster x cohort table, each cluster row summing to 1 (fraction by cohort)."""
    ct = pd.crosstab(labels, cohorts)
    ct = ct.reindex(columns=COHORTS, fill_value=0)
    return ct.div(ct.sum(axis=1), axis=0)


def match_to_reference(mine: pd.DataFrame, ref: pd.DataFrame) -> tuple[dict, float]:
    """Greedily match my cluster rows to reference rows by L1 distance.

    KMeans cluster ids are arbitrary, so we align up to a permutation and report
    the worst residual. Returns (my_cluster -> ref_cluster, max_L1_residual).
    """
    ref_indexed = ref.set_index("Cluster") if "Cluster" in ref.columns else ref
    ref_vecs = {int(c): ref_indexed.loc[c, COHORTS].to_numpy(float)
                for c in ref_indexed.index}
    mapping, residuals, used = {}, [], set()
    for my_c in mine.index:
        v = mine.loc[my_c, COHORTS].to_numpy(float)
        best_ref, best_d = None, np.inf
        for ref_c, rv in ref_vecs.items():
            if ref_c in used:
                continue
            d = np.abs(v - rv).sum()
            if d < best_d:
                best_ref, best_d = ref_c, d
        mapping[int(my_c)] = int(best_ref)
        used.add(best_ref)
        residuals.append(best_d)
    return mapping, float(max(residuals))


def main() -> int:
    if not EMB_PATH.exists():
        sys.exit(f"Missing embedding: {EMB_PATH}")

    emb = pd.read_parquet(EMB_PATH)
    cohorts = emb.index.get_level_values("cohort")
    print(f"Loaded embedding: {emb.shape[0]:,} patients x {emb.shape[1]} dims")
    print(f"  per cohort: {pd.Series(cohorts).value_counts().to_dict()}")

    # ---- reproduce their clustering (identical recipe) -------------------
    km = KMeans(n_clusters=BEST_K, random_state=RANDOM_STATE, n_init=N_INIT)
    labels = pd.Series(km.fit_predict(emb.to_numpy(np.float64)),
                       index=emb.index, name="cluster")

    # ---- verify against their published contingency ----------------------
    ref = pd.read_csv(REF_CONTINGENCY)
    mine = row_normalized_contingency(labels, pd.Series(cohorts, index=labels.index))
    mapping, max_resid = match_to_reference(mine, ref)
    print(f"\nContingency match (k={BEST_K}, up to cluster-id permutation):")
    print(f"  my_cluster -> their_cluster: {mapping}")
    print(f"  max per-cluster L1 residual: {max_resid:.4f}")
    ok = max_resid < 0.02
    print(f"  REPRODUCED: {'YES' if ok else 'NO (residual too large)'}")

    print("\nReproduced cluster x cohort (row-normalized):")
    show = mine.copy()
    show.insert(0, "n", labels.value_counts().reindex(mine.index).values)
    print(show.round(3).to_string())

    # ---- emit verified V0 anchor, keyed to OUR ids -----------------------
    # Their key: (cohort lowercase, patient_id str). Ours: (cohort upper, usubjid int).
    anchor = labels.rename("their_cluster").reset_index()
    anchor["cohort"] = anchor["cohort"].str.upper()
    anchor = anchor[anchor["cohort"].isin(OUR_COHORTS)].copy()
    anchor["usubjid_patients"] = anchor["patient_id"].astype(np.int64)
    anchor["patient_uid"] = anchor["cohort"] + "::" + anchor["usubjid_patients"].astype(str)
    # Relabel clusters to their canonical ids for downstream readability
    anchor["cluster"] = anchor["their_cluster"].map(mapping).astype(int)
    anchor = anchor[["patient_uid", "cohort", "usubjid_patients", "cluster"]]

    OUT_PATH.parent.mkdir(exist_ok=True)
    anchor.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")
    print(f"  V0 anchor: {len(anchor):,} BP/SZ/DR patients")
    print(f"  per cohort: {anchor['cohort'].value_counts().to_dict()}")
    print(f"  cluster sizes: {anchor['cluster'].value_counts().sort_index().to_dict()}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
