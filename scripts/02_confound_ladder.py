"""Reproduce the §3.1 "confound ladder" — why naïve clustering of raw psychiatric
records recovers nuisance axes, and the controls that remove them.

Four rungs (LABBOOK E4 / FINDINGS §2.1), all on the same engine
(MultipartiteSpectral embedding → k=6 k-means), differing only in the feature
configuration:

  1. ALL features, RAW, with the birth-date kept — `brthdtc` parses to a
     datetime64[ns] integer ≈3.7e17 that dominates cosine → a *spurious* but very
     stable clustering (target: bootstrap ARI ~0.96, ARI-vs-sister ~0.31).
  2. ALL features, robustly scaled (date dropped) → raw labs/anthropometry now
     dominate and a sex×age stratification begins to emerge (qualitative).
  3. Clinical sections, age/sex-residualized, but *_mhoccur kept → the clusters
     are a sex×age stratification (target: cluster↔sex ARI ~0.32 > ↔cohort ~0.19),
     carried by the physical-comorbidity flags.
  4. + *_mhoccur excluded → the sex/age confound collapses (target: ↔sex ~0.005,
     ↔age ~0.008): the principled configuration.

Writes results/confound_ladder.csv (+ prints achieved-vs-target).
Run:  python3 scripts/02_confound_ladder.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sklearn.metrics import adjusted_rand_score  # noqa: E402

from trans_diag import (  # noqa: E402
    ADMINISTRATIVE_FEATURES,
    CLINICAL_SECTIONS,
    build_unified_dataframe,
    load_variables,
    to_harmonized_dataset,
)
from trans_diag.engine import (  # noqa: E402
    FeatureSchema,
    HarmonizedDataset,
    MultipartiteSpectralEmbedding,
    bootstrap_stability,
    run_kmeans,
)

DATA, DICT = REPO / "data", REPO / "face-common-vars.xlsx"
RESULTS = REPO / "results"
K = 6
N_BOOT = 25
EMBED_CONFIG = dict(min_coverage=0.30, min_features_per_partition=3,
                    n_components_per_partition=8, k_neighbours=10, include_4way=True,
                    include_mask_columns=True, l2_normalize=True,
                    feature_mode="cumulative", partition_weighting="sqrt_info")
# §3.1 / E4 landmark numbers (rung 2 = "labs dominate", qualitative — no ARI target;
# the sex×age stratification 0.32>0.19 appears once clinical features are residualized
# but the physical-comorbidity *_mhoccur flags are still in — that is rung 3).
TARGETS = {
    1: {"bootstrap_ari": 0.96, "ari_sister": 0.31},
    3: {"ari_sex": 0.32, "ari_cohort": 0.19},
    4: {"ari_sex": 0.005, "ari_age": 0.008},
}


def _ari(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    m = pd.notna(a) & pd.notna(b)
    return float(adjusted_rand_score(a[m], b[m])) if m.sum() > 1 else float("nan")


def wrap_stub(X: pd.DataFrame, metadata: pd.DataFrame) -> HarmonizedDataset:
    """Wrap an arbitrary numeric matrix in a HarmonizedDataset with a stub schema
    (so the embedding runs on feature sets the dictionary schema would drop)."""
    feats = [{"id": c, "label_fr": c, "block": "all", "type": "continuous",
              "temporal_scope": "current", "cohorts": ("bp", "sz", "dr")} for c in X.columns]
    schema = FeatureSchema.model_validate(
        {"version": "confound-ladder-0.1",
         "blocks": [{"id": "all", "label_fr": "all", "description": "all features"}],
         "features": feats})
    fm = pd.DataFrame([{"feature_id": c, "label_fr": c, "block": "all", "type": "continuous",
                        "temporal_scope": "current", "unit": None, "direction": "none",
                        "cohorts": "bp,sz,dr"} for c in X.columns]).set_index("feature_id")
    return HarmonizedDataset(X=X, metadata=metadata, feature_metadata=fm, schema=schema)


def embed(dataset) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MultipartiteSpectralEmbedding(**EMBED_CONFIG).fit(dataset).transform().values


def metrics(emb, sex, age_tert, sister) -> dict:
    labels = run_kmeans(emb, n_clusters=K, random_state=0).labels
    lab = labels.to_numpy()
    cohort = np.array([c for c, _ in emb.index])
    sh = sister.reindex(emb.index)
    return {
        "n": len(emb), "dims": emb.shape[1],
        "bootstrap_ari": round(bootstrap_stability(emb, n_clusters=K, n_bootstraps=N_BOOT,
                                                    random_state=0)["mean_ari"], 3),
        "ari_sex": round(_ari(sex.reindex(emb.index).to_numpy(), lab), 3),
        "ari_cohort": round(_ari(cohort, lab), 3),
        "ari_age": round(_ari(age_tert.reindex(emb.index).to_numpy(), lab), 3),
        "ari_sister": round(_ari(sh.to_numpy(), lab), 3),
    }


def main() -> int:
    variables = load_variables(DICT)
    mhoccur = {v.canonical_name for v in variables if v.canonical_name.endswith("_mhoccur")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(DATA, DICT, readiness=["READY", "PARTIAL"], format="long")
        # raw, all features (dates dropped by the adapter), for rung 1's base + brthdtc inject
        raw_all = to_harmonized_dataset(df, variables, visit="V0", normalize=False,
                                        exclude=ADMINISTRATIVE_FEATURES)
        full = to_harmonized_dataset(df, variables, visit="V0", exclude=ADMINISTRATIVE_FEATURES)

    sex = full.X["sex"]
    age_tert = pd.Series(pd.qcut(full.X["age"], 3, labels=False, duplicates="drop"),
                         index=full.X.index)
    # sister reference
    ref = pd.read_csv(RESULTS / "v0_clusters_anchor.csv")
    sister = pd.Series(
        ref["cluster"].to_numpy(),
        index=pd.MultiIndex.from_arrays([ref["cohort"].str.lower(), ref["usubjid_patients"].astype(str)],
                                        names=("cohort", "patient_id")))
    sister = sister[~sister.index.duplicated()]

    # birthdate as the dominating ~3.7e17 column (datetime64[ns] → int64)
    v0 = df[df["visit"] == "V0"].copy()
    v0_idx = pd.MultiIndex.from_arrays([v0["cohort"].str.lower(), v0["usubjid_patients"].astype(str)],
                                       names=("cohort", "patient_id"))
    brth = pd.Series(pd.to_datetime(v0["brthdtc"], errors="coerce").astype("int64").to_numpy(),
                     index=v0_idx)
    brth = brth[~brth.index.duplicated()]

    rows = []
    print("Building the confound ladder (engine = MultipartiteSpectral → k=6)...\n")

    # Rung 1: all features RAW + brthdtc (1e17 dominates cosine)
    X1 = raw_all.X.copy()
    X1["brthdtc"] = brth.reindex(X1.index)
    m1 = metrics(embed(wrap_stub(X1, raw_all.metadata)), sex, age_tert, sister)
    m1["rung"] = 1; m1["config"] = "all feat, RAW, +brthdtc(1e17)"; rows.append(m1)

    # Rung 2: all features robustly scaled (date dropped), no residualize
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds2 = to_harmonized_dataset(df, variables, visit="V0", normalize=True,
                                    exclude=ADMINISTRATIVE_FEATURES)
    m2 = metrics(embed(ds2), sex, age_tert, sister)
    m2["rung"] = 2; m2["config"] = "all feat, robust-scaled"; rows.append(m2)

    # Rung 3: clinical sections, residualized, *_mhoccur KEPT
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds3 = to_harmonized_dataset(df, variables, visit="V0", sections=CLINICAL_SECTIONS,
                                    residualize_on=("age", "sex"), normalize=True,
                                    exclude=ADMINISTRATIVE_FEATURES)
    m3 = metrics(embed(ds3), sex, age_tert, sister)
    m3["rung"] = 3; m3["config"] = "clinical, resid, +mhoccur"; rows.append(m3)

    # Rung 4: + *_mhoccur excluded (principled config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds4 = to_harmonized_dataset(df, variables, visit="V0", sections=CLINICAL_SECTIONS,
                                    residualize_on=("age", "sex"), normalize=True,
                                    exclude=set(ADMINISTRATIVE_FEATURES) | mhoccur)
    m4 = metrics(embed(ds4), sex, age_tert, sister)
    m4["rung"] = 4; m4["config"] = "clinical, resid, -mhoccur (129 feat)"; rows.append(m4)

    out = pd.DataFrame(rows).set_index("rung")[
        ["config", "n", "dims", "bootstrap_ari", "ari_sister", "ari_sex", "ari_cohort", "ari_age"]]
    out.to_csv(RESULTS / "confound_ladder.csv")
    print(out.to_string())
    print("\nAchieved vs §3.1 target:")
    for rung, tgt in TARGETS.items():
        for metric, t in tgt.items():
            got = out.loc[rung, metric]
            print(f"  rung {rung} {metric:13s}: got {got:+.3f}  target {t:+.3f}  "
                  f"{'OK' if abs(got - t) <= 0.06 else 'CHECK'}")
    print("\nWrote results/confound_ladder.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
