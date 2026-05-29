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
    ADMINISTRATIVE_FEATURES,  # patient IDs, site codes — labels, not features
    CLINICAL_SECTIONS,        # the subset of sections containing clinical questionnaires only
    build_unified_dataframe,
    load_variables,
    to_harmonized_dataset,
)
from trans_diag.engine import (  # noqa: E402
    FeatureSchema,
    HarmonizedDataset,
    MultipartiteSpectralEmbedding,  # the spectral graph embedding used on all four rungs
    bootstrap_stability,            # measures how reproducible the clusters are across data subsamples
    run_kmeans,
)

DATA, DICT = REPO / "data", REPO / "data" / "face-common-vars.xlsx"
RESULTS = REPO / "results"

# k=6 is inherited from the sister project (their non-ASP cluster count).
# It is only a placeholder here — the real k selection happens in script 03.
K = 6
N_BOOT = 25  # number of bootstrap subsamples for stability measurement (more = more reliable, slower)

# Hyperparameters for the spectral embedding — same across all four rungs so that
# any difference in the output is purely due to the feature configuration, not the algorithm.
EMBED_CONFIG = dict(min_coverage=0.30, min_features_per_partition=3,
                    n_components_per_partition=8, k_neighbours=10, include_4way=True,
                    include_mask_columns=True, l2_normalize=True,
                    feature_mode="cumulative", partition_weighting="sqrt_info")

# Known landmark numbers from LABBOOK E4. After each rung we compare the achieved
# value to these targets and print OK / CHECK — this is a reproducibility assertion,
# not a tuning criterion. Rung 2 has no numeric target because its result is qualitative
# ("labs dominate", sex×age stratification begins to emerge).
TARGETS = {
    1: {"bootstrap_ari": 0.96, "ari_sister": 0.31},
    3: {"ari_sex": 0.32, "ari_cohort": 0.19},
    4: {"ari_sex": 0.005, "ari_age": 0.008},
}


def _ari(a, b) -> float:
    # ARI = Adjusted Rand Index. Measures overlap between two labellings of the same patients.
    #   0  → no more agreement than random chance
    #   1  → identical labellings
    #  <0  → less agreement than random (very rare in practice)
    # We use it to ask: "do our clusters accidentally align with sex, age, or cohort?"
    # If yes, the clustering is recovering a demographic confound, not psychiatric structure.
    # NaNs are dropped because not every patient has every variable (e.g. sister labels
    # only cover patients present in that project).
    a, b = np.asarray(a), np.asarray(b)
    m = pd.notna(a) & pd.notna(b)
    return float(adjusted_rand_score(a[m], b[m])) if m.sum() > 1 else float("nan")


def wrap_stub(X: pd.DataFrame, metadata: pd.DataFrame) -> HarmonizedDataset:
    # The embedding engine expects a HarmonizedDataset with a validated FeatureSchema.
    # Rung 1 uses a feature set (all raw columns + birthdate) that doesn't exist in the
    # dictionary schema, so we build a minimal "stub" schema on the fly that just declares
    # every column as a generic continuous feature. This lets us run the engine on
    # arbitrary feature sets without touching the real dictionary.
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
    # Run the multipartite spectral embedding and return the low-dimensional patient coordinates.
    # The result is a (n_patients × n_dims) DataFrame that k-means will cluster.
    # Warnings are suppressed because the engine emits expected convergence notices.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MultipartiteSpectralEmbedding(**EMBED_CONFIG).fit(dataset).transform().values


def metrics(emb, sex, age_tert, sister) -> dict:
    # Run k-means once (fixed seed) to get cluster labels, then measure
    # how much those labels agree with known demographic and reference groupings.
    labels = run_kmeans(emb, n_clusters=K, random_state=0).labels
    lab = labels.to_numpy()
    cohort = np.array([c for c, _ in emb.index])  # "bp"/"sz"/"dr" per patient, from the MultiIndex
    sh = sister.reindex(emb.index)                 # align sister labels to this embedding's patient order

    return {
        "n": len(emb), "dims": emb.shape[1],

        # bootstrap_ari: run k-means on N_BOOT random 80% subsamples of the data, re-cluster,
        # then measure ARI between each subsample's labels and the full-data labels projected back.
        # High value (→1) = same patients always end up in the same cluster regardless of which
        # subsample is used → the clusters are stable, not noise. Rung 1 scores ~0.96 — very stable,
        # but that stability is driven by birthdate, not genuine psychiatric structure.
        "bootstrap_ari": round(bootstrap_stability(emb, n_clusters=K, n_bootstraps=N_BOOT,
                                                    random_state=0)["mean_ari"], 3),

        # ari_sex: do the clusters align with patient sex?
        # Should be ~0. A high value means the algorithm is recovering a sex stratification,
        # not a disease-relevant structure.
        "ari_sex": round(_ari(sex.reindex(emb.index).to_numpy(), lab), 3),

        # ari_cohort: do the clusters align with BP / SZ / DR diagnosis?
        # Should be low for trans-diagnostic discovery. A high value means clusters = diagnoses,
        # which is circular (we already knew patients' diagnoses from their referral).
        "ari_cohort": round(_ari(cohort, lab), 3),

        # ari_age: do the clusters align with age tertile (young / middle / older)?
        # Age tertile is used instead of raw age because ARI requires discrete labels.
        # Should be ~0 for the same reason as ari_sex.
        "ari_age": round(_ari(age_tert.reindex(emb.index).to_numpy(), lab), 3),

        # ari_sister: do our clusters agree with the sister 4-cohort project's clusters?
        # Used only for rung 1: a value of ~0.31 shows the two projects accidentally produced
        # similar clusters when both used raw features dominated by birthdate.
        # Once confounds are removed (rung 4) this metric becomes irrelevant.
        "ari_sister": round(_ari(sh.to_numpy(), lab), 3),
    }


def main() -> int:
    variables = load_variables(DICT)

    # Collect all physical-comorbidity flag variable names (those ending in "_mhoccur").
    # These are binary flags like "lupus_mhoccur", "myocardial_infarction_mhoccur".
    # They are strongly sex- and age-dimorphic (lupus → female, MI → older male),
    # so they carry the demographic confound even after residualization. Rung 4 excludes them.
    mhoccur = {v.canonical_name for v in variables if v.canonical_name.endswith("_mhoccur")}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(DATA, DICT, readiness=["READY", "PARTIAL"], format="long")

        # normalize=False: keep raw values (no scaling). Needed for rung 1 to demonstrate
        # that unscaled birthdate (≈3.7e17) catastrophically dominates cosine similarity.
        raw_all = to_harmonized_dataset(df, variables, visit="V0", normalize=False,
                                        exclude=ADMINISTRATIVE_FEATURES)

        # normalize=True (default): robust scaling (subtract median, divide by IQR).
        # Used to extract sex and age for the ARI checks — they need to be aligned
        # to the same patient index as the embedding, which this call provides.
        full = to_harmonized_dataset(df, variables, visit="V0", exclude=ADMINISTRATIVE_FEATURES)

    sex = full.X["sex"]

    # Bin age into 3 equal-frequency tertiles (young / middle / older).
    # ARI needs discrete labels, not a continuous number.
    # duplicates="drop" handles ties at the bin edges.
    age_tert = pd.Series(pd.qcut(full.X["age"], 3, labels=False, duplicates="drop"),
                         index=full.X.index)

    # Sister reference: cluster labels from the separate 4-cohort (BP+SZ+DR+ASP) project.
    # This file is gitignored (per-patient derived data, never committed).
    # If absent, ari_sister will be NaN — all other metrics still compute correctly.
    _anchor = RESULTS / "v0_clusters_anchor.csv"
    if _anchor.exists():
        ref = pd.read_csv(_anchor)
        sister = pd.Series(
            ref["cluster"].to_numpy(),
            index=pd.MultiIndex.from_arrays([ref["cohort"].str.lower(), ref["usubjid_patients"].astype(str)],
                                            names=("cohort", "patient_id")))
        sister = sister[~sister.index.duplicated()]
    else:
        print("  [warn] results/v0_clusters_anchor.csv not found — ari_sister will be NaN")
        sister = pd.Series(dtype=float,
                           index=pd.MultiIndex.from_tuples([], names=("cohort", "patient_id")))

    # Prepare birthdate as an integer column for rung 1.
    # datetime64[ns] parsed from "1978-03-15" becomes nanoseconds since epoch ≈ 3.7×10^17.
    # Everything else in the matrix is on the 0–100 scale, so this single column
    # completely dominates cosine similarity → patients with similar birthyears
    # cluster together regardless of their symptoms. This is the trap rung 1 demonstrates.
    v0 = df[df["visit"] == "V0"].copy()
    v0_idx = pd.MultiIndex.from_arrays([v0["cohort"].str.lower(), v0["usubjid_patients"].astype(str)],
                                       names=("cohort", "patient_id"))
    brth = pd.Series(pd.to_datetime(v0["brthdtc"], errors="coerce").astype("int64").to_numpy(),
                     index=v0_idx)
    brth = brth[~brth.index.duplicated()]

    rows = []
    print("Building the confound ladder (engine = MultipartiteSpectral → k=6)...\n")

    # ── Rung 1: ALL features, RAW, birthdate injected ────────────────────────────
    # Purpose: show that a completely spurious but stable clustering emerges when
    # one column dominates the distance metric. bootstrap_ari ≈ 0.96 looks impressive
    # but is meaningless — the algorithm is clustering by birth year, not by illness.
    X1 = raw_all.X.copy()
    X1["brthdtc"] = brth.reindex(X1.index)            # inject the 1e17 column
    m1 = metrics(embed(wrap_stub(X1, raw_all.metadata)), sex, age_tert, sister)
    m1["rung"] = 1; m1["config"] = "all feat, RAW, +brthdtc(1e17)"; rows.append(m1)

    # ── Rung 2: ALL features, robustly scaled, no birthdate ──────────────────────
    # Purpose: fix the scale problem (robust z-score removes the magnitude difference).
    # But now lab values and anthropometry dominate — these are strongly sex/age-dimorphic,
    # so clusters still reflect demographics rather than psychiatric structure.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds2 = to_harmonized_dataset(df, variables, visit="V0", normalize=True,
                                    exclude=ADMINISTRATIVE_FEATURES)
    m2 = metrics(embed(ds2), sex, age_tert, sister)
    m2["rung"] = 2; m2["config"] = "all feat, robust-scaled"; rows.append(m2)

    # ── Rung 3: Clinical sections only, age/sex residualized, *_mhoccur KEPT ────
    # Purpose: restrict to clinical questionnaires (no raw labs) and explicitly remove
    # the linear effect of age and sex from every feature before embedding.
    # You'd expect this to fix the confound — but it doesn't, because the _mhoccur
    # flags (physical comorbidities) still carry the age/sex signal: lupus is predominantly
    # female, MI is predominantly older and male. ari_sex ≈ 0.32 > ari_cohort ≈ 0.19.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds3 = to_harmonized_dataset(df, variables, visit="V0", sections=CLINICAL_SECTIONS,
                                    residualize_on=("age", "sex"), normalize=True,
                                    exclude=ADMINISTRATIVE_FEATURES)
    m3 = metrics(embed(ds3), sex, age_tert, sister)
    m3["rung"] = 3; m3["config"] = "clinical, resid, +mhoccur"; rows.append(m3)

    # ── Rung 4: Same as rung 3, but *_mhoccur excluded ──────────────────────────
    # Purpose: removing physical comorbidity flags finally kills the demographic confound.
    # ari_sex collapses from 0.32 → ~0.005, ari_age from visible → ~0.008.
    # This is the principled configuration carried forward to all subsequent analyses.
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

    # Verify that the achieved numbers match the LABBOOK E4 targets within ±0.06 tolerance.
    # Prints OK or CHECK — a CHECK means the data or code has diverged from the published result.
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
