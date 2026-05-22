"""Tests for Stage B review: clustering, metrics, ablation, enrichment, medoids."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from face_stratification import (
    build_harmonized_dataset,
    ConcatenatedEmbedding,
    fit_embedding,
    kmeans_sweep,
    run_kmeans,
    bootstrap_stability,
    compute_cluster_metrics,
)
from face_stratification.harmonization.normalization import (
    fit_normalization,
    fit_per_cohort_normalization,
    transform_normalization,
    transform_per_cohort_normalization,
)
from face_stratification.analysis.enrichment import (
    compute_cluster_feature_enrichment,
    _benjamini_hochberg,
    _rank_biserial,
)
from face_stratification.analysis.medoids import (
    ClusterMedoid,
    extract_cluster_medoids,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATHS = {
    "bp": REPO_ROOT / "data" / "BP.csv",
    "sz": REPO_ROOT / "data" / "SZ.csv",
    "dr": REPO_ROOT / "data" / "DR.csv",
    "asp": REPO_ROOT / "data" / "ASP.csv",
}


def _require_csvs():
    missing = [c for c, p in CSV_PATHS.items() if not p.is_file()]
    if missing:
        pytest.skip(f"FACE CSVs missing: {missing}")
    return CSV_PATHS


# ─── Normalization ablation (pure synthetic) ─────────────────────────────────


def test_per_cohort_normalization_centers_each_cohort_independently():
    """Per-cohort normalization should produce a ~0 median within each cohort."""
    from face_stratification import load_feature_schema
    from face_stratification.harmonization.harmonizer import HarmonizedDataset

    schema = load_feature_schema()
    rng = np.random.default_rng(0)
    cohorts = ("bp", "sz", "dr", "asp")
    # Simulate a cohort shift: BP patients have high MADRS, SZ has low MADRS
    n_per = 30
    rows = []
    index_tuples = []
    for i_c, c in enumerate(cohorts):
        for i in range(n_per):
            index_tuples.append((c, f"p{i_c}_{i}"))
            row = {f.id: rng.normal() for f in schema.features}
            if "inst_madrs_total" in row:
                # Add per-cohort shift
                row["inst_madrs_total"] += 10 * i_c
            rows.append(row)

    index = pd.MultiIndex.from_tuples(index_tuples, names=("cohort", "patient_id"))
    X = pd.DataFrame(rows, index=index, dtype="float64")
    metadata = pd.DataFrame(
        {
            "cohort": [c for c, _ in index_tuples],
            "patient_id": [p for _, p in index_tuples],
            "dsm_diagnosis": [c.upper() for c, _ in index_tuples],
        },
        index=index,
    )
    feature_metadata = pd.DataFrame(index=[f.id for f in schema.features])
    ds = HarmonizedDataset(
        X=X, metadata=metadata, feature_metadata=feature_metadata, schema=schema
    )

    # Global: BP's MADRS should NOT be zero-median
    g = fit_normalization(ds.X, schema)
    Xg = transform_normalization(ds.X, g)
    bp_global = Xg.xs("bp", level="cohort")["inst_madrs_total"].median()

    # Per-cohort: BP's MADRS SHOULD be ~zero-median (within BP)
    p = fit_per_cohort_normalization(ds.X, schema)
    Xp = transform_per_cohort_normalization(ds.X, p)
    bp_per = Xp.xs("bp", level="cohort")["inst_madrs_total"].median()

    assert abs(bp_per) < 0.1, f"per-cohort median should be ~0 but was {bp_per}"
    assert abs(bp_global) > abs(bp_per), (
        "global normalization should leave a non-trivial BP offset when the cohort is shifted"
    )


# ─── Clustering + metrics ────────────────────────────────────────────────────


def test_kmeans_sweep_returns_all_metric_columns():
    csvs = _require_csvs()
    ds = build_harmonized_dataset(csv_paths=csvs, max_rows_per_cohort=40)
    model = ConcatenatedEmbedding.build_default(
        pca_dim=4, td_spectral_dim=6, multiplex_spectral_dim=8
    )
    emb, _ = fit_embedding(ds, model=model)

    sweep = kmeans_sweep(
        emb.values,
        k_values=[3, 4, 5],
        reference_labels=ds.metadata["cohort"].values,
    )
    assert len(sweep) == 3
    expected = {"k", "silhouette", "ari", "nmi", "v_measure", "inertia"}
    assert expected.issubset(sweep.columns)


def test_bootstrap_stability_is_bounded_in_reasonable_range():
    csvs = _require_csvs()
    ds = build_harmonized_dataset(csv_paths=csvs, max_rows_per_cohort=30)
    model = ConcatenatedEmbedding.build_default(
        pca_dim=4, td_spectral_dim=6, multiplex_spectral_dim=8
    )
    emb, _ = fit_embedding(ds, model=model)
    bs = bootstrap_stability(emb.values, n_clusters=4, n_bootstraps=6)
    assert 0.0 <= bs["mean_ari"] <= 1.0
    assert bs["std_ari"] >= 0.0
    assert bs["n_pairs"] > 0


def test_cluster_metrics_match_expected_for_trivial_case():
    """A perfect clustering of a tiny synthetic example should give ARI=1."""
    # 4 patients in 2 perfectly-separated clusters
    embedding = np.array([[0, 0], [0, 0.1], [5, 5], [5, 5.1]], dtype=np.float64)
    labels = np.array([0, 0, 1, 1])
    reference = np.array(["a", "a", "b", "b"])
    m = compute_cluster_metrics(
        embedding, labels, reference, silhouette_sample_size=None
    )
    assert m.ari_vs_reference == pytest.approx(1.0)
    assert m.homogeneity == pytest.approx(1.0)
    assert m.completeness == pytest.approx(1.0)


# ─── Feature enrichment ──────────────────────────────────────────────────────


def test_benjamini_hochberg_basic_case():
    # 5 p-values: the smallest 3 should pass BH at q=0.05
    pvals = np.array([0.001, 0.005, 0.01, 0.5, 0.9])
    reject = _benjamini_hochberg(pvals, 0.05)
    assert reject[0] and reject[1] and reject[2]
    assert not reject[3]
    assert not reject[4]


def test_rank_biserial_bounds_and_sign():
    """Standard Wendt (1972) convention: positive = first group higher."""
    from scipy.stats import mannwhitneyu

    # Test 1: inside > outside → rb should be +1
    inside = np.array([10, 11, 12, 13, 14])
    outside = np.array([1, 2, 3, 4, 5])
    u, _ = mannwhitneyu(inside, outside, alternative="two-sided")
    eff = _rank_biserial(u, inside.size, outside.size)
    assert eff == pytest.approx(1.0)

    # Test 2: inside < outside → rb should be −1
    inside = np.array([1, 2, 3, 4, 5])
    outside = np.array([10, 11, 12, 13, 14])
    u, _ = mannwhitneyu(inside, outside, alternative="two-sided")
    eff = _rank_biserial(u, inside.size, outside.size)
    assert eff == pytest.approx(-1.0)

    # Test 3: identical → rb ≈ 0
    a = np.array([10, 11, 12])
    u, _ = mannwhitneyu(a, a, alternative="two-sided")
    eff = _rank_biserial(u, a.size, a.size)
    assert abs(eff) < 1e-6


def test_enrichment_identifies_planted_signal_with_correct_direction():
    """A feature with an artificial cluster effect should come out significant
    AND have the correct direction (positive because cluster 0 is HIGHER)."""
    from face_stratification import load_feature_schema
    schema = load_feature_schema()
    rng = np.random.default_rng(0)
    index = pd.MultiIndex.from_tuples(
        [("bp", f"p{i}") for i in range(60)], names=("cohort", "patient_id")
    )
    data = {f.id: rng.normal(size=60) for f in schema.features}
    X = pd.DataFrame(data, index=index)

    # Plant a cluster effect on MADRS: first 20 patients have +5, rest 0
    X.loc[X.index[:20], "inst_madrs_total"] += 5.0
    labels = pd.Series([0] * 20 + [1] * 40, index=index)

    enr = compute_cluster_feature_enrichment(X, labels, q_threshold=0.05, min_samples_per_side=5)
    sig = enr.table[enr.table["significant"]]
    assert "inst_madrs_total" in set(sig["feature_id"]), "planted signal should be detected"

    # Cluster 0 was planted ABOVE baseline → effect must be positive
    cluster0_madrs = enr.table[
        (enr.table["cluster"] == 0) & (enr.table["feature_id"] == "inst_madrs_total")
    ].iloc[0]
    assert cluster0_madrs["effect_rank_biserial"] > 0, (
        "Cluster 0 has higher MADRS than cluster 1 — effect should be positive"
    )
    # And cluster 1 should have the opposite sign
    cluster1_madrs = enr.table[
        (enr.table["cluster"] == 1) & (enr.table["feature_id"] == "inst_madrs_total")
    ].iloc[0]
    assert cluster1_madrs["effect_rank_biserial"] < 0


# ─── Medoid extraction ───────────────────────────────────────────────────────


def test_extract_medoids_returns_one_per_cluster():
    index = pd.MultiIndex.from_tuples(
        [("bp", f"p{i}") for i in range(10)], names=("cohort", "patient_id")
    )
    emb = pd.DataFrame(
        np.random.RandomState(0).normal(size=(10, 3)),
        index=index,
        columns=["x0", "x1", "x2"],
    )
    labels = pd.Series([0] * 5 + [1] * 5, index=index)
    medoids = extract_cluster_medoids(emb, labels, n_per_cluster=1)
    assert len(medoids) == 2
    assert {m.cluster for m in medoids} == {0, 1}
    # Each medoid belongs to its cluster
    for m in medoids:
        assert m.cohort == "bp"
        assert m.cluster_size == 5


def test_extract_medoids_respects_n_per_cluster():
    index = pd.MultiIndex.from_tuples(
        [("bp", f"p{i}") for i in range(20)], names=("cohort", "patient_id")
    )
    emb = pd.DataFrame(
        np.random.RandomState(0).normal(size=(20, 3)),
        index=index,
        columns=["x0", "x1", "x2"],
    )
    labels = pd.Series([0] * 10 + [1] * 10, index=index)
    medoids = extract_cluster_medoids(emb, labels, n_per_cluster=3)
    assert len(medoids) == 6
    # First 3 should be cluster 0
    cluster_ids = [m.cluster for m in medoids]
    assert cluster_ids.count(0) == 3
    assert cluster_ids.count(1) == 3


def test_extract_medoids_excludes_noise():
    index = pd.MultiIndex.from_tuples(
        [("bp", f"p{i}") for i in range(10)], names=("cohort", "patient_id")
    )
    emb = pd.DataFrame(
        np.random.RandomState(0).normal(size=(10, 3)),
        index=index,
        columns=["x0", "x1", "x2"],
    )
    labels = pd.Series([-1] * 5 + [0] * 5, index=index)
    medoids = extract_cluster_medoids(emb, labels)
    # Noise cluster -1 should not appear
    assert len(medoids) == 1
    assert medoids[0].cluster == 0


# ─── End-to-end ablation ─────────────────────────────────────────────────────


def test_ablation_runs_both_variants_and_agrees_reasonably():
    csvs = _require_csvs()
    ds = build_harmonized_dataset(csv_paths=csvs, max_rows_per_cohort=30)

    from face_stratification.analysis.ablation import run_normalization_ablation
    result = run_normalization_ablation(
        ds,
        k_clusters=4,
        pca_dim=4,
        td_spectral_dim=6,
        multiplex_spectral_dim=8,
        n_bootstraps=4,
    )
    # Both variants should produce valid metrics
    assert result.global_result.metrics.n_clusters == 4
    assert result.per_cohort_result.metrics.n_clusters == 4
    # Cross-variant ARI should be a valid number in [-1, 1]
    assert -1.0 <= result.variant_vs_variant_ari <= 1.0
    # Summary table has both rows
    summary = result.summary_table()
    assert set(summary.index) == {"global", "per_cohort"}
    assert "silhouette" in summary.columns
    assert "bootstrap_mean_ari" in summary.columns
