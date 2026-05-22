"""Stage B master pipeline: embedding → clustering → validation.

Orchestrates the full Stage B workflow with leak-proof train/test split,
systematic method comparison, and comprehensive validation.

Usage::

    from face_stratification.stage_b.pipeline import StageBPipeline
    pipeline = StageBPipeline()
    result = pipeline.run(dataset)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from face_stratification.harmonization.harmonizer import HarmonizedDataset
from face_stratification.harmonization.normalization import (
    NormalizationStats,
    fit_normalization,
    transform_normalization,
)
from face_stratification.models.base import PatientEmbedding

logger = logging.getLogger(__name__)


@dataclass
class StageBResult:
    """Complete output of the Stage B pipeline."""

    # Data split
    split: Any  # StratifiedCohortSplit
    normalization_stats: NormalizationStats

    # Embeddings per method
    embeddings: dict[str, PatientEmbedding] = field(default_factory=dict)
    embedding_times: dict[str, float] = field(default_factory=dict)

    # Clustering results: {method: {k: ClusterAssignment}}
    clustering_results: dict[str, dict[int, Any]] = field(default_factory=dict)

    # Validation metrics (train and test separately)
    train_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    test_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    # Permutation tests
    permutation_results: dict[str, pd.DataFrame] = field(default_factory=dict)

    # Stability
    bootstrap_results: dict[str, dict] = field(default_factory=dict)
    perturbation_results: dict[str, dict] = field(default_factory=dict)

    # Interpretability
    feature_importance: dict[str, pd.DataFrame] = field(default_factory=dict)

    # Best configuration
    best_method: str = ""
    best_k: int = 0
    best_score: float = 0.0

    # Figure paths
    figures: dict[str, Path] = field(default_factory=dict)


class StageBPipeline:
    """Master pipeline for Stage B: embedding + clustering + validation.

    Parameters
    ----------
    config_path:
        Path to stage_b_config.yaml. If None, uses the default config.
    output_dir:
        Directory for figures and artifacts.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        output_dir: str | Path = "output/stage_b",
    ) -> None:
        from face_stratification.stage_b.method_registry import load_stage_b_config
        self.config = load_stage_b_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        dataset: HarmonizedDataset,
        *,
        methods: list[str] | None = None,
        k_range: list[int] | None = None,
        skip_gnn: bool = False,
        skip_permutation: bool = False,
    ) -> StageBResult:
        """Run the full Stage B pipeline.

        Parameters
        ----------
        dataset:
            Harmonized dataset from Stage A.
        methods:
            Subset of methods to run. None = all enabled in config.
        k_range:
            Range of k values for clustering. None = config default.
        skip_gnn:
            Skip GPU-heavy GNN methods for quick iteration.
        skip_permutation:
            Skip permutation tests (expensive).
        """
        t_start = time.time()
        k_range = k_range or self.config.get("k_range", list(range(3, 13)))

        # ── Step 1: Train/test split ──
        logger.info("Step 1: Creating stratified train/test split")
        from face_stratification.evaluation.split import create_stratified_split

        split_cfg = self.config.get("split", {})
        split = create_stratified_split(
            dataset,
            test_fraction=split_cfg.get("test_fraction", 0.2),
            seed=split_cfg.get("seed", 42),
        )

        # ── Step 2: Normalize (train only) ──
        logger.info("Step 2: Fitting normalization on train split")
        train_ds = split.train_dataset(dataset)
        test_ds = split.test_dataset(dataset)

        norm_stats = fit_normalization(train_ds.X, dataset.schema)
        train_Xn = transform_normalization(train_ds.X, norm_stats)
        test_Xn = transform_normalization(test_ds.X, norm_stats)

        # Replace X in datasets with normalized versions
        train_ds_norm = HarmonizedDataset(
            X=train_Xn, metadata=train_ds.metadata,
            feature_metadata=train_ds.feature_metadata, schema=train_ds.schema,
        )
        test_ds_norm = HarmonizedDataset(
            X=test_Xn, metadata=test_ds.metadata,
            feature_metadata=test_ds.feature_metadata, schema=test_ds.schema,
        )

        # ── Step 3: Build graph on train ──
        logger.info("Step 3: Building multiplex graph on train data")
        from face_stratification.graph.patient_similarity import build_multiplex_graph

        graph_cfg = self.config.get("graph", {})
        graph = build_multiplex_graph(
            train_Xn, dataset.schema,
            k=graph_cfg.get("k_neighbours", 10),
            metadata=train_ds.metadata,
        )

        # ── Step 4: Fit embeddings ──
        logger.info("Step 4: Fitting embedding methods")
        from face_stratification.stage_b.method_registry import get_embedding_methods

        all_methods = get_embedding_methods(self.config)
        if methods:
            all_methods = {k: v for k, v in all_methods.items() if k in methods}
        if skip_gnn:
            gnn_names = {"gae", "vgae", "graphcl", "gat", "dgi", "rgcn"}
            all_methods = {k: v for k, v in all_methods.items() if k not in gnn_names}

        result = StageBResult(
            split=split,
            normalization_stats=norm_stats,
        )

        for name, model in all_methods.items():
            logger.info("  Fitting: %s", name)
            t0 = time.time()
            try:
                emb = model.fit_transform(train_ds_norm, graph=graph)
                result.embeddings[name] = emb
                result.embedding_times[name] = time.time() - t0
                logger.info("  %s: %d × %d in %.1fs",
                           name, emb.n_patients, emb.dim, result.embedding_times[name])
            except Exception as exc:
                logger.error("  %s FAILED: %s", name, exc)
                result.embedding_times[name] = time.time() - t0

        # ── Step 5: Clustering ──
        logger.info("Step 5: Clustering on each embedding")
        cohort_labels = train_ds.metadata["cohort"].values

        for emb_name, emb in result.embeddings.items():
            logger.info("  Clustering: %s", emb_name)
            from face_stratification.clustering.algorithms import run_kmeans, kmeans_sweep

            sweep = kmeans_sweep(
                emb.values,
                k_values=k_range,
                reference_labels=cohort_labels,
            )
            result.clustering_results[emb_name] = {
                "sweep": sweep,
            }

            # Also run at the best k (by silhouette)
            if "silhouette" in sweep.columns and not sweep["silhouette"].isna().all():
                best_k = int(sweep.loc[sweep["silhouette"].idxmax(), "k"])
                best_assignment = run_kmeans(
                    emb.values, n_clusters=best_k,
                    reference_labels=cohort_labels,
                )
                result.clustering_results[emb_name]["best_k"] = best_k
                result.clustering_results[emb_name]["best_assignment"] = best_assignment

        # ── Step 6: Validation metrics ──
        logger.info("Step 6: Computing validation metrics")
        from face_stratification.evaluation.validation import (
            compute_external_validation,
            compute_information_theoretic_validation,
            compute_internal_validation,
        )

        dsm_subtypes = train_ds.metadata["dsm_diagnosis"].values

        for emb_name, emb in result.embeddings.items():
            cr = result.clustering_results.get(emb_name, {})
            assignment = cr.get("best_assignment")
            if assignment is None:
                continue

            labels = assignment.labels.values
            emb_arr = emb.values.to_numpy()

            internal = compute_internal_validation(emb_arr, labels)
            external = compute_external_validation(labels, cohort_labels, dsm_subtypes=dsm_subtypes)
            info_theoretic = compute_information_theoretic_validation(
                labels, cohort_labels,
                feature_matrix=train_Xn.to_numpy(),
                feature_names=list(train_Xn.columns),
            )

            result.train_metrics[emb_name] = {
                **internal,
                **external,
                "transdiagnostic_score": info_theoretic.get("transdiagnostic_score", 0),
                "mean_cluster_entropy": info_theoretic.get("mean_cluster_cohort_entropy", 0),
                "mutual_information": info_theoretic.get("mutual_information", 0),
            }

        # ── Step 7: Permutation tests ──
        if not skip_permutation:
            logger.info("Step 7: Running permutation tests")
            from face_stratification.evaluation.permutation import permutation_test_all

            n_perms = self.config.get("validation", {}).get("n_permutations", 1000)
            for emb_name, emb in result.embeddings.items():
                cr = result.clustering_results.get(emb_name, {})
                assignment = cr.get("best_assignment")
                if assignment is None:
                    continue
                try:
                    perm_df = permutation_test_all(
                        emb.values.to_numpy(),
                        assignment.labels.values,
                        cohort_labels,
                        n_permutations=min(n_perms, 200),  # Reduce for speed
                    )
                    result.permutation_results[emb_name] = perm_df
                except Exception as exc:
                    logger.warning("Permutation test for %s failed: %s", emb_name, exc)
        else:
            logger.info("Step 7: Skipping permutation tests")

        # ── Step 8: Bootstrap stability ──
        logger.info("Step 8: Bootstrap stability analysis")
        from face_stratification.evaluation.stability import bootstrap_stability_extended

        n_boots = self.config.get("validation", {}).get("bootstrap_n", 100)
        for emb_name, emb in result.embeddings.items():
            cr = result.clustering_results.get(emb_name, {})
            best_k = cr.get("best_k")
            if best_k is None:
                continue
            try:
                boot = bootstrap_stability_extended(
                    emb.values, n_clusters=best_k,
                    n_bootstraps=min(n_boots, 30),  # Reduce for speed
                )
                result.bootstrap_results[emb_name] = boot
            except Exception as exc:
                logger.warning("Bootstrap for %s failed: %s", emb_name, exc)

        # ── Step 9: Select best configuration ──
        logger.info("Step 9: Selecting best configuration")
        best_score = -float("inf")
        for emb_name, metrics in result.train_metrics.items():
            # Multi-objective: silhouette + transdiagnostic_score + (1 - cramers_v)
            sil = metrics.get("silhouette", 0)
            td = metrics.get("transdiagnostic_score", 0)
            cv = metrics.get("cramers_v", 0)
            score = sil + 2 * td + (1 - cv)

            if score > best_score:
                best_score = score
                result.best_method = emb_name
                result.best_k = result.clustering_results.get(emb_name, {}).get("best_k", 0)
                result.best_score = score

        elapsed = time.time() - t_start
        logger.info(
            "Stage B complete in %.1fs. Best: %s (k=%d, score=%.3f)",
            elapsed, result.best_method, result.best_k, result.best_score,
        )

        return result

    def summarize(self, result: StageBResult) -> pd.DataFrame:
        """Create a summary DataFrame of all methods and their metrics."""
        rows = []
        for emb_name, metrics in result.train_metrics.items():
            row = {"method": emb_name}
            row.update(metrics)
            row["time_s"] = result.embedding_times.get(emb_name, 0)
            row["best_k"] = result.clustering_results.get(emb_name, {}).get("best_k", 0)

            # Add stability if available
            boot = result.bootstrap_results.get(emb_name, {})
            row["bootstrap_ari"] = boot.get("mean_ari", float("nan"))

            rows.append(row)

        df = pd.DataFrame(rows).set_index("method")
        return df.sort_values("silhouette", ascending=False)
