"""Stratified shuffle-split validation of minimum clinical-feature panels.

Regenerates ``clinical_panel_validation.json`` in place of the legacy
``biomarker_validation.json``. Two versions are written:

- ``clinical_panel_validation.json`` — **leakage-safe default** using the
  sanitised whitelist (the eight universally-measured features that seed
  the Stage A transdiagnostic similarity graph are excluded from the
  candidate pool). This is the version that should be cited in the
  article's Section 3.7.
- ``clinical_panel_validation_leaky.json`` — unsanitised legacy version
  that keeps all 49 whitelist features, retained only for audit /
  comparison (it reproduces the inflated C3 AUC ≈ 1.000 artefact
  reported in the verification pass).

The script uses stratified shuffle splits with joint (cluster, cohort)
strata (``validate_all_clinical_feature_panels_cv``) and honours the
lowered ``MIN_PANEL_POSITIVES = 10`` threshold so that small consensus
clusters (e.g. C0) are not silently dropped.

Run:

    python scripts/validate_clinical_panels_cv.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from face_stratification import PatientEmbedding, build_harmonized_dataset  # noqa: E402
from face_stratification.stage_c.clinical_panels import (  # noqa: E402
    MIN_PANEL_POSITIVES,
    default_clinical_feature_whitelist,
    validate_all_clinical_feature_panels_cv,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("clinical_panel_validation")


CSV_PATHS = {
    "bp": REPO / "data" / "BP.csv",
    "sz": REPO / "data" / "SZ.csv",
    "dr": REPO / "data" / "DR.csv",
    "asp": REPO / "data" / "ASP.csv",
}

OUT_DIR = REPO / "output" / "stratification" / "stage_c"
DEEP_DIR = OUT_DIR / "deep_analysis"
EMBED_CACHE = REPO / "output" / "stratification" / "stage_b_review" / "embedding_cache"


def main() -> None:
    t0 = time.time()
    DEEP_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Stage C consensus + Stage A harmonized data")
    emb = PatientEmbedding.load(EMBED_CACHE)
    ds = build_harmonized_dataset(csv_paths=CSV_PATHS)
    cluster_labels = (
        pd.read_parquet(OUT_DIR / "consensus_labels.parquet")["cluster"].astype(int)
    )

    idx = emb.values.index
    cluster_labels = cluster_labels.loc[idx]
    X = ds.X.loc[idx]
    cohort_labels = ds.metadata.loc[idx, "cohort"]

    logger.info(
        "n_patients=%d  n_features=%d  n_clusters=%d  min_panel_positives=%d",
        len(idx),
        X.shape[1],
        cluster_labels.nunique(),
        MIN_PANEL_POSITIVES,
    )

    for cid, count in cluster_labels.value_counts().sort_index().items():
        logger.info("  cluster %d: %d patients", int(cid), int(count))

    # ── Sanitised (leakage-safe) run — this is the one the article cites ──
    sanitised_wl = default_clinical_feature_whitelist(exclude_embedding_inputs=True)
    logger.info(
        "Running sanitised validation (%d whitelist features; "
        "embedding inputs excluded)",
        len(sanitised_wl),
    )
    sanitised = validate_all_clinical_feature_panels_cv(
        X,
        cluster_labels,
        cohort_labels,
        n_splits=5,
        test_fraction=0.2,
        max_panel_size=6,
        feature_whitelist=sanitised_wl,
        exclude_embedding_inputs=True,
        random_state=0,
    )
    sanitised_out = {str(cid): r.as_dict() for cid, r in sanitised.items()}
    (DEEP_DIR / "clinical_panel_validation.json").write_text(
        json.dumps(sanitised_out, indent=2, default=str)
    )
    logger.info(
        "  wrote clinical_panel_validation.json (%d clusters)", len(sanitised_out)
    )

    skipped = sorted(set(int(c) for c in cluster_labels.unique() if c >= 0) - set(sanitised))
    if skipped:
        logger.warning(
            "Sanitised run: clusters with no validation result = %s "
            "(likely too small, < MIN_PANEL_POSITIVES=%d)",
            skipped,
            MIN_PANEL_POSITIVES,
        )

    # ── Unsanitised (legacy, leakage-prone) run — audit copy ──────────────
    leaky_wl = default_clinical_feature_whitelist(exclude_embedding_inputs=False)
    logger.info(
        "Running LEAKY audit validation (%d whitelist features; "
        "embedding inputs INCLUDED)",
        len(leaky_wl),
    )
    leaky = validate_all_clinical_feature_panels_cv(
        X,
        cluster_labels,
        cohort_labels,
        n_splits=5,
        test_fraction=0.2,
        max_panel_size=6,
        feature_whitelist=leaky_wl,
        exclude_embedding_inputs=False,
        random_state=0,
    )
    leaky_out = {str(cid): r.as_dict() for cid, r in leaky.items()}
    (DEEP_DIR / "clinical_panel_validation_leaky.json").write_text(
        json.dumps(leaky_out, indent=2, default=str)
    )
    logger.info(
        "  wrote clinical_panel_validation_leaky.json (%d clusters)", len(leaky_out)
    )

    # Keep a legacy symlink / copy so older scripts that still reference
    # ``biomarker_validation.json`` do not break during the migration. We
    # point it at the sanitised version, because that is the honest one.
    legacy_path = DEEP_DIR / "biomarker_validation.json"
    legacy_path.write_text(json.dumps(sanitised_out, indent=2, default=str))
    logger.info(
        "  updated legacy biomarker_validation.json → sanitised payload"
    )

    # ── Side-by-side summary ───────────────────────────────────────────────
    summary_rows = []
    all_cids = sorted(set(sanitised) | set(leaky))
    for cid in all_cids:
        s = sanitised.get(cid)
        ly = leaky.get(cid)
        summary_rows.append(
            {
                "cluster_id": int(cid),
                "sanitised_test_auc_mean": s.test_auc_mean if s else None,
                "sanitised_test_auc_std": s.test_auc_std if s else None,
                "leaky_test_auc_mean": ly.test_auc_mean if ly else None,
                "leaky_test_auc_std": ly.test_auc_std if ly else None,
                "auc_inflation": (
                    (ly.test_auc_mean - s.test_auc_mean)
                    if (s and ly) else None
                ),
                "sanitised_stable_features": (
                    [
                        f
                        for f, v in s.feature_selection_stability.items()
                        if v >= 0.8
                    ]
                    if s else []
                ),
                "leaky_stable_features": (
                    [
                        f
                        for f, v in ly.feature_selection_stability.items()
                        if v >= 0.8
                    ]
                    if ly else []
                ),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        DEEP_DIR / "clinical_panel_validation_summary.csv", index=False
    )
    logger.info(
        "  wrote clinical_panel_validation_summary.csv (%d rows)", len(summary_df)
    )
    # Print the side-by-side table
    if not summary_df.empty:
        for _, row in summary_df.iterrows():
            logger.info(
                "  C%d: sanitised AUC=%.3f ± %.3f | leaky AUC=%.3f ± %.3f | "
                "inflation=%s",
                row["cluster_id"],
                row["sanitised_test_auc_mean"] or float("nan"),
                row["sanitised_test_auc_std"] or float("nan"),
                row["leaky_test_auc_mean"] or float("nan"),
                row["leaky_test_auc_std"] or float("nan"),
                (
                    f"{row['auc_inflation']:+.3f}"
                    if row["auc_inflation"] is not None
                    else "n/a"
                ),
            )

    logger.info("Validation complete in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
