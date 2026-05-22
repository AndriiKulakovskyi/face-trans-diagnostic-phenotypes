"""Phase 2 threshold sweep + feature-quality analysis.

For every (variable_threshold, patient_threshold) cell in a configured grid:
  - run select_v0_anchor()
  - record n_features, per-cohort patient counts, V1..V4 carry, per-section
    feature retention, feature variance/entropy, near-constant flag
  - record per-cohort missingness profile of the surviving feature matrix
    (raw NaN cells after the filter — feeds Phase-2 imputation decision)

All numbers are written to `results/phase2_sweep.csv` and
`results/phase2_features.csv` (per-feature variance/entropy at the user's
chosen primary threshold).

The companion HTML report is built by `scripts/phase2_report.py` from these
CSVs (kept separate so re-rendering the report doesn't re-sweep).

Run:  python3 scripts/phase2_sweep.py [--primary-var 0.75 --primary-pt 0.75]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from face_common import (  # noqa: E402
    IDENTIFIER_COLUMNS,
    build_unified_dataframe,
    load_variables,
    select_v0_anchor,
)


DATA_DIR = REPO_ROOT / "data"
DICT_PATH = REPO_ROOT / "face-common-vars.xlsx"
RESULTS_DIR = REPO_ROOT / "results"


# Grid we explore. Includes intentionally permissive (0.5) through strict (0.9).
THRESHOLD_GRID: list[float] = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]


def feature_quality(series: pd.Series) -> dict:
    """For a single column, compute marginal-distribution quality metrics.

    Returns dict with `metric` in {std, entropy_bits}, `value`, `modal_share`
    (fraction of non-null observations equal to the modal value),
    `near_constant` (modal_share > 0.95).
    """
    non_null = series.dropna()
    n = len(non_null)
    if n == 0:
        return {"metric": "empty", "value": 0.0,
                "modal_share": 1.0, "near_constant": True,
                "n_unique": 0, "n_non_null": 0}
    n_unique = int(non_null.nunique())
    counts = non_null.value_counts(normalize=True)
    modal_share = float(counts.iloc[0]) if not counts.empty else 1.0
    if pd.api.types.is_numeric_dtype(non_null) and n_unique > 10:
        return {"metric": "std", "value": float(non_null.std(ddof=0)),
                "modal_share": modal_share,
                "near_constant": False,
                "n_unique": n_unique, "n_non_null": n}
    # categorical / binary / ordinal
    p = counts.values
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())
    return {"metric": "entropy_bits", "value": entropy,
            "modal_share": modal_share,
            "near_constant": modal_share > 0.95,
            "n_unique": n_unique, "n_non_null": n}


def per_section_breakdown(
    kept_features: Iterable[str], variable_lookup: dict[str, "Variable"],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for f in kept_features:
        sec = variable_lookup[f].section if f in variable_lookup else "—"
        counts[sec] = counts.get(sec, 0) + 1
    return counts


def per_cohort_missingness(df_v0: pd.DataFrame, features: list[str]) -> dict:
    out: dict = {}
    for cohort in ("BP", "SZ", "DR"):
        sub = df_v0[df_v0["cohort"] == cohort][features]
        n_cells = sub.size
        n_nan = int(sub.isna().sum().sum())
        out[cohort] = {
            "n_cells": int(n_cells),
            "n_nan": n_nan,
            "nan_rate": (n_nan / n_cells) if n_cells else float("nan"),
        }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)
    p.add_argument("--primary-var", type=float, default=0.75,
                   help="primary variable threshold used for per-feature dump")
    p.add_argument("--primary-pt", type=float, default=0.75,
                   help="primary patient threshold used for per-feature dump")
    p.add_argument("--grid", type=float, nargs="+", default=THRESHOLD_GRID,
                   help="threshold grid (same for both axes)")
    p.add_argument("--readiness", nargs="+", default=["READY", "PARTIAL"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Loading unified frame (readiness={args.readiness})...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(
            DATA_DIR, DICT_PATH,
            readiness=args.readiness, format="long",
        )
    variables = load_variables(DICT_PATH)
    var_lookup = {v.canonical_name: v for v in variables}
    v0 = df[df["visit"] == "V0"]
    print(f"  rows {len(df):,} · V0 rows {len(v0):,} · "
          f"unique V0 patients {v0['usubjid_patients'].nunique():,}")

    sweep_rows = []
    section_rows = []
    print(f"\nSweeping {len(args.grid)}×{len(args.grid)} threshold grid...")
    for var_thr in args.grid:
        for pt_thr in args.grid:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                v0_filtered, anchor = select_v0_anchor(
                    df, variable_threshold=var_thr,
                    patient_threshold=pt_thr,
                )
            kept_per_cohort = (v0_filtered.groupby("cohort")["usubjid_patients"]
                               .nunique().to_dict())
            row = {
                "var_threshold": var_thr,
                "pt_threshold": pt_thr,
                "n_features": anchor.n_features,
                "n_patients": anchor.n_patients,
                "bp_patients": int(kept_per_cohort.get("BP", 0)),
                "sz_patients": int(kept_per_cohort.get("SZ", 0)),
                "dr_patients": int(kept_per_cohort.get("DR", 0)),
            }
            for visit in ("V1", "V2", "V3", "V4"):
                proj = anchor.apply(df, restrict_visits=[visit])
                row[f"{visit}_n"] = int(proj["usubjid_patients"].nunique())
                per_co = proj.groupby("cohort")["usubjid_patients"].nunique()
                for c in ("BP", "SZ", "DR"):
                    row[f"{visit}_{c}"] = int(per_co.get(c, 0))
            sweep_rows.append(row)
            sect = per_section_breakdown(anchor.feature_columns, var_lookup)
            for s, n in sect.items():
                section_rows.append({
                    "var_threshold": var_thr,
                    "pt_threshold": pt_thr,
                    "section": s,
                    "n_features": n,
                })
            print(f"  var={var_thr:.2f} pt={pt_thr:.2f} "
                  f"→ {anchor.n_features:>3} feats, "
                  f"{anchor.n_patients:>5} pts "
                  f"(BP={row['bp_patients']:>4} SZ={row['sz_patients']:>4} "
                  f"DR={row['dr_patients']:>3})")

    sweep_df = pd.DataFrame(sweep_rows)
    section_df = pd.DataFrame(section_rows)
    sweep_path = RESULTS_DIR / "phase2_sweep.csv"
    section_path = RESULTS_DIR / "phase2_sweep_sections.csv"
    sweep_df.to_csv(sweep_path, index=False)
    section_df.to_csv(section_path, index=False)
    print(f"\nWrote {sweep_path}")
    print(f"Wrote {section_path}")

    # ----- per-feature analysis at the primary thresholds -------------------
    print(f"\nFeature-quality dump at primary "
          f"({args.primary_var:.2f}/{args.primary_pt:.2f})...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v0_primary, anchor_primary = select_v0_anchor(
            df, variable_threshold=args.primary_var,
            patient_threshold=args.primary_pt,
        )
    feature_rows = []
    for f in anchor_primary.feature_columns:
        q = feature_quality(v0_primary[f])
        var = var_lookup.get(f)
        feature_rows.append({
            "canonical_name": f,
            "section": var.section if var else "—",
            "dtype": var.dtype if var else "—",
            "readiness": var.cluster_readiness.split(" ")[0] if var else "—",
            **q,
        })
    features_df = pd.DataFrame(feature_rows)
    features_df = features_df.sort_values(["section", "canonical_name"])
    features_path = RESULTS_DIR / "phase2_features.csv"
    features_df.to_csv(features_path, index=False)
    print(f"Wrote {features_path}")

    # ----- per-cohort missingness in the primary feature matrix -------------
    miss = per_cohort_missingness(v0_primary,
                                  list(anchor_primary.feature_columns))
    meta = {
        "primary_var_threshold": args.primary_var,
        "primary_pt_threshold": args.primary_pt,
        "primary_n_features": anchor_primary.n_features,
        "primary_n_patients": anchor_primary.n_patients,
        "grid": list(args.grid),
        "readiness": args.readiness,
        "primary_cohort_missingness": miss,
        "near_constant_features": (
            features_df.loc[features_df["near_constant"], "canonical_name"]
            .tolist()
        ),
    }
    meta_path = RESULTS_DIR / "phase2_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Wrote {meta_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
