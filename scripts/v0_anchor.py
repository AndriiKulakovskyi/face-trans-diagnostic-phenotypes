"""Build and persist the V0-anchored feature/patient selection.

This script is the Phase-1 deliverable referenced in ROADMAP.md.

  1. Loads the unified long-format frame (READY + PARTIAL).
  2. Calls `select_v0_anchor(variable_threshold, patient_threshold)` to
     identify the V0 feature set and V0 patient roster.
  3. Persists three artefacts to `results/`:
        - v0_anchor_features.csv   variable × completeness × kept
        - v0_anchor_patients.csv   usubjid_patients × completeness × kept × cohort × arm
        - v0_anchor_meta.json      thresholds, counts, timestamp, git rev
  4. Prints a summary including per-cohort patient counts and an example
     V1 projection (sanity check that `anchor.apply()` works).

The default thresholds (75% / 75%) are PLACEHOLDERS — see ROADMAP.md Q2.1.
Override on the command line:

    python3 scripts/v0_anchor.py --var-threshold 0.85 --pt-threshold 0.80

The CSV artefacts are the canonical inputs for the Phase-3 clustering
scripts (cluster_v0.py).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "archive"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from face_common import (  # noqa: E402
    build_unified_dataframe,
    select_v0_anchor,
)

DATA_DIR = REPO_ROOT / "data"
DICT_PATH = REPO_ROOT / "face-common-vars.xlsx"
RESULTS_DIR = REPO_ROOT / "results"


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
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--var-threshold", type=float, default=0.75,
                        help="completeness floor for variables at V0 (default 0.75)")
    parser.add_argument("--pt-threshold", type=float, default=0.75,
                        help="completeness floor for patients at V0 (default 0.75)")
    parser.add_argument("--readiness", nargs="+",
                        default=["READY", "PARTIAL"],
                        help="cluster_readiness prefixes (default: READY PARTIAL)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"Loading unified frame (readiness={args.readiness}, format='long')...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(
            DATA_DIR, DICT_PATH,
            readiness=args.readiness, format="long",
        )
    print(f"  rows: {len(df):,}   columns: {df.shape[1]}")
    print(f"  V0 rows per cohort: "
          f"{df[df['visit'] == 'V0'].groupby('cohort').size().to_dict()}")

    print(f"\nSelecting V0 anchor with var_threshold={args.var_threshold}, "
          f"pt_threshold={args.pt_threshold}...")
    v0_filtered, anchor = select_v0_anchor(
        df,
        variable_threshold=args.var_threshold,
        patient_threshold=args.pt_threshold,
    )
    print(f"  {anchor}")

    # ----- persist artefacts ------------------------------------------------
    RESULTS_DIR.mkdir(exist_ok=True)
    features_path = RESULTS_DIR / "v0_anchor_features.csv"
    patients_path = RESULTS_DIR / "v0_anchor_patients.csv"
    meta_path = RESULTS_DIR / "v0_anchor_meta.json"

    anchor.variable_report.table.to_csv(features_path, index=False)
    # enrich the patient report with cohort+arm for downstream convenience.
    # Merge on patient_uid (globally unique) — NOT usubjid_patients, which
    # collides across cohorts.
    pt_table = anchor.patient_report.table.copy()
    cohort_arm = (df[df["visit"] == "V0"]
                  [["patient_uid", "cohort", "arm"]]
                  .drop_duplicates("patient_uid"))
    pt_table = pt_table.merge(cohort_arm, on="patient_uid", how="left")
    pt_table.to_csv(patients_path, index=False)

    kept_cohort_counts = (v0_filtered.groupby("cohort")["usubjid_patients"]
                          .nunique().to_dict())
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_rev": _git_rev(),
        "readiness": args.readiness,
        "variable_threshold": args.var_threshold,
        "patient_threshold": args.pt_threshold,
        "n_features_input": int(len(anchor.variable_report.table)),
        "n_features_kept": int(anchor.n_features),
        "n_patients_input_v0": int(len(anchor.patient_report.table)),
        "n_patients_kept_v0": int(anchor.n_patients),
        "kept_patients_per_cohort": {k: int(v) for k, v in kept_cohort_counts.items()},
        "feature_columns": list(anchor.feature_columns),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"\nWrote:")
    print(f"  {features_path}  ({len(anchor.variable_report.table):,} variables)")
    print(f"  {patients_path}  ({len(pt_table):,} V0 patients)")
    print(f"  {meta_path}")

    # ----- sanity-check apply() to V1..V4 -----------------------------------
    print("\nV1..V4 projection (patients × visits via anchor.apply):")
    n_ids = sum(c in df.columns for c in
                ("patient_uid", "usubjid_patients", "cohort", "arm",
                 "visitnum", "visit"))
    for visit in ("V1", "V2", "V3", "V4"):
        v_df = anchor.apply(df, restrict_visits=[visit])
        print(f"  {visit}: {len(v_df):>5} rows, "
              f"{v_df['patient_uid'].nunique():>5} patients, "
              f"{v_df.shape[1] - n_ids:>4} features")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
