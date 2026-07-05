#!/usr/bin/env python3
"""Numeric fingerprint of the confidential per-patient arrays — the reproduction target.

The 22 GB of posteriors cannot be committed and are too large to diff. This recipe reduces
each per-patient array to a tiny, rank-based, tolerance-banded signature that is (a) safe to
commit (no patient rows, only moments / correlations / hashes), (b) invariant to benign row
reordering (rows are canonically sorted first), and (c) sensitive to any scientifically
meaningful drift (moments, cross-factor correlation, and per-patient rank hashes all move).

Recompute from ANY results root and diff field-by-field against the frozen oracle:

    python reference/oracle/fingerprint.py --results results --out /tmp/new_fingerprint.json
    # then compare /tmp/new_fingerprint.json to reference/oracle/FINGERPRINT.json

Determinism: rows sorted by (cohort, patient_id[, visit]); values rounded before hashing so
tiny numeric jitter does not false-alarm; the coordinate check downstream is "fraction of
patients within the frozen band", not a bitwise hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

FACTORS = [
    "overall_severity", "cognition", "immunometabolic", "sleep",
    "suicidality", "developmental_risk", "mania_activation", "substance",
]
ARCH_W = [f"arch_w{i}" for i in range(5)]
PCTS = [1, 5, 25, 50, 75, 95, 99]
KEYCOLS = ["cohort", "patient_id"]


def _canonical(df: pd.DataFrame) -> pd.DataFrame:
    sort = [c for c in ["cohort", "patient_id", "visit"] if c in df.columns]
    return df.sort_values(sort).reset_index(drop=True) if sort else df.reset_index(drop=True)


def _hash_floats(a: np.ndarray, dp: int = 3) -> str:
    q = np.round(np.asarray(a, dtype="float64"), dp)
    q[np.isnan(q)] = 0.0
    return hashlib.sha256(q.tobytes()).hexdigest()[:16]


def _hash_ints(a: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(a, dtype="int64").tobytes()).hexdigest()[:16]


def _col_moments(s: pd.Series) -> dict:
    x = s.to_numpy(dtype="float64")
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {"n": 0}
    out = {"n": int(x.size), "mean": round(float(x.mean()), 6),
           "sd": round(float(x.std(ddof=1)) if x.size > 1 else 0.0, 6),
           "skew": round(float(pd.Series(x).skew()) if x.size > 2 else 0.0, 6)}
    out["pct"] = {str(p): round(float(np.percentile(x, p)), 6) for p in PCTS}
    return out


def _coord_fingerprint(df: pd.DataFrame, label: str) -> dict:
    df = _canonical(df)
    present = [f for f in FACTORS if f"{f}__mean" in df.columns]
    fp: dict = {"label": label, "shape": list(df.shape), "factors": present}
    fp["moments"] = {f: _col_moments(df[f"{f}__mean"]) for f in present}
    # cross-factor correlation of the posterior-mean coordinates
    M = df[[f"{f}__mean" for f in present]].to_numpy(dtype="float64")
    ok = ~np.isnan(M).any(axis=1)
    corr = np.corrcoef(M[ok].T) if ok.sum() > 2 else np.full((len(present), len(present)), np.nan)
    fp["corr"] = np.round(corr, 4).tolist()
    # per-factor per-patient rank hash (robust to tiny jitter, catches reorder/mislabel)
    fp["rank_hash"] = {f: _hash_ints(pd.Series(df[f"{f}__mean"]).rank(method="average").fillna(0).round().to_numpy())
                       for f in present}
    # per-factor posterior-mean float hash (dp=3) for exact-ish provenance
    fp["mean_hash"] = {f: _hash_floats(df[f"{f}__mean"].to_numpy()) for f in present}
    return fp


def _archetype_fingerprint(df: pd.DataFrame, label: str) -> dict:
    df = _canonical(df)
    fp: dict = {"label": label, "shape": list(df.shape)}
    wcols = [c for c in ARCH_W if c in df.columns]
    fp["weight_moments"] = {c: _col_moments(df[c]) for c in wcols}
    if "arch_dominant" in df.columns:
        dom = df["arch_dominant"].to_numpy()
        vals, counts = np.unique(dom[~pd.isna(dom)], return_counts=True)
        fp["dominant_sizes"] = sorted(int(c) for c in counts)  # label-permutation invariant
        try:
            fp["dominant_hash"] = _hash_ints(pd.Series(dom).fillna(-1).astype("int64").to_numpy())
        except Exception:
            fp["dominant_hash"] = _hash_ints(pd.factorize(pd.Series(dom).fillna("NA"))[0])
    return fp


def build(results_root: Path) -> dict:
    R = results_root
    targets = {
        "m1m2_coordinates": (R / "m2_strata/coordinates/coordinates_full.parquet", _coord_fingerprint),
        "m2_patient_strata": (R / "m2_strata/consolidate/patient_strata.parquet", _archetype_fingerprint),
        "m3_patient_panel": (R / "m3_temporal/consolidate/patient_panel.parquet", _coord_fingerprint),
        "m4_patient_risk": (R / "m4_prognosis/consolidate/prognosis_patient_risk.parquet", _archetype_fingerprint),
    }
    out: dict = {"results_root": str(R), "arrays": {}}
    for key, (path, fn) in targets.items():
        if path.exists():
            out["arrays"][key] = fn(pd.read_parquet(path), key)
        else:
            out["arrays"][key] = {"label": key, "MISSING": str(path)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", type=Path)
    ap.add_argument("--out", default="reference/oracle/FINGERPRINT.json", type=Path)
    args = ap.parse_args()
    fp = build(args.results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fp, indent=2, sort_keys=True))
    n = sum(1 for v in fp["arrays"].values() if "MISSING" not in v)
    print(f"fingerprint: {n}/{len(fp['arrays'])} arrays -> {args.out}")
