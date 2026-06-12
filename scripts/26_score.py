#!/usr/bin/env python3
"""26 — M2.5 consolidation: the unified per-patient strata hand-off (§8).

Merges the two soft views into one object for M3/M4: archetype simplex weights (lead) + tessellation
responsibilities, each with uncertainty and a dominant label. Diagnosis is carried for validation only.
This is the M2 deliverable the later milestones score against.

    python3 scripts/26_score.py
Reads results/face/m2/{archetypes.parquet, tessellation.parquet, coordinates_full.parquet,
validation_table.parquet}. Writes results/face/patient_strata.parquet (gitignored).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
M2 = REPO / "results" / "face" / "m2"
KEYS = ["cohort", "patient_id"]


def main():
    arch = pd.read_parquet(M2 / "archetypes.parquet")
    tess = pd.read_parquet(M2 / "tessellation.parquet")
    vt = pd.read_parquet(M2 / "validation_table.parquet")[KEYS + ["arm"]]

    A = sum(c.endswith("_mean") and c.startswith("w") for c in arch.columns)
    K = sum(c.startswith("r") and c[1:].isdigit() for c in tess.columns)
    arch = arch.rename(columns={**{f"w{a}_mean": f"arch_w{a}" for a in range(A)},
                                **{f"w{a}_sd": f"arch_w{a}_sd" for a in range(A)},
                                "dominant": "arch_dominant", "dominant_name": "arch_dominant_name",
                                "entropy": "arch_entropy"})
    tess = tess.rename(columns={**{f"r{k}": f"tess_r{k}" for k in range(K)},
                                "MAP": "tess_MAP", "MAP_name": "tess_MAP_name", "entropy": "tess_entropy"})

    df = arch.merge(tess, on=KEYS, validate="1:1").merge(vt, on=KEYS, validate="1:1")
    out = REPO / "results" / "face" / "patient_strata.parquet"
    df.to_parquet(out)

    print(f"wrote {out.relative_to(REPO)}  shape={df.shape}")
    print(f"  archetypes A={A} · tessellation K={K} · N={len(df):,}")
    print(f"  columns: {list(df.columns)}")
    print("\n  archetype dominant shares:")
    print(df["arch_dominant_name"].value_counts(normalize=True).round(3).to_string())
    print("\n  tessellation MAP shares:")
    print(df["tess_MAP_name"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
