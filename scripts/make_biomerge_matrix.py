#!/usr/bin/env python
"""Derive the 'biology-merged' prior matrix from v3: collapse the metabolic + inflammatory factors
into a single 'immunometabolic' factor (FACE soft-prior candidate #5, "Metabolism / Immunometabolism").

This is the one-factor arm of the one-vs-two model comparison for the biology block: the prior ontology
(and the official FACE dimension-readiness workbook) treat metabolism/immunometabolism as a SINGLE
construct; v3 split it into two. We test which the data prefer (WAIC/LOO) by fitting both.

    python scripts/make_biomerge_matrix.py
    -> configs/loading_matrix.immunometabolic.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "configs" / "loading_matrix.csv"
OUT = REPO / "configs" / "loading_matrix.immunometabolic.csv"
MERGE = ["metabolic", "inflammatory"]
NEW = "immunometabolic"
# permissiveness order for collapsing a non-home item's two cross rows into one
PRIORITY = {"primary": 4, "g_anchor": 4, "g_anchor_on_specific": 3, "plausible_cross": 2, "unlikely_cross": 1}


def main() -> None:
    pm = pd.read_csv(SRC)
    home = (pm[pm.prior_type.isin(["primary", "g_anchor"])]
            .drop_duplicates("item").set_index("item").factor.to_dict())

    keep = pm[~pm.factor.isin(MERGE)].copy()          # all non-merged factor rows, untouched
    merged_rows = []
    for item, g in pm[pm.factor.isin(MERGE)].groupby("item", sort=False):
        if home.get(item) in MERGE:
            # item's home is metabolic or inflammatory -> it now anchors the merged factor
            src = g[g.prior_type.isin(["primary", "g_anchor"])].iloc[0]
            row = src.copy()
            row["factor"] = NEW
            row["rationale"] = "home factor immunometabolic (merged metabolic+inflammatory)"
        else:
            # non-biology item: collapse its two cross rows into the most-permissive single one
            src = g.sort_values("prior_type", key=lambda s: s.map(PRIORITY), ascending=False).iloc[0]
            row = src.copy()
            row["factor"] = NEW
            row["rationale"] = "merged metabolic+inflammatory -> immunometabolic"
        merged_rows.append(row)

    out = pd.concat([keep, pd.DataFrame(merged_rows)], ignore_index=True)
    out = out.sort_values(["item", "factor"]).reset_index(drop=True)
    out.to_csv(OUT, index=False)

    # report
    hm = (out[out.prior_type.isin(["primary", "g_anchor"])]
          .drop_duplicates("item").set_index("item").factor.value_counts())
    print(f"[biomerge] wrote {OUT.relative_to(REPO)}  ({len(out)} rows, {out.item.nunique()} items)")
    print(f"[biomerge] factors: {sorted(out.factor.unique())}")
    print(f"[biomerge] immunometabolic home items: {int(hm.get(NEW, 0))} "
          f"(v3 had metabolic {pd.Series(home).eq('metabolic').sum()} + "
          f"inflammatory {pd.Series(home).eq('inflammatory').sum()})")


if __name__ == "__main__":
    main()
