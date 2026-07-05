#!/usr/bin/env python3
"""01 — build the model-ready V0 baseline and persist it (no imputation).

    python3 scripts/01_build_data.py

Loads the harmonized FACE V0 baseline (FULL sample, all 3 cohorts — no completeness
selection, methods §3.6), applies deterministic skip-logic structural-zero decoding, and
restricts to the modeled indicators declared in configs/loading_matrix.csv. Persists:

  data/processed/baseline_v0.parquet         one row per patient × modeled indicator
                                             (raw harmonized values, NaN = missing, NEVER imputed)
  data/processed/indicator_metadata.parquet  per indicator: home factor, likelihood family,
                                             modeling block, burden sign
  reports/01_build_data.md                   aggregate QC summary (no per-patient data)
  reports/01_coverage_by_indicator.csv       per-indicator observed coverage (aggregate)

Per-patient parquet is confidential -> gitignored. The aggregate reports are shareable.
This is the input the staged-fit engine (scripts/04) consumes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np  # noqa: F401  (kept for parity / future use)
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from face.data import build_unified_dataframe, load_variables, to_harmonized_dataset  # noqa: E402

XLSX = REPO / "data" / "face-common-vars.xlsx"
MATRIX = REPO / "configs" / "loading_matrix.csv"
PROC = REPO / "data" / "processed"
REPORTS = REPO / "reports"


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # ---- modeled-indicator metadata from the prior matrix ----
    m = pd.read_csv(MATRIX)
    home = (m[m.prior_type.isin(["primary", "g_anchor"])].drop_duplicates("item")
            .set_index("item")["factor"])
    meta = (m.drop_duplicates("item").set_index("item")
            [["likelihood_family", "modeling_block", "item_sign"]].copy())
    meta["home_factor"] = home                      # NaN for cross-loading windows (no home)
    modeled = list(meta.index)

    # ---- harmonized full-sample V0 (skip-logic decoded; NaN = missing, never imputed) ----
    variables = load_variables(str(XLSX))
    df = build_unified_dataframe("data", str(XLSX), readiness=["READY", "PARTIAL"], format="long")
    ds = to_harmonized_dataset(df, variables, visit="V0", normalize=False, apply_skip_logic=True)
    X = ds.X

    present = [it for it in modeled if it in X.columns]
    absent = [it for it in modeled if it not in X.columns]
    B = X[present].apply(pd.to_numeric, errors="coerce")        # raw harmonized; NaN = missing
    cohort = pd.Series(X.index.get_level_values("cohort"), index=X.index)

    # ---- site (administrative; NEVER modeled) — persisted as a side table so the §8 site
    # bootstrap (scripts/08_robustness) can cluster-resample recruitment sites. siteid_city is the
    # canonical fondacode-derived network code (src/face/data/rules.py::derive_siteid_city); it is
    # computed by harmonization but excluded from the indicator matrix (not in the prior matrix). ----
    site = pd.to_numeric(X["siteid_city"], errors="coerce") if "siteid_city" in X.columns else None

    # ---- persist model-ready tables (per-patient -> gitignored) ----
    B.to_parquet(PROC / "baseline_v0.parquet")
    meta.loc[present].reset_index(names="item").to_parquet(PROC / "indicator_metadata.parquet",
                                                           index=False)
    if site is not None:
        site.rename("siteid_city").to_frame().to_parquet(PROC / "site_v0.parquet")  # gitignored
        site_cov = pd.crosstab(site.round().astype("Int64"), cohort.values)
        site_cov.columns = [f"n_{c}" for c in site_cov.columns]
        site_cov["n_total"] = site_cov.sum(axis=1)
        site_cov.index.name = "site_code"
        site_cov.to_csv(REPORTS / "01_site_coverage.csv")

    # ---- aggregate QC (committable: counts/fractions only, no per-patient values) ----
    cov = pd.DataFrame({"indicator": present,
                        "home_factor": [meta.loc[it, "home_factor"] for it in present],
                        "block": [meta.loc[it, "modeling_block"] for it in present],
                        "family": [meta.loc[it, "likelihood_family"] for it in present]})
    cov["n_obs"] = [int(B[it].notna().sum()) for it in present]
    cov["frac_obs"] = (cov["n_obs"] / len(B)).round(3)
    for c in ("bp", "sz", "dr"):
        mask = (cohort.values == c)
        cov[f"obs_{c}"] = [round(float(B.loc[mask, it].notna().mean()), 2) for it in present]
    cov.sort_values(["home_factor", "indicator"]).to_csv(
        REPORTS / "01_coverage_by_indicator.csv", index=False)

    ncoh = cohort.value_counts().to_dict()
    lines = [
        "# 01 — model-ready V0 baseline (build + persistence)", "",
        f"- **N = {len(B):,} patients** (full V0, no completeness selection): "
        + " · ".join(f"{k.upper()} {v:,}" for k, v in ncoh.items()), "",
        f"- **Modeled indicators: {len(present)}** "
        f"(continuous {int((cov.block == 'continuous').sum())} · "
        f"explicit {int((cov.block == 'explicit').sum())}).",
        f"- Mean cell missingness across modeled indicators: "
        f"**{float(B.isna().mean().mean()):.1%}** — NaN preserved, never imputed.",
        "- Skip-logic structural-zero decoding applied (`apply_skip_logic=True`).",
        f"- Best/worst 3-cohort coverage: "
        f"max {cov.frac_obs.max():.2f} ({cov.loc[cov.frac_obs.idxmax(), 'indicator']}), "
        f"min {cov.frac_obs.min():.2f} ({cov.loc[cov.frac_obs.idxmin(), 'indicator']}).",
        f"- **{int((cov.n_obs < 30).sum())} indicator(s) with < 30 obs** "
        f"({', '.join(cov[cov.n_obs < 30].indicator) or 'none'}) — below the engine's "
        f"min-observation guard, auto-skipped at fit; effective modeled set "
        f"**{int((cov.n_obs >= 30).sum())}**.",
    ]
    if site is not None:
        per = {c: int(site[(cohort == c).values].dropna().nunique()) for c in ("bp", "sz", "dr")}
        lines.append(
            f"- **Recruitment sites: {int(site.dropna().nunique())}** (administrative — persisted to "
            "`data/processed/site_v0.parquet` for the §8 site bootstrap; NOT modeled): "
            + " · ".join(f"{k.upper()} {v}" for k, v in per.items()) + " distinct sites per cohort.")
    else:
        lines.append("- ⚠ `siteid_city` absent from the harmonized matrix — site side table not "
                     "written (the §8 site bootstrap will need an alternate site source).")
    if absent:
        lines.append(f"- ⚠ declared in matrix but absent from data ({len(absent)}): {absent}")
    lines += ["", "Artifacts: `data/processed/{baseline_v0,indicator_metadata}.parquet`"
              + (" + `site_v0.parquet`" if site is not None else "")
              + " (gitignored) · `reports/01_coverage_by_indicator.csv`"
              + (" · `01_site_coverage.csv`" if site is not None else "") + "."]
    (REPORTS / "01_build_data.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote data/processed/baseline_v0.parquet  shape={B.shape}")

    # ---- follow-up visits V1/V2 (same pipeline; modeled indicators, NaN = missing) — M3/M4 inputs ----
    # Identical harmonization to V0 (skip-logic on, no normalization), parameterized by visit; the
    # per-visit roster is the completers at that visit. Downstream engines align columns on read.
    for v in ("V1", "V2"):
        dsv = to_harmonized_dataset(df, variables, visit=v, normalize=False, apply_skip_logic=True)
        presv = [it for it in modeled if it in dsv.X.columns]
        Bv = dsv.X[presv].apply(pd.to_numeric, errors="coerce")
        Bv.to_parquet(PROC / f"baseline_{v.lower()}.parquet")
        print(f"wrote data/processed/baseline_{v.lower()}.parquet  shape={Bv.shape}")


if __name__ == "__main__":
    main()
