#!/usr/bin/env python
"""Export immunometabolic loadings + Phi from the BMI/weight/waist-EXCLUDED sensitivity refit
(reviewer R1), and compare to the canonical fit by Tucker congruence on the shared items.

    PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_export_loadings_nobmi.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from face.measurement.engine import (  # noqa: E402
    DEFAULT_EXPLICIT_FACTORS,
    MeasurementConfig,
    MeasurementDataset,
)
from face.measurement.synthetic import export_loadings_summary, export_phi  # noqa: E402

XC = REPO / "configs" / "loading_matrix.immunometabolic_crossload.csv"
IDATA = REPO / "results/face/oop_measurement/copula/horseshoe_8d_excl-bmi-weight-wstcir/hs_s5_merged_xc/idata.nc"
F8 = ["overall_severity", "cognition", "immunometabolic", "sleep", "suicidality",
      "developmental_risk", "mania_activation", "substance"]
EXCL = ("bmi", "weight", "wstcir")


class _IData:
    def __init__(self, post): self.posterior = post


def _load(idata_path: Path) -> _IData:
    post = xr.open_dataset(idata_path, group="posterior")
    needed = ["Lam", "Phi", "sigma"] + [v for v in post.data_vars if str(v).startswith(("lh_", "lg_"))]
    sub = post[[v for v in needed if v in post.data_vars]].load()
    post.close()
    return _IData(sub)


def tucker(x: np.ndarray, y: np.ndarray) -> float:
    return float((x @ y) / np.sqrt((x @ x) * (y @ y)))


def main() -> None:
    config = MeasurementConfig().with_gaussian_copula()
    config = replace(config, prior_matrix=XC, cohort_weighted=False, exclude_items=EXCL)
    ds = MeasurementDataset(config)
    mixed = ds.mixed(F8, explicit_factors=list(DEFAULT_EXPLICIT_FACTORS),
                     min_cohorts=2, balanced=True, n_subsample=2000, seed=20260605)
    base = mixed.base
    print(f"[rebuild] N={base.M.shape[0]} J={base.M.shape[1]} (canonical mixed J=88); "
          f"excluded={EXCL}")
    assert not any(it in base.items for it in EXCL), "excluded items leaked into the data block!"

    idata = _load(IDATA)
    loadings = export_loadings_summary(idata, mixed, config, hdi_prob=0.95,
                                       specific_cross=True, cross_sd_scale=1.0)
    phi = export_phi(idata, base.factor_cols)

    imm = loadings[(loadings.factor == "immunometabolic") & (loadings.kind == "primary")].copy()
    imm = imm.reindex(imm.loading.abs().sort_values(ascending=False).index)
    out = REPO / "reports" / "copula_8factor_nobmi_loadings.csv"
    loadings.to_csv(out, index=False)
    phi.to_csv(REPO / "reports" / "copula_8factor_nobmi_phi.csv")
    print(f"\n[nobmi] immunometabolic primary loadings (n={len(imm)}, "
          f"{int(imm.excludes_zero.sum())} exclude 0):")
    for _, r in imm.iterrows():
        print(f"  {r['item']:18s} {r['loading']:+.3f} [{r['ci_low']:+.3f},{r['ci_high']:+.3f}] "
              f"{'*' if r['excludes_zero'] else ' '}")

    # --- Tucker congruence vs canonical on the shared (non-anthropometric) items ---
    canon = pd.read_csv(REPO / "reports" / "copula_8factor_loadings.csv")
    canon = canon[(canon.factor == "immunometabolic") & (canon.kind == "primary")][["item", "loading"]]
    merged = canon.merge(imm[["item", "loading"]], on="item", suffixes=("_canon", "_nobmi"))
    phi_t = tucker(merged.loading_canon.to_numpy(), merged.loading_nobmi.to_numpy())
    r_pear = float(np.corrcoef(merged.loading_canon, merged.loading_nobmi)[0, 1])
    print(f"\n[congruence] shared non-anthropometric items: {len(merged)}")
    print(f"[congruence] Tucker phi (canonical vs no-BMI immunometabolic column) = {phi_t:.3f}")
    print(f"[congruence] Pearson r of loadings = {r_pear:.3f}")

    # --- Phi: immunometabolic's correlations with the other specifics (coherence/distinctness) ---
    print("\n[phi] immunometabolic row of the correlated-factor block (no-BMI fit):")
    if "immunometabolic" in phi.index:
        row = phi.loc["immunometabolic"].drop("immunometabolic", errors="ignore")
        for k, v in row.items():
            print(f"  immunometabolic <-> {k:20s} {v:+.3f}")


if __name__ == "__main__":
    main()
