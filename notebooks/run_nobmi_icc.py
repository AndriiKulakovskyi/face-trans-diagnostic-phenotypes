#!/usr/bin/env python
"""No-anthropometry durability check (reviewer R1): score the immunometabolic axis at V0/V1/V2 under the
FIXED BMI/weight/waist-EXCLUDED M1 fit and compute its trait/state ICC. If the ICC stays high, the 0.91
durability of the canonical axis is not merely BMI's test-retest stability.

    PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_nobmi_icc.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from face.temporal.engine import (  # noqa: E402
    F8_FIT,
    CopulaPanelScorer,
    TemporalConfig,
    TemporalData,
)
from face.temporal.variance import decompose, patient_patterns, raw_icc  # noqa: E402

XC = REPO / "configs" / "loading_matrix.immunometabolic_crossload.csv"
EXCL_MAP = REPO / "results/m1_measurement/copula/horseshoe_8d_excl-bmi-weight-wstcir"
EXCL = ("bmi", "weight", "wstcir")


class NoBMITemporalData(TemporalData):
    """TemporalData pointed at the BMI-excluded fold fit: exclude_items set, substance NOT pinned, balanced
    fit (cohort_weighted=False), folded matrix. Item/factor structure aligns to the excluded idata's Lam."""

    def copula_mixed(self):
        if self._mp is not None:
            return self._mp, self._idata
        import arviz as az

        from face.measurement.engine import (
            DEFAULT_EXPLICIT_FACTORS,
            MeasurementConfig,
            MeasurementDataset,
        )
        mcfg = MeasurementConfig(likelihood_mode="gaussian_copula", cohort_weighted=False,
                                 prior_matrix=XC, exclude_items=EXCL, output_dir=self.config.map_dir)
        dataset = MeasurementDataset(mcfg)
        self._dataset = dataset
        self._mp = dataset.mixed(F8_FIT, explicit_factors=DEFAULT_EXPLICIT_FACTORS, min_cohorts=2,
                                 balanced=False, n_subsample=None)
        self._idata = az.from_netcdf(str(self.config.map_dir / "hs_s5_merged_xc" / "idata.nc"))
        return self._mp, self._idata


def main() -> None:
    cfg = replace(TemporalConfig(), map_dir=EXCL_MAP, proj_draws=600, proj_tune=600, proj_chains=2)
    data = NoBMITemporalData(cfg)
    scorer = CopulaPanelScorer(cfg)
    scorer.data = data  # inject the excluded-fit data object

    rows = []
    for visit in ("V0", "V1", "V2"):
        c = scorer.score_continuous(visit)
        ci = c["cidx"]["immunometabolic"]
        idx = c["index"]
        uid = [f"{co}|{pa}" for co, pa in zip(idx.get_level_values("cohort"),
                                              idx.get_level_values("patient_id"), strict=False)]
        df = pd.DataFrame({"patient_uid": uid, "visit": visit,
                           "immunometabolic__mean": c["mean"][:, ci],
                           "immunometabolic__sd": c["sd"][:, ci]})
        df = df[np.isfinite(df["immunometabolic__mean"])]
        print(f"[score] {visit}: N={len(df)}  mean(sd)={df['immunometabolic__sd'].mean():.3f}")
        rows.append(df)
    panel = pd.concat(rows, ignore_index=True)
    nv = panel.groupby("patient_uid")["visit"].transform("nunique")
    panel["n_visits"] = nv

    patterns = patient_patterns(panel)
    ts = decompose(panel, ["immunometabolic"], patterns, draws=cfg.proj_draws,
                   tune=cfg.proj_tune, chains=cfg.proj_chains, seed=cfg.seed)
    naive = raw_icc(panel, ["immunometabolic"])
    out = REPO / "results/m1_measurement/copula/horseshoe_8d_excl-bmi-weight-wstcir" / "nobmi_immunometabolic_icc.csv"
    ts.to_csv(out, index=False)
    print("\n=== no-anthropometry immunometabolic trait/state ===")
    print(ts.to_string(index=False))
    print(f"\nraw (uncorrected) ICC = {naive['immunometabolic']}")
    print("\n[compare] canonical full-axis immunometabolic ICC = 0.91 (reports trait_state).")
    print(f"[written] {out}")


if __name__ == "__main__":
    main()
