"""Driver for the M4 representation-benchmark recovery-gap SHAP diagnostic.

Regenerates ``recovery_gap_shap_{V1,V2}.csv`` (the per-feature TreeSHAP block-mass table that shows how much
of raw's recovery-predictive signal is *within-factor* vs *off-map*). Runs on the reported copula **A=5**
latent block (``ARCH = arch_w0..arch_w4``); previously these CSVs had no committed writer and were stale A=4.

    PYTHONPATH=$PWD/src python notebooks/run_repbench_recovery_shap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from face.benchmark import diagnostic  # noqa: E402

OUT = ROOT / "results" / "face" / "m4_repbench"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for horizon in ("V1", "V2"):
        res = diagnostic.recovery_gap_shap(target="egf_recovery", horizon=horizon, scope="pooled")
        res["table"].to_csv(OUT / f"recovery_gap_shap_{horizon}.csv", index=False)
        print(f"=== recovery_gap_shap {horizon} (n={res['n']}, events={res['events']}) — block mass ===")
        print(res["mass"].to_string())
        print(res["raw_top"][["feature", "mean_abs_shap", "home_factor", "on_map"]].head(8).to_string(index=False))
    print(f"\nwrote recovery_gap_shap_{{V1,V2}}.csv to {OUT}")


if __name__ == "__main__":
    main()
