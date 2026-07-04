"""Driver for the M4 representation benchmark (Phase 1).

Fits the six arms (REF · REF+RAW · REF+LAT-{mu,sigma,A} · REF+RAW+LAT) on the continuous-GAF backbone for
recovery & deterioration, pooled and BP+DR, and writes the scalar / net-benefit / sufficiency tables.

    PYTHONPATH=$PWD/src python notebooks/run_representation_benchmark.py            # full (detached)
    PYTHONPATH=$PWD/src python notebooks/run_representation_benchmark.py --smoke    # fast sanity
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from face.benchmark import harness  # noqa: E402

OUT = ROOT / "results" / "face" / "m4_repbench"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="fast subsampled run (structure check)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        res = harness.run_all(n_boot=200, n_splits=3, n_repeats=1, subsample=1500)
        tag = "smoke_"
    else:
        res = harness.run_all(n_boot=2000, n_splits=5, n_repeats=2)
        tag = ""

    for name, df in res.items():
        df.to_csv(OUT / f"{tag}{name}.csv", index=False)

    print("=== sufficiency (full vs latent, by contrast) ===")
    print(res["sufficiency"].round(4).to_string(index=False))
    print("\n=== scalar metrics (CRPS / AUC by arm) ===")
    s = res["scalar"][["target", "horizon", "scope", "arm", "n", "events", "crps", "auc", "brier", "cal_slope"]]
    print(s.round(4).to_string(index=False))
    print(f"\nwrote {tag or 'full '}tables to {OUT}")


if __name__ == "__main__":
    main()
