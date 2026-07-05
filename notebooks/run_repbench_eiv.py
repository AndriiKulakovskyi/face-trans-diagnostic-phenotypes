"""P2b EIV-GLM uncertainty arm (Bayesian NUTS; slow — run detached).

    PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_repbench_eiv.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from face.benchmark import eiv  # noqa: E402

OUT = ROOT / "results" / "face" / "m4_repbench"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summ = []
    for tgt in ("egf_recovery", "egf_deterioration"):
        r = eiv.eiv_uncertainty(target=tgt, horizon="V2", scope="pooled")
        print(f"\n##### {tgt} V2 pooled  (N={r['n']}, events={r['events']}) #####")
        print("-- incremental over REF (ΔELPD) --")
        print(r["vs_ref"].round(2).to_string())
        print("-- EIV vs MU: does honest uncertainty add? (ΔELPD) --")
        print(r["eiv_vs_mu"].round(2).to_string())
        print("convergence:", r["diag"])
        for comp, df in (("vs_ref", r["vs_ref"]), ("eiv_vs_mu", r["eiv_vs_mu"])):
            d = df.reset_index()
            d["comparison"], d["target"] = comp, tgt
            summ.append(d)
    pd.concat(summ, ignore_index=True).to_csv(OUT / "eiv_uncertainty.csv", index=False)
    print(f"\nwrote eiv_uncertainty.csv to {OUT}")


if __name__ == "__main__":
    main()
