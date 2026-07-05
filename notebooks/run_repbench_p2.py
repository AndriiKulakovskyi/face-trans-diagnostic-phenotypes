"""P2 efficiency + transport — learning curves and LOCO (XGBoost; fast, foreground).

    PYTHONPATH=$PWD/src python notebooks/run_repbench_p2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from face.benchmark import curves  # noqa: E402

OUT = ROOT / "results" / "face" / "m4_repbench"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== learning curves — recovery pooled V2 (AUC vs N) ===")
    lc = curves.learning_curve(target="egf_recovery", horizon="V2", scope="pooled")
    print(lc.round(3).to_string(index=False))
    lc.to_csv(OUT / "learning_curve_recovery_V2.csv", index=False)

    for tgt in ("egf_recovery", "egf_deterioration"):
        print(f"\n=== LOCO transport — {tgt} V2 (out-of-sample AUC on the held-out cohort) ===")
        lo = curves.loco(target=tgt, horizon="V2")
        print(lo.round(3).to_string(index=False))
        lo.to_csv(OUT / f"loco_{tgt}_V2.csv", index=False)
    print(f"\nwrote learning_curve / loco tables to {OUT}")


if __name__ == "__main__":
    main()
