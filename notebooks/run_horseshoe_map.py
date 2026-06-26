#!/usr/bin/env python
"""The 8-factor sparse-ESEM map: metabolic + inflammatory merged into one immunometabolic factor,
with the off-home specific cross-loadings freed under a REGULARIZED HORSESHOE.

The horseshoe is a sparsity-inducing prior — a sharp spike at 0 crushes the many noise cross-loadings
(so a low-instrument factor's column is not diluted and its identity is protected: substance n=4, mania
n=2) while heavy tails let a GENUINELY supported small cross-loading escape and become credible. This is
the principled way to let instruments load weakly on several axes without collapsing thin factors.

Ladder (each rung warm-starts the next):
  hs_s1_merged  continuous core (G, cognition, immunometabolic, sleep)        — hard-zero, clean backbone
  hs_s3_merged  + developmental_risk, mania_activation (continuous)           — hard-zero
  hs_s5_merged  full 8-factor MIXED map (+ suicidality, substance explicit)   — HORSESHOE cross-loadings,
                warm-started from the clean hard-zero backbone (identified, unlike free cross-loadings).

    HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_horseshoe_map.py --smoke
    python3 scripts/run_job.py horseshoe -- env HDF5_USE_FILE_LOCKING=FALSE \
        python notebooks/run_horseshoe_map.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for c in (start, *start.parents):
        if (c / "src" / "face" / "models").exists() and (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("repo root not found")


REPO = _find_repo_root(Path(__file__).resolve())
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
for m in [n for n in sys.modules if n == "face" or n.startswith("face.")]:
    f = getattr(sys.modules[m], "__file__", None)
    if f and SRC not in Path(f).resolve().parents:
        del sys.modules[m]

from face.models.bayesian.measurement_model_oop import (  # noqa: E402
    DEFAULT_EXPLICIT_FACTORS, MeasurementConfig, StageDefinition, StageRunner,
)

MERGED_MATRIX = REPO / "configs" / "prior_loading_matrix_v3_biomerge.csv"
F1 = ["overall_severity", "cognition", "immunometabolic", "sleep"]
F3 = F1 + ["developmental_risk", "mania_activation"]
F8 = ["overall_severity", "cognition", "immunometabolic", "sleep",
      "suicidality", "developmental_risk", "mania_activation", "substance"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=2000, help="balanced subsample (default 2000)")
    p.add_argument("--tau0", type=float, default=0.05, help="horseshoe global-shrinkage scale")
    p.add_argument("--slab-c", type=float, default=0.30, help="horseshoe slab width (caps escaped |λ|)")
    p.add_argument("--smoke", action="store_true", help="tiny wiring check")
    p.add_argument("--final", choices=["horseshoe", "hardzero"], default="horseshoe",
                   help="cross-loading treatment for the final 8-factor MIXED map: horseshoe (sparse ESEM) "
                        "or hardzero (the operational map for M2-M5, converges like the certified fit).")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    cont = dict(correlated=True, windows=True, mixed=False, balanced=True, n_subsample=a.n, seed=20260605)
    mixed = dict(correlated=True, windows=True, mixed=True, explicit_factors=list(DEFAULT_EXPLICIT_FACTORS),
                 min_cohorts=2, balanced=True, n_subsample=a.n, seed=20260605)
    dc, dt, mc_, mt = (1000, 1000, 1500, 2000) if not a.smoke else (40, 40, 40, 40)
    ch = 4 if not a.smoke else 2
    s5name = "hs_s5_merged" if a.final == "horseshoe" else "hs_s5_merged_hz"
    s1 = StageDefinition("hs_s1_merged", F1, draws=dc, tune=dt, chains=ch, target_accept=0.95, **cont)
    s3 = StageDefinition("hs_s3_merged", F3, draws=dc, tune=dt, chains=ch, target_accept=0.95, **cont)
    s5 = StageDefinition(s5name, F8, draws=mc_, tune=mt, chains=ch, target_accept=0.95, **mixed)

    base = MeasurementConfig().with_gaussian_copula()
    base = replace(base, prior_matrix=MERGED_MATRIX,
                   output_dir=base.output_dir / "copula" / "horseshoe_8d",
                   figure_dir=base.figure_dir / "copula" / "horseshoe_8d")
    hs = base.with_horseshoe(tau0=a.tau0, slab_c=a.slab_c)

    final_cfg, final_mode = (hs, "HORSESHOE") if a.final == "horseshoe" else (base, "hard-zero")
    print(f"[hs-map] 8-factor merged map; final={a.final} (tau0={a.tau0} slab_c={a.slab_c}); N={a.n}", flush=True)
    print(f"[hs-map] factors: {F8}", flush=True)

    # backbone always hard-zero (clean, fast); final stage per --final, all same output_dir for warm-start
    runner_hz = StageRunner(base)
    prev = None
    for st, runner, mode in [(s1, runner_hz, "hard-zero"), (s3, runner_hz, "hard-zero"),
                             (s5, StageRunner(final_cfg), final_mode)]:
        print(f"\n=== {st.name} ({mode}, {len(st.factors)} factors{' MIXED' if st.mixed else ''}) ===", flush=True)
        t0 = time.time()
        _idata, man = runner.run_stage(st, overwrite=a.overwrite, prev_stage=prev)
        print(f"[{st.name}] diagnostics={man.get('diagnostics', {})}  elapsed={round(time.time()-t0)}s", flush=True)
        prev = st

    out = base.output_dir / s5.name
    print(f"\n[hs-map] DONE -> {out}/idata.nc", flush=True)
    print(f"[hs-map] next: export CI-aware loadings: "
          f"`python notebooks/run_export_loadings.py --idata {out}/idata.nc`", flush=True)


if __name__ == "__main__":
    main()
