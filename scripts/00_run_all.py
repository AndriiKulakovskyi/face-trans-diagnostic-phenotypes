"""Reproduce the entire FACE manuscript pipeline from the raw data, in order.

Each step writes to results/ + reports/; later steps consume earlier outputs.
Run from the repo root:  python3 scripts/00_run_all.py
Requires the full extras:  pip install -e ".[full]"   (torch, neuroHarmonize, kaleido)

The pipeline is deterministic (fixed seeds) apart from ~1e-9 BLAS round-off and
the PyTorch autoencoder (CPU-deterministic but BLAS-sensitive); all
manuscript-reported digits reproduce.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# (script, args) — run strictly in numeric order; the filename prefix IS the
# execution order, so a reviewer can also run them one-by-one top to bottom.
# The only script that runs twice is 10_phase5_outcomes (once per follow-up visit).
STEPS: list[tuple[str, list[str]]] = [
    ("01_manuscript_table1.py", []),        # cohort demographics (Table 1)
    ("02_confound_ladder.py", []),          # §3.1 confound trap: why naïve clustering recovers nuisance
    ("03_cluster_domains.py", []),          # → residualized domain scores + masked-cosine embedding
    ("04_structure_test.py", []),           # discrete-vs-dimensional verdict (uses embedding)
    ("05_dimensional_axes.py", []),         # classical varimax FA (parallel analysis) — AE reference
    ("06_dimensional_ae.py", []),           # masked autoencoder (no-imputation cross-check)
    ("07_dimensional_refine.py", []),       # LOCKED K=6 axes (split-half congruence) → final scores
    ("08_longitudinal_axes.py", []),        # trait–state stability V0→V4
    ("09_longitudinal_coherence.py", []),   # discrete-flow negative result + DSM contingency
    ("10_phase5_outcomes.py", ["--visit", "V1"]),   # head-to-head (primary)
    ("10_phase5_outcomes.py", ["--visit", "V2"]),   # head-to-head (follow-up)
    ("11_phase5_ci.py", []),                # repeated-CV confidence intervals
    ("12_phase5_decircularized.py", []),    # de-circularization sensitivity
    ("13_robustness_site.py", []),          # ComBat site harmonization
    ("14_cognition_bpsz.py", []),           # BP/SZ cognition (g + speed)
    ("15_review_checks.py", []),            # cohort/site eta^2, CCA null, missingness, rho CI, figS2
    ("16_manuscript_figures.py", []),       # Figures 1–5 (static)
    ("17_export_longitudinal_figure.py", []),   # Suppl. Fig S1 (discrete flow)
    ("18_export_dimensional_flow.py", []),  # Figure 6 (dimensional flow + DSM η²)
    ("19_face_score.py", []),               # §3.9 FACE profile (FACE-D + FACE-M) + validation
    ("20_robustness_cvrefit.py", []),       # Limitation 10: re-fit axes inside CV folds (optimism)
]


def main() -> int:
    t0 = time.time()
    for i, (script, args) in enumerate(STEPS, 1):
        cmd = [sys.executable, str(REPO / "scripts" / script), *args]
        label = f"{script} {' '.join(args)}".strip()
        print(f"\n{'='*72}\n[{i}/{len(STEPS)}] {label}\n{'='*72}", flush=True)
        t = time.time()
        r = subprocess.run(cmd, cwd=REPO)
        if r.returncode != 0:
            print(f"\n!!! STEP FAILED: {label} (exit {r.returncode}) — stopping.", flush=True)
            return r.returncode
        print(f"   ...done in {time.time()-t:.0f}s", flush=True)
    print(f"\n{'='*72}\nALL {len(STEPS)} STEPS OK in {time.time()-t0:.0f}s\n{'='*72}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
