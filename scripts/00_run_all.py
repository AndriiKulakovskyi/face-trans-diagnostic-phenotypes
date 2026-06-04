"""Reproduce the entire FACE **v2** manuscript pipeline from the raw data, in order.

Each step writes aggregate artifacts to ``results/hfa/`` (+ ``results/reports/``); later steps consume
earlier outputs. Run from the repo root:  ``python3 scripts/00_run_all.py``
Requires the confidential cohort CSVs in ``data/`` and ``pip install -e ".[full]"``.

The order follows ``docs/PIPELINE.md`` and the inter-script dependencies (e.g. ``08`` derives the
relapse outcome consumed by ``12``–``15``; ``11`` reuses Stage-0/2 logic; ``15`` reuses ``11``). The
pipeline is deterministic (fixed seeds) apart from ~1e-9 BLAS round-off.

The manuscript ``.docx`` is built separately (after the figures) via ``scripts/build_manuscript.py``
(needs pandoc); it is intentionally not part of this analysis run.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# (script, args) — strict execution order; the filename prefix IS the order within each group.
STEPS: list[tuple[str, list[str]]] = [
    ("qa_harmonization.py", []),            # 3-part data-processing QA report (gate before analysis)
    # ── hierarchical / bifactor measurement model — Stages 0–4 ──
    ("01_hfa_stage0_itemset.py", []),    # freeze the 194-item V0 set + factorability
    ("02_hfa_stage1_efa.py", []),        # exploratory first-order EFA (Horn parallel analysis)
    ("03_hfa_stage2.py", []),            # hybrid first-order constructs (94) → Φ₁
    ("04_hfa_stage3.py", []),            # second-order: K=4 axes; Schmid–Leiman ECV (no p-factor)
    ("05_hfa_kselect.py", []),           # per-factor split-half K-selection deep dive
    ("06_hfa_stage4.py", []),            # validation: confound η² / leave-cohort-out / granularity
    # ── stratification arm ──
    ("07_phase5_stratify.py", []),       # discrete-vs-continuum battery → dimensional
    # ── validation A–D (08 derives the relapse outcome used by 12–15) ──
    ("08_v1v4_inventory.py", []),        # V1–V4 inventory + LOCKED CGI-S relapse derivation
    ("09_cohort_confound.py", []),       # Study A — cohort confound
    ("10_orthogonality_pfactor.py", []), # Study B — symptom⊥biology / p-factor (headline)
    ("11_longitudinal_coherence.py", []),# Study C — measurement invariance + score stability
    ("12_predictive_validity.py", []),   # Study D — prognosis vs DSM
    ("13_predictive_survival.py", []),   # Study D-refined — remission-based discrete-time survival
    ("14_relapse_richbaseline.py", []),  # Study D3 — richer baseline vs 6 axes
    ("15_relapse_trajectory.py", []),    # Study D4 — early-course prognosis
    # ── sensitivity analyses ──
    ("sensitivity_aggregation.py", []),  # granularity invariance / conditioning audit
    ("sensitivity_comorbidity.py", []),  # the 24 *_mhoccur flags decomposition
    ("sensitivity_polychoric.py", []),   # tetrachoric sensitivity of the K=4 structure
    # ── figures ──
    ("figures_manuscript.py", []),       # 6 manuscript figures from results/hfa/
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
    print(f"\n{'='*72}\nALL {len(STEPS)} STEPS OK in {time.time()-t0:.0f}s")
    print("Next: python3 scripts/build_manuscript.py   # build the .docx (needs pandoc)")
    print(f"{'='*72}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
