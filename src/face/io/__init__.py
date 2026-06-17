"""I/O utilities for reproducible, observable runs.

- ``runstate``  — atomic read/write of ``run/<job>.json`` job-state files (used by the detached
  launcher ``scripts/run_job.py`` and the dashboard ``scripts/status.py``).
- ``progress``  — ``heartbeat(...)`` so long fits report stage/progress into their run-state.
- ``manifest``  — persist exact fit metadata (N, cohort counts, seed, patient-index hash, commit,
  package versions, diagnostics) + the exact sampled patient index, so scoring/QC loads the real
  index instead of reconstructing it from a seed (issue P2-03).
"""
from __future__ import annotations

from . import manifest, progress, runstate

__all__ = ["runstate", "progress", "manifest"]
