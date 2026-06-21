#!/usr/bin/env python
"""Generate the QA distribution report for the FACE modeled indicators.

Self-contained HTML (per-indicator raw + rank-INT-Gaussianized panels, named empirical form,
declared family, recommended copula tier) + a committable aggregate summary CSV. Reads the
model-ready data built by ``scripts/01_build_data.py``.

    PYTHONPATH=$PWD/src python notebooks/run_distribution_report.py

Writes:
  results/reports/qa_distributions.html        (self-contained; gitignored results dir)
  reports/qa_distributions_summary.csv         (aggregate; committable)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src" / "face").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not locate FACE repository root from {start}")


REPO = _find_repo_root(Path(__file__).resolve())
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from face.reporting.distribution_report import DistributionReport  # noqa: E402


def main() -> None:
    report = DistributionReport()
    out = report.run()
    print(json.dumps(out, indent=2))
    # quick tier breakdown for the console
    summary = report.to_frame(report.analyze_all())
    print("\nrecommended-tier counts:")
    print(summary["recommended_tier"].value_counts().to_string())
    print("\nempirical-form counts:")
    print(summary["empirical_form"].value_counts().to_string())


if __name__ == "__main__":
    main()
