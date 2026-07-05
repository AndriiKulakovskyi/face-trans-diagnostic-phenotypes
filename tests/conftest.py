"""Pytest path bootstrap for the ``src/`` layout + a CI skip-on-missing-data guard.

Only ``src/`` needs to be importable to run the tests without installing the
project (``import face`` resolves from there).

The confidential 3-cohort data (``data/processed/``) and the large model results
(``results/m1_measurement/`` … ``results/m5_treatment/``) are gitignored and therefore **absent in
CI**. The unit + golden tests run on
synthetic fixtures, but a handful of integration tests load those artifacts directly; rather than let
them *fail* in CI, a missing-data ``FileNotFoundError`` pointing at those paths is reclassified as a
**skip** (locally, where the data is present, the tests run normally).
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_src = str(REPO_ROOT / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# Path fragments that mark a confidential / gitignored artifact absent in CI.
_CI_ABSENT = ("data/processed", "results/m1_measurement", "results/m2_strata",
              "results/m3_temporal", "results/m4_prognosis", "results/m5_treatment",
              "results/analyses", "baseline_v")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):  # noqa: ARG001
    outcome = yield
    rep = outcome.get_result()
    if (rep.when == "call" and rep.failed and call.excinfo is not None
            and call.excinfo.errisinstance((FileNotFoundError, OSError))):
        msg = str(call.excinfo.value).replace("\\", "/")
        if any(frag in msg for frag in _CI_ABSENT):
            rep.outcome = "skipped"
            rep.longrepr = f"skipped — needs confidential data/results (absent in CI): {msg}"
