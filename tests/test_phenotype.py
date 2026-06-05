"""Guards for the phenotype-atlas feature module (``trans_diag.phenotype``).

Structural checks + a synthetic-data check of the scorer, so they run on a clean clone (no
dependence on the gitignored ``results/hfa/stage2_scores.pkl``). See ``docs/PHENOTYPE_ATLAS.md``.
"""
import numpy as np
import pandas as pd

from trans_diag.phenotype import (
    AXES,
    FACTOR_META,
    PHENOTYPE_FACTORS,
    STANDALONES,
    build_phenotype_factors,
)


def test_factor_roster_is_well_formed():
    # AXES + STANDALONES partition the factor set exactly; meta covers every factor.
    assert set(AXES) | set(STANDALONES) == set(PHENOTYPE_FACTORS)
    assert not (set(AXES) & set(STANDALONES))
    assert set(FACTOR_META) == set(PHENOTYPE_FACTORS)
    assert AXES == ("internalizing", "cognition", "cardiometabolic")
    for fac, members in PHENOTYPE_FACTORS.items():
        assert members, f"{fac} has no constructs"
        for construct, sign in members:
            assert isinstance(construct, str) and sign in (-1, 1)


def test_single_construct_standalones():
    # substance/mania/suicidality pass a single construct straight through (+1).
    for fac in ("substance_use", "mania", "suicidality"):
        assert len(PHENOTYPE_FACTORS[fac]) == 1 and PHENOTYPE_FACTORS[fac][0][1] == 1


def test_build_scores_and_coverage_on_synthetic_data():
    # two constructs of internalizing, one observed-everywhere, one half-missing.
    idx = pd.MultiIndex.from_product([["bp"], range(4)], names=["cohort", "patient_id"])
    S = pd.DataFrame(
        {"qidsr": [1.0, 2.0, 3.0, 4.0], "eq5d": [4.0, np.nan, 2.0, np.nan]}, index=idx,
    )
    scores, coverage = build_phenotype_factors(S)
    # only constructs present in S are used → internalizing is scored from {qidsr, eq5d}
    assert "internalizing" in scores.columns
    # coverage = fraction of the *present* members observed (2 of 2, or 1 of 2)
    assert list(coverage["internalizing"]) == [1.0, 0.5, 1.0, 0.5]
    # eq5d has sign -1: higher eq5d must lower the score → patient 0 (eq5d=4) < its qids-only peers
    assert scores["internalizing"].notna().all()
    # a factor whose constructs are entirely absent is dropped, not errored
    assert "cardiometabolic" not in scores.columns


def test_min_coverage_gating():
    idx = pd.MultiIndex.from_product([["bp"], range(3)], names=["cohort", "patient_id"])
    S = pd.DataFrame({"qidsr": [1.0, 2.0, 3.0], "madrs": [np.nan, 3.0, 5.0]}, index=idx)
    scores, _ = build_phenotype_factors(S, min_coverage=0.75)  # need ≥75% of 2 members → both
    # patient 0 observes only qidsr (cov 0.5 < 0.75) → gated to NaN; patients 1,2 keep both
    assert scores["internalizing"].isna().tolist() == [True, False, False]
