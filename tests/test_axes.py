"""Guards for the v2 axis-name source of truth (``trans_diag.axes``).

These catch axis-name/order drift: the v2 hierarchical model (scripts 01–15, LABBOOK V2-9..V2-20)
locked the structure at **K=3** — internalizing, cognition, cardiometabolic — after the
substance_use_disorder construct collapsed the earlier K=4 (illness_course is no longer a stable
standalone axis). mania, suicidality & substance-use disorder are *orthogonal standalone* dimensions
(|loading| < 0.10), NOT axes. ``trans_diag.axes`` is the canonical map written for the manuscript and
downstream code. (The superseded v1 6-axis solution is archived at git tag ``v1-archive-2026-05-30``.)
"""
from trans_diag.axes import (
    AXIS_INDEX_TO_NAME,
    AXIS_LABELS,
    AXIS_NAMES,
    AXIS_SHORT,
    ORTHOGONAL_DIMENSIONS,
)


def test_three_unique_axes():
    assert AXIS_NAMES == ["internalizing", "cognition", "cardiometabolic"]
    assert len(set(AXIS_NAMES)) == 3


def test_label_dicts_cover_exactly_the_axes():
    assert set(AXIS_SHORT) == set(AXIS_NAMES)
    assert set(AXIS_LABELS) == set(AXIS_NAMES)


def test_index_map_matches_paf_order():
    # stage 3 writes dim1..dim3 in PAF / descending-eigenvalue order; the map must mirror AXIS_NAMES.
    assert AXIS_INDEX_TO_NAME == {f"dim{i + 1}": n for i, n in enumerate(AXIS_NAMES)}
    assert AXIS_INDEX_TO_NAME["dim1"] == "internalizing"
    assert AXIS_INDEX_TO_NAME["dim3"] == "cardiometabolic"


def test_orthogonal_standalones_are_not_axes():
    # mania / suicidality / substance-use are valid constructs but ORTHOGONAL → not axes.
    assert set(ORTHOGONAL_DIMENSIONS) == {"mania_activation", "suicidal_ideation", "substance_use_disorder"}
    for name in ORTHOGONAL_DIMENSIONS:
        assert name not in AXIS_NAMES


def test_no_legacy_v1_axis_name():
    # superseded v1 axis names must not reappear in the v2 constant (v2 has 4 axes, not 6).
    for legacy in ("later_onset", "mania_activation", "illness_burden",
                   "cognition_verbal", "metabolic", "depression_severity"):
        assert legacy not in AXIS_NAMES
        assert legacy not in AXIS_SHORT
        assert legacy not in AXIS_LABELS
