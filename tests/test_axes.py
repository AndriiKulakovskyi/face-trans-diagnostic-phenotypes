"""Guards for the shared axis-name constant (trans_diag.axes).

These catch the class of drift bug that motivated the constant: a renamed axis (e.g. axis 6:
adhd_impulsivity_trauma → work_disability) leaving stale copies, or the label dicts falling out
of sync with the canonical name list.
"""
from trans_diag import AXIS_INDEX_TO_NAME, AXIS_LABELS, AXIS_NAMES, AXIS_SHORT


def test_six_unique_axes():
    assert len(AXIS_NAMES) == 6
    assert len(set(AXIS_NAMES)) == 6


def test_label_dicts_cover_exactly_the_axes():
    assert set(AXIS_SHORT) == set(AXIS_NAMES)
    assert set(AXIS_LABELS) == set(AXIS_NAMES)


def test_index_map_matches_ss_order():
    # 07_dimensional_refine writes axis1..axis6 in SS order; the map must mirror AXIS_NAMES.
    assert AXIS_INDEX_TO_NAME == {f"axis{i + 1}": n for i, n in enumerate(AXIS_NAMES)}
    assert AXIS_INDEX_TO_NAME["axis6"] == "work_disability"   # the renamed axis, pinned


def test_no_legacy_axis_name():
    # the superseded mean-fill axis must not reappear anywhere in the constant
    assert "adhd_impulsivity_trauma" not in AXIS_NAMES
    assert "adhd_impulsivity_trauma" not in AXIS_SHORT
    assert "adhd_impulsivity_trauma" not in AXIS_LABELS
