"""Guards for the shared axis-name constant (trans_diag.axes).

These catch the class of drift bug that motivated the constant: a renamed/re-ordered axis
(e.g. the 2026-05 cognition integration that re-locked the model at K=6 — folding in one
verbal/working-memory cognitive axis, re-merging mania with externalizing, and dropping the
separate work-disability axis) leaving stale copies, or the label dicts falling out of sync
with the canonical name list.
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
    assert AXIS_INDEX_TO_NAME["axis5"] == "cognition_verbal"   # the integrated cognitive axis
    assert AXIS_INDEX_TO_NAME["axis6"] == "metabolic"


def test_no_legacy_axis_name():
    # superseded / pre-cognition axis names must not reappear in the constant
    for legacy in ("adhd_impulsivity_trauma", "externalizing", "work_disability",
                   "cognition_fluency"):
        assert legacy not in AXIS_NAMES
        assert legacy not in AXIS_SHORT
        assert legacy not in AXIS_LABELS
