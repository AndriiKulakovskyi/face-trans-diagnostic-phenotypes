"""M3 G4 — spine-vs-corner + membership persistence (pure; synthetic panel)."""
from __future__ import annotations

import pandas as pd

from face.temporal import CANON
from face.temporal.persistence import membership_persistence, reliable_change_rate, spine_corner


def _panel():
    # p1 moves big on severity (0→3, sd 0.3 → reliable), biology flat; p2 flat everywhere. Both stay arch 0.
    rows = []
    for uid, sev2 in [("A::1", 3.0), ("A::2", 0.0)]:
        for visit, sev in [("V0", 0.0), ("V2", sev2)]:
            r = {"patient_uid": uid, "cohort": "bp", "visit": visit, "n_visits": 2}
            for ax in CANON:
                r[f"{ax}__mean"] = sev if ax == "overall_severity" else 0.0
                r[f"{ax}__sd"] = 0.3
            for k in range(8):
                r[f"archB_w{k}"] = 1.0 if k == 0 else 0.0
            r["archB_dominant"] = 0
            rows.append(r)
    return pd.DataFrame(rows)


def test_reliable_change_rate_picks_the_mover():
    rcr = reliable_change_rate(_panel(), CANON, s="V0", t="V2").set_index("axis")
    assert rcr.loc["overall_severity", "frac_reliable"] == 0.5     # p1 moves, p2 doesn't
    assert rcr.loc["overall_severity", "frac_increase"] == 0.5
    assert rcr.loc["metabolic", "frac_reliable"] == 0.0            # biology flat


def test_spine_corner_spine_moves_biology_holds():
    sc = spine_corner(_panel(), s="V0", t="V2")
    assert sc["n"] == 2
    assert sc["spine_rate"] == 0.5 and sc["bio_corner_rate"] == 0.0
    assert sc["spine_not_bio"] == 0.5                             # the §1.4 cell: p1 moves on spine, biology holds


def test_membership_persistence_stable():
    mb = membership_persistence(_panel(), arm="archB", A=8, s="V0", t="V2")
    assert mb["dominant_agree"] == 1.0                            # both keep archetype 0
    assert mb["transition"].shape == (8, 8)
    assert mb["transition"][0, 0] == 1.0 and mb["cos_median"] == 1.0
