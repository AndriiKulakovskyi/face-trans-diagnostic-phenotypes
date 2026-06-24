"""Phase-1 harness tests — block construction, derived probability, sufficiency verdict logic, and a small
end-to-end XGBoost smoke (structure + invariants, not values)."""
from __future__ import annotations

import numpy as np

from face.prognosis.repbench import data, harness


def test_derive_p_monotone_and_bounds():
    mu = np.array([40.0, 60.0, 80.0])
    p_rec = harness._derive_p("egf_recovery", mu, 10.0, np.zeros(3))
    # higher predicted GAF → higher recovery probability; valid probabilities
    assert np.all(np.diff(p_rec) > 0) and np.all((p_rec >= 0) & (p_rec <= 1))
    gaf0 = np.array([70.0, 70.0, 70.0])
    p_det = harness._derive_p("egf_deterioration", mu, 10.0, gaf0)
    # higher predicted GAF → lower deterioration probability
    assert np.all(np.diff(p_det) < 0) and np.all((p_det >= 0) & (p_det <= 1))


def test_verdict_logic():
    assert harness._verdict(-0.01, 0.01, 0.05) == "sufficient"      # CI inside ±margin → tie
    assert harness._verdict(0.1, 0.3, 0.05) == "raw-adds"           # CI above +margin
    assert harness._verdict(-0.3, -0.1, 0.05) == "latent-better"    # CI below −margin
    assert harness._verdict(-0.2, 0.2, 0.05) == "inconclusive"      # CI straddles the band


def test_boot_diff_zero_for_identical():
    a = np.linspace(0, 1, 200)
    d, lo, hi = harness._boot_diff(a, a.copy(), n_boot=100, seed=0)
    assert d == 0.0 and lo == 0.0 and hi == 0.0


def test_run_cell_smoke_structure():
    raw = data.load_raw()
    frame = data.assemble().sample(1500, random_state=0)        # enough deterioration events for 3-fold
    r = harness.run_cell(frame, raw, target="egf_deterioration", horizon="V2", scope="smoke",
                         n_boot=30, n_splits=3, n_repeats=1)
    scal, nb, suff = r["scalar"], r["net_benefit"], r["sufficiency"]
    # every arm scored (incremental ladder + the no-GAF sensitivity arms)
    assert set(scal["arm"]) == set(harness.ARM_SPECS)
    assert np.isfinite(scal["crps"]).all()
    assert ((scal["auc"] >= 0) & (scal["auc"] <= 1)).all()
    # the no-GAF base (REF0) drops one feature vs REF (the baseline-GAF column)
    blocks = harness._blocks(frame[data.eligible(frame, "egf_deterioration", "V2")], raw.reindex(frame.index))
    assert blocks["ref0"].shape[1] == blocks["ref"].shape[1] - 1
    # net-benefit curve present for every arm over the threshold band
    assert set(nb["arm"]) == set(harness.ARM_SPECS) and nb["threshold"].between(0.05, 0.50).all()
    # sufficiency under both contrasts (with/without baseline GAF) for crps / brier / auc
    assert set(suff["contrast"]) == {"with_gaf", "no_gaf"}
    assert set(suff["metric"]) == {"crps", "brier", "auc"}
    assert set(suff["verdict"]) <= {"sufficient", "raw-adds", "latent-better", "inconclusive", "tie"}


def test_v1_endpoints_exist():
    panel = data.load_outcomes()
    for h in ("V1", "V2"):
        assert f"ep_egf_recovery__{h}" in panel.columns
        assert f"ep_egf_deterioration__{h}" in panel.columns
    # V1 has more eligible patients than V2 (milder attrition)
    n1 = int(panel["ep_egf_recovery__V1"].notna().sum())
    n2 = int(panel["ep_egf_recovery__V2"].notna().sum())
    assert n1 > n2
