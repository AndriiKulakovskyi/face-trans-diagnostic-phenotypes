#!/usr/bin/env python3
"""34 — G2: score the longitudinal panel (V0 → V1 → V2) onto the FIXED M1/M2 model.

The substrate every M3 headline acts on. Per (patient, visit): the 9-dim coordinates with uncertainty (6
continuous axes analytically from the certified loadings; the 3 explicit axes by full-N projection under
fixed certified parameters), frozen-V0-scaled (§3.1) so genuine change is preserved; archetype memberships
(Arm B G-residualized = primary, Arm A = contextual); and the per-axis G1 invariance license. V0 is reused
from M2.0 (`coordinates_full.parquet`) — the prep was validated bit-exact, so only V1/V2 are newly projected
(cached → resumable). No re-fit, no imputation. Methods: docs/TEMPORAL_MODEL.md §3.

    python3 scripts/34_score_panel.py

Writes results/face/m3/{panel_coords.parquet, panel_draws.npz, proj_V{1,2}.npz} (gitignored) ·
reports/34_score_panel.md · docs/figures/34_trajectories.png.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

from face.temporal import CANON, VISITS  # noqa: E402
from face.temporal.dropout import patient_retention  # noqa: E402
from face.temporal.membership import archetype_membership  # noqa: E402

PROC = REPO / "data" / "processed"
M2 = REPO / "results" / "face" / "m2"
M3 = REPO / "results" / "face" / "m3"
REPORTS, FIGS = REPO / "reports", REPO / "docs" / "figures"
CONT6 = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep", "mania_activation"]
EXPL3 = ["suicidality", "developmental_risk", "substance"]
EXPLICIT9 = ["overall_severity", "suicidality", "developmental_risk", "substance"]
SPECIFICS = [f for f in CANON if f != "overall_severity"]            # the 8 corner axes (Arm B)
AXCOLS = [f"{f}__{s}" for f in CANON for s in ("mean", "sd", "hdi_lo", "hdi_hi", "n_obs", "reliability")]
NDRAW_CONT, DRAWS, TUNE, CHAINS, NKEEP, SEED = 200, 400, 500, 2, 200, 20260609


def _tier(n):
    return np.where(n >= 3, "well", np.where(n >= 1, "partial", "prior-dominated"))


def _score_followup(visit, spec, post, idata, mp_v0, cert_index, B_v0):
    """Score one follow-up visit: 6 continuous axes (analytic) + 3 explicit axes (cached projection).
    Returns (coords_df indexed by (cohort,patient_id) with the AXCOLS schema, draws [NKEEP, N, 9])."""
    from scipy.stats import norm

    from face.scoring import reliability_flags
    from face.strata.scoring import (
        conditional_gaussian_draws,
        explicit_nobs,
        project_explicit_full_n,
    )
    from face.temporal.standardize import prep_visit_continuous, prep_visit_mixed
    z = float(norm.ppf(0.97))                                        # 94% HDI
    B = pd.read_parquet(PROC / f"baseline_{visit.lower()}.parquet")

    prep_c = prep_visit_continuous(spec, B)                          # continuous block, V0-scaled
    cg = conditional_gaussian_draws(prep_c.M, post, prep_c.factor_cols, n_draws=NDRAW_CONT, seed=SEED)
    nobs_c, tier_c = reliability_flags(prep_c.M, prep_c.items, prep_c.home, prep_c.factor_cols)
    cidx = {f: i for i, f in enumerate(prep_c.factor_cols)}

    mp_vis = prep_visit_mixed(spec, mp_v0, B, cert_index=cert_index, B_v0=B_v0)
    cache = M3 / f"proj_{visit}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        res = {"mean": d["mean"], "sd": d["sd"], "draws": d["draws"], "fcols": list(d["fcols"]),
               "diag": {"max_rhat": float(d["max_rhat"]), "divergences": int(d["divergences"])}}
        print(f"  [{visit}] explicit projection [cached]", flush=True)
    else:
        res = project_explicit_full_n(mp_vis, idata, draws=DRAWS, tune=TUNE, chains=CHAINS, seed=SEED)
        step = max(1, res["draws"].shape[0] // NKEEP)
        res["draws"] = res["draws"][::step][:NKEEP]
        np.savez_compressed(cache, mean=res["mean"], sd=res["sd"], draws=res["draws"],
                            fcols=np.array(res["fcols"]), max_rhat=res["diag"]["max_rhat"],
                            divergences=res["diag"]["divergences"])
    ecols = list(res["fcols"])
    en = explicit_nobs(mp_vis)
    ne = pd.DataFrame(en["n_obs"], index=mp_vis.base.index, columns=en["fcols"])

    N = len(B)
    df = pd.DataFrame(index=B.index)
    draws = np.full((NKEEP, N, len(CANON)), np.nan, dtype="float32")
    for f in CANON:
        di = CANON.index(f)
        if f in CONT6:
            i = cidx[f]; m, s, n = cg["mean"][:, i], cg["sd"][:, i], nobs_c[:, i]
            rel = tier_c[:, i]; draws[:, :, di] = cg["draws"][:NKEEP, :, i]
        else:
            k = ecols.index(f); m, s, n = res["mean"][:, k], res["sd"][:, k], ne[f].to_numpy()
            rel = _tier(n); draws[:, :, di] = res["draws"][:NKEEP, :, k]
        df[f"{f}__mean"] = np.round(m, 3); df[f"{f}__sd"] = np.round(s, 3)
        df[f"{f}__hdi_lo"] = np.round(m - z * s, 3); df[f"{f}__hdi_hi"] = np.round(m + z * s, 3)
        df[f"{f}__n_obs"] = n.astype(int); df[f"{f}__reliability"] = rel
    return df[AXCOLS], draws, res["diag"]


def main():
    import arviz as az

    from face.models.bayesian.continuous_core import S5_FACTORS, prepare_mixed
    from face.temporal.standardize import load_spec
    M3.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    idata = az.from_netcdf(REPO / "results/face/s5_cert9_s1/idata.nc")
    post = idata.posterior
    spec = load_spec(PROC / "v0_standardization_spec.json")
    lic = pd.read_parquet(M3 / "invariance_license.parquet").set_index("axis")["license"].to_dict()

    # V0 explicit structure (built once; cert_index defines the ordinal coding) — reused for V1/V2
    print("[setup] V0 mixed structure...", flush=True)
    mp_v0 = prepare_mixed(S5_FACTORS, explicit_factors=EXPLICIT9, min_cohorts=2)
    mpC = prepare_mixed(S5_FACTORS, explicit_factors=EXPLICIT9, min_cohorts=2, balanced=True,
                        n_subsample=2000, seed=20260605)
    B_v0 = pd.read_parquet(PROC / "baseline_v0.parquet")

    # archetype profiles (Arm A all-9, Arm B specifics)
    zdf = pd.read_csv(M2 / "archetype_profiles.csv")
    Z_A = zdf[zdf.arm == "A_all9"][list(CANON)].to_numpy()
    Z_B = zdf[zdf.arm == "B_specifics"][SPECIFICS].to_numpy()
    names_A = zdf[zdf.arm == "A_all9"]["name"].tolist()
    names_B = zdf[zdf.arm == "B_specifics"]["name"].tolist()
    b_cols = [CANON.index(s) for s in SPECIFICS]

    # ---- per-visit coords (+ draws): V0 reused from M2.0, V1/V2 newly scored ----
    coords, draws_by_visit, diags = {}, {}, {}
    v0 = pd.read_parquet(M2 / "coordinates_full.parquet").set_index(["cohort", "patient_id"])
    coords["V0"] = v0[AXCOLS]
    dz = np.load(M2 / "coordinates_draws.npz", allow_pickle=True)
    draws_by_visit["V0"] = dz["draws"]
    for visit in [v for v in VISITS if v != "V0"]:
        print(f"[score] {visit} (continuous analytic + explicit projection)...", flush=True)
        coords[visit], draws_by_visit[visit], diags[visit] = _score_followup(
            visit, spec, post, idata, mp_v0, mpC.base.index, B_v0)

    # ---- memberships per visit (both arms) + assemble the panel ----
    parts, draws_stack = [], []
    for visit in VISITS:
        cdf, dr = coords[visit], draws_by_visit[visit]
        mA = archetype_membership(cdf[[f"{f}__mean" for f in CANON]].to_numpy(), dr, list(range(len(CANON))),
                                  Z_A, names_A, prefix="archA", index=cdf.index, seed=SEED)
        mB = archetype_membership(cdf[[f"{f}__mean" for f in SPECIFICS]].to_numpy(), dr, b_cols,
                                  Z_B, names_B, prefix="archB", index=cdf.index, seed=SEED)
        vp = pd.concat([cdf, mA, mB], axis=1)
        vp["visit"] = visit
        parts.append(vp)
        draws_stack.append(dr[:, :, :].astype("float32"))
    panel = pd.concat(parts).reset_index()
    panel["patient_uid"] = panel["cohort"].astype(str).str.upper() + "::" + panel["patient_id"].astype(str)
    for f in CANON:
        panel[f"{f}__license"] = lic.get(f, "not-tested")

    # retention + arm (validation-only)
    from face.temporal.panel import load_long
    long_ret = patient_retention(load_long()).reset_index()
    panel = panel.merge(long_ret[["cohort", "patient_id", "n_visits", "last_visit"]],
                        on=["cohort", "patient_id"], how="left")
    arm = pd.read_parquet(M2 / "validation_table.parquet")[["cohort", "patient_id", "arm"]]
    panel = panel.merge(arm, on=["cohort", "patient_id"], how="left")

    front = ["patient_uid", "cohort", "patient_id", "visit", "arm", "n_visits", "last_visit"]
    panel = panel[front + [c for c in panel.columns if c not in front]]
    panel.to_parquet(M3 / "panel_coords.parquet", index=False)
    draws_all = np.concatenate(draws_stack, axis=1)                  # [NKEEP, total_rows, 9]
    np.savez_compressed(M3 / "panel_draws.npz", draws=draws_all, dims=np.array(CANON),
                        patient_uid=panel["patient_uid"].to_numpy(), visit=panel["visit"].to_numpy())

    # ---- QC: Arm-A dominant on V0 reproduces M2's patient_strata ----
    ps = pd.read_parquet(REPO / "results/face/patient_strata.parquet").set_index(["cohort", "patient_id"])
    v0p = panel[panel.visit == "V0"].set_index(["cohort", "patient_id"]).reindex(ps.index)
    agree = float((v0p["archA_dominant"].to_numpy() == ps["arch_dominant"].to_numpy()).mean())

    _figure(panel)
    n_by_v = panel.groupby("visit").size().reindex(list(VISITS))
    md = ["# 34 — G2: longitudinal coordinate + membership panel (V0 → V1 → V2)", "",
          "Per (patient, visit) on the **frozen V0 scale**: 9-dim coordinates + uncertainty, Arm-B "
          "(G-residualized) + Arm-A archetype memberships, and the per-axis G1 license. V0 reused from M2.0 "
          "(prep validated bit-exact); V1/V2 newly projected under fixed certified parameters (no re-fit).", "",
          f"- **Panel rows:** {len(panel):,} — " + " · ".join(f"{v} {int(n_by_v[v]):,}" for v in VISITS) + ".",
          f"- **V0 QC:** Arm-A dominant archetype reproduces M2's `patient_strata` at **{agree:.1%}** "
          "agreement (validates the frozen-scale scoring + the simplex projector).",
          "- **Explicit-projection convergence (V1/V2):** "
          + " · ".join(f"{v} R-hat {diags[v]['max_rhat']:.3f}/div {diags[v]['divergences']}"
                       for v in diags) + ".", "",
          "## Coordinate trajectories (cohort-mean, frozen scale)",
          "Mean coordinate per axis per visit — licensed axes carry patient-change meaning; "
          "inflammatory partial, the 3 explicit axes descriptive (per the G1 license).", ""]
    traj = pd.DataFrame({f: [panel[panel.visit == v][f"{f}__mean"].mean() for v in VISITS] for f in CANON},
                        index=list(VISITS)).round(3)
    md += [traj.T.to_markdown(), "",
           "- License attached per axis: " + ", ".join(f"{f}={lic.get(f,'not-tested')}" for f in CANON) + ".",
           "", "## Artifacts (results/face/m3/, gitignored)",
           "- `panel_coords.parquet` — the tidy (patient_uid, visit) substrate (coords + memberships + "
           "license + retention).",
           f"- `panel_draws.npz` — [{NKEEP}, {len(panel):,}, 9] posterior draws (the uncertainty arm for "
           "G3/G4).",
           "- `proj_V{1,2}.npz` — cached explicit projections.", "",
           f"Runtime {(time.time()-t0)/60:.1f} min. Next: stage 35 (G3 trait/state)."]
    (REPORTS / "34_score_panel.md").write_text("\n".join(md))
    print("\n".join(md))


def _figure(panel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    for f in CANON:
        y = [panel[panel.visit == v][f"{f}__mean"].mean() for v in VISITS]
        ax.plot(list(VISITS), y, "o-", label=f)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("cohort-mean coordinate (frozen V0 z-scale)")
    ax.set_title("Coordinate trajectories V0 → V1 → V2 (population mean)")
    ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "34_trajectories.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
