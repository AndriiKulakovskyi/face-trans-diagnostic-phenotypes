#!/usr/bin/env python3
"""46 — M4.6 clinical value: does the map change decisions? (AUC · calibration · net benefit)

The clinician's-currency complement to the ΔELPD ladder. For each clinical endpoint we
patient-level-cross-validate a logistic model and ask whether adding the transdiagnostic map (Arm-B
archetype memberships) to the clinician's reference (DSM-5 arm + severity + baseline outcome + age/sex)
improves **discrimination** (AUC, with a bootstrap CI on the paired gain), **calibration** (Brier),
and **net clinical benefit** (decision-curve analysis). Reference points: DSM-5-only and map-only.
Proper internal validation (K-fold CV), no in-sample optimism. Methods: docs/PROGNOSIS_MODEL.md (M4.6).

    python3 scripts/46_clinical_value.py

Writes results/face/m4/clinical_value.csv, docs/figures/46_{auc,decision_curve,calibration}.png,
reports/46_clinical_value.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from face.prognosis.clinical_value import (  # noqa: E402
    auc,
    brier,
    cv_predict,
    net_benefit,
    paired_auc_delta,
)
from face.prognosis.endpoints import ENDPOINTS, build_endpoints  # noqa: E402
from face.prognosis.frame import load_outcome_config  # noqa: E402
from face.prognosis.reference import (  # noqa: E402
    arm_block,
    armB_block,
    design_for_rung,
    foundation_design,
    severity_column,
)

CONFIG = REPO / "configs" / "m4_outcomes.yaml"
M4 = REPO / "results" / "face" / "m4"
PROFILES = REPO / "results" / "face" / "m2" / "archetype_profiles.csv"
REPORTS = REPO / "reports"
FIGS = REPO / "docs" / "figures"
CGI_BASELINE = "cgi_s__V0"
MODELLED = ("egf_remission", "egf_deterioration", "egf_sustained_impair", "cgi_relapse")
DCA_EPS = ("egf_deterioration", "cgi_relapse")          # adverse endpoints → decision to flag/intervene
COVARS = ("age", "sex", "siteid_city", "arm")


def _designs(sub, spec, *, sev, horizon):
    """Return the four model design matrices for an endpoint, all on the same rows."""
    found, _ = foundation_design(sub, spec, severity_col=sev, horizon=horizon)  # age+sex+severity+baseline
    arm, _ = arm_block(sub)
    archB, _ = armB_block(sub, profiles_path=PROFILES)
    nuis, _ = design_for_rung(sub, spec, "R0", severity_col=sev, horizon=horizon)  # age+sex
    return {
        "DSM-5 only": np.column_stack([nuis, arm]),
        "map only": np.column_stack([nuis, archB]),
        "reference (DSM-5+severity)": np.column_stack([found, arm]),
        "reference + map": np.column_stack([found, arm, archB]),
    }


def main() -> None:
    M4.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    cfg = load_outcome_config(CONFIG)
    horizon = cfg.meta.get("primary_horizon", "V2")
    seed = int(cfg.meta.get("seed", 20260610))
    f = build_endpoints(pd.read_parquet(M4 / "analysis_frame.parquet"))
    ep_by_name = {e.name: e for e in ENDPOINTS}

    rows, preds, dca = [], {}, {}
    for ep in MODELLED:
        e = ep_by_name[ep]
        spec = cfg.by_name(e.outcome)
        sev = severity_column(spec, cgi_baseline_col=CGI_BASELINE)
        sub = f.dropna(subset=[f"ep_{ep}", *COVARS, sev, f"{spec.name}__V0"]).copy()
        y = sub[f"ep_{ep}"].to_numpy("int64")
        designs = _designs(sub, spec, sev=sev, horizon=horizon)
        pmodels = {name: cv_predict(X, y, seed=seed) for name, X in designs.items()}
        preds[ep] = (y, pmodels)
        for name, p in pmodels.items():
            rows.append({"endpoint": ep, "polarity": e.polarity, "n": int(len(y)),
                         "prevalence": round(float(y.mean()), 3), "model": name,
                         "auc": round(auc(y, p), 3), "brier": round(brier(y, p), 3)})
        d, lo, hi, pgt = paired_auc_delta(y, pmodels["reference (DSM-5+severity)"],
                                          pmodels["reference + map"], seed=seed)
        rows.append({"endpoint": ep, "polarity": e.polarity, "n": int(len(y)),
                     "prevalence": round(float(y.mean()), 3), "model": "ΔAUC (map added)",
                     "auc": round(d, 3), "brier": np.nan,
                     "ci": f"[{lo:+.3f},{hi:+.3f}]", "p_gain>0": round(pgt, 3)})
        if ep in DCA_EPS:
            thr = np.linspace(0.02, max(0.5, float(y.mean()) * 3), 40)
            dca[ep] = {"ref": net_benefit(y, pmodels["reference (DSM-5+severity)"], thr),
                       "refmap": net_benefit(y, pmodels["reference + map"], thr)}

    res = pd.DataFrame(rows)
    res.to_csv(M4 / "clinical_value.csv", index=False)
    _fig_auc(res)
    _fig_decision_curves(dca)
    _fig_calibration(preds)
    _report(res, dca)


def _fig_auc(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = ["DSM-5 only", "map only", "reference (DSM-5+severity)", "reference + map"]
    eps = list(dict.fromkeys(res["endpoint"]))
    colors = {"DSM-5 only": "#d6604d", "map only": "#92c5de",
              "reference (DSM-5+severity)": "#888", "reference + map": "#2c7fb8"}
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(eps))
    w = 0.2
    for i, m in enumerate(models):
        vals = [res[(res.endpoint == e) & (res.model == m)]["auc"].iloc[0] for e in eps]
        ax.bar(x + (i - 1.5) * w, vals, w, label=m, color=colors[m])
    ax.axhline(0.5, color="k", lw=0.8, ls="--", label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(eps, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("cross-validated AUC")
    ax.set_ylim(0.45, 0.85)
    ax.set_title("Discrimination of 2-year clinical endpoints (5-fold CV)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "46_auc.png", dpi=130)
    plt.close(fig)


def _fig_decision_curves(dca):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, len(dca), figsize=(6.2 * len(dca), 5), squeeze=False)
    for j, (ep, d) in enumerate(dca.items()):
        a = ax[0][j]
        t = d["ref"]["thresholds"]
        a.plot(t, d["refmap"]["model"], color="#2c7fb8", lw=2, label="reference + map")
        a.plot(t, d["ref"]["model"], color="#888", lw=2, label="reference (DSM-5+severity)")
        a.plot(t, d["ref"]["treat_all"], color="#d6604d", lw=1, ls="--", label="treat all")
        a.plot(t, d["ref"]["treat_none"], color="k", lw=1, ls=":", label="treat none")
        a.set_xlabel("threshold probability")
        a.set_ylabel("net benefit")
        a.set_title(f"Decision curve — {ep}")
        a.set_ylim(min(-0.02, float(np.min(d["ref"]["treat_all"]))), None)
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "46_decision_curve.png", dpi=130)
    plt.close(fig)


def _fig_calibration(preds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    eps = list(preds)
    fig, ax = plt.subplots(1, len(eps), figsize=(3.6 * len(eps), 3.8), squeeze=False)
    for j, ep in enumerate(eps):
        y, pmodels = preds[ep]
        p = pmodels["reference + map"]
        frac, mean_pred = calibration_curve(y, p, n_bins=5, strategy="quantile")
        a = ax[0][j]
        a.plot([0, 1], [0, 1], "k:", lw=1)
        a.plot(mean_pred, frac, "o-", color="#2c7fb8", lw=1.5)
        a.set_title(ep, fontsize=8)
        a.set_xlabel("predicted", fontsize=8)
        if j == 0:
            a.set_ylabel("observed", fontsize=8)
        a.grid(alpha=0.3)
    fig.suptitle("Calibration — reference + map (5-fold CV, quantile bins)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "46_calibration.png", dpi=130)
    plt.close(fig)


def _report(res, dca):
    wide = res[res.model != "ΔAUC (map added)"].pivot_table(
        index=["endpoint", "n", "prevalence"], columns="model", values="auc").reset_index()
    delta = res[res.model == "ΔAUC (map added)"][["endpoint", "auc", "ci", "p_gain>0"]].rename(
        columns={"auc": "dAUC_map"})
    md = [
        "# 46 — M4.6 clinical value: AUC · calibration · net benefit", "",
        "Does adding the transdiagnostic map to the clinician's model change *decisions*? Patient-level "
        "5-fold cross-validated logistic models; the map = Arm-B archetype memberships (⊥G).", "",
        "## Discrimination (cross-validated AUC)", "",
        wide.round(3).to_markdown(index=False), "",
        "Paired AUC gain from adding the map to the reference (bootstrap CI):", "",
        delta.to_markdown(index=False), "",
        "## Net benefit (decision-curve analysis)", "",
        "For the adverse endpoints (flag-for-intervention decisions), the net benefit of "
        "`reference + map` vs `reference` and the treat-all / treat-none defaults across decision "
        "thresholds — `docs/figures/46_decision_curve.png`. A curve above the others over a clinically "
        "plausible threshold band = acting on that model yields more true flags net of false ones.", "",
        "## Read", "",
        "- **AUC**: `reference + map` vs the clinician's `reference` — the ΔAUC + CI says whether the "
        "map adds discrimination over diagnosis+severity+baseline; `map only` vs `DSM-5 only` is the "
        "raw classifier head-to-head.",
        "- **Calibration** (`46_calibration.png`): predicted vs observed risk — usable risks, not just "
        "ranking.",
        "- **Net benefit** translates discrimination into a decision: is the map worth acting on.", "",
        "## Decision for the gate",
        "Confirm the clinical-value verdicts; fold the AUC / net-benefit numbers into "
        "`docs/PROGNOSIS_ATLAS.md §5`, then proceed to the robustness sweep (47).", "",
        "Artifacts: `results/face/m4/clinical_value.csv` · "
        "`docs/figures/46_{auc,decision_curve,calibration}.png`.",
    ]
    (REPORTS / "46_clinical_value.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
