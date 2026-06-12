#!/usr/bin/env python3
"""50 — M5.0 treatment-response inventory (feasibility + circularity + the severity-confound gate).

Before modelling response heterogeneity we inventory: (1) coverage of the raw CGI response signals
(cgi02/03a/03b, cgi01, mars) by visit and cohort; (2) the prevalence of the derived endpoints
(response / therapeutic_effect / resistance / side_effects / low_adherence); (3) the **circularity
audit** — do any response signals double as M1 map indicators?; and (4) the load-bearing
**severity-confound audit** — how entangled is each endpoint with *baseline* severity (CGI-S at V0)?
That last item decides M5: response is regression-to-the-mean-prone (sicker patients have more room to
"improve"), so a "the stratum responds" signal must clear a diagnosis+severity bar. M5.0 quantifies the
bar's burden up front. No model, no imputation. Methods: docs/TREATMENT_MODEL.md (M5.0).

    python3 scripts/50_inventory.py

Writes reports/50_inventory.md (+ 50_signal_coverage.csv, 50_endpoint_prevalence.csv,
50_severity_confound.csv) and docs/figures/50_response_inventory.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from face.data import build_unified_dataframe, load_variables, to_harmonized_dataset  # noqa: E402
from face.treatment import RESPONSE_SIGNALS  # noqa: E402
from face.treatment.endpoints import ENDPOINTS, build_endpoints, load_m5_config  # noqa: E402

XLSX = REPO / "data" / "face-common-vars.xlsx"
MATRIX = REPO / "configs" / "prior_loading_matrix_v3.csv"
CONFIG = REPO / "configs" / "m5_outcomes.yaml"
REPORTS = REPO / "reports"
FIGS = REPO / "docs" / "figures"
COHORTS = ("bp", "sz", "dr")


def _signals(df, variables, visit):
    ds = to_harmonized_dataset(df, variables, visit=visit, normalize=False, apply_skip_logic=True)
    cols = [c for c in RESPONSE_SIGNALS if c in ds.X.columns]
    return ds.X[cols].apply(pd.to_numeric, errors="coerce")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    cfg = load_m5_config(CONFIG)
    horizon = cfg["meta"].get("primary_horizon", "V2")
    repl = cfg["meta"].get("secondary_horizon", "V1")
    mars_low = cfg["meta"].get("mars_low_threshold", 5)
    rcgis = cfg["meta"].get("resistance_cgis_min", 4)

    variables = load_variables(str(XLSX))
    df = build_unified_dataframe("data", str(XLSX), readiness=["READY", "PARTIAL"], format="long")
    S = {v: _signals(df, variables, v) for v in ("V0", repl, horizon)}

    # ---- 1) signal coverage (n non-null per signal × visit, by cohort) ----
    rows = []
    Xh = S[horizon]
    cohh = Xh.index.get_level_values("cohort")
    for sig in RESPONSE_SIGNALS:
        rec = {"signal": sig}
        for v in ("V0", repl, horizon):
            X = S[v]
            rec[f"n_{v}"] = int(X[sig].notna().sum()) if sig in X.columns else 0
        for c in COHORTS:                                  # per-cohort at the horizon (surfaces the DR gap)
            rec[f"{horizon}_{c}"] = (int(pd.to_numeric(Xh.loc[cohh == c, sig], errors="coerce").notna().sum())
                                     if sig in Xh.columns else 0)
        rows.append(rec)
    cov = pd.DataFrame(rows)
    cov.to_csv(REPORTS / "50_signal_coverage.csv", index=False)
    # data-QC: DR lacks the CGI efficacy index; DR MARS is on a different scale than BP/SZ
    dr_cgi = {s: int(cov.loc[cov.signal == s, f"{horizon}_dr"].iloc[0]) for s in ("cgi02", "cgi03a", "cgi03b")}
    mars_mean = {c: float(pd.to_numeric(Xh.loc[cohh == c, "mars"], errors="coerce").mean())
                 for c in COHORTS if "mars" in Xh.columns}

    # ---- 2) endpoint prevalence at the horizon (overall + by cohort) ----
    epH = build_endpoints(S[horizon], mars_low=mars_low, resistance_cgis=rcgis)
    coh = epH.index.get_level_values("cohort")
    prev = []
    for e in ENDPOINTS:
        s = epH[f"ep_{e.name}"].dropna()
        rec = {"endpoint": e.name, "label": e.label, "polarity": e.polarity, "role": e.role,
               "n": int(len(s)), "rate": round(float(s.mean()), 3) if len(s) else np.nan}
        for c in COHORTS:
            sc = epH.loc[coh == c, f"ep_{e.name}"].dropna()
            rec[f"rate_{c}"] = round(float(sc.mean()), 3) if len(sc) else np.nan
        prev.append(rec)
    prev = pd.DataFrame(prev)
    prev.to_csv(REPORTS / "50_endpoint_prevalence.csv", index=False)

    # ---- 3) circularity audit: are response signals M1 map indicators? ----
    matrix_items = set(pd.read_csv(MATRIX)["item"].astype(str))
    circ = {sig: (sig in matrix_items) for sig in RESPONSE_SIGNALS}

    # ---- 4) severity-confound audit: endpoint vs BASELINE CGI-S (cgi01 at V0) ----
    base_cgis = S["V0"]["cgi01"].rename("cgis0") if "cgi01" in S["V0"].columns else None
    conf_rows = []
    if base_cgis is not None:
        j = epH.join(base_cgis, how="inner")
        tert = pd.qcut(j["cgis0"], 3, labels=["low", "mid", "high"], duplicates="drop")
        for e in ENDPOINTS:
            col = f"ep_{e.name}"
            sub = j[[col, "cgis0"]].dropna()
            if len(sub) < 50:
                conf_rows.append({"endpoint": e.name, "n": len(sub), "note": "thin"})
                continue
            r = float(np.corrcoef(sub[col], sub["cgis0"])[0, 1])
            t = j.dropna(subset=[col]).groupby(tert, observed=True)[col].mean()
            conf_rows.append({"endpoint": e.name, "n": int(len(sub)), "polarity": e.polarity,
                              "corr_baseline_cgis": round(r, 3),
                              "rate_lowsev": round(float(t.get("low", np.nan)), 3),
                              "rate_highsev": round(float(t.get("high", np.nan)), 3)})
    conf = pd.DataFrame(conf_rows)
    conf.to_csv(REPORTS / "50_severity_confound.csv", index=False)

    _figure(prev, conf)
    _report(cfg, cov, prev, circ, conf, horizon, repl, mars_low, rcgis, dr_cgi, mars_mean)


def _figure(prev, conf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    names = list(prev["endpoint"])
    colors = ["#1a9850" if p == "good" else "#d73027" for p in prev["polarity"]]
    ax[0].barh(range(len(names)), prev["rate"].values, color=colors)
    ax[0].set_yticks(range(len(names))); ax[0].set_yticklabels(names, fontsize=8)
    ax[0].invert_yaxis(); ax[0].set_xlabel("prevalence at horizon")
    ax[0].set_title("Treatment-response endpoint prevalence (green good · red adverse)")
    for i, (r, n) in enumerate(zip(prev["rate"], prev["n"])):
        ax[0].text(r + 0.01, i, f"{r:.2f} (n={int(n)})", va="center", fontsize=7)
    ax[0].grid(axis="x", alpha=0.3)
    if len(conf) and "rate_lowsev" in conf.columns:
        c = conf.dropna(subset=["rate_lowsev"])
        x = np.arange(len(c)); w = 0.38
        ax[1].bar(x - w / 2, c["rate_lowsev"], w, label="low baseline severity", color="#9ecae1")
        ax[1].bar(x + w / 2, c["rate_highsev"], w, label="high baseline severity", color="#08519c")
        ax[1].set_xticks(x); ax[1].set_xticklabels(c["endpoint"], rotation=30, ha="right", fontsize=8)
        ax[1].set_ylabel("endpoint rate"); ax[1].set_title("Severity confound: endpoint rate by baseline CGI-S")
        ax[1].legend(fontsize=8)
        ax[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "50_response_inventory.png", dpi=130)
    plt.close(fig)


def _report(cfg, cov, prev, circ, conf, horizon, repl, mars_low, rcgis, dr_cgi, mars_mean):
    clean = [s for s, inmap in circ.items() if not inmap]
    inmap = [s for s, inmap in circ.items() if inmap]
    hi_conf = conf[conf.get("corr_baseline_cgis", pd.Series(dtype=float)).abs() >= 0.2]["endpoint"].tolist() \
        if "corr_baseline_cgis" in conf.columns else []
    md = [
        "# 50 — M5.0 treatment-response inventory (feasibility + circularity + severity-confound)", "",
        "What treatment-response signal is available, what endpoints it yields, and how badly each is "
        "confounded with baseline severity (the hazard that decides M5). M5 = response *heterogeneity* "
        "(stratify response to treatment-as-usual), not treatment *selection* — TAU is unobserved. No "
        "model, no imputation.", "",
        "## Response-signal coverage (raw harmonized layer, by visit)", "",
        cov.to_markdown(index=False), "",
        "- The signals are present at follow-up (the modelling visits) but absent from the processed "
        "tables — stage 51 extracts them into the M5 frame.",
        f"- **Data QC (gate catches):** DR has **no CGI efficacy index** at {horizon} "
        f"(cgi02/03a/03b $n={dr_cgi['cgi02']}/{dr_cgi['cgi03a']}/{dr_cgi['cgi03b']}$) → the response / "
        "therapeutic_effect / resistance / side_effects endpoints are **BP/SZ only** (DR generalization "
        "untestable, as for the M4 two-cohort outcomes). And **DR MARS is mis-scaled** "
        f"(mean {mars_mean.get('dr', float('nan')):.1f} vs BP {mars_mean.get('bp', float('nan')):.1f} on "
        "0–10) → the DR low\\_adherence rate is a harmonization artefact; **exclude DR from the adherence "
        "endpoint** pending a data-layer fix.", "",
        f"## Endpoint prevalence at {horizon} (overall + by cohort)", "",
        prev[["endpoint", "polarity", "role", "n", "rate", "rate_bp", "rate_sz", "rate_dr"]].to_markdown(index=False),
        "",
        f"- Definitions: response `cgi02∈{{1,2}}`; therapeutic_effect `cgi03a∈{{1,2}}`; resistance "
        f"`cgi01≥{rcgis} & cgi02≥3`; side_effects `cgi03b≥3`; low_adherence `mars≤{mars_low}`.", "",
        "## Circularity audit — are response signals M1 map indicators?", "",
        f"- **Clean (not map indicators, no overlap):** {', '.join(clean)}.",
        f"- **In the map:** {', '.join(inmap) if inmap else 'none'} — `cgi01` (CGI-S) is the G anchor, "
        "so it enters M5 only as the *severity adjustment* and inside the *resistance* definition (which "
        "is severity-entangled by construction, see below); it is never a credited response predictor.", "",
        "## Severity-confound audit (the make-or-break for Q2)", "",
        "Correlation of each endpoint with **baseline** CGI-S, and its rate in the low- vs high-baseline-"
        "severity tertile. A large gap means baseline severity drives the endpoint — so the map must beat "
        "a diagnosis+severity bar (R2/R3), not raw prevalence.", "",
        (conf.to_markdown(index=False) if len(conf) else "_baseline CGI-S unavailable_"), "",
        f"- **Most severity-confounded** (|corr| ≥ 0.2): {', '.join(hi_conf) if hi_conf else 'none'} — "
        "`resistance` is confounded by design (it contains CGI-S). These set the bar's burden; "
        "tolerability/adherence should be the *least* severity-driven (cleaner map tests).", "",
        "## Decision for the gate",
        "Confirm the endpoint set + the severity-confound profile before building the M5 frame (stage 51). "
        "The beyond-severity gate (Q2) is the milestone's crux; M5.0 shows exactly how much work it must do.", "",
        "Artifacts: `reports/50_{signal_coverage,endpoint_prevalence,severity_confound}.csv` · "
        "`docs/figures/50_response_inventory.png`.",
    ]
    (REPORTS / "50_inventory.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
