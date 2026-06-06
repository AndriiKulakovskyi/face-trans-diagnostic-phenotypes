"""V3 — sleep ↔ affective sensitivity (is sleep a separable dimension or a depression facet?).

The extended model (V3-6) found sleep×affective = 0.68 in the mood cohorts. This decomposes that
coupling at the PSQI sub-item level (pairwise-complete correlations, no imputation), to test
whether it is genuine sleep–depression co-occurrence or PSQI method overlap with depression symptoms.

Verdict logic: if the coupling is concentrated in the SUBJECTIVE / daytime-dysfunction items (which
overlap with depression: fatigue, anhedonia), while OBJECTIVE items (efficiency / duration / latency)
are weakly affect-coupled, then a sleep factor built from objective items is separable from affect.

Aggregate outputs only:
  results/v3/sleep_sensitivity/psqi_affect_items.csv
  docs/figures/v3/sleep_affect_items.png
Run:  python3 scripts/v3/06_sleep_affect_sensitivity.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
from v3.data import build_unified_dataframe, load_variables, to_harmonized_dataset  # noqa: E402

OUT = ROOT / "results" / "v3" / "sleep_sensitivity"
FIG = ROOT / "docs" / "figures" / "v3"
# PSQI sub-items: objective (sleep parameters) vs subjective (overlap with depression)
OBJECTIVE = {"efficiency(11)": "psqi11", "duration(12)": "psqi12", "latency(14)": "psqi14"}
SUBJECTIVE = {"disturbance(13)": "psqi13", "quality(15)": "psqi15", "daytime-dysf(17)": "psqi17"}


def z(s):
    s = pd.to_numeric(s, errors="coerce")
    return (s - s.mean()) / s.std()


def mcorr(a, b):
    m = a.notna() & b.notna()
    return (float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())) if m.sum() > 30 else (np.nan, int(m.sum()))


def main():
    vs = load_variables(str(ROOT / "data" / "face-common-vars.xlsx"))
    df = build_unified_dataframe("data", str(ROOT / "data" / "face-common-vars.xlsx"),
                                 readiness=["READY", "PARTIAL"], format="long")
    X = to_harmonized_dataset(df, vs, visit="V0", normalize=False, apply_skip_logic=True).X
    bpdr = pd.Series(X.index.get_level_values("cohort"), index=X.index).isin(["bp", "dr"])
    aff = pd.concat([z(X[c]) for c in ["madrs", "qidsr120", "staya"] if c in X], axis=1).mean(axis=1, skipna=True)

    rows = []
    for kind, items in [("objective", OBJECTIVE), ("subjective", SUBJECTIVE)]:
        for lab, v in items.items():
            if v in X.columns:
                r, n = mcorr(z(X[v])[bpdr], aff[bpdr])
                rows.append({"item": lab, "kind": kind, "r_affect": round(r, 2), "n": n})
    if "psqi" in X.columns:
        r, n = mcorr(z(X["psqi"])[bpdr], aff[bpdr]); rows.append({"item": "PSQI total", "kind": "total", "r_affect": round(r, 2), "n": n})
    tab = pd.DataFrame(rows)

    def comp(items):
        return pd.concat([z(X[v]) for v in items if v in X.columns], axis=1).mean(axis=1, skipna=True)
    obj, subj = comp(OBJECTIVE.values()), comp(SUBJECTIVE.values())
    r_obj = mcorr(obj[bpdr], aff[bpdr])[0]
    r_subj = mcorr(subj[bpdr], aff[bpdr])[0]
    r_os = mcorr(obj[bpdr], subj[bpdr])[0]

    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "psqi_affect_items.csv", index=False)

    # figure: per-item bar, colored by kind
    t = tab[tab["kind"] != "total"].sort_values("r_affect")
    cmap = {"objective": "#76B7B2", "subjective": "#E15759"}
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.barh(range(len(t)), t["r_affect"], color=[cmap[k] for k in t["kind"]])
    ax.set_yticks(range(len(t))); ax.set_yticklabels(t["item"])
    ax.set_xlabel("pairwise-complete correlation with affective severity (BP/DR)")
    ax.set_title("Sleep↔affect is driven by SUBJECTIVE PSQI items\n"
                 f"objective composite r={r_obj:.2f} · subjective r={r_subj:.2f} (obj×subj {r_os:.2f})", fontsize=10.5)
    ax.axvline(0, color="#999", lw=0.6)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=cmap[k]) for k in ["objective", "subjective"]],
              ["objective (sleep parameters)", "subjective (depression-overlap)"], fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "sleep_affect_items.png", dpi=140); plt.close(fig)

    print(tab.to_string(index=False))
    print(f"\nOBJECTIVE sleep × affect = {r_obj:.2f}   SUBJECTIVE × affect = {r_subj:.2f}   obj×subj = {r_os:.2f}")
    verdict = ("separable — coupling is PSQI method overlap (subjective/daytime items); "
               "an objective-item sleep factor is largely independent of affect"
               if r_obj < 0.4 and r_subj - r_obj > 0.15 else "entangled — coupling is broad across items")
    print("VERDICT:", verdict)
    print("wrote:", (OUT / "psqi_affect_items.csv").relative_to(ROOT), "·", (FIG / "sleep_affect_items.png").relative_to(ROOT))


if __name__ == "__main__":
    main()
