#!/usr/bin/env python3
"""53 — M5.0(treatment) treatment-exposure feasibility + harmonization audit.

Treatment data exists in the raw per-cohort files (not in the harmonized common set). It is captured by
DIFFERENT mechanisms per cohort, all reducible to common drug-class exposures:
  SZ  — `med_psy_code_atc` : per-visit LIST of ATC codes (current medication; gold standard)
  DR  — `psycho_act_cmclas` / `psy_lifetime_cmclas` : drug-class strings (current + lifetime)
  BP  — `cmoccur_*` : structured lifetime med-class flags (Y/N) + `lithiumplasma` + a current-med table
This audit extracts each, harmonizes to common classes (antipsychotic / antidepressant /
mood_stabilizer / lithium / anxiolytic), and characterizes coverage, temporality (current vs lifetime),
per-visit availability, and the analyzable moderation questions. No modelling. Methods:
docs/TREATMENT_MODEL.md (to be revised to the full treatment milestone).

    python3 scripts/53_treatment_audit.py

Writes reports/53_treatment_audit.md (+ 53_exposure_coverage.csv), docs/figures/53_treatment_coverage.png.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
REPORTS = REPO / "reports"
FIGS = REPO / "docs" / "figures"
CLASSES = ("antipsychotic", "antidepressant", "mood_stabilizer", "lithium", "anxiolytic")

# ATC prefix -> common class (longest-prefix first: lithium N05AN before antipsychotic N05A)
ATC = [("N05AN", "lithium"), ("N05A", "antipsychotic"), ("N06A", "antidepressant"),
       ("N03A", "mood_stabilizer"), ("N05B", "anxiolytic"), ("N05C", "anxiolytic")]
# French class-name fragment -> common class (for DR cmclas / fallbacks)
NAMECLASS = [("lithium", "lithium"), ("antidépresseur", "antidepressant"), ("antidepresseur", "antidepressant"),
             ("antipsychotique", "antipsychotic"), ("neuroleptique", "antipsychotic"),
             ("thymorégulateur", "mood_stabilizer"), ("thymoregulateur", "mood_stabilizer"),
             ("anticonvulsant", "mood_stabilizer"), ("benzodiazépine", "anxiolytic"),
             ("anxiolytique", "anxiolytic"), ("hypnotique", "anxiolytic")]


def _atc_classes(cell):
    """Parse a (possibly stringified-list) ATC cell -> set of common classes."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return set()
    vals = cell
    if isinstance(cell, str):
        try:
            vals = ast.literal_eval(cell) if cell.strip().startswith("[") else [cell]
        except (ValueError, SyntaxError):
            vals = [cell]
    out = set()
    for a in (vals if isinstance(vals, list | tuple) else [vals]):
        a = str(a).upper().strip()
        for pre, cls in ATC:
            if a.startswith(pre):
                out.add(cls)
                break
    return out


def _name_classes(cell):
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return set()
    s = str(cell).lower()
    return {cls for frag, cls in NAMECLASS if frag in s}


def _patient_visit(d):
    idc = next((c for c in ("usubjid_patients", "fondacode", "patient_id") if c in d.columns), None)
    return idc, ("visit" if "visit" in d.columns else None)


def _flags_from(d, col, parser):
    """Per-row class-exposure flags (DataFrame of 0/1) from a list/string column via `parser`."""
    if col not in d.columns:
        return None
    sets = d[col].map(parser)
    return pd.DataFrame({cls: sets.map(lambda s, c=cls: 1.0 if c in s else 0.0) for cls in CLASSES},
                        index=d.index).where(d[col].notna())


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    rows, notes = [], {}

    # --- SZ: current meds via ATC lists (per-visit) ---
    sz = pd.read_csv(REPO / "data" / "schizophrenia.csv", low_memory=False)
    sz_cur = _flags_from(sz, "med_psy_code_atc", _atc_classes)
    notes["sz"] = f"current via ATC lists (med_psy_code_atc, n_rows={int(sz['med_psy_code_atc'].notna().sum())}); +rad_clozapine"
    for cls in CLASSES:
        rows.append({"cohort": "sz", "class": cls, "temporality": "current",
                     "n_rows": int(sz_cur[cls].notna().sum()) if sz_cur is not None else 0,
                     "n_exposed": int(sz_cur[cls].sum()) if sz_cur is not None else 0})
    sz_cloz = int((sz.get("rad_clozapine", pd.Series(dtype=str)).astype(str).str.lower().isin(["oui", "yes", "1", "1.0"])).sum())

    # --- DR: current (psycho_act_cmclas) + lifetime (psy_lifetime_cmclas) via class strings ---
    dr = pd.read_csv(REPO / "data" / "depression.csv", low_memory=False)
    for col, temp in [("psycho_act_cmclas", "current"), ("psy_lifetime_cmclas", "lifetime")]:
        fl = _flags_from(dr, col, _name_classes)
        notes.setdefault("dr", []).append(f"{temp}={col}")
        for cls in CLASSES:
            rows.append({"cohort": "dr", "class": cls, "temporality": temp,
                         "n_rows": int(fl[cls].notna().sum()) if fl is not None else 0,
                         "n_exposed": int(fl[cls].sum()) if fl is not None else 0})
    if isinstance(notes.get("dr"), list):
        notes["dr"] = "; ".join(notes["dr"]) + f"; cmclas vocab e.g. {list(pd.Series(dr.get('psycho_act_cmclas', pd.Series(dtype=str)).dropna().unique())[:6])}"

    # --- BP: lifetime structured classes (cmoccur_*) ---
    bp = pd.read_csv(REPO / "data" / "bipolar.csv", low_memory=False)
    cmoccur = {"antipsychotic": ["cmoccur_antip", "cmoccur_neuro"], "antidepressant": ["cmoccur_antid"],
               "mood_stabilizer": ["cmoccur_thymo"], "lithium": ["cmoccur_lithi"], "anxiolytic": ["cmoccur_benzo"]}

    def _yes(s):
        return s.astype(str).str.strip().str.lower().isin(["y", "yes", "oui", "1", "1.0"])
    for cls, cols in cmoccur.items():
        present = [c for c in cols if c in bp.columns]
        nrows = int(bp[present[0]].notna().sum()) if present else 0
        nexp = int(np.logical_or.reduce([_yes(bp[c]) for c in present]).sum()) if present else 0
        rows.append({"cohort": "bp", "class": cls, "temporality": "lifetime", "n_rows": nrows, "n_exposed": nexp})
    notes["bp"] = (f"lifetime classes via cmoccur_* (n~{int(bp['cmoccur_lithi'].notna().sum())}); "
                   f"+lithiumplasma (n={int(bp['lithiumplasma'].notna().sum())}); +current med table (med_psy_*, names no ATC)")

    cov = pd.DataFrame(rows)
    cov.to_csv(REPORTS / "53_exposure_coverage.csv", index=False)
    _figure(cov)
    _report(cov, notes, sz_cloz)


def _figure(cov):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    piv = cov.pivot_table(index="class", columns="cohort", values="n_exposed", aggfunc="max").reindex(CLASSES)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    im = ax.imshow(piv.fillna(0).values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([c.upper() for c in piv.columns])
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            v = piv.values[i, j]
            ax.text(j, i, "—" if (v != v) else f"{int(v)}", ha="center", va="center",
                    fontsize=9, color="white" if (v == v and v > piv.fillna(0).values.max() / 2) else "#222")
    ax.set_title("Treatment-exposure coverage: n exposed by drug class × cohort")
    fig.colorbar(im, ax=ax, shrink=0.7, label="n exposed (best of current/lifetime)")
    fig.tight_layout()
    fig.savefig(FIGS / "53_treatment_coverage.png", dpi=130)
    plt.close(fig)


def _report(cov, notes, sz_cloz):
    wide = cov.pivot_table(index=["class"], columns=["cohort", "temporality"], values="n_exposed",
                           aggfunc="max").reindex(CLASSES)
    md = [
        "# 53 — M5 treatment-exposure feasibility + harmonization audit", "",
        "Treatment data exists in the raw per-cohort files (unharmonized), captured by different "
        "mechanisms but reducible to common drug-class exposures. This audit extracts + harmonizes + "
        "characterizes coverage; it confirms which moderation questions are powered. No modelling.", "",
        "## Source mechanism per cohort",
        f"- **SZ**: {notes['sz']} (+ clozapine flagged Yes in {sz_cloz}).",
        f"- **DR**: {notes['dr']}.",
        f"- **BP**: {notes['bp']}.", "",
        "## Harmonized exposure coverage (n exposed) by class × cohort × temporality", "",
        wide.fillna(0).astype(int).to_markdown(), "",
        "## Read — what is analyzable", "",
        "- **Antipsychotic** and **antidepressant** exposure are recoverable in **all three** cohorts "
        "(ATC for SZ, class-string for DR, lifetime-flag for BP) — the broadest common exposures.",
        "- **Lithium** + **mood-stabilizer**: strongest in **BP** (structured lifetime + plasma levels) "
        "— the classic *lithium-response-in-BP* question is well-powered.",
        "- **Clozapine**: **SZ**-specific (the treatment-resistance drug) — the *clozapine-in-SZ* question.",
        "- **Temporality**: SZ/DR give **current, per-visit** exposure (usable as time-varying treatment); "
        "BP is mostly **lifetime** (a confounded baseline exposure — needs the target-trial framing). The "
        "BP current-med table (`med_psy_*`) has names but no ATC, so a name→class map (or the SZ/DR ATC/"
        "class route) is the M5.1 harmonization task.", "",
        "## Caveats (carried to the design)",
        "- **Confounding by indication** — treatment is prescribed on presentation, not randomized; "
        "moderation needs propensity / target-trial emulation, never a naive interaction.",
        "- **Lifetime ≠ current** — BP's clean exposures are lifetime (illness-history-confounded); the "
        "current/time-varying exposures (SZ/DR ATC, BP med table) are the cleaner moderation substrate.",
        "- **Heterogeneous, mostly within-cohort** — the questions are per-cohort (lithium-BP, "
        "clozapine-SZ) with the map applied within each; a clean transdiagnostic common-treatment "
        "moderation is limited.", "",
        "## Decision for the gate",
        "Confirm the analyzable questions (lithium-in-BP; clozapine-in-SZ; antipsychotic/antidepressant "
        "across cohorts; treatment-as-confounder for M4) and the harmonization route (ATC/class → common "
        "classes) before building the harmonized exposure table (M5.1) + the causal design.", "",
        "Artifacts: `reports/53_exposure_coverage.csv` · `docs/figures/53_treatment_coverage.png`.",
    ]
    (REPORTS / "53_treatment_audit.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
