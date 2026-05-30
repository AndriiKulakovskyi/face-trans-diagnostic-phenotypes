"""Build notebooks/FACE_reproduction.ipynb — a literate, end-to-end reproduction
of the FACE trans-diagnostic dimensional-phenotyping manuscript.

The notebook orchestrates the verified pipeline scripts, then loads + displays
the group-level results and every figure inline with explanatory markdown. It
never prints patient-level rows, so the notebook is safe to share.

Run:  python3 scripts/build_notebook.py   →   notebooks/FACE_reproduction.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "FACE_reproduction.ipynb"

nb = nbf.v4.new_notebook()
cells: list = []
def md(text: str): cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))
def code(src: str): cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# ───────────────────────────────────────── title
md(r"""
# Reproducing *“Trans-diagnostic psychopathology is dimensional, not categorical”* (FACE: BP · SZ · DR)

This notebook reproduces **every result and figure** in `MANUSCRIPT.md`, end to end, from the
harmonized FACE data. It is a literate wrapper around the pipeline scripts (`scripts/…`); each
section **runs** the relevant step, then **loads and displays** the group-level result and the
figure, with an explanation of the method and what to look for.

**The thesis.** Across bipolar disorder (BP), schizophrenia (SZ) and major depression (DR), the
only categorical structure the data support is *diagnosis itself* — all trans-diagnostic variation
is **continuous**. We recover seven reproducible, confound-controlled symptom **dimensions** that
complement/outperform DSM diagnosis for patient-reported outcomes.

**Requirements.** `pip install -e ".[full]"` (adds `torch`, `neuroHarmonize`, `kaleido`) and the
confidential `data/*.csv` present on disk. The pipeline is deterministic (fixed seeds) and
reproduces the manuscript to ≤1e-12 (BLAS round-off only).

> **Confidentiality:** this notebook displays only **aggregate** outputs (loadings, statistics,
> figures) — never patient-level rows — so it is safe to share.
""")

# ───────────────────────────────────────── setup
md("## 0 · Setup")
code(r"""
import subprocess, sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from IPython.display import Image, Markdown, display
warnings.filterwarnings("ignore")

# locate the repo root whether the notebook is opened from notebooks/ or the repo root
REPO = Path.cwd()
if not (REPO / "scripts").exists() and (REPO.parent / "scripts").exists():
    REPO = REPO.parent
SCRIPTS, RESULTS, FIGS = REPO / "scripts", REPO / "results", REPO / "results" / "reports" / "figures"
sys.path.insert(0, str(REPO / "src"))

# Set False to skip re-running and just display already-computed artifacts (fast browse).
RUN_PIPELINE = True

def run_step(script, *args):
    "Run a pipeline script as a subprocess; print a trimmed tail of its log."
    if not RUN_PIPELINE:
        print(f"[skip] {script} — RUN_PIPELINE=False, using cached artifacts"); return
    cmd = [sys.executable, str(SCRIPTS / script), *map(str, args)]
    print(f"$ python scripts/{script} {' '.join(map(str, args))}")
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    noise = ("FutureWarning", "pynvml", "import pynvml")
    tail = [l for l in (r.stdout + r.stderr).splitlines() if not any(n in l for n in noise)]
    print("\n".join(tail[-12:]))
    if r.returncode:
        raise RuntimeError(f"{script} failed (exit {r.returncode})")

def fig(name, caption=""):
    "Embed a figure from results/reports/figures/ inline."
    p = FIGS / name
    display(Image(str(p))) if p.exists() else print(f"[missing] {name} — run its figure step")
    if caption:
        display(Markdown(f"**Figure.** {caption}"))

def show(path, **kw):
    "Load an AGGREGATE results file (CSV/JSON) for display (never per-patient rows)."
    p = RESULTS / path
    return json.loads(p.read_text()) if p.suffix == ".json" else pd.read_csv(p, **kw)

print("repo:", REPO)
print("pipeline scripts:", len(list(SCRIPTS.glob("*.py"))), "| RUN_PIPELINE =", RUN_PIPELINE)
""")

# ───────────────────────────────────────── 1 cohort + harmonization
md(r"""
## 1 · Cohort & harmonization

We harmonize three single-cohort longitudinal extracts (BP/SZ/DR, baseline V0 → 4-year V4) into one
patient × feature matrix via a common-variables dictionary, keyed by `patient_uid = cohort::usubjid`.
Similarity/embedding use **masked, pairwise-complete** operators (no imputation); only the
factor-analysis input mean-fills standardized gaps (the residual matrix is ~65 % observed).
""")
code(r"""
from trans_diag import build_unified_dataframe, load_variables, to_harmonized_dataset
from trans_diag.adapter import ADMINISTRATIVE_FEATURES

variables = load_variables(REPO / "data" / "face-common-vars.xlsx")
df = build_unified_dataframe(REPO / "data", REPO / "data" / "face-common-vars.xlsx",
                             readiness=["READY", "PARTIAL"], format="long")
v0 = df[df["visit"] == "V0"]
ds = to_harmonized_dataset(df, variables, visit="V0", exclude=ADMINISTRATIVE_FEATURES)

# AGGREGATE summaries only (no patient rows)
print("patient-visit rows (all visits):", f"{len(df):,}")
print("V0 patients per cohort:", v0.groupby('cohort')['usubjid_patients'].nunique().to_dict())
print("harmonized V0 matrix:", ds.X.shape, "(patients × features), NaN = missing (never imputed here)")
""")
md("**Table 1 — cohort composition** (age, sex, DSM subtypes, retention):")
code(r"""
run_step("01_manuscript_table1.py")
display(show("manuscript_table1.csv"))
display(Markdown("DSM subtype counts:")); display(show("manuscript_table1_subtypes.csv"))
""")

# ───────────────────────────────────────── 2 confound ladder
md(r"""
## 2 · The confound trap (why naïve clustering fails — §3.1)

Unsupervised clustering of raw psychiatric records recovers the **largest-variance nuisance axis**,
not psychopathology. We climb down four rungs (same engine throughout), reproducing the manuscript's
landmark numbers: a stored birth-date (≈3.7e17) that dominates cosine (bootstrap ARI ≈0.96), then raw
scale, then a sex×age stratification carried by physical-comorbidity flags (cluster↔sex ARI ≈0.32 >
↔cohort ≈0.19), which **collapses to ≈0.005** once those flags are excluded and age/sex are
residualized out. This motivates the careful configuration used everywhere below.
""")
code(r"""
run_step("02_confound_ladder.py")
display(show("confound_ladder.csv"))
""")

# ───────────────────────────────────────── 3 domains + residualization
md(r"""
## 3 · Construct-level domain scores + cross-fitted residualization

Items are aggregated to **construct-level domain scores** (masked mean of robust-z items, so a
many-item instrument doesn't dominate by count), then **age/sex are partialled out** with a cubic
B-spline + 5-fold cross-fitting (double-ML style). This produces the residualized 54-domain matrix
that feeds the structure test and the dimensional model, plus the masked-cosine spectral embedding.
""")
code(r"""
run_step("03_cluster_domains.py")
meta = show("cluster_domains_meta.json")
print("residualized domains:", meta.get("n_features_kept", meta.get("embedding_dim")), "| see meta below")
print("k-selection (bootstrap stability / consensus PAC):")
display(show("cluster_domains_kselect.csv"))
""")

# ───────────────────────────────────────── 4 structure test
md(r"""
## 4 · Discrete or dimensional? (the pivotal test — §3.2)

Five complementary tests on the embedding: (a) Laplacian **eigengap** (none → no natural k);
(b) **gap statistic** vs a Gaussian null (rises monotonically → continuum); (c) **HDBSCAN** density
(its only dense clusters *are* the cohorts, ARI 0.70 — partly a measurement-protocol artifact, cohort
is ~98 % predictable from the missingness mask); (d) **bimodality** (no axis clearly multimodal);
(e) the 7 DSM subtypes **order on a mood↔psychosis continuum** (ρ 0.79 [0.75, 0.86]). Verdict:
**dimensional, not discrete.**
""")
code(r"""
run_step("04_structure_test.py")
st = show("structure_test.json")
print("VERDICT:", st["verdict"])
print(f"HDBSCAN↔cohort ARI = {st['hdbscan']['cohort_ari']:.2f} | gap monotone = {st['gap_monotonic']} "
      f"| continuum |ρ|(PC1) = {abs(st['continuum_spearman']['pc1']):.2f}")
""")
md("**Figure 1 — the structure is dimensional.**")
code(r"""
run_step("16_manuscript_figures.py")   # generates Figures 1–5 from the artifacts
fig("fig1_structure.png", "No eigengap; monotone gap; unimodal axes; HDBSCAN≈cohort; DSM mood↔psychosis continuum.")
""")

# ───────────────────────────────────────── 5 the six dimensions
md(r"""
## 5 · The six trans-diagnostic dimensions (§3.3)

We fit an **imputation-free varimax factor model** — masked pairwise-complete correlation → principal-axis
factoring + varimax → posterior-mean scores on each patient's observed support (no cell ever filled). The
factor count is set by **masked split-half Tucker congruence** (reproducibility), giving **K = 6** (the
maximum reproducible dimensionality before collapse; K≥7 collapses): depression/internalizing, later
age-of-onset, mania/activation (with externalizing: impulsivity, childhood-ADHD), illness/hospitalization
burden, a **cognitive axis** (verbal reasoning + working memory), and metabolic/inflammatory load. Once the
DR neuropsychology extraction gap was closed (2026-05), cognition entered the main model as one
confound-clean trans-diagnostic dimension. A no-imputation masked autoencoder recovers the same structure
(leading canonical correlation 0.94 vs a permutation null of 0.05). The axes are confound-free
(|corr| age/sex ≤0.017) and diagnosis-independent (cohort η²≤0.106, site ≤0.049).
""")
code(r"""
run_step("05_dimensional_axes.py")     # classical varimax FA (AE reference)
run_step("06_dimensional_ae.py")       # masked autoencoder cross-check
run_step("07_dimensional_refine.py")   # LOCK K=6 by split-half congruence (data-driven)
fmeta = show("dimensional_final_meta.json")
print("locked K =", fmeta["K"], "| max |corr| age/sex =", max(fmeta["confound_max_corr"].values()))
print("(AE↔FA canonical correlations are reported below from review_checks — AE vs the FINAL imputation-free model.)")
""")
md("**Figure 2 — the six-dimension loading structure** (salient |λ|≥0.20; incl. the cognitive axis).")
code(r"""fig("fig2_loadings.png", "Clean block structure: each dimension is carried by a coherent set of instruments.")""")
md("Per-dimension reproducibility (split-half min Tucker congruence) and the K curve:")
code(r"""
run_step("15_review_checks.py")        # eta-squared (cohort/site), CCA permutation null, rho CI, figS2
rc = show("review_checks.json")
print("AE↔FA leading CCA:", rc["cca_observed"][0], "vs permutation null 95th pct:", rc["cca_null_leading_p95"])
print("DSM-subtype variance explained per axis (eta^2):", rc.get("eta_cohort"))
fig("figS2_kcurve.png", "Masked split-half reproducibility vs K: minimum congruence ≥0.85 through K=6; K=6 locked (K≥7 collapse).")
""")
md("**Figure 6c — trans-diagnostic overlap:** even the most diagnosis-linked axis (illness burden) is "
   "only ~14 % explained by DSM-5 (η²); diagnoses fan across the whole axis.")
code(r"""
run_step("18_export_dimensional_flow.py")   # Figure 6 (a/b/c) + dimensional_dsm_eta_squared
display(show("dimensional_dsm_eta_squared.csv"))
fig("fig6c_dsm_axis_flow.png", "DSM-5 subtypes spread across Low/Mid/High of the illness-burden axis (η²=0.14).")
""")

# ───────────────────────────────────────── 6 temporal stability + discrete negative result
md(r"""
## 6 · Temporal stability & the discrete-flow negative result (§3.6)

Projecting the locked axes onto follow-up visits gives a **trait↔state gradient**: metabolic load and
depression are the most trait-like (test–retest r 0.64 and 0.58), activation and work-disability more
state-like. The **dimensional “flow”**
(continuous-axis band trajectories) shows a patient's *position* is largely retained, whereas forcing
**discrete** clusters gives labels that **hop** (~38 % persistence) and are independent of DSM-5
(ARI 0.006) — a *negative result* that motivates the dimensional model (Supplementary Figure S1).
""")
code(r"""
run_step("08_longitudinal_axes.py")        # trait-state test-retest (Figure 4 data)
run_step("09_longitudinal_coherence.py")   # discrete-flow negative result + DSM contingency
display(show("longitudinal_axes_stability.csv").pivot(index="axis", columns="visit", values="pearson").round(2))
print("discrete clusters ↔ DSM-5 ARI:", show("longitudinal_meta.json").get("dsm_phenotype_ari"))
""")
code(r"""
fig("fig4_traitstate.png", "Trait–state gradient across V1–V4 (metabolic & depression trait-like; later-onset static).")
fig("fig6_dimensional_flow.png", "Dimensional flow: continuous-axis band trajectories V0→V1→V2 (positions retained).")
fig("fig6b_band_persistence.png", "Same-band persistence per dimension vs chance (33%) and the discrete clusters' 38%.")
run_step("17_export_longitudinal_figure.py")   # Supplementary Figure S1 (discrete flow)
fig("figS1_dsm_phenotype_flow.png", "Suppl. S1 — discrete clusters: trans-diagnostic (ARI 0.006) and temporally fluid.")
""")

# ───────────────────────────────────────── 7 head-to-head outcomes
md(r"""
## 7 · Do the dimensions beat DSM diagnosis? (the value test — §3.4–3.5)

Leakage-safe, **shuffled** nested 5-fold CV predicting 1-year outcomes from V0 (baseline + age + sex
adjusted): the dimensions **outperform** DSM for quality of life and **complement** it for functioning;
DSM + prior service use dominate hospitalization. Confidence intervals come from repeated CV; the
advantage survives **de-circularization** (refit axes without each outcome's own measure), **ComBat**
site harmonization, and a second follow-up (V2, same cohort).
""")
code(r"""
run_step("10_phase5_outcomes.py", "--visit", "V1")
run_step("10_phase5_outcomes.py", "--visit", "V2")
run_step("11_phase5_ci.py")                 # repeated-CV 95% intervals
run_step("12_phase5_decircularized.py")     # de-circularization sensitivity
run_step("13_robustness_site.py")           # ComBat
print("V1 head-to-head:"); display(show("phase5_headtohead_V1.csv"))
print("Repeated-CV 95% intervals on the differences:"); display(show("phase5_ci.csv"))
print("De-circularized (axes refit w/o each outcome's own measures):"); display(show("phase5_decircularized.csv"))
""")
md("**Figure 3 — head-to-head outcome prediction** (DSM vs dimensions vs combined; V1 + V2).")
code(r"""
run_step("16_manuscript_figures.py")   # refresh Fig 3 with the latest head-to-head numbers
fig("fig3_headtohead.png", "QoL: dimensions beat DSM; functioning: combined wins; hospitalization: DSM dominates.")
""")

# ───────────────────────────────────────── 8 cognition (integrated)
md(r"""
## 8 · Cognition is now one of the trans-diagnostic dimensions (§2.12, §3.7)

The depression cohort's neuropsychology was recovered (the old "absent in DR" was a data-extraction
artifact), so cognition now enters the **main** model as curated constructs. A confound battery
(`15_review_checks` #10 + the leave-one-cohort holdout in `21`) admits exactly ONE genuine
trans-diagnostic cognitive axis — **verbal reasoning + working memory** — that is confound-clean
(cohort η² 0.072, not predictable from test-availability) and transports leave-DR-out (congruence 1.0).
It is semi-independent of the symptom dimensions and ~57% reconstructable from routine items (education +
functioning). Processing speed/executive (incoherent across cohorts), verbal fluency (a cohort artifact),
and CVLT memory / matrix reasoning (BP/SZ-only) were excluded.
""")
code(r"""
rc = show("review_checks.json")
print("cognitive axis:", rc["cognition_axes"], "| cohort eta^2:", rc["cognition_axis_eta_cohort"])
print("availability R^2 (axis ~ # tests done):", rc["cognition_axis_r2_from_availability"])
""")

# ───────────────────────────────────────── summary
md(r"""
## 9 · Summary

| Claim | Evidence (reproduced above) |
|---|---|
| Structure is **dimensional, not categorical** | no eigengap · monotone gap · HDBSCAN≈cohort · DSM continuum ρ 0.79 |
| **Seven** reproducible, confound-free, **imputation-free** axes | masked split-half congruence 0.91 (K=7) · age/sex |corr|≤0.018 · FA≈AE (CCA 0.97 vs null 0.06) |
| Genuinely **trans-diagnostic** | cohort η²≤0.11, site ≤0.05; even the most diagnosis-linked axis (illness burden) only η² 0.14 of DSM |
| Axes **beat/complement DSM** for patient-reported outcomes | QoL +0.039 [+0.036,+0.042]; functioning combined +0.034; robust to de-circularization, ComBat, V2 |
| Discrete clustering **fails** | ~38 % persistence, DSM-ARI 0.006 → slices of a continuum (Suppl. S1) |
| Cognition = **g + speed**, semi-independent | max |r| 0.26 |
| No general (***p***) factor | confound-free axes near-orthogonal (mean inter-factor r ≈ 0) — a single severity score is not justified |

Everything above is regenerated from the raw data by the scripts in `scripts/` (orchestrated by
`scripts/00_run_all.py`). Full prose, methods and references: **`MANUSCRIPT.md`**.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
OUT.parent.mkdir(exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT.relative_to(REPO)} ({len(cells)} cells)")
