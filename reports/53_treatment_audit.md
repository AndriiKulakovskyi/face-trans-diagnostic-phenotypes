# 53 — M5 treatment-exposure feasibility + harmonization audit

Treatment data exists in the raw per-cohort files (unharmonized), captured by different mechanisms but reducible to common drug-class exposures. This audit extracts + harmonizes + characterizes coverage; it confirms which moderation questions are powered. No modelling.

## Source mechanism per cohort
- **SZ**: current via ATC lists (med_psy_code_atc, n_rows=5578); +rad_clozapine (+ clozapine flagged Yes in 511).
- **DR**: current=psycho_act_cmclas; lifetime=psy_lifetime_cmclas; cmclas vocab e.g. ["['TRANQUILLISANTS']", "['AUTRES ANTIDEPRESSEURS', 'TRANQUILLISANTS']", "['ANTIDEPRESSEURS INHIB DE LA RECAPTURE DE LA SEROTONINE ET DE LA NORADRENALINE']", "['ANTIPSYCHOTIQUES ATYPIQUES', 'NORMOTHYMIQUES', 'TRANQUILLISANTS']", "['ANTIPSYCHOTIQUES ATYPIQUES', 'AUTRES ANTIDEPRESSEURS', 'TRANQUILLISANTS']", "['NORMOTHYMIQUES']"].
- **BP**: lifetime classes via cmoccur_* (n~9132); +lithiumplasma (n=4475); +current med table (med_psy_*, names no ATC).

## Harmonized exposure coverage (n exposed) by class × cohort × temporality

| class           |   ('bp', 'lifetime') |   ('dr', 'current') |   ('dr', 'lifetime') |   ('sz', 'current') |
|:----------------|---------------------:|--------------------:|---------------------:|--------------------:|
| antipsychotic   |                 6824 |                 370 |                  188 |                1793 |
| antidepressant  |                 8094 |                1118 |                  999 |                 847 |
| mood_stabilizer |                 6669 |                   0 |                    0 |                 480 |
| lithium         |                 4224 |                   0 |                    0 |                  77 |
| anxiolytic      |                 6356 |                   0 |                    0 |                 817 |

## Read — what is analyzable

- **Antipsychotic** and **antidepressant** exposure are recoverable in **all three** cohorts (ATC for SZ, class-string for DR, lifetime-flag for BP) — the broadest common exposures.
- **Lithium** + **mood-stabilizer**: strongest in **BP** (structured lifetime + plasma levels) — the classic *lithium-response-in-BP* question is well-powered.
- **Clozapine**: **SZ**-specific (the treatment-resistance drug) — the *clozapine-in-SZ* question.
- **Temporality**: SZ/DR give **current, per-visit** exposure (usable as time-varying treatment); BP is mostly **lifetime** (a confounded baseline exposure — needs the target-trial framing). The BP current-med table (`med_psy_*`) has names but no ATC, so a name→class map (or the SZ/DR ATC/class route) is the M5.1 harmonization task.

## Caveats (carried to the design)
- **Confounding by indication** — treatment is prescribed on presentation, not randomized; moderation needs propensity / target-trial emulation, never a naive interaction.
- **Lifetime ≠ current** — BP's clean exposures are lifetime (illness-history-confounded); the current/time-varying exposures (SZ/DR ATC, BP med table) are the cleaner moderation substrate.
- **Heterogeneous, mostly within-cohort** — the questions are per-cohort (lithium-BP, clozapine-SZ) with the map applied within each; a clean transdiagnostic common-treatment moderation is limited.

## Decision for the gate
Confirm the analyzable questions (lithium-in-BP; clozapine-in-SZ; antipsychotic/antidepressant across cohorts; treatment-as-confounder for M4) and the harmonization route (ATC/class → common classes) before building the harmonized exposure table (M5.1) + the causal design.

Artifacts: `reports/53_exposure_coverage.csv` · `docs/figures/53_treatment_coverage.png`.