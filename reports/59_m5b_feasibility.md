# 59 — M5b feasibility check: what FACE can and cannot support for treatment *selection*

A true **M5b** — treatment **selection** ("which drug for which phenotype"), the precision-psychiatry
payoff M5 could not deliver observationally — needs randomized treatment assignment. This is the
**evidenced** version of "check the FondaMental treatment data": does FACE contain any randomization, and
what stronger design does the existing data already support?

## (1) Randomization / trial-arm: confirmed **absent**

Scanned all three raw cohort CSVs **and** the per-cohort thesauri for `random*`, `trial`/`essai`,
`protocol*`, `allocat*`, `assign*`, `placebo`, `blind`/`aveugle`, `bras`/`groupe` — **none found**. FACE is
an **observational cohort** (FondaMental Centres Experts) by design; the enrolment `arm` is a **DSM-5
subtype**, not an assigned treatment. M5 already confirmed treatment is treatment-as-usual.

> **A randomized M5b is not possible from the FACE data itself.** It requires **external randomized /
> trial-arm data** — a treatment RCT or a randomized sub-study linkable to FACE patients. That is the
> concrete data-team ask (below), not something derivable from what is in hand.

## (2) A *stronger observational* M5b **is** possible now (no new data)

M5 used **baseline (V0) exposure** only. But the per-cohort medication tables are **per-visit with
start/end dates** — richer than M5 used:

| cohort | per-visit med records (V0 / V1 / V2) | date fields | longitudinal? |
|---|---|---|---|
| BP | 4,596 / 2,540 / 1,908 | `med_psy_cmstdtc/cmendtc` | **yes** |
| SZ | 1,889 / 915 / 571 | `med_psy_cmstdtc/cmendtc` | **yes** |
| DR | 387 / 1 / 0 | `psycho_act_*` / `psy_lifetime_*` | **no** (no follow-up Rx) |

So **BP and SZ support a longitudinal, time-varying-treatment design** — medication **switches,
augmentation, discontinuation, and time-on-drug** over V0→V2 → outcome — analyzable with **g-methods**
(e.g. a marginal structural model with time-varying IPTW). That is a genuine step up from M5's
baseline-exposure propensity: it exploits *within-patient* treatment changes and time-varying confounding
control. It is **still observational** (confounding by indication remains; not randomization), so it
sharpens rather than removes the M5 caveat. DR is out (no follow-up medication).

## The data-team ask (precise, evidenced)

1. **For true selection:** is there a **randomized / trial-arm** dataset — a treatment RCT or a randomized
   sub-study — that can be **linked to FACE patients**? (Required; FACE itself has none.)
2. **For the stronger-observational M5b:** the per-visit dose/titration detail already in `med_psy_*` is
   sufficient — no new acquisition needed; confirm the date fields are reliable enough for switch
   detection.

## Recommendation

- **Near-term (no new data):** a longitudinal observational M5b on BP/SZ medication trajectories
  (g-methods) — a real upgrade over M5, still honestly observational.
- **True treatment selection:** contingent on external randomized data (ask #1). Until then, M5's earned
  boundary stands: the map does not demonstrably guide treatment on observational data.
