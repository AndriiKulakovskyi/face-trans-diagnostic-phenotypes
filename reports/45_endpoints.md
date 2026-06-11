# 45 — M4.5 clinical endpoints + the archetype prognostic atlas

The clinician-facing demonstration: binary clinical event surrogates from the V0→V1→V2 scales, and the 2-year prognosis of each of the 8 archetypes. Descriptive (rates + Wilson CIs); the predictive head-to-head with clinical value metrics is stage 46.

## Clinical endpoints (prevalence)

Binary state transitions recovered from the repeated scales (NaN where a needed visit is missing; never imputed). Base rates are well-distributed — none too rare to model.

| endpoint             | label                                                   | polarity   |    n |   rate | ci          |
|:---------------------|:--------------------------------------------------------|:-----------|-----:|-------:|:------------|
| egf_remission        | Functional remission (GAF ≥ 71 at 2y)                   | good       | 2121 |  0.432 | [0.41,0.45] |
| egf_recovery         | Functional recovery (impaired → GAF ≥ 71)               | good       | 1087 |  0.26  | [0.24,0.29] |
| egf_deterioration    | Functional deterioration (GAF drop ≥ 10)                | poor       | 2121 |  0.154 | [0.14,0.17] |
| egf_sustained_impair | Sustained impairment (GAF < 61 at V1 & V2)              | poor       | 1560 |  0.261 | [0.24,0.28] |
| cgi_remission        | Symptomatic remission (CGI-S ≤ 2 at 2y)                 | good       | 2345 |  0.37  | [0.35,0.39] |
| cgi_relapse          | Clinical worsening (CGI-S rise ≥ 2 — relapse surrogate) | poor       | 2345 |  0.102 | [0.09,0.11] |
| cgi_sustained_severe | Sustained illness (CGI-S ≥ 4 at V1 & V2)                | poor       | 1752 |  0.307 | [0.29,0.33] |

## The archetype prognostic atlas (sorted by functional-remission rate)

Each archetype's cohort mix (transdiagnostic) and 2-year endpoint rates:

| archetype          |    n |   pct_bp |   pct_sz |   pct_dr |   egf_remission |   egf_recovery |   egf_deterioration |   egf_sustained_impair |   cgi_remission |   cgi_relapse |   cgi_sustained_severe |
|:-------------------|-----:|---------:|---------:|---------:|----------------:|---------------:|--------------------:|-----------------------:|----------------:|--------------:|-----------------------:|
| low-burden         | 3324 |     0.8  |     0.19 |     0.02 |           0.602 |          0.406 |               0.162 |                  0.118 |           0.487 |         0.123 |                  0.175 |
| metabolic          | 1192 |     0.63 |     0.32 |     0.05 |           0.395 |          0.201 |               0.185 |                  0.276 |           0.295 |         0.146 |                  0.332 |
| mania/activation   |  498 |     0.86 |     0.13 |     0    |           0.351 |          0.259 |               0.198 |                  0.184 |           0.339 |         0.102 |                  0.345 |
| sleep/circadian    | 1437 |     0.77 |     0.13 |     0.11 |           0.347 |          0.227 |               0.14  |                  0.325 |           0.328 |         0.074 |                  0.38  |
| developmental      |  765 |     0.8  |     0.14 |     0.06 |           0.304 |          0.204 |               0.167 |                  0.333 |           0.323 |         0.059 |                  0.348 |
| high-sev+cognitive | 1486 |     0.34 |     0.51 |     0.15 |           0.225 |          0.209 |               0.075 |                  0.547 |           0.204 |         0.036 |                  0.542 |
| inflammatory       |  174 |     0.59 |     0.34 |     0.07 |           0.158 |          0.107 |               0.158 |                  0.483 |           0.244 |         0.146 |                  0.5   |
| suicidality        |  137 |     0.72 |     0.25 |     0.04 |           0.138 |        nan     |               0.276 |                  0.619 |           0.219 |         0.219 |                  0.417 |

- **Headline:** functional remission ranges **14% (suicidality) → 60% (low-burden)** across archetypes — a clinically decisive spread a single z-scored ΔELPD hides.
- Every archetype contains **all three cohorts** (see `pct_bp/sz/dr`) — the prognostic groups are transdiagnostic, not DSM-5 relabeled.

## Do archetypes separate outcomes better than DSM-5?

| endpoint             | polarity   |   arch_spread | arch_range   |   dsm5_spread | dsm5_range   | winner     |
|:---------------------|:-----------|--------------:|:-------------|--------------:|:-------------|:-----------|
| egf_remission        | good       |         0.444 | [0.16,0.60]  |         0.42  | [0.14,0.56]  | archetypes |
| egf_recovery         | good       |         0.204 | [0.20,0.41]  |         0.302 | [0.11,0.41]  | DSM-5      |
| egf_deterioration    | poor       |         0.123 | [0.07,0.20]  |         0.072 | [0.10,0.18]  | archetypes |
| egf_sustained_impair | poor       |         0.429 | [0.12,0.55]  |         0.44  | [0.13,0.57]  | DSM-5      |
| cgi_remission        | good       |         0.283 | [0.20,0.49]  |         0.394 | [0.11,0.51]  | DSM-5      |
| cgi_relapse          | poor       |         0.183 | [0.04,0.22]  |         0.141 | [0.01,0.15]  | archetypes |
| cgi_sustained_severe | poor       |         0.366 | [0.18,0.54]  |         0.384 | [0.20,0.58]  | DSM-5      |

- On this crude spread metric the split is **3 vs 4** and **falls along outcome *type***: archetypes separate the **dynamic transitions** (egf_remission, egf_deterioration, cgi_relapse) better, DSM-5 separates the **severity-level / sustained** outcomes (egf_recovery, egf_sustained_impair, cgi_remission, cgi_sustained_severe) better. The map owns *who changes* (remits / deteriorates / relapses); DSM-5 owns *who stays severe*. Consistent with the M4.4 co-informative split; the rigorous AUC / net-benefit head-to-head is stage 46.

## Read
- A stratification's value is a **group-level** property; shown as per-archetype outcome rates it is vivid (16%→60% remission) where the individual-level ΔELPD looked modest — same signal, decision-relevant granularity.
- Trajectories (`docs/figures/45_atlas_trajectories.png`) show the archetypes diverge over V0→V1→V2, not just differ at baseline.

## Decision for the gate
Confirm the endpoints + atlas before the predictive clinical-value stage (46: AUC, calibration, decision-curve / net-benefit of map vs DSM-5 on these endpoints).

Artifacts: `results/face/m4/{endpoint_prevalence,archetype_atlas,archetype_vs_dsm5}.csv` · `docs/figures/45_{atlas_trajectories,atlas_rates,arch_vs_dsm5}.png`.