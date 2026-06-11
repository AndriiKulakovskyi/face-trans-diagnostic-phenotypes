# 48 — M4.8 consolidation + the M5 hand-off

Closes M4. The consolidated per-outcome verdict and the per-patient prognostic-risk object for M5.

## Consolidated verdict (M4.1–4.7)

| outcome           | verdict                                                 | vs_dsm5                                           | generalization                                               | robust                                                                  |
|:------------------|:--------------------------------------------------------|:--------------------------------------------------|:-------------------------------------------------------------|:------------------------------------------------------------------------|
| EGF (functioning) | PREDICTIVE (functional), robust, complementary to DSM-5 | co-informative (B−A +47, B−C +40)                 | course-dependent: BP/DR yes, SZ null (foundation saturation) | survives IPW + reliability + permutation (p=0.001); weakens dropping BP |
| CGI-S (severity)  | NOT incremental (severity is baseline-determined)       | co-informative, DSM-5-leaning (B−C +35 > B−A +15) | BP only; SZ/DR null                                          | n/a (no headline gain)                                                  |

Full detail (ELPD / β / AUC) in `results/face/m4/prognosis_summary.csv`.

## Per-patient prognostic risk (M5 hand-off) — 2114 patients

Out-of-fold (5-fold CV) `reference + map` predicted probabilities, with each patient's archetype + cohort:

| cohort   |   patient_id | archetype                                    |   p_remission |   p_deterioration |
|:---------|-------------:|:---------------------------------------------|--------------:|------------------:|
| bp       |       100016 | ↑metabolic ↓suicidality ↓developmental_risk  |         0.568 |             0.178 |
| bp       |       100017 | ↓overall_severity ↓sleep ↓developmental_risk |         0.306 |             0.04  |
| bp       |       100052 | ↓overall_severity ↓sleep ↓developmental_risk |         0.812 |             0.118 |
| bp       |       100056 | ↑sleep ↓cognition ↓developmental_risk        |         0.099 |             0.063 |
| bp       |       100068 | ↑cognition ↑overall_severity ↓suicidality    |         0.394 |             0.039 |
| bp       |        10007 | ↑metabolic ↓suicidality ↓developmental_risk  |         0.779 |             0.416 |

- Columns: `cohort, patient_id, archetype, p_remission, p_deterioration`.
- `results/face/m4/prognosis_patient_risk.parquet` — the patient-level object M5 (treatment) consumes: stratum + prognostic risk per patient.

## M4 is complete

- **Map adds robust prognostic value for functioning** beyond diagnosis+severity (metabolic/inflammatory ⊥G; archetypes stratify 14%→60% functional remission), surviving attrition/reliability/permutation.
- **Co-informative with DSM-5** (complements, not replaces) and **course-dependent** (episodic BP/DR, not baseline-saturated SZ).
- **Severity (CGI-S) is autoregression-determined** — the map adds little there.
- Honest limits: scale trajectories not events; internal validity; 2-year horizon. Next: **M5 treatment** (does stratum moderate treatment response?).

Artifacts: `results/face/m4/{prognosis_summary.csv, prognosis_patient_risk.parquet}`. Docs: `docs/PROGNOSIS_{MODEL,FINDINGS,RESULTS,ATLAS}.md`.