# M3 — Temporal coherence: per-stage results (dev record)

> Development log for Milestone 3. Paper-facing synthesis: [`TEMPORAL_FINDINGS.md`](TEMPORAL_FINDINGS.md);
> methods: [`TEMPORAL_MODEL.md`](TEMPORAL_MODEL.md). Per-stage reports + figures under `reports/3N_*` and
> `docs/figures/3N_*`. Engine `src/face/temporal/`; pipeline `scripts/30–37`. Updated 2026-06-10.

## Pipeline & engine (reuse-first)

`src/face/temporal/`: `dropout` (G6) · `standardize` (V0 spec) · `panel` (per-visit tables) · `score`-in-34
· `membership` (archetype projection) · `invariance` (G1) · `variance` (G3) · `persistence` (G4). Reused
unchanged from M1/M2: `conditional_gaussian_draws`, `project_explicit_full_n`, `align_ordinals_to_fit`,
`project_to_Z`/`project_draws`, `tucker_phi` + the §6 harness, `prepare`/`prepare_mixed`. Two minimal core
additions to `prepare()` — `emit_moments` (capture the V0 transform) and `visit=` (read a per-visit table) —
both default-off, proven non-disruptive (90 v3 tests green). **36 temporal tests.**

## Stage log

- **30 — inventory (M3.0).** Retention V0 9,013 → V1 4,270 (47%) → V2 2,958 (33%), all cohorts present.
  8/9 axes well-covered at V1/V2; mania thin (2 indicators). **Correction the gate caught:** the V0→V1
  change rate measures *re-administration*, not state (CTQ hits ~0.9 from recall noise) → every axis is
  scored per visit; trait/state deferred to G3.
- **31 — attrition (G6).** Logistic retention ~ V0 coords: **mildly informative** — severity ~neutral
  (OR 0.97), strongest signal cognition (OR 0.83, impaired leave), biology ~flat. IPW weights saved. Fixed
  two data bugs: death-date **sentinel** `1900-01-01` (faked 966 SZ deaths → real ~12); diagnosis-change
  exits are **BP+SZ** (60+17), not BP-only.
- **32 — substrate + V0 spec.** `emit_moments` → `V0StdSpec`; **round-trip QC bit-exact** (apply_spec(V0) =
  prepare().M, maxdiff 0) so V1/V2 score on the frozen scale. Built `baseline_v{1,2}.parquet`.
- **33 — invariance (G1).** S3A backbone, 3 seeds, converged-only φ: **5 invariant + 1 partial** — severity
  0.99, cognition 0.99, metabolic 0.99, sleep 1.00, developmental 0.96 invariant; **inflammatory partial
  0.90** (WBC differential, acute-phase). Reuses §6 via `prepare(visit=)`.
- **34 — score panel (G2).** Continuous analytic + explicit projection per visit; V0 reused from M2.0,
  V1/V2 projected (R-hat 1.04/1.07). **V0 reproduces M2 `patient_strata` at 99.9%.** `prep_visit_mixed`
  bit-exact on V0. `panel_coords.parquet` (16,241 × 108) + `panel_draws.npz`. Bug: V2 `lym=0` → log(0)=−inf
  → `apply_spec` maps non-finite → NaN (out-of-V0-support, not imputed).
- **35 — trait/state (G3).** Marginalized measurement-error random-intercept (visit fixed effects, known
  var plugged). **Two lenses:** population slide (symptoms slide, biology static) + individual ICC (metabolic
  0.93, cognition 0.78 trait; sleep/suicidality/developmental mixed/state; severity 0.66 trait-by-rank).
  Survivorship robust (max |Δ completers| 0.14). developmental "state" = CTQ recall noise.
- **36 — persistence + spine-vs-corner (G4).** Spine moves 34.5% > biology corner 20.2%; §1.4 cell 25.8% vs
  anti-pattern 11.5% (2.2×). Arm-B archetype identity persists 52% (κ 0.27, chance 12.5%, cosine 0.81).
  **G3⟷G4:** core split agrees both ways; ρ=−0.33 diluted by 2 principled exceptions (severity slide,
  developmental recall-noise — G4 robust there).
- **37 — consolidation (M3.7).** `patient_panel.parquet` (16,241 × 117) = panel + trait/state broadcast;
  `37_axis_summary.csv` (per-axis G1/G3/G4 verdict). Durable stratify-on axes for M4: **cognition, metabolic,
  inflammatory**; monitoring axes: severity (spine), suicidality, sleep.

## Verdict

The transdiagnostic map + strata are **temporally coherent** (G1 precondition holds; G3⟷G4 agree on
biology-durable / symptoms-move; G6 survivorship mild). *Stratify on the durable biology, monitor the moving
symptoms.* Persists ≠ predicts → M4.
