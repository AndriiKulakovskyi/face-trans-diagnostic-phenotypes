# Certification tiers (resolves P0-01)

> **Map of record (read first).** The measurement map is the **8-factor immunometabolic map** (G + 7 specifics;
> immunometabolic a single biology factor; substance orthogonal; 3 earned cross-loadings) with **A = 5
> archetypes** — converged at full N = 9,013 (R-hat 1.03, 0 divergences); canonical
> [`HORSESHOE_ESEM.md`](HORSESHOE_ESEM.md) (see also STATE.md / the OOP findings docs). The tiers below are a
> *wording convention* for how strictly the word "certified" may be used; they were drawn up historically on an
> earlier staged mixed fit and still govern usage. Everything that follows is a usage convention and a record of
> that staged-fit history — not a description of the model of record, which is the 8-factor map above.

The word **"certified"** was overloaded — applied both to the full-N continuous backbone (which passes the
strict gate) and to the joint mixed block (which does not). Use these tiers consistently across the
manuscript, README, docs, and generated reports.

| Tier | What | Gate met | Approved wording |
|---|---|---|---|
| **Full-N certified continuous backbone** | S1/S2 Gaussian block, N = 9,013 | **strict**: R-hat ≤ 1.01, ESS ≥ 400, 0 div, no Heywood, PPC ok | "full-N certified continuous backbone" |
| **Largest-N-documented mixed block** *(historical staged fit)* | the staged-era joint fit (explicit latents for suicidality/developmental/substance), N ≈ 2,000 (1,884 explicit) | R-hat ≤ 1.04, ESS ≥ 112, 0 div, **cross-seed Tucker φ 0.993** | "largest-N documented; point estimates resample-stable, precision provisional" |
| **Internally validated downstream** | M2 strata · M3 temporal · M4 prognosis · M5 treatment | internal validation only | "internally validated" |

## Facts *(historical — staged-fit record)*

- Continuous backbone **passes** the strict gate: `results/face/stage1/diagnostics.json` → R-hat 1.01,
  ESS 1939, 0 div, `certified: true`.
- The staged-era mixed block did **not**: `results/face/stage5/diagnostics.json` → ESS ~112,
  `certified: false`. The code's `certified` flag (`scripts/04_fit.py:45-159`, `s5_certify9.py:131`) was
  **already correctly False** for that mixed block — only the prose over-reached.
- The 2026-06-17 re-fit confirmed the original plain-NB staged fit (`s5_cert9_s1/s2`, R-hat 1.01/1.04) was
  the **reported staged map**; the suicidality hurdle was an opt-in sensitivity (it destabilizes the
  suicidality↔developmental Φ cell).

## Rule

Reserve unqualified **"certified"** for the full-N continuous backbone. Tag the staged-era joint mixed map as
**"largest-N documented / resample-stable, precision provisional"**, and surface the R-hat-1.04 caveat in
the abstract/results, not only Methods. Downstream layers are **"internally validated"**. (Consistent with
the project's own QC guidance: prefer "converged" over "certified" for MCMC convergence.)
