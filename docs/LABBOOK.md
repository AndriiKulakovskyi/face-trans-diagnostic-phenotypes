# LABBOOK — FACE trans-diagnostic study (v2) — research notebook

Chronological trace of the v2 study — what we did, observed, decided, and **why**.
> The v1 notebook (entries E1–E26) is archived at git tag `v1-archive-2026-05-30`.

## V2-1 · Restart on the re-curated dictionary — 2026-05-30
Stopped trusting the v1 common-variables set. Snapshotted the full v1 state (tag
`v1-archive-2026-05-30`, branch `archive/v1-research`) and started branch `v2-study`.
Decision: **re-derive every result from zero** on v2; keep the method code + engine.

## V2-2 · v2 dictionary finalized + cognition reconciled — 2026-05-30
- v2 = **214 usable variables** (subset of v1's 361), with structured sanity bounds + coverage.
- Promoted v2 to the canonical `data/face-common-vars.xlsx` (loader auto-detects v2 → sanity
  bounds + v2 rules + fondacode site); archived v1 dictionary.
- Cognition reconciled to `docs/neuropsy_features.yaml`: 6 primary 3-cohort features (verbal
  reasoning, working memory, processing speed ×2, TMT-A/B), education as a covariate; fixed the
  bogus "mmHg"/"free text" labels; re-curated `domains.py` COGNITIVE_COMPOSITES → 5 constructs.

## V2-3 · QA-driven dictionary corrections — 2026-05-30
Dropped (NOT USABLE): `brthdtc` (date→1e18 artifact, redundant with age), `clozapin` (SZ-specific
treatment marker), `hcg_lbstresc` (pregnancy test, ~0 + 333000 sentinel), `mdq` (absent at BP
yearly), `ltsv03` (DR n=0), and 12 near-zero-variance `*_mhoccur` flags. Fixed within-column unit
mixing for `mchc`/`hct` (g/L ÷10 → g/dL; L/L ×100 → %) via new v2 rules. QA: 190/190 pass, 0 fail.

## V2-4 · Preprocessing debug + type-aware scaling — 2026-05-30
- Fixed the robust-z **explosion**: `prolactin` domain |z|≈106 → ≤5 (log1p heavy-skewed labs + clip ±5).
- `normalize_for_embedding` → **type-aware bounded scaling to [−1, 1]** (binary/ordinal min-max;
  continuous robust-z-clip). All 190 post-processed features land in [−1, 1] (0 out of range).
- Confirmed V0 **within-cohort** missingness (97/190 >25%; pooled 128 was inflated by 31 structural
  2-cohort vars). Decision: **keep masked design, no hard missingness drop.**

## V2-5 · QA report (3 parts) — 2026-05-30
`scripts/qa_harmonization.py` → `results/reports/qa_harmonization.html`:
Part 1 harmonized variables (native scale) + sanity + missingness · Part 2 post-processed
variables (type-aware [−1, 1], all 190 in Part-1 order) · Part 3 aggregated V0 domain scores
(the ~69 model inputs). 190/190 pass.

## Next
- **Phase 4** — dimensional analysis on v2 (checkpoints: structure test, K-selection).
- **Phase 5** — patient stratification on v2 (checkpoint: verdict).
- **Phase 6** — fresh manuscript + re-baselined golden tests + verify.py thresholds.
