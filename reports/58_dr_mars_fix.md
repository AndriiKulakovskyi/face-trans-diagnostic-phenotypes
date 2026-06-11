# 58 — DR MARS harmonization fix (reverse-coding correction)

A data-layer follow-up flagged at M5.0: DR's medication-adherence score (MARS) was on a different
effective scale from BP/SZ, making any cross-cohort adherence comparison — and the M5 `low_adherence`
endpoint for DR — an artifact.

## The bug

`mars` is read from the raw `mars_` column in all three cohorts (it was a plain float identity cast). The
raw distributions:

| cohort | n | min–max | mean | median | shape |
|---|--:|:--:|--:|--:|---|
| BP | 15,476 | 0–10 | 7.38 | 8 | skewed **high** (good adherence) |
| SZ | 4,018 | 0–10 | 6.77 | 7 | skewed **high** |
| DR | 987 | 0–10 | **3.23** | 3 | skewed **low** — the mirror image |

All three are the **same instrument on the same range**: the MARS-10 (10 binary items MARS01–MARS10,
`1:Oui/0:Non`, total `mars_`), confirmed identical in all three per-cohort thesauri. Yet DR's distribution
is the mirror image of BP/SZ.

## The diagnosis — reverse-coding

DR's `mars_` total is summed with the **opposite item polarity**: in BP/SZ higher = better adherence; in
DR higher = worse. The evidence is decisive: reflecting DR (`10 − x`) gives mean **6.77 — matching SZ
exactly (6.77)** and close to BP (7.38), and the reflected *shape* matches BP/SZ (skewed toward good
adherence). A near-exact mirror to SZ is not plausibly genuine low adherence; it is a scoring-direction
difference.

## The fix

A cohort-conditional harmonization rule (`src/face/data/rules.py::harmonize_mars`): reflect DR onto the
common **0–10, higher = better adherence** scale, BP/SZ unchanged.

```python
base = pd.to_numeric(series, errors="coerce").astype("float64")
return (10.0 - base) if cohort == "DR" else base   # BP/SZ == prior identity_cast
```

**Verified end-to-end** (harmonized V0): DR mean **3.2 → 6.28**, now aligned with BP (6.84) and SZ (6.29);
reflection stays in 0–10 (no clipping). Tests: `tests/v3/test_mars_harmonization.py` (4, pass).

## Downstream

- `mars` is a **covariate/outcome**, not an M1 model indicator — so M1–M4 are unaffected.
- **M5**: the M5.0 audit flagged DR `low_adherence` as a MARS artifact and the M5 frame **defensively
  excluded DR from adherence**. With this fix DR's MARS is now valid (higher = better), so DR
  `low_adherence` (MARS ≤ 5) is usable. M5 is locked/merged; the exclusion is harmless and conservative,
  and re-including DR adherence is a small, optional M5 refresh — not re-run here.

## Caveat

This is an **evidence-based inference** of the coding direction (identical instrument + exact mirror to
SZ), not a documented per-cohort scoring key. **Flag for PI confirmation** against the DR source scoring
before any DR-adherence claim is reported.
