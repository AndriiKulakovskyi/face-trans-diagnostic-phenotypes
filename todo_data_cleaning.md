# TODO — data cleaning: variables to clarify

Bilan biologique (BP / SZ / DR). Issues found while setting sanity bounds in
`data/face-common-vars-v2.xlsx`. Variable names are CSV columns (BP appends `_lbstresc`).
Canonical unit in parentheses.

## Needs clinician decision — left WITHOUT sanity bounds
- **Hématocrite — `hct`** (%): packed-cell volume. Mostly % (median ~40–41) but a subset looks like L/L fractions (0.xx) plus gross high outliers (BP max 3700, DR max 391). Confirm canonical scale (% vs L/L).
- **CCMH / MCHC — `mchc`** (g/dL or g/L): mean corpuscular Hb concentration. Bimodal — median ~34 (g/dL) but ~30% of values ~330–360 (g/L). Confirm and split by scale.

## Unit mixing within a single column — normalize before modelling
- **Hémoglobine — `hgb`** (g/dL): g/dL (median ~14) mixed with g/L entries (~140–160; BP max 464). ~1–4 % are g/L.
- **Glycémie à jeun — `gluc`** (mmol/L): DR mixes mg/dL into mmol/L (DR p99 = 252, max = 488 ≈ 14–27 mmol/L). In-range mg/dL slips (≈90 → "90") can't be auto-detected by a bound.
- **HbA1c — `hba1c`** (%): DR mixes IFCC mmol/mol (~30–130) with NGSP % (median ~5.5).
- **Cholestérol HDL — `hdl`** (mmol/L or g/L): dictionary lists two units; DR shows scale outliers (p99 = 26) + a `33330` sentinel. Confirm mmol/L vs g/L.
- **Cholestérol LDL — `ldl`** (mmol/L or g/L): same as HDL (DR p99 = 30, `33330` sentinel).
- **25-OH Vitamine D — `vitd`** (ng/mL or nmol/L): DR mixes nmol/L with ng/mL (SZ median ~49 ng/mL) plus gross outliers (DR max 26 800 / 63 000). No BP column.
- **Prolactine — `prolctn`** (ng/mL or mIU/L): unit ambiguity — SZ max 24 000 may be mIU/L *or* a real giant adenoma; DR has a `3333333` sentinel.

## Sentinel / placeholder values
- **Recurring DR sentinel `33333` / `333333` / `3333333`**: recurs across DR bilan columns (`hba1c`, `hdl`, `ldl`, `hcg`, `prolctn`, …). Treat as missing, not a value.
- **bHCG — `hcg`** (UI/L): huge dynamic range (BP real pregnancies to ~193 000); DR `333333` is the sentinel, not a titer. Cannot be cleanly bounded.
