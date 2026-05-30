# TODO — data cleaning: variables to clarify

Issues found while setting sanity bounds in `data/face-common-vars-v2.xlsx` (BP / SZ / DR).
Variable names are CSV columns (BP bilan columns append `_lbstresc`). Canonical unit in parentheses.

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

## ECG intervals — convert milliseconds → seconds during processing
Each column mixes seconds and milliseconds. Rule: **if value > 5 → ÷ 1000**, then apply the bound.
- **Mesure du QT — `qt`** (s): ms in BP 3.6 % / DR 9.0 % of values. After conversion → bound `[0.20, 0.70]`.
- **Mesure du RR — `rr`** (s): ms in BP 1.0 % / DR 9.5 %. After conversion → bound `[0.30, 2.0]`.
- **QT corrigé QTc — `qtc`** (s): ms in BP 2.3 % / DR 9.1 % (SZ already clean). After conversion → bound `[0.25, 0.70]`.

## Categorical encoding — encode BP/SZ text to DR's numeric codes
PERINATALITE: BP/SZ store these as French text; DR already uses numeric codes. Encode BP/SZ **in coherence with DR**, then apply the bound.
- **Naissance préma/terme — `prembrth`**: "Ne sais pas"→0, "Prématurité"→1, "Né à terme"→2, "Post-maturité"→3. Then bound `[0, 3]`.
- **Type de naissance — `naisstyp`**: "Ne sais pas"→0, "Voie basse"→1, "Césarienne"→2. Then bound `[0, 2]`.
- **Hospitalisation néonat — `honeonat`**: "Non"→0, "Oui"→1, "Ne sais pas"→2. Then bound `[0, 2]`. Dictionary codage lists only 0/1 — add code `2 = ne sais pas` (DR already uses it).

SUICIDE: same pattern (BP/SZ text, DR numeric codes). Encode, then bound. **Map "Ne sais pas" / DR code `2` → NaN** (missing, not a category) so cohorts are comparable.
- **Yes/no — `isf01`–`isf05`, `ltsv09`, `css0101`–`css0105`** (DR cols `cssrs01`–`05`): "Oui"→1, "Non"→0. Then bound `[0, 1]`.
- **Yes/no with don't-know — `isf08`, `isf09`**: "Oui"→1, "Non"→0, "Ne sais pas"/`2`→NaN. Then bound `[0, 1]`.
- **Already bounded, comparable as-is** (numeric in all cohorts): attempt counts `isf07`/`isf08a`/`isf09a` `[0,100]`; C-SSRS intensity `css0106`/`0108`/`0109` `[1,5]` and `css0110`/`0111`/`0112` `[0,5]`.
- **Dropped — not comparable:** `isf01a`–`isf05a` (timing): BP/SZ have 4 categories vs DR 3, and SZ codes are shifted 1–3 vs 0–2. _(removed manually)_
- **Pending separate review:** `ltsg07` (asymmetric DK: BP yes/no, DR yes/no/DK 10.6%) and the sparse method/lethality block `ltsv01`–`07` / `ltsg01`–`06`.

SUBSTANCES: BP/SZ free-text vs DR numeric codes.
- **Statut tabagique — `suncf_cigarettes_lt`**: "Non fumeur"→1, "Ex-fumeur"→2, "Fumeur actuel"→3, "Statut inconnu"→NaN (asymmetric — DR has no unknown code). Then bound `[1, 3]`. (DR already coded 1/2/3.)
- _Already bounded — watch:_ pack-years `sudose_cigarettes_lt` `[0,200]` — SZ has an inflated tail (p99≈272 + `9999` sentinel), likely cigs/day mis-entered; `agedebut`/`agefin` `[5,70]`/`[10,85]`, plus a cross-field check **quit age ≥ start age** (BP 4, SZ 2 records fail).

NEUROPSYCHOLOGIE — context variables (`cclin*`): BP/SZ text (Y/N), DR codes. These are **covariates / validity filters, not cognitive features** — encode only to use them as such.
- **Latéralité — `cclin02`**: "Droitier"→1, "Gaucher"→2, "Ambidextre"→3 (DR codes). SZ absent; nominal → low value.
- **Maîtrise français `cclin03` / Absence daltonisme `cclin09` / Absence troubles auditifs `cclin10` / Pas d'ECT `cclin11`**: "Oui"→1, "Non"→0. Use as **exclusion/validity filters** (verbal/visual test validity, ECT memory confound).
- **Temps depuis dernière éval — `cclin04`**: "Moins d'une semaine"→1, "Plus d'une semaine"→2 (admin covariate).
- **Traitement psychotrope — `cclin12`**: "Oui"→1, "Non"→0, "Ne sais pas"/U→NaN. Medication **confound** (BP column absent).

## NEUROPSYCHOLOGIE — feature selection & rescues
Most of this section is **redundant or defective**. Within every test the *percentile* and *note Z*
columns are derived from the raw/standard score — keep the raw/standard, drop the Z/percentile twins.

**Use as trans-diagnostic features (clean, all 3 cohorts) — use the WAIS *standard* score:**
- Similitudes (verbal reasoning), Mémoire des chiffres (working memory), Code (processing speed) standard scores `[1,19]` (+ `valeur standardisée [-3,3]`); IVT index `[40,160]` / percentile `[0,100]`.
- TMT-A & TMT-B raw times (seconds).

**Covariates, NOT features (residualize on these):** age, sex, years of education.

**Rescues (pre-processing):**
- Cap WAIS standard scores to `[1,19]` (SZ Similitudes has a stray `20`).
- BP Mémoire-des-chiffres **raw** (`nbrut`) & **empan endroit** are corrupted (median 2 / span 0–46) → **use the standard score; do not rescue the raw**.
- DR letter-P fluency columns are **shifted by one** (`fv01`=flag, `fv02`=count, `fv03`=Z, `fv04`=percentile…) → remap before any use — *but fluency is a known cohort artifact, so low priority*.
- Z-scores & percentile text-bands: only worth parsing/sign-fixing if a test's raw score is missing — otherwise redundant. If rescued, verify per-cohort **sign convention** (TMT errors-Z is +ve in DR but −ve in SZ) and strip `99`/`inf` sentinels.

**Drop — not useful:**
- All Z-scores & percentiles (redundant + sign/sentinel defects).
- Demographic filters/confounds: laterality, French mastery, colour/hearing, ECT, psychotropic treatment (latter two are med/validity confounds).
- Developmental disorders (`trbapp*`) — BP-only, single cohort.
- Verbal fluency (cohort artifact + DR shifted).
- CVLT Mardi / Reconnaissances / Discriminabilité d' (scale anomalies, e.g. d'→1000).

**2-cohort only — sensitivity analyses, not the main trans-diagnostic features (cohort-aligned missingness risk):**
- CVLT, Test des commissions, fluency Animaux (BP/SZ).
- WAIS Matrices (BP/SZ), Arithmétique (SZ/DR), Symboles (BP/DR), IMT index (SZ/DR); drop IMT `somme` (DR exceeds the 2-subtest max).

## EVALUATION MEDICALE — anchor caveats (2 retained features; 8 DSM specifiers removed)
Section pruned to 2 course/staging variables (present in all 3 cohorts, not DSM-5 criteria); the 8 DSM-5
mood-episode specifiers (episode type ± psychotic, peripartum, seasonal, melancholic/atypical/catatonic,
severity, remission, chronicity) were **removed** — DSM-circular *and* SZ-absent (would pollute the
DSM-vs-dimensional comparison). Both kept vars bounded `[5, 90]` years; data ↔ thesauri verified consistent.
- **Âge premier traitement — `agetrt`** (all 3): comparable years. Caveat: SZ anchors on first **antipsychotic**, BP/DR on first **psychotrope** (any). Construct aligned; SZ tends later/narrower.
- **Âge premier épisode — `agedebutpremier_episode` (BP/DR) / `agepisod` (SZ)** (all 3): comparable scale, but anchor differs **by design** — first **mood** episode (BP/DR) vs first **psychotic** episode (SZ). Underlies the "later-onset" axis → run a **sensitivity check** (axis stability with vs without SZ `agepisod`).
- _Option, not applied:_ DR also has `agediagpremier_episode` (age at first *diagnosed/treated* episode, med 33) which matches SZ's diagnosed-episode anchor better than `agedebutpremier_episode` (first *symptomatic*, med 31) — but BP lacks an equivalent, so switching trades one asymmetry for another.
- _Logic check (clean):_ onset→treatment median gap = 0 yr all cohorts; only ~5% treated >1 yr before recorded onset (recall noise).

## ANTECEDENTS — encode binary flags + rescued family history (13 dropped)
All retained somatic flags are **Y/N/U text (BP/SZ) vs 0/1 codes (DR)** — encode "Oui"→1 / "Non"→0 / "Ne sais pas"/U→NaN, then bound `[0,1]`. DR has no "U" (forced choice). 27 disease flags + `pregnn_rporres` `[0,20]` + menopause/hormonal.
- **Family psychiatric history — `mere_structure` / `pere_structure`** `[0,1]`: thesauri confirm "parent has a psychiatric disorder Y/N" in **all 3** (v2's old "composite" label was wrong, now corrected). Encode Oui→1/Non→0/NSP→NaN.
- **Family suicidality — `mere_suicide` / `pere_suicide`** `[0,2]` (ordinal): "Aucun"→0, "Tentative de suicide"→1, "Suicide abouti"→2, NSP→NaN (DR codes 0/1/2, 3=NSP→NaN).
- **Dropped — not comparable / not usable (13):**
  - `asthme_mhoccur`, `allergie_mhoccur`, `lupus_mhoccur`, `polyarthr_mhoccur` — 3 incompatible encodings: BP numeric 0/1 over all ~21k visit rows, SZ presence-only/no-"No", DR split via `somat_all_specifier_*`. Different construct *and* denominator.
  - `mere_trouble` / `pere_trouble` — BP/SZ multi-category type (Aucun/EDM/Bipolaire/Schizophrène); DR splits into separate `mere_thy`/`mere_schi` columns → structurally different, not harmonizable as one column.
  - 7 free-text/type detail fields (`autneuro/autcardv/dysthyro/toxidermi/inflachro/hvc/genetique _mhmodify`) — descriptive, redundant with their parent Y/N flag.
- _Note:_ `somat_all` kept `[0,1]` but it is a section flag ("any allergic/inflammatory pathology"), redundant with the specific items — consider dropping to avoid double-counting.

## SOIN SUIVI HOSP ARRET TRAVAIL — hospitalization + work leave (3 dropped)
Pruned 8 -> 5. Hospitalization burden is comparable across cohorts (BP unsuffixed column = lifetime, confirmed by separate `_ly` last-year variant; SZ/DR use `_lt`).
- **Âge 1ère hospitalisation — `agedebut_hospitalisation`(BP)/`_first`(SZ)/`_lt`(DR)** `[5,90]`: comparable years; DR has a `0` sentinel to treat as missing.
- **Nb hospitalisations vie — `nboccur_hospitalisation(_lt)`** `[0,100]`: comparable. BP median 0 (39% never hospitalized) vs SZ ~3 is **clinically real** (BP outpatients vs near-universal SZ admission), not a scale artifact.
- **Durée hospitalisations (mois) — `hodur_hospitalisation(_lt)`** `[0,600]`: comparable months.
- **Arrêt travail actuel — `hooccur_arret_travail_actuel`** `[0,1]` (all 3): encode Oui->1/Non->0; **"Non applicable"/NA (not in workforce) -> NaN, NOT 0**. DR codes 0/1/2 (2=NA->NaN).
- **Arrêt longue durée — `hodur_arret_travail_actuel`** `[0,1]` (BP+DR; SZ absent): Y/N, same construct.
- **Dropped — not comparable (3):**
  - `hooccur_hospitalisation_lt` (r225 "déjà hospitalisé") — **DR-only** (BP/SZ absent; for them it must be derived from nb>0).
  - `hooccur_arret_travail` (r227) — **temporal mismatch**: BP thesaurus = general lifetime Y/N/U, DR = "au cours de l'année" Y/N/NA; SZ uses a different `_LY` column not mapped here.
  - `hodur_arret_travail` (r228) — **different time window + scale**: BP "durée totale" (max 208 wk, 2% >52) vs DR "semaines sur l'année" (med 41, 15% >52); also a `-4` invalid value in BP.

## SOCIAL — sociodemographics (3 dropped)
Pruned 7 -> 4. BP/DR use uniform numeric codes; SZ stores several as text and lives in a different tab ('SOCIAL - PERINATALITE').
- **Statut professionnel — `stprof`** `[0,6]`: same grid all 3 (0=sans emploi,1=actif,2=retraité,3=étudiant,4=pension,5=foyer,6=autres). SZ folds lycéen into 3. Comparable.
- **Emploi temps plein — `jobclas`** `[0,1]`: "Oui"/"Non" (BP/SZ) vs 0/1 (DR). Encode Oui->1/Non->0.
- **Mode de vie — `lvsbjind`(BP/DR) / `vie`(SZ)** `[1,8]`: BP/DR numeric 1-8 (seul/parents/foyer/...); **SZ stores TEXT** ("Chez ses parents","Seul",...) -> encode SZ text to the 1-8 scheme before use.
- **Éducation — `edulevel`** `[1,20]`: BP/DR uniform ordinal (CP-CM2=1-5...BAC+n=15-19,Doctorat=20). **SZ is MIXED text/number** (~23% numeric; rest "BAC" etc.) -> parse SZ text to the ordinal, else SZ unusable.
- **Dropped — not comparable (3):**
  - `maristat` — **SZ is missing the 'marié/concubin/pacsé' category entirely** (code 2 = 0 of 1652 in SZ, vs 48% BP / 58% DR). Either not collected or text->code dropped it -> near-perfect cohort discriminator. Not usable until SZ source is re-checked.
  - `jobdur` — **units differ**: BP years (as text bands "entre 10 et 20 ans"), SZ free text, DR months (numeric). Three incompatible formats.
  - `empjob` (INSEE class) — nominal 1-8, **very sparse** (BP n=359, SZ n=46) and BP only has classes 1/4/5; not usable as a comparable feature.

## SITEID ("Numéro du centre principal") — decode via fondacode, not raw siteid
The per-cohort SITEID codebooks are partial and disjoint (BP/SZ list different site subsets; DR has NO codebook), and raw `siteid` is occasionally mislabeled. **Solution: the site code = the 1-2 leading digits of `fondacode`** (the FONDAMENTAL network-wide patient ID) — the SAME numbering scheme in all 3 cohorts.
- Canonical lookup written to **`data/site_lookup.csv`** (site_code -> city + per-cohort presence). 21 sites; codes 1-3,6,10,13 are the shared (all-3) sites.
- Raw `siteid` agrees with the fondacode head 99.2% (BP) / 99.9% (SZ) / 100% (DR); the few mismatches are data-entry errors -> **prefer fondacode-derived code**.
- One real mislabel: SZ `siteid=16` is tagged "CHU Toulouse" but its fondacode says network code **19** (=Toulouse); BP/DR `16`=Besançon. Decoding via fondacode resolves this collision.
- Same city, two hospitals: codes 4 and 15 both = Grenoble (collapse to city). Code 3 = Montpellier (Lapeyronie in BP, Colombière in SZ).
- DR-only sites **17, 18** have no BP/SZ text label yet — assign their city names (network codes are unambiguous).
- Pipeline: `rules.py::_siteid_city` currently warns + falls back to raw numeric (the bug). Register a real mapping that derives the code from fondacode and joins `site_lookup.csv`. Site is used as the ComBat batch / confound (scripts 13, 15, 21), so this matters for harmonization.
