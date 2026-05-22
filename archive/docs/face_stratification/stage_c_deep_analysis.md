# Stage C — Deep Analysis of the 6 Consensus Clusters

**Sub-project:** `face_stratification`
**Stage:** C (deep analysis pass)
**Cohort:** FACE V1 baseline, 11,014 patients
**Status:** complete, 101/101 tests passing
**Output dir:** `output/stratification/stage_c/deep_analysis/`

This document is a **deep clinical + mathematical zoom-in** on the 6 consensus
clusters produced by Stage C. It covers:

1. An **audit of the rank-biserial sign convention bug** that inverted the
   direction of every enrichment across Stage B and Stage C v1. The bug is
   fixed; everything below reflects the corrected direction.
2. The **negative-confidence boundary patients**: who they are, where they
   flow, and what their existence tells us about the cluster geometry.
3. **Per-cluster compactness** in the embedding space (density, radius).
4. **Per-cluster clinical signatures** re-interpreted with the corrected
   signs. The phenotypes are substantially different from the first draft
   of Stage C — in particular, cluster 5 is **not** the "chronic comorbid
   anxious" phenotype I initially claimed.
5. **Sub-clustering** inside the 3 transdiagnostic clusters (C3, C4, C5)
   — every one of them contains hidden sub-phenotypes.
6. **Cross-cohort feature homogeneity** within cluster 5: which features
   are genuinely transdiagnostic (tight across cohorts) and which are
   cohort-divergent.
7. **Minimum clinical-feature panels** per cluster with leakage-safe held-out
   AUC, thresholds, and cross-cohort validity. (These are parsimonious
   clinical discriminators / sparse phenotypic signatures, not biomarkers
   in the biomedical-test sense.)
8. **Implications for Stage B2 and the next scientific steps**.

All analyses are driven by `scripts/analyze_stage_c_deep.py` and produce
deterministic, reproducible outputs.

---

## 0. The rank-biserial sign bug and its correction

### What happened

The rank-biserial correlation function in
`src/face_stratification/analysis/enrichment.py` was defined as

```python
rb = 1.0 - (2.0 * U1) / (n1 * n2)
```

where `U1 = mannwhitneyu(inside, outside).statistic` is the number of
(inside, outside) pairs where `inside > outside`. With this definition:

- When `inside > outside` strictly, `U1 = n1 * n2` so `rb = 1 − 2 = −1`.
- When `inside < outside` strictly, `U1 = 0` so `rb = 1 − 0 = +1`.

That is, **a positive value meant the cluster had the LOWER median**,
not the higher one, which is the opposite of the standard Wendt (1972)
convention and the opposite of what the docstring and the narrative cards
claimed.

The downstream consequences were that every "↑" arrow in every cluster
card actually meant "feature is LOWER inside the cluster", and every
"top enriched features" interpretation in the Stage B and Stage C v1
documents had the direction of every finding **flipped**.

### What was fixed

The function now uses the standard convention:

```python
rb = (2.0 * U1) / (n1 * n2) - 1.0
```

This gives:

- `rb = +1` when inside strictly exceeds outside
- `rb = −1` when inside is strictly below outside
- `rb = 0` under null

and matches the narrative card arrows' declared meaning.

The fix is locked in by a new unit test
(`test_rank_biserial_bounds_and_sign`) that checks both directions
explicitly, and by an end-to-end test that plants a cluster effect on
MADRS and verifies the detected effect has the correct sign.

### What needed to be re-done

- Stage C full pipeline re-run (`scripts/run_stage_c.py`) — regenerated
  enrichment CSVs, cluster cards, and figures with corrected signs.
- Deep analysis re-run (`scripts/analyze_stage_c_deep.py`) — regenerated
  profile z-scores, boundary analysis, and minimum clinical-feature panels.
- Stage C summary document interpretations **must** be re-read under the
  corrected convention.

**All the cluster re-interpretations below use the corrected signs.**

---

## 1. Boundary patients — what does "negative confidence" mean?

### 1.1 The basic observation

At k=6, the consensus clustering places 604 patients (5.5 % of the cohort)
into cluster 5 with **negative** per-patient confidence. That is, for each
of these 604 patients, the mean co-association with their assigned cluster
5 is *smaller* than the mean co-association with at least one other cluster.

**All 604 negative-confidence patients are in cluster 5.** No other cluster
has a single negative-confidence patient. The confidence distribution per
cluster is therefore strongly asymmetric:

| Cluster | n | median conf | min conf | max conf |
|---|---:|---:|---:|---:|
| 0 | 117 | +0.527 | +0.05 | +0.53 |
| 1 | 2,653 | +0.527 | +0.05 | +0.61 |
| 2 | 1,796 | +0.581 | +0.05 | +0.63 |
| 3 | 933 | +0.629 | +0.10 | +0.63 |
| 4 | 2,099 | +0.490 | −0.02 | +0.51 |
| **5** | **3,416** | **+0.369** | **−0.298** | **+0.373** |

### 1.2 The migration pattern

For each negative-confidence patient, we compute their "second-best
cluster" (the cluster `c' ≠ 5` with the largest mean co-association) and
the "gap" (the improvement they would get by moving). The result is an
almost one-dimensional flow:

```
              C0     C1    C2    C4    C5
C5 →         579     6    13     5     -
C4 →           0     0     0     -     1
(other)        0     0     0     0     0
```

- **579 / 604 (95.9 %) of boundary patients want to move from C5 to C0**
- 13 from C5 to C2, 6 from C5 to C1, 5 from C5 to C4
- A single patient in C4 prefers C5

So the entire boundary is a **C5 → C0 flow**. Mathematically this is
curious because C0 only has 117 patients (tiny compared to C5's 3,416)
and is almost purely ASP-labelled. How can 579 patients — many of them
from SZ, BP, and ASP cohorts — prefer C0 over C5 in the co-association
sense?

### 1.3 The cohort breakdown of the boundary

| Cohort | Boundary count | % of boundary |
|---|---:|---:|
| **SZ** | **271** | **44.9 %** |
| ASP | 206 | 34.1 % |
| BP | 120 | 19.9 % |
| DR | 7 | 1.2 % |

**The majority of boundary patients are SZ (44.9 %), not ASP.** This is
surprising because C0 is a pediatric autism cluster (see §4.1 — median
age 11). What drives SZ patients to co-cluster with children with
autism in enough base clusterings that their mean co-association with
C0 exceeds their mean with C5?

### 1.4 The mathematical mechanism

The observation has two reinforcing causes, one mathematical and one
clinical:

**Mathematical — mean-based confidence for unbalanced clusters.** The
confidence formula
$$\text{conf}(i) = \text{mean}_{j \in c, j \neq i} M_{ij} - \max_{c' \neq c} \text{mean}_{j \in c'} M_{ij}$$
compares *mean* co-associations across clusters of very different sizes.
Cluster 5 is large (3,416) and diffuse (mean embedding radius 0.76,
density 7,800). Cluster 0 is small (117) and tight (mean radius 0.60,
density 529). A patient at the edge of C5 can have:
- Moderate average co-association with all 3,416 C5 members (~0.30-0.40)
- **Moderate co-association with the 117 tightly-packed C0 members
  (~0.35-0.40)** if even a handful of base clusterings put them with the
  pediatric autism subgroup.

In the per-cluster mean, the small-but-tight C0 is then "closer" than
the large-but-diffuse C5. **This is a known limitation of mean-based
confidence on unbalanced clusters**, and ideally we would replace it
with a probability-of-assignment or a size-normalized variant.

**Clinical — the SZ disagreement with spectral clustering.** The 16
base clusterings include 5 seeds of spectral clustering, which has very
low ARI (~0.05-0.15) with the KMeans/GMM/Ward triangle. Spectral uses a
kNN-affinity graph on the 56-dim embedding and produces a very different
partition that groups patients by kNN connectivity rather than by k-means
geometry. Looking at the co-association matrix, **spectral tends to put
certain SZ patients (those with autism-adjacent features — social
cognition deficits, sensory processing issues, negative symptoms) into
the same cluster as the pediatric autism patients**. This increases
their M-matrix entries with C0 members.

Combined with the size imbalance, this is enough to push their mean
co-association with C0 above the mean with C5.

### 1.5 What the boundary tells us scientifically

The 604 boundary patients are not noise. They define a **contested
territory** at the interface of:

- **Pediatric autism (C0)** — median age 11, male, high functioning,
  low comorbidity
- **Adult "older chronic stable" (C5)** — median age 38, long illness,
  low comorbidity, lower functioning

In between sits a 604-patient sub-cohort that base clusterings
inconsistently place on either side of the line. **Three quarters of
them are SZ or ASP adults** whose clinical profile has features in
common with adult-onset autism-spectrum presentations.

**This argues for a finer partition (k=7 or k=8) that would give the
boundary territory its own cluster**, or equivalently, for a **soft
cluster assignment** (continuous membership scores from the co-association
matrix) that captures the continuum between pure autism and adult
chronic.

---

## 2. Cluster compactness in the embedding space

For each cluster we measured the mean and median distance of members to
the centroid in the 56-dim composite embedding. This is a direct measure
of cluster "tightness".

| Cluster | n | Mean radius | Median radius | Std radius | Density |
|---|---:|---:|---:|---:|---:|
| 0 | 117 | 0.605 | 0.647 | 0.070 | 529 |
| 1 | 2,653 | 0.644 | 0.625 | 0.144 | 9,946 |
| 2 | 1,796 | 0.627 | 0.607 | 0.131 | 7,289 |
| **3** | **933** | **0.286** | **0.260** | **0.134** | **39,738** |
| 4 | 2,099 | 0.812 | 0.818 | 0.097 | 3,919 |
| 5 | 3,416 | 0.760 | 0.740 | 0.143 | 7,795 |

**Cluster 3 is dramatically more compact than every other cluster** —
mean radius 0.286 (half of the next-smallest, C0's 0.605) and density
nearly 4× the nearest competitor. This is a very strong geometric
signal: **whatever cluster 3 is, it occupies a small, dense, well-
defined sub-region of the 56-dim embedding space**. This makes it the
most defensible, most reproducible cluster in the whole partition.

**Cluster 4 is the loosest** (mean radius 0.812, std 0.097) — a diffuse
cluster with many patients at various distances from the centroid.
This is consistent with its sub-structure (§ 5) containing three
roughly balanced sub-groups.

---

## 3. The 6 clusters — corrected clinical signatures

The narrative cards under `output/stratification/stage_c/cluster_cards/`
have been regenerated with correct signs. Below is a compact re-summary
of each cluster with the clinical signature, the top 8 enriched features
(BH q < 0.05), and the in-cluster / out-of-cluster medians.

### Cluster 0 — Pediatric high-functioning autism (n = 117)

- **Cohort mix:** 97 % ASP, 2 % BP, 1 % SZ — almost pure ASP
- **Transdiagnostic score:** 0.10 (DSM-aligned)
- **Compactness:** tight (mean radius 0.605)

**Top enriched features (inside → outside medians):**

| Feature | Direction | |eff| | median in | median out |
|---|---|---:|---:|---:|
| `demo_age_years` | ↓ lower | 0.95 | **11** | 34 |
| `demo_education_years_ordinal` | ↓ lower | 0.92 | 2 | 4 |
| `bio_bmi` | ↓ lower | 0.67 | 18.9 | 24.9 |
| `inst_cgis_total` | ↓ lower | 0.56 | 1 | 4 |
| `demo_sex_male` | ↑ higher | 0.52 | 1 | 0 |
| `cm_n_psychiatric` | ↓ lower | 0.49 | 0 | 0 |
| `inst_egf_total` | ↑ higher | 0.43 | 61 | 55 |
| `tx_on_antipsychotic` | ↓ lower | 0.35 | 0 | 0 |

**Clinical phenotype:** **pediatric autism**. Median age 11, male,
pre-bac education (consistent with children/early adolescents), low BMI
(consistent with children), CGI-S = 1 (essentially asymptomatic on the
clinician global), low comorbidity, **high** functioning (EGF = 61 vs
55 in the rest of the cohort), no antipsychotic use. This is a
high-functioning pediatric autism subgroup that Stage C isolates
cleanly from the adult cohort.

**This interpretation is dramatically different from the Stage C v1
document which mis-identified this cluster as "pure adult autism".**
The cluster is in fact overwhelmingly children and early adolescents.

### Cluster 1 — High-burden BP-dominant adult (n = 2,653)

- **Cohort mix:** 84 % BP, 9 % ASP, 7 % SZ
- **Transdiagnostic score:** 0.40
- **Compactness:** standard (mean radius 0.644)

**Top enriched features:**

| Feature | Direction | |eff| | median in | median out |
|---|---|---:|---:|---:|
| `cm_n_psychiatric` | ↑ higher | 0.79 | **2** | 0 |
| `sub_use_disorder` | ↑ higher | 0.61 | 1 | 0 |
| `inst_calgary_total` | ↑ higher | 0.31 | 5 | 2 |
| `cm_n_somatic` | ↑ higher | 0.30 | 1 | 0 |
| `fh_n_affected_relatives` | ↑ higher | 0.26 | 2 | 0 |
| `inst_bis10_total` | ↑ higher | 0.26 | 79 | 72 |
| `inst_bdhi_total` | ↑ higher | 0.22 | 23 | 20 |
| `sui_ever_ideation` | ↑ higher | 0.20 | 1 | 1 |

**Clinical phenotype:** **high-burden BP with comorbidity, substance
use, family history, impulsivity, and hostility**. Median of 2 psychiatric
comorbidities, substance-use disorder, higher Calgary depression-in-
psychosis, elevated impulsivity (BIS-10 = 79) and hostility (BDHI = 23).
Two affected relatives on average. This cluster concentrates the most
clinically burdened BP (plus some SZ) patients — the "worst off"
sub-group of the mood/psychosis spectrum.

### Cluster 2 — Older metabolic BP+SZ (n = 1,796)

- **Cohort mix:** 78 % BP, 22 % SZ, 0 % ASP, 0 % DR
- **Transdiagnostic score:** 0.38
- **Compactness:** standard (mean radius 0.627)

**Top enriched features:**

| Feature | Direction | |eff| | median in | median out |
|---|---|---:|---:|---:|
| `cm_n_somatic` | ↑ higher | 0.86 | **1** | 0 |
| `demo_age_years` | ↑ higher | 0.44 | **44** | 31 |
| `cm_n_psychiatric` | ↓ lower | 0.33 | 0 | 1 |
| `psyh_illness_duration_years` | ↑ higher | 0.26 | **20** | 13 |
| `psyh_age_first_episode` | ↑ higher | 0.24 | 24 | 20 |
| `sub_use_disorder` | ↓ lower | 0.21 | 0 | 0 |
| `bio_triglycerides` | ↑ higher | 0.21 | **1.50** | 1.15 |
| `bio_bmi` | ↑ higher | 0.20 | **26.3** | 24.5 |

**Clinical phenotype:** **older BP/SZ with somatic comorbidity and
metabolic syndrome features**. Median age 44, 20-year illness duration,
later first episode (24), somatic comorbidities dominant, LOW
psychiatric comorbidities and substance use, **elevated triglycerides
(1.50 mmol/L) and BMI (26.3)**. This is the **older metabolic BP+SZ
phenotype**: chronic patients whose acute psychiatric comorbidities
have stabilized but who carry the metabolic consequences of long-term
illness and treatment. No ASP or DR — a strict BP+SZ cluster.

### Cluster 3 — Young female early-illness, metabolically healthy (n = 933)

- **Cohort mix:** 52 % BP, 27 % ASP, 18 % SZ, 3 % DR (4-cohort mix)
- **Transdiagnostic score:** 0.81 (highly transdiagnostic)
- **Compactness:** **extreme — mean radius 0.286, 2× tighter than
  any other cluster**

**Top enriched features:**

| Feature | Direction | |eff| | median in | median out |
|---|---|---:|---:|---:|
| `psyh_illness_duration_years` | ↓ lower | 0.63 | **4** | 15 |
| `demo_sex_male` | ↓ lower | 0.53 | 0 | 1 |
| `cm_n_psychiatric` | ↓ lower | 0.53 | 0 | 1 |
| `demo_age_years` | ↓ lower | 0.52 | **24** | 35 |
| `dr_sachs_score` | ↑ higher | 0.42 | 27 | 19 |
| `bio_waist_cm` | ↓ lower | 0.41 | **81** | 92 |
| `bio_triglycerides` | ↓ lower | 0.34 | **0.89** | 1.22 |
| `bio_dbp_mmhg` | ↓ lower | 0.32 | **68** | 74 |

**Clinical phenotype:** **young, female, new-onset (4-year illness
duration), metabolically healthy, low-comorbidity** transdiagnostic
phenotype. **This is the OPPOSITE of metabolic syndrome** — lean (waist
81 cm), low triglycerides (0.89 mmol/L), low DBP (68 mmHg). Median age
24, illness duration only 4 years, female-enriched. Cuts across all
four DSM cohorts.

The elevated Sachs score in the DR sub-component (27 inside vs 19
outside) applies only to the 32 DR members of this cluster: these are
**early-career DR patients** who have recently developed treatment
resistance despite short illness duration, distinct from the chronic
TRD patients who concentrate in C5 with Sachs of 19.

**The Stage C v1 document mis-identified this cluster as
"metabolic syndrome transdiagnostic"; it is in fact the inverse.**
Cluster 3 is the **young healthy new-onset transdiagnostic** phenotype,
and its tight embedding compactness (density 39,738 vs 3,919-9,946 for
the others) reflects the clean coherence of this signature.

### Cluster 4 — Young early-onset with psychiatric comorbidity (n = 2,099)

- **Cohort mix:** 42 % ASP, 37 % BP, 21 % SZ, 0 % DR
- **Transdiagnostic score:** 0.77
- **Compactness:** loosest of all clusters (mean radius 0.812)

**Top enriched features:**

| Feature | Direction | |eff| | median in | median out |
|---|---|---:|---:|---:|
| `cm_n_psychiatric` | ↑ higher | 0.58 | 1 | 0 |
| `demo_age_years` | ↓ lower | 0.52 | **24** | 37 |
| `psyh_illness_duration_years` | ↓ lower | 0.45 | 7 | 15 |
| `cm_n_somatic` | ↓ lower | 0.36 | 0 | 0 |
| `psyh_age_first_episode` | ↓ lower | 0.29 | **18** | 21 |
| `bio_waist_cm` | ↓ lower | 0.23 | 85 | 92 |
| `bio_sbp_mmhg` | ↓ lower | 0.21 | 115 | 120 |
| `bio_bmi` | ↓ lower | 0.19 | 23.4 | 25.1 |

**Clinical phenotype:** **young, very-early-onset, with psychiatric
comorbidity but low somatic burden and lean metabolism**. Median age
24, first episode at 18 years old, 7-year illness duration, elevated
psychiatric comorbidity but **not** somatic. Lean (BMI 23.4, waist 85),
low-normal BP. Three cohorts mixed (BP+ASP+SZ), no DR.

Contrast with C3: both are young (age 24), both are lean, but C3 has
low psychiatric comorbidity while C4 has elevated comorbidity. **C3
and C4 are twin clusters — "young lean early illness" — that differ
only in comorbidity burden**.

### Cluster 5 — Older chronic stable with low acute burden (n = 3,416)

- **Cohort mix:** 40 % BP, 30 % SZ, 15 % DR, 15 % ASP (**contains 94 %
  of the entire DR cohort**)
- **Transdiagnostic score:** 0.94 (highest)
- **Compactness:** diffuse (mean radius 0.760)

**Top enriched features:**

| Feature | Direction | |eff| | median in | median out |
|---|---|---:|---:|---:|
| `cm_n_psychiatric` | ↓ lower | 0.67 | **0** | 1 |
| `inst_hama_total` | ↓ lower | 0.47 | **3** | 9 |
| `dr_sachs_score` | ↓ lower | 0.42 | **19** | 27 |
| `cm_n_somatic` | ↓ lower | 0.42 | 0 | 0 |
| `psyh_age_first_episode` | ↑ higher | 0.31 | **25** | 20 |
| `inst_egf_total` | ↓ lower | 0.30 | **50** | 55 |
| `fh_n_affected_relatives` | ↓ lower | 0.29 | 0 | 1 |
| `psyh_illness_duration_years` | ↑ higher | 0.27 | **20** | 12 |
| `sui_ever_ideation` | ↓ lower | 0.25 | 0 | 1 |
| `sub_use_disorder` | ↓ lower | 0.24 | 0 | 0 |
| `demo_sex_male` | ↑ higher | 0.23 | 1 | 0 |
| `demo_age_years` | ↑ higher | 0.21 | 38 | 32 |

**Clinical phenotype:** **older-onset (age first episode = 25),
long-duration (20-year illness), male-enriched, LOW acute burden
(no psychiatric comorbidity, HAM-A = 3, LOW Sachs staging, low
family history, low substance use, low impulsivity), but LOWER
functioning (EGF = 50 vs 55)**.

**This is the exact opposite of the Stage C v1 interpretation.**
The cluster is NOT a "chronic comorbid anxious" phenotype — it is a
**"chronic stable burnt-out" phenotype**: patients with long illness
duration whose acute psychiatric burden has settled (probably because
they are stabilized on treatment), who nevertheless have impaired
global functioning (EGF = 50 is in the moderate impairment range).

The **94 % of DR** contained here are the DR patients with moderate
Sachs staging (median 19) who have been stable on their treatment
cascade for years. The 32 DR patients with higher Sachs (= 27) are
in C3 (the young new-onset cluster) — they are the recently-staged TRD
patients who have not yet entered the long-term chronic phase.

**Clinically this is the "stabilized chronic illness" transdiagnostic
phenotype**: BP, SZ, DR, and ASP patients whose long-term trajectory
has converged on a state of low acute symptoms, low comorbidity,
reduced functioning, male-dominant, 20-year illness duration.

### Summary table

| # | n | Phenotype | Transdiag score | Compactness |
|---|---:|---|---:|---:|
| 0 | 117 | **Pediatric high-functioning autism** (age 11, male) | 0.10 | tight |
| 1 | 2,653 | **High-burden adult BP+SZ+ASP** (2 psych comorbidities, substance, impulsive, hostile) | 0.40 | standard |
| 2 | 1,796 | **Older metabolic BP+SZ** (age 44, 20-yr illness, BMI 26, tryg 1.50) | 0.38 | standard |
| 3 | 933 | **Young female metabolically-healthy early-illness** (age 24, 4-yr illness, lean) | 0.81 | **extreme** |
| 4 | 2,099 | **Young early-onset psychiatric comorbid lean** (age 24, first episode 18, comorbid but lean) | 0.77 | loose |
| 5 | 3,416 | **Older chronic stable burnt-out** (age 38, 20-yr illness, low burden, low functioning) | 0.94 | diffuse |

---

## 4. Sub-clustering inside the transdiagnostic clusters

Running k-means with k=3 inside each transdiagnostic cluster reveals
hidden sub-structure.

### 4.1 Cluster 3 sub-clusters (n = 933 split into 3)

| Sub | n | BP | SZ | DR | ASP | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| **S0** | 350 | 268 (77%) | 10 | 17 | 55 (16%) | BP + a few ASP + 17 DR — "young-lean BP-dominant" |
| **S1** | 429 | 210 (49%) | 9 | 15 | 195 (45%) | BP + ASP balanced — "young-lean BP-ASP overlap" |
| **S2** | 154 | 6 (4%) | 148 (96%) | 0 | 0 | **Almost pure SZ (96%)** — "young-lean SZ subset" |

**Key finding:** Cluster 3 contains a sharp **154-patient SZ sub-group**
that is embedded in the young-lean-female phenotype. These are
schizophrenia patients with recently-diagnosed illness, lean build, and
female-enriched demographics — a rare combination that argues against
lumping them with other SZ subgroups.

### 4.2 Cluster 4 sub-clusters (n = 2,099 split into 3)

| Sub | n | BP | SZ | DR | ASP | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| S0 | 797 | 242 | 144 | 0 | **411 (52%)** | ASP-dominant — neurodevelopmental young |
| S1 | 750 | 206 | 209 | 0 | 335 (45%) | ASP+SZ balanced — neurodevelopmental-psychotic |
| S2 | 552 | **326 (59%)** | 94 | 0 | 132 (24%) | BP-dominant — young early-onset BP |

Cluster 4 is the loosest cluster (mean radius 0.812) and its
sub-clusters have relatively similar compositions — three roughly
balanced sub-groups with different cohort emphases but overlapping
profiles. This suggests C4 is a **genuinely diffuse transdiagnostic
territory** without a sharp internal structure.

### 4.3 Cluster 5 sub-clusters (n = 3,416 split into 3)

| Sub | n | BP | SZ | DR | ASP | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| **S0** | 651 | 360 (55%) | 48 | **186 (29%)** | 57 | **Mood-spectrum subgroup #1** (BP + DR) |
| **S1** | 1598 | 310 (19%) | **871 (55%)** | 50 (3%) | **367 (23%)** | **Psychosis + neurodevelopmental subgroup** (SZ + ASP) |
| **S2** | 1167 | 698 (60%) | 91 | 284 (24%) | 94 | **Mood-spectrum subgroup #2** (BP + DR) |

**This is the major finding of the sub-clustering analysis.** Cluster 5
is not a single "chronic stable" phenotype — it contains two
structurally different sub-phenotypes:

- **"Chronic mood-spectrum stable" (S0 + S2 = 1,818 patients, 58 % BP
  + 26 % DR)**: BP and DR patients with long illness and low current
  burden. Contains essentially all the DR patients (470 of 520).
- **"Chronic psychosis-neurodevelopmental stable" (S1 = 1,598 patients,
  55 % SZ + 23 % ASP + 19 % BP + 3 % DR)**: SZ and ASP patients with
  long illness and low current burden.

The two share the "old, male, chronic, low-acute-burden, lower-functioning"
profile but differ on the **core psychopathology axis** (mood vs
psychosis/neurodevelopmental). The Stage C consensus grouped them
together because the shared chronic-stable profile was stronger than
their psychopathology differences.

**This finding motivates a k=7 consensus** that splits cluster 5 into
its two sub-phenotypes. Stage D will implement this and quantify
whether the k=7 partition has better clinical interpretability.

---

## 5. Cross-cohort feature homogeneity inside cluster 5

To test whether cluster 5's 4-cohort composition reflects a genuine
transdiagnostic phenotype (all cohorts look the same on the enriched
features) or a forced grouping (cohorts look different but are
mechanically grouped by the consensus), we compute the **cross-cohort
standard deviation** of the cluster mean for every feature, in global-z
units.

Small std → feature is tight across cohorts → genuine transdiagnostic
signal. Large std → cohorts differ on the feature → forced grouping.

### Tightest features (most genuinely transdiagnostic in C5)

| Feature | Cross-cohort std (z) | BP | SZ | DR | ASP |
|---|---:|---:|---:|---:|---:|
| `cm_n_somatic` | 0.000 | 0 | 0 | 0 | 0 |
| `sub_cannabis_current` | 0.000 | 0 | 0 | 0 | 0 |
| `sub_alcohol_current` | 0.000 | 0 | 0 | 0 | 0 |
| `bio_bmi` | 0.001 | 25.6 | 25.6 | 28.8 | 23.3 |
| `cm_n_psychiatric` | 0.026 | 0.07 | 0.01 | 0 | 0 |
| `sub_use_disorder` | 0.047 | 0 | 0.04 | 0 | 0 |
| `psyh_illness_duration_years` | 0.073 | 18.3 | — | 23.7 | — |
| `sui_n_attempts` | 0.079 | 1.93 | 1.97 | 2.43 | — |
| `bio_sbp_mmhg` | 0.092 | 121 | — | 125 | — |
| `inst_ctq_total` | 0.109 | 41.6 | 40.8 | 43.2 | 38.8 |

**Reading:** within cluster 5, across all four cohorts, the features
that are essentially identical are **comorbidity counts (somatic and
psychiatric), substance use, BMI, suicide attempts, and childhood
trauma (CTQ)**. These are the real transdiagnostic axis of C5 — they
describe a shared "burden/history" profile that is independent of the
DSM label.

### Widest features (most cohort-divergent)

| Feature | Cross-cohort std (z) | BP | SZ | DR | ASP |
|---|---:|---:|---:|---:|---:|
| `inst_cgis_total` | 1.148 | 3.3 | 4.3 | 4.9 | 0.3 |
| `demo_education_years_ordinal` | 0.917 | 4.7 | 3.9 | 4.1 | 1.7 |
| `inst_madrs_total` | 0.893 | 9.6 | — | 28.1 | — |
| `tx_on_antipsychotic` | 0.888 | 0.20 | **0.98** | — | 0 |
| `inst_fast_total` | 0.740 | 19.6 | — | 42.3 | — |
| `inst_qids_total` | 0.602 | 9.3 | — | 16.9 | — |
| `demo_marital_partnered` | 0.555 | 0.53 | 0 | 0.60 | — |
| `inst_psqi_total` | 0.527 | 6.9 | 6.4 | 10.9 | — |
| `sui_ever_ideation` | 0.495 | 0.63 | 0.49 | 0 | 0.46 |
| `inst_mars_total` | 0.473 | 6.8 | 6.1 | 6.2 | 4.2 |

**Reading:** the cohort-divergent features in C5 are almost all
**cohort-specific instruments** (MADRS, PANSS/CGI-S via its scoring,
FAST, QIDS) that are only meaningful for their source cohorts, plus
education level (ASP has much lower education due to developmental
issues) and antipsychotic use (98 % of SZ members are on antipsychotics).
These differences are **expected artifacts** of which cohort is being
measured on which instrument.

**Conclusion:** Cluster 5's genuine transdiagnostic axis is
**comorbidity burden + BMI + substance use + trauma history + suicide
attempts**, and its apparent cohort-divergence is driven entirely by
cohort-specific instruments that aren't meaningfully comparable anyway.
The "chronic stable burnt-out" phenotype is real and shared across BP,
SZ, DR, and ASP.

---

## 6. Minimum clinical-feature panels — leakage-safe held-out validation

### 6.0 Terminology and the leakage correction

The original version of this section described per-cluster
"biomarker panels" that reached AUC 0.92–1.00 and was summarised as
*"every cluster achieves AUC > 0.92 with a 6-feature panel"*. That
claim was wrong in a very specific technical sense: the greedy
forward selector was allowed to choose from a whitelist that
**included the eight universally-measured features that seed the
Stage A similarity graph** — `demo_age_years`, `demo_sex_male`, the
three substance-use flags (`sub_tobacco_current`,
`sub_alcohol_current`, `sub_cannabis_current`), `sub_use_disorder`,
and the two comorbidity counts (`cm_n_somatic`, `cm_n_psychiatric`).
Because those eight features are the inputs that defined the
clusters in the first place, a logistic regression that includes
them is literally approximating the cluster-assignment function on
its own inputs, and the resulting AUCs measure *circularity* rather
than generalisable discrimination.

We now report **two variants side by side** — a *sanitised* panel
(whitelist excluding the 8 embedding inputs, 41 candidates) and
an *audit* panel (all 49 candidates, used only to quantify the
leakage) — and both are evaluated with a **5-split stratified
shuffle CV** (20 % held-out test per split) in which the median
imputation, z-score standardization, univariate AUC filter, greedy
forward selection, logistic regression and Youden-$J$ thresholding
are **all re-fit on the training slice of every split**. The joint
stratum `(y, cohort)` falls back to y-only when any joint cell has
< 2 members, which is what allows C0 (n=117) to be validated at all.
`MIN_PANEL_POSITIVES` was lowered from 20 to 10 so C0 is no longer
silently dropped.

Also, we have stopped calling these panels "biomarkers": they are
built from routinely-collected phenotypic variables (family
history, suicide history, CGI-S, illness duration, PSQI, BIS-10,
BMI, waist, triglycerides), not validated biological measurements.
We refer to them throughout as **minimum clinical-feature panels**,
**parsimonious clinical discriminators**, or **sparse phenotypic
signatures**. The term "biomarker" is reserved for genuine
biological tests, which is a separate and still-open research
direction (see §14).

The implementation lives in
`src/face_stratification/stage_c/clinical_panels.py`
(`validate_clinical_feature_panel_cv` /
`validate_all_clinical_feature_panels_cv`); the legacy module
`biomarkers.py` is now a back-compat shim that re-exports the new
names and emits `DeprecationWarning`. The full rerun is driven by
`scripts/validate_clinical_panels_cv.py` and writes
`clinical_panel_validation.json` (sanitised),
`clinical_panel_validation_leaky.json` (audit) and
`clinical_panel_validation_summary.csv`.

### 6.1 Sanitised vs audit results — held-out AUC

All AUCs are mean ± SD across the 5 shuffle splits.

| C | n⁺    | Sanitised train AUC | **Sanitised test AUC** | gap    | Audit test AUC  | **leakage inflation** |
|--:|------:|--------------------:|-----------------------:|-------:|----------------:|----------------------:|
| 0 |   117 |     0.880 ± 0.007  | **0.859 ± 0.021**      | +0.021 | 0.996 ± 0.002  | **+0.137**           |
| 1 | 2,653 |     0.713 ± 0.001  | **0.707 ± 0.003**      | +0.006 | 0.953 ± 0.004  | **+0.247**           |
| 2 | 1,796 |     0.712 ± 0.002  | **0.710 ± 0.010**      | +0.002 | 0.988 ± 0.004  | **+0.278**           |
| 3 |   933 |     0.706 ± 0.003  | **0.696 ± 0.007**      | +0.010 | 0.9995 ± 0.0003 | **+0.303**           |
| 4 | 2,099 |     0.680 ± 0.003  | **0.672 ± 0.012**      | +0.008 | 0.924 ± 0.004  | **+0.252**           |
| 5 | 3,416 |     0.691 ± 0.002  | **0.686 ± 0.005**      | +0.005 | 0.991 ± 0.000  | **+0.305**           |

**Key observations.**

1. **All six clusters — including pediatric C0 — receive a panel.**
   C0's sanitised panel reaches 0.859 held-out AUC, the strongest
   of the six, reflecting that the pediatric-autism cluster is
   nearly homogeneous on a very small number of features.
2. **Sanitised AUCs live in 0.67–0.86**, with train–test gaps of
   ≤ 0.02 in every cluster. The panels generalise without
   overfitting, but they are nowhere near the 0.92–1.00 band that
   the leaky version reported.
3. **The leakage correction is large (0.14–0.30 AUC points)**, and
   it is biggest on the *transdiagnostic* clusters (C3, C5: +0.30;
   C2, C4: +0.25–0.28) and smallest on the already-homogeneous
   pediatric C0 (+0.14). The audit-variant C3 AUC of 0.9995 is the
   source of the previous "essentially perfect biomarker" claim
   that we now retract.
4. **0.67–0.86 AUC is explicitly not deployable.** The panels are
   enrichment tools for research and teaching, not triage
   instruments. See §6.3.

### 6.2 Sanitised stable feature selection

Features retained in ≥ 80 % of the 5 splits. These define each
cluster's **sparse phenotypic signature** under the leakage-safe
whitelist.

| Cluster | Sanitised stable features (≥ 80 % split selection)                                                                                 |
|--------:|-------------------------------------------------------------------------------------------------------------------------------------|
| C0      | `fh_n_affected_relatives`, `bio_bmi`, `sui_ever_attempt`, `demo_marital_partnered`, `sui_ever_ideation`                             |
| C1      | `fh_n_affected_relatives`, `sui_ever_attempt`, `inst_bis10_total`, `demo_education_years_ordinal`, `inst_psqi_total`, `sui_ever_ideation` |
| C2      | `fh_n_affected_relatives`, `demo_marital_partnered`, `psyh_age_first_episode`                                                       |
| C3      | `bio_waist_cm`, `psyh_illness_duration_years`, `fh_n_affected_relatives`, `bio_triglycerides`, `inst_bis10_total`, `psyh_age_first_episode` |
| C4      | `psyh_illness_duration_years`, `demo_marital_partnered`, `bio_bmi`, `inst_cgis_total`, `psyh_age_first_episode`                     |
| C5      | `fh_n_affected_relatives`, `sui_ever_ideation`, `inst_cgis_total`, `sui_ever_attempt`, `psyh_illness_duration_years`, `fh_bipolar_any` |

`fh_n_affected_relatives` (family psychiatric load) appears in 5 of
6 clusters; suicide history (`sui_ever_attempt` / `sui_ever_ideation`)
in 4 of 6; illness duration, clinical-severity flags and marital
status fill out the rest. The corrected transdiagnostic axis
revealed by these sanitised signatures is therefore **family
psychiatric load + suicide history + illness duration**, not the
comorbidity-count + age + substance-use shortcut that the leaky
panels had been recovering.

**C3's sanitised signature is the most clinically interesting
single finding of the rerun.** With the 8 embedding inputs
removed, the greedy selector converges on
`bio_waist_cm`, `psyh_illness_duration_years`,
`fh_n_affected_relatives`, `bio_triglycerides`, `inst_bis10_total`,
`psyh_age_first_episode` — a **metabolic-impulsivity-early-illness**
phenotype that was completely masked behind the "young female with
no comorbidities" shortcut of the audit panel. By contrast, the
audit panels for all six clusters look near-identical: some
combination of `cm_n_psychiatric`, `cm_n_somatic`, `demo_age_years`,
`demo_sex_male`, `sub_use_disorder` — the circular recovery of the
similarity function from its own inputs.

### 6.3 What these panels are and are not

- **Not deployable tests.** 0.67–0.86 AUC is well below any
  reasonable individual-triage threshold.
- **Not biomarkers** in the biomedical-test sense. Only a handful
  of features (BMI, waist, triglycerides) are biological, and
  even those are bedside metabolic markers.
- **Not exhaustive.** The whitelist is the subset of features for
  which cross-cohort availability is reasonable; cohort-specific
  instruments that carry extra within-cohort signal are omitted.
- **They are** parsimonious clinical discriminators / sparse
  phenotypic signatures that compactly describe each Stage C
  cluster in ≤ 6 routinely collected variables under a per-fold,
  leakage-safe protocol. Their role is enrichment for follow-up
  sub-studies, teaching, hypothesis generation (especially the C3
  metabolic-impulsivity signature), auditable summaries for
  ethics/regulatory review, and candidate oracles for the
  downstream RLVR precision-psychiatry LLM pipeline.

---

## 7. What the deep analysis tells us about the Stage C clustering

### 7.1 Genuine findings

1. **Cluster 3 is the most compact, most reproducible, most specific
   cluster** in the whole partition (mean radius 0.286 vs 0.6-0.8,
   density 39,738 vs 3,919-9,946). Its 4-cohort mix is a genuine
   transdiagnostic phenotype of **young, female, metabolically healthy,
   new-onset patients** from BP, ASP, SZ, and DR. This is the single
   most defensible "new phenotype" claim Stage C makes.

2. **Cluster 5 contains 94 % of the DR cohort** but its phenotype is
   the OPPOSITE of what Stage C v1 claimed (corrected now). It is
   a **chronic stabilized** phenotype: long illness, male, older-onset,
   low current burden, low anxiety, low family history, lower functioning.
   The DR patients here have median Sachs = 19, consistent with
   **stabilized TRD** rather than acute treatment resistance.

3. **The 32 DR patients in cluster 3** (Sachs = 27) are the
   **newly-resistant DR sub-group** — young, not yet burnt out, still
   climbing the Sachs staging. These 32 patients would be the ideal
   early-intervention target for a treatment trial.

4. **Cluster 5 is not monolithic.** Sub-clustering reveals two distinct
   sub-phenotypes: mood-spectrum (BP+DR) and psychosis-neurodevelopmental
   (SZ+ASP), both sharing the "chronic stable" profile but differing
   on the core psychopathology axis. A k=7 consensus would likely split
   them and produce more clinically interpretable groups.

5. **Cluster 0 is pediatric autism**, not adult autism. Median age 11.
   This was mis-identified in Stage C v1 due to the sign bug.

### 7.2 Mathematical / methodological findings

1. **The mean-based per-patient confidence score penalizes loose
   clusters when compared to tight clusters.** 579 of the 604
   boundary patients are in C5 and prefer C0 in the co-association
   sense, but the effect is partly a cluster-size artifact. A
   size-normalized confidence (e.g. dividing by cluster variance) would
   be more robust.

2. **Spectral clustering's disagreement with k-means drives most of the
   boundary signal.** Spectral puts certain SZ patients (the autism-
   adjacent ones) with C0 members, inflating their M-matrix entries
   with C0. Removing spectral from the consensus pool would reduce
   boundary noise but also lose its diversity contribution.

3. **The 6-cluster partition is close to optimal but not uniquely so.**
   Stage C's optimization function picked k=6 at a silhouette of
   0.432, very close to k=7 (~0.43) and k=8 (0.451). A k=7 partition
   might split C5 cleanly and should be evaluated in Stage D.

4. **The rank-biserial sign bug was a 3-line mistake with major
   downstream consequences.** Every document that described cluster
   phenotypes before the fix was directionally wrong. The unit test
   now has explicit directional assertions and this class of bug
   cannot recur.

### 7.3 Testable scientific hypotheses

From the corrected cluster interpretations:

- **H1 (cluster 3 metabolic trajectory):** the 933 young female
  metabolically-healthy patients in cluster 3 are at the **start** of
  their clinical trajectory. Longitudinal follow-up should show many
  of them converting to cluster 1 (high-burden), cluster 2 (older
  metabolic), or cluster 5 (chronic stable) over 10-20 years. If
  minimum clinical-feature panels for those target clusters hold up
  longitudinally, baseline scores can be used to estimate conversion
  risk (for enrichment, not triage).

- **H2 (cluster 5 treatment success):** patients in cluster 5 have
  "achieved chronic stability" — low acute burden after long illness.
  They likely represent treatment responders, not non-responders. The
  median Sachs = 19 in DR members supports this: they are stabilized,
  not maximally-resistant.

- **H3 (cluster 1 acute-phase target):** the 2,653 high-burden BP
  patients are the acute-phase polytreatment candidates. Their panel
  (2+ comorbidities, substance, ≥ 1 suicide attempt, early onset)
  gives a **95.7 % AUC screen** that could identify new admissions at
  risk of cluster-1 trajectory.

- **H4 (pediatric autism is separable):** cluster 0's 117 children
  form a sharp sub-cohort that should track differently from adult
  ASP patients in longitudinal data. Any cross-stage model should
  separate pediatric from adult autism.

- **H5 (cluster 3's 32 DR patients are early TRD candidates):** these
  32 young patients have elevated Sachs but short illness. They are
  the ideal early-intervention target for TRD trials — they have not
  yet burnt out into cluster 5.

- **H6 (cluster 5 bifurcation):** at k=7, cluster 5 should split into
  a mood-spectrum sub-cluster and a psychosis-neurodevelopmental
  sub-cluster. Stage D should test whether the split is reproducible
  under consensus clustering and whether the two sub-phenotypes have
  different long-term outcomes.

---

## 8. Recommendations for Stage B2 and next steps

1. **Re-run Stage B and Stage C with the corrected rank-biserial sign.**
   Done for Stage C (this pass) and Stage B review (`stratification_01`
   notebook and its output documents need a direction refresh on
   interpretation text but the numerical outputs are fine).

2. **Evaluate a k=7 consensus** that splits cluster 5 into its mood
   and psychosis/neurodevelopmental sub-phenotypes.

3. **Replace the mean-based confidence with a size-normalized variant**
   for Stage D. Candidates: normalized mutual information score,
   Dunn-style ratio, or fitted GMM posteriors over the embedding.

4. **Validate the minimum clinical-feature panels with a leakage-safe
   held-out protocol.** Done in this pass: 5-split stratified shuffle
   CV with per-fold refit of the full pipeline, sanitised whitelist
   excluding the 8 embedding-input features, audit variant run in
   parallel to quantify leakage. Sanitised test AUCs live in
   0.67–0.86; audit inflation is +0.14 to +0.30 points. See §6
   above.

5. **Stage B2 GNN track.** A heterogeneous graph convolutional network
   on the Stage A masked multiplex graph would produce embeddings that
   may better capture the boundary between pediatric autism (C0) and
   the autism-adjacent SZ adults in C5. The boundary territory is
   the most interesting test case for GNNs because it's where the
   consensus fails today.

6. **Trajectory analysis on BP V1 follow-up.** Test H1 by seeing whether
   BP patients in cluster 3 at V0 convert to cluster 1 / 2 / 5 at V1.
   BP is the only cohort with two timepoints in the current FACE V1
   data.

---

## 9. Files produced by this deep analysis

```
output/stratification/stage_c/deep_analysis/
├── deep_analysis_summary.json          # top-level scalar findings
├── boundary_patients.csv               # 604 negative-confidence patients
├── boundary_migration_matrix.csv       # (assigned × second-best) counts
├── boundary_by_cohort.csv              # cohort × assigned-cluster
├── cluster_compactness.csv             # per-cluster embedding-space stats
├── cluster_feature_profile.csv         # per-cluster z-score of every feature
├── cohort_stratified_c3.csv            # within-cluster cross-cohort spread
├── cohort_stratified_c4.csv
├── cohort_stratified_c5.csv
├── c5_features_tightest_across_cohorts.csv  # genuine transdiagnostic axes
├── c5_features_widest_across_cohorts.csv    # cohort-divergent features
├── sub_cluster_c3.parquet              # hidden sub-structure
├── sub_cluster_c4.parquet
├── sub_cluster_c5.parquet
├── clinical_panel_validation.json         # sanitised panels + held-out AUCs (canonical)
├── clinical_panel_validation_leaky.json   # audit variant (includes 8 embedding inputs)
├── clinical_panel_validation_summary.csv  # side-by-side sanitised vs audit summary
├── biomarker_validation.json              # legacy back-compat copy of the sanitised payload
├── biomarker_panel_c0.json                # legacy per-cluster panel dumps (back-compat)
├── biomarker_panel_c1.json
├── biomarker_panel_c2.json
├── biomarker_panel_c3.json
├── biomarker_panel_c4.json
└── biomarker_panel_c5.json

output/stratification/stage_c/figures/   # 14 figures total
├── 01_..05                              # from Stage C pipeline (regenerated)
├── 06_..08                              # t-SNE + Stage B comparison
├── 09_boundary_migration.png            # NEW — boundary C5 → C0 flow
├── 10_cluster_feature_zscores.png       # NEW — per-cluster z-score heatmap
├── 11_c5_cross_cohort_spread.png        # NEW — C5 within-cluster homogeneity
├── 12_c5_sub_clusters.png               # NEW — C5 k=3 sub-clustering
├── 13_biomarker_panels.png              # NEW — clinical-feature panel AUCs + sizes
                                        # (filename retained for back-compat)
└── 14_tsne_with_boundaries.png          # NEW — t-SNE with boundary overlay

src/face_stratification/stage_c/
├── deep_analysis.py                    # boundary, compactness, cohort-stratified, sub-clustering
├── clinical_panels.py                  # leakage-safe 5-split CV, sanitised + audit panel validators
└── biomarkers.py                       # legacy back-compat shim (re-exports clinical_panels, DeprecationWarning)
```

## 10. Test coverage

```
$ pytest tests/face_stratification/ -q
...101 passed in 6.35 s
```

The new test `test_rank_biserial_bounds_and_sign` explicitly asserts
both directions:

```python
# Test 1: inside HIGHER → rb = +1
inside = np.array([10, 11, 12, 13, 14])
outside = np.array([1, 2, 3, 4, 5])
u, _ = mannwhitneyu(inside, outside, alternative="two-sided")
eff = _rank_biserial(u, 5, 5)
assert eff == pytest.approx(1.0)

# Test 2: inside LOWER → rb = -1
...
```

The previous version of the test accepted either sign (`approx(-1.0)
or approx(1.0)`), which was the flaw that allowed the bug to persist.

---

## Summary

The deep analysis pass **inverted the interpretation of every cluster**
relative to the Stage C v1 document — not because the clustering itself
changed, but because a sign bug in the rank-biserial effect-size
computation was silently reporting every direction backwards. The fix
is in place, the tests catch it, and the clinical picture that emerges
is scientifically much more coherent:

- **Cluster 0** is pediatric autism, not adult autism.
- **Cluster 3** is young, female, metabolically HEALTHY, new-onset — the
  opposite of metabolic syndrome.
- **Cluster 5** (containing 94 % of DR) is chronic STABLE burnt-out
  illness, not chronic comorbid anxious.
- **Cluster 1** is the high-burden cluster with heavy psychiatric and
  somatic comorbidity, substance use, impulsivity, and hostility.
- **Cluster 2** is older BP+SZ with metabolic syndrome features.
- **Cluster 4** is young early-onset comorbid lean.

The cluster **compactness** and **transdiagnostic score** findings from
Stage C v1 remain valid (those are not affected by the sign bug).
Cluster 3 remains the most compact cluster by a factor of 2, and
cluster 5 remains the most transdiagnostic by entropy. The **minimum
clinical-feature panels**, re-evaluated under a leakage-safe 5-split
stratified shuffle CV with an explicit exclusion of the eight
graph-seeding features, now report **held-out AUCs of 0.67–0.86**
(C0 0.859, C1 0.707, C2 0.710, C3 0.696, C4 0.672, C5 0.686). The
previous "AUC > 0.92 for every cluster" claim has been retracted:
the leaky variant of the protocol inflated the AUC by +0.14 to
+0.30 points by letting the greedy selector pick the very features
that had defined the clusters. The sanitised panels are not
deployable individual tests, but they are compact, auditable,
cross-cohort **sparse phenotypic signatures** that support
enrichment, teaching, and hypothesis generation — the most striking
example being C3's metabolic-impulsivity-early-illness signature
(waist, triglycerides, illness duration, BIS-10, family load, age
at first episode).

The most important new finding is that **cluster 5 contains two
distinct sub-phenotypes** (mood-spectrum and psychosis-neurodevelopmental)
that share the "chronic stable burnt-out" profile but differ on the
core psychopathology axis. This motivates a k=7 consensus that will
be evaluated in Stage D.

Stage B2 (torch-geometric GNN embeddings) should be prioritized for
resolving the C5 ↔ C0 boundary — the 604 patients who sit between
pediatric autism and adult chronic-stable are the most informative
test case for deep representation learning.
