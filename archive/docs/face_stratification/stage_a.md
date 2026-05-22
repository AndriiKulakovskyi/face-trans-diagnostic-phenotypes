# Stage A — Feature Engineering and Harmonization

## Objective

Unify heterogeneous clinical data from 4 FACE psychiatric cohorts (BP 6,252 / SZ 2,209 / ASP 2,001 / DR 552 = 11,014 patients) into a single normalized feature matrix suitable for graph construction and embedding learning.

## Method

### Feature Schema

- **184 unified features** across **21 clinical blocks**.
- Features span three resolution levels: instrument totals, clinical sub-scales (the level at which clinicians interpret results), and derived composites.
- Each feature annotated with type (continuous / ordinal / binary), temporal scope (current / lifetime / static), direction (higher_is_worse / higher_is_better / none), and cohort availability.
- Each block declares its own distance metric (cosine, euclidean, or Gower) and minimum overlap constraints for graph construction.

### Block Catalog

| Block | Features | Cohorts | Metric | Description |
|-------|----------|---------|--------|-------------|
| **demographics** | 5 | BP SZ DR ASP | gower | Age, sex, education, marital status, employment. Gower handles the mix of continuous, ordinal, and binary types. |
| **mood** | 9 | BP SZ DR ASP | cosine | Depression and mania severity. MADRS, YMRS, CGI-S, QIDS, MAThyS (BP/DR), ASRM (BP), BDI-II (ASP), SHAPS (DR), Calgary (SZ). Different cohorts use different instruments but all contribute to the same mood similarity layer. |
| **psychosis** | 11 | SZ | cosine | PANSS total + 3 traditional subscales (P/N/G) + Wallwork 5-factor decomposition (positive, negative, disorganized, excited, depressed) + AIMS involuntary movements + BARS akathisia. The Wallwork depressed factor is near-orthogonal to P/N, capturing a distinct dimension. |
| **anxiety_impulsivity** | 8 | BP DR ASP | cosine | State anxiety (STAI-YA), clinician-rated anxiety (HAM-A for ASP), social anxiety (LSAS for ASP), total impulsivity (BIS-10) and 3 dimensions (attentional, motor, non-planning), affective lability (ALS for BP). |
| **hostility_aggression** | 10 | BP | cosine | BDHI total + 9 subscales: assault, indirect hostility, irritability, negativism, resentment, suspicion, verbal hostility, guilt, attitudinal. Separated from anxiety/impulsivity because hostile attribution bias is a distinct dimension from impulsivity. |
| **functioning** | 4 | BP SZ DR ASP | cosine | Daily-life disability and health utility: FAST (BP/DR), PSP (SZ), EQ-5D (BP/SZ/ASP), EGF (DR/ASP). |
| **sleep_circadian** | 3 | BP SZ DR ASP | cosine | Sleep quality (PSQI), daytime somnolence (ESS), chronotype (CSM). Sleep disturbance is a transdiagnostic risk factor across all four pathologies. |
| **cognition** | 17 | BP SZ DR | euclidean | Core neuropsychological battery: TMT-A/B + derived indices (B−A, B/A), Stroop (word, color, color-word, interference), CVLT (learning, short/long delay, recognition), phonemic/semantic fluency, WAIS (similarities, vocabulary, working memory). |
| **neuropsych** | 10 | BP SZ | euclidean | Extended battery: WAIS-IV subtests (matrices, code, symbol search), digit span (forward, backward, total), CPT sustained attention (omissions, commissions, hit RT, variability). |
| **biology** | 13 | BP SZ DR ASP | euclidean | Somatic panel: BMI, waist, BP, HR, QTc, fasting glucose, lipids (cholesterol, HDL, TG, TG/HDL ratio), derived metabolic syndrome flag (IDF/ATP-III), waist-to-height ratio. |
| **treatment** | 11 | BP SZ DR ASP | cosine | Medication classes (binary flags), adherence (MARS), plasma levels (lithium, valproate, clozapine), polypharmacy index (count of concurrent classes — trans-cohort eligible). |
| **substance** | 5 | BP SZ DR ASP | gower | Tobacco (current + CPD), alcohol, cannabis, lifetime substance use disorder. All measured across all 4 cohorts. |
| **trauma** | 7 | BP SZ DR ASP | cosine | CTQ total + 5 subscales (emotional abuse, physical abuse, sexual abuse, emotional neglect, physical neglect) + PCL-5 PTSD (DR). Subscales have differential associations with disorders and treatment response. |
| **family_history** | 4 | BP SZ DR ASP | gower | Family history of bipolar, suicide, substance use (binary) + count of affected relatives. ASP lacks family history (all NaN). |
| **comorbidities** | 2 | BP SZ DR ASP | gower | Count of somatic and psychiatric comorbidities. |
| **suicide_history** | 4 | BP SZ DR ASP | gower | Lifetime ideation, attempt (binary), attempt count, violent attempt flag. ASP partial (BDI-II item 9 only). |
| **psychiatric_history** | 8 | BP SZ DR | cosine | Age at onset, illness duration, episode counts (depressive/manic for BP), rapid cycling (BP), hospitalizations, onset category (early/typical/late), illness burden composite (log-scaled). |
| **insight** | 10 | SZ | cosine | SUMD mean + 9 awareness items (illness, medication, social consequences, hallucinations, delusions, thought disorder, flat affect, anhedonia, asociality). Item-level profiles distinguish subtypes. |
| **autism_profile** | 32 | ASP | cosine | DSM-5 domains, ADI-R 4 domains, RBS-R total + 6 subscales, BRIEF GEC + 9 subscales, WAIS-IV 4 indices, ADHD-RS 2 dimensions, LSAS anxiety/avoidance. Largest block — autism requires multi-dimensional characterization. |
| **treatment_resistance** | 6 | DR | cosine | Resistance flag, Sachs staging, C-SSRS ideation, Rosenberg self-esteem, ERD emotional reactivity, LEAPS work functioning. |
| **personality** | 5 | DR | cosine | BFI Big Five: openness, conscientiousness, extraversion, agreeableness, neuroticism. Only administered to DR. |

### Cohort Adapters

- One adapter per cohort (`adapt_{bp,sz,dr,asp}_profile`) maps `PatientData` dataclasses to the unified feature dict.
- Sub-scale extraction via `_subscale(scores, instrument_key, subscale_name)` helper reading French-language keys from `ScoreInterpretation.subscales` (populated from glossary YAML `subscale_columns`).
- The `ignore_suspect` parameter bypasses the suspect-value filter for instruments like ADI-R whose domain subscales are independently valid even when the total trips the suspect check.
- Derived features computed inline: `tmt_b_minus_a`, `tmt_ratio_ba`, `bio_tg_hdl_ratio`, `tx_polypharmacy_index`, `psyh_illness_burden`, `psyh_onset_category`, `bio_waist_height_ratio`.
- PANSS Wallwork 5-factor scores read from pre-computed columns in SZ.csv (`pansspow`, `panssnew`, `panssdiw`, `panssexw`, `panssdew`).

### Feature Counts Per Cohort

| Cohort | Features emitted | % of schema |
|--------|-----------------|-------------|
| BP     | 104             | 57%         |
| SZ     | 98              | 53%         |
| DR     | 94              | 51%         |
| ASP    | 69              | 38%         |

### Trans-cohort Feature Selection

A feature is "trans-cohort" if it has ≥50% observed coverage in **every** cohort. This strict criterion ensures that when two patients from different cohorts are compared in the cross-cohort similarity graph, the comparison is based on actually observed measurements, not imputed or absent values.

#### Selection mechanism

For each of the 184 features, we compute the fraction of patients with a non-NaN value separately in each of the 4 cohorts. The feature enters the trans-cohort set only if that fraction is ≥50% in BP **and** SZ **and** DR **and** ASP. One cohort at 49% and the feature is excluded.

#### Results: 9 trans-cohort features (Tier 1)

| Feature | Block | BP | SZ | DR | ASP |
|---------|-------|----|----|----|----|
| `demo_age_years` | demographics | 100% | 100% | 100% | 100% |
| `demo_sex_male` | demographics | 100% | 100% | 100% | 100% |
| `sub_tobacco_current` | substance | 100% | 100% | 100% | 100% |
| `sub_alcohol_current` | substance | 100% | 100% | 100% | 100% |
| `sub_cannabis_current` | substance | 100% | 100% | 100% | 100% |
| `sub_use_disorder` | substance | 100% | 100% | 100% | 100% |
| `cm_n_somatic` | comorbidities | 100% | 100% | 100% | 100% |
| `cm_n_psychiatric` | comorbidities | 100% | 100% | 100% | 100% |
| **`tx_polypharmacy_index`** | treatment | 100% | 100% | 100% | 100% |

The polypharmacy index (count of concurrent medication classes) is the only clinically actionable feature that entered the trans-cohort set — and it only exists because it was added as a derived composite in Phase 5. The remaining 8 are demographics, substance use, and comorbidity counts.

#### Near-misses

Only one feature comes close to qualifying: `sui_ever_ideation` (76% BP, 95% SZ, 72% DR, **46% ASP**). ASP uses BDI-II item 9 as a proxy instead of the dedicated C-SSRS/ISF scales, dropping coverage 4 percentage points below the threshold.

Many clinically important features miss the threshold specifically because of ASP:

| Feature | BP | SZ | DR | ASP (bottleneck) |
|---------|----|----|----|----|
| `inst_ctq_total` | 91% | 95% | 81% | **17%** |
| `inst_psqi_total` | 93% | 63% | 56% | **1%** |
| `inst_cgis_total` | 84% | 96% | 91% | **4%** |
| `inst_mars_total` | 81% | 92% | 63% | **14%** |
| `demo_education_years_ordinal` | 72% | 82% | 65% | **19%** |

#### Tiered structure

The strict all-4-cohort threshold is informative but limiting. A tiered analysis reveals the underlying cohort overlap topology:

**Tier 2 (≥50% in at least 3 cohorts): 31 features.** Every Tier 2 feature that fails Tier 1 fails because of ASP. This means BP, SZ, and DR share substantial instrument overlap — CTQ subscales, treatment flags, family history, sleep quality, mood scales — while ASP is assessed on a fundamentally different battery. Tier 2 adds clinically meaningful features: CTQ 5 trauma subscales, CGI-S, PSQI, MARS adherence, treatment medication flags, family history, and suicide indicators.

**Tier 3 (≥50% in exactly 2 cohorts): 24 features.** These reveal the cohort-pair structure:

- **BP ↔ DR** share the most (13 features): MADRS, QIDS, MAThyS, STAI-YA, BIS-10, FAST, ESS, CSM, waist, blood pressure, lithium, illness duration, illness burden. This reflects the clinical overlap between bipolar and treatment-resistant depression as mood disorders with shared assessment protocols.
- **BP ↔ SZ** share cognition (5 features): CVLT long-delay and recognition, phonemic fluency, WAIS matrices, hospitalizations. The shared neuropsychological battery bridges these cohorts.
- **SZ ↔ DR** share metabolic labs (4 features): glucose, HDL, triglycerides, TG/HDL ratio.
- **BP ↔ DR** share biology (2 features): waist/height ratio, blood pressure.

#### Interpretation

The trans-cohort analysis reveals a fundamental structural property of the FACE data:

1. **The four cohorts have very little instrument overlap.** Despite belonging to the same research network, each cohort uses mostly different clinical instruments. Only demographic, substance, and comorbidity variables are universally measured.

2. **ASP is the systematic bottleneck.** The autism cohort is assessed on a fundamentally different battery (BRIEF, RBS-R, ADI-R, WAIS-IV indices, ADHD-RS) that shares almost nothing with the mood/psychosis instruments used for BP, SZ, and DR. This is not a data quality issue — it reflects the clinical reality that autism assessment requires domain-specific instruments.

3. **BP and DR are the closest pair.** They share 13 features at Tier 3, reflecting their overlap as mood disorders. This predicts that BP and DR patients may be more easily mixed within data-driven clusters, while SZ and ASP will remain more cohort-specific.

4. **The trans-cohort graph alone cannot discover clinical phenotypes.** Nine features (demographics + substance + comorbidities + polypharmacy) carry almost no mood, cognitive, or symptom information. Any clusters found using only these features would reflect age/sex/smoking/comorbidity patterns, not psychiatric phenotypes.

5. **The multiplex graph architecture is essential.** The pipeline builds separate block-level similarity graphs (mood graph, cognition graph, biology graph, etc.) and combines them via multiplex spectral embedding. Within each block graph, a BP patient and a DR patient connect through shared MADRS scores; the same BP patient connects to an SZ patient through shared CVLT scores. The multiplex approach lets clinical information flow across cohorts through **domain-specific bridges** rather than requiring a single universal feature set.

6. **ASP will form a relatively isolated community** in the multiplex graph. Its cross-cohort bridges are limited to the 9 trans-cohort features. This is scientifically honest — forcing comparability through imputation would mask the genuine measurement heterogeneity.

### Normalization

- Robust z-scoring: winsorize to 1st–99th percentile, then (x − median) / (MAD × 1.4826).
- Sign flip: features with `direction=higher_is_worse` are negated so higher = more pathological everywhere.
- Continuous features only — ordinal, binary, and categorical pass through unchanged.
- This normalized matrix feeds all embedding methods (PCA, UMAP, spectral, GNN).

### Missingness Treatment

- FACE missingness is **MNAR** (missing not at random): ASP systematically lacks labs, family history, and suicide assessment; DR lacks psychosis scales; SZ lacks BFI.
- `characterize_missingness()` produces per-feature per-cohort rates, missingness correlation matrix, Little's MCAR test per block.
- Default path: **pairwise-complete masked similarity** — no imputation. Block-level graphs use only the observed feature subspace for each patient pair. The semantic overlap constraint (`min_shared_features`) ensures edges are only drawn between patients with sufficient shared measurements.
- Missingness indicator columns (`miss_<block>`) augment the feature matrix to preserve MNAR patterns as first-class signals.

### Graph Construction

#### Why build graphs at all?

The harmonized feature matrix is 11,014 patients × 184 features with ~62% NaN. Three structural problems make direct matrix-based clustering (e.g. k-means on the raw matrix) inadequate:

1. **Massive structured missingness.** ASP patients have no cognition scores, SZ patients have no BFI personality, DR patients have no PANSS. Any algorithm that requires a complete feature vector — or imputes missing values to create one — either discards most patients or fabricates data where none was collected. A patient similarity graph sidesteps this entirely: similarity is computed **only on the features both patients actually have**, and if two patients share no measurements in a domain, they simply receive no edge in that domain's graph. The missingness is encoded as *absence of connection*, not as an imputed zero.

2. **Feature space incomparability across domains.** A 1-unit change in MADRS total (mood severity, 0–60 scale) is clinically unrelated to a 1-unit change in TMT-B time (executive function, seconds) or a 1-unit change in fasting glucose (mg/dL). Euclidean distance on the concatenated vector treats them as equivalent. Block-level graphs solve this by computing similarity **within each clinical domain** using the metric that matches that domain's data structure (cosine for symptom profiles, euclidean for cognitive scores, Gower for mixed types), then combining the domain-level similarities into a multiplex structure.

3. **Mixed feature types.** The 184 features include continuous scores, ordinal scales, binary flags, and categorical codes. No single distance metric handles all types correctly. Gower distance (used for demographics, substance, family history, comorbidities, suicide history) is explicitly designed for mixed types, while cosine (symptom profiles) and euclidean (cognitive/biological measurements) are used where their assumptions hold.

The graph representation converts these three data-level problems into a single, well-defined object — a weighted multiplex network — that downstream embedding models (PCA on the adjacency spectrum, spectral embedding, GNN) can consume without ever encountering the raw missingness or type heterogeneity.

#### How block-level kNN graphs are built

Each of the 21 clinical blocks produces its own similarity graph through a 5-step process:

**Step 1 — Candidate filtering.** Patients with too few observed values in a block are excluded from that block's graph. The threshold is `min_fraction_present` (declared per block in `feature_schema.yaml`; typically 0.3–0.6). A patient with no cognition assessments receives no edges in the cognition graph — this is the desired honest behaviour, not a flaw.

**Step 2 — Pairwise-complete masked similarity.** For every pair of candidate patients, similarity is computed strictly on the subset of features where *both* patients have an observed value. The mathematical core replaces NaN with 0 in the value matrix and maintains a separate boolean mask, enabling fully vectorized computation:

- **Cosine** (mood, psychosis, anxiety, hostility, functioning, sleep, trauma, treatment, psychiatric history, insight, autism profile, treatment resistance, personality): measures the angle between two patient profiles in their shared feature subspace. Two patients with identical relative severity patterns get similarity 1.0 regardless of absolute scale. Used for blocks where the *profile shape* matters more than the absolute level.
- **Euclidean** (cognition, neuropsych, biology): root-mean-square difference over shared features, normalized by overlap count so pairs with different numbers of shared features are comparable. Used for blocks where absolute magnitude differences are clinically meaningful (a TMT-B time of 120s vs 60s is a real difference, not just a profile shape difference).
- **Gower** (demographics, substance, family history, comorbidities, suicide history): per-feature contribution is `|x_i − x_j| / range_f`, averaged over shared features. Handles the mix of continuous (age), ordinal (education), and binary (sex, tobacco use) features by normalizing each to its empirical range.

**Step 3 — Semantic overlap constraint.** An edge (i, j) is only created if the pair shares at least `min_shared_features` observed measurements. This prevents spurious high-similarity edges between patients who happen to share 1 observed feature with similar values. The default threshold is `ceil(n_features / 2)` or a per-block override from the schema. For example, in the cognition block (17 features), two patients must share at least 9 observed cognitive scores to be connected.

**Step 4 — k-nearest-neighbour selection.** Among the pairs passing the overlap constraint, each patient retains edges only to its k nearest neighbours (default k=10). This sparsifies the graph while preserving local structure. The k-NN selection is undirected: if patient A selects B as a neighbour OR B selects A, the edge exists.

**Step 5 — Edge weighting.** Raw similarity is converted to a final edge weight via two factors:
- **Gaussian kernel**: `w_gauss = exp(−d² / 2σ²)` where d is the masked distance and σ (bandwidth) is the median distance across all edges in the block. This soft-thresholds distant neighbours.
- **Overlap confidence**: `w_conf = overlap_count / n_features_in_block`. A pair sharing 100% of a block's features gets full weight; a pair at the minimum overlap threshold gets proportionally less.
- **Final weight**: `w = w_gauss × w_conf`. This ensures that edges supported by more complete data are stronger in the downstream embedding.

#### The 21 block-level graphs and their metrics

| Block | Features | Metric | Rationale |
|-------|----------|--------|-----------|
| demographics | 5 | gower | Mixed types (continuous age, binary sex, ordinal education) |
| mood | 9 | cosine | Severity profile shape across different mood instruments |
| psychosis | 11 | cosine | PANSS profile shape (P/N/G + Wallwork 5 factors) |
| anxiety_impulsivity | 8 | cosine | Anxiety–impulsivity dimensional profile |
| hostility_aggression | 10 | cosine | BDHI hostility dimension profile |
| functioning | 4 | cosine | Disability profile across instruments |
| sleep_circadian | 3 | cosine | Sleep quality–somnolence–chronotype profile |
| cognition | 17 | euclidean | Absolute performance differences matter (120s vs 60s on TMT) |
| neuropsych | 10 | euclidean | Extended cognitive battery, same rationale as cognition |
| biology | 13 | euclidean | Lab values where absolute magnitude is clinically meaningful |
| treatment | 11 | cosine | Medication class profile (which drugs, not how many) |
| substance | 5 | gower | Mixed binary flags + continuous (cigarettes per day) |
| trauma | 7 | cosine | Trauma dimension profile across CTQ subscales |
| family_history | 4 | gower | Binary flags + count |
| comorbidities | 2 | gower | Counts of somatic and psychiatric conditions |
| suicide_history | 4 | gower | Binary flags + count |
| psychiatric_history | 8 | cosine | Illness trajectory profile |
| insight | 10 | cosine | SUMD awareness profile across 9 domains |
| autism_profile | 32 | cosine | Multi-dimensional autism characterization profile |
| treatment_resistance | 6 | cosine | Depression resistance staging profile |
| personality | 5 | cosine | Big Five personality profile shape |

Each block produces a separate sparse graph. Some blocks are cohort-specific (psychosis → SZ only, autism_profile → ASP only, personality → DR only), others span 2–4 cohorts. This is not a limitation — it is the accurate representation of what the data supports.

#### Multiplex assembly

The 21 block-level graphs are combined into a single **multiplex graph** (a `NetworkX MultiGraph` with one edge type per block). A 22nd edge type — `transdiagnostic` — is built from the 9 trans-cohort features that pass the ≥50% all-cohort coverage threshold, using the same masked kNN procedure.

The multiplex structure preserves the distinction between *why* two patients are similar. Patient A and patient B may be connected by a mood edge (similar depression severity), a biology edge (similar metabolic profile), and a cognition edge (similar TMT scores) — or by only one of these. The downstream spectral or GNN embedding learns from the full multiplex, weighting each edge type by its informativeness.

Cross-cohort bridges form naturally within blocks where instruments overlap:
- **Cognition block**: 38,832 BP ↔ SZ cross-cohort edges through shared CVLT and phonemic fluency scores
- **Neuropsych block**: 51,737 BP ↔ SZ cross-cohort edges through shared WAIS-IV matrices
- **Mood block**: 9,259 BP ↔ DR cross-cohort edges through shared MADRS, QIDS, MAThyS scores
- **Anxiety block**: 17,702 BP ↔ DR cross-cohort edges through shared STAI and BIS-10
- **Biology block**: 14,389 BP ↔ DR cross-cohort edges through shared metabolic labs
- **Functioning block**: 43,317 cross-cohort edges across all 6 cohort pairs (BP↔ASP 4,855; BP↔SZ 24,699; BP↔DR 7,200; ASP↔DR 3,357; ASP↔SZ 3,206)
- **Transdiagnostic layer**: all 4 cohorts connect through demographics, substance use, comorbidities, and polypharmacy

#### Per-block edge weight normalization

Without normalization, low-dimensional Gower blocks (comorbidities with 2 features: 108,916 total weight) dominate the multiplex adjacency 490× more than sparse blocks (treatment_resistance: 222 total weight). This causes the spectral embedding to be driven almost entirely by comorbidity patterns rather than clinical profiles.

The multiplex builder normalizes each block's total edge weight to the median across active blocks. After normalization, every block contributes equally (ratio max/min = 1.0). Individual edge weight ratios within each block are preserved — only the cross-block scale is equalized.

#### Measuring graph quality

Three complementary metrics assess whether a block graph is well-constructed:

1. **Edge density and mean degree.** A block with too few edges provides insufficient signal for embedding; a block where every patient connects to every other is uninformative. `GraphSummary.edges_per_type` and `mean_degree_per_type` are tracked per block. Blocks with zero edges (e.g. because too few patients pass the candidate filter) are flagged in `blocks_with_zero_edges`.

2. **Cohort assortativity coefficient.** Newman's attribute assortativity on the cohort label measures whether edges tend to connect patients within the same cohort (+1) or across cohorts (−1). For multi-cohort blocks (mood, cognition, biology, treatment), low or negative assortativity indicates genuine cross-cohort mixing — patients from different diagnostic categories connect because they share similar clinical profiles. For single-cohort blocks (psychosis, insight, autism_profile), assortativity is +1 by construction — all edges are within one cohort.

3. **Candidate node coverage.** `GraphSummary.candidate_nodes_per_type` reports how many of the 11,014 patients participate in each block's graph. The complement (patients excluded by `min_fraction_present`) quantifies how many patients are invisible to that domain. If a block excludes >80% of patients, it provides signal for only a small subgroup and its weight in the multiplex should be interpreted accordingly.

#### Empirical graph statistics

The final multiplex graph has **11,014 nodes**, **22 edge types**, and **1,154,471 edges**:

| Block | Edges | Mean Degree | Assortativity | Candidates | Bridging? |
|-------|-------|-------------|---------------|------------|-----------|
| comorbidities | 108,916 | 19.8 | 0.066 | 11,014 | All 4 cohorts |
| substance | 109,875 | 20.0 | −0.081 | 11,014 | All 4 cohorts |
| functioning | 94,514 | 17.2 | 0.175 | 9,477 | All 4 cohorts |
| family_history | 89,260 | 16.2 | 0.229 | 9,013 | BP/SZ/DR |
| suicide_history | 81,015 | 14.7 | −0.060 | 8,148 | BP/SZ/DR |
| treatment | 75,248 | 13.7 | 0.655 | 7,872 | BP/SZ/DR |
| transdiagnostic | 72,792 | 13.2 | 0.498 | 11,014 | All 4 cohorts |
| neuropsych | 59,423 | 10.8 | **−0.743** | 6,072 | BP↔SZ |
| demographics | 56,204 | 10.2 | 0.282 | 7,383 | All 4 cohorts |
| anxiety_impulsivity | 55,067 | 10.0 | **−0.086** | 6,328 | BP↔DR |
| trauma | 54,724 | 9.9 | 0.039 | 8,122 | BP/SZ/DR |
| sleep_circadian | 53,083 | 9.6 | 0.026 | 6,264 | BP↔DR |
| psychiatric_history | 51,091 | 9.3 | −0.064 | 5,494 | BP/SZ/DR |
| mood | 44,901 | 8.2 | −0.004 | 6,041 | BP↔DR |
| cognition | 44,586 | 8.1 | **−0.750** | 5,151 | BP↔SZ |
| hostility_aggression | 31,807 | 5.8 | 1.000 | 4,494 | BP only |
| biology | 29,063 | 5.3 | −0.292 | 3,841 | BP↔DR |
| insight | 14,735 | 2.7 | 1.000 | 2,055 | SZ only |
| psychosis | 14,458 | 2.6 | 1.000 | 2,093 | SZ only |
| autism_profile | 10,214 | 1.9 | 1.000 | 1,256 | ASP only |
| personality | 2,511 | 0.5 | 1.000 | 384 | DR only |
| treatment_resistance | 984 | 0.2 | 1.000 | 159 | DR only |

Key observations:
- **Cognition (−0.750) and neuropsych (−0.743)** have strong negative assortativity — BP and SZ patients preferentially connect across cohorts through shared cognitive tests.
- **Anxiety_impulsivity (−0.086)** now bridges BP and DR through shared STAI and BIS-10, after lowering `min_fraction_present` from 0.50 to 0.20.
- **Functioning (0.175)** bridges all 4 cohorts with assortativity well below 1.0, after lowering `min_shared_features` from 2 to 1.
- **6 blocks** remain cohort-specific by design (assortativity = 1.0): psychosis, insight, hostility_aggression, autism_profile, personality, treatment_resistance. This is correct — these blocks use instruments administered to a single cohort.
- **Cross-cohort edge quality**: cross-cohort edges have comparable weights to intra-cohort edges (ratio 0.68–1.36×), with the lower ratios correctly reflecting fewer shared features.

#### Why graph construction belongs in Stage A

Graph construction is sometimes placed in the "modelling" stage of a machine learning pipeline. In this project, it belongs in Stage A (data preparation) for three reasons:

1. **It is deterministic and parameter-light.** The entire graph construction has exactly two free parameters per block: `k` (number of neighbours, default 10) and `min_shared_features` (overlap constraint, default `ceil(n_features / 2)`). Everything else — the metric, the candidate threshold, the bandwidth — is either declared in the YAML schema or derived from the data (median distance). There is no training, no loss function, no gradient. Given the same harmonized matrix and schema, the graph is identical every time.

2. **It solves data-level problems, not modelling problems.** The three problems the graph addresses — structured missingness, feature incomparability, mixed types — are properties of the data collection protocol, not of any downstream learning task. Whether the embedding model is PCA, spectral decomposition, or a GCN, it will face the same missingness and heterogeneity. The graph normalizes these away once, as a preprocessing step.

3. **It is the contract between Stage A and Stage B.** The graph is the input to all embedding models. Stage B receives a weighted multiplex graph and produces low-dimensional patient embeddings — it never touches the raw feature matrix, the YAML schema, or the per-cohort adapters. This clean interface means Stage B can iterate on embedding architectures (PCA → spectral → GNN → contrastive SSL) without re-running harmonization or graph construction.

The graph is not a model of patient similarity — it is a **structured representation of what the data supports**. An edge between two patients in the cognition block means: "these two patients were both assessed on cognition, shared enough measurements for a meaningful comparison, and their cognitive profiles are among the most similar in the cohort." That is a statement about data, not a learned parameter.

### DSM Subtypes

- Fine-grained labels extracted: BP-I/II (from bipolar_type), SZ positive/negative/mixed (from PANSS profile), DR resistance staging (from Sachs score), ASP functioning level (from EGF).
- ~10–12 subtypes replace the coarse 4-cohort labels for all comparison analyses.

## Diagnostic Inspection

The notebook `notebooks/stage_a_feature_inspection.ipynb` produces comprehensive diagnostics:

1. **Coverage heatmap** — per-cohort % non-NaN for all 184 features, grouped by block.
2. **Distribution violin plots** — one figure per block (21 figures), showing every continuous/ordinal feature faceted by cohort. Checks for floor/ceiling effects, multimodality, and extreme skew.
3. **Correlation heatmaps** — Spearman rank correlation within each block (15 figures for blocks with ≥3 continuous features). Flags pairs with |r| > 0.85 for redundancy review.
4. **Trans-cohort selection analysis** — bar chart of total vs trans-cohort features per block.
5. **Missingness patterns** — per-block MCAR/MAR/MNAR mechanism classification + missingness correlation heatmaps (17 figures) showing whether subscales go missing together (instrument-level dropout) or independently.
6. **t-SNE projection** — PCA → t-SNE on trans-cohort features, colored by cohort and by individual clinical features (polypharmacy, CTQ, BMI).
7. **PANSS Wallwork validation** — correlation matrix between traditional P/N/G and Wallwork 5 factors, confirming differential structure.
8. **Subscale fix validation** — CTQ total vs sum(5 subscales) r=1.000, BIS-10 total vs sum(3 subscales) r=1.000, confirming the French key fix is correct.
9. **Derived feature distributions** — histograms of polypharmacy index, onset category, illness burden, waist/height ratio per cohort.
10. **Summary statistics** — mean, std, coverage per cohort for all 184 features, exported to CSV.
11. **Assortativity before/after** — grouped bar chart comparing cohort assortativity per block before and after parameter tuning. Highlights the 4 blocks that went from assortativity=1.0 (no cross-cohort edges) to negative assortativity (active bridging).
12. **Cross-cohort bridge heatmap** — block × cohort-pair heatmap showing the number of cross-cohort edges per block. Reveals which clinical domains connect which cohort pairs.
13. **Block weight normalization** — side-by-side bar chart of total edge weight per block before (490× range) and after (1.0× range) normalization.
14. **Cohort participation** — block × cohort heatmap showing what fraction of each cohort's patients are candidates in each block's graph.

## Key Findings

### Data engineering

- **Subscale keys were broken**: The previous adapters used English shorthand keys ("attentional", "emotional_abuse") but the actual `ScoreInterpretation.subscales` dict uses French names from the glossary YAML ("Impulsivité attentionnelle", "Abus émotionnel"). All CTQ, BIS-10, and BFI subscales were silently returning None. Fixed by mapping to the correct French keys. Verified with sum-to-total checks: CTQ total vs sum(5 subscales) r=1.000, BIS-10 total vs sum(3 subscales) r=1.000.
- **ADI-R requires ignore_suspect**: The ADI-R diagnostic algorithm score trips the suspect-value filter, but its domain subscales (social interaction, communication, restricted behaviors, development) are independently valid clinical measures. Added `ignore_suspect=True` parameter to `_subscale()`, recovering 62% coverage for ADI-R domains in ASP.
- **SCIP coverage too low**: Only 1.8% of BP patients have SCIP cognitive screening data — excluded from schema. The standard neuropsychological battery (TMT, Stroop, CVLT, WAIS) has >80% coverage and is preferred.
- **SUMD items 4–9 had wrong column names**: The SZ column map specified `sumd04`–`sumd09` but the CSV has `sumd04a`–`sumd09a` (awareness ratings) and `sumd04c`–`sumd09c` (attribution ratings). Fixed to use the 'a' (awareness) variants, recovering 93% coverage for all 9 SUMD items and reviving the insight block from 0 edges.
- **BFI trips suspect_value**: The BFI instrument definition uses `bfi_extr` (Extraversion subscale) as `total_column`, making the BFI always flag `suspect_value=True`. Added `ignore_suspect=True` for BFI subscale extraction in the DR adapter, recovering 66% coverage for all 5 Big Five dimensions and reviving the personality block from 0 edges.

### Instrument validation

- **PANSS Wallwork 5-factor model adds information beyond P/N/G**: The Wallwork depressed factor correlates r=0.16 with traditional Positive and r=0.08 with Negative — it is nearly orthogonal to both and captures a mood dimension within psychosis not visible from P/N/G alone. The Wallwork excited factor (r=0.48 with P, r=0.18 with N) captures agitation independently. Wallwork-negative correlates r=0.96 with traditional N (expected — largely the same items), confirming convergent validity.
- **High-correlation pairs to monitor**: PANSS-N vs Wallwork-negative (r=0.96), CVLT long-delay vs recognition (r=0.93), RBS-R total vs sameness subscale (r=0.92), TMT-B vs B−A (r=0.89), BDHI resentment vs attitudinal (r=0.89). These represent expected total-subscale or method-overlap redundancy, not measurement problems.

### Trans-cohort analysis

- **Only 9 features pass the strict all-4-cohort ≥50% threshold**: demographics (age, sex), substance use (tobacco, alcohol, cannabis, SUD), comorbidity counts (somatic, psychiatric), and the derived polypharmacy index. No mood, cognitive, biological, or symptom severity features are trans-cohort.
- **ASP is the systematic bottleneck**: Of the 175 excluded features, the vast majority fail because of ASP. CTQ has 91/95/81% in BP/SZ/DR but 17% in ASP. PSQI has 93/63/56% but 1% in ASP. The autism cohort uses a fundamentally different assessment battery.
- **The cohort-pair topology follows clinical lines**: BP↔DR share the most features (13 at Tier 3, all mood/anxiety instruments), BP↔SZ bridge through cognition (5 shared neuropsych features), SZ↔DR bridge through metabolic labs (4 features). These pairings predict which cohorts will intermix most easily in the similarity graph.
- **The multiplex graph architecture is essential**: A single trans-cohort graph from 9 features cannot discover psychiatric phenotypes. The block-level multiplex approach lets clinical information flow across cohorts through domain-specific bridges — mood instruments connect BP↔DR, cognition connects BP↔SZ — without requiring every feature to be measured in every cohort.

### Graph construction

- **Initial graph had 2 dead blocks and 4 failing bridges**: The insight block (SUMD column name mismatch) and personality block (BFI suspect_value) produced zero edges. Four multi-cohort blocks (cognition, neuropsych, anxiety_impulsivity, functioning) had assortativity=1.0 despite spanning multiple cohorts — `min_fraction_present` was too high for cohorts with partial instrument coverage, and `min_shared_features` was too high for cross-cohort pairs sharing only 1–3 features.
- **Parameter tuning created 4 new cross-cohort bridges**: Lowering `min_fraction_present` (cognition 0.50→0.15, neuropsych 0.50→0.10, anxiety_impulsivity 0.50→0.20) and `min_shared_features` (cognition 5→2, neuropsych 4→1, functioning 2→1) allowed patients from different cohorts to enter the graph and connect through their shared instruments. Cognition assortativity dropped from 1.0 to −0.75, neuropsych from 1.0 to −0.74, anxiety_impulsivity from 1.0 to −0.09, functioning from 1.0 to 0.18.
- **Cross-cohort edge quality is comparable to intra-cohort**: For the newly bridging blocks, cross-cohort edges have 0.68–1.36× the mean weight of intra-cohort edges. The lower ratios correctly reflect lower overlap confidence (fewer shared features), not lower data quality.
- **Block weight normalization eliminates 490× imbalance**: Without normalization, comorbidities (2 features, Gower) has 490× more total edge weight than treatment_resistance (6 features, cosine). After normalization, all 22 blocks contribute equal total weight to the multiplex adjacency, preventing low-dimensional blocks from dominating the spectral embedding.
- **The dominant cross-cohort link is BP↔SZ through cognition**: Cognition (38,832 edges) and neuropsych (51,737 edges) together provide ~90,000 BP↔SZ cross-cohort edges — by far the largest bridge in the multiplex. This predicts that BP and SZ patients with similar cognitive profiles will cluster together, which is clinically meaningful (cognitive deficits are a shared endophenotype).
- **ASP remains relatively isolated**: ASP participates in functioning (61%), demographics (13%), and universal blocks (comorbidities, substance, transdiagnostic), but has no dedicated cross-cohort bridge with BP, SZ, or DR in the clinical assessment blocks. ASP's cross-cohort connections are limited to the thin transdiagnostic layer.

### Graph topology analysis — data science conclusions

The 22-layer multiplex graph reveals a three-tier taxonomy of clinical blocks, defined by two orthogonal dimensions of transdiagnosticity: **quantity** (percentage of cross-cohort edges) and **quality** (mean weight ratio of cross-cohort vs intra-cohort edges).

#### Tier 1 — Strongly transdiagnostic blocks

Cognition (87% cross-cohort, weight ratio 0.79), neuropsych (87% cross, ratio 0.68), substance (53% cross, ratio ~1.0), and comorbidities (48% cross, ratio ~1.0) form the backbone of cross-cohort connectivity. These blocks produce the majority of inter-diagnostic edges and do so with edge weights that are comparable to (or even exceed) intra-cohort edges.

Among these, the cognitive blocks are remarkable: their assortativity is **strongly negative** (cognition −0.750, neuropsych −0.743). Negative assortativity means that, within these blocks, a patient is *more likely* to connect to someone from a different cohort than from their own. This is not an artifact — it reflects the empirical structure of cognitive performance distributions: BP and SZ patients scored on the same CVLT, TMT, and WAIS battery produce overlapping performance distributions, and kNN selects the genuinely nearest neighbours regardless of diagnosis.

The substance and comorbidities blocks have near-zero assortativity (−0.081 and +0.066 respectively), consistent with random mixing. These blocks connect all 4 cohorts but carry limited clinical specificity — they reflect lifestyle and medical burden patterns that are largely orthogonal to the primary psychiatric diagnosis.

#### Tier 2 — Selective bridge blocks

Mood (42% cross, ratio 1.11), anxiety_impulsivity (35% cross, ratio 1.36), biology (54% cross, ratio ~0.9), functioning (45% cross, ratio ~1.0), sleep_circadian (35% cross, ratio ~0.9), trauma (31% cross, ratio ~0.8), and psychiatric_history (48% cross, ratio ~0.9) produce fewer cross-cohort edges than Tier 1, but their edges are **clinically informative** — they connect specific cohort pairs through shared instruments. Notably, anxiety_impulsivity cross-cohort edges have the highest weight ratio (1.36×) of any block, meaning that when a BP patient and a DR patient share similar STAI and BIS-10 scores, their similarity is stronger than typical intra-cohort pairs. This is a hallmark of genuine phenotypic overlap: these patients are not "barely similar" across diagnoses — they are more similar to each other than to many same-diagnosis peers.

The key cohort-pair topology emerges from these blocks:

| Cohort pair | Primary bridge blocks | Edges | Total weight |
|-------------|----------------------|-------|-------------|
| **BP ↔ SZ** | cognition, neuropsych, functioning | 292,157 | 83,037 |
| **BP ↔ DR** | mood, anxiety, biology, sleep, trauma | 87,649 | 34,842 |
| **ASP ↔ BP** | functioning, comorbidities, substance | ~40,000 | ~9,600 |
| **ASP ↔ SZ** | functioning, comorbidities, substance | ~17,000 | ~4,200 |
| **DR ↔ SZ** | biology, psychiatric_history | 14,901 | 3,443 |
| **ASP ↔ DR** | functioning, comorbidities, substance | ~3,400 | ~500 |

The BP↔SZ link dominates the cross-cohort topology by an order of magnitude, driven almost entirely by the shared neuropsychological battery. The BP↔DR link is second, driven by shared mood and anxiety instruments. DR↔SZ is the weakest inter-cohort bridge — these two cohorts share very few instruments, connecting only through metabolic labs and hospitalization history.

#### Tier 3 — Cohort-specific blocks

Six blocks have assortativity = 1.0 by construction: psychosis (SZ), insight (SZ), hostility_aggression (BP), autism_profile (ASP), personality (DR), treatment_resistance (DR). These blocks produce **zero cross-cohort edges** — every edge connects two patients from the same cohort.

This is not a failure. These blocks use instruments that are administered to only one cohort because they assess domain-specific pathology. PANSS measures psychotic symptoms that only SZ patients have; BRIEF and RBS-R measure executive and repetitive behaviour profiles specific to autism; BFI personality was administered only to DR. Cohort-specific blocks serve a different function in the multiplex: they provide **within-cohort resolution**. While transdiagnostic blocks determine which patients from different cohorts are similar, cohort-specific blocks differentiate patients *within* each diagnostic group. In the embedding, these blocks determine the fine-grained positions of patients within their cohort's region of the latent space.

#### Implications for embedding learning

The three-tier structure predicts specific properties of the downstream embeddings:

1. **Block weight normalization is essential.** Without normalization, comorbidities (109K total weight, 2 features) would overwhelm treatment_resistance (222 total weight, 6 features) by 490×. The spectral embedding would be driven almost entirely by somatic and psychiatric comorbidity counts — a clinically shallow representation. After normalization to equal total weight, every block contributes proportionally to the graph Laplacian, and the spectral decomposition reflects the full clinical depth of the 21-block structure.

2. **The first spectral components will separate ASP from {BP, SZ, DR}.** ASP participates in only 8 of 22 blocks (vs 16–18 for BP/SZ/DR), and its cross-cohort edges are limited to thin universal blocks (comorbidities, substance, demographics). In the graph Laplacian, ASP patients form a loosely connected sub-community — the leading eigenvectors will capture this separation.

3. **Intermediate spectral components should reveal the BP↔SZ cognitive axis.** The negative-assortativity cognition and neuropsych blocks contribute ~90K BP↔SZ cross-cohort edges with strong weights. Spectral decomposition of this structure should produce components along which BP and SZ patients intermix, with position determined by cognitive performance rather than diagnosis.

4. **The risk of recapitulating nosology is moderate.** Treatment (assortativity +0.655) and transdiagnostic (+0.498) blocks are the most assortative multi-cohort blocks. These blocks, combined with the 6 cohort-specific blocks, exert an inward pull — grouping patients by diagnosis. The question for Stage B is whether the 10 bridging blocks (particularly cognition, mood, and anxiety) generate enough cross-cohort mixing to overcome this inward pull and produce genuinely transdiagnostic clusters.

5. **0% patient isolation across all cohorts.** Every patient has at least one edge in the multiplex. DR patients have the highest cross-cohort degree (mean 198.8), reflecting their connection to BP through mood, anxiety, and biology. ASP patients have the lowest (mean 30.1), consistent with their limited instrument overlap. This ensures that no patient is invisible to the embedding, though ASP patients' positions will be determined primarily by the thin universal blocks.

### Graph topology analysis — clinical conclusions

#### Cognition as the strongest transdiagnostic bridge

The strongest empirical link in the multiplex graph is BP↔SZ through shared cognitive deficits. This finding is consistent with a large body of neuropsychological literature showing that cognitive impairment — particularly in executive function, processing speed, and verbal memory — is a shared endophenotype across the schizophrenia–bipolar spectrum.

The negative assortativity in cognition (−0.750) means that a BP patient with severe executive dysfunction (slow TMT-B, low Stroop interference) is more likely to be connected to an SZ patient with a similar profile than to a BP patient who happens to be cognitively intact. The graph is encoding what the RDoC framework calls a *dimensional cognitive system deficit* that cuts across categorical diagnoses. If downstream clustering finds a "cognitive impairment cluster" containing both BP and SZ patients, this is not a confound — it is the central hypothesis of dimensional psychiatry, materialized in the data.

Conversely, cognitively spared BP patients (fast TMT, good CVLT) will **not** bridge to SZ through this layer. The bridge is selective — it connects specific patient subtypes, not entire diagnostic groups. This selectivity is what makes the kNN graph more informative than a simple cohort-overlap indicator.

#### Mood and anxiety bridge BP to treatment-resistant depression

The BP↔DR link through mood (MADRS, QIDS, MAThyS), anxiety (STAI-YA, BIS-10), and biology (metabolic panel) reflects the clinical continuum between bipolar depression and treatment-resistant unipolar depression. Patients with bipolar II disorder presenting in a depressive episode are historically difficult to distinguish from treatment-resistant depression patients — many DR patients are eventually rediagnosed as BP-II upon emergence of hypomanic symptoms.

The graph captures this ambiguity: BP patients with high MADRS and STAI scores connect to DR patients through the mood and anxiety layers, while BP patients currently manic (high YMRS, low MADRS) do not. The graph encodes *state-dependent similarity*, not trait diagnosis. This is clinically important: if Stage B clustering places some BP-II depressed patients in the same cluster as DR patients, it would suggest that these patients share a treatment-relevant phenotype regardless of their nosological label.

The anxiety_impulsivity block is particularly informative: its cross-cohort edges have the highest weight ratio (1.36×) of any block, meaning BP↔DR anxiety-impulsivity similarity is *stronger* than within-cohort similarity. This suggests a subgroup of BP and DR patients who share an anxiety-impulsivity dimension that is more prominent than what is typical within either diagnostic group alone — potentially a high-impulsivity, high-anxiety phenotype relevant to treatment selection (e.g., preference for mood stabilizers with anxiolytic properties, caution with activating antidepressants).

#### Treatment follows nosology

The treatment block (assortativity +0.655) is the most assortative multi-cohort block. This makes clinical sense: medication regimens are prescribed **based on diagnosis**, not on transdiagnostic symptom profiles. BP patients receive mood stabilizers and atypical antipsychotics; SZ patients receive antipsychotics with different dosing profiles; DR patients receive antidepressant combinations; ASP patients may receive various off-label medications. Even when two patients from different cohorts have similar symptom profiles, their medication regimens will differ because guidelines are diagnosis-specific.

This finding has a practical implication for Stage B: if clusters are discovered that mix BP and SZ patients (through the cognitive bridge), those clusters will have **heterogeneous treatment profiles** within them. This is not a weakness — it is a hypothesis-generating finding. A "cognitively impaired" transdiagnostic cluster containing both BP and SZ patients, who are treated differently despite similar cognitive profiles, raises the question of whether cognitive-profile-matched treatment strategies could be more effective than diagnosis-matched ones.

#### ASP isolation reflects genuine measurement heterogeneity

ASP patients connect to other cohorts only through thin universal blocks (comorbidities, substance, demographics, functioning). This isolation is not an artifact of parameter tuning — it reflects the fundamental reality that autism assessment uses a completely different clinical battery (BRIEF, RBS-R, ADI-R, WAIS-IV indices, ADHD-RS) from the mood/psychosis batteries used for BP, SZ, and DR.

Clinically, this raises an important question about the scope of "transdiagnostic" stratification. The FACE consortium measures autism on fundamentally different dimensions than it measures mood or psychosis. Cognitive tests (TMT, CVLT, Stroop) — which are the strongest bridges for BP↔SZ — are not administered to ASP patients. If the consortium were to add a shared cognitive battery for ASP, the cross-cohort connectivity would likely increase substantially, given that cognitive variability in autism is well-documented.

For the current analysis, ASP's relative isolation means that any clusters containing ASP patients will be driven primarily by demographics, substance use, comorbidity burden, and functioning — coarse dimensions that may not reflect ASP-specific clinical phenotypes. Within-ASP clustering (driven by the 32-feature autism_profile block) will provide fine-grained autism subtypes, but these subtypes will not directly compare to BP/SZ/DR subtypes in the cross-cohort embedding.

#### The dual structure: bridging blocks for phenotypes, cohort-specific blocks for resolution

The multiplex graph encodes a dual structure that mirrors an important clinical distinction:

- **16 bridging blocks** (demographics through psychiatric_history, plus transdiagnostic) determine *whether* patients from different diagnostic groups share clinical features. These blocks test the transdiagnostic hypothesis — do some BP patients resemble some SZ patients more than they resemble other BP patients?

- **6 cohort-specific blocks** (psychosis, insight, hostility, autism_profile, personality, treatment_resistance) determine *how* patients within a diagnostic group differ from each other. These blocks provide intra-cohort resolution — within SZ, they separate patients with good vs poor insight, predominantly positive vs predominantly negative symptoms, high vs low extrapyramidal burden.

Both roles are essential. A clustering that only uses bridging blocks would find cross-diagnostic groupings but lack the resolution to distinguish clinically meaningful subtypes within each diagnosis. A clustering that only uses cohort-specific blocks would produce DSM-aligned subtypes but miss the cross-diagnostic patterns. The multiplex embedding integrates both, and the balance between them is determined by the block weight normalization (equal total weight) and the number of blocks in each category (16 bridging vs 6 cohort-specific).

This 16:6 ratio predicts a moderate bias toward cross-diagnostic clustering: the transdiagnostic signal has more "votes" in the multiplex than the within-cohort signal. Whether this produces genuinely novel phenotypes or simply reflects instrument overlap between cohorts is the central question for Stage B validation (particularly the DSM subtype recovery test and leave-one-cohort-out stability analysis).

## What the graph cannot tell us

The graph encodes **observed similarity** — it does not encode causation, prognosis, or treatment response. Two patients connected by a strong mood edge share similar depression severity today, but may have entirely different illness trajectories. The graph does not model temporal dynamics.

Furthermore, the graph's cross-cohort bridges are constrained by which instruments happen to be shared. The BP↔SZ bridge through cognition is strong because both cohorts receive the same cognitive battery — but this is a data collection decision, not necessarily a statement about which clinical dimension is most important for cross-diagnostic similarity. If both cohorts also received the same anxiety instruments (e.g., STAI-YA administered to SZ), the BP↔SZ bridge might include an anxiety layer that could be equally or more informative.

The graph is an honest representation of what the FACE data supports, not an unbiased estimate of all possible cross-diagnostic similarities.

## What the three tiers tell us

### Tier 1 — Strongly transdiagnostic blocks
**Cognition, neuropsych, substance, comorbidities**

These blocks say: **"diagnosis doesn't predict who your nearest neighbours are."**

Cognition (assortativity −0.75) and neuropsych (−0.74) are the headline finding. A BP patient with slow TMT-B and poor CVLT recall is closer to an SZ patient with the same cognitive profile than to a cognitively intact BP patient. The kNN algorithm doesn't know which cohort a patient belongs to — it just finds the nearest neighbours in the shared feature space, and in these blocks, the nearest neighbour is more often from a *different* diagnostic group.

**What this means clinically:** Cognitive impairment is a genuine *dimensional* trait that cuts across the bipolar–schizophrenia boundary. This is the core claim of the RDoC framework and of the "psychosis continuum" literature — and the graph materializes it empirically from FACE data. If Stage B clustering finds a "cognitively impaired" cluster containing both BP and SZ patients, that's not a confound; it's the transdiagnostic phenotype we're looking for.

Substance and comorbidities also mix freely across cohorts, but they carry less clinical specificity — smoking rates and comorbidity counts don't distinguish psychiatric subtypes well. They're transdiagnostic but *shallow*.

**Bottom line for Tier 1:** Cognition is the strongest empirical argument that DSM categories don't carve nature at its joints. The other Tier 1 blocks confirm cross-cohort mixing but at lower clinical resolution.

---

### Tier 2 — Selective bridge blocks
**Mood, anxiety_impulsivity, biology, functioning, sleep, trauma, psychiatric_history**

These blocks say: **"specific cohort *pairs* share a clinical dimension, but not all cohorts mix."**

Unlike Tier 1 where everyone mixes, Tier 2 blocks create *targeted* bridges:
- **Mood** (MADRS, QIDS, MAThyS) bridges **BP ↔ DR** — bipolar depressed patients connect to treatment-resistant depression patients through shared depression severity. But SZ and ASP patients don't participate meaningfully in this bridge.
- **Anxiety_impulsivity** (STAI, BIS-10) bridges **BP ↔ DR** with the highest weight ratio of any block (~1.36×). This is striking: the cross-cohort edges are *stronger* than intra-cohort ones. It means there's a subgroup of BP + DR patients who share an anxiety-impulsivity profile that is more extreme than what's typical within either diagnosis alone.
- **Biology** (metabolic panel) bridges **BP ↔ DR** and **SZ ↔ DR** — metabolic risk is shared across mood and psychosis.

**What this means clinically:** Tier 2 reveals the *topology* of diagnostic overlap. BP and DR are closest (they're both mood disorders — this makes clinical sense). BP and SZ bridge through cognition (Tier 1) but *not* through mood or anxiety (different symptom profiles). DR and SZ barely connect at all — they share metabolic labs but little else.

The anxiety_impulsivity finding is particularly actionable: a high-anxiety, high-impulsivity subgroup spanning BP and DR might benefit from the same treatment approach (mood stabilizers with anxiolytic properties, caution with activating antidepressants) regardless of which DSM label they carry.

**Bottom line for Tier 2:** These blocks tell us *where* the diagnostic boundaries are blurry and *where* they hold. The graph doesn't say "all diagnoses are the same" — it says "BP and DR overlap on mood/anxiety, BP and SZ overlap on cognition, and DR and SZ barely overlap at all."

---

### Tier 3 — Cohort-specific blocks
**Psychosis, insight, hostility, autism_profile, personality, treatment_resistance**

These blocks say: **"within each diagnosis, patients differ on dimensions that are invisible to other cohorts."**

Every edge is intra-cohort (assortativity = 1.0). PANSS distinguishes positive-predominant from negative-predominant SZ. SUMD separates insightful from anosognostic SZ patients. BRIEF and RBS-R differentiate autism subtypes. BFI personality and treatment resistance staging characterize DR subgroups.

**What this means clinically:** These blocks don't contribute to the transdiagnostic question at all — but they're essential for the *other* half of the scientific question: **within-cohort resolution**. A clustering that only uses Tier 1 + 2 blocks would find cross-diagnostic groups but couldn't distinguish a paranoid SZ patient from a disorganized one, or a rigid-routine ASP patient from a sensory-seeking one.

**Bottom line for Tier 3:** These blocks are the reason the graph is *multiplex* rather than a single-layer cross-cohort graph. They preserve the clinical subtlety within each diagnostic group.

---

### The three tiers together

The tiers compose into a **dual-purpose graph**:

| Purpose | Driven by | Blocks | Effect on clustering |
|---|---|---|---|
| Cross-diagnostic phenotyping | Tier 1 + 2 (16 blocks) | Cognition, mood, anxiety, biology, ... | Mix patients across DSM boundaries |
| Within-diagnostic subtyping | Tier 3 (6 blocks) | PANSS, SUMD, BRIEF, BFI, ... | Separate subtypes within each diagnosis |

The 16:6 ratio means the graph carries a moderate **transdiagnostic bias** — there are more "votes" for cross-cohort mixing than for within-cohort separation. Whether this produces genuinely novel phenotypes or just reflects instrument overlap is the central question for Stage B.

The key prediction: **Stage B clustering will produce clusters that are partially cross-diagnostic** (driven by Tier 1 cognition bridges between BP and SZ, and Tier 2 mood bridges between BP and DR), **with ASP forming relatively isolated sub-clusters** (it only connects through thin Tier 1 blocks), and **with within-cluster heterogeneity on Tier 3 dimensions** (e.g., a mixed BP+SZ cognitive cluster will still contain patients with very different PANSS or hostility profiles).

## Limitations

- **Coverage inhomogeneity**: ASP emits 69/184 features (38%), BP emits 104/184 (57%). The autism cohort will have sparser similarity graphs and may form relatively isolated communities in the multiplex embedding.
- **Trans-cohort set is clinically thin**: 9 features covering only demographics, substance use, comorbidities, and polypharmacy. No direct mood, cognitive, or symptom severity measures cross all 4 cohorts, limiting the strength of cross-cohort patient comparisons.
- **Floor effects in some subscales**: RBS-R self-injury, CTQ sexual abuse, and several BRIEF subscales pile up near zero. These features have low discriminative power for the majority of patients and primarily identify a small high-scoring subgroup.
- **WAIS-IV ICV in ASP**: Only ~16% coverage — too sparse for reliable similarity computation. The other 3 WAIS-IV indices (IRP 53%, IMT, IVT) have moderate but usable coverage.
- **Personality (BFI)** is DR-only: neuroticism is a well-established transdiagnostic risk factor but cannot be compared across cohorts without the instrument being administered to BP, SZ, or ASP.
- **No longitudinal features**: The schema is V1-only (baseline visit). BP has partial follow-up data (`_n1` suffix) but it is intentionally excluded to maintain the cross-sectional design. Longitudinal trajectory features would require a separate pipeline.

## Conclusion

Stage A transforms 4 heterogeneous psychiatric cohorts into a single, well-defined multiplex patient graph — the input to all downstream embedding and clustering. The pipeline has three layers:

1. **Feature engineering**: 184 features across 21 clinical blocks, spanning instrument totals, clinical sub-scales, and derived composites. The critical fix of the French subscale keys means that CTQ trauma dimensions, BIS-10 impulsivity facets, BFI personality traits, and PANSS Wallwork factors now carry real patient data instead of the all-None values silently produced by the previous adapters.

2. **Trans-cohort analysis**: Only 9 features pass the strict all-4-cohort threshold. The four cohorts share very few instruments — this reflects clinical reality, not a data quality issue. The cohort-pair topology (BP↔DR through mood, BP↔SZ through cognition, SZ↔DR through metabolic labs) predicts where cross-diagnostic bridges will form.

3. **Graph construction**: 22 edge types (21 blocks + 1 transdiagnostic), 1,154,471 edges, 0 dead blocks. Parameter tuning created 4 new cross-cohort bridges (cognition, neuropsych, anxiety, functioning) where none existed before. Per-block weight normalization eliminated a 490× imbalance. The graph solves the three fundamental data-level problems (structured missingness, feature incomparability, mixed types) deterministically and with minimal parameters. No imputation is ever performed.

The graph is the contract between Stage A and Stage B. Stage B receives a weighted multiplex graph and produces patient embeddings — it never touches the raw feature matrix, the missingness patterns, or the per-cohort adapters. This separation means embedding architectures can iterate freely (PCA → spectral → GNN → contrastive SSL) without re-running harmonization or graph construction.

The key scientific question for Stage B is whether this multiplex structure — with domain-specific cross-cohort bridges and ASP as a relatively isolated community — is sufficient to discover data-driven phenotypes that cut across diagnostic boundaries, or whether the measurement heterogeneity will cause the clusters to recapitulate the cohort labels.

## Next Steps

- Run the full pipeline with the expanded schema: block graph construction → multiplex assembly → spectral/GNN embedding → clustering.
- Evaluate whether the block restructuring (`cohort_specific` → 4 domain blocks) improves within-cohort graph connectivity and edge quality.
- DSM subtype recovery test: do discovered clusters separate BP-I from BP-II, paranoid from disorganized SZ, or do they reflect entirely novel groupings?
- Leave-one-cohort-out stability: remove each cohort in turn, re-cluster, measure assignment consistency. If clusters are truly cross-diagnostic, they should be stable.
- SHAP feature importance per cluster: identify which features drive each cluster assignment, producing clinically interpretable cluster signatures.
- Consider relaxing the trans-cohort threshold to build a tiered graph (Tier 1 at ≥50% all 4 cohorts, Tier 2 at ≥50% in 3 cohorts, Tier 3 at ≥50% in 2 cohorts) to enrich the cross-cohort edges, particularly between BP↔DR and BP↔SZ.
