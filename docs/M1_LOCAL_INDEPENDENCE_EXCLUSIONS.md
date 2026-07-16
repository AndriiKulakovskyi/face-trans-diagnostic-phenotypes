# M1 local-independence deletion profile

Decision date: 2026-07-15

The corrected-v3 M1 diagnostic exposed severe residual dependence within repeated
instruments, positional vital signs, and overlapping assay families.  The current
deletion-only model retains one prespecified representative from each cluster and
excludes the alternatives before constructing any data arrays or likelihood terms.
Its refit is versioned as `corrected_v4_composite_independence` so the failed
pre-composite diagnostic remains available for provenance.

| Cluster | Retained | Excluded | Selection rule |
|---|---|---|---|
| CVLT | `cvlt_total_recall` | `cvlt_short_delay_free_recall`, `cvlt_long_delay_free_recall` | Broad cognition target; avoid triple weighting one test family. |
| WAIS processing speed | `wais_code_std` | `wais_ivt_index` | Direct subtest, greater coverage; avoid component/composite overlap. |
| CTQ burden | `ctq29`, `ctq31`, `ctq33`, `ctq35`, `ctq37` | `ctq39` | `ctq39` is exactly the sum of the five retained CTQ subscales in all 8,122 jointly observed baseline records. Retaining the subscales preserves domain information without deterministically duplicating their total. |
| CTQ minimization/denial | `ctq40` | `ctq41` | In every jointly observed record, `ctq41 = 1(ctq40 > 0)`. Retain the more informative ordinal score and remove its binary recode. |
| Sleep quality | `psqi11`, `psqi12`, `psqi13`, `psqi14`, `psqi15`, `psqi17` | `psqi` | The PSQI total contains the retained component scores; keep the components and avoid weighting the same responses again through the total. |
| Functional impairment | `fast25`, `fast26`, `fast27`, `fast28`, `fast30` | `fast` | The FAST total contains the retained domain scores; keep the domains and avoid total/component overlap. |
| Anthropometry | `bmi`, `wstcir` | `weight` | BMI represents overall adiposity and waist circumference central adiposity. Raw weight is height-dependent and adds a third strongly overlapping body-size indicator. |
| Heart rate | `hrsupine` | `hrstanding`, `eghrmn` | Resting measurement with the greatest coverage. |
| Systolic blood pressure | `sysbpsupine` | `sysbpstanding` | Resting measurement with the greatest coverage. |
| Diastolic blood pressure | `diabpsupine` | `diabpstanding` | Resting measurement with the greatest coverage. |
| White-cell measures | `wbc` | `neut` | Avoid total/component double weighting. |
| Lipids | `ldl` | `chol` | Retain the more clinically actionable lipid measure. |
| Liver enzymes | `alt_lbstresc` | `ast_lbstresc` | ALT is more liver-specific when only one marker is retained. |

## Interpretation constraint

This profile is a model-specification decision, not evidence that every excluded
indicator is intrinsically invalid.  CVLT delay scores, total scores, raw weight,
standing vital signs, and AST contain clinically meaningful information that the
broad eight-coordinate map does not separately represent.  CTQ total/subscale and
CTQ ordinal/binary relationships are deterministic in the model-ready baseline
data.  PSQI and FAST exclusions instead enforce a prespecified component-over-total
granularity; they are not claims that the totals are invalid clinical measures.  LDL
assay provenance is unresolved: the source data dictionary must establish whether
LDL is directly assayed or calculated before the manuscript calls the
cholesterol/LDL relationship mathematical duplication.

The exclusion profile is accepted only if the staged refit converges, improves
chain-wise loading congruence, reduces posterior-predictive residual dependence,
and leaves the retained factor scores stable.  Old M1 artifacts remain noncanonical.

## Separate data-quality quarantine

The corrected-v5 primary recipe additionally excludes `suoccur_alcool` and
`suoccur_cannabis`, for a reason distinct from local independence. The aggregate
audit in `reports/00_substance_harmonization_audit.md` shows that the current
CSV export did not reliably unpack the lifetime substance-disorder branches:

- in BP, 1,647 of 1,880 parent-positive baseline records have no positive
  substance child, and many alcohol/cannabis lifetime negatives conflict with
  recorded current symptoms or disorder-onset ages;
- in SZ, both baseline summaries are constant negative despite positive parent
  and detailed diagnostic branches; later visits also expose unpacked false
  negatives against the retained checkbox text;
- DR has no comparable lifetime alcohol/cannabis disorder fields.

The exposure variables `suoccur_alcoollt` and `suoccur_cannabislt` are not valid
replacements because they ask whether the patient consumed the substance, not
whether a disorder was present. Detailed criterion counts are retained as
contradiction evidence but are not converted into diagnoses without a validated
scoring and time-window rule. Consequently, both summary indicators are
quarantined from primary M1 until a corrected upstream export or clinical data
adjudication is available.

The smoking branch is handled separately. Fagerstrom remains missing outside
current smokers because the instrument is not applicable there. Missing
pack-years are set to structural zero only for a known never-smoker; existing
values are never overwritten.
