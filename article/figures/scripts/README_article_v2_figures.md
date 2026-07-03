# article_v2 figure-generation code

Every figure in `article_v2/` is reproducible from code in this repository. This
directory is the single source of truth for the figure→script mapping (the
generator function names predate the manuscript numbering, so the mapping is made
explicit here rather than left to filename inference).

## One-command regeneration

```bash
cd article
../.venv/bin/python figures/scripts/regenerate_article_v2_figures.py
```

This regenerates all 12 figures into `article/figures/` and copies them into
`article_v2/figures/` under their manuscript names.

## Mapping (manuscript figure → generator → output)

| Manuscript | label | generator | source PNG | → article_v2 |
|-----------|-------|-----------|------------|--------------|
| Fig 1 | fig:atlas | `make_figures_copula:fig2_map` (+ flagship-title override in the wrapper) | fig2_map.png | fig1_atlas.png |
| Fig 2 | fig:continuum | `make_figures_copula:fig4_continuum` | fig4_continuum.png | fig2_continuum.png |
| Fig 3 | fig:hinge | `make_figures_copula:fig3_biology_g` | fig3_biology_g.png | fig3_biology_g.png |
| Fig 4 | fig:persistence | `make_figures_copula:fig5_persistence` | fig5_persistence.png | fig4_persistence.png |
| **Fig 5** | **fig:money** | **`scripts/fig5_archetype_prognosis.py`** (standalone) | fig5_archetype_prognosis.png | fig5_archetype_prognosis.png |
| Fig 6 | fig:prognosis_quant | `scripts/fig6_prognosis_rebuilt.py` (treatment-free) | fig6_prognosis_rebuilt.png | fig6_prognosis_quant.png |
| ED E1 | edfig:consort | `make_figures_copula:edfig_consort` | edfig_consort.png | edfig_consort.png |
| ED E2 | edfig:fullatlas | `make_figures_copula:edfig_full_atlas` | edfig_full_atlas.png | edfig_fullatlas.png |
| ED E3 | edfig:invariance | `make_figures_copula:edfig_invariance` | edfig_invariance.png | edfig_invariance.png |
| ED E4 | edfig:loso | `scripts/edfig_loso.py` | edfig_loso.png | edfig_loso.png |
| ED E5 | edfig:repbench | `make_figures_copula:edfig_repbench` | edfig_repbench.png | edfig_repbench.png |
| ED E6 | edfig:robustness | `make_figures_copula:edfig_robustness` | edfig_robustness.png | edfig_robustness.png |

## Data sources

- Main copula figures read `reports/*.csv` and `results/face/{strata_oop,temporal_oop,prognosis_oop}/…`.
- **Fig 5 (money)** reads `results/face/prognosis_oop/consolidate/prognosis_patient_risk.parquet`
  — per-patient archetype weights and two-year outcomes together (N=9,013), so no join.
- **Fig 6** reads `results/face/prognosis_oop/endpoints/archetype_atlas.csv`.
- **ED E4 (LOSO)** reads `figures/data/loso_summary_production.csv` (a committed copy of a
  summary that otherwise lives only in the artifact store).

## Notes on this session's hygiene pass (2026-07-03)

- Fixed a real label error in `src/face/reporting/loading_atlas.py`: the Fig 1 subtitle
  said "specific axes D1–D8" but the model has 7 specific axes → corrected to **D1–D7**.
- Harmonized Fig 1's embedded title to its flagship caption (done in the wrapper's
  `_fig1_with_flagship_title`, so the shared module default is untouched).
- Fixed a label/annotation collision in Fig 4b (`fig5_persistence`): per-point label
  placement + moved the explanatory note to the empty bottom-right.
- Made `fig5_archetype_prognosis.py`, `fig6_prognosis_rebuilt.py` and `edfig_loso.py`
  path-robust (derive paths from `__file__`, write to `article/figures/` explicitly)
  so they no longer depend on the current working directory or a machine-specific path.
