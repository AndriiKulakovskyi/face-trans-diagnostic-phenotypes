#!/usr/bin/env python
"""
Regenerate the full article_v2 figure deck, in place, into article_v2/figures/.

article_v2 is the main article. This wrapper and every generator it calls live
under article_v2/figures/scripts/ and write into article_v2/figures/, so the
manuscript is self-contained: no dependency on the legacy article/ folder.

Run with the repo venv:
    cd article_v2 && ../.venv/bin/python figures/scripts/regenerate_figures.py

The generator function names predate the manuscript numbering; MANIFEST below is
the single source of truth mapping manuscript figure -> generator -> output file.
Generators that emit a non-manuscript filename are renamed in place after they run.
"""
import os
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

_HERE = os.path.dirname(os.path.abspath(__file__))            # article_v2/figures/scripts
FIG = os.path.dirname(_HERE)                                   # article_v2/figures
MANUSCRIPT = os.path.dirname(FIG)                              # article_v2
ROOT = os.path.dirname(MANUSCRIPT)                             # repo root
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, _HERE)                                      # so `import make_figures_copula` resolves

# MANIFEST: manuscript label -> (final filename in article_v2/figures,
#                                generator-emitted filename, "module:function" | "script:file")
MANIFEST = [
    ("Fig 1 fig:atlas",           "fig1_atlas.png",              "fig2_map.png",              "make_figures_copula:fig2_map"),
    ("Fig 2 fig:continuum",       "fig2_continuum.png",          "fig4_continuum.png",        "make_figures_copula:fig4_continuum"),
    ("Fig 3 fig:hinge",           "fig3_biology_g.png",          "fig3_biology_g.png",        "make_figures_copula:fig3_biology_g"),
    ("Fig 4 fig:persistence",     "fig4_persistence.png",        "fig5_persistence.png",      "make_figures_copula:fig5_persistence"),
    ("Fig 5 fig:money",           "fig5_archetype_prognosis.png","fig5_archetype_prognosis.png","script:fig5_archetype_prognosis.py"),
    ("Fig 6 fig:prognosis_quant", "fig6_prognosis_quant.png",    "fig6_prognosis_rebuilt.png","script:fig6_prognosis_rebuilt.py"),
    ("ED E1 edfig:consort",       "edfig_consort.png",           "edfig_consort.png",         "make_figures_copula:edfig_consort"),
    ("ED E2 edfig:fullatlas",     "edfig_fullatlas.png",         "edfig_full_atlas.png",      "make_figures_copula:edfig_full_atlas"),
    ("ED E3 edfig:invariance",    "edfig_invariance.png",        "edfig_invariance.png",      "make_figures_copula:edfig_invariance"),
    ("ED E4 edfig:loso",          "edfig_loso.png",              "edfig_loso.png",            "script:edfig_loso.py"),
    ("ED E5 edfig:repbench",      "edfig_repbench.png",          "edfig_repbench.png",        "make_figures_copula:edfig_repbench"),
    ("ED E6 edfig:robustness",    "edfig_robustness.png",        "edfig_robustness.png",      "make_figures_copula:edfig_robustness"),
]

def _fig1_with_flagship_title():
    """Fig 1's embedded title must match its flagship caption; override at render
    time so the shared module default (used by the technical report) is untouched."""
    import make_figures_copula as m
    import matplotlib.pyplot as plt

    from face.reporting import loading_atlas as LA
    L = m._load_loadings()
    rows = LA.atlas_rows(L, m.AXES, 8)
    fig, axA = plt.subplots(figsize=(9.6, max(8.5, 2.4 + 0.135*len(rows))))
    sc = LA.draw_dot_atlas(axA, L, m.AXES, rows, **m.ASTYLE)
    axA.set_title("A diagnosis-blind, biology-inclusive map of severe mental illness",
                  loc="left", fontsize=12, fontweight="bold")
    LA.atlas_legends(fig, axA, sc, window_color=m.OI["yellow"])
    fig.savefig(os.path.join(FIG, "fig2_map.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "fig2_map.pdf"), bbox_inches="tight")

def main():
    import importlib
    m = importlib.import_module("make_figures_copula")
    done = set()
    for label, final, emitted, gen in MANIFEST:
        kind, ref = gen.split(":", 1)
        if label.startswith("Fig 1"):
            _fig1_with_flagship_title()
        elif kind == "make_figures_copula":
            if ref not in done:
                getattr(m, ref)()
                done.add(ref)
        elif kind == "script":
            subprocess.run([sys.executable, os.path.join(_HERE, ref)], check=True, cwd=MANUSCRIPT)
        # rename generator-emitted file -> manuscript name (in place, same dir)
        if emitted != final:
            shutil.move(os.path.join(FIG, emitted), os.path.join(FIG, final))
        assert os.path.exists(os.path.join(FIG, final)), f"missing {final}"
        print(f"  {label:28s} -> {final}")
    print("\nAll 12 manuscript figures regenerated into article_v2/figures/.")

if __name__ == "__main__":
    main()
