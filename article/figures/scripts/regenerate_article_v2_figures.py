#!/usr/bin/env python
"""
Regenerate the full article_v2 figure deck from source, into article/figures/,
then copy to article_v2/figures/ under the manuscript figure names.

This wrapper resolves the source-file / manuscript-number drift: each manuscript
figure is produced by a generator function whose *internal* name does not match
the manuscript number (historical). The MANIFEST below is the single source of
truth mapping manuscript figure -> generator -> output file.

Run with the repo venv from the article/ directory:
    cd article && ../.venv/bin/python figures/scripts/regenerate_article_v2_figures.py

Main figures 1-4 come from make_figures_copula.py (the frozen copula fit).
Figure 5 (money) is its own standalone script (fig5_archetype_prognosis.py).
Figure 6 is the treatment-free rebuild (fig6_prognosis_rebuilt.py).
Extended Data E1-E6 come from make_figures_copula.py.
"""
import os
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

_HERE = os.path.dirname(os.path.abspath(__file__))
ARTICLE = os.path.dirname(os.path.dirname(_HERE))          # article/
ROOT = os.path.dirname(ARTICLE)                             # repo root
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ARTICLE)

FIG = os.path.join(ARTICLE, "figures")
V2FIG = os.path.join(ROOT, "article_v2", "figures")

# MANIFEST: manuscript label -> (article_v2 filename, source PNG in article/figures,
#                                generator "module:function" or "script:path")
MANIFEST = [
    ("Fig 1 fig:atlas",            "fig1_atlas.png",              "fig2_map.png",             "make_figures_copula:fig2_map"),
    ("Fig 2 fig:continuum",        "fig2_continuum.png",          "fig4_continuum.png",       "make_figures_copula:fig4_continuum"),
    ("Fig 3 fig:hinge",            "fig3_biology_g.png",          "fig3_biology_g.png",       "make_figures_copula:fig3_biology_g"),
    ("Fig 4 fig:persistence",      "fig4_persistence.png",        "fig5_persistence.png",     "make_figures_copula:fig5_persistence"),
    ("Fig 5 fig:money",            "fig5_archetype_prognosis.png","fig5_archetype_prognosis.png","script:fig5_archetype_prognosis.py"),
    ("Fig 6 fig:prognosis_quant",  "fig6_prognosis_quant.png",    "fig6_prognosis_rebuilt.png","script:fig6_prognosis_rebuilt.py"),
    ("ED E1 edfig:consort",        "edfig_consort.png",           "edfig_consort.png",        "make_figures_copula:edfig_consort"),
    ("ED E2 edfig:fullatlas",      "edfig_fullatlas.png",         "edfig_full_atlas.png",     "make_figures_copula:edfig_full_atlas"),
    ("ED E3 edfig:invariance",     "edfig_invariance.png",        "edfig_invariance.png",     "make_figures_copula:edfig_invariance"),
    ("ED E4 edfig:loso",           "edfig_loso.png",              "edfig_loso.png",           "script:edfig_loso.py"),
    ("ED E5 edfig:repbench",       "edfig_repbench.png",          "edfig_repbench.png",       "make_figures_copula:edfig_repbench"),
    ("ED E6 edfig:robustness",     "edfig_robustness.png",        "edfig_robustness.png",     "make_figures_copula:edfig_robustness"),
]

# Fig 1 needs the flagship title override (embedded title must match the caption).
def _fig1_with_flagship_title():
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
    for label, v2name, srcpng, gen in MANIFEST:
        kind, ref = gen.split(":", 1)
        if label.startswith("Fig 1"):
            _fig1_with_flagship_title()                 # title-overridden variant
        elif kind == "make_figures_copula":
            if ref not in done:
                getattr(m, ref)()
                done.add(ref)
        elif kind == "script":
            subprocess.run([sys.executable, os.path.join(_HERE, ref)], check=True,
                           cwd=ARTICLE)
        # copy source PNG -> article_v2 name
        shutil.copy(os.path.join(FIG, srcpng), os.path.join(V2FIG, v2name))
        print(f"  {label:28s} <- {srcpng:28s} -> article_v2/figures/{v2name}")
    print("\nAll 12 manuscript figures regenerated and installed into article_v2/figures/.")

if __name__ == "__main__":
    main()
