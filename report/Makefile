# FACE-ATLAS technical report — build
# Usage:  make            (build the PDF, two passes for TOC/refs)
#         make clean      (remove LaTeX aux files)
#         make cleanall   (also remove the PDF)
PDF  = FACE-ATLAS.pdf
MAIN = FACE-ATLAS.tex
SRC  = $(MAIN) faceatlas.sty references.tex $(wildcard sections/*.tex) figures/prior_atlas.png

$(PDF): $(SRC)
	pdflatex -interaction=nonstopmode $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN)

.PHONY: clean cleanall
clean:
	rm -f *.aux *.log *.out *.toc sections/*.aux
cleanall: clean
	rm -f $(PDF)
