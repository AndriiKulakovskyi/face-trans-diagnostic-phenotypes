# FACE-ATLAS technical report — build
# Usage:  make            (regenerate figures, then build the PDF — two passes for TOC/refs)
#         make figures    (regenerate the data-driven figures from committed aggregate CSVs)
#         make pdf         (build the PDF only, assuming figures exist)
#         make clean      (remove LaTeX aux files)
#         make cleanall   (also remove the PDF)
PDF  = FACE-ATLAS.pdf
MAIN = FACE-ATLAS.tex
PY   ?= python3
SRC  = $(MAIN) faceatlas.sty references.tex $(wildcard sections/*.tex) $(wildcard figures/*.png)

.PHONY: all figures pdf clean cleanall
all: figures pdf

figures:
	$(PY) make_figures.py

pdf: $(PDF)

$(PDF): $(SRC)
	pdflatex -interaction=nonstopmode $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN)

clean:
	rm -f *.aux *.log *.out *.toc sections/*.aux
cleanall: clean
	rm -f $(PDF)
