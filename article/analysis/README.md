# Analysis code supporting the article

- `run_loso_validation.py` — leave-one-site-out external-validity runner. Refits the
  variational GLLVM (VI path, not NUTS) per held-out site, exports per-fold loading
  congruence (Tucker φ) and the immunometabolic-vs-general-burden score decoupling.
  Produces the numbers behind the LOSO Results subsection and `edfig_loso.png`.
  (Model-training code itself lives in `src/face/models/variational/`, already tracked.)
- `canonical_numbers.md` — single-source-of-truth number reference used during the rewrite.
