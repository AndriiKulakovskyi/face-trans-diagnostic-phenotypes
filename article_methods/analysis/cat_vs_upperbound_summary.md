# CAT (patient-adaptive) vs population-mean upper-bound reliability — FACE-ATLAS methods paper

Across all eight dimensional axes, a fully patient-adaptive test (CAT: Fisher information re-evaluated at each
patient's provisional posterior mean, 500 simulated patients drawn from the standardized N(0,1) factor prior,
seed 20240704) reaches the **same items-to-reliability-0.80 as the fixed population-mean order** on every axis — the
fixed "most-informative-at-the-mean" battery is already near-optimal, so the article's upper-bound framing holds.

For the five Gaussian-topped axes (overall_severity, cognition, immunometabolic, sleep, developmental_risk),
realized CAT reliability matches the upper bound to **within 0.0002 reliability at every step**: Gaussian item
information does not depend on theta, so re-ordering by the provisional estimate cannot change the curve. Items to
reach reliability 0.80 are therefore unchanged — immunometabolic 2, developmental risk 1, cognition and sleep 3,
overall severity 6 — under both methods.

The two axes whose banks contain Bernoulli/ordinal items show a small, genuine adaptive signature.
**Suicidality** (19 of its top-20 items are Bernoulli/ordinal): CAT shows an early-estimation *penalty* of −0.012
at 2 items (the provisional theta is still noisy), then **exceeds** the fixed order by ~+0.003 from 4 items onward.
**Substance**: the two bank-capping Bernoulli items give CAT a +0.0015 edge at the 4-item ceiling. Both effects
are below the resolution that would change items-to-0.80 (both axes still behave as reported: suicidality crosses
0.80 at 2 items; substance never reaches 0.80). The remaining axes match to within 0.0002.

The two bank-limited axes remain capped regardless of adaptivity: **mania/activation** tops out at **0.408**
(2 items in the bank) and **substance** at **0.429** (4 items) — a battery-content limitation, not an algorithmic
one, so neither the fixed order nor CAT can reach 0.80.

**Definition of "ceiling" (read before comparing to the plotted curve):** the per-axis `ceiling` reported in the
results table and metadata is the **full-item-bank reliability** — the value reached when the ENTIRE candidate pool
is administered. The saved curves in `cat_vs_upperbound.csv` and the preview figure are capped at min(20, pool)
items per the analysis spec, so for the three axes whose pool exceeds 20 the plotted n=20 endpoint is BELOW the
reported full-bank ceiling: overall_severity 0.874 (n=20) vs 0.881 (full 138-item bank); immunometabolic 0.886 vs
0.888 (42 items); suicidality 0.912 vs 0.914 (32 items). For every axis with pool ≤ 20 (cognition, sleep,
developmental_risk, mania, substance) the ceiling equals the plotted endpoint. Both values are tabulated per axis
in `cat_vs_upperbound_meta.json` (`full_pool_ceiling` and `reliability_at_item_cap`).

**Headline:** patient-adaptive administration does not reduce the number of items needed versus the population-mean
upper bound on these eight axes — the fixed most-informative order is already near-optimal at the prior mean, and
the small realized CAT gains (≤0.003 reliability, confined to the two Bernoulli-containing banks: suicidality and
substance) confirm rather than overturn the article's use of that fixed-order curve as a tight upper bound on
adaptive efficiency.
