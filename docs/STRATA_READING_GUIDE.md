# M2 reading guide — how to read the transdiagnostic map

> **What M2 is (reframed).** M2 tested whether the FACE map contains patient *types* (biotypes) and found it
> does **not** — the 9-dim copula space is a **continuum** (single-Gaussian falsification null). So the
> load-bearing object of M2 is **not a typology**; it is the **continuous coordinate system** itself — each
> patient's validated, transdiagnostic position on nine axes, with uncertainty — plus this **reading guide** and a
> **patient-similarity** tool for interpreting positions. The four "archetype" corners and the K-region
> tessellation are **interpretation lenses** on that continuum, not discovered subgroups. (M4 confirms this
> empirically: *operative K = none* — no hard partition adds predictive value beyond the continuous coordinates.)

## How to read a patient
A patient is a **point** on the map. Three complementary lenses describe where they are:
1. **Coordinates** (the truth) — their score on each of the nine axes, with error bars. This is what M3/M4/M5 use.
2. **The four corners** (§ reading guide) — the poles of the cloud; a patient is a *blend* of them ("pulled toward
   the biological corner"). The corners carry the biology⊥severity structure.
3. **Nearest neighbours** (§ similarity tool) — the real patients most similar to this one.

A **corner** is a *pure extreme* almost nobody sits exactly at; its **typical representative** is the centroid of
patients who lean toward it (what a representative patient actually looks like); its **medoid** is a real exemplar
patient. Reporting the triplet keeps both the defining signal (pole) and the realism (typical member + real person).

## The four representative profiles (Arm A, all nine axes)

![reading guide](figures/strata_oop/reading_guide.png)

| corner | plain clinical label | share | cohorts (bp/sz/dr) | defining signal |
|---|---|---|---|---|
| **A0** | **High biological load** | 25% | 64/27/8 | elevated cardiometabolic markers, systemic inflammation and substance use, with above-average overall severity |
| **A1** | **Low burden (relatively well)** | 36% | 74/23/2 | mild across symptoms and biology — comparatively well-functioning |
| **A2** | **Severe, low biological load** | 17% | 49/36/14 | high overall clinical severity *without* the metabolic / inflammatory / substance signature |
| **A3** | **Symptom-driven** | 22% | 81/14/4 | disrupted sleep/circadian, high early-life adversity and mood activation/suicidality, near-average severity, low metabolic |

`results/face/strata_oop/reading_guide/reading_guide.csv` carries the full pole + typical-member vectors and the
medoid patient id per corner. The exemplars are **cohort-diverse** (A2's medoid is a schizophrenia patient; A0/A1/A3
bipolar) — the corners are **transdiagnostic**, not relabelled diagnoses.

## The headline, made pointable
**A0 and A2 sit at essentially the same overall severity (+0.69 vs +0.76) but are biologically inverse**: A0's
typical member is high on metabolic/inflammatory/substance (≈ +0.70 each), A2's is *low* on all of them
(inflammatory −1.00, substance −0.95). Two "typical patients," equally ill, biologically opposite. That single
contrast **is** the project's biology⊥severity result, rendered as two representative profiles a clinician can hold
— and, crucially, **biology stays the *defining* axis of A0's representative** (its top axes), diluting the pure
pole only ~3× (metabolic +1.99 → +0.70). A hard tessellation, by contrast, splits the cloud on *suicidality* and
shrinks biology to a faint co-loading (~6× dilution) — which is exactly why the reading guide is anchored to the
archetype corners, not to region centroids.

## The patient-similarity tool (the continuum-honest core)
For any patient: their axis profile + their **nearest real neighbours**, computed with an uncertainty-aware
distance over diagonal posteriors,

    d²(i,j) = Σ_d (μ_i,d − μ_j,d)² / (σ_i,d² + σ_j,d² + ε)

so an axis a patient is uncertain about contributes little (an ill-measured patient gets a *fuzzy* neighbourhood).
Two spaces are exported: **Arm A** = all nine axes ("similar overall, incl. severity"); **Arm B** = the eight
specifics with severity removed ("similar in clinical *kind*, regardless of how ill"). Hand-off:
`results/face/strata_oop/similarity/neighbors.parquet` (9,013 patients × top-10, both arms; neighbours keyed as
`cohort/patient_id`).

Two things the demonstration on the four medoid exemplars shows: (a) neighbours are **transdiagnostic** (e.g. the
biological-pole patient's nearest neighbours mix BP/SZ/DR), and (b) the **Arm-B "in-kind" space recovers the
phenotype** where Arm-A mixes in severity — clearest for the symptom-driven exemplar (4/5 in-kind neighbours are
symptom-driven vs a mixed overall set). This is the transdiagnostic promise made concrete: "your most similar
patients, across diagnoses, look like this."

## Honest framing (do not over-read)
- These are **reading lenses on a continuum**, not natural kinds. Most patients are genuine blends (high corner
  entropy). "A0 vs A2" is a way to *talk about positions*, not a claim that patients fall into four boxes.
- The **pole** is the pure extreme (rare); the **typical member** centroid is ~3× milder; report the triplet.
- The **tessellation** (K=2/3/4) is an **optional** convenience for workflows that need hard boxes; its borders are
  arbitrary (no privileged K) and it is *not* the load-bearing object.
- Copula vertical throughout; biology⊥severity is confound-robust (validation chapter §6.2).

## Reproduce
```
PYTHONPATH=$PWD/src python notebooks/m2_reframe/build_reading_guide.py   # corners + medoids + figure + csv
PYTHONPATH=$PWD/src python notebooks/m2_reframe/build_similarity.py      # nearest-neighbour hand-off (both arms)
```
Both read existing strata_oop coordinates/profiles — no fitting. (`patient_id` is per-cohort, not global; all joins
key on `(cohort, patient_id)`.)

## Open / next
- Fold this reading guide + similarity tool into the report's M2 chapter under the coordinate-system framing
  (demote the tessellation to optional).
- Deferred from earlier: within-cohort prognosis breakdown; the full archetype-robustness battery.
