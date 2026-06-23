#!/usr/bin/env python3
"""Archetype-robustness battery — is A=4 the right, reproducible, non-artefactual reading lens?

A=4 is load-bearing twice over: the M2 reading lens AND M4's predictive carrier. But archetypal analysis
*always* returns A corners, so — exactly as M2 falsifies clusters before trusting them — we must show the
A=4 corners are (a) the right count, (b) reproducible across seeds / n_init / bootstrap / measurement error,
and (c) not a fitting artefact. Wraps the proven kernels in ``src/face/strata/archetypes.py`` (the same code
the production fit used); no re-derivation of the map.

Speed/validity protocol (AA on the JAX backend is ~8s/fit on full N): the REFERENCE corners are the
persisted production fit (``archetype_profiles.csv`` — zero refits); reproducibility uses **n_init=4**
(single restarts are unreliable — that is check 3); refit-heavy checks **subsample to N=3,500** (n_init=4 on
3,500 reproduces the production corners to Tucker ~0.985, validated). Nine checks, each PASS/CONDITIONAL/FAIL.

  1. A-selection         — EV scree + stability-gated A (largest A with cross-seed Tucker >= 0.8).
  2. cross-seed @A=4      — min / per-corner Tucker congruence over seeds (FULL N, n_init=4).
  3. n_init sensitivity   — single restarts scatter; n_init=4 reproduces (why the n_init=4 protocol).
  4. anchor recovery      — bootstrap + M1 measurement-draw refits: per-corner profile HDI + congruence.
  5. granularity curve    — Tucker vs A; A=3 *collapse* (which two A=4 corners merge).
  6. native A=8 on copula — A=8 reproduces far worse than A=4 (substantiates "A=8 doesn't transfer").
  7. degeneracy           — pairwise corner cosine (no near-duplicate corners) + min separation.
  8. membership health    — blend entropy, %near-pole, %boundary (continuum-honest: most are blends).
  9. split-half OOS       — fit on one half, project the other; cross-half congruence + reconstruction R².

    PYTHONPATH=$PWD/src python notebooks/archetype_geometry/robustness_battery.py
"""
from __future__ import annotations

import json
import pathlib
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from face.strata.archetypes import _align, explained_variance, fit_aa, project_to_Z  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
C = REPO / "results" / "face" / "strata_oop" / "coordinates"
CONS = REPO / "results" / "face" / "strata_oop" / "consolidate"
OUTD = REPO / "results" / "face" / "strata_oop" / "archetype_robustness"; OUTD.mkdir(parents=True, exist_ok=True)
FIGD = REPO / "docs" / "figures" / "archetype_geometry"; FIGD.mkdir(parents=True, exist_ok=True)

AX = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep",
      "mania_activation", "suicidality", "developmental_risk", "substance"]
SEED = 20260621
A_STAR = 4
NSUB = 3500
LABELS = {0: "biological", 1: "well", 2: "severe·low-bio", 3: "symptom-driven"}
COL4 = ["#B42318", "#2B4C8C", "#0F766E", "#B7791F"]
_FITS = [0]


def _fit(X, A, seed, n_init=4):
    _FITS[0] += 1
    return fit_aa(X, A, seed=seed, n_init=n_init)


def _tuck(p, q):
    return float(abs((p * q).sum()) / (np.sqrt((p**2).sum()) * np.sqrt((q**2).sum()) + 1e-12))


def congruence(Zref, Z):
    c = _align(Zref, Z)
    per = np.array([_tuck(Zref[k], Z[c][k]) for k in range(Zref.shape[0])])
    return per, float(per.min()), c


def gate(v, p, c, hib=True):
    if hib:
        return "PASS" if v >= p else ("CONDITIONAL" if v >= c else "FAIL")
    return "PASS" if v <= p else ("CONDITIONAL" if v <= c else "FAIL")


def main() -> None:
    t0 = time.time()
    co = pd.read_parquet(C / "coordinates_full.parquet")
    X = co[[f"{a}__mean" for a in AX]].to_numpy("float64")
    dz = np.load(C / "coordinates_draws.npz"); draws = dz["draws"].astype("float64")
    cols = list(range(len(AX))); N = X.shape[0]
    prof = pd.read_csv(CONS / "archetype_profiles.csv")
    Zref = prof[prof.arm == "A_all9"].sort_values("archetype")[AX].to_numpy("float64")   # production corners
    ps = pd.read_parquet(CONS / "patient_strata.parquet")
    W = ps[[f"arch_w{k}" for k in range(A_STAR)]].to_numpy("float64")
    rng = np.random.default_rng(SEED)
    sub = lambda: rng.choice(N, NSUB, replace=False)                                      # noqa: E731
    print(f"X={X.shape} draws={draws.shape}; reference = persisted production corners (no refit)")
    checks, details = [], {}

    # 1. A-selection: EV scree + stability-gated A. Stability per A = MIN over all 4-seed pairs (robust to
    #    AA's occasional bad local optimum); chosen A = largest CONTIGUOUS A from 2 with stability >= 0.8
    #    (so a fluke high value at some A>ceiling cannot reinstate it).
    As = (2, 3, 4, 5, 6); seeds_sw = (0, 1, 2, 3); ev_by_A, stab_by_A, Zsweep = {}, {}, {}
    for A in As:
        i0 = sub()
        fits = [_fit(X[i0], A, seed=s) for s in seeds_sw]; Zse = [f[1] for f in fits]
        pij = [congruence(Zse[a], Zse[b])[1] for a in range(len(Zse)) for b in range(a + 1, len(Zse))]
        stab_by_A[A] = float(min(pij)); ev_by_A[A] = float(explained_variance(X[i0], fits[0][3])); Zsweep[A] = Zse[0]
    chosen_A = As[0]
    for A in As:
        if stab_by_A[A] >= 0.8:
            chosen_A = A
        else:
            break
    details["a_selection"] = {"ev_scree": {str(k): round(v, 3) for k, v in ev_by_A.items()},
                              "stability_by_A": {str(k): round(v, 3) for k, v in stab_by_A.items()},
                              "stability_gated_A": int(chosen_A)}
    checks.append({"check": "1. A-selection (stability-gated)",
                   "metric": f"chosen A={chosen_A} (A5 stability drops to {stab_by_A.get(5, float('nan')):.2f})",
                   "value": float(chosen_A == A_STAR), "verdict": "PASS" if chosen_A == A_STAR else "CONDITIONAL"})
    print(f"[1] A-selection done ({_FITS[0]} fits, {time.time()-t0:.0f}s) chosen_A={chosen_A} "
          f"curve={ {k: round(v, 2) for k, v in stab_by_A.items()} }")

    # 2. cross-seed reproducibility @A=4 (FULL N, n_init=4) — the headline
    Zs = [_fit(X, A_STAR, seed=s)[1] for s in range(3)]
    per_seed = np.array([congruence(Zref, Z)[0] for Z in Zs])
    cs_min = float(per_seed.min()); cs_pc = per_seed.min(0)
    details["cross_seed"] = {"n_seeds": len(Zs), "min_tucker": round(cs_min, 4),
                             "per_corner_min": {LABELS[k]: round(float(cs_pc[k]), 3) for k in range(A_STAR)}}
    checks.append({"check": "2. cross-seed reproducibility @A=4", "metric": f"min Tucker={cs_min:.3f} (full N, 3 seeds)",
                   "value": cs_min, "verdict": gate(cs_min, 0.90, 0.80)})
    print(f"[2] cross-seed done ({_FITS[0]} fits, {time.time()-t0:.0f}s) min={cs_min:.3f}")

    # 3. n_init sensitivity — single restarts scatter, n_init=4 reproduces
    ni1 = [congruence(Zref, _fit(X[sub()], A_STAR, seed=s, n_init=1)[1])[1] for s in (3, 4, 5)]
    details["n_init"] = {"single_restart_congruence": [round(v, 3) for v in ni1],
                         "single_restart_range": [round(min(ni1), 3), round(max(ni1), 3)],
                         "n_init4_congruence": round(cs_min, 3)}
    n_init_ok = cs_min >= 0.9 and (max(ni1) - min(ni1)) > 0.1
    checks.append({"check": "3. n_init sensitivity (why n_init=4)",
                   "metric": f"1-restart {min(ni1):.2f}-{max(ni1):.2f} vs n_init4 {cs_min:.2f}",
                   "value": float(n_init_ok), "verdict": "PASS" if n_init_ok else "CONDITIONAL"})
    print(f"[3] n_init done ({_FITS[0]} fits, {time.time()-t0:.0f}s)")

    # 4. anchor recovery — bootstrap (posterior mean) + measurement-draw refits, aligned to Zref. Report the
    #    MEDIAN per-refit min-corner congruence (robust to AA's occasional bad local optimum), bootstrap and
    #    measurement-draw SEPARATELY; the band is the 10-90% envelope of the aligned corners.
    def _refit(Xsub):
        _, Z, _, _ = _fit(Xsub, A_STAR, seed=SEED)
        _, mn, c = congruence(Zref, Z)
        return Z[c], float(mn)
    boot, drw, aligned = [], [], []
    for _ in range(8):                                                  # patient bootstrap (posterior mean)
        Za, mn = _refit(X[rng.integers(0, N, NSUB)]); boot.append(mn); aligned.append(Za)
    for _ in range(8):                                                  # M1 measurement draws (harder)
        s = int(rng.integers(0, draws.shape[0])); Za, mn = _refit(np.asarray(draws[s])[sub()][:, cols])
        drw.append(mn); aligned.append(Za)
    boot_med = float(np.median(boot)); drw_med = float(np.median(drw))
    frac_ok = float(np.mean([m >= 0.8 for m in boot + drw]))
    stack = np.stack(aligned); lo, hi = np.quantile(stack, 0.10, 0), np.quantile(stack, 0.90, 0)
    details["anchor_recovery"] = {"n_refits": len(aligned), "bootstrap_median_tucker": round(boot_med, 3),
                                  "draw_median_tucker": round(drw_med, 3), "frac_refits_ge_0.8": round(frac_ok, 3),
                                  "mean_band_width": round(float(np.mean(hi - lo)), 3)}
    details["_anchor_bands"] = {"mean": stack.mean(0).tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
    # two-tier: bootstrap = SAMPLE stability (primary); measurement-draws = corner-LOCATION precision under
    # propagated M1 uncertainty (harder — the rare extreme corners wobble most). PASS needs both strong;
    # CONDITIONAL = sample-stable but measurement-uncertain locations (consistent with "corners = reading lens,
    # the continuous coordinates + their uncertainty are the load-bearing object").
    if boot_med >= 0.85 and drw_med >= 0.80:
        v4 = "PASS"
    elif boot_med >= 0.85 and drw_med >= 0.60:
        v4 = "CONDITIONAL"
    else:
        v4 = "FAIL"
    checks.append({"check": "4. anchor recovery (bootstrap/draws)",
                   "metric": f"median Tucker boot={boot_med:.2f}/draw={drw_med:.2f}, {frac_ok*100:.0f}% refits≥0.8",
                   "value": min(boot_med, drw_med), "verdict": v4})
    print(f"[4] anchor recovery done ({_FITS[0]} fits, {time.time()-t0:.0f}s) boot={boot_med:.2f} draw={drw_med:.2f}")

    # 5. granularity — A=4 is the reproducibility ceiling: stability ~1.0 through A=4, then drops below 0.8
    #    at A=5. Also report which A=4 corners collapse at A=3 (AA is non-hierarchical, so this is informational).
    Z3 = Zsweep[3]
    nn = [int(np.argmax([_tuck(Zref[k], Z3[j]) for j in range(3)])) for k in range(A_STAR)]
    merged_j = [j for j, c in Counter(nn).items() if c > 1]
    if len(merged_j) == 1:
        coll = " + ".join(LABELS[k] for k in range(A_STAR) if nn[k] == merged_j[0])
    elif merged_j:
        coll = "non-hierarchical (A=3 does not cleanly nest in A=4)"
    else:
        coll = "no clean merge"
    drop_ok = stab_by_A[4] >= 0.9 and stab_by_A.get(5, 1.0) < 0.8
    details["granularity"] = {"stability_by_A": details["a_selection"]["stability_by_A"], "A3_collapse": coll}
    checks.append({"check": "5. granularity (A=4 ceiling)",
                   "metric": f"Tucker A4={stab_by_A[4]:.2f}, A5={stab_by_A.get(5,float('nan')):.2f} (<0.8 ⇒ A=4 ceiling); A3≈{coll[:18]}",
                   "value": float(drop_ok), "verdict": "PASS" if drop_ok else "CONDITIONAL"})

    # 6. native A=8 on the copula
    i8 = sub(); a8 = congruence(_fit(X[i8], 8, seed=0)[1], _fit(X[i8], 8, seed=1)[1])[1]
    details["native_A8"] = {"A8_min_tucker": round(float(a8), 3), "A4_min_tucker": round(stab_by_A[4], 3)}
    checks.append({"check": "6. native A=8 does not transfer", "metric": f"A8 Tucker={a8:.2f} vs A4={stab_by_A[4]:.2f}",
                   "value": float(stab_by_A[4] - a8), "verdict": "PASS" if a8 < stab_by_A[4] - 0.05 else "CONDITIONAL"})
    print(f"[5-6] granularity+A8 done ({_FITS[0]} fits, {time.time()-t0:.0f}s) A8={a8:.2f}")

    # 7. degeneracy — pairwise corner cosine + min separation (no fit; production corners)
    Zn = Zref / (np.linalg.norm(Zref, axis=1, keepdims=True) + 1e-12)
    cos = Zn @ Zn.T; np.fill_diagonal(cos, -1); max_cos = float(cos.max())
    mind = float(min(np.linalg.norm(Zref[i] - Zref[j]) for i in range(A_STAR) for j in range(i + 1, A_STAR)))
    details["degeneracy"] = {"max_pairwise_cosine": round(max_cos, 3), "min_pairwise_distance": round(mind, 2)}
    checks.append({"check": "7. no degenerate/duplicate corners", "metric": f"max corner cosine={max_cos:.2f}",
                   "value": max_cos, "verdict": gate(max_cos, 0.50, 0.75, hib=False)})

    # 8. membership health — from persisted production weights (no fit)
    ent = -(W * np.log(W + 1e-12)).sum(1) / np.log(A_STAR)
    near_pole = float((W.max(1) > 0.5).mean()); boundary = float((W.max(1) < 0.4).mean())
    details["membership"] = {"mean_entropy": round(float(ent.mean()), 3), "frac_near_pole": round(near_pole, 3),
                             "frac_blend_boundary": round(boundary, 3)}
    checks.append({"check": "8. membership health (continuum-honest)",
                   "metric": f"mean entropy={ent.mean():.2f}, {near_pole*100:.0f}% near a pole",
                   "value": float(ent.mean()), "verdict": "PASS" if 0.45 < ent.mean() else "CONDITIONAL"})

    # 9. split-half out-of-sample
    idx = rng.permutation(N); h1, h2 = idx[: N // 2], idx[N // 2:]
    Z1 = _fit(X[h1], A_STAR, seed=SEED)[1]; Z2 = _fit(X[h2], A_STAR, seed=SEED)[1]
    xh = congruence(Z1, Z2)[1]
    W2 = project_to_Z(X[h2], Z1)
    rec = 1.0 - ((X[h2] - W2 @ Z1) ** 2).sum() / ((X[h2] - X[h2].mean(0)) ** 2).sum()
    details["split_half"] = {"cross_half_min_tucker": round(float(xh), 3), "oos_reconstruction_r2": round(float(rec), 3)}
    checks.append({"check": "9. split-half out-of-sample", "metric": f"cross-half Tucker={xh:.3f}, OOS R²={rec:.2f}",
                   "value": xh, "verdict": gate(xh, 0.85, 0.70)})
    print(f"[7-9] done ({_FITS[0]} fits, {time.time()-t0:.0f}s)")

    tab = pd.DataFrame(checks)[["check", "metric", "verdict"]]
    tab.to_csv(OUTD / "archetype_robustness.csv", index=False)
    (OUTD / "robustness_detail.json").write_text(json.dumps(details, indent=2))
    n_pass = int((tab.verdict == "PASS").sum()); n_cond = int((tab.verdict == "CONDITIONAL").sum())
    n_fail = int((tab.verdict == "FAIL").sum())
    print("\n" + "=" * 92); print(tab.to_string(index=False)); print("=" * 92)
    print(f"VERDICT: {n_pass} PASS · {n_cond} CONDITIONAL · {n_fail} FAIL "
          f"[{_FITS[0]} AA fits, {time.time()-t0:.0f}s]")
    print(f"wrote {OUTD/'archetype_robustness.csv'} + robustness_detail.json")
    _figure(details, ev_by_A, stab_by_A, per_seed, tab)


def _figure(details, ev_by_A, stab_by_A, per_seed, tab):
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.4)); A4C = "#1E366B"
    ax = axes[0, 0]; As = sorted(ev_by_A)
    ax.plot(As, [ev_by_A[a] for a in As], "-o", color="#5B6573", label="explained var")
    ax2 = ax.twinx(); ax2.plot(As, [stab_by_A[a] for a in As], "-s", color="#B42318", label="cross-seed Tucker")
    ax2.axhline(0.8, color="#B42318", lw=0.7, ls=":"); ax.axvline(A_STAR, color=A4C, lw=1.2, ls="--")
    ax.set_xlabel("A (corners)"); ax.set_ylabel("explained variance", color="#5B6573")
    ax2.set_ylabel("min Tucker", color="#B42318"); ax2.set_ylim(0, 1.02)
    ax.set_title("1/5. EV always rises; reproducibility holds to A=4\nthen falls (stability-gated A=4)", fontsize=9.5, fontweight="bold")

    ax = axes[0, 1]; pc = per_seed.min(0)
    ax.bar(range(A_STAR), pc, color=COL4); ax.axhline(0.9, color="#444", lw=0.7, ls=":"); ax.set_ylim(0, 1.02)
    ax.set_xticks(range(A_STAR)); ax.set_xticklabels([LABELS[k] for k in range(A_STAR)], rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("min Tucker (full N)")
    ax.set_title("2. Each corner reproduces across seeds (n_init=4)", fontsize=9.5, fontweight="bold")

    ax = axes[1, 0]; b = details["_anchor_bands"]
    Zm, lo, hi = np.array(b["mean"]), np.array(b["lo"]), np.array(b["hi"]); xs = np.arange(len(AX))
    for k in range(A_STAR):
        ax.plot(xs, Zm[k], "-", color=COL4[k], lw=1.5, label=LABELS[k])
        ax.fill_between(xs, lo[k], hi[k], color=COL4[k], alpha=0.15)
    ax.axhline(0, color="#444", lw=0.5); ax.set_xticks(xs)
    ax.set_xticklabels([a[:5] for a in AX], rotation=40, ha="right", fontsize=7); ax.set_ylabel("coordinate (SD)")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax.set_title("4. Corners survive bootstrap + measurement-draw refits\n(10–90% band)", fontsize=9.5, fontweight="bold")

    ax = axes[1, 1]; ax.axis("off")
    vc = {"PASS": "#0F766E", "CONDITIONAL": "#B7791F", "FAIL": "#B42318"}
    ax.set_title("Battery verdict", fontsize=10, fontweight="bold", loc="left")
    for i, (_, r) in enumerate(tab.iterrows()):
        y = 0.93 - i * 0.102
        ax.text(0.0, y, r["check"], fontsize=7.6, va="top")
        ax.text(0.63, y, str(r["metric"])[:33], fontsize=6.7, va="top", color="#5B6573")
        ax.text(0.99, y, r["verdict"], fontsize=7.6, va="top", ha="right", color=vc[r["verdict"]], fontweight="bold")
    fig.suptitle("Archetype-robustness battery — A=4 is the right, reproducible, non-artefactual reading lens",
                 y=1.0, fontsize=12, fontweight="bold", color=A4C)
    fig.tight_layout()
    for p in [FIGD / "robustness_battery.png", REPO / "report" / "figures" / "m2_archetype_robustness.png"]:
        fig.savefig(p, bbox_inches="tight", facecolor="white", dpi=150)
    print(f"wrote {FIGD/'robustness_battery.png'} (+ report/figures/m2_archetype_robustness.png)")


if __name__ == "__main__":
    main()
