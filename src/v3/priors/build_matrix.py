"""Build the V3 soft-prior loading matrix from configs (config-first, no hard-coded SPEC).

Expands ``configs/dimensions.yaml`` (modeling ontology + full indicator pools) through
``configs/priors.yaml`` (tier rules) into a FULL (item x factor) prior loading matrix:
every modeled indicator gets a prior cell on EVERY modeled factor — primary on its home
factor, plausible on theory-motivated cross factors, near-zero (soft) everywhere else.
The general factor G (overall_severity) is identified by dedicated anchors that load on G
only (near-hard zero on every specific factor).

This matrix is the single source consumed by the Bayesian ESEM-bifactor engine to
parameterize Lambda — the mechanism by which theory proposes and data confirms / splits /
merges / cross-loads. Nothing here imputes data; it only describes priors on loadings.

Outputs (configs/):
  * prior_loading_matrix_v3.csv   — one row per (item, factor) cell
  * likelihoods.yaml              — canonical per-item likelihood family (mirrors map)
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[3]
CONFIGS = REPO / "configs"

# Gaussian-marginalizable families go in the continuous block; the rest need an
# explicit latent (Z) block (ordinal / binary / count likelihoods).
CONTINUOUS_FAMILIES = {"gaussian", "lognormal", "student_t"}


def _norm_family(raw: str) -> str:
    """Strip data-layer notes (e.g. 'lognormal:parse') to the bare likelihood family."""
    return str(raw).split(":")[0].strip()


def modeling_block(family: str) -> str:
    return "continuous" if _norm_family(family) in CONTINUOUS_FAMILIES else "explicit"


# --------------------------------------------------------------------------- config IO
def _load(name: str) -> dict[str, Any]:
    with open(CONFIGS / name) as fh:
        return yaml.safe_load(fh)


def load_configs(configs_dir: Path | None = None) -> tuple[dict, dict, dict]:
    """Return (dimensions, priors, likelihoods) config dicts."""
    global CONFIGS
    if configs_dir is not None:
        CONFIGS = Path(configs_dir)
    dims = _load("dimensions.yaml")
    priors = _load("priors.yaml")
    # likelihood_map_v3.yaml is the hand-maintained SOURCE; likelihoods.yaml is a derived
    # artifact this builder EMITS (never reads back — that would let it go stale).
    liks = _load("likelihood_map_v3.yaml")["likelihoods"]
    liks = {k: _norm_family(v) for k, v in liks.items()}   # strip ':parse' data-layer notes
    return dims, priors, liks


# ------------------------------------------------------------------ ontology resolution
def _factor_items(fac: dict) -> dict[str, int]:
    """Items that are PRIMARY indicators of a specific factor, with sign (+1/-1).

    Pulls from primary / extended / binary_anchors / subjective_items pos|neg lists.
    Excludes covariate_only and plausible-cross (those are handled separately).
    """
    items: dict[str, int] = {}
    for block in ("primary", "extended", "binary_anchors"):
        b = fac.get(block) or {}
        for it in b.get("pos", []):
            items[it] = +1
        for it in b.get("neg", []):
            items[it] = -1
    # subjective items are primary on the home factor too (with an affective cross added later)
    subj = fac.get("subjective_items") or {}
    for it in subj.get("pos", []):
        items.setdefault(it, +1)
    for it in subj.get("neg", []):
        items.setdefault(it, -1)
    return items


def _g_anchor_items(g: dict) -> dict[str, int]:
    """General-factor anchor items with sign."""
    items: dict[str, int] = {}
    for block in ("anchors", "extended", "count_anchors"):
        b = g.get(block) or {}
        for it in b.get("pos", []):
            items[it] = +1
        for it in b.get("neg", []):
            items[it] = -1
    return items


def resolve_ontology(dims: dict) -> dict[str, Any]:
    """Flatten dimensions.yaml into modeling primitives.

    Returns dict with:
      factors            ordered list of specific-factor keys (model_factor true)
      g_key              general-factor key (or None)
      all_factors        [g_key] + factors  (column order for the matrix)
      item_home          {item: home_factor_key}
      item_sign          {item: +1|-1}
      item_cross         {item: [cross_factor_key, ...]}  (plausible)
      g_anchors          {item: sign}
      covariate_only     set of items excluded from modeling
    """
    g = dims.get("general_factor") or {}
    g_key = g.get("key") if g.get("model_factor") else None
    g_anchors = _g_anchor_items(g) if g_key else {}

    factors: list[str] = []
    item_home: dict[str, str] = {}
    item_sign: dict[str, int] = {}
    item_cross: dict[str, list[str]] = {}
    covariate_only: set[str] = set()

    for fac in dims.get("factors", []):
        if not fac.get("model_factor"):
            continue
        key = fac["key"]
        factors.append(key)
        prim = _factor_items(fac)
        for it, sgn in prim.items():
            item_home[it] = key
            item_sign[it] = sgn
        # factor-level plausible cross -> every primary item of this factor
        for cross in fac.get("plausible_cross", []) or []:
            for it in prim:
                item_cross.setdefault(it, [])
                if cross not in item_cross[it]:
                    item_cross[it].append(cross)
        # item-level subjective cross (e.g. PSQI subjective -> affective)
        subj = fac.get("subjective_items") or {}
        for it in (subj.get("pos", []) + subj.get("neg", [])):
            for cross in subj.get("cross", []) or []:
                item_cross.setdefault(it, [])
                if cross not in item_cross[it]:
                    item_cross[it].append(cross)
        for it in fac.get("covariate_only", []) or []:
            covariate_only.add(it)

    all_factors = ([g_key] if g_key else []) + factors
    return {
        "factors": factors,
        "g_key": g_key,
        "all_factors": all_factors,
        "item_home": item_home,
        "item_sign": item_sign,
        "item_cross": item_cross,
        "g_anchors": g_anchors,
        "covariate_only": covariate_only,
    }


# ----------------------------------------------------------------------- matrix builder
def build_prior_matrix(configs_dir: Path | None = None,
                       out_csv: Path | None = None) -> list[dict[str, Any]]:
    """Expand the ontology into the full (item x factor) prior matrix; write CSV."""
    dims, priors, liks = load_configs(configs_dir)
    onto = resolve_ontology(dims)
    tiers = priors["tiers"]
    g_key = onto["g_key"]
    all_factors = onto["all_factors"]

    # full modeled item set = specific-factor primaries + G anchors (minus covariate_only)
    items = sorted(set(onto["item_home"]) | set(onto["g_anchors"]))
    items = [it for it in items if it not in onto["covariate_only"]]

    rows: list[dict[str, Any]] = []
    missing_lik: list[str] = []

    def tier_row(item, factor, tier_name, sign, lik, why):
        t = tiers[tier_name]
        return {
            "item": item,
            "factor": factor,
            "prior_type": tier_name,
            "prior_mean": t["mean"],
            "prior_sd": t["sd"],
            "dist": t["dist"],
            "sign_constraint": t["sign_constraint"],
            "item_sign": sign,
            "likelihood_family": lik,
            "modeling_block": modeling_block(lik),
            "rationale": why,
        }

    for it in items:
        lik = liks.get(it)
        if lik is None:
            missing_lik.append(it)
            lik = "gaussian"  # safe default; flagged below
        is_g_anchor = it in onto["g_anchors"]
        home = onto["item_home"].get(it)
        sign = onto["item_sign"].get(it, onto["g_anchors"].get(it, +1))
        crosses = onto["item_cross"].get(it, [])

        for fac in all_factors:
            if is_g_anchor:
                if fac == g_key:
                    rows.append(tier_row(it, fac, "g_anchor", sign, lik, "G dedicated anchor"))
                else:
                    rows.append(tier_row(it, fac, "g_anchor_on_specific", sign, lik,
                                         "severity anchor ~0 on specific (bifactor id)"))
                continue
            # specific-factor indicator
            if fac == home:
                rows.append(tier_row(it, fac, "primary", sign, lik, f"home factor {home}"))
            elif fac == g_key:
                # specific indicators MAY load on G (plausible) — this is how G becomes "general"
                rows.append(tier_row(it, fac, "plausible_cross", sign, lik,
                                     "specific item may load on G"))
            elif fac in crosses:
                rows.append(tier_row(it, fac, "plausible_cross", sign, lik,
                                     "theory-motivated cross-loading"))
            else:
                rows.append(tier_row(it, fac, "unlikely_cross", sign, lik, "soft-zero complement"))

    if out_csv is None:
        out_csv = CONFIGS / "prior_loading_matrix_v3.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # canonical likelihoods.yaml (mirror the per-item families actually modeled)
    lik_view = {it: liks.get(it, "gaussian") for it in items}
    _write_likelihoods_yaml(lik_view)

    _print_summary(rows, items, all_factors, missing_lik, out_csv)
    return rows


def _write_likelihoods_yaml(lik_view: dict[str, str]) -> None:
    path = CONFIGS / "likelihoods.yaml"
    header = ("# configs/likelihoods.yaml — canonical per-item likelihood family (V3).\n"
              "# Generated by src/v3/priors/build_matrix.py from likelihood_map_v3.yaml +\n"
              "# the modeled item set in dimensions.yaml. Single source for the engine.\n")
    body = yaml.safe_dump({"likelihoods": dict(sorted(lik_view.items()))},
                          sort_keys=False, default_flow_style=False)
    path.write_text(header + body)


def _print_summary(rows, items, all_factors, missing_lik, out_csv) -> None:
    from collections import Counter
    tier_counts = Counter(r["prior_type"] for r in rows)
    lik_counts = Counter(r["likelihood_family"] for r in {r["item"]: r for r in rows}.values())
    print(f"[prior-matrix] wrote {out_csv.relative_to(REPO)}")
    print(f"  items modeled : {len(items)}")
    print(f"  factors       : {len(all_factors)}  {all_factors}")
    print(f"  cells         : {len(rows)} (= {len(items)} x {len(all_factors)})")
    print(f"  tier counts   : {dict(tier_counts)}")
    print(f"  likelihoods   : {dict(lik_counts)}")
    if missing_lik:
        print(f"  ⚠ missing likelihood (defaulted gaussian): {missing_lik}")


if __name__ == "__main__":
    build_prior_matrix()
