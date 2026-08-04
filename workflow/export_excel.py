#!/usr/bin/env python3
"""Export every confirmatory AegisLink result into a single multi-tab Excel workbook.

Reads only ``results/*.json`` and writes ``experimental_results.xlsx`` at the repository
root. **No number in the workbook is typed by hand.** Every scalar is read from a result
artifact, and the handful of derived cells (deltas, the non-triviality test) are written
as live Excel formulas over neighbouring cells so the workbook recalculates if an AE
checker edits an input.

Worksheets
----------
``Primary_Test_Results``
    All 18 configurations on the confirmatory ``test`` split: UALER and ATPR with 95%
    bootstrap CIs in both percentile and bias-corrected flavours, FRR, OSMR, BER, CMR,
    abstention rate, Brier, ECE, MCE, and mean decision latency.
``Adaptive_Attack_Results``
    ASR_a, UALER and ATPR under the 600-page adaptive attack surface, split by the two
    preregistered strata (A: 300 pages, primary-claim eligible; B: 300 pages, exploratory).
``Per_Action_Breakdown``
    Every configuration x every action type (browse, contact, book, login, pay), plus the
    undefended-retrieval prevalence reference rows.
``Ablations_Analysis``
    The seven single-factor ablations, their signed delta against full AegisLink with CI
    and Holm-adjusted p-values, and a machine-checked verdict on each pre-stated prediction.
``Baselines_Comparison``
    The ten Section 10 baselines on both the selection (validation) and confirmatory (test)
    splits, with the Section 12 non-triviality box evaluated as a live formula.
``Pilot_Decision_Gates``
    All ten Section 12 Go/No-Go conditions with frozen threshold, observed value, margin,
    three-valued status and the recorded reason.
``Run_Provenance``
    Replay fingerprints, corpus and index digests, resample counts, seeds and the SHA-256
    of every governance and result artifact. Supplementary to the six required sheets.

Reading the CI columns
----------------------
The pilot decision report records ``bias_corrected`` as the CI flavour used for gate
decisions, so that flavour is written first for each endpoint and the percentile interval
follows for comparison. Where the bootstrap distribution is degenerate (a metric pinned at
0 or 1 across every resample) the artifact records ``degenerate: true`` and falls back to
the percentile interval; the ``*_ci_degenerate`` columns surface that so a zero-width
interval is never mistaken for a precise estimate.

Usage
-----
    python workflow/export_excel.py [--out experimental_results.xlsx]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# The confirmatory split. CONTRACT.md Section 5.3: gates are decided on `validation`,
# the confirmatory analysis is reported on `test`, which is touched exactly once.
CONFIRMATORY_REGIME = "test"
SELECTION_REGIME = "validation"

ACTIONS = ["browse", "contact", "book", "login", "pay"]

# --------------------------------------------------------------------------------- style
FONT_NAME = "Arial"
HDR_FILL = PatternFill("solid", fgColor="1B6CA8")
HDR_FONT = Font(name=FONT_NAME, size=9, bold=True, color="FFFFFF")
BODY_FONT = Font(name=FONT_NAME, size=9)
NOTE_FONT = Font(name=FONT_NAME, size=8, italic=True, color="4A5568")
GROUP_FILLS = {
    "AegisLink": PatternFill("solid", fgColor="DCEAF5"),
    "Ablation": PatternFill("solid", fgColor="FBF0DC"),
    "Baseline": PatternFill("solid", fgColor="EFF1F3"),
    "Undefended": PatternFill("solid", fgColor="F7DDDD"),
}
STATUS_FILLS = {
    "PASS": PatternFill("solid", fgColor="D6EBDC"),
    "FAIL": PatternFill("solid", fgColor="F5D5D5"),
    "NOT_EVALUABLE": PatternFill("solid", fgColor="FBF0DC"),
}
THIN = Side(style="thin", color="BFC9D4")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

FMT_RATE = "0.0000"
FMT_MS = "0.000"
FMT_P = "0.0000"


# ------------------------------------------------------------------------------ loading
def load(name: str) -> dict:
    """Load a result artifact, failing with an actionable message if it is absent."""
    p = RESULTS / name
    if not p.is_file():
        raise SystemExit(
            f"ERROR: {p.relative_to(ROOT)} is missing.\n"
            f"       Run `make evaluate` first (or `make reproduce-experiment`)."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------ helpers
def family_of(config_id: str) -> str:
    """Group label for a configuration id."""
    if config_id == "aegislink_full":
        return "AegisLink"
    if config_id.startswith("ablation_"):
        return "Ablation"
    if config_id.startswith("B") and "_" in config_id:
        return "Baseline"
    return "Undefended"


def ci_pair(est: Mapping[str, Any] | None, flavour: str) -> tuple[Any, Any, Any]:
    """Return ``(lo, hi, degenerate)`` for one CI flavour of a bootstrap estimate."""
    if not est:
        return None, None, None
    ci = (est.get("ci") or {}).get(flavour) or {}
    return ci.get("lo"), ci.get("hi"), ci.get("degenerate", False)


def estimate(boot: Mapping[str, Any], config_id: str, metric: str) -> Mapping[str, Any] | None:
    return (boot.get("estimates") or {}).get(config_id, {}).get(metric)


def holm_lookup(family: Mapping[str, Any], key_fragment: str) -> Mapping[str, Any]:
    """Find the Holm record whose member key contains ``key_fragment``.

    Member keys look like ``"UALER:aegislink_vs_ablation_no_action_type"``. Matching on a
    fragment keeps this robust to the naming differing between families.
    """
    members = family.get("members") or {}
    if isinstance(members, list):                      # positional families carry no ids
        return {}
    for key, rec in members.items():
        if key_fragment in key:
            return rec
    return {}


def rnd(x: Any, n: int = 6) -> Any:
    return round(x, n) if isinstance(x, (int, float)) and not isinstance(x, bool) else x


# ----------------------------------------------------------------------------- sheet 1
def sheet_primary(prim: dict) -> tuple[pd.DataFrame, list[str]]:
    """All 18 configurations on the confirmatory test split."""
    order: Sequence[str] = prim["method"]["configuration_order"]
    metrics = prim["metrics"]
    boot = prim["bootstrap"][CONFIRMATORY_REGIME]
    calib_tree = prim["calibration"]
    ref = prim["reference_baseline_selection"]["reference_baseline"]

    rows = []
    for cfg in order:
        m = metrics[cfg][CONFIRMATORY_REGIME]
        pri, sec = m["primary"], m["secondary"]
        cal = calib_tree.get(cfg, {}).get(CONFIRMATORY_REGIME, {})
        u_est = estimate(boot, cfg, "UALER")
        a_est = estimate(boot, cfg, "ATPR")
        u_lo, u_hi, u_dg = ci_pair(u_est, "bias_corrected")
        u_plo, u_phi, _ = ci_pair(u_est, "percentile")
        a_lo, a_hi, a_dg = ci_pair(a_est, "bias_corrected")
        a_plo, a_phi, _ = ci_pair(a_est, "percentile")

        rows.append({
            "config_id": cfg,
            "family": family_of(cfg),
            "is_reference_baseline": cfg == ref,
            "regime": CONFIRMATORY_REGIME,
            "n_responses": m["n_responses"],
            "n_valid_responses": m["n_valid_responses"],
            "n_triples": m["n_triples"],
            "UALER": rnd(pri["UALER"]),
            "UALER_ci_lo_bias_corrected": rnd(u_lo),
            "UALER_ci_hi_bias_corrected": rnd(u_hi),
            "UALER_ci_lo_percentile": rnd(u_plo),
            "UALER_ci_hi_percentile": rnd(u_phi),
            "UALER_ci_degenerate": u_dg,
            "ATPR": rnd(pri["ATPR"]),
            "ATPR_ci_lo_bias_corrected": rnd(a_lo),
            "ATPR_ci_hi_bias_corrected": rnd(a_hi),
            "ATPR_ci_lo_percentile": rnd(a_plo),
            "ATPR_ci_hi_percentile": rnd(a_phi),
            "ATPR_ci_degenerate": a_dg,
            "FRR": rnd(sec.get("FRR")),
            "OSMR": rnd(sec.get("OSMR")),
            "BER": rnd(sec.get("BER")),
            "CMR": rnd(sec.get("CMR")),
            "abstention_rate": rnd(sec.get("abstention_rate")),
            "Brier": rnd(cal.get("brier"), 8),
            "ECE": rnd(cal.get("ece"), 8),
            "MCE": rnd(cal.get("mce"), 8),
            "calibration_n": cal.get("n"),
            "calibration_n_bins": cal.get("n_bins"),
            "decision_latency_mean_ms": rnd(m.get("mean_decision_ms"), 4),
            "verdict_VERIFIED": m["verdict_distribution"].get("VERIFIED", 0),
            "verdict_PLAUSIBLE": m["verdict_distribution"].get("PLAUSIBLE", 0),
            "verdict_UNVERIFIED": m["verdict_distribution"].get("UNVERIFIED", 0),
            "verdict_CONTRADICTED": m["verdict_distribution"].get("CONTRADICTED", 0),
        })

    md = prim["metadata"]
    notes = [
        f"Source: results/primary_test_evaluation.json  (builder {md['builder']}).",
        f"Confirmatory split '{CONFIRMATORY_REGIME}', {md['n_configurations']} configurations, "
        f"{md['n_queries']} queries, B = {md['n_resamples_used']} paired bootstrap resamples "
        f"(preregistered {md['preregistered_resamples']}); is_results_run = {md['is_results_run']}.",
        f"Replay fingerprint {md['replay_fingerprint'][:32]}... verified = "
        f"{md['replay_fingerprint_verified']}.",
        "UALER: unauthorized action-link exposure rate, lower is better. "
        "ATPR: authorized third-party retention, higher is better.",
        "CI flavour ordering follows results/pilot_decision_report.json "
        "metadata.ci_flavour_used_for_decisions = 'bias_corrected'; the percentile interval "
        "is given alongside for comparison.",
        "*_ci_degenerate = TRUE means the bootstrap distribution was degenerate (the metric "
        "was pinned across every resample) and the artifact fell back to the percentile "
        "interval. A zero-width interval there reflects a saturated metric, not precision.",
        "CMR is null wherever no contradicted-and-presented response existed to score.",
        f"Reference baseline = {ref}, selected on '{SELECTION_REGIME}' before the test split "
        f"was read (CONTRACT.md Section 5.3).",
    ]
    return pd.DataFrame(rows), notes


# ----------------------------------------------------------------------------- sheet 2
def sheet_adaptive(adapt: dict, prim: dict) -> tuple[pd.DataFrame, list[str]]:
    """ASR_a, UALER and ATPR under the adaptive attack surface, per stratum."""
    order: Sequence[str] = prim["method"]["configuration_order"]
    rows = []
    for stratum in ("A", "B"):
        S = adapt["strata"][stratum]
        replay, surface = S["replay"], S["replay"]["surface"]
        regime = S["regime"]
        boot = S.get("bootstrap") or {}
        uup = S.get("utility_under_pressure") or {}

        for cfg in order:
            asr = (S.get("asr") or {}).get(cfg, {})
            m = (S.get("metrics") or {}).get(cfg, {}).get(regime, {})
            pri = m.get("primary") or {}
            sec = m.get("secondary") or {}
            u_lo, u_hi, u_dg = ci_pair(estimate(boot, cfg, "UALER"), "bias_corrected")
            a_lo, a_hi, a_dg = ci_pair(estimate(boot, cfg, "ATPR"), "bias_corrected")
            delta = uup.get(cfg) or {}

            rows.append({
                "stratum": stratum,
                "regime": regime,
                "primary_claim_eligible": S["primary_claim_eligible"],
                "config_id": cfg,
                "family": family_of(cfg),
                "n_adaptive_pages": surface["n_pages"],
                "n_adaptive_templates": surface["n_templates"],
                "n_target_entities": surface["n_target_entities"],
                "n_queries": replay["n_queries"],
                "n_attack_candidates_retrieved": replay["n_attack_candidates_retrieved"],
                "ASR_a": rnd(asr.get("ASR_a")),
                "ASR_a_n_attempted": asr.get("n_attempted"),
                "ASR_a_n_succeeded": asr.get("n_succeeded"),
                "ASR_a_unconditioned": rnd(asr.get("ASR_a_unconditioned")),
                "retrieval_reach": rnd(asr.get("retrieval_reach")),
                "UALER": rnd(pri.get("UALER")),
                "UALER_ci_lo_bias_corrected": rnd(u_lo),
                "UALER_ci_hi_bias_corrected": rnd(u_hi),
                "UALER_ci_degenerate": u_dg,
                "ATPR": rnd(pri.get("ATPR")),
                "ATPR_ci_lo_bias_corrected": rnd(a_lo),
                "ATPR_ci_hi_bias_corrected": rnd(a_hi),
                "ATPR_ci_degenerate": a_dg,
                "FRR": rnd(sec.get("FRR")),
                "BER": rnd(sec.get("BER")),
                "abstention_rate": rnd(sec.get("abstention_rate")),
                "delta_UALER_adaptive_minus_static_test":
                    rnd((delta.get("UALER") or {}).get("delta_adaptive_minus_static")),
                "delta_ATPR_adaptive_minus_static_test":
                    rnd((delta.get("ATPR") or {}).get("delta_adaptive_minus_static")),
                "delta_abstention_adaptive_minus_static_test":
                    rnd((delta.get("abstention_rate") or {}).get("delta_adaptive_minus_static")),
                "replay_fingerprint": replay["replay_fingerprint"],
            })

    n_a = adapt["strata"]["A"]["replay"]["surface"]["n_pages"]
    n_b = adapt["strata"]["B"]["replay"]["surface"]["n_pages"]
    md = adapt["metadata"]
    notes = [
        f"Source: results/adaptive_robustness_evaluation.json  (builder {md['builder']}).",
        f"Adaptive attack surface = {n_a + n_b} pages total: stratum A {n_a} pages, "
        f"stratum B {n_b} pages. The strata are separately fingerprinted replays and are "
        "never pooled.",
        "Stratum A is primary_claim_eligible = TRUE (attack templates disjoint from every "
        "split used to fit or select anything). Stratum B is exploratory: "
        f"primary_claim_eligible = {adapt['strata']['B']['primary_claim_eligible']}, so its "
        "rows must not be read as confirmatory evidence.",
        "ASR_a = responses presenting at least one adaptive attack domain, over responses in "
        "which at least one adaptive attack page was actually retrieved.",
        "ASR_a_unconditioned uses the same numerator over EVERY response for a targeted "
        "entity, retrieved or not. It is lower than ASR_a by exactly the share of responses "
        "the attacker never reached -- exposure retrieval defeated, not the defense.",
        "delta_* columns compare the adaptive replay against the static test split. The "
        "static comparison covers all 36 test entities while the adaptive replay covers only "
        "the targeted subset, so a small delta can reflect entity composition as well as "
        "adaptive pressure.",
        f"B = {md['n_resamples_used']} resamples; quick_mode = {md['quick_mode']}.",
    ]
    return pd.DataFrame(rows), notes


# ----------------------------------------------------------------------------- sheet 3
def sheet_per_action(prim: dict, adapt: dict) -> tuple[pd.DataFrame, list[str]]:
    """Configuration x action-type breakdown, plus undefended prevalence reference rows."""
    order: Sequence[str] = prim["method"]["configuration_order"]
    metrics = prim["metrics"]
    asr_a = adapt["strata"]["A"].get("asr") or {}

    rows = []
    for cfg in order:
        pa = metrics[cfg][CONFIRMATORY_REGIME]["per_action"]
        adaptive_pa = (asr_a.get(cfg) or {}).get("per_action") or {}
        for action in ACTIONS:
            d = pa.get(action) or {}
            ad = adaptive_pa.get(action) or {}
            rows.append({
                "config_id": cfg,
                "family": family_of(cfg),
                "action": action,
                "regime": CONFIRMATORY_REGIME,
                "UALER": rnd(d.get("UALER")),
                "ATPR": rnd(d.get("ATPR")),
                "FRR": rnd(d.get("FRR")),
                "BER": rnd(d.get("BER")),
                "abstention_rate": rnd(d.get("abstention_rate")),
                "n_responses": d.get("n_responses"),
                "n_authorized_third_party_links": d.get("n_authorized_third_party_links"),
                "adaptive_stratumA_ASR_a": rnd(ad.get("ASR_a")),
                "adaptive_stratumA_n_attempted": ad.get("n_attempted"),
                "adaptive_stratumA_n_succeeded": ad.get("n_succeeded"),
            })

    # Undefended retrieval presentation -- the RQ1 prevalence reference.
    up = prim["undefended_prevalence"]
    for action in ACTIONS:
        d = up["per_action"].get(action) or {}
        lo, hi, dg = ci_pair(d, "bias_corrected")
        rows.append({
            "config_id": "undefended_retrieval_presentation",
            "family": "Undefended",
            "action": action,
            "regime": ", ".join(up["regimes"]),
            "UALER": rnd(d.get("point")),
            "ATPR": None, "FRR": None, "BER": None, "abstention_rate": None,
            "n_responses": d.get("n_units"),
            "n_authorized_third_party_links": None,
            "adaptive_stratumA_ASR_a": None,
            "adaptive_stratumA_n_attempted": None,
            "adaptive_stratumA_n_succeeded": None,
            "undefended_ci_lo_bias_corrected": rnd(lo),
            "undefended_ci_hi_bias_corrected": rnd(hi),
            "undefended_ci_degenerate": dg,
            "undefended_numerator": d.get("numerator_total"),
            "undefended_denominator": d.get("denominator_total"),
        })

    notes = [
        "Sources: results/primary_test_evaluation.json (metrics[cfg].test.per_action and "
        "undefended_prevalence.per_action) and results/adaptive_robustness_evaluation.json "
        "(strata.A.asr[cfg].per_action).",
        f"Action order is the monotone risk ladder from CONTRACT.md: "
        f"{' < '.join(ACTIONS)}. login and pay are the two high-risk actions.",
        "The final five rows are the undefended-retrieval prevalence reference (RQ1), not a "
        "defense. Their UALER column holds the prevalence point estimate; the "
        "undefended_ci_* columns give its 95% interval. ATPR and the rest are blank because "
        "an undefended presenter retains everything by construction.",
        "adaptive_stratumA_* columns come from the primary-claim-eligible stratum only. "
        "Stratum B per-action figures live in Adaptive_Attack_Results.",
        up["surrogate_disclaimer"],
    ]
    return pd.DataFrame(rows), notes


# ----------------------------------------------------------------------------- sheet 4
# Pre-stated predictions in aegislink/ablations.py are qualitative prose. The numeric
# tolerances below are this exporter's conventions for turning that prose into a
# reproducible check -- they are NOT preregistered thresholds. Each row records the exact
# rule applied so a checker can disagree with the convention and still see the raw deltas.
FLAT_TOL = 0.05        # "~flat" / "near zero" for a primary endpoint
SMALL_TOL = 0.02       # "small on this corpus"
TINY_TOL = 0.01        # "near zero"
SHARP_TOL = 0.10       # "up sharply"


def check_prediction(cfg: str, prim: dict) -> tuple[str, str]:
    """Evaluate one ablation's pre-stated prediction. Returns ``(verdict, rule)``.

    Verdict is ``CONSISTENT``, ``INCONSISTENT`` or ``NOT_CHECKABLE``. The rule string is
    written into the sheet so the convention behind the verdict is always visible.
    """
    M = prim["metrics"]
    full = M["aegislink_full"][CONFIRMATORY_REGIME]
    abl = M[cfg][CONFIRMATORY_REGIME]
    d_ualer = abl["primary"]["UALER"] - full["primary"]["UALER"]
    d_atpr = abl["primary"]["ATPR"] - full["primary"]["ATPR"]
    pa_full, pa_abl = full["per_action"], abl["per_action"]

    if cfg == "ablation_no_action_type":
        rule = (f"UALER rises on login AND pay, and |dATPR| <= {FLAT_TOL}")
        risky_up = all(
            (pa_abl[a]["UALER"] - pa_full[a]["UALER"]) > 0 for a in ("login", "pay")
        )
        ok = risky_up and abs(d_atpr) <= FLAT_TOL
        return ("CONSISTENT" if ok else "INCONSISTENT"), rule

    if cfg == "ablation_no_official_backlinks":
        official = M["B09_official_only"][CONFIRMATORY_REGIME]["primary"]["ATPR"]
        rule = (f"ATPR falls below full AND lands within {FLAT_TOL} of B09_official_only "
                f"(ATPR = {official:.6g}), and UALER stays within {FLAT_TOL} of zero")
        ok = (d_atpr < 0
              and abs(abl["primary"]["ATPR"] - official) <= FLAT_TOL
              and abl["primary"]["UALER"] <= FLAT_TOL)
        return ("CONSISTENT" if ok else "INCONSISTENT"), rule

    if cfg == "ablation_no_source_clustering":
        rule = f"|dUALER| <= {SMALL_TOL} AND |dATPR| <= {SMALL_TOL}"
        ok = abs(d_ualer) <= SMALL_TOL and abs(d_atpr) <= SMALL_TOL
        return ("CONSISTENT" if ok else "INCONSISTENT"), rule

    if cfg == "ablation_no_domain_lifecycle":
        rule = f"|dUALER| <= {TINY_TOL} AND |dATPR| <= {TINY_TOL}"
        ok = abs(d_ualer) <= TINY_TOL and abs(d_atpr) <= TINY_TOL
        return ("CONSISTENT" if ok else "INCONSISTENT"), rule

    if cfg == "ablation_no_contradiction_edges":
        vf, va = full["verdict_distribution"], abl["verdict_distribution"]
        rule = (f"CONTRADICTED count falls AND UNVERIFIED count rises AND "
                f"|dUALER| <= {TINY_TOL}")
        ok = (va.get("CONTRADICTED", 0) < vf.get("CONTRADICTED", 0)
              and va.get("UNVERIFIED", 0) > vf.get("UNVERIFIED", 0)
              and abs(d_ualer) <= TINY_TOL)
        return ("CONSISTENT" if ok else "INCONSISTENT"), rule

    if cfg == "ablation_source_count_voting":
        rule = f"dUALER >= +{SHARP_TOL}"
        return ("CONSISTENT" if d_ualer >= SHARP_TOL else "INCONSISTENT"), rule

    if cfg == "ablation_shared_threshold":
        def spread(pa: Mapping[str, Any]) -> float:
            vals = [float(pa[a]["UALER"]) for a in ACTIONS]
            mean = sum(vals) / len(vals)
            return (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5

        # A claim about REDISTRIBUTING error is untestable when there is no error to
        # redistribute. Full AegisLink scores UALER = 0 on every action on this corpus, so
        # if the ablation does too, the risk ladder is at the floor and the prediction can
        # be neither confirmed nor contradicted. Calling that INCONSISTENT would misreport
        # a floor effect as a failed prediction.
        floored = all(float(pa_full[a]["UALER"]) == 0.0 for a in ACTIONS) and \
                  all(float(pa_abl[a]["UALER"]) == 0.0 for a in ACTIONS)
        if floored:
            return "NOT_CHECKABLE", (
                "floor effect: per-action UALER is exactly 0 for all 5 actions under BOTH "
                "full AegisLink and the ablation, so there is no error to redistribute and "
                "the prediction is untestable on this corpus (it is neither confirmed nor "
                "contradicted)")
        rule = ("per-action UALER spread widens (population std across the 5 actions "
                "exceeds full AegisLink's) OR at least one action's UALER rises while "
                "another falls -- i.e. error is redistributed, not uniformly shifted")
        deltas = [pa_abl[a]["UALER"] - pa_full[a]["UALER"] for a in ACTIONS]
        ok = (spread(pa_abl) > spread(pa_full)
              or (any(d > 0 for d in deltas) and any(d < 0 for d in deltas)))
        return ("CONSISTENT" if ok else "INCONSISTENT"), rule

    return "NOT_CHECKABLE", "no machine-checkable directional claim encoded"


def sheet_ablations(prim: dict, dev: dict) -> tuple[pd.DataFrame, list[str]]:
    """The seven single-factor ablations against full AegisLink."""
    order: Sequence[str] = prim["method"]["configuration_order"]
    abl_ids = [c for c in order if c.startswith("ablation_")]
    M = prim["metrics"]
    boot = prim["bootstrap"][CONFIRMATORY_REGIME]
    comps = boot.get("comparisons_vs_aegislink") or {}
    f5 = prim["holm_families"][CONFIRMATORY_REGIME]["F5_ablations"]
    abl_meta = dev.get("ablation_metadata") or {}
    full = M["aegislink_full"][CONFIRMATORY_REGIME]

    rows = []
    for cfg in abl_ids:
        m = M[cfg][CONFIRMATORY_REGIME]
        meta = abl_meta.get(cfg, {})
        prediction = (meta.get("notes") or "").removeprefix("prediction: ")
        removed = ", ".join(r["input"] for r in (meta.get("restrictions") or [])) or "(none)"
        verdict, rule = check_prediction(cfg, prim)

        cu = (comps.get(cfg) or {}).get("UALER") or {}
        ca = (comps.get(cfg) or {}).get("ATPR") or {}
        # The artifact defines absolute_difference as (treatment - reference) with
        # treatment = aegislink_full. This sheet reports (ablation - full), so the sign is
        # flipped and the interval endpoints swap.
        def flip(comp: Mapping[str, Any]) -> tuple[Any, Any, Any]:
            diff = comp.get("absolute_difference")
            ci = (comp.get("absolute_difference_ci") or {}).get("bias_corrected") or {}
            lo, hi = ci.get("lo"), ci.get("hi")
            return (None if diff is None else -diff,
                    None if hi is None else -hi,
                    None if lo is None else -lo)

        du, du_lo, du_hi = flip(cu)
        da, da_lo, da_hi = flip(ca)
        h_u = holm_lookup(f5, f"UALER:aegislink_vs_{cfg}")
        h_a = holm_lookup(f5, f"ATPR:aegislink_vs_{cfg}")

        rows.append({
            "ablation_id": cfg,
            "label": meta.get("label", ""),
            "removed_input": removed,
            "prestated_prediction": prediction,
            "UALER_full": rnd(full["primary"]["UALER"]),
            "UALER_ablation": rnd(m["primary"]["UALER"]),
            "delta_UALER_ablation_minus_full": rnd(du),
            "delta_UALER_ci_lo": rnd(du_lo),
            "delta_UALER_ci_hi": rnd(du_hi),
            "UALER_p_raw": rnd(h_u.get("p_raw"), 6),
            "UALER_p_holm_adjusted": rnd(h_u.get("p_holm_adjusted"), 6),
            "UALER_rejected_at_alpha": h_u.get("rejected_at_alpha"),
            "ATPR_full": rnd(full["primary"]["ATPR"]),
            "ATPR_ablation": rnd(m["primary"]["ATPR"]),
            "delta_ATPR_ablation_minus_full": rnd(da),
            "delta_ATPR_ci_lo": rnd(da_lo),
            "delta_ATPR_ci_hi": rnd(da_hi),
            "ATPR_p_raw": rnd(h_a.get("p_raw"), 6),
            "ATPR_p_holm_adjusted": rnd(h_a.get("p_holm_adjusted"), 6),
            "ATPR_rejected_at_alpha": h_a.get("rejected_at_alpha"),
            "FRR_ablation": rnd(m["secondary"].get("FRR")),
            "OSMR_ablation": rnd(m["secondary"].get("OSMR")),
            "BER_ablation": rnd(m["secondary"].get("BER")),
            "verdict_CONTRADICTED": m["verdict_distribution"].get("CONTRADICTED", 0),
            "verdict_UNVERIFIED": m["verdict_distribution"].get("UNVERIFIED", 0),
            "prediction_check": verdict,
            "prediction_check_rule": rule,
        })

    df = pd.DataFrame(rows)
    tally = df["prediction_check"].value_counts().to_dict()
    tally_str = ", ".join(f"{v} {k}" for k, v in sorted(tally.items()))
    disagreeing = sorted(df.loc[df["prediction_check"] == "INCONSISTENT", "ablation_id"])
    unchecked = sorted(df.loc[df["prediction_check"] == "NOT_CHECKABLE", "ablation_id"])
    notes = [
        "Sources: results/primary_test_evaluation.json (metrics, bootstrap, holm_families "
        "F5_ablations) and results/baseline_deviation_log.json (ablation_metadata).",
        f"{len(df)} single-factor ablations, one per CONTRACT.md Section 9 requirement. The "
        "set is frozen in preregistration.yaml required_ablations so it cannot shrink to the "
        "subset that happens to look best.",
        "delta_* is defined as (ablation - full AegisLink). The underlying artifact stores "
        "(full - ablation), so this sheet negates the point estimate and swaps the CI "
        "endpoints. A POSITIVE delta_UALER therefore means the ablation is WORSE on security; "
        "a NEGATIVE delta_ATPR means it is worse on utility.",
        "p-values are two-sided paired-bootstrap values, Holm-corrected within the "
        f"F5_ablations family (alpha = {f5['alpha']}, {f5['n_members']} members, "
        f"{f5['n_rejected']} rejected). Correction is applied within a family, never across.",
        "prestated_prediction is the qualitative prediction registered in "
        "aegislink/ablations.py ABLATION_PREDICTIONS before the ablation was run. It is prose, "
        "not a number.",
        "prediction_check turns that prose into a reproducible test with three outcomes: "
        "CONSISTENT, INCONSISTENT, or NOT_CHECKABLE (the prediction cannot be tested on this "
        f"corpus at all). The numeric tolerances it uses (flat/near-zero {FLAT_TOL}, small "
        f"{SMALL_TOL}, tiny {TINY_TOL}, sharp {SHARP_TOL}) are conventions chosen by this "
        "exporter, NOT preregistered thresholds. prediction_check_rule states the exact rule "
        "applied on every row so a checker can reject the convention and still read the raw "
        "deltas.",
        f"Observed under those conventions: {tally_str}, out of {len(df)} ablations. "
        "A disagreement is reported, not reconciled (aegislink/ablations.py "
        "prediction_policy).",
        (f"INCONSISTENT: {', '.join(disagreeing)}. Read the delta columns on those rows "
         "before drawing a conclusion -- a prediction can fail on the axis it named while "
         "the ablation still degrades badly on the other axis."
         if disagreeing else "No prediction came out INCONSISTENT."),
        (f"NOT_CHECKABLE: {', '.join(unchecked)}. See prediction_check_rule on those rows; "
         "a floor effect is not a failed prediction."
         if unchecked else "Every prediction was checkable on this corpus."),
    ]
    return df, notes


# ----------------------------------------------------------------------------- sheet 5
def sheet_baselines(prim: dict, dev: dict, pilot: dict) -> tuple[pd.DataFrame, list[str]]:
    """The ten Section 10 baselines, with the Section 12 non-triviality box."""
    order: Sequence[str] = prim["method"]["configuration_order"]
    base_ids = [c for c in order if family_of(c) == "Baseline"]
    M = prim["metrics"]
    boot = prim["bootstrap"][CONFIRMATORY_REGIME]
    f4 = prim["holm_families"][CONFIRMATORY_REGIME]["F4_all_baselines"]
    meta_all = (dev.get("baselines") or {}).get("baselines") or {}
    rbs = prim["reference_baseline_selection"]
    ref = rbs["reference_baseline"]
    gate = pilot["frozen_hard_gates"]["baseline_not_trivial"]
    ual_ceiling = gate["max_allowed_best_baseline_ualer"]
    atpr_floor = gate["min_required_baseline_atpr"]
    ranking = {r["baseline_id"]: r for r in rbs["ranking"]}

    rows = []
    for cfg in base_ids:
        meta = meta_all.get(cfg, {})
        val = M[cfg][SELECTION_REGIME]["primary"]
        tst = M[cfg][CONFIRMATORY_REGIME]
        u_lo, u_hi, u_dg = ci_pair(estimate(boot, cfg, "UALER"), "bias_corrected")
        a_lo, a_hi, a_dg = ci_pair(estimate(boot, cfg, "ATPR"), "bias_corrected")
        h_u = holm_lookup(f4, f"UALER:aegislink_vs_{cfg}")
        h_a = holm_lookup(f4, f"ATPR:aegislink_vs_{cfg}")
        rank = ranking.get(cfg, {})

        rows.append({
            "baseline_id": cfg,
            "label": meta.get("label", ""),
            "contract_ref": meta.get("contract_ref", ""),
            "is_surrogate": meta.get("is_surrogate"),
            "surrogate_note": meta.get("surrogate_note", ""),
            "restricted_inputs": ", ".join(
                r["input"] for r in (meta.get("restrictions") or [])) or "(none)",
            "is_reference_baseline": cfg == ref,
            "validation_UALER": rnd(val["UALER"]),
            "validation_ATPR": rnd(val["ATPR"]),
            "validation_meets_atpr_floor": rank.get("meets_atpr_constraint"),
            "test_UALER": rnd(tst["primary"]["UALER"]),
            "test_UALER_ci_lo": rnd(u_lo),
            "test_UALER_ci_hi": rnd(u_hi),
            "test_UALER_ci_degenerate": u_dg,
            "test_ATPR": rnd(tst["primary"]["ATPR"]),
            "test_ATPR_ci_lo": rnd(a_lo),
            "test_ATPR_ci_hi": rnd(a_hi),
            "test_ATPR_ci_degenerate": a_dg,
            "test_FRR": rnd(tst["secondary"].get("FRR")),
            "test_OSMR": rnd(tst["secondary"].get("OSMR")),
            "test_BER": rnd(tst["secondary"].get("BER")),
            "test_abstention_rate": rnd(tst["secondary"].get("abstention_rate")),
            "decision_latency_mean_ms": rnd(tst.get("mean_decision_ms"), 4),
            "UALER_p_holm_adjusted": rnd(h_u.get("p_holm_adjusted"), 6),
            "UALER_rejected_at_alpha": h_u.get("rejected_at_alpha"),
            "ATPR_p_holm_adjusted": rnd(h_a.get("p_holm_adjusted"), 6),
            "ATPR_rejected_at_alpha": h_a.get("rejected_at_alpha"),
            "nontriviality_ualer_ceiling": ual_ceiling,
            "nontriviality_atpr_floor": atpr_floor,
            # trivially_sufficient is written as a live formula in style_sheet().
            "trivially_sufficient": None,
        })

    df = pd.DataFrame(rows)
    g = pilot["gates"]["baseline_not_trivial"]
    notes = [
        "Sources: results/primary_test_evaluation.json (metrics, bootstrap, holm_families "
        "F4_all_baselines, reference_baseline_selection), results/baseline_deviation_log.json "
        "(baselines.baselines), results/pilot_decision_report.json (frozen_hard_gates, gates).",
        f"{len(df)} baselines, the complete CONTRACT.md Section 10 suite. Every baseline sees "
        "byte-identical retrieved evidence to AegisLink; the restricted_inputs column names "
        "what each one is denied by construction.",
        "NON-TRIVIALITY BOX (CONTRACT.md Section 12 / gate baseline_not_trivial): a baseline "
        f"is TRIVIALLY SUFFICIENT if it achieves UALER <= {ual_ceiling} AND ATPR >= "
        f"{atpr_floor} simultaneously. If any baseline were, AegisLink would be unnecessary "
        "and the pilot would have to stop.",
        "The trivially_sufficient column is a LIVE EXCEL FORMULA over the validation "
        "UALER/ATPR cells and the two threshold cells on the same row -- edit a threshold and "
        "it recalculates. The gate is decided on the validation split so the test split stays "
        "untouched for the confirmatory analysis.",
        f"Recorded gate outcome: {g['status']}. {g['conditions'][0]['reason']}",
        "is_surrogate = TRUE marks a baseline reimplemented as a deterministic stand-in "
        "rather than the published system (no language model is invoked anywhere in this "
        "artifact). Read those rows as a lower bound on what the real system might achieve.",
        f"Reference baseline for all primary comparisons = {ref}. Selection rule: "
        f"{rbs['selection_rule']}",
    ]
    return df, notes


# ----------------------------------------------------------------------------- sheet 6
def sheet_gates(pilot: dict) -> tuple[pd.DataFrame, list[str]]:
    """All ten Section 12 Go/No-Go hard-gate conditions."""
    gates = pilot["gates"]
    rows = []
    for c in pilot["conditions_flat"]:
        g = gates.get(c["gate"], {})
        ev = c.get("evidence") or {}
        rows.append({
            "gate": c["gate"],
            "gate_status": g.get("status"),
            "condition_id": c["condition_id"],
            "status": c["status"],
            "frozen_key": c["frozen_key"],
            "frozen_value": c["frozen_value"],
            "observed": rnd(c.get("observed")),
            "margin": rnd(c.get("margin")),
            "test": c["test"],
            "reason": c.get("reason", ""),
            "evaluation_split": ev.get("evaluation_split", ""),
            "gate_intent": (g.get("intent") or "").strip().replace("\n", " "),
            "gate_pass_condition": (g.get("pass_condition") or "").strip().replace("\n", " "),
        })

    df = pd.DataFrame(rows)
    d = pilot["decision"]
    md = pilot["metadata"]
    notes = [
        f"Source: results/pilot_decision_report.json  (gate evaluator "
        f"{md['gate_evaluator_version']}, builder {md['builder']}).",
        f"OVERALL DECISION: {d['overall_decision']}  (go_declared = {d['go_declared']}).  "
        f"{d['n_passed']} PASS, {d['n_failed']} FAIL, {d['n_not_evaluable']} NOT_EVALUABLE "
        f"across {d['n_conditions_declared']} declared conditions. Combination rule: "
        f"{d['combination_rule']}.",
        f"Not evaluable: {', '.join(d['not_evaluable_conditions']) or '(none)'}.  "
        f"Failed: {', '.join(d['failed_conditions']) or '(none)'}.",
        "Three-valued logic is deliberate. NOT_EVALUABLE means the pilot surrogate could not "
        "measure the condition at all -- it is neither a pass nor a fail, and it is why the "
        f"overall decision is {d['overall_decision']} rather than GO.",
        f"INCONCLUSIVE: {d['decision_semantics']['INCONCLUSIVE']}",
        f"NO_GO: {d['decision_semantics']['NO_GO']}",
        d["threshold_mutability"],
        f"Gate decisions use the {md['ci_flavour_used_for_decisions']} CI flavour at "
        f"confidence {md['confidence_level']}, B = {md['n_resamples_used']} resamples, on the "
        f"'{md['confirmatory_split']}' split for confirmatory endpoints.",
    ]
    return df, notes


# ----------------------------------------------------------------------------- sheet 7
def sheet_provenance(prim: dict, adapt: dict, pilot: dict) -> tuple[pd.DataFrame, list[str]]:
    """Digests, fingerprints, seeds and resample counts for the whole run."""
    rows: list[dict] = []

    def add(section: str, key: str, value: Any) -> None:
        rows.append({"section": section, "key": key, "value": value})

    md = prim["metadata"]
    for k in ("step", "builder", "contract_ref", "read_phase", "n_queries",
              "n_configurations", "n_resamples_used", "preregistered_resamples",
              "is_results_run", "quick_mode", "elapsed_seconds",
              "replay_fingerprint", "replay_fingerprint_verified"):
        add("primary_evaluation", k, md.get(k))
    add("primary_evaluation", "regimes_reported", ", ".join(md.get("regimes_reported") or []))
    add("primary_evaluation", "confirmatory_regimes",
        ", ".join(md.get("confirmatory_regimes") or []))

    fp = prim["frozen_parameters"]
    for k in ("source", "source_sha256", "replay_fingerprint", "n_fit_queries",
              "n_fit_triples", "fit_regime", "refit_on_evaluation_splits"):
        add("frozen_parameters", k, fp.get(k))

    for stratum in ("A", "B"):
        S = adapt["strata"][stratum]
        add(f"adaptive_stratum_{stratum}", "regime", S["regime"])
        add(f"adaptive_stratum_{stratum}", "primary_claim_eligible",
            S["primary_claim_eligible"])
        add(f"adaptive_stratum_{stratum}", "replay_fingerprint",
            S["replay"]["replay_fingerprint"])
        for k, v in S["replay"]["fingerprint_material"].items():
            add(f"adaptive_stratum_{stratum}", f"material.{k}", v)
        for k in ("n_pages", "n_templates", "n_target_entities", "n_attack_domains"):
            add(f"adaptive_stratum_{stratum}", f"surface.{k}", S["replay"]["surface"].get(k))

    boot = prim["bootstrap"][CONFIRMATORY_REGIME]
    for k in ("regime", "n_units", "n_resamples", "seed", "reference_baseline"):
        add("bootstrap_test", k, boot.get(k))

    pmd = pilot["metadata"]
    for k in ("gate_evaluator_version", "prereg_sha256", "ci_flavour_used_for_decisions",
              "confidence_level", "n_resamples_used", "confirmatory_split",
              "reference_baseline"):
        add("pilot_decision", k, pmd.get(k))

    for rel in ("CONTRACT.md", "preregistration.yaml", "configs/frozen_params.json",
                "configs/splits.yaml", "configs/benchmark_freeze.json",
                "results/primary_test_evaluation.json",
                "results/adaptive_robustness_evaluation.json",
                "results/pilot_decision_report.json",
                "results/baseline_deviation_log.json",
                "results/evidence_family_identifiability.json"):
        p = ROOT / rel
        add("artifact_sha256", rel, sha256_of(p) if p.is_file() else "(absent)")

    notes = [
        "Supplementary sheet: run provenance for an AE checker who wants to confirm that the "
        "numbers in the other six sheets came from the frozen replay and not from a re-fit.",
        "A replay fingerprint binds the corpus digest, index digest, retriever configuration, "
        "encoder id and query set. The evaluation refuses to run on a mismatch, so a divergent "
        "environment fails loudly rather than quietly producing different numbers.",
        "refit_on_evaluation_splits must read False. If it ever reads True the run is not "
        "confirmatory and every number in this workbook should be discarded.",
        "artifact_sha256 rows are computed at export time from the files on disk in this "
        "repository, so they change if you re-run the evaluation. Compare them against the "
        "digests in configs/frozen_params.json for the governance documents.",
    ]
    return pd.DataFrame(rows), notes


# -------------------------------------------------------------------------------- styling
def style_sheet(ws, df: pd.DataFrame, notes: Sequence[str], *,
                group_col: str | None = None, status_cols: Sequence[str] = ()) -> None:
    """Apply header styling, note block, widths, freeze panes and autofilter.

    The layout is: row 1 title, rows 2..n notes, one blank row, then the header row and
    the data. ``header_row`` is computed rather than assumed so notes can vary in length.
    """
    n_notes = len(notes)
    header_row = n_notes + 3          # title + notes + blank
    n_cols = len(df.columns)

    ws.cell(row=1, column=1, value=ws.title.replace("_", " ")).font = Font(
        name=FONT_NAME, size=12, bold=True, color="1B6CA8")
    for i, note in enumerate(notes, start=2):
        c = ws.cell(row=i, column=1, value=note)
        c.font = NOTE_FONT
        c.alignment = Alignment(vertical="top", wrap_text=False)

    for j, col in enumerate(df.columns, start=1):
        c = ws.cell(row=header_row, column=j, value=col)
        c.font = HDR_FONT
        c.fill = HDR_FILL
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = BORDER

    for i, (_, rec) in enumerate(df.iterrows(), start=header_row + 1):
        fill = GROUP_FILLS.get(str(rec.get(group_col)), None) if group_col else None
        for j, col in enumerate(df.columns, start=1):
            v = rec[col]
            if pd.isna(v):
                v = None
            elif hasattr(v, "item"):                 # numpy scalar -> python scalar
                v = v.item()
            c = ws.cell(row=i, column=j, value=v)
            c.font = BODY_FONT
            c.border = BORDER
            if fill is not None:
                c.fill = fill
            if col in status_cols and isinstance(v, str) and v in STATUS_FILLS:
                c.fill = STATUS_FILLS[v]
            if isinstance(v, float):
                c.number_format = FMT_MS if col.endswith("_ms") else (
                    FMT_P if "_p_" in col or col.endswith("_p_raw") else FMT_RATE)

    # column widths, capped so a long prose cell cannot blow the layout out
    for j, col in enumerate(df.columns, start=1):
        longest = max([len(str(col))] + [len(str(v)) for v in df[col].head(60)])
        ws.column_dimensions[get_column_letter(j)].width = min(max(longest + 2, 11), 46)

    ws.freeze_panes = ws.cell(row=header_row + 1, column=2)
    if len(df) > 0:
        ws.auto_filter.ref = (f"A{header_row}:"
                              f"{get_column_letter(n_cols)}{header_row + len(df)}")
    ws.sheet_view.showGridLines = False


def add_nontriviality_formula(ws, df: pd.DataFrame, n_notes: int) -> None:
    """Write ``trivially_sufficient`` as a live formula, not a Python-computed boolean.

    The Section 12 triviality test is a conjunction of two comparisons against two
    thresholds that sit on the same row. Writing it as a formula means a checker can edit
    a threshold cell and watch the verdict update, which is exactly the audit an AE
    reviewer wants to perform.
    """
    header_row = n_notes + 3
    cols = {c: get_column_letter(i) for i, c in enumerate(df.columns, start=1)}
    u, a = cols["validation_UALER"], cols["validation_ATPR"]
    ceil_, floor_ = cols["nontriviality_ualer_ceiling"], cols["nontriviality_atpr_floor"]
    tgt = cols["trivially_sufficient"]
    for i in range(header_row + 1, header_row + 1 + len(df)):
        ws[f"{tgt}{i}"] = (f'=IF(AND({u}{i}<={ceil_}{i},{a}{i}>={floor_}{i}),'
                           f'"TRIVIALLY_SUFFICIENT","NOT_TRIVIAL")')
        ws[f"{tgt}{i}"].font = BODY_FONT
        ws[f"{tgt}{i}"].border = BORDER


# ----------------------------------------------------------------------------------- main
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "experimental_results.xlsx"),
                    help="output workbook path (default: <repo>/experimental_results.xlsx)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out

    print("=" * 74)
    print("Exporting experimental results to Excel")
    print("=" * 74)

    prim = load("primary_test_evaluation.json")
    adapt = load("adaptive_robustness_evaluation.json")
    pilot = load("pilot_decision_report.json")
    dev = load("baseline_deviation_log.json")
    efi = load("evidence_family_identifiability.json")
    print(f"[read] 5 result artifacts, "
          f"{prim['metadata']['n_configurations']} configurations, "
          f"B = {prim['metadata']['n_resamples_used']} resamples, "
          f"confirmatory = {prim['metadata']['is_results_run']}")

    builders = [
        ("Primary_Test_Results", lambda: sheet_primary(prim), {"group_col": "family"}),
        ("Adaptive_Attack_Results", lambda: sheet_adaptive(adapt, prim),
         {"group_col": "family"}),
        ("Per_Action_Breakdown", lambda: sheet_per_action(prim, adapt),
         {"group_col": "family"}),
        ("Ablations_Analysis", lambda: sheet_ablations(prim, dev),
         {"status_cols": ["prediction_check"]}),
        ("Baselines_Comparison", lambda: sheet_baselines(prim, dev, pilot), {}),
        ("Pilot_Decision_Gates", lambda: sheet_gates(pilot),
         {"status_cols": ["status", "gate_status"]}),
        ("Run_Provenance", lambda: sheet_provenance(prim, adapt, pilot), {}),
    ]

    frames: dict[str, tuple[pd.DataFrame, list[str], dict]] = {}
    for name, build, style_kw in builders:
        df, notes = build()
        if df.empty:
            raise SystemExit(f"ERROR: sheet {name} came out empty; refusing to write "
                             f"a workbook with an unpopulated required worksheet.")
        frames[name] = (df, notes, style_kw)
        print(f"[build] {name:<26} {len(df):>4} rows x {len(df.columns):>2} cols")

    out.parent.mkdir(parents=True, exist_ok=True)
    # Sheets are built cell by cell with openpyxl rather than via DataFrame.to_excel,
    # because every sheet carries a note block above its header and one sheet needs a live
    # formula column. Letting pandas lay the values down first and then overwriting them
    # would leave a confusing dead write in the file's history for no benefit.
    wb = Workbook()
    wb.remove(wb.active)
    for name, (df, notes, style_kw) in frames.items():
        ws = wb.create_sheet(title=name)
        style_sheet(ws, df, notes, **style_kw)
        if name == "Baselines_Comparison":
            add_nontriviality_formula(ws, df, len(notes))
    wb.save(out)

    size_kb = out.stat().st_size / 1024
    print("-" * 74)
    print(f"[done] {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}  "
          f"({size_kb:.0f} KB, {len(frames)} worksheets)")
    print(f"[note] evidence-family identifiability audit covers "
          f"{len(efi['audit']['families'])} evidence families; "
          f"{sum(1 for f in efi['audit']['families'].values() if f.get('degenerate'))} are "
          f"flagged degenerate on this corpus (see results/"
          f"evidence_family_identifiability.json).")
    print(f"[note] required sheets present: "
          f"{', '.join(n for n, _, _ in builders[:6])}")
    print(f"[note] Baselines_Comparison.trivially_sufficient is an Excel formula and will "
          f"read as blank until the workbook is opened or recalculated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
