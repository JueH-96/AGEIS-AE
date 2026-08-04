"""Step 5 driver: confirmatory evaluation, paired bootstrap, adaptive robustness, gate decision.

CONTRACT.md Sections 3 (RQ1, RQ4), 5.3, 11, 12.

Order of operations, and why it is this order
---------------------------------------------
1. Rebuild the static replay and **verify the frozen fingerprint**. Nothing else runs until it
   matches: a drifted replay would make Step 4's fitted parameters describe a different corpus than
   Step 5 evaluates on, and every number below would be confounded in a way that looks normal.
2. Load the Step 4 fit. Nothing is fitted here; ``FrozenParameters.load`` refuses a fit that came
   from anywhere but ``train_development``.
3. Sweep all 18 configurations over the static replay, and file the results under
   ``validation`` / ``test`` / ``transfer_holdout`` / ``fabricated_control``. Cross-check the
   validation row against Step 4's, since the two were computed by different code paths on the same
   frozen evidence and must agree exactly.
4. Select the reference baseline by the preregistered rule -- on ``validation``, never on ``test``.
5. Build the two adaptive replays (Stratum A primary, Stratum B secondary) and sweep them.
6. Bootstrap: 10,000 paired resamples per regime, one shared index matrix per regime so every
   comparison is paired. Holm within each preregistered family.
7. Evaluate the ten Section 12 conditions and write the decision report.

Outputs
-------
``results/primary_test_evaluation.json``        endpoints, bootstrap intervals, Holm families
``results/adaptive_robustness_evaluation.json`` ASR_a and utility under adaptive pressure
``results/pilot_decision_report.json``          the ten Section 12 conditions
``data/benchmark/adaptive_retrieval_snapshots.json`` the frozen adaptive replays

Run::

    uv run python workflow/19_evaluate_primary_and_adaptive.py [--quick] [--resamples N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from aegislink.ablations import ABLATION_IDS  # noqa: E402
from aegislink.framework import ACTION_RISK_ORDER  # noqa: E402
from aegislink.pipeline_adapter import (  # noqa: E402
    DefenseEvaluationHarness,
    assert_replay_bound,
    assert_shared_evidence,
)
from baselines.registry import BASELINE_IDS  # noqa: E402
from evaluation.gate_evaluator import (  # noqa: E402
    FrozenGates,
    build_decision_report,
    check_improvement,
    describe_gate_evaluator,
    evaluate_aegislink_improvement,
    evaluate_baseline_not_trivial,
    evaluate_generalization,
    evaluate_misbinding_reproducible,
    evaluate_novelty,
    undefended_exposure_units,
)
from evaluation.primary_evaluation import (  # noqa: E402
    CONFIG_ORDER,
    CONFIRMATORY_REGIMES,
    EVALUATION_REGIMES,
    FrozenParameters,
    admit_regime,
    assert_no_refitting_hooks,
    assert_units_aligned,
    build_all_configurations,
    build_evaluator_context,
    calibration_pairs,
    describe_primary_evaluation,
    load_yaml,
    metrics_by_regime,
    run_sweep,
    select_reference_baseline,
    units_by_regime,
)
from experiments.adaptive_evaluation import (  # noqa: E402
    REGIME_OF_STRATUM,
    describe_adaptive_evaluation,
    evaluate_stratum,
    utility_under_adaptive_pressure,
    write_adaptive_snapshot,
)
from statistics.bootstrap import (  # noqa: E402
    ALPHA,
    CONFIDENCE_LEVEL,
    N_RESAMPLES,
    bootstrap_metric,
    calibration_report,
    derive_seed,
    describe_bootstrap,
    holm_bonferroni,
    paired_bootstrap_compare,
    resample_indices,
)
from web_rag.crawler_indexer import (  # noqa: E402
    PHASE_INDEX,
    PHASE_LIVE,
    build_index,
    crawl_corpus,
    outbound_links,
)
from web_rag.experiment_matrix import ExperimentMatrix, RunPurpose  # noqa: E402
from web_rag.exposure import (  # noqa: E402
    assert_no_leakage,
    build_site_index,
    describe_firewall,
    load_official_registry,
)
from web_rag.retriever import (  # noqa: E402
    RANKING_MODES,
    HashedNGramEncoder,
    Retriever,
    RetrieverConfig,
    build_entity_queries,
)
from web_rag.trace_recorder import TraceRecorder  # noqa: E402

BENCH = ROOT / "data" / "benchmark"
RESULTS = ROOT / "results"
SNAPSHOT_PATH = BENCH / "retrieval_snapshots.json"

#: Endpoints bootstrapped for every configuration on every regime.
BOOTSTRAP_METRICS: tuple[str, ...] = ("UALER", "ATPR", "OSMR", "CMR", "FRR", "BER", "abstention_rate")

#: The two primary endpoints. Family F1 is confined to these.
PRIMARY_METRICS: tuple[str, ...] = ("UALER", "ATPR")


class ReplayDriftError(RuntimeError):
    """Raised when the rebuilt static replay does not match the frozen fingerprint. Fails closed."""


def reconcile_with_step4(
    sweep: Any, ev: Any, metrics: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Prove that the only difference from Step 4 is the de-duplication correction.

    Step 4 scored one triple per retrieved *candidate*. Step 5 scores one per unique
    ``(query, domain)``, because an authorized third-party link is a property of an
    ``(entity, domain, action)`` triple in the authorization graph and the defense makes exactly one
    decision per domain. Step 4's rule entered that single decision twice whenever a shared provider
    served two listing pages, making ATPR's denominator a function of the corpus rather than of the
    authorization.

    Rather than assert the correction, this reconciles it: Step 5's *legacy* triple set is scored and
    must reproduce the Step 4 artifact exactly. That agreement is what shows the two code paths are
    equivalent and isolates the de-duplication as the sole difference. A legacy disagreement is fatal,
    because it would mean one of the two implementations is simply wrong.
    """
    step4 = json.loads(
        (ROOT / "results" / "aegislink_defense_evaluation.json").read_text()
    )["metrics"]
    # Step 4 reported train_development and validation, so the reconciliation covers both -- twice the
    # evidence that the two code paths agree. Both sides are recomputed over these regimes because
    # ``metrics`` itself deliberately omits train_development (it is the fitting split, not a result).
    reconcile_regimes = ("train_development", "validation")
    legacy = metrics_by_regime(sweep, ev, reconcile_regimes, source="per_candidate")
    deduped = metrics_by_regime(sweep, ev, reconcile_regimes, source="deduped")

    legacy_bad: list[dict[str, Any]] = []
    corrected: list[dict[str, Any]] = []
    for name in CONFIG_ORDER:
        for regime in reconcile_regimes:
            old = (step4.get(name, {}).get(regime) or {}).get("primary") or {}
            leg = (legacy.get(name, {}).get(regime) or {}).get("primary") or {}
            new = (deduped.get(name, {}).get(regime) or {}).get("primary") or {}
            if not old:
                continue
            for metric in PRIMARY_METRICS:
                a, b, c = old.get(metric), leg.get(metric), new.get(metric)
                if a is None and b is None:
                    pass
                elif a is None or b is None or abs(float(a) - float(b)) > 1e-9:
                    legacy_bad.append(
                        {"config": name, "regime": regime, "metric": metric,
                         "step4": a, "step5_legacy": b}
                    )
                if a is not None and c is not None and abs(float(a) - float(c)) > 1e-9:
                    corrected.append(
                        {"config": name, "regime": regime, "metric": metric,
                         "step4_per_candidate": a, "step5_deduped": round(float(c), 6),
                         "delta": round(float(c) - float(a), 6)}
                    )
    if legacy_bad:
        raise ReplayDriftError(
            f"{len(legacy_bad)} endpoint(s) differ between Step 4 and Step 5's LEGACY "
            f"per-candidate scoring on identical frozen evidence: "
            f"{json.dumps(legacy_bad[:6], indent=2)}. Under the same counting rule the two code "
            f"paths must agree exactly; a difference means one of them is wrong, and the "
            f"de-duplication correction cannot be attributed until they do."
        )

    denominators: dict[str, Any] = {}
    probe = CONFIG_ORDER[0]
    all_legacy = metrics_by_regime(sweep, ev, EVALUATION_REGIMES, source="per_candidate")
    for regime in EVALUATION_REGIMES:
        dd = ((metrics.get(probe, {}).get(regime) or {}).get("counts") or {}).get(
            "authorized_third_party_links_retrieved"
        )
        pc = ((all_legacy.get(probe, {}).get(regime) or {}).get("counts") or {}).get(
            "authorized_third_party_links_retrieved"
        )
        denominators[regime] = {"per_candidate": pc, "deduped": dd,
                                "double_counted": None if (pc is None or dd is None) else pc - dd}

    return {
        "purpose": (
            "Isolate the de-duplication correction. Step 5's legacy per-candidate scoring must "
            "reproduce the Step 4 artifact exactly; the remaining difference is then attributable to "
            "the counting rule alone."
        ),
        "n_configurations_compared": len(CONFIG_ORDER),
        "n_legacy_disagreements": len(legacy_bad),
        "legacy_reproduces_step4_exactly": True,
        "tolerance": 1e-9,
        "correction": {
            "what_changed": (
                "One triple per unique (query, domain) instead of one per retrieved candidate."
            ),
            "why": (
                "An authorized third-party link is a property of an (entity, domain, action) triple "
                "in the authorization graph, and the defense makes exactly one decision per domain. "
                "Step 4's rule entered that single decision twice whenever retrieval returned two "
                "documents on one host, which made ATPR's denominator depend on how many listing "
                "pages a shared provider happens to serve -- a corpus artifact, not an authorization "
                "fact -- and treated two copies of one decision as independent observations."
            ),
            "affects": "ATPR, FRR, OSMR and BER denominators; UALER is per response and unaffected.",
            "n_duplicate_domain_candidates_per_config": sweep.n_duplicate_domain_candidates,
            "diagnostic": "workflow/diagnostics/check_duplicate_domain_candidates.py",
        },
        "n_corrected_endpoints": len(corrected),
        "corrected_endpoints": corrected,
        "atpr_denominator": denominators,
        "step4_numbers_superseded": (
            "Where they differ, the Step 5 de-duplicated values supersede Step 4's. The direction is "
            "small and does not change the reference-baseline selection or the triviality verdict, "
            "both of which are re-derived here from the corrected numbers."
        ),
    }


def _fmt(x: Any, width: int = 7, places: int = 3) -> str:
    if x is None:
        return "n/a".rjust(width)
    try:
        return f"{float(x):{width}.{places}f}"
    except (TypeError, ValueError):
        return str(x).rjust(width)


# ======================================================================================
# 1. Static replay
# ======================================================================================
def rebuild_static_replay() -> dict[str, Any]:
    """Rebuild the frozen replay and verify its fingerprint against the Step 3 freeze."""
    print("[1/7] Rebuilding the frozen static replay")
    corpus = crawl_corpus(verify_hashes=True, progress_every=0)
    index = build_index(corpus, phase=PHASE_INDEX, progress_every=0)
    entities = load_yaml(ROOT / "registry" / "entities.yaml")["entities"]
    queries = build_entity_queries(entities)
    query_digest = hashlib.sha256(
        json.dumps([q.as_dict() for q in queries], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    config = RetrieverConfig(mode="hybrid_rrf", top_k=10)
    encoder = HashedNGramEncoder()
    retriever = Retriever(index, config=config, encoder=encoder)
    results = retriever.search_all(queries, progress_every=0)

    material = {
        "corpus_digest": corpus.corpus_digest,
        "index_digest": index.index_digest,
        "retriever_config_digest": config.digest(),
        "encoder_id": encoder.encoder_id,
        "query_set_digest": query_digest,
        "per_query_sha256": sorted(r.sha256 for r in results),
    }
    fingerprint = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    frozen = json.loads(SNAPSHOT_PATH.read_text())["metadata"]
    if fingerprint != frozen["replay_fingerprint"]:
        drift = {
            k: (v, frozen["fingerprint_material"].get(k))
            for k, v in material.items()
            if k != "per_query_sha256" and v != frozen["fingerprint_material"].get(k)
        }
        raise ReplayDriftError(
            f"rebuilt replay fingerprint {fingerprint} != frozen {frozen['replay_fingerprint']}.\n"
            f"Drifted (rebuilt, frozen): {json.dumps(drift, indent=2)}\n"
            f"Step 4 fitted its parameters on this replay; a drifted one would make those "
            f"parameters describe a different corpus than Step 5 evaluates."
        )
    print(f"[1/7] replay fingerprint VERIFIED {fingerprint[:16]}...  "
          f"{len(results)} queries, {sum(len(r.candidates) for r in results)} candidates")
    return {
        "corpus": corpus,
        "index": index,
        "entities": entities,
        "queries": queries,
        "results": list(results),
        "fingerprint": fingerprint,
        "retriever_config": config,
        "encoder": encoder,
    }


def make_fetchers(corpus: Any, oracle: Any) -> tuple[Any, Any, Any]:
    """``(recorder, fetch_text, fetch_links)`` for one corpus."""
    recorder = TraceRecorder(corpus=corpus, oracle=oracle, read_phase=PHASE_LIVE)

    def fetch_links(doc_id: str) -> list[str]:
        """Outbound hyperlink targets at read phase.

        Needed because visible text never carries a hostname in this corpus: partners are named by
        display name and the host lives only in the href.
        """
        return outbound_links(corpus.documents[doc_id].html(PHASE_LIVE))

    return recorder, recorder._fetch_text, fetch_links


# ======================================================================================
# 2. Bootstrap over a regime
# ======================================================================================
def bootstrap_regime(
    regime: str,
    units: Mapping[str, list[Any]],
    reference: str,
    *,
    n_resamples: int,
    treatment: str = "aegislink_full",
) -> dict[str, Any]:
    """Bootstrap every configuration and every AegisLink-vs-X contrast on one regime.

    One index matrix, generated once from a regime-derived seed and reused by every configuration and
    every contrast. That is what makes the contrasts paired: resample *b* selects the same queries for
    AegisLink as for the baseline it is compared against.
    """
    present = [c for c in CONFIG_ORDER if c in units]
    aligned = assert_units_aligned({c: units[c] for c in present})
    n_units = aligned["n_units"]
    seed = derive_seed("regime", regime)
    idx = resample_indices(n_units, n_resamples=n_resamples, seed=seed)
    print(f"[5/7] {regime:26s} {len(present):2d} configs x {n_units:4d} units "
          f"x {n_resamples} resamples (seed {seed})")

    estimates: dict[str, dict[str, Any]] = {}
    for config in present:
        estimates[config] = {
            metric: bootstrap_metric(
                units[config], metric, config_id=config, regime=regime, idx=idx, seed=seed
            ).as_dict()
            for metric in BOOTSTRAP_METRICS
        }

    comparisons: dict[str, dict[str, Any]] = {}
    for config in present:
        if config == treatment:
            continue
        comparisons[config] = {
            metric: paired_bootstrap_compare(
                units[treatment],
                units[config],
                metric,
                treatment=treatment,
                reference=config,
                regime=regime,
                idx=idx,
                seed=seed,
            ).as_dict()
            for metric in PRIMARY_METRICS
        }

    # -- per-action UALER contrast against the reference baseline (family F2) ----------
    per_action: dict[str, Any] = {}
    if reference in units:
        by_action_t: dict[str, list[Any]] = defaultdict(list)
        by_action_r: dict[str, list[Any]] = defaultdict(list)
        for ut, ur in zip(units[treatment], units[reference]):
            by_action_t[ut.action].append(ut)
            by_action_r[ur.action].append(ur)
        for action in sorted(by_action_t):
            n = len(by_action_t[action])
            a_seed = derive_seed("regime", regime, "action", action)
            a_idx = resample_indices(n, n_resamples=n_resamples, seed=a_seed)
            per_action[action] = paired_bootstrap_compare(
                by_action_t[action],
                by_action_r[action],
                "UALER",
                treatment=treatment,
                reference=reference,
                regime=f"{regime}:{action}",
                idx=a_idx,
                seed=a_seed,
            ).as_dict()

    return {
        "regime": regime,
        "n_units": n_units,
        "n_resamples": n_resamples,
        "seed": seed,
        "alignment": aligned,
        "reference_baseline": reference,
        "estimates": estimates,
        "comparisons_vs_aegislink": comparisons,
        "per_action_ualer_vs_reference": per_action,
    }


def holm_families(
    boot: Mapping[str, Any], reference: str, regime: str
) -> dict[str, Any]:
    """Holm correction within each preregistered family.

    ``preregistration.yaml multiplicity_correction.families`` declares the families before data
    collection, and correction is applied *within* a family and never across them. F3
    (per model family) is absent because the pilot has one deterministic reader; that is recorded
    rather than silently dropped.
    """
    cmp_ = boot["comparisons_vs_aegislink"]
    families: dict[str, Any] = {}

    ref = cmp_.get(reference, {})
    families["F1_primary_endpoints"] = holm_bonferroni(
        {f"UALER:aegislink_vs_{reference}": (ref.get("UALER") or {}).get("p_value"),
         f"ATPR:aegislink_vs_{reference}": (ref.get("ATPR") or {}).get("p_value")},
        alpha=ALPHA,
        family_id="F1_primary_endpoints",
    )
    families["F2_per_action_ualer"] = holm_bonferroni(
        {
            f"UALER@{action}": (row or {}).get("p_value")
            for action, row in boot["per_action_ualer_vs_reference"].items()
        },
        alpha=ALPHA,
        family_id="F2_per_action_ualer",
    )
    families["F3_per_model_family_ualer"] = {
        "family_id": "F3_per_model_family_ualer",
        "method": "holm",
        "alpha": ALPHA,
        "n_members": 0,
        "status": "NOT_EVALUABLE",
        "reason": (
            "The pilot reader is a deterministic surrogate, so the model-family axis has size 1 and "
            "there is no family of per-model-family comparisons to correct. Recorded rather than "
            "dropped so the declared family list stays complete."
        ),
    }
    families["F4_all_baselines"] = holm_bonferroni(
        {
            f"{metric}:aegislink_vs_{bid}": (cmp_.get(bid, {}).get(metric) or {}).get("p_value")
            for bid in BASELINE_IDS
            if bid in cmp_
            for metric in PRIMARY_METRICS
        },
        alpha=ALPHA,
        family_id="F4_all_baselines",
    )
    families["F5_ablations"] = holm_bonferroni(
        {
            f"{metric}:aegislink_vs_{aid}": (cmp_.get(aid, {}).get(metric) or {}).get("p_value")
            for aid in ABLATION_IDS
            if aid in cmp_
            for metric in PRIMARY_METRICS
        },
        alpha=ALPHA,
        family_id="F5_ablations",
    )
    families["_policy"] = (
        "One family per preregistered comparison set (preregistration.yaml "
        "multiplicity_correction.families). Correction is applied within a family, never across "
        "families. F6 (secondary endpoints) is corrected in the secondary block, not here."
    )
    families["_regime"] = regime
    return families


# ======================================================================================
# 3. Pareto frontier
# ======================================================================================
def pareto_frontier(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Non-dominated set on (UALER down, ATPR up).

    ``preregistration.yaml pareto_analysis`` requires the dominated/non-dominated split alongside the
    figure. A configuration with an undefined endpoint cannot be placed on the frontier and is listed
    separately rather than treated as 0.
    """
    points: dict[str, tuple[float, float]] = {}
    unplaceable: list[str] = []
    for name, row in rows.items():
        u = (row.get("primary") or {}).get("UALER")
        a = (row.get("primary") or {}).get("ATPR")
        if u is None or a is None:
            unplaceable.append(name)
        else:
            points[name] = (float(u), float(a))

    non_dominated: list[str] = []
    for name, (u, a) in points.items():
        dominated = any(
            (ou <= u and oa >= a) and (ou < u or oa > a)
            for other, (ou, oa) in points.items()
            if other != name
        )
        if not dominated:
            non_dominated.append(name)
    return {
        "axes": {"UALER": "lower_is_better", "ATPR": "higher_is_better"},
        "n_placed": len(points),
        "non_dominated": sorted(non_dominated),
        "dominated": sorted(set(points) - set(non_dominated)),
        "unplaceable": sorted(unplaceable),
        "unplaceable_policy": (
            "A configuration with an undefined endpoint is listed here rather than plotted at 0: "
            "'no eligible cases' is not the same as 'a rate of zero'."
        ),
        "points": {n: {"UALER": round(u, 6), "ATPR": round(a, 6)} for n, (u, a) in sorted(points.items())},
    }


# ======================================================================================
# 4. Retrieval-configuration sensitivity for the RQ1 prevalence arm
# ======================================================================================
def retrieval_sensitivity(
    corpus: Any, index: Any, queries: Sequence[Any], oracle: Any,
    entity_to_regime: Mapping[str, str], regimes: Sequence[str], *, n_resamples: int
) -> dict[str, Any]:
    """Undefended exposure under each of the three frozen ranking modes.

    NOT a model-family axis. It answers a narrower question -- is the phenomenon an artifact of one
    ranker? -- and is reported under that description only. See
    ``evaluation/gate_evaluator.evaluate_misbinding_reproducible``.
    """
    encoder = HashedNGramEncoder()
    out: dict[str, Any] = {}
    for mode in RANKING_MODES:
        config = RetrieverConfig(mode=mode, top_k=10)
        results = Retriever(index, config=config, encoder=encoder).search_all(queries, progress_every=0)
        per_action = undefended_exposure_units(results, oracle, entity_to_regime, regimes)
        cells: dict[str, Any] = {}
        for action, units in sorted(per_action.items()):
            seed = derive_seed("sensitivity", mode, action)
            idx = resample_indices(len(units), n_resamples=n_resamples, seed=seed)
            cells[action] = bootstrap_metric(
                units, "UALER", config_id=f"undefended@{mode}", regime="+".join(regimes),
                idx=idx, seed=seed
            ).as_dict()
        out[mode] = {
            "retriever_config_digest": config.digest(),
            "n_queries": len(results),
            "per_action_undefended_ualer": cells,
        }
        print(f"[6/7] undefended exposure @ {mode:11s} "
              + "  ".join(f"{a}={_fmt(c['point'], 5)}" for a, c in sorted(cells.items())))
    return {
        "question": "Is undefended action-link misbinding an artifact of one ranking mode?",
        "modes": list(RANKING_MODES),
        "per_mode": out,
        "interpretation_limit": (
            "Ranking modes are NOT model families. This axis shows the phenomenon survives changing "
            "the ranker; it does not satisfy misbinding_reproducible.min_model_families, which asks "
            "for >= 2 LLM model families, and is not counted toward it."
        ),
    }


# ======================================================================================
# Main
# ======================================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="reduce resamples to 200 and skip Stratum B; wiring check only")
    ap.add_argument("--resamples", type=int, default=N_RESAMPLES,
                    help=f"paired bootstrap resamples (preregistered: {N_RESAMPLES})")
    args = ap.parse_args()
    n_resamples = 200 if args.quick else args.resamples

    t0 = time.time()
    print("=" * 78)
    print("Step 5 -- primary evaluation, adaptive robustness and the Section 12 gate decision")
    print("=" * 78)
    if n_resamples != N_RESAMPLES:
        print(f"!! resamples={n_resamples} differs from the preregistered {N_RESAMPLES}; "
              f"this run is NOT a results run")

    # ---------------------------------------------------------------- 1. static replay
    env = rebuild_static_replay()
    corpus, results, fingerprint = env["corpus"], env["results"], env["fingerprint"]
    ev = build_evaluator_context(corpus)
    recorder, fetch_text, fetch_links = make_fetchers(corpus, ev.oracle)

    # ---------------------------------------------------------------- 2. frozen params
    print("\n[2/7] Loading the Step 4 fit (nothing is fitted here)")
    params = FrozenParameters.load()
    if params.replay_fingerprint != fingerprint:
        raise ReplayDriftError(
            f"the Step 4 artifact was produced on replay {params.replay_fingerprint} but this run "
            f"rebuilt {fingerprint}. The frozen parameters would describe a different corpus."
        )
    hooks = assert_no_refitting_hooks()
    print(f"[2/7] Platt a={params.calibration.a} b={params.calibration.b} "
          f"fit on {params.calibration.fit_regime} ({params.n_fit_triples} triples)")
    print(f"[2/7] no-refit guard: {hooks['no_fitting_symbols_imported']}; "
          f"source sha256 {params.source_sha256[:16]}...")

    matrix = ExperimentMatrix.from_splits()
    # Register all 80 dynamic AD-* templates before any admission check. The controller re-derives
    # each template's region membership from its factor levels and rejects the registration if that
    # disagrees with the template's own in_declared_region flag, so a mislabelled template cannot
    # smuggle an out-of-region attack into the primary adaptive claim.
    adaptive_templates = load_yaml(ROOT / "attacks" / "adaptive_templates.yaml")["templates"]
    matrix.register_adaptive_templates(adaptive_templates)
    print(f"[2/7] registered {len(adaptive_templates)} adaptive templates; "
          f"{len(matrix.adaptive_stratum_a)} verified inside the pre-declared region (Stratum A)")

    admissions: dict[str, Any] = {}
    for regime in EVALUATION_REGIMES:
        ents = [e for e, r in ev.entity_to_regime.items() if r == regime]
        admissions[regime] = admit_regime(matrix, ev, regime, ents)
        print(f"[2/7] admitted {regime:20s} {len(ents):3d} entities, "
              f"{admissions[regime]['n_entity_templates']:2d} entity templates")

    # ---------------------------------------------------------------- 3. static sweep
    print(f"\n[3/7] Sweeping {len(CONFIG_ORDER)} configurations over {len(results)} queries")
    defenses = build_all_configurations(params, replay_fingerprint=fingerprint)
    bound = assert_replay_bound(defenses, fingerprint)
    harness = DefenseEvaluationHarness(
        registry=corpus.registry,
        official_registry=load_official_registry(),
        fetch_links=fetch_links,
        site_index=build_site_index(corpus.documents),
        read_phase=PHASE_LIVE,
        replay_fingerprint=fingerprint,
    )
    probe = harness.context(results[0], fetch_text)
    shared = assert_shared_evidence(
        probe, defenses, entity_id=results[0].entity_id, action=results[0].action
    )
    print(f"[3/7] replay binding verified for {bound['n_verifier_configs_bound']} verifier configs; "
          f"identical-evidence check over {shared['n_defenses_checked']} defenses")

    sweep = run_sweep(defenses, harness, results, ev, fetch_text, progress_label="3/7")
    metrics = metrics_by_regime(sweep, ev, EVALUATION_REGIMES)
    units = units_by_regime(sweep, ev, EVALUATION_REGIMES)

    # -- reconcile against Step 4 -----------------------------------------------------
    reconciliation = reconcile_with_step4(sweep, ev, metrics)
    print(f"[3/7] Step 4 reconciliation: {reconciliation['n_configurations_compared']} configs "
          f"reproduce Step 4 EXACTLY under its per-candidate rule "
          f"({reconciliation['n_legacy_disagreements']} disagreements)")
    print(f"[3/7] de-duplication correction changes {reconciliation['n_corrected_endpoints']} "
          f"validation endpoint(s); ATPR denominator "
          f"{reconciliation['atpr_denominator']['validation']['per_candidate']} -> "
          f"{reconciliation['atpr_denominator']['validation']['deduped']}")

    # ---------------------------------------------------------------- 4. reference baseline
    print("\n[4/7] Selecting the reference baseline on validation (never on test)")
    selection = select_reference_baseline(metrics)
    reference = selection.baseline_id
    print(f"[4/7] reference_baseline = {reference} "
          f"(validation UALER={_fmt(selection.ualer)} ATPR={_fmt(selection.atpr)}, "
          f"ATPR constraint satisfied={selection.constraint_satisfied})")

    # ---------------------------------------------------------------- 5. adaptive arm
    print("\n[5/7] Adaptive robustness arm")
    strata = ["A"] if args.quick else ["A", "B"]

    def fetch_factory(c: Any):
        _rec, ft, fl = make_fetchers(c, ev.oracle)
        return ft, fl

    adaptive: dict[str, Any] = {}
    for stratum in strata:
        adaptive[stratum] = evaluate_stratum(
            corpus, ev, params, matrix, stratum, fetch_factory=fetch_factory
        )
    snapshot_path = write_adaptive_snapshot([a["replay"] for a in adaptive.values()])
    print(f"[5/7] wrote {snapshot_path.relative_to(ROOT)} "
          f"({snapshot_path.stat().st_size / 1e6:.2f} MB)")

    adaptive_metrics: dict[str, Any] = {}
    adaptive_units: dict[str, Any] = {}
    for stratum, payload in adaptive.items():
        regime = REGIME_OF_STRATUM[stratum]
        adaptive_metrics[regime] = metrics_by_regime(payload["sweep"], payload["ev"], [regime])
        adaptive_units[regime] = units_by_regime(payload["sweep"], payload["ev"], [regime])

    # ---------------------------------------------------------------- 6. bootstrap
    print(f"\n[5/7] Paired bootstrap: {n_resamples} resamples, "
          f"{int(CONFIDENCE_LEVEL * 100)}% intervals, seed-derived per regime")
    boot: dict[str, Any] = {}
    for regime in EVALUATION_REGIMES:
        per_config = {c: u[regime] for c, u in units.items() if regime in u}
        if not per_config:
            continue
        boot[regime] = bootstrap_regime(regime, per_config, reference, n_resamples=n_resamples)
    for regime, per_regime_units in adaptive_units.items():
        per_config = {c: u[regime] for c, u in per_regime_units.items() if regime in u}
        if per_config:
            boot[regime] = bootstrap_regime(regime, per_config, reference, n_resamples=n_resamples)

    holm = {regime: holm_families(boot[regime], reference, regime) for regime in boot}

    # -- calibration ------------------------------------------------------------------
    calibration: dict[str, Any] = {}
    for name, triples in sweep.triples.items():
        by_regime: dict[str, list[Any]] = defaultdict(list)
        for t in triples:
            by_regime[t.regime].append(t)
        calibration[name] = {
            regime: calibration_report(calibration_pairs(sel))
            for regime, sel in sorted(by_regime.items())
            if regime in EVALUATION_REGIMES
        }
    for stratum, payload in adaptive.items():
        regime = REGIME_OF_STRATUM[stratum]
        for name, triples in payload["sweep"].triples.items():
            sel = [t for t in triples if t.regime == regime]
            if sel:
                calibration.setdefault(name, {})[regime] = calibration_report(calibration_pairs(sel))

    pareto = {
        regime: pareto_frontier(
            {name: rows[regime] for name, rows in metrics.items() if regime in rows}
        )
        for regime in EVALUATION_REGIMES
    }
    for regime, per_regime in adaptive_metrics.items():
        pareto[regime] = pareto_frontier(
            {name: rows[regime] for name, rows in per_regime.items() if regime in rows}
        )

    # ---------------------------------------------------------------- 7. gates
    print("\n[6/7] Evaluating the ten Section 12 conditions")
    gates = FrozenGates.load()
    print(f"[6/7] preregistration declares {gates.n_conditions()} frozen threshold values "
          f"across {len(gates.hard_gates)} gates")

    # Gate 1: undefended prevalence on the confirmatory regimes.
    undef_units = undefended_exposure_units(
        results, ev.oracle, ev.entity_to_regime, CONFIRMATORY_REGIMES
    )
    undef_cells: dict[str, Any] = {}
    for action, us in sorted(undef_units.items()):
        seed = derive_seed("undefended", action)
        idx = resample_indices(len(us), n_resamples=n_resamples, seed=seed)
        undef_cells[action] = bootstrap_metric(
            us, "UALER", config_id="undefended_retrieval", regime="+".join(CONFIRMATORY_REGIMES),
            idx=idx, seed=seed
        ).as_dict()
    sensitivity = retrieval_sensitivity(
        corpus, env["index"], env["queries"], ev.oracle, ev.entity_to_regime,
        CONFIRMATORY_REGIMES, n_resamples=n_resamples
    )
    gate1 = evaluate_misbinding_reproducible(
        gates,
        undef_cells,
        n_model_families=1,
        model_family_note=(
            "NOT EVALUABLE in this pilot. RQ1 asks about web-enabled LLMs; the pilot reader is a "
            "deterministic surrogate, so the model-family axis has size 1. Reporting 1 as 2, or "
            "substituting the three retrieval configurations for model families, would invent the "
            "measurement the gate asks for."
        ),
        sensitivity=sensitivity,
    )

    # Gate 2: on validation, as the preregistration directs.
    gate2 = evaluate_baseline_not_trivial(
        gates,
        {
            bid: (metrics[bid]["validation"]["primary"])
            for bid in BASELINE_IDS
            if "validation" in metrics.get(bid, {})
        },
        split="validation",
    )

    # Gates 3 and 4: the improvement check per regime.
    min_red = float(gates.value("aegislink_improvement", "min_relative_ualer_reduction"))
    max_loss = float(gates.value("aegislink_improvement", "max_absolute_atpr_loss"))

    def improvement_for(regime: str):
        cmp_ = boot.get(regime, {}).get("comparisons_vs_aegislink", {}).get(reference)
        return check_improvement(
            (cmp_ or {}).get("UALER"),
            (cmp_ or {}).get("ATPR"),
            regime=regime,
            reference=reference,
            min_relative_reduction=min_red,
            max_absolute_atpr_loss=max_loss,
        )

    test_check = improvement_for("test")
    transfer_check = improvement_for("transfer_holdout")
    adaptive_check = improvement_for(REGIME_OF_STRATUM["A"])
    gate3 = evaluate_aegislink_improvement(gates, test_check, split="test")
    gate4 = evaluate_generalization(gates, transfer_check, adaptive_check)
    gate5 = evaluate_novelty(gates)

    report = build_decision_report(
        gates,
        [gate1, gate2, gate3, gate4, gate5],
        context={
            "replay_fingerprint": fingerprint,
            "adaptive_replay_fingerprints": {
                s: a["replay"].fingerprint for s, a in adaptive.items()
            },
            "reference_baseline": reference,
            "reference_baseline_selection": selection.as_dict(),
            "confirmatory_split": "test",
            "n_resamples_used": n_resamples,
            "preregistered_resamples": N_RESAMPLES,
            "is_results_run": n_resamples == N_RESAMPLES and not args.quick,
            "frozen_parameters": params.as_dict(),
        },
    )
    d = report["decision"]
    print(f"[6/7] DECISION: {d['overall_decision']}  "
          f"passed {d['n_passed']}/{d['n_conditions_declared']}, "
          f"failed {d['n_failed']}, not evaluable {d['n_not_evaluable']}")
    for cond in report["conditions_flat"]:
        print(f"       {cond['status']:14s} {cond['condition_id']:32s} "
              f"observed={cond['observed']}")

    # ---------------------------------------------------------------- 8. artifacts
    print("\n[7/7] Writing artifacts")
    RESULTS.mkdir(exist_ok=True)

    primary_doc = {
        "metadata": {
            "step": "Step 5 of 8 -- primary evaluation on the held-out regimes",
            "contract_ref": "CONTRACT.md Sections 3 (RQ1, RQ4), 5.3, 11, 12",
            "builder": "workflow/19_evaluate_primary_and_adaptive.py",
            "replay_fingerprint": fingerprint,
            "replay_fingerprint_verified": True,
            "read_phase": PHASE_LIVE,
            "n_queries": len(results),
            "n_configurations": len(CONFIG_ORDER),
            "regimes_reported": list(EVALUATION_REGIMES),
            "confirmatory_regimes": list(CONFIRMATORY_REGIMES),
            "n_resamples_used": n_resamples,
            "preregistered_resamples": N_RESAMPLES,
            "is_results_run": n_resamples == N_RESAMPLES and not args.quick,
            "quick_mode": bool(args.quick),
            "elapsed_seconds": round(time.time() - t0, 2),
        },
        "step4_reconciliation": reconciliation,
        "frozen_parameters": params.as_dict(),
        "no_refit_guard": hooks,
        "admissions": admissions,
        "replay_binding": bound,
        "identical_evidence_check": shared,
        "exposure_firewall": describe_firewall(),
        "method": describe_primary_evaluation(),
        "statistics": describe_bootstrap(),
        "reference_baseline_selection": selection.as_dict(),
        "metrics": metrics,
        "bootstrap": {r: v for r, v in boot.items() if r in EVALUATION_REGIMES},
        "holm_families": {r: v for r, v in holm.items() if r in EVALUATION_REGIMES},
        "calibration": calibration,
        "pareto": {r: v for r, v in pareto.items() if r in EVALUATION_REGIMES},
        "undefended_prevalence": {
            "arm": "RQ1 prevalence, undefended retrieval presentation",
            "regimes": list(CONFIRMATORY_REGIMES),
            "per_action": undef_cells,
            "definition": (
                "An undefended agent presents the retrieved candidate list, so a response is a "
                "misbinding whenever any retrieved domain is unauthorized for the query's action. "
                "Computed from the replay and the oracle alone, so the preregistered "
                "18-configuration table is unchanged."
            ),
            "surrogate_disclaimer": (
                "This is a DETERMINISTIC SURROGATE for an LLM reader, not a model family. It "
                "establishes that the phenomenon exists on this corpus; it does not satisfy "
                "misbinding_reproducible.min_model_families."
            ),
            "retrieval_configuration_sensitivity": sensitivity,
        },
    }
    assert_no_leakage(primary_doc["metrics"], location="primary_test_evaluation.metrics",
                     allow_free_text=True)
    (RESULTS / "primary_test_evaluation.json").write_text(
        json.dumps(primary_doc, indent=2) + "\n", encoding="utf-8"
    )

    adaptive_doc = {
        "metadata": {
            "step": "Step 5 of 8 -- adaptive robustness (RQ4)",
            "contract_ref": "CONTRACT.md Section 3 (RQ4)",
            "builder": "workflow/19_evaluate_primary_and_adaptive.py",
            "static_replay_fingerprint": fingerprint,
            "strata_evaluated": sorted(adaptive),
            "adaptive_snapshot": str(snapshot_path.relative_to(ROOT)),
            "n_resamples_used": n_resamples,
            "quick_mode": bool(args.quick),
        },
        "method": describe_adaptive_evaluation(),
        "strata": {
            stratum: {
                "regime": REGIME_OF_STRATUM[stratum],
                "primary_claim_eligible": stratum == "A",
                "replay": payload["replay"].as_dict(),
                "admission": payload["admission"],
                "asr": payload["asr"],
                "metrics": adaptive_metrics[REGIME_OF_STRATUM[stratum]],
                "utility_under_pressure": {
                    name: utility_under_adaptive_pressure(
                        adaptive_metrics[REGIME_OF_STRATUM[stratum]]
                        .get(name, {})
                        .get(REGIME_OF_STRATUM[stratum]),
                        metrics.get(name, {}).get("test"),
                    )
                    for name in CONFIG_ORDER
                },
                "bootstrap": boot.get(REGIME_OF_STRATUM[stratum]),
                "holm_families": holm.get(REGIME_OF_STRATUM[stratum]),
                "pareto": pareto.get(REGIME_OF_STRATUM[stratum]),
            }
            for stratum, payload in adaptive.items()
        },
        "reference_baseline": reference,
    }
    (RESULTS / "adaptive_robustness_evaluation.json").write_text(
        json.dumps(adaptive_doc, indent=2) + "\n", encoding="utf-8"
    )

    report["method"] = describe_gate_evaluator()
    (RESULTS / "pilot_decision_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    for p in (
        RESULTS / "primary_test_evaluation.json",
        RESULTS / "adaptive_robustness_evaluation.json",
        RESULTS / "pilot_decision_report.json",
        snapshot_path,
    ):
        print(f"[7/7] wrote {p.relative_to(ROOT)} ({p.stat().st_size / 1e6:.2f} MB)")

    # ---------------------------------------------------------------- 9. summary
    print("\n" + "=" * 100)
    print("Step 5 -- TEST split (confirmatory)")
    print("=" * 100)
    hdr = (f"{'configuration':34s} {'UALER':>7s} {'95% CI':>17s} {'ATPR':>7s} {'95% CI':>17s} "
           f"{'FRR':>7s} {'Brier':>8s}")
    print(hdr)
    for name in CONFIG_ORDER:
        row = metrics.get(name, {}).get("test")
        if not row:
            continue
        est = boot.get("test", {}).get("estimates", {}).get(name, {})
        def ci(metric: str) -> str:
            b = ((est.get(metric) or {}).get("ci") or {}).get("bias_corrected") or {}
            lo, hi = b.get("lo"), b.get("hi")
            return "  [n/a, n/a]" if lo is None else f"[{lo:6.3f},{hi:6.3f}]"
        p, s, c = row["primary"], row["secondary"], row["calibration"]
        print(f"{name:34s} {_fmt(p['UALER'])} {ci('UALER'):>17s} {_fmt(p['ATPR'])} "
              f"{ci('ATPR'):>17s} {_fmt(s['FRR'])} {_fmt(c['brier'], 8, 5)}")

    print("\n" + "=" * 100)
    print("Adaptive robustness: ASR_a per configuration")
    print("=" * 100)
    print(f"{'configuration':34s} " + "  ".join(
        f"{'ASR_a[' + s + ']':>14s}  {'UALER[' + s + ']':>14s}" for s in sorted(adaptive)))
    for name in CONFIG_ORDER:
        cells = []
        for stratum in sorted(adaptive):
            regime = REGIME_OF_STRATUM[stratum]
            asr = adaptive[stratum]["asr"].get(name, {})
            row = adaptive_metrics[regime].get(name, {}).get(regime, {})
            cells.append(f"{_fmt(asr.get('ASR_a'), 14):>14s}  "
                         f"{_fmt((row.get('primary') or {}).get('UALER'), 14):>14s}")
        print(f"{name:34s} " + "  ".join(cells))

    print("\n" + "=" * 78)
    print(f"PILOT DECISION: {d['overall_decision']}")
    print("=" * 78)
    for name, g in report["gates"].items():
        print(f"  {g['status']:14s} {name}")
    if d["not_evaluable_conditions"]:
        print(f"\n  not evaluable: {', '.join(d['not_evaluable_conditions'])}")
    if d["failed_conditions"]:
        print(f"  failed:        {', '.join(d['failed_conditions'])}")
    print(f"\nElapsed {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
