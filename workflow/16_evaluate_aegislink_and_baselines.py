"""Step 4 driver: run AegisLink, its seven ablations and the ten baselines on the frozen replay.

CONTRACT.md Sections 3 (RQ2, RQ3), 9, 10, 11.

Order of operations, and why it is this order
---------------------------------------------
1. Rebuild the corpus, index and retrieval, then **verify the replay fingerprint** against the
   frozen snapshot. Nothing else runs until that matches: if the evidence has drifted, every
   between-defense comparison below is confounded, so the mismatch is fatal rather than warned.
2. Fit AegisLink's thresholds and probability calibration on ``train_development`` **only**, under
   the admission controller. ``test``, ``transfer_holdout`` and ``adaptive_holdout`` are never read.
3. Run all 18 configurations over the frozen replay through the RQ2 trace recorder.
4. Compute the preregistered endpoints on ``train_development`` and ``validation``.
5. Audit the evidence families for identifiability -- which of them are separable on this corpus at
   all -- and record it as a limitation rather than discovering it later.

Outputs
-------
``results/aegislink_defense_evaluation.json``   endpoints, per action, for all 18 configurations
``results/baseline_deviation_log.json``         Section 10 deviation log
``results/evidence_family_identifiability.json`` degeneracy audit + the shortcuts AegisLink declines
``results/defense_stage_traces_summary.json``   RQ2 failure-origin distribution per defense
``data/benchmark/defense_traces.json``          full five-stage traces per defense

Run::

    uv run python workflow/16_evaluate_aegislink_and_baselines.py [--quick]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from aegislink.ablations import (  # noqa: E402
    ABLATION_IDS,
    ABLATION_PREDICTIONS,
    ablation_metadata,
    assert_single_factor_ablations,
    build_ablations,
    describe_ablations,
)
from aegislink.framework import (  # noqa: E402
    ACTION_RISK_ORDER,
    RiskThresholds,
    Verdict,
    describe_framework,
    fit_platt,
)
from aegislink.pipeline_adapter import (  # noqa: E402
    DefenseEvaluationHarness,
    assert_replay_bound,
    assert_shared_evidence,
    describe_adapter,
)
from aegislink.verifier import AegisLink, AegisLinkConfig  # noqa: E402
from baselines.registry import (  # noqa: E402
    BASELINE_IDS,
    assert_suite_complete,
    baseline_metadata,
    build_baselines,
    describe_baselines,
    deviation_log,
)
from evaluation.defense_metrics import (  # noqa: E402
    TripleOutcome,
    compute_metrics,
    describe_metrics,
)
from web_rag.crawler_indexer import (  # noqa: E402
    PHASE_INDEX,
    PHASE_LIVE,
    build_index,
    crawl_corpus,
    outbound_links,
)
from web_rag.experiment_matrix import (  # noqa: E402
    ExperimentMatrix,
    RunPurpose,
    RunSpec,
)
from web_rag.exposure import (  # noqa: E402
    ExposureTier,
    assert_no_leakage,
    build_site_index,
    describe_firewall,
    load_official_registry,
)
from web_rag.retriever import (  # noqa: E402
    HashedNGramEncoder,
    Retriever,
    RetrieverConfig,
    build_entity_queries,
)
from web_rag.trace_recorder import (  # noqa: E402
    GroundTruthOracle,
    TraceRecorder,
    summarize_traces,
    validate_trace_schema,
)

BENCH = ROOT / "data" / "benchmark"
RESULTS = ROOT / "results"
SNAPSHOT_PATH = BENCH / "retrieval_snapshots.json"

#: Regimes Step 4 is allowed to read. The evaluation splits are Step 5's.
READABLE_REGIMES: tuple[str, ...] = ("train_development", "validation")
FIT_REGIME = "train_development"


#: Coarse direction each ablation was predicted to move, derived from the pre-stated predictions in
#: ``aegislink/ablations.py``. Kept as a separate literal so the prose predictions stay immutable and
#: the machine-checkable claim is explicit.
PREDICTED_DIRECTION: Mapping[str, str] = {
    "ablation_no_action_type": "over_grant",
    "ablation_no_official_backlinks": "under_grant",
    "ablation_no_source_clustering": "negligible",
    "ablation_no_domain_lifecycle": "negligible",
    "ablation_no_contradiction_edges": "negligible",
    "ablation_source_count_voting": "over_grant",
    "ablation_shared_threshold": "negligible",
}

#: Mechanisms observed during the run that were not anticipated. Recorded here so the results
#: artifact carries the explanation rather than leaving a reader to wonder at a failed prediction.
OBSERVED_MECHANISMS: Mapping[str, str] = {
    "ablation_source_count_voting": (
        "PREDICTION FAILED, and the reason is informative. Voting was predicted to over-grant "
        "because Sybil corroborators would outvote the official page. Measured, it UNDER-grants: "
        "the forged 'Listed by the official site' backlink points at the entity's GENUINE official "
        "host, so in a link-counting vote every Sybil page casts its vote FOR the official domain. "
        "The forgery that defeats a provenance checker therefore helps a vote counter. Voting's "
        "failure on this corpus is lost third-party utility (ATPR collapses), not admitted attacks."
    ),
    "ablation_no_contradiction_edges": (
        "Negligible on UALER as predicted, but for a reason worth stating: the authority gate "
        "already withholds impersonators, so contradiction edges add the ability to say "
        "CONTRADICTED (a positive detection) rather than UNVERIFIED (an evidence gap). Their value "
        "shows up in verdict semantics, CMR and OSMR, not in blocking."
    ),
}


def _direction(delta_ualer: float | None, delta_atpr: float | None) -> str | None:
    """Classify an ablation's realised effect. ``0.02`` is the negligible band on both endpoints."""
    if delta_ualer is None or delta_atpr is None:
        return None
    if delta_ualer > 0.02:
        return "over_grant"
    if delta_atpr < -0.02:
        return "under_grant"
    return "negligible"


class ReplayDriftError(RuntimeError):
    """Raised when the rebuilt replay does not match the frozen fingerprint. Fails closed."""


def load_yaml(path: Path) -> Any:
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    return yaml.load(path.read_text(encoding="utf-8"), Loader=loader)


# ======================================================================================
# 1. Environment + fingerprint verification
# ======================================================================================
def rebuild_replay() -> dict[str, Any]:
    """Rebuild corpus, index and retrieval, and verify the frozen replay fingerprint."""
    print("[1/6] Rebuilding the frozen replay environment")
    corpus = crawl_corpus(verify_hashes=True)
    index = build_index(corpus, phase=PHASE_INDEX)
    entities = load_yaml(ROOT / "registry" / "entities.yaml")["entities"]
    queries = build_entity_queries(entities)
    query_digest = hashlib.sha256(
        json.dumps([q.as_dict() for q in queries], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    config = RetrieverConfig(mode="hybrid_rrf", top_k=10)
    encoder = HashedNGramEncoder()
    retriever = Retriever(index, config=config, encoder=encoder)
    results = retriever.search_all(queries, progress_every=400)

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
    expected = frozen["replay_fingerprint"]
    if fingerprint != expected:
        drift = {
            k: (v, frozen["fingerprint_material"].get(k))
            for k, v in material.items()
            if k != "per_query_sha256" and v != frozen["fingerprint_material"].get(k)
        }
        raise ReplayDriftError(
            f"rebuilt replay fingerprint {fingerprint} != frozen {expected}.\n"
            f"Drifted components (rebuilt, frozen): {json.dumps(drift, indent=2)}\n"
            f"CONTRACT.md Section 6 freezes retrieval so the defenses are compared on identical "
            f"evidence; a drifted replay would confound every comparison below."
        )
    print(f"[1/6] replay fingerprint VERIFIED: {fingerprint[:16]}...{fingerprint[-4:]}")
    print(f"[1/6] {len(results)} queries, {sum(len(r.candidates) for r in results)} candidates")

    return {
        "corpus": corpus,
        "index": index,
        "entities": entities,
        "results": results,
        "retriever": retriever,
        "fingerprint": fingerprint,
    }


# ======================================================================================
# 2. Ground truth + split bookkeeping (EVALUATOR tier)
# ======================================================================================
def build_evaluator_side(corpus: Any) -> dict[str, Any]:
    """Assemble the oracle and the per-entity regime map. Never handed to a defense."""
    domains_raw = load_yaml(ROOT / "registry" / "domains.yaml")
    graph_edges = load_yaml(ROOT / "registry" / "authorization_graph.yaml")["edges"]
    aliases = load_yaml(ROOT / "registry" / "aliases.yaml")["aliases"]
    entities = load_yaml(ROOT / "registry" / "entities.yaml")["entities"]
    splits = load_yaml(ROOT / "configs" / "splits.yaml")

    oracle = GroundTruthOracle.from_registry(
        graph_edges=graph_edges,
        delegations=domains_raw["delegations"],
        domain_hosts={d.domain_id: d.domain for d in corpus.registry.domains.values()},
        entities=entities,
        aliases=aliases,
    )
    # Delegated third parties, from the registry's delegation table. EVALUATOR tier: used only to
    # decide which authorized links count toward ATPR, never exposed to a defense.
    third_party: set[tuple[str, str]] = {
        (str(d["entity_id"]), str(d["domain_id"])) for d in domains_raw["delegations"]
    }
    official_of: dict[str, str] = {
        str(rec["owner_entity_id"]): str(did)
        for did, rec in domains_raw["domains"].items()
        if rec.get("owner_entity_id")
    }
    return {
        "oracle": oracle,
        "third_party_pairs": third_party,
        "official_of": official_of,
        "entity_status": {str(e["entity_id"]): str(e["status"]) for e in entities},
        "entity_to_regime": {str(k): str(v) for k, v in splits["entity_to_regime"].items()},
        "graph_edges": graph_edges,
    }


def unconditioned_denominators(
    ev: Mapping[str, Any], regime: str
) -> tuple[int, int]:
    """Every authorized (third-party, any) link in ``regime``, retrieved or not.

    Used only for the ``*_incl_unretrieved`` columns, so the conditioning choice in
    ``evaluation/defense_metrics.py`` is inspectable rather than asserted.
    """
    e2r = ev["entity_to_regime"]
    tp = ev["third_party_pairs"]
    n_tp = n_all = 0
    for edge in ev["graph_edges"]:
        if not edge["authorized"]:
            continue
        eid, did = str(edge["entity_id"]), str(edge["domain_id"])
        if e2r.get(eid) != regime:
            continue
        n_all += 1
        if (eid, did) in tp:
            n_tp += 1
    return n_tp, n_all


# ======================================================================================
# 3. Threshold + calibration fitting (train_development ONLY)
# ======================================================================================
def fit_on_development(
    harness: DefenseEvaluationHarness,
    results: Sequence[Any],
    ev: Mapping[str, Any],
    fetch,
    matrix: ExperimentMatrix,
) -> dict[str, Any]:
    """Fit AegisLink's calibration on ``train_development``, under admission control.

    The threshold *vector* is the frozen design default: the preregistration fixes its shape, and
    tuning five thresholds on the corpus would be fitting the decision rule to the benchmark. What
    is fitted is the two-parameter Platt transform, which is monotone and therefore cannot change
    any ranking -- only the probability scale that Section 11's Brier and calibration table reads.
    """
    print(f"\n[2/6] Fitting probability calibration on {FIT_REGIME} only")
    run = RunSpec(
        run_id="step4_threshold_selection",
        regime=FIT_REGIME,
        purpose=RunPurpose.THRESHOLD_SELECTION,
        notes="Step 4 calibration fit for AegisLink; no test/holdout template is read",
    )
    report = matrix.admit(run).raise_if_rejected()
    print(f"[2/6] admission controller: {run.run_id} admitted "
          f"(rules {', '.join(report.checked_rules)})")

    e2r = ev["entity_to_regime"]
    oracle = ev["oracle"]
    dev = [r for r in results if e2r.get(r.entity_id) == FIT_REGIME]
    print(f"[2/6] {len(dev)} development queries")

    probe = AegisLink(config=AegisLinkConfig())
    pairs: list[tuple[float, bool]] = []
    for i, r in enumerate(dev, start=1):
        ctx = harness.context(r, fetch)
        for c in r.candidates:
            res = probe.verify(r.entity_id, c.domain_id, r.action, ctx)
            pairs.append(
                (res.probability, oracle.is_authorized(r.entity_id, c.domain_id, r.action))
            )
        if i % 100 == 0:
            print(f"[2/6] ... scored {i}/{len(dev)} development queries")

    platt = fit_platt(pairs, regime=FIT_REGIME)
    print(f"[2/6] Platt fit a={platt.a} b={platt.b} on {platt.n_fit} triples; "
          f"Brier {platt.brier_before:.5f} -> {platt.brier_after:.5f}")

    thresholds = RiskThresholds.design_default()
    return {
        "admission_report": report.as_dict(),
        "calibration": platt,
        "thresholds": thresholds,
        "n_fit_queries": len(dev),
        "n_fit_triples": len(pairs),
        "policy": {
            "threshold_vector": "frozen design default; shape fixed by preregistration, not fitted",
            "why_not_fitted": (
                "Tuning five thresholds on the corpus would fit the decision rule to the "
                "benchmark. Only the two-parameter Platt transform is fitted, and being monotone "
                "it cannot change any ranking -- only the probability scale Section 11 reads."
            ),
            "regimes_read": [FIT_REGIME],
            "regimes_refused": ["test", "transfer_holdout", "adaptive_holdout"],
        },
    }


# ======================================================================================
# 4. The sweep
# ======================================================================================
def build_all_defenses(
    thresholds: RiskThresholds, calibration: Any, fingerprint: str
) -> dict[str, Any]:
    """One full method, seven ablations, ten baselines -- all sharing the fitted calibration."""
    base = AegisLinkConfig(thresholds=thresholds, calibration=calibration)
    defenses: dict[str, Any] = {
        "aegislink_full": AegisLink(config=base, expected_replay_fingerprint=fingerprint)
    }
    defenses.update(build_ablations(base, expected_replay_fingerprint=fingerprint))
    # Baselines share the same Platt transform so the calibration column is comparable; their
    # underlying signals are untouched.
    for bid, d in build_baselines().items():
        defenses[bid] = d.with_calibration(calibration)
    return defenses


def run_sweep(
    defenses: Mapping[str, Any],
    harness: DefenseEvaluationHarness,
    results: Sequence[Any],
    ev: Mapping[str, Any],
    recorder: TraceRecorder,
) -> dict[str, Any]:
    """Run every defense over the frozen replay, recording traces and triple outcomes."""
    print(f"\n[3/6] Sweeping {len(defenses)} configurations over {len(results)} queries")
    oracle = ev["oracle"]
    e2r = ev["entity_to_regime"]
    status = ev["entity_status"]
    third_party = ev["third_party_pairs"]
    official_of = ev["official_of"]

    per_defense: dict[str, Any] = {}
    t_start = time.time()
    for n, (name, defense) in enumerate(defenses.items(), start=1):
        t0 = time.time()
        pipeline = harness.pipeline(defense)
        traces = recorder.record_all(results, pipeline, progress_every=0)
        for t in traces:
            validate_trace_schema(t.as_dict())

        triples: list[TripleOutcome] = []
        for r in results:
            decisions = pipeline.decisions_for(r, recorder._fetch_text)
            for c in r.candidates:
                dec = decisions.get(c.domain_id)
                if dec is None:
                    continue
                triples.append(
                    TripleOutcome(
                        query_id=r.query_id,
                        entity_id=r.entity_id,
                        action=r.action,
                        domain_id=c.domain_id,
                        rank=c.rank,
                        decision=dec,
                        authorized=oracle.is_authorized(r.entity_id, c.domain_id, r.action),
                        is_official_domain=official_of.get(r.entity_id) == c.domain_id,
                        is_authorized_third_party=(r.entity_id, c.domain_id) in third_party,
                        entity_status=status.get(r.entity_id, "unknown"),
                        regime=e2r.get(r.entity_id, "unassigned"),
                    )
                )

        per_defense[name] = {
            "traces": traces,
            "trace_summary": summarize_traces(traces),
            "triples": triples,
            "pipeline": pipeline.describe(),
        }
        dt = time.time() - t0
        print(f"[3/6] {n:2d}/{len(defenses)} {name:34s} {len(traces)} traces, "
              f"{len(triples)} triples, {dt:.1f}s")
    print(f"[3/6] sweep complete in {time.time() - t_start:.1f}s; "
          f"caches: {harness.cache_stats()}")
    return per_defense


# ======================================================================================
# 5. Endpoints
# ======================================================================================
def compute_all_metrics(
    per_defense: Mapping[str, Any], ev: Mapping[str, Any]
) -> dict[str, Any]:
    print("\n[4/6] Computing preregistered endpoints "
          f"({', '.join(READABLE_REGIMES)}; test never read)")
    out: dict[str, Any] = {}
    for name, payload in per_defense.items():
        triples = payload["triples"]
        by_regime: dict[str, list[TripleOutcome]] = defaultdict(list)
        for t in triples:
            by_regime[t.regime].append(t)
        rows: dict[str, Any] = {}
        for regime in READABLE_REGIMES:
            sel = by_regime.get(regime, [])
            if not sel:
                continue
            n_tp, n_all = unconditioned_denominators(ev, regime)
            rows[regime] = compute_metrics(
                name,
                regime,
                sel,
                mean_decision_ms=payload["pipeline"].get("mean_decision_ms"),
                authorized_third_party_total=n_tp,
                authorized_total=n_all,
            ).as_dict()
        out[name] = rows
        # Report which regime the numbers came from. An earlier version labelled the line
        # "validation" while silently falling back to train_development when validation was empty,
        # which is exactly the kind of mislabelled table this project cannot afford.
        shown = "validation" if "validation" in rows else next(iter(rows), None)
        v = rows.get(shown, {}) if shown else {}
        p = v.get("primary", {})
        print(f"[4/6] {name:34s} [{shown or 'no data':17s}] UALER={_fmt(p.get('UALER'))} "
              f"ATPR={_fmt(p.get('ATPR'))} "
              f"FRR={_fmt(v.get('secondary', {}).get('FRR'))} "
              f"Brier={_fmt(v.get('calibration', {}).get('brier'))}")
    return out


def _fmt(x: Any) -> str:
    return "  n/a" if x is None else f"{float(x):.3f}"


# ======================================================================================
# 6. Evidence-family identifiability audit
# ======================================================================================
def audit_evidence_families(ev: Mapping[str, Any]) -> dict[str, Any]:
    """Measure whether each evidence family is separable on this corpus at all.

    This exists because two families are degenerate here, and finding that out from an ablation
    table would be finding it out too late. Both degeneracies are properties of the corpus, not of
    the method, and both are shortcuts AegisLink deliberately declines -- see
    ``aegislink/verifier.py`` "Two deliberate abstentions".
    """
    print("\n[5/6] Auditing evidence-family identifiability")
    domains_raw = load_yaml(ROOT / "registry" / "domains.yaml")
    domains = domains_raw["domains"]

    auth_any: dict[tuple[str, str], bool] = defaultdict(bool)
    for e in ev["graph_edges"]:
        k = (str(e["entity_id"]), str(e["domain_id"]))
        auth_any[k] = auth_any[k] or bool(e["authorized"])

    lifecycle_domains = {d for d, rec in domains.items() if isinstance(rec.get("lifecycle"), dict)}
    changed = {
        d
        for d, rec in domains.items()
        if isinstance(rec.get("lifecycle"), dict)
        and rec["lifecycle"].get("ownership_change_at_snapshot")
    }

    def contingency(flagged: set[str]) -> dict[str, int]:
        c = Counter()
        for (_eid, did), ok in auth_any.items():
            c[(did in flagged, ok)] += 1
        return {
            "flagged_and_authorized": c[(True, True)],
            "flagged_and_unauthorized": c[(True, False)],
            "unflagged_and_authorized": c[(False, True)],
            "unflagged_and_unauthorized": c[(False, False)],
        }

    lc = contingency(lifecycle_domains)
    oc = contingency(changed)

    return {
        "question": (
            "Is each RQ3 evidence family separable on this corpus, or is it a disguised label?"
        ),
        "method": (
            "For each family, cross the raw signal against authorized-for-some-action over all "
            "entity-domain pairs. A signal with an empty (flagged, authorized) cell is a perfect "
            "predictor, i.e. degenerate."
        ),
        "families": {
            "domain_lifecycle_presence": {
                "signal": "a lifecycle record exists for the domain",
                "contingency": lc,
                "degenerate": lc["flagged_and_authorized"] == 0,
                "interpretation": (
                    f"{len(lifecycle_domains)} of {len(domains)} domains carry a lifecycle record, "
                    f"and {lc['flagged_and_authorized']} of them are authorized for any action. "
                    f"Presence alone therefore predicts 'unauthorized' perfectly."
                ),
                "aegislink_response": (
                    "Declined. The lifecycle family may only WITHDRAW authority that the registry "
                    "or a published delegation granted, and takeover domains hold neither, so the "
                    "family contributes ~nothing here. ablation_no_domain_lifecycle is therefore "
                    "expected to be ~0, which is a corpus property and not evidence that lifecycle "
                    "reasoning is worthless."
                ),
                "corpus_remediation": (
                    "Add benign ownership-change controls: domains that change registrant and "
                    "RETAIN authorization via a re-published delegation. Then ownership change "
                    "alone stops being sufficient and the family becomes identifiable."
                ),
            },
            "ownership_change": {
                "signal": "lifecycle records an ownership change",
                "contingency": oc,
                "degenerate": oc["flagged_and_authorized"] == 0,
                "interpretation": (
                    f"{oc['flagged_and_authorized']} benign domains changed hands while retaining "
                    f"authorization, so this signal has no false-positive region in this corpus."
                ),
            },
            "prompt_injection_markers": {
                "signal": "page carries instruction-shaped content aimed at an automated reader",
                "degenerate": True,
                "interpretation": (
                    "prompt_injection is a Section 7 attack FACTOR held at roughly half the "
                    "adversarial templates, so its presence classifies the factor level rather "
                    "than the authorization."
                ),
                "aegislink_response": (
                    "Declined as a scoring feature. Counted in the evidence graph's notes for the "
                    "audit trail only. AegisLink is injection-immune by construction because it "
                    "follows no instruction found in retrieved text."
                ),
            },
            "registry_delegation_table": {
                "signal": "domains.yaml delegations (which third party holds which grant)",
                "degenerate": True,
                "interpretation": "It IS the RQ3 answer key.",
                "aegislink_response": (
                    "Never imported by any defense, ablation or baseline. Enforced statically by "
                    "tests/test_aegislink_and_baselines.py::"
                    "test_no_defense_imports_the_delegation_table."
                ),
            },
            "official_registry": {
                "signal": "entity -> official domain, via domains.yaml owner_entity_id",
                "degenerate": False,
                "interpretation": (
                    "Required by CONTRACT.md Section 3 (RQ3) evidence family 1 and Section 10 "
                    "baseline 9. Informationally equivalent to role == official_site and nothing "
                    "more: it names the official domain and leaves all 1,630 non-official domains "
                    "indistinguishable, exposing no per-action third-party grant. The "
                    "authorized-vs-forged third-party discrimination under study is untouched."
                ),
            },
            "official_backlink_direction": {
                "signal": "the official page publishes a delegation naming this domain",
                "degenerate": False,
                "interpretation": (
                    "The load-bearing family. Genuine partners and forgers emit byte-identical "
                    "inbound claims, with the forged link pointing at the real official host, so "
                    "only direction separates them. Extraction verified at 176/176 official pages "
                    "across all ten site templates by "
                    "workflow/diagnostics/check_extraction_assumptions.py."
                ),
            },
            "domain_age": {
                "signal": "how many snapshots the host has been observed for",
                "contingency": {
                    "domains_with_observed_first_seen": len(lifecycle_domains),
                    "domains_without": len(domains) - len(lifecycle_domains),
                },
                "degenerate": True,
                "interpretation": (
                    f"Only {len(lifecycle_domains)} of {len(domains)} hosts carry a first-seen "
                    f"snapshot, so age is unobservable for the other "
                    f"{len(domains) - len(lifecycle_domains)}. B02 therefore reduces in practice to "
                    f"'is it a takeover, and is it a shared provider'."
                ),
                "corpus_remediation": (
                    "Give every domain a first-seen snapshot spread over T0/T1/T2, uncorrelated "
                    "with role, so age becomes a genuine signal with a false-positive region."
                ),
            },
            "shared_host_flag": {
                "signal": "registry marks the host as serving many listings",
                "degenerate": False,
                "interpretation": (
                    "True for exactly the 6 booking providers, 4 directories and 4 neutral "
                    "publishers. It marks 'is a real multi-listing provider', NOT 'is authorized': "
                    "each shared provider is authorized for only about half the entities it serves "
                    "(160 authorized vs 160 unauthorized entity-provider pairs), so entity "
                    "relativity survives it and it cannot substitute for a delegation check."
                ),
            },
        },
        "corpus_separability_caveat": (
            "The corpus is deterministically generated, so a defense that reads the generating "
            "asymmetry correctly can separate it almost perfectly. AegisLink reaching UALER ~0 with "
            "ATPR ~1 on the static replay should therefore be read as evidence that the MECHANISM is "
            "correct, not as an estimate of how it degrades under noise or under an attacker "
            "optimising against it. The adaptive-attack arm (RQ4, Step 5) is where degradation is "
            "measured; a near-zero Brier score on the static replay is a property of a separable "
            "benchmark, not a claim of real-world calibration."
        ),
        "link_structure_note": (
            "Visible page text never contains a hostname in this corpus -- partners are named by "
            "display name and the host lives in the href. Every 'does page A reference domain B' "
            "test therefore has to read outbound links (VerificationContext.references_domain). "
            "Written against visible text instead, B10's corroboration layer failed on 100% of "
            "authorized third-party links and B05/B07 collapsed to self-votes, which is measured in "
            "workflow/diagnostics/debug_baselines.py."
        ),
        "consequence_for_the_ablation_table": (
            "Two of the seven ablations (no_domain_lifecycle, and to a lesser extent "
            "no_contradiction_edges) are expected to show small effects for reasons that lie in the "
            "corpus rather than in the architecture. Predicted before measurement in "
            "aegislink/ablations.py ABLATION_PREDICTIONS, so the table reads as a test."
        ),
    }


# ======================================================================================
# Main
# ======================================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="restrict to the first 60 queries; for wiring checks, not for results",
    )
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 78)
    print("Step 4 -- AegisLink, 7 ablations, 10 baselines on the frozen Web-RAG replay")
    print("=" * 78)

    env = rebuild_replay()
    corpus, results, fingerprint = env["corpus"], env["results"], env["fingerprint"]
    if args.quick:
        results = results[:60]
        print(f"[quick] restricted to {len(results)} queries -- NOT a results run")

    ev = build_evaluator_side(corpus)
    recorder = TraceRecorder(corpus=corpus, oracle=ev["oracle"], read_phase=PHASE_LIVE)
    fetch = recorder._fetch_text

    official_registry = load_official_registry()
    site_index = build_site_index(corpus.documents)

    def fetch_links(doc_id: str) -> list[str]:
        """Outbound hyperlink targets at the read phase. ``READER``-tier link structure.

        Needed because visible text never carries a hostname: the corpus names partners by display
        name and keeps the host in the ``href``. Every "does page A reference domain B" test in the
        baselines depends on this channel.
        """
        return outbound_links(corpus.documents[doc_id].html(PHASE_LIVE))

    harness = DefenseEvaluationHarness(
        registry=corpus.registry,
        official_registry=official_registry,
        fetch_links=fetch_links,
        site_index=site_index,
        read_phase=PHASE_LIVE,
        replay_fingerprint=fingerprint,
    )

    matrix = ExperimentMatrix.from_splits()
    fit = fit_on_development(harness, results, ev, fetch, matrix)

    defenses = build_all_defenses(fit["thresholds"], fit["calibration"], fingerprint)
    bound = assert_replay_bound(defenses, fingerprint)
    print(f"[2/6] replay binding verified for {bound['n_verifier_configs_bound']} "
          f"verifier configurations")

    # Prove the identical-evidence rule on a live context before the sweep.
    probe_ctx = harness.context(results[0], fetch)
    shared = assert_shared_evidence(
        probe_ctx, defenses, entity_id=results[0].entity_id, action=results[0].action
    )
    print(f"[2/6] identical-evidence check: {shared['n_defenses_checked']} defenses, "
          f"one shared context, firewall clean")

    per_defense = run_sweep(defenses, harness, results, ev, recorder)
    metrics = compute_all_metrics(per_defense, ev)
    identifiability = audit_evidence_families(ev)

    # ------------------------------------------------------------------ ablation effects
    full_rows = metrics.get("aegislink_full", {})
    effect_regime = "validation" if "validation" in full_rows else next(iter(full_rows), None)
    full = full_rows.get(effect_regime, {}) if effect_regime else {}
    ablation_effects: dict[str, Any] = {"_regime": effect_regime}
    for name in ABLATION_IDS:
        row = metrics.get(name, {}).get(effect_regime, {}) if effect_regime else {}
        f_ualer = (full.get("primary") or {}).get("UALER")
        f_atpr = (full.get("primary") or {}).get("ATPR")
        a_ualer = (row.get("primary") or {}).get("UALER")
        a_atpr = (row.get("primary") or {}).get("ATPR")
        d_ualer = None if (a_ualer is None or f_ualer is None) else round(a_ualer - f_ualer, 6)
        d_atpr = None if (a_atpr is None or f_atpr is None) else round(a_atpr - f_atpr, 6)
        ablation_effects[name] = {
            "prediction": ABLATION_PREDICTIONS[name],
            "delta_ualer_vs_full": d_ualer,
            "delta_atpr_vs_full": d_atpr,
            "predicted_direction": PREDICTED_DIRECTION.get(name),
            "realised_direction": _direction(d_ualer, d_atpr),
            "prediction_holds": (
                None
                if (d_ualer is None or d_atpr is None)
                else _direction(d_ualer, d_atpr) == PREDICTED_DIRECTION.get(name)
            ),
            "verdict_distribution": row.get("verdict_distribution"),
        }
    ablation_effects["_prediction_policy"] = (
        "Directions were stated in aegislink/ablations.py before any measurement. Where "
        "prediction_holds is false the disagreement is reported and explained, not reconciled by "
        "adjusting the mechanism -- see observed_mechanisms below."
    )
    ablation_effects["_observed_mechanisms"] = OBSERVED_MECHANISMS

    # ------------------------------------------------------------------ write artifacts
    print("\n[6/6] Writing artifacts")
    RESULTS.mkdir(exist_ok=True)

    evaluation_doc = {
        "metadata": {
            "step": "Step 4 of 8 -- AegisLink defense framework and baseline suite",
            "contract_ref": "CONTRACT.md Sections 3 (RQ2, RQ3), 9, 10, 11",
            "builder": "workflow/16_evaluate_aegislink_and_baselines.py",
            "replay_fingerprint": fingerprint,
            "replay_fingerprint_verified": True,
            "read_phase": PHASE_LIVE,
            "n_queries": len(results),
            "n_configurations": len(defenses),
            "regimes_reported": list(READABLE_REGIMES),
            "regimes_refused": ["test", "transfer_holdout", "adaptive_holdout"],
            "scope_note": (
                "Step 4 establishes that every defense runs and is comparable on identical frozen "
                "evidence. The confirmatory test-split evaluation, paired bootstrap, Holm "
                "correction and Pareto frontier are Step 5."
            ),
            "quick_mode": bool(args.quick),
            "elapsed_seconds": round(time.time() - t0, 2),
        },
        "framework": describe_framework(),
        "verifier": defenses["aegislink_full"].describe(),
        "ablations": describe_ablations(),
        "adapter": describe_adapter(),
        "metric_definitions": describe_metrics(),
        "fitting": {
            "admission_report": fit["admission_report"],
            "calibration": fit["calibration"].as_dict(),
            "thresholds": fit["thresholds"].as_dict(),
            "n_fit_queries": fit["n_fit_queries"],
            "n_fit_triples": fit["n_fit_triples"],
            "policy": fit["policy"],
        },
        "replay_binding": bound,
        "identical_evidence_check": shared,
        "exposure_firewall": describe_firewall(),
        "metrics": metrics,
        "ablation_effects": ablation_effects,
        "latency": {
            name: payload["pipeline"] for name, payload in per_defense.items()
        },
    }
    assert_no_leakage(
        evaluation_doc["metrics"],
        location="aegislink_defense_evaluation.metrics",
        allow_free_text=True,
    )
    (RESULTS / "aegislink_defense_evaluation.json").write_text(
        json.dumps(evaluation_doc, indent=2) + "\n", encoding="utf-8"
    )

    (RESULTS / "baseline_deviation_log.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "contract_ref": "CONTRACT.md Section 10",
                    "builder": "workflow/16_evaluate_aegislink_and_baselines.py",
                    "n_baselines": len(BASELINE_IDS),
                    "suite_check": assert_suite_complete(),
                },
                "deviation_log": deviation_log(),
                "baselines": describe_baselines(),
                "ablation_metadata": {
                    k: v.as_dict() for k, v in ablation_metadata().items()
                },
                "baseline_metadata": {
                    k: v.as_dict() for k, v in baseline_metadata().items()
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (RESULTS / "evidence_family_identifiability.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "contract_ref": "CONTRACT.md Section 3 (RQ3) evidence families",
                    "builder": "workflow/16_evaluate_aegislink_and_baselines.py",
                    "purpose": (
                        "Report which evidence families are separable on this corpus BEFORE the "
                        "ablation table is read, and record the shortcuts AegisLink declines."
                    ),
                },
                "audit": identifiability,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    stage_summary = {
        "metadata": {
            "contract_ref": "CONTRACT.md Section 3 (RQ2)",
            "replay_fingerprint": fingerprint,
            "n_queries": len(results),
            "attribution_note": (
                "Outcome-anchored attribution (web_rag/trace_recorder.py). A defense that presents "
                "nothing and omits nothing has no failure to attribute, so 'none' rising is the "
                "intended direction of improvement."
            ),
        },
        "per_defense": {
            name: payload["trace_summary"] for name, payload in per_defense.items()
        },
    }
    (RESULTS / "defense_stage_traces_summary.json").write_text(
        json.dumps(stage_summary, indent=2) + "\n", encoding="utf-8"
    )

    traces_doc = {
        "metadata": {
            "contract_ref": "CONTRACT.md Section 3 (RQ2)",
            "replay_fingerprint": fingerprint,
            "schema": "one entry per defense; five stages per query",
            "n_defenses": len(per_defense),
            "n_queries": len(results),
        },
        "traces": {
            name: [t.as_dict() for t in payload["traces"]]
            for name, payload in per_defense.items()
        },
    }
    # Compact separators: 18 defenses x 1,008 five-stage traces is large, and pretty-printing it
    # tripled the file for no reader benefit -- it is a machine-read artifact.
    (BENCH / "defense_traces.json").write_text(
        json.dumps(traces_doc, separators=(",", ":")) + "\n", encoding="utf-8"
    )

    for p in (
        RESULTS / "aegislink_defense_evaluation.json",
        RESULTS / "baseline_deviation_log.json",
        RESULTS / "evidence_family_identifiability.json",
        RESULTS / "defense_stage_traces_summary.json",
        BENCH / "defense_traces.json",
    ):
        print(f"[6/6] wrote {p.relative_to(ROOT)} ({p.stat().st_size / 1e6:.2f} MB)")

    # ------------------------------------------------------------------ summary
    report_regime = effect_regime or "train_development"
    print("\n" + "=" * 78)
    print(f"Step 4 summary ({report_regime} split; test never read)")
    print("=" * 78)
    print(f"{'configuration':34s} {'UALER':>7s} {'ATPR':>7s} {'FRR':>7s} {'OSMR':>7s} "
          f"{'abst':>7s} {'Brier':>7s}")
    order = ["aegislink_full", *ABLATION_IDS, *BASELINE_IDS]
    for name in order:
        row = metrics.get(name, {}).get(report_regime)
        if not row:
            continue
        p, s, c = row["primary"], row["secondary"], row["calibration"]
        print(f"{name:34s} {_fmt(p['UALER']):>7s} {_fmt(p['ATPR']):>7s} "
              f"{_fmt(s['FRR']):>7s} {_fmt(s['OSMR']):>7s} "
              f"{_fmt(s['abstention_rate']):>7s} {_fmt(c['brier']):>7s}")

    print(f"\nElapsed {time.time() - t0:.1f}s")
    print("Artifacts in results/ ; full traces in data/benchmark/defense_traces.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
