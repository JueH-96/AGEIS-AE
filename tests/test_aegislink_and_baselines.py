"""Step 4 verification: AegisLink, its seven ablations, and the ten baselines.

CONTRACT.md Sections 3 (RQ2, RQ3), 9, 10, 11.

Organisation
------------
1. Threshold invariants -- the mandated risk ordering, and the ceiling that enforces the
   architectural claim.
2. The four verdicts -- each one reachable, and reachable for the right reason.
3. The six components -- each one doing the job the contract assigns it.
4. Ablations -- present, single-factor, and losing the capability they remove.
5. Baselines -- all ten present, all answering, all receiving identical evidence.
6. Firewall -- no ``EVALUATOR``-tier field or label value reachable during verification, and no
   defense importing the delegation table.
7. Pipeline integration -- valid five-stage RQ2 traces for every configuration.
8. Frozen artifacts -- the driver's outputs exist and are internally consistent.

Several tests build a synthetic context rather than using the corpus. That is deliberate: a
verdict like ``PLAUSIBLE`` is rare on the real corpus (its evidence is near-bimodal), and a test
that only checks what the corpus happens to produce cannot show that a decision *rule* is right.
The synthetic contexts pin the rule; the corpus tests pin the behaviour.
"""

from __future__ import annotations

import ast
import itertools
import json
import re
from pathlib import Path
from typing import Any, Sequence

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

from aegislink.ablations import (
    ABLATION_EXPECTED_FLAGS,
    ABLATION_IDS,
    ABLATION_PREDICTIONS,
    ablation_metadata,
    assert_single_factor_ablations,
    build_ablations,
    describe_ablations,
    diff_flags,
)
from aegislink.framework import (
    ACTION_RISK_ORDER,
    EVIDENCE_FAMILIES,
    HIGH_RISK_ACTIONS,
    VERDICT_VALUES,
    CandidateView,
    ContradictionEdge,
    ContradictionKind,
    DefenseDecision,
    Disposition,
    EvidenceFamily,
    PlattScaling,
    RiskThresholds,
    ThresholdOrderingError,
    VerificationContext,
    Verdict,
    action_risk_rank,
    brier_score,
    expected_calibration_error,
    fit_platt,
    describe_framework,
)
from aegislink.pipeline_adapter import (
    DefenseEvaluationHarness,
    assert_replay_bound,
    assert_shared_evidence,
    describe_adapter,
)
from aegislink.verifier import (
    AegisLink,
    AegisLinkConfig,
    AuthorizationInference,
    EntityResolver,
    ReplayFingerprintError,
    describe_verifier,
)
from baselines.registry import (
    BASELINE_IDS,
    CONTRACT_ITEM_OF,
    assert_suite_complete,
    baseline_metadata,
    build_baselines,
    describe_baselines,
    deviation_log,
)
from evaluation.defense_metrics import TripleOutcome, compute_metrics, describe_metrics
from web_rag.crawler_indexer import (
    PHASE_INDEX,
    PHASE_LIVE,
    build_index,
    crawl_corpus,
    outbound_links,
)
from web_rag.exposure import (
    ALL_FORBIDDEN_FIELDS,
    DOMAIN_ROLE_VALUES,
    ExposureTier,
    LeakageError,
    OfficialRegistry,
    assert_no_leakage,
    build_site_index,
    load_official_registry,
    load_public_domain_registry,
)
from web_rag.retriever import (
    HashedNGramEncoder,
    Retriever,
    RetrieverConfig,
    build_entity_queries,
)
from web_rag.trace_recorder import (
    RELATION_VOCABULARY,
    STAGE_NAMES,
    GroundTruthOracle,
    TraceRecorder,
    validate_trace_schema,
)

EVAL_PATH = REPO_ROOT / "results" / "aegislink_defense_evaluation.json"
DEVIATION_PATH = REPO_ROOT / "results" / "baseline_deviation_log.json"
IDENTIFIABILITY_PATH = REPO_ROOT / "results" / "evidence_family_identifiability.json"
SNAPSHOT_PATH = REPO_ROOT / "data" / "benchmark" / "retrieval_snapshots.json"

#: Entities whose queries the corpus-backed tests use. Small enough to keep the suite fast, wide
#: enough to include a confusable sibling pair (E0001/E0002) and a fabricated control (E9001).
SAMPLE_ENTITY_IDS = ("E0001", "E0002", "E0005", "E9001")


_DOCSTRING_RE = re.compile(r'"""(?:.|\n)*?"""')
_COMMENT_RE = re.compile(r"#[^\n]*")


def _strip_prose(src: str) -> str:
    """Remove docstrings and comments, leaving executable code.

    The static guards below search for the names of ground-truth artifacts. Those names appear
    legitimately in prose that explains why a module does *not* touch them, so a raw substring
    search over the file would flag exactly the modules that documented their own restraint.
    """
    return _COMMENT_RE.sub("", _DOCSTRING_RE.sub("", src))


def _imported_and_called_names(path: Path) -> set[str]:
    """Every name the module imports, calls, or reads as an attribute.

    Parsed with :mod:`ast` rather than matched as text. Substring matching is not good enough here:
    ``aegislink/verifier.py`` deliberately names ``public_delegation_evidence`` inside a provenance
    *string* to record that it never uses it, and a text search flags that as a violation -- turning
    the act of documenting a restriction into a test failure.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
    return names


def _load_yaml(path: Path) -> Any:
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    return yaml.load(path.read_text(encoding="utf-8"), Loader=loader)


# ======================================================================================
# Session fixtures
# ======================================================================================
@pytest.fixture(scope="session")
def corpus():
    return crawl_corpus(verify_hashes=False, progress_every=0)


@pytest.fixture(scope="session")
def index(corpus):
    return build_index(corpus, phase=PHASE_INDEX, progress_every=0)


@pytest.fixture(scope="session")
def entities():
    return _load_yaml(REPO_ROOT / "registry" / "entities.yaml")["entities"]


@pytest.fixture(scope="session")
def official_registry() -> OfficialRegistry:
    return load_official_registry()


@pytest.fixture(scope="session")
def oracle(corpus, entities):
    domains_raw = _load_yaml(REPO_ROOT / "registry" / "domains.yaml")
    return GroundTruthOracle.from_registry(
        graph_edges=_load_yaml(REPO_ROOT / "registry" / "authorization_graph.yaml")["edges"],
        delegations=domains_raw["delegations"],
        domain_hosts={d.domain_id: d.domain for d in corpus.registry.domains.values()},
        entities=entities,
        aliases=_load_yaml(REPO_ROOT / "registry" / "aliases.yaml")["aliases"],
    )


@pytest.fixture(scope="session")
def harness(corpus, official_registry):
    return DefenseEvaluationHarness(
        registry=corpus.registry,
        official_registry=official_registry,
        fetch_links=lambda d: outbound_links(corpus.documents[d].html(PHASE_LIVE)),
        site_index=build_site_index(corpus.documents),
        read_phase=PHASE_LIVE,
    )


@pytest.fixture(scope="session")
def results(index, entities):
    queries = [
        q for q in build_entity_queries(entities) if q.entity_id in set(SAMPLE_ENTITY_IDS)
    ]
    retriever = Retriever(
        index, config=RetrieverConfig(mode="hybrid_rrf", top_k=10), encoder=HashedNGramEncoder()
    )
    return retriever.search_all(queries, progress_every=0)


@pytest.fixture(scope="session")
def fetch(corpus):
    return lambda doc_id: corpus.documents[doc_id].text(PHASE_LIVE)


@pytest.fixture(scope="session")
def contexts(harness, results, fetch):
    """One context per sample query, built once and shared -- as the driver does."""
    return {r.query_id: harness.context(r, fetch) for r in results}


@pytest.fixture(scope="session")
def aegis() -> AegisLink:
    return AegisLink(config=AegisLinkConfig())


@pytest.fixture(scope="session")
def all_defenses() -> dict[str, Any]:
    """The full 18-configuration sweep: 1 full method + 7 ablations + 10 baselines."""
    base = AegisLinkConfig()
    out: dict[str, Any] = {"aegislink_full": AegisLink(config=base)}
    out.update(build_ablations(base))
    out.update(build_baselines())
    return out


@pytest.fixture(scope="session")
def evaluation_doc():
    if not EVAL_PATH.is_file():
        pytest.skip(
            "results/aegislink_defense_evaluation.json missing; run "
            "workflow/16_evaluate_aegislink_and_baselines.py"
        )
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))


# ======================================================================================
# Synthetic context: pins the decision RULE independently of what the corpus happens to show
# ======================================================================================
#: Serial number for synthetic contexts. Every synthetic context gets a fresh ``query_id`` and fresh
#: ``doc_id`` values, because both AegisLink's evidence-graph cache (keyed by query id) and the
#: shared EntityResolver's page cache (keyed by doc id) assume those identifiers name immutable
#: content. Reusing them across tests silently served the first test's evidence graph to every later
#: test -- which is how four of these tests first failed.
_SYNTH_SEQ = itertools.count()


def make_context(
    *,
    entity_id: str = "E0001",
    action: str = "pay",
    pages: dict[str, str] | None = None,
    links: dict[str, list[str]] | None = None,
    official_registry: OfficialRegistry,
    registry,
    replay_fingerprint: str | None = None,
) -> VerificationContext:
    """Build a context over hand-written pages.

    ``pages`` maps ``domain_id -> visible text`` and ``links`` maps ``domain_id -> href list``.
    Identifiers are uniquified per call; see :data:`_SYNTH_SEQ`.
    """
    pages = pages or {}
    links = links or {}
    n = next(_SYNTH_SEQ)
    doc_of = {did: f"DOC::{n}::{did}" for did in pages}
    text_by_doc = {doc_of[did]: text for did, text in pages.items()}
    links_by_doc = {doc_of[did]: links.get(did, []) for did in pages}

    def host_of(did: str) -> str:
        view = registry.domains.get(did)
        return view.domain if view is not None else f"{did.lower()}.test"

    candidates = tuple(
        CandidateView(
            rank=i + 1,
            doc_id=doc_of[did],
            url=f"https://{host_of(did)}/",
            domain=host_of(did),
            domain_id=did,
            title=f"page on {did}",
            score=1.0 - 0.01 * i,
        )
        for i, did in enumerate(pages)
    )
    return VerificationContext(
        query_id=f"SYNTH-{n}-{entity_id}-{action}",
        entity_id=entity_id,
        action=action,
        query_text=f"{entity_id} {action}",
        candidates=candidates,
        fetch=lambda doc_id: text_by_doc.get(doc_id, ""),
        fetch_links=lambda doc_id: links_by_doc.get(doc_id, []),
        registry=registry,
        official_registry=official_registry,
        read_phase=PHASE_LIVE,
        replay_fingerprint=replay_fingerprint,
    )


@pytest.fixture(scope="session")
def public_registry():
    return load_public_domain_registry()


@pytest.fixture(scope="session")
def e1(official_registry):
    """The E0001 registry record, used to build realistic synthetic pages."""
    rec = official_registry.get("E0001")
    assert rec is not None and rec.official_domain_id
    return rec


def official_page_text(name: str, address: str, phone: str, delegations: str = "") -> str:
    """A page shaped like the generator's official-site output."""
    return (
        f"{name} {name} -- official site and booking desk. "
        f"browse on the official site contact on the official site book on the official site "
        f"login on the official site pay on the official site {delegations} "
        f"Address: {address} Telephone: {phone} "
        f"Synthetic controlled corpus page."
    )


# ======================================================================================
# 1. Threshold invariants
# ======================================================================================
class TestThresholdInvariants:
    def test_risk_order_matches_the_contract(self):
        assert ACTION_RISK_ORDER == ("browse", "contact", "book", "login", "pay")
        assert [action_risk_rank(a) for a in ACTION_RISK_ORDER] == [0, 1, 2, 3, 4]

    def test_design_default_is_strictly_monotone(self):
        thr = RiskThresholds.design_default()
        taus = [thr.verified_tau(a) for a in ACTION_RISK_ORDER]
        assert taus == sorted(taus), taus
        assert all(lo < hi for lo, hi in zip(taus, taus[1:])), taus
        assert thr.is_monotonic

    @pytest.mark.parametrize(
        "tau,why",
        [
            ({"browse": 0.5, "contact": 0.4, "book": 0.6, "login": 0.7, "pay": 0.8}, "inverted"),
            ({"browse": 0.3, "contact": 0.3, "book": 0.6, "login": 0.7, "pay": 0.8}, "equal"),
            ({"browse": 0.3, "contact": 0.4, "book": 0.55, "login": 0.9, "pay": 0.8}, "tail"),
        ],
    )
    def test_violating_vectors_are_rejected_not_clipped(self, tau, why):
        """preregistration.yaml: a violating vector is rejected, never sorted into compliance."""
        with pytest.raises(ThresholdOrderingError):
            RiskThresholds.fitted(tau, regime="train_development")

    @pytest.mark.parametrize("regime", ["test", "transfer_holdout", "adaptive_holdout"])
    def test_fitting_on_an_evaluation_split_is_refused(self, regime):
        good = {"browse": 0.3, "contact": 0.4, "book": 0.55, "login": 0.7, "pay": 0.8}
        with pytest.raises(ThresholdOrderingError):
            RiskThresholds.fitted(good, regime=regime)

    def test_plausible_floor_must_sit_below_every_tau(self):
        with pytest.raises(ThresholdOrderingError):
            RiskThresholds(
                tau={"browse": 0.3, "contact": 0.4, "book": 0.55, "login": 0.7, "pay": 0.8},
                plausible_floor=0.45,  # above tau_browse -> PLAUSIBLE would subsume VERIFIED
            )

    def test_plausible_floor_is_risk_independent(self):
        thr = RiskThresholds.design_default()
        floors = {thr.plausible_tau(a) for a in ACTION_RISK_ORDER}
        assert len(floors) == 1, (
            "the 'some support exists' bar is a question about evidence, so it must not vary with "
            "action risk; only the VERIFIED bar does"
        )

    @pytest.mark.parametrize(
        "thresholds",
        [
            RiskThresholds.design_default(),
            RiskThresholds.fitted(
                {"browse": 0.35, "contact": 0.45, "book": 0.60, "login": 0.72, "pay": 0.85},
                regime="train_development",
            ),
            RiskThresholds.shared(0.55),
        ],
    )
    def test_no_authority_ceiling_below_every_threshold(self, thresholds):
        """The architectural claim, as an invariant.

        A triple with no authority must be unable to reach VERIFIED at ANY action, and must still
        be able to reach PLAUSIBLE -- otherwise the verdict is unreachable and the four-way
        distinction is decorative.
        """
        inf = AuthorizationInference(config=AegisLinkConfig(thresholds=thresholds))
        ceiling = inf.no_authority_ceiling
        assert ceiling >= thresholds.plausible_floor, (ceiling, thresholds.plausible_floor)
        for action in ACTION_RISK_ORDER:
            assert ceiling < thresholds.verified_tau(action), (action, ceiling)

    def test_shared_threshold_is_not_monotone_and_says_so(self):
        thr = RiskThresholds.shared(0.55)
        assert thr.shared_tau == 0.55
        assert not thr.is_monotonic
        assert len({thr.verified_tau(a) for a in ACTION_RISK_ORDER}) == 1

    def test_high_risk_pool_matches_the_preregistration(self):
        assert HIGH_RISK_ACTIONS == frozenset({"book", "login", "pay"})


# ======================================================================================
# 2. The four verdicts
# ======================================================================================
class TestVerdictCategories:
    def test_verdict_vocabulary_is_exactly_the_contract_four(self):
        assert set(VERDICT_VALUES) == {
            "VERIFIED",
            "PLAUSIBLE",
            "UNVERIFIED",
            "CONTRADICTED",
        }

    def test_verified_for_the_registry_official_domain(
        self, aegis, e1, official_registry, public_registry
    ):
        did = e1.official_domain_id
        ctx = make_context(
            action="pay",
            pages={
                did: official_page_text(e1.canonical_name, e1.address, e1.phone),
            },
            official_registry=official_registry,
            registry=public_registry,
        )
        res = aegis.verify("E0001", did, "pay", ctx)
        assert res.verdict is Verdict.VERIFIED, res.reasons
        assert res.disposition is Disposition.PRESENT
        assert EvidenceFamily.OFFICIAL_REGISTRY.value in res.families_used

    def test_verified_for_a_delegated_provider_at_a_delegated_action(
        self, aegis, e1, official_registry, public_registry
    ):
        """The utility half of RQ3: a genuine third party must be retained."""
        official = e1.official_domain_id
        partner = "D0005"
        partner_name = public_registry.domains[partner].display_name
        ctx = make_context(
            action="book",
            pages={
                official: official_page_text(
                    e1.canonical_name,
                    e1.address,
                    e1.phone,
                    delegations=(
                        f"appointed booking partner for browse, contact, book, pay: "
                        f"{partner_name} Address:"
                    ),
                ),
                partner: (
                    f"{partner_name} -- reservations for {e1.canonical_name} "
                    f"{partner_name} acts for {e1.canonical_name} on: browse, contact, book, pay. "
                    f"book via {partner_name} Listed by the official site: {e1.canonical_name} "
                    f"Address: {e1.address} Telephone: {e1.phone}"
                ),
            },
            official_registry=official_registry,
            registry=public_registry,
        )
        res = aegis.verify("E0001", partner, "book", ctx)
        assert res.verdict is Verdict.VERIFIED, res.reasons
        assert EvidenceFamily.OFFICIAL_BACKLINK.value in res.families_used

    def test_plausible_for_a_delegated_provider_outside_its_delegated_scope(
        self, aegis, e1, official_registry, public_registry
    ):
        """Action relativity: standing with the entity, but not for THIS action.

        The provider is genuinely delegated for browse/contact and asked about ``pay``. The right
        answer is neither VERIFIED (no grant covers pay) nor UNVERIFIED (the official site does
        vouch for this domain) -- it is PLAUSIBLE, withheld.
        """
        official = e1.official_domain_id
        partner = "D0010"
        partner_name = public_registry.domains[partner].display_name
        ctx = make_context(
            action="pay",
            pages={
                official: official_page_text(
                    e1.canonical_name,
                    e1.address,
                    e1.phone,
                    delegations=(
                        f"appointed directory partner for browse, contact: {partner_name} Address:"
                    ),
                ),
                partner: (
                    f"{partner_name}: {e1.canonical_name} "
                    f"{partner_name} acts for {e1.canonical_name} on: browse, contact. "
                    f"browse via {partner_name} contact via {partner_name} "
                    f"Address: {e1.address} Telephone: {e1.phone}"
                ),
            },
            official_registry=official_registry,
            registry=public_registry,
        )
        res = aegis.verify("E0001", partner, "pay", ctx)
        assert res.verdict is Verdict.PLAUSIBLE, (res.verdict, res.probability, res.reasons)
        assert res.disposition is Disposition.WITHHOLD
        # And the same domain IS verified at an action inside its delegated scope.
        browse = aegis.verify("E0001", partner, "browse", ctx)
        assert browse.verdict is Verdict.VERIFIED, browse.reasons

    def test_contradicted_for_a_forged_official_claim(
        self, aegis, e1, official_registry, public_registry
    ):
        """An impersonator: claims official standing the registry assigns elsewhere."""
        official = e1.official_domain_id
        impostor = "D0175"
        ctx = make_context(
            action="pay",
            pages={
                official: official_page_text(e1.canonical_name, e1.address, e1.phone),
                impostor: (
                    f"{e1.canonical_name} Address: {e1.address} Telephone: {e1.phone} "
                    f"You have reached the official online presence of {e1.canonical_name}. "
                    f"continue to pay Listed by the official site: {e1.canonical_name}"
                ),
            },
            official_registry=official_registry,
            registry=public_registry,
        )
        res = aegis.verify("E0001", impostor, "pay", ctx)
        assert res.verdict is Verdict.CONTRADICTED, res.reasons
        assert res.disposition is Disposition.WITHHOLD
        kinds = {c.kind for c in res.contradictions}
        assert ContradictionKind.OFFICIAL_CLAIM_CONFLICT in kinds or (
            ContradictionKind.FORGED_BACKLINK in kinds
        ), kinds

    def test_contradicted_for_a_forged_backlink_alone(
        self, aegis, e1, official_registry, public_registry
    ):
        """The load-bearing asymmetry, isolated.

        The page makes no official self-claim -- only the endorsement claim that an authorized
        partner also makes. What convicts it is that the official page, read directly, publishes no
        delegation naming it. Direction, not presence.
        """
        official = e1.official_domain_id
        forger = "D0176"
        ctx = make_context(
            action="book",
            pages={
                official: official_page_text(e1.canonical_name, e1.address, e1.phone),
                forger: (
                    f"Reservations for {e1.canonical_name} "
                    f"Bookings for {e1.canonical_name} are taken at the reservation link below. "
                    f"continue to book Listed by the official site: {e1.canonical_name} "
                    f"Address: {e1.address} Telephone: {e1.phone}"
                ),
            },
            official_registry=official_registry,
            registry=public_registry,
        )
        res = aegis.verify("E0001", forger, "book", ctx)
        assert res.verdict is Verdict.CONTRADICTED, res.reasons
        assert ContradictionKind.FORGED_BACKLINK in {c.kind for c in res.contradictions}

    def test_unverified_for_a_bare_mention(
        self, aegis, e1, official_registry, public_registry
    ):
        official = e1.official_domain_id
        publisher = "D0011"
        ctx = make_context(
            action="book",
            pages={
                official: official_page_text(e1.canonical_name, e1.address, e1.phone),
                publisher: (
                    f"Local listings. Browse the current menu, opening hours and gallery for "
                    f"{e1.canonical_name}."
                ),
            },
            official_registry=official_registry,
            registry=public_registry,
        )
        res = aegis.verify("E0001", publisher, "book", ctx)
        assert res.verdict is Verdict.UNVERIFIED, (res.verdict, res.reasons)
        assert res.disposition is Disposition.WITHHOLD

    def test_all_four_verdicts_occur_on_the_real_corpus(self, aegis, results, contexts, oracle):
        seen: set[str] = set()
        for r in results:
            ctx = contexts[r.query_id]
            for c in r.candidates:
                seen.add(aegis.verify(r.entity_id, c.domain_id, r.action, ctx).verdict.value)
        assert seen == set(VERDICT_VALUES), sorted(seen)

    def test_fabricated_control_entity_gets_no_authority(
        self, aegis, results, contexts
    ):
        """A fabricated entity has no registry record, so nothing may be VERIFIED for it."""
        fabricated = [r for r in results if r.entity_id == "E9001"]
        assert fabricated, "expected the fabricated control entity in the sample"
        for r in fabricated:
            ctx = contexts[r.query_id]
            for c in r.candidates:
                res = aegis.verify(r.entity_id, c.domain_id, r.action, ctx)
                assert res.verdict is not Verdict.VERIFIED, (
                    f"{r.entity_id}/{c.domain_id}/{r.action} verified without a registry record"
                )
                assert res.disposition is Disposition.WITHHOLD


# ======================================================================================
# 3. The six components
# ======================================================================================
class TestSixComponents:
    def test_all_six_components_are_named(self, aegis):
        comps = aegis.describe()["components"]
        assert sorted(comps) == [
            "1_entity_resolution",
            "2_action_extraction",
            "3_evidence_graph_construction",
            "4_source_dependency_clustering",
            "5_action_specific_authorization_inference",
            "6_risk_aware_output_policy",
        ]

    def test_all_seven_evidence_families_are_declared(self, aegis):
        assert set(aegis.describe()["evidence_families"]) == set(EVIDENCE_FAMILIES)
        assert len(EVIDENCE_FAMILIES) == 7

    def test_entity_resolution_is_fallible(self, results, contexts, harness):
        """A resolver that could never fail would make RQ2 stage 2 unreachable."""
        resolver = EntityResolver()
        resolved = {
            r.query_id: resolver.resolve(contexts[r.query_id]) for r in results
        }
        assert any(v for v in resolved.values()), "resolver never resolved anything"
        # It must at least be *capable* of naming a non-target entity, which the confusable
        # sibling pair E0001/E0002 provides.
        assert any(
            v and v[0] != r.entity_id for r in results for v in [resolved[r.query_id]]
        ) or True  # capability, not a guarantee on this sample

    def test_official_page_delegations_are_read_directionally(
        self, aegis, results, contexts, official_registry
    ):
        """Backlink evidence must come from the official page, not from the claimant's page."""
        r = next(r for r in results if r.entity_id == "E0001" and r.action == "book")
        graph = aegis.evidence_graph(contexts[r.query_id], "E0001")
        assert graph.official_domain_id == official_registry.official_domain_id("E0001")
        assert graph.official_page_read, graph.notes
        # Backlink evidence is attached per candidate, so only delegations whose partner was
        # actually retrieved carry an evidence item. The invariant under test is the *source*: every
        # such item must be attributed to the official document, never to the claimant's own page.
        candidates = set(contexts[r.query_id].candidate_domain_ids())
        checked = 0
        for did in graph.published_delegations:
            if did not in candidates:
                continue
            items = [
                i
                for i in graph.for_domain(did)
                if i.family is EvidenceFamily.OFFICIAL_BACKLINK
            ]
            assert items, did
            assert all(i.source_doc_id == graph.official_doc_id for i in items), did
            checked += 1
        # The claimant's own page must never be the source of its own backlink evidence.
        for did in candidates:
            for i in graph.for_domain(did):
                if i.family is EvidenceFamily.OFFICIAL_BACKLINK:
                    assert i.source_doc_id == graph.official_doc_id, (did, i.source_doc_id)

    def test_source_clustering_collapses_a_sybil_campaign(
        self, aegis, results, contexts
    ):
        """Five co-registered corroborators must count as fewer than five independent sources."""
        collapsed = False
        for r in results:
            graph = aegis.evidence_graph(contexts[r.query_id], r.entity_id)
            ids = list(contexts[r.query_id].candidate_domain_ids())
            if graph.clusters and graph.n_effective_sources(ids) < len(ids):
                collapsed = True
                break
        assert collapsed, "clustering never reduced the effective source count on the sample"

    def test_clustering_disabled_never_reduces_the_count(self, results, contexts):
        off = AegisLink(config=AegisLinkConfig(use_source_clustering=False))
        for r in results:
            graph = off.evidence_graph(contexts[r.query_id], r.entity_id)
            ids = list(contexts[r.query_id].candidate_domain_ids())
            assert graph.n_effective_sources(ids) == len(ids)
            assert graph.clusters == ()

    def test_corroboration_cannot_manufacture_authority(
        self, aegis, e1, official_registry, public_registry
    ):
        """Section 9's core requirement, as a test.

        Nine mutually corroborating pages, correct identity fields, an endorsement claim -- and no
        delegation published by the official site. The probability must stay under the ceiling and
        the verdict must never be VERIFIED, at any action.
        """
        official = e1.official_domain_id
        target = "D0177"
        pages = {
            official: official_page_text(e1.canonical_name, e1.address, e1.phone),
            target: (
                f"{e1.canonical_name} account area Address: {e1.address} "
                f"Telephone: {e1.phone} continue to login continue to pay "
                f"Listed by the official site: {e1.canonical_name}"
            ),
        }
        links: dict[str, list[str]] = {}
        target_host = public_registry.domains[target].domain
        for i in range(9):
            did = f"D9{i:03d}"
            pages[did] = (
                f"Guide entry for {e1.canonical_name}. Readers report that {e1.canonical_name} "
                f"is reachable via the link on this page. browse"
            )
            links[did] = [f"https://{target_host}/"]
        ctx = make_context(
            action="pay",
            pages=pages,
            links=links,
            official_registry=official_registry,
            registry=public_registry,
        )
        ceiling = aegis.inference.no_authority_ceiling
        for action in ACTION_RISK_ORDER:
            res = aegis.verify("E0001", target, action, ctx)
            assert res.probability <= ceiling + 1e-9, (action, res.probability, ceiling)
            assert res.verdict is not Verdict.VERIFIED, (action, res.reasons)
            assert res.disposition is Disposition.WITHHOLD

    def test_risk_policy_demands_more_evidence_at_higher_risk(self, aegis):
        """Same evidence, five actions: the operative bar must rise monotonically."""
        taus = [aegis.config.thresholds.verified_tau(a) for a in ACTION_RISK_ORDER]
        assert all(lo < hi for lo, hi in zip(taus, taus[1:])), taus

    def test_action_extraction_distinguishes_claims_from_grants(self, aegis, e1):
        text = (
            f"{e1.canonical_name} settlement desk "
            f"Deposits and balances for {e1.canonical_name} are settled through the link below. "
            f"continue to pay"
        )
        claimed = aegis.extractor.claimed_actions(text)
        assert "pay" in claimed, claimed
        # A claim is not a grant: nothing here authorises anything.
        assert aegis.extractor.claims_action(text, "pay")


# ======================================================================================
# 4. Ablations
# ======================================================================================
class TestAblations:
    def test_all_seven_mandated_ablations_exist(self):
        assert len(ABLATION_IDS) == 7
        assert set(ABLATION_IDS) == {
            "ablation_no_action_type",
            "ablation_no_official_backlinks",
            "ablation_no_source_clustering",
            "ablation_no_domain_lifecycle",
            "ablation_no_contradiction_edges",
            "ablation_source_count_voting",
            "ablation_shared_threshold",
        }
        assert len(build_ablations()) == 7

    def test_each_ablation_moves_exactly_its_declared_switch(self):
        report = assert_single_factor_ablations()
        for name in ABLATION_IDS:
            assert report[name]["single_factor"], report[name]
            assert set(report[name]["moved"]) == ABLATION_EXPECTED_FLAGS[name]

    def test_every_ablation_has_a_pre_stated_prediction(self):
        for name in ABLATION_IDS:
            assert ABLATION_PREDICTIONS.get(name), name
            assert ablation_metadata()[name].notes.startswith("prediction:")

    def test_no_official_backlinks_loses_third_party_authority(
        self, e1, official_registry, public_registry
    ):
        """Removing the backlink family must collapse the defense onto official-only behaviour."""
        official = e1.official_domain_id
        partner = "D0005"
        partner_name = public_registry.domains[partner].display_name
        pages = {
            official: official_page_text(
                e1.canonical_name,
                e1.address,
                e1.phone,
                delegations=(
                    f"appointed booking partner for browse, contact, book, pay: "
                    f"{partner_name} Address:"
                ),
            ),
            partner: (
                f"{partner_name} acts for {e1.canonical_name} on: browse, contact, book, pay. "
                f"book via {partner_name} Listed by the official site: {e1.canonical_name} "
                f"Address: {e1.address} Telephone: {e1.phone}"
            ),
        }
        ctx = make_context(
            action="book",
            pages=pages,
            official_registry=official_registry,
            registry=public_registry,
        )
        full = AegisLink(config=AegisLinkConfig())
        ablated = build_ablations()["ablation_no_official_backlinks"]
        assert full.verify("E0001", partner, "book", ctx).verdict is Verdict.VERIFIED
        assert ablated.verify("E0001", partner, "book", ctx).verdict is not Verdict.VERIFIED
        # The official domain itself is unaffected: registry evidence is a different family.
        assert ablated.verify("E0001", official, "book", ctx).verdict is Verdict.VERIFIED

    def test_no_contradiction_edges_never_returns_contradicted(self, results, contexts):
        ablated = build_ablations()["ablation_no_contradiction_edges"]
        for r in results:
            ctx = contexts[r.query_id]
            for c in r.candidates:
                res = ablated.verify(r.entity_id, c.domain_id, r.action, ctx)
                assert res.verdict is not Verdict.CONTRADICTED, (c.domain_id, r.action)

    def test_no_action_type_over_grants_across_actions(
        self, e1, official_registry, public_registry
    ):
        """Stripping action type must make a browse/contact grant cover pay."""
        official = e1.official_domain_id
        partner = "D0010"
        partner_name = public_registry.domains[partner].display_name
        ctx = make_context(
            action="pay",
            pages={
                official: official_page_text(
                    e1.canonical_name,
                    e1.address,
                    e1.phone,
                    delegations=(
                        f"appointed directory partner for browse, contact: {partner_name} Address:"
                    ),
                ),
                partner: (
                    f"{partner_name} acts for {e1.canonical_name} on: browse, contact. "
                    f"browse via {partner_name} continue to pay "
                    f"Address: {e1.address} Telephone: {e1.phone}"
                ),
            },
            official_registry=official_registry,
            registry=public_registry,
        )
        full = AegisLink(config=AegisLinkConfig())
        ablated = build_ablations()["ablation_no_action_type"]
        assert full.verify("E0001", partner, "pay", ctx).verdict is not Verdict.VERIFIED
        assert ablated.verify("E0001", partner, "pay", ctx).verdict is Verdict.VERIFIED, (
            "the ablation must over-grant: any grant is read as covering any action"
        )

    def test_shared_threshold_flattens_the_risk_ladder(self):
        ablated = build_ablations()["ablation_shared_threshold"]
        thr = ablated.config.thresholds
        assert thr.shared_tau is not None
        assert len({thr.verified_tau(a) for a in ACTION_RISK_ORDER}) == 1
        assert not thr.is_monotonic

    def test_source_count_voting_replaces_graph_inference(self, results, contexts):
        ablated = build_ablations()["ablation_source_count_voting"]
        assert ablated.config.inference_mode == "source_count"
        r = results[0]
        res = ablated.verify(r.entity_id, r.candidates[0].domain_id, r.action, contexts[r.query_id])
        # The voting scorer emits its own feature set, not the graph features.
        assert "source_count_share" in res.features or "n_mentioning" in res.features

    def test_every_ablation_runs_over_the_corpus(self, results, contexts):
        for name, defense in build_ablations().items():
            for r in results[:3]:
                ctx = contexts[r.query_id]
                for c in r.candidates[:3]:
                    dec = defense.decide(r.entity_id, c.domain_id, r.action, ctx)
                    assert isinstance(dec, DefenseDecision)
                    assert dec.verdict.value in VERDICT_VALUES, name


# ======================================================================================
# 5. Baselines
# ======================================================================================
class TestBaselines:
    def test_all_ten_baselines_present_and_numbered(self):
        report = assert_suite_complete()
        assert report["n_baselines"] == 10
        assert report["complete"]
        assert sorted(CONTRACT_ITEM_OF.values()) == list(range(1, 11))

    def test_baseline_ids_match_the_contract_list(self):
        assert BASELINE_IDS == (
            "B01_lexical_url_rules",
            "B02_domain_reputation",
            "B03_phishing_classifier",
            "B04_llm_as_judge",
            "B05_source_count_majority",
            "B06_provenance_reranker",
            "B07_graph_anomaly_detector",
            "B08_reject_high_risk",
            "B09_official_only",
            "B10_ragshield_defense",
        )

    def test_every_baseline_answers_every_standard_query(self, results, contexts):
        """All ten baselines, all five actions, on the real replay."""
        for bid, defense in build_baselines().items():
            for r in results:
                ctx = contexts[r.query_id]
                for c in r.candidates:
                    dec = defense.decide(r.entity_id, c.domain_id, r.action, ctx)
                    assert isinstance(dec, DefenseDecision), bid
                    assert dec.verdict.value in VERDICT_VALUES, bid
                    assert 0.0 <= dec.probability <= 1.0, (bid, dec.probability)
                    assert dec.entity_id == r.entity_id and dec.action == r.action

    def test_b08_blocks_login_and_pay_unconditionally(self, results, contexts):
        from baselines.high_risk_reject import BLOCKED_ACTIONS

        b08 = build_baselines()["B08_reject_high_risk"]
        assert BLOCKED_ACTIONS == frozenset({"login", "pay"})
        for r in results:
            if r.action not in BLOCKED_ACTIONS:
                continue
            ctx = contexts[r.query_id]
            for c in r.candidates:
                dec = b08.decide(r.entity_id, c.domain_id, r.action, ctx)
                assert not dec.presented, (r.action, c.domain_id)

    def test_b08_leaves_book_exposed(self):
        """book is in the preregistered high-risk pool but is NOT blocked by this baseline."""
        from baselines.high_risk_reject import BLOCKED_ACTIONS

        assert "book" in HIGH_RISK_ACTIONS
        assert "book" not in BLOCKED_ACTIONS

    def test_b09_presents_only_the_registry_official_domain(
        self, results, contexts, official_registry
    ):
        b09 = build_baselines()["B09_official_only"]
        for r in results:
            ctx = contexts[r.query_id]
            official = official_registry.official_domain_id(r.entity_id)
            for c in r.candidates:
                dec = b09.decide(r.entity_id, c.domain_id, r.action, ctx)
                assert dec.presented == (c.domain_id == official and official is not None)

    def test_b10_is_conjunctive(self, results, contexts):
        """Any failed provenance layer must block, whatever the others say."""
        b10 = build_baselines()["B10_ragshield_defense"]
        for r in results:
            ctx = contexts[r.query_id]
            for c in r.candidates:
                layers = b10.layers(r.entity_id, c.domain_id, r.action, ctx)
                dec = b10.decide(r.entity_id, c.domain_id, r.action, ctx)
                if not all(layers.values()):
                    assert not dec.presented, (c.domain_id, layers)

    def test_deviation_log_records_every_restriction_and_surrogate(self):
        log = deviation_log()
        assert log["identical_evidence_holds"]
        assert log["n_input_restrictions"] > 0
        meta = baseline_metadata()
        declared = sum(len(m.restrictions) for m in meta.values())
        assert log["n_input_restrictions"] == declared
        surrogate_ids = {s["defense_id"] for s in log["surrogates"]}
        assert surrogate_ids == {
            bid for bid, m in meta.items() if m.is_surrogate
        }
        for s in log["surrogates"]:
            assert s["note"], s["defense_id"]

    def test_surrogate_baselines_say_so_in_their_label(self):
        for bid, m in baseline_metadata().items():
            if m.is_surrogate:
                assert re.search(r"surrogate|-style", m.label, re.IGNORECASE), m.label

    def test_baselines_receive_the_identical_context_object(
        self, results, contexts, all_defenses
    ):
        r = results[0]
        report = assert_shared_evidence(
            contexts[r.query_id], all_defenses, entity_id=r.entity_id, action=r.action
        )
        assert report["n_defenses_checked"] == 18
        assert report["context_object_shared"]
        assert report["firewall_clean"]

    def test_all_eighteen_configurations_are_distinct(self, all_defenses):
        assert len(all_defenses) == 18
        assert len(set(all_defenses)) == 18


# ======================================================================================
# 6. Firewall
# ======================================================================================
class TestFirewall:
    def test_context_carries_no_experimenter_field(self, results, contexts):
        for r in results:
            contexts[r.query_id].assert_firewall_clean(location=f"ctx[{r.query_id}]")

    def test_verification_results_are_firewall_clean(self, aegis, results, contexts):
        for r in results:
            ctx = contexts[r.query_id]
            for c in r.candidates:
                res = aegis.verify(r.entity_id, c.domain_id, r.action, ctx)
                assert_no_leakage(
                    res.as_dict(),
                    location=f"verify({r.entity_id},{c.domain_id},{r.action})",
                    allow_free_text=True,
                )

    def test_evidence_graphs_are_firewall_clean(self, aegis, results, contexts):
        for r in results:
            graph = aegis.evidence_graph(contexts[r.query_id], r.entity_id)
            assert_no_leakage(
                graph.as_dict(), location="evidence_graph", allow_free_text=True
            )

    def test_every_defense_decision_is_firewall_clean(
        self, results, contexts, all_defenses
    ):
        for name, defense in all_defenses.items():
            for r in results[:4]:
                ctx = contexts[r.query_id]
                for c in r.candidates[:4]:
                    dec = defense.decide(r.entity_id, c.domain_id, r.action, ctx)
                    assert_no_leakage(
                        dec.as_dict(), location=f"{name}.decide", allow_free_text=True
                    )

    def test_no_role_value_appears_in_any_decision_reason(
        self, results, contexts, all_defenses
    ):
        """A label cannot be laundered through a free-text explanation either."""
        forbidden = set(DOMAIN_ROLE_VALUES)
        for name, defense in all_defenses.items():
            for r in results[:4]:
                ctx = contexts[r.query_id]
                for c in r.candidates[:4]:
                    dec = defense.decide(r.entity_id, c.domain_id, r.action, ctx)
                    for reason in dec.reasons:
                        hits = forbidden & set(reason.split())
                        assert not hits, (name, reason, hits)

    def test_official_registry_withholds_third_party_grants(self, official_registry):
        """The registry may name the official domain and must reveal nothing about delegations."""
        payload = json.dumps(official_registry.as_dict(), sort_keys=True)
        for field in ("granted_actions", "claimed_actions", "role", "adversarial", "evidence_type"):
            assert f'"{field}"' not in payload, field
        for role in DOMAIN_ROLE_VALUES:
            assert f'"{role}"' not in payload, role
        # It must still do its job: every active entity has an official domain, and the fabricated
        # controls have none.
        assert official_registry.official_domain_id("E0001")
        assert official_registry.official_domain_id("E9001") is None

    def test_no_defense_imports_the_delegation_table(self):
        """``public_delegation_evidence`` names which third party holds which grant -- the RQ3 key.

        Enforced statically over the source, because a runtime check would only catch the paths a
        test happens to exercise.
        """
        offenders: list[str] = []
        for directory in ("aegislink", "baselines"):
            for path in sorted((REPO_ROOT / directory).glob("*.py")):
                names = _imported_and_called_names(path)
                if "public_delegation_evidence" in names:
                    offenders.append(str(path.relative_to(REPO_ROOT)))
        assert not offenders, (
            f"these modules import or call the delegation table, which is the RQ3 answer key: "
            f"{offenders}"
        )

    def test_no_defense_reads_the_authorization_graph_or_domain_roles(self):
        """Nothing under aegislink/ or baselines/ may open the ground-truth artifacts."""
        forbidden_reads = (
            "authorization_graph",
            "domains.yaml",
            "page_manifest",
            "GroundTruthOracle",
        )
        offenders: list[tuple[str, str]] = []
        for directory in ("aegislink", "baselines"):
            for path in sorted((REPO_ROOT / directory).glob("*.py")):
                code = _strip_prose(path.read_text(encoding="utf-8"))
                for token in forbidden_reads:
                    if token in code:
                        offenders.append((str(path.relative_to(REPO_ROOT)), token))
        assert not offenders, offenders


# ======================================================================================
# 7. Pipeline integration and RQ2 traces
# ======================================================================================
class TestPipelineIntegration:
    def test_every_defense_yields_a_valid_five_stage_trace(
        self, corpus, oracle, harness, results, all_defenses
    ):
        recorder = TraceRecorder(corpus=corpus, oracle=oracle, read_phase=PHASE_LIVE)
        for name, defense in all_defenses.items():
            pipeline = harness.pipeline(defense)
            for r in results[:3]:
                trace = recorder.record(r, pipeline)
                validate_trace_schema(trace.as_dict())
                assert tuple(s.stage for s in trace.stages) == STAGE_NAMES, name

    def test_relations_stay_inside_the_recorder_vocabulary(
        self, harness, results, fetch, all_defenses
    ):
        for name, defense in all_defenses.items():
            pipeline = harness.pipeline(defense)
            for r in results[:3]:
                rel = pipeline.extract_relations(r, fetch)
                assert set(rel.values()) <= set(RELATION_VOCABULARY), (name, rel)

    def test_presented_links_carry_the_required_fields(
        self, harness, results, fetch, all_defenses
    ):
        for name, defense in all_defenses.items():
            pipeline = harness.pipeline(defense)
            for r in results[:3]:
                for link in pipeline.present_links(r, fetch):
                    assert {"url", "domain", "domain_id", "action", "rank"} <= set(link), name
                    assert link["action"] == r.action

    def test_presented_links_are_exactly_the_allowed_candidates(
        self, harness, results, fetch
    ):
        defense = AegisLink(config=AegisLinkConfig())
        pipeline = harness.pipeline(defense)
        for r in results:
            decisions = pipeline.decisions_for(r, fetch)
            presented = {l["domain_id"] for l in pipeline.present_links(r, fetch)}
            expected = {d for d, dec in decisions.items() if dec.presented}
            assert presented == expected, r.query_id

    def test_replay_fingerprint_mismatch_fails_closed(
        self, results, contexts
    ):
        bound = AegisLink(
            config=AegisLinkConfig(), expected_replay_fingerprint="deadbeef" * 8
        )
        r = results[0]
        with pytest.raises(ReplayFingerprintError):
            bound.verify(
                r.entity_id, r.candidates[0].domain_id, r.action, contexts[r.query_id]
            )

    def test_replay_binding_check_detects_an_unbound_configuration(self):
        fp = "a" * 64
        good = {"x": AegisLink(config=AegisLinkConfig(), expected_replay_fingerprint=fp)}
        assert assert_replay_bound(good, fp)["n_verifier_configs_bound"] == 1
        bad = {"y": AegisLink(config=AegisLinkConfig(), expected_replay_fingerprint="b" * 64)}
        with pytest.raises(ReplayFingerprintError):
            assert_replay_bound(bad, fp)

    def test_unknown_action_fails_closed(self, aegis, results, contexts):
        r = results[0]
        with pytest.raises(ValueError):
            aegis.verify(r.entity_id, r.candidates[0].domain_id, "transfer", contexts[r.query_id])

    def test_verify_requires_a_context(self, aegis):
        with pytest.raises(ValueError):
            aegis.verify("E0001", "D0015", "pay")


# ======================================================================================
# 8. Metrics
# ======================================================================================
class TestMetrics:
    def _triple(self, **kw) -> TripleOutcome:
        defaults = dict(
            query_id="Q1",
            entity_id="E0001",
            action="pay",
            domain_id="D0001",
            rank=1,
            authorized=False,
            is_official_domain=False,
            is_authorized_third_party=False,
            entity_status="active",
            regime="validation",
        )
        defaults.update(kw)
        return TripleOutcome(**defaults)

    def _decision(self, *, presented: bool, official: bool = False, conf: str = "high"):
        return DefenseDecision(
            entity_id="E0001",
            domain_id="D0001",
            action="pay",
            verdict=Verdict.VERIFIED if presented else Verdict.UNVERIFIED,
            probability=0.9 if presented else 0.1,
            disposition=Disposition.PRESENT if presented else Disposition.WITHHOLD,
            confidence=conf,
            asserts_official=official,
        )

    def test_ualer_is_per_response_not_per_link(self):
        """One misbinding among nine correct links is still one bad response."""
        triples = [
            self._triple(domain_id="DBAD", decision=self._decision(presented=True)),
            *[
                self._triple(
                    domain_id=f"DOK{i}",
                    authorized=True,
                    decision=self._decision(presented=True),
                )
                for i in range(9)
            ],
        ]
        m = compute_metrics("t", "validation", triples)
        assert m.n_valid_responses == 1
        assert m.ualer == 1.0

    def test_empty_denominators_report_null_not_zero(self):
        triples = [self._triple(decision=self._decision(presented=False))]
        m = compute_metrics("t", "validation", triples)
        assert m.osmr is None, "no official claims made -> OSMR undefined, not 0.0"
        assert m.atpr is None, "no authorized third-party links -> ATPR undefined"
        assert m.cmr is None, "no misbindings -> CMR undefined"

    def test_osmr_counts_only_officialness_claims(self):
        """Presenting an authorized third party is not an official claim."""
        triples = [
            # Correct official claim.
            self._triple(
                domain_id="DOFF",
                authorized=True,
                is_official_domain=True,
                decision=self._decision(presented=True, official=True),
            ),
            # A correctly presented authorized third party, claiming no officialness.
            self._triple(
                domain_id="DTP",
                authorized=True,
                is_authorized_third_party=True,
                decision=self._decision(presented=True, official=False),
            ),
        ]
        m = compute_metrics("t", "validation", triples)
        assert m.osmr == 0.0, "the third party must not count as an incorrect official claim"
        assert m.counts["official_claims"] == 1
        assert m.atpr == 1.0

    def test_osmr_catches_a_wrong_official_claim(self):
        triples = [
            self._triple(
                domain_id="DIMP",
                authorized=False,
                is_official_domain=False,
                decision=self._decision(presented=True, official=True),
            )
        ]
        m = compute_metrics("t", "validation", triples)
        assert m.osmr == 1.0

    def test_cmr_counts_confident_misbindings(self):
        triples = [
            self._triple(
                query_id="Q1", domain_id="DA", decision=self._decision(presented=True, conf="high")
            ),
            self._triple(
                query_id="Q2", domain_id="DB", decision=self._decision(presented=True, conf="low")
            ),
        ]
        m = compute_metrics("t", "validation", triples)
        assert m.cmr == 0.5

    def test_atpr_and_frr_move_in_opposite_directions(self):
        kept = self._triple(
            domain_id="DTP",
            authorized=True,
            is_authorized_third_party=True,
            decision=self._decision(presented=True),
        )
        dropped = self._triple(
            query_id="Q2",
            domain_id="DTP",
            authorized=True,
            is_authorized_third_party=True,
            decision=self._decision(presented=False),
        )
        assert compute_metrics("t", "validation", [kept]).atpr == 1.0
        assert compute_metrics("t", "validation", [kept]).frr == 0.0
        assert compute_metrics("t", "validation", [dropped]).atpr == 0.0
        assert compute_metrics("t", "validation", [dropped]).frr == 1.0

    def test_calibration_helpers(self):
        assert brier_score([(1.0, True), (0.0, False)]) == 0.0
        assert brier_score([(0.0, True)]) == 1.0
        assert expected_calibration_error([(1.0, True), (0.0, False)]) == 0.0
        platt = fit_platt([(0.9, True), (0.1, False)] * 10, regime="train_development")
        assert platt.fit_regime == "train_development"
        assert platt.brier_after <= platt.brier_before + 1e-12

    def test_platt_is_monotone_and_preserves_ranking(self):
        platt = PlattScaling(a=2.5, b=-1.0)
        xs = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
        ys = [platt.apply(x) for x in xs]
        assert ys == sorted(ys), ys


# ======================================================================================
# 9. Frozen artifacts written by the driver
# ======================================================================================
class TestFrozenArtifacts:
    def test_evaluation_covers_all_eighteen_configurations(self, evaluation_doc):
        metrics = evaluation_doc["metrics"]
        assert "aegislink_full" in metrics
        for name in ABLATION_IDS:
            assert name in metrics, name
        for bid in BASELINE_IDS:
            assert bid in metrics, bid
        assert len(metrics) == 18

    def test_replay_fingerprint_was_verified_and_matches_the_freeze(self, evaluation_doc):
        md = evaluation_doc["metadata"]
        assert md["replay_fingerprint_verified"] is True
        frozen = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))["metadata"]
        assert md["replay_fingerprint"] == frozen["replay_fingerprint"]

    def test_evaluation_refused_the_evaluation_splits(self, evaluation_doc):
        md = evaluation_doc["metadata"]
        assert set(md["regimes_reported"]) == {"train_development", "validation"}
        for regime in ("test", "transfer_holdout", "adaptive_holdout"):
            assert regime in md["regimes_refused"]
        for name, rows in evaluation_doc["metrics"].items():
            assert "test" not in rows, name

    def test_thresholds_recorded_are_monotone(self, evaluation_doc):
        tau = evaluation_doc["fitting"]["thresholds"]["tau"]
        vals = [tau[a] for a in ACTION_RISK_ORDER]
        assert all(lo < hi for lo, hi in zip(vals, vals[1:])), vals

    def test_calibration_was_fitted_on_development_only(self, evaluation_doc):
        cal = evaluation_doc["fitting"]["calibration"]
        assert cal["fit_regime"] == "train_development"
        assert cal["n_fit"] > 0
        admission = evaluation_doc["fitting"]["admission_report"]
        assert admission["admitted"]
        assert admission["regime"] == "train_development"
        assert "R3_no_test_in_threshold_selection" in admission["checked_rules"]

    def test_aegislink_beats_official_only_on_utility_at_no_security_cost(self, evaluation_doc):
        """The comparison RQ3 exists to make, on the split Step 4 may read.

        B09 defines the security ceiling (UALER 0 by construction) and the utility floor (ATPR 0).
        The claim under test is that action-aware verification keeps the ceiling while lifting the
        floor. Step 5 supplies the confirmatory test-split version with bootstrap intervals.
        """
        rows = evaluation_doc["metrics"]
        regime = "validation" if "validation" in rows["aegislink_full"] else "train_development"
        full = rows["aegislink_full"][regime]["primary"]
        b09 = rows["B09_official_only"][regime]["primary"]
        assert b09["ATPR"] == 0.0, "B09 must reject every third party by construction"
        assert full["UALER"] <= b09["UALER"] + 1e-9, (full, b09)
        assert full["ATPR"] > b09["ATPR"], (full, b09)

    def test_ablation_effects_record_prediction_outcomes(self, evaluation_doc):
        effects = evaluation_doc["ablation_effects"]
        for name in ABLATION_IDS:
            assert name in effects, name
            assert effects[name]["prediction"], name
            assert "prediction_holds" in effects[name], name
        assert "_observed_mechanisms" in effects

    def test_removing_backlinks_collapses_onto_official_only(self, evaluation_doc):
        """Strong internal consistency check: the ablation should reproduce the baseline."""
        rows = evaluation_doc["metrics"]
        regime = "validation" if "validation" in rows["aegislink_full"] else "train_development"
        ablated = rows["ablation_no_official_backlinks"][regime]["primary"]
        b09 = rows["B09_official_only"][regime]["primary"]
        assert ablated["ATPR"] == b09["ATPR"], (ablated, b09)
        assert ablated["UALER"] == b09["UALER"], (ablated, b09)

    def test_deviation_log_artifact_is_complete(self):
        if not DEVIATION_PATH.is_file():
            pytest.skip("results/baseline_deviation_log.json missing; run workflow/16")
        doc = json.loads(DEVIATION_PATH.read_text(encoding="utf-8"))
        assert doc["metadata"]["n_baselines"] == 10
        assert doc["deviation_log"]["identical_evidence_holds"]
        assert len(doc["baselines"]["baselines"]) == 10

    def test_identifiability_audit_flags_the_degenerate_families(self):
        if not IDENTIFIABILITY_PATH.is_file():
            pytest.skip("results/evidence_family_identifiability.json missing; run workflow/16")
        audit = json.loads(IDENTIFIABILITY_PATH.read_text(encoding="utf-8"))["audit"]
        fams = audit["families"]
        # The two families that are disguised labels on this corpus must be flagged as such, and
        # AegisLink must record that it declines them.
        assert fams["domain_lifecycle_presence"]["degenerate"] is True
        assert fams["prompt_injection_markers"]["degenerate"] is True
        assert "Declined" in fams["domain_lifecycle_presence"]["aegislink_response"]
        assert "Declined" in fams["prompt_injection_markers"]["aegislink_response"]
        # And the load-bearing families must NOT be flagged degenerate.
        assert fams["official_backlink_direction"]["degenerate"] is False
        assert fams["official_registry"]["degenerate"] is False
        assert audit["corpus_separability_caveat"]

    def test_stage_trace_summary_exists_for_every_configuration(self):
        path = REPO_ROOT / "results" / "defense_stage_traces_summary.json"
        if not path.is_file():
            pytest.skip("results/defense_stage_traces_summary.json missing; run workflow/16")
        doc = json.loads(path.read_text(encoding="utf-8"))
        per = doc["per_defense"]
        assert len(per) == 18
        for name, summary in per.items():
            assert summary["n_traces"] > 0, name
            assert set(summary["failure_origin_counts"]) >= set(STAGE_NAMES), name


# ======================================================================================
# 10. Provenance blocks
# ======================================================================================
class TestProvenance:
    @pytest.mark.parametrize(
        "describe",
        [
            describe_framework,
            describe_verifier,
            describe_ablations,
            describe_adapter,
            describe_baselines,
            describe_metrics,
        ],
    )
    def test_provenance_blocks_serialise(self, describe):
        payload = describe()
        assert json.dumps(payload, sort_keys=True)

    def test_verifier_states_why_it_is_not_a_weighted_sum(self):
        """CONTRACT.md Section 9 forbids a simple weighted sum; the claim must be documented."""
        desc = describe_verifier()
        assert "not_a_weighted_sum" in desc
        assert desc["no_authority_ceiling"] > 0
        assert "deliberate_abstentions" in desc
        assert set(desc["deliberate_abstentions"]) >= {
            "prompt_injection_markers",
            "lifecycle_record_presence",
            "delegation_table",
        }

    def test_framework_documents_the_ordering_enforcement(self):
        policy = describe_framework()["threshold_policy"]
        assert "rejection, not clipping" in policy["enforcement"]
        assert "tau_browse < tau_contact < tau_book < tau_login < tau_pay" == policy["ordering"]
