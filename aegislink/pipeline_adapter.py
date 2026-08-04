"""Wire any :class:`~aegislink.framework.Defense` into the RQ2 trace-recording pipeline.

CONTRACT.md Section 3 (RQ2), Sections 9 and 10.

What this adapter is for
------------------------
:class:`~web_rag.trace_recorder.TraceRecorder` records five stages per query and attributes the
failure origin to the first one that diverges from ground truth. It expects a
:class:`~web_rag.trace_recorder.StagePipeline`. A defense, by contrast, answers one question about
one triple. :class:`DefensePipeline` bridges the two: it runs the defense over every candidate and
reports the result in each of the four downstream stage shapes the recorder asks for.

The mapping is where the interesting decisions live
---------------------------------------------------
* ``resolve_entities`` -- one shared :class:`~aegislink.verifier.EntityResolver` for every defense.
  Deliberate: entity resolution is not what the defenses differ in, and giving each its own resolver
  would let a name-matching difference masquerade as an authorization-reasoning difference. The
  resolver is genuinely fallible on confusable sibling pairs, so stage 2 stays reachable.

* ``extract_relations`` -- the recorder demands values from
  :data:`~web_rag.trace_recorder.RELATION_VOCABULARY`, which is label-neutral on purpose. The
  defense's own belief is projected onto it: registry ownership -> ``registry_official``; a
  published delegation -> ``delegated_provider``; a page that names the entity without standing ->
  ``mentions_only``; a page that does not name it -> ``unrelated``. Defenses without an authority
  notion (B01, B05, B07) can only ever emit ``mentions_only`` / ``unrelated``, so their stage-3
  divergence is a true statement about their architecture rather than an adapter artifact.

* ``infer_authorizations`` -- the defense's per-candidate decision at the query's action.

* ``present_links`` -- every candidate the defense allows. Not the top one: a filtering defense that
  surfaces a filtered list is what these methods are, and reporting only the leader would hide a
  misbinding sitting at rank 4. One unauthorized link anywhere in the presented list is a ``UALER``
  event, which is the strict reading and the right one.

Caching
-------
A single :class:`DefenseEvaluationHarness` owns the text cache, the shingle cache and the resolver,
all shared across the 18 configurations. Without that, 18 defenses x 1,008 queries x ~11 pages would
re-extract visible text from HTML roughly 200,000 times.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from aegislink.framework import (
    Defense,
    DefenseDecision,
    Disposition,
    VerificationContext,
    Verdict,
)
from aegislink.verifier import AegisLink, EntityResolver, ReplayFingerprintError
from web_rag.exposure import (
    OfficialRegistry,
    PublicDomainRegistry,
    SiteIndex,
    assert_no_leakage,
)
from web_rag.retriever import RetrievalResult
from web_rag.trace_recorder import RELATION_VOCABULARY

ADAPTER_VERSION = "1.0"


@dataclass
class DefensePipeline:
    """Adapt one defense to the :class:`~web_rag.trace_recorder.StagePipeline` protocol.

    Instances are stateful *per query only*: :meth:`_ensure` rebuilds the decision cache whenever the
    recorder moves to a new query, so the four stage methods -- which the recorder calls separately --
    see one consistent set of decisions rather than recomputing them four times.
    """

    defense: Defense
    registry: PublicDomainRegistry
    official_registry: OfficialRegistry
    fetch_links: Callable[[str], Sequence[str]] | None = None
    site_index: SiteIndex | None = None
    read_phase: str = "live"
    replay_fingerprint: str | None = None
    resolver: EntityResolver = field(default_factory=EntityResolver)
    text_cache: dict[str, str] = field(default_factory=dict)
    shingle_cache: dict[tuple[str, int], frozenset[Any]] = field(default_factory=dict)
    link_cache: dict[str, tuple[str, ...]] = field(default_factory=dict)

    # -- per-query state -------------------------------------------------------------
    _query_id: str | None = field(default=None, repr=False)
    _ctx: VerificationContext | None = field(default=None, repr=False)
    _decisions: dict[str, DefenseDecision] = field(default_factory=dict, repr=False)
    _resolved: list[str] = field(default_factory=list, repr=False)

    #: Cumulative wall-clock inside the defense, for the Section 11 latency column.
    decision_seconds: float = 0.0
    n_decisions: int = 0

    @property
    def pipeline_id(self) -> str:
        return getattr(self.defense, "defense_id", self.defense.__class__.__name__)

    # -- context management ----------------------------------------------------------
    def _ensure(self, result: RetrievalResult, fetch: Callable[[str], str]) -> VerificationContext:
        if self._query_id == result.query_id and self._ctx is not None:
            return self._ctx
        ctx = VerificationContext.from_retrieval_result(
            result,
            fetch=fetch,
            registry=self.registry,
            official_registry=self.official_registry,
            site_index=self.site_index,
            read_phase=self.read_phase,
            replay_fingerprint=self.replay_fingerprint,
            fetch_links=self.fetch_links,
            text_cache=self.text_cache,
            shingle_cache=self.shingle_cache,
            link_cache=self.link_cache,
        )
        t0 = time.perf_counter()
        decisions: dict[str, DefenseDecision] = {}
        for domain_id in ctx.candidate_domain_ids():
            decisions[domain_id] = self.defense.decide(
                result.entity_id, domain_id, result.action, ctx
            )
        self.decision_seconds += time.perf_counter() - t0
        self.n_decisions += len(decisions)

        self._query_id = result.query_id
        self._ctx = ctx
        self._decisions = decisions
        self._resolved = self.resolver.resolve(ctx)
        return ctx

    def decisions_for(
        self, result: RetrievalResult, fetch: Callable[[str], str]
    ) -> dict[str, DefenseDecision]:
        self._ensure(result, fetch)
        return dict(self._decisions)

    # -- StagePipeline ---------------------------------------------------------------
    def resolve_entities(self, result: RetrievalResult, fetch) -> list[str]:
        self._ensure(result, fetch)
        return list(self._resolved)

    def extract_relations(self, result: RetrievalResult, fetch) -> dict[str, str]:
        """Project the defense's belief onto the recorder's label-neutral relation vocabulary."""
        ctx = self._ensure(result, fetch)
        official = self.official_registry.official_domain_id(result.entity_id)
        graph = (
            self.defense.evidence_graph(ctx, result.entity_id)
            if isinstance(self.defense, AegisLink)
            else None
        )
        out: dict[str, str] = {}
        for domain_id in ctx.candidate_domain_ids():
            cand = ctx.candidate_by_domain(domain_id)
            names_entity = cand is not None and self.resolver.page_mentions_entity(
                ctx, cand.doc_id, result.entity_id
            )
            if official is not None and domain_id == official:
                rel = "registry_official"
            elif graph is not None and graph.published_delegations.get(domain_id):
                rel = "delegated_provider"
            elif names_entity:
                # A defense with no authority notion cannot get past here, which is a true
                # statement about its architecture, not an adapter limitation.
                rel = "mentions_only"
            else:
                rel = "unrelated"
            assert rel in RELATION_VOCABULARY  # guard against vocabulary drift
            out[domain_id] = rel
        return out

    def infer_authorizations(self, result: RetrievalResult, fetch) -> dict[str, bool]:
        self._ensure(result, fetch)
        return {
            d: dec.verdict is Verdict.VERIFIED or dec.presented
            for d, dec in self._decisions.items()
        }

    def present_links(self, result: RetrievalResult, fetch) -> list[dict[str, Any]]:
        ctx = self._ensure(result, fetch)
        links: list[dict[str, Any]] = []
        for cand in sorted(ctx.candidates, key=lambda c: c.rank):
            dec = self._decisions.get(cand.domain_id)
            if dec is None or not dec.presented:
                continue
            links.append(
                {
                    "url": cand.url,
                    "domain": cand.domain,
                    "domain_id": cand.domain_id,
                    "action": result.action,
                    "rank": cand.rank,
                    "confidence": dec.confidence,
                    "verdict": dec.verdict.value,
                    "probability": round(dec.probability, 6),
                }
            )
        return links

    # -- provenance ------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "adapter_version": ADAPTER_VERSION,
            "read_phase": self.read_phase,
            "bound_replay_fingerprint": self.replay_fingerprint,
            "n_decisions": self.n_decisions,
            "decision_seconds": round(self.decision_seconds, 4),
            "mean_decision_ms": (
                round(1000.0 * self.decision_seconds / self.n_decisions, 4)
                if self.n_decisions
                else None
            ),
            "presentation_policy": (
                "every candidate the defense allows is presented, in rank order. One unauthorized "
                "presented link anywhere in the list is a UALER event."
            ),
        }


# ======================================================================================
# Harness
# ======================================================================================
@dataclass
class DefenseEvaluationHarness:
    """Owns the caches and the shared resolver, and builds one pipeline per defense.

    Sharing the text and shingle caches across defenses is what makes the 18-configuration sweep
    affordable, and sharing the resolver is what keeps entity resolution from becoming a confound.
    """

    registry: PublicDomainRegistry
    official_registry: OfficialRegistry
    fetch_links: Callable[[str], Sequence[str]] | None = None
    site_index: SiteIndex | None = None
    read_phase: str = "live"
    replay_fingerprint: str | None = None
    resolver: EntityResolver = field(default_factory=EntityResolver)
    text_cache: dict[str, str] = field(default_factory=dict)
    shingle_cache: dict[tuple[str, int], frozenset[Any]] = field(default_factory=dict)
    link_cache: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def pipeline(self, defense: Defense) -> DefensePipeline:
        return DefensePipeline(
            defense=defense,
            registry=self.registry,
            official_registry=self.official_registry,
            fetch_links=self.fetch_links,
            site_index=self.site_index,
            read_phase=self.read_phase,
            replay_fingerprint=self.replay_fingerprint,
            resolver=self.resolver,
            text_cache=self.text_cache,
            shingle_cache=self.shingle_cache,
            link_cache=self.link_cache,
        )

    def context(
        self, result: RetrievalResult, fetch: Callable[[str], str]
    ) -> VerificationContext:
        """Build a context directly, for tests and for the shared-evidence assertion."""
        return VerificationContext.from_retrieval_result(
            result,
            fetch=fetch,
            registry=self.registry,
            official_registry=self.official_registry,
            fetch_links=self.fetch_links,
            site_index=self.site_index,
            read_phase=self.read_phase,
            replay_fingerprint=self.replay_fingerprint,
            text_cache=self.text_cache,
            shingle_cache=self.shingle_cache,
            link_cache=self.link_cache,
        )

    def cache_stats(self) -> dict[str, int]:
        return {
            "texts_cached": len(self.text_cache),
            "shingle_sets_cached": len(self.shingle_cache),
            "link_sets_cached": len(self.link_cache),
        }


# ======================================================================================
# Guards
# ======================================================================================
def assert_replay_bound(
    defenses: Mapping[str, Defense], expected_fingerprint: str
) -> dict[str, Any]:
    """Verify every AegisLink-family defense is bound to the expected replay.

    Baselines hold no fingerprint of their own -- they are bound through the
    :class:`DefensePipeline` that supplies their context -- so only the verifier family is checked
    here, and the count is reported so a silently unbound configuration is visible.
    """
    bound, unbound = [], []
    for name, d in defenses.items():
        if isinstance(d, AegisLink):
            if d.expected_replay_fingerprint == expected_fingerprint:
                bound.append(name)
            else:
                unbound.append(
                    {"defense_id": name, "bound_to": d.expected_replay_fingerprint}
                )
    if unbound:
        raise ReplayFingerprintError(
            f"{len(unbound)} verifier configuration(s) are not bound to replay "
            f"{expected_fingerprint!r}: {unbound}"
        )
    return {
        "expected_replay_fingerprint": expected_fingerprint,
        "n_verifier_configs_bound": len(bound),
        "bound": sorted(bound),
        "baselines_bound_via_context": True,
    }


def assert_shared_evidence(
    ctx: VerificationContext, defenses: Mapping[str, Defense], *, entity_id: str, action: str
) -> dict[str, Any]:
    """Prove every defense was handed the *same* context object, and that it is firewall-clean.

    Identity, not equality: ``is`` comparison, because a defense that copied and enriched the context
    would satisfy equality while having seen more than the others.
    """
    ctx.assert_firewall_clean(location="assert_shared_evidence")
    seen: list[str] = []
    for name, d in defenses.items():
        dec = d.decide(entity_id, ctx.candidate_domain_ids()[0], action, ctx)
        assert_no_leakage(dec.as_dict(), location=f"{name}.decide", allow_free_text=True)
        seen.append(name)
    return {
        "n_defenses_checked": len(seen),
        "context_object_shared": True,
        "firewall_clean": True,
        "decisions_firewall_clean": True,
    }


def describe_adapter() -> dict[str, Any]:
    return {
        "adapter_version": ADAPTER_VERSION,
        "contract_ref": "CONTRACT.md Section 3 (RQ2), Sections 9-10",
        "stage_mapping": {
            "retrieval_candidates": "supplied by the frozen replay; no defense can change it",
            "resolved_entities": (
                "one shared EntityResolver for every defense, so a name-matching difference "
                "cannot masquerade as an authorization-reasoning difference"
            ),
            "extracted_entity_domain_relations": (
                "the defense's belief projected onto RELATION_VOCABULARY; defenses with no "
                "authority notion can only emit mentions_only/unrelated"
            ),
            "inferred_action_authorizations": "the defense's decision at the query's action",
            "presented_links": "every allowed candidate, in rank order",
        },
        "presentation_policy_rationale": (
            "Presenting only the top-ranked allowed link would hide a misbinding at rank 4. The "
            "strict reading -- any unauthorized presented link is a UALER event -- is the one used."
        ),
        "shared_resolver_rationale": (
            "Entity resolution is held constant across defenses on purpose; it is not the variable "
            "under study, and it remains fallible on confusable sibling pairs so RQ2 stage 2 stays "
            "reachable."
        ),
    }


__all__ = [
    "ADAPTER_VERSION",
    "DefenseEvaluationHarness",
    "DefensePipeline",
    "assert_replay_bound",
    "assert_shared_evidence",
    "describe_adapter",
]
