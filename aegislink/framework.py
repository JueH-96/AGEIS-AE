"""AegisLink core data model: verdicts, evidence, risk thresholds, defense protocol.

CONTRACT.md Sections 3 (RQ3), 9, 10.

What this module fixes
----------------------
Every defense in this project -- AegisLink, its seven ablations, and the ten baselines -- answers
the same question about the same triple ``(entity, domain, action)`` and is handed the same
evidence. That symmetry is a contract requirement (Section 10: "Every baseline MUST receive
identical evidence, splits, and attack budgets unless its design explicitly prohibits an input
type; deviations must be logged"), so it is expressed as a type rather than as a convention:

* :class:`VerificationContext` is the *only* evidence channel. It is constructed once per query
  and passed to every defense unchanged.
* :class:`Defense` is the protocol every defense implements.
* :class:`InputRestriction` records, per defense, which inputs its design forbids it to use. That
  is the "deviations must be logged" clause, machine-readable.

The action-risk ordering
------------------------
CONTRACT.md Section 3 (RQ3) mandates ``tau_browse < tau_contact < tau_book < tau_login < tau_pay``.
:class:`RiskThresholds` refuses to exist unless that holds. The preregistration is explicit that a
fitted vector violating the ordering is *rejected, not clipped* (``preregistration.yaml``
``risk_thresholds``), so :meth:`RiskThresholds.fitted` raises instead of sorting its input.

Verdicts vs. presentation
-------------------------
``verify`` returns one of four epistemic states (Section 3, RQ3)::

    VERIFIED | PLAUSIBLE | UNVERIFIED | CONTRADICTED

A verdict is a statement about evidence. Whether a link is *shown* is a separate policy question,
because the same evidential state warrants different behaviour at different risk levels: a
PLAUSIBLE browse link is worth showing, a PLAUSIBLE payment endpoint is not. Keeping the two apart
is what lets the risk-aware output policy be ablated (``ablation_shared_threshold``) without
touching the evidence layer.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from web_rag.exposure import (
    ExposureTier,
    LifecycleObservation,
    OfficialRegistry,
    PublicDomainRegistry,
    SiteIndex,
    assert_no_leakage,
)
from web_rag.retriever import CANONICAL_ACTIONS, RetrievalResult

FRAMEWORK_VERSION = "1.0"

#: Risk order, lowest to highest. Identical to ``CANONICAL_ACTIONS`` by construction, asserted
#: below so a future reordering of the retriever's tuple cannot silently invert the risk ladder.
ACTION_RISK_ORDER: tuple[str, ...] = ("browse", "contact", "book", "login", "pay")
assert ACTION_RISK_ORDER == tuple(CANONICAL_ACTIONS), (
    "ACTION_RISK_ORDER must match web_rag.retriever.CANONICAL_ACTIONS; the risk ladder in "
    "CONTRACT.md Section 3 (RQ3) is stated over exactly those five actions"
)

#: Actions the preregistration designates high risk (``high_risk_action_pool``).
HIGH_RISK_ACTIONS: frozenset[str] = frozenset({"book", "login", "pay"})


def action_risk_rank(action: str) -> int:
    """0 for ``browse`` ... 4 for ``pay``. Raises on an unknown action (fail closed)."""
    try:
        return ACTION_RISK_ORDER.index(action)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"unknown action {action!r}; the frozen ontology is {ACTION_RISK_ORDER}"
        ) from exc


# ======================================================================================
# Verdicts and decisions
# ======================================================================================
class Verdict(str, Enum):
    """The four RQ3 outcome categories.

    ``UNVERIFIED`` and ``CONTRADICTED`` are deliberately distinct. "I found no authority" and
    "I found authority pointing the other way" carry different information: the first is an
    evidence gap that better retrieval could close, the second is a positive detection. Collapsing
    them would make the contradiction-edge ablation unmeasurable.
    """

    VERIFIED = "VERIFIED"
    PLAUSIBLE = "PLAUSIBLE"
    UNVERIFIED = "UNVERIFIED"
    CONTRADICTED = "CONTRADICTED"


VERDICT_VALUES: tuple[str, ...] = tuple(v.value for v in Verdict)


class Disposition(str, Enum):
    """What the reader does with a candidate link once the verdict is known."""

    PRESENT = "present"
    WITHHOLD = "withhold"


@dataclass(frozen=True)
class DefenseDecision:
    """One defense's decision about one ``(entity, domain, action)`` triple.

    ``probability`` is required (not optional) because CONTRACT.md Section 11 mandates
    "calibration error and Brier score for authorization probability". A defense that has no
    natural probability must state a degenerate one and say so via ``calibrated``; that is more
    honest than omitting the field and silently dropping out of the calibration table.
    """

    entity_id: str
    domain_id: str
    action: str
    verdict: Verdict
    probability: float
    disposition: Disposition
    confidence: str = "medium"
    reasons: tuple[str, ...] = ()
    calibrated: bool = False
    #: Whether this decision asserts the domain is the entity's OFFICIAL site, as opposed to merely
    #: authorized for the action. The two are different claims and ``OSMR`` measures only the first.
    #:
    #: Conflating them was a real bug: scoring every VERIFIED link as an official claim charged
    #: AegisLink an OSMR of 0.43 for correctly presenting authorized booking providers, which are
    #: not claiming to be anyone's official site. A defense with no notion of officialness leaves
    #: this ``False`` throughout, and its OSMR is then reported as null rather than as zero -- an
    #: undefined metric, not a perfect score.
    asserts_official: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(
                f"probability {self.probability!r} outside [0, 1] for "
                f"({self.entity_id}, {self.domain_id}, {self.action})"
            )
        if self.confidence not in ("low", "medium", "high"):
            raise ValueError(f"confidence must be low/medium/high, got {self.confidence!r}")

    @property
    def presented(self) -> bool:
        return self.disposition is Disposition.PRESENT

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "domain_id": self.domain_id,
            "action": self.action,
            "verdict": self.verdict.value,
            "probability": round(self.probability, 6),
            "disposition": self.disposition.value,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "calibrated": self.calibrated,
            "asserts_official": self.asserts_official,
        }


# ======================================================================================
# Risk thresholds
# ======================================================================================
class ThresholdOrderingError(ValueError):
    """Raised when a threshold vector violates the mandated risk ordering. Fails closed."""


@dataclass(frozen=True)
class RiskThresholds:
    """Per-action authorization thresholds, monotone in action risk.

    ``tau`` is the bar for :attr:`Verdict.VERIFIED`, and it rises with action risk.
    ``plausible_floor`` is the single bar for "there is *some* support", and it does not.

    That asymmetry is the design. ``PLAUSIBLE`` answers "is there any support at all?", which is a
    question about the evidence and therefore risk-independent; ``VERIFIED`` answers "is there
    enough support to act at this risk level?", which is not. An earlier version derived the lower
    bar as a fixed fraction of each ``tau``, which made the ``PLAUSIBLE`` floor for ``pay`` higher
    than the ceiling an unauthorized domain can reach -- so every unauthorized triple collapsed to
    ``UNVERIFIED`` at high risk and the four-way distinction silently became three-way.

    A single floor also keeps the fit honest: one free vector plus one scalar, so the fitting
    procedure cannot trade the two bars against each other and quietly relax the ordering
    constraint.
    """

    tau: Mapping[str, float]
    plausible_floor: float = 0.20
    #: When ``None``, per-action thresholds apply. When set, ONE threshold is used for every
    #: action -- the ``ablation_shared_threshold`` and ``ablation_no_action_type`` configuration.
    shared_tau: float | None = None
    fit_regime: str = "unfitted_design_default"
    fit_note: str = ""

    def __post_init__(self) -> None:
        missing = [a for a in ACTION_RISK_ORDER if a not in self.tau]
        if missing:
            raise ThresholdOrderingError(f"threshold vector missing actions {missing}")
        for a, v in self.tau.items():
            if not 0.0 < float(v) < 1.0:
                raise ThresholdOrderingError(f"tau_{a}={v!r} must lie strictly inside (0, 1)")
        if self.shared_tau is None:
            self.assert_monotonic()
        lowest_tau = (
            float(self.shared_tau)
            if self.shared_tau is not None
            else min(float(v) for v in self.tau.values())
        )
        if not 0.0 < self.plausible_floor < lowest_tau:
            raise ThresholdOrderingError(
                f"plausible_floor={self.plausible_floor!r} must lie strictly between 0 and the "
                f"lowest tau ({lowest_tau}); otherwise PLAUSIBLE either subsumes VERIFIED or "
                f"becomes unreachable"
            )

    # -- ordering --------------------------------------------------------------------
    def assert_monotonic(self) -> "RiskThresholds":
        """Enforce ``tau_browse < tau_contact < tau_book < tau_login < tau_pay``.

        Strict inequality, as written in the contract. Equal thresholds would make two adjacent
        risk levels indistinguishable and would let a degenerate "one threshold" vector pass as a
        risk-aware one.
        """
        vals = [(a, float(self.tau[a])) for a in ACTION_RISK_ORDER]
        for (lo_a, lo), (hi_a, hi) in zip(vals, vals[1:]):
            if not lo < hi:
                raise ThresholdOrderingError(
                    f"risk ordering violated: tau_{lo_a}={lo} is not strictly less than "
                    f"tau_{hi_a}={hi}. CONTRACT.md Section 3 (RQ3) mandates "
                    f"tau_browse < tau_contact < tau_book < tau_login < tau_pay; "
                    f"preregistration.yaml requires rejection rather than clipping."
                )
        return self

    @property
    def is_monotonic(self) -> bool:
        try:
            self.assert_monotonic()
        except ThresholdOrderingError:
            return False
        return True

    # -- lookups ---------------------------------------------------------------------
    def verified_tau(self, action: str) -> float:
        if self.shared_tau is not None:
            return float(self.shared_tau)
        action_risk_rank(action)  # validate
        return float(self.tau[action])

    def plausible_tau(self, action: str) -> float:
        """The "some support exists" bar. Risk-independent by design; see the class docstring."""
        action_risk_rank(action)  # validate
        return float(self.plausible_floor)

    # -- constructors ----------------------------------------------------------------
    @classmethod
    def design_default(cls) -> "RiskThresholds":
        """The pre-fit design vector.

        Spaced evenly across the risk ladder rather than tuned, so the *shape* of the policy is
        fixed by design and only the scale is fitted later. Values are the starting point for
        :meth:`fitted`, never a result.
        """
        return cls(
            tau={"browse": 0.30, "contact": 0.40, "book": 0.55, "login": 0.70, "pay": 0.80},
            fit_regime="unfitted_design_default",
            fit_note=(
                "Evenly spaced design prior. Fitting adjusts the scale on train_development "
                "only; the ordering constraint is never relaxed."
            ),
        )

    @classmethod
    def fitted(
        cls,
        tau: Mapping[str, float],
        *,
        regime: str,
        note: str = "",
        plausible_floor: float = 0.20,
    ) -> "RiskThresholds":
        """Build a fitted vector, rejecting any ordering violation.

        Parameters
        ----------
        regime
            The split the vector was fitted on. ``test``, ``transfer_holdout`` and
            ``adaptive_holdout`` are refused here as well as by the admission controller, because
            a threshold fitted on an evaluation split invalidates the primary endpoint whether or
            not a ``RunSpec`` was declared.
        """
        forbidden = {"test", "transfer_holdout", "adaptive_holdout"}
        if regime in forbidden:
            raise ThresholdOrderingError(
                f"thresholds may not be fitted on regime {regime!r} "
                f"(preregistration.yaml threshold_selection_constraint; "
                f"web_rag.experiment_matrix rule R3_no_test_in_threshold_selection)"
            )
        return cls(
            tau=dict(tau), plausible_floor=plausible_floor, fit_regime=regime, fit_note=note
        )

    @classmethod
    def shared(cls, value: float, *, note: str = "") -> "RiskThresholds":
        """One threshold for all five actions -- the ablated output policy.

        The per-action vector is kept (filled with ``value``) so the object still serialises the
        same way; ``shared_tau`` is what actually governs lookups, and its presence is what
        ``assert_monotonic`` is deliberately not applied to.
        """
        return cls(
            tau={a: value for a in ACTION_RISK_ORDER},
            shared_tau=value,
            plausible_floor=min(0.20, value * 0.5),
            fit_regime="ablation_shared_threshold",
            fit_note=note or "single global threshold; risk ordering intentionally destroyed",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "tau": {a: round(float(self.tau[a]), 6) for a in ACTION_RISK_ORDER},
            "tau_plausible": {a: round(self.plausible_tau(a), 6) for a in ACTION_RISK_ORDER},
            "plausible_floor": self.plausible_floor,
            "shared_tau": self.shared_tau,
            "is_monotonic": self.is_monotonic,
            "ordering_constraint": "tau_browse < tau_contact < tau_book < tau_login < tau_pay",
            "fit_regime": self.fit_regime,
            "fit_note": self.fit_note,
        }


# ======================================================================================
# Evidence model
# ======================================================================================
class EvidenceFamily(str, Enum):
    """The seven evidence families required by CONTRACT.md Section 3 (RQ3).

    ``IDENTITY_CONSISTENCY`` covers the contract's "name, address, phone, entity ID" as one
    family, matching the way the contract lists it.
    """

    OFFICIAL_REGISTRY = "official_registry"
    OFFICIAL_BACKLINK = "official_backlink"
    IDENTITY_CONSISTENCY = "identity_consistency"
    DOMAIN_LIFECYCLE = "domain_lifecycle"
    SOURCE_DEPENDENCY = "source_dependency"
    ACTION_SPECIFIC = "action_specific"
    CONTRADICTION = "contradiction"


EVIDENCE_FAMILIES: tuple[str, ...] = tuple(f.value for f in EvidenceFamily)


class Polarity(str, Enum):
    SUPPORTS = "supports"
    REFUTES = "refutes"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class EvidenceItem:
    """One extracted, attributable piece of evidence.

    ``source_doc_id`` records *where the evidence was read*, which is what makes the evidence
    graph auditable: a claim with no source doc is an inference, not an observation.
    ``actions`` is empty for action-independent evidence and populated for action-scoped evidence,
    which is how a directory's browse/contact-only grant stays distinguishable from a booking
    provider's payment grant.
    """

    family: EvidenceFamily
    polarity: Polarity
    detail: str
    source_doc_id: str | None = None
    source_domain_id: str | None = None
    actions: tuple[str, ...] = ()
    weight: float = 1.0

    def applies_to_action(self, action: str) -> bool:
        return not self.actions or action in self.actions

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family.value,
            "polarity": self.polarity.value,
            "detail": self.detail,
            "source_doc_id": self.source_doc_id,
            "source_domain_id": self.source_domain_id,
            "actions": list(self.actions),
            "weight": round(self.weight, 6),
        }


class ContradictionKind(str, Enum):
    """The contradiction edges AegisLink can draw.

    Each is a *positive* detection with a named mechanism, not a low score. ``FORGED_BACKLINK`` is
    the load-bearing one: a page asserting the official site lists it, where the official site --
    read directly -- does not. That asymmetry is unavailable to the attacker, who controls its own
    bytes and nothing else.
    """

    OFFICIAL_CLAIM_CONFLICT = "official_claim_conflict"
    FORGED_BACKLINK = "forged_backlink"
    IDENTITY_FIELD_CONFLICT = "identity_field_conflict"
    LIFECYCLE_WITHDRAWAL = "lifecycle_withdrawal"
    ACTION_SCOPE_OVERREACH = "action_scope_overreach"


@dataclass(frozen=True)
class ContradictionEdge:
    kind: ContradictionKind
    detail: str
    source_doc_id: str | None = None
    #: True when the edge alone is sufficient to return CONTRADICTED. Non-decisive edges still
    #: lower the score; making decisiveness explicit keeps the output policy readable.
    decisive: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "detail": self.detail,
            "source_doc_id": self.source_doc_id,
            "decisive": self.decisive,
        }


@dataclass(frozen=True)
class SourceCluster:
    """A set of candidate sources judged mutually dependent.

    Dependence is the point: ``n_effective_sources`` counts clusters, not pages, so producing
    five lexically diverse pages that all originate from one campaign buys one unit of
    corroboration rather than five. Section 7's ``corroborating_sources`` factor (1/3/5) is
    exactly the attack this defeats.
    """

    cluster_id: str
    domain_ids: tuple[str, ...]
    reason: str
    similarity: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "domain_ids": list(self.domain_ids),
            "reason": self.reason,
            "similarity": round(self.similarity, 4),
        }


@dataclass
class EvidenceGraph:
    """Evidence about one entity, assembled once per query and reused for all five actions.

    Nodes are the entity, the candidate domains and the source documents; edges are the evidence
    items, the contradiction edges and the dependency clusters. Materialising it per *entity*
    rather than per triple matters: the official page is read once, and the source clustering sees
    the whole candidate set at once, which is what clustering needs to be meaningful.
    """

    entity_id: str
    #: domain_id -> evidence about that domain
    items: dict[str, list[EvidenceItem]] = field(default_factory=dict)
    #: domain_id -> contradiction edges against that domain
    contradictions: dict[str, list[ContradictionEdge]] = field(default_factory=dict)
    clusters: tuple[SourceCluster, ...] = ()
    #: domain_id -> cluster_id
    cluster_of: dict[str, str] = field(default_factory=dict)
    #: The registry-listed official domain, if the entity has one.
    official_domain_id: str | None = None
    #: True when the official page was actually read (backlink evidence is only available then).
    official_page_read: bool = False
    official_doc_id: str | None = None
    #: domain_id -> actions the official page publishes for it (directional backlink evidence).
    published_delegations: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: domain_id -> actions the *page itself* claims. Claims, not grants.
    claimed_actions: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: Free-text notes for the audit trail.
    notes: tuple[str, ...] = ()

    def add(self, domain_id: str, item: EvidenceItem) -> None:
        self.items.setdefault(domain_id, []).append(item)

    def add_contradiction(self, domain_id: str, edge: ContradictionEdge) -> None:
        self.contradictions.setdefault(domain_id, []).append(edge)

    def for_domain(self, domain_id: str) -> list[EvidenceItem]:
        return self.items.get(domain_id, [])

    def contradictions_for(self, domain_id: str) -> list[ContradictionEdge]:
        return self.contradictions.get(domain_id, [])

    def families_present(self, domain_id: str) -> set[str]:
        fams = {i.family.value for i in self.for_domain(domain_id)}
        if self.contradictions_for(domain_id):
            fams.add(EvidenceFamily.CONTRADICTION.value)
        return fams

    def n_effective_sources(self, domain_ids: Iterable[str]) -> int:
        """Number of *independent* clusters spanned by ``domain_ids``.

        Unclustered domains count individually; that is the conservative direction (it can only
        raise the effective count, never suppress a genuine independent source).
        """
        seen: set[str] = set()
        for d in domain_ids:
            seen.add(self.cluster_of.get(d, f"singleton::{d}"))
        return len(seen)

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "official_domain_id": self.official_domain_id,
            "official_page_read": self.official_page_read,
            "official_doc_id": self.official_doc_id,
            "published_delegations": {
                k: list(v) for k, v in sorted(self.published_delegations.items())
            },
            "claimed_actions": {k: list(v) for k, v in sorted(self.claimed_actions.items())},
            "items": {
                k: [i.as_dict() for i in v] for k, v in sorted(self.items.items())
            },
            "contradictions": {
                k: [c.as_dict() for c in v] for k, v in sorted(self.contradictions.items())
            },
            "clusters": [c.as_dict() for c in self.clusters],
            "n_clusters": len(self.clusters),
            "notes": list(self.notes),
        }


# ======================================================================================
# Verification result
# ======================================================================================
@dataclass(frozen=True)
class VerificationResult:
    """The full, auditable answer to ``verify(e, d, a)``."""

    entity_id: str
    domain_id: str
    action: str
    verdict: Verdict
    probability: float
    score_logit: float
    tau_verified: float
    tau_plausible: float
    disposition: Disposition
    features: Mapping[str, float]
    evidence: tuple[EvidenceItem, ...]
    contradictions: tuple[ContradictionEdge, ...]
    families_used: tuple[str, ...]
    n_effective_sources: int
    n_raw_sources: int
    reasons: tuple[str, ...]
    config_id: str

    @property
    def confidence(self) -> str:
        """Coarse confidence band, used for the ``CMR`` metric (unwarned high-confidence errors).

        Anchored on distance from the operative threshold rather than on the raw probability, so
        "high confidence" means "far from the decision boundary the policy actually used".
        """
        margin = abs(self.probability - self.tau_verified)
        if self.verdict is Verdict.CONTRADICTED:
            return "high"
        if margin >= 0.25:
            return "high"
        if margin >= 0.10:
            return "medium"
        return "low"

    def to_decision(self) -> DefenseDecision:
        # An official claim is made only when the decision rests on registry ownership. Presenting
        # an authorized booking provider is a claim about authorization, not about officialness.
        asserts_official = bool(
            self.disposition is Disposition.PRESENT
            and float(self.features.get("registry_official", 0.0)) >= 1.0
        )
        return DefenseDecision(
            entity_id=self.entity_id,
            domain_id=self.domain_id,
            action=self.action,
            verdict=self.verdict,
            probability=self.probability,
            disposition=self.disposition,
            confidence=self.confidence,
            reasons=self.reasons,
            calibrated=True,
            asserts_official=asserts_official,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "domain_id": self.domain_id,
            "action": self.action,
            "verdict": self.verdict.value,
            "probability": round(self.probability, 6),
            "score_logit": round(self.score_logit, 6),
            "tau_verified": round(self.tau_verified, 6),
            "tau_plausible": round(self.tau_plausible, 6),
            "disposition": self.disposition.value,
            "confidence": self.confidence,
            "features": {k: round(float(v), 6) for k, v in sorted(self.features.items())},
            "evidence": [e.as_dict() for e in self.evidence],
            "contradictions": [c.as_dict() for c in self.contradictions],
            "families_used": list(self.families_used),
            "n_effective_sources": self.n_effective_sources,
            "n_raw_sources": self.n_raw_sources,
            "reasons": list(self.reasons),
            "config_id": self.config_id,
        }


# ======================================================================================
# The shared evidence channel
# ======================================================================================
@dataclass(frozen=True)
class CandidateView:
    """What every defense may see about one retrieved candidate. ``READER`` tier."""

    rank: int
    doc_id: str
    url: str
    domain: str
    domain_id: str
    title: str
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "doc_id": self.doc_id,
            "url": self.url,
            "domain": self.domain,
            "domain_id": self.domain_id,
            "title": self.title,
            "score": self.score,
            "bm25_rank": self.bm25_rank,
            "dense_rank": self.dense_rank,
        }


@dataclass
class VerificationContext:
    """The single evidence channel handed to every defense. Constructed once per query.

    Nothing on this object is above ``VERIFIER`` tier, and :meth:`assert_firewall_clean` proves
    it. In particular there is no oracle, no page role, no ``adversarial`` flag and no delegation
    table: third-party authorization must be *established*, never looked up.

    ``fetch`` returns the visible text of a document at ``read_phase``. It is the two-phase
    payoff: the retriever ranked ``indexed`` bytes, the verifier reads ``live`` bytes, so a page
    that turned adversarial after indexing is seen in its adversarial form.
    """

    query_id: str
    entity_id: str
    action: str
    query_text: str
    candidates: tuple[CandidateView, ...]
    fetch: Callable[[str], str]
    registry: PublicDomainRegistry
    official_registry: OfficialRegistry
    #: Outbound hyperlinks of a document. ``READER`` tier: ``CrawledDocument.public_view`` already
    #: publishes ``outbound_links`` at ``RETRIEVER`` tier, because a crawler plainly sees the anchors
    #: on a page it fetched.
    #:
    #: This channel is not a convenience. Visible text never contains a hostname -- the corpus names
    #: partners by display name and puts the host only in the ``href`` -- so any "does page A
    #: reference domain B?" test written against visible text silently returns False. That defect
    #: made B10's corroboration layer fail on 100% of authorized third parties and quietly
    #: reduced B05, B06 and B07 to self-votes.
    fetch_links: Callable[[str], Sequence[str]] | None = None
    site_index: SiteIndex | None = None
    read_phase: str = "live"
    replay_fingerprint: str | None = None
    #: Caches so N defenses over the same query do not re-read or re-shingle the same page N
    #: times. Passed in by the driver as *shared* dicts, which is what makes the saving real: a
    #: per-context cache would be discarded between defenses.
    _text_cache: dict[str, str] = field(default_factory=dict, repr=False)
    _shingle_cache: dict[tuple[str, int], frozenset[Any]] = field(
        default_factory=dict, repr=False
    )
    _link_cache: dict[str, tuple[str, ...]] = field(default_factory=dict, repr=False)

    # -- reads -----------------------------------------------------------------------
    def text(self, doc_id: str) -> str:
        """Visible text of ``doc_id`` at :attr:`read_phase`, memoised."""
        cached = self._text_cache.get(doc_id)
        if cached is None:
            cached = self.fetch(doc_id)
            self._text_cache[doc_id] = cached
        return cached

    def shingle_set(self, doc_id: str, n: int = 5) -> frozenset[Any]:
        """Memoised word-``n``-gram set of a document, for near-duplicate detection."""
        key = (doc_id, n)
        cached = self._shingle_cache.get(key)
        if cached is None:
            cached = shingles(self.text(doc_id), n)
            self._shingle_cache[key] = cached
        return cached

    def links(self, doc_id: str) -> tuple[str, ...]:
        """Outbound hyperlink targets of ``doc_id`` at :attr:`read_phase`, memoised.

        Empty when no link fetcher was supplied, which keeps hand-built test contexts usable. A
        defense that needs link structure should treat an empty result as "no links observed", not
        as "no links exist".
        """
        if self.fetch_links is None:
            return ()
        cached = self._link_cache.get(doc_id)
        if cached is None:
            cached = tuple(self.fetch_links(doc_id))
            self._link_cache[doc_id] = cached
        return cached

    def references_domain(self, doc_id: str, domain_id: str) -> bool:
        """Whether ``doc_id`` links to, or names, the host behind ``domain_id``.

        Checks the outbound links first, because a hyperlink is an unambiguous reference. Falls back
        to the hostname appearing in visible text, which happens only in a page's own footer, so the
        fallback is mainly a self-reference test.

        Display-name matching is deliberately NOT used here. Every entity's official domain has the
        entity's own name as its display name, so matching on display name would make every page
        that mentions the business "reference" its official domain and would inflate corroboration
        for exactly the domain that needs no corroborating.
        """
        host = None
        view = self.registry.domains.get(domain_id)
        if view is not None:
            host = view.domain
        else:
            cand = self.candidate_by_domain(domain_id)
            host = cand.domain if cand else None
        if not host:
            return False
        for target in self.links(doc_id):
            if host in target:
                return True
        return host in self.text(doc_id)

    def candidate_domain_ids(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for c in self.candidates:
            seen.setdefault(c.domain_id, None)
        return tuple(seen)

    def candidate_by_domain(self, domain_id: str) -> CandidateView | None:
        for c in self.candidates:
            if c.domain_id == domain_id:
                return c
        return None

    def lifecycle(self, domain_id: str) -> LifecycleObservation | None:
        """Derived, crawl-observable lifecycle facts. ``VERIFIER`` tier."""
        return self.registry.lifecycle.get(domain_id)

    def hostname(self, domain_id: str) -> str:
        view = self.registry.domains.get(domain_id)
        if view is not None:
            return view.domain
        cand = self.candidate_by_domain(domain_id)
        return cand.domain if cand else domain_id

    def display_name(self, domain_id: str) -> str:
        view = self.registry.domains.get(domain_id)
        return view.display_name if view is not None else self.hostname(domain_id)

    def is_shared_host(self, domain_id: str) -> bool:
        """Whether the registry marks the host as serving multiple listings."""
        view = self.registry.domains.get(domain_id)
        return bool(view is not None and view.shared)

    # -- firewall --------------------------------------------------------------------
    def assert_firewall_clean(self, *, location: str = "VerificationContext") -> None:
        """Prove no experimenter-only field or label value is reachable from this context.

        Checked on the serialisable projection: ``fetch`` is a closure over corpus bytes and page
        prose legitimately contains words like "official", so free text is exempted from *value*
        checking exactly as :func:`web_rag.exposure.assert_no_leakage` documents. Field-name
        checking is never relaxed.
        """
        assert_no_leakage(self.describe(), location=location, allow_free_text=True)

    def describe(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "entity_id": self.entity_id,
            "action": self.action,
            "query_text": self.query_text,
            "read_phase": self.read_phase,
            "replay_fingerprint": self.replay_fingerprint,
            "candidates": [c.as_dict() for c in self.candidates],
            "registry": self.registry.as_dict(ExposureTier.VERIFIER),
            "official_registry": self.official_registry.as_dict(),
            "site_index": None if self.site_index is None else self.site_index.as_dict(),
        }

    # -- constructor -----------------------------------------------------------------
    @classmethod
    def from_retrieval_result(
        cls,
        result: RetrievalResult,
        fetch: Callable[[str], str],
        *,
        registry: PublicDomainRegistry,
        official_registry: OfficialRegistry,
        fetch_links: Callable[[str], Sequence[str]] | None = None,
        site_index: SiteIndex | None = None,
        read_phase: str = "live",
        replay_fingerprint: str | None = None,
        text_cache: dict[str, str] | None = None,
        shingle_cache: dict[tuple[str, int], frozenset[Any]] | None = None,
        link_cache: dict[str, tuple[str, ...]] | None = None,
    ) -> "VerificationContext":
        return cls(
            query_id=result.query_id,
            entity_id=result.entity_id,
            action=result.action,
            query_text=result.query_text,
            candidates=tuple(
                CandidateView(
                    rank=c.rank,
                    doc_id=c.doc_id,
                    url=c.url,
                    domain=c.domain,
                    domain_id=c.domain_id,
                    title=c.title,
                    score=c.score,
                    bm25_rank=c.bm25_rank,
                    dense_rank=c.dense_rank,
                )
                for c in result.candidates
            ),
            fetch=fetch,
            registry=registry,
            official_registry=official_registry,
            fetch_links=fetch_links,
            site_index=site_index,
            read_phase=read_phase,
            replay_fingerprint=replay_fingerprint,
            _text_cache=text_cache if text_cache is not None else {},
            _shingle_cache=shingle_cache if shingle_cache is not None else {},
            _link_cache=link_cache if link_cache is not None else {},
        )


# ======================================================================================
# Defense protocol
# ======================================================================================
class InputKind(str, Enum):
    """Evidence inputs a defense may or may not consult.

    Exists so "identical evidence unless the design explicitly prohibits an input type"
    (CONTRACT.md Section 10) is a declaration rather than a comment. Every defense declares its
    restrictions; the evaluation driver writes them to the deviation log.
    """

    URL_LEXICAL = "url_lexical"
    PAGE_TEXT = "page_text"
    REGISTRY_OFFICIAL = "registry_official"
    LIFECYCLE = "lifecycle"
    OFFICIAL_BACKLINK_DIRECTIONAL = "official_backlink_directional"
    SOURCE_CLUSTERING = "source_clustering"
    ACTION_TYPE = "action_type"
    RETRIEVAL_RANK = "retrieval_rank"


@dataclass(frozen=True)
class InputRestriction:
    """One logged deviation from the identical-evidence rule."""

    kind: InputKind
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"input": self.kind.value, "reason": self.reason}


@runtime_checkable
class Defense(Protocol):
    """What AegisLink, every ablation and every baseline implement.

    One method. A defense is handed the shared context and asked about one triple; everything
    else -- graph construction, caching, ordering -- is its own business.
    """

    defense_id: str

    def decide(
        self, entity_id: str, domain_id: str, action: str, ctx: VerificationContext
    ) -> DefenseDecision: ...


@dataclass
class DefenseMetadata:
    """Registry entry for one defense, for the results tables and the deviation log."""

    defense_id: str
    family: str
    label: str
    contract_ref: str
    restrictions: tuple[InputRestriction, ...] = ()
    notes: str = ""
    is_surrogate: bool = False
    surrogate_note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "defense_id": self.defense_id,
            "family": self.family,
            "label": self.label,
            "contract_ref": self.contract_ref,
            "restrictions": [r.as_dict() for r in self.restrictions],
            "notes": self.notes,
            "is_surrogate": self.is_surrogate,
            "surrogate_note": self.surrogate_note,
        }


# ======================================================================================
# Small numeric helpers shared by the verifier and the learned baselines
# ======================================================================================
def sigmoid(x: float) -> float:
    """Numerically stable logistic. Clamped so a huge logit cannot produce exactly 0.0 or 1.0.

    An exact 0 or 1 would make the Brier/calibration analysis of CONTRACT.md Section 11 degenerate
    (infinite log loss on a single error), so the output is kept strictly inside the open interval.
    """
    if x >= 0:
        p = 1.0 / (1.0 + math.exp(-min(x, 60.0)))
    else:
        e = math.exp(max(x, -60.0))
        p = e / (1.0 + e)
    return min(max(p, 1e-6), 1.0 - 1e-6)


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class PlattScaling:
    """Post-hoc probability calibration: ``p' = sigmoid(a * logit(p) + b)``.

    Two parameters, fitted by deterministic grid search on a development split only. Platt scaling
    is used rather than isotonic regression because with one free monotone transform the ranking
    -- and therefore every threshold comparison -- is preserved exactly, so calibration cannot be
    confused with a change in the decision rule.
    """

    a: float = 1.0
    b: float = 0.0
    fit_regime: str = "identity"
    n_fit: int = 0
    brier_before: float | None = None
    brier_after: float | None = None

    def apply(self, p: float) -> float:
        return sigmoid(self.a * logit(p) + self.b)

    @property
    def is_identity(self) -> bool:
        return self.a == 1.0 and self.b == 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "a": round(self.a, 6),
            "b": round(self.b, 6),
            "fit_regime": self.fit_regime,
            "n_fit": self.n_fit,
            "brier_before": None if self.brier_before is None else round(self.brier_before, 6),
            "brier_after": None if self.brier_after is None else round(self.brier_after, 6),
            "is_identity": self.is_identity,
        }


def brier_score(pairs: Sequence[tuple[float, bool]]) -> float:
    """Mean squared error of probabilistic predictions. ``nan``-free on empty input."""
    if not pairs:
        return float("nan")
    return sum((p - (1.0 if y else 0.0)) ** 2 for p, y in pairs) / len(pairs)


def expected_calibration_error(pairs: Sequence[tuple[float, bool]], n_bins: int = 10) -> float:
    """Equal-width binned ECE. Bins with no mass contribute nothing."""
    if not pairs:
        return float("nan")
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for p, y in pairs:
        idx = min(n_bins - 1, int(p * n_bins))
        bins[idx].append((p, y))
    total = len(pairs)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        conf = sum(p for p, _ in b) / len(b)
        acc = sum(1.0 for _, y in b if y) / len(b)
        ece += (len(b) / total) * abs(conf - acc)
    return ece


def fit_platt(
    pairs: Sequence[tuple[float, bool]],
    *,
    regime: str,
    a_grid: Sequence[float] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0),
    b_grid: Sequence[float] = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0),
) -> PlattScaling:
    """Deterministic grid-search Platt fit minimising the Brier score.

    A grid rather than gradient descent so the result is bit-reproducible without pinning an
    optimiser's implementation details -- Section 15 requires a clean machine to reproduce the
    tables exactly.
    """
    if not pairs:
        return PlattScaling(fit_regime=f"{regime}::empty")
    before = brier_score(pairs)
    best = (float("inf"), 1.0, 0.0)
    for a in a_grid:
        for b in b_grid:
            cal = PlattScaling(a=a, b=b)
            s = brier_score([(cal.apply(p), y) for p, y in pairs])
            if s < best[0] - 1e-12:
                best = (s, a, b)
    _, a, b = best
    fitted = PlattScaling(
        a=a, b=b, fit_regime=regime, n_fit=len(pairs), brier_before=before, brier_after=best[0]
    )
    return fitted


# ======================================================================================
# Text utilities shared by evidence extraction and by the baselines
# ======================================================================================
_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def word_tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def shingles(text: str, n: int = 5) -> frozenset[tuple[str, ...]]:
    """Word ``n``-gram set, for near-duplicate detection between corroborating pages."""
    toks = word_tokens(text)
    if len(toks) < n:
        return frozenset({tuple(toks)}) if toks else frozenset()
    return frozenset(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))


def jaccard(a: frozenset[Any], b: frozenset[Any]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def slugify_name(name: str) -> str:
    """``"Harbour Lantern Bistro"`` -> ``"harbour-lantern-bistro"``."""
    return "-".join(word_tokens(name))


def describe_framework() -> dict[str, Any]:
    """Provenance block for the results artifacts."""
    return {
        "framework_version": FRAMEWORK_VERSION,
        "contract_ref": "CONTRACT.md Sections 3 (RQ3), 9, 10",
        "verdicts": list(VERDICT_VALUES),
        "action_risk_order": list(ACTION_RISK_ORDER),
        "high_risk_actions": sorted(HIGH_RISK_ACTIONS),
        "evidence_families": list(EVIDENCE_FAMILIES),
        "contradiction_kinds": [k.value for k in ContradictionKind],
        "threshold_policy": {
            "ordering": "tau_browse < tau_contact < tau_book < tau_login < tau_pay",
            "enforcement": (
                "RiskThresholds.__post_init__ calls assert_monotonic; a violating vector raises "
                "ThresholdOrderingError. preregistration.yaml requires rejection, not clipping."
            ),
            "fitting_constraint": (
                "RiskThresholds.fitted refuses regime in {test, transfer_holdout, "
                "adaptive_holdout}, independently of the admission controller."
            ),
        },
        "identical_evidence_rule": (
            "One VerificationContext per query is passed unchanged to every defense. A defense "
            "that by design ignores an input declares an InputRestriction, which the driver "
            "writes to the deviation log (CONTRACT.md Section 10)."
        ),
        "probability_requirement": (
            "Every DefenseDecision carries a probability so Section 11's Brier score and "
            "calibration error are computable for every defense, including rule-based ones."
        ),
    }


__all__ = [
    "ACTION_RISK_ORDER",
    "EVIDENCE_FAMILIES",
    "FRAMEWORK_VERSION",
    "HIGH_RISK_ACTIONS",
    "VERDICT_VALUES",
    "CandidateView",
    "ContradictionEdge",
    "ContradictionKind",
    "Defense",
    "DefenseDecision",
    "DefenseMetadata",
    "Disposition",
    "EvidenceFamily",
    "EvidenceGraph",
    "EvidenceItem",
    "InputKind",
    "InputRestriction",
    "PlattScaling",
    "Polarity",
    "RiskThresholds",
    "SourceCluster",
    "ThresholdOrderingError",
    "VerificationContext",
    "VerificationResult",
    "Verdict",
    "action_risk_rank",
    "brier_score",
    "describe_framework",
    "expected_calibration_error",
    "fit_platt",
    "jaccard",
    "logit",
    "normalize",
    "shingles",
    "sigmoid",
    "slugify_name",
    "word_tokens",
]
