"""AegisLink: action-aware authorization verification.

CONTRACT.md Sections 3 (RQ3), 9.

The claim this implements
------------------------
Whether a domain may perform an action for an entity is not a property of the domain. It is a
property of the *triple*, and it is established by evidence that the domain cannot manufacture.
The corpus is built so that exactly one such asymmetry exists, and AegisLink is built around it:

    An attacker controls its own page bytes and nothing else.

Concretely: an authorized booking provider's page says "Listed by the official site: <Entity>" and
links to the official host. An impersonating page says the *same words* and links to the *same*
host -- the generator emits both lines from one template precisely so no local lexical feature
separates them (``site_generator/generator.py``, ``official_backlink`` factor). What separates them
is direction. Only the official page can publish::

    appointed booking partner for browse, contact, book, pay: Cinderpost Reserve

So AegisLink reads the *official* page and asks whether it names this domain for this action. A
forged inbound link is worthless; a genuine outbound delegation is decisive.

The six mandated components (Section 9)
---------------------------------------
1. :class:`EntityResolver`          -- which entity is the query about, and does this page mean it?
2. :class:`ActionExtractor`         -- what does this page claim it can do?
3. :class:`EvidenceGraphBuilder`    -- assemble all seven evidence families into one graph.
4. :class:`SourceDependencyClusterer` -- collapse mutually dependent corroborators.
5. :class:`AuthorizationInference`  -- action-specific, graph-gated, calibrated probability.
6. :class:`RiskAwareOutputPolicy`   -- monotone thresholds map probability to verdict and action.

Why this is not a weighted sum of URL reputation signals
--------------------------------------------------------
Section 9 forbids that explicitly. Two structural properties make the difference, and both are
testable:

* **Gating, not addition.** Authority comes from registry ownership or a published delegation.
  No amount of corroboration, page polish or identity consistency can substitute for it: the
  features that lack authority enter a *capped* branch that cannot reach ``tau_book`` and above.
  ``test_corroboration_cannot_manufacture_authority`` asserts this.
* **Contradiction is a detection, not a low score.** A forged backlink yields
  :attr:`~aegislink.framework.Verdict.CONTRADICTED` through a named edge, so the verdict survives
  any reweighting of the continuous features.

Two deliberate abstentions
--------------------------
Both are shortcuts the corpus would reward and that would inflate the result:

* **Prompt-injection markers are not a feature.** Roughly half the attack templates carry an
  injected instruction (Section 7 ``prompt_injection``), so "page contains an instruction to the
  assistant" is a strong classifier of *that factor level* rather than of authorization. AegisLink
  is immune to injection by construction -- it follows no instructions found in retrieved text --
  but it does not score their presence. ``results/evidence_family_identifiability.json`` records
  what the shortcut would have been worth.
* **Lifecycle presence is not a feature.** In this corpus a lifecycle record exists for exactly
  the 160 takeover domains, so the mere *presence* of one perfectly predicts "unauthorized"
  (audited in the same file). Using it would be reading a label. The lifecycle family is therefore
  wired so it can only ever *withdraw* authority that another family granted -- see
  :meth:`EvidenceGraphBuilder._add_lifecycle_evidence`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from aegislink.framework import (
    ACTION_RISK_ORDER,
    CandidateView,
    ContradictionEdge,
    ContradictionKind,
    DefenseDecision,
    Disposition,
    EvidenceFamily,
    EvidenceGraph,
    EvidenceItem,
    PlattScaling,
    Polarity,
    RiskThresholds,
    SourceCluster,
    VerificationContext,
    VerificationResult,
    Verdict,
    action_risk_rank,
    jaccard,
    logit,
    normalize,
    shingles,
    sigmoid,
    slugify_name,
    word_tokens,
)

VERIFIER_VERSION = "1.0"


class ReplayFingerprintError(RuntimeError):
    """Raised when a defense is run against a replay it was not bound to. Fails closed.

    CONTRACT.md Section 6 freezes retrieval so that ten baselines and the defense are compared on
    identical evidence. A silent mismatch would confound a measured difference between two
    defenses with a difference in what they were shown, so the mismatch is fatal rather than
    warned.
    """


# ======================================================================================
# Extraction patterns
# ======================================================================================
#: The official page's delegation block. This is the only genuine grant statement in the corpus.
#: Shape: ``appointed booking partner for browse, contact, book, pay: Cinderpost Reserve``
#:
#: ``partner`` is captured inside a *lookahead* so the match consumes nothing past the colon. An
#: earlier consuming version swallowed the following delegation line whenever two partners were
#: published back to back, silently dropping the directory grant on 160/176 official pages and
#: capping ATPR. The partner name is then resolved by longest-display-name prefix rather than by
#: guessing where the name ends, because the surrounding text differs across all five page
#: structures (FAQ headings, list items, narrative prose).
_DELEGATION_RE = re.compile(
    r"appointed\s+(?P<kind>booking|directory)\s+partner\s+for\s+"
    r"(?P<actions>[a-z,\s]+?)\s*:\s*(?=(?P<partner>.{0,90}))",
    re.IGNORECASE,
)

#: A page's own assertion that the official site lists it. Emitted identically by genuine partners
#: and by forgers -- which is the point. Presence is a *claim*; direction decides.
_INBOUND_BACKLINK_CLAIM_RE = re.compile(
    r"listed by the official site|appointed by|partner of", re.IGNORECASE
)

#: A page's assertion that it *is itself* the official presence.
#:
#: Must be a *self* predication. An earlier version matched the bare phrase "official site", which
#: also occurs in "Listed by the official site: <Entity>" -- a line every authorized partner page
#: carries -- and in "Official site: <Entity>" on benign directory pages. That false positive
#: contradicted every legitimate third party and cut recall on authorized links to 0.54, so the
#: patterns below are anchored on the four self-claim shapes in ``registry/lexicon.py``
#: ``OFFICIAL_CLAIM_PHRASES`` and deliberately exclude the pointing-at-someone-else forms.
#:
#: Detecting the claim does not distinguish honest from dishonest: the generator draws genuine
#: official pages and impersonators from the SAME phrase bank (``site_generator/generator.py``
#: lines 579 and 630). Only the registry and the backlink direction separate them.
_OFFICIAL_SELF_CLAIM_RE = re.compile(
    r"this is the official (?:website|site|page)\b"
    r"|official home page for\b"
    r"|official site and booking desk"
    r"|you have reached the official"
    r"|official online presence of",
    re.IGNORECASE,
)

#: ``<Display Name> acts for <Entity> on: browse, contact`` -- a third party's own scope claim.
_ACTS_FOR_RE = re.compile(
    r"(?P<who>[^.]{2,80}?)\s+acts\s+for\s+(?P<entity>[^:]{2,120}?)\s+on\s*:\s*"
    r"(?P<actions>[a-z,\s]+?)\s*\.",
    re.IGNORECASE,
)

#: Action cues, used for the claimed-action extraction. Deliberately the same vocabulary the
#: generator draws its action-claim phrases from, so extraction is not a hidden capability test.
_ACTION_CUE_RE: dict[str, re.Pattern[str]] = {
    "browse": re.compile(
        r"\bbrowse\b|opening hours|\bmenu\b|\bgallery\b|read (?:about|more)|"
        r"services and visiting", re.IGNORECASE
    ),
    "contact": re.compile(
        r"\bcontact\b|telephone|\bphone\b|enquir|get in touch", re.IGNORECASE
    ),
    "book": re.compile(
        r"\bbook\b|\bbooking\b|reserv|\btable\b|availability", re.IGNORECASE
    ),
    "login": re.compile(
        r"\blog ?in\b|\bsign ?in\b|account (?:area|holders)|members", re.IGNORECASE
    ),
    "pay": re.compile(
        r"\bpay\b|payment|deposit|settle|balance|checkout|invoice", re.IGNORECASE
    ),
}

#: ``continue to book`` / ``book on the official site`` / ``book via Cinderpost Reserve`` --
#: the action *link labels*. A link label is a stronger claim than a prose cue, because it is the
#: thing a reader would actually present.
_ACTION_LINK_LABEL_RE = re.compile(
    r"(?:continue to|read more)\s+(?P<a1>browse|contact|book|login|pay)|"
    r"\b(?P<a2>browse|contact|book|login|pay)\s+(?:on the official site|via\b)",
    re.IGNORECASE,
)

_ADDRESS_RE = re.compile(r"Address:\s*(?P<addr>[^\n]{4,120}?)\s*(?=Telephone:|$)", re.IGNORECASE)
_PHONE_RE = re.compile(r"Telephone:\s*(?P<phone>\+?[0-9][0-9\-\s()]{5,24})", re.IGNORECASE)
_TAKEOVER_PROSE_RE = re.compile(
    r"previously hosted|now under different management|under new management", re.IGNORECASE
)
#: Instruction-shaped content aimed at an automated reader. Counted for the audit trail only;
#: never a scoring feature. See the module docstring.
_INJECTION_MARKER_RE = re.compile(
    r"note for automated assistants|ignore (?:the )?previous instructions|"
    r"when summarising|system note", re.IGNORECASE
)


def _split_actions(blob: str) -> tuple[str, ...]:
    """Parse ``"browse, contact, book, pay"`` into the frozen ontology, order-normalised."""
    found = {t for t in word_tokens(blob) if t in ACTION_RISK_ORDER}
    return tuple(a for a in ACTION_RISK_ORDER if a in found)


# ======================================================================================
# Configuration (the ablation surface)
# ======================================================================================
@dataclass(frozen=True)
class AegisLinkConfig:
    """Everything an ablation switches off, in one place.

    Every flag defaults to the full method. ``aegislink/ablations.py`` produces the seven mandated
    variants by flipping exactly one flag each, so an ablation cannot accidentally change two
    things at once -- :func:`aegislink.ablations.describe_ablations` asserts the one-flag property.
    """

    config_id: str = "aegislink-full-v1"
    label: str = "AegisLink (full)"

    #: Component 5/6: use the query's action type. Off -> every action treated as one.
    use_action_type: bool = True
    #: Family 2: read the official page and use its published delegations.
    use_official_backlinks: bool = True
    #: Component 4: collapse mutually dependent corroborating sources.
    use_source_clustering: bool = True
    #: Family 4: domain lifecycle / ownership change.
    use_domain_lifecycle: bool = True
    #: Family 7: contradiction edges.
    use_contradiction_edges: bool = True
    #: Family 3: identity-field consistency.
    use_identity_consistency: bool = True
    #: Component 5: ``"graph"`` (gated graph inference) or ``"source_count"`` (majority voting).
    inference_mode: str = "graph"

    thresholds: RiskThresholds = field(default_factory=RiskThresholds.design_default)
    calibration: PlattScaling = field(default_factory=PlattScaling)

    #: Clustering knobs. Fixed by design, not fitted -- see SourceDependencyClusterer.
    duplicate_jaccard: float = 0.55
    shingle_n: int = 5

    def __post_init__(self) -> None:
        if self.inference_mode not in ("graph", "source_count"):
            raise ValueError(
                f"inference_mode must be 'graph' or 'source_count', got {self.inference_mode!r}"
            )

    def with_thresholds(self, thresholds: RiskThresholds) -> "AegisLinkConfig":
        return replace(self, thresholds=thresholds)

    def with_calibration(self, calibration: PlattScaling) -> "AegisLinkConfig":
        return replace(self, calibration=calibration)

    def flag_state(self) -> dict[str, Any]:
        """The ablation-relevant switches only, for the one-flag-per-ablation assertion."""
        return {
            "use_action_type": self.use_action_type,
            "use_official_backlinks": self.use_official_backlinks,
            "use_source_clustering": self.use_source_clustering,
            "use_domain_lifecycle": self.use_domain_lifecycle,
            "use_contradiction_edges": self.use_contradiction_edges,
            "use_identity_consistency": self.use_identity_consistency,
            "inference_mode": self.inference_mode,
            "shared_threshold": self.thresholds.shared_tau is not None,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "label": self.label,
            **self.flag_state(),
            "thresholds": self.thresholds.as_dict(),
            "calibration": self.calibration.as_dict(),
            "duplicate_jaccard": self.duplicate_jaccard,
            "shingle_n": self.shingle_n,
        }

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]


# ======================================================================================
# Component 1 -- entity resolution
# ======================================================================================
@dataclass
class EntityResolver:
    """Resolve the query to an entity, and decide which candidate pages are about it.

    Resolution is longest-surface-form matching over the page's visible text, using the registry's
    published aliases. It is genuinely fallible on the corpus's confusable sibling pairs (``Harbour
    Lantern Bistro`` vs ``Harbour Mere Bistro``), which is required: a resolution stage that could
    never fail would make RQ2's stage attribution vacuous.

    The machine-readable ``x-aegis-entity-id`` meta channel would make this trivial. It is
    ``EVALUATOR`` tier and is not in the visible text the verifier reads, by design
    (``web_rag/exposure.py`` ``meta_channel_policy``).
    """

    resolver_id: str = "aegislink-entity-resolver-v1"
    #: ``doc_id -> (longest match length, entity ids achieving it)``. Shared across defenses so
    #: 18 configurations over 1,008 queries scan each page once rather than 18 times.
    #:
    #: Correctness rests on one assumption, stated because it is load-bearing: a ``doc_id``
    #: identifies immutable bytes at a fixed ``read_phase``. That holds for the frozen corpus, where
    #: a document slot is a file. It does NOT hold for hand-built contexts that reuse a doc id with
    #: different text, so tests must mint fresh doc ids per context -- ``make_context`` in
    #: ``tests/test_aegislink_and_baselines.py`` does.
    _doc_best: dict[str, tuple[int, frozenset[str]]] = field(default_factory=dict, repr=False)
    _forms: tuple[tuple[str, str], ...] | None = field(default=None, repr=False)

    def surface_forms(self, ctx: VerificationContext, entity_id: str) -> tuple[str, ...]:
        return ctx.official_registry.surface_forms(entity_id)

    def _forms_index(self, ctx: VerificationContext) -> tuple[tuple[str, str], ...]:
        """``(normalised surface form, entity_id)`` over every registry entity, longest first."""
        if self._forms is None:
            pairs: list[tuple[str, str]] = []
            for eid in sorted(ctx.official_registry.records):
                for f in self.surface_forms(ctx, eid):
                    nf = normalize(f)
                    if nf:
                        pairs.append((nf, eid))
            self._forms = tuple(sorted(pairs, key=lambda t: (-len(t[0]), t[0], t[1])))
        return self._forms

    def mentions(self, text: str, forms: Sequence[str]) -> int:
        """Length of the longest published surface form present in ``text``; 0 if none."""
        low = normalize(text)
        best = 0
        for f in forms:
            fl = normalize(f)
            if fl and fl in low:
                best = max(best, len(fl))
        return best

    def _best_matches(
        self, ctx: VerificationContext, doc_id: str
    ) -> tuple[int, frozenset[str]]:
        """Longest surface-form match on the page, and every entity achieving that length.

        One pass over the whole alias index per document. The result is what both
        :meth:`resolve` and :meth:`page_mentions_entity` need, so it is computed once and cached.
        """
        cached = self._doc_best.get(doc_id)
        if cached is not None:
            return cached
        low = normalize(ctx.text(doc_id))
        best_len = 0
        winners: set[str] = set()
        for form, eid in self._forms_index(ctx):
            if len(form) < best_len:
                break  # sorted longest-first: nothing further can win
            if form in low:
                if len(form) > best_len:
                    best_len, winners = len(form), {eid}
                else:
                    winners.add(eid)
        result = (best_len, frozenset(winners))
        self._doc_best[doc_id] = result
        return result

    def resolve(self, ctx: VerificationContext) -> list[str]:
        """Resolve the query to entity ids by voting over the top candidates.

        Votes over the top three candidates and takes the strongest match, exactly as an undefended
        reader would; the defense's advantage is meant to come from authorization reasoning, not
        from a better name matcher. Comparing all registry entities (not just the query's) is what
        allows a confusable sibling to win, i.e. what allows this stage to be wrong -- a resolution
        stage that could never fail would make RQ2's attribution vacuous.
        """
        votes: dict[str, int] = {}
        order: list[str] = []
        for cand in ctx.candidates[:3]:
            best_len, winners = self._best_matches(ctx, cand.doc_id)
            if not best_len or not winners:
                continue
            eid = min(winners)  # deterministic tie-break
            votes[eid] = votes.get(eid, 0) + 1
            order.append(eid)
        if not votes:
            return []
        top = max(votes.items(), key=lambda kv: (kv[1], -order.index(kv[0])))
        return [top[0]]

    def page_mentions_entity(
        self, ctx: VerificationContext, doc_id: str, entity_id: str
    ) -> bool:
        """Whether the page names the target entity, and no other entity more specifically."""
        best_len, winners = self._best_matches(ctx, doc_id)
        return bool(best_len) and entity_id in winners


# ======================================================================================
# Component 2 -- action extraction
# ======================================================================================
@dataclass
class ActionExtractor:
    """Extract which actions a page *claims* it can perform.

    A claim is not a grant. This component exists so the action-specific evidence family has
    something to test the grant against: an unauthorized payment portal claims ``pay``, and the
    question is whether anything authoritative agrees.

    Link labels are separated from prose cues because they differ in strength. ``continue to pay``
    is an offer of a payment route; the word "balance" appearing in prose is not.
    """

    extractor_id: str = "aegislink-action-extractor-v1"

    def claimed_actions(self, text: str) -> tuple[str, ...]:
        """Actions offered as a route (link labels). The strong form of the claim."""
        found: set[str] = set()
        for m in _ACTION_LINK_LABEL_RE.finditer(text):
            found.add((m.group("a1") or m.group("a2")).lower())
        # A third party's own scope statement is also an explicit route claim.
        for m in _ACTS_FOR_RE.finditer(text):
            found.update(_split_actions(m.group("actions")))
        return tuple(a for a in ACTION_RISK_ORDER if a in found)

    def cued_actions(self, text: str) -> tuple[str, ...]:
        """Actions merely alluded to in prose. The weak form."""
        return tuple(a for a in ACTION_RISK_ORDER if _ACTION_CUE_RE[a].search(text))

    def claims_action(self, text: str, action: str) -> bool:
        return action in self.claimed_actions(text) or bool(_ACTION_CUE_RE[action].search(text))


# ======================================================================================
# Component 4 -- source-dependency clustering
# ======================================================================================
@dataclass
class SourceDependencyClusterer:
    """Group candidate sources that are not independent.

    Three dependence signals, combined by union-find. All three are crawl-observable and none of
    them consults a label:

    1. **Near-duplicate content.** Word-5-gram Jaccard above ``duplicate_jaccard``. Catches the
       ``lexical_diversity=low`` corroboration arm directly.
    2. **Shared referent in the hostname.** Hosts whose name embeds the *same entity slug* are
       registrations made for one listing. A pre-existing independent publisher's hostname does not
       contain the business it reviews. Catches the ``lexical_diversity=high`` arm, where the prose
       is varied but the registrations are still one campaign.
    3. **Single-listing scope.** A host the crawl only ever sees discussing one entity is not an
       independent directory. Used as a corroborating signal for (2), never alone, because a
       legitimately small publisher would also look like this.

    Thresholds are design constants, not fitted parameters. Fitting them on the corpus would let
    the clusterer memorise the corroborator naming scheme, which would be a benchmark artifact
    rather than a mechanism.
    """

    duplicate_jaccard: float = 0.55
    shingle_n: int = 5
    clusterer_id: str = "aegislink-source-clusterer-v1"

    def cluster(
        self, ctx: VerificationContext, entity_id: str, domain_ids: Sequence[str]
    ) -> tuple[tuple[SourceCluster, ...], dict[str, str]]:
        """Return ``(clusters, domain_id -> cluster_id)``. Singletons are omitted from clusters."""
        parent: dict[str, str] = {d: d for d in domain_ids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        reasons: dict[frozenset[str], tuple[str, float]] = {}

        # -- signal 1: near-duplicate visible text ------------------------------------
        cand_by_domain = {d: ctx.candidate_by_domain(d) for d in domain_ids}
        sh: dict[str, frozenset[Any]] = {}
        for d in domain_ids:
            c = cand_by_domain[d]
            sh[d] = ctx.shingle_set(c.doc_id, self.shingle_n) if c else frozenset()
        ids = list(domain_ids)
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                if not sh[a] or not sh[b]:
                    continue
                j = jaccard(sh[a], sh[b])
                if j >= self.duplicate_jaccard:
                    union(a, b)
                    reasons[frozenset({a, b})] = ("near_duplicate_content", j)

        # -- signal 2+3: hostname embeds the entity slug, host serves one listing ------
        rec = ctx.official_registry.get(entity_id)
        slug = slugify_name(rec.canonical_name) if rec else ""
        official_id = rec.official_domain_id if rec else None
        if slug:
            # The generator truncates the entity slug when composing corroborator hostnames, so
            # match on a prefix rather than the whole slug. 12 characters is long enough that a
            # collision between two different businesses is not plausible, and short enough to
            # survive truncation.
            probe = slug[:12]
            embedded = [
                d
                for d in domain_ids
                # The official domain also embeds the slug -- it is the entity's own name. It is
                # excluded because it is identified by the registry, not by its spelling.
                if d != official_id and probe and probe in ctx.hostname(d)
                # A host the registry marks as serving many listings is a real shared provider.
                and not ctx.is_shared_host(d)
            ]
            for i, a in enumerate(embedded):
                for b in embedded[i + 1 :]:
                    union(a, b)
                    reasons.setdefault(
                        frozenset({a, b}), ("shared_referent_in_hostname", 0.0)
                    )

        groups: dict[str, list[str]] = {}
        for d in domain_ids:
            groups.setdefault(find(d), []).append(d)

        clusters: list[SourceCluster] = []
        cluster_of: dict[str, str] = {}
        for i, (root, members) in enumerate(sorted(groups.items())):
            if len(members) < 2:
                continue
            cid = f"SC{i:03d}"
            why, sim = "co_registered_campaign", 0.0
            for pair, (r, s) in reasons.items():
                if pair <= set(members):
                    why, sim = r, max(sim, s)
                    break
            clusters.append(
                SourceCluster(
                    cluster_id=cid, domain_ids=tuple(sorted(members)), reason=why, similarity=sim
                )
            )
            for m in members:
                cluster_of[m] = cid
        return tuple(clusters), cluster_of


# ======================================================================================
# Component 3 -- evidence graph construction
# ======================================================================================
@dataclass
class EvidenceGraphBuilder:
    """Assemble all seven evidence families for one entity into one :class:`EvidenceGraph`."""

    config: AegisLinkConfig
    resolver: EntityResolver
    extractor: ActionExtractor
    clusterer: SourceDependencyClusterer
    builder_id: str = "aegislink-graph-builder-v1"

    # -- official page ---------------------------------------------------------------
    def _locate_official_doc(
        self, ctx: VerificationContext, official_domain_id: str
    ) -> str | None:
        """Find a document slot on the entity's official host.

        Prefers a retrieved candidate. Falls back to the crawl-observable
        :class:`~web_rag.exposure.SiteIndex`, because RQ3's backlink family requires the verifier
        to be able to consult the official site even when retrieval did not surface it -- a
        verifier that knows the official hostname can fetch it, and pretending otherwise would cap
        ATPR on a retrieval artifact rather than on the defense.
        """
        cand = ctx.candidate_by_domain(official_domain_id)
        if cand is not None:
            return cand.doc_id
        if ctx.site_index is None:
            return None
        slots = ctx.site_index.doc_ids_for_domain(official_domain_id)
        return slots[0] if slots else None

    def _read_published_delegations(
        self, ctx: VerificationContext, official_text: str
    ) -> dict[str, tuple[str, ...]]:
        """Parse the official page's delegation block into ``domain_id -> actions``.

        Partners are named on the page by display name, so they are resolved through the public
        registry's ``display_name`` index, longest name first. Ambiguous display names are dropped
        rather than guessed: 176 of the 190 display names in the registry are shared by several
        domains (every entity's own name is the display name of its official site *and* of the five
        adversarial hosts built around it), so guessing would hand authority to an impersonator.
        The ten domains that ever appear in a delegation all have unique display names, verified by
        ``workflow/diagnostics/check_extraction_assumptions.py``.
        """
        names = self._display_name_index(ctx)
        out: dict[str, tuple[str, ...]] = {}
        for m in _DELEGATION_RE.finditer(official_text):
            actions = _split_actions(m.group("actions"))
            if not actions:
                continue
            tail = normalize(m.group("partner"))
            did = next((d for name, d in names if tail.startswith(name)), None)
            if did is None:
                continue
            out[did] = tuple(
                a for a in ACTION_RISK_ORDER if a in set(out.get(did, ())) | set(actions)
            )
        return out

    def _display_name_index(
        self, ctx: VerificationContext
    ) -> tuple[tuple[str, str], ...]:
        """``(normalised display name, domain_id)`` for unambiguous names, longest first.

        Cached on the builder: the registry holds 1,790 domains and the index is identical for
        every query, so rebuilding it per query would dominate the runtime.
        """
        cached = getattr(self, "_name_index_cache", None)
        if cached is not None:
            return cached
        counts: dict[str, list[str]] = {}
        for did, view in ctx.registry.domains.items():
            counts.setdefault(normalize(view.display_name), []).append(did)
        index = tuple(
            sorted(
                ((name, ids[0]) for name, ids in counts.items() if len(ids) == 1 and name),
                key=lambda t: (-len(t[0]), t[0]),
            )
        )
        self._name_index_cache = index
        return index

    # -- per-family evidence ---------------------------------------------------------
    def _add_registry_evidence(
        self, graph: EvidenceGraph, ctx: VerificationContext, domain_id: str
    ) -> None:
        if graph.official_domain_id is None:
            graph.add(
                domain_id,
                EvidenceItem(
                    family=EvidenceFamily.OFFICIAL_REGISTRY,
                    polarity=Polarity.NEUTRAL,
                    detail="entity has no authoritative registry record (fabricated control)",
                ),
            )
            return
        if domain_id == graph.official_domain_id:
            graph.add(
                domain_id,
                EvidenceItem(
                    family=EvidenceFamily.OFFICIAL_REGISTRY,
                    polarity=Polarity.SUPPORTS,
                    detail=f"authoritative registry names {domain_id} as the entity's own domain",
                    source_domain_id=domain_id,
                    weight=1.0,
                ),
            )
        else:
            graph.add(
                domain_id,
                EvidenceItem(
                    family=EvidenceFamily.OFFICIAL_REGISTRY,
                    polarity=Polarity.NEUTRAL,
                    detail=(
                        f"registry names {graph.official_domain_id} as the official domain, so "
                        f"{domain_id} needs a published delegation to hold any authority"
                    ),
                    source_domain_id=graph.official_domain_id,
                ),
            )

    def _add_backlink_evidence(
        self, graph: EvidenceGraph, ctx: VerificationContext, domain_id: str
    ) -> None:
        """Directional backlink evidence, plus the forged-backlink contradiction."""
        if not self.config.use_official_backlinks:
            return
        granted = graph.published_delegations.get(domain_id, ())
        cand = ctx.candidate_by_domain(domain_id)
        page_text = ctx.text(cand.doc_id) if cand else ""
        asserts_backlink = bool(_INBOUND_BACKLINK_CLAIM_RE.search(page_text))

        if granted:
            graph.add(
                domain_id,
                EvidenceItem(
                    family=EvidenceFamily.OFFICIAL_BACKLINK,
                    polarity=Polarity.SUPPORTS,
                    detail=(
                        f"the entity's official page publishes {domain_id} as an appointed "
                        f"partner for {', '.join(granted)}"
                    ),
                    source_doc_id=graph.official_doc_id,
                    source_domain_id=graph.official_domain_id,
                    actions=granted,
                    weight=1.0,
                ),
            )
            return

        if asserts_backlink and graph.official_page_read and domain_id != graph.official_domain_id:
            # The decisive asymmetry: the page says the official site lists it; the official page,
            # read directly, does not.
            graph.add(
                domain_id,
                EvidenceItem(
                    family=EvidenceFamily.OFFICIAL_BACKLINK,
                    polarity=Polarity.REFUTES,
                    detail=(
                        "page asserts the official site lists it, but the official page "
                        "publishes no delegation naming this domain"
                    ),
                    source_doc_id=graph.official_doc_id,
                    source_domain_id=graph.official_domain_id,
                    weight=1.0,
                ),
            )
            if self.config.use_contradiction_edges:
                graph.add_contradiction(
                    domain_id,
                    ContradictionEdge(
                        kind=ContradictionKind.FORGED_BACKLINK,
                        detail=(
                            "inbound backlink claim unsupported by the official site's own "
                            "outbound delegations (direction, not presence, is what authorises)"
                        ),
                        source_doc_id=graph.official_doc_id,
                        decisive=True,
                    ),
                )

    def _add_official_claim_contradiction(
        self, graph: EvidenceGraph, ctx: VerificationContext, domain_id: str
    ) -> None:
        """A domain claiming to be the official presence when the registry names another one."""
        cand = ctx.candidate_by_domain(domain_id)
        if cand is None:
            return
        text = ctx.text(cand.doc_id)
        if not _OFFICIAL_SELF_CLAIM_RE.search(text):
            return
        if graph.official_domain_id is None or domain_id == graph.official_domain_id:
            return
        if not self.resolver.page_mentions_entity(ctx, cand.doc_id, graph.entity_id):
            return
        graph.add(
            domain_id,
            EvidenceItem(
                family=EvidenceFamily.CONTRADICTION,
                polarity=Polarity.REFUTES,
                detail=(
                    f"page claims to be the official presence, but the registry names "
                    f"{graph.official_domain_id}"
                ),
                source_doc_id=cand.doc_id,
                source_domain_id=domain_id,
                weight=1.0,
            ),
        )
        if self.config.use_contradiction_edges:
            graph.add_contradiction(
                domain_id,
                ContradictionEdge(
                    kind=ContradictionKind.OFFICIAL_CLAIM_CONFLICT,
                    detail=(
                        f"self-declared official presence conflicts with the authoritative "
                        f"registry record ({graph.official_domain_id})"
                    ),
                    source_doc_id=cand.doc_id,
                    decisive=True,
                ),
            )

    def _add_identity_evidence(
        self, graph: EvidenceGraph, ctx: VerificationContext, domain_id: str
    ) -> None:
        """Compare page-published name/address/phone against the authoritative record."""
        if not self.config.use_identity_consistency:
            return
        cand = ctx.candidate_by_domain(domain_id)
        rec = ctx.official_registry.get(graph.entity_id)
        if cand is None or rec is None:
            return
        text = ctx.text(cand.doc_id)

        fields_checked, fields_agree, mismatches = 0, 0, []
        if rec.address:
            m = _ADDRESS_RE.search(text)
            if m:
                fields_checked += 1
                if normalize(m.group("addr")) == normalize(rec.address):
                    fields_agree += 1
                else:
                    mismatches.append("address")
        if rec.phone:
            m = _PHONE_RE.search(text)
            if m:
                fields_checked += 1
                if normalize(m.group("phone")) == normalize(rec.phone):
                    fields_agree += 1
                else:
                    mismatches.append("phone")

        if not fields_checked:
            graph.add(
                domain_id,
                EvidenceItem(
                    family=EvidenceFamily.IDENTITY_CONSISTENCY,
                    polarity=Polarity.NEUTRAL,
                    detail="page publishes no address or telephone to compare",
                    source_doc_id=cand.doc_id,
                    source_domain_id=domain_id,
                ),
            )
            return

        if mismatches:
            graph.add(
                domain_id,
                EvidenceItem(
                    family=EvidenceFamily.IDENTITY_CONSISTENCY,
                    polarity=Polarity.REFUTES,
                    detail=f"published {'/'.join(mismatches)} disagrees with the registry record",
                    source_doc_id=cand.doc_id,
                    source_domain_id=domain_id,
                    weight=float(len(mismatches)) / max(1, fields_checked),
                ),
            )
            if self.config.use_contradiction_edges:
                graph.add_contradiction(
                    domain_id,
                    ContradictionEdge(
                        kind=ContradictionKind.IDENTITY_FIELD_CONFLICT,
                        detail=(
                            f"identity fields {'/'.join(mismatches)} conflict with the "
                            f"authoritative record"
                        ),
                        source_doc_id=cand.doc_id,
                        # Not decisive on its own: a stale phone on a genuine partner page is a
                        # data-quality problem, not proof of misbinding. It lowers the score.
                        decisive=False,
                    ),
                )
        else:
            graph.add(
                domain_id,
                EvidenceItem(
                    family=EvidenceFamily.IDENTITY_CONSISTENCY,
                    polarity=Polarity.SUPPORTS,
                    detail=f"all {fields_agree} published identity field(s) match the registry",
                    source_doc_id=cand.doc_id,
                    source_domain_id=domain_id,
                    weight=float(fields_agree) / max(1, fields_checked),
                ),
            )

    def _add_lifecycle_evidence(
        self, graph: EvidenceGraph, ctx: VerificationContext, domain_id: str
    ) -> None:
        """Lifecycle evidence, deliberately restricted to *withdrawal*.

        In this corpus a lifecycle record exists for exactly the takeover domains, so a rule of
        the form "has a lifecycle record -> reject" would be a perfect classifier and a disguised
        label read (audited in ``results/evidence_family_identifiability.json``). The family is
        therefore wired so it can only remove authority that another family granted: an ownership
        change after a grant invalidates *that grant*, and where there is no grant there is
        nothing for it to say.

        The consequence is honest and expected: because takeover domains hold no current
        delegation, this family changes almost no verdict on this corpus, and
        ``ablation_no_domain_lifecycle`` will show almost no effect. That is a property of the
        corpus (no benign domain ever changes hands), not of the mechanism, and the fix belongs in
        a future corpus revision that adds benign ownership-change controls.
        """
        if not self.config.use_domain_lifecycle:
            return
        obs = ctx.lifecycle(domain_id)
        if obs is None:
            return
        granted = graph.published_delegations.get(domain_id, ())
        is_official = domain_id == graph.official_domain_id
        if not granted and not is_official:
            # No authority to withdraw. Recorded as neutral so the audit trail shows the family
            # was consulted and deliberately declined to contribute.
            graph.add(
                domain_id,
                EvidenceItem(
                    family=EvidenceFamily.DOMAIN_LIFECYCLE,
                    polarity=Polarity.NEUTRAL,
                    detail=(
                        "lifecycle observed but no current authority to withdraw; presence of a "
                        "lifecycle record is deliberately not treated as evidence of misbinding"
                    ),
                    source_domain_id=domain_id,
                ),
            )
            return
        if obs.ownership_changed or obs.expired:
            what = []
            if obs.expired:
                what.append(f"expired at {obs.expired_at_snapshot}")
            if obs.ownership_changed:
                what.append(f"changed hands at {obs.ownership_change_at_snapshot}")
            graph.add(
                domain_id,
                EvidenceItem(
                    family=EvidenceFamily.DOMAIN_LIFECYCLE,
                    polarity=Polarity.REFUTES,
                    detail=(
                        f"domain {'; '.join(what)}; a grant does not survive a change of "
                        f"registrant"
                    ),
                    source_domain_id=domain_id,
                    weight=1.0,
                ),
            )
            if self.config.use_contradiction_edges:
                graph.add_contradiction(
                    domain_id,
                    ContradictionEdge(
                        kind=ContradictionKind.LIFECYCLE_WITHDRAWAL,
                        detail=(
                            f"authority withdrawn for {domain_id}: {'; '.join(what)}"
                        ),
                        decisive=True,
                    ),
                )

    def _add_action_specific_evidence(
        self, graph: EvidenceGraph, ctx: VerificationContext, domain_id: str
    ) -> None:
        """Claimed-vs-granted action scope, and the overreach contradiction.

        This is where action relativity becomes a mechanism rather than a label. A directory whose
        published delegation covers ``browse, contact`` and whose page offers ``pay`` is
        overreaching *for pay only*, and stays fully authorized for browse.
        """
        if not self.config.use_action_type:
            # With action type stripped there is no scope to overreach: any grant is read as
            # covering any action, so this family has nothing to say. Firing it anyway would make
            # ablation_no_action_type block the very over-grants it is meant to commit, and the
            # ablation would appear to lose nothing.
            return
        cand = ctx.candidate_by_domain(domain_id)
        if cand is None:
            return
        claimed = graph.claimed_actions.get(domain_id, ())
        granted = graph.published_delegations.get(domain_id, ())
        if domain_id == graph.official_domain_id:
            return
        if not granted or not claimed:
            return
        over = tuple(a for a in claimed if a not in granted)
        if not over:
            return
        graph.add(
            domain_id,
            EvidenceItem(
                family=EvidenceFamily.ACTION_SPECIFIC,
                polarity=Polarity.REFUTES,
                detail=(
                    f"page offers {', '.join(over)} but the published delegation covers only "
                    f"{', '.join(granted)}"
                ),
                source_doc_id=cand.doc_id,
                source_domain_id=domain_id,
                actions=over,
                weight=1.0,
            ),
        )
        if self.config.use_contradiction_edges:
            graph.add_contradiction(
                domain_id,
                ContradictionEdge(
                    kind=ContradictionKind.ACTION_SCOPE_OVERREACH,
                    detail=(
                        f"claimed actions {', '.join(over)} exceed the delegated scope "
                        f"{', '.join(granted)}"
                    ),
                    source_doc_id=cand.doc_id,
                    # Scoped to the overreaching actions; the inference layer applies it per action.
                    decisive=False,
                ),
            )

    def _add_source_dependency_evidence(
        self, graph: EvidenceGraph, ctx: VerificationContext, domain_id: str
    ) -> None:
        """Record how much *independent* corroboration this domain actually has."""
        corroborators = [
            d
            for d in ctx.candidate_domain_ids()
            if d != domain_id and self._mentions_target(ctx, graph.entity_id, d)
        ]
        raw = len(corroborators)
        eff = (
            graph.n_effective_sources(corroborators)
            if self.config.use_source_clustering
            else raw
        )
        graph.add(
            domain_id,
            EvidenceItem(
                family=EvidenceFamily.SOURCE_DEPENDENCY,
                polarity=Polarity.NEUTRAL,
                detail=(
                    f"{raw} corroborating page(s) reduce to {eff} independent source(s)"
                    if self.config.use_source_clustering
                    else f"{raw} corroborating page(s), independence not assessed"
                ),
                weight=float(eff),
            ),
        )

    def _mentions_target(
        self, ctx: VerificationContext, entity_id: str, domain_id: str
    ) -> bool:
        """Does the candidate page on ``domain_id`` name ``entity_id``? ``False`` if not retrieved."""
        cand = ctx.candidate_by_domain(domain_id)
        if cand is None:
            return False
        return self.resolver.page_mentions_entity(ctx, cand.doc_id, entity_id)

    # -- driver ----------------------------------------------------------------------
    def build(self, ctx: VerificationContext, entity_id: str) -> EvidenceGraph:
        """Build the entity's evidence graph once; all five actions reuse it."""
        rec = ctx.official_registry.get(entity_id)
        graph = EvidenceGraph(
            entity_id=entity_id,
            official_domain_id=rec.official_domain_id if rec else None,
        )
        notes: list[str] = []

        # -- read the official page (the directional backlink source) -----------------
        if graph.official_domain_id and self.config.use_official_backlinks:
            doc_id = self._locate_official_doc(ctx, graph.official_domain_id)
            if doc_id is not None:
                text = ctx.text(doc_id)
                # Self-validate: a slot on the official host that does not name the entity is the
                # wrong slot, and trusting it would import another entity's delegations.
                if self.resolver.page_mentions_entity(ctx, doc_id, entity_id):
                    graph.official_page_read = True
                    graph.official_doc_id = doc_id
                    graph.published_delegations = self._read_published_delegations(ctx, text)
                    notes.append(
                        f"read official page {doc_id} at phase={ctx.read_phase}; "
                        f"{len(graph.published_delegations)} published delegation(s)"
                    )
                else:
                    notes.append(
                        f"located slot {doc_id} on the official host but it does not name "
                        f"{entity_id}; backlink evidence withheld"
                    )
            else:
                notes.append("official host not reachable from candidates or site index")
        elif not self.config.use_official_backlinks:
            notes.append("official backlink family disabled by configuration (ablation)")

        # -- claimed actions per candidate --------------------------------------------
        domain_ids = list(ctx.candidate_domain_ids())
        for d in domain_ids:
            cand = ctx.candidate_by_domain(d)
            if cand is not None:
                graph.claimed_actions[d] = self.extractor.claimed_actions(ctx.text(cand.doc_id))

        # -- clustering ---------------------------------------------------------------
        if self.config.use_source_clustering:
            clusters, cluster_of = self.clusterer.cluster(ctx, entity_id, domain_ids)
            graph.clusters, graph.cluster_of = clusters, cluster_of
            notes.append(
                f"{len(domain_ids)} candidate domains -> "
                f"{graph.n_effective_sources(domain_ids)} independent source(s)"
            )
        else:
            notes.append("source-dependency clustering disabled by configuration (ablation)")

        # -- per-domain families ------------------------------------------------------
        for d in domain_ids:
            self._add_registry_evidence(graph, ctx, d)
            self._add_backlink_evidence(graph, ctx, d)
            self._add_official_claim_contradiction(graph, ctx, d)
            self._add_identity_evidence(graph, ctx, d)
            self._add_lifecycle_evidence(graph, ctx, d)
            self._add_action_specific_evidence(graph, ctx, d)
            self._add_source_dependency_evidence(graph, ctx, d)

        # -- audit-only counters (never scored) ---------------------------------------
        n_inject = 0
        for d in domain_ids:
            cand = ctx.candidate_by_domain(d)
            if cand is not None and _INJECTION_MARKER_RE.search(ctx.text(cand.doc_id)):
                n_inject += 1
        if n_inject:
            notes.append(
                f"{n_inject} candidate page(s) carry instruction-shaped content aimed at an "
                f"automated reader; recorded for audit, deliberately not scored"
            )

        graph.notes = tuple(notes)
        return graph


# ======================================================================================
# Component 5 -- action-specific authorization inference
# ======================================================================================
#: Feature weights. Fixed by design and documented, then Platt-calibrated on train_development.
#:
#: The magnitudes are a design statement, not a fit. Authority-bearing features (registry ownership,
#: a delegation published by the official page for *this* action) are an order of magnitude larger
#: than every corroborative feature, so corroboration can shade a belief but not create one.
#:
#: Enforcement of that claim does not rest on the weights, though. It rests on
#: :attr:`AuthorizationInference.no_authority_ceiling`, a hard cap applied after scoring, which
#: keeps any no-authority triple strictly below the lowest ``tau``. Because the cap is structural,
#: the weights are free to express graded belief *within* the unauthorized region -- which is what
#: makes ``PLAUSIBLE`` a meaningful state rather than an unreachable one -- without ever letting
#: that belief become permission.
FEATURE_WEIGHTS: Mapping[str, float] = {
    "bias": -2.60,
    "registry_official": 6.20,
    "delegated_this_action": 5.60,
    # Standing with the entity, but not for the requested action. Raises belief (the official site
    # does vouch for this domain) while the cap still forbids authorisation. Sized so this cell
    # clears ``plausible_floor`` on its OWN standing, at every action, without needing corroborating
    # pages to push it over: "the official site vouches for this domain, just not for this action"
    # is meaningful evidence however many other pages happen to have been retrieved. It is the cell
    # PLAUSIBLE exists to name, and a verdict nobody can reach is not a verdict.
    #
    # Raising it cannot cause over-granting, because the no-authority ceiling sits below every tau
    # by construction: this weight only moves a triple between UNVERIFIED and PLAUSIBLE, and neither
    # is presented.
    "delegated_other_action_only": 2.00,
    "no_authority": -0.90,
    "identity_consistent": 0.80,
    "identity_conflict": -1.60,
    "log_effective_sources": 0.70,
    "claims_official_without_registry": -2.40,
    "claims_action_without_grant": -1.00,
    "lifecycle_withdrawal": -3.40,
    # Evidence-side risk aversion, distinct from the policy-side tau ladder: riskier actions need
    # more support for the same belief. Stripped by ablation_no_action_type.
    "action_risk": -0.28,
}


@dataclass
class AuthorizationInference:
    """Turn the graph into a calibrated, action-specific authorization probability.

    Two branches, and the split is the substance of Section 9's "not a simple weighted sum":

    * **Authority present** (registry ownership, or a delegation published by the official page
      covering this action). The logistic score runs free and can exceed any threshold.
    * **No authority.** The score is *capped* below ``tau_book`` regardless of every other
      feature, so corroboration, identity polish and lexical respectability cannot add up to
      permission for a high-risk action. This is a structural gate, not a large negative weight,
      so it cannot be undone by reweighting.
    """

    config: AegisLinkConfig
    weights: Mapping[str, float] = field(default_factory=lambda: dict(FEATURE_WEIGHTS))
    inference_id: str = "aegislink-authorization-inference-v1"

    @property
    def no_authority_ceiling(self) -> float:
        """Hard cap on the probability a triple with no authority may reach.

        Derived from the live threshold vector rather than hard-coded, so refitting the thresholds
        cannot silently break the invariant. It is placed inside the ``PLAUSIBLE`` band of the
        *lowest-risk* action, which states the architectural claim exactly:

            a domain with no authority can at best be PLAUSIBLE, and only at the lowest risk
            level -- never VERIFIED, at any action, on any amount of corroboration.

        ``test_no_authority_ceiling_below_every_threshold`` asserts the ordering holds for the
        design default, for a refitted vector and for the shared-threshold ablation.
        """
        thr = self.config.thresholds
        lowest = ACTION_RISK_ORDER[0]
        return 0.5 * (thr.plausible_tau(lowest) + thr.verified_tau(lowest))

    def features(
        self, graph: EvidenceGraph, ctx: VerificationContext, domain_id: str, action: str
    ) -> dict[str, float]:
        """Extract the feature vector for one triple. All features are graph-derived."""
        is_official = domain_id == graph.official_domain_id
        granted = graph.published_delegations.get(domain_id, ())
        # With action type stripped, the system cannot tell one grant from another, so ANY grant
        # is read as covering the requested action. That is the ablation's whole content, and it
        # is what makes it over-grant login and pay.
        delegated_here = bool(granted) if not self.config.use_action_type else action in granted
        delegated_elsewhere = bool(granted) and not delegated_here
        authority = is_official or delegated_here

        items = graph.for_domain(domain_id)
        id_support = any(
            i.family is EvidenceFamily.IDENTITY_CONSISTENCY and i.polarity is Polarity.SUPPORTS
            for i in items
        )
        id_conflict = any(
            i.family is EvidenceFamily.IDENTITY_CONSISTENCY and i.polarity is Polarity.REFUTES
            for i in items
        )
        lifecycle_bad = any(
            i.family is EvidenceFamily.DOMAIN_LIFECYCLE and i.polarity is Polarity.REFUTES
            for i in items
        )
        claims_official_no_registry = any(
            i.family is EvidenceFamily.CONTRADICTION and i.polarity is Polarity.REFUTES
            for i in items
        ) or any(
            c.kind is ContradictionKind.OFFICIAL_CLAIM_CONFLICT
            for c in graph.contradictions_for(domain_id)
        )
        claims_action = action in graph.claimed_actions.get(domain_id, ())

        eff = 1.0
        for i in items:
            if i.family is EvidenceFamily.SOURCE_DEPENDENCY:
                eff = max(1.0, float(i.weight))
                break

        risk = action_risk_rank(action) if self.config.use_action_type else 2

        return {
            "bias": 1.0,
            "registry_official": 1.0 if is_official else 0.0,
            "delegated_this_action": 1.0 if delegated_here else 0.0,
            "delegated_other_action_only": 1.0 if delegated_elsewhere else 0.0,
            "no_authority": 0.0 if authority else 1.0,
            "identity_consistent": 1.0 if id_support else 0.0,
            "identity_conflict": 1.0 if id_conflict else 0.0,
            "log_effective_sources": math.log1p(eff),
            "claims_official_without_registry": 1.0 if claims_official_no_registry else 0.0,
            "claims_action_without_grant": 1.0 if (claims_action and not authority) else 0.0,
            "lifecycle_withdrawal": 1.0 if lifecycle_bad else 0.0,
            "action_risk": float(risk),
        }

    def has_authority(self, graph: EvidenceGraph, domain_id: str, action: str) -> bool:
        if domain_id == graph.official_domain_id:
            return True
        granted = graph.published_delegations.get(domain_id, ())
        if not self.config.use_action_type:
            # Action type stripped: any grant counts for any action. This is the ablation's whole
            # point -- it destroys action relativity and should over-grant login/pay.
            return bool(granted)
        return action in granted

    def score(
        self, graph: EvidenceGraph, ctx: VerificationContext, domain_id: str, action: str
    ) -> tuple[float, float, dict[str, float]]:
        """Return ``(probability, logit, features)``."""
        if self.config.inference_mode == "source_count":
            return self._score_source_count(graph, ctx, domain_id, action)

        feats = self.features(graph, ctx, domain_id, action)
        z = sum(self.weights.get(k, 0.0) * v for k, v in feats.items())
        p = sigmoid(z)
        p = self.config.calibration.apply(p)
        if not self.has_authority(graph, domain_id, action):
            p = min(p, self.no_authority_ceiling)
        return p, z, feats

    def _score_source_count(
        self, graph: EvidenceGraph, ctx: VerificationContext, domain_id: str, action: str
    ) -> tuple[float, float, dict[str, float]]:
        """Ablation: replace graph inference with unweighted source-count voting.

        The rule is the one the ablation names: every retrieved page casts one equally weighted
        vote for whichever candidate it backs, and the plurality winner is authorized. A page backs
        ``domain_id`` when it mentions the entity and either is that domain's own page offering the
        action, or links to that domain.

        Votes are counted from *outbound links* (``ctx.references_domain``), not from hostnames in
        visible text: the corpus names partners by display name and keeps the host in the ``href``,
        so a text-based test would register only self-nominations and the ablation would look like
        an abstention rather than a mis-authorisation.

        No authority, no direction, no clustering. Five Sybil pages therefore outvote one official
        page, which is precisely the failure mode the full method's clustering and gating remove.
        """
        rec = ctx.official_registry.get(graph.entity_id)
        if rec is None:
            return 1e-6, logit(1e-6), {"bias": 1.0, "n_mentioning": 0.0}

        voters = [
            d
            for d in ctx.candidate_domain_ids()
            if (cand := ctx.candidate_by_domain(d)) is not None
            and normalize(rec.canonical_name) in normalize(ctx.text(cand.doc_id))
        ]
        counts: dict[str, int] = {}
        for target in ctx.candidate_domain_ids():
            votes = 0
            for voter in voters:
                if voter == target:
                    if action in graph.claimed_actions.get(voter, ()):
                        votes += 1
                else:
                    cand = ctx.candidate_by_domain(voter)
                    if cand is not None and ctx.references_domain(cand.doc_id, target):
                        votes += 1
            counts[target] = votes

        total = sum(counts.values())
        mine = counts.get(domain_id, 0)
        share = (mine / total) if total else 0.0
        winner = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0] if total else None
        p = (
            min(1.0 - 1e-6, 0.55 + 0.45 * share)
            if (winner == domain_id and mine > 0)
            else max(1e-6, 0.45 * share)
        )
        feats = {
            "bias": 1.0,
            "source_count_share": share,
            "n_votes": float(mine),
            "n_votes_total": float(total),
            "is_plurality_winner": 1.0 if winner == domain_id else 0.0,
        }
        return p, logit(p), feats


# ======================================================================================
# Component 6 -- risk-aware output policy
# ======================================================================================
@dataclass
class RiskAwareOutputPolicy:
    """Map (probability, contradictions, action) to a verdict and a presentation decision.

    Presentation follows from ``VERIFIED`` alone. That is deliberate and it is where the monotone
    thresholds do their work: one and the same evidential strength clears ``tau_browse`` and fails
    ``tau_pay``, so the *same* page is presented as a way to read a menu and withheld as a way to
    submit a payment. Risk awareness lives in the ladder, not in a second presentation rule layered
    on top of it -- which keeps ``ablation_shared_threshold`` a clean single-factor change: flatten
    the ladder and the risk-dependence disappears with it.

    Showing ``PLAUSIBLE`` links was considered and rejected. On this corpus every page that reaches
    ``PLAUSIBLE`` is unauthorized for the action by the Section 5.2 graph, so presenting it would
    manufacture ``UALER`` events in exchange for no recall -- the state is worth *reporting* to a
    user, not acting on.
    """

    config: AegisLinkConfig
    policy_id: str = "aegislink-risk-policy-v1"

    def decide(
        self,
        *,
        probability: float,
        contradictions: Sequence[ContradictionEdge],
        action: str,
    ) -> tuple[Verdict, Disposition, float, float, list[str]]:
        thr = self.config.thresholds
        tau_v = thr.verified_tau(action)
        tau_p = thr.plausible_tau(action)
        reasons: list[str] = []

        decisive = [c for c in contradictions if c.decisive]
        if decisive and self.config.use_contradiction_edges:
            reasons.append(
                "contradicted by: " + "; ".join(sorted({c.kind.value for c in decisive}))
            )
            return Verdict.CONTRADICTED, Disposition.WITHHOLD, tau_v, tau_p, reasons

        if probability >= tau_v:
            reasons.append(f"p={probability:.3f} >= tau_{action}={tau_v:.3f}")
            return Verdict.VERIFIED, Disposition.PRESENT, tau_v, tau_p, reasons

        if probability >= tau_p:
            reasons.append(
                f"p={probability:.3f} in [{tau_p:.3f}, {tau_v:.3f}): some support, not enough "
                f"for {action}; reported, not acted on"
            )
            return Verdict.PLAUSIBLE, Disposition.WITHHOLD, tau_v, tau_p, reasons

        reasons.append(f"p={probability:.3f} < tau_plausible_{action}={tau_p:.3f}")
        return Verdict.UNVERIFIED, Disposition.WITHHOLD, tau_v, tau_p, reasons


# ======================================================================================
# The verifier
# ======================================================================================
@dataclass
class AegisLink:
    """Action-aware authorization verification. ``verify(e, d, a)`` is the contract entry point.

    Bind to a frozen replay by passing ``expected_replay_fingerprint``; every ``verify`` call then
    checks the context against it and raises :class:`ReplayFingerprintError` on drift, so a stale
    snapshot cannot silently mix evidence across defenses.
    """

    config: AegisLinkConfig = field(default_factory=AegisLinkConfig)
    expected_replay_fingerprint: str | None = None
    resolver: EntityResolver = field(default_factory=EntityResolver)
    extractor: ActionExtractor = field(default_factory=ActionExtractor)
    verifier_version: str = VERIFIER_VERSION

    def __post_init__(self) -> None:
        self.clusterer = SourceDependencyClusterer(
            duplicate_jaccard=self.config.duplicate_jaccard, shingle_n=self.config.shingle_n
        )
        self.graph_builder = EvidenceGraphBuilder(
            config=self.config,
            resolver=self.resolver,
            extractor=self.extractor,
            clusterer=self.clusterer,
        )
        self.inference = AuthorizationInference(config=self.config)
        self.policy = RiskAwareOutputPolicy(config=self.config)
        # Per-(query, entity) graph cache: the official page is read once per query, not once per
        # (domain, action) triple. 10 candidates x 5 actions would otherwise re-read it 50 times.
        self._graph_cache: dict[tuple[str, str, str], EvidenceGraph] = {}

    # -- identity --------------------------------------------------------------------
    @property
    def defense_id(self) -> str:
        return self.config.config_id

    # -- fingerprint -----------------------------------------------------------------
    def assert_replay_fingerprint(self, ctx: VerificationContext) -> None:
        """Fail closed when the context is not the replay this instance was bound to."""
        if self.expected_replay_fingerprint is None:
            return
        got = ctx.replay_fingerprint
        if got != self.expected_replay_fingerprint:
            raise ReplayFingerprintError(
                f"replay fingerprint mismatch for query {ctx.query_id!r}: context carries "
                f"{got!r} but this verifier is bound to "
                f"{self.expected_replay_fingerprint!r}. CONTRACT.md Section 6 requires every "
                f"defense to be compared on identical frozen evidence."
            )

    # -- graph -----------------------------------------------------------------------
    def evidence_graph(self, ctx: VerificationContext, entity_id: str) -> EvidenceGraph:
        key = (ctx.query_id, ctx.read_phase, entity_id)
        cached = self._graph_cache.get(key)
        if cached is None:
            cached = self.graph_builder.build(ctx, entity_id)
            self._graph_cache[key] = cached
        return cached

    # -- the contract entry point ----------------------------------------------------
    def verify(
        self,
        entity_id: str,
        domain_id: str,
        action: str,
        ctx: VerificationContext | None = None,
    ) -> VerificationResult:
        """Verify whether ``domain_id`` may perform ``action`` for ``entity_id``.

        Parameters
        ----------
        entity_id, domain_id, action
            The triple. ``action`` must be in the frozen ontology.
        ctx
            The shared evidence channel. Required; the parameter is keyword-optional only so the
            signature reads as the contract writes it (``verify(e, d, a)``).
        """
        if ctx is None:
            raise ValueError(
                "verify(e, d, a) needs a VerificationContext: evidence is supplied through the "
                "shared channel so every defense provably sees the same inputs"
            )
        self.assert_replay_fingerprint(ctx)
        action_risk_rank(action)  # fail closed on an unknown action

        graph = self.evidence_graph(ctx, entity_id)
        p, z, feats = self.inference.score(graph, ctx, domain_id, action)

        contradictions = list(graph.contradictions_for(domain_id))
        # Action-scoped edges apply only at the actions they were raised for.
        scoped: list[ContradictionEdge] = []
        for c in contradictions:
            if c.kind is ContradictionKind.ACTION_SCOPE_OVERREACH:
                granted = graph.published_delegations.get(domain_id, ())
                if action in granted:
                    continue  # this action is inside the delegated scope
                scoped.append(replace(c, decisive=True))
                continue
            scoped.append(c)

        verdict, disposition, tau_v, tau_p, reasons = self.policy.decide(
            probability=p, contradictions=scoped, action=action
        )

        items = tuple(
            i for i in graph.for_domain(domain_id) if i.applies_to_action(action) or not i.actions
        )
        corroborators = [d for d in ctx.candidate_domain_ids() if d != domain_id]
        return VerificationResult(
            entity_id=entity_id,
            domain_id=domain_id,
            action=action,
            verdict=verdict,
            probability=p,
            score_logit=z,
            tau_verified=tau_v,
            tau_plausible=tau_p,
            disposition=disposition,
            features=feats,
            evidence=items,
            contradictions=tuple(scoped),
            families_used=tuple(sorted(graph.families_present(domain_id))),
            n_effective_sources=graph.n_effective_sources(corroborators),
            n_raw_sources=len(corroborators),
            reasons=tuple(reasons),
            config_id=self.config.config_id,
        )

    # -- Defense protocol ------------------------------------------------------------
    def decide(
        self, entity_id: str, domain_id: str, action: str, ctx: VerificationContext
    ) -> DefenseDecision:
        return self.verify(entity_id, domain_id, action, ctx).to_decision()

    # -- provenance ------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {
            "defense_id": self.defense_id,
            "verifier_version": self.verifier_version,
            "contract_ref": "CONTRACT.md Sections 3 (RQ3), 9",
            "config": self.config.as_dict(),
            "config_digest": self.config.digest(),
            "bound_replay_fingerprint": self.expected_replay_fingerprint,
            "components": {
                "1_entity_resolution": self.resolver.resolver_id,
                "2_action_extraction": self.extractor.extractor_id,
                "3_evidence_graph_construction": self.graph_builder.builder_id,
                "4_source_dependency_clustering": self.clusterer.clusterer_id,
                "5_action_specific_authorization_inference": self.inference.inference_id,
                "6_risk_aware_output_policy": self.policy.policy_id,
            },
            "evidence_families": list(f.value for f in EvidenceFamily),
            "feature_weights": dict(self.inference.weights),
            "no_authority_ceiling": round(self.inference.no_authority_ceiling, 6),
            "not_a_weighted_sum": (
                "Authority -- registry ownership, or a delegation published by the entity's own "
                "official page covering THIS action -- gates the score. Without it the probability "
                f"is capped at {self.inference.no_authority_ceiling:.4f}, which sits strictly "
                f"below every tau, so no amount of corroboration, identity polish or lexical "
                f"respectability can authorise any action at any risk level. The cap is applied "
                f"after scoring and is derived from the threshold vector, so it cannot be undone "
                f"by reweighting the features. Contradiction edges are named detections that "
                f"short-circuit the score entirely."
            ),
            "deliberate_abstentions": {
                "prompt_injection_markers": (
                    "Not a feature. Injection presence is a Section 7 factor level, so scoring it "
                    "would classify the factor rather than the authorization."
                ),
                "lifecycle_record_presence": (
                    "Not a feature. In this corpus a lifecycle record exists for exactly the "
                    "takeover domains, so presence alone is a label. The family may only withdraw "
                    "authority another family granted."
                ),
                "delegation_table": (
                    "web_rag.exposure.public_delegation_evidence is never imported: it names which "
                    "third party holds which grant, i.e. the answer to RQ3."
                ),
            },
        }


def describe_verifier() -> dict[str, Any]:
    """Module-level provenance for the results artifacts."""
    return AegisLink().describe()


__all__ = [
    "FEATURE_WEIGHTS",
    "VERIFIER_VERSION",
    "ActionExtractor",
    "AegisLink",
    "AegisLinkConfig",
    "AuthorizationInference",
    "EntityResolver",
    "EvidenceGraphBuilder",
    "ReplayFingerprintError",
    "RiskAwareOutputPolicy",
    "SourceDependencyClusterer",
    "describe_verifier",
]
