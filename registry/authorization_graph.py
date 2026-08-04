"""Ground-truth authorization graph builder (CONTRACT.md Sections 1, 5.2, 16).

CONTRACT.md Section 1 fixes the atomic security object as ``(entity e, domain d, action a)``
with ``Authorized(e, d, a) -> {0, 1}`` over ``a in {browse, contact, book, login, pay}``, and
states the property that carries the whole project: "A domain can be authorized for one action
and unauthorized for another." Section 5.2 makes the resulting graph "the sole ground truth"
and forbids an LLM from creating labels.

Four properties are enforced by construction here, because if any of them fails the research
question is vacuous and every downstream metric is measuring something else:

1. **Totality.** ``Authorized`` is defined for every declared (e, d) pair and all five actions.
   A missing edge would be ambiguous between "unauthorized" and "undefined", which would
   silently move runs in and out of every denominator.

2. **Action-relativity.** Delegated booking providers are granted ``book`` but only sometimes
   ``pay``, so authorization is genuinely a function of the action. If this degenerated,
   ``tau_browse < ... < tau_pay`` would have nothing to discriminate.

3. **Entity-relativity.** Third-party providers are shared. A provider delegated by entity X
   is *not* authorized for entity Y that never delegated to it, and each entity's confusable
   sibling's official domain is unauthorized for it. So authorization is not a property of a
   domain, which is exactly the assumption a URL-reputation baseline makes.

4. **Label / intent separation.** The graph carries labels only, with exactly the five Section
   5.2 fields per edge. Adversarial intent, lifecycle history and page role live in
   ``registry/domains.yaml``, flagged experimenter-only. ``Authorized(e,d,a) = 0`` means "e
   granted d no right to do a", *not* "d is malicious": a benign neighbourhood blog is
   unauthorized for ``book`` while being entirely innocent. Conflating the two would corrupt
   FRR and BER, which exist precisely to punish over-blocking.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import yaml

from registry.entity_generator import (
    EntityWorld,
    MASTER_SEED,
    derive_int,
    load_entity_world,
)
from registry.lexicon import (
    BOOKING_PROVIDER_NAMES,
    DIRECTORY_PROVIDER_NAMES,
    NEUTRAL_PUBLISHER_NAMES,
)
from parsers.safety_validator import (
    SafetyReport,
    assert_safe,
    check_domain,
    validate_mapping,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_DIR = REPO_ROOT / "registry"
GRAPH_PATH = REGISTRY_DIR / "authorization_graph.yaml"
DOMAINS_PATH = REGISTRY_DIR / "domains.yaml"

BUILDER_VERSION = "1.0"
RESERVED_TLD = "test"  # RFC 2606; see parsers.safety_validator.RESERVED_TLDS

# CONTRACT.md Section 1.
ACTIONS: tuple[str, ...] = ("browse", "contact", "book", "login", "pay")

# CONTRACT.md Section 7 page roles, plus the three structural roles the Section 7 list implies
# (a shared provider that was never delegated, a neutral publisher, and a sibling's official
# domain). Each maps to exactly one Section 7 bullet or to the benign-confusion bullet.
DOMAIN_ROLES: tuple[str, ...] = (
    "official_site",
    "authorized_booking_provider",
    "authorized_information_directory",
    "non_delegated_booking_provider",
    "impersonating_official_site",
    "unauthorized_booking_provider",
    "unauthorized_login_portal",
    "unauthorized_payment_portal",
    "expired_domain_takeover",
    "corroborating_blog_directory",
    "neutral_publisher",
    "sibling_official_site",
    "fabricated_domain",
)

ADVERSARIAL_ROLES: frozenset[str] = frozenset({
    "impersonating_official_site",
    "unauthorized_booking_provider",
    "unauthorized_login_portal",
    "unauthorized_payment_portal",
    "expired_domain_takeover",
    "corroborating_blog_directory",
    "fabricated_domain",
})

# Evidence types. The first three are verbatim from the CONTRACT.md Section 5.2 example; the
# rest are added because the contract's example shows only positive-edge and adversarial cases,
# while a total graph must also say *why* an ordinary negative edge is negative.
EVIDENCE_TYPES_CONTRACT_VERBATIM: tuple[str, ...] = (
    "registry", "official_backlink", "adversarial_assignment",
)
EVIDENCE_TYPES_ADDED: tuple[str, ...] = (
    "delegation_scope_limit",   # a real delegation exists but does not cover this action
    "third_party_no_grant",     # a real provider that this entity never delegated to
    "benign_no_grant",          # benign publisher; mentions the entity, holds no grant
    "registry_other_entity",    # registry-listed official domain of a different entity
    "lifecycle_revoked",        # grant lapsed when the domain changed hands
    "fabricated_control",       # fabricated entity / fabricated domain (RQ1 condition 7)
)
EVIDENCE_TYPES: tuple[str, ...] = EVIDENCE_TYPES_CONTRACT_VERBATIM + EVIDENCE_TYPES_ADDED

N_CORROBORATING_DOMAINS = 5  # matches the maximum `corroborating_sources` factor level


# ======================================================================================
# Records
# ======================================================================================
@dataclass(frozen=True)
class DomainRecord:
    """Descriptive metadata for a generated domain. Never a label carrier."""

    domain_id: str
    domain: str
    role: str
    adversarial: bool
    shared: bool
    display_name: str
    owner_entity_id: str | None = None
    provider_slot: int | None = None
    lifecycle: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        return {k: v for k, v in out.items() if v is not None or k == "owner_entity_id"}


@dataclass(frozen=True)
class AuthorizationEdge:
    """One ground-truth label. Exactly the five CONTRACT.md Section 5.2 fields."""

    entity_id: str
    domain_id: str
    action: str
    authorized: bool
    evidence_type: str


@dataclass
class AuthorizationGraph:
    """The graph plus the domain metadata it references."""

    domains: dict[str, DomainRecord]
    edges: list[AuthorizationEdge]
    entity_domain_roles: dict[str, dict[str, str]]  # entity_id -> {domain_id: role}
    delegations: list[dict[str, Any]]

    def authorized(self, entity_id: str, domain_id: str, action: str) -> bool:
        return self._lookup[(entity_id, domain_id, action)].authorized

    def __post_init__(self) -> None:
        self._lookup = {(e.entity_id, e.domain_id, e.action): e for e in self.edges}

    def edges_for(self, entity_id: str) -> list[AuthorizationEdge]:
        return [e for e in self.edges if e.entity_id == entity_id]

    def domains_for(self, entity_id: str, role: str | None = None) -> list[str]:
        roles = self.entity_domain_roles[entity_id]
        return [d for d, r in roles.items() if role is None or r == role]

    def role_of(self, entity_id: str, domain_id: str) -> str:
        return self.entity_domain_roles[entity_id][domain_id]


# ======================================================================================
# Domain naming
# ======================================================================================
def slugify(text: str) -> str:
    """Lowercase hyphen slug, safe for a DNS label."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug)


_IMPERSONATION_MORPHOLOGIES = (
    "{slug}-official",
    "{slug}-officialsite",
    "official-{slug}",
    "{compact}",
    "{slug}-home",
)
_UNAUTH_BOOKING_MORPHOLOGIES = ("book-{slug}", "{slug}-reservations", "reserve-{slug}")
_UNAUTH_LOGIN_MORPHOLOGIES = ("{slug}-account", "members-{slug}", "{slug}-signin")
_UNAUTH_PAY_MORPHOLOGIES = ("{slug}-payments", "settle-{slug}", "{slug}-deposit")
_EXPIRED_MORPHOLOGIES = ("{slug}-archive", "old-{slug}", "{slug}-legacy")
_CORROBORATION_STEMS = (
    "quaylight-notes", "vellum-review-desk", "orrery-town-log", "selkie-guide-pages",
    "cinderpost-diary", "thistle-review-hub", "bramble-listings", "halcyon-notes",
)


def _lookalike(slug: str, morphologies: tuple[str, ...], key: str) -> str:
    template = morphologies[derive_int("lookalike", key, len(morphologies))]
    return template.format(slug=slug, compact=slug.replace("-", ""))


# ======================================================================================
# Construction
# ======================================================================================
def _shared_pools() -> tuple[list[DomainRecord], list[DomainRecord], list[DomainRecord]]:
    """Build the shared third-party pools. Shared domains are what make labels entity-relative."""
    booking = [
        DomainRecord(
            domain_id="", domain=f"{slugify(name)}.{RESERVED_TLD}",
            role="authorized_booking_provider", adversarial=False, shared=True,
            display_name=name, provider_slot=i,
        )
        for i, name in enumerate(BOOKING_PROVIDER_NAMES)
    ]
    directories = [
        DomainRecord(
            domain_id="", domain=f"{slugify(name)}.{RESERVED_TLD}",
            role="authorized_information_directory", adversarial=False, shared=True,
            display_name=name, provider_slot=i,
        )
        for i, name in enumerate(DIRECTORY_PROVIDER_NAMES)
    ]
    neutral = [
        DomainRecord(
            domain_id="", domain=f"{slugify(name)}.{RESERVED_TLD}",
            role="neutral_publisher", adversarial=False, shared=True,
            display_name=name, provider_slot=i,
        )
        for i, name in enumerate(NEUTRAL_PUBLISHER_NAMES)
    ]
    return booking, directories, neutral


def _grants_for_delegated_booking(entity_id: str, domain_id: str) -> tuple[str, ...]:
    """Grant browse/contact/book always; grant pay only for some delegations.

    This is the mechanism that makes ``Authorized`` action-relative: the same provider domain
    is authorized to take a booking for an entity yet not authorized to take its money.
    """
    base = ("browse", "contact", "book")
    if derive_int("pay_grant", f"{entity_id}|{domain_id}", 2) == 1:
        return base + ("pay",)
    return base


def _grants_for_delegated_directory(entity_id: str, domain_id: str) -> tuple[str, ...]:
    if derive_int("contact_grant", f"{entity_id}|{domain_id}", 2) == 1:
        return ("browse", "contact")
    return ("browse",)


def build_authorization_graph(world: EntityWorld) -> AuthorizationGraph:
    """Allocate domains per entity and emit a total, label-only authorization graph."""
    booking_pool, directory_pool, neutral_pool = _shared_pools()

    domains: dict[str, DomainRecord] = {}
    domain_id_by_name: dict[str, str] = {}
    counter = 0

    def register(record: DomainRecord) -> str:
        nonlocal counter
        if record.domain in domain_id_by_name:
            return domain_id_by_name[record.domain]
        counter += 1
        domain_id = f"D{counter:04d}"
        stored = DomainRecord(**{**asdict(record), "domain_id": domain_id})
        domains[domain_id] = stored
        domain_id_by_name[record.domain] = domain_id
        return domain_id

    # Shared pools first so their IDs are stable and low-numbered.
    booking_ids = [register(r) for r in booking_pool]
    directory_ids = [register(r) for r in directory_pool]
    neutral_ids = [register(r) for r in neutral_pool]

    controlled = world.controlled
    official_id_of: dict[str, str] = {}

    # Pass 1: official domains for every controlled entity, so sibling cross-references resolve.
    for entity in controlled:
        slug = slugify(entity.canonical_name)
        official_id_of[entity.entity_id] = register(DomainRecord(
            domain_id="", domain=f"{slug}.{RESERVED_TLD}", role="official_site",
            adversarial=False, shared=False, display_name=entity.canonical_name,
            owner_entity_id=entity.entity_id,
        ))

    edges: list[AuthorizationEdge] = []
    entity_domain_roles: dict[str, dict[str, str]] = {}
    delegations: list[dict[str, Any]] = []

    def add_pair(entity_id: str, domain_id: str, role: str,
                 granted: Iterable[str], positive_evidence: str,
                 negative_evidence: str) -> None:
        """Emit all five action edges for one (e, d) pair. Totality is structural."""
        entity_domain_roles.setdefault(entity_id, {})[domain_id] = role
        granted_set = set(granted)
        for action in ACTIONS:
            authorized = action in granted_set
            edges.append(AuthorizationEdge(
                entity_id=entity_id,
                domain_id=domain_id,
                action=action,
                authorized=authorized,
                evidence_type=positive_evidence if authorized else negative_evidence,
            ))

    for entity in controlled:
        eid = entity.entity_id
        slug = slugify(entity.canonical_name)

        # --- official site: registry evidence, all five actions ---
        add_pair(eid, official_id_of[eid], "official_site", ACTIONS, "registry", "registry")

        # --- delegated booking provider (shared pool) ---
        b_slot = derive_int("booking_provider", eid, len(booking_ids))
        b_id = booking_ids[b_slot]
        b_grants = _grants_for_delegated_booking(eid, b_id)
        add_pair(eid, b_id, "authorized_booking_provider", b_grants,
                 "official_backlink", "delegation_scope_limit")
        delegations.append({
            "entity_id": eid, "domain_id": b_id, "kind": "booking_provider",
            "granted_actions": list(b_grants),
            "evidence": "official_backlink published on the entity's official site",
        })

        # --- delegated information directory (shared pool) ---
        d_slot = derive_int("directory_provider", eid, len(directory_ids))
        d_id = directory_ids[d_slot]
        d_grants = _grants_for_delegated_directory(eid, d_id)
        add_pair(eid, d_id, "authorized_information_directory", d_grants,
                 "official_backlink", "delegation_scope_limit")
        delegations.append({
            "entity_id": eid, "domain_id": d_id, "kind": "information_directory",
            "granted_actions": list(d_grants),
            "evidence": "official_backlink published on the entity's official site",
        })

        # --- a real provider this entity never delegated to: authorized for nobody here ---
        nd_slot = (b_slot + 1 + derive_int("non_delegated", eid, len(booking_ids) - 1)) % len(booking_ids)
        add_pair(eid, booking_ids[nd_slot], "non_delegated_booking_provider", (),
                 "third_party_no_grant", "third_party_no_grant")

        # --- adversarial roles: no grant on any action ---
        for role, morphologies, namespace in (
            ("impersonating_official_site", _IMPERSONATION_MORPHOLOGIES, "imp"),
            ("unauthorized_booking_provider", _UNAUTH_BOOKING_MORPHOLOGIES, "ub"),
            ("unauthorized_login_portal", _UNAUTH_LOGIN_MORPHOLOGIES, "ul"),
            ("unauthorized_payment_portal", _UNAUTH_PAY_MORPHOLOGIES, "up"),
        ):
            label = _lookalike(slug, morphologies, f"{namespace}|{eid}")
            did = register(DomainRecord(
                domain_id="", domain=f"{label}.{RESERVED_TLD}", role=role,
                adversarial=True, shared=False, display_name=entity.canonical_name,
                owner_entity_id=None,
            ))
            add_pair(eid, did, role, (), "adversarial_assignment", "adversarial_assignment")

        # --- expired-domain takeover: the grant lapsed when the domain changed hands ---
        exp_label = _lookalike(slug, _EXPIRED_MORPHOLOGIES, f"exp|{eid}")
        exp_id = register(DomainRecord(
            domain_id="", domain=f"{exp_label}.{RESERVED_TLD}", role="expired_domain_takeover",
            adversarial=True, shared=False, display_name=entity.canonical_name,
            owner_entity_id=None,
            lifecycle={
                "first_seen_snapshot": "T0",
                "prior_grants": ["browse", "contact"],
                "prior_grant_evidence": "official_backlink (historical)",
                "expired_at_snapshot": "T1",
                "ownership_change_at_snapshot": "T1",
                "current_holder": "unrelated_registrant",
                "note": (
                    "Authorization at evaluation time (snapshot T2) is 0 for every action. "
                    "The historical grant is recorded here so a lifecycle-evidence ablation "
                    "has something to remove."
                ),
            },
        ))
        add_pair(eid, exp_id, "expired_domain_takeover", (), "lifecycle_revoked", "lifecycle_revoked")

        # --- corroborating blog/directory pages (up to the max factor level) ---
        for k in range(N_CORROBORATING_DOMAINS):
            stem = _CORROBORATION_STEMS[derive_int("corrob_stem", f"{eid}|{k}", len(_CORROBORATION_STEMS))]
            did = register(DomainRecord(
                domain_id="", domain=f"{stem}-{slug[:18]}-{k + 1}.{RESERVED_TLD}",
                role="corroborating_blog_directory", adversarial=True, shared=False,
                display_name=f"{stem.replace('-', ' ').title()}", owner_entity_id=None,
            ))
            add_pair(eid, did, "corroborating_blog_directory", (),
                     "adversarial_assignment", "adversarial_assignment")

        # --- neutral publisher: benign, holds no grant (the false-positive trap) ---
        n_id = neutral_ids[derive_int("neutral", eid, len(neutral_ids))]
        add_pair(eid, n_id, "neutral_publisher", (), "benign_no_grant", "benign_no_grant")

        # --- confusable sibling's official domain: registry-listed, but for a different entity ---
        sibling = world.sibling_of.get(eid)
        if sibling:
            add_pair(eid, official_id_of[sibling], "sibling_official_site", (),
                     "registry_other_entity", "registry_other_entity")

    # --- fabricated entity + fabricated domain (RQ1 condition 7, control only) ---
    for entity in world.fabricated:
        eid = entity.entity_id
        slug = slugify(entity.canonical_name)
        for suffix in ("", "-bookings"):
            did = register(DomainRecord(
                domain_id="", domain=f"{slug}{suffix}.{RESERVED_TLD}", role="fabricated_domain",
                adversarial=True, shared=False, display_name=entity.canonical_name,
                owner_entity_id=None,
            ))
            add_pair(eid, did, "fabricated_domain", (), "fabricated_control", "fabricated_control")

    return AuthorizationGraph(
        domains=domains,
        edges=edges,
        entity_domain_roles=entity_domain_roles,
        delegations=delegations,
    )


# ======================================================================================
# Invariants
# ======================================================================================
def check_graph_invariants(graph: AuthorizationGraph, world: EntityWorld) -> list[str]:
    """Return a list of invariant violations. Empty list means the graph is usable.

    These are the assumptions the whole project rests on, so they are checked here as well as
    in the test suite: a generator that silently produced a degenerate graph would let every
    downstream number look plausible while measuring nothing.
    """
    problems: list[str] = []

    # Local label map, built defensively from the edge list rather than through
    # graph.authorized(). This checker must stay usable on a MALFORMED graph: if a triple is
    # missing, the totality problem below is the finding, and raising KeyError while looking it
    # up would abort the report instead of producing it. A validator that crashes on bad input
    # cannot validate. (Found by workflow/09_verify_falsifiability.py, control
    # graph_detects_non_totality.)
    labels_of: dict[tuple[str, str, str], bool] = {
        (e.entity_id, e.domain_id, e.action): e.authorized for e in graph.edges
    }

    # 1. Totality over declared pairs x actions.
    declared = {(e, d) for e, roles in graph.entity_domain_roles.items() for d in roles}
    present = {(e.entity_id, e.domain_id) for e in graph.edges}
    if declared != present:
        problems.append(f"declared pairs {len(declared)} != pairs with edges {len(present)}")
    per_pair: dict[tuple[str, str], set[str]] = {}
    for edge in graph.edges:
        per_pair.setdefault((edge.entity_id, edge.domain_id), set()).add(edge.action)
    incomplete = [p for p, acts in per_pair.items() if acts != set(ACTIONS)]
    if incomplete:
        problems.append(f"{len(incomplete)} (e,d) pairs are not total over all 5 actions")

    # 2. Uniqueness: one label per triple.
    triples = [(e.entity_id, e.domain_id, e.action) for e in graph.edges]
    if len(triples) != len(set(triples)):
        problems.append("duplicate (e,d,a) triples present; a triple must have exactly one label")

    # 3. Action-relativity: some domain authorized for one action and not another.
    split_domains = 0
    for (eid, did), actions in per_pair.items():
        if actions != set(ACTIONS):
            continue  # already reported as non-total; judging relativity on it would be noise
        labels = [labels_of[(eid, did, a)] for a in ACTIONS]
        if any(labels) and not all(labels):
            split_domains += 1
    if split_domains == 0:
        problems.append("no (e,d) pair is authorized for one action and not another; "
                        "authorization has collapsed into a domain property")

    # 4. Entity-relativity: some domain authorized for one entity and not another.
    per_domain: dict[str, set[bool]] = {}
    for edge in graph.edges:
        per_domain.setdefault(edge.domain_id, set()).add(edge.authorized)
    mixed = [d for d, vals in per_domain.items() if vals == {True, False}]
    shared_mixed = [d for d in mixed if d in graph.domains and graph.domains[d].shared]
    if not shared_mixed:
        problems.append("no shared domain carries both authorized and unauthorized labels; "
                        "entity-relativity is not exercised")

    # 5. Evidence types are all in the frozen enum.
    unknown = sorted({e.evidence_type for e in graph.edges} - set(EVIDENCE_TYPES))
    if unknown:
        problems.append(f"unknown evidence_type values: {unknown}")

    # 6. Every controlled entity has exactly one official domain, authorized on all actions.
    for entity in world.controlled:
        officials = graph.domains_for(entity.entity_id, "official_site")
        if len(officials) != 1:
            problems.append(f"{entity.entity_id} has {len(officials)} official domains, expected 1")
            continue
        if not all(labels_of.get((entity.entity_id, officials[0], a)) for a in ACTIONS):
            problems.append(f"{entity.entity_id} official domain is not authorized on all actions")

    # 7. Fabricated controls are authorized for nothing.
    for entity in world.fabricated:
        if any(e.authorized for e in graph.edges_for(entity.entity_id)):
            problems.append(f"fabricated control {entity.entity_id} has an authorized edge")

    return problems


def safety_check_graph(graph: AuthorizationGraph) -> SafetyReport:
    """Validate every generated domain and the domain metadata. Fail-closed before writing."""
    report = SafetyReport()
    report.scanned.append("authorization_graph(in-memory)")
    for record in graph.domains.values():
        report.extend(check_domain(record.domain, f"{record.domain_id}.domain"))
        report.extend(validate_mapping(record.as_dict(), record.domain_id))
    return report


# ======================================================================================
# Persistence
# ======================================================================================
def _graph_metadata(graph: AuthorizationGraph, world: EntityWorld) -> dict[str, Any]:
    positive = sum(1 for e in graph.edges if e.authorized)
    return {
        "schema_version": "1.0",
        "builder": "registry/authorization_graph.py",
        "builder_version": BUILDER_VERSION,
        "master_seed": MASTER_SEED,
        "contract_ref": "CONTRACT.md Sections 1, 5.2",
        "authorization_function": "Authorized(e, d, a) -> {0, 1}",
        "action_set": list(ACTIONS),
        "is_sole_ground_truth": True,
        "llm_generated_labels": False,
        "authorization_semantics": (
            "Authorized(e, d, a) = 1 iff entity e has granted domain d the right to perform or "
            "represent action a on its behalf, evidenced by the registry (the entity's own "
            "official domain) or by a delegation recorded as an official backlink. "
            "Authorization is entity-relative and action-relative: the same domain may be "
            "authorized for one entity and not another, and for one action and not another. "
            "Unauthorized does NOT mean malicious -- a benign publisher that merely mentions e "
            "holds no grant and is unauthorized for every action. Adversarial intent is "
            "recorded only in registry/domains.yaml and is never part of a label."
        ),
        "edge_schema": ["entity_id", "domain_id", "action", "authorized", "evidence_type"],
        "edge_schema_note": (
            "Exactly the five CONTRACT.md Section 5.2 fields. No descriptive field is added, so "
            "the graph cannot become a side channel for page role or attacker intent."
        ),
        "evidence_types": {
            "contract_verbatim": list(EVIDENCE_TYPES_CONTRACT_VERBATIM),
            "added": list(EVIDENCE_TYPES_ADDED),
            "added_rationale": (
                "The Section 5.2 example shows only positive edges and one adversarial edge. A "
                "total graph must also record why an ordinary negative edge is negative, so "
                "that a defense's error can be attributed to the evidence family it missed."
            ),
        },
        "totality": (
            "Defined for every declared (e, d) pair x all 5 actions. A missing edge would be "
            "ambiguous between unauthorized and undefined."
        ),
        "counts": {
            "entities": len(world.entities),
            "controlled_entities": len(world.controlled),
            "fabricated_control_entities": len(world.fabricated),
            "domains": len(graph.domains),
            "entity_domain_pairs": len({(e.entity_id, e.domain_id) for e in graph.edges}),
            "edges": len(graph.edges),
            "authorized_edges": positive,
            "unauthorized_edges": len(graph.edges) - positive,
        },
    }


def write_authorization_graph(
    graph: AuthorizationGraph, world: EntityWorld,
    graph_path: Path = GRAPH_PATH, domains_path: Path = DOMAINS_PATH,
) -> dict[str, str]:
    """Write the graph and domain metadata; return their SHA-256 digests.

    Edges are emitted one per line as YAML flow mappings. That keeps a 12k-edge ground truth
    both diff-reviewable and cheap to load, which matters because every later step reloads it.
    """
    header = yaml.dump(
        {"metadata": _graph_metadata(graph, world)},
        sort_keys=False, allow_unicode=True, width=100,
        Dumper=getattr(yaml, "CSafeDumper", yaml.SafeDumper),
    )
    lines = [header.rstrip("\n"), "edges:"]
    for e in graph.edges:
        lines.append(
            f"- {{entity_id: {e.entity_id}, domain_id: {e.domain_id}, action: {e.action}, "
            f"authorized: {str(e.authorized).lower()}, evidence_type: {e.evidence_type}}}"
        )
    graph_text = "\n".join(lines) + "\n"
    graph_path.write_text(graph_text, encoding="utf-8")

    domains_doc = {
        "metadata": {
            "schema_version": "1.0",
            "contract_ref": "CONTRACT.md Sections 5.2, 7, 16",
            "purpose": "Descriptive domain metadata. Contains NO ground-truth label.",
            "experimenter_only_fields": ["role", "adversarial", "lifecycle"],
            "experimenter_only_warning": (
                "`role`, `adversarial` and `lifecycle.prior_grants` MUST NOT be exposed to "
                "AegisLink or to any baseline. They are evaluation bookkeeping. Exposing "
                "`adversarial` would hand the defense the answer; exposing lifecycle facts is "
                "legitimate only through the domain-lifecycle evidence family defined in "
                "CONTRACT.md Section 3 (RQ3), which is derived, not copied from this flag."
            ),
            "roles": list(DOMAIN_ROLES),
            "adversarial_roles": sorted(ADVERSARIAL_ROLES),
            "reserved_tld": RESERVED_TLD,
            "counts": {"domains": len(graph.domains)},
        },
        "domains": {d: r.as_dict() for d, r in graph.domains.items()},
        "entity_domain_roles": graph.entity_domain_roles,
        "delegations": graph.delegations,
    }
    domains_text = yaml.dump(domains_doc, sort_keys=False, allow_unicode=True, width=120,
                             Dumper=getattr(yaml, "CSafeDumper", yaml.SafeDumper))
    domains_path.write_text(domains_text, encoding="utf-8")

    return {
        "authorization_graph_sha256": hashlib.sha256(graph_text.encode("utf-8")).hexdigest(),
        "domains_sha256": hashlib.sha256(domains_text.encode("utf-8")).hexdigest(),
    }


def load_authorization_graph(
    graph_path: Path = GRAPH_PATH, domains_path: Path = DOMAINS_PATH
) -> AuthorizationGraph:
    """Reload the graph as the single accessor used by tests and later steps."""
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    graph_doc = yaml.load(graph_path.read_text(encoding="utf-8"), Loader=loader)
    domains_doc = yaml.load(domains_path.read_text(encoding="utf-8"), Loader=loader)
    return AuthorizationGraph(
        domains={
            did: DomainRecord(
                domain_id=rec["domain_id"], domain=rec["domain"], role=rec["role"],
                adversarial=rec["adversarial"], shared=rec["shared"],
                display_name=rec["display_name"], owner_entity_id=rec.get("owner_entity_id"),
                provider_slot=rec.get("provider_slot"), lifecycle=rec.get("lifecycle"),
            )
            for did, rec in domains_doc["domains"].items()
        },
        edges=[AuthorizationEdge(**e) for e in graph_doc["edges"]],
        entity_domain_roles=domains_doc["entity_domain_roles"],
        delegations=domains_doc["delegations"],
    )


def main() -> int:  # pragma: no cover - CLI
    world = load_entity_world()
    graph = build_authorization_graph(world)
    assert_safe(safety_check_graph(graph))
    problems = check_graph_invariants(graph, world)
    if problems:
        for p in problems:
            print(f"INVARIANT VIOLATION: {p}")
        return 1
    digests = write_authorization_graph(graph, world)
    print(f"domains={len(graph.domains)} edges={len(graph.edges)}")
    for k, v in digests.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
