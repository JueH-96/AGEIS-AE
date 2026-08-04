"""Exposure firewall between experimenter bookkeeping and the Web-RAG pipeline.

CONTRACT.md Sections 5.2, 6 and ``registry/domains.yaml`` metadata.

Why this module exists
----------------------
``registry/domains.yaml`` carries three fields that are *evaluation bookkeeping*, not world
facts a crawler could observe:

``role``
    The generator's page-role assignment (``impersonating_official_site``,
    ``authorized_booking_provider``, ...). Thirteen of these strings name the answer.
``adversarial``
    A boolean that literally is "is this an attack". Handing it to the defense would make
    every reported authorization number an artifact of the generator.
``lifecycle.prior_grants``
    A list of actions a now-expired domain *used* to be granted. A grant statement is an
    authorization statement, so copying it across the boundary is copying part of the label.

The naive implementation of this boundary is a blacklist -- ``del record["role"]``. That fails
open: a field added to the registry in a later step is exposed by default, and nothing detects
it. This module therefore does the opposite in three layers:

1. **Whitelist construction.** Public views are built by *naming the allowed fields*. A new
   registry field is invisible until someone deliberately adds it to a whitelist here.
2. **Runtime guard.** :func:`assert_no_leakage` walks any object about to cross a public API
   boundary and raises on a forbidden key *or a forbidden value*. Value-level checking matters
   because a leak can be laundered: ``{"note": "impersonating_official_site"}`` carries the
   label without using the key ``role``.
3. **Tiering.** :class:`ExposureTier` makes "who is allowed to see what" an explicit, testable
   parameter instead of a convention.

Exposure tiers
--------------
``RETRIEVER``
    Visible page text, URL, and public domain descriptors. This is what an indexer and a
    ranker get. Deliberately *excludes* the ``x-aegis-entity-id`` meta channel: CONTRACT.md
    Section 8 publishes that channel so the *automatic evaluator* can score without manual
    reading, and feeding it to the retriever would trivialise entity resolution, which is one
    of the five RQ2 stages under measurement.
``READER``
    Retriever tier plus rank and score. The LLM reader sees exactly what a real web-RAG reader
    sees: ranked page text.
``VERIFIER``
    Reader tier plus the public registry projection and *derived* lifecycle observations. RQ3
    lists "domain lifecycle and ownership-change metadata" as a required evidence family;
    ``domains.yaml`` permits it "only through the domain-lifecycle evidence family ... which is
    derived, not copied from this flag". :class:`LifecycleObservation` is that derivation, and
    it hard-strips the grant statements.
``EVALUATOR``
    Experimenter tier. Ground-truth manifest, page roles, adversarial flags, the meta channel.
    Nothing at this tier may be passed to a defense or a baseline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

EXPOSURE_VERSION = "1.0"

REPO_ROOT = Path(__file__).resolve().parent.parent
DOMAINS_PATH = REPO_ROOT / "registry" / "domains.yaml"


# ======================================================================================
# The forbidden surface
# ======================================================================================
#: Top-level ``domains.yaml`` record fields that must never cross the boundary.
EXPERIMENTER_ONLY_FIELDS: frozenset[str] = frozenset({"role", "adversarial"})

#: ``lifecycle`` sub-fields that must never cross the boundary. ``prior_grants`` and
#: ``prior_grant_evidence`` are grant statements; ``note`` is generator commentary that
#: spells out the intended label in prose.
EXPERIMENTER_ONLY_LIFECYCLE_SUBFIELDS: frozenset[str] = frozenset(
    {"prior_grants", "prior_grant_evidence", "note"}
)

#: Fields from the *page manifest* that are experimenter-only. The manifest exists precisely
#: because these must not live in the page bytes (CONTRACT.md Section 6); they must not be
#: smuggled back in through a Python object either.
EXPERIMENTER_ONLY_MANIFEST_FIELDS: frozenset[str] = frozenset(
    {"page_role", "attack_template_id", "site_template_id", "entity_template_id", "snapshot"}
)

#: Ground-truth vocabulary. ``authorized`` / ``evidence_type`` belong to the label schema.
EXPERIMENTER_ONLY_LABEL_FIELDS: frozenset[str] = frozenset(
    {"authorized", "evidence_type", "granted_actions", "adversarial_assignment"}
)

ALL_FORBIDDEN_FIELDS: frozenset[str] = (
    EXPERIMENTER_ONLY_FIELDS
    | EXPERIMENTER_ONLY_LIFECYCLE_SUBFIELDS
    | EXPERIMENTER_ONLY_MANIFEST_FIELDS
    | EXPERIMENTER_ONLY_LABEL_FIELDS
)

#: Every ``role`` string the generator can assign. Checked as a *value* so a leak cannot be
#: laundered through an innocuously named key.
DOMAIN_ROLE_VALUES: tuple[str, ...] = (
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

#: Page roles used by the site generator. Overlaps ``DOMAIN_ROLE_VALUES`` plus benign pages.
PAGE_ROLE_VALUES: tuple[str, ...] = DOMAIN_ROLE_VALUES + ("benign_confusing_page",)

_FORBIDDEN_VALUE_SET: frozenset[str] = frozenset(DOMAIN_ROLE_VALUES) | frozenset(PAGE_ROLE_VALUES)

#: Public, crawl-observable descriptors of a domain. Whitelist -- anything not named here is
#: excluded by construction.
PUBLIC_DOMAIN_FIELDS: tuple[str, ...] = (
    "domain_id",
    "domain",
    "display_name",
    "shared",
)

#: Lifecycle sub-fields that a crawler could legitimately observe across snapshots.
PUBLIC_LIFECYCLE_SUBFIELDS: tuple[str, ...] = (
    "first_seen_snapshot",
    "expired_at_snapshot",
    "ownership_change_at_snapshot",
    "current_holder",
)

#: Attack-template ids look like ``A0001`` / ``AD-0001``; template ids are experimenter-only
#: provenance. Detect them as values, not just as keys.
_TEMPLATE_ID_PATTERN = re.compile(r"^(?:A|AD-|ET|FT|S|PG)\d{3,6}$")


class LeakageError(AssertionError):
    """Raised when an experimenter-only field or value reaches a public boundary.

    Deliberately an :class:`AssertionError` subclass: a leak is a correctness violation of the
    experiment, not a recoverable runtime condition. Callers must not catch it in production
    paths.
    """


class ExposureTier(str, Enum):
    """Who is allowed to see what. See module docstring."""

    RETRIEVER = "retriever"
    READER = "reader"
    VERIFIER = "verifier"
    EVALUATOR = "evaluator"

    @property
    def is_public(self) -> bool:
        """True for tiers subject to the firewall (everything except ``EVALUATOR``)."""
        return self is not ExposureTier.EVALUATOR


#: Tiers that the firewall guards. ``EVALUATOR`` is the experimenter side of the boundary.
PUBLIC_TIERS: tuple[ExposureTier, ...] = (
    ExposureTier.RETRIEVER,
    ExposureTier.READER,
    ExposureTier.VERIFIER,
)


# ======================================================================================
# Recursive guard
# ======================================================================================
def _iter_paths(obj: Any, prefix: str = "$") -> Iterable[tuple[str, str | None, Any]]:
    """Yield ``(json_path, key_or_None, value)`` for every node in a nested structure.

    Dataclasses and objects exposing ``as_dict`` / ``__dict__`` are walked too, so a leak
    hidden in an attribute rather than a dict key is still found.
    """
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            path = f"{prefix}.{k}"
            yield path, str(k), v
            yield from _iter_paths(v, path)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for i, v in enumerate(obj):
            path = f"{prefix}[{i}]"
            yield path, None, v
            yield from _iter_paths(v, path)
    elif hasattr(obj, "as_dict") and callable(obj.as_dict):
        yield from _iter_paths(obj.as_dict(), prefix)
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        vars_ = vars(obj)
        if vars_:
            for k, v in vars_.items():
                path = f"{prefix}.{k}"
                yield path, str(k), v
                yield from _iter_paths(v, path)


def assert_no_leakage(
    obj: Any,
    *,
    location: str = "<object>",
    forbidden_fields: Iterable[str] | None = None,
    forbidden_values: Iterable[str] | None = None,
    allow_free_text: bool = False,
) -> None:
    """Raise :class:`LeakageError` if ``obj`` carries an experimenter-only field or value.

    Parameters
    ----------
    obj
        Any nested structure or dataclass about to cross a public API boundary.
    location
        Human-readable origin, used in the error message.
    forbidden_fields, forbidden_values
        Overrides for testing. Default to the module-level sets.
    allow_free_text
        When ``True``, long strings are exempt from *value* checking. Needed for page prose:
        the corpus deliberately contains words like ``official``, and a benign directory page
        may legitimately contain the substring ``site``. Value checking then applies only to
        short strings, i.e. to enum-like payloads, which is where a real leak would live.
        Field-name checking is never relaxed.

    Notes
    -----
    Value checking uses exact equality on stripped strings rather than substring search.
    Substring search over page prose produces false positives (``"official_site"`` is not a
    substring of natural prose, but ``"neutral_publisher"``-style tokens can appear in
    generated slugs), and the leak this guard defends against is a *structured* one: an enum
    value copied into a field. Prose-level label-vocabulary leakage is a different guarantee,
    enforced separately by ``site_generator.LABEL_VOCABULARY_FORBIDDEN_IN_PAGES`` and by
    ``parsers/safety_validator.py``.
    """
    bad_fields = frozenset(forbidden_fields) if forbidden_fields is not None else ALL_FORBIDDEN_FIELDS
    bad_values = (
        frozenset(forbidden_values) if forbidden_values is not None else _FORBIDDEN_VALUE_SET
    )

    for path, key, value in _iter_paths(obj):
        if key is not None and key in bad_fields:
            raise LeakageError(
                f"experimenter-only field '{key}' reached a public boundary at "
                f"{location}{path}. Whitelist the field explicitly in web_rag/exposure.py "
                f"if it is genuinely crawl-observable."
            )
        if isinstance(value, str):
            stripped = value.strip()
            if stripped in bad_values and not (allow_free_text and len(stripped) > 64):
                raise LeakageError(
                    f"experimenter-only VALUE {stripped!r} reached a public boundary at "
                    f"{location}{path}. A label cannot be laundered through a benign key."
                )
            if _TEMPLATE_ID_PATTERN.match(stripped) and key not in (None, "doc_id"):
                # Template/page provenance ids are experimenter-only (Section 6: they live in
                # the side manifest precisely so the served page cannot leak them).
                if key in EXPERIMENTER_ONLY_MANIFEST_FIELDS or key in ("template_id",):
                    raise LeakageError(
                        f"template provenance id {stripped!r} reached a public boundary at "
                        f"{location}{path}."
                    )


def guarded(obj: Any, *, location: str, allow_free_text: bool = True) -> Any:
    """Run :func:`assert_no_leakage` and return ``obj`` unchanged.

    Sugar for ``return guarded(view, location="Retriever.search")`` so every public return
    path is guarded in one visually obvious place.
    """
    assert_no_leakage(obj, location=location, allow_free_text=allow_free_text)
    return obj


# ======================================================================================
# Public projections of registry/domains.yaml
# ======================================================================================
@dataclass(frozen=True)
class PublicDomainView:
    """Crawl-observable descriptors of a domain. No role, no adversarial flag."""

    domain_id: str
    domain: str
    display_name: str
    shared: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "domain": self.domain,
            "display_name": self.display_name,
            "shared": self.shared,
        }


@dataclass(frozen=True)
class LifecycleObservation:
    """Derived, crawl-observable lifecycle facts for the RQ3 lifecycle evidence family.

    Derived, not copied: the source record's ``prior_grants``, ``prior_grant_evidence`` and
    ``note`` are dropped, and ``ownership_changed`` / ``expired`` are recomputed from snapshot
    labels rather than read from a flag. What survives is what a crawler comparing two crawl
    snapshots of the same host could actually see.
    """

    domain_id: str
    first_seen_snapshot: str | None
    expired_at_snapshot: str | None
    ownership_change_at_snapshot: str | None
    current_holder: str | None

    @property
    def expired(self) -> bool:
        return self.expired_at_snapshot is not None

    @property
    def ownership_changed(self) -> bool:
        return self.ownership_change_at_snapshot is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "first_seen_snapshot": self.first_seen_snapshot,
            "expired_at_snapshot": self.expired_at_snapshot,
            "ownership_change_at_snapshot": self.ownership_change_at_snapshot,
            "current_holder": self.current_holder,
            "expired": self.expired,
            "ownership_changed": self.ownership_changed,
        }


@dataclass(frozen=True)
class PublicDomainRegistry:
    """The whole of ``registry/domains.yaml`` as the pipeline is allowed to see it."""

    domains: dict[str, PublicDomainView]
    lifecycle: dict[str, LifecycleObservation]
    source_digest: str | None = None
    stripped_fields: tuple[str, ...] = field(default=())

    def view(self, domain_id: str) -> PublicDomainView:
        return self.domains[domain_id]

    def by_hostname(self, hostname: str) -> PublicDomainView | None:
        for v in self.domains.values():
            if v.domain == hostname:
                return v
        return None

    def as_dict(self, tier: ExposureTier = ExposureTier.RETRIEVER) -> dict[str, Any]:
        """Serialise at ``tier``. Lifecycle observations appear only at ``VERIFIER``."""
        out: dict[str, Any] = {
            "domains": {k: v.as_dict() for k, v in sorted(self.domains.items())},
        }
        if tier is ExposureTier.VERIFIER:
            out["lifecycle"] = {k: v.as_dict() for k, v in sorted(self.lifecycle.items())}
        return out


def project_domain_record(record: Mapping[str, Any]) -> PublicDomainView:
    """Whitelist-project one raw ``domains.yaml`` record.

    Missing optional descriptors default rather than raise, so a record that legitimately
    lacks ``display_name`` does not force the caller to special-case it.
    """
    return PublicDomainView(
        domain_id=str(record["domain_id"]),
        domain=str(record["domain"]),
        display_name=str(record.get("display_name") or record["domain"]),
        shared=bool(record.get("shared", False)),
    )


def derive_lifecycle_observation(record: Mapping[str, Any]) -> LifecycleObservation | None:
    """Derive a :class:`LifecycleObservation`, or ``None`` if the domain has no lifecycle."""
    lc = record.get("lifecycle")
    if not isinstance(lc, Mapping):
        return None
    return LifecycleObservation(
        domain_id=str(record["domain_id"]),
        first_seen_snapshot=_opt_str(lc.get("first_seen_snapshot")),
        expired_at_snapshot=_opt_str(lc.get("expired_at_snapshot")),
        ownership_change_at_snapshot=_opt_str(lc.get("ownership_change_at_snapshot")),
        current_holder=_opt_str(lc.get("current_holder")),
    )


def _opt_str(v: Any) -> str | None:
    return None if v is None else str(v)


def load_public_domain_registry(path: Path | None = None) -> PublicDomainRegistry:
    """Load ``registry/domains.yaml`` and return only the public projection.

    The raw mapping is never returned and never stored on the result, so a caller cannot reach
    the experimenter fields through this function even by accident.
    """
    src = path or DOMAINS_PATH
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    raw = yaml.load(src.read_text(encoding="utf-8"), Loader=loader)

    views: dict[str, PublicDomainView] = {}
    lifecycles: dict[str, LifecycleObservation] = {}
    for domain_id, record in raw["domains"].items():
        views[str(domain_id)] = project_domain_record(record)
        obs = derive_lifecycle_observation(record)
        if obs is not None:
            lifecycles[str(domain_id)] = obs

    registry = PublicDomainRegistry(
        domains=views,
        lifecycle=lifecycles,
        stripped_fields=tuple(sorted(ALL_FORBIDDEN_FIELDS)),
    )
    # Self-check: the projection must survive its own guard at the strictest public tier.
    assert_no_leakage(
        registry.as_dict(ExposureTier.RETRIEVER), location="load_public_domain_registry"
    )
    return registry


def public_delegation_evidence(path: Path | None = None) -> list[dict[str, Any]]:
    """Backlink evidence as published *by the entity's own official site*.

    RQ3 lists "official-domain-to-third-party backlinks" as a required evidence family, so
    delegations are legitimately observable -- but only in the form the corpus actually
    publishes: an official page naming a partner and an action. ``granted_actions`` is renamed
    to ``claimed_actions`` and the ``evidence`` provenance string is dropped, because the
    verifier must *read the backlink off the page*, not trust the registry's word for it.

    This function therefore returns the *addressing* of backlink evidence (which official
    domain points at which third party), leaving action extraction to the reader. It is a
    ``VERIFIER``-tier artifact.
    """
    src = path or DOMAINS_PATH
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    raw = yaml.load(src.read_text(encoding="utf-8"), Loader=loader)
    out: list[dict[str, Any]] = []
    for d in raw.get("delegations", []):
        out.append(
            {
                "entity_id": str(d["entity_id"]),
                "domain_id": str(d["domain_id"]),
                "kind": str(d.get("kind", "unspecified")),
                "claimed_actions": sorted(str(a) for a in d.get("granted_actions", [])),
            }
        )
    out.sort(key=lambda r: (r["entity_id"], r["domain_id"]))
    assert_no_leakage(out, location="public_delegation_evidence")
    return out


# ======================================================================================
# The authoritative registry (VERIFIER tier)
# ======================================================================================
#: Why an official-domain registry is a legitimate public artifact and not a laundered label.
#:
#: CONTRACT.md Section 3 (RQ3) lists "generated authoritative registry" as the FIRST required
#: evidence family, and Section 10 mandates a baseline that allows "only the registry-listed
#: official domain". Both are impossible without an artifact that names, per entity, which
#: domain is its official one. So this projection is required by the contract, not a convenience.
#:
#: The boundary is narrow and worth stating precisely, because in ``registry/domains.yaml`` the
#: field ``owner_entity_id`` is non-null for exactly the 160 ``official_site`` records and null
#: everywhere else. Publishing it is therefore informationally equivalent to publishing
#: ``role == "official_site"`` -- and nothing more. What it does NOT reveal:
#:
#:   * whether any OTHER domain is authorized for the entity. Third-party authorization lives in
#:     ``domains.yaml.delegations`` and is never exposed here; a verifier must establish it by
#:     reading the official site's published backlinks.
#:   * the ``adversarial`` flag, or which of the non-official domains is impersonating, an
#:     unauthorized booking provider, a takeover, or a Sybil corroborator. All 1,630 non-official
#:     domains are indistinguishable in this projection.
#:   * any per-action grant. Action relativity (a directory authorized for browse/contact but
#:     not for pay) is invisible here.
#:
#: The discrimination this benchmark actually measures -- authorized third party vs. forged
#: third party, at a specific action -- is untouched. See
#: ``OFFICIAL_REGISTRY_EXCLUSIONS`` for the machine-readable statement of what is withheld.
OFFICIAL_REGISTRY_RATIONALE = (
    "CONTRACT.md Section 3 (RQ3) evidence family 1 and Section 10 baseline 9 both require an "
    "authoritative registry naming each entity's official domain. This projection publishes "
    "that and the entity's true identity fields, and withholds every third-party grant, so the "
    "authorized-vs-forged third-party discrimination under study is unaffected."
)

#: Named so a reader can check the claim above rather than take it on trust.
OFFICIAL_REGISTRY_EXCLUSIONS: tuple[str, ...] = (
    "domains.yaml.delegations (which third party holds which grant)",
    "granted_actions / claimed_actions (per-action grants)",
    "role and adversarial for every non-official domain",
    "lifecycle.prior_grants and lifecycle.note",
    "page_role, attack_template_id, site_template_id",
)


@dataclass(frozen=True)
class OfficialRegistryRecord:
    """One authoritative registry record: an entity, its official domain, its identity fields.

    ``official_domain_id`` is ``None`` for a fabricated control entity, which by construction has
    no registry entry at all. That absence is itself the correct public fact: a verifier asked
    about a fabricated entity should find no authority, not a silent default.
    """

    entity_id: str
    canonical_name: str
    category: str
    official_domain_id: str | None
    official_domain: str | None
    address: str | None
    phone: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "category": self.category,
            "official_domain_id": self.official_domain_id,
            "official_domain": self.official_domain,
            "address": self.address,
            "phone": self.phone,
        }


@dataclass(frozen=True)
class OfficialRegistry:
    """``VERIFIER``-tier view of the authoritative registry, keyed by entity id."""

    records: dict[str, OfficialRegistryRecord]
    aliases: dict[str, tuple[str, ...]]

    def get(self, entity_id: str) -> OfficialRegistryRecord | None:
        return self.records.get(entity_id)

    def official_domain_id(self, entity_id: str) -> str | None:
        rec = self.records.get(entity_id)
        return None if rec is None else rec.official_domain_id

    def entity_of_official_domain(self, domain_id: str) -> str | None:
        """Reverse lookup. ``None`` when the domain is not any entity's official domain."""
        for eid, rec in self.records.items():
            if rec.official_domain_id == domain_id:
                return eid
        return None

    def surface_forms(self, entity_id: str) -> tuple[str, ...]:
        """Canonical name plus published aliases, longest first (for entity resolution)."""
        rec = self.records.get(entity_id)
        if rec is None:
            return ()
        forms = [rec.canonical_name, *self.aliases.get(entity_id, ())]
        return tuple(sorted({f for f in forms if f}, key=lambda f: (-len(f), f)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "records": {k: v.as_dict() for k, v in sorted(self.records.items())},
            "aliases": {k: list(v) for k, v in sorted(self.aliases.items())},
        }


def load_official_registry(
    *,
    domains_path: Path | None = None,
    entities_path: Path | None = None,
    addresses_path: Path | None = None,
    phones_path: Path | None = None,
    aliases_path: Path | None = None,
) -> OfficialRegistry:
    """Build the ``VERIFIER``-tier authoritative registry.

    Reads ``domains.yaml`` only to find ``owner_entity_id``; the raw records, their ``role`` and
    their ``adversarial`` flag are never returned. Reads the identity tables so the
    identity-field-consistency evidence family (RQ3: "name, address, phone, entity ID") has an
    authoritative reference to compare page-published fields against.
    """
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

    def _load(p: Path) -> Any:
        return yaml.load(p.read_text(encoding="utf-8"), Loader=loader)

    root = REPO_ROOT / "registry"
    raw_domains = _load(domains_path or DOMAINS_PATH)["domains"]
    entities = _load(entities_path or root / "entities.yaml")["entities"]
    addresses = _load(addresses_path or root / "addresses.yaml")["addresses"]
    phones = _load(phones_path or root / "phones.yaml")["phones"]
    alias_doc = _load(aliases_path or root / "aliases.yaml")["aliases"]

    # entity -> official domain, via ownership. Nothing else from the domain record travels.
    official_of: dict[str, tuple[str, str]] = {}
    for did, rec in raw_domains.items():
        owner = rec.get("owner_entity_id")
        if owner:
            owner = str(owner)
            if owner in official_of:
                raise ValueError(
                    f"entity {owner} owns more than one domain ({official_of[owner][0]}, {did}); "
                    f"the authoritative registry assumes at most one official domain per entity"
                )
            official_of[owner] = (str(did), str(rec["domain"]))

    def _addr_text(aid: Any) -> str | None:
        """Render the address exactly as ``EntityWorld.address_text`` does.

        The identity-consistency evidence family compares a page-published address against this
        string, so any formatting drift would manufacture a false contradiction on every page.
        Kept byte-identical to ``registry/entity_generator.py::EntityWorld.address_text``.
        """
        a = addresses.get(str(aid)) if aid else None
        if not isinstance(a, Mapping):
            return None
        return f"{a['street']}, {a['city']}, {a['region']} {a['postal_code']}"

    def _phone_text(pid: Any) -> str | None:
        p = phones.get(str(pid)) if pid else None
        if isinstance(p, Mapping):  # tolerate a future richer phone record
            return _opt_str(p.get("number") or p.get("phone"))
        return _opt_str(p)

    records: dict[str, OfficialRegistryRecord] = {}
    for e in entities:
        eid = str(e["entity_id"])
        did, host = official_of.get(eid, (None, None))
        records[eid] = OfficialRegistryRecord(
            entity_id=eid,
            canonical_name=str(e["canonical_name"]),
            category=str(e.get("category", "unknown")),
            official_domain_id=did,
            official_domain=host,
            address=_addr_text(e.get("address_id")),
            phone=_phone_text(e.get("phone_id")),
        )

    registry = OfficialRegistry(
        records=records,
        aliases={str(k): tuple(str(a) for a in v) for k, v in alias_doc.items()},
    )
    # Must survive its own guard: an identity field is public, a label is not.
    assert_no_leakage(registry.as_dict(), location="load_official_registry", allow_free_text=True)
    return registry


# ======================================================================================
# Crawl-observable site index (RETRIEVER tier)
# ======================================================================================
@dataclass(frozen=True)
class SiteIndex:
    """Which document slots a crawler holds per host. ``RETRIEVER`` tier.

    A crawler that has fetched the corpus knows, for every hostname, which pages it holds. That
    is exactly what :func:`strip_manifest_entry` already declares public (url, domain,
    domain_id). This index makes the same information addressable, so a verifier can consult a
    domain it did not happen to retrieve -- specifically, the entity's official site, which RQ3's
    backlink evidence family requires it to read.

    Provenance fields (``page_role``, ``snapshot``, ``*_template_id``) are absent by construction:
    the index is built from stripped entries only.
    """

    doc_ids_by_domain_id: dict[str, tuple[str, ...]]
    urls_by_doc_id: dict[str, str]

    def doc_ids_for_domain(self, domain_id: str) -> tuple[str, ...]:
        return self.doc_ids_by_domain_id.get(domain_id, ())

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_ids_by_domain_id": {
                k: list(v) for k, v in sorted(self.doc_ids_by_domain_id.items())
            },
            "n_domains": len(self.doc_ids_by_domain_id),
            "n_docs": len(self.urls_by_doc_id),
        }


def build_site_index(documents: Mapping[str, Any]) -> SiteIndex:
    """Build a :class:`SiteIndex` from crawled document slots.

    Takes a mapping ``doc_id -> object`` exposing ``domain_id`` and ``url`` (i.e. a
    ``CrawledDocument``), and keeps only those two fields. Passing the corpus's document mapping
    is safe because nothing else is read off the objects.
    """
    by_domain: dict[str, list[str]] = {}
    urls: dict[str, str] = {}
    for doc_id, doc in documents.items():
        did = str(getattr(doc, "domain_id"))
        by_domain.setdefault(did, []).append(str(doc_id))
        urls[str(doc_id)] = str(getattr(doc, "url"))
    index = SiteIndex(
        doc_ids_by_domain_id={k: tuple(sorted(v)) for k, v in sorted(by_domain.items())},
        urls_by_doc_id=dict(sorted(urls.items())),
    )
    assert_no_leakage(index.as_dict(), location="build_site_index")
    return index


def strip_manifest_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce a page-manifest entry to its ``RETRIEVER``-tier fields.

    Keeps only what a crawler learns by fetching: an opaque document handle, the URL and the
    hostname. Drops ``page_role``, ``snapshot``, ``attack_template_id``, ``site_template_id``,
    ``entity_template_id`` and ``entity_id``.
    """
    return {
        "url": str(entry["url"]),
        "domain": str(entry["domain"]),
        "domain_id": str(entry["domain_id"]),
    }


def describe_firewall() -> dict[str, Any]:
    """Machine-readable description of the firewall, for the frozen snapshot provenance."""
    return {
        "exposure_version": EXPOSURE_VERSION,
        "contract_ref": "CONTRACT.md Sections 5.2, 6; registry/domains.yaml metadata",
        "experimenter_only_fields": sorted(EXPERIMENTER_ONLY_FIELDS),
        "experimenter_only_lifecycle_subfields": sorted(EXPERIMENTER_ONLY_LIFECYCLE_SUBFIELDS),
        "experimenter_only_manifest_fields": sorted(EXPERIMENTER_ONLY_MANIFEST_FIELDS),
        "experimenter_only_label_fields": sorted(EXPERIMENTER_ONLY_LABEL_FIELDS),
        "public_domain_fields": list(PUBLIC_DOMAIN_FIELDS),
        "public_lifecycle_subfields": list(PUBLIC_LIFECYCLE_SUBFIELDS),
        "forbidden_role_values": list(DOMAIN_ROLE_VALUES),
        "tiers": {
            "retriever": "visible page text + url + public domain descriptors",
            "reader": "retriever tier + rank + score",
            "verifier": "reader tier + public registry + DERIVED lifecycle observations",
            "evaluator": "experimenter tier: ground truth, page roles, meta channel",
        },
        "meta_channel_policy": (
            "x-aegis-entity-id / x-aegis-page-id are EVALUATOR-tier. CONTRACT.md Section 8 "
            "publishes them so scoring needs no manual reading; exposing them to the "
            "retriever or reader would trivialise the entity-resolution stage that RQ2 "
            "measures."
        ),
        "enforcement": [
            "whitelist construction (new registry fields excluded by default)",
            "recursive assert_no_leakage guard on field names and enum values",
            "explicit ExposureTier on every serialisation path",
        ],
        "authoritative_registry": {
            "artifact": "load_official_registry() -> OfficialRegistry",
            "tier": ExposureTier.VERIFIER.value,
            "contract_ref": "CONTRACT.md Section 3 (RQ3) evidence family 1; Section 10 baseline 9",
            "publishes": [
                "entity_id, canonical_name, category",
                "official_domain_id / official_domain (from domains.yaml owner_entity_id)",
                "authoritative address and phone (identity-consistency reference)",
                "published aliases (entity-resolution surface forms)",
            ],
            "withholds": list(OFFICIAL_REGISTRY_EXCLUSIONS),
            "rationale": OFFICIAL_REGISTRY_RATIONALE,
            "equivalence_note": (
                "owner_entity_id is non-null for exactly the official_site records, so this "
                "projection is informationally equivalent to role == 'official_site' and to "
                "nothing else. Every non-official domain remains indistinguishable, and no "
                "per-action third-party grant is exposed."
            ),
        },
        "site_index": {
            "artifact": "build_site_index() -> SiteIndex",
            "tier": ExposureTier.RETRIEVER.value,
            "publishes": ["doc_id -> url", "domain_id -> doc_ids"],
            "rationale": (
                "A crawler knows which pages it holds per host. Same field set as "
                "strip_manifest_entry, made addressable so a verifier can read the official "
                "site it did not happen to retrieve."
            ),
        },
        "delegation_evidence_policy": (
            "public_delegation_evidence() exists for provenance and for tests, but it names "
            "which third party holds which grant -- i.e. the answer to RQ3. No defense, "
            "ablation or baseline may import it, and tests/test_aegislink_and_baselines.py "
            "enforces that statically."
        ),
    }


__all__ = [
    "ALL_FORBIDDEN_FIELDS",
    "DOMAIN_ROLE_VALUES",
    "OFFICIAL_REGISTRY_EXCLUSIONS",
    "OFFICIAL_REGISTRY_RATIONALE",
    "OfficialRegistry",
    "OfficialRegistryRecord",
    "SiteIndex",
    "build_site_index",
    "load_official_registry",
    "EXPERIMENTER_ONLY_FIELDS",
    "EXPERIMENTER_ONLY_LABEL_FIELDS",
    "EXPERIMENTER_ONLY_LIFECYCLE_SUBFIELDS",
    "EXPERIMENTER_ONLY_MANIFEST_FIELDS",
    "EXPOSURE_VERSION",
    "PAGE_ROLE_VALUES",
    "PUBLIC_DOMAIN_FIELDS",
    "PUBLIC_LIFECYCLE_SUBFIELDS",
    "PUBLIC_TIERS",
    "ExposureTier",
    "LeakageError",
    "LifecycleObservation",
    "PublicDomainRegistry",
    "PublicDomainView",
    "assert_no_leakage",
    "derive_lifecycle_observation",
    "describe_firewall",
    "guarded",
    "load_public_domain_registry",
    "project_domain_record",
    "public_delegation_evidence",
    "strip_manifest_entry",
]
