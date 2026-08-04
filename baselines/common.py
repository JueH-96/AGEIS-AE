"""Shared machinery for the ten comparative baselines. CONTRACT.md Section 10.

The identical-evidence rule
--------------------------
Section 10: "Every baseline MUST receive identical evidence, splits, and attack budgets unless its
design explicitly prohibits an input type; deviations must be logged."

Mechanised, not promised:

* every baseline is handed the same :class:`~aegislink.framework.VerificationContext` object that
  AegisLink is handed, on the same frozen replay, at the same ``read_phase``;
* a baseline that by design ignores an input declares an
  :class:`~aegislink.framework.InputRestriction`, and
  :func:`baselines.registry.deviation_log` collects them into a machine-readable log;
* :class:`BaseDefense` gives every baseline the same probability-to-verdict mapping, so a
  difference between two baselines is a difference in their *signal*, never in their output policy.

Why every baseline emits a probability
--------------------------------------
Section 11 requires Brier score and calibration error "for authorization probability". A rule-based
baseline has no natural probability, so each one states an explicit, documented pseudo-probability
and declares ``calibrated=False``. Reporting a crude number and saying it is crude is more useful
than omitting the baseline from the calibration table, which would silently make the table a
comparison among only the probabilistic methods.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from aegislink.framework import (
    ACTION_RISK_ORDER,
    DefenseDecision,
    DefenseMetadata,
    Disposition,
    InputKind,
    InputRestriction,
    PlattScaling,
    VerificationContext,
    Verdict,
    action_risk_rank,
    normalize,
    sigmoid,
    slugify_name,
    word_tokens,
)

BASELINES_VERSION = "1.0"

#: One threshold for every baseline. Deliberately uniform: the baselines differ in what they
#: measure, and giving each its own tuned cut-off would turn the comparison into a tuning contest.
#: 0.5 is the natural cut for a probability.
BASELINE_DECISION_THRESHOLD = 0.5


# ======================================================================================
# Shared surface cues
# ======================================================================================
#: Hostname tokens a lexical URL heuristic would flag. Generic phishing folklore, deliberately not
#: tuned to this corpus: these are the tokens such rules use in the wild.
SUSPICIOUS_HOST_TOKENS: tuple[str, ...] = (
    "official", "login", "signin", "account", "accounts", "secure", "verify",
    "pay", "payment", "deposit", "settlement", "billing", "reserve", "reservations",
    "booking", "bookings", "archive", "old", "new", "support", "update", "confirm",
)

#: Words a generic phishing classifier keys on in page content.
CREDENTIAL_CUES: tuple[str, ...] = (
    "sign in", "log in", "login", "account area", "account holders", "members",
    "password", "credentials",
)
PAYMENT_CUES: tuple[str, ...] = (
    "deposit", "settlement", "balance", "balances", "payment", "outstanding amounts",
    "settled", "checkout", "invoice",
)
URGENCY_CUES: tuple[str, ...] = (
    "immediately", "as soon as possible", "urgent", "act now", "continue to",
)
_OFFICIAL_CLAIM_ANY_RE = re.compile(
    r"this is the official (?:website|site|page)\b"
    r"|official home page for\b"
    r"|official site and booking desk"
    r"|you have reached the official"
    r"|official online presence of",
    re.IGNORECASE,
)
_BACKLINK_CLAIM_RE = re.compile(
    r"listed by the official site|appointed by|partner of", re.IGNORECASE
)
_ADDRESS_RE = re.compile(r"Address:\s*(?P<addr>[^\n]{4,120}?)\s*(?=Telephone:|$)", re.IGNORECASE)
_PHONE_RE = re.compile(r"Telephone:\s*(?P<phone>\+?[0-9][0-9\-\s()]{5,24})", re.IGNORECASE)
_ACTION_CUE_RE: dict[str, re.Pattern[str]] = {
    "browse": re.compile(r"\bbrowse\b|opening hours|\bmenu\b|\bgallery\b|read (?:about|more)", re.I),
    "contact": re.compile(r"\bcontact\b|telephone|\bphone\b|enquir|get in touch", re.I),
    "book": re.compile(r"\bbook\b|\bbooking\b|reserv|\btable\b|availability", re.I),
    "login": re.compile(r"\blog ?in\b|\bsign ?in\b|account (?:area|holders)|members", re.I),
    "pay": re.compile(r"\bpay\b|payment|deposit|settle|balance|checkout|invoice", re.I),
}


def host_labels(hostname: str) -> list[str]:
    """Split a hostname into hyphen/dot-separated labels."""
    return [t for t in re.split(r"[.\-]", hostname.lower()) if t]


def page_text_for(ctx: VerificationContext, domain_id: str) -> str:
    """Visible text of the candidate on ``domain_id``, or ``""`` if it was not retrieved."""
    cand = ctx.candidate_by_domain(domain_id)
    return ctx.text(cand.doc_id) if cand is not None else ""


def claims_official(text: str) -> bool:
    return bool(_OFFICIAL_CLAIM_ANY_RE.search(text))


def claims_backlink(text: str) -> bool:
    return bool(_BACKLINK_CLAIM_RE.search(text))


def claims_action(text: str, action: str) -> bool:
    return bool(_ACTION_CUE_RE[action].search(text))


def identity_agreement(ctx: VerificationContext, entity_id: str, text: str) -> float | None:
    """Fraction of published identity fields that match the authoritative record.

    ``None`` when the page publishes nothing comparable, which is different from publishing
    something wrong -- a distinction several baselines need.
    """
    rec = ctx.official_registry.get(entity_id)
    if rec is None:
        return None
    checked = agree = 0
    if rec.address:
        m = _ADDRESS_RE.search(text)
        if m:
            checked += 1
            agree += int(normalize(m.group("addr")) == normalize(rec.address))
    if rec.phone:
        m = _PHONE_RE.search(text)
        if m:
            checked += 1
            agree += int(normalize(m.group("phone")) == normalize(rec.phone))
    return (agree / checked) if checked else None


def mentions_entity(ctx: VerificationContext, entity_id: str, text: str) -> bool:
    """Whether any published surface form of the entity appears in the text."""
    low = normalize(text)
    for f in ctx.official_registry.surface_forms(entity_id):
        if normalize(f) in low:
            return True
    return False


def references_domain(ctx: VerificationContext, doc_domain_id: str, target_domain_id: str) -> bool:
    """Whether the page on ``doc_domain_id`` references the host behind ``target_domain_id``.

    Delegates to :meth:`~aegislink.framework.VerificationContext.references_domain`, which checks
    outbound hyperlinks first. Written as a helper rather than inlined because every corroboration
    test in the suite needs it and an earlier version of each of them tested ``host in visible_text``
    -- which is never true, since the corpus names partners by display name and keeps the host in the
    ``href``. That defect silently reduced B05, B06, B07 and B10 to self-votes.
    """
    cand = ctx.candidate_by_domain(doc_domain_id)
    if cand is None:
        return False
    return ctx.references_domain(cand.doc_id, target_domain_id)


def count_corroborators(
    ctx: VerificationContext, entity_id: str, target_domain_id: str
) -> int:
    """Retrieved pages, other than the target's own, that name the entity and reference the target."""
    n = 0
    for d in ctx.candidate_domain_ids():
        if d == target_domain_id:
            continue
        text = page_text_for(ctx, d)
        if not text or not mentions_entity(ctx, entity_id, text):
            continue
        if references_domain(ctx, d, target_domain_id):
            n += 1
    return n


def entity_slug_in_host(ctx: VerificationContext, entity_id: str, domain_id: str) -> bool:
    """Whether the hostname embeds (a prefix of) the entity's name slug."""
    rec = ctx.official_registry.get(entity_id)
    if rec is None:
        return False
    probe = slugify_name(rec.canonical_name)[:12]
    return bool(probe) and probe in ctx.hostname(domain_id)


# ======================================================================================
# Base class
# ======================================================================================
@dataclass
class BaseDefense:
    """Common output policy for the baselines.

    A subclass implements :meth:`risk_probability` -- its estimate that the triple is authorized --
    and inherits the mapping to verdict and disposition. Sharing the mapping is what makes the
    comparison fair: any measured difference is a difference in signal, not in how a score was
    turned into a decision.
    """

    defense_id: str = "baseline"
    label: str = "baseline"
    contract_ref: str = "CONTRACT.md Section 10"
    restrictions: tuple[InputRestriction, ...] = ()
    threshold: float = BASELINE_DECISION_THRESHOLD
    calibration: PlattScaling = field(default_factory=PlattScaling)
    is_surrogate: bool = False
    surrogate_note: str = ""
    notes: str = ""

    # -- subclass hook ---------------------------------------------------------------
    def probability(
        self, entity_id: str, domain_id: str, action: str, ctx: VerificationContext
    ) -> tuple[float, tuple[str, ...]]:
        """Return ``(probability that the triple is authorized, reasons)``."""
        raise NotImplementedError

    # -- officialness ----------------------------------------------------------------
    def asserts_official(
        self, entity_id: str, domain_id: str, ctx: VerificationContext
    ) -> bool:
        """Whether presenting this link asserts the domain is the entity's OFFICIAL site.

        ``False`` by default, because most baselines have no notion of officialness at all. Their
        ``OSMR`` is then reported as null -- undefined, not zero. Overridden by the three baselines
        that do make the claim: B09 (its entire rule), and B04/B06/B10, which accept and propagate
        the page's own official self-description.
        """
        return False

    # -- shared policy ---------------------------------------------------------------
    def decide(
        self, entity_id: str, domain_id: str, action: str, ctx: VerificationContext
    ) -> DefenseDecision:
        action_risk_rank(action)  # fail closed on an unknown action
        p, reasons = self.probability(entity_id, domain_id, action, ctx)
        p = self.calibration.apply(min(max(p, 0.0), 1.0))
        if p >= self.threshold:
            verdict, disp = Verdict.VERIFIED, Disposition.PRESENT
        elif p >= self.threshold * 0.5:
            verdict, disp = Verdict.PLAUSIBLE, Disposition.WITHHOLD
        else:
            verdict, disp = Verdict.UNVERIFIED, Disposition.WITHHOLD
        margin = abs(p - self.threshold)
        confidence = "high" if margin >= 0.25 else ("medium" if margin >= 0.10 else "low")
        return DefenseDecision(
            entity_id=entity_id,
            domain_id=domain_id,
            action=action,
            verdict=verdict,
            probability=p,
            disposition=disp,
            confidence=confidence,
            reasons=reasons,
            calibrated=not self.calibration.is_identity,
            asserts_official=(
                disp is Disposition.PRESENT
                and self.asserts_official(entity_id, domain_id, ctx)
            ),
        )

    def with_calibration(self, cal: PlattScaling) -> "BaseDefense":
        self.calibration = cal
        return self

    def metadata(self) -> DefenseMetadata:
        return DefenseMetadata(
            defense_id=self.defense_id,
            family="baseline",
            label=self.label,
            contract_ref=self.contract_ref,
            restrictions=self.restrictions,
            notes=self.notes,
            is_surrogate=self.is_surrogate,
            surrogate_note=self.surrogate_note,
        )


def describe_baseline_common() -> dict[str, Any]:
    return {
        "baselines_version": BASELINES_VERSION,
        "contract_ref": "CONTRACT.md Section 10",
        "shared_decision_threshold": BASELINE_DECISION_THRESHOLD,
        "shared_threshold_rationale": (
            "One cut-off for every baseline. Per-baseline tuning would make the comparison a "
            "tuning contest rather than a comparison of signals; 0.5 is the natural cut for a "
            "probability."
        ),
        "identical_evidence_mechanism": (
            "Every baseline receives the same VerificationContext instance as AegisLink, on the "
            "same frozen replay at the same read_phase. Design-mandated blindness is declared as "
            "an InputRestriction and collected into the deviation log."
        ),
        "probability_requirement": (
            "Rule-based baselines emit an explicit pseudo-probability and declare calibrated=False "
            "so Section 11's Brier and calibration table can include every defense."
        ),
        "shared_cue_lexicons": {
            "suspicious_host_tokens": list(SUSPICIOUS_HOST_TOKENS),
            "credential_cues": list(CREDENTIAL_CUES),
            "payment_cues": list(PAYMENT_CUES),
            "urgency_cues": list(URGENCY_CUES),
        },
    }


__all__ = [
    "BASELINES_VERSION",
    "BASELINE_DECISION_THRESHOLD",
    "CREDENTIAL_CUES",
    "PAYMENT_CUES",
    "SUSPICIOUS_HOST_TOKENS",
    "URGENCY_CUES",
    "BaseDefense",
    "claims_action",
    "claims_backlink",
    "claims_official",
    "count_corroborators",
    "describe_baseline_common",
    "entity_slug_in_host",
    "host_labels",
    "identity_agreement",
    "mentions_entity",
    "page_text_for",
    "references_domain",
]
