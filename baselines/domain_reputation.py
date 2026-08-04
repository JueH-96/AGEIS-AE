"""B02 -- domain age and reputation rules. CONTRACT.md Section 10, item 2.

Reputation heuristics ask how *established* a host is: how long it has been observed, whether it
has changed hands, whether it lapsed, and whether it serves many listings or exactly one. All of
these are crawl-observable through :class:`~web_rag.exposure.LifecycleObservation` and the public
registry's ``shared`` flag.

Note on what this baseline gets for free
----------------------------------------
On this corpus a lifecycle record exists for exactly the 160 takeover domains, so "has a lifecycle
record" is a perfect predictor of "unauthorized" -- a degeneracy audited in
``results/evidence_family_identifiability.json``. This baseline is *allowed* to exploit it, and
deliberately so: it is what a reputation rule would actually do, and letting the baseline take the
free win makes the comparison harder for AegisLink rather than easier. AegisLink declines the same
shortcut (see ``aegislink/verifier.py``), so any advantage it shows is achieved without it.

The structural limit is the same as B01's: reputation is a property of a host, not of a triple. A
long-established shared booking provider scores well for every entity, including the 50% of
entity-provider pairs it holds no delegation for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aegislink.framework import InputKind, InputRestriction, VerificationContext, sigmoid
from baselines.common import BaseDefense

#: Crawl snapshot labels in chronological order; index doubles as an age proxy.
SNAPSHOT_ORDER: tuple[str, ...] = ("T0", "T1", "T2")


def snapshot_age(label: str | None) -> int:
    """Age in snapshots: 3 for a host first seen at ``T0``, 1 for one first seen at ``T2``.

    ``0`` means *unknown*, not *new*. That distinction matters here: in this corpus only the 160
    takeover domains carry a lifecycle record at all, so 1,630 of 1,790 hosts have no observed
    first-seen snapshot. An earlier version defaulted the unknown case to ``T0`` (maximum age),
    which silently asserted that every host without a record was long-established and made the age
    feature a constant. The unknown case is now carried as its own feature so the baseline's
    blindness is visible in its weights rather than hidden in a default.
    """
    if label is None:
        return 0
    try:
        return len(SNAPSHOT_ORDER) - SNAPSHOT_ORDER.index(label)
    except ValueError:
        return 0


@dataclass
class DomainReputation(BaseDefense):
    """Score a triple from host age, ownership stability and breadth of use."""

    defense_id: str = "B02_domain_reputation"
    label: str = "B02 domain age / reputation rules"
    contract_ref: str = "CONTRACT.md Section 10, item 2"
    restrictions: tuple[InputRestriction, ...] = field(
        default_factory=lambda: (
            InputRestriction(
                InputKind.PAGE_TEXT,
                "a reputation rule scores the host, not its content",
            ),
            InputRestriction(
                InputKind.OFFICIAL_BACKLINK_DIRECTIONAL,
                "no cross-source evidence by design",
            ),
            InputRestriction(
                InputKind.ACTION_TYPE,
                "host reputation is action-independent by construction",
            ),
        )
    )
    notes: str = (
        "Permitted to exploit the corpus's lifecycle degeneracy (a lifecycle record exists only "
        "for takeover domains). AegisLink declines that shortcut, so the comparison is "
        "conservative in the baseline's favour."
    )

    w_bias: float = 0.35
    w_age: float = 0.55
    w_age_unknown: float = -0.25
    w_expired: float = -2.90
    w_ownership_change: float = -2.40
    w_shared: float = 1.05
    w_unregistered: float = -0.80

    def probability(
        self, entity_id: str, domain_id: str, action: str, ctx: VerificationContext
    ) -> tuple[float, tuple[str, ...]]:
        obs = ctx.lifecycle(domain_id)
        registered = ctx.registry.domains.get(domain_id) is not None
        observed_first_seen = obs.first_seen_snapshot if obs else None
        age = snapshot_age(observed_first_seen)
        age_unknown = observed_first_seen is None
        expired = bool(obs and obs.expired)
        changed = bool(obs and obs.ownership_changed)
        shared = ctx.is_shared_host(domain_id)
        z = (
            self.w_bias
            + self.w_age * age
            + self.w_age_unknown * age_unknown
            + self.w_expired * expired
            + self.w_ownership_change * changed
            + self.w_shared * shared
            + self.w_unregistered * (0.0 if registered else 1.0)
        )
        reasons = (
            f"first_seen={observed_first_seen or 'unobserved'} (age={age}); expired={expired}; "
            f"ownership_changed={changed}; shared_host={shared}",
        )
        return sigmoid(z), reasons


def describe() -> dict[str, Any]:
    d = DomainReputation()
    return {
        **d.metadata().as_dict(),
        "weights": {
            "bias": d.w_bias,
            "age_per_snapshot": d.w_age,
            "age_unknown": d.w_age_unknown,
            "expired": d.w_expired,
            "ownership_change": d.w_ownership_change,
            "shared_host": d.w_shared,
            "unregistered_host": d.w_unregistered,
        },
        "snapshot_order": list(SNAPSHOT_ORDER),
        "knife_edge_note": (
            "w_age_unknown was -0.45 in a first pass, which put a plain registered non-shared host "
            "at p=0.475 -- just under the 0.5 cut -- so the rule rejected every official site and "
            "reported FRR 0.57 for a reason that was an artefact of a boundary, not of the "
            "heuristic. A reputation rule with no age signal has no basis to reject, so the weight "
            "was set so the unknown-age case lands just above the cut."
        ),
        "corpus_limitation": (
            "Only the 160 takeover domains carry a lifecycle record, so 1,630 of 1,790 hosts have "
            "no observed first-seen snapshot and the age feature is uninformative for them. "
            "Reported in results/evidence_family_identifiability.json; this baseline therefore "
            "reduces in practice to 'is it a takeover, and is it a shared provider'."
        ),
    }


__all__ = ["SNAPSHOT_ORDER", "DomainReputation", "describe", "snapshot_age"]
