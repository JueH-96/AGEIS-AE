"""B09 -- allow only the registry-listed official domain. CONTRACT.md Section 10, item 9.

The maximally conservative policy: present a link only when the host is the entity's own official
domain according to the authoritative registry. Everything else is refused, for every action.

Why it is the most important baseline in the suite
-------------------------------------------------
It is the one baseline whose ``UALER`` is zero by construction, so it defines the security ceiling.
Any claim that AegisLink reduces unauthorized exposure has to be read against it: beating B09 on
``UALER`` is impossible, and that is the point. What AegisLink must beat it on is ``ATPR``.

The corpus makes that a real contest rather than a formality. Every entity delegates to exactly two
third parties (one booking provider, one information directory), and those delegations are genuine
Section 5.2 authorizations. B09 rejects all of them, so its ``ATPR`` is 0 and its ``FRR`` on
authorized third-party links is 1.0 by construction. The preregistered gates make this concrete:
``min_required_baseline_atpr: 0.90`` and ``max_absolute_atpr_loss: 0.05`` mean a defense cannot buy
security by becoming B09.

This is the direct operationalisation of novelty claim 5 in CONTRACT.md Section 4.2, "explicit
preservation of authorized third-party services": B09 is what happens without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aegislink.framework import InputKind, InputRestriction, VerificationContext
from baselines.common import BaseDefense


@dataclass
class OfficialOnly(BaseDefense):
    """Allow the registry-listed official domain; refuse every third party."""

    defense_id: str = "B09_official_only"
    label: str = "B09 allow only the registry-listed official domain"
    contract_ref: str = "CONTRACT.md Section 10, item 9"
    restrictions: tuple[InputRestriction, ...] = field(
        default_factory=lambda: (
            InputRestriction(
                InputKind.PAGE_TEXT,
                "the rule is a registry lookup; page content cannot change its answer",
            ),
            InputRestriction(
                InputKind.OFFICIAL_BACKLINK_DIRECTIONAL,
                "third-party delegations are refused unread, which is the rule's definition",
            ),
            InputRestriction(
                InputKind.ACTION_TYPE,
                "the same host is allowed for every action; the rule has no action notion",
            ),
            InputRestriction(
                InputKind.SOURCE_CLUSTERING, "no corroboration is consulted"
            ),
        )
    )
    notes: str = (
        "UALER = 0 and ATPR = 0, both by construction. Defines the security ceiling and the "
        "utility floor; the preregistered ATPR gates exist to stop a defense collapsing to it."
    )

    def asserts_official(
        self, entity_id: str, domain_id: str, ctx: VerificationContext
    ) -> bool:
        """Every link this baseline presents is presented *because* it is the official domain."""
        return ctx.official_registry.official_domain_id(entity_id) == domain_id

    def probability(
        self, entity_id: str, domain_id: str, action: str, ctx: VerificationContext
    ) -> tuple[float, tuple[str, ...]]:
        official = ctx.official_registry.official_domain_id(entity_id)
        if official is None:
            return 0.0, (
                f"{entity_id} has no authoritative registry record, so nothing is allowed",
            )
        if domain_id == official:
            return 0.98, (f"registry names {domain_id} as the official domain of {entity_id}",)
        return 0.02, (
            f"{domain_id} is not the registry-listed official domain ({official}); refused "
            f"without reading the page",
        )


def describe() -> dict[str, Any]:
    return {
        **OfficialOnly().metadata().as_dict(),
        "by_construction": {
            "ualer": 0.0,
            "atpr": 0.0,
            "frr_on_authorized_third_party_links": 1.0,
        },
        "role_in_the_suite": (
            "Defines the security ceiling. AegisLink cannot beat it on UALER; the comparison that "
            "matters is ATPR at equal UALER."
        ),
        "contract_link": (
            "Direct operationalisation of CONTRACT.md Section 4.2 claim 5 (explicit preservation "
            "of authorized third-party services): B09 is the counterfactual without it. Bounded "
            "by gates min_required_baseline_atpr=0.90 and max_absolute_atpr_loss=0.05."
        ),
    }


__all__ = ["OfficialOnly", "describe"]
