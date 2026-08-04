"""B08 -- reject all ``login`` and ``pay`` links. CONTRACT.md Section 10, item 8.

A policy baseline, not a detector. It hard-blocks the two highest-risk actions and is otherwise
undefended: for ``browse``, ``contact`` and ``book`` it presents the top-ranked candidate that offers
the action, exactly as a reader with no verification would.

Why the suite needs it
----------------------
It marks the security-utility frontier's degenerate corner and makes two things measurable that no
detector baseline can:

* it drives ``UALER`` on ``login``/``pay`` to zero *by construction*, so any defense that fails to
  beat zero there is not earning its complexity;
* it leaves ``book`` completely exposed, which shows that "block the scary verbs" is not a strategy:
  ``book`` is a high-risk action in the preregistered pool (``high_risk_action_pool``) and a
  misbound reservation endpoint is a real harm.

Its ``FRR`` on ``login``/``pay`` is 1.0 by construction: every genuinely authorized login and payment
route is rejected along with the forged ones. That is the cost the frontier plot has to show.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aegislink.framework import (
    HIGH_RISK_ACTIONS,
    InputKind,
    InputRestriction,
    VerificationContext,
)
from baselines.common import BaseDefense, claims_action, mentions_entity, page_text_for

#: The two actions the contract names for hard rejection. ``book`` is deliberately NOT included:
#: item 8 says "reject all login and pay links", and widening it would quietly turn a stated
#: baseline into a stronger one.
BLOCKED_ACTIONS: frozenset[str] = frozenset({"login", "pay"})
assert BLOCKED_ACTIONS < HIGH_RISK_ACTIONS, (
    "the blocked set must be a strict subset of the preregistered high-risk pool; that gap "
    "(book) is what makes this baseline's limitation visible"
)


@dataclass
class RejectHighRisk(BaseDefense):
    """Block ``login`` and ``pay`` outright; undefended elsewhere."""

    defense_id: str = "B08_reject_high_risk"
    label: str = "B08 reject all login and pay links"
    contract_ref: str = "CONTRACT.md Section 10, item 8"
    restrictions: tuple[InputRestriction, ...] = field(
        default_factory=lambda: (
            InputRestriction(
                InputKind.PAGE_TEXT,
                "for login/pay the decision is unconditional, so no evidence is consulted",
            ),
            InputRestriction(
                InputKind.REGISTRY_OFFICIAL,
                "a policy baseline: it verifies nothing, it only refuses",
            ),
            InputRestriction(
                InputKind.OFFICIAL_BACKLINK_DIRECTIONAL,
                "no verification of any kind by design",
            ),
        )
    )
    notes: str = (
        "FRR on login/pay is 1.0 by construction, and book is left fully exposed although it is in "
        "the preregistered high-risk pool. Marks the degenerate corner of the Pareto frontier."
    )

    def probability(
        self, entity_id: str, domain_id: str, action: str, ctx: VerificationContext
    ) -> tuple[float, tuple[str, ...]]:
        if action in BLOCKED_ACTIONS:
            return 0.0, (f"action {action!r} is unconditionally blocked",)

        # Undefended elsewhere: the highest-ranked candidate that offers the action wins.
        for cand in sorted(ctx.candidates, key=lambda c: c.rank):
            text = page_text_for(ctx, cand.domain_id)
            if not text or not mentions_entity(ctx, entity_id, text):
                continue
            if claims_action(text, action):
                if cand.domain_id == domain_id:
                    return 0.95, (
                        f"undefended for {action!r}: highest-ranked page offering it (rank "
                        f"{cand.rank})",
                    )
                return 0.05, (
                    f"undefended for {action!r}: outranked by {cand.domain_id} at rank {cand.rank}",
                )
        return 0.05, (f"no retrieved page offers {action!r} for this entity",)


def describe() -> dict[str, Any]:
    return {
        **RejectHighRisk().metadata().as_dict(),
        "blocked_actions": sorted(BLOCKED_ACTIONS),
        "unblocked_high_risk_actions": sorted(HIGH_RISK_ACTIONS - BLOCKED_ACTIONS),
        "behaviour_when_unblocked": "undefended: presents the top-ranked page offering the action",
        "by_construction": {
            "ualer_on_login_and_pay": 0.0,
            "frr_on_login_and_pay": 1.0,
        },
        "why_included": (
            "Sets the floor any real defense must beat on login/pay, and exposes that blocking "
            "the two scariest verbs leaves book -- also in the preregistered high-risk pool -- "
            "entirely open."
        ),
    }


__all__ = ["BLOCKED_ACTIONS", "RejectHighRisk", "describe"]
