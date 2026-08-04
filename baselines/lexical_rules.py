"""B01 -- lexical URL risk rules. CONTRACT.md Section 10, item 1.

The classic first line of defense: judge a link by how its URL looks. Hyphen count, hostname
length, label count, and a token blocklist ("official", "login", "secure", "pay", ...).

Why it is a real baseline and not a straw man on this corpus: the adversarial hosts genuinely do
look suspicious (``harbour-lantern-bistro-official.test``, ``harbour-lantern-bistro-deposit.test``),
so the heuristic has real signal. Its structural limit is that a URL carries no relation to an
*entity*: ``cinderpost-reserve.test`` looks equally clean whether or not this particular business
appointed it, and the entity's own official host embeds the business name and so looks equally
suspicious to a token blocklist. Entity relativity is invisible to it by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aegislink.framework import InputKind, InputRestriction, VerificationContext, sigmoid
from baselines.common import (
    SUSPICIOUS_HOST_TOKENS,
    BaseDefense,
    host_labels,
)


@dataclass
class LexicalUrlRules(BaseDefense):
    """Score a triple from URL surface features alone."""

    defense_id: str = "B01_lexical_url_rules"
    label: str = "B01 lexical URL risk rules"
    contract_ref: str = "CONTRACT.md Section 10, item 1"
    restrictions: tuple[InputRestriction, ...] = field(
        default_factory=lambda: (
            InputRestriction(
                InputKind.PAGE_TEXT,
                "a lexical URL rule inspects the URL only; reading page content would make it a "
                "different method",
            ),
            InputRestriction(
                InputKind.REGISTRY_OFFICIAL,
                "no registry lookup: the heuristic is defined over URL surface form alone",
            ),
            InputRestriction(
                InputKind.OFFICIAL_BACKLINK_DIRECTIONAL,
                "no cross-source evidence by design",
            ),
        )
    )
    notes: str = (
        "Has genuine signal on adversarial hostnames but cannot represent entity relativity: the "
        "same URL scores identically for an entity that appointed it and one that did not."
    )

    #: Weights on the surface features. Chosen to reproduce the shape of published URL heuristics
    #: (length and token penalties dominate), not fitted to this corpus.
    w_bias: float = 2.60
    w_suspicious_token: float = -1.10
    w_hyphens: float = -0.34
    w_length: float = -0.020
    w_labels: float = -0.45

    def probability(
        self, entity_id: str, domain_id: str, action: str, ctx: VerificationContext
    ) -> tuple[float, tuple[str, ...]]:
        host = ctx.hostname(domain_id)
        labels = host_labels(host)
        tokens = [t for t in labels if t in SUSPICIOUS_HOST_TOKENS]
        n_hyphen = host.count("-")
        z = (
            self.w_bias
            + self.w_suspicious_token * len(tokens)
            + self.w_hyphens * n_hyphen
            + self.w_length * len(host)
            + self.w_labels * max(0, host.count(".") - 1)
        )
        reasons = (
            f"host={host}; suspicious tokens={sorted(tokens) or 'none'}; "
            f"hyphens={n_hyphen}; length={len(host)}",
        )
        return sigmoid(z), reasons


def describe() -> dict[str, Any]:
    d = LexicalUrlRules()
    return {
        **d.metadata().as_dict(),
        "weights": {
            "bias": d.w_bias,
            "suspicious_token": d.w_suspicious_token,
            "hyphen": d.w_hyphens,
            "length_char": d.w_length,
            "extra_label": d.w_labels,
        },
        "blocklist_size": len(SUSPICIOUS_HOST_TOKENS),
    }


__all__ = ["LexicalUrlRules", "describe"]
