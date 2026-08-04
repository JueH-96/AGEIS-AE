"""B06 -- provenance-weighted reranking. CONTRACT.md Section 10, item 6.

Rather than counting sources equally (B05), weight each source by how good its provenance looks, and
present the top-ranked domain if its weighted support clears the bar. Provenance weight comes from
signals a reranker can observe: whether the host is registry-known, whether it serves many listings,
how long it has been observed, its retrieval rank, and whether it carries an endorsement claim.

Where it breaks, and why that is the interesting part
----------------------------------------------------
"Carries an endorsement claim" is a *presence* test. This corpus is built so that presence and
direction come apart: an authorized partner page and an impersonating page emit the byte-identical
line "Listed by the official site: <Entity>", the forged one pointing at the entity's genuine
official host so that even a link-target check passes. A reranker that treats the claim as
provenance therefore upgrades the forgery for the same reason it upgrades the genuine partner.

That is the precise gap AegisLink closes, and it is why B06 and B10 both appear in the suite: they
are the two strongest provenance-shaped defenses, and both are non-directional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aegislink.framework import InputKind, InputRestriction, VerificationContext
from baselines.common import (
    BaseDefense,
    claims_action,
    claims_backlink,
    claims_official,
    identity_agreement,
    mentions_entity,
    page_text_for,
)
from baselines.domain_reputation import snapshot_age


@dataclass
class ProvenanceReranker(BaseDefense):
    """Provenance-weighted support, normalised over the candidate set."""

    defense_id: str = "B06_provenance_reranker"
    label: str = "B06 provenance-weighted reranking"
    contract_ref: str = "CONTRACT.md Section 10, item 6"
    restrictions: tuple[InputRestriction, ...] = field(
        default_factory=lambda: (
            InputRestriction(
                InputKind.OFFICIAL_BACKLINK_DIRECTIONAL,
                "provenance weighting tests whether an endorsement claim is PRESENT, not whether "
                "the endorsing site actually publishes it; directionality is outside the design",
            ),
            InputRestriction(
                InputKind.SOURCE_CLUSTERING,
                "sources are weighted, not clustered",
            ),
        )
    )
    notes: str = (
        "Prediction before measurement: better than B05 on ATPR, still exposed on the "
        "official_backlink attack arm, because a forged endorsement is upgraded exactly like a "
        "genuine one."
    )

    w_registry_known: float = 1.00
    w_shared_host: float = 0.90
    w_age: float = 0.30
    w_rank: float = 0.55
    w_endorsement_claim: float = 0.80
    w_identity_match: float = 0.55
    w_identity_conflict: float = -1.00
    w_official_claim: float = 0.35

    def asserts_official(
        self, entity_id: str, domain_id: str, ctx: VerificationContext
    ) -> bool:
        """Officialness is one of the provenance signals this reranker credits, so a presented
        page carrying an official self-claim is an official claim the defense has endorsed."""
        official = ctx.official_registry.official_domain_id(entity_id)
        return domain_id == official or claims_official(page_text_for(ctx, domain_id))

    def provenance_weight(
        self, entity_id: str, domain_id: str, ctx: VerificationContext
    ) -> float:
        text = page_text_for(ctx, domain_id)
        obs = ctx.lifecycle(domain_id)
        cand = ctx.candidate_by_domain(domain_id)
        agree = identity_agreement(ctx, entity_id, text)
        n = max(1, len(ctx.candidates))
        rank_credit = (n - (cand.rank - 1)) / n if cand is not None else 0.0
        w = (
            self.w_registry_known * (1.0 if ctx.registry.domains.get(domain_id) else 0.0)
            + self.w_shared_host * (1.0 if ctx.is_shared_host(domain_id) else 0.0)
            + self.w_age * snapshot_age(obs.first_seen_snapshot if obs else None)
            + self.w_rank * rank_credit
            + self.w_endorsement_claim * (1.0 if claims_backlink(text) else 0.0)
            + self.w_official_claim * (1.0 if claims_official(text) else 0.0)
            + (
                self.w_identity_match
                if (agree is not None and agree >= 1.0)
                else (self.w_identity_conflict if agree is not None else 0.0)
            )
        )
        return max(0.0, w)

    def probability(
        self, entity_id: str, domain_id: str, action: str, ctx: VerificationContext
    ) -> tuple[float, tuple[str, ...]]:
        # Weighted support for every candidate that offers the requested action, then normalise:
        # the reranked leader wins, so the score is the target's share of provenance weight.
        weights: dict[str, float] = {}
        for d in ctx.candidate_domain_ids():
            text = page_text_for(ctx, d)
            if not text or not mentions_entity(ctx, entity_id, text):
                continue
            if not claims_action(text, action):
                continue
            weights[d] = self.provenance_weight(entity_id, d, ctx)
        total = sum(weights.values())
        if not total:
            return 0.0, ("no candidate offers the requested action for this entity",)
        share = weights.get(domain_id, 0.0) / total
        best = max(weights.items(), key=lambda kv: (kv[1], kv[0]))
        # A share is diluted by the number of contenders, so the decision is expressed as
        # "is this the reranked leader?" smoothed by its share -- the behaviour of a reranker that
        # presents its top result.
        p = 0.55 + 0.45 * share if best[0] == domain_id else 0.45 * share
        return min(1.0, p), (
            f"provenance weight {weights.get(domain_id, 0.0):.2f} of {total:.2f} "
            f"(share={share:.3f}); reranked leader={best[0]}",
        )


def describe() -> dict[str, Any]:
    d = ProvenanceReranker()
    return {
        **d.metadata().as_dict(),
        "weights": {
            "registry_known": d.w_registry_known,
            "shared_host": d.w_shared_host,
            "age_per_snapshot": d.w_age,
            "retrieval_rank": d.w_rank,
            "endorsement_claim_present": d.w_endorsement_claim,
            "official_self_claim": d.w_official_claim,
            "identity_match": d.w_identity_match,
            "identity_conflict": d.w_identity_conflict,
        },
        "structural_gap": (
            "Treats the presence of an endorsement claim as provenance. The corpus emits the "
            "byte-identical claim on authorized and impersonating pages, with the forged link "
            "pointing at the genuine official host, so presence cannot separate them."
        ),
    }


__all__ = ["ProvenanceReranker", "describe"]
