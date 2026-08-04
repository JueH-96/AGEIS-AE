"""B05 -- unweighted source-count majority vote. CONTRACT.md Section 10, item 5.

One page, one vote. A domain is authorized for an action if a majority of the retrieved pages that
discuss the entity support that domain for that action -- where "support" means the domain's own page
offers the action, or another page references the domain.

This is the baseline the corpus's ``corroborating_sources`` factor (1 / 3 / 5) exists to break. It
has no notion of source independence, so a campaign of five lexically diverse pages built for one
listing carries five times the weight of the entity's own official site. AegisLink's clustering
component is the direct answer to it, and ``ablation_source_count_voting`` is the same idea applied
inside the full method.

Distinct from ``ablation_source_count_voting``: that ablation keeps AegisLink's evidence extraction
and swaps only the inference rule, so the two together separate "voting is the wrong rule" from
"voting on the wrong evidence".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aegislink.framework import InputKind, InputRestriction, VerificationContext
from baselines.common import (
    BaseDefense,
    claims_action,
    mentions_entity,
    page_text_for,
    references_domain,
)


@dataclass
class SourceCountMajority(BaseDefense):
    """Majority vote over retrieved pages, every page weighted equally."""

    defense_id: str = "B05_source_count_majority"
    label: str = "B05 unweighted source-count majority vote"
    contract_ref: str = "CONTRACT.md Section 10, item 5"
    restrictions: tuple[InputRestriction, ...] = field(
        default_factory=lambda: (
            InputRestriction(
                InputKind.SOURCE_CLUSTERING,
                "unweighted by definition: assessing independence would make it a different rule",
            ),
            InputRestriction(
                InputKind.REGISTRY_OFFICIAL,
                "the vote is over retrieved pages; consulting the registry would add an "
                "authority notion the rule does not have",
            ),
            InputRestriction(
                InputKind.OFFICIAL_BACKLINK_DIRECTIONAL,
                "no directional verification by design",
            ),
        )
    )
    notes: str = (
        "Prediction before measurement: high UALER on the 3- and 5-corroborator attack arms, "
        "because a one-listing campaign outvotes the entity's own site."
    )

    def support_counts(
        self, entity_id: str, action: str, ctx: VerificationContext
    ) -> dict[str, int]:
        """Votes for each candidate domain: how many retrieved pages back it for this action."""
        counts: dict[str, int] = {}
        for target in ctx.candidate_domain_ids():
            votes = 0
            for voter in ctx.candidate_domain_ids():
                text = page_text_for(ctx, voter)
                if not text or not mentions_entity(ctx, entity_id, text):
                    continue
                if voter == target:
                    if claims_action(text, action):
                        votes += 1  # self-nomination
                elif references_domain(ctx, voter, target):
                    votes += 1
            counts[target] = votes
        return counts

    def probability(
        self, entity_id: str, domain_id: str, action: str, ctx: VerificationContext
    ) -> tuple[float, tuple[str, ...]]:
        """Plurality among the candidates, not a share of the whole electorate.

        The share-of-electorate reading was implemented first and measured: with ten candidates the
        largest share any domain reached was 0.6 and the median was 0.125, so no domain ever cleared
        a threshold and the baseline collapsed to "reject everything". A defense that always abstains
        has UALER 0 for the wrong reason and never exercises the Sybil vulnerability this baseline
        exists to expose.

        A vote is a contest, so the operative question is which candidate the sources *favour*. The
        winner is presented; its margin sets the confidence. Five co-registered corroborators now
        outvote one official page, which is exactly the failure mode.
        """
        counts = self.support_counts(entity_id, action, ctx)
        total = sum(counts.values())
        if not total:
            return 0.0, ("no retrieved page backs any candidate for this action",)
        mine = counts.get(domain_id, 0)
        share = mine / total
        winner, top = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
        if domain_id == winner and mine > 0:
            p = min(1.0, 0.55 + 0.45 * share)
        else:
            p = 0.45 * share
        return p, (
            f"{mine} of {total} votes (share={share:.3f}); plurality winner={winner} "
            f"with {top} vote(s)",
        )


def describe() -> dict[str, Any]:
    return {
        **SourceCountMajority().metadata().as_dict(),
        "rule": (
            "plurality among the candidates: a page votes for the target if it is the target's own "
            "page offering the action, or it names the entity and links to the target host. The "
            "plurality winner is presented."
        ),
        "why_plurality_not_share": (
            "The share-of-electorate reading was measured first and never cleared a threshold "
            "(max share 0.6, median 0.125 over ten candidates), so the baseline collapsed to "
            "'reject everything' -- UALER 0 for the wrong reason, and no exposure to the Sybil "
            "arm it exists to expose."
        ),
        "known_weakness": (
            "no independence notion, so the Section 7 corroborating_sources factor (1/3/5) "
            "translates directly into vote share"
        ),
        "contrast_with_ablation": (
            "ablation_source_count_voting keeps AegisLink's extraction and swaps only the "
            "inference rule; together the two separate a wrong rule from wrong evidence"
        ),
    }


__all__ = ["SourceCountMajority", "describe"]
