"""B04 -- LLM-as-a-judge. CONTRACT.md Section 10, item 4.

LOGGED DEVIATION
----------------
This baseline is a **deterministic rubric surrogate**, not a language model. The substitution is
recorded in ``results/baseline_deviation_log.json`` via ``is_surrogate=True``, and it is the only
deviation in the suite that changes a baseline's *nature* rather than its inputs.

Why:

* Section 15 requires a clean machine to reproduce every primary table from one command, with a
  recorded model checksum, tokenizer checksum and decoding configuration. A hosted model satisfies
  none of those; a pinned open-weight checkpoint would satisfy them but is a Step 5 deliverable
  (``results/web_rag_environment.json`` already records ``llm-reader: Step 4 (pinned open-weight)``
  as pending).
* The frozen replay is 1,008 queries x 10 candidates x 5 actions. A judged-per-triple LLM baseline
  is ~50k generations per configuration, which is not affordable to run repeatedly during
  development and would make the ablation sweep non-reproducible in practice.

What the surrogate does, and why it is not a straw man
------------------------------------------------------
It implements the rubric an instructed judge plausibly applies: read the page, weigh how
authoritatively it presents itself, whether its identity details check out, whether it says the
official site vouches for it, and how many other retrieved pages agree. It takes those claims
largely at face value, discounting for internal inconsistency.

That is a *strong* baseline on this corpus, and it fails for a specific, reportable reason rather
than through weakness: every one of its inputs is a property of the page being judged, and the page
is exactly what the attacker controls. An impersonating page copies the true address and phone,
asserts official standing, states "Listed by the official site", and is corroborated by up to five
Sybil pages -- so it satisfies the entire rubric. The judge cannot fetch the official site and check
the direction of the claim, which is the one thing that separates the forged case from the genuine
one.

Because it is deliberately credulous about page self-description, it is expected to be one of the
*worst* baselines on UALER while scoring well on ATPR. Both halves of that prediction are recorded
before measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aegislink.framework import (
    InputKind,
    InputRestriction,
    VerificationContext,
    normalize,
    sigmoid,
)
from baselines.common import (
    BaseDefense,
    claims_action,
    claims_backlink,
    claims_official,
    count_corroborators,
    identity_agreement,
    mentions_entity,
    page_text_for,
)

#: Rubric items and their credit, as an instructed judge would weigh them.
JUDGE_RUBRIC: dict[str, float] = {
    "bias": -1.90,
    "names_the_entity": 1.30,
    "presents_as_official": 1.55,
    "asserts_official_endorsement": 1.45,
    "identity_fields_match": 1.25,
    "identity_fields_conflict": -1.85,
    "offers_the_requested_action": 1.10,
    "corroborated_by_other_pages": 0.42,
    "internally_inconsistent": -1.20,
}


@dataclass
class LlmAsJudge(BaseDefense):
    """Deterministic surrogate of an instructed LLM judge scoring one page at a time."""

    defense_id: str = "B04_llm_as_judge"
    label: str = "B04 LLM-as-a-judge (deterministic rubric surrogate)"
    contract_ref: str = "CONTRACT.md Section 10, item 4"
    is_surrogate: bool = True
    surrogate_note: str = (
        "Not a language model. A deterministic rubric standing in for an instructed judge, because "
        "Section 15 requires a recorded model/tokenizer checksum and one-command reproduction, and "
        "a per-triple LLM judge is ~50k generations per configuration. Replacing it with a pinned "
        "open-weight judge is a Step 5 deliverable and will invalidate this row, not the method."
    )
    restrictions: tuple[InputRestriction, ...] = field(
        default_factory=lambda: (
            InputRestriction(
                InputKind.OFFICIAL_BACKLINK_DIRECTIONAL,
                "the judge reads the page it is judging; it cannot fetch the official site to "
                "check whether the endorsement it claims actually exists",
            ),
            InputRestriction(
                InputKind.SOURCE_CLUSTERING,
                "corroboration is counted, not assessed for independence",
            ),
        )
    )
    notes: str = (
        "Prediction before measurement: among the worst baselines on UALER (credulous about page "
        "self-description) while strong on ATPR."
    )
    rubric: dict[str, float] = field(default_factory=lambda: dict(JUDGE_RUBRIC))

    def asserts_official(
        self, entity_id: str, domain_id: str, ctx: VerificationContext
    ) -> bool:
        """The judge takes the page's official self-description at face value, so presenting a
        page that claims official standing propagates that claim."""
        return claims_official(page_text_for(ctx, domain_id))

    def features(
        self, entity_id: str, domain_id: str, action: str, ctx: VerificationContext
    ) -> dict[str, float]:
        text = page_text_for(ctx, domain_id)
        agree = identity_agreement(ctx, entity_id, text)
        names = mentions_entity(ctx, entity_id, text)
        official = claims_official(text)
        endorsed = claims_backlink(text)

        # Corroboration: other retrieved pages that name the entity and reference this host.
        # Counted through link structure, because visible text never carries a hostname.
        corroborators = count_corroborators(ctx, entity_id, domain_id)

        # A judge would notice a page that claims to be the official site *and* claims the official
        # site endorses it -- those cannot both be true of the same page.
        inconsistent = official and endorsed
        return {
            "bias": 1.0,
            "names_the_entity": 1.0 if names else 0.0,
            "presents_as_official": 1.0 if official else 0.0,
            "asserts_official_endorsement": 1.0 if endorsed else 0.0,
            "identity_fields_match": 1.0 if (agree is not None and agree >= 1.0) else 0.0,
            "identity_fields_conflict": 1.0 if (agree is not None and agree < 1.0) else 0.0,
            "offers_the_requested_action": 1.0 if claims_action(text, action) else 0.0,
            "corroborated_by_other_pages": float(corroborators),
            "internally_inconsistent": 1.0 if inconsistent else 0.0,
        }

    def probability(
        self, entity_id: str, domain_id: str, action: str, ctx: VerificationContext
    ) -> tuple[float, tuple[str, ...]]:
        feats = self.features(entity_id, domain_id, action, ctx)
        z = sum(self.rubric.get(k, 0.0) * v for k, v in feats.items())
        fired = [k for k, v in feats.items() if k != "bias" and v]
        return sigmoid(z), (f"rubric items satisfied: {', '.join(fired) or 'none'}",)


def describe() -> dict[str, Any]:
    d = LlmAsJudge()
    return {
        **d.metadata().as_dict(),
        "rubric": dict(JUDGE_RUBRIC),
        "deviation": {
            "kind": "surrogate_substitution",
            "declared": True,
            "reason": d.surrogate_note,
            "remediation": "replace with a pinned open-weight judge in Step 5 and re-run this row",
        },
        "why_it_fails_structurally": (
            "Every rubric input is a property of the page under judgement, and the page is what "
            "the attacker controls. A forged 'Listed by the official site' line satisfies the "
            "endorsement item exactly as a genuine one does."
        ),
    }


__all__ = ["JUDGE_RUBRIC", "LlmAsJudge", "describe"]
