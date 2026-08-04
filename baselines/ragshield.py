"""B10 -- RAGShield-style provenance-verified defense. CONTRACT.md Section 10, item 10.

Prior work: Patil, arXiv:2604.00387 (provenance-based defense-in-depth), listed in CONTRACT.md
Section 4.1 as work that cannot be claimed as novel. Item 10 asks for "the strongest reproducible
related-work defense applicable to this setting", and this is it: of the eight priors, RAGShield is
the only one whose mechanism transfers directly to deciding whether a retrieved link may be
presented.

The mechanism, as a defense-in-depth chain
------------------------------------------
A link passes only if every layer passes:

1. **Provenance chain exists** -- the host is registry-known, or is referenced by a registry-known
   host in the retrieved set.
2. **Identity attestation** -- the published identity fields agree with the authoritative record.
3. **Endorsement** -- the page carries an official endorsement claim, or is the official domain.
4. **Independent corroboration** -- at least ``min_corroborators`` other retrieved pages support it.
5. **Lifecycle integrity** -- the host has not lapsed or changed registrant.

Conjunctive by design: defense-in-depth means every layer must hold, so it is strictly stronger than
any single one of B01-B07 and is the hardest baseline for AegisLink to beat.

The one gap, stated precisely
-----------------------------
Layer 3 tests that an endorsement is *asserted on the page being judged*. It does not fetch the
endorsing site to ask whether the endorsement is published there. On this corpus that distinction is
the whole attack: the forged line is byte-identical to the genuine one and its link points at the
entity's real official host, so it passes layer 3, and copied identity fields carry it through layer
2, while the Sybil arm supplies layer 4.

So B10 and AegisLink differ in exactly one respect -- the direction in which backlink evidence is
read -- which makes the B10-vs-AegisLink contrast the cleanest available test of RQ3's central claim.
Implemented from the described mechanism, not from released code, and labelled RAGShield-STYLE for
that reason.
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
    references_domain,
)


@dataclass
class RagShieldDefense(BaseDefense):
    """Conjunctive provenance chain over the retrieved evidence."""

    defense_id: str = "B10_ragshield_defense"
    label: str = "B10 RAGShield-style provenance-verified defense"
    contract_ref: str = "CONTRACT.md Section 10, item 10; prior work arXiv:2604.00387"
    restrictions: tuple[InputRestriction, ...] = field(
        default_factory=lambda: (
            InputRestriction(
                InputKind.OFFICIAL_BACKLINK_DIRECTIONAL,
                "provenance is verified as ASSERTED on the judged page; the endorsing site is not "
                "fetched to confirm the assertion. This single gap is the B10-vs-AegisLink "
                "contrast.",
            ),
            InputRestriction(
                InputKind.SOURCE_CLUSTERING,
                "corroborators are counted, not tested for mutual independence",
            ),
        )
    )
    notes: str = (
        "Strongest baseline in the suite: conjunctive, so strictly stronger than any single "
        "B01-B07 layer. Prediction before measurement: strong on ATPR, exposed on the "
        "official_backlink attack arm."
    )
    is_surrogate: bool = True
    surrogate_note: str = (
        "Implemented from the provenance-chain mechanism described in arXiv:2604.00387, not from "
        "released code. Labelled RAGShield-STYLE; it is not a reproduction of the published system."
    )

    min_corroborators: int = 1

    def asserts_official(
        self, entity_id: str, domain_id: str, ctx: VerificationContext
    ) -> bool:
        """Layer 3 accepts the page's own endorsement/official claim, so a passed link on a page
        claiming official standing is an official claim the chain has certified."""
        official = ctx.official_registry.official_domain_id(entity_id)
        return domain_id == official or claims_official(page_text_for(ctx, domain_id))

    def layers(
        self, entity_id: str, domain_id: str, action: str, ctx: VerificationContext
    ) -> dict[str, bool]:
        text = page_text_for(ctx, domain_id)
        official = ctx.official_registry.official_domain_id(entity_id)
        is_official = official is not None and domain_id == official
        host = ctx.hostname(domain_id)

        # Layer 1: a provenance chain into the registry. Layer 4: independent-looking
        # corroboration. Both need the same walk over the retrieved set, so it is done once.
        #
        # Reference detection goes through ctx.references_domain, i.e. outbound hyperlinks. An
        # earlier version tested `host in visible_text`, which is never true in this corpus (partners
        # are named by display name; the host lives in the href), so this layer failed on 100% of
        # authorized third parties and B10 was indistinguishable from B09.
        registry_known = ctx.registry.domains.get(domain_id) is not None
        referenced_by_known = False
        corroborators = 0
        for d in ctx.candidate_domain_ids():
            if d == domain_id:
                continue
            other = page_text_for(ctx, d)
            if not other or not mentions_entity(ctx, entity_id, other):
                continue
            if references_domain(ctx, d, domain_id):
                corroborators += 1
                if ctx.registry.domains.get(d) is not None:
                    referenced_by_known = True

        agree = identity_agreement(ctx, entity_id, text)
        obs = ctx.lifecycle(domain_id)

        return {
            "provenance_chain": bool(registry_known or referenced_by_known),
            "identity_attested": bool(agree is not None and agree >= 1.0) or is_official,
            "endorsed": bool(is_official or claims_backlink(text) or claims_official(text)),
            "corroborated": corroborators >= self.min_corroborators or is_official,
            "lifecycle_intact": not (obs is not None and (obs.expired or obs.ownership_changed)),
            "offers_action": bool(is_official or claims_action(text, action)),
        }

    def probability(
        self, entity_id: str, domain_id: str, action: str, ctx: VerificationContext
    ) -> tuple[float, tuple[str, ...]]:
        layers = self.layers(entity_id, domain_id, action, ctx)
        failed = sorted(k for k, ok in layers.items() if not ok)
        if failed:
            # Conjunctive: any failed layer blocks. The score still degrades with the number of
            # failures so the Brier/calibration table sees a graded signal rather than a step.
            p = max(0.02, 0.40 - 0.08 * len(failed))
            return p, (f"provenance chain broken at: {', '.join(failed)}",)
        return 0.94, ("all provenance layers passed: " + ", ".join(sorted(layers)),)


def describe() -> dict[str, Any]:
    d = RagShieldDefense()
    return {
        **d.metadata().as_dict(),
        "layers": [
            "provenance_chain (registry-known, or referenced by a registry-known host)",
            "identity_attested (published fields match the authoritative record)",
            "endorsed (official endorsement claim present, or is the official domain)",
            f"corroborated (>= {d.min_corroborators} other retrieved page supports it)",
            "lifecycle_intact (not lapsed, not transferred)",
            "offers_action (the page actually offers the requested action)",
        ],
        "combination": "conjunctive (defense-in-depth): every layer must pass",
        "why_strongest": (
            "Conjunctive, so strictly stronger than any single B01-B07 layer, and the only prior "
            "of the eight whose mechanism transfers directly to link presentation."
        ),
        "single_gap": (
            "Endorsement is tested as ASSERTED on the judged page, not as PUBLISHED by the "
            "endorsing site. AegisLink differs from B10 in that one respect, which makes the pair "
            "the cleanest test of RQ3's central claim."
        ),
    }


__all__ = ["RagShieldDefense", "describe"]
