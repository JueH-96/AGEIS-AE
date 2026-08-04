"""B03 -- generic phishing classifier. CONTRACT.md Section 10, item 3.

A logistic classifier over the feature set generic phishing detectors use: URL surface form,
brand-name-in-hostname, credential and payment vocabulary on the page, urgency phrasing, and
identity-field mismatch. Weights are fixed by design in the shape published detectors use
(brand-in-host and credential-harvesting cues dominate); the intercept is Platt-calibrated on
``train_development`` only, so it is a *calibrated* classifier rather than an arbitrary one.

Deliberately generic
--------------------
"Generic" is the contract's word and it is the point of the baseline. This classifier asks "does
this look like a phishing page?", which is a different question from "is this domain authorized to
take a payment for this business?". The two come apart in both directions on this corpus, and both
directions are informative:

* An impersonating page with correct identity fields and no urgency looks benign -- a false
  negative that is a genuine misbinding.
* The entity's own official site embeds the business name in its hostname and hosts a login area --
  the two strongest generic phishing features -- so it looks like a false positive.

No external model is used: a pinned classifier checkpoint would have to be shipped and hashed
(Section 15), and a fixed-weight logistic model over declared features is reproducible by
inspection.
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
    CREDENTIAL_CUES,
    PAYMENT_CUES,
    SUSPICIOUS_HOST_TOKENS,
    URGENCY_CUES,
    BaseDefense,
    entity_slug_in_host,
    host_labels,
    identity_agreement,
    page_text_for,
)

#: Fixed feature weights, in the shape generic phishing detectors use.
PHISHING_WEIGHTS: dict[str, float] = {
    "bias": 2.30,
    "brand_in_host": -1.60,
    "suspicious_host_token": -0.85,
    "hyphens": -0.22,
    "credential_cue": -1.15,
    "payment_cue": -0.95,
    "urgency_cue": -0.45,
    "identity_mismatch": -1.70,
    "identity_match": 0.60,
    "no_identity_published": -0.35,
}


@dataclass
class PhishingClassifier(BaseDefense):
    """Generic phishing likelihood, inverted into an authorization probability."""

    defense_id: str = "B03_phishing_classifier"
    label: str = "B03 generic phishing classifier"
    contract_ref: str = "CONTRACT.md Section 10, item 3"
    restrictions: tuple[InputRestriction, ...] = field(
        default_factory=lambda: (
            InputRestriction(
                InputKind.OFFICIAL_BACKLINK_DIRECTIONAL,
                "a generic phishing classifier scores one page in isolation; directional "
                "cross-source verification is outside its design",
            ),
            InputRestriction(
                InputKind.ACTION_TYPE,
                "generic: it classifies the page, not the requested action",
            ),
        )
    )
    notes: str = (
        "Fixed-weight logistic model over declared features, Platt-calibrated on "
        "train_development. No external checkpoint, so Section 15 reproducibility holds by "
        "inspection."
    )
    weights: dict[str, float] = field(default_factory=lambda: dict(PHISHING_WEIGHTS))

    def features(
        self, entity_id: str, domain_id: str, action: str, ctx: VerificationContext
    ) -> dict[str, float]:
        host = ctx.hostname(domain_id)
        text = page_text_for(ctx, domain_id)
        low = normalize(text)
        agree = identity_agreement(ctx, entity_id, text)
        labels = host_labels(host)
        return {
            "bias": 1.0,
            "brand_in_host": 1.0 if entity_slug_in_host(ctx, entity_id, domain_id) else 0.0,
            "suspicious_host_token": float(
                sum(1 for t in labels if t in SUSPICIOUS_HOST_TOKENS)
            ),
            "hyphens": float(host.count("-")),
            "credential_cue": float(sum(1 for c in CREDENTIAL_CUES if c in low)),
            "payment_cue": float(sum(1 for c in PAYMENT_CUES if c in low)),
            "urgency_cue": float(sum(1 for c in URGENCY_CUES if c in low)),
            "identity_mismatch": 1.0 if (agree is not None and agree < 1.0) else 0.0,
            "identity_match": 1.0 if (agree is not None and agree >= 1.0) else 0.0,
            "no_identity_published": 1.0 if agree is None else 0.0,
        }

    def probability(
        self, entity_id: str, domain_id: str, action: str, ctx: VerificationContext
    ) -> tuple[float, tuple[str, ...]]:
        feats = self.features(entity_id, domain_id, action, ctx)
        z = sum(self.weights.get(k, 0.0) * v for k, v in feats.items())
        top = sorted(
            ((k, self.weights.get(k, 0.0) * v) for k, v in feats.items() if k != "bias"),
            key=lambda kv: kv[1],
        )[:3]
        reasons = ("strongest negative features: " + ", ".join(f"{k}({c:+.2f})" for k, c in top),)
        return sigmoid(z), reasons


def describe() -> dict[str, Any]:
    d = PhishingClassifier()
    return {
        **d.metadata().as_dict(),
        "weights": dict(PHISHING_WEIGHTS),
        "feature_set": sorted(PHISHING_WEIGHTS),
        "why_generic_matters": (
            "It asks whether the page looks like phishing, not whether the domain is authorized "
            "for the action. The two come apart in both directions on this corpus: an "
            "impersonator with correct identity fields looks benign, and the entity's own "
            "official site trips brand-in-host plus credential cues."
        ),
    }


__all__ = ["PHISHING_WEIGHTS", "PhishingClassifier", "describe"]
