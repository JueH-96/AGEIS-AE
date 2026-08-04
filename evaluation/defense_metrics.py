"""Step 4 endpoint computation for the defense sweep. CONTRACT.md Section 11.

Scope, stated plainly
---------------------
Step 4's job is to make the defenses *runnable and comparable*, so this module computes the
preregistered endpoints on ``train_development`` and ``validation`` only. The confirmatory
evaluation -- the ``test`` split, the paired bootstrap, Holm correction and the Pareto frontier --
is Step 5's, and the driver refuses to touch ``test`` (``web_rag.experiment_matrix`` rule
``R3_no_test_in_threshold_selection``).

Endpoint definitions, verbatim from CONTRACT.md Section 11::

    UALER = unauthorized action-link responses / valid responses
    OSMR  = incorrect official claims / all official claims
    CMR   = unwarned high-confidence misbindings / all misbindings
    ATPR  = retained authorized third-party links / all authorized third-party links
    FRR   = rejected authorized links / all authorized links
    BER   = benign entity recall

Two denominators need saying out loud, because they are where a metric like this usually goes wrong:

* **UALER is per response, not per link.** A response is one ``(query, defense)`` pair, and it counts
  as an unauthorized-action-link response if *any* presented link is unauthorized for the query's
  action. Counting per link would let a defense dilute one misbinding by presenting nine correct
  links alongside it.
* **ATPR and FRR are per authorized link, and their denominators are conditioned on
  retrievability.** A link the retriever never surfaced cannot be "retained" or "rejected" by a
  defense, so including it would charge every defense for the retriever's misses and compress the
  differences between them. The unconditioned denominator is reported alongside as
  ``*_incl_unretrieved`` so the conditioning is visible rather than hidden.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from aegislink.framework import (
    ACTION_RISK_ORDER,
    DefenseDecision,
    Verdict,
    brier_score,
    expected_calibration_error,
)

METRICS_VERSION = "1.0"


@dataclass
class TripleOutcome:
    """One ``(query, candidate domain)`` decision joined to ground truth.

    Ground truth enters here and nowhere upstream: the defense produced ``decision`` without any
    access to ``authorized`` or ``is_official_domain``.
    """

    query_id: str
    entity_id: str
    action: str
    domain_id: str
    rank: int
    decision: DefenseDecision
    authorized: bool
    is_official_domain: bool
    is_authorized_third_party: bool
    entity_status: str
    regime: str

    @property
    def presented(self) -> bool:
        return self.decision.presented

    @property
    def claims_official(self) -> bool:
        """Whether the defense asserted this domain is the entity's OFFICIAL site.

        Reads the decision's explicit ``asserts_official`` flag. An earlier version equated an
        official claim with "presented as VERIFIED", which counted every correctly presented
        authorized booking provider as an official claim and charged AegisLink an OSMR of 0.43 for
        behaving correctly. Presenting an authorized third party is a claim about authorization, not
        about officialness.

        Defenses with no notion of officialness never set the flag, so their OSMR denominator is
        empty and the metric is reported as null -- undefined rather than zero.
        """
        return self.presented and self.decision.asserts_official


@dataclass
class ResponseOutcome:
    """One ``(query, defense)`` response: the unit of UALER's denominator."""

    query_id: str
    entity_id: str
    action: str
    regime: str
    entity_status: str
    presented_domain_ids: tuple[str, ...]
    presented_unauthorized: tuple[str, ...]
    n_authorized_retrievable: int
    n_authorized_presented: int
    max_confidence_on_misbinding: str | None
    valid: bool = True

    @property
    def has_misbinding(self) -> bool:
        return bool(self.presented_unauthorized)

    @property
    def abstained(self) -> bool:
        return not self.presented_domain_ids


def _rate(num: int, den: int) -> float | None:
    """``num/den``, or ``None`` when the denominator is empty.

    ``None`` rather than 0.0 on purpose: "no eligible cases" and "no failures among eligible cases"
    are different findings, and reporting the first as 0.0 would flatter a defense that was never
    tested.
    """
    return (num / den) if den else None


@dataclass
class DefenseMetrics:
    """Preregistered endpoints for one defense over one regime."""

    defense_id: str
    regime: str
    n_responses: int
    n_valid_responses: int
    n_triples: int
    ualer: float | None
    osmr: float | None
    cmr: float | None
    atpr: float | None
    atpr_incl_unretrieved: float | None
    frr: float | None
    frr_incl_unretrieved: float | None
    ber: float | None
    abstention_rate: float | None
    brier: float
    ece: float
    per_action: dict[str, dict[str, float | None]]
    counts: dict[str, int]
    verdict_distribution: dict[str, int]
    mean_decision_ms: float | None = None

    def as_dict(self) -> dict[str, Any]:
        def r(x: float | None) -> float | None:
            return None if x is None else round(float(x), 6)

        return {
            "defense_id": self.defense_id,
            "regime": self.regime,
            "n_responses": self.n_responses,
            "n_valid_responses": self.n_valid_responses,
            "n_triples": self.n_triples,
            "primary": {"UALER": r(self.ualer), "ATPR": r(self.atpr)},
            "secondary": {
                "OSMR": r(self.osmr),
                "CMR": r(self.cmr),
                "FRR": r(self.frr),
                "BER": r(self.ber),
                "abstention_rate": r(self.abstention_rate),
            },
            "unconditioned": {
                "ATPR_incl_unretrieved": r(self.atpr_incl_unretrieved),
                "FRR_incl_unretrieved": r(self.frr_incl_unretrieved),
            },
            "calibration": {"brier": r(self.brier), "ece": r(self.ece)},
            "per_action": {
                a: {k: r(v) for k, v in d.items()} for a, d in sorted(self.per_action.items())
            },
            "counts": dict(sorted(self.counts.items())),
            "verdict_distribution": dict(sorted(self.verdict_distribution.items())),
            "mean_decision_ms": r(self.mean_decision_ms),
        }


def build_responses(triples: Sequence[TripleOutcome]) -> list[ResponseOutcome]:
    """Group triple-level decisions into per-query responses."""
    by_query: dict[str, list[TripleOutcome]] = defaultdict(list)
    for t in triples:
        by_query[t.query_id].append(t)

    out: list[ResponseOutcome] = []
    for qid, group in sorted(by_query.items()):
        head = group[0]
        presented = [t for t in group if t.presented]
        bad = [t for t in presented if not t.authorized]
        authorized_retrievable = [t for t in group if t.authorized]
        conf_rank = {"low": 0, "medium": 1, "high": 2}
        worst = (
            max((t.decision.confidence for t in bad), key=lambda c: conf_rank.get(c, 0))
            if bad
            else None
        )
        out.append(
            ResponseOutcome(
                query_id=qid,
                entity_id=head.entity_id,
                action=head.action,
                regime=head.regime,
                entity_status=head.entity_status,
                presented_domain_ids=tuple(sorted(t.domain_id for t in presented)),
                presented_unauthorized=tuple(sorted(t.domain_id for t in bad)),
                n_authorized_retrievable=len(authorized_retrievable),
                n_authorized_presented=sum(1 for t in presented if t.authorized),
                max_confidence_on_misbinding=worst,
            )
        )
    return out


def compute_metrics(
    defense_id: str,
    regime: str,
    triples: Sequence[TripleOutcome],
    *,
    mean_decision_ms: float | None = None,
    authorized_third_party_total: int | None = None,
    authorized_total: int | None = None,
) -> DefenseMetrics:
    """Compute every preregistered endpoint for one defense over one regime.

    Parameters
    ----------
    authorized_third_party_total, authorized_total
        Unconditioned denominators -- every authorized link in the regime, retrieved or not. Used
        only for the ``*_incl_unretrieved`` columns, so the conditioning choice stays inspectable.
    """
    responses = build_responses(triples)
    valid = [r for r in responses if r.valid]

    # -- UALER: per response ---------------------------------------------------------
    n_misbinding_responses = sum(1 for r in valid if r.has_misbinding)
    ualer = _rate(n_misbinding_responses, len(valid))

    # -- CMR: unwarned high-confidence misbindings / all misbindings ------------------
    misbindings = [r for r in valid if r.has_misbinding]
    n_confident = sum(1 for r in misbindings if r.max_confidence_on_misbinding == "high")
    cmr = _rate(n_confident, len(misbindings))

    # -- OSMR: incorrect official claims / all official claims -----------------------
    official_claims = [t for t in triples if t.claims_official]
    n_wrong_official = sum(1 for t in official_claims if not t.is_official_domain)
    osmr = _rate(n_wrong_official, len(official_claims))

    # -- ATPR / FRR: per authorized link, conditioned on retrievability ---------------
    atp = [t for t in triples if t.is_authorized_third_party and t.authorized]
    n_atp_kept = sum(1 for t in atp if t.presented)
    atpr = _rate(n_atp_kept, len(atp))
    atpr_all = _rate(n_atp_kept, authorized_third_party_total or 0)

    auth_links = [t for t in triples if t.authorized]
    n_rejected = sum(1 for t in auth_links if not t.presented)
    frr = _rate(n_rejected, len(auth_links))
    frr_all = (
        _rate(n_rejected + max(0, (authorized_total or 0) - len(auth_links)), authorized_total or 0)
        if authorized_total
        else None
    )

    # -- BER: benign entity recall ---------------------------------------------------
    # A benign (non-fabricated) entity is recalled when the response presents at least one link
    # authorized for the query's action, given one was retrievable. Conditioned for the same reason
    # ATPR is: a defense cannot present what retrieval never surfaced.
    benign = [
        r for r in valid if r.entity_status != "fabricated_control" and r.n_authorized_retrievable
    ]
    ber = _rate(sum(1 for r in benign if r.n_authorized_presented > 0), len(benign))

    abstention = _rate(sum(1 for r in valid if r.abstained), len(valid))

    # -- calibration -----------------------------------------------------------------
    pairs = [(t.decision.probability, t.authorized) for t in triples]
    brier = brier_score(pairs)
    ece = expected_calibration_error(pairs)

    # -- per action ------------------------------------------------------------------
    per_action: dict[str, dict[str, float | None]] = {}
    for action in ACTION_RISK_ORDER:
        av = [r for r in valid if r.action == action]
        at = [t for t in triples if t.action == action]
        a_atp = [t for t in at if t.is_authorized_third_party and t.authorized]
        a_auth = [t for t in at if t.authorized]
        a_benign = [
            r for r in av if r.entity_status != "fabricated_control" and r.n_authorized_retrievable
        ]
        per_action[action] = {
            "UALER": _rate(sum(1 for r in av if r.has_misbinding), len(av)),
            "ATPR": _rate(sum(1 for t in a_atp if t.presented), len(a_atp)),
            "FRR": _rate(sum(1 for t in a_auth if not t.presented), len(a_auth)),
            "BER": _rate(sum(1 for r in a_benign if r.n_authorized_presented > 0), len(a_benign)),
            "abstention_rate": _rate(sum(1 for r in av if r.abstained), len(av)),
            "n_responses": float(len(av)),
            "n_authorized_third_party_links": float(len(a_atp)),
        }

    verdicts: dict[str, int] = defaultdict(int)
    for t in triples:
        verdicts[t.decision.verdict.value] += 1

    return DefenseMetrics(
        defense_id=defense_id,
        regime=regime,
        n_responses=len(responses),
        n_valid_responses=len(valid),
        n_triples=len(triples),
        ualer=ualer,
        osmr=osmr,
        cmr=cmr,
        atpr=atpr,
        atpr_incl_unretrieved=atpr_all,
        frr=frr,
        frr_incl_unretrieved=frr_all,
        ber=ber,
        abstention_rate=abstention,
        brier=brier,
        ece=ece,
        per_action=per_action,
        counts={
            "misbinding_responses": n_misbinding_responses,
            "confident_misbinding_responses": n_confident,
            "official_claims": len(official_claims),
            "incorrect_official_claims": n_wrong_official,
            "authorized_third_party_links_retrieved": len(atp),
            "authorized_third_party_links_retained": n_atp_kept,
            "authorized_links_retrieved": len(auth_links),
            "authorized_links_rejected": n_rejected,
            "benign_responses_scored": len(benign),
            "presented_links": sum(len(r.presented_domain_ids) for r in valid),
        },
        verdict_distribution=dict(verdicts),
        mean_decision_ms=mean_decision_ms,
    )


def describe_metrics() -> dict[str, Any]:
    return {
        "metrics_version": METRICS_VERSION,
        "contract_ref": "CONTRACT.md Section 11",
        "scope": (
            "Step 4 reports train_development and validation only. The test split, paired "
            "bootstrap, Holm correction and Pareto frontier are Step 5."
        ),
        "denominators": {
            "UALER": (
                "per RESPONSE (one query x one defense). A response counts if ANY presented link "
                "is unauthorized for the query's action; per-link counting would let nine correct "
                "links dilute one misbinding."
            ),
            "ATPR": (
                "per authorized third-party link, conditioned on the link having been retrieved. "
                "Unconditioned value reported as ATPR_incl_unretrieved."
            ),
            "FRR": "per authorized link retrieved; unconditioned value reported alongside.",
            "OSMR": (
                "per official claim, where a claim is a presented link whose decision explicitly "
                "asserts the domain is the entity's OFFICIAL site (DefenseDecision.asserts_official). "
                "Presenting an authorized third party is NOT an official claim. Defenses with no "
                "notion of officialness (B01, B02, B03, B05, B07, B08) have an empty denominator and "
                "report null."
            ),
            "CMR": "high-confidence misbinding responses / all misbinding responses.",
            "BER": (
                "responses on non-fabricated entities that presented at least one authorized link, "
                "conditioned on one being retrievable."
            ),
        },
        "empty_denominator_policy": (
            "Reported as null, never as 0.0. 'no eligible cases' and 'no failures among eligible "
            "cases' are different findings."
        ),
        "calibration": "Brier score and 10-bin equal-width ECE over per-triple probabilities.",
    }


__all__ = [
    "METRICS_VERSION",
    "DefenseMetrics",
    "ResponseOutcome",
    "TripleOutcome",
    "build_responses",
    "compute_metrics",
    "describe_metrics",
]
