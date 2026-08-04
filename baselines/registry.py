"""The ten-baseline registry and the Section 10 deviation log.

CONTRACT.md Section 10 requires *at least* ten named baselines and requires deviations from the
identical-evidence rule to be logged. This module is the single place that:

* instantiates all ten in contract order;
* asserts the suite is complete and correctly numbered, so a missing or misnumbered baseline is a
  hard failure rather than a quietly shorter results table;
* collects every declared :class:`~aegislink.framework.InputRestriction` and every surrogate
  substitution into one machine-readable log.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from aegislink.framework import DefenseMetadata, InputRestriction
from baselines.common import BASELINES_VERSION, BaseDefense, describe_baseline_common
from baselines.domain_reputation import DomainReputation
from baselines.graph_anomaly import GraphAnomalyDetector
from baselines.high_risk_reject import RejectHighRisk
from baselines.lexical_rules import LexicalUrlRules
from baselines.llm_judge import LlmAsJudge
from baselines.official_only import OfficialOnly
from baselines.phishing_classifier import PhishingClassifier
from baselines.provenance_reranker import ProvenanceReranker
from baselines.ragshield import RagShieldDefense
from baselines.source_voting import SourceCountMajority

#: Contract order. The ids embed the B-number so a results table cannot silently reorder them.
BASELINE_IDS: tuple[str, ...] = (
    "B01_lexical_url_rules",
    "B02_domain_reputation",
    "B03_phishing_classifier",
    "B04_llm_as_judge",
    "B05_source_count_majority",
    "B06_provenance_reranker",
    "B07_graph_anomaly_detector",
    "B08_reject_high_risk",
    "B09_official_only",
    "B10_ragshield_defense",
)

BASELINE_FACTORIES: Mapping[str, Callable[[], BaseDefense]] = {
    "B01_lexical_url_rules": LexicalUrlRules,
    "B02_domain_reputation": DomainReputation,
    "B03_phishing_classifier": PhishingClassifier,
    "B04_llm_as_judge": LlmAsJudge,
    "B05_source_count_majority": SourceCountMajority,
    "B06_provenance_reranker": ProvenanceReranker,
    "B07_graph_anomaly_detector": GraphAnomalyDetector,
    "B08_reject_high_risk": RejectHighRisk,
    "B09_official_only": OfficialOnly,
    "B10_ragshield_defense": RagShieldDefense,
}

#: Which contract item each baseline answers (Section 10 list positions).
CONTRACT_ITEM_OF: Mapping[str, int] = {bid: i + 1 for i, bid in enumerate(BASELINE_IDS)}


def build_baselines() -> dict[str, BaseDefense]:
    """Instantiate all ten baselines, in contract order."""
    return {bid: BASELINE_FACTORIES[bid]() for bid in BASELINE_IDS}


def assert_suite_complete() -> dict[str, Any]:
    """Fail closed unless all ten baselines exist, are numbered B01..B10 and self-identify.

    A results table with nine rows would still look plausible, so completeness is asserted rather
    than eyeballed.
    """
    problems: list[str] = []
    if len(BASELINE_IDS) != 10:
        problems.append(f"expected 10 baselines, registry declares {len(BASELINE_IDS)}")
    built = build_baselines()
    for i, bid in enumerate(BASELINE_IDS, start=1):
        expected_prefix = f"B{i:02d}_"
        if not bid.startswith(expected_prefix):
            problems.append(f"position {i} holds {bid!r}, expected prefix {expected_prefix!r}")
        d = built[bid]
        if d.defense_id != bid:
            problems.append(f"{bid}: instance reports defense_id={d.defense_id!r}")
        if not d.contract_ref.startswith("CONTRACT.md Section 10"):
            problems.append(f"{bid}: contract_ref does not cite Section 10 ({d.contract_ref!r})")
    if problems:
        raise AssertionError(
            "baseline suite is incomplete or misnumbered:\n  " + "\n  ".join(problems)
        )
    return {
        "n_baselines": len(BASELINE_IDS),
        "baseline_ids": list(BASELINE_IDS),
        "contract_items_covered": sorted(CONTRACT_ITEM_OF.values()),
        "complete": True,
    }


def baseline_metadata() -> dict[str, DefenseMetadata]:
    return {bid: d.metadata() for bid, d in build_baselines().items()}


def deviation_log() -> dict[str, Any]:
    """The Section 10 deviation log: every declared input restriction and every surrogate.

    Two kinds of deviation, kept separate because they differ in seriousness:

    * ``input_restrictions`` -- the baseline's *design* forbids an input. Section 10 explicitly
      permits this ("unless its design explicitly prohibits an input type"); a URL heuristic that
      read page content would not be a URL heuristic.
    * ``surrogates`` -- the implementation is not the thing the contract names. Only B04 (LLM judge)
      is a substitution of kind; B07 and B10 are re-implementations from published mechanism
      descriptions rather than released code, which is a weaker claim than a reproduction and is
      labelled as such in their row names.
    """
    meta = baseline_metadata()
    restrictions: dict[str, list[dict[str, Any]]] = {}
    surrogates: list[dict[str, Any]] = []
    for bid, m in meta.items():
        if m.restrictions:
            restrictions[bid] = [r.as_dict() for r in m.restrictions]
        if m.is_surrogate:
            surrogates.append(
                {
                    "defense_id": bid,
                    "label": m.label,
                    "note": m.surrogate_note,
                }
            )
    n_restr = sum(len(v) for v in restrictions.values())
    return {
        "contract_ref": (
            "CONTRACT.md Section 10: every baseline receives identical evidence, splits and attack "
            "budgets unless its design explicitly prohibits an input type; deviations must be "
            "logged."
        ),
        "identical_evidence_holds": True,
        "identical_evidence_mechanism": (
            "One VerificationContext per query, constructed once and passed unchanged to AegisLink, "
            "all seven ablations and all ten baselines, on the same frozen replay at the same "
            "read_phase."
        ),
        "n_input_restrictions": n_restr,
        "input_restrictions": restrictions,
        "n_surrogates": len(surrogates),
        "surrogates": surrogates,
        "surrogate_policy": (
            "A surrogate is declared in the row label as well as here, so no table can present one "
            "as the thing it stands in for."
        ),
    }


def describe_baselines() -> dict[str, Any]:
    """Full provenance block for the results artifacts."""
    from baselines import (  # noqa: PLC0415 - imported here to keep module import cheap
        domain_reputation,
        graph_anomaly,
        high_risk_reject,
        lexical_rules,
        llm_judge,
        official_only,
        phishing_classifier,
        provenance_reranker,
        ragshield,
        source_voting,
    )

    describers = {
        "B01_lexical_url_rules": lexical_rules.describe,
        "B02_domain_reputation": domain_reputation.describe,
        "B03_phishing_classifier": phishing_classifier.describe,
        "B04_llm_as_judge": llm_judge.describe,
        "B05_source_count_majority": source_voting.describe,
        "B06_provenance_reranker": provenance_reranker.describe,
        "B07_graph_anomaly_detector": graph_anomaly.describe,
        "B08_reject_high_risk": high_risk_reject.describe,
        "B09_official_only": official_only.describe,
        "B10_ragshield_defense": ragshield.describe,
    }
    return {
        "baselines_version": BASELINES_VERSION,
        "suite_check": assert_suite_complete(),
        "common": describe_baseline_common(),
        "deviation_log": deviation_log(),
        "baselines": {bid: describers[bid]() for bid in BASELINE_IDS},
    }


__all__ = [
    "BASELINE_FACTORIES",
    "BASELINE_IDS",
    "CONTRACT_ITEM_OF",
    "assert_suite_complete",
    "baseline_metadata",
    "build_baselines",
    "describe_baselines",
    "deviation_log",
]
