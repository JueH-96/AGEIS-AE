"""The seven architectural ablations mandated by CONTRACT.md Section 9.

Each ablation is the full method with exactly one thing removed. That is enforced, not merely
intended: :func:`assert_single_factor_ablations` diffs every ablation's flag state against the full
configuration and fails if more than one switch moved. An ablation that changed two things at once
would not identify which one mattered, and the ablation table would be uninterpretable.

The seven, in the contract's order::

    remove action type                     -> ablation_no_action_type
    remove official backlink evidence      -> ablation_no_official_backlinks
    remove source-dependency clustering    -> ablation_no_source_clustering
    remove domain lifecycle evidence       -> ablation_no_domain_lifecycle
    remove contradiction edges             -> ablation_no_contradiction_edges
    replace graph inference with voting    -> ablation_source_count_voting
    one shared threshold for all actions   -> ablation_shared_threshold

What each is expected to cost, stated before measurement
--------------------------------------------------------
Predictions are recorded here so the ablation table is read as a test rather than as a description.
``workflow/16_evaluate_aegislink_and_baselines.py`` writes the realised effect next to the
prediction, and disagreements are reported rather than reconciled.

* ``no_action_type`` -- should *over-grant*. Any grant is read as covering every action, so a
  directory delegated for browse/contact becomes a payment endpoint. Expect UALER to rise on
  ``login``/``pay`` and ATPR to stay flat.
* ``no_official_backlinks`` -- should *under-grant*. Third parties lose their only source of
  authority, so expect ATPR to collapse toward the ``official_only`` baseline while UALER stays
  near zero.
* ``no_source_clustering`` -- expected to be *small on this corpus*. Corroboration is capped below
  every threshold whether or not it is deduplicated, so removing the cluster collapse changes the
  probability but rarely the verdict. It matters for the voting ablation and for the baselines that
  do count sources, which is where the Sybil arm actually bites.
* ``no_domain_lifecycle`` -- expected to be *near zero*, and the reason is a property of the corpus,
  not of the mechanism: takeover domains hold no current delegation, so there is no authority for
  the lifecycle family to withdraw. See ``results/evidence_family_identifiability.json``.
* ``no_contradiction_edges`` -- expected to change *verdict semantics more than blocking*. The
  authority gate already withholds impersonators; contradiction edges are what let the system say
  ``CONTRADICTED`` (a positive detection) instead of ``UNVERIFIED`` (an evidence gap), which moves
  CMR and OSMR rather than UALER.
* ``source_count_voting`` -- should *fail hard*. Five Sybil pages outvote one official page, so
  expect UALER to rise sharply.
* ``shared_threshold`` -- should redistribute error across the risk ladder: whichever single value
  is chosen, low-risk actions become stricter or high-risk actions become laxer than the monotone
  vector.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Mapping

from aegislink.framework import (
    ACTION_RISK_ORDER,
    DefenseMetadata,
    InputKind,
    InputRestriction,
    RiskThresholds,
)
from aegislink.verifier import AegisLink, AegisLinkConfig

ABLATIONS_VERSION = "1.0"

#: The seven ablation ids, in contract order.
ABLATION_IDS: tuple[str, ...] = (
    "ablation_no_action_type",
    "ablation_no_official_backlinks",
    "ablation_no_source_clustering",
    "ablation_no_domain_lifecycle",
    "ablation_no_contradiction_edges",
    "ablation_source_count_voting",
    "ablation_shared_threshold",
)

#: Pre-stated expected direction per ablation, for the falsification table.
ABLATION_PREDICTIONS: Mapping[str, str] = {
    "ablation_no_action_type": "UALER up on login/pay (over-grants); ATPR ~flat",
    "ablation_no_official_backlinks": "ATPR collapses toward official_only; UALER ~flat near zero",
    "ablation_no_source_clustering": "small on this corpus; corroboration is capped either way",
    "ablation_no_domain_lifecycle": "near zero; takeover domains hold no grant to withdraw",
    "ablation_no_contradiction_edges": "CONTRADICTED -> UNVERIFIED; moves CMR/OSMR, not UALER",
    "ablation_source_count_voting": "UALER up sharply; Sybil pages outvote the official page",
    "ablation_shared_threshold": "error redistributed across the risk ladder",
}


def _shared_threshold_value() -> float:
    """The single value used by ``ablation_shared_threshold``.

    The mean of the monotone design vector. Any single value privileges some actions over others --
    that is the point of the ablation -- so the mean is chosen because it is the one value that
    cannot be accused of having been picked to favour a particular action.
    """
    d = RiskThresholds.design_default()
    return round(sum(float(d.tau[a]) for a in ACTION_RISK_ORDER) / len(ACTION_RISK_ORDER), 6)


# ======================================================================================
# The seven ablations
# ======================================================================================
def ablation_no_action_type(base: AegisLinkConfig | None = None) -> AegisLinkConfig:
    """Strip action type: one global risk threshold, and any grant covers any action."""
    base = base or AegisLinkConfig()
    return replace(
        base,
        config_id="ablation_no_action_type",
        label="AegisLink -- action type removed",
        use_action_type=False,
        thresholds=RiskThresholds.shared(
            _shared_threshold_value(),
            note=(
                "action type stripped, so no per-action threshold is definable; the shared value "
                "is the mean of the monotone design vector"
            ),
        ),
    )


def ablation_no_official_backlinks(base: AegisLinkConfig | None = None) -> AegisLinkConfig:
    """Drop official-site-to-third-party backlink evidence."""
    base = base or AegisLinkConfig()
    return replace(
        base,
        config_id="ablation_no_official_backlinks",
        label="AegisLink -- official backlink evidence removed",
        use_official_backlinks=False,
    )


def ablation_no_source_clustering(base: AegisLinkConfig | None = None) -> AegisLinkConfig:
    """Disable source-dependency clustering: every page counts as an independent source."""
    base = base or AegisLinkConfig()
    return replace(
        base,
        config_id="ablation_no_source_clustering",
        label="AegisLink -- source-dependency clustering removed",
        use_source_clustering=False,
    )


def ablation_no_domain_lifecycle(base: AegisLinkConfig | None = None) -> AegisLinkConfig:
    """Drop domain lifecycle and ownership-change evidence."""
    base = base or AegisLinkConfig()
    return replace(
        base,
        config_id="ablation_no_domain_lifecycle",
        label="AegisLink -- domain lifecycle evidence removed",
        use_domain_lifecycle=False,
    )


def ablation_no_contradiction_edges(base: AegisLinkConfig | None = None) -> AegisLinkConfig:
    """Ignore contradiction edges: no verdict may be ``CONTRADICTED``."""
    base = base or AegisLinkConfig()
    return replace(
        base,
        config_id="ablation_no_contradiction_edges",
        label="AegisLink -- contradiction edges removed",
        use_contradiction_edges=False,
    )


def ablation_source_count_voting(base: AegisLinkConfig | None = None) -> AegisLinkConfig:
    """Replace graph inference with unweighted source-count voting."""
    base = base or AegisLinkConfig()
    return replace(
        base,
        config_id="ablation_source_count_voting",
        label="AegisLink -- graph inference replaced by source-count voting",
        inference_mode="source_count",
    )


def ablation_shared_threshold(base: AegisLinkConfig | None = None) -> AegisLinkConfig:
    """Apply one shared threshold across all five actions, keeping the graph inference."""
    base = base or AegisLinkConfig()
    return replace(
        base,
        config_id="ablation_shared_threshold",
        label="AegisLink -- single shared threshold",
        thresholds=RiskThresholds.shared(
            _shared_threshold_value(),
            note="risk ordering deliberately removed; graph inference otherwise unchanged",
        ),
    )


ABLATION_FACTORIES: Mapping[str, Callable[..., AegisLinkConfig]] = {
    "ablation_no_action_type": ablation_no_action_type,
    "ablation_no_official_backlinks": ablation_no_official_backlinks,
    "ablation_no_source_clustering": ablation_no_source_clustering,
    "ablation_no_domain_lifecycle": ablation_no_domain_lifecycle,
    "ablation_no_contradiction_edges": ablation_no_contradiction_edges,
    "ablation_source_count_voting": ablation_source_count_voting,
    "ablation_shared_threshold": ablation_shared_threshold,
}

#: Which switch each ablation is *allowed* to move. Anything else is a bug.
ABLATION_EXPECTED_FLAGS: Mapping[str, frozenset[str]] = {
    # Stripping action type makes a per-action threshold undefinable, so this ablation necessarily
    # also collapses the threshold vector. That is one conceptual change, and it is declared here
    # rather than silently tolerated.
    "ablation_no_action_type": frozenset({"use_action_type", "shared_threshold"}),
    "ablation_no_official_backlinks": frozenset({"use_official_backlinks"}),
    "ablation_no_source_clustering": frozenset({"use_source_clustering"}),
    "ablation_no_domain_lifecycle": frozenset({"use_domain_lifecycle"}),
    "ablation_no_contradiction_edges": frozenset({"use_contradiction_edges"}),
    "ablation_source_count_voting": frozenset({"inference_mode"}),
    "ablation_shared_threshold": frozenset({"shared_threshold"}),
}


# ======================================================================================
# Construction and self-checks
# ======================================================================================
def build_ablations(
    base: AegisLinkConfig | None = None,
    *,
    expected_replay_fingerprint: str | None = None,
) -> dict[str, AegisLink]:
    """Instantiate all seven ablations from one base configuration.

    Passing the *fitted* base config is important: an ablation compared against a full method with
    different thresholds would confound the removed component with the threshold change.
    """
    base = base or AegisLinkConfig()
    out: dict[str, AegisLink] = {}
    for name in ABLATION_IDS:
        cfg = ABLATION_FACTORIES[name](base)
        out[name] = AegisLink(
            config=cfg, expected_replay_fingerprint=expected_replay_fingerprint
        )
    return out


def diff_flags(base: AegisLinkConfig, ablated: AegisLinkConfig) -> set[str]:
    """Which ablation-relevant switches differ between two configurations."""
    b, a = base.flag_state(), ablated.flag_state()
    return {k for k in b if b[k] != a[k]}


def assert_single_factor_ablations(base: AegisLinkConfig | None = None) -> dict[str, Any]:
    """Prove each ablation moves only the switch it is named for.

    Raises :class:`AssertionError` on any extra or missing change. Returns the realised diff per
    ablation for the provenance record.
    """
    base = base or AegisLinkConfig()
    report: dict[str, Any] = {}
    problems: list[str] = []
    for name in ABLATION_IDS:
        cfg = ABLATION_FACTORIES[name](base)
        moved = diff_flags(base, cfg)
        allowed = ABLATION_EXPECTED_FLAGS[name]
        report[name] = {
            "moved": sorted(moved),
            "allowed": sorted(allowed),
            "single_factor": moved == allowed,
        }
        if moved != allowed:
            problems.append(
                f"{name}: moved {sorted(moved)} but is declared to move {sorted(allowed)}"
            )
    if problems:
        raise AssertionError(
            "ablations must change exactly the declared switch:\n  " + "\n  ".join(problems)
        )
    return report


def ablation_metadata() -> dict[str, DefenseMetadata]:
    """Registry metadata for the results tables and the deviation log."""
    restrictions = {
        "ablation_no_action_type": (
            InputRestriction(
                InputKind.ACTION_TYPE, "ablation removes the action type by construction"
            ),
        ),
        "ablation_no_official_backlinks": (
            InputRestriction(
                InputKind.OFFICIAL_BACKLINK_DIRECTIONAL,
                "ablation removes directional backlink evidence by construction",
            ),
        ),
        "ablation_no_source_clustering": (
            InputRestriction(
                InputKind.SOURCE_CLUSTERING,
                "ablation removes source-dependency clustering by construction",
            ),
        ),
        "ablation_no_domain_lifecycle": (
            InputRestriction(
                InputKind.LIFECYCLE, "ablation removes lifecycle evidence by construction"
            ),
        ),
    }
    return {
        name: DefenseMetadata(
            defense_id=name,
            family="aegislink_ablation",
            label=ABLATION_FACTORIES[name]().label,
            contract_ref="CONTRACT.md Section 9 (required ablations)",
            restrictions=restrictions.get(name, ()),
            notes=f"prediction: {ABLATION_PREDICTIONS[name]}",
        )
        for name in ABLATION_IDS
    }


def describe_ablations() -> dict[str, Any]:
    """Provenance block for the results artifacts."""
    base = AegisLinkConfig()
    return {
        "ablations_version": ABLATIONS_VERSION,
        "contract_ref": "CONTRACT.md Section 9 (required ablations)",
        "n_ablations": len(ABLATION_IDS),
        "ablation_ids": list(ABLATION_IDS),
        "base_config": base.as_dict(),
        "shared_threshold_value": _shared_threshold_value(),
        "shared_threshold_rationale": (
            "The mean of the monotone design vector. Any single value privileges some actions, "
            "which is the ablation's content; the mean is the one choice that cannot be accused "
            "of having been selected to favour a particular action."
        ),
        "single_factor_check": assert_single_factor_ablations(base),
        "predictions": dict(ABLATION_PREDICTIONS),
        "prediction_policy": (
            "Directions are stated here before measurement. The driver writes the realised effect "
            "beside the prediction; a disagreement is reported, not reconciled."
        ),
    }


__all__ = [
    "ABLATIONS_VERSION",
    "ABLATION_EXPECTED_FLAGS",
    "ABLATION_FACTORIES",
    "ABLATION_IDS",
    "ABLATION_PREDICTIONS",
    "ablation_metadata",
    "ablation_no_action_type",
    "ablation_no_contradiction_edges",
    "ablation_no_domain_lifecycle",
    "ablation_no_official_backlinks",
    "ablation_no_source_clustering",
    "ablation_shared_threshold",
    "ablation_source_count_voting",
    "assert_single_factor_ablations",
    "build_ablations",
    "describe_ablations",
    "diff_flags",
]
