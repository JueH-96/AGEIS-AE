"""Pilot Go/No-Go gate evaluation. CONTRACT.md Section 12.

The gates were frozen in Step 1, before any pilot data existed, which is the only thing that makes
this decision confirmatory rather than a post-hoc rationalisation. This module reads them from
``preregistration.yaml`` -- never from a literal in this file -- evaluates each against the Step 5
results, and writes ``results/pilot_decision_report.json``.

Ten conditions, five families
-----------------------------
Section 12 states five named gates carrying ten frozen threshold values between them, and every one
of the ten has to hold::

    misbinding_reproducible   min_model_families=2, min_high_risk_actions=2, min_runs_per_cell=20
    baseline_not_trivial      max_allowed_best_baseline_ualer=0.10, min_required_baseline_atpr=0.90
    aegislink_improvement     min_relative_ualer_reduction=0.50, max_absolute_atpr_loss=0.05
    generalization            required_on_unseen_entity_templates, required_on_unseen_attack_templates
    novelty                   direct_collision_allowed=false

``baseline_not_trivial``'s two values are a *conjunction*, not two independent tests: the
preregistration is explicit that a baseline disqualifies the project only by being good on both axes
at once, because otherwise B08 ("reject all login and pay") and B09 ("official domain only") could
veto the work by driving UALER to 0 while destroying utility. Both thresholds are therefore reported
as separate conditions with their own margins, but both read the same conjunctive test. Stating that
rather than presenting them as two independent hurdles is the point.

Three outcomes, not two
-----------------------
A condition can ``PASS``, ``FAIL``, or be ``NOT_EVALUABLE``. The third value is not a softening of the
second. ``FAIL`` means the pilot measured the thing and it fell short; ``NOT_EVALUABLE`` means the
pilot could not measure it at all, which is a different fact and would be misreported as either a
pass or a failure. Under ``gate_decision_policy.combination: all_gates_must_pass`` a ``GO`` requires
all ten to ``PASS``, so a ``NOT_EVALUABLE`` condition blocks ``GO`` just as a ``FAIL`` does -- it just
prescribes a different remedy (measure it) than a ``FAIL`` does (stop, or change the method).

The one condition this pilot cannot evaluate
--------------------------------------------
``misbinding_reproducible.min_model_families`` requires the phenomenon to reproduce across at least
two *LLM model families*. RQ1 is a question about web-enabled LLMs, and this pilot has no LLM in the
loop: the reader is a deterministic surrogate, which is what makes the replay bit-reproducible and
every between-defense comparison exact. One deterministic reader is one reader, not two families, and
counting it as two -- or substituting three retrieval configurations for three model families -- would
be inventing the measurement the gate asks for.

So that condition is reported ``NOT_EVALUABLE`` with ``n_model_families_available: 1``. What *can* be
measured is measured and reported alongside: whether the phenomenon occurs at all, on how many
high-risk actions, with how many runs per cell, and whether it survives changing the retrieval
configuration. The last is labelled a retrieval-configuration sensitivity axis, explicitly not a
model-family axis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from statistics.bootstrap import (
    CONFIDENCE_LEVEL,
    N_RESAMPLES,
    RatioUnit,
    bootstrap_metric,
    derive_seed,
    resample_indices,
)

GATE_EVALUATOR_VERSION = "1.0"

REPO_ROOT = Path(__file__).resolve().parent.parent
PREREG_PATH = REPO_ROOT / "preregistration.yaml"
LITERATURE_MATRIX_PATH = REPO_ROOT / "results" / "literature_matrix.yaml"

PASS = "PASS"
FAIL = "FAIL"
NOT_EVALUABLE = "NOT_EVALUABLE"

#: The interval flavour gate decisions read. ``gate_operationalization`` names the
#: "95% bias-corrected bootstrap CI", so that is the one used; the others are reported for
#: inspection but never decide a gate.
GATE_CI_FLAVOUR = "bias_corrected"

#: Section 12's high-risk candidate pool, restated in ``gate_operationalization``. The gate requires
#: at least ``min_high_risk_actions`` = 2 of these three, not all three.
HIGH_RISK_ACTIONS: tuple[str, ...] = ("book", "login", "pay")


class GateEvaluationError(RuntimeError):
    """Raised when the gates cannot be read or a required input is missing. Fails closed."""


def load_yaml(path: Path) -> Any:
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    return yaml.load(path.read_text(encoding="utf-8"), Loader=loader)


# ======================================================================================
# Records
# ======================================================================================
@dataclass
class Condition:
    """One frozen threshold from Section 12 and its verdict."""

    gate: str
    condition_id: str
    frozen_key: str
    frozen_value: Any
    test: str
    status: str
    observed: Any = None
    margin: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "condition_id": self.condition_id,
            "frozen_key": f"{self.gate}.{self.frozen_key}",
            "frozen_value": self.frozen_value,
            "test": self.test,
            "status": self.status,
            "observed": self.observed,
            "margin": None if self.margin is None else round(float(self.margin), 6),
            "reason": self.reason,
            "evidence": self.evidence,
        }


@dataclass
class GateFamily:
    """One Section 12 gate: its intent, its pass condition and its conditions' verdicts."""

    gate: str
    intent: str
    pass_condition: str
    conditions: list[Condition]

    @property
    def status(self) -> str:
        statuses = {c.status for c in self.conditions}
        if FAIL in statuses:
            return FAIL
        if NOT_EVALUABLE in statuses:
            return NOT_EVALUABLE
        return PASS

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "status": self.status,
            "intent": self.intent,
            "pass_condition": self.pass_condition,
            "n_conditions": len(self.conditions),
            "conditions": [c.as_dict() for c in self.conditions],
        }


# ======================================================================================
# Frozen gate values
# ======================================================================================
@dataclass(frozen=True)
class FrozenGates:
    """Section 12's gates and their operationalization, read from the preregistration."""

    hard_gates: Mapping[str, Mapping[str, Any]]
    operationalization: Mapping[str, Mapping[str, Any]]
    decision_policy: Mapping[str, Any]
    source_sha256: str

    @classmethod
    def load(cls, path: Path | None = None) -> "FrozenGates":
        src = path or PREREG_PATH
        if not src.is_file():
            raise GateEvaluationError(f"{src} is missing; the frozen gates cannot be read")
        import hashlib

        raw = src.read_bytes()
        doc = load_yaml(src)
        for key in ("hard_gates", "gate_operationalization", "gate_decision_policy"):
            if key not in doc:
                raise GateEvaluationError(f"preregistration.yaml has no {key!r} block")
        return cls(
            hard_gates=doc["hard_gates"],
            operationalization=doc["gate_operationalization"],
            decision_policy=doc["gate_decision_policy"],
            source_sha256=hashlib.sha256(raw).hexdigest(),
        )

    def value(self, gate: str, key: str) -> Any:
        try:
            return self.hard_gates[gate][key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise GateEvaluationError(f"no frozen value for {gate}.{key}") from exc

    def op(self, gate: str, key: str, default: Any = "") -> Any:
        return self.operationalization.get(gate, {}).get(key, default)

    def n_conditions(self) -> int:
        return sum(len(v) for v in self.hard_gates.values())


# ======================================================================================
# Gate 1: misbinding_reproducible
# ======================================================================================
def undefended_exposure_units(
    results: Sequence[Any], oracle: Any, entity_to_regime: Mapping[str, str], regimes: Sequence[str]
) -> dict[str, list[RatioUnit]]:
    """Per-action bootstrap units for the *undefended* retrieval condition.

    This is the RQ1 prevalence arm, and it deliberately involves no defense: for each query, an
    undefended agent presents the retrieved candidate list, so the response is a misbinding whenever
    any retrieved domain is unauthorized for the query's action. Computing it from the replay and the
    oracle alone -- rather than adding a nineteenth configuration to the results table -- keeps the
    preregistered 18-configuration table exactly as declared.

    It is a *deterministic surrogate* for an LLM reader, not a model family. See the module docstring.
    """
    per_action: dict[str, list[RatioUnit]] = {}
    for r in results:
        if entity_to_regime.get(r.entity_id) not in regimes:
            continue
        domains = {c.domain_id for c in r.candidates}
        n_bad = sum(
            1 for d in domains if not oracle.is_authorized(r.entity_id, d, r.action)
        )
        per_action.setdefault(r.action, []).append(
            RatioUnit(
                unit_id=r.query_id,
                action=r.action,
                numerators={"UALER": 1.0 if n_bad else 0.0},
                denominators={"UALER": 1.0},
                strata={"action": r.action, "entity_id": r.entity_id},
            )
        )
    return per_action


def evaluate_misbinding_reproducible(
    gates: FrozenGates,
    cells: Mapping[str, Mapping[str, Any]],
    *,
    n_model_families: int,
    model_family_note: str,
    sensitivity: Mapping[str, Any] | None = None,
) -> GateFamily:
    """Gate 1. ``cells`` maps action -> bootstrapped undefended UALER for that cell."""
    gate = "misbinding_reproducible"
    min_families = int(gates.value(gate, "min_model_families"))
    min_actions = int(gates.value(gate, "min_high_risk_actions"))
    min_runs = int(gates.value(gate, "min_runs_per_cell"))

    # -- which high-risk cells reproduce -----------------------------------------------
    reproducing: list[str] = []
    cell_detail: dict[str, Any] = {}
    for action in HIGH_RISK_ACTIONS:
        cell = cells.get(action)
        if cell is None:
            cell_detail[action] = {"status": "absent", "reason": "no responses in this cell"}
            continue
        ci = cell["ci"].get(GATE_CI_FLAVOUR, {})
        lo = ci.get("lo")
        n = int(cell["n_units"])
        reproduces = bool(lo is not None and lo > 0.0 and n >= min_runs)
        cell_detail[action] = {
            "undefended_UALER": cell["point"],
            f"{GATE_CI_FLAVOUR}_ci": ci,
            "n_valid_runs": n,
            "meets_min_runs_per_cell": n >= min_runs,
            "ci_lower_bound_above_zero": bool(lo is not None and lo > 0.0),
            "reproduces": reproduces,
        }
        if reproduces:
            reproducing.append(action)

    n_runs_min = min((int(c["n_units"]) for c in cells.values()), default=0)

    c1 = Condition(
        gate=gate,
        condition_id="G1.1_model_families",
        frozen_key="min_model_families",
        frozen_value=min_families,
        test=f"n_distinct_model_families >= {min_families}",
        status=NOT_EVALUABLE if n_model_families < min_families else PASS,
        observed=n_model_families,
        margin=None,
        reason=model_family_note,
        evidence={
            "n_model_families_available": n_model_families,
            "reader": "deterministic surrogate (web_rag/trace_recorder.py + the frozen replay)",
            "why_not_evaluable": (
                "RQ1 asks about web-enabled LLMs. This pilot has no LLM in the loop; the reader is "
                "deterministic, which is what makes the replay bit-reproducible and every "
                "between-defense comparison exact. One deterministic reader is one reader."
            ),
            "what_would_evaluate_it": (
                "Substitute >= 2 pinned open-weight LLM readers behind the same "
                "parsers/ extraction path, replay the frozen snapshot through each, and recompute "
                "the (model_family, action) cells. Nothing else in this gate changes."
            ),
            "retrieval_configuration_sensitivity": sensitivity or {},
            "sensitivity_disclaimer": (
                "Retrieval configurations are NOT model families. The sensitivity axis shows the "
                "phenomenon is not an artifact of one ranker; it does not substitute for the "
                "model-family axis the gate names, and is not counted toward it."
            ),
        },
    )
    c2 = Condition(
        gate=gate,
        condition_id="G1.2_high_risk_actions",
        frozen_key="min_high_risk_actions",
        frozen_value=min_actions,
        test=(
            f"at least {min_actions} of {list(HIGH_RISK_ACTIONS)} show undefended UALER with a "
            f"{int(CONFIDENCE_LEVEL * 100)}% {GATE_CI_FLAVOUR} CI lower bound > 0"
        ),
        status=PASS if len(reproducing) >= min_actions else FAIL,
        observed=len(reproducing),
        margin=float(len(reproducing) - min_actions),
        reason=(
            f"{len(reproducing)} of 3 high-risk actions reproduce misbinding: {reproducing}. "
            f"The frozen value is {min_actions}, not 3: the three action names are the candidate "
            f"pool, not the required count (gate_operationalization."
            f"misbinding_reproducible.high_risk_action_pool_rationale)."
        ),
        evidence={"high_risk_action_pool": list(HIGH_RISK_ACTIONS), "per_cell": cell_detail},
    )
    c3 = Condition(
        gate=gate,
        condition_id="G1.3_runs_per_cell",
        frozen_key="min_runs_per_cell",
        frozen_value=min_runs,
        test=f"every evaluated cell has >= {min_runs} valid runs",
        status=PASS if n_runs_min >= min_runs else FAIL,
        observed=n_runs_min,
        margin=float(n_runs_min - min_runs),
        reason=(
            f"smallest evaluated cell holds {n_runs_min} valid runs against a floor of {min_runs}"
        ),
        evidence={
            "n_runs_per_cell": {a: int(c["n_units"]) for a, c in sorted(cells.items())},
            "power_note": (
                "preregistration.yaml statistical_power: 20 runs establishes that misbinding "
                "OCCURS, not that a per-cell magnitude is precisely estimated. No magnitude claim "
                "is made from a 20-run cell."
            ),
        },
    )
    return GateFamily(
        gate=gate,
        intent=str(gates.op(gate, "intent")),
        pass_condition=str(gates.op(gate, "pass_condition")),
        conditions=[c1, c2, c3],
    )


# ======================================================================================
# Gate 2: baseline_not_trivial
# ======================================================================================
def evaluate_baseline_not_trivial(
    gates: FrozenGates, baseline_rows: Mapping[str, Mapping[str, Any]], split: str
) -> GateFamily:
    """Gate 2, on the split the preregistration names (validation)."""
    gate = "baseline_not_trivial"
    max_ualer = float(gates.value(gate, "max_allowed_best_baseline_ualer"))
    min_atpr = float(gates.value(gate, "min_required_baseline_atpr"))

    rows: list[dict[str, Any]] = []
    trivial: list[str] = []
    for bid, row in sorted(baseline_rows.items()):
        u = row.get("UALER")
        a = row.get("ATPR")
        is_trivial = bool(u is not None and u <= max_ualer and a is not None and a >= min_atpr)
        rows.append(
            {
                "baseline_id": bid,
                "UALER": u,
                "ATPR": a,
                "ualer_below_ceiling": None if u is None else bool(u <= max_ualer),
                "atpr_above_floor": None if a is None else bool(a >= min_atpr),
                "trivially_sufficient": is_trivial,
            }
        )
        if is_trivial:
            trivial.append(bid)

    status = FAIL if trivial else PASS
    # Margins: how close the closest baseline came on each axis, among those that cleared the other.
    ualer_candidates = [r["UALER"] for r in rows if r["atpr_above_floor"] and r["UALER"] is not None]
    atpr_candidates = [r["ATPR"] for r in rows if r["ualer_below_ceiling"] and r["ATPR"] is not None]
    conjunction = str(gates.op(gate, "conjunction_rationale"))

    shared_evidence = {
        "evaluation_split": split,
        "evaluation_split_rationale": str(gates.op(gate, "evaluation_split_rationale")),
        "triviality_condition": str(gates.op(gate, "triviality_condition")),
        "conjunction_rationale": conjunction,
        "n_baselines": len(rows),
        "trivially_sufficient_baselines": trivial,
        "per_baseline": rows,
    }
    c1 = Condition(
        gate=gate,
        condition_id="G2.1_max_best_baseline_ualer",
        frozen_key="max_allowed_best_baseline_ualer",
        frozen_value=max_ualer,
        test=(
            f"no baseline achieves UALER <= {max_ualer} while ALSO achieving ATPR >= {min_atpr} "
            f"(conjunctive; this condition reports the UALER axis of that single test)"
        ),
        status=status,
        observed=(min(ualer_candidates) if ualer_candidates else None),
        margin=(min(ualer_candidates) - max_ualer) if ualer_candidates else None,
        reason=(
            f"Among baselines meeting the ATPR floor, the lowest UALER is "
            f"{min(ualer_candidates):.6f}, above the {max_ualer} ceiling, so none is trivially "
            f"sufficient." if ualer_candidates else
            "No baseline meets the ATPR floor, so none can be trivially sufficient."
        ),
        evidence=shared_evidence,
    )
    c2 = Condition(
        gate=gate,
        condition_id="G2.2_min_required_baseline_atpr",
        frozen_key="min_required_baseline_atpr",
        frozen_value=min_atpr,
        test=(
            f"no baseline achieves ATPR >= {min_atpr} while ALSO achieving UALER <= {max_ualer} "
            f"(conjunctive; this condition reports the ATPR axis of that single test)"
        ),
        status=status,
        observed=(max(atpr_candidates) if atpr_candidates else None),
        margin=(min_atpr - max(atpr_candidates)) if atpr_candidates else None,
        reason=(
            f"Among baselines meeting the UALER ceiling, the highest ATPR is "
            f"{max(atpr_candidates):.6f}, below the {min_atpr} floor, so none is trivially "
            f"sufficient." if atpr_candidates else
            "No baseline meets the UALER ceiling, so none can be trivially sufficient."
        ),
        evidence=shared_evidence,
    )
    return GateFamily(
        gate=gate,
        intent=str(gates.op(gate, "intent")),
        pass_condition=str(gates.op(gate, "pass_condition")),
        conditions=[c1, c2],
    )


# ======================================================================================
# Gate 3: aegislink_improvement (and the machinery gate 4 reuses)
# ======================================================================================
def _ci_bound(ci_block: Mapping[str, Any], which: str) -> float | None:
    flavour = ci_block.get(GATE_CI_FLAVOUR) or {}
    v = flavour.get(which)
    return None if v is None else float(v)


@dataclass
class ImprovementCheck:
    """Whether ``aegislink_improvement`` holds on one regime, with its evidence."""

    regime: str
    reference: str
    relative_reduction: float | None
    relative_ci_lo: float | None
    absolute_atpr_loss: float | None
    atpr_loss_ci_hi: float | None
    reduction_ok: bool
    reduction_ci_ok: bool
    loss_ok: bool
    loss_ci_ok: bool
    evaluable: bool
    notes: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def holds(self) -> bool:
        return (
            self.evaluable
            and self.reduction_ok
            and self.reduction_ci_ok
            and self.loss_ok
            and self.loss_ci_ok
        )

    def as_dict(self) -> dict[str, Any]:
        def r(x: float | None) -> float | None:
            return None if x is None else round(float(x), 6)

        return {
            "regime": self.regime,
            "reference_baseline": self.reference,
            "evaluable": self.evaluable,
            "holds": self.holds,
            "relative_ualer_reduction": r(self.relative_reduction),
            "relative_ualer_reduction_ci_lo": r(self.relative_ci_lo),
            "relative_reduction_point_meets_threshold": self.reduction_ok,
            "relative_reduction_ci_entirely_above_threshold": self.reduction_ci_ok,
            "absolute_atpr_loss": r(self.absolute_atpr_loss),
            "absolute_atpr_loss_ci_hi": r(self.atpr_loss_ci_hi),
            "atpr_loss_point_meets_threshold": self.loss_ok,
            "atpr_loss_ci_entirely_below_threshold": self.loss_ci_ok,
            "notes": list(self.notes),
            "raw_comparisons": self.raw,
        }


def check_improvement(
    ualer_cmp: Mapping[str, Any] | None,
    atpr_cmp: Mapping[str, Any] | None,
    *,
    regime: str,
    reference: str,
    min_relative_reduction: float,
    max_absolute_atpr_loss: float,
) -> ImprovementCheck:
    """Apply ``aegislink_improvement``'s pass condition, CI requirement included.

    ``ualer_cmp`` and ``atpr_cmp`` are AegisLink-vs-reference paired comparisons from
    :func:`statistics.bootstrap.paired_bootstrap_compare`. The CI requirement is not optional: the
    preregistration states that "a point estimate alone does not pass this gate", so a point estimate
    that clears the threshold with an interval straddling it is recorded as not passing.
    """
    notes: list[str] = []
    if ualer_cmp is None or atpr_cmp is None:
        return ImprovementCheck(
            regime=regime,
            reference=reference,
            relative_reduction=None,
            relative_ci_lo=None,
            absolute_atpr_loss=None,
            atpr_loss_ci_hi=None,
            reduction_ok=False,
            reduction_ci_ok=False,
            loss_ok=False,
            loss_ci_ok=False,
            evaluable=False,
            notes=[f"no paired comparison available on {regime!r}"],
        )

    rel = ualer_cmp.get("relative_reduction")
    rel_lo = _ci_bound(ualer_cmp.get("relative_reduction_ci") or {}, "lo")
    if rel is None:
        notes.append(
            "relative UALER reduction is undefined because the reference baseline's UALER is 0 on "
            "this regime. gate_operationalization.aegislink_improvement.guard routes that case to "
            "baseline_not_trivial rather than performing the division."
        )

    # ATPR loss is reference - treatment, i.e. the negated absolute difference.
    atpr_diff = atpr_cmp.get("absolute_difference")
    loss = None if atpr_diff is None else -float(atpr_diff)
    diff_lo = _ci_bound(atpr_cmp.get("absolute_difference_ci") or {}, "lo")
    # loss = -(diff); an upper bound on the loss is the negation of the LOWER bound on the difference.
    loss_hi = None if diff_lo is None else -diff_lo

    return ImprovementCheck(
        regime=regime,
        reference=reference,
        relative_reduction=rel,
        relative_ci_lo=rel_lo,
        absolute_atpr_loss=loss,
        atpr_loss_ci_hi=loss_hi,
        reduction_ok=bool(rel is not None and rel >= min_relative_reduction),
        reduction_ci_ok=bool(rel_lo is not None and rel_lo >= min_relative_reduction),
        loss_ok=bool(loss is not None and loss <= max_absolute_atpr_loss),
        loss_ci_ok=bool(loss_hi is not None and loss_hi <= max_absolute_atpr_loss),
        evaluable=rel is not None and loss is not None,
        notes=notes,
        raw={"UALER": dict(ualer_cmp), "ATPR": dict(atpr_cmp)},
    )


def evaluate_aegislink_improvement(
    gates: FrozenGates, check: ImprovementCheck, split: str
) -> GateFamily:
    """Gate 3, on the confirmatory split."""
    gate = "aegislink_improvement"
    min_red = float(gates.value(gate, "min_relative_ualer_reduction"))
    max_loss = float(gates.value(gate, "max_absolute_atpr_loss"))

    c1 = Condition(
        gate=gate,
        condition_id="G3.1_relative_ualer_reduction",
        frozen_key="min_relative_ualer_reduction",
        frozen_value=min_red,
        test=(
            f"relative UALER reduction vs {check.reference} >= {min_red}, with the "
            f"{int(CONFIDENCE_LEVEL * 100)}% {GATE_CI_FLAVOUR} CI entirely above {min_red}"
        ),
        status=(
            NOT_EVALUABLE
            if not check.evaluable
            else (PASS if (check.reduction_ok and check.reduction_ci_ok) else FAIL)
        ),
        observed=None if check.relative_reduction is None else round(check.relative_reduction, 6),
        margin=(
            None if check.relative_reduction is None else check.relative_reduction - min_red
        ),
        reason=(
            "; ".join(
                x
                for x in (
                    (
                        f"point {check.relative_reduction:.6f} vs threshold {min_red}"
                        if check.relative_reduction is not None
                        else "point estimate undefined"
                    ),
                    (
                        f"CI lower bound {check.relative_ci_lo:.6f}"
                        if check.relative_ci_lo is not None
                        else "CI lower bound unavailable"
                    ),
                    *check.notes,
                )
            )
        ),
        evidence={"evaluation_split": split, "check": check.as_dict(),
                  "ci_requirement": str(gates.op(gate, "ci_requirement"))},
    )
    c2 = Condition(
        gate=gate,
        condition_id="G3.2_absolute_atpr_loss",
        frozen_key="max_absolute_atpr_loss",
        frozen_value=max_loss,
        test=(
            f"absolute ATPR loss vs {check.reference} <= {max_loss}, with the "
            f"{int(CONFIDENCE_LEVEL * 100)}% {GATE_CI_FLAVOUR} CI entirely below {max_loss}"
        ),
        status=(
            NOT_EVALUABLE
            if check.absolute_atpr_loss is None
            else (PASS if (check.loss_ok and check.loss_ci_ok) else FAIL)
        ),
        observed=None if check.absolute_atpr_loss is None else round(check.absolute_atpr_loss, 6),
        margin=(
            None if check.absolute_atpr_loss is None else max_loss - check.absolute_atpr_loss
        ),
        reason=(
            f"loss {check.absolute_atpr_loss:.6f} (negative means AegisLink retains MORE authorized "
            f"third-party links than the reference); CI upper bound "
            f"{check.atpr_loss_ci_hi if check.atpr_loss_ci_hi is None else round(check.atpr_loss_ci_hi, 6)}"
            if check.absolute_atpr_loss is not None
            else "ATPR loss undefined"
        ),
        evidence={"evaluation_split": split, "check": check.as_dict()},
    )
    return GateFamily(
        gate=gate,
        intent=str(gates.op(gate, "intent")),
        pass_condition=str(gates.op(gate, "pass_condition")),
        conditions=[c1, c2],
    )


# ======================================================================================
# Gate 4: generalization
# ======================================================================================
def evaluate_generalization(
    gates: FrozenGates,
    entity_check: ImprovementCheck,
    attack_check: ImprovementCheck,
) -> GateFamily:
    """Gate 4: the improvement must hold independently on both holdouts."""
    gate = "generalization"
    req_entity = bool(gates.value(gate, "required_on_unseen_entity_templates"))
    req_attack = bool(gates.value(gate, "required_on_unseen_attack_templates"))

    def cond(cid: str, key: str, required: bool, chk: ImprovementCheck, what: str) -> Condition:
        if not required:
            status = PASS
            reason = f"frozen value is false, so {what} is not required"
        elif not chk.evaluable:
            status = NOT_EVALUABLE
            reason = f"the improvement check on {chk.regime!r} is not evaluable: {'; '.join(chk.notes) or 'no data'}"
        else:
            status = PASS if chk.holds else FAIL
            reason = (
                f"aegislink_improvement {'holds' if chk.holds else 'does not hold'} on "
                f"{chk.regime!r}: relative UALER reduction "
                f"{chk.relative_reduction if chk.relative_reduction is None else round(chk.relative_reduction, 6)} "
                f"(CI lo {chk.relative_ci_lo if chk.relative_ci_lo is None else round(chk.relative_ci_lo, 6)}), "
                f"ATPR loss "
                f"{chk.absolute_atpr_loss if chk.absolute_atpr_loss is None else round(chk.absolute_atpr_loss, 6)} "
                f"(CI hi {chk.atpr_loss_ci_hi if chk.atpr_loss_ci_hi is None else round(chk.atpr_loss_ci_hi, 6)})"
            )
        return Condition(
            gate=gate,
            condition_id=cid,
            frozen_key=key,
            frozen_value=required,
            test=f"aegislink_improvement holds on the {what} regime ({chk.regime})",
            status=status,
            observed=chk.holds if chk.evaluable else None,
            reason=reason,
            evidence={"check": chk.as_dict()},
        )

    return GateFamily(
        gate=gate,
        intent=str(gates.op(gate, "intent")),
        pass_condition=str(gates.op(gate, "pass_condition")),
        conditions=[
            cond("G4.1_unseen_entity_templates", "required_on_unseen_entity_templates",
                 req_entity, entity_check, "unseen entity template"),
            cond("G4.2_unseen_attack_templates", "required_on_unseen_attack_templates",
                 req_attack, attack_check, "unseen attack template"),
        ],
    )


# ======================================================================================
# Gate 5: novelty
# ======================================================================================
def evaluate_novelty(gates: FrozenGates, matrix_path: Path | None = None) -> GateFamily:
    """Gate 5: the literature collision verdict."""
    gate = "novelty"
    allowed = bool(gates.value(gate, "direct_collision_allowed"))
    src = matrix_path or LITERATURE_MATRIX_PATH
    if not src.is_file():
        cond = Condition(
            gate=gate,
            condition_id="G5.1_direct_collision",
            frozen_key="direct_collision_allowed",
            frozen_value=allowed,
            test="results/literature_matrix.yaml reports verdict == NO_DIRECT_COLLISION",
            status=NOT_EVALUABLE,
            reason=f"{src} is missing; the collision verdict cannot be read",
        )
        return GateFamily(gate, str(gates.op(gate, "intent")), str(gates.op(gate, "pass_condition")), [cond])

    doc = load_yaml(src)
    verdict = str(doc.get("verdict", "<absent>"))
    # Both scored populations count: the 8 declared prior works and the candidates the six frozen
    # search concepts surfaced. Reading only prior_works would let a newly discovered collision pass.
    records = [
        r
        for key in ("prior_works", "discovered_candidates")
        for r in (doc.get(key) or [])
        if isinstance(r, Mapping)
    ]
    colliding = sorted({str(r.get("paper_id")) for r in records if bool(r.get("direct_collision"))})
    near = sorted({str(r.get("paper_id")) for r in records if bool(r.get("near_collision"))})
    # The matrix also publishes its own aggregate lists; a disagreement between them and the
    # per-record flags is itself a failure, so both are checked rather than one trusted.
    declared_colliding = sorted(str(x) for x in (doc.get("colliding_paper_ids") or []))
    declared_near = sorted(str(x) for x in (doc.get("near_collision_paper_ids") or []))
    aggregate_mismatch = declared_colliding != colliding
    ok = (
        verdict == "NO_DIRECT_COLLISION"
        and not colliding
        and not bool(doc.get("direct_collision_any"))
        and not aggregate_mismatch
    )
    cond = Condition(
        gate=gate,
        condition_id="G5.1_direct_collision",
        frozen_key="direct_collision_allowed",
        frozen_value=allowed,
        test="results/literature_matrix.yaml reports verdict == NO_DIRECT_COLLISION",
        status=PASS if ok else FAIL,
        observed=verdict,
        reason=(
            f"verdict {verdict!r}; {len(colliding)} direct collision(s), {len(near)} near "
            f"collision(s) flagged for re-check"
            + ("; AGGREGATE MISMATCH between per-record flags and colliding_paper_ids"
               if aggregate_mismatch else "")
        ),
        evidence={
            "verdict": verdict,
            "n_scored_papers": len(records),
            "n_prior_works": len(doc.get("prior_works") or []),
            "n_discovered_candidates": len(doc.get("discovered_candidates") or []),
            "direct_collisions": colliding,
            "near_collisions": near,
            "declared_colliding_paper_ids": declared_colliding,
            "declared_near_collision_paper_ids": declared_near,
            "per_record_and_aggregate_agree": not aggregate_mismatch,
            "checker": str(gates.op(gate, "checker")),
            "rubric": str(gates.op(gate, "rubric")),
            "recheck_schedule": gates.op(gate, "recheck_schedule", []),
            "post_pilot_recheck": (
                "This evaluation is the 'post_pilot' checkpoint on the frozen recheck schedule."
            ),
            "coverage_caveat": (
                "A NO_DIRECT_COLLISION verdict is evidence of no DETECTED collision over the six "
                "frozen search concepts, not proof that none exists."
            ),
        },
    )
    return GateFamily(gate, str(gates.op(gate, "intent")), str(gates.op(gate, "pass_condition")), [cond])


# ======================================================================================
# Report
# ======================================================================================
def build_decision_report(
    gates: FrozenGates, families: Sequence[GateFamily], *, context: Mapping[str, Any]
) -> dict[str, Any]:
    """Assemble ``results/pilot_decision_report.json``.

    ``overall_decision`` is ``GO`` only when all ten conditions pass. ``NO_GO`` is reserved for a
    measured shortfall, and ``INCONCLUSIVE`` for the case where something could not be measured, so
    the remedy each implies stays distinguishable: ``NO_GO`` says stop or change the method,
    ``INCONCLUSIVE`` says complete the measurement.
    """
    conditions = [c for f in families for c in f.conditions]
    n_expected = gates.n_conditions()
    if len(conditions) != n_expected:
        raise GateEvaluationError(
            f"evaluated {len(conditions)} conditions but preregistration.yaml hard_gates declares "
            f"{n_expected} frozen threshold values. Every frozen value must be evaluated or the "
            f"report would present a partial decision as a complete one."
        )

    failed = [c.condition_id for c in conditions if c.status == FAIL]
    unmeasured = [c.condition_id for c in conditions if c.status == NOT_EVALUABLE]
    passed = [c.condition_id for c in conditions if c.status == PASS]

    if failed:
        decision = "NO_GO"
    elif unmeasured:
        decision = "INCONCLUSIVE"
    else:
        decision = "GO"

    return {
        "metadata": {
            "step": "Step 5 of 8 -- primary evaluation, adaptive robustness and gate decision",
            "contract_ref": "CONTRACT.md Section 12",
            "prereg_ref": "preregistration.yaml hard_gates, gate_operationalization",
            "prereg_sha256": gates.source_sha256,
            "gate_evaluator_version": GATE_EVALUATOR_VERSION,
            "builder": "evaluation/gate_evaluator.py",
            "ci_flavour_used_for_decisions": GATE_CI_FLAVOUR,
            "n_resamples": N_RESAMPLES,
            "confidence_level": CONFIDENCE_LEVEL,
            **dict(context),
        },
        "decision": {
            "overall_decision": decision,
            "go_declared": decision == "GO",
            "combination_rule": str(gates.decision_policy.get("combination")),
            "n_conditions_declared": n_expected,
            "n_passed": len(passed),
            "n_failed": len(failed),
            "n_not_evaluable": len(unmeasured),
            "failed_conditions": failed,
            "not_evaluable_conditions": unmeasured,
            "decision_semantics": {
                "GO": "all frozen conditions measured and met",
                "NO_GO": (
                    "at least one condition was measured and fell short. "
                    + str(gates.decision_policy.get("no_go_action", "")).strip()
                ),
                "INCONCLUSIVE": (
                    "no condition failed, but at least one could not be measured by this pilot. "
                    "Distinguished from NO_GO because the remedy differs: complete the measurement "
                    "rather than stop or change the method. It is NOT a GO."
                ),
            },
            "threshold_mutability": str(gates.decision_policy.get("threshold_mutability", "")).strip(),
        },
        "gates": {f.gate: f.as_dict() for f in families},
        "conditions_flat": [c.as_dict() for c in conditions],
        "frozen_hard_gates": {k: dict(v) for k, v in gates.hard_gates.items()},
    }


def describe_gate_evaluator() -> dict[str, Any]:
    return {
        "gate_evaluator_version": GATE_EVALUATOR_VERSION,
        "contract_ref": "CONTRACT.md Section 12",
        "gate_source": "preregistration.yaml hard_gates (never a literal in this module)",
        "n_families": 5,
        "n_conditions": 10,
        "condition_status_values": [PASS, FAIL, NOT_EVALUABLE],
        "why_three_valued": (
            "FAIL means the pilot measured the thing and it fell short. NOT_EVALUABLE means the "
            "pilot could not measure it, which is a different fact and would be misreported as "
            "either a pass or a failure. Both block GO."
        ),
        "ci_flavour": GATE_CI_FLAVOUR,
        "baseline_not_trivial_conjunction": (
            "Its two frozen values read ONE conjunctive test. Reported as two conditions with "
            "separate margins, but a baseline disqualifies the project only by being good on both "
            "axes at once -- otherwise B08 and B09 could veto the work while failing the utility "
            "requirement Section 4.2 makes central."
        ),
        "known_unevaluable": {
            "misbinding_reproducible.min_model_families": (
                "requires >= 2 LLM model families; this pilot's reader is a deterministic surrogate, "
                "so the axis has size 1. Reported NOT_EVALUABLE rather than satisfied by "
                "substituting retrieval configurations for model families."
            )
        },
    }


__all__ = [
    "FAIL",
    "GATE_CI_FLAVOUR",
    "GATE_EVALUATOR_VERSION",
    "HIGH_RISK_ACTIONS",
    "NOT_EVALUABLE",
    "PASS",
    "Condition",
    "FrozenGates",
    "GateEvaluationError",
    "GateFamily",
    "ImprovementCheck",
    "build_decision_report",
    "check_improvement",
    "describe_gate_evaluator",
    "evaluate_aegislink_improvement",
    "evaluate_baseline_not_trivial",
    "evaluate_generalization",
    "evaluate_misbinding_reproducible",
    "evaluate_novelty",
    "undefended_exposure_units",
]
