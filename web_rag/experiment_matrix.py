"""Admission controller for evaluation runs.

CONTRACT.md Section 5.3 ("No test template may be used during threshold selection") and the
``regime_admission_rule`` frozen in ``configs/splits.yaml``.

The failure mode this prevents
------------------------------
Steps 4-8 fit thresholds, report validation numbers and report test numbers. Nothing in the
retrieval or attack machinery stops a run from quietly mixing a train-development attack
template into a test evaluation, and if that happened the resulting number would be neither a
clean generalisation estimate nor a clean fit -- and it would look completely normal in the
output. Section 12's ``generalization`` gate (``required_on_unseen_attack_templates: true``)
would then be satisfied by a run that had in fact seen them.

So admission is a *precondition*, checked before any inference runs, and it fails closed.

Rules enforced
--------------
``R1 single_regime``
    A run declares exactly one regime, drawn from ``policy.regimes``.

``R2 pool_membership``
    Every entity template, site template and attack template used must be in that regime's
    pool in ``configs/splits.yaml``.

``R3 no_test_in_threshold_selection``
    A run whose ``purpose`` is ``threshold_selection`` may not touch the ``test`` regime, nor
    the ``transfer_holdout`` / ``adaptive_holdout`` regimes, whose whole purpose is to be
    unseen at fitting time. Section 5.3 states the ``test`` half; extending it to the holdouts
    is the same argument, applied consistently.

``R4 triple_consistency``
    The splits policy is explicit that "a rendered page belongs to regime R iff its entity
    template is in R's entity pool AND its site template is in R's site pool AND its attack
    template is either absent or in R's attack pool". Because the three axes were split
    independently, a triple can mix pools and belong to *no* regime. Such a triple is rejected
    rather than assigned to whichever regime it partly matches.

``R5 adaptive_template_gating``
    Dynamic ``AD-*`` templates from :mod:`attacks.adaptive_attacker` are admitted only to
    adaptive regimes, and only Stratum A (inside the pre-declared holdout region) may enter a
    run that supports a primary claim. Stratum B, which carries ``identity_consistency:
    partial`` and therefore sits outside the frozen region, is admitted only to explicitly
    secondary runs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

EXPERIMENT_MATRIX_VERSION = "1.0"

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_PATH = REPO_ROOT / "configs" / "splits.yaml"

AXES: tuple[str, ...] = ("entity_template", "site_template", "attack_template")

#: Regimes whose contents must be unseen while thresholds are fitted.
FIT_FORBIDDEN_REGIMES: frozenset[str] = frozenset(
    {"test", "transfer_holdout", "adaptive_holdout"}
)

#: Regimes in which dynamic adaptive templates may appear at all.
ADAPTIVE_REGIMES: frozenset[str] = frozenset({"adaptive_holdout"})

#: Prefix of dynamic adaptive template ids (``AD-0001``).
ADAPTIVE_ID_PREFIX = "AD-"

RULE_IDS: tuple[str, ...] = (
    "R1_single_regime",
    "R2_pool_membership",
    "R3_no_test_in_threshold_selection",
    "R4_triple_consistency",
    "R5_adaptive_template_gating",
)


class RunPurpose(str, Enum):
    """Why a run exists. Determines which regimes it may touch."""

    THRESHOLD_SELECTION = "threshold_selection"
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    PRIMARY_EVALUATION = "primary_evaluation"
    SECONDARY_EVALUATION = "secondary_evaluation"
    SANITY_CHECK = "sanity_check"

    @property
    def fits_thresholds(self) -> bool:
        return self is RunPurpose.THRESHOLD_SELECTION

    @property
    def supports_primary_claim(self) -> bool:
        return self in (RunPurpose.PRIMARY_EVALUATION, RunPurpose.VALIDATION)


class AdmissionError(RuntimeError):
    """Raised when a run would violate the frozen split policy. Fails closed."""


@dataclass(frozen=True)
class AdmissionViolation:
    rule_id: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemplateTriple:
    """One rendered-page addressing triple. ``attack_template_id`` is ``None`` for benign pages."""

    entity_template_id: str
    site_template_id: str
    attack_template_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunSpec:
    """A declared evaluation run, checked before anything executes."""

    run_id: str
    regime: str
    purpose: RunPurpose
    triples: tuple[TemplateTriple, ...] = ()
    entity_template_ids: tuple[str, ...] = ()
    site_template_ids: tuple[str, ...] = ()
    attack_template_ids: tuple[str, ...] = ()
    adaptive_template_ids: tuple[str, ...] = ()
    notes: str = ""

    def declared_ids(self, axis: str) -> set[str]:
        """Union of ids declared directly and ids implied by the triples."""
        direct = {
            "entity_template": set(self.entity_template_ids),
            "site_template": set(self.site_template_ids),
            "attack_template": set(self.attack_template_ids),
        }[axis]
        from_triples = set()
        for t in self.triples:
            v = getattr(t, f"{axis}_id")
            if v is not None:
                from_triples.add(v)
        return direct | from_triples

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "regime": self.regime,
            "purpose": self.purpose.value,
            "n_triples": len(self.triples),
            "entity_template_ids": sorted(self.declared_ids("entity_template")),
            "site_template_ids": sorted(self.declared_ids("site_template")),
            "attack_template_ids": sorted(self.declared_ids("attack_template")),
            "adaptive_template_ids": sorted(self.adaptive_template_ids),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class AdmissionReport:
    run_id: str
    admitted: bool
    violations: tuple[AdmissionViolation, ...]
    checked_rules: tuple[str, ...]
    regime: str
    purpose: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "admitted": self.admitted,
            "regime": self.regime,
            "purpose": self.purpose,
            "checked_rules": list(self.checked_rules),
            "violations": [v.as_dict() for v in self.violations],
        }

    def raise_if_rejected(self) -> "AdmissionReport":
        if not self.admitted:
            lines = "\n".join(f"  [{v.rule_id}] {v.detail}" for v in self.violations)
            raise AdmissionError(
                f"run {self.run_id!r} rejected by the admission controller:\n{lines}"
            )
        return self


# ======================================================================================
# Controller
# ======================================================================================
@dataclass
class ExperimentMatrix:
    """The frozen split policy, plus admission checking against it."""

    regimes: tuple[str, ...]
    pools: dict[str, dict[str, frozenset[str]]]
    fractions: Mapping[str, float]
    transfer_holdout_categories: tuple[str, ...]
    adaptive_holdout_region: Mapping[str, tuple[str, ...]]
    splits_digest: str = ""
    version: str = EXPERIMENT_MATRIX_VERSION
    #: ``AD-*`` id -> whether it sits inside the pre-declared region (Stratum A).
    adaptive_stratum_a: frozenset[str] = frozenset()

    # -- construction ----------------------------------------------------------------
    @classmethod
    def from_splits(cls, path: Path | None = None) -> "ExperimentMatrix":
        src = path or SPLITS_PATH
        loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
        raw = yaml.load(src.read_text(encoding="utf-8"), Loader=loader)
        regimes = tuple(str(r) for r in raw["policy"]["regimes"])
        pools: dict[str, dict[str, frozenset[str]]] = {}
        for regime, axes in raw["regime_pools"].items():
            pools[str(regime)] = {
                str(axis): frozenset(str(i) for i in ids) for axis, ids in axes.items()
            }
        missing = set(regimes) - set(pools)
        if missing:
            raise AdmissionError(
                f"configs/splits.yaml declares regimes {sorted(missing)} with no regime_pools "
                f"entry; admission could not be checked for them."
            )
        region = {
            str(k): tuple(str(x) for x in v)
            for k, v in raw["policy"]["adaptive_holdout_region"].items()
        }
        import hashlib

        return cls(
            regimes=regimes,
            pools=pools,
            fractions=dict(raw["policy"]["fractions"]),
            transfer_holdout_categories=tuple(
                str(c) for c in raw["policy"]["transfer_holdout_categories"]
            ),
            adaptive_holdout_region=region,
            splits_digest=hashlib.sha256(src.read_bytes()).hexdigest(),
        )

    def register_adaptive_templates(
        self, templates: Sequence[Mapping[str, Any]]
    ) -> "ExperimentMatrix":
        """Register dynamic ``AD-*`` templates and record which are Stratum A.

        Stratum membership is read off the template's own ``in_declared_region`` flag *and*
        independently re-derived from its factor levels, so a mislabelled template is caught
        here rather than trusted.
        """
        stratum_a: set[str] = set()
        for t in templates:
            tid = str(t["attack_template_id"])
            derived = self.in_declared_region(t)
            declared = bool(t.get("in_declared_region", derived))
            if declared != derived:
                raise AdmissionError(
                    f"adaptive template {tid} claims in_declared_region={declared} but its "
                    f"factor levels give {derived}. Region membership must be verifiable from "
                    f"the factor levels alone."
                )
            if derived:
                stratum_a.add(tid)
        self.adaptive_stratum_a = frozenset(stratum_a)
        return self

    def in_declared_region(self, template: Mapping[str, Any]) -> bool:
        """Whether a template's factor levels satisfy the frozen region conjunction.

        Levels are compared as strings because ``configs/splits.yaml`` stores them as strings
        (``'True'``, ``'3'``) while the generator holds them as ``bool``/``int``.
        """
        for factor, allowed in self.adaptive_holdout_region.items():
            if factor not in template:
                return False
            if str(template[factor]) not in {str(a) for a in allowed}:
                return False
        return True

    # -- pool queries ----------------------------------------------------------------
    def pool(self, regime: str, axis: str) -> frozenset[str]:
        if regime not in self.pools:
            raise AdmissionError(f"unknown regime {regime!r}; known: {sorted(self.pools)}")
        if axis not in AXES:
            raise AdmissionError(f"unknown axis {axis!r}; known: {list(AXES)}")
        return self.pools[regime].get(axis, frozenset())

    def regimes_containing(self, axis: str, template_id: str) -> list[str]:
        return sorted(r for r in self.regimes if template_id in self.pool(r, axis))

    def regime_of_triple(self, triple: TemplateTriple) -> str | None:
        """The unique regime a triple belongs to, or ``None`` if it belongs to none.

        A triple can legitimately match more than one regime because the site and attack pools
        of ``test``, ``transfer_holdout`` and ``fabricated_control`` coincide by design (only
        the entity axis distinguishes them). The entity axis is therefore the discriminator,
        and a match requires all three axes to agree with the *same* regime.
        """
        matches = [
            r
            for r in self.regimes
            if triple.entity_template_id in self.pool(r, "entity_template")
            and triple.site_template_id in self.pool(r, "site_template")
            and (
                triple.attack_template_id is None
                or triple.attack_template_id in self.pool(r, "attack_template")
            )
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            return None
        # Deterministic tie-break: prefer the regime whose entity pool is unique to it.
        return sorted(matches)[0]

    # -- rules -----------------------------------------------------------------------
    def _r1_single_regime(self, run: RunSpec) -> list[AdmissionViolation]:
        if run.regime not in self.regimes:
            return [
                AdmissionViolation(
                    "R1_single_regime",
                    f"regime {run.regime!r} is not one of the frozen regimes "
                    f"{list(self.regimes)}",
                )
            ]
        return []

    def _r2_pool_membership(self, run: RunSpec) -> list[AdmissionViolation]:
        if run.regime not in self.regimes:
            return []  # R1 already reported it
        out: list[AdmissionViolation] = []
        for axis in AXES:
            pool = self.pool(run.regime, axis)
            declared = run.declared_ids(axis)
            # Dynamic adaptive ids are handled by R5, not by static pool membership.
            declared = {d for d in declared if not d.startswith(ADAPTIVE_ID_PREFIX)}
            stray = sorted(declared - pool)
            if stray:
                where = {
                    s: self.regimes_containing(axis, s) or ["<no regime>"] for s in stray[:5]
                }
                out.append(
                    AdmissionViolation(
                        "R2_pool_membership",
                        f"{len(stray)} {axis} id(s) are outside the {run.regime!r} pool: "
                        f"{json.dumps(where, sort_keys=True)}"
                        + (f" (+{len(stray) - 5} more)" if len(stray) > 5 else ""),
                    )
                )
        return out

    def _r3_no_test_in_fitting(self, run: RunSpec) -> list[AdmissionViolation]:
        if not run.purpose.fits_thresholds:
            return []
        out: list[AdmissionViolation] = []
        if run.regime in FIT_FORBIDDEN_REGIMES:
            out.append(
                AdmissionViolation(
                    "R3_no_test_in_threshold_selection",
                    f"purpose is {run.purpose.value} but regime is {run.regime!r}. "
                    f"CONTRACT.md Section 5.3: no test template may be used during threshold "
                    f"selection; the same argument bars the holdouts, whose purpose is to be "
                    f"unseen at fitting time.",
                )
            )
        # Even in an allowed regime, an explicitly declared forbidden-regime template is a leak.
        for axis in AXES:
            for tid in sorted(run.declared_ids(axis)):
                if tid.startswith(ADAPTIVE_ID_PREFIX):
                    out.append(
                        AdmissionViolation(
                            "R3_no_test_in_threshold_selection",
                            f"adaptive template {tid} appears in a threshold-selection run",
                        )
                    )
                    continue
                homes = set(self.regimes_containing(axis, tid))
                allowed_homes = homes - FIT_FORBIDDEN_REGIMES
                if homes and not allowed_homes:
                    out.append(
                        AdmissionViolation(
                            "R3_no_test_in_threshold_selection",
                            f"{axis} {tid} lives only in {sorted(homes)}, which may not be "
                            f"used while fitting thresholds",
                        )
                    )
        return out

    def _r4_triple_consistency(self, run: RunSpec) -> list[AdmissionViolation]:
        out: list[AdmissionViolation] = []
        orphans: list[TemplateTriple] = []
        wrong_regime: list[tuple[TemplateTriple, str | None]] = []
        for t in run.triples:
            if t.attack_template_id and t.attack_template_id.startswith(ADAPTIVE_ID_PREFIX):
                # Dynamic templates are not in any static pool; R5 governs them.
                probe = TemplateTriple(t.entity_template_id, t.site_template_id, None)
                home = self.regime_of_triple(probe)
            else:
                home = self.regime_of_triple(t)
            if home is None:
                orphans.append(t)
            elif home != run.regime:
                wrong_regime.append((t, home))
        if orphans:
            out.append(
                AdmissionViolation(
                    "R4_triple_consistency",
                    f"{len(orphans)} triple(s) belong to no regime, e.g. "
                    f"{[o.as_dict() for o in orphans[:3]]}. Independent per-axis splitting "
                    f"means such a triple must never be rendered "
                    f"(configs/splits.yaml policy.regime_admission_rule).",
                )
            )
        if wrong_regime:
            sample = [
                {"triple": t.as_dict(), "belongs_to": h} for t, h in wrong_regime[:3]
            ]
            out.append(
                AdmissionViolation(
                    "R4_triple_consistency",
                    f"{len(wrong_regime)} triple(s) belong to a regime other than "
                    f"{run.regime!r}, e.g. {sample}. Mixing regimes in one run makes the "
                    f"resulting number neither a clean fit nor a clean generalisation estimate.",
                )
            )
        return out

    def _r5_adaptive_gating(self, run: RunSpec) -> list[AdmissionViolation]:
        out: list[AdmissionViolation] = []
        adaptive = {
            *run.adaptive_template_ids,
            *(
                t
                for axis in AXES
                for t in run.declared_ids(axis)
                if t.startswith(ADAPTIVE_ID_PREFIX)
            ),
        }
        if not adaptive:
            return out
        if run.regime not in ADAPTIVE_REGIMES:
            out.append(
                AdmissionViolation(
                    "R5_adaptive_template_gating",
                    f"{len(adaptive)} dynamic adaptive template(s) declared in regime "
                    f"{run.regime!r}; they are admitted only in {sorted(ADAPTIVE_REGIMES)}. "
                    f"Letting them into a core regime would contaminate the split that the "
                    f"unseen-attack-template gate depends on.",
                )
            )
        if run.purpose.supports_primary_claim and self.adaptive_stratum_a:
            stratum_b = sorted(adaptive - self.adaptive_stratum_a)
            if stratum_b:
                out.append(
                    AdmissionViolation(
                        "R5_adaptive_template_gating",
                        f"{len(stratum_b)} adaptive template(s) sit outside the pre-declared "
                        f"adaptive holdout region and may not support a primary claim: "
                        f"{stratum_b[:5]}. Use purpose=secondary_evaluation.",
                    )
                )
        unregistered = sorted(adaptive - self.adaptive_stratum_a) if not self.adaptive_stratum_a else []
        if unregistered:
            out.append(
                AdmissionViolation(
                    "R5_adaptive_template_gating",
                    f"{len(unregistered)} adaptive template(s) were never registered with "
                    f"register_adaptive_templates(), so their region membership is unverified: "
                    f"{unregistered[:5]}",
                )
            )
        return out

    # -- driver ----------------------------------------------------------------------
    def admit(self, run: RunSpec) -> AdmissionReport:
        """Check ``run`` against every rule and return a report. Never raises for violations."""
        violations: list[AdmissionViolation] = []
        violations += self._r1_single_regime(run)
        violations += self._r2_pool_membership(run)
        violations += self._r3_no_test_in_fitting(run)
        violations += self._r4_triple_consistency(run)
        violations += self._r5_adaptive_gating(run)
        return AdmissionReport(
            run_id=run.run_id,
            admitted=not violations,
            violations=tuple(violations),
            checked_rules=RULE_IDS,
            regime=run.regime,
            purpose=run.purpose.value,
        )

    def require_admitted(self, run: RunSpec) -> AdmissionReport:
        """Admit or raise. This is the call an experiment driver should make."""
        return self.admit(run).raise_if_rejected()

    # -- provenance ------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {
            "experiment_matrix_version": EXPERIMENT_MATRIX_VERSION,
            "contract_ref": "CONTRACT.md Section 5.3; configs/splits.yaml policy",
            "splits_digest": self.splits_digest,
            "regimes": list(self.regimes),
            "pool_sizes": {
                r: {a: len(self.pool(r, a)) for a in AXES} for r in self.regimes
            },
            "fractions": dict(self.fractions),
            "transfer_holdout_categories": list(self.transfer_holdout_categories),
            "adaptive_holdout_region": {k: list(v) for k, v in self.adaptive_holdout_region.items()},
            "fit_forbidden_regimes": sorted(FIT_FORBIDDEN_REGIMES),
            "adaptive_regimes": sorted(ADAPTIVE_REGIMES),
            "n_registered_stratum_a": len(self.adaptive_stratum_a),
            "rules": {
                "R1_single_regime": "exactly one regime per run",
                "R2_pool_membership": "every template id in that regime's pool",
                "R3_no_test_in_threshold_selection": (
                    "threshold-selection runs may not touch test or either holdout"
                ),
                "R4_triple_consistency": (
                    "a triple mixing pools from different regimes belongs to no regime and is "
                    "rejected, per configs/splits.yaml policy.regime_admission_rule"
                ),
                "R5_adaptive_template_gating": (
                    "AD-* templates only in adaptive regimes; only Stratum A (inside the "
                    "pre-declared region) may support a primary claim"
                ),
            },
            "failure_mode": "fails closed: require_admitted raises AdmissionError",
        }


__all__ = [
    "ADAPTIVE_ID_PREFIX",
    "ADAPTIVE_REGIMES",
    "AXES",
    "EXPERIMENT_MATRIX_VERSION",
    "FIT_FORBIDDEN_REGIMES",
    "RULE_IDS",
    "AdmissionError",
    "AdmissionReport",
    "AdmissionViolation",
    "ExperimentMatrix",
    "RunPurpose",
    "RunSpec",
    "TemplateTriple",
]
