"""Confirmatory evaluation on the held-out regimes, under frozen Step 4 parameters.

CONTRACT.md Sections 3 (RQ1, RQ4), 5.3, 11, 12; ``preregistration.yaml`` ``primary_comparison``.

What makes this the *confirmatory* step
---------------------------------------
Step 4 fitted AegisLink's probability calibration on ``train_development`` and reported
``train_development`` and ``validation``. Step 5 reads ``test``, ``transfer_holdout`` and
``adaptive_holdout``, and its whole scientific value rests on those splits never having influenced a
parameter. So this module does not *re-fit anything*. It loads the Step 4 fit from
``results/aegislink_defense_evaluation.json``, verifies its provenance, and refuses to run if the fit
came from anywhere but ``train_development``.

Three guards, because "we didn't refit" is otherwise an unverifiable claim
-------------------------------------------------------------------------
``FrozenParameters.load``
    Reads the Step 4 artifact, records its SHA-256, and raises :class:`RefitError` unless the Platt
    transform's ``fit_regime`` is ``train_development`` and the threshold vector is the unfitted
    design default with its ordering constraint intact.

``assert_no_refitting_hooks``
    Asserts that this module never calls :func:`aegislink.framework.fit_platt`. The assertion is
    executable rather than documentary: ``tests/test_primary_evaluation.py`` replaces ``fit_platt``
    with a function that raises and then runs a full evaluation, so a future edit that reintroduces
    fitting fails the suite instead of quietly producing an optimistic number.

the admission controller
    Every regime is evaluated under a declared :class:`~web_rag.experiment_matrix.RunSpec` with
    ``purpose=PRIMARY_EVALUATION`` and *all three* template axes declared, so rule ``R2`` checks the
    entity, site and attack templates actually used against that regime's frozen pool. Declaring
    nothing would pass R2 vacuously; the pools were verified to be cleanly scoped by
    ``workflow/diagnostics/check_regime_template_scoping.py``, which is what makes the strict
    declaration possible.

The regimes
-----------
``validation``
    Re-run here even though Step 4 reported it, for two reasons the preregistration forces. The
    reference baseline is *selected* on validation (``primary_comparison.reference_baseline_selection``)
    and the ``baseline_not_trivial`` gate is *decided* on validation
    (``gate_operationalization.baseline_not_trivial.evaluation_split``). Recomputing them inside the
    same process that computes the test numbers also cross-checks Step 4: a disagreement means the
    replay drifted.
``test``
    The confirmatory split. 30% of entity templates, unseen at fitting time.
``transfer_holdout``
    Disjoint entity *categories* (fitness_studio, bookshop). Supplies the
    "unseen entity templates" half of the ``generalization`` gate.
``fabricated_control``
    Reported as a control, never as a primary result: these entities have no authorized domain at
    all, so a defense should abstain. Included because a defense that presents links for a
    fabricated business is failing in a way no rate on the other regimes would show.

``adaptive_holdout`` is not here. It needs a different corpus -- the 600 adaptive pages -- and so
lives in :mod:`experiments.adaptive_evaluation`.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from aegislink.ablations import ABLATION_IDS, build_ablations
from aegislink.framework import (
    ACTION_RISK_ORDER,
    PlattScaling,
    RiskThresholds,
    ThresholdOrderingError,
)
from aegislink.pipeline_adapter import DefenseEvaluationHarness
from aegislink.verifier import AegisLink, AegisLinkConfig
from baselines.registry import BASELINE_IDS, build_baselines
from evaluation.defense_metrics import TripleOutcome, compute_metrics
from statistics.bootstrap import RatioUnit
from web_rag.experiment_matrix import ExperimentMatrix, RunPurpose, RunSpec

PRIMARY_EVALUATION_VERSION = "1.0"

REPO_ROOT = Path(__file__).resolve().parent.parent
STEP4_ARTIFACT = REPO_ROOT / "results" / "aegislink_defense_evaluation.json"
SPLITS_PATH = REPO_ROOT / "configs" / "splits.yaml"

#: The regime the Step 4 fit is allowed to have come from. Anything else is a refit.
PERMITTED_FIT_REGIME = "train_development"

#: Regimes this module evaluates, in report order. ``adaptive_holdout`` is handled by
#: :mod:`experiments.adaptive_evaluation` because it needs the adaptive corpus.
EVALUATION_REGIMES: tuple[str, ...] = (
    "validation",
    "test",
    "transfer_holdout",
    "fabricated_control",
)

#: Regimes whose numbers support a confirmatory claim. ``validation`` is a selection/gate split and
#: ``fabricated_control`` is a control, so neither is a primary result.
CONFIRMATORY_REGIMES: tuple[str, ...] = ("test", "transfer_holdout")

#: All 18 preregistered configurations, in report order.
CONFIG_ORDER: tuple[str, ...] = ("aegislink_full", *ABLATION_IDS, *BASELINE_IDS)


class RefitError(RuntimeError):
    """Raised when a parameter used on a held-out split was not frozen on train_development."""


class RegimeError(RuntimeError):
    """Raised when a regime is empty or unknown -- an empty regime must not report as 0.0."""


def load_yaml(path: Path) -> Any:
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    return yaml.load(path.read_text(encoding="utf-8"), Loader=loader)


# ======================================================================================
# Frozen parameters
# ======================================================================================
@dataclass(frozen=True)
class FrozenParameters:
    """AegisLink's Step 4 parameters, loaded rather than refitted.

    ``source_sha256`` is the digest of the whole Step 4 artifact, so the exact file these numbers
    came from is recoverable from the Step 5 output.
    """

    calibration: PlattScaling
    thresholds: RiskThresholds
    replay_fingerprint: str
    source_path: str
    source_sha256: str
    n_fit_triples: int
    n_fit_queries: int

    @classmethod
    def load(cls, path: Path | None = None) -> "FrozenParameters":
        src = path or STEP4_ARTIFACT
        if not src.is_file():
            raise RefitError(
                f"{src} is missing. Step 5 must reuse the Step 4 fit; it may not create one, "
                f"because a parameter fitted now would have seen the held-out splits."
            )
        raw = src.read_bytes()
        doc = json.loads(raw)
        fit = doc["fitting"]
        cal_raw = fit["calibration"]

        if str(cal_raw["fit_regime"]) != PERMITTED_FIT_REGIME:
            raise RefitError(
                f"Platt calibration reports fit_regime={cal_raw['fit_regime']!r}, expected "
                f"{PERMITTED_FIT_REGIME!r}. CONTRACT.md Section 5.3 forbids a held-out split from "
                f"influencing a parameter, so an evaluation under this fit would not be "
                f"confirmatory."
            )
        calibration = PlattScaling(
            a=float(cal_raw["a"]),
            b=float(cal_raw["b"]),
            fit_regime=str(cal_raw["fit_regime"]),
            n_fit=int(cal_raw["n_fit"]),
            brier_before=cal_raw.get("brier_before"),
            brier_after=cal_raw.get("brier_after"),
        )

        # ``tau_plausible`` in the artifact is derived from ``plausible_floor``, not a field, so it
        # is deliberately not passed back in: reconstructing it as an input would let a future
        # artifact carry an inconsistent pair without anything noticing.
        thr_raw = fit["thresholds"]
        thresholds = RiskThresholds(
            tau={str(k): float(v) for k, v in thr_raw["tau"].items()},
            plausible_floor=float(thr_raw["plausible_floor"]),
            shared_tau=thr_raw.get("shared_tau"),
            fit_regime=str(thr_raw["fit_regime"]),
            fit_note=str(thr_raw.get("fit_note", "")),
        )
        rebuilt = thresholds.as_dict()
        drift = {
            k: (rebuilt.get(k), thr_raw.get(k))
            for k in ("tau", "tau_plausible", "plausible_floor", "shared_tau")
            if rebuilt.get(k) != thr_raw.get(k)
        }
        if drift:
            raise RefitError(
                f"the threshold vector rebuilt from the Step 4 artifact does not round-trip: "
                f"{json.dumps(drift, default=str)}. Step 5 would then be evaluating a different "
                f"decision rule than Step 4 fitted."
            )
        if not thresholds.is_monotonic:
            raise ThresholdOrderingError(
                "the frozen threshold vector violates tau_browse < tau_contact < tau_book < "
                "tau_login < tau_pay; preregistration.yaml marks the ordering a hard constraint"
            )

        return cls(
            calibration=calibration,
            thresholds=thresholds,
            replay_fingerprint=str(doc["metadata"]["replay_fingerprint"]),
            source_path=str(src.relative_to(REPO_ROOT)),
            source_sha256=hashlib.sha256(raw).hexdigest(),
            n_fit_triples=int(fit["n_fit_triples"]),
            n_fit_queries=int(fit["n_fit_queries"]),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source_path,
            "source_sha256": self.source_sha256,
            "replay_fingerprint": self.replay_fingerprint,
            "calibration": self.calibration.as_dict(),
            "thresholds": self.thresholds.as_dict(),
            "n_fit_queries": self.n_fit_queries,
            "n_fit_triples": self.n_fit_triples,
            "fit_regime": self.calibration.fit_regime,
            "refit_on_evaluation_splits": False,
            "policy": (
                "Loaded from the Step 4 artifact and used verbatim. No parameter is fitted on "
                "test, transfer_holdout or adaptive_holdout; FrozenParameters.load raises "
                "RefitError unless the Platt fit_regime is train_development, and "
                "tests/test_primary_evaluation.py runs a full evaluation with fit_platt replaced "
                "by a function that raises."
            ),
        }


def assert_no_refitting_hooks() -> dict[str, Any]:
    """Assert this module holds no reference to a fitting routine.

    Checked by inspecting the module's own globals rather than by reading the source, so an import
    added later is caught even if it is never called on the happy path.
    """
    import sys as _sys

    mod = _sys.modules[__name__]
    forbidden = [n for n in ("fit_platt", "fit_thresholds", "tune_thresholds") if hasattr(mod, n)]
    if forbidden:
        raise RefitError(
            f"evaluation/primary_evaluation.py imports fitting routine(s) {forbidden}. The "
            f"confirmatory evaluation must consume frozen parameters only."
        )
    return {
        "no_fitting_symbols_imported": True,
        "checked": ["fit_platt", "fit_thresholds", "tune_thresholds"],
        "runtime_check": (
            "tests/test_primary_evaluation.py::test_evaluation_never_refits monkeypatches "
            "aegislink.framework.fit_platt to raise and runs a full regime evaluation."
        ),
    }


# ======================================================================================
# Evaluator-side bookkeeping
# ======================================================================================
@dataclass
class EvaluatorContext:
    """Ground truth and split bookkeeping. Never handed to a defense."""

    oracle: Any
    third_party_pairs: set[tuple[str, str]]
    official_of: dict[str, str]
    entity_status: dict[str, str]
    entity_category: dict[str, str]
    entity_to_regime: dict[str, str]
    graph_edges: list[Mapping[str, Any]]
    entity_template_of: dict[str, str]
    site_templates_of_regime: dict[str, set[str]]
    attack_templates_of_regime: dict[str, set[str]]

    def regime_of(self, entity_id: str) -> str:
        return self.entity_to_regime.get(entity_id, "unassigned")

    def unconditioned_denominators(self, regime: str) -> tuple[int, int]:
        """Every authorized (third-party, any) link in ``regime``, retrieved or not."""
        n_tp = n_all = 0
        for edge in self.graph_edges:
            if not edge["authorized"]:
                continue
            eid, did = str(edge["entity_id"]), str(edge["domain_id"])
            if self.entity_to_regime.get(eid) != regime:
                continue
            n_all += 1
            if (eid, did) in self.third_party_pairs:
                n_tp += 1
        return n_tp, n_all


def build_evaluator_context(corpus: Any) -> EvaluatorContext:
    """Assemble the evaluator side from the frozen registry artifacts."""
    from web_rag.trace_recorder import GroundTruthOracle

    domains_raw = load_yaml(REPO_ROOT / "registry" / "domains.yaml")
    graph_edges = load_yaml(REPO_ROOT / "registry" / "authorization_graph.yaml")["edges"]
    aliases = load_yaml(REPO_ROOT / "registry" / "aliases.yaml")["aliases"]
    entities = load_yaml(REPO_ROOT / "registry" / "entities.yaml")["entities"]
    splits = load_yaml(SPLITS_PATH)
    pages = load_yaml(REPO_ROOT / "data" / "benchmark" / "page_manifest.yaml")["pages"]

    oracle = GroundTruthOracle.from_registry(
        graph_edges=graph_edges,
        delegations=domains_raw["delegations"],
        domain_hosts={d.domain_id: d.domain for d in corpus.registry.domains.values()},
        entities=entities,
        aliases=aliases,
    )
    e2r = {str(k): str(v) for k, v in splits["entity_to_regime"].items()}

    # entities.yaml carries no entity_template_id, so the entity -> template map is derived from
    # the page manifest. Verified single-valued by
    # workflow/diagnostics/check_regime_template_scoping.py.
    tmpl_of: dict[str, str] = {}
    site_of_regime: dict[str, set[str]] = defaultdict(set)
    attack_of_regime: dict[str, set[str]] = defaultdict(set)
    for p in pages:
        eid = str(p["entity_id"])
        tmpl_of.setdefault(eid, str(p["entity_template_id"]))
        regime = e2r.get(eid)
        if regime is None:
            continue
        site_of_regime[regime].add(str(p["site_template_id"]))
        if p["attack_template_id"]:
            attack_of_regime[regime].add(str(p["attack_template_id"]))

    return EvaluatorContext(
        oracle=oracle,
        third_party_pairs={
            (str(d["entity_id"]), str(d["domain_id"])) for d in domains_raw["delegations"]
        },
        official_of={
            str(rec["owner_entity_id"]): str(did)
            for did, rec in domains_raw["domains"].items()
            if rec.get("owner_entity_id")
        },
        entity_status={str(e["entity_id"]): str(e["status"]) for e in entities},
        entity_category={str(e["entity_id"]): str(e.get("category", "unknown")) for e in entities},
        entity_to_regime=e2r,
        graph_edges=graph_edges,
        entity_template_of=tmpl_of,
        site_templates_of_regime=dict(site_of_regime),
        attack_templates_of_regime=dict(attack_of_regime),
    )


# ======================================================================================
# Admission
# ======================================================================================
def admit_regime(
    matrix: ExperimentMatrix,
    ev: EvaluatorContext,
    regime: str,
    entity_ids: Sequence[str],
    *,
    run_id_prefix: str = "step5",
    purpose: RunPurpose = RunPurpose.PRIMARY_EVALUATION,
) -> dict[str, Any]:
    """Declare and admit one regime's evaluation run, with all three template axes declared.

    All three axes are declared on purpose. R2 checks declared ids against the regime's frozen pool,
    so declaring nothing would pass it vacuously and the "no test template during threshold
    selection" guarantee would rest on nothing checkable.
    """
    entity_templates = sorted({ev.entity_template_of[e] for e in entity_ids if e in ev.entity_template_of})
    run = RunSpec(
        run_id=f"{run_id_prefix}_{regime}",
        regime=regime,
        purpose=purpose,
        entity_template_ids=tuple(entity_templates),
        site_template_ids=tuple(sorted(ev.site_templates_of_regime.get(regime, set()))),
        attack_template_ids=tuple(sorted(ev.attack_templates_of_regime.get(regime, set()))),
        notes=(
            f"Step 5 confirmatory evaluation of {regime}. Parameters are the frozen Step 4 fit; "
            f"nothing is fitted here."
        ),
    )
    report = matrix.admit(run).raise_if_rejected()
    return {
        "run": run.as_dict(),
        "admission": report.as_dict(),
        "n_entities": len(entity_ids),
        "n_entity_templates": len(entity_templates),
    }


# ======================================================================================
# Defenses
# ======================================================================================
def build_all_configurations(
    params: FrozenParameters, *, replay_fingerprint: str
) -> dict[str, Any]:
    """The 18 preregistered configurations, all sharing the frozen calibration.

    Baselines receive the same Platt transform so the calibration column is comparable across the
    table; their underlying signals are untouched, and the transform is monotone so no ranking moves.
    """
    base = AegisLinkConfig(thresholds=params.thresholds, calibration=params.calibration)
    out: dict[str, Any] = {
        "aegislink_full": AegisLink(config=base, expected_replay_fingerprint=replay_fingerprint)
    }
    out.update(build_ablations(base, expected_replay_fingerprint=replay_fingerprint))
    for bid, d in build_baselines().items():
        out[bid] = d.with_calibration(params.calibration)
    missing = set(CONFIG_ORDER) - set(out)
    if missing:
        raise RegimeError(f"configuration set is incomplete: missing {sorted(missing)}")
    return out


# ======================================================================================
# Sweep
# ======================================================================================
@dataclass
class SweepOutcome:
    """Per-configuration triple outcomes and pipeline provenance for one query set.

    Two triple sets are carried, and the difference between them is a correction to Step 4 rather
    than an implementation detail:

    ``triples``
        One triple per unique ``(query, domain)``. This is the canonical set. An authorized
        third-party *link* is a property of an ``(entity, domain, action)`` triple in the
        authorization graph, and the defense makes exactly one decision per domain, so a domain
        retrieved twice is one link.
    ``triples_per_candidate``
        One triple per retrieved *candidate*, which is what Step 4 scored. Retained solely to
        reconcile against the Step 4 artifact: it reproduces those numbers exactly, which is what
        proves the two code paths agree and isolates the de-duplication as the only difference.

    Step 4's rule made ATPR's denominator a function of how many listing pages a shared provider
    happens to serve -- a corpus artifact, not an authorization fact -- and entered one decision twice
    as if it were two independent observations. Measured on the frozen replay by
    ``workflow/diagnostics/check_duplicate_domain_candidates.py``: 6 of 102 authorized third-party
    links on ``validation``, 1 of 148 on ``test``.
    """

    triples: dict[str, list[TripleOutcome]] = field(default_factory=dict)
    triples_per_candidate: dict[str, list[TripleOutcome]] = field(default_factory=dict)
    pipelines: dict[str, dict[str, Any]] = field(default_factory=dict)
    n_duplicate_domain_candidates: int = 0
    query_order: list[str] = field(default_factory=list)


def run_sweep(
    defenses: Mapping[str, Any],
    harness: DefenseEvaluationHarness,
    results: Sequence[Any],
    ev: EvaluatorContext,
    fetch: Callable[[str], str],
    *,
    progress_label: str = "sweep",
) -> SweepOutcome:
    """Run every configuration over ``results``, joining decisions to ground truth.

    Emits both triple sets described on :class:`SweepOutcome` in a single pass, so the Step 4
    reconciliation costs no extra sweep.
    """
    out = SweepOutcome(query_order=[r.query_id for r in results])
    dupes = 0
    t_start = time.time()

    def make(r: Any, c: Any, dec: Any) -> TripleOutcome:
        return TripleOutcome(
            query_id=r.query_id,
            entity_id=r.entity_id,
            action=r.action,
            domain_id=c.domain_id,
            rank=c.rank,
            decision=dec,
            authorized=ev.oracle.is_authorized(r.entity_id, c.domain_id, r.action),
            is_official_domain=ev.official_of.get(r.entity_id) == c.domain_id,
            is_authorized_third_party=(r.entity_id, c.domain_id) in ev.third_party_pairs,
            entity_status=ev.entity_status.get(r.entity_id, "unknown"),
            regime=ev.regime_of(r.entity_id),
        )

    for n, (name, defense) in enumerate(defenses.items(), start=1):
        t0 = time.time()
        pipeline = harness.pipeline(defense)
        triples: list[TripleOutcome] = []
        per_candidate: list[TripleOutcome] = []
        for r in results:
            decisions = pipeline.decisions_for(r, fetch)
            best_rank: dict[str, Any] = {}
            for c in r.candidates:
                dec = decisions.get(c.domain_id)
                if dec is None:
                    continue
                per_candidate.append(make(r, c, dec))
                prev = best_rank.get(c.domain_id)
                if prev is None:
                    best_rank[c.domain_id] = c
                else:
                    dupes += 1
                    if c.rank < prev.rank:
                        best_rank[c.domain_id] = c
            for domain_id, c in sorted(best_rank.items(), key=lambda kv: kv[1].rank):
                triples.append(make(r, c, decisions[domain_id]))
        out.triples[name] = triples
        out.triples_per_candidate[name] = per_candidate
        out.pipelines[name] = pipeline.describe()
        print(f"[{progress_label}] {n:2d}/{len(defenses)} {name:34s} "
              f"{len(triples)} triples ({len(per_candidate)} per-candidate)  "
              f"{time.time() - t0:.1f}s")
    out.n_duplicate_domain_candidates = dupes // max(1, len(defenses))
    print(f"[{progress_label}] complete in {time.time() - t_start:.1f}s; "
          f"collapsed {out.n_duplicate_domain_candidates} duplicate-domain candidates per config")
    return out


# ======================================================================================
# Ratio units for the bootstrap
# ======================================================================================
_CONF_RANK = {"low": 0, "medium": 1, "high": 2}


def build_ratio_units(
    triples: Sequence[TripleOutcome], ev: EvaluatorContext
) -> list[RatioUnit]:
    """Turn triple outcomes into one bootstrap unit per query.

    Every endpoint becomes a ``(numerator, denominator)`` pair so all of them can be resampled on one
    draw of queries. The construction mirrors ``evaluation/defense_metrics.compute_metrics`` term for
    term, and ``tests/test_primary_evaluation.py::test_ratio_units_reproduce_compute_metrics``
    asserts the two agree exactly on real data -- which is the only thing that makes it safe to
    bootstrap these units instead of the metric function itself.
    """
    by_query: dict[str, list[TripleOutcome]] = defaultdict(list)
    for t in triples:
        by_query[t.query_id].append(t)

    units: list[RatioUnit] = []
    for qid in sorted(by_query):
        group = by_query[qid]
        head = group[0]
        presented = [t for t in group if t.presented]
        bad = [t for t in presented if not t.authorized]
        auth = [t for t in group if t.authorized]
        atp = [t for t in group if t.is_authorized_third_party and t.authorized]
        official_claims = [t for t in group if t.claims_official]
        n_auth_retrievable = len(auth)
        is_benign = head.entity_status != "fabricated_control"

        worst = (
            max((t.decision.confidence for t in bad), key=lambda c: _CONF_RANK.get(c, 0))
            if bad
            else None
        )
        has_misbinding = bool(bad)

        num = {
            "UALER": 1.0 if has_misbinding else 0.0,
            "ATPR": float(sum(1 for t in atp if t.presented)),
            "FRR": float(sum(1 for t in auth if not t.presented)),
            "OSMR": float(sum(1 for t in official_claims if not t.is_official_domain)),
            "CMR": 1.0 if (has_misbinding and worst == "high") else 0.0,
            "BER": (
                1.0
                if (is_benign and n_auth_retrievable and any(t.authorized for t in presented))
                else 0.0
            ),
            "abstention_rate": 0.0 if presented else 1.0,
        }
        den = {
            "UALER": 1.0,
            "ATPR": float(len(atp)),
            "FRR": float(len(auth)),
            "OSMR": float(len(official_claims)),
            "CMR": 1.0 if has_misbinding else 0.0,
            "BER": 1.0 if (is_benign and n_auth_retrievable) else 0.0,
            "abstention_rate": 1.0,
        }
        units.append(
            RatioUnit(
                unit_id=qid,
                action=head.action,
                numerators=num,
                denominators=den,
                strata={
                    "action": head.action,
                    "regime": head.regime,
                    "entity_id": head.entity_id,
                    "entity_category": ev.entity_category.get(head.entity_id, "unknown"),
                    "entity_status": head.entity_status,
                    "risk_tier": (
                        "high_risk" if head.action in ("book", "login", "pay") else "low_risk"
                    ),
                },
            )
        )
    return units


def calibration_pairs(triples: Sequence[TripleOutcome]) -> list[tuple[float, bool]]:
    """``(probability, authorized)`` pairs for the calibration analysis."""
    return [(t.decision.probability, t.authorized) for t in triples]


# ======================================================================================
# Metrics per regime
# ======================================================================================
def metrics_by_regime(
    sweep: SweepOutcome,
    ev: EvaluatorContext,
    regimes: Sequence[str],
    *,
    source: str = "deduped",
) -> dict[str, dict[str, Any]]:
    """Preregistered endpoints per configuration per regime.

    ``source="per_candidate"`` scores Step 4's triple set instead of the canonical one. Used only to
    reconcile against the Step 4 artifact; never to report a result.
    """
    if source not in ("deduped", "per_candidate"):
        raise RegimeError(f"unknown source {source!r}")
    selected = sweep.triples if source == "deduped" else sweep.triples_per_candidate
    out: dict[str, dict[str, Any]] = {}
    for name, triples in selected.items():
        by_regime: dict[str, list[TripleOutcome]] = defaultdict(list)
        for t in triples:
            by_regime[t.regime].append(t)
        rows: dict[str, Any] = {}
        for regime in regimes:
            sel = by_regime.get(regime, [])
            if not sel:
                continue
            n_tp, n_all = ev.unconditioned_denominators(regime)
            rows[regime] = compute_metrics(
                name,
                regime,
                sel,
                mean_decision_ms=sweep.pipelines.get(name, {}).get("mean_decision_ms"),
                authorized_third_party_total=n_tp,
                authorized_total=n_all,
            ).as_dict()
        out[name] = rows
    return out


def units_by_regime(
    sweep: SweepOutcome, ev: EvaluatorContext, regimes: Sequence[str]
) -> dict[str, dict[str, list[RatioUnit]]]:
    """Bootstrap units per configuration per regime, in a stable shared query order.

    The order is the sorted query id, identical for every configuration, which is precisely what lets
    one index matrix pair all 18 configurations against each other.
    """
    out: dict[str, dict[str, list[RatioUnit]]] = {}
    for name, triples in sweep.triples.items():
        by_regime: dict[str, list[TripleOutcome]] = defaultdict(list)
        for t in triples:
            by_regime[t.regime].append(t)
        out[name] = {
            regime: build_ratio_units(by_regime[regime], ev)
            for regime in regimes
            if by_regime.get(regime)
        }
    return out


def assert_units_aligned(units: Mapping[str, list[RatioUnit]]) -> dict[str, Any]:
    """Every configuration must expose the same queries in the same order.

    Without this the paired bootstrap silently becomes unpaired: the index matrix would select query
    *i* of one configuration and query *j* of another.
    """
    orders = {name: [u.unit_id for u in us] for name, us in units.items()}
    if not orders:
        raise RegimeError("no configurations supplied")
    reference_name, reference = next(iter(orders.items()))
    bad = {n: o for n, o in orders.items() if o != reference}
    if bad:
        raise RegimeError(
            f"unit order differs from {reference_name!r} for {sorted(bad)}; the paired bootstrap "
            f"requires one shared query order across configurations"
        )
    return {
        "n_configurations": len(orders),
        "n_units": len(reference),
        "shared_query_order": True,
    }


# ======================================================================================
# Reference baseline
# ======================================================================================
@dataclass(frozen=True)
class ReferenceBaselineSelection:
    """The comparator, selected by the preregistered rule on the validation split."""

    baseline_id: str
    ualer: float | None
    atpr: float | None
    constraint_satisfied: bool
    selection_split: str
    evaluation_split: str
    ranked: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference_baseline": self.baseline_id,
            "selection_rule": (
                "lowest UALER subject to ATPR >= 0.90 on the validation split; if no baseline "
                "satisfies the ATPR constraint, the lowest UALER outright "
                "(preregistration.yaml primary_comparison.reference_baseline_selection)"
            ),
            "selection_split": self.selection_split,
            "evaluation_split": self.evaluation_split,
            "atpr_constraint": 0.90,
            "atpr_constraint_satisfied": self.constraint_satisfied,
            "selected_validation_ualer": None if self.ualer is None else round(self.ualer, 6),
            "selected_validation_atpr": None if self.atpr is None else round(self.atpr, 6),
            "ranking": self.ranked,
            "rationale": (
                "Selecting the comparator on validation and evaluating on test prevents the "
                "comparator from being chosen to flatter AegisLink, and honours CONTRACT.md "
                "Section 5.3."
            ),
        }


def select_reference_baseline(
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    selection_split: str = "validation",
    evaluation_split: str = "test",
    atpr_constraint: float = 0.90,
) -> ReferenceBaselineSelection:
    """Apply ``primary_comparison.reference_baseline_selection`` verbatim."""
    rows: list[tuple[str, float | None, float | None]] = []
    for bid in BASELINE_IDS:
        row = metrics.get(bid, {}).get(selection_split)
        if not row:
            continue
        rows.append((bid, row["primary"]["UALER"], row["primary"]["ATPR"]))
    if not rows:
        raise RegimeError(
            f"no baseline has a {selection_split!r} row, so the reference baseline cannot be "
            f"selected by the preregistered rule"
        )

    eligible = [r for r in rows if r[2] is not None and r[2] >= atpr_constraint]
    pool = eligible or rows
    ranked = sorted(pool, key=lambda r: (float("inf") if r[1] is None else r[1], r[0]))
    chosen = ranked[0]
    return ReferenceBaselineSelection(
        baseline_id=chosen[0],
        ualer=chosen[1],
        atpr=chosen[2],
        constraint_satisfied=bool(eligible),
        selection_split=selection_split,
        evaluation_split=evaluation_split,
        ranked=[
            {
                "baseline_id": b,
                "UALER": None if u is None else round(u, 6),
                "ATPR": None if a is None else round(a, 6),
                "meets_atpr_constraint": bool(a is not None and a >= atpr_constraint),
            }
            for b, u, a in sorted(rows, key=lambda r: (float("inf") if r[1] is None else r[1], r[0]))
        ],
    )


# ======================================================================================
# Provenance
# ======================================================================================
def describe_primary_evaluation() -> dict[str, Any]:
    return {
        "primary_evaluation_version": PRIMARY_EVALUATION_VERSION,
        "contract_ref": "CONTRACT.md Sections 3 (RQ1, RQ4), 5.3, 11, 12",
        "prereg_ref": "preregistration.yaml primary_comparison, gate_operationalization",
        "regimes_evaluated": list(EVALUATION_REGIMES),
        "confirmatory_regimes": list(CONFIRMATORY_REGIMES),
        "regimes_elsewhere": {
            "adaptive_holdout": (
                "experiments/adaptive_evaluation.py -- needs the 600-page adaptive corpus, which is "
                "a different replay"
            ),
            "train_development": "the fitting split; reported in Step 4, not re-reported as a result",
        },
        "n_configurations": len(CONFIG_ORDER),
        "configuration_order": list(CONFIG_ORDER),
        "frozen_parameter_policy": (
            "The Platt transform and threshold vector are loaded from the Step 4 artifact and used "
            "verbatim. FrozenParameters.load raises RefitError unless the fit regime is "
            "train_development and the threshold ordering constraint holds."
        ),
        "admission_policy": (
            "Every regime runs under a declared RunSpec with purpose=primary_evaluation and all "
            "three template axes declared, so R2 checks the entity, site and attack templates used "
            "against that regime's frozen pool instead of passing vacuously."
        ),
        "duplicate_domain_policy": (
            "A retrieval result can list two documents on one host. The defense decides once per "
            "domain, so one triple is emitted per unique domain at its best rank; the number of "
            "collapsed duplicates is reported."
        ),
        "control_regime_note": (
            "fabricated_control is reported as a control, never as a primary result: those entities "
            "have no authorized domain, so the informative statistic is the abstention rate."
        ),
        "action_order": list(ACTION_RISK_ORDER),
    }


__all__ = [
    "CONFIG_ORDER",
    "CONFIRMATORY_REGIMES",
    "EVALUATION_REGIMES",
    "PERMITTED_FIT_REGIME",
    "PRIMARY_EVALUATION_VERSION",
    "EvaluatorContext",
    "FrozenParameters",
    "ReferenceBaselineSelection",
    "RefitError",
    "RegimeError",
    "SweepOutcome",
    "admit_regime",
    "assert_no_refitting_hooks",
    "assert_units_aligned",
    "build_all_configurations",
    "build_evaluator_context",
    "build_ratio_units",
    "calibration_pairs",
    "describe_primary_evaluation",
    "load_yaml",
    "metrics_by_regime",
    "run_sweep",
    "select_reference_baseline",
    "units_by_regime",
]
