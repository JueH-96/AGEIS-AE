"""Paired cluster bootstrap, Holm-Bonferroni correction and calibration metrics.

CONTRACT.md Section 11; ``preregistration.yaml`` ``statistical_analysis_plan``.

Every choice below is fixed by the preregistration, not selected after seeing the data:
10,000 resamples, 95% two-sided intervals, bias correction on, Holm within a declared family,
master seed 42, and 10 equal-width ECE bins.

The resampling unit, and why it is the query
--------------------------------------------
``statistical_analysis_plan.confidence_intervals.resampling_unit`` fixes the unit as the paired run
-- "(query, entity, attack condition, retrieval snapshot, model, seed)". In this pilot one query on
the frozen replay *is* one run: the snapshot is frozen, the reader is deterministic, and there is one
seed. So the unit is the query, and resampling draws whole queries with replacement.

That makes this a **cluster** bootstrap, which matters for the link-level endpoints. ATPR's
denominator is authorized third-party links, and several such links can belong to one query. Drawing
links independently would treat them as independent observations when they in fact share a retrieval
snapshot and an entity; drawing whole queries keeps the cluster intact. The plan's
``assumption_checks`` calls for exactly this ("runs within a cell share a retrieval snapshot and page
corpus. Resampling is therefore performed at the run level").

Every endpoint as a ratio of sums
---------------------------------
To resample UALER and ATPR on the same draw they must be expressed in one shape, so each unit
carries a ``(numerator, denominator)`` pair per metric and the metric is
``sum(numerators) / sum(denominators)`` over the units in the resample. This is not a convenience
re-definition: :func:`ratio_estimate` over all units reproduces
:func:`evaluation.defense_metrics.compute_metrics` exactly, and
``tests/test_primary_evaluation.py::test_ratio_units_reproduce_compute_metrics`` asserts it. A unit
with a zero denominator contributes nothing to that metric and does not inflate it -- which is how
"no eligible cases" stays distinguishable from "no failures among eligible cases".

Pairing
-------
One index matrix per regime, generated once and reused by every configuration. Two defenses
compared on resample *b* therefore see the *same* queries, which is what makes the interval on their
difference a paired interval. Generating fresh indices per configuration would silently convert
every paired comparison into an unpaired one and widen the intervals.

Interval flavours reported
--------------------------
``percentile``
    The plain percentile interval.
``bias_corrected``
    Percentile with the median-bias correction ``z0`` only. This is the flavour the preregistration
    names (``bias_correction: true``, and the ``misbinding_reproducible`` gate's "95% bias-corrected
    bootstrap CI"), so it is the one gate decisions read.
``bca``
    ``z0`` plus the jackknife acceleration. Reported alongside because it is the stronger interval
    and a disagreement between the two is worth seeing rather than hiding.

Where a bootstrap distribution is degenerate -- every resample identical, which happens whenever a
defense scores exactly 0.000 on every query -- ``z0`` is undefined. That case is detected and
reported as ``degenerate: true`` with a fallback to the percentile interval, never silently patched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import norm

BOOTSTRAP_VERSION = "1.0"

#: Frozen by ``preregistration.yaml`` ``statistical_analysis_plan``.
MASTER_SEED = 42
N_RESAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95
ALPHA = 0.05
ECE_BINS = 10

#: Endpoints carried as ratio-of-sums units. ``direction`` is the preregistered
#: ``direction_of_benefit`` and is used only for reporting, never to choose a test.
METRIC_DIRECTION: Mapping[str, str] = {
    "UALER": "lower_is_better",
    "ATPR": "higher_is_better",
    "OSMR": "lower_is_better",
    "CMR": "lower_is_better",
    "FRR": "lower_is_better",
    "BER": "higher_is_better",
    "ASR_a": "lower_is_better",
    "abstention_rate": "neutral",
}

RATIO_METRICS: tuple[str, ...] = ("UALER", "ATPR", "OSMR", "CMR", "FRR", "BER", "abstention_rate")


class BootstrapError(RuntimeError):
    """Raised on a misuse that would silently invalidate an interval."""


# ======================================================================================
# Resampling units
# ======================================================================================
@dataclass(frozen=True)
class RatioUnit:
    """One resampling unit (one query) with per-metric numerator and denominator counts.

    ``numerators``/``denominators`` are keyed by endpoint name. A metric absent from both maps is
    simply not measured on this unit.
    """

    unit_id: str
    action: str
    numerators: Mapping[str, float]
    denominators: Mapping[str, float]
    #: Free-form stratifiers used for the per-cell breakdowns Section 11 requires.
    strata: Mapping[str, str] = field(default_factory=dict)

    def num(self, metric: str) -> float:
        return float(self.numerators.get(metric, 0.0))

    def den(self, metric: str) -> float:
        return float(self.denominators.get(metric, 0.0))


def unit_arrays(units: Sequence[RatioUnit], metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Numerator and denominator vectors for ``metric``, aligned to ``units`` order."""
    num = np.fromiter((u.num(metric) for u in units), dtype=np.float64, count=len(units))
    den = np.fromiter((u.den(metric) for u in units), dtype=np.float64, count=len(units))
    return num, den


def ratio_estimate(units: Sequence[RatioUnit], metric: str) -> float | None:
    """``sum(numerators) / sum(denominators)``, or ``None`` on an empty denominator.

    ``None`` rather than 0.0, matching ``evaluation/defense_metrics._rate``: a defense that was
    never given an eligible case has an undefined rate, not a perfect one.
    """
    num, den = unit_arrays(units, metric)
    total = float(den.sum())
    return float(num.sum() / total) if total > 0 else None


# ======================================================================================
# Index matrices
# ======================================================================================
def resample_indices(
    n_units: int, *, n_resamples: int = N_RESAMPLES, seed: int = MASTER_SEED
) -> np.ndarray:
    """``(n_resamples, n_units)`` matrix of unit indices drawn with replacement.

    Deterministic given ``(n_units, n_resamples, seed)``: ``np.random.default_rng`` with an explicit
    integer seed is reproducible across platforms and NumPy versions, which Section 15 requires.
    """
    if n_units <= 0:
        raise BootstrapError("cannot bootstrap zero units")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_units, size=(n_resamples, n_units), dtype=np.int64)


def derive_seed(*parts: Any, master: int = MASTER_SEED) -> int:
    """Deterministic per-stream seed from the master seed and a label.

    Uses BLAKE2b rather than :func:`hash`, whose randomised seed varies per interpreter run and
    would make a "fixed seed" claim false.
    """
    import hashlib

    label = "|".join(str(p) for p in parts)
    h = hashlib.blake2b(f"{master}|{label}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % (2**31 - 1)


def _ratio_over_resamples(
    num: np.ndarray, den: np.ndarray, idx: np.ndarray
) -> np.ndarray:
    """Ratio estimate per resample. ``nan`` where the resampled denominator is empty."""
    num_sums = num[idx].sum(axis=1)
    den_sums = den[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den_sums > 0, num_sums / np.where(den_sums > 0, den_sums, 1.0), np.nan)
    return out


# ======================================================================================
# Intervals
# ======================================================================================
@dataclass(frozen=True)
class Interval:
    """One confidence interval, with enough provenance to audit how it was formed."""

    method: str
    lo: float | None
    hi: float | None
    level: float = CONFIDENCE_LEVEL
    degenerate: bool = False
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        def r(x: float | None) -> float | None:
            return None if x is None or not math.isfinite(x) else round(float(x), 6)

        d = {"method": self.method, "lo": r(self.lo), "hi": r(self.hi), "level": self.level}
        if self.degenerate:
            d["degenerate"] = True
        if self.note:
            d["note"] = self.note
        return d


def percentile_interval(samples: np.ndarray, level: float = CONFIDENCE_LEVEL) -> Interval:
    good = samples[np.isfinite(samples)]
    if good.size == 0:
        return Interval("percentile", None, None, level, True, "no finite resamples")
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(good, [tail, 1.0 - tail])
    return Interval("percentile", float(lo), float(hi), level)


def _z0(samples: np.ndarray, point: float) -> tuple[float | None, str]:
    """Median-bias correction. ``None`` when the bootstrap distribution is degenerate."""
    good = samples[np.isfinite(samples)]
    if good.size == 0:
        return None, "no finite resamples"
    frac = float(np.mean(good < point))
    if frac <= 0.0 or frac >= 1.0:
        # Every resample on one side of the estimate: happens when a defense scores the same value
        # on every query (e.g. exactly 0.000). z0 would be +/-inf, so the correction is refused.
        return None, f"degenerate bootstrap distribution (P(theta* < theta_hat) = {frac})"
    return float(norm.ppf(frac)), ""


def jackknife_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Leave-one-unit-out ratio estimates. ``nan`` where the remaining denominator is empty."""
    num_tot, den_tot = float(num.sum()), float(den.sum())
    den_jack = den_tot - den
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            den_jack > 0, (num_tot - num) / np.where(den_jack > 0, den_jack, 1.0), np.nan
        )


def _acceleration(theta_jack: np.ndarray) -> tuple[float | None, str]:
    """BCa acceleration from leave-one-out replicates of the *same* statistic.

    Takes the replicates rather than ``(num, den)`` on purpose: the statistic being accelerated is
    sometimes a difference or a ratio of two ratios, and feeding such a statistic a single
    numerator/denominator pair would compute the acceleration of a different quantity than the one
    the interval is for.
    """
    good = theta_jack[np.isfinite(theta_jack)]
    if good.size < 3:
        return None, "fewer than 3 finite jackknife replicates"
    d = good.mean() - good
    s2 = float(np.sum(d**2))
    if s2 <= 0.0:
        return None, "zero jackknife variance"
    return float(np.sum(d**3) / (6.0 * s2**1.5)), ""


def _adjusted_interval(
    samples: np.ndarray,
    point: float,
    *,
    method: str,
    z0: float | None,
    accel: float,
    level: float,
    note: str,
) -> Interval:
    if z0 is None:
        fallback = percentile_interval(samples, level)
        return Interval(method, fallback.lo, fallback.hi, level, True,
                        f"{note}; fell back to percentile interval")
    good = samples[np.isfinite(samples)]
    tail = (1.0 - level) / 2.0
    out: list[float] = []
    for q in (tail, 1.0 - tail):
        z = norm.ppf(q)
        denom = 1.0 - accel * (z0 + z)
        if abs(denom) < 1e-12:
            fallback = percentile_interval(samples, level)
            return Interval(method, fallback.lo, fallback.hi, level, True,
                            "acceleration singularity; fell back to percentile interval")
        adj = norm.cdf(z0 + (z0 + z) / denom)
        out.append(float(np.quantile(good, min(max(adj, 0.0), 1.0))))
    return Interval(method, out[0], out[1], level, note=note)


def interval_set(
    samples: np.ndarray,
    point: float | None,
    jackknife: np.ndarray | None = None,
    *,
    level: float = CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """All three preregistered interval flavours for one statistic.

    ``jackknife`` holds leave-one-unit-out replicates *of the statistic the interval is for*, and is
    needed only for the acceleration term behind ``bca``. Omit it and ``bca`` is reported as
    unavailable with the reason recorded, rather than silently falling back to a different interval
    under the ``bca`` label.
    """
    result: dict[str, Any] = {"percentile": percentile_interval(samples, level).as_dict()}
    if point is None or not math.isfinite(point):
        for name in ("bias_corrected", "bca"):
            result[name] = Interval(name, None, None, level, True, "undefined point estimate").as_dict()
        return result

    z0, z_note = _z0(samples, point)
    result["bias_corrected"] = _adjusted_interval(
        samples, point, method="bias_corrected", z0=z0, accel=0.0, level=level, note=z_note
    ).as_dict()
    result["z0"] = None if z0 is None else round(z0, 6)

    if jackknife is None:
        result["bca"] = Interval(
            "bca", None, None, level, True, "no jackknife replicates supplied"
        ).as_dict()
        result["acceleration"] = None
        return result
    accel, a_note = _acceleration(jackknife)
    result["bca"] = _adjusted_interval(
        samples,
        point,
        method="bca",
        z0=z0,
        accel=0.0 if accel is None else accel,
        level=level,
        note="; ".join(x for x in (z_note, a_note) if x),
    ).as_dict()
    result["acceleration"] = None if accel is None else round(accel, 6)
    return result


# ======================================================================================
# Single-configuration estimate
# ======================================================================================
@dataclass
class MetricEstimate:
    """Point estimate plus bootstrap intervals for one (configuration, regime, metric)."""

    config_id: str
    regime: str
    metric: str
    point: float | None
    n_units: int
    n_resamples: int
    denominator_total: float
    numerator_total: float
    intervals: dict[str, Any]
    n_effective_resamples: int
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "regime": self.regime,
            "metric": self.metric,
            "direction_of_benefit": METRIC_DIRECTION.get(self.metric, "unspecified"),
            "point": None if self.point is None else round(self.point, 6),
            "numerator_total": self.numerator_total,
            "denominator_total": self.denominator_total,
            "n_units": self.n_units,
            "n_resamples": self.n_resamples,
            "n_effective_resamples": self.n_effective_resamples,
            "seed": self.seed,
            "ci": self.intervals,
        }


def bootstrap_metric(
    units: Sequence[RatioUnit],
    metric: str,
    *,
    config_id: str,
    regime: str,
    idx: np.ndarray,
    seed: int,
) -> MetricEstimate:
    """Bootstrap one endpoint for one configuration on a *given* index matrix."""
    num, den = unit_arrays(units, metric)
    if idx.shape[1] != num.size:
        raise BootstrapError(
            f"index matrix has {idx.shape[1]} columns but {num.size} units were supplied; "
            f"a mismatched matrix would break the pairing this interval depends on"
        )
    point = ratio_estimate(units, metric)
    samples = _ratio_over_resamples(num, den, idx)
    return MetricEstimate(
        config_id=config_id,
        regime=regime,
        metric=metric,
        point=point,
        n_units=len(units),
        n_resamples=int(idx.shape[0]),
        denominator_total=float(den.sum()),
        numerator_total=float(num.sum()),
        intervals=interval_set(samples, point, jackknife_ratio(num, den)),
        n_effective_resamples=int(np.isfinite(samples).sum()),
        seed=seed,
    )


# ======================================================================================
# Paired comparison
# ======================================================================================
def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for two proportions: ``2*asin(sqrt(p1)) - 2*asin(sqrt(p2))``."""
    c = lambda p: 2.0 * math.asin(math.sqrt(min(max(p, 0.0), 1.0)))  # noqa: E731
    return c(p1) - c(p2)


@dataclass
class PairedComparison:
    """One paired contrast on one endpoint, with intervals and a bootstrap p-value.

    ``p_value`` is the two-sided bootstrap achieved significance level: the bootstrap difference
    distribution is recentred on zero and the p-value is the mass at least as extreme as the observed
    difference. The ``(1 + k) / (1 + B)`` form is used so a p-value is never reported as exactly 0,
    which would overstate what 10,000 resamples can resolve.
    """

    metric: str
    regime: str
    treatment: str
    reference: str
    point_treatment: float | None
    point_reference: float | None
    absolute_difference: float | None
    difference_ci: dict[str, Any]
    relative_reduction: float | None
    relative_reduction_ci: dict[str, Any]
    cohens_h: float | None
    p_value: float | None
    n_units: int
    n_resamples: int
    n_effective_resamples: int
    seed: int
    comparison_id: str = ""
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        def r(x: float | None) -> float | None:
            return None if x is None or not math.isfinite(x) else round(float(x), 6)

        return {
            "comparison_id": self.comparison_id,
            "metric": self.metric,
            "regime": self.regime,
            "treatment": self.treatment,
            "reference": self.reference,
            "direction_of_benefit": METRIC_DIRECTION.get(self.metric, "unspecified"),
            "point_treatment": r(self.point_treatment),
            "point_reference": r(self.point_reference),
            "absolute_difference": r(self.absolute_difference),
            "absolute_difference_definition": "treatment - reference",
            "absolute_difference_ci": self.difference_ci,
            "relative_reduction": r(self.relative_reduction),
            "relative_reduction_definition": "(reference - treatment) / reference",
            "relative_reduction_ci": self.relative_reduction_ci,
            "cohens_h": r(self.cohens_h),
            "p_value": r(self.p_value),
            "p_value_method": "two-sided paired bootstrap ASL, (1+k)/(1+B)",
            "n_units": self.n_units,
            "n_resamples": self.n_resamples,
            "n_effective_resamples": self.n_effective_resamples,
            "seed": self.seed,
            "notes": list(self.notes),
        }


def paired_bootstrap_compare(
    treatment_units: Sequence[RatioUnit],
    reference_units: Sequence[RatioUnit],
    metric: str,
    *,
    treatment: str,
    reference: str,
    regime: str,
    idx: np.ndarray,
    seed: int,
) -> PairedComparison:
    """Paired bootstrap of ``treatment - reference`` on one endpoint.

    Both unit sequences must be in the *same* order, one entry per query, so that column ``j`` of
    the index matrix selects the same query for both configurations. The unit ids are checked, not
    assumed: an unaligned pair would produce a plausible-looking unpaired interval.
    """
    if len(treatment_units) != len(reference_units):
        raise BootstrapError(
            f"{treatment} has {len(treatment_units)} units, {reference} has "
            f"{len(reference_units)}; a paired comparison needs one unit per shared query"
        )
    mismatched = [
        (a.unit_id, b.unit_id)
        for a, b in zip(treatment_units, reference_units)
        if a.unit_id != b.unit_id
    ]
    if mismatched:
        raise BootstrapError(
            f"unit ids are not aligned between {treatment} and {reference} "
            f"({len(mismatched)} positions differ, e.g. {mismatched[:3]}); the pairing would be "
            f"silently broken"
        )

    num_t, den_t = unit_arrays(treatment_units, metric)
    num_r, den_r = unit_arrays(reference_units, metric)
    point_t = ratio_estimate(treatment_units, metric)
    point_r = ratio_estimate(reference_units, metric)

    samples_t = _ratio_over_resamples(num_t, den_t, idx)
    samples_r = _ratio_over_resamples(num_r, den_r, idx)
    diff_samples = samples_t - samples_r

    notes: list[str] = []
    abs_diff = None if (point_t is None or point_r is None) else point_t - point_r
    # Jackknife the difference itself: leave one query out of BOTH arms, since that is the statistic
    # the interval is for.
    jack_t = jackknife_ratio(num_t, den_t)
    jack_r = jackknife_ratio(num_r, den_r)
    diff_ci = interval_set(diff_samples, abs_diff, jack_t - jack_r)

    # -- relative reduction: only defined where the reference is non-zero -------------
    rel_point: float | None = None
    if point_r is not None and point_t is not None:
        if point_r > 0:
            rel_point = (point_r - point_t) / point_r
        else:
            notes.append(
                "relative reduction undefined: the reference endpoint is 0, so there is nothing to "
                "reduce. preregistration.yaml gate_operationalization.aegislink_improvement.guard "
                "forbids performing the division."
            )
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_samples = np.where(
            samples_r > 0, (samples_r - samples_t) / np.where(samples_r > 0, samples_r, 1.0), np.nan
        )
    n_rel = int(np.isfinite(rel_samples).sum())
    if rel_point is not None and n_rel < idx.shape[0]:
        notes.append(
            f"{idx.shape[0] - n_rel} of {idx.shape[0]} resamples had a zero reference endpoint and "
            f"are excluded from the relative-reduction interval"
        )
    if rel_point is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            jack_rel = np.where(
                jack_r > 0, (jack_r - jack_t) / np.where(jack_r > 0, jack_r, 1.0), np.nan
            )
        rel_ci = interval_set(rel_samples, rel_point, jack_rel)
    else:
        rel_ci = {
            "percentile": Interval(
                "percentile", None, None, degenerate=True, note="undefined: reference endpoint is 0"
            ).as_dict()
        }

    # -- two-sided bootstrap ASL -----------------------------------------------------
    p_value: float | None = None
    good = diff_samples[np.isfinite(diff_samples)]
    if abs_diff is not None and good.size:
        centred = good - abs_diff
        k = int(np.sum(np.abs(centred) >= abs(abs_diff) - 1e-15))
        p_value = (1.0 + k) / (1.0 + good.size)

    return PairedComparison(
        metric=metric,
        regime=regime,
        treatment=treatment,
        reference=reference,
        point_treatment=point_t,
        point_reference=point_r,
        absolute_difference=abs_diff,
        difference_ci=diff_ci,
        relative_reduction=rel_point,
        relative_reduction_ci=rel_ci,
        cohens_h=(
            None if (point_t is None or point_r is None) else cohens_h(point_t, point_r)
        ),
        p_value=p_value,
        n_units=len(treatment_units),
        n_resamples=int(idx.shape[0]),
        n_effective_resamples=int(good.size),
        seed=seed,
        comparison_id=f"{metric}:{treatment}_vs_{reference}@{regime}",
        notes=notes,
    )


# ======================================================================================
# Holm-Bonferroni
# ======================================================================================
def holm_bonferroni(
    p_values: Mapping[str, float | None], *, alpha: float = ALPHA, family_id: str = ""
) -> dict[str, Any]:
    """Holm step-down correction within one declared family.

    The adjusted p-value for the ``i``-th smallest raw p (0-indexed) is
    ``max_{j <= i} (m - j) * p_j``, clipped at 1. The running maximum is what enforces
    monotonicity; computing ``(m - i) * p_i`` alone can produce an adjusted sequence that decreases,
    which would let a larger raw p be declared more significant than a smaller one.

    Members whose p-value is ``None`` (an undefined endpoint) are reported as ``testable: false``
    and excluded from ``m``. Counting them would deflate every other member's correction.
    """
    testable = {k: float(v) for k, v in p_values.items() if v is not None and math.isfinite(v)}
    untestable = sorted(set(p_values) - set(testable))
    m = len(testable)
    order = sorted(testable.items(), key=lambda kv: (kv[1], kv[0]))

    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(order):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running

    # Holm rejects the smallest p first and stops at the first non-rejection.
    rejected: dict[str, bool] = {}
    still_rejecting = True
    for name, _ in order:
        if still_rejecting and adjusted[name] <= alpha:
            rejected[name] = True
        else:
            still_rejecting = False
            rejected[name] = False

    return {
        "family_id": family_id,
        "method": "holm",
        "alpha": alpha,
        "n_members": len(p_values),
        "n_testable": m,
        "n_untestable": len(untestable),
        "untestable_members": untestable,
        "untestable_policy": (
            "Members with an undefined endpoint are excluded from the family size m. Counting them "
            "would deflate every other member's correction."
        ),
        "members": {
            name: {
                "p_raw": round(testable[name], 8),
                "p_holm_adjusted": round(adjusted[name], 8),
                "rejected_at_alpha": bool(rejected[name]),
                "rank": i + 1,
                "testable": True,
            }
            for i, (name, _) in enumerate(order)
        }
        | {name: {"p_raw": None, "testable": False} for name in untestable},
        "n_rejected": sum(1 for v in rejected.values() if v),
    }


# ======================================================================================
# Calibration
# ======================================================================================
def calibration_report(
    pairs: Sequence[tuple[float, bool]], *, n_bins: int = ECE_BINS
) -> dict[str, Any]:
    """Brier score, equal-width ECE, maximum calibration error and reliability bins.

    ``preregistration.yaml`` fixes ``ece_bins: 10`` and requires all three metrics plus a
    reliability diagram, so the per-bin table is emitted here rather than recomputed in the plotting
    script from rounded numbers.
    """
    if not pairs:
        return {
            "n": 0,
            "brier": None,
            "ece": None,
            "mce": None,
            "n_bins": n_bins,
            "bins": [],
            "note": "no predictions",
        }
    p = np.asarray([float(a) for a, _ in pairs], dtype=np.float64)
    y = np.asarray([1.0 if b else 0.0 for _, b in pairs], dtype=np.float64)
    brier = float(np.mean((p - y) ** 2))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-closed on the last bin so p == 1.0 lands in bin n-1 rather than out of range.
    which = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    bins: list[dict[str, Any]] = []
    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        mask = which == b
        count = int(mask.sum())
        entry: dict[str, Any] = {
            "bin": b,
            "lo": round(float(edges[b]), 4),
            "hi": round(float(edges[b + 1]), 4),
            "count": count,
        }
        if count:
            conf = float(p[mask].mean())
            acc = float(y[mask].mean())
            gap = abs(conf - acc)
            ece += (count / p.size) * gap
            mce = max(mce, gap)
            entry |= {
                "mean_predicted": round(conf, 6),
                "empirical_rate": round(acc, 6),
                "gap": round(gap, 6),
            }
        else:
            entry |= {"mean_predicted": None, "empirical_rate": None, "gap": None}
        bins.append(entry)

    return {
        "n": int(p.size),
        "brier": round(brier, 8),
        "ece": round(ece, 8),
        "mce": round(mce, 8),
        "n_bins": n_bins,
        "binning": "equal_width",
        "positive_rate": round(float(y.mean()), 6),
        "mean_predicted": round(float(p.mean()), 6),
        "bins": bins,
    }


# ======================================================================================
# Provenance
# ======================================================================================
def describe_bootstrap() -> dict[str, Any]:
    return {
        "bootstrap_version": BOOTSTRAP_VERSION,
        "contract_ref": "CONTRACT.md Section 11",
        "prereg_ref": "preregistration.yaml statistical_analysis_plan",
        "n_resamples": N_RESAMPLES,
        "confidence_level": CONFIDENCE_LEVEL,
        "alpha": ALPHA,
        "master_seed": MASTER_SEED,
        "rng": "numpy.random.default_rng(seed); integer-seeded and platform-stable",
        "resampling_unit": (
            "the query. On the frozen replay one query is one run: the snapshot is frozen, the "
            "reader is deterministic and there is a single seed, so the preregistered unit "
            "'(query, entity, attack condition, retrieval snapshot, model, seed)' collapses to the "
            "query."
        ),
        "cluster_structure": (
            "Whole queries are drawn with replacement, so the several authorized third-party links "
            "belonging to one query stay together. Drawing links independently would treat "
            "observations that share a retrieval snapshot and an entity as independent."
        ),
        "estimator_shape": (
            "Every endpoint is carried as a ratio of sums over units, so UALER (per response) and "
            "ATPR (per link) can be resampled on the same draw. Over all units the ratio reproduces "
            "evaluation.defense_metrics.compute_metrics exactly."
        ),
        "pairing": (
            "One index matrix per regime, reused by every configuration, so two defenses compared "
            "on resample b see the same queries."
        ),
        "intervals": {
            "percentile": "plain percentile interval",
            "bias_corrected": (
                "percentile with median-bias correction z0 only. This is the flavour named by the "
                "preregistration (bias_correction: true) and the one gate decisions read."
            ),
            "bca": "z0 plus jackknife acceleration; reported alongside as the stronger interval",
            "degenerate_policy": (
                "When every resample falls on one side of the point estimate, z0 is undefined and "
                "the interval is reported as degenerate with a percentile fallback, never patched."
            ),
        },
        "p_value": (
            "two-sided paired bootstrap achieved significance level, computed by recentring the "
            "difference distribution on zero; reported as (1+k)/(1+B) so it is never exactly 0"
        ),
        "multiplicity": "Holm step-down within each preregistered family; never across families",
        "effect_sizes": [
            "absolute_risk_difference (treatment - reference)",
            "relative_risk_reduction ((reference - treatment) / reference)",
            "cohens_h (2*asin(sqrt(p1)) - 2*asin(sqrt(p2)))",
        ],
        "calibration": {
            "metrics": ["brier_score", "expected_calibration_error", "maximum_calibration_error"],
            "ece_bins": ECE_BINS,
            "binning": "equal_width",
        },
    }


__all__ = [
    "ALPHA",
    "BOOTSTRAP_VERSION",
    "CONFIDENCE_LEVEL",
    "ECE_BINS",
    "MASTER_SEED",
    "METRIC_DIRECTION",
    "N_RESAMPLES",
    "RATIO_METRICS",
    "BootstrapError",
    "Interval",
    "MetricEstimate",
    "PairedComparison",
    "RatioUnit",
    "bootstrap_metric",
    "calibration_report",
    "cohens_h",
    "derive_seed",
    "describe_bootstrap",
    "holm_bonferroni",
    "interval_set",
    "jackknife_ratio",
    "paired_bootstrap_compare",
    "percentile_interval",
    "ratio_estimate",
    "resample_indices",
    "unit_arrays",
]
