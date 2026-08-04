"""Template-level dataset splitter (CONTRACT.md Section 5.3).

CONTRACT.md Section 5.3: "Split by entity template, site template, and attack template, not by
individual rendered pages." Required splits are train/development 50%, validation 20%, test
30%, an adaptive holdout of disjoint attacker templates, and a transfer holdout of disjoint
entity categories, with "No test template may be used during threshold selection."

The design decision that matters most here is how three *independently* split axes combine.
Splitting each axis 50/20/30 and then stamping one label on a rendered page is incoherent: a
page built from a train entity template, a test site template and a validation attack template
belongs to no split, and quietly assigning it one leaks the holdout. So splits are declared as
**evaluation regimes over template triples**:

    train_development : entity, site and attack templates all from the train pool
    validation        : all three from the validation pool
    test              : all three from the test pool
    transfer_holdout  : entity template from held-out CATEGORIES, site/attack from test pools
    adaptive_holdout  : entity/site from test pools, attack template from the held-out region
    fabricated_control: fabricated-entity templates, site/attack from test pools

Two consequences are made explicit rather than left implicit:

* The 50/20/30 fractions apply to each axis's **core pool** -- the templates left after the
  mandated holdouts are removed. Section 5.3 lists the holdouts alongside the three fractions
  and characterises them by disjointness rather than by a percentage, so folding them into the
  denominator would make the stated fractions unachievable.

* Benign pages have no attack template. A page is admitted to regime R iff its entity template
  is in R's entity pool, its site template is in R's site pool, and its attack template is
  either absent or in R's attack pool.

The anti-leak invariant the tests assert is per-axis pool disjointness: the train pool of an
axis shares no template with that axis's validation, test, or holdout pools.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

from registry.entity_generator import (
    EntityWorld,
    MASTER_SEED,
    derive_int,
)
from registry.lexicon import CATEGORIES
from site_generator.generator import (
    ATTACK_FACTORS,
    PAGE_ROLES,
    ADAPTIVE_HOLDOUT_REGION,
    AttackDesign,
    SiteTemplate,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_PATH = REPO_ROOT / "configs" / "splits.yaml"

BUILDER_VERSION = "1.0"

# preregistration.yaml -> splits.fractions (frozen in Step 1).
SPLIT_NAMES: tuple[str, ...] = ("train_development", "validation", "test")
SPLIT_FRACTIONS: dict[str, float] = {
    "train_development": 0.50,
    "validation": 0.20,
    "test": 0.30,
}

# Transfer holdout: whole entity categories, declared before any result existed. The last two
# entries of registry.lexicon.CATEGORIES. Chosen because they are the two categories whose
# service vocabulary overlaps least with the other six, which makes the unseen-entity-template
# gate a genuine transfer test rather than a relabelling.
TRANSFER_HOLDOUT_CATEGORIES: tuple[str, ...] = ("fitness_studio", "bookshop")

REGIMES: tuple[str, ...] = (
    "train_development", "validation", "test",
    "transfer_holdout", "adaptive_holdout", "fabricated_control",
)


@dataclass
class AxisSplit:
    """Per-axis pools."""

    axis: str
    core_pool: list[str]
    splits: dict[str, list[str]]
    holdout_name: str | None = None
    holdout: list[str] | None = None

    def all_templates(self) -> list[str]:
        out = list(self.core_pool)
        if self.holdout:
            out += self.holdout
        return sorted(set(out))

    def achieved_fractions(self) -> dict[str, float]:
        n = len(self.core_pool)
        return {k: (len(v) / n if n else 0.0) for k, v in self.splits.items()}


@dataclass
class SplitPlan:
    """The full split plan across all three axes plus the derived regimes."""

    entity: AxisSplit
    site: AxisSplit
    attack: AxisSplit
    fabricated_entity_templates: list[str]
    regime_pools: dict[str, dict[str, list[str]]]
    regime_of_entity_template: dict[str, str]


# ======================================================================================
# Exact stratified allocation
# ======================================================================================
def _largest_remainder(n: int, fractions: dict[str, float]) -> dict[str, int]:
    """Allocate ``n`` items across ``fractions`` summing exactly to ``n``.

    Exact when ``n`` is a multiple of 10; otherwise the largest-remainder rule keeps the total
    exact and the deviation below one item, and the achieved fractions are written out so the
    difference is visible rather than assumed away.
    """
    raw = {k: n * f for k, f in fractions.items()}
    base = {k: int(v) for k, v in raw.items()}
    shortfall = n - sum(base.values())
    order = sorted(raw, key=lambda k: (-(raw[k] - base[k]), k))
    for k in order[:shortfall]:
        base[k] += 1
    return base


def _stratified_split(
    strata: dict[str, list[str]], fractions: dict[str, float], namespace: str
) -> dict[str, list[str]]:
    """Split each stratum independently so every split keeps the stratum composition.

    Assignment inside a stratum is by a deterministic per-template sort key derived from
    SHA-256, so the split is reproducible and does not depend on input ordering.
    """
    out: dict[str, list[str]] = {k: [] for k in fractions}
    for stratum, members in sorted(strata.items()):
        ordered = sorted(members, key=lambda t: (derive_int(namespace, f"{stratum}|{t}", 10**9), t))
        counts = _largest_remainder(len(ordered), fractions)
        cursor = 0
        for split in SPLIT_NAMES:
            take = counts[split]
            out[split].extend(ordered[cursor:cursor + take])
            cursor += take
    return {k: sorted(v) for k, v in out.items()}


# ======================================================================================
# Axis builders
# ======================================================================================
def build_entity_axis(world: EntityWorld) -> tuple[AxisSplit, list[str]]:
    """Split entity templates, holding out whole categories for the transfer holdout."""
    controlled = [t for t in world.templates if t.morphology != "fabricated_control"]
    fabricated = sorted(t.entity_template_id for t in world.templates
                        if t.morphology == "fabricated_control")

    holdout = sorted(t.entity_template_id for t in controlled
                     if t.category in TRANSFER_HOLDOUT_CATEGORIES)
    core = [t for t in controlled if t.category not in TRANSFER_HOLDOUT_CATEGORIES]

    strata: dict[str, list[str]] = {}
    for t in core:
        strata.setdefault(t.category, []).append(t.entity_template_id)

    return AxisSplit(
        axis="entity_template",
        core_pool=sorted(t.entity_template_id for t in core),
        splits=_stratified_split(strata, SPLIT_FRACTIONS, "entity_split"),
        holdout_name="transfer_holdout",
        holdout=holdout,
    ), fabricated


def build_site_axis(site_templates: Sequence[SiteTemplate]) -> AxisSplit:
    """Split site templates, stratified by page role.

    Stratification is required, not cosmetic: an unstratified split could leave a Section 7
    page role absent from a split, and that regime could then never render that role.
    """
    strata: dict[str, list[str]] = {}
    for t in site_templates:
        strata.setdefault(t.page_role, []).append(t.site_template_id)
    return AxisSplit(
        axis="site_template",
        core_pool=sorted(t.site_template_id for t in site_templates),
        splits=_stratified_split(strata, SPLIT_FRACTIONS, "site_split"),
    )


def build_attack_axis(design: AttackDesign) -> AxisSplit:
    """Split the core attack fraction, stratified by action_claim; hold out the declared region."""
    by_id = design.by_id
    strata: dict[str, list[str]] = {}
    for tid in design.core_ids:
        strata.setdefault(by_id[tid].action_claim, []).append(tid)
    return AxisSplit(
        axis="attack_template",
        core_pool=sorted(design.core_ids),
        splits=_stratified_split(strata, SPLIT_FRACTIONS, "attack_split"),
        holdout_name="adaptive_holdout",
        holdout=sorted(design.adaptive_holdout_ids),
    )


# ======================================================================================
# Regimes
# ======================================================================================
def build_split_plan(
    world: EntityWorld, site_templates: Sequence[SiteTemplate], design: AttackDesign
) -> SplitPlan:
    """Assemble the three axis splits and derive the six evaluation regimes."""
    entity_axis, fabricated = build_entity_axis(world)
    site_axis = build_site_axis(site_templates)
    attack_axis = build_attack_axis(design)

    regime_pools: dict[str, dict[str, list[str]]] = {
        "train_development": {
            "entity_template": entity_axis.splits["train_development"],
            "site_template": site_axis.splits["train_development"],
            "attack_template": attack_axis.splits["train_development"],
        },
        "validation": {
            "entity_template": entity_axis.splits["validation"],
            "site_template": site_axis.splits["validation"],
            "attack_template": attack_axis.splits["validation"],
        },
        "test": {
            "entity_template": entity_axis.splits["test"],
            "site_template": site_axis.splits["test"],
            "attack_template": attack_axis.splits["test"],
        },
        # Unseen entity templates: whole categories never seen in train or validation.
        "transfer_holdout": {
            "entity_template": list(entity_axis.holdout or []),
            "site_template": site_axis.splits["test"],
            "attack_template": attack_axis.splits["test"],
        },
        # Unseen attack templates: the pre-declared strongest-attack region.
        "adaptive_holdout": {
            "entity_template": entity_axis.splits["test"],
            "site_template": site_axis.splits["test"],
            "attack_template": list(attack_axis.holdout or []),
        },
        # RQ1 condition 7 control: fabricated entity + fabricated domain.
        "fabricated_control": {
            "entity_template": fabricated,
            "site_template": site_axis.splits["test"],
            "attack_template": attack_axis.splits["test"],
        },
    }

    # One regime per entity template, used by the renderer so no page can straddle a boundary.
    regime_of_entity_template: dict[str, str] = {}
    for regime in ("train_development", "validation", "test", "transfer_holdout",
                   "fabricated_control"):
        for tid in regime_pools[regime]["entity_template"]:
            regime_of_entity_template[tid] = regime

    return SplitPlan(
        entity=entity_axis,
        site=site_axis,
        attack=attack_axis,
        fabricated_entity_templates=fabricated,
        regime_pools=regime_pools,
        regime_of_entity_template=regime_of_entity_template,
    )


def check_split_invariants(plan: SplitPlan, world: EntityWorld) -> list[str]:
    """Return leakage / ratio violations. Empty list means the plan is usable."""
    problems: list[str] = []

    for axis in (plan.entity, plan.site, plan.attack):
        pools = {name: set(ids) for name, ids in axis.splits.items()}
        if axis.holdout is not None:
            pools[axis.holdout_name or "holdout"] = set(axis.holdout)

        # Pairwise disjointness: this is the anti-leak property.
        names = sorted(pools)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                overlap = pools[a] & pools[b]
                if overlap:
                    problems.append(
                        f"{axis.axis}: pools {a} and {b} share {len(overlap)} template(s), "
                        f"e.g. {sorted(overlap)[:3]}")

        # Core pool is exactly partitioned by the three splits.
        union = set().union(*(set(v) for v in axis.splits.values()))
        if union != set(axis.core_pool):
            problems.append(
                f"{axis.axis}: the three splits do not partition the core pool "
                f"({len(union)} vs {len(axis.core_pool)})")

        # Exact fractions.
        for name, target in SPLIT_FRACTIONS.items():
            achieved = len(axis.splits[name]) / len(axis.core_pool)
            if abs(achieved - target) > 1e-12:
                problems.append(
                    f"{axis.axis}: {name} fraction {achieved:.6f} != required {target}")

    # Transfer holdout must be disjoint in CATEGORY, not merely in template.
    template_category = {t.entity_template_id: t.category for t in world.templates}
    seen_categories = {
        template_category[t]
        for name in SPLIT_NAMES for t in plan.entity.splits[name]
    }
    holdout_categories = {template_category[t] for t in (plan.entity.holdout or [])}
    if seen_categories & holdout_categories:
        problems.append(
            f"transfer holdout categories {sorted(holdout_categories & seen_categories)} also "
            "appear in train/validation/test; the holdout is not category-disjoint")

    # Every controlled entity template is assigned to exactly one regime.
    controlled = [t.entity_template_id for t in world.templates]
    unassigned = sorted(set(controlled) - set(plan.regime_of_entity_template))
    if unassigned:
        problems.append(f"{len(unassigned)} entity template(s) belong to no regime, "
                        f"e.g. {unassigned[:3]}")
    return problems


def check_region_disjointness(plan: SplitPlan, design: AttackDesign) -> list[str]:
    """Verify no core attack template falls inside the pre-declared adaptive-holdout region."""
    by_id = design.by_id
    problems: list[str] = []
    for tid in plan.attack.core_pool:
        t = by_id[tid]
        if all(getattr(t, f) in levels for f, levels in ADAPTIVE_HOLDOUT_REGION.items()):
            problems.append(f"core attack template {tid} lies inside the adaptive-holdout region")
    return problems


# ======================================================================================
# Persistence
# ======================================================================================
def _axis_doc(axis: AxisSplit, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "axis": axis.axis,
        "total_templates": len(axis.all_templates()),
        "core_pool_size": len(axis.core_pool),
        "required_fractions": dict(SPLIT_FRACTIONS),
        "achieved_fractions": {k: round(v, 10) for k, v in axis.achieved_fractions().items()},
        "split_sizes": {k: len(v) for k, v in axis.splits.items()},
        "splits": {k: list(v) for k, v in axis.splits.items()},
    }
    if axis.holdout is not None:
        doc["holdout_name"] = axis.holdout_name
        doc["holdout_size"] = len(axis.holdout)
        doc["holdout"] = list(axis.holdout)
    if extra:
        doc.update(extra)
    return doc


def write_splits(
    plan: SplitPlan, world: EntityWorld, design: AttackDesign,
    site_templates: Sequence[SiteTemplate], path: Path = SPLITS_PATH,
) -> str:
    """Write configs/splits.yaml and return its SHA-256."""
    template_category = {t.entity_template_id: t.category for t in world.templates}
    per_split_categories = {
        name: sorted({template_category[t] for t in ids})
        for name, ids in plan.entity.splits.items()
    }
    role_of_site = {t.site_template_id: t.page_role for t in site_templates}
    per_split_roles = {
        name: sorted({role_of_site[t] for t in ids}) for name, ids in plan.site.splits.items()
    }
    by_id = design.by_id
    per_split_actions = {
        name: sorted({by_id[t].action_claim for t in ids})
        for name, ids in plan.attack.splits.items()
    }

    doc = {
        "metadata": {
            "schema_version": "1.0",
            "builder": "registry/split_builder.py",
            "builder_version": BUILDER_VERSION,
            "master_seed": MASTER_SEED,
            "contract_ref": "CONTRACT.md Section 5.3",
            "preregistration_ref": "preregistration.yaml -> splits",
            "split_unit": ["entity_template", "site_template", "attack_template"],
            "split_unit_note": (
                "Splitting is by template, never by rendered page. Two entities instantiated "
                "from one entity template therefore always land in the same split, which is "
                "what stops a page-level split from leaking a memorised template."
            ),
            "fraction_denominator": (
                "The 50/20/30 fractions apply to each axis's CORE pool, i.e. the templates "
                "remaining after the mandated holdouts are removed. CONTRACT.md Section 5.3 "
                "lists the adaptive and transfer holdouts alongside the three fractions and "
                "characterises them by disjointness rather than by a share, so including them "
                "in the denominator would make the stated fractions unachievable."
            ),
            "threshold_selection_constraint": (
                "No test template may be used during threshold selection (CONTRACT.md Section "
                "5.3). Gate decisions and threshold fitting use train_development and "
                "validation only."
            ),
        },
        "policy": {
            "fractions": dict(SPLIT_FRACTIONS),
            "regimes": list(REGIMES),
            "regime_admission_rule": (
                "A rendered page belongs to regime R iff its entity template is in R's entity "
                "pool AND its site template is in R's site pool AND its attack template is "
                "either absent (benign page) or in R's attack pool. Independent per-axis "
                "splitting means a triple mixing pools from different regimes belongs to no "
                "regime and must never be rendered."
            ),
            "transfer_holdout_categories": list(TRANSFER_HOLDOUT_CATEGORIES),
            "transfer_holdout_rationale": (
                "Whole entity categories are withheld, so the unseen-entity-template gate tests "
                "transfer to a category the defense never saw rather than to a sibling template "
                "of one it did."
            ),
            "adaptive_holdout_region": {k: [str(x) for x in v]
                                        for k, v in ADAPTIVE_HOLDOUT_REGION.items()},
            "adaptive_holdout_rationale": (
                "A pre-declared factor-level region (the strongest attacks), fixed before any "
                "result existed. Randomly held-out cells of one grid would each have a trained "
                "near neighbour, making the unseen-attack-template gate close to vacuous."
            ),
            "adaptive_holdout_cost": (
                "Excising a factor-level region skews the ELIGIBLE pool for "
                "identity_consistency, official_backlink, corroborating_sources and "
                "lexical_diversity. Whether that skew propagates into the SELECTED core "
                "fraction was checked rather than assumed: the per-factor quotas remained "
                "satisfiable after the exclusion, so the selected core fraction is exactly "
                "uniform on every main effect. The achieved marginals for both the core pool "
                "and the holdout are recorded under "
                "axes.attack_template.achieved_factor_marginals so this can be verified rather "
                "than taken on trust. The holdout pool itself is deliberately non-uniform -- "
                "that is what makes it the strongest-attack region."
            ),
        },
        "axes": {
            "entity_template": _axis_doc(plan.entity, {
                "stratified_by": "category",
                "categories_per_split": per_split_categories,
                "fabricated_control_templates": plan.fabricated_entity_templates,
                "all_categories": list(CATEGORIES),
            }),
            "site_template": _axis_doc(plan.site, {
                "stratified_by": "page_role",
                "page_roles_per_split": per_split_roles,
                "all_page_roles": list(PAGE_ROLES),
                "stratification_rationale": (
                    "Every split must contain every Section 7 page role, or that regime could "
                    "not render the role at all."
                ),
            }),
            "attack_template": _axis_doc(plan.attack, {
                "stratified_by": "action_claim",
                "action_claims_per_split": per_split_actions,
                "all_action_claims": [str(a) for a in ATTACK_FACTORS["action_claim"]],
                "achieved_factor_marginals": _attack_marginals(plan, design),
                "balance": design.balance_report,
            }),
        },
        "regime_pools": plan.regime_pools,
        "regime_pool_sizes": {
            r: {axis: len(ids) for axis, ids in pools.items()}
            for r, pools in plan.regime_pools.items()
        },
        "regime_of_entity_template": plan.regime_of_entity_template,
        "entity_to_regime": {
            eid: plan.regime_of_entity_template[tid]
            for eid, tid in sorted(world.entity_to_template.items())
        },
    }
    text = yaml.dump(doc, sort_keys=False, allow_unicode=True, width=100,
                     Dumper=getattr(yaml, "CSafeDumper", yaml.SafeDumper))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _attack_marginals(plan: SplitPlan, design: AttackDesign) -> dict[str, Any]:
    by_id = design.by_id
    out: dict[str, Any] = {}
    for scope, ids in (("core_pool", plan.attack.core_pool),
                       ("adaptive_holdout", plan.attack.holdout or [])):
        scope_out: dict[str, Any] = {}
        for factor, levels in ATTACK_FACTORS.items():
            counts = {str(level): 0 for level in levels}
            for tid in ids:
                counts[str(getattr(by_id[tid], factor))] += 1
            uniform = len(ids) / len(levels) if ids else 0.0
            scope_out[factor] = {
                "counts": counts,
                "uniform_expectation": uniform,
                "max_deviation_from_uniform": (
                    max(abs(v - uniform) for v in counts.values()) if ids else 0.0),
            }
        out[scope] = scope_out
    return out


def load_splits(path: Path = SPLITS_PATH) -> dict[str, Any]:
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    return yaml.load(path.read_text(encoding="utf-8"), Loader=loader)
