"""Step 2.5 -- build template-level splits and write configs/splits.yaml.

CONTRACT.md Section 5.3. Fails closed on any leakage or ratio violation.

Usage:
    uv run python workflow/07_build_splits.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from registry.entity_generator import load_entity_world  # noqa: E402
from registry.split_builder import (  # noqa: E402
    SPLIT_FRACTIONS,
    TRANSFER_HOLDOUT_CATEGORIES,
    build_split_plan,
    check_region_disjointness,
    check_split_invariants,
    write_splits,
)
from site_generator.generator import (  # noqa: E402
    load_attack_design,
    load_site_templates,
)

RESULTS_DIR = REPO_ROOT / "results"


def main() -> int:
    print("=" * 78)
    print("Step 2.5 -- template-level splits (CONTRACT.md Section 5.3)")
    print("=" * 78)

    world = load_entity_world()
    site_templates = load_site_templates()
    design = load_attack_design()
    print(f"[1/3] loaded {len(world.templates)} entity templates, "
          f"{len(site_templates)} site templates, "
          f"{len(design.core_ids)} core + {len(design.adaptive_holdout_ids)} holdout attack "
          f"templates")

    plan = build_split_plan(world, site_templates, design)

    print("[2/3] per-axis split sizes and achieved fractions:")
    for axis in (plan.entity, plan.site, plan.attack):
        achieved = axis.achieved_fractions()
        sizes = {k: len(v) for k, v in axis.splits.items()}
        print(f"       {axis.axis:<16} core={len(axis.core_pool):<4} sizes={sizes}")
        print(f"       {'':<16} achieved={{" + ", ".join(
            f"{k}: {v:.4f}" for k, v in achieved.items()) + "}")
        exact = all(abs(achieved[k] - SPLIT_FRACTIONS[k]) < 1e-12 for k in SPLIT_FRACTIONS)
        print(f"       {'':<16} exactly 50/20/30: {exact}")
        if axis.holdout is not None:
            print(f"       {'':<16} {axis.holdout_name}={len(axis.holdout)}")

    print(f"       transfer holdout categories: {list(TRANSFER_HOLDOUT_CATEGORIES)}")
    print("       regime pool sizes:")
    for regime, pools in plan.regime_pools.items():
        print(f"         {regime:<20} " + " ".join(
            f"{axis.split('_')[0]}={len(ids)}" for axis, ids in pools.items()))

    print("[3/3] checking split invariants (fail-closed) ...")
    problems = check_split_invariants(plan, world) + check_region_disjointness(plan, design)
    if problems:
        for p in problems:
            print(f"  SPLIT VIOLATION: {p}")
        return 1
    print("       all invariants hold (pairwise pool disjointness, exact partition, exact "
          "50/20/30, category-disjoint transfer holdout, region-disjoint adaptive holdout, "
          "every template assigned)")

    digest = write_splits(plan, world, design, site_templates)
    print(f"       wrote configs/splits.yaml")
    print(f"       splits_sha256: {digest}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "benchmark_splits_summary.json").write_text(json.dumps({
        "step": "2.5",
        "axes": {
            axis.axis: {
                "core_pool_size": len(axis.core_pool),
                "split_sizes": {k: len(v) for k, v in axis.splits.items()},
                "achieved_fractions": axis.achieved_fractions(),
                "holdout_name": axis.holdout_name,
                "holdout_size": len(axis.holdout) if axis.holdout else 0,
            }
            for axis in (plan.entity, plan.site, plan.attack)
        },
        "transfer_holdout_categories": list(TRANSFER_HOLDOUT_CATEGORIES),
        "regime_pool_sizes": {
            r: {a: len(ids) for a, ids in pools.items()}
            for r, pools in plan.regime_pools.items()
        },
        "splits_sha256": digest,
        "leakage_violations": 0,
    }, indent=2), encoding="utf-8")
    print(f"       wrote results/benchmark_splits_summary.json")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
