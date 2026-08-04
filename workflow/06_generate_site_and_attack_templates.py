"""Step 2.4a -- generate the site templates and the frozen attack experiment matrix.

CONTRACT.md Section 7 requires a precomputed experiment matrix and forbids post-hoc selection
of successful attacks. This script therefore runs BEFORE any model exists.

Usage:
    uv run python workflow/06_generate_site_and_attack_templates.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from parsers.safety_validator import assert_safe  # noqa: E402
from site_generator.generator import (  # noqa: E402
    ATTACK_FACTORS,
    PAGE_ROLES,
    TEMPLATES_PER_ACTION,
    achieved_core_marginals,
    build_attack_design,
    build_site_templates,
    safety_check_templates,
    write_attack_design,
    write_site_templates,
)

RESULTS_DIR = REPO_ROOT / "results"


def main() -> int:
    print("=" * 78)
    print("Step 2.4a -- site templates and frozen attack experiment matrix")
    print("=" * 78)

    print("[1/4] building site templates ...")
    site_templates = build_site_templates()
    role_counts = Counter(t.page_role for t in site_templates)
    print(f"       site templates    : {len(site_templates)}")
    print(f"       page roles        : {len(PAGE_ROLES)} "
          f"(all Section 7 roles present: {set(role_counts) == set(PAGE_ROLES)})")
    print(f"       per role          : {sorted(set(role_counts.values()))}")

    print("[2/4] enumerating and fractionating the Section 7 attack design ...")
    full_cells = 1
    for levels in ATTACK_FACTORS.values():
        full_cells *= len(levels)
    print(f"       full factorial    : {full_cells} cells "
          f"({' x '.join(str(len(v)) for v in ATTACK_FACTORS.values())})")
    design = build_attack_design()
    print(f"       enumerated        : {len(design.full)}")
    print(f"       core fraction     : {len(design.core_ids)} "
          f"({TEMPLATES_PER_ACTION} per action_claim, action fully crossed)")
    print(f"       adaptive holdout  : {len(design.adaptive_holdout_ids)} "
          f"(pre-declared region {design.balance_report['adaptive_holdout_region']})")
    overlap = set(design.core_ids) & set(design.adaptive_holdout_ids)
    print(f"       core/holdout overlap: {len(overlap)} (must be 0)")
    if overlap:
        return 1

    print("       per-action balance:")
    for action, balance in design.balance_report["per_action"].items():
        print(f"         {action:<8} main effects exact={balance['main_effects_exactly_balanced']} "
              f"max pairwise cell deviation={balance['max_pairwise_cell_deviation']}")

    print("[3/4] achieved core marginals (holdout-induced skew is expected and reported):")
    for factor, stats in achieved_core_marginals(design).items():
        print(f"         {factor:<32} {stats['counts']} "
              f"(uniform={stats['uniform_expectation']:.0f}, "
              f"max dev={stats['max_deviation_from_uniform']:.0f})")

    print("[4/4] safety-validating templates and writing (fail-closed) ...")
    assert_safe(safety_check_templates(site_templates, design))
    site_hash = write_site_templates(site_templates)
    attack_hashes = write_attack_design(design)
    print(f"       wrote site_generator/site_templates.yaml")
    print(f"       wrote attacks/attack_matrix_full.yaml")
    print(f"       wrote attacks/attack_templates.yaml")
    print(f"       site_templates_sha256: {site_hash}")
    for k, v in attack_hashes.items():
        print(f"       {k}: {v}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "benchmark_design_summary.json").write_text(json.dumps({
        "step": "2.4a",
        "site_templates": len(site_templates),
        "site_templates_per_role": dict(sorted(role_counts.items())),
        "attack_full_factorial_cells": len(design.full),
        "attack_core_templates": len(design.core_ids),
        "attack_adaptive_holdout_templates": len(design.adaptive_holdout_ids),
        "attack_factors": {k: [str(x) for x in v] for k, v in ATTACK_FACTORS.items()},
        "balance": design.balance_report,
        "achieved_core_marginals": achieved_core_marginals(design),
        "site_templates_sha256": site_hash,
        **attack_hashes,
    }, indent=2), encoding="utf-8")
    print(f"       wrote results/benchmark_design_summary.json")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
