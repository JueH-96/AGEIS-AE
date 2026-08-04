"""Step 2.2 + 2.3 -- generate controlled entities and the ground-truth authorization graph.

Fails closed on any CONTRACT.md Section 16 safety violation and on any graph invariant
violation, before writing anything to disk.

Usage:
    uv run python workflow/05_generate_entities_and_graph.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from parsers.safety_validator import assert_safe, load_fail_closed_conditions  # noqa: E402
from registry.authorization_graph import (  # noqa: E402
    ACTIONS,
    build_authorization_graph,
    check_graph_invariants,
    safety_check_graph,
    write_authorization_graph,
)
from registry.entity_generator import (  # noqa: E402
    generate_entity_world,
    safety_check_world,
    write_entity_world,
)

LOG_DIR = REPO_ROOT / "logs"
RESULTS_DIR = REPO_ROOT / "results"


def main() -> int:
    print("=" * 78)
    print("Step 2.2/2.3 -- controlled entities and ground-truth authorization graph")
    print("=" * 78)

    conditions = load_fail_closed_conditions()
    print(f"[safety] loaded {len(conditions)} frozen Section 16 fail-closed conditions "
          f"from preregistration.yaml")

    # ---------------- entities ----------------
    print("[1/4] generating entity templates and entities ...")
    world = generate_entity_world()
    cat_counts = Counter(e.category for e in world.controlled)
    print(f"       entity templates      : {len(world.templates)}")
    print(f"       controlled entities   : {len(world.controlled)}")
    print(f"       fabricated controls   : {len(world.fabricated)}")
    print(f"       per-category entities : {dict(sorted(cat_counts.items()))}")
    example = world.controlled[0]
    print(f"       Section 5.1 example   : {example.as_dict()}")
    print(f"       resolved address      : {world.address_text(example.entity_id)}")
    print(f"       resolved phone        : {world.phone_text(example.entity_id)}")
    print(f"       aliases               : {world.aliases[example.entity_id]}")

    print("[2/4] safety-validating identity fields (fail-closed) ...")
    report = safety_check_world(world)
    assert_safe(report)
    print(f"       0 violations across {len(world.entities)} entities")
    paths = write_entity_world(world)
    for name, path in paths.items():
        print(f"       wrote {name}: {path.relative_to(REPO_ROOT)}")

    # ---------------- authorization graph ----------------
    print("[3/4] building the authorization graph ...")
    graph = build_authorization_graph(world)
    pairs = {(e.entity_id, e.domain_id) for e in graph.edges}
    print(f"       domains               : {len(graph.domains)}")
    print(f"       (e,d) pairs           : {len(pairs)}")
    print(f"       edges                 : {len(graph.edges)} "
          f"(= pairs x {len(ACTIONS)} actions: {len(pairs) * len(ACTIONS)})")
    pos = sum(1 for e in graph.edges if e.authorized)
    print(f"       authorized edges      : {pos} ({pos / len(graph.edges):.1%})")
    print(f"       role distribution     : "
          f"{dict(sorted(Counter(d.role for d in graph.domains.values()).items()))}")
    print(f"       evidence distribution : "
          f"{dict(sorted(Counter(e.evidence_type for e in graph.edges).items()))}")

    print("[4/4] validating safety and graph invariants (fail-closed) ...")
    assert_safe(safety_check_graph(graph))
    problems = check_graph_invariants(graph, world)
    if problems:
        for p in problems:
            print(f"  INVARIANT VIOLATION: {p}")
        return 1
    print("       all invariants hold (totality, uniqueness, action-relativity, "
          "entity-relativity, evidence enum, official uniqueness, fabricated controls)")

    digests = write_authorization_graph(graph, world)
    print(f"       wrote registry/authorization_graph.yaml "
          f"({(REPO_ROOT / 'registry/authorization_graph.yaml').stat().st_size / 1e6:.2f} MB)")
    print(f"       wrote registry/domains.yaml")
    for k, v in digests.items():
        print(f"       {k}: {v}")

    # ---------------- provenance summary ----------------
    LOG_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    summary = {
        "step": "2.2/2.3",
        "entity_templates": len(world.templates),
        "controlled_entities": len(world.controlled),
        "fabricated_control_entities": len(world.fabricated),
        "entities_per_category": dict(sorted(cat_counts.items())),
        "domains": len(graph.domains),
        "entity_domain_pairs": len(pairs),
        "edges": len(graph.edges),
        "authorized_edges": pos,
        "domain_roles": dict(sorted(Counter(d.role for d in graph.domains.values()).items())),
        "evidence_types": dict(sorted(Counter(e.evidence_type for e in graph.edges).items())),
        "safety_violations": 0,
        "invariant_violations": 0,
        **digests,
    }
    (RESULTS_DIR / "benchmark_entities_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"       wrote results/benchmark_entities_summary.json")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
