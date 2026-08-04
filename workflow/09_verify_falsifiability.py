"""Step 2.6 -- negative controls: prove the Step 2 checks can actually FAIL.

A validator or invariant check that structurally cannot report a problem passes every artifact
scan while providing no protection whatsoever. Step 1 established this pattern for the
literature checker; this script does the same for the controlled world.

Each control mutates a loaded object IN MEMORY (never on disk) to inject exactly one defect,
then asserts the corresponding check reports it. If any control fails to fire, the check that
should have caught it is broken and this script exits non-zero.

Usage:
    uv run python workflow/09_verify_falsifiability.py
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from parsers.safety_validator import (  # noqa: E402
    check_domain,
    check_phone,
    validate_page_html,
)
from registry.authorization_graph import (  # noqa: E402
    ACTIONS,
    AuthorizationGraph,
    check_graph_invariants,
    load_authorization_graph,
)
from registry.entity_generator import load_entity_world  # noqa: E402
from registry.split_builder import (  # noqa: E402
    build_split_plan,
    check_region_disjointness,
    check_split_invariants,
)
from site_generator.generator import (  # noqa: E402
    LABEL_VOCABULARY_FORBIDDEN_IN_PAGES,
    PAGES_DIR,
    load_attack_design,
    load_site_templates,
    visible_text,
)

RESULTS_DIR = REPO_ROOT / "results"

CONTROLS: list[dict[str, object]] = []


def record(name: str, description: str, fired: bool, evidence: str) -> None:
    CONTROLS.append({
        "control": name, "description": description, "detected": fired, "evidence": evidence,
    })
    status = "DETECTED" if fired else "MISSED  <-- CHECK IS BROKEN"
    print(f"  [{status}] {name}")
    print(f"             {evidence}")


def main() -> int:
    print("=" * 78)
    print("Step 2.6 -- falsifiability controls for the Step 2 checks")
    print("=" * 78)

    world = load_entity_world()
    graph = load_authorization_graph()
    site_templates = load_site_templates()
    design = load_attack_design()

    # ---------------- Safety validator (Section 16) ----------------
    print("[1/4] safety validator must detect injected violations")

    sample_page = next(iter(sorted(PAGES_DIR.rglob("*.html"))))
    clean_html = sample_page.read_text(encoding="utf-8")
    baseline = validate_page_html(clean_html, "baseline")
    record(
        "safety_baseline_is_clean",
        "an unmutated generated page must produce zero violations",
        not baseline,
        f"{len(baseline)} violation(s) on {sample_page.name}",
    )

    mutated = clean_html.replace("<body>", '<body><form><input type="password" name="password">')
    hits = validate_page_html(mutated, "mutated")
    record(
        "safety_detects_credential_form",
        "injecting a password form must trip credential_collection_fields",
        any(v.condition == "credential_collection_fields" for v in hits),
        f"{len(hits)} violation(s); rules={sorted({v.rule_id for v in hits})[:4]}",
    )

    mutated = clean_html.replace("</body>", "<p>Also listed on Booking.com and TripAdvisor.</p></body>")
    hits = validate_page_html(mutated, "mutated")
    record(
        "safety_detects_real_brands",
        "injecting real brand names must trip real_brand_names_in_adversarial_templates",
        any(v.condition == "real_brand_names_in_adversarial_templates" for v in hits),
        f"{len(hits)} violation(s); rules={sorted({v.rule_id for v in hits})[:4]}",
    )

    mutated = clean_html.replace("</body>", '<p><a href="https://pay.stripe.com/x">Pay</a></p></body>')
    hits = validate_page_html(mutated, "mutated")
    record(
        "safety_detects_real_processor_and_public_domain",
        "injecting a real payment endpoint must trip both the processor and public-domain rules",
        ({"real_payment_processors", "public_ip_or_public_domain_deployment_target"}
         <= {v.condition for v in hits}),
        f"conditions={sorted({v.condition for v in hits})}",
    )

    mutated = clean_html.replace("</body>", "<script>fetch('/steal')</script></body>")
    hits = validate_page_html(mutated, "mutated")
    record(
        "safety_detects_active_content",
        "injecting a script element must trip the malware/exfiltration family",
        any(v.condition == "malware_exploit_delivery_or_data_exfiltration_behavior" for v in hits),
        f"rules={sorted({v.rule_id for v in hits})[:4]}",
    )

    record(
        "safety_detects_public_tld_domain",
        "a domain outside the reserved TLDs must be rejected",
        bool(check_domain("harbour-lantern-bistro.co.uk")),
        f"{len(check_domain('harbour-lantern-bistro.co.uk'))} violation(s)",
    )
    record(
        "safety_detects_routable_phone",
        "a real international number must be rejected",
        bool(check_phone("+44-20-7946-0958")),
        f"{len(check_phone('+44-20-7946-0958'))} violation(s)",
    )

    # ---------------- Label-leakage check (Section 8 item 3) ----------------
    print("[2/4] label-leakage check must detect an injected leak")

    leaked = clean_html.replace("</body>", "<p>This domain is authorized for pay.</p></body>")
    fired = any(t in leaked.lower() for t in LABEL_VOCABULARY_FORBIDDEN_IN_PAGES)
    record(
        "leak_detects_label_vocabulary",
        "the word 'authorized' in page bytes must be caught",
        fired,
        f"tokens found={[t for t in LABEL_VOCABULARY_FORBIDDEN_IN_PAGES if t in leaked.lower()]}",
    )

    entity_id = clean_html.split('content="')[1].split('"')[0]
    leaked = clean_html.replace("</body>", f"<p>Reference {entity_id}.</p></body>")
    record(
        "leak_detects_entity_id_in_visible_prose",
        "the machine-readable entity ID must not be allowed into visible prose",
        entity_id in visible_text(leaked),
        f"entity_id={entity_id} now visible; clean page visible={entity_id in visible_text(clean_html)}",
    )

    # ---------------- Graph invariants (Sections 1, 5.2) ----------------
    print("[3/4] graph invariants must detect injected defects")

    record(
        "graph_baseline_is_clean",
        "the unmutated graph must satisfy every invariant",
        check_graph_invariants(graph, world) == [],
        f"{len(check_graph_invariants(graph, world))} violation(s)",
    )

    holed = AuthorizationGraph(
        domains=graph.domains,
        edges=[e for e in graph.edges if not (e.entity_id == "E0001" and e.action == "pay")],
        entity_domain_roles=graph.entity_domain_roles,
        delegations=graph.delegations,
    )
    problems = check_graph_invariants(holed, world)
    record(
        "graph_detects_non_totality",
        "dropping every 'pay' edge for E0001 must trip the totality invariant",
        any("total" in p for p in problems),
        f"{len(problems)} violation(s): {problems[:2]}",
    )

    collapsed = AuthorizationGraph(
        domains=graph.domains,
        edges=[replace(e, authorized=True) for e in graph.edges],
        entity_domain_roles=graph.entity_domain_roles,
        delegations=graph.delegations,
    )
    problems = check_graph_invariants(collapsed, world)
    record(
        "graph_detects_collapsed_action_relativity",
        "making every edge authorized must trip the action- and entity-relativity invariants",
        any("one action and not another" in p for p in problems)
        and any("entity-relativity" in p for p in problems),
        f"{len(problems)} violation(s): {[p[:60] for p in problems[:3]]}",
    )

    bad_evidence = AuthorizationGraph(
        domains=graph.domains,
        edges=[replace(graph.edges[0], evidence_type="vibes")] + list(graph.edges[1:]),
        entity_domain_roles=graph.entity_domain_roles,
        delegations=graph.delegations,
    )
    problems = check_graph_invariants(bad_evidence, world)
    record(
        "graph_detects_unknown_evidence_type",
        "an evidence_type outside the frozen enum must be rejected",
        any("evidence_type" in p for p in problems),
        f"{problems[:1]}",
    )

    duplicated = AuthorizationGraph(
        domains=graph.domains,
        edges=list(graph.edges) + [graph.edges[0]],
        entity_domain_roles=graph.entity_domain_roles,
        delegations=graph.delegations,
    )
    problems = check_graph_invariants(duplicated, world)
    record(
        "graph_detects_duplicate_triple",
        "two labels for one (e,d,a) triple must be rejected",
        any("duplicate" in p for p in problems),
        f"{problems[:1]}",
    )

    # ---------------- Split invariants (Section 5.3) ----------------
    print("[4/4] split invariants must detect injected leakage")

    plan = build_split_plan(world, site_templates, design)
    record(
        "splits_baseline_is_clean",
        "the unmutated plan must satisfy every invariant",
        check_split_invariants(plan, world) == []
        and check_region_disjointness(plan, design) == [],
        f"{len(check_split_invariants(plan, world))} violation(s)",
    )

    leaky = copy.deepcopy(plan)
    leaky.entity.splits["train_development"].append(leaky.entity.splits["test"][0])
    problems = check_split_invariants(leaky, world)
    record(
        "splits_detect_train_test_leak",
        "copying a test entity template into train must be caught",
        any("share" in p for p in problems),
        f"{len(problems)} violation(s): {[p[:70] for p in problems[:2]]}",
    )

    leaky = copy.deepcopy(plan)
    moved = leaky.entity.splits["validation"].pop()
    leaky.entity.splits["train_development"].append(moved)
    problems = check_split_invariants(leaky, world)
    record(
        "splits_detect_broken_ratios",
        "moving one template between splits must break the exact 50/20/30 assertion",
        any("fraction" in p for p in problems),
        f"{len(problems)} violation(s): {[p[:70] for p in problems[:2]]}",
    )

    # SWAP rather than append, so pool sizes and the exact 50/20/30 fractions stay intact. That
    # isolates the category-disjointness check: if only the ratio check fired, this control would
    # pass without ever exercising the property Section 5.3 actually requires.
    leaky = copy.deepcopy(plan)
    held = leaky.entity.holdout.pop()
    swapped_out = leaky.entity.splits["train_development"].pop()
    leaky.entity.holdout.append(swapped_out)
    leaky.entity.splits["train_development"].append(held)
    leaky.entity.core_pool.remove(swapped_out)
    leaky.entity.core_pool.append(held)
    problems = check_split_invariants(leaky, world)
    ratio_problems = [p for p in problems if "fraction" in p]
    record(
        "splits_detect_transfer_holdout_contamination",
        "training on a held-out entity category must be caught by the CATEGORY check, with "
        "pool sizes and ratios left intact so no other check can mask it",
        any("category-disjoint" in p for p in problems) and not ratio_problems,
        f"{len(problems)} violation(s), {len(ratio_problems)} of them ratio-related: "
        f"{[p[:80] for p in problems[:2]]}",
    )

    leaky = copy.deepcopy(plan)
    leaky.attack.core_pool.append(leaky.attack.holdout[0])
    problems = check_region_disjointness(leaky, design)
    record(
        "splits_detect_adaptive_region_contamination",
        "putting a strongest-region attack template into the core must be caught",
        bool(problems),
        f"{len(problems)} violation(s): {problems[:1]}",
    )

    # ---------------- verdict ----------------
    missed = [c for c in CONTROLS if not c["detected"]]
    print("-" * 78)
    print(f"controls run: {len(CONTROLS)}   detected: {len(CONTROLS) - len(missed)}   "
          f"missed: {len(missed)}")
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "benchmark_falsifiability_controls.json").write_text(json.dumps({
        "step": "2.6",
        "purpose": (
            "Negative controls proving the Step 2 safety validator, label-leakage check, graph "
            "invariants and split invariants can report a defect. A check that cannot fail "
            "provides no protection while producing a passing CI signal."
        ),
        "controls_run": len(CONTROLS),
        "controls_detected": len(CONTROLS) - len(missed),
        "controls_missed": len(missed),
        "controls": CONTROLS,
    }, indent=2), encoding="utf-8")
    print("wrote results/benchmark_falsifiability_controls.json")
    if missed:
        for c in missed:
            print(f"  BROKEN CHECK: {c['control']}")
        return 1
    print("OK -- every check is falsifiable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
