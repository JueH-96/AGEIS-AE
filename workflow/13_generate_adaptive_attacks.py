"""Step 3 driver: generate the adaptive attack family in the pre-declared holdout region.

CONTRACT.md Section 3 (RQ4), Section 7, Section 16.

Outputs
-------
``attacks/adaptive_templates.yaml``
    The frozen adaptive design: ``AD-0001``+ templates, their factor levels, the chosen prose
    realisation, and the full optimisation trace per template.
``data/benchmark/adaptive_pages/<entity_id>/AP*.html``
    Rendered adaptive pages, two-phase where ``content_change_after_indexing`` is true.
``data/benchmark/adaptive_page_manifest.yaml``
    Side manifest (page kind, template, snapshot) held OUTSIDE the page bytes.
``results/adaptive_attacker_report.json``
    Invariant checks, safety scan, optimisation statistics, admission-controller wiring.

Run::

    uv run python workflow/13_generate_adaptive_attacks.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from attacks.adaptive_attacker import (  # noqa: E402
    ADAPTIVE_ID_PREFIX,
    AdaptiveAttacker,
    AttackerWorld,
    HeuristicDefensePanel,
    assert_disjoint_from_core,
    assert_ground_truth_untouched,
    assert_no_label_vocabulary,
    build_adaptive_design,
    build_adaptive_manifest,
    build_adaptive_template_doc,
    safety_check_pages,
    verify_two_phase,
    write_adaptive_pages,
)
from web_rag.experiment_matrix import (  # noqa: E402
    ExperimentMatrix,
    RunPurpose,
    RunSpec,
    TemplateTriple,
)

BENCH = ROOT / "data" / "benchmark"
RESULTS = ROOT / "results"
TEMPLATES_PATH = ROOT / "attacks" / "adaptive_templates.yaml"
MANIFEST_PATH = BENCH / "adaptive_page_manifest.yaml"


def load_yaml(path: Path) -> Any:
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    return yaml.load(path.read_text(encoding="utf-8"), Loader=loader)


def main() -> int:
    t0 = time.time()
    print("=" * 78)
    print("Step 3 -- Adaptive attacker: defense-optimised templates (AD-0001+)")
    print("=" * 78)

    # ---------------------------------------------------------------- 1. world + design
    print("\n[1/6] Loading the attacker's world view")
    world = AttackerWorld.load()
    targets = world.adaptive_target_entities()
    print(f"[1/6] {len(targets)} eligible target entities in the adaptive holdout pool "
          f"(each has an official domain and >=5 corroborating domains)")

    print("\n[2/6] Enumerating the adaptive design")
    templates = build_adaptive_design(world)
    strata = Counter(t.stratum for t in templates)
    print(f"[2/6] {len(templates)} templates: "
          f"Stratum A (in declared region) = {strata['A']}, "
          f"Stratum B (partial identity, secondary only) = {strata['B']}")
    print(f"[2/6] action_claim balance: "
          f"{json.dumps(dict(sorted(Counter(t.action_claim for t in templates).items())))}")
    print(f"[2/6] corroborating_sources: "
          f"{json.dumps({str(k): v for k, v in sorted(Counter(t.corroborating_sources for t in templates).items())})}")
    print(f"[2/6] content_change_after_indexing: "
          f"{json.dumps({str(k): v for k, v in sorted(Counter(t.content_change_after_indexing for t in templates).items())})}")

    # ---------------------------------------------------------------- 3. optimise + render
    print("\n[3/6] Optimising prose against the heuristic defense panel (a PROXY)")
    panel = HeuristicDefensePanel()
    attacker = AdaptiveAttacker(world=world, panel=panel)
    pages, traces, realisations = attacker.generate(templates, progress_every=10)
    improvements = [tr.improvement for tr in traces]
    print(f"[3/6] {len(pages)} pages rendered, {sum(tr.evaluations for tr in traces)} panel "
          f"evaluations")
    print(f"[3/6] suspicion: mean {sum(tr.initial_suspicion for tr in traces) / len(traces):.4f} "
          f"-> {sum(tr.final_suspicion for tr in traces) / len(traces):.4f} "
          f"(mean improvement {sum(improvements) / len(improvements):+.4f}, "
          f"{sum(1 for x in improvements if x > 0)}/{len(improvements)} improved)")

    # ---------------------------------------------------------------- 4. invariants
    print("\n[4/6] Invariant checks")
    static_design = load_yaml(ROOT / "attacks" / "attack_templates.yaml")
    core_ids = set(static_design["core_template_ids"]) | set(
        static_design["adaptive_holdout_template_ids"]
    )
    assert_disjoint_from_core(templates, core_ids)
    print(f"   [OK] adaptive ids disjoint from all {len(core_ids)} static matrix ids")

    gt = assert_ground_truth_untouched(world, templates)
    print(f"   [OK] ground truth untouched: {gt['edges_checked']} edges checked, "
          f"0 new domains, 0 new edges, every target already unauthorized for its claimed action")

    assert_no_label_vocabulary(pages)
    print(f"   [OK] no ground-truth label vocabulary in any of {len(pages)} page(s)")

    two_phase = verify_two_phase(pages)
    print(f"   [OK] two-phase fidelity: {two_phase['n_two_phase_slots']} indexed/live slot(s), "
          f"0 with identical bytes")

    # Region membership must be verifiable from factor levels alone.
    matrix = ExperimentMatrix.from_splits()
    matrix.register_adaptive_templates([
        {**t.factor_levels(), "attack_template_id": t.attack_template_id,
         "in_declared_region": t.in_declared_region}
        for t in templates
    ])
    print(f"   [OK] region membership re-derived from factor levels for all "
          f"{len(templates)} template(s); {len(matrix.adaptive_stratum_a)} in Stratum A")

    # ---------------------------------------------------------------- 5. safety
    print("\n[5/6] Section 16 fail-closed safety scan")
    report = safety_check_pages(pages)
    if report.violations:
        print(f"   [FAIL] {len(report.violations)} safety violation(s):")
        for v in report.violations[:10]:
            print(f"      {v.condition} / {v.rule_id}: {v.detail} @ {v.location}")
        return 1
    print(f"   [OK] {len(report.scanned)} page(s) scanned, 0 violations")

    # ---------------------------------------------------------------- 5b. admission wiring
    print("\n[5/6] Admission-controller wiring checks")
    ad_ids = tuple(t.attack_template_id for t in templates)
    stratum_a = tuple(t.attack_template_id for t in templates if t.stratum == "A")
    stratum_b = tuple(t.attack_template_id for t in templates if t.stratum == "B")
    ad_pool = matrix.pool("adaptive_holdout", "entity_template")
    site_pool = sorted(matrix.pool("adaptive_holdout", "site_template"))
    probe_triples = (
        TemplateTriple(sorted(ad_pool)[0], site_pool[0], None),
    )
    checks: list[dict[str, Any]] = []

    def check(name: str, run: RunSpec, expect_admitted: bool) -> None:
        rep = matrix.admit(run)
        ok = rep.admitted == expect_admitted
        checks.append(
            {
                "check": name,
                "expected_admitted": expect_admitted,
                "actual_admitted": rep.admitted,
                "pass": ok,
                "violations": [v.as_dict() for v in rep.violations],
            }
        )
        print(f"   [{'OK  ' if ok else 'FAIL'}] {name}: admitted={rep.admitted} "
              f"(expected {expect_admitted})")

    check(
        "Stratum A in adaptive_holdout primary run is admitted",
        RunSpec("AD-PRIMARY", "adaptive_holdout", RunPurpose.PRIMARY_EVALUATION,
                triples=probe_triples, adaptive_template_ids=stratum_a),
        True,
    )
    check(
        "Stratum B in a primary run is REJECTED (outside the declared region)",
        RunSpec("AD-B-PRIMARY", "adaptive_holdout", RunPurpose.PRIMARY_EVALUATION,
                triples=probe_triples, adaptive_template_ids=stratum_b),
        False,
    )
    check(
        "Stratum B in a secondary run is admitted",
        RunSpec("AD-B-SECONDARY", "adaptive_holdout", RunPurpose.SECONDARY_EVALUATION,
                triples=probe_triples, adaptive_template_ids=stratum_b),
        True,
    )
    check(
        "adaptive templates in the test regime are REJECTED",
        RunSpec("AD-IN-TEST", "test", RunPurpose.PRIMARY_EVALUATION,
                adaptive_template_ids=ad_ids[:3]),
        False,
    )
    check(
        "adaptive templates during threshold selection are REJECTED",
        RunSpec("AD-IN-FIT", "train_development", RunPurpose.THRESHOLD_SELECTION,
                adaptive_template_ids=ad_ids[:3]),
        False,
    )
    if any(not c["pass"] for c in checks):
        print("   [FAIL] admission-controller wiring did not behave as specified")
        return 1

    # ---------------------------------------------------------------- 6. persist
    print("\n[6/6] Writing artifacts")
    write_adaptive_pages(pages)
    doc = build_adaptive_template_doc(templates, traces, realisations, panel)
    TEMPLATES_PATH.write_text(
        yaml.safe_dump(doc, sort_keys=False, width=120, default_flow_style=False),
        encoding="utf-8",
    )
    manifest = build_adaptive_manifest(pages, templates)
    MANIFEST_PATH.write_text(
        yaml.safe_dump(manifest, sort_keys=False, width=120, default_flow_style=False),
        encoding="utf-8",
    )

    corpus_digest = hashlib.sha256()
    for p in sorted(pages, key=lambda x: x.page_id):
        corpus_digest.update(p.page_id.encode())
        corpus_digest.update(p.sha256.encode())

    report_doc = {
        "metadata": {
            "step": "Step 3 of 8 -- adaptive attacker",
            "contract_ref": "CONTRACT.md Section 3 (RQ4), Section 7, Section 16",
            "builder": "workflow/13_generate_adaptive_attacks.py",
            "id_prefix": ADAPTIVE_ID_PREFIX,
            "adaptive_corpus_digest": corpus_digest.hexdigest(),
            "adaptive_templates_sha256": hashlib.sha256(
                TEMPLATES_PATH.read_bytes()
            ).hexdigest(),
            "adaptive_manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
            "elapsed_seconds": round(time.time() - t0, 2),
        },
        "design": {
            "n_templates": len(templates),
            "stratum_a": strata["A"],
            "stratum_b": strata["B"],
            "n_pages": len(pages),
            "n_target_entities": len({t.target_entity_id for t in templates}),
            "n_target_domains": len({t.target_domain_id for t in templates}),
            "action_claim_balance": dict(sorted(Counter(t.action_claim for t in templates).items())),
            "endpoint_replacement_actions": dict(
                sorted(Counter(str(t.endpoint_replacement_action) for t in templates).items())
            ),
            "corroboration_cluster_sizes": {
                str(k): v for k, v in sorted(Counter(t.corroborating_sources for t in templates).items())
            },
            "identity_consistency_levels": dict(
                sorted(Counter(t.identity_consistency for t in templates).items())
            ),
            "page_kinds": dict(sorted(Counter(p.page_kind for p in pages).items())),
            "snapshots": dict(sorted(Counter(p.snapshot for p in pages).items())),
        },
        "optimisation": {
            "objective": "minimise HeuristicDefensePanel score",
            "panel": panel.describe(),
            "mean_initial_suspicion": round(
                sum(tr.initial_suspicion for tr in traces) / len(traces), 6
            ),
            "mean_final_suspicion": round(
                sum(tr.final_suspicion for tr in traces) / len(traces), 6
            ),
            "mean_improvement": round(sum(improvements) / len(improvements), 6),
            "templates_improved": sum(1 for x in improvements if x > 0),
            "templates_converged": sum(1 for tr in traces if tr.converged),
            "total_panel_evaluations": sum(tr.evaluations for tr in traces),
            "mean_detector_before": {
                k: round(sum(tr.initial_detector_scores[k] for tr in traces) / len(traces), 6)
                for k in traces[0].initial_detector_scores
            },
            "mean_detector_after": {
                k: round(sum(tr.final_detector_scores[k] for tr in traces) / len(traces), 6)
                for k in traces[0].final_detector_scores
            },
        },
        "invariants": {
            "disjoint_from_static_matrix": True,
            "n_static_ids_checked": len(core_ids),
            "ground_truth_untouched": gt,
            "no_label_vocabulary_in_pages": True,
            "two_phase": two_phase,
            "region_membership_rederived_from_factor_levels": True,
            "n_stratum_a_registered": len(matrix.adaptive_stratum_a),
        },
        "safety": {
            "contract_ref": "CONTRACT.md Section 16 (fail-closed)",
            "pages_scanned": len(report.scanned),
            "violations": len(report.violations),
        },
        "admission_controller_wiring": checks,
    }
    (RESULTS / "adaptive_attacker_report.json").write_text(
        json.dumps(report_doc, indent=2) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print(f"Adaptive attacker complete in {time.time() - t0:.1f}s")
    print(f"  templates           : {len(templates)} "
          f"(A={strata['A']} primary-eligible, B={strata['B']} secondary-only)")
    print(f"  pages rendered      : {len(pages)}")
    print(f"  two-phase slots     : {two_phase['n_two_phase_slots']}")
    print(f"  safety violations   : 0")
    print(f"  mean suspicion drop : {sum(improvements) / len(improvements):+.4f}")
    print(f"  corpus digest       : {corpus_digest.hexdigest()[:32]}...")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
