"""Step 2.4b -- render the controlled base page corpus.

Renders every (entity, domain) pair the authorization graph declares, drawing site and attack
templates from the entity's own evaluation regime so no page can straddle a split boundary.
Every page is safety-validated before it reaches disk (CONTRACT.md Section 16), and page
SHA-256 hashes are stored (CONTRACT.md Sections 6, 15).

Usage:
    uv run python workflow/08_render_base_corpus.py
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from parsers.safety_validator import assert_safe  # noqa: E402
from registry.authorization_graph import load_authorization_graph  # noqa: E402
from registry.entity_generator import MACHINE_READABLE_KEY, load_entity_world  # noqa: E402
from registry.split_builder import REGIMES, load_splits  # noqa: E402
from site_generator.generator import (  # noqa: E402
    LABEL_VOCABULARY_FORBIDDEN_IN_PAGES,
    PAGE_ROLES,
    PAGES_DIR,
    load_attack_design,
    load_site_templates,
    render_base_corpus,
    safety_check_pages,
    visible_text,
    write_pages,
)

RESULTS_DIR = REPO_ROOT / "results"


def main() -> int:
    print("=" * 78)
    print("Step 2.4b -- render the controlled base page corpus")
    print("=" * 78)

    world = load_entity_world()
    graph = load_authorization_graph()
    site_templates = load_site_templates()
    design = load_attack_design()
    splits = load_splits()
    print(f"[1/5] loaded {len(world.entities)} entities, {len(graph.domains)} domains, "
          f"{len(graph.edges)} graph edges, {len(site_templates)} site templates")

    regime_of_entity_template = splits["regime_of_entity_template"]
    site_pool = {r: list(splits["regime_pools"][r]["site_template"]) for r in REGIMES}
    attack_pool = {r: list(splits["regime_pools"][r]["attack_template"]) for r in REGIMES}

    print("[2/5] rendering ...")
    pages = render_base_corpus(
        world, graph, site_templates, design,
        regime_of_entity_template, site_pool, attack_pool,
    )
    role_counts = Counter(p.page_role for p in pages)
    snapshot_counts = Counter(p.snapshot for p in pages)
    print(f"       pages rendered      : {len(pages)}")
    print(f"       distinct entities   : {len({p.entity_id for p in pages})}")
    print(f"       per page role       : {dict(sorted(role_counts.items()))}")
    print(f"       per snapshot        : {dict(sorted(snapshot_counts.items()))}")
    missing_roles = set(PAGE_ROLES) - set(role_counts)
    print(f"       Section 7 roles all present: {not missing_roles} "
          f"{sorted(missing_roles) if missing_roles else ''}")
    if missing_roles:
        return 1

    print("[3/5] safety-validating every page (fail-closed) ...")
    assert_safe(safety_check_pages(pages))
    print(f"       0 violations across {len(pages)} pages")

    print("[4/5] checking the machine-readable channel does not leak (Section 8 item 3) ...")
    leaks = 0
    label_tokens = LABEL_VOCABULARY_FORBIDDEN_IN_PAGES
    for page in pages:
        prose = visible_text(page.html)
        if page.entity_id in prose:
            print(f"  LEAK: entity id {page.entity_id} visible in prose of {page.page_id}")
            leaks += 1
        lowered = page.html.lower()
        for token in label_tokens:
            if token in lowered:
                print(f"  LEAK: label token {token!r} present in served bytes of {page.page_id}")
                leaks += 1
                break
    if leaks:
        return 1
    print(f"       entity IDs reach parsers only through <meta name=\"{MACHINE_READABLE_KEY}\">; "
          f"0 prose leaks, 0 label-token leaks")

    print("[5/5] writing pages, manifest and hashes ...")
    if PAGES_DIR.exists():
        shutil.rmtree(PAGES_DIR)  # avoid stale pages from an earlier scale
    digests = write_pages(pages)
    total_bytes = sum(len(p.html.encode("utf-8")) for p in pages)
    print(f"       wrote {len(pages)} pages under data/benchmark/pages/ "
          f"({total_bytes / 1e6:.2f} MB)")
    print(f"       wrote data/benchmark/page_manifest.yaml")
    print(f"       wrote data/benchmark/page_hashes.yaml")
    for k, v in digests.items():
        print(f"       {k}: {v}")

    # Regime consistency: recomputed from the rendered pages rather than trusted from the render.
    straddling = 0
    for page in pages:
        regime = regime_of_entity_template[page.entity_template_id]
        if page.site_template_id not in site_pool[regime]:
            straddling += 1
        elif page.attack_template_id and page.attack_template_id not in attack_pool[regime]:
            straddling += 1
    print(f"       pages straddling a regime boundary: {straddling} (must be 0)")
    if straddling:
        return 1

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "benchmark_corpus_summary.json").write_text(json.dumps({
        "step": "2.4b",
        "pages": len(pages),
        "entities_with_pages": len({p.entity_id for p in pages}),
        "pages_per_role": dict(sorted(role_counts.items())),
        "pages_per_snapshot": dict(sorted(snapshot_counts.items())),
        "corpus_bytes": total_bytes,
        "safety_violations": 0,
        "metadata_leaks": 0,
        "pages_straddling_regime": 0,
        "distinct_site_templates_used": len({p.site_template_id for p in pages}),
        "distinct_attack_templates_used": len(
            {p.attack_template_id for p in pages if p.attack_template_id}),
        **digests,
    }, indent=2), encoding="utf-8")
    print(f"       wrote results/benchmark_corpus_summary.json")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
