"""Step 2.7 -- repository-wide Section 16 safety scan and the Step 2 reproducibility freeze.

Two jobs:

1. Scan every generated artifact against CONTRACT.md Section 16 and fail closed on any hit.
2. Write ``configs/benchmark_freeze.json`` with the SHA-256 of every Step 2 artifact, supplying
   the ``authorization_graph_hash`` and ``page_hashes`` entries CONTRACT.md Section 15 requires.

``configs/frozen_params.json`` is NOT touched: it records the Step 1 preregistration freeze, and
``preregistration.yaml`` must keep its recorded hash. The Step 2 artifacts are generated data, not
preregistered constraints, so they get their own freeze record.

Usage:
    uv run python workflow/10_safety_scan_and_freeze.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from parsers.safety_validator import (  # noqa: E402
    SAFETY_VALIDATOR_VERSION,
    SCAN_LIMITATIONS,
    SafetyReport,
    load_fail_closed_conditions,
    validate_page_html,
    validate_tree,
)
from registry.authorization_graph import BUILDER_VERSION as GRAPH_BUILDER_VERSION  # noqa: E402
from registry.entity_generator import GENERATOR_VERSION as ENTITY_GENERATOR_VERSION  # noqa: E402
from registry.lexicon import LEXICON_VERSION  # noqa: E402
from site_generator.generator import (  # noqa: E402
    GENERATOR_VERSION as SITE_GENERATOR_VERSION,
    PAGES_DIR,
)

CONFIGS_DIR = REPO_ROOT / "configs"
RESULTS_DIR = REPO_ROOT / "results"

# Explicit, not a wildcard: a repo-wide sweep would pull in the validator's own blocklists and
# results/literature_matrix.yaml, which legitimately cites real arXiv URLs.
#
# Scope note. CONTRACT.md Section 16 says the pipeline must fail closed "if any CONFIGURATION
# contains" a forbidden item, so the scan targets generated configuration and data, not
# implementation source. Scanning Python source produced only false positives, and they were
# structural rather than fixable by tuning: site_generator/generator.py documents the very tags
# it forbids ("<form>", "<script>"), and `plan.site` in an attribute access reads as a hostname
# under the real .site TLD. Loosening the detector to accommodate source text would blunt it for
# the data it exists to police.
#
# Source is still covered, and covered more strongly than a substring scan would manage: every
# module's OUTPUT is scanned here (all generated YAML plus all 2631 rendered pages), so a
# forbidden string reaching a page is caught however it was constructed -- including assembled at
# runtime from fragments, which a source-text scan cannot see. registry/lexicon.py is the one
# exception scanned directly, because its entire content is human-readable prose destined for
# page bodies, and it contains no self-referential rule documentation.
SCAN_TARGETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("registry", ("*.yaml", "lexicon.py")),
    ("configs", ("splits.yaml",)),
    ("attacks", ("*.yaml",)),
    ("site_generator", ("*.yaml",)),
    ("data/benchmark", ("*.yaml",)),
)

FREEZE_FILES: tuple[str, ...] = (
    "registry/entities.yaml",
    "registry/entity_templates.yaml",
    "registry/addresses.yaml",
    "registry/phones.yaml",
    "registry/coordinates.yaml",
    "registry/aliases.yaml",
    "registry/authorization_graph.yaml",
    "registry/domains.yaml",
    "site_generator/site_templates.yaml",
    "attacks/attack_matrix_full.yaml",
    "attacks/attack_templates.yaml",
    "configs/splits.yaml",
    "data/benchmark/page_manifest.yaml",
    "data/benchmark/page_hashes.yaml",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=10, check=False)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
        return None


def main() -> int:
    print("=" * 78)
    print("Step 2.7 -- Section 16 artifact scan and Step 2 reproducibility freeze")
    print("=" * 78)

    conditions = load_fail_closed_conditions()
    print(f"[1/3] scanning against {len(conditions)} frozen fail-closed conditions "
          f"(validator v{SAFETY_VALIDATOR_VERSION})")

    total = SafetyReport()
    for subdir, patterns in SCAN_TARGETS:
        root = REPO_ROOT / subdir
        if not root.exists():
            print(f"       MISSING TARGET: {subdir}")
            return 1
        report = validate_tree([root], patterns)
        total.scanned.extend(report.scanned)
        total.exempted.extend(report.exempted)
        total.extend(report.violations)
        print(f"       {subdir:<22} files={len(report.scanned):<4} "
              f"exempt={len(report.exempted):<2} violations={len(report.violations)}")

    pages = sorted(PAGES_DIR.rglob("*.html"))
    print(f"       scanning {len(pages)} rendered pages ...")
    for i, path in enumerate(pages, start=1):
        total.scanned.append(str(path.relative_to(REPO_ROOT)))
        total.extend(validate_page_html(path.read_text(encoding="utf-8"), path.name))
        if i % 500 == 0:
            print(f"         {i}/{len(pages)} pages scanned, "
                  f"{len(total.violations)} violations so far")

    print(f"       total scanned={len(total.scanned)} exempted={len(total.exempted)} "
          f"violations={len(total.violations)}")
    if not total.ok:
        for v in total.violations[:25]:
            print(f"  VIOLATION: {v}")
        print("FAIL-CLOSED: CONTRACT.md Section 16")
        return 1
    print("       0 violations across every generated artifact")

    print("[2/3] computing artifact digests (CONTRACT.md Section 15) ...")
    digests: dict[str, str] = {}
    for rel in FREEZE_FILES:
        path = REPO_ROOT / rel
        if not path.exists():
            print(f"       MISSING ARTIFACT: {rel}")
            return 1
        digests[rel] = sha256_file(path)
        print(f"       {rel:<42} {digests[rel][:16]}...")

    corpus_digest = hashlib.sha256(
        "".join(f"{p.relative_to(PAGES_DIR).as_posix()}:{sha256_file(p)}" for p in pages)
        .encode("utf-8")).hexdigest()
    print(f"       {'data/benchmark/pages/** (corpus)':<42} {corpus_digest[:16]}...")

    print("[3/3] writing configs/benchmark_freeze.json ...")
    summaries = {}
    for name in ("benchmark_entities_summary", "benchmark_design_summary",
                 "benchmark_splits_summary", "benchmark_corpus_summary",
                 "benchmark_falsifiability_controls"):
        path = RESULTS_DIR / f"{name}.json"
        if path.exists():
            summaries[name] = json.loads(path.read_text(encoding="utf-8"))

    freeze = {
        "benchmark_freeze_version": "1.0",
        "step": 2,
        "step_name": "Controlled world, authorization graph, and data splits",
        "contract_ref": "CONTRACT.md Sections 5, 7, 8, 15, 16",
        "purpose": (
            "Reproducibility and tamper-evidence record for the Step 2 generated benchmark. "
            "Supplies the authorization_graph_hash and page_hashes entries required by "
            "CONTRACT.md Section 15. Distinct from configs/frozen_params.json, which records "
            "the Step 1 PREREGISTRATION freeze: these artifacts are generated data that "
            "legitimately changes when the generator version changes, whereas the "
            "preregistration must not change at all after the pilot begins."
        ),
        "hash_algorithm": "sha256",
        "is_frozen_constraint": False,
        "generator_versions": {
            "lexicon": LEXICON_VERSION,
            "entity_generator": ENTITY_GENERATOR_VERSION,
            "authorization_graph_builder": GRAPH_BUILDER_VERSION,
            "site_generator": SITE_GENERATOR_VERSION,
            "safety_validator": SAFETY_VALIDATOR_VERSION,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_commit": git_commit(),
        },
        "master_seed": 42,
        "artifact_sha256": digests,
        "authorization_graph_hash": digests["registry/authorization_graph.yaml"],
        "page_corpus_hash": corpus_digest,
        "page_count": len(pages),
        "safety": {
            "fail_closed_conditions": conditions,
            "files_scanned": len(total.scanned),
            "files_exempted": len(total.exempted),
            "exempted_files": sorted(total.exempted),
            "exemption_rationale": (
                "The validator module and its negative-control tests contain the real brand and "
                "real payment-processor strings that constitute the blocklist, so scanning them "
                "would always fail. The exemption is a short explicit list, never a wildcard."
            ),
            "violations": len(total.violations),
            "known_limitations": list(SCAN_LIMITATIONS),
        },
        "step2_summaries": summaries,
    }
    CONFIGS_DIR.mkdir(exist_ok=True)
    (CONFIGS_DIR / "benchmark_freeze.json").write_text(
        json.dumps(freeze, indent=2), encoding="utf-8")
    print(f"       wrote configs/benchmark_freeze.json")
    print(f"       authorization_graph_hash: {freeze['authorization_graph_hash']}")
    print(f"       page_corpus_hash        : {corpus_digest}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
