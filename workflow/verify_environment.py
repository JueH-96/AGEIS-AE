#!/usr/bin/env python3
"""Pre-flight check for ``make reproduce-experiment``.

Confirms, before any long computation starts, that:

* the interpreter is Python 3.12 or newer,
* every required third-party package imports, and prints the version actually loaded
  (not the version pinned in ``requirements.txt``),
* the governance documents and the frozen benchmark corpus are present,
* ``CONTRACT.md`` is byte-identical to the immutable source under ``user_data/``,
* the SHA-256 digests of ``CONTRACT.md``, ``preregistration.yaml`` and the collision
  rubric match the values frozen into ``configs/frozen_params.json`` before the pilot ran.

Exits non-zero on the first hard failure, so the pipeline fails fast and loudly rather
than producing numbers from a drifted contract.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_PACKAGES = ["numpy", "pandas", "openpyxl", "scipy", "matplotlib", "yaml", "pytest"]

REQUIRED_FILES = [
    "CONTRACT.md",
    "preregistration.yaml",
    "conftest.py",
    "user_data/AegisLink_ARIS_CONTRACT.md",
    "configs/splits.yaml",
    "configs/frozen_params.json",
    "configs/benchmark_freeze.json",
    "evaluation/collision_rubric.yaml",
    "data/benchmark/page_manifest.yaml",
    "data/benchmark/page_hashes.yaml",
    "data/benchmark/retrieval_snapshots.json",
    "data/benchmark/adaptive_retrieval_snapshots.json",
    "data/benchmark/stage_traces.json",
    "data/benchmark/defense_traces.json",
    "data/benchmark/adaptive_page_manifest.yaml",
]

REQUIRED_DIRS = ["data/benchmark/pages", "data/benchmark/adaptive_pages"]

# CONTRACT.md Section 17 mandates these; tests/test_preregistration.py enforces it.
CONTRACT_DIRS = [
    "configs", "registry", "site_generator", "attacks", "web_rag", "aegislink",
    "baselines", "parsers", "evaluation", "statistics", "experiments", "results",
    "figures", "claim_ledger", "manuscript", "docker", "tests",
]

# Keys inside configs/frozen_params.json that carry a digest, mapped to the file they
# should describe. Resolved defensively: the block is searched rather than assumed.
DIGEST_TARGETS = {
    "CONTRACT.md": "CONTRACT.md",
    "preregistration.yaml": "preregistration.yaml",
    "evaluation/collision_rubric.yaml": "evaluation/collision_rubric.yaml",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def walk_for_digests(obj, found: dict[str, str], prefix: str = "") -> None:
    """Collect every ``{... 'sha256': hex ...}`` record keyed by any path-looking sibling."""
    if isinstance(obj, dict):
        sha = obj.get("sha256") or obj.get("sha256_digest")
        if isinstance(sha, str) and len(sha) == 64:
            for k in ("path", "file", "filename", "name", "relative_path"):
                v = obj.get(k)
                if isinstance(v, str):
                    found[v.lstrip("./")] = sha
                    break
            else:
                if prefix:
                    found[prefix.lstrip("./")] = sha
        for k, v in obj.items():
            if isinstance(v, str) and len(v) == 64 and all(
                    c in "0123456789abcdef" for c in v):
                found.setdefault(k.lstrip("./"), v)
            walk_for_digests(v, found, prefix=k)
    elif isinstance(obj, list):
        for item in obj:
            walk_for_digests(item, found, prefix=prefix)


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []

    print("=" * 74)
    print("Environment verification")
    print("=" * 74)
    print(f"[env] repository root : {ROOT}")
    print(f"[env] interpreter     : {sys.executable}")
    print(f"[env] python          : {sys.version.split()[0]}")
    if sys.version_info < (3, 12):
        failures.append(f"Python {sys.version.split()[0]} is older than the required 3.12")

    # ------------------------------------------------------------------- packages
    for name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(name)
        except ImportError as exc:
            failures.append(f"cannot import {name}: {exc}")
            print(f"[pkg] {name:<12} MISSING")
            continue
        ver = getattr(mod, "__version__", "(no __version__)")
        print(f"[pkg] {name:<12} {ver}")

    # ---------------------------------------------------------------------- files
    missing = [f for f in REQUIRED_FILES if not (ROOT / f).is_file()]
    if missing:
        failures.append(f"missing required files: {missing}")
        print(f"[files] MISSING: {missing}")
    else:
        print(f"[files] all {len(REQUIRED_FILES)} required inputs present")

    for d in REQUIRED_DIRS:
        p = ROOT / d
        if not p.is_dir():
            failures.append(f"missing corpus directory: {d}")
            print(f"[corpus] MISSING {d}")
        else:
            # Count recursively: these trees are nested one level deep per site.
            n = sum(1 for q in p.rglob("*") if q.is_file())
            n_top = sum(1 for _ in p.iterdir())
            print(f"[corpus] {d:<34} {n:>5} files in {n_top} subtrees")

    missing_dirs = [d for d in CONTRACT_DIRS if not (ROOT / d).is_dir()]
    if missing_dirs:
        failures.append(f"missing CONTRACT.md Section 17 directories: {missing_dirs}")
        print(f"[structure] MISSING {missing_dirs}")
    else:
        print(f"[structure] all {len(CONTRACT_DIRS)} CONTRACT.md Section 17 "
              f"directories present")

    # ------------------------------------------------- contract byte-identity
    contract = ROOT / "CONTRACT.md"
    source = ROOT / "user_data" / "AegisLink_ARIS_CONTRACT.md"
    if contract.is_file() and source.is_file():
        h_c, h_s = sha256_of(contract), sha256_of(source)
        print(f"[digest] CONTRACT.md            sha256 {h_c}")
        print(f"[digest] user_data source       sha256 {h_s}")
        if h_c != h_s:
            failures.append("CONTRACT.md is NOT byte-identical to "
                            "user_data/AegisLink_ARIS_CONTRACT.md")
            print("[digest] FAIL: contract copy has drifted from its source")
        else:
            print("[digest] PASS: contract is byte-identical to its immutable source")

    prereg = ROOT / "preregistration.yaml"
    if prereg.is_file():
        print(f"[digest] preregistration.yaml   sha256 {sha256_of(prereg)}")

    # ------------------------------------------ digests vs frozen_params.json
    frozen_path = ROOT / "configs" / "frozen_params.json"
    if frozen_path.is_file():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        recorded: dict[str, str] = {}
        walk_for_digests(frozen, recorded)
        checked = 0
        for rel, _ in DIGEST_TARGETS.items():
            p = ROOT / rel
            if not p.is_file():
                continue
            actual = sha256_of(p)
            match = [k for k, v in recorded.items() if v == actual]
            if match:
                checked += 1
                print(f"[frozen] {rel:<34} digest matches the pre-pilot freeze record")
            else:
                same_name = [k for k in recorded if Path(k).name == Path(rel).name]
                if same_name:
                    failures.append(
                        f"{rel} digest {actual[:16]}... does not match the freeze record "
                        f"{recorded[same_name[0]][:16]}...; the file has been modified "
                        f"since the preregistration was frozen")
                    print(f"[frozen] {rel:<34} DIGEST MISMATCH")
                else:
                    warnings.append(f"{rel} carries no digest in configs/frozen_params.json")
                    print(f"[frozen] {rel:<34} no digest recorded (skipped)")
        print(f"[frozen] {checked} of {len(DIGEST_TARGETS)} governance digests verified "
              f"against configs/frozen_params.json")
    else:
        failures.append("configs/frozen_params.json is missing; digests cannot be verified")

    # --------------------------------------------------------------- no LaTeX
    tex = sorted(p.relative_to(ROOT) for p in ROOT.rglob("*")
                 if p.is_file() and p.suffix in {".tex", ".cls", ".bst", ".bib"})
    if tex:
        warnings.append(f"unexpected TeX source files in the bundle: {tex[:5]}")
        print(f"[latex] WARNING: TeX files present: {tex[:5]}")
    else:
        print("[latex] PASS: no .tex/.cls/.bst/.bib anywhere -- no TeX toolchain needed")

    print("-" * 74)
    for w in warnings:
        print(f"WARNING: {w}")
    if failures:
        print(f"VERIFICATION FAILED -- {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("VERIFICATION PASSED -- environment and inputs are ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
