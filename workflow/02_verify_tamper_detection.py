"""Negative control for the preregistration tamper-detection test.

A checksum test that cannot fail provides no protection. This script proves the guard has
teeth by mutating ``preregistration.yaml``, confirming that
``test_preregistration_checksum_matches`` FAILS, then restoring the original bytes and
confirming it passes again.

Two mutations are exercised:

1. A whitespace-only append -- proves even a semantically inert edit is caught.
2. A gate-value change (min_runs_per_cell 20 -> 5) -- proves the scientifically dangerous
   edit, silently loosening a Go/No-Go threshold after the fact, is caught by both the
   checksum test and the contract-comparison test.

The original file is restored in a ``finally`` block so an interrupted run cannot leave the
preregistration modified.

Usage
-----
    uv run python workflow/02_verify_tamper_detection.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PREREG_PATH = REPO_ROOT / "preregistration.yaml"

CHECKSUM_TEST = "tests/test_preregistration.py::test_preregistration_checksum_matches"
GATE_TESTS = (
    "tests/test_preregistration.py::test_gates_match_contract_section_12_verbatim",
    "tests/test_preregistration.py::"
    "test_each_hard_gate_value[misbinding_reproducible-min_runs_per_cell-20]",
)


def run_tests(node_ids: tuple[str, ...] | list[str]) -> bool:
    """Run the given pytest node ids; return True if all passed."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *node_ids, "-q", "--no-header"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    return proc.returncode == 0


def main() -> int:
    print("=" * 78)
    print("Negative control: preregistration tamper detection")
    print("=" * 78)

    original = PREREG_PATH.read_bytes()
    failures: list[str] = []

    try:
        # ---- Control: unmodified file must pass ------------------------------------
        if run_tests([CHECKSUM_TEST]):
            print("  [ok]   baseline           : checksum test PASSES on the frozen file")
        else:
            failures.append("baseline checksum test failed on an unmodified file")
            print("  [FAIL] baseline           : checksum test failed before any mutation")

        # ---- Mutation 1: whitespace-only append -------------------------------------
        PREREG_PATH.write_bytes(original + b"\n# tamper probe\n")
        if not run_tests([CHECKSUM_TEST]):
            print("  [ok]   whitespace tamper  : checksum test FAILS as required")
        else:
            failures.append("checksum test passed despite a modified file (whitespace append)")
            print("  [FAIL] whitespace tamper  : checksum test still passed -- guard is inert")
        PREREG_PATH.write_bytes(original)

        # ---- Mutation 2: silently loosen a Go/No-Go threshold ------------------------
        text = original.decode("utf-8")
        needle = "    min_runs_per_cell: 20"
        if needle not in text:
            failures.append(f"could not locate {needle!r} to mutate")
            print(f"  [FAIL] gate tamper        : anchor {needle!r} not found")
        else:
            PREREG_PATH.write_text(
                text.replace(needle, "    min_runs_per_cell: 5", 1), encoding="utf-8"
            )
            checksum_caught = not run_tests([CHECKSUM_TEST])
            gates_caught = not run_tests(GATE_TESTS)
            if checksum_caught:
                print("  [ok]   gate tamper        : checksum test FAILS as required")
            else:
                failures.append("checksum test passed despite a changed gate value")
                print("  [FAIL] gate tamper        : checksum test still passed")
            if gates_caught:
                print("  [ok]   gate tamper        : contract-comparison tests FAIL as required")
            else:
                failures.append("contract-comparison tests passed despite a changed gate value")
                print("  [FAIL] gate tamper        : contract-comparison tests still passed")
            PREREG_PATH.write_bytes(original)

    finally:
        # Restore unconditionally: an aborted run must never leave the file mutated.
        PREREG_PATH.write_bytes(original)
        restored = PREREG_PATH.read_bytes() == original
        print(f"  [{'ok' if restored else 'FAIL'}]   restore            : "
              f"original bytes restored = {restored}")
        if not restored:
            failures.append("failed to restore the original preregistration bytes")

    # ---- Post-restore control ------------------------------------------------------
    if run_tests([CHECKSUM_TEST, *GATE_TESTS]):
        print("  [ok]   post-restore       : all guards PASS again on the frozen file")
    else:
        failures.append("guards did not pass after restore")
        print("  [FAIL] post-restore       : guards did not recover")

    print("=" * 78)
    if failures:
        print(f"RESULT: TAMPER DETECTION UNVERIFIED -- {len(failures)} problem(s)")
        for f in failures:
            print(f"  - {f}")
        print("=" * 78)
        return 1
    print("RESULT: TAMPER DETECTION VERIFIED")
    print("  The frozen preregistration cannot be edited without failing the test suite.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
