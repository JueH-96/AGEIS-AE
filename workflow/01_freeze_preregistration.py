"""Freeze the AegisLink preregistration by recording SHA-256 integrity hashes.

Implements the integrity requirement of ``CONTRACT.md`` Section 12 ("Threshold values may
be changed only before pilot execution and MUST be committed in preregistration.yaml") and
the reproducibility metadata requirement of Section 15.

Writes ``configs/frozen_params.json`` containing:

* the SHA-256 of ``preregistration.yaml`` -- the tamper-evidence anchor;
* the SHA-256 of ``CONTRACT.md`` plus proof it is byte-identical to the user-supplied
  contract in ``user_data/``;
* the SHA-256 of the frozen scoring rubric ``evaluation/collision_rubric.yaml``;
* a flattened copy of every Section 12 hard gate, so a drift between the YAML and the
  recorded gates is detectable without parsing the YAML;
* a clearly-separated checkpoint snapshot of the literature matrix, which is regenerated at
  each Section 13 checkpoint and is therefore provenance rather than a frozen constraint.

Usage
-----
    uv run python workflow/01_freeze_preregistration.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PREREG_PATH = REPO_ROOT / "preregistration.yaml"
CONTRACT_PATH = REPO_ROOT / "CONTRACT.md"
CONTRACT_SOURCE = REPO_ROOT / "user_data" / "AegisLink_ARIS_CONTRACT.md"
RUBRIC_PATH = REPO_ROOT / "evaluation" / "collision_rubric.yaml"
MATRIX_PATH = REPO_ROOT / "results" / "literature_matrix.yaml"
FROZEN_PATH = REPO_ROOT / "configs" / "frozen_params.json"


def sha256_file(path: Path) -> str:
    """Streaming SHA-256 of a file, so the function is size-independent."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def flatten_gates(gates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Flatten the nested Section 12 gate tree to ``group.param`` keys."""
    flat: dict[str, Any] = {}
    for group, params in gates.items():
        for name, value in params.items():
            flat[f"{group}.{name}"] = value
    return flat


def main() -> int:
    print("=" * 78)
    print("AegisLink preregistration freeze -- CONTRACT.md Sections 12, 15")
    print("=" * 78)

    missing = [p for p in (PREREG_PATH, CONTRACT_PATH, CONTRACT_SOURCE, RUBRIC_PATH) if not p.exists()]
    if missing:
        for p in missing:
            print(f"[error] required file missing: {p}", file=sys.stderr)
        return 1

    with PREREG_PATH.open(encoding="utf-8") as fh:
        prereg = yaml.safe_load(fh)

    prereg_hash = sha256_file(PREREG_PATH)
    contract_hash = sha256_file(CONTRACT_PATH)
    source_hash = sha256_file(CONTRACT_SOURCE)
    rubric_hash = sha256_file(RUBRIC_PATH)

    byte_identical = contract_hash == source_hash
    if not byte_identical:
        print("[error] CONTRACT.md is NOT byte-identical to the user-supplied contract.",
              file=sys.stderr)
        print(f"        CONTRACT.md          : {contract_hash}", file=sys.stderr)
        print(f"        user_data source     : {source_hash}", file=sys.stderr)
        print("        The contract is the immutable basis of the project; refusing to freeze.",
              file=sys.stderr)
        return 2

    flat_gates = flatten_gates(prereg["hard_gates"])

    frozen: dict[str, Any] = {
        "frozen_params_version": "1.0",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": (
            "Tamper-evidence and reproducibility record for the AegisLink preregistration. "
            "CONTRACT.md Section 12 permits threshold changes only before pilot execution; "
            "the preregistration SHA-256 below makes any later edit detectable."
        ),
        "hash_algorithm": "sha256",
        "pilot_executed": prereg["pilot_executed"],

        "preregistration": {
            "file": "preregistration.yaml",
            "sha256": prereg_hash,
            "size_bytes": PREREG_PATH.stat().st_size,
            "preregistration_version": prereg["preregistration_version"],
            "amendment_number": prereg["amendment_number"],
            "frozen_date": prereg["frozen_date"],
        },

        "contract": {
            "file": "CONTRACT.md",
            "sha256": contract_hash,
            "size_bytes": CONTRACT_PATH.stat().st_size,
            "source_file": "user_data/AegisLink_ARIS_CONTRACT.md",
            "source_sha256": source_hash,
            "byte_identical_to_source": byte_identical,
            "contract_version": prereg["contract_version"],
            "contract_frozen_date": prereg["contract_frozen_date"],
        },

        "collision_rubric": {
            "file": "evaluation/collision_rubric.yaml",
            "sha256": rubric_hash,
            "note": (
                "The rubric is frozen alongside the preregistration because it fixes how "
                "the Section 12 novelty gate is decided. Changing it after the pilot would "
                "change a gate outcome."
            ),
        },

        "primary_endpoints": [ep["name"] for ep in prereg["primary_endpoints"]],
        "secondary_endpoints": [ep["name"] for ep in prereg["secondary_endpoints"]],

        "hard_gates_flat": flat_gates,
        "hard_gates_nested": prereg["hard_gates"],
        "hard_gates_contract_ref": "CONTRACT.md Section 12",
        "gate_decision_policy": prereg["gate_decision_policy"]["combination"],
        "gate_failure_outcome": prereg["gate_decision_policy"]["failure_outcome"],

        "risk_threshold_ordering": prereg["risk_thresholds"]["ordering_constraint"],
        "bootstrap_resamples": prereg["statistical_analysis_plan"]["confidence_intervals"]["n_resamples"],
        "multiplicity_correction": prereg["statistical_analysis_plan"]["multiplicity_correction"]["method"],
        "random_seed": prereg["statistical_analysis_plan"]["random_seed"],
    }

    # Kept apart from the frozen constraints above: CONTRACT.md Section 13 requires the
    # literature matrix to be regenerated at four checkpoints, so its hash is expected to
    # change over the project's life. Recording it as a frozen constraint would produce a
    # spurious tamper alert at every legitimate re-check.
    if MATRIX_PATH.exists():
        with MATRIX_PATH.open(encoding="utf-8") as fh:
            matrix = yaml.safe_load(fh)
        frozen["literature_matrix_checkpoint_snapshot"] = {
            "file": "results/literature_matrix.yaml",
            "sha256": sha256_file(MATRIX_PATH),
            "checkpoint": matrix.get("checkpoint"),
            "verdict": matrix.get("verdict"),
            "novelty_gate_satisfied": matrix.get("verdict") == "NO_DIRECT_COLLISION",
            "is_frozen_constraint": False,
            "note": (
                "Provenance, not a frozen constraint. Section 13 mandates re-running the "
                "check at project_start, post_pilot, pre_full_experiments and "
                "pre_manuscript_freeze, so this hash legitimately changes at each "
                "checkpoint. The frozen artifact governing the check is the rubric above."
            ),
        }
    else:
        frozen["literature_matrix_checkpoint_snapshot"] = {
            "file": "results/literature_matrix.yaml",
            "status": "absent at freeze time; run evaluation/literature_checker.py",
        }

    FROZEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FROZEN_PATH.open("w", encoding="utf-8") as fh:
        json.dump(frozen, fh, indent=2, sort_keys=False)
        fh.write("\n")

    print(f"  preregistration.yaml sha256 : {prereg_hash}")
    print(f"  CONTRACT.md sha256          : {contract_hash}")
    print(f"  contract byte-identical     : {byte_identical}")
    print(f"  collision_rubric sha256     : {rubric_hash}")
    print(f"  primary endpoints           : {frozen['primary_endpoints']}")
    print(f"  hard gate parameters frozen : {len(flat_gates)}")
    for k, v in flat_gates.items():
        print(f"      {k:<58} = {v}")
    snap = frozen["literature_matrix_checkpoint_snapshot"]
    print(f"  literature verdict          : {snap.get('verdict', snap.get('status'))}")
    print(f"  written                     : {FROZEN_PATH.relative_to(REPO_ROOT)}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
