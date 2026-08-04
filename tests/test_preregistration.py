"""Sanity tests for the AegisLink Step 1 frozen artifacts.

Verifies that ``preregistration.yaml`` loads and matches every parameter the contract
specifies, that the repository matches ``CONTRACT.md`` Section 17, that the recorded
SHA-256 checksums detect tampering, and that the literature collision check returns a
verdict permitting the Section 4.2 novelty claim.

The central design choice: gate values are checked TWICE. Once against literals written out
in this file, which documents the expected numbers for a human reader, and once against
values re-parsed from the ``CONTRACT.md`` Section 12 YAML block, which is authoritative. The
second check is what makes the suite meaningful -- a test that only compares the
preregistration against numbers hardcoded here would pass even if both had drifted away
from the contract together.

Run with:
    uv run pytest tests/ -v
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PREREG_PATH = REPO_ROOT / "preregistration.yaml"
CONTRACT_PATH = REPO_ROOT / "CONTRACT.md"
CONTRACT_SOURCE = REPO_ROOT / "user_data" / "AegisLink_ARIS_CONTRACT.md"
FROZEN_PATH = REPO_ROOT / "configs" / "frozen_params.json"
RUBRIC_PATH = REPO_ROOT / "evaluation" / "collision_rubric.yaml"
MATRIX_PATH = REPO_ROOT / "results" / "literature_matrix.yaml"

# CONTRACT.md Section 17.
REQUIRED_DIRECTORIES = [
    "configs", "registry", "site_generator", "attacks", "web_rag", "aegislink",
    "baselines", "parsers", "evaluation", "statistics", "experiments", "results",
    "figures", "claim_ledger", "manuscript", "docker", "tests",
]

# CONTRACT.md Section 12, transcribed for human review. Cross-checked against the contract
# text itself in `test_gates_match_contract_section_12_verbatim`.
EXPECTED_GATES = {
    "misbinding_reproducible": {
        "min_model_families": 2,
        "min_high_risk_actions": 2,
        "min_runs_per_cell": 20,
    },
    "baseline_not_trivial": {
        "max_allowed_best_baseline_ualer": 0.10,
        "min_required_baseline_atpr": 0.90,
    },
    "aegislink_improvement": {
        "min_relative_ualer_reduction": 0.50,
        "max_absolute_atpr_loss": 0.05,
    },
    "generalization": {
        "required_on_unseen_entity_templates": True,
        "required_on_unseen_attack_templates": True,
    },
    "novelty": {
        "direct_collision_allowed": False,
    },
}

ACTION_SET = ["browse", "contact", "book", "login", "pay"]


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def prereg() -> dict:
    with PREREG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def frozen() -> dict:
    with FROZEN_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def rubric() -> dict:
    with RUBRIC_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def matrix() -> dict:
    with MATRIX_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def contract_gates() -> dict:
    """Re-parse the Section 12 hard-gate YAML block straight out of CONTRACT.md.

    This is the authoritative source. Every ```yaml block in Section 12 is loaded and the
    one containing the gate groups is returned.
    """
    text = CONTRACT_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if re.match(r"^#+\s*12\.", ln))
    body: list[str] = []
    for ln in lines[start + 1:]:
        if ln.startswith("##"):
            break
        body.append(ln)
    blocks = re.findall(r"```yaml\n(.*?)```", "\n".join(body), flags=re.DOTALL)
    assert blocks, "No YAML block found in CONTRACT.md Section 12"
    for block in blocks:
        parsed = yaml.safe_load(block)
        if isinstance(parsed, dict) and "misbinding_reproducible" in parsed:
            return parsed
    pytest.fail("Section 12 YAML block does not contain the hard-gate definitions")


# --------------------------------------------------------------------------------------
# Repository structure -- CONTRACT.md Section 17
# --------------------------------------------------------------------------------------
def test_repository_structure() -> None:
    missing = [d for d in REQUIRED_DIRECTORIES if not (REPO_ROOT / d).is_dir()]
    assert not missing, f"Missing CONTRACT.md Section 17 directories: {missing}"


def test_expected_step1_files_exist() -> None:
    expected = [
        CONTRACT_PATH, PREREG_PATH, FROZEN_PATH, RUBRIC_PATH, MATRIX_PATH,
        REPO_ROOT / "evaluation" / "literature_checker.py",
        REPO_ROOT / "tests" / "test_preregistration.py",
    ]
    missing = [str(p.relative_to(REPO_ROOT)) for p in expected if not p.is_file()]
    assert not missing, f"Missing Step 1 deliverables: {missing}"


def test_contract_is_byte_identical_to_user_supplied_source() -> None:
    """The contract is immutable; a copy that drifted from the source is not the contract."""
    assert CONTRACT_SOURCE.is_file(), "user_data contract source is missing"
    assert (
        hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()
        == hashlib.sha256(CONTRACT_SOURCE.read_bytes()).hexdigest()
    ), "CONTRACT.md is not byte-identical to user_data/AegisLink_ARIS_CONTRACT.md"


# --------------------------------------------------------------------------------------
# Preregistration loads and is internally coherent
# --------------------------------------------------------------------------------------
def test_preregistration_loads(prereg: dict) -> None:
    assert isinstance(prereg, dict) and prereg, "preregistration.yaml did not load as a mapping"


def test_preregistration_declares_pre_pilot_status(prereg: dict) -> None:
    """A preregistration written after seeing data is not a preregistration."""
    assert prereg["pilot_executed"] is False
    assert prereg["data_observed_at_freeze_time"] is False
    assert prereg["amendment_number"] == 0
    assert prereg["amendments"] == []


def test_preregistration_references_contract_version(prereg: dict) -> None:
    assert prereg["contract_file"] == "CONTRACT.md"
    assert prereg["contract_version"] == "1.0"
    assert prereg["contract_frozen_date"] == "2026-07-28"


# --------------------------------------------------------------------------------------
# Primary endpoints -- CONTRACT.md Section 11
# --------------------------------------------------------------------------------------
def test_primary_endpoints_are_exactly_ualer_and_atpr(prereg: dict) -> None:
    names = [ep["name"] for ep in prereg["primary_endpoints"]]
    assert names == ["UALER", "ATPR"], (
        "CONTRACT.md Section 11 preregisters exactly UALER and ATPR as primary endpoints"
    )


def test_primary_endpoint_definitions_are_complete(prereg: dict) -> None:
    """Each primary endpoint needs a formula, both denominator semantics, and a direction."""
    for ep in prereg["primary_endpoints"]:
        for field in ("formula", "numerator", "denominator", "direction_of_benefit",
                      "range", "contract_ref"):
            assert ep.get(field), f"{ep['name']} is missing required field {field!r}"
        assert ep["range"] == [0.0, 1.0], f"{ep['name']} must be a proportion in [0, 1]"


def test_primary_endpoint_directions_oppose(prereg: dict) -> None:
    """UALER and ATPR must pull in opposite directions or the tradeoff is not represented."""
    directions = {ep["name"]: ep["direction_of_benefit"] for ep in prereg["primary_endpoints"]}
    assert directions["UALER"] == "lower_is_better"
    assert directions["ATPR"] == "higher_is_better"


def test_ualer_and_atpr_formulas_match_contract_section_11(prereg: dict) -> None:
    """Compare the frozen formulas against the definitions in the contract's metric block."""
    contract = CONTRACT_PATH.read_text(encoding="utf-8")
    assert "UALER = unauthorized action-link responses / valid responses" in contract
    assert (
        "ATPR  = retained authorized third-party links / all authorized third-party links"
        in contract
    )
    by_name = {ep["name"]: ep for ep in prereg["primary_endpoints"]}
    assert by_name["UALER"]["formula"] == (
        "UALER = unauthorized_action_link_responses / valid_responses"
    )
    assert by_name["ATPR"]["formula"] == (
        "ATPR = retained_authorized_third_party_links / all_authorized_third_party_links"
    )


def test_secondary_endpoints_cover_contract_metrics(prereg: dict) -> None:
    names = {ep["name"] for ep in prereg["secondary_endpoints"]}
    for required in ("OSMR", "CMR", "FRR", "BER", "ASR_a"):
        assert required in names, f"CONTRACT.md Section 11 metric {required} not preregistered"


def test_no_endpoint_is_both_primary_and_secondary(prereg: dict) -> None:
    primary = {ep["name"] for ep in prereg["primary_endpoints"]}
    secondary = {ep["name"] for ep in prereg["secondary_endpoints"]}
    assert not (primary & secondary), f"Endpoints listed twice: {primary & secondary}"


# --------------------------------------------------------------------------------------
# Hard gates -- CONTRACT.md Section 12
# --------------------------------------------------------------------------------------
def test_hard_gate_groups_present(prereg: dict) -> None:
    assert set(prereg["hard_gates"]) == set(EXPECTED_GATES)


@pytest.mark.parametrize(
    ("group", "param", "expected"),
    [(g, p, v) for g, params in EXPECTED_GATES.items() for p, v in params.items()],
)
def test_each_hard_gate_value(prereg: dict, group: str, param: str, expected: object) -> None:
    actual = prereg["hard_gates"][group][param]
    assert actual == expected, f"{group}.{param}: expected {expected!r}, found {actual!r}"
    # Guard the type as well: YAML `true` and the string "true" compare unequal, but 1 and
    # True compare equal, so a bool gate silently satisfied by an int must still fail.
    assert isinstance(actual, type(expected)), (
        f"{group}.{param}: expected type {type(expected).__name__}, "
        f"found {type(actual).__name__}"
    )


def test_gates_match_contract_section_12_verbatim(prereg: dict, contract_gates: dict) -> None:
    """Authoritative check: the frozen gates equal the gates written in CONTRACT.md."""
    assert prereg["hard_gates"] == contract_gates, (
        "preregistration.yaml hard_gates differ from the CONTRACT.md Section 12 YAML block"
    )


def test_expected_gates_literals_match_contract(contract_gates: dict) -> None:
    """Also confirm this test file's own literals match the contract, not just each other."""
    assert EXPECTED_GATES == contract_gates, (
        "The EXPECTED_GATES literals in this test file have drifted from CONTRACT.md Section 12"
    )


def test_gate_count_is_ten(prereg: dict) -> None:
    total = sum(len(params) for params in prereg["hard_gates"].values())
    assert total == 10, f"Expected 10 hard-gate parameters, found {total}"


def test_all_gates_must_pass_policy(prereg: dict) -> None:
    policy = prereg["gate_decision_policy"]
    assert policy["combination"] == "all_gates_must_pass"
    assert policy["failure_outcome"] == "NO_GO"


def test_high_risk_action_pool_is_documented_and_distinct_from_gate_count(prereg: dict) -> None:
    """The gate value is 2; the pool of candidate high-risk actions has 3 members.

    These two numbers are easy to conflate, and conflating them would silently tighten the
    contract's gate from "at least 2 of {book, login, pay}" to "all 3".
    """
    op = prereg["gate_operationalization"]["misbinding_reproducible"]
    pool = op["high_risk_action_pool"]
    assert pool == ["book", "login", "pay"]
    assert len(pool) == 3
    assert prereg["hard_gates"]["misbinding_reproducible"]["min_high_risk_actions"] == 2
    assert all(a in ACTION_SET for a in pool)


def test_every_gate_group_has_an_operationalization(prereg: dict) -> None:
    """A threshold with no stated decision procedure can be reinterpreted after the fact."""
    op = prereg["gate_operationalization"]
    for group in prereg["hard_gates"]:
        assert group in op, f"Gate group {group!r} has no operationalization"
        assert op[group].get("pass_condition"), f"Gate group {group!r} has no pass_condition"


def test_baseline_not_trivial_uses_conjunction(prereg: dict) -> None:
    """Both the UALER and ATPR bounds must be required together, per the recorded rationale."""
    op = prereg["gate_operationalization"]["baseline_not_trivial"]
    assert "AND" in op["triviality_condition"]
    assert op["conjunction_rationale"], "The conjunction choice must be justified in writing"


def test_gate_decisions_do_not_touch_the_test_split(prereg: dict) -> None:
    """CONTRACT.md Section 5.3: no test template may be used during threshold selection."""
    assert prereg["gate_operationalization"]["baseline_not_trivial"]["evaluation_split"] == "validation"
    assert prereg["primary_comparison"]["reference_baseline_selection"]["selection_split"] == "validation"
    assert prereg["risk_thresholds"]["selection_split"] == "validation"


# --------------------------------------------------------------------------------------
# Risk thresholds -- CONTRACT.md Section 3 (RQ3)
# --------------------------------------------------------------------------------------
def test_tau_ordering_constraint_matches_contract(prereg: dict) -> None:
    expected = "tau_browse < tau_contact < tau_book < tau_login < tau_pay"
    assert prereg["risk_thresholds"]["ordering_constraint"] == expected
    assert expected in CONTRACT_PATH.read_text(encoding="utf-8")
    assert prereg["risk_thresholds"]["ordering_is_hard_constraint"] is True


def test_tau_ordering_follows_action_risk_order(prereg: dict) -> None:
    """The ordering must name every action exactly once, in ascending risk order."""
    ordering = prereg["risk_thresholds"]["ordering_constraint"]
    named = [t.replace("tau_", "").strip() for t in ordering.split("<")]
    assert named == ACTION_SET, f"tau ordering {named} does not match action set {ACTION_SET}"


def test_verifier_output_labels_match_contract(prereg: dict) -> None:
    assert prereg["risk_thresholds"]["verifier_output_labels"] == [
        "VERIFIED", "PLAUSIBLE", "UNVERIFIED", "CONTRADICTED",
    ]


# --------------------------------------------------------------------------------------
# Statistical analysis plan -- CONTRACT.md Section 11
# --------------------------------------------------------------------------------------
def test_bootstrap_configuration(prereg: dict) -> None:
    ci = prereg["statistical_analysis_plan"]["confidence_intervals"]
    assert ci["method"] == "paired_bootstrap"
    assert ci["n_resamples"] == 10000, "CONTRACT.md Section 11 requires 10,000 resamples"
    assert ci["confidence_level"] == 0.95


def test_holm_correction_with_declared_families(prereg: dict) -> None:
    mc = prereg["statistical_analysis_plan"]["multiplicity_correction"]
    assert mc["method"] == "holm"
    assert len(mc["families"]) >= 1
    # Families must be declared in advance, and the primary family must be the two endpoints.
    primary = [f for f in mc["families"] if f["name"] == "primary_endpoints"]
    assert primary, "No primary_endpoints comparison family declared"
    assert len(primary[0]["members"]) == 2


def test_calibration_metrics_required(prereg: dict) -> None:
    cal = prereg["statistical_analysis_plan"]["calibration"]
    assert cal["required"] is True
    assert "brier_score" in cal["metrics"], "CONTRACT.md Section 11 requires a Brier score"


def test_pareto_frontier_required(prereg: dict) -> None:
    pareto = prereg["statistical_analysis_plan"]["pareto_analysis"]
    assert pareto["required"] is True
    assert set(pareto["axes"]) == {"UALER", "ATPR"}


def test_random_seed_fixed(prereg: dict) -> None:
    assert prereg["statistical_analysis_plan"]["random_seed"] == 42


def test_statistical_power_is_addressed(prereg: dict) -> None:
    """A 20-run gate cell establishes existence, not magnitude; that must be stated."""
    power = prereg["statistical_power"]
    assert power["gate_min_runs_per_cell"] == prereg["hard_gates"]["misbinding_reproducible"]["min_runs_per_cell"]
    for field in ("detectable_at_n20", "full_experiment_target", "underpowered_reporting"):
        assert power.get(field), f"statistical_power is missing {field!r}"


# --------------------------------------------------------------------------------------
# Splits -- CONTRACT.md Section 5.3
# --------------------------------------------------------------------------------------
def test_split_fractions_match_contract_and_sum_to_one(prereg: dict) -> None:
    fractions = prereg["splits"]["fractions"]
    assert fractions["train_development"] == 0.50
    assert fractions["validation"] == 0.20
    assert fractions["test"] == 0.30
    assert sum(fractions.values()) == pytest.approx(1.0)


def test_splits_are_by_template_not_by_page(prereg: dict) -> None:
    assert set(prereg["splits"]["split_by"]) == {
        "entity_template", "site_template", "attack_template",
    }


def test_both_generalization_holdouts_declared(prereg: dict) -> None:
    holdouts = prereg["splits"]["holdouts"]
    assert "adaptive_holdout" in holdouts and "transfer_holdout" in holdouts


# --------------------------------------------------------------------------------------
# Parser exclusion rule -- CONTRACT.md Section 8
# --------------------------------------------------------------------------------------
def test_parser_disagreement_rule_excludes_and_reports(prereg: dict) -> None:
    rule = prereg["parser_disagreement_rule"]
    assert rule["primary_denominator_treatment"] == "exclude"
    assert rule["exclusion_reporting"], "The exclusion rate must be reported"
    assert 0 < rule["max_tolerated_exclusion_rate"] <= 1
    assert rule["exclusion_rate_breach_action"], (
        "A tolerance threshold with no breach action is unenforceable"
    )


def test_manual_validation_is_prohibited(prereg: dict) -> None:
    assert prereg["failure_stage_attribution"]["manual_validation_allowed"] is False


def test_failure_stage_trace_fields_match_contract(prereg: dict) -> None:
    assert prereg["failure_stage_attribution"]["required_trace_fields"] == [
        "retrieval_candidates",
        "resolved_entities",
        "extracted_entity_domain_relations",
        "inferred_action_authorizations",
        "presented_links",
    ]


# --------------------------------------------------------------------------------------
# Scope, ablations, baselines
# --------------------------------------------------------------------------------------
def test_prohibited_scope_redefinitions_recorded(prereg: dict) -> None:
    """All nine Section 2 prohibitions must be frozen so scope drift is detectable."""
    assert len(prereg["prohibited_scope_redefinitions"]) == 9


def test_all_seven_ablations_required(prereg: dict) -> None:
    assert len(prereg["required_ablations"]["items"]) == 7, (
        "CONTRACT.md Section 9 lists seven required ablations"
    )


def test_all_ten_baselines_required(prereg: dict) -> None:
    assert len(prereg["required_baselines"]["items"]) == 10, (
        "CONTRACT.md Section 10 requires at least ten baselines"
    )
    items = prereg["required_baselines"]["items"]
    # The two degenerate defenses are what give the ATPR endpoint teeth.
    assert "reject_all_login_and_pay_links" in items
    assert "allow_only_registry_listed_official_domain" in items


def test_safety_fail_closed_conditions_complete(prereg: dict) -> None:
    """All seven Section 16 fail-closed conditions must be frozen for the CI validator."""
    safety = prereg["safety_fail_closed_conditions"]
    assert len(safety["fail_closed_on"]) == 7
    assert prereg["safety_fail_closed_conditions"]["deployment_constraints"][
        "public_deceptive_deployment_allowed"
    ] is False


# --------------------------------------------------------------------------------------
# Checksums and tamper detection -- CONTRACT.md Sections 12, 15
# --------------------------------------------------------------------------------------
def test_frozen_params_loads(frozen: dict) -> None:
    assert frozen["hash_algorithm"] == "sha256"


def test_preregistration_checksum_matches(frozen: dict) -> None:
    """Recompute the hash and compare. Fails if preregistration.yaml was edited post-freeze."""
    recomputed = hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()
    recorded = frozen["preregistration"]["sha256"]
    assert recomputed == recorded, (
        "preregistration.yaml has changed since it was frozen.\n"
        f"  recorded   : {recorded}\n"
        f"  recomputed : {recomputed}\n"
        "If this is a legitimate PRE-PILOT amendment, increment amendment_number, add an "
        "amendments entry, and re-run workflow/01_freeze_preregistration.py. Post-pilot "
        "threshold edits invalidate the preregistration."
    )
    assert len(recorded) == 64 and re.fullmatch(r"[0-9a-f]{64}", recorded)


def test_contract_checksum_matches(frozen: dict) -> None:
    recomputed = hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()
    assert recomputed == frozen["contract"]["sha256"]
    assert frozen["contract"]["byte_identical_to_source"] is True
    assert frozen["contract"]["sha256"] == frozen["contract"]["source_sha256"]


def test_rubric_checksum_matches(frozen: dict) -> None:
    """The rubric decides the novelty gate, so it is frozen alongside the thresholds."""
    recomputed = hashlib.sha256(RUBRIC_PATH.read_bytes()).hexdigest()
    assert recomputed == frozen["collision_rubric"]["sha256"]


def test_frozen_gates_match_preregistration(prereg: dict, frozen: dict) -> None:
    assert frozen["hard_gates_nested"] == prereg["hard_gates"]
    flat = frozen["hard_gates_flat"]
    assert len(flat) == 10
    for group, params in prereg["hard_gates"].items():
        for name, value in params.items():
            assert flat[f"{group}.{name}"] == value


def test_frozen_endpoints_match_preregistration(prereg: dict, frozen: dict) -> None:
    assert frozen["primary_endpoints"] == [ep["name"] for ep in prereg["primary_endpoints"]]


def test_frozen_params_records_pre_pilot_state(frozen: dict) -> None:
    assert frozen["pilot_executed"] is False


def test_literature_matrix_hash_is_not_a_frozen_constraint(frozen: dict) -> None:
    """Section 13 mandates re-running the check, so its hash must not be a tamper anchor."""
    snap = frozen["literature_matrix_checkpoint_snapshot"]
    assert snap["is_frozen_constraint"] is False


# --------------------------------------------------------------------------------------
# Literature collision check -- CONTRACT.md Sections 4.2, 13
# --------------------------------------------------------------------------------------
def test_literature_matrix_exists_and_loads(matrix: dict) -> None:
    assert matrix["literature_matrix_version"]


def test_literature_matrix_covers_all_eight_prior_works(matrix: dict) -> None:
    assert len(matrix["prior_works"]) == 8, (
        "CONTRACT.md Section 4.1 declares eight prior works that cannot be claimed as novel"
    )
    assert matrix["counts"]["prior_works_declared"] == 8


def test_literature_matrix_verdict_is_no_direct_collision(matrix: dict) -> None:
    assert matrix["verdict"] == "NO_DIRECT_COLLISION"
    assert matrix["direct_collision_any"] is False
    assert matrix["colliding_paper_ids"] == []
    assert matrix["novelty_claim_permitted"] is True


def test_every_prior_work_record_has_all_section_13_fields(matrix: dict) -> None:
    required = [
        "paper_id", "problem_overlap", "attack_overlap", "method_overlap",
        "evaluation_overlap", "direct_collision", "required_project_change",
    ]
    for rec in matrix["prior_works"]:
        for field in required:
            assert field in rec, f"{rec.get('paper_id')} missing Section 13 field {field!r}"


def test_no_prior_work_has_direct_collision(matrix: dict) -> None:
    colliding = [r["paper_id"] for r in matrix["prior_works"] if r["direct_collision"]]
    assert not colliding, f"Direct collision detected for {colliding}"


def test_overlap_scores_are_in_range(matrix: dict) -> None:
    axes = ["problem_overlap", "attack_overlap", "method_overlap", "evaluation_overlap"]
    for rec in matrix["prior_works"] + [r for r in matrix["discovered_candidates"] if r.get("scored")]:
        for axis in axes:
            score = rec[axis]
            assert isinstance(score, int) and 0 <= score <= 3, (
                f"{rec['paper_id']} {axis} = {score!r} is outside the Section 13 range 0-3"
            )


def test_collision_verdict_is_consistent_with_the_frozen_rule(matrix: dict, rubric: dict) -> None:
    """Recompute each record's collision flag from its own scores and the frozen rule."""
    p_thresh = rubric["collision_rule"]["problem_threshold"]
    m_thresh = rubric["collision_rule"]["method_threshold"]
    for rec in matrix["prior_works"] + [r for r in matrix["discovered_candidates"] if r.get("scored")]:
        expected = rec["problem_overlap"] >= p_thresh and rec["method_overlap"] >= m_thresh
        assert rec["direct_collision"] is expected, (
            f"{rec['paper_id']}: direct_collision={rec['direct_collision']} but scores "
            f"P{rec['problem_overlap']}/M{rec['method_overlap']} imply {expected}"
        )
    any_collision = any(
        r["direct_collision"]
        for r in matrix["prior_works"] + matrix["discovered_candidates"]
        if "direct_collision" in r
    )
    assert matrix["direct_collision_any"] is any_collision


def test_collision_rule_is_conjunctive_over_the_two_named_axes(rubric: dict) -> None:
    """Section 13 excludes shared attacks alone from constituting a collision."""
    rule = rubric["collision_rule"]
    assert "problem_overlap" in rule["expression"] and "method_overlap" in rule["expression"]
    assert "AND" in rule["expression"]
    assert set(rule["axes_excluded_from_rule"]) == {"attack_overlap", "evaluation_overlap"}


def test_all_six_search_concepts_are_defined(matrix: dict) -> None:
    assert matrix["counts"]["search_concepts_defined"] == 6, (
        "CONTRACT.md Section 13 lists six search concepts"
    )


def test_matrix_records_provenance_and_limitations(matrix: dict) -> None:
    """A novelty verdict without stated limitations invites over-reading."""
    assert matrix["rubric"]["version"]
    assert "no LLM judgement" in matrix["rubric"]["scored_by"]
    assert len(matrix["limitations"]) >= 3
    assert matrix["checkpoint"] in matrix["checkpoint_schedule"]


def test_citation_integrity_is_reported(matrix: dict) -> None:
    """Mismatched citations must surface rather than hide behind a resolved identifier."""
    integrity = matrix["citation_integrity"]
    assert "title_mismatch_paper_ids" in integrity
    for rec in matrix["prior_works"]:
        assert "title_consistent" in rec
        if not rec["title_consistent"]:
            assert rec.get("title_mismatch_caveat"), (
                f"{rec['paper_id']} is flagged inconsistent but carries no caveat"
            )


def test_novelty_gate_is_wired_to_the_matrix_verdict(prereg: dict, matrix: dict) -> None:
    """The Section 12 novelty gate must be decided by the artifact the checker writes."""
    gate = prereg["gate_operationalization"]["novelty"]
    assert "NO_DIRECT_COLLISION" in gate["pass_condition"]
    assert gate["checker"] == "evaluation/literature_checker.py"
    assert prereg["hard_gates"]["novelty"]["direct_collision_allowed"] is False
    # The gate as written is satisfied by the current matrix.
    assert matrix["verdict"] == "NO_DIRECT_COLLISION"


def test_novelty_recheck_schedule_matches_contract(prereg: dict) -> None:
    """Section 13: run at project start, after pilot, before full experiments, before freeze."""
    assert prereg["gate_operationalization"]["novelty"]["recheck_schedule"] == [
        "project_start", "post_pilot", "pre_full_experiments", "pre_manuscript_freeze",
    ]


def test_novelty_wording_is_gated(prereg: dict) -> None:
    gate = prereg["claim_gating"]["novelty_wording_gate"]
    assert "NO_DIRECT_COLLISION" in gate
    assert "first" in gate and "novel" in gate
