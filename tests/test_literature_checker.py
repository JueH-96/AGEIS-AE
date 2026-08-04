"""Unit and falsifiability tests for the literature collision checker.

The most important tests here are the two controls in the FALSIFIABILITY section. A novelty
check that structurally cannot return ``DIRECT_COLLISION`` would pass every test in
``test_preregistration.py`` while providing no protection at all against a false novelty
claim. These tests feed the frozen rubric a synthetic abstract engineered to occupy the
``CONTRACT.md`` Section 4.2 contribution and assert that the verdict flips.

Run with:
    uv run pytest tests/ -v
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from evaluation.literature_checker import (
    TITLE_MATCH_THRESHOLD,
    build_haystack,
    decide_collision,
    evaluate_overlap,
    load_rubric,
    parse_prior_works,
    parse_search_concepts,
    score_axis,
    title_similarity,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = REPO_ROOT / "CONTRACT.md"

AXES = ["problem_overlap", "attack_overlap", "method_overlap", "evaluation_overlap"]


@pytest.fixture(scope="module")
def rubric() -> dict:
    return load_rubric()


@pytest.fixture(scope="module")
def contract_text() -> str:
    return CONTRACT_PATH.read_text(encoding="utf-8")


# ======================================================================================
# FALSIFIABILITY -- the checker must be able to say "collision"
# ======================================================================================

# Engineered to occupy the Section 4.2 contribution on both decisive axes: a legitimate
# entity bound to an unauthorized action endpoint by an LLM (problem), defended by
# action-level authorization inference over an evidence graph (method).
SYNTHETIC_COLLIDING_ABSTRACT = """
We study action-link misbinding in web-enabled LLM recommendations. A large language model
correctly identifies a legitimate business entity but presents an attacker-controlled
payment link as its official website, so the entity is right while the action endpoint is
unauthorized. We formalise an entity-domain authorization relation and propose an
action-aware defense that performs authorization inference per action across browse,
contact, book, login and pay, using an evidence graph built from cross-source verification
of official backlinks and domain ownership records. We evaluate on a synthetic benchmark
whose authorization graph provides ground truth, reporting per-action exposure rate and the
security-utility trade-off against a false rejection rate baseline.
"""

# A representative prior work: shares the attack substrate but defends nothing at the
# action-authorization level. Must NOT be a collision.
SYNTHETIC_POISONING_ABSTRACT = """
We show that a single poisoned web page is enough to manipulate a generative recommender
into promoting a fabricated product. Our attack pollutes the retrieval corpus with multiple
mutually corroborating pages and coordinated Sybil sources, and we benchmark poisoning
success across retrieval settings.
"""


def test_synthetic_colliding_paper_is_detected(rubric: dict) -> None:
    """POSITIVE CONTROL: a paper occupying the contribution must be flagged."""
    hay = build_haystack("Action-Link Misbinding and Action-Aware Authorization Verification",
                         SYNTHETIC_COLLIDING_ABSTRACT)
    scores, markers = evaluate_overlap(hay, rubric)
    direct, _near = decide_collision(scores, rubric)

    assert scores["problem_overlap"] >= 3, (
        f"Synthetic colliding paper scored only {scores['problem_overlap']} on problem_overlap; "
        f"markers fired: {markers['problem_overlap']}"
    )
    assert scores["method_overlap"] >= 3, (
        f"Synthetic colliding paper scored only {scores['method_overlap']} on method_overlap; "
        f"markers fired: {markers['method_overlap']}"
    )
    assert direct is True, (
        "The rubric failed to flag a paper deliberately written to occupy the Section 4.2 "
        "contribution. The novelty check would be incapable of ever returning "
        "DIRECT_COLLISION, making every NO_DIRECT_COLLISION verdict meaningless."
    )


def test_pure_poisoning_paper_is_not_a_collision(rubric: dict) -> None:
    """NEGATIVE CONTROL: Section 13 says shared poisoning alone is not a collision."""
    hay = build_haystack("One Polluted Page Is Enough", SYNTHETIC_POISONING_ABSTRACT)
    scores, _markers = evaluate_overlap(hay, rubric)
    direct, _near = decide_collision(scores, rubric)
    assert direct is False, "A pure web-poisoning paper must not register as a direct collision"
    assert scores["attack_overlap"] >= 2, "Attack overlap should still be recognised and reported"


def test_high_attack_overlap_alone_cannot_trigger_collision(rubric: dict) -> None:
    """Section 13 explicitly excludes 'merely web poisoning or phishing' from collision."""
    saturated = {"problem_overlap": 0, "attack_overlap": 3,
                 "method_overlap": 0, "evaluation_overlap": 3}
    direct, near = decide_collision(saturated, rubric)
    assert direct is False and near is False


def test_collision_requires_both_decisive_axes(rubric: dict) -> None:
    """Verify the rule is a conjunction, not a disjunction, at the exact boundary."""
    assert decide_collision(
        {"problem_overlap": 3, "attack_overlap": 0, "method_overlap": 2,
         "evaluation_overlap": 0}, rubric)[0] is False
    assert decide_collision(
        {"problem_overlap": 2, "attack_overlap": 0, "method_overlap": 3,
         "evaluation_overlap": 0}, rubric)[0] is False
    assert decide_collision(
        {"problem_overlap": 3, "attack_overlap": 0, "method_overlap": 3,
         "evaluation_overlap": 0}, rubric)[0] is True


def test_near_collision_fires_between_thresholds_and_is_mutually_exclusive(rubric: dict) -> None:
    direct, near = decide_collision(
        {"problem_overlap": 2, "attack_overlap": 0, "method_overlap": 2,
         "evaluation_overlap": 0}, rubric)
    assert direct is False and near is True
    direct, near = decide_collision(
        {"problem_overlap": 3, "attack_overlap": 0, "method_overlap": 3,
         "evaluation_overlap": 0}, rubric)
    assert direct is True and near is False, "near_collision must not co-fire with direct"


# ======================================================================================
# SEPARATOR CONVENTION -- regression lock on the rubric v1.0 defect
# ======================================================================================
@pytest.mark.parametrize(
    ("text", "expected_marker"),
    [
        ("we study misbinding of entity links", "misbinding_framing"),
        ("we study mis-binding of entity links", "misbinding_framing"),
        ("official site misattribution by llms", "misbinding_framing"),
        ("official site mis-attribution by llms", "misbinding_framing"),
    ],
)
def test_compound_terms_match_joined_spelling(
    rubric: dict, text: str, expected_marker: str
) -> None:
    """Regression lock: rubric v1.0 used `.` as separator and missed joined spellings.

    "misbinding" and "misattribution" are the most natural spellings of this project's
    central terms. Missing them under-detected overlap on the decisive problem axis, which
    biases the check toward a false novelty claim.
    """
    _score, matched = score_axis(text, rubric["axes"]["problem_overlap"])
    assert expected_marker in matched, (
        f"{expected_marker!r} did not fire on {text!r}; markers fired: {matched}"
    )


def test_all_rubric_patterns_compile(rubric: dict) -> None:
    """A malformed pattern would silently never match, weakening the check invisibly."""
    for axis_name, axis in rubric["axes"].items():
        for marker_name, marker in axis["markers"].items():
            for pattern in marker["patterns"]:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    pytest.fail(f"{axis_name}.{marker_name}: invalid regex {pattern!r}: {exc}")


def test_no_rubric_pattern_uses_bare_dot_as_word_separator(rubric: dict) -> None:
    """Guard against reintroducing the v1.0 defect.

    Flags a `.` sitting between two word characters, which is the defective form. A `.`
    inside a character class or followed by a quantifier is left alone.
    """
    offenders: list[str] = []
    for axis_name, axis in rubric["axes"].items():
        for marker_name, marker in axis["markers"].items():
            for pattern in marker["patterns"]:
                stripped = re.sub(r"\[[^\]]*\]", "", pattern)  # ignore character classes
                if re.search(r"\w\.(?![*+?{])\w", stripped):
                    offenders.append(f"{axis_name}.{marker_name}: {pattern}")
    assert not offenders, (
        "Patterns use a bare '.' between word characters, which cannot match joined "
        f"spellings: {offenders}"
    )


def test_rubric_declares_its_separator_class_and_version(rubric: dict) -> None:
    assert rubric["separator_class"] == r"[-_\s]?"
    assert rubric["rubric_version"] == "1.1"
    assert len(rubric["revision_history"]) >= 2, (
        "The v1.0 -> v1.1 correction must remain documented in the rubric"
    )


# ======================================================================================
# SCORING MECHANICS
# ======================================================================================
def test_score_is_capped_at_three(rubric: dict) -> None:
    """All four markers firing must yield exactly 3, matching the Section 13 range."""
    hay = build_haystack("", SYNTHETIC_COLLIDING_ABSTRACT)
    score, matched = score_axis(hay, rubric["axes"]["problem_overlap"])
    assert len(matched) == 4, f"Expected all 4 problem markers to fire, got {matched}"
    assert score == 3, "Score must saturate at 3, not exceed the Section 13 range"


def test_empty_text_scores_zero_on_every_axis(rubric: dict) -> None:
    scores, markers = evaluate_overlap("", rubric)
    assert all(scores[a] == 0 for a in AXES)
    assert all(markers[a] == [] for a in AXES)


def test_marker_counted_once_even_if_several_patterns_match(rubric: dict) -> None:
    """Duplicate evidence for one concept must not inflate the score."""
    text = "booking link and payment link and login portal and official website"
    score, matched = score_axis(text, rubric["axes"]["problem_overlap"])
    assert matched == ["action_endpoint_focus"]
    assert score == 1


def test_all_axes_scored_and_in_range(rubric: dict) -> None:
    scores, _ = evaluate_overlap(build_haystack("", SYNTHETIC_COLLIDING_ABSTRACT), rubric)
    assert set(scores) == set(AXES)
    assert all(0 <= scores[a] <= 3 for a in AXES)


def test_build_haystack_includes_contract_characterization(rubric: dict) -> None:
    """The union-of-evidence behaviour is deliberate and must not regress silently."""
    hay = build_haystack("", "", "Provenance-based defense-in-depth")
    assert "provenance" in hay
    score, matched = score_axis(hay, rubric["axes"]["method_overlap"])
    assert "cross_source_evidence_graph" in matched


def test_build_haystack_is_lowercased_and_normalized() -> None:
    hay = build_haystack("TITLE   Here", "Abstract\n\nText")
    assert hay == hay.lower()
    assert "  " not in hay


def test_authorization_marker_ignores_bare_threat_model_mention(rubric: dict) -> None:
    """"unauthorized access" in a threat model is not an authorization defense."""
    score, matched = score_axis(
        "an attacker gains unauthorized access to the corpus",
        rubric["axes"]["method_overlap"],
    )
    assert "authorization_as_defense" not in matched
    assert score == 0


def test_authorization_marker_fires_on_genuine_defense(rubric: dict) -> None:
    _score, matched = score_axis(
        "we verify the authorization status of each domain before presenting a link",
        rubric["axes"]["method_overlap"],
    )
    assert "authorization_as_defense" in matched


# ======================================================================================
# CITATION INTEGRITY
# ======================================================================================
def test_title_similarity_identical_is_one() -> None:
    t = "TopoGuard: Graph Theory Based Defenses Against Split-Knowledge Attacks on RAG"
    assert title_similarity(t, t) == pytest.approx(1.0)


def test_title_similarity_detects_a_different_paper() -> None:
    """The real discrepancy found at project_start: arXiv:2604.00387."""
    cited = ("RAGShield: Provenance-Verified Defense-in-Depth Against Knowledge Base "
             "Poisoning in Government Retrieval-Augmented Generation Systems")
    retrieved = "RAGShield: Detecting Numerical Claim Manipulation in Government RAG Systems"
    sim = title_similarity(cited, retrieved)
    assert sim < TITLE_MATCH_THRESHOLD, (
        f"similarity {sim} should fall below the {TITLE_MATCH_THRESHOLD} mismatch threshold"
    )


def test_title_similarity_tolerates_punctuation_and_case() -> None:
    a = "Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications"
    b = "not what you have signed up for compromising real world llm integrated applications"
    assert title_similarity(a, b) >= TITLE_MATCH_THRESHOLD


def test_title_similarity_handles_empty_input() -> None:
    assert title_similarity("", "anything") == 0.0
    assert title_similarity("anything", "") == 0.0


# ======================================================================================
# CONTRACT PARSING
# ======================================================================================
def test_parses_exactly_eight_prior_works(contract_text: str) -> None:
    works = parse_prior_works(contract_text)
    assert len(works) == 8, "CONTRACT.md Section 4.1 declares eight prior works"


def test_prior_work_ids_are_unique_and_well_formed(contract_text: str) -> None:
    works = parse_prior_works(contract_text)
    assert len({w.paper_id for w in works}) == 8
    assert len({w.arxiv_id for w in works}) == 8
    for w in works:
        assert re.fullmatch(r"\d{4}\.\d{4,5}", w.arxiv_id), f"bad arXiv id {w.arxiv_id!r}"
        assert w.short_name and w.contract_characterization


def test_named_prior_works_are_recognised(contract_text: str) -> None:
    """The contract's parenthesised short names must survive parsing."""
    short_names = {w.short_name for w in parse_prior_works(contract_text)}
    for expected in ("FORGE", "SIREN", "TopoGuard", "RAGShield", "Greshake"):
        assert expected in short_names, f"{expected} not parsed from Section 4.1"


def test_prior_works_are_joined_to_section_20_citations(contract_text: str) -> None:
    """Every Section 4.1 entry must resolve to a Section 20 title, or the citation is broken."""
    for w in parse_prior_works(contract_text):
        assert w.contract_cited_title, (
            f"{w.paper_id} (arXiv:{w.arxiv_id}) has no matching Section 20 citation"
        )


def test_parses_all_six_search_concepts(contract_text: str) -> None:
    concepts = parse_search_concepts(contract_text)
    assert len(concepts) == 6, "CONTRACT.md Section 13 lists six search concepts"
    assert any("misbinding" in c for c in concepts)


def test_parser_raises_on_missing_section() -> None:
    """Silent failure on a malformed contract would produce an empty, passing matrix."""
    with pytest.raises(ValueError):
        parse_prior_works("# 1. Nothing here\n\nno sections\n")


# ======================================================================================
# GENERATED MATRIX AGREES WITH A FRESH RECOMPUTATION
# ======================================================================================
def test_matrix_scores_are_reproducible_from_the_rubric(rubric: dict) -> None:
    """Recompute each verified record's scores from its stored markers and re-derive collision.

    Confirms the emitted matrix is internally consistent: the marker lists, the integer
    scores, and the collision flags must all agree with the frozen rule.
    """
    matrix_path = REPO_ROOT / "results" / "literature_matrix.yaml"
    with matrix_path.open(encoding="utf-8") as fh:
        matrix = yaml.safe_load(fh)

    scored = matrix["prior_works"] + [
        r for r in matrix["discovered_candidates"] if r.get("scored")
    ]
    assert scored, "matrix contains no scored records"

    for rec in scored:
        for axis in AXES:
            markers = rec["matched_markers"][axis] or []
            expected = min(3, len(markers))
            assert rec[axis] == expected, (
                f"{rec['paper_id']} {axis}: stored score {rec[axis]} does not equal "
                f"min(3, len({markers})) = {expected}"
            )
            # Every reported marker must be a real marker name from the rubric.
            for m in markers:
                assert m in rubric["axes"][axis]["markers"], (
                    f"{rec['paper_id']} {axis}: unknown marker {m!r}"
                )
        expected_direct, expected_near = decide_collision(
            {a: rec[a] for a in AXES}, rubric
        )
        assert rec["direct_collision"] is expected_direct
        assert rec["near_collision"] is expected_near
