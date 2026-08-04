"""Unit and property tests for the controlled world, authorization graph and splits.

Covers CONTRACT.md Sections 1, 5.1, 5.2, 5.3, 7 and 8.

The tests are organised around the failure modes that would let every downstream number look
plausible while measuring nothing:

* a graph that is not total, so runs drift in and out of a denominator;
* a graph where authorization has collapsed into a property of the domain, making RQ3 vacuous;
* a corpus where a label is recoverable by substring match, making any defense accuracy an
  artifact of the generator;
* splits that leak, so a "holdout" result is a memorisation result;
* an attack design whose "fractional factorial" is not actually balanced.

Run with:
    uv run pytest tests/test_benchmark_and_splits.py -v
"""

from __future__ import annotations

import itertools
from collections import Counter
from pathlib import Path

import pytest
import yaml

from registry.authorization_graph import (
    ACTIONS,
    EVIDENCE_TYPES,
    EVIDENCE_TYPES_CONTRACT_VERBATIM,
    AuthorizationGraph,
    check_graph_invariants,
    load_authorization_graph,
)
from registry.entity_generator import (
    ENTITIES_PER_TEMPLATE,
    ENTITY_STATUSES,
    MACHINE_READABLE_KEY,
    TEMPLATES_PER_CATEGORY,
    EntityWorld,
    generate_entity_world,
    load_entity_world,
)
from registry.lexicon import CATEGORIES, OFFICIAL_CLAIM_PHRASES

# Distinctive fragment of each official-claim phrase, with the {name} placeholder removed, so a
# claim can be detected in rendered prose without matching the bare word "official".
OFFICIAL_CLAIM_FRAGMENTS: tuple[str, ...] = tuple(
    max(phrase.split("{name}"), key=len).strip(" .-").lower()
    for phrase in OFFICIAL_CLAIM_PHRASES
)
from registry.split_builder import (
    REGIMES,
    SPLIT_FRACTIONS,
    SPLIT_NAMES,
    TRANSFER_HOLDOUT_CATEGORIES,
    build_split_plan,
    check_region_disjointness,
    check_split_invariants,
    load_splits,
)
from site_generator.generator import (
    ADAPTIVE_HOLDOUT_REGION,
    ATTACK_FACTORS,
    DOMAIN_ROLE_TO_PAGE_ROLE,
    FRACTION_FACTORS,
    LABEL_VOCABULARY_FORBIDDEN_IN_PAGES,
    PAGE_ROLES,
    AttackDesign,
    build_site_templates,
    in_adaptive_holdout_region,
    load_attack_design,
    load_page_manifest,
    load_site_templates,
    visible_text,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = REPO_ROOT / "CONTRACT.md"
PAGES_ROOT = REPO_ROOT / "data" / "benchmark" / "pages"


# ======================================================================================
# Fixtures
# ======================================================================================
@pytest.fixture(scope="module")
def world() -> EntityWorld:
    return load_entity_world()


@pytest.fixture(scope="module")
def graph() -> AuthorizationGraph:
    return load_authorization_graph()


@pytest.fixture(scope="module")
def design() -> AttackDesign:
    return load_attack_design()


@pytest.fixture(scope="module")
def site_templates() -> list:
    return load_site_templates()


@pytest.fixture(scope="module")
def splits() -> dict:
    return load_splits()


@pytest.fixture(scope="module")
def manifest() -> list[dict]:
    return load_page_manifest()


@pytest.fixture(scope="module")
def contract_text() -> str:
    return CONTRACT_PATH.read_text(encoding="utf-8")


# ======================================================================================
# Section 5.1 -- entity records
# ======================================================================================
SECTION_5_1_FIELDS = (
    "entity_id", "canonical_name", "category", "address_id", "phone_id", "coordinates_id",
    "status",
)


def test_entity_record_has_exactly_the_contract_fields(world: EntityWorld) -> None:
    """Exactly the seven Section 5.1 fields, no more: an extra field could carry a label."""
    for entity in world.entities:
        assert tuple(entity.as_dict()) == SECTION_5_1_FIELDS


def test_contract_section_5_1_example_is_reproduced(world: EntityWorld) -> None:
    """E0001 reproduces the worked example in the contract, so the schema is demonstrably read."""
    first = world.by_id("E0001").as_dict()
    assert first == {
        "entity_id": "E0001",
        "canonical_name": "Harbour Lantern Bistro",
        "category": "restaurant",
        "address_id": "A0001",
        "phone_id": "P0001",
        "coordinates_id": "C0001",
        "status": "active",
    }


def test_entity_ids_are_unique_sequential_and_contract_formatted(world: EntityWorld) -> None:
    ids = [e.entity_id for e in world.entities]
    assert len(ids) == len(set(ids))
    for eid in ids:
        assert eid.startswith("E") and eid[1:].isdigit() and len(eid) == 5


def test_canonical_names_are_unique(world: EntityWorld) -> None:
    """A duplicate name would make entity resolution ambiguous and corrupt every label."""
    names = [e.canonical_name for e in world.entities]
    assert len(names) == len(set(names))


def test_every_id_reference_resolves(world: EntityWorld) -> None:
    for entity in world.entities:
        assert entity.address_id in world.addresses
        assert entity.phone_id in world.phones
        assert entity.coordinates_id in world.coordinates
        assert entity.entity_id in world.aliases


def test_statuses_are_in_the_frozen_enum(world: EntityWorld) -> None:
    assert {e.status for e in world.entities} <= set(ENTITY_STATUSES)


def test_all_categories_are_populated(world: EntityWorld) -> None:
    counts = Counter(e.category for e in world.controlled)
    assert set(counts) == set(CATEGORIES)
    assert set(counts.values()) == {TEMPLATES_PER_CATEGORY * ENTITIES_PER_TEMPLATE}


def test_fabricated_controls_exist_for_rq1_condition_7(world: EntityWorld) -> None:
    """CONTRACT.md Section 3 RQ1 condition 7: fabricated entity + fabricated domain, control."""
    assert len(world.fabricated) >= 1
    assert all(e.status == "fabricated_control" for e in world.fabricated)


def test_confusable_siblings_share_a_template_and_a_lead_word(world: EntityWorld) -> None:
    """The benign-confusion condition must be structural, and siblings must not straddle splits."""
    assert world.sibling_of
    for eid, sib in world.sibling_of.items():
        assert world.entity_to_template[eid] == world.entity_to_template[sib]
        a, b = world.by_id(eid).canonical_name, world.by_id(sib).canonical_name
        assert a != b
        assert a.split()[0] == b.split()[0], (a, b)


def test_generation_is_deterministic() -> None:
    """Regenerating must reproduce the world byte-for-byte, or nothing downstream reproduces."""
    a, b = generate_entity_world(), generate_entity_world()
    assert [e.as_dict() for e in a.entities] == [e.as_dict() for e in b.entities]
    assert a.addresses == b.addresses and a.phones == b.phones
    assert a.coordinates == b.coordinates and a.aliases == b.aliases


def test_generated_world_matches_the_persisted_registry(world: EntityWorld) -> None:
    fresh = generate_entity_world()
    assert [e.as_dict() for e in fresh.entities] == [e.as_dict() for e in world.entities]


# ======================================================================================
# Section 1 / 5.2 -- authorization graph
# ======================================================================================
def test_action_set_matches_contract_section_1(contract_text: str) -> None:
    assert "a in {browse, contact, book, login, pay}" in contract_text
    assert ACTIONS == ("browse", "contact", "book", "login", "pay")


def test_edge_schema_is_exactly_the_five_contract_fields() -> None:
    doc = yaml.safe_load(
        (REPO_ROOT / "registry" / "authorization_graph.yaml").read_text(encoding="utf-8"))
    expected = ["entity_id", "domain_id", "action", "authorized", "evidence_type"]
    assert doc["metadata"]["edge_schema"] == expected
    for edge in doc["edges"][:200]:
        assert list(edge) == expected


def test_graph_declares_itself_the_sole_ground_truth() -> None:
    doc = yaml.safe_load(
        (REPO_ROOT / "registry" / "authorization_graph.yaml").read_text(encoding="utf-8"))
    assert doc["metadata"]["is_sole_ground_truth"] is True
    assert doc["metadata"]["llm_generated_labels"] is False


def test_graph_is_total_over_pairs_and_actions(graph: AuthorizationGraph) -> None:
    """A missing edge would be ambiguous between unauthorized and undefined."""
    per_pair: dict[tuple[str, str], set[str]] = {}
    for edge in graph.edges:
        per_pair.setdefault((edge.entity_id, edge.domain_id), set()).add(edge.action)
    assert per_pair
    for pair, actions in per_pair.items():
        assert actions == set(ACTIONS), pair
    assert len(graph.edges) == len(per_pair) * len(ACTIONS)


def test_every_triple_has_exactly_one_label(graph: AuthorizationGraph) -> None:
    triples = [(e.entity_id, e.domain_id, e.action) for e in graph.edges]
    assert len(triples) == len(set(triples))


def test_authorized_is_boolean(graph: AuthorizationGraph) -> None:
    assert {type(e.authorized) for e in graph.edges} == {bool}


def test_evidence_types_are_in_the_frozen_enum(graph: AuthorizationGraph) -> None:
    assert {e.evidence_type for e in graph.edges} <= set(EVIDENCE_TYPES)


def test_contract_verbatim_evidence_types_are_all_used(graph: AuthorizationGraph) -> None:
    """The three Section 5.2 example evidence types must actually appear."""
    used = {e.evidence_type for e in graph.edges}
    assert set(EVIDENCE_TYPES_CONTRACT_VERBATIM) <= used


def test_authorization_is_action_relative(graph: AuthorizationGraph) -> None:
    """CONTRACT.md Section 1: a domain may be authorized for one action and not another.

    If this collapsed, the risk ordering tau_browse < ... < tau_pay would have nothing to
    discriminate and RQ3 would be vacuous.
    """
    split_pairs = []
    for (eid, did) in {(e.entity_id, e.domain_id) for e in graph.edges}:
        labels = [graph.authorized(eid, did, a) for a in ACTIONS]
        if any(labels) and not all(labels):
            split_pairs.append((eid, did))
    assert len(split_pairs) >= 100, f"only {len(split_pairs)} action-split pairs"


def test_book_authorized_but_pay_unauthorized_exists(graph: AuthorizationGraph) -> None:
    """The concrete case the contract's motivation rests on: bookings yes, money no."""
    hits = [
        (eid, did) for (eid, did) in {(e.entity_id, e.domain_id) for e in graph.edges}
        if graph.authorized(eid, did, "book") and not graph.authorized(eid, did, "pay")
    ]
    assert hits, "no domain is authorized to take a booking but not a payment"


def test_authorization_is_entity_relative(graph: AuthorizationGraph) -> None:
    """The same domain must be authorized for one entity and unauthorized for another.

    This is what a URL-reputation baseline cannot represent, so it must be exercised.
    """
    per_domain: dict[str, set[bool]] = {}
    for edge in graph.edges:
        per_domain.setdefault(edge.domain_id, set()).add(edge.authorized)
    mixed = [d for d, vals in per_domain.items() if vals == {True, False}]
    assert mixed
    shared_mixed = [d for d in mixed if graph.domains[d].shared]
    assert shared_mixed, "no shared third-party domain carries both labels"


def test_sibling_official_domain_is_unauthorized_for_the_other_entity(
    world: EntityWorld, graph: AuthorizationGraph
) -> None:
    """The confusable sibling's registry-listed domain must not be authorized here."""
    checked = 0
    for eid, sib in world.sibling_of.items():
        sib_official = graph.domains_for(sib, "official_site")
        if not sib_official:
            continue
        did = sib_official[0]
        if did not in graph.entity_domain_roles[eid]:
            continue
        assert not any(graph.authorized(eid, did, a) for a in ACTIONS)
        assert all(graph.authorized(sib, did, a) for a in ACTIONS)
        checked += 1
    assert checked >= 50


def test_official_domain_is_authorized_on_every_action(
    world: EntityWorld, graph: AuthorizationGraph
) -> None:
    for entity in world.controlled:
        officials = graph.domains_for(entity.entity_id, "official_site")
        assert len(officials) == 1
        assert all(graph.authorized(entity.entity_id, officials[0], a) for a in ACTIONS)


def test_adversarial_domains_are_never_authorized(graph: AuthorizationGraph) -> None:
    for edge in graph.edges:
        if graph.domains[edge.domain_id].adversarial:
            assert not edge.authorized, (edge.entity_id, edge.domain_id, edge.action)


def test_fabricated_controls_are_authorized_for_nothing(
    world: EntityWorld, graph: AuthorizationGraph
) -> None:
    for entity in world.fabricated:
        assert not any(e.authorized for e in graph.edges_for(entity.entity_id))


def test_benign_but_unauthorized_domains_exist(graph: AuthorizationGraph) -> None:
    """Unauthorized must not be a synonym for malicious, or FRR and BER become meaningless."""
    benign_unauthorized = [
        edge for edge in graph.edges
        if not edge.authorized and not graph.domains[edge.domain_id].adversarial
    ]
    assert benign_unauthorized
    assert {"benign_no_grant", "third_party_no_grant", "registry_other_entity"} <= {
        e.evidence_type for e in benign_unauthorized
    }


def test_graph_metadata_separates_labels_from_intent() -> None:
    """`adversarial` must live in domains.yaml, never in the graph."""
    graph_doc = yaml.safe_load(
        (REPO_ROOT / "registry" / "authorization_graph.yaml").read_text(encoding="utf-8"))
    assert "adversarial" not in graph_doc["metadata"]["edge_schema"]
    domains_doc = yaml.safe_load(
        (REPO_ROOT / "registry" / "domains.yaml").read_text(encoding="utf-8"))
    assert "adversarial" in domains_doc["metadata"]["experimenter_only_fields"]
    assert "MUST NOT be exposed" in domains_doc["metadata"]["experimenter_only_warning"]


def test_all_graph_invariants_hold(world: EntityWorld, graph: AuthorizationGraph) -> None:
    assert check_graph_invariants(graph, world) == []


# ======================================================================================
# Section 7 -- page roles and attack design
# ======================================================================================
def test_all_ten_contract_page_roles_are_declared(contract_text: str) -> None:
    section = contract_text.split("## 7. Page and attack generation")[1].split("Attack factors")[0]
    for phrase in ("official site", "authorized booking provider",
                   "authorized information directory", "impersonating official site",
                   "unauthorized booking provider", "unauthorized login portal",
                   "unauthorized payment portal", "expired-domain takeover simulation",
                   "corroborating blog/directory pages", "benign confusing pages"):
        assert phrase in section, phrase
    assert len(PAGE_ROLES) == 10


def test_attack_factors_match_contract_section_7(contract_text: str) -> None:
    """Factor names and levels are re-read from the contract, not trusted from the module."""
    block = contract_text.split("Attack factors:")[1].split("```yaml")[1].split("```")[0]
    declared = yaml.safe_load(block)
    assert set(declared) == set(ATTACK_FACTORS)
    for factor, levels in declared.items():
        assert list(ATTACK_FACTORS[factor]) == list(levels), factor


def test_full_factorial_is_completely_enumerated(design: AttackDesign) -> None:
    expected = 1
    for levels in ATTACK_FACTORS.values():
        expected *= len(levels)
    assert len(design.full) == expected
    combos = {tuple(t.factor_levels().values()) for t in design.full}
    assert len(combos) == expected


def test_attack_template_ids_are_unique(design: AttackDesign) -> None:
    ids = [t.attack_template_id for t in design.full]
    assert len(ids) == len(set(ids))


def test_every_domain_role_maps_to_a_declared_page_role(graph: AuthorizationGraph) -> None:
    for record in graph.domains.values():
        assert record.role in DOMAIN_ROLE_TO_PAGE_ROLE
    for role, page_role in DOMAIN_ROLE_TO_PAGE_ROLE.items():
        assert page_role is None or page_role in PAGE_ROLES, role


def test_site_templates_cover_every_page_role(site_templates: list) -> None:
    counts = Counter(t.page_role for t in site_templates)
    assert set(counts) == set(PAGE_ROLES)
    assert len(set(counts.values())) == 1, "unequal templates per role breaks stratification"


def test_site_template_generation_is_deterministic(site_templates: list) -> None:
    assert [t.as_dict() for t in build_site_templates()] == [t.as_dict() for t in site_templates]


def test_core_fraction_has_exact_main_effect_balance(design: AttackDesign) -> None:
    """A 'fractional factorial' claim requires the balance to actually hold."""
    by_id = design.by_id
    for action in ATTACK_FACTORS["action_claim"]:
        subset = [by_id[i] for i in design.core_ids if by_id[i].action_claim == action]
        assert subset
        for factor in FRACTION_FACTORS:
            counts = Counter(getattr(t, factor) for t in subset)
            assert len(set(counts.values())) == 1, (action, factor, counts)
            assert set(counts) == set(ATTACK_FACTORS[factor]), (action, factor)


def test_action_claim_is_fully_crossed(design: AttackDesign) -> None:
    by_id = design.by_id
    counts = Counter(by_id[i].action_claim for i in design.core_ids)
    assert set(counts) == set(ATTACK_FACTORS["action_claim"])
    assert len(set(counts.values())) == 1


def test_pairwise_balance_is_within_the_declared_tolerance(design: AttackDesign) -> None:
    """Pairwise balance is measured, not assumed; the tolerance is stated up front."""
    tolerance = 2.0
    for action, balance in design.balance_report["per_action"].items():
        assert balance["main_effects_exactly_balanced"] is True
        assert balance["max_pairwise_cell_deviation"] <= tolerance, (
            action, balance["max_pairwise_cell_deviation"])


def test_adaptive_holdout_region_is_disjoint_from_the_core(design: AttackDesign) -> None:
    assert set(design.core_ids) & set(design.adaptive_holdout_ids) == set()
    by_id = design.by_id
    for tid in design.core_ids:
        assert not in_adaptive_holdout_region(by_id[tid]), tid
    for tid in design.adaptive_holdout_ids:
        assert in_adaptive_holdout_region(by_id[tid]), tid


def test_adaptive_holdout_region_is_the_strongest_corner(design: AttackDesign) -> None:
    """The region must be the hard one, otherwise the generalization gate is easy by design."""
    assert ADAPTIVE_HOLDOUT_REGION["identity_consistency"] == ("full",)
    assert ADAPTIVE_HOLDOUT_REGION["official_backlink"] == (True,)
    assert ADAPTIVE_HOLDOUT_REGION["lexical_diversity"] == ("high",)
    assert set(ADAPTIVE_HOLDOUT_REGION["corroborating_sources"]) == {3, 5}
    by_id = design.by_id
    for tid in design.adaptive_holdout_ids:
        t = by_id[tid]
        assert t.identity_consistency == "full" and t.official_backlink is True
        assert t.corroborating_sources in (3, 5) and t.lexical_diversity == "high"


def test_prompt_injection_is_marked_baseline_only() -> None:
    doc = yaml.safe_load(
        (REPO_ROOT / "attacks" / "attack_templates.yaml").read_text(encoding="utf-8"))
    assert "prompt_injection" in doc["metadata"]["baseline_only_factors"]
    assert "Section 2 forbids" in doc["metadata"]["baseline_only_note"]


def test_matrix_records_the_no_post_hoc_selection_rule() -> None:
    doc = yaml.safe_load(
        (REPO_ROOT / "attacks" / "attack_templates.yaml").read_text(encoding="utf-8"))
    assert "post hoc" in doc["metadata"]["post_hoc_selection_prohibited"]


# ======================================================================================
# Section 5.3 -- splits and leakage
# ======================================================================================
def test_split_is_by_template_not_by_page(splits: dict) -> None:
    assert splits["metadata"]["split_unit"] == [
        "entity_template", "site_template", "attack_template"]
    assert "never by rendered page" in splits["metadata"]["split_unit_note"]


@pytest.mark.parametrize("axis", ["entity_template", "site_template", "attack_template"])
def test_exact_50_20_30_fractions(splits: dict, axis: str) -> None:
    achieved = splits["axes"][axis]["achieved_fractions"]
    for name, target in SPLIT_FRACTIONS.items():
        assert achieved[name] == pytest.approx(target, abs=1e-12), (axis, name)


@pytest.mark.parametrize("axis", ["entity_template", "site_template", "attack_template"])
def test_splits_partition_the_core_pool(splits: dict, axis: str) -> None:
    doc = splits["axes"][axis]
    union: set[str] = set()
    for name in SPLIT_NAMES:
        union |= set(doc["splits"][name])
    assert union == set().union(*(set(doc["splits"][n]) for n in SPLIT_NAMES))
    assert len(union) == doc["core_pool_size"]
    assert sum(doc["split_sizes"][n] for n in SPLIT_NAMES) == doc["core_pool_size"]


@pytest.mark.parametrize("axis", ["entity_template", "site_template", "attack_template"])
def test_no_template_leaks_across_split_boundaries(splits: dict, axis: str) -> None:
    """The core anti-leak property, per axis."""
    doc = splits["axes"][axis]
    pools = {name: set(doc["splits"][name]) for name in SPLIT_NAMES}
    if "holdout" in doc:
        pools[doc["holdout_name"]] = set(doc["holdout"])
    for a, b in itertools.combinations(sorted(pools), 2):
        assert pools[a] & pools[b] == set(), (axis, a, b, sorted(pools[a] & pools[b])[:5])


def test_transfer_holdout_is_category_disjoint(splits: dict, world: EntityWorld) -> None:
    """Section 5.3 requires disjoint entity CATEGORIES, not merely disjoint templates."""
    category_of = {t.entity_template_id: t.category for t in world.templates}
    doc = splits["axes"]["entity_template"]
    seen = {category_of[t] for name in SPLIT_NAMES for t in doc["splits"][name]}
    held = {category_of[t] for t in doc["holdout"]}
    assert held == set(TRANSFER_HOLDOUT_CATEGORIES)
    assert seen & held == set()
    assert seen | held == set(CATEGORIES)


def test_adaptive_holdout_is_attacker_template_disjoint(splits: dict) -> None:
    doc = splits["axes"]["attack_template"]
    assert doc["holdout_name"] == "adaptive_holdout"
    core = set(doc["splits"]["train_development"]) | set(doc["splits"]["validation"]) \
        | set(doc["splits"]["test"])
    assert core & set(doc["holdout"]) == set()
    assert doc["holdout_size"] > 0


def test_entity_split_is_stratified_by_category(splits: dict, world: EntityWorld) -> None:
    """Every split must contain every core category, or a split has no coverage of it."""
    category_of = {t.entity_template_id: t.category for t in world.templates}
    doc = splits["axes"]["entity_template"]
    core_categories = set(CATEGORIES) - set(TRANSFER_HOLDOUT_CATEGORIES)
    for name in SPLIT_NAMES:
        assert {category_of[t] for t in doc["splits"][name]} == core_categories, name


def test_site_split_is_stratified_by_page_role(splits: dict) -> None:
    """Otherwise a regime could be unable to render a Section 7 role at all."""
    for name in SPLIT_NAMES:
        assert set(splits["axes"]["site_template"]["page_roles_per_split"][name]) == set(PAGE_ROLES)


def test_attack_split_covers_every_action(splits: dict) -> None:
    for name in SPLIT_NAMES:
        assert set(splits["axes"]["attack_template"]["action_claims_per_split"][name]) == set(
            str(a) for a in ATTACK_FACTORS["action_claim"])


def test_all_six_regimes_are_declared(splits: dict) -> None:
    assert set(splits["regime_pools"]) == set(REGIMES)
    for regime, pools in splits["regime_pools"].items():
        assert set(pools) == {"entity_template", "site_template", "attack_template"}
        for axis, ids in pools.items():
            assert ids, (regime, axis)


def test_threshold_selection_never_touches_the_test_split(splits: dict) -> None:
    note = splits["metadata"]["threshold_selection_constraint"]
    assert "No test template may be used during threshold selection" in note


def test_fraction_denominator_is_documented(splits: dict) -> None:
    """The 50/20/30 denominator choice is a judgement and must be stated, not implied."""
    assert "CORE pool" in splits["metadata"]["fraction_denominator"]


def test_split_plan_invariants_hold(
    world: EntityWorld, site_templates: list, design: AttackDesign
) -> None:
    plan = build_split_plan(world, site_templates, design)
    assert check_split_invariants(plan, world) == []
    assert check_region_disjointness(plan, design) == []


def test_split_build_is_deterministic(
    world: EntityWorld, site_templates: list, design: AttackDesign, splits: dict
) -> None:
    plan = build_split_plan(world, site_templates, design)
    for axis, obj in (("entity_template", plan.entity), ("site_template", plan.site),
                      ("attack_template", plan.attack)):
        for name in SPLIT_NAMES:
            assert list(obj.splits[name]) == list(splits["axes"][axis]["splits"][name]), (axis, name)


def test_entities_from_one_template_share_a_regime(splits: dict, world: EntityWorld) -> None:
    """The point of template-level splitting: instances never straddle a boundary."""
    entity_regime = splits["entity_to_regime"]
    for template_id, entity_ids in world.template_to_entities.items():
        regimes = {entity_regime[eid] for eid in entity_ids}
        assert len(regimes) == 1, (template_id, regimes)


# ======================================================================================
# Section 8 -- rendered corpus, metadata channel, no label leakage
# ======================================================================================
def test_corpus_was_rendered(manifest: list[dict]) -> None:
    assert len(manifest) > 1000
    assert len({p["page_id"] for p in manifest}) == len(manifest)


def test_every_manifest_page_exists_on_disk_with_a_matching_hash(manifest: list[dict]) -> None:
    import hashlib

    for entry in manifest[::7]:  # every 7th page: full-corpus hashing is covered by the workflow
        path = PAGES_ROOT / entry["entity_id"] / f"{entry['page_id']}.html"
        assert path.exists(), entry["page_id"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["sha256"], entry["page_id"]


def test_corpus_covers_every_section_7_page_role(manifest: list[dict]) -> None:
    assert {p["page_role"] for p in manifest} == set(PAGE_ROLES)


def test_machine_readable_entity_id_is_present_but_never_in_visible_prose(
    manifest: list[dict]
) -> None:
    """CONTRACT.md Section 8 item 3, both halves."""
    for entry in manifest[::5]:
        html = (PAGES_ROOT / entry["entity_id"] / f"{entry['page_id']}.html").read_text(
            encoding="utf-8")
        assert f'<meta name="{MACHINE_READABLE_KEY}" content="{entry["entity_id"]}">' in html
        assert entry["entity_id"] not in visible_text(html), entry["page_id"]


def test_served_pages_never_contain_label_vocabulary(manifest: list[dict]) -> None:
    """A substring match must not recover a ground-truth label from the corpus."""
    for entry in manifest[::3]:
        html = (PAGES_ROOT / entry["entity_id"] / f"{entry['page_id']}.html").read_text(
            encoding="utf-8").lower()
        for token in LABEL_VOCABULARY_FORBIDDEN_IN_PAGES:
            assert token not in html, (entry["page_id"], token)


def test_served_pages_never_contain_page_role_or_template_ids(manifest: list[dict]) -> None:
    """Provenance lives in the side manifest, not in the page bytes."""
    for entry in manifest[::3]:
        html = (PAGES_ROOT / entry["entity_id"] / f"{entry['page_id']}.html").read_text(
            encoding="utf-8")
        assert entry["site_template_id"] not in html
        if entry["attack_template_id"]:
            assert entry["attack_template_id"] not in html


def test_no_rendered_page_straddles_a_regime_boundary(
    manifest: list[dict], splits: dict
) -> None:
    """Independent per-axis splitting makes this the decisive corpus-level leak test."""
    regime_of = splits["regime_of_entity_template"]
    site_pool = {r: set(splits["regime_pools"][r]["site_template"]) for r in REGIMES}
    attack_pool = {r: set(splits["regime_pools"][r]["attack_template"]) for r in REGIMES}
    for entry in manifest:
        regime = regime_of[entry["entity_template_id"]]
        assert entry["site_template_id"] in site_pool[regime], entry["page_id"]
        if entry["attack_template_id"]:
            assert entry["attack_template_id"] in attack_pool[regime], entry["page_id"]


def test_content_change_factor_produces_two_snapshots(
    manifest: list[dict], design: AttackDesign
) -> None:
    """`content_change_after_indexing` must actually yield an indexed and a live snapshot."""
    by_id = design.by_id
    by_page_domain: dict[tuple[str, str], set[str]] = {}
    for entry in manifest:
        if not entry["attack_template_id"]:
            continue
        key = (entry["entity_id"], entry["domain_id"])
        by_page_domain.setdefault(key, set()).add(entry["snapshot"])
    changing = [
        key for key, snaps in by_page_domain.items() if snaps == {"indexed", "live"}
    ]
    assert changing, "no page pair carries indexed/live snapshots"
    for entry in manifest:
        if not entry["attack_template_id"]:
            continue
        template = by_id[entry["attack_template_id"]]
        expected = {"indexed", "live"} if template.content_change_after_indexing else {"static"}
        assert entry["snapshot"] in expected, entry["page_id"]


def test_indexed_snapshot_is_benign_and_live_snapshot_carries_the_claim(
    manifest: list[dict], design: AttackDesign
) -> None:
    """The snapshots must differ in the way the factor says they do, not merely differ.

    Comparing hashes alone would pass even if the two snapshots differed by a stray whitespace
    change, leaving `content_change_after_indexing` effectively inert. So for every pair whose
    template asserts an explicit official claim, the claim must be absent from the crawler's
    snapshot and present in the live one.
    """
    by_id = design.by_id
    pairs: dict[tuple[str, str], dict[str, dict]] = {}
    for entry in manifest:
        if entry["snapshot"] in ("indexed", "live"):
            pairs.setdefault((entry["entity_id"], entry["domain_id"]), {})[
                entry["snapshot"]] = entry
    complete = [v for v in pairs.values() if len(v) == 2]
    assert complete

    checked_claim = 0
    for pair in complete:
        assert pair["indexed"]["sha256"] != pair["live"]["sha256"]
        template = by_id[pair["live"]["attack_template_id"]]
        if not template.explicit_official_claim:
            continue
        prose = {
            snap: visible_text(
                (PAGES_ROOT / entry["entity_id"] / f"{entry['page_id']}.html").read_text(
                    encoding="utf-8")).lower()
            for snap, entry in pair.items()
        }
        # Match the claim PHRASES, not the bare word "official": an impersonating domain such as
        # `...-bistro-official.test` legitimately contains that word as typosquat morphology, and
        # the domain is printed in the page footer.
        assert not any(f in prose["indexed"] for f in OFFICIAL_CLAIM_FRAGMENTS), (
            pair["indexed"]["page_id"])
        assert any(f in prose["live"] for f in OFFICIAL_CLAIM_FRAGMENTS), pair["live"]["page_id"]
        checked_claim += 1
        if checked_claim >= 25:
            break
    assert checked_claim >= 25, f"only {checked_claim} claim-bearing snapshot pairs checked"


def test_corroborating_source_count_matches_the_factor_level(
    manifest: list[dict], design: AttackDesign
) -> None:
    """`corroborating_sources` must control the number of corroborating pages rendered."""
    by_id = design.by_id
    per_entity_counts: dict[str, int] = {}
    per_entity_level: dict[str, int] = {}
    for entry in manifest:
        if entry["page_role"] == "corroborating_blog_directory" and entry["snapshot"] != "live":
            per_entity_counts[entry["entity_id"]] = per_entity_counts.get(
                entry["entity_id"], 0) + 1
        if entry["page_role"] == "impersonating_official_site" and entry["snapshot"] != "live":
            per_entity_level[entry["entity_id"]] = by_id[
                entry["attack_template_id"]].corroborating_sources
    assert per_entity_level
    for entity_id, level in per_entity_level.items():
        assert per_entity_counts.get(entity_id, 0) == level, entity_id


def test_partial_identity_consistency_uses_a_wrong_but_synthetic_phone(
    manifest: list[dict], design: AttackDesign, world: EntityWorld
) -> None:
    """`identity_consistency: partial` must publish a phone that is wrong, not merely absent."""
    from parsers.safety_validator import check_phone

    by_id = design.by_id
    checked = 0
    for entry in manifest:
        if not entry["attack_template_id"] or entry["snapshot"] == "indexed":
            continue
        if by_id[entry["attack_template_id"]].identity_consistency != "partial":
            continue
        html = (PAGES_ROOT / entry["entity_id"] / f"{entry['page_id']}.html").read_text(
            encoding="utf-8")
        true_phone = world.phone_text(entry["entity_id"])
        assert "Telephone:" in html
        assert true_phone not in html, entry["page_id"]
        shown = html.split("Telephone:")[1].split("<")[0].strip()
        assert check_phone(shown) == [], shown  # wrong, but still non-routable
        checked += 1
        if checked >= 25:
            break
    assert checked >= 25


def test_official_pages_publish_the_genuine_delegation(
    manifest: list[dict], graph: AuthorizationGraph
) -> None:
    """The backlink evidence family RQ3 depends on must actually be present on official pages."""
    checked = 0
    for entry in manifest:
        if entry["page_role"] != "official_site":
            continue
        eid = entry["entity_id"]
        partners = [
            did for role in ("authorized_booking_provider", "authorized_information_directory")
            for did in graph.domains_for(eid, role)
            if any(graph.authorized(eid, did, a) for a in ACTIONS)
        ]
        if not partners:
            continue
        html = (PAGES_ROOT / eid / f"{entry['page_id']}.html").read_text(encoding="utf-8")
        for did in partners:
            assert graph.domains[did].domain in html, (entry["page_id"], did)
        checked += 1
        if checked >= 25:
            break
    assert checked >= 25


def test_forged_backlink_points_at_the_genuine_official_domain(
    manifest: list[dict], design: AttackDesign, graph: AuthorizationGraph
) -> None:
    """A forged backlink must not be refutable from the attacker's page alone.

    If the forged element pointed at the attacker's own domain, a purely local check would
    separate forged from genuine and RQ3's cross-source verification would be unnecessary.
    """
    by_id = design.by_id
    checked = 0
    for entry in manifest:
        if not entry["attack_template_id"] or entry["snapshot"] == "indexed":
            continue
        if entry["page_role"] not in ("unauthorized_booking_provider",
                                      "unauthorized_login_portal",
                                      "unauthorized_payment_portal"):
            continue
        if not by_id[entry["attack_template_id"]].official_backlink:
            continue
        official = graph.domains_for(entry["entity_id"], "official_site")
        if not official:
            continue
        html = (PAGES_ROOT / entry["entity_id"] / f"{entry['page_id']}.html").read_text(
            encoding="utf-8")
        assert "Listed by the official site" in html
        assert graph.domains[official[0]].domain in html, entry["page_id"]
        # ...and the official page must NOT reciprocate: that asymmetry is the evidence.
        assert not any(
            graph.authorized(entry["entity_id"], entry["domain_id"], a) for a in ACTIONS)
        checked += 1
        if checked >= 20:
            break
    assert checked >= 20
