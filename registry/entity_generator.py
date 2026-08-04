"""Controlled entity template generator (CONTRACT.md Sections 5.1, 8, 16).

CONTRACT.md Section 5.1 fixes the entity record schema:

    entity_id: E0001
    canonical_name: "Harbour Lantern Bistro"
    category: restaurant
    address_id: A0001
    phone_id: P0001
    coordinates_id: C0001
    status: active

Those seven fields are emitted verbatim and nothing else is added to the record. The literal
address, phone and coordinate values live in resolvable side tables keyed by the ID
references. The indirection is load-bearing rather than decorative: RQ3 requires
identity-field-consistency evidence, so an attack page at ``identity_consistency: partial``
must be able to quote a correct name and address while quoting a *wrong* phone. Separate ID
spaces make that expressible and machine-checkable instead of a matter of prose.

Two structural choices deserve emphasis:

* **Template -> instance relation.** A template fixes the category, the suffix and the leading
  word; each of its two entities draws a different second word. The two entities of one
  template are therefore lexically confusable siblings ("Harbour Lantern Bistro" and "Harbour
  Quay Bistro"), and because splitting is by template they can never straddle a split
  boundary. The benign-confusion condition of Section 7 is thus a structural property of the
  design rather than an extra pass.

* **Per-key deterministic derivation.** Randomness comes from SHA-256 over a namespaced key,
  not from a sequential RNG. Adding or removing an entity therefore cannot perturb any other
  entity's address, phone or coordinates, so regenerating after a change produces a minimal
  diff instead of a wholesale reshuffle.

Machine-readable entity IDs (Section 8, item 3) are the ``entity_id`` values themselves. The
site generator embeds them in a ``<meta>`` channel named by :data:`MACHINE_READABLE_KEY` and
never in visible prose; ``tests/test_benchmark_and_splits.py`` asserts both halves of that.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterator

import yaml

from registry.lexicon import (
    CATEGORIES,
    CATEGORY_SUFFIXES,
    LEXICON_VERSION,
    SYNTHETIC_CITIES,
    SYNTHETIC_STREETS,
    WORD_BANK_LEAD,
    WORD_BANK_SECOND,
)
from parsers.safety_validator import (
    COORDINATE_GRID_MAX,
    COORDINATE_SYSTEM,
    SafetyReport,
    assert_safe,
    check_coordinates,
    check_phone,
    check_postal_code,
    validate_text,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_DIR = REPO_ROOT / "registry"

GENERATOR_VERSION = "1.0"
MASTER_SEED = 42  # preregistration.yaml -> statistical_analysis_plan.random_seed

# --------------------------------------------------------------------------------------
# Scale. Chosen so that every mandated split ratio is exact rather than rounded:
#   8 categories x 10 templates = 80 entity templates
#   6 core categories x 10 = 60 core templates -> 30 / 12 / 18 = exactly 50 / 20 / 30
#   and per category 5 / 2 / 3, so the split is exactly stratified by category too.
# --------------------------------------------------------------------------------------
TEMPLATES_PER_CATEGORY = 10
ENTITIES_PER_TEMPLATE = 2  # the mutually confusable sibling pair
N_FABRICATED_CONTROLS = 8  # RQ1 condition 7: fabricated entity + fabricated domain

ENTITY_STATUSES = ("active", "fabricated_control")
MACHINE_READABLE_KEY = "x-aegis-entity-id"

# Fabricated controls use a visually distinct ID block so they cannot be silently pooled with
# real controlled entities in a later step. The format still matches the contract's E%04d.
FABRICATED_ID_BASE = 9000


# ======================================================================================
# Deterministic per-key derivation
# ======================================================================================
def derive_int(namespace: str, key: str, modulus: int, salt: int = MASTER_SEED) -> int:
    """Derive a stable integer in ``[0, modulus)`` from a namespaced key.

    Uses SHA-256 rather than a seeded RNG so that the value for one key is independent of how
    many other keys were drawn before it.
    """
    digest = hashlib.sha256(f"{salt}|{namespace}|{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


# ======================================================================================
# Records
# ======================================================================================
@dataclass(frozen=True)
class EntityTemplate:
    """A template from which entities are instantiated. The split unit for the entity axis."""

    entity_template_id: str
    category: str
    lead_word: str
    name_suffix: str
    morphology: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EntityRecord:
    """CONTRACT.md Section 5.1 record. Exactly seven fields, in contract order."""

    entity_id: str
    canonical_name: str
    category: str
    address_id: str
    phone_id: str
    coordinates_id: str
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "category": self.category,
            "address_id": self.address_id,
            "phone_id": self.phone_id,
            "coordinates_id": self.coordinates_id,
            "status": self.status,
        }


@dataclass
class EntityWorld:
    """Everything the entity layer produces."""

    templates: list[EntityTemplate]
    entities: list[EntityRecord]
    addresses: dict[str, dict[str, Any]]
    phones: dict[str, str]
    coordinates: dict[str, dict[str, Any]]
    aliases: dict[str, list[str]]
    entity_to_template: dict[str, str]
    template_to_entities: dict[str, list[str]]
    sibling_of: dict[str, str]

    def by_id(self, entity_id: str) -> EntityRecord:
        return self._index[entity_id]

    def __post_init__(self) -> None:
        self._index = {e.entity_id: e for e in self.entities}

    @property
    def controlled(self) -> list[EntityRecord]:
        return [e for e in self.entities if e.status == "active"]

    @property
    def fabricated(self) -> list[EntityRecord]:
        return [e for e in self.entities if e.status == "fabricated_control"]

    def address_text(self, entity_id: str) -> str:
        a = self.addresses[self.by_id(entity_id).address_id]
        return f"{a['street']}, {a['city']}, {a['region']} {a['postal_code']}"

    def phone_text(self, entity_id: str) -> str:
        return self.phones[self.by_id(entity_id).phone_id]


# ======================================================================================
# Generation
# ======================================================================================
def build_entity_templates() -> list[EntityTemplate]:
    """Build the 80 entity templates: 8 categories x 10 templates."""
    templates: list[EntityTemplate] = []
    for ci, category in enumerate(CATEGORIES):
        suffixes = CATEGORY_SUFFIXES[category]
        for ti in range(TEMPLATES_PER_CATEGORY):
            index = ci * TEMPLATES_PER_CATEGORY + ti
            lead = WORD_BANK_LEAD[(index * 7) % len(WORD_BANK_LEAD)]
            suffix = suffixes[ti % len(suffixes)]
            templates.append(EntityTemplate(
                entity_template_id=f"ET{index + 1:03d}",
                category=category,
                lead_word=lead,
                name_suffix=suffix,
                morphology="<lead_word> <second_word> <name_suffix>",
            ))
    return templates


def _instance_names(template: EntityTemplate, index: int) -> list[str]:
    """Pick distinct second words for a template's entities, keeping the shared lead word."""
    names: list[str] = []
    used: set[int] = set()
    for j in range(ENTITIES_PER_TEMPLATE):
        # Deterministic start, then linear probing so a collision cannot silently duplicate.
        slot = derive_int("second_word", f"{template.entity_template_id}|{j}", len(WORD_BANK_SECOND))
        while slot in used:
            slot = (slot + 1) % len(WORD_BANK_SECOND)
        used.add(slot)
        second = WORD_BANK_SECOND[slot]
        names.append(f"{template.lead_word} {second} {template.name_suffix}")
    # Template 0 / entity 0 must reproduce the CONTRACT.md Section 5.1 worked example.
    if index == 0:
        names[0] = f"{template.lead_word} Lantern {template.name_suffix}"
    return names


def _make_address(address_id: str, key: str) -> dict[str, Any]:
    street_no = 1 + derive_int("street_no", key, 240)
    street = SYNTHETIC_STREETS[derive_int("street", key, len(SYNTHETIC_STREETS))]
    city = SYNTHETIC_CITIES[derive_int("city", key, len(SYNTHETIC_CITIES))]
    postal = f"ZZ-{derive_int('postal', key, 10000):04d}"
    return {
        "address_id": address_id,
        "street": f"{street_no} {street}",
        "city": city,
        "region": "ZZ",
        "postal_code": postal,
        "country_note": "synthetic ISO-3166 user-assigned region ZZ; not a real jurisdiction",
    }


def _make_coordinates(coordinates_id: str, key: str) -> dict[str, Any]:
    return {
        "coordinates_id": coordinates_id,
        "grid_system": COORDINATE_SYSTEM,
        "grid_x": derive_int("grid_x", key, COORDINATE_GRID_MAX + 1),
        "grid_y": derive_int("grid_y", key, COORDINATE_GRID_MAX + 1),
        "units": "synthetic_metres",
        "note": "planar synthetic grid, deliberately not WGS84; cannot be reverse-geocoded",
    }


def _make_phone(key: str) -> str:
    return f"+999-555-0{derive_int('phone', key, 10000):04d}"


def _aliases(canonical: str, template: EntityTemplate, city: str) -> list[str]:
    """Generated aliases for entity-mention resolution (CONTRACT.md Section 8, item 3)."""
    short = canonical[: -(len(template.name_suffix) + 1)]
    out = [canonical, short, f"The {canonical}", f"{canonical}, {city}", f"{short} {city}"]
    seen, unique = set(), []
    for a in out:
        if a not in seen:
            seen.add(a)
            unique.append(a)
    return unique


def generate_entity_world() -> EntityWorld:
    """Generate templates, entities, side tables, aliases and the sibling map."""
    templates = build_entity_templates()
    entities: list[EntityRecord] = []
    addresses: dict[str, dict[str, Any]] = {}
    phones: dict[str, str] = {}
    coordinates: dict[str, dict[str, Any]] = {}
    aliases: dict[str, list[str]] = {}
    entity_to_template: dict[str, str] = {}
    template_to_entities: dict[str, list[str]] = {}
    sibling_of: dict[str, str] = {}

    used_names: set[str] = set()
    counter = 0
    for index, template in enumerate(templates):
        names = _instance_names(template, index)
        ids_for_template: list[str] = []
        for j, name in enumerate(names):
            # Uniqueness is asserted, not assumed: a duplicate canonical name would make
            # entity resolution ambiguous and silently corrupt every label.
            if name in used_names:
                raise ValueError(f"duplicate canonical name generated: {name!r}")
            used_names.add(name)

            counter += 1
            entity_id = f"E{counter:04d}"
            key = f"{template.entity_template_id}|{j}|{entity_id}"
            address_id, phone_id, coordinates_id = (
                f"A{counter:04d}", f"P{counter:04d}", f"C{counter:04d}",
            )
            addresses[address_id] = _make_address(address_id, key)
            phones[phone_id] = _make_phone(key)
            coordinates[coordinates_id] = _make_coordinates(coordinates_id, key)

            entities.append(EntityRecord(
                entity_id=entity_id,
                canonical_name=name,
                category=template.category,
                address_id=address_id,
                phone_id=phone_id,
                coordinates_id=coordinates_id,
                status="active",
            ))
            aliases[entity_id] = _aliases(name, template, addresses[address_id]["city"])
            entity_to_template[entity_id] = template.entity_template_id
            ids_for_template.append(entity_id)

        template_to_entities[template.entity_template_id] = ids_for_template
        # The two entities of a template are each other's confusable sibling.
        if len(ids_for_template) == 2:
            sibling_of[ids_for_template[0]] = ids_for_template[1]
            sibling_of[ids_for_template[1]] = ids_for_template[0]

    # ---- Fabricated controls (RQ1 condition 7). Separate template family and ID block. ----
    for k in range(N_FABRICATED_CONTROLS):
        category = CATEGORIES[k % len(CATEGORIES)]
        template_id = f"FT{k + 1:03d}"
        lead = WORD_BANK_LEAD[(k * 5 + 2) % len(WORD_BANK_LEAD)]
        second = WORD_BANK_SECOND[(k * 9 + 1) % len(WORD_BANK_SECOND)]
        suffix = CATEGORY_SUFFIXES[category][k % 5]
        name = f"{lead} {second} {suffix}"
        while name in used_names:  # keep canonical names globally unique
            second = WORD_BANK_SECOND[(WORD_BANK_SECOND.index(second) + 1) % len(WORD_BANK_SECOND)]
            name = f"{lead} {second} {suffix}"
        used_names.add(name)

        entity_id = f"E{FABRICATED_ID_BASE + k + 1:04d}"
        key = f"{template_id}|0|{entity_id}"
        suffix_num = FABRICATED_ID_BASE + k + 1
        address_id, phone_id, coordinates_id = (
            f"A{suffix_num:04d}", f"P{suffix_num:04d}", f"C{suffix_num:04d}",
        )
        addresses[address_id] = _make_address(address_id, key)
        phones[phone_id] = _make_phone(key)
        coordinates[coordinates_id] = _make_coordinates(coordinates_id, key)
        entities.append(EntityRecord(
            entity_id=entity_id,
            canonical_name=name,
            category=category,
            address_id=address_id,
            phone_id=phone_id,
            coordinates_id=coordinates_id,
            status="fabricated_control",
        ))
        aliases[entity_id] = _aliases(name, EntityTemplate(
            template_id, category, lead, suffix, ""), addresses[address_id]["city"])
        entity_to_template[entity_id] = template_id
        template_to_entities[template_id] = [entity_id]
        templates.append(EntityTemplate(
            entity_template_id=template_id,
            category=category,
            lead_word=lead,
            name_suffix=suffix,
            morphology="fabricated_control",
        ))

    return EntityWorld(
        templates=templates,
        entities=entities,
        addresses=addresses,
        phones=phones,
        coordinates=coordinates,
        aliases=aliases,
        entity_to_template=entity_to_template,
        template_to_entities=template_to_entities,
        sibling_of=sibling_of,
    )


# ======================================================================================
# Safety gate (CONTRACT.md Section 16) -- runs before anything is written to disk
# ======================================================================================
def safety_check_world(world: EntityWorld) -> SafetyReport:
    """Validate every generated identity field. Fail-closed before persisting."""
    report = SafetyReport()
    report.scanned.append("entity_world(in-memory)")
    for entity in world.entities:
        loc = entity.entity_id
        report.extend(validate_text(entity.canonical_name, f"{loc}.canonical_name"))
        report.extend(check_phone(world.phones[entity.phone_id], f"{loc}.phone"))
        report.extend(check_postal_code(
            world.addresses[entity.address_id]["postal_code"], f"{loc}.postal_code"))
        report.extend(check_coordinates(
            world.coordinates[entity.coordinates_id], f"{loc}.coordinates"))
        report.extend(validate_text(world.address_text(entity.entity_id), f"{loc}.address"))
        for alias in world.aliases[entity.entity_id]:
            report.extend(validate_text(alias, f"{loc}.alias"))
    return report


# ======================================================================================
# Persistence
# ======================================================================================
def _dump(path: Path, payload: Any) -> None:
    text = yaml.dump(payload, sort_keys=False, allow_unicode=True, width=100,
                     Dumper=getattr(yaml, "CSafeDumper", yaml.SafeDumper))
    path.write_text(text, encoding="utf-8")


def write_entity_world(world: EntityWorld, out_dir: Path = REGISTRY_DIR) -> dict[str, Path]:
    """Write the entity registry and side tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": "1.0",
        "generator": "registry/entity_generator.py",
        "generator_version": GENERATOR_VERSION,
        "lexicon_version": LEXICON_VERSION,
        "master_seed": MASTER_SEED,
        "contract_ref": "CONTRACT.md Section 5.1",
        "machine_readable_entity_id_channel": MACHINE_READABLE_KEY,
        "machine_readable_note": (
            "Entity IDs are published to parsers through an HTML <meta> channel and never "
            "appear in visible prose (CONTRACT.md Section 8, item 3). The channel carries the "
            "target entity ID and an opaque page ID only; page role, attack template and "
            "factor levels are held in the side manifest so the channel cannot leak a label."
        ),
        "status_enum": list(ENTITY_STATUSES),
        "counts": {
            "entity_templates": len([t for t in world.templates if t.morphology != "fabricated_control"]),
            "fabricated_control_templates": len([t for t in world.templates if t.morphology == "fabricated_control"]),
            "controlled_entities": len(world.controlled),
            "fabricated_control_entities": len(world.fabricated),
        },
    }
    paths = {
        "entities": out_dir / "entities.yaml",
        "entity_templates": out_dir / "entity_templates.yaml",
        "addresses": out_dir / "addresses.yaml",
        "phones": out_dir / "phones.yaml",
        "coordinates": out_dir / "coordinates.yaml",
        "aliases": out_dir / "aliases.yaml",
    }
    _dump(paths["entities"], {"metadata": meta, "entities": [e.as_dict() for e in world.entities]})
    _dump(paths["entity_templates"], {
        "metadata": {**meta, "split_unit": "entity_template"},
        "entity_templates": [t.as_dict() for t in world.templates],
        "template_to_entities": world.template_to_entities,
        "confusable_sibling_of": world.sibling_of,
    })
    _dump(paths["addresses"], {"metadata": meta, "addresses": world.addresses})
    _dump(paths["phones"], {
        "metadata": {**meta, "numbering_plan": "+999-555-0NNNN (ITU country code 999 unassigned)"},
        "phones": world.phones,
    })
    _dump(paths["coordinates"], {"metadata": meta, "coordinates": world.coordinates})
    _dump(paths["aliases"], {"metadata": meta, "aliases": world.aliases})
    return paths


def load_entity_world(out_dir: Path = REGISTRY_DIR) -> EntityWorld:
    """Reload a previously written entity world (used by tests and later steps)."""
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

    def read(name: str) -> dict[str, Any]:
        return yaml.load((out_dir / name).read_text(encoding="utf-8"), Loader=loader)

    entities_doc = read("entities.yaml")
    templates_doc = read("entity_templates.yaml")
    entity_to_template = {
        eid: tid
        for tid, eids in templates_doc["template_to_entities"].items()
        for eid in eids
    }
    return EntityWorld(
        templates=[EntityTemplate(**t) for t in templates_doc["entity_templates"]],
        entities=[EntityRecord(**e) for e in entities_doc["entities"]],
        addresses=read("addresses.yaml")["addresses"],
        phones=read("phones.yaml")["phones"],
        coordinates=read("coordinates.yaml")["coordinates"],
        aliases=read("aliases.yaml")["aliases"],
        entity_to_template=entity_to_template,
        template_to_entities=templates_doc["template_to_entities"],
        sibling_of=templates_doc["confusable_sibling_of"],
    )


def iter_categories(world: EntityWorld) -> Iterator[tuple[str, list[str]]]:
    """Yield (category, entity_template_ids) for the controlled templates."""
    by_category: dict[str, list[str]] = {}
    for template in world.templates:
        if template.morphology == "fabricated_control":
            continue
        by_category.setdefault(template.category, []).append(template.entity_template_id)
    for category in CATEGORIES:
        yield category, by_category.get(category, [])


def main() -> int:  # pragma: no cover - CLI
    world = generate_entity_world()
    assert_safe(safety_check_world(world))
    paths = write_entity_world(world)
    print(f"entities={len(world.controlled)} fabricated_controls={len(world.fabricated)} "
          f"templates={len(world.templates)}")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
