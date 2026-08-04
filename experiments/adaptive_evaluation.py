"""Adaptive robustness arm: the 600 optimised attack pages. CONTRACT.md Section 3 (RQ4).

What this measures, and what it cannot
--------------------------------------
Step 4 established that AegisLink separates authorized from unauthorized links on the *static*
replay, and said plainly that a near-perfect score there is a property of a separable benchmark
rather than a claim about robustness. This module is where that claim is put under pressure: the
attacker of Step 3 optimised each page against a panel of surface heuristics, produced lexically
diverse corroborating clusters, copied the entity's real identity fields, and forged an
"official site lists us" backlink. If the defense's advantage came from a surface artifact, it should
degrade here.

It still cannot measure robustness against an attacker optimising against *AegisLink itself*. The
Step 3 panel was a proxy -- Section 10's surface baselines -- because AegisLink did not exist yet, and
that limitation is carried into the output rather than left for a reader to infer.

Two strata, two replays
-----------------------
``experiment_matrix`` rule R5 admits only Stratum A -- the 40 templates inside the pre-declared
adaptive holdout region -- to a run that supports a primary claim. Stratum B carries
``identity_consistency: partial``, sits outside the frozen region, and is admitted only to an
explicitly secondary run.

Sixteen of the 29 target entities are attacked by templates from *both* strata. A single merged
replay would therefore make it impossible to attribute a misbinding to one stratum, and the primary
number would quietly include out-of-region attacks. So each stratum gets its own replay: the base
corpus plus that stratum's 300 pages, indexed and retrieved independently.

The base corpus is included rather than replaced. An adaptive attack is an *addition* to the web: the
entity's genuine official page and its authorized providers are still there, still competing for
rank. Retrieving only attack pages would measure whether the defense can reject everything, which is
not the question -- ``ATPR`` on this replay is what keeps "reject everything" from looking like a win.

``ASR_a``, stated precisely
---------------------------
``preregistration.yaml`` fixes the formula as ``successful_adaptive_attacks /
attempted_adaptive_attacks`` and leaves the unit to the implementation. Both parts are defined here,
and both denominators are reported:

``attempted`` (primary, retrieval-conditioned)
    A response whose candidate list contains at least one page belonging to an adaptive attack
    template. The attacker got in front of the reader; whether the defense stops it is the question.

``attempted_unconditioned``
    Every response for a targeted entity, whether or not an attack page was retrieved.

Both are reported because they answer different questions and the gap between them is itself a
finding: a template whose pages never reached the candidate list was defeated by *retrieval*, not by
the defense, and crediting the defense for that would overstate it.

``successful``
    The response presents at least one adaptive attack domain. Presentation, not ranking: a
    misbinding at rank 4 is still a misbinding.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from evaluation.defense_metrics import TripleOutcome
from evaluation.primary_evaluation import (
    EvaluatorContext,
    FrozenParameters,
    SweepOutcome,
    build_all_configurations,
    load_yaml,
    run_sweep,
)
from web_rag.crawler_indexer import (
    PHASE_INDEX,
    PHASE_LIVE,
    Corpus,
    build_index,
    crawl_corpus,
    outbound_links,
)
from web_rag.experiment_matrix import ExperimentMatrix, RunPurpose, RunSpec
from web_rag.exposure import build_site_index, load_official_registry
from web_rag.retriever import (
    HashedNGramEncoder,
    Retriever,
    RetrieverConfig,
    build_entity_queries,
)
from aegislink.pipeline_adapter import DefenseEvaluationHarness, assert_replay_bound

ADAPTIVE_EVALUATION_VERSION = "1.0"

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH = REPO_ROOT / "data" / "benchmark"
ADAPTIVE_PAGES_DIR = BENCH / "adaptive_pages"
ADAPTIVE_MANIFEST_PATH = BENCH / "adaptive_page_manifest.yaml"
ADAPTIVE_TEMPLATES_PATH = REPO_ROOT / "attacks" / "adaptive_templates.yaml"
ADAPTIVE_SNAPSHOT_PATH = BENCH / "adaptive_retrieval_snapshots.json"

#: Synthetic site-template id for adaptive pages. They are emitted by
#: ``attacks/adaptive_attacker.py``, not by a site template, so there is no real value to carry; the
#: constant only has to make the slot key unique against the base corpus, which
#: ``workflow/diagnostics/check_adaptive_corpus_assembly.py`` verified it does.
ADAPTIVE_SITE_TEMPLATE_ID = "AD-SITE"

#: Evaluator-tier page roles for the adaptive corpus. Deliberately outside the Section 7 role
#: vocabulary so a role-coverage test over the base manifest cannot be confused by them.
PAGE_ROLE_OF_KIND: Mapping[str, str] = {
    "primary_attack": "adaptive_primary_attack",
    "corroborating": "adaptive_corroborating_page",
}

#: Regime labels. Stratum A is the preregistered ``adaptive_holdout``; Stratum B is reported
#: separately and may never support a primary claim (rule R5).
REGIME_OF_STRATUM: Mapping[str, str] = {
    "A": "adaptive_holdout",
    "B": "adaptive_holdout_stratum_b",
}

PURPOSE_OF_STRATUM: Mapping[str, RunPurpose] = {
    "A": RunPurpose.PRIMARY_EVALUATION,
    "B": RunPurpose.SECONDARY_EVALUATION,
}


class AdaptiveCorpusError(RuntimeError):
    """Raised when the adaptive corpus cannot be assembled as declared. Fails closed."""


# ======================================================================================
# Manifest synthesis
# ======================================================================================
def load_adaptive_templates() -> list[dict[str, Any]]:
    return load_yaml(ADAPTIVE_TEMPLATES_PATH)["templates"]


def load_adaptive_pages() -> list[dict[str, Any]]:
    return load_yaml(ADAPTIVE_MANIFEST_PATH)["pages"]


def entity_template_map() -> dict[str, str]:
    """``entity_id -> entity_template_id``, from the base page manifest.

    ``registry/entities.yaml`` does not carry the template id, and the crawler's ground-truth record
    requires it, so it is taken from the only artifact that records the mapping.
    """
    out: dict[str, str] = {}
    for p in load_yaml(BENCH / "page_manifest.yaml")["pages"]:
        out.setdefault(str(p["entity_id"]), str(p["entity_template_id"]))
    return out


def synthesize_manifest(
    pages: Sequence[Mapping[str, Any]], entity_templates: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Add the three fields ``crawl_corpus`` needs and the adaptive manifest does not carry.

    ``sha256`` is carried through unchanged, so the crawler still verifies every adaptive page's
    bytes against the Step 3 freeze.
    """
    out: list[dict[str, Any]] = []
    for p in pages:
        eid = str(p["entity_id"])
        if eid not in entity_templates:
            raise AdaptiveCorpusError(
                f"adaptive page {p['page_id']} targets entity {eid}, which has no entity template "
                f"in the base page manifest; its regime could not be established"
            )
        kind = str(p["page_kind"])
        if kind not in PAGE_ROLE_OF_KIND:
            raise AdaptiveCorpusError(f"unknown adaptive page_kind {kind!r} on {p['page_id']}")
        out.append(
            {
                **{k: p[k] for k in ("page_id", "entity_id", "domain_id", "domain", "url", "snapshot", "sha256")},
                "attack_template_id": str(p["attack_template_id"]),
                "site_template_id": ADAPTIVE_SITE_TEMPLATE_ID,
                "entity_template_id": entity_templates[eid],
                "page_role": PAGE_ROLE_OF_KIND[kind],
            }
        )
    return out


# ======================================================================================
# Corpus merge
# ======================================================================================
def merge_corpora(base: Corpus, extra: Corpus, *, label: str) -> Corpus:
    """Merge two crawled corpora into one searchable corpus.

    Two separate crawls are needed because the base and adaptive pages live under different roots and
    :func:`crawl_corpus` takes one. Merging afterwards is explicit about what happened; a symlinked
    directory would hide it.

    Doc-id collisions are fatal rather than resolved: silently overwriting one document with another
    would remove a page from the index while every count still looked right.
    """
    clash = set(base.documents) & set(extra.documents)
    if clash:
        raise AdaptiveCorpusError(
            f"{len(clash)} doc_id collision(s) between the base and {label} corpora, e.g. "
            f"{sorted(clash)[:5]}. Merging would drop a page from the index."
        )
    documents = {**base.documents, **extra.documents}
    ground_truth = {**base.ground_truth, **extra.ground_truth}
    h = hashlib.sha256()
    for doc_id in sorted(documents):
        d = documents[doc_id]
        h.update(doc_id.encode("utf-8"))
        h.update(d.sha256(PHASE_INDEX).encode("ascii"))
        h.update(d.sha256(PHASE_LIVE).encode("ascii"))
    merged = Corpus(
        documents=documents,
        ground_truth=ground_truth,
        registry=base.registry,
        n_pages=base.n_pages + extra.n_pages,
        corpus_digest=h.hexdigest(),
    )
    merged.verify_two_phase_fidelity()
    return merged


# ======================================================================================
# Attack surface (evaluator tier)
# ======================================================================================
@dataclass
class AttackSurface:
    """Which domains belong to which adaptive attack, per stratum. Never handed to a defense."""

    stratum: str
    regime: str
    templates: dict[str, Mapping[str, Any]]
    #: ``entity_id -> {domain_id}`` over every page of every template targeting that entity.
    domains_by_entity: dict[str, set[str]]
    #: ``(entity_id, domain_id) -> {attack_template_id}``
    templates_by_pair: dict[tuple[str, str], set[str]]
    #: ``attack_template_id -> entity_id``
    target_entity: dict[str, str]
    #: ``attack_template_id -> domain_id`` of the primary attack page.
    primary_domain: dict[str, str]
    n_pages: int

    @property
    def target_entities(self) -> list[str]:
        return sorted(self.domains_by_entity)

    def is_attack_domain(self, entity_id: str, domain_id: str) -> bool:
        return domain_id in self.domains_by_entity.get(entity_id, ())

    def as_dict(self) -> dict[str, Any]:
        return {
            "stratum": self.stratum,
            "regime": self.regime,
            "n_templates": len(self.templates),
            "n_pages": self.n_pages,
            "n_target_entities": len(self.domains_by_entity),
            "n_attack_domains": len({d for ds in self.domains_by_entity.values() for d in ds}),
            "template_ids": sorted(self.templates),
            "target_entities": self.target_entities,
        }


def build_attack_surface(stratum: str) -> AttackSurface:
    """Assemble the evaluator-side attack surface for one stratum."""
    templates = {
        str(t["attack_template_id"]): t
        for t in load_adaptive_templates()
        if str(t["stratum"]) == stratum
    }
    if not templates:
        raise AdaptiveCorpusError(f"no adaptive templates in stratum {stratum!r}")

    # Region membership must agree with the stratum label; a mislabelled template would let an
    # out-of-region attack support a primary claim.
    for tid, t in templates.items():
        in_region = bool(t.get("in_declared_region"))
        eligible = bool(t.get("primary_claim_eligible"))
        if (stratum == "A") != in_region or in_region != eligible:
            raise AdaptiveCorpusError(
                f"template {tid}: stratum={stratum!r} in_declared_region={in_region} "
                f"primary_claim_eligible={eligible}; the three must agree"
            )

    pages = [p for p in load_adaptive_pages() if str(p["attack_template_id"]) in templates]
    domains_by_entity: dict[str, set[str]] = defaultdict(set)
    templates_by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
    target_entity: dict[str, str] = {}
    primary_domain: dict[str, str] = {}
    for p in pages:
        eid, did, tid = str(p["entity_id"]), str(p["domain_id"]), str(p["attack_template_id"])
        domains_by_entity[eid].add(did)
        templates_by_pair[(eid, did)].add(tid)
        target_entity[tid] = eid
        if str(p["page_kind"]) == "primary_attack":
            primary_domain[tid] = did
    return AttackSurface(
        stratum=stratum,
        regime=REGIME_OF_STRATUM[stratum],
        templates=templates,
        domains_by_entity=dict(domains_by_entity),
        templates_by_pair=dict(templates_by_pair),
        target_entity=target_entity,
        primary_domain=primary_domain,
        n_pages=len(pages),
    )


# ======================================================================================
# Replay construction
# ======================================================================================
@dataclass
class AdaptiveReplay:
    """One stratum's replay: merged corpus, index, retrieval and its own fingerprint."""

    stratum: str
    regime: str
    corpus: Corpus
    index: Any
    results: list[Any]
    queries: list[Any]
    fingerprint: str
    fingerprint_material: dict[str, Any]
    surface: AttackSurface
    n_attack_candidates: int
    n_queries_with_attack_candidate: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "stratum": self.stratum,
            "regime": self.regime,
            "replay_fingerprint": self.fingerprint,
            "fingerprint_material": {
                k: v for k, v in self.fingerprint_material.items() if k != "per_query_sha256"
            },
            "n_document_slots": len(self.corpus.documents),
            "n_queries": len(self.results),
            "n_candidates": sum(len(r.candidates) for r in self.results),
            "n_attack_candidates_retrieved": self.n_attack_candidates,
            "n_queries_with_attack_candidate": self.n_queries_with_attack_candidate,
            "surface": self.surface.as_dict(),
        }


def build_adaptive_replay(
    base_corpus: Corpus,
    stratum: str,
    *,
    retriever_config: RetrieverConfig | None = None,
) -> AdaptiveReplay:
    """Crawl, merge, index and retrieve one stratum's replay.

    The retriever configuration and encoder are the same objects the frozen base replay used, so the
    only thing that differs between this replay and the static one is the presence of the attack
    pages. Changing the retriever here would confound the adaptive effect with a retrieval change.
    """
    surface = build_attack_surface(stratum)
    print(f"\n[adaptive:{stratum}] {len(surface.templates)} templates, {surface.n_pages} pages, "
          f"{len(surface.domains_by_entity)} target entities")

    entries = synthesize_manifest(
        [p for p in load_adaptive_pages() if str(p["attack_template_id"]) in surface.templates],
        entity_template_map(),
    )
    adaptive_corpus = crawl_corpus(
        manifest=entries,
        pages_dir=ADAPTIVE_PAGES_DIR,
        verify_hashes=True,
        registry=base_corpus.registry,
        progress_every=0,
    )
    merged = merge_corpora(base_corpus, adaptive_corpus, label=f"stratum-{stratum}")
    print(f"[adaptive:{stratum}] merged corpus: {len(merged.documents)} slots "
          f"({len(base_corpus.documents)} base + {len(adaptive_corpus.documents)} adaptive)")

    index = build_index(merged, phase=PHASE_INDEX, progress_every=0)

    entities = load_yaml(REPO_ROOT / "registry" / "entities.yaml")["entities"]
    targets = set(surface.domains_by_entity)
    queries = build_entity_queries([e for e in entities if str(e["entity_id"]) in targets])
    if not queries:
        raise AdaptiveCorpusError(f"stratum {stratum!r} produced no queries")

    config = retriever_config or RetrieverConfig(mode="hybrid_rrf", top_k=10)
    encoder = HashedNGramEncoder()
    retriever = Retriever(index, config=config, encoder=encoder)
    results = retriever.search_all(queries, progress_every=0)

    query_digest = hashlib.sha256(
        json.dumps([q.as_dict() for q in queries], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    material = {
        "corpus_digest": merged.corpus_digest,
        "index_digest": index.index_digest,
        "retriever_config_digest": config.digest(),
        "encoder_id": encoder.encoder_id,
        "query_set_digest": query_digest,
        "adaptive_stratum": stratum,
        "per_query_sha256": sorted(r.sha256 for r in results),
    }
    fingerprint = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    n_attack = 0
    n_q_attack = 0
    for r in results:
        hit = sum(1 for c in r.candidates if surface.is_attack_domain(r.entity_id, c.domain_id))
        n_attack += hit
        n_q_attack += 1 if hit else 0
    print(f"[adaptive:{stratum}] {len(results)} queries, "
          f"{sum(len(r.candidates) for r in results)} candidates, "
          f"{n_attack} attack candidates retrieved across {n_q_attack} queries")
    print(f"[adaptive:{stratum}] replay fingerprint {fingerprint[:16]}...")

    return AdaptiveReplay(
        stratum=stratum,
        regime=surface.regime,
        corpus=merged,
        index=index,
        results=list(results),
        queries=list(queries),
        fingerprint=fingerprint,
        fingerprint_material=material,
        surface=surface,
        n_attack_candidates=n_attack,
        n_queries_with_attack_candidate=n_q_attack,
    )


def admit_adaptive_run(matrix: ExperimentMatrix, replay: AdaptiveReplay, ev: EvaluatorContext) -> dict[str, Any]:
    """Declare and admit one stratum's run under rule R5.

    The AD-* ids are declared through ``adaptive_template_ids``, which R5 governs; they are
    deliberately *not* declared on the ``attack_template`` axis, which R2 checks against the static
    pool they were never part of. Stratum A runs declare ``primary_evaluation``; Stratum B declares
    ``secondary_evaluation``, and R5 rejects it if it ever claims otherwise.
    """
    targets = replay.surface.target_entities
    entity_templates = sorted({ev.entity_template_of[e] for e in targets if e in ev.entity_template_of})
    # Site templates come from the target entities' BASE pages: the adaptive pages carry the
    # synthetic AD-SITE id, which is not in any pool and would trip R2 if declared.
    site_templates = sorted(ev.site_templates_of_regime.get("test", set()))
    run = RunSpec(
        run_id=f"step5_adaptive_stratum_{replay.stratum.lower()}",
        regime="adaptive_holdout",
        purpose=PURPOSE_OF_STRATUM[replay.stratum],
        entity_template_ids=tuple(entity_templates),
        site_template_ids=tuple(site_templates),
        adaptive_template_ids=tuple(sorted(replay.surface.templates)),
        notes=(
            f"Step 5 adaptive robustness, stratum {replay.stratum}. "
            f"{'Primary claim eligible (inside the pre-declared region).' if replay.stratum == 'A' else 'Secondary only: outside the pre-declared region.'}"
        ),
    )
    report = matrix.admit(run).raise_if_rejected()
    return {"run": run.as_dict(), "admission": report.as_dict()}


# ======================================================================================
# ASR_a
# ======================================================================================
@dataclass
class AdaptiveOutcome:
    """One response's adaptive-attack outcome for one defense."""

    query_id: str
    entity_id: str
    action: str
    n_attack_retrieved: int
    n_attack_presented: int
    attack_templates_retrieved: tuple[str, ...]
    attack_templates_presented: tuple[str, ...]

    @property
    def attempted(self) -> bool:
        return self.n_attack_retrieved > 0

    @property
    def succeeded(self) -> bool:
        return self.n_attack_presented > 0


def adaptive_outcomes(
    triples: Sequence[TripleOutcome], surface: AttackSurface
) -> list[AdaptiveOutcome]:
    """Group triple decisions into per-response adaptive outcomes."""
    by_query: dict[str, list[TripleOutcome]] = defaultdict(list)
    for t in triples:
        by_query[t.query_id].append(t)

    out: list[AdaptiveOutcome] = []
    for qid in sorted(by_query):
        group = by_query[qid]
        head = group[0]
        retrieved: set[str] = set()
        presented: set[str] = set()
        n_r = n_p = 0
        for t in group:
            if not surface.is_attack_domain(t.entity_id, t.domain_id):
                continue
            n_r += 1
            retrieved |= surface.templates_by_pair.get((t.entity_id, t.domain_id), set())
            if t.presented:
                n_p += 1
                presented |= surface.templates_by_pair.get((t.entity_id, t.domain_id), set())
        out.append(
            AdaptiveOutcome(
                query_id=qid,
                entity_id=head.entity_id,
                action=head.action,
                n_attack_retrieved=n_r,
                n_attack_presented=n_p,
                attack_templates_retrieved=tuple(sorted(retrieved)),
                attack_templates_presented=tuple(sorted(presented)),
            )
        )
    return out


def _rate(num: int, den: int) -> float | None:
    return (num / den) if den else None


def compute_asr(
    outcomes: Sequence[AdaptiveOutcome], surface: AttackSurface
) -> dict[str, Any]:
    """``ASR_a`` on both denominators, plus per-action and per-template breakdowns."""
    attempted = [o for o in outcomes if o.attempted]
    succeeded = [o for o in attempted if o.succeeded]

    per_action: dict[str, Any] = {}
    for action in sorted({o.action for o in outcomes}):
        sel = [o for o in outcomes if o.action == action]
        att = [o for o in sel if o.attempted]
        per_action[action] = {
            "ASR_a": _rate(sum(1 for o in att if o.succeeded), len(att)),
            "n_attempted": len(att),
            "n_responses": len(sel),
            "n_succeeded": sum(1 for o in att if o.succeeded),
        }

    # A template counts as successful if any response it reached was compromised.
    reached: dict[str, int] = Counter()
    compromised: set[str] = set()
    for o in outcomes:
        for tid in o.attack_templates_retrieved:
            reached[tid] += 1
        compromised |= set(o.attack_templates_presented)
    n_reached = len(reached)

    return {
        "ASR_a": _rate(len(succeeded), len(attempted)),
        "ASR_a_definition": (
            "responses presenting >= 1 adaptive attack domain / responses in which >= 1 adaptive "
            "attack page was retrieved"
        ),
        "n_attempted": len(attempted),
        "n_succeeded": len(succeeded),
        "ASR_a_unconditioned": _rate(len(succeeded), len(outcomes)),
        "ASR_a_unconditioned_definition": (
            "same numerator over EVERY response for a targeted entity, retrieved or not. Lower than "
            "the conditioned rate by exactly the share of responses the attacker never reached, "
            "which retrieval defeated rather than the defense."
        ),
        "n_responses": len(outcomes),
        "retrieval_reach": _rate(len(attempted), len(outcomes)),
        "per_action": per_action,
        "per_template": {
            "n_templates": len(surface.templates),
            "n_templates_reaching_a_response": n_reached,
            "n_templates_compromising_a_response": len(compromised),
            "template_success_rate": _rate(len(compromised), n_reached),
            "compromising_template_ids": sorted(compromised),
        },
    }


def utility_under_adaptive_pressure(
    adaptive_row: Mapping[str, Any] | None, static_row: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Utility deltas between the adaptive replay and the static ``test`` replay.

    Both rows describe the same defense; the adaptive row is restricted to the targeted entities, so
    the comparison isolates the effect of adding the attack pages from any difference between
    entities. Read as "what did the attacker cost the defense in utility", not as an absolute level.
    """
    def get(row: Mapping[str, Any] | None, group: str, key: str) -> float | None:
        if not row:
            return None
        v = row.get(group, {}).get(key)
        return None if v is None else float(v)

    out: dict[str, Any] = {}
    for group, key in (
        ("primary", "ATPR"),
        ("primary", "UALER"),
        ("secondary", "FRR"),
        ("secondary", "BER"),
        ("secondary", "abstention_rate"),
    ):
        a, s = get(adaptive_row, group, key), get(static_row, group, key)
        out[key] = {
            "adaptive": None if a is None else round(a, 6),
            "static_test": None if s is None else round(s, 6),
            "delta_adaptive_minus_static": None if (a is None or s is None) else round(a - s, 6),
        }
    out["_note"] = (
        "The static comparison covers all 36 test entities while the adaptive replay covers only the "
        "targeted subset, so a small delta can reflect entity composition as well as adaptive "
        "pressure. The entity-matched comparison is the per-defense delta, not the absolute level."
    )
    return out


# ======================================================================================
# Driver-facing helper
# ======================================================================================
def evaluate_stratum(
    base_corpus: Corpus,
    ev: EvaluatorContext,
    params: FrozenParameters,
    matrix: ExperimentMatrix,
    stratum: str,
    *,
    fetch_factory: Callable[[Corpus], tuple[Callable[[str], str], Callable[[str], list[str]]]],
) -> dict[str, Any]:
    """Build one stratum's replay, sweep all 18 configurations over it, and score ``ASR_a``.

    Returns the replay, per-configuration triples, adaptive outcomes and an evaluator context whose
    regime map labels the targeted entities with this stratum's regime -- so the shared metric code
    reports them under ``adaptive_holdout`` rather than under ``test``.
    """
    replay = build_adaptive_replay(base_corpus, stratum)
    admission = admit_adaptive_run(matrix, replay, ev)
    print(f"[adaptive:{stratum}] admitted as {admission['run']['purpose']} "
          f"(rules {', '.join(admission['admission']['checked_rules'])})")

    fetch_text, fetch_links = fetch_factory(replay.corpus)
    harness = DefenseEvaluationHarness(
        registry=replay.corpus.registry,
        official_registry=load_official_registry(),
        fetch_links=fetch_links,
        site_index=build_site_index(replay.corpus.documents),
        read_phase=PHASE_LIVE,
        replay_fingerprint=replay.fingerprint,
    )
    defenses = build_all_configurations(params, replay_fingerprint=replay.fingerprint)
    assert_replay_bound(defenses, replay.fingerprint)

    # Relabel the targeted entities so the shared metric code files them under this regime.
    ev_stratum = replace(
        ev,
        entity_to_regime={
            **ev.entity_to_regime,
            **{e: replay.regime for e in replay.surface.target_entities},
        },
    )
    sweep = run_sweep(
        defenses,
        harness,
        replay.results,
        ev_stratum,
        fetch_text,
        progress_label=f"adaptive:{stratum}",
    )
    asr = {
        name: compute_asr(adaptive_outcomes(triples, replay.surface), replay.surface)
        for name, triples in sweep.triples.items()
    }
    return {
        "replay": replay,
        "admission": admission,
        "sweep": sweep,
        "ev": ev_stratum,
        "asr": asr,
    }


def write_adaptive_snapshot(replays: Sequence[AdaptiveReplay], path: Path | None = None) -> Path:
    """Freeze the adaptive retrieval replays, in the shape of the base snapshot.

    Written so the adaptive arm is replayable independently of the corpus: a reader can verify the
    fingerprint without re-running the attacker.
    """
    dst = path or ADAPTIVE_SNAPSHOT_PATH
    doc = {
        "metadata": {
            "schema_version": "1.0",
            "builder": "experiments/adaptive_evaluation.py",
            "contract_ref": "CONTRACT.md Section 3 (RQ4), Section 6",
            "purpose": (
                "Frozen retrieval replay for the adaptive arm. One replay per stratum: the base "
                "corpus plus that stratum's 300 optimised pages, indexed and retrieved with the "
                "same retriever configuration and encoder as the static replay, so the only "
                "difference between the two is the presence of the attack pages."
            ),
            "stratum_policy": (
                "Stratum A (inside the pre-declared adaptive holdout region) carries the primary "
                "adaptive claim. Stratum B sits outside the region and is secondary only "
                "(web_rag/experiment_matrix.py rule R5). They are separate replays because 16 of the "
                "29 target entities are attacked by both strata, so a merged replay could not "
                "attribute a misbinding to one of them."
            ),
            "n_strata": len(replays),
            "two_phase": {
                "index_phase": PHASE_INDEX,
                "read_phase": PHASE_LIVE,
            },
        },
        "strata": {
            r.stratum: {
                **r.as_dict(),
                "per_query_sha256": r.fingerprint_material["per_query_sha256"],
                "query_index": [q.as_dict() for q in r.queries],
                "snapshots": [
                    {
                        "query_id": res.query_id,
                        "query_text": res.query_text,
                        "mode": res.mode,
                        "sha256": res.sha256,
                        "candidates": [c.as_dict() for c in res.candidates],
                    }
                    for res in r.results
                ],
            }
            for r in replays
        },
    }
    dst.write_text(json.dumps(doc, separators=(",", ":")) + "\n", encoding="utf-8")
    return dst


def describe_adaptive_evaluation() -> dict[str, Any]:
    return {
        "adaptive_evaluation_version": ADAPTIVE_EVALUATION_VERSION,
        "contract_ref": "CONTRACT.md Section 3 (RQ4)",
        "prereg_ref": "preregistration.yaml secondary_endpoints (ASR_a)",
        "replay_design": (
            "Base corpus PLUS one stratum's adaptive pages, re-indexed and re-retrieved. The base "
            "corpus is kept because an adaptive attack is an addition to the web: the genuine "
            "official page and the authorized providers still compete for rank. Retrieving only "
            "attack pages would measure whether a defense can reject everything."
        ),
        "stratum_separation": (
            "Two replays, one per stratum. Rule R5 admits only Stratum A to a primary claim, and 16 "
            "of the 29 target entities are attacked by both strata, so a merged replay could not "
            "attribute a misbinding to one stratum."
        ),
        "asr_definitions": {
            "attempted": (
                "a response whose candidate list contains >= 1 page of an adaptive attack template"
            ),
            "attempted_unconditioned": "every response for a targeted entity",
            "succeeded": "the response presents >= 1 adaptive attack domain, at any rank",
            "why_both_denominators": (
                "A template whose pages never reached the candidate list was defeated by retrieval, "
                "not by the defense. Reporting only the unconditioned rate would credit the defense "
                "for retrieval's work."
            ),
        },
        "retriever_note": (
            "Retriever configuration, encoder and top_k are identical to the static replay. Changing "
            "them here would confound the adaptive effect with a retrieval change."
        ),
        "limitation": (
            "The Step 3 attacker optimised against a panel of Section 10 surface heuristics, not "
            "against AegisLink, which did not exist when the pages were generated. This arm "
            "therefore measures robustness to a strong surface-optimised attacker and NOT to an "
            "attacker optimising against the evidence-graph verifier itself. A white-box adaptive "
            "attack against the official-backlink direction check -- the family Step 4 identified as "
            "load-bearing -- remains unmeasured."
        ),
    }


__all__ = [
    "ADAPTIVE_EVALUATION_VERSION",
    "ADAPTIVE_SNAPSHOT_PATH",
    "ADAPTIVE_SITE_TEMPLATE_ID",
    "PURPOSE_OF_STRATUM",
    "REGIME_OF_STRATUM",
    "AdaptiveCorpusError",
    "AdaptiveOutcome",
    "AdaptiveReplay",
    "AttackSurface",
    "adaptive_outcomes",
    "admit_adaptive_run",
    "build_adaptive_replay",
    "build_attack_surface",
    "compute_asr",
    "describe_adaptive_evaluation",
    "entity_template_map",
    "evaluate_stratum",
    "load_adaptive_pages",
    "load_adaptive_templates",
    "merge_corpora",
    "synthesize_manifest",
    "utility_under_adaptive_pressure",
    "write_adaptive_snapshot",
]
