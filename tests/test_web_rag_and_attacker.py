"""Step 3 tests: Web-RAG replay environment and adaptive attacker.

CONTRACT.md Sections 3 (RQ2, RQ4), 5.3, 6, 7, 16.

The suite is organised around the four guarantees Step 3 claims, and every guarantee has both a
positive test and a *negative falsifiability control* -- a deliberately broken input that the
guard must reject. A guard that has never been seen to fire is not evidence of anything.

1. Two-phase replay fidelity (indexed vs live)
2. Zero leakage of experimenter-only fields
3. Trace schema, first-divergence attribution, and snapshot hash reproducibility
4. Adaptive templates disjoint from the core matrix and inside the declared region
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attacks.adaptive_attacker import (  # noqa: E402
    ACTIONS,
    ADAPTIVE_ID_PREFIX,
    OFFICIAL_CLAIM_VARIANTS,
    OFFICIAL_CLAIM_VARIANTS_KEYWORD_FREE,
    OFFICIAL_CLAIM_VARIANTS_WITH_KEYWORD,
    STRATUM_A_PINNED,
    AdaptiveAttacker,
    AttackerWorld,
    HeuristicDefensePanel,
    Realisation,
    assert_disjoint_from_core,
    assert_ground_truth_untouched,
    assert_no_label_vocabulary,
    build_adaptive_design,
    render_corroborating_page,
    render_primary_page,
    safety_check_pages,
    verify_two_phase,
)
from web_rag.crawler_indexer import (  # noqa: E402
    PHASE_INDEX,
    PHASE_LIVE,
    CorpusIntegrityError,
    build_index,
    crawl_corpus,
    load_page_manifest,
    meta_channel,
    visible_text,
)
from web_rag.experiment_matrix import (  # noqa: E402
    ADAPTIVE_REGIMES,
    FIT_FORBIDDEN_REGIMES,
    AdmissionError,
    ExperimentMatrix,
    RunPurpose,
    RunSpec,
    TemplateTriple,
)
from web_rag.exposure import (  # noqa: E402
    ALL_FORBIDDEN_FIELDS,
    DOMAIN_ROLE_VALUES,
    EXPERIMENTER_ONLY_FIELDS,
    EXPERIMENTER_ONLY_LIFECYCLE_SUBFIELDS,
    ExposureTier,
    LeakageError,
    assert_no_leakage,
    load_public_domain_registry,
    project_domain_record,
    public_delegation_evidence,
)
from web_rag.retriever import (  # noqa: E402
    RANKING_MODES,
    SCORE_PRECISION,
    HashedNGramEncoder,
    Retriever,
    RetrieverConfig,
    build_entity_queries,
    candidate_hash,
)
from web_rag.trace_recorder import (  # noqa: E402
    NO_FAILURE,
    RELATION_VOCABULARY,
    STAGE_NAMES,
    ReferenceStagePipeline,
    ScriptedPipeline,
    Stage,
    TraceRecorder,
    assert_all_stages_reachable,
    build_oracle_and_pipeline,
    summarize_traces,
    validate_trace_schema,
)

BENCH = REPO_ROOT / "data" / "benchmark"
SNAPSHOT_PATH = BENCH / "retrieval_snapshots.json"
TRACES_PATH = BENCH / "stage_traces.json"
ADAPTIVE_TEMPLATES_PATH = REPO_ROOT / "attacks" / "adaptive_templates.yaml"
ADAPTIVE_MANIFEST_PATH = BENCH / "adaptive_page_manifest.yaml"

EXPECTED_MANIFEST_PAGES = 2631
EXPECTED_SLOTS = 2002
EXPECTED_CHANGING = 629


def _load_yaml(path: Path) -> Any:
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    return yaml.load(path.read_text(encoding="utf-8"), Loader=loader)


# ======================================================================================
# Session fixtures -- the corpus is large, so build it once
# ======================================================================================
@pytest.fixture(scope="session")
def corpus():
    return crawl_corpus(verify_hashes=True, progress_every=0)


@pytest.fixture(scope="session")
def index(corpus):
    return build_index(corpus, phase=PHASE_INDEX, progress_every=0)


@pytest.fixture(scope="session")
def entities():
    return _load_yaml(REPO_ROOT / "registry" / "entities.yaml")["entities"]


@pytest.fixture(scope="session")
def small_retriever(index):
    """Top-5 hybrid retriever, reused across tests to avoid re-encoding 2,002 documents."""
    return Retriever(index, config=RetrieverConfig(mode="hybrid_rrf", top_k=5))


@pytest.fixture(scope="session")
def sample_queries(entities):
    return build_entity_queries(entities[:6])


@pytest.fixture(scope="session")
def oracle_and_pipeline(corpus, entities):
    domains_raw = _load_yaml(REPO_ROOT / "registry" / "domains.yaml")
    return build_oracle_and_pipeline(
        corpus,
        graph_edges=_load_yaml(REPO_ROOT / "registry" / "authorization_graph.yaml")["edges"],
        delegations=domains_raw["delegations"],
        entities=entities,
        aliases=_load_yaml(REPO_ROOT / "registry" / "aliases.yaml")["aliases"],
        read_phase=PHASE_LIVE,
    )


@pytest.fixture(scope="session")
def frozen_snapshot():
    assert SNAPSHOT_PATH.is_file(), (
        "data/benchmark/retrieval_snapshots.json missing; run "
        "workflow/12_build_web_rag_index_and_snapshots.py"
    )
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def frozen_traces():
    assert TRACES_PATH.is_file(), "data/benchmark/stage_traces.json missing; run workflow/12"
    return json.loads(TRACES_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def attacker_world():
    return AttackerWorld.load()


@pytest.fixture(scope="session")
def adaptive_design(attacker_world):
    return build_adaptive_design(attacker_world)


@pytest.fixture(scope="session")
def frozen_adaptive():
    assert ADAPTIVE_TEMPLATES_PATH.is_file(), (
        "attacks/adaptive_templates.yaml missing; run workflow/13_generate_adaptive_attacks.py"
    )
    return _load_yaml(ADAPTIVE_TEMPLATES_PATH)


@pytest.fixture(scope="session")
def matrix():
    return ExperimentMatrix.from_splits()


# ======================================================================================
# 1. Crawler and two-phase replay fidelity
# ======================================================================================
class TestTwoPhaseCrawling:
    def test_corpus_cardinality(self, corpus):
        assert corpus.n_pages == EXPECTED_MANIFEST_PAGES
        assert len(corpus) == EXPECTED_SLOTS
        assert len(corpus.changing_doc_ids()) == EXPECTED_CHANGING
        assert len(corpus.static_doc_ids()) == EXPECTED_SLOTS - EXPECTED_CHANGING

    def test_index_covers_every_slot(self, corpus, index):
        assert len(index) == len(corpus)
        assert set(index.doc_ids) == set(corpus.doc_ids)
        assert index.phase_indexed == PHASE_INDEX

    def test_every_changing_slot_differs_across_phases(self, corpus):
        """The Section 7 factor is inert unless the two phases differ in bytes."""
        identical = [d.doc_id for d in corpus if d.changing and not d.content_changed]
        assert identical == [], f"{len(identical)} changing slots serve identical bytes"

    def test_every_static_slot_identical_across_phases(self, corpus):
        differing = [d.doc_id for d in corpus if not d.changing and d.content_changed]
        assert differing == []

    def test_indexed_and_live_hashes_differ_for_changing_slots(self, corpus):
        for doc_id in corpus.changing_doc_ids():
            d = corpus.documents[doc_id]
            assert d.sha256(PHASE_INDEX) != d.sha256(PHASE_LIVE)

    def test_index_holds_phase1_text_not_phase2(self, corpus, index):
        """The whole point: the ranker's evidence is the benign snapshot.

        For at least some changing slot the two phases must differ in *visible text* (not merely
        in the page-id meta tag), and the index must hold the phase-1 text.
        """
        text_differs = 0
        for doc_id in corpus.changing_doc_ids():
            d = corpus.documents[doc_id]
            if d.text(PHASE_INDEX) != d.text(PHASE_LIVE):
                text_differs += 1
                assert index.documents[doc_id].text == d.text(PHASE_INDEX)
                assert index.documents[doc_id].text != d.text(PHASE_LIVE)
        assert text_differs == EXPECTED_CHANGING, (
            f"only {text_differs}/{EXPECTED_CHANGING} changing slots differ in VISIBLE text; "
            f"a difference confined to the meta channel would not affect a reader"
        )

    def test_fetch_returns_the_requested_phase(self, corpus):
        doc_id = corpus.changing_doc_ids()[0]
        d = corpus.documents[doc_id]
        assert corpus.fetch(doc_id, PHASE_INDEX) == d.indexed_html
        assert corpus.fetch(doc_id, PHASE_LIVE) == d.live_html
        assert corpus.fetch(doc_id, PHASE_INDEX) != corpus.fetch(doc_id, PHASE_LIVE)

    def test_unknown_phase_rejected(self, corpus):
        with pytest.raises(ValueError, match="unknown phase"):
            corpus.documents[corpus.doc_ids[0]].html("yesterday")

    def test_shared_urls_do_not_collapse_documents(self, corpus):
        """A URL is not a key: shared provider hosts serve one page per business."""
        by_url = Counter(d.url for d in corpus)
        assert max(by_url.values()) > 1, "expected shared hosts in this corpus"
        assert len(corpus) > len(by_url), (
            "document count collapsed to URL count; indexing by URL would have destroyed the "
            "entity-relative structure"
        )

    def test_two_phase_diff_report_has_no_exploitable_regularity(self, corpus):
        """If the live phase only ever added text, a defense could key on that alone."""
        rep = corpus.two_phase_diff_report()
        assert rep["n_changing"] == EXPECTED_CHANGING
        assert rep["live_added_and_removed"] > 0
        assert rep["min_token_delta"] < 0 < rep["max_token_delta"], (
            "token deltas are single-signed, so 'did the page grow?' would be a free signal"
        )

    # -- negative controls ------------------------------------------------------------
    def test_half_pair_manifest_is_rejected(self, tmp_path):
        """A slot with 'indexed' but no 'live' cannot render the factor and must fail."""
        manifest = load_page_manifest()
        first_indexed = next(e for e in manifest if e["snapshot"] == "indexed")
        broken = [
            e
            for e in manifest
            if not (
                e["snapshot"] == "live"
                and e["entity_id"] == first_indexed["entity_id"]
                and e["domain_id"] == first_indexed["domain_id"]
                and e["site_template_id"] == first_indexed["site_template_id"]
                and e["attack_template_id"] == first_indexed["attack_template_id"]
            )
        ]
        with pytest.raises(CorpusIntegrityError, match="snapshot labels"):
            crawl_corpus(manifest=broken, verify_hashes=False, progress_every=0)

    def test_tampered_page_bytes_are_rejected(self, tmp_path):
        """Corpus drift must fail loudly: it would invalidate every frozen snapshot."""
        manifest = load_page_manifest()[:4]
        pages_root = tmp_path / "pages"
        for e in manifest:
            d = pages_root / str(e["entity_id"])
            d.mkdir(parents=True, exist_ok=True)
            src = BENCH / "pages" / str(e["entity_id"]) / f"{e['page_id']}.html"
            d.joinpath(f"{e['page_id']}.html").write_text(
                src.read_text(encoding="utf-8"), encoding="utf-8"
            )
        # Untampered copy passes.
        crawl_corpus(manifest=manifest, pages_dir=pages_root, verify_hashes=True, progress_every=0)
        # One flipped byte must be caught.
        victim = pages_root / str(manifest[0]["entity_id"]) / f"{manifest[0]['page_id']}.html"
        victim.write_text(victim.read_text(encoding="utf-8") + "<!-- drift -->", encoding="utf-8")
        with pytest.raises(CorpusIntegrityError, match="hash mismatch"):
            crawl_corpus(
                manifest=manifest, pages_dir=pages_root, verify_hashes=True, progress_every=0
            )

    def test_missing_page_file_is_rejected(self, tmp_path):
        manifest = load_page_manifest()[:2]
        with pytest.raises(CorpusIntegrityError, match="missing rendered page"):
            crawl_corpus(
                manifest=manifest, pages_dir=tmp_path / "empty", verify_hashes=False, progress_every=0
            )


# ======================================================================================
# 2. Exposure firewall -- zero leakage
# ======================================================================================
class TestExposureFirewall:
    def test_public_domain_view_has_only_whitelisted_fields(self):
        raw = _load_yaml(REPO_ROOT / "registry" / "domains.yaml")["domains"]
        record = next(iter(raw.values()))
        view = project_domain_record(record).as_dict()
        assert set(view) == {"domain_id", "domain", "display_name", "shared"}
        for forbidden in ("role", "adversarial", "lifecycle", "provider_slot", "owner_entity_id"):
            assert forbidden not in view

    def test_public_registry_strips_role_and_adversarial(self):
        reg = load_public_domain_registry()
        assert len(reg.domains) == 1790
        assert_no_leakage(reg.as_dict(ExposureTier.RETRIEVER), location="registry")
        assert_no_leakage(reg.as_dict(ExposureTier.VERIFIER), location="registry")

    def test_lifecycle_observations_drop_prior_grants(self):
        reg = load_public_domain_registry()
        assert reg.lifecycle, "expected lifecycle records for expired-domain-takeover cases"
        for obs in reg.lifecycle.values():
            d = obs.as_dict()
            for banned in EXPERIMENTER_ONLY_LIFECYCLE_SUBFIELDS:
                assert banned not in d, f"{banned} survived the lifecycle derivation"
            assert set(d) == {
                "domain_id",
                "first_seen_snapshot",
                "expired_at_snapshot",
                "ownership_change_at_snapshot",
                "current_holder",
                "expired",
                "ownership_changed",
            }

    def test_lifecycle_derivation_is_derived_not_copied(self):
        """``expired`` / ``ownership_changed`` must be recomputed, not read off a flag."""
        reg = load_public_domain_registry()
        raw = _load_yaml(REPO_ROOT / "registry" / "domains.yaml")["domains"]
        for domain_id, obs in reg.lifecycle.items():
            src = raw[domain_id]["lifecycle"]
            assert obs.expired == (src.get("expired_at_snapshot") is not None)
            assert obs.ownership_changed == (
                src.get("ownership_change_at_snapshot") is not None
            )

    def test_delegation_evidence_renames_granted_actions(self):
        """The verifier must read a backlink off a page, not trust the registry's grant field."""
        ev = public_delegation_evidence()
        assert ev
        for rec in ev:
            assert set(rec) == {"entity_id", "domain_id", "kind", "claimed_actions"}
            assert "granted_actions" not in rec
            assert "evidence" not in rec

    def test_retriever_tier_document_views_are_clean(self, corpus):
        for doc_id in corpus.doc_ids[:400]:
            view = corpus.documents[doc_id].public_view(PHASE_INDEX, ExposureTier.RETRIEVER)
            assert set(view) == {
                "doc_id", "url", "domain", "domain_id", "title", "text", "outbound_links"
            }
            assert_no_leakage(view, location=f"doc/{doc_id}")

    def test_meta_channel_is_stripped_from_visible_text(self, corpus):
        """The x-aegis channel is EVALUATOR tier; it must not reach retriever-tier text."""
        checked = 0
        for doc_id in corpus.doc_ids[:300]:
            d = corpus.documents[doc_id]
            channel = meta_channel(d.indexed_html)
            assert "x-aegis-entity-id" in channel, "expected the Section 8 channel in the bytes"
            text = visible_text(d.indexed_html)
            assert "x-aegis" not in text
            assert channel["x-aegis-entity-id"] not in text
            checked += 1
        assert checked == 300

    def test_index_documents_are_clean(self, index):
        payload = {k: v.as_dict() for k, v in list(index.documents.items())[:400]}
        assert_no_leakage(payload, location="index")

    def test_retrieval_candidates_are_clean(self, small_retriever, sample_queries):
        for q in sample_queries:
            res = small_retriever.search(q)
            payload = res.as_dict(ExposureTier.READER)
            assert_no_leakage(payload, location=f"search/{q.query_id}")
            for cand in payload["candidates"]:
                assert set(cand) == {
                    "rank", "doc_id", "url", "domain", "domain_id", "title", "score",
                    "bm25_score", "dense_score", "bm25_rank", "dense_rank", "snippet",
                }

    def test_reader_tier_omits_experiment_addressing(self, small_retriever, sample_queries):
        """entity_id / action are evaluator addressing, not evidence the reader may use."""
        res = small_retriever.search(sample_queries[0])
        reader = res.as_dict(ExposureTier.READER)
        assert "entity_id" not in reader and "action" not in reader
        evaluator = res.as_dict(ExposureTier.EVALUATOR)
        assert evaluator["entity_id"] == res.entity_id

    def test_frozen_snapshot_file_is_clean(self, frozen_snapshot):
        """Brute-force sweep of the artifact on disk, independent of the guard."""
        text = json.dumps(frozen_snapshot["snapshots"], sort_keys=True)
        for field in ALL_FORBIDDEN_FIELDS:
            assert f'"{field}":' not in text, f"forbidden key {field} in frozen snapshots"
        for role in DOMAIN_ROLE_VALUES:
            assert f'"{role}"' not in text, f"forbidden role value {role} in frozen snapshots"
        assert_no_leakage(frozen_snapshot["snapshots"], location="frozen snapshots")

    def test_leakage_audit_artifact_reports_all_clean(self):
        audit = json.loads((REPO_ROOT / "results" / "leakage_audit.json").read_text())
        assert audit["metadata"]["all_clean"] is True
        assert len(audit["audits"]) >= 8
        for a in audit["audits"]:
            assert a["clean"], f"{a['location']} not clean: {a}"

    # -- negative controls ------------------------------------------------------------
    @pytest.mark.parametrize("field", sorted(EXPERIMENTER_ONLY_FIELDS))
    def test_guard_catches_forbidden_key(self, field):
        with pytest.raises(LeakageError, match="experimenter-only field"):
            assert_no_leakage({"doc_id": "DOC1", field: "whatever"}, location="probe")

    @pytest.mark.parametrize("role", DOMAIN_ROLE_VALUES)
    def test_guard_catches_laundered_role_value(self, role):
        """A label copied into a benign key is still a label.

        ``category`` is chosen deliberately: it is a *whitelist-clean* key name, so this control
        exercises the value arm of the guard rather than the key arm.
        """
        assert "category" not in ALL_FORBIDDEN_FIELDS
        with pytest.raises(LeakageError, match="experimenter-only VALUE"):
            assert_no_leakage({"doc_id": "DOC1", "category": role}, location="probe")

    def test_guard_catches_laundered_role_value_in_a_list(self):
        """Values inside a list have no key at all, so the value arm must still fire."""
        with pytest.raises(LeakageError, match="experimenter-only VALUE"):
            assert_no_leakage(
                {"doc_id": "DOC1", "tags": ["ordinary", "impersonating_official_site"]},
                location="probe",
            )

    def test_guard_walks_nested_structures(self):
        payload = {"a": [{"b": {"c": [{"adversarial": True}]}}]}
        with pytest.raises(LeakageError, match="adversarial"):
            assert_no_leakage(payload, location="probe")

    def test_guard_walks_dataclass_attributes(self):
        class Sneaky:
            def __init__(self) -> None:
                self.doc_id = "DOC1"
                self.role = "impersonating_official_site"

        with pytest.raises(LeakageError):
            assert_no_leakage({"wrapped": Sneaky()}, location="probe")

    def test_guard_catches_lifecycle_prior_grants(self):
        payload = {"lifecycle": {"prior_grants": ["browse", "contact"]}}
        with pytest.raises(LeakageError, match="prior_grants"):
            assert_no_leakage(payload, location="probe")

    def test_clean_payload_passes(self):
        assert_no_leakage(
            {"doc_id": "DOC1", "url": "https://x.test/", "text": "Some ordinary prose."},
            location="probe",
        )


# ======================================================================================
# 3. Retriever determinism and snapshot integrity
# ======================================================================================
class TestRetrieverDeterminism:
    @pytest.mark.parametrize("mode", RANKING_MODES)
    def test_same_query_same_hash(self, index, sample_queries, mode):
        cfg = RetrieverConfig(mode=mode, top_k=5)
        a = Retriever(index, config=cfg).search(sample_queries[0])
        b = Retriever(index, config=cfg).search(sample_queries[0])
        assert a.sha256 == b.sha256
        assert a.doc_ids == b.doc_ids

    def test_encoder_is_not_pythonhashseed_dependent(self):
        """blake2b, not hash(): a per-process seed would break every frozen hash."""
        enc = HashedNGramEncoder()
        v = enc.encode("Harbour Lantern Bistro official website")
        expected_dim = hashlib.blake2b(b"w:harbour", digest_size=8).digest()
        raw = int.from_bytes(expected_dim, "big")
        assert raw % enc.dimensions in v

    def test_encoded_vectors_are_l2_normalised(self):
        enc = HashedNGramEncoder()
        v = enc.encode("book a table at Cinder Court Kitchen")
        norm = sum(x * x for x in v.values()) ** 0.5
        assert abs(norm - 1.0) < 1e-9

    def test_scores_are_rounded_to_declared_precision(self, small_retriever, sample_queries):
        res = small_retriever.search(sample_queries[0])
        for c in res.candidates:
            assert c.score == round(c.score, SCORE_PRECISION)

    def test_ranks_are_dense_and_ordered(self, small_retriever, sample_queries):
        for q in sample_queries:
            res = small_retriever.search(q)
            assert [c.rank for c in res.candidates] == list(range(1, len(res.candidates) + 1))
            scores = [c.score for c in res.candidates]
            assert scores == sorted(scores, reverse=True)

    def test_ties_break_on_doc_id(self, index):
        """A total order is required, else replay depends on dict iteration order."""
        r = Retriever(index, config=RetrieverConfig(mode="bm25", top_k=50, candidate_pool=50))
        from web_rag.retriever import Query

        res = r.search(Query("Q-tie", "opening hours", "E0001", "browse", "action"))
        groups: dict[float, list[str]] = {}
        for c in res.candidates:
            groups.setdefault(c.score, []).append(c.doc_id)
        tied = [ids for ids in groups.values() if len(ids) > 1]
        assert tied, "expected at least one tie group on a generic query"
        for ids in tied:
            assert ids == sorted(ids), "tied candidates not ordered by doc_id"

    def test_hybrid_uses_both_signals(self, index, sample_queries):
        res = Retriever(index, config=RetrieverConfig(mode="hybrid_rrf", top_k=5)).search(
            sample_queries[0]
        )
        assert any(c.bm25_rank is not None for c in res.candidates)
        assert any(c.dense_rank is not None for c in res.candidates)
        assert any(c.dense_score != 0.0 for c in res.candidates)

    def test_config_digest_changes_with_config(self):
        a = RetrieverConfig(mode="bm25", top_k=10)
        b = RetrieverConfig(mode="bm25", top_k=10, k1=1.5)
        assert a.digest() != b.digest()

    def test_invalid_config_rejected(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            RetrieverConfig(mode="magic")
        with pytest.raises(ValueError, match="candidate_pool >= top_k"):
            RetrieverConfig(top_k=100, candidate_pool=10)


class TestFrozenSnapshots:
    def test_snapshot_file_shape(self, frozen_snapshot):
        for key in ("metadata", "environment", "retriever", "exposure_firewall",
                    "query_index", "snapshots"):
            assert key in frozen_snapshot
        md = frozen_snapshot["metadata"]
        assert md["replay_fingerprint"]
        assert md["counts"]["document_slots"] == EXPECTED_SLOTS
        assert md["counts"]["manifest_pages"] == EXPECTED_MANIFEST_PAGES
        assert md["two_phase"]["index_phase"] == PHASE_INDEX
        assert md["two_phase"]["read_phase"] == PHASE_LIVE

    def test_every_query_has_a_sha256(self, frozen_snapshot):
        snaps = frozen_snapshot["snapshots"]
        assert len(snaps) == frozen_snapshot["metadata"]["counts"]["queries"]
        for s in snaps:
            assert len(s["sha256"]) == 64
            int(s["sha256"], 16)  # must be hex

    def test_per_query_hashes_recompute(self, frozen_snapshot):
        """The recorded hash must be a function of the recorded candidates."""
        from web_rag.retriever import Candidate

        for s in frozen_snapshot["snapshots"]:
            cands = [
                Candidate(
                    rank=c["rank"], doc_id=c["doc_id"], url=c["url"], domain=c["domain"],
                    domain_id=c["domain_id"], title=c["title"], score=c["score"],
                    bm25_score=c["bm25_score"], dense_score=c["dense_score"],
                    bm25_rank=c["bm25_rank"], dense_rank=c["dense_rank"],
                    snippet=c.get("snippet", ""),
                )
                for c in s["candidates"]
            ]
            assert candidate_hash(s["query_id"], cands) == s["sha256"], (
                f"recorded hash for {s['query_id']} does not match its candidate list"
            )

    def test_replay_reproduces_frozen_hashes(self, index, entities, frozen_snapshot):
        """Re-run retrieval from scratch; every per-query hash must match the freeze.

        This is the guarantee Steps 4-8 depend on: the same evidence, every run.
        """
        cfg = RetrieverConfig(**{
            k: v for k, v in frozen_snapshot["retriever"]["config"].items()
            if k != "retriever_version"
        })
        r = Retriever(index, config=cfg, encoder=HashedNGramEncoder())
        queries = build_entity_queries(entities)
        frozen = {s["query_id"]: s["sha256"] for s in frozen_snapshot["snapshots"]}
        assert len(queries) == len(frozen)
        mismatches = []
        for q in queries:
            got = r.search(q).sha256
            if got != frozen[q.query_id]:
                mismatches.append((q.query_id, frozen[q.query_id], got))
        assert mismatches == [], f"{len(mismatches)} query hashes drifted: {mismatches[:3]}"

    def test_fingerprint_covers_config_and_encoder(self, frozen_snapshot):
        material = frozen_snapshot["metadata"]["fingerprint_material"]
        for key in ("corpus_digest", "index_digest", "retriever_config_digest",
                    "encoder_id", "query_set_digest"):
            assert material[key], f"{key} missing from the replay fingerprint"

    def test_fingerprint_changes_if_any_input_changes(self, frozen_snapshot):
        """A stale replay must fail loudly rather than silently mix evidence."""
        base = dict(frozen_snapshot["metadata"]["fingerprint_material"])
        base["per_query_sha256"] = sorted(s["sha256"] for s in frozen_snapshot["snapshots"])

        def fp(material: dict[str, Any]) -> str:
            return hashlib.sha256(
                json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()

        original = fp(base)
        assert original == frozen_snapshot["metadata"]["replay_fingerprint"]
        for key in ("corpus_digest", "index_digest", "retriever_config_digest",
                    "encoder_id", "query_set_digest"):
            perturbed = {**base, key: str(base[key]) + "x"}
            assert fp(perturbed) != original, f"fingerprint insensitive to {key}"

    def test_snapshot_hashes_ignore_snippet_text(self, small_retriever, sample_queries):
        """Documented property: the hash covers ordering and scores, not derived snippets."""
        res = small_retriever.search(sample_queries[0])
        import dataclasses

        altered = [dataclasses.replace(c, snippet="") for c in res.candidates]
        assert candidate_hash(res.query_id, altered) == res.sha256

    def test_snapshot_hashes_are_sensitive_to_order(self, small_retriever, sample_queries):
        res = small_retriever.search(sample_queries[0])
        import dataclasses

        swapped = list(res.candidates)
        swapped[0], swapped[1] = (
            dataclasses.replace(swapped[1], rank=1),
            dataclasses.replace(swapped[0], rank=2),
        )
        assert candidate_hash(res.query_id, swapped) != res.sha256


# ======================================================================================
# 4. Trace recorder
# ======================================================================================
class TestTraceRecorder:
    def test_stage_names_match_contract(self):
        assert STAGE_NAMES == (
            "retrieval_candidates",
            "resolved_entities",
            "extracted_entity_domain_relations",
            "inferred_action_authorizations",
            "presented_links",
        )

    def test_recorded_trace_has_all_five_stages_in_order(
        self, corpus, small_retriever, sample_queries, oracle_and_pipeline
    ):
        oracle, pipeline = oracle_and_pipeline
        rec = TraceRecorder(corpus=corpus, oracle=oracle, read_phase=PHASE_LIVE)
        for q in sample_queries[:6]:
            trace = rec.record(small_retriever.search(q), pipeline)
            assert tuple(s.stage for s in trace.stages) == STAGE_NAMES
            validate_trace_schema(trace.as_dict())

    def test_failure_origin_is_first_diverging_stage(
        self, corpus, small_retriever, sample_queries, oracle_and_pipeline
    ):
        oracle, pipeline = oracle_and_pipeline
        rec = TraceRecorder(corpus=corpus, oracle=oracle, read_phase=PHASE_LIVE)
        for q in sample_queries:
            trace = rec.record(small_retriever.search(q), pipeline)
            first = next(
                (s.stage for s in trace.stages if not s.matches_ground_truth), NO_FAILURE
            )
            assert trace.failure_origin == first

    def test_relations_stay_in_the_declared_vocabulary(
        self, corpus, small_retriever, sample_queries, oracle_and_pipeline
    ):
        oracle, pipeline = oracle_and_pipeline
        rec = TraceRecorder(corpus=corpus, oracle=oracle, read_phase=PHASE_LIVE)
        for q in sample_queries[:6]:
            trace = rec.record(small_retriever.search(q), pipeline)
            s3 = trace.stage(Stage.EXTRACTED_ENTITY_DOMAIN_RELATIONS)
            for v in list(s3.output.values()) + list(s3.expected.values()):
                assert v in RELATION_VOCABULARY

    def test_trace_vocabulary_is_label_neutral(self, frozen_traces):
        """A regex over trace files must not recover ground truth."""
        text = json.dumps(frozen_traces["traces"][:200])
        for banned in ("authorized_booking_provider", "impersonating_official_site",
                       "unauthorized_login_portal", "expired_domain_takeover"):
            assert banned not in text

    def test_reader_sees_the_live_phase(self, corpus, oracle_and_pipeline):
        """Two-phase payoff: the recorder's fetch must return phase-2 bytes."""
        oracle, _ = oracle_and_pipeline
        rec = TraceRecorder(corpus=corpus, oracle=oracle, read_phase=PHASE_LIVE)
        doc_id = corpus.changing_doc_ids()[0]
        d = corpus.documents[doc_id]
        assert rec._fetch_text(doc_id) == d.text(PHASE_LIVE)
        assert rec._fetch_text(doc_id) != d.text(PHASE_INDEX)

    def test_all_five_stages_are_reachable(
        self, corpus, small_retriever, sample_queries, oracle_and_pipeline
    ):
        """The attribution instrument must not be degenerate.

        The first version of the comparison rule compared every candidate and made stage 3
        absorb 94% of attributions, leaving stages 4 and 5 unreachable. This asserts the
        outcome-anchored scope fixed that, and guards against a regression.
        """
        oracle, _ = oracle_and_pipeline
        rec = TraceRecorder(corpus=corpus, oracle=oracle, read_phase=PHASE_LIVE)
        results = [small_retriever.search(q) for q in sample_queries]
        report = assert_all_stages_reachable(rec, results)
        assert report["all_five_stages_reachable"] is True
        assert set(report["witness_query_per_stage"]) == set(STAGE_NAMES)

    @pytest.mark.parametrize(
        "stage",
        [
            Stage.RESOLVED_ENTITIES,
            Stage.EXTRACTED_ENTITY_DOMAIN_RELATIONS,
            Stage.INFERRED_ACTION_AUTHORIZATIONS,
            Stage.PRESENTED_LINKS,
        ],
    )
    def test_scripted_pipeline_is_attributed_to_the_corrupted_stage(
        self, corpus, small_retriever, sample_queries, oracle_and_pipeline, stage
    ):
        oracle, _ = oracle_and_pipeline
        rec = TraceRecorder(corpus=corpus, oracle=oracle, read_phase=PHASE_LIVE)
        probe = ScriptedPipeline(oracle, corrupt_stage=stage)
        origins = set()
        for q in sample_queries:
            r = small_retriever.search(q)
            t = rec.record(r, probe)
            # Skip queries where retrieval itself already failed: stage 1 legitimately wins.
            if t.failure_origin != Stage.RETRIEVAL_CANDIDATES.value:
                origins.add(t.failure_origin)
        assert stage.value in origins, (
            f"corrupting {stage.value} never produced that attribution; got {origins}"
        )

    def test_faithful_pipeline_produces_no_failure(
        self, corpus, small_retriever, sample_queries, oracle_and_pipeline
    ):
        """A pipeline told the truth must not be blamed for anything."""
        oracle, _ = oracle_and_pipeline
        rec = TraceRecorder(corpus=corpus, oracle=oracle, read_phase=PHASE_LIVE)
        faithful = ScriptedPipeline(oracle, corrupt_stage=None)
        clean = 0
        for q in sample_queries:
            t = rec.record(small_retriever.search(q), faithful)
            if t.failure_origin == Stage.RETRIEVAL_CANDIDATES.value:
                continue  # retrieval recall failure is not the pipeline's fault
            assert t.failure_origin == NO_FAILURE, (
                f"faithful pipeline blamed at {t.failure_origin}: "
                f"{t.stage(t.failure_origin).divergence}"
            )
            assert t.presented_unauthorized is False
            clean += 1
        assert clean > 0

    def test_unauthorized_presentation_always_flags_stage5(self, frozen_traces):
        for t in frozen_traces["traces"]:
            if t["presented_unauthorized"]:
                assert t["stages"][-1]["matches_ground_truth"] is False

    def test_scope_reason_consistency(self, frozen_traces):
        for t in frozen_traces["traces"]:
            reason = t["attribution_scope_reason"]
            if reason == "no_failure":
                assert t["attribution_scope"] == []
            else:
                assert t["attribution_scope"], f"{reason} with empty scope"

    def test_frozen_traces_are_not_degenerate(self, frozen_traces):
        """At least three distinct outcomes must appear, or the instrument says nothing."""
        counts = frozen_traces["summary"]["failure_origin_counts"]
        nonzero = {k: v for k, v in counts.items() if v > 0}
        assert len(nonzero) >= 3, f"attribution collapsed onto {nonzero}"
        top = max(counts.values()) / sum(counts.values())
        assert top < 0.95, f"one outcome absorbs {top:.1%} of traces"

    def test_frozen_traces_validate(self, frozen_traces):
        for t in frozen_traces["traces"]:
            validate_trace_schema(t)

    def test_summary_matches_traces(self, frozen_traces):
        counts = Counter(t["failure_origin"] for t in frozen_traces["traces"])
        for k, v in counts.items():
            assert frozen_traces["summary"]["failure_origin_counts"][k] == v

    # -- schema negative controls ------------------------------------------------------
    def test_schema_rejects_missing_key(self, frozen_traces):
        bad = dict(frozen_traces["traces"][0])
        bad.pop("failure_origin")
        with pytest.raises(ValueError, match="missing keys"):
            validate_trace_schema(bad)

    def test_schema_rejects_wrong_stage_order(self, frozen_traces):
        bad = json.loads(json.dumps(frozen_traces["traces"][0]))
        bad["stages"][0], bad["stages"][1] = bad["stages"][1], bad["stages"][0]
        with pytest.raises(ValueError, match="mandated"):
            validate_trace_schema(bad)

    def test_schema_rejects_inconsistent_failure_origin(self, frozen_traces):
        bad = json.loads(json.dumps(frozen_traces["traces"][0]))
        bad["failure_origin"] = "presented_links"
        bad["stages"][-1]["matches_ground_truth"] = True
        bad["stages"][-1]["divergence"] = None
        with pytest.raises(ValueError):
            validate_trace_schema(bad)

    def test_schema_rejects_divergence_without_explanation(self, frozen_traces):
        bad = json.loads(json.dumps(frozen_traces["traces"][0]))
        bad["stages"][0]["matches_ground_truth"] = False
        bad["stages"][0]["divergence"] = None
        bad["failure_origin"] = "retrieval_candidates"
        with pytest.raises(ValueError, match="no divergence explanation"):
            validate_trace_schema(bad)

    def test_schema_rejects_silent_unauthorized_presentation(self, frozen_traces):
        bad = json.loads(json.dumps(frozen_traces["traces"][0]))
        bad["presented_unauthorized"] = True
        for s in bad["stages"]:
            s["matches_ground_truth"] = True
            s["divergence"] = None
        bad["failure_origin"] = "none"
        with pytest.raises(ValueError, match="presented_unauthorized"):
            validate_trace_schema(bad)

    def test_pipeline_emitting_bad_relation_is_rejected(
        self, corpus, small_retriever, sample_queries, oracle_and_pipeline
    ):
        oracle, _ = oracle_and_pipeline
        rec = TraceRecorder(corpus=corpus, oracle=oracle, read_phase=PHASE_LIVE)

        class BadPipeline(ScriptedPipeline):
            def extract_relations(self, result, fetch):
                return {c.domain_id: "definitely_the_official_one" for c in result.candidates}

        with pytest.raises(ValueError, match="outside RELATION_VOCABULARY"):
            rec.record(small_retriever.search(sample_queries[0]), BadPipeline(oracle))


# ======================================================================================
# 5. Experiment matrix admission controller
# ======================================================================================
class TestAdmissionController:
    def test_loads_all_regimes(self, matrix):
        assert set(matrix.regimes) == {
            "train_development", "validation", "test",
            "transfer_holdout", "adaptive_holdout", "fabricated_control",
        }
        for r in matrix.regimes:
            assert matrix.pool(r, "entity_template")
            assert matrix.pool(r, "attack_template")

    def _clean_run(self, matrix, regime, purpose):
        et = sorted(matrix.pool(regime, "entity_template"))[0]
        st = sorted(matrix.pool(regime, "site_template"))[0]
        at = sorted(matrix.pool(regime, "attack_template"))[0]
        return RunSpec(
            f"RUN-{regime}", regime, purpose,
            triples=(TemplateTriple(et, st, at),),
        )

    def test_clean_run_is_admitted(self, matrix):
        run = self._clean_run(matrix, "train_development", RunPurpose.DEVELOPMENT)
        rep = matrix.admit(run)
        assert rep.admitted, rep.as_dict()

    def test_unknown_regime_rejected(self, matrix):
        rep = matrix.admit(RunSpec("R", "production", RunPurpose.DEVELOPMENT))
        assert not rep.admitted
        assert any(v.rule_id == "R1_single_regime" for v in rep.violations)

    def test_foreign_template_rejected(self, matrix):
        """R2: a validation attack template must not appear in a train run."""
        et = sorted(matrix.pool("train_development", "entity_template"))[0]
        st = sorted(matrix.pool("train_development", "site_template"))[0]
        foreign = sorted(matrix.pool("validation", "attack_template"))[0]
        rep = matrix.admit(
            RunSpec("R", "train_development", RunPurpose.DEVELOPMENT,
                    attack_template_ids=(foreign,),
                    triples=(TemplateTriple(et, st, None),))
        )
        assert not rep.admitted
        assert any(v.rule_id == "R2_pool_membership" for v in rep.violations)

    @pytest.mark.parametrize("regime", sorted(FIT_FORBIDDEN_REGIMES))
    def test_threshold_selection_cannot_touch_forbidden_regimes(self, matrix, regime):
        """CONTRACT.md Section 5.3: no test template during threshold selection."""
        rep = matrix.admit(self._clean_run(matrix, regime, RunPurpose.THRESHOLD_SELECTION))
        assert not rep.admitted
        assert any(v.rule_id == "R3_no_test_in_threshold_selection" for v in rep.violations)

    def test_threshold_selection_allowed_on_train_and_validation(self, matrix):
        for regime in ("train_development", "validation"):
            rep = matrix.admit(self._clean_run(matrix, regime, RunPurpose.THRESHOLD_SELECTION))
            assert rep.admitted, rep.as_dict()

    def test_cross_regime_triple_rejected(self, matrix):
        """R4: independently split axes can form a triple belonging to no regime."""
        et = sorted(matrix.pool("train_development", "entity_template"))[0]
        st = sorted(matrix.pool("validation", "site_template"))[0]
        at = sorted(matrix.pool("test", "attack_template"))[0]
        rep = matrix.admit(
            RunSpec("R", "train_development", RunPurpose.DEVELOPMENT,
                    triples=(TemplateTriple(et, st, at),))
        )
        assert not rep.admitted
        assert any(v.rule_id == "R4_triple_consistency" for v in rep.violations)

    def test_require_admitted_raises(self, matrix):
        with pytest.raises(AdmissionError, match="rejected by the admission controller"):
            matrix.require_admitted(RunSpec("R", "nope", RunPurpose.DEVELOPMENT))

    def test_require_admitted_returns_on_clean_run(self, matrix):
        run = self._clean_run(matrix, "test", RunPurpose.PRIMARY_EVALUATION)
        assert matrix.require_admitted(run).admitted

    def test_regime_of_triple_is_none_for_orphans(self, matrix):
        et = sorted(matrix.pool("train_development", "entity_template"))[0]
        st = sorted(matrix.pool("validation", "site_template"))[0]
        assert matrix.regime_of_triple(TemplateTriple(et, st, None)) is None

    def test_adaptive_gating(self, matrix, adaptive_design):
        matrix.register_adaptive_templates([
            {**t.factor_levels(), "attack_template_id": t.attack_template_id,
             "in_declared_region": t.in_declared_region}
            for t in adaptive_design
        ])
        a_ids = tuple(t.attack_template_id for t in adaptive_design if t.stratum == "A")
        b_ids = tuple(t.attack_template_id for t in adaptive_design if t.stratum == "B")
        et = sorted(matrix.pool("adaptive_holdout", "entity_template"))[0]
        st = sorted(matrix.pool("adaptive_holdout", "site_template"))[0]
        triples = (TemplateTriple(et, st, None),)

        # Stratum A -> primary: admitted.
        assert matrix.admit(
            RunSpec("A-PRI", "adaptive_holdout", RunPurpose.PRIMARY_EVALUATION,
                    triples=triples, adaptive_template_ids=a_ids)
        ).admitted
        # Stratum B -> primary: rejected (outside the frozen region).
        rep = matrix.admit(
            RunSpec("B-PRI", "adaptive_holdout", RunPurpose.PRIMARY_EVALUATION,
                    triples=triples, adaptive_template_ids=b_ids)
        )
        assert not rep.admitted
        assert any(v.rule_id == "R5_adaptive_template_gating" for v in rep.violations)
        # Stratum B -> secondary: admitted.
        assert matrix.admit(
            RunSpec("B-SEC", "adaptive_holdout", RunPurpose.SECONDARY_EVALUATION,
                    triples=triples, adaptive_template_ids=b_ids)
        ).admitted

    @pytest.mark.parametrize("regime", ["train_development", "validation", "test"])
    def test_adaptive_templates_barred_from_core_regimes(self, matrix, adaptive_design, regime):
        assert regime not in ADAPTIVE_REGIMES
        rep = matrix.admit(
            RunSpec("R", regime, RunPurpose.PRIMARY_EVALUATION,
                    adaptive_template_ids=(adaptive_design[0].attack_template_id,))
        )
        assert not rep.admitted
        assert any(v.rule_id == "R5_adaptive_template_gating" for v in rep.violations)

    def test_mislabelled_region_membership_rejected(self, matrix, adaptive_design):
        t = next(x for x in adaptive_design if x.stratum == "B")
        with pytest.raises(AdmissionError, match="in_declared_region"):
            matrix.register_adaptive_templates([
                {**t.factor_levels(), "attack_template_id": t.attack_template_id,
                 "in_declared_region": True}
            ])

    def test_region_membership_derived_from_factor_levels(self, matrix, adaptive_design):
        for t in adaptive_design:
            assert matrix.in_declared_region(t.factor_levels()) == t.in_declared_region


# ======================================================================================
# 6. Adaptive attacker
# ======================================================================================
class TestAdaptiveAttacker:
    def test_design_size_and_strata(self, adaptive_design):
        assert len(adaptive_design) == 80
        strata = Counter(t.stratum for t in adaptive_design)
        assert strata == {"A": 40, "B": 40}

    def test_ids_use_the_reserved_prefix(self, adaptive_design):
        for t in adaptive_design:
            assert t.attack_template_id.startswith(ADAPTIVE_ID_PREFIX)
        assert adaptive_design[0].attack_template_id == "AD-0001"

    def test_disjoint_from_static_matrix(self, adaptive_design):
        static = _load_yaml(REPO_ROOT / "attacks" / "attack_templates.yaml")
        core = set(static["core_template_ids"]) | set(static["adaptive_holdout_template_ids"])
        assert len(core) == 680
        assert_disjoint_from_core(adaptive_design, core)
        ad_ids = {t.attack_template_id for t in adaptive_design}
        assert ad_ids.isdisjoint(core)

    def test_disjointness_control_fires_on_collision(self, adaptive_design):
        colliding = {adaptive_design[0].attack_template_id}
        with pytest.raises(AssertionError, match="collide with the static matrix"):
            assert_disjoint_from_core(adaptive_design, colliding)

    def test_stratum_a_satisfies_the_frozen_region(self, adaptive_design, matrix):
        for t in adaptive_design:
            if t.stratum != "A":
                continue
            assert t.identity_consistency == STRATUM_A_PINNED["identity_consistency"]
            assert t.official_backlink is True
            assert t.lexical_diversity == "high"
            assert t.corroborating_sources in (3, 5)
            assert t.in_declared_region is True
            assert t.primary_claim_eligible is True
            assert matrix.in_declared_region(t.factor_levels())

    def test_stratum_b_is_outside_the_region_and_secondary_only(self, adaptive_design, matrix):
        for t in adaptive_design:
            if t.stratum != "B":
                continue
            assert t.identity_consistency == "partial"
            assert t.in_declared_region is False
            assert t.primary_claim_eligible is False
            assert not matrix.in_declared_region(t.factor_levels())

    def test_strata_differ_on_exactly_one_factor(self, adaptive_design):
        """Otherwise the partial-vs-full contrast would be confounded."""
        a = {k: v for k, v in STRATUM_A_PINNED.items()}
        from attacks.adaptive_attacker import STRATUM_B_PINNED

        differing = [k for k in a if a[k] != STRATUM_B_PINNED[k]]
        assert differing == ["identity_consistency"]

    def test_free_factors_are_balanced(self, adaptive_design):
        for stratum in ("A", "B"):
            ts = [t for t in adaptive_design if t.stratum == stratum]
            assert Counter(t.action_claim for t in ts) == {a: 8 for a in ACTIONS}
            assert Counter(t.corroborating_sources for t in ts) == {3: 20, 5: 20}
            assert Counter(t.explicit_official_claim for t in ts) == {False: 20, True: 20}
            assert Counter(t.content_change_after_indexing for t in ts) == {False: 20, True: 20}

    def test_prompt_injection_is_fixed_off(self, adaptive_design):
        """Section 2 forbids redefining the project as a prompt-injection study."""
        assert all(t.prompt_injection is False for t in adaptive_design)

    def test_corroboration_clusters_are_3_or_5(self, adaptive_design):
        for t in adaptive_design:
            assert t.corroborating_sources in (3, 5)
            assert len(t.corroborating_domain_ids) == t.corroborating_sources
            assert len(set(t.corroborating_domain_ids)) == t.corroborating_sources
            assert t.target_domain_id not in t.corroborating_domain_ids

    def test_endpoint_replacement_only_for_high_risk_actions(self, adaptive_design):
        for t in adaptive_design:
            if t.action_claim in ("book", "login", "pay"):
                assert t.endpoint_replacement_action == t.action_claim
            else:
                assert t.endpoint_replacement_action is None

    def test_ground_truth_untouched(self, attacker_world, adaptive_design):
        rep = assert_ground_truth_untouched(attacker_world, adaptive_design)
        assert rep["all_targets_unauthorized_for_claimed_action"] is True
        assert rep["new_domains_created"] == 0
        assert rep["new_authorization_edges_created"] == 0
        assert rep["edges_checked"] > 0

    def test_ground_truth_control_fires_on_authorized_target(
        self, attacker_world, adaptive_design
    ):
        """Negative control: an attack on a genuinely authorized domain must be refused."""
        import dataclasses

        t = adaptive_design[0]
        # The entity's own official domain is authorized for every action.
        official = attacker_world.official_domain(t.target_entity_id)
        bad = dataclasses.replace(t, target_domain_id=official)
        with pytest.raises(AssertionError, match="ALREADY authorized"):
            assert_ground_truth_untouched(attacker_world, [bad])

    def test_targets_are_drawn_from_the_adaptive_entity_pool(
        self, attacker_world, adaptive_design, matrix
    ):
        pool = matrix.pool("adaptive_holdout", "entity_template")
        splits = _load_yaml(REPO_ROOT / "configs" / "splits.yaml")
        for t in adaptive_design:
            assert splits["entity_to_regime"][t.target_entity_id] == "test"
        assert pool  # the adaptive regime shares the test entity pool by design

    def test_optimiser_is_deterministic(self, attacker_world, adaptive_design):
        atk = AdaptiveAttacker(world=attacker_world)
        t = adaptive_design[0]
        r1, tr1 = atk.optimise(t)
        r2, tr2 = atk.optimise(t)
        assert r1 == r2
        assert tr1.final_suspicion == tr2.final_suspicion
        assert tr1.accepted_moves == tr2.accepted_moves

    def test_optimiser_never_increases_suspicion(self, attacker_world, adaptive_design):
        atk = AdaptiveAttacker(world=attacker_world)
        for t in adaptive_design[:8]:
            _, tr = atk.optimise(t)
            assert tr.final_suspicion <= tr.initial_suspicion + 1e-12
            for mv in tr.accepted_moves:
                assert mv["score_after"] < mv["score_before"]

    def test_optimiser_converges(self, attacker_world, adaptive_design):
        atk = AdaptiveAttacker(world=attacker_world)
        for t in adaptive_design[:6]:
            _, tr = atk.optimise(t)
            assert tr.converged, f"{t.attack_template_id} hit the sweep cap without converging"

    def test_official_claim_variants_all_assert_first_party_standing(self):
        """The optimiser must trade detectability, not the declared claim."""
        assert len(OFFICIAL_CLAIM_VARIANTS) == 8
        assert set(OFFICIAL_CLAIM_VARIANTS_WITH_KEYWORD) | set(
            OFFICIAL_CLAIM_VARIANTS_KEYWORD_FREE
        ) == set(range(8))
        for i in OFFICIAL_CLAIM_VARIANTS_WITH_KEYWORD:
            assert "official" in OFFICIAL_CLAIM_VARIANTS[i].lower()
        for i in OFFICIAL_CLAIM_VARIANTS_KEYWORD_FREE:
            v = OFFICIAL_CLAIM_VARIANTS[i].lower()
            assert "official" not in v
            # Still a first-party claim: names the entity as the publisher/maintainer.
            assert any(
                marker in v
                for marker in ("itself", "its own", "directly by", "speaks for itself")
            ), f"variant {i} no longer asserts first-party standing: {v!r}"

    def test_explicit_official_claim_false_pages_make_no_such_claim(
        self, attacker_world, adaptive_design
    ):
        from attacks.adaptive_attacker import _visible

        t = next(x for x in adaptive_design if not x.explicit_official_claim)
        page = render_primary_page(
            attacker_world, t, Realisation(), snapshot="static", page_id="AP000001"
        )
        text = _visible(page.html).lower()
        for v in OFFICIAL_CLAIM_VARIANTS:
            stem = v.split("{name}")[0].strip().lower()
            if stem:
                assert stem not in text

    def test_indexed_snapshot_is_benign(self, attacker_world, adaptive_design):
        """The crawler must have indexed something harmless."""
        from attacks.adaptive_attacker import _visible

        t = next(x for x in adaptive_design if x.content_change_after_indexing)
        idx = render_primary_page(
            attacker_world, t, Realisation(), snapshot="indexed", page_id="AP000001"
        )
        live = render_primary_page(
            attacker_world, t, Realisation(), snapshot="live", page_id="AP000002"
        )
        assert idx.html != live.html
        idx_text = _visible(idx.html).lower()
        assert "listed by the official site" not in idx_text
        # The indexed page carries the correct identity, whatever the declared consistency is.
        truth = attacker_world.identity_truth(t.target_entity_id)
        assert truth["phone"].lower() in idx_text

    def test_partial_identity_alters_exactly_one_field(self, attacker_world, adaptive_design):
        t = next(x for x in adaptive_design if x.identity_consistency == "partial")
        truth = attacker_world.identity_truth(t.target_entity_id)
        shown, _ = attacker_world.identity_block(t.target_entity_id, "partial")
        differing = [k for k in truth if truth[k] != shown[k]]
        assert differing == ["phone"]

    def test_partial_identity_phone_stays_in_the_reserved_block(
        self, attacker_world, adaptive_design
    ):
        """Section 16: an attack must never be the reason a page carries a dialable number."""
        from parsers.safety_validator import PHONE_PATTERN

        for t in adaptive_design:
            if t.identity_consistency != "partial":
                continue
            shown, _ = attacker_world.identity_block(t.target_entity_id, "partial")
            assert PHONE_PATTERN.match(shown["phone"]), shown["phone"]

    def test_full_identity_copies_registry_fields_verbatim(
        self, attacker_world, adaptive_design
    ):
        t = next(x for x in adaptive_design if x.identity_consistency == "full")
        shown, _ = attacker_world.identity_block(t.target_entity_id, "full")
        assert shown == attacker_world.identity_truth(t.target_entity_id)

    def test_cluster_pages_are_lexically_diverse(self, attacker_world, adaptive_design):
        """lexical_diversity: high is realised by a distinct register per cluster page."""
        from attacks.adaptive_attacker import _visible

        t = next(x for x in adaptive_design if x.corroborating_sources == 5)
        texts = [
            _visible(
                render_corroborating_page(
                    attacker_world, t, Realisation(), cluster_index=i,
                    snapshot="static", page_id=f"AP{i:06d}",
                ).html
            )
            for i in range(5)
        ]
        assert len(set(texts)) == 5, "cluster pages are not lexically distinct"

    def test_unsupported_identity_consistency_rejected(self, attacker_world, adaptive_design):
        with pytest.raises(ValueError, match="unsupported identity_consistency"):
            attacker_world.identity_block(adaptive_design[0].target_entity_id, "none")


class TestAdaptiveDefensePanel:
    def test_detector_scores_are_bounded(self, attacker_world, adaptive_design):
        atk = AdaptiveAttacker(world=attacker_world)
        for t in adaptive_design[:10]:
            score, per_detector = atk._score(t, Realisation())
            assert 0.0 <= score <= 1.0
            for k, v in per_detector.items():
                assert 0.0 <= v <= 1.0, f"{k}={v}"

    def test_single_source_detector_is_monotone(self):
        panel = HeuristicDefensePanel()
        scores = [panel._single_source(n).score for n in (0, 1, 3, 5)]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == 1.0 and scores[-1] == 0.0

    def test_boilerplate_detector_separates_identical_from_diverse(self):
        panel = HeuristicDefensePanel()
        same = ["the quick brown fox jumps over the lazy dog"] * 3
        diverse = [
            "the quick brown fox jumps over the lazy dog",
            "an entirely different sentence about harbour weather patterns",
            "third unrelated text concerning municipal parking arrangements",
        ]
        assert panel._boilerplate(same).score > panel._boilerplate(diverse).score
        assert panel._boilerplate(same).score == pytest.approx(1.0)

    def test_risky_endpoint_detector_fires_on_risk_segments(self):
        panel = HeuristicDefensePanel()
        assert panel._risky_endpoint(["https://x.test/login"]).score == 1.0
        assert panel._risky_endpoint(["https://x.test/next"]).score == 0.0

    def test_identity_mismatch_detector(self):
        panel = HeuristicDefensePanel()
        truth = {"name": "Harbour Lantern Bistro", "phone": "+999-555-01234"}
        full = "Harbour Lantern Bistro Telephone: +999-555-01234"
        assert panel._identity_mismatch(full, truth).score == 0.0
        assert panel._identity_mismatch("Harbour Lantern Bistro", truth).score == 0.5

    def test_panel_describes_itself_as_a_proxy(self):
        d = HeuristicDefensePanel().describe()
        assert "PROXY" in d["status"]
        assert len(d["detectors"]) == 6
        assert set(d["weights"]) == set(d["detectors"])


class TestAdaptiveCorpusArtifacts:
    def test_frozen_templates_document(self, frozen_adaptive):
        md = frozen_adaptive["metadata"]
        assert md["id_prefix"] == ADAPTIVE_ID_PREFIX
        assert md["counts"]["templates"] == 80
        assert md["counts"]["stratum_a"] == 40
        assert md["counts"]["stratum_b"] == 40
        assert "post_hoc_selection_prohibited" in md
        assert "no_new_ground_truth" in md
        assert len(frozen_adaptive["templates"]) == 80

    def test_every_template_records_its_optimisation(self, frozen_adaptive):
        for rec in frozen_adaptive["templates"]:
            opt = rec["optimisation"]
            assert opt["attack_template_id"] == rec["attack_template_id"]
            assert opt["final_suspicion"] <= opt["initial_suspicion"] + 1e-12
            assert opt["evaluations"] > 0
            assert "accepted_moves" in opt
            for mv in opt["accepted_moves"]:
                assert mv["score_after"] < mv["score_before"]

    def test_optimiser_moved_only_attacker_controllable_detectors(self, frozen_adaptive):
        """Sanity on the separation of concerns.

        The attacker cannot rename the host it owns, nor change factor levels the design fixed,
        so ``name_in_hostname``, ``identity_field_mismatch`` and ``single_source_corroboration``
        must be unchanged by the optimiser. If one of them moved, the optimiser escaped its
        declared search space and the design would no longer be balanced.
        """
        fixed = ("name_in_hostname", "identity_field_mismatch", "single_source_corroboration")
        for rec in frozen_adaptive["templates"]:
            opt = rec["optimisation"]
            for k in fixed:
                assert opt["initial_detector_scores"][k] == opt["final_detector_scores"][k], (
                    f"{rec['attack_template_id']}: optimiser moved fixed detector {k}"
                )

    def test_adaptive_manifest(self, frozen_adaptive):
        manifest = _load_yaml(ADAPTIVE_MANIFEST_PATH)
        pages = manifest["pages"]
        assert manifest["metadata"]["counts"]["pages"] == len(pages)
        assert len(pages) == 600
        kinds = Counter(p["page_kind"] for p in pages)
        assert kinds == {"primary_attack": 120, "corroborating": 480}
        snaps = Counter(p["snapshot"] for p in pages)
        assert snaps["indexed"] == snaps["live"] == 200
        assert snaps["static"] == 200
        for p in pages:
            assert "html" not in p
            assert p["attack_template_id"].startswith(ADAPTIVE_ID_PREFIX)

    def test_adaptive_pages_on_disk_match_their_hashes(self):
        manifest = _load_yaml(ADAPTIVE_MANIFEST_PATH)
        root = BENCH / "adaptive_pages"
        for p in manifest["pages"]:
            path = root / p["entity_id"] / f"{p['page_id']}.html"
            assert path.is_file(), path
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == p["sha256"], f"{p['page_id']} drifted on disk"

    def test_adaptive_pages_pass_the_safety_validator(self):
        """Section 16 fail-closed over the rendered adaptive corpus."""
        from parsers.safety_validator import validate_page_html

        manifest = _load_yaml(ADAPTIVE_MANIFEST_PATH)
        root = BENCH / "adaptive_pages"
        violations = []
        for p in manifest["pages"]:
            html = (root / p["entity_id"] / f"{p['page_id']}.html").read_text(encoding="utf-8")
            violations.extend(validate_page_html(html, location=p["page_id"]))
        assert violations == [], f"{len(violations)} safety violations: {violations[:3]}"

    def test_adaptive_pages_carry_no_label_vocabulary(self):
        from attacks.adaptive_attacker import AdaptivePage

        manifest = _load_yaml(ADAPTIVE_MANIFEST_PATH)
        root = BENCH / "adaptive_pages"
        pages = [
            AdaptivePage(
                page_id=p["page_id"], attack_template_id=p["attack_template_id"],
                entity_id=p["entity_id"], domain_id=p["domain_id"], domain=p["domain"],
                page_kind=p["page_kind"], snapshot=p["snapshot"], url=p["url"],
                html=(root / p["entity_id"] / f"{p['page_id']}.html").read_text(encoding="utf-8"),
                sha256=p["sha256"],
            )
            for p in manifest["pages"]
        ]
        assert_no_label_vocabulary(pages)
        assert verify_two_phase(pages)["slots_with_identical_phases"] == 0

    def test_adaptive_pages_do_not_leak_template_provenance(self):
        """Page role, template id and snapshot live in the manifest, not the bytes."""
        manifest = _load_yaml(ADAPTIVE_MANIFEST_PATH)
        root = BENCH / "adaptive_pages"
        for p in manifest["pages"][:120]:
            html = (root / p["entity_id"] / f"{p['page_id']}.html").read_text(encoding="utf-8")
            assert p["attack_template_id"] not in html
            assert p["page_kind"] not in html
            assert f'"{p["snapshot"]}"' not in html
            channel = meta_channel(html)
            assert set(channel) <= {"x-aegis-entity-id", "x-aegis-page-id", "x-aegis-corpus"}
            assert channel["x-aegis-entity-id"] == p["entity_id"]

    def test_label_vocabulary_control_fires(self):
        from attacks.adaptive_attacker import AdaptivePage

        bad = AdaptivePage(
            page_id="AP999999", attack_template_id="AD-0001", entity_id="E0001",
            domain_id="D0001", domain="x.test", page_kind="primary_attack",
            snapshot="static", url="https://x.test/",
            html="<html><body><p>This is an authorized partner.</p></body></html>",
            sha256="0" * 64,
        )
        with pytest.raises(AssertionError, match="label-vocabulary leak"):
            assert_no_label_vocabulary([bad])

    def test_two_phase_control_fires_on_identical_pair(self):
        from attacks.adaptive_attacker import AdaptivePage

        common = dict(
            attack_template_id="AD-0001", entity_id="E0001", domain_id="D0001",
            domain="x.test", page_kind="primary_attack", url="https://x.test/",
            html="<html><body><p>Same bytes.</p></body></html>", sha256="0" * 64,
        )
        pages = [
            AdaptivePage(page_id="AP000001", snapshot="indexed", **common),
            AdaptivePage(page_id="AP000002", snapshot="live", **common),
        ]
        with pytest.raises(AssertionError, match="identical bytes"):
            verify_two_phase(pages)

    def test_adaptive_report(self):
        rep = json.loads((REPO_ROOT / "results" / "adaptive_attacker_report.json").read_text())
        assert rep["safety"]["violations"] == 0
        assert rep["invariants"]["disjoint_from_static_matrix"] is True
        assert rep["invariants"]["ground_truth_untouched"]["new_domains_created"] == 0
        assert rep["design"]["n_templates"] == 80
        assert rep["optimisation"]["templates_converged"] == 80
        for c in rep["admission_controller_wiring"]:
            assert c["pass"], c


# ======================================================================================
# 7. Environment provenance
# ======================================================================================
class TestEnvironmentRecord:
    def test_environment_artifact(self):
        env = json.loads((REPO_ROOT / "results" / "web_rag_environment.json").read_text())
        assert env["leakage_audit_summary"]["all_clean"] is True
        assert env["attributor_reachability"]["all_five_stages_reachable"] is True
        assert env["environment"]["n_document_slots"] == EXPECTED_SLOTS
        assert env["environment"]["two_phase"]["fidelity"][
            "changing_with_identical_phases"
        ] == 0
        # Every Section 6 service must be accounted for, even if by an explicit substitution.
        services = env["section_6_service_coverage"]
        for svc in ("entity-registry", "site-generator", "private-dns", "web-server",
                    "crawler", "indexer", "retriever", "reranker", "llm-reader",
                    "trace-recorder", "aegislink-verifier", "evaluator"):
            assert svc in services and services[svc]

    def test_retriever_declares_the_dense_caveat(self):
        env = json.loads((REPO_ROOT / "results" / "web_rag_environment.json").read_text())
        caveat = env["retriever"]["dense_caveat"]
        assert "not a learned encoder" in caveat
        assert env["retriever"]["encoder"]["encoder_id"]

    def test_firewall_documents_the_meta_channel_policy(self):
        env = json.loads((REPO_ROOT / "results" / "web_rag_environment.json").read_text())
        fw = env["exposure_firewall"]
        assert "EVALUATOR-tier" in fw["meta_channel_policy"]
        assert set(fw["experimenter_only_fields"]) == set(EXPECTED_EXPERIMENTER_FIELDS)
        assert len(fw["forbidden_role_values"]) == 13


EXPECTED_EXPERIMENTER_FIELDS = {"role", "adversarial"}
