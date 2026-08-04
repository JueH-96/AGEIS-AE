"""Step 3 driver: build the Web-RAG index, run retrieval, record traces, freeze snapshots.

CONTRACT.md Sections 3 (RQ2), 6.

Outputs
-------
``data/benchmark/retrieval_snapshots.json``
    The frozen replay: per-query candidate lists with a SHA-256 each, plus a global
    ``replay_fingerprint`` over corpus, index, retriever config, encoder id and query set.
``data/benchmark/stage_traces.json``
    The five RQ2 stages per query with the failure origin attributed programmatically.
``results/web_rag_environment.json``
    Provenance: crawler/indexer/retriever/recorder descriptions, exposure firewall, leakage
    audit, two-phase fidelity, admission-controller description.
``results/leakage_audit.json``
    The zero-leakage sweep over every public artifact.

Run::

    uv run python workflow/12_build_web_rag_index_and_snapshots.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from web_rag.crawler_indexer import (  # noqa: E402
    PHASE_INDEX,
    PHASE_LIVE,
    build_index,
    crawl_corpus,
    describe_environment,
)
from web_rag.exposure import (  # noqa: E402
    ALL_FORBIDDEN_FIELDS,
    DOMAIN_ROLE_VALUES,
    ExposureTier,
    LeakageError,
    assert_no_leakage,
    describe_firewall,
    public_delegation_evidence,
)
from web_rag.experiment_matrix import ExperimentMatrix  # noqa: E402
from web_rag.retriever import (  # noqa: E402
    HashedNGramEncoder,
    Retriever,
    RetrieverConfig,
    build_entity_queries,
)
from web_rag.trace_recorder import (  # noqa: E402
    TraceRecorder,
    assert_all_stages_reachable,
    build_oracle_and_pipeline,
    describe_recorder,
    summarize_traces,
    validate_trace_schema,
)

BENCH = ROOT / "data" / "benchmark"
RESULTS = ROOT / "results"
SNAPSHOT_PATH = BENCH / "retrieval_snapshots.json"
TRACES_PATH = BENCH / "stage_traces.json"

RETRIEVAL_MODE = "hybrid_rrf"
TOP_K = 10


def load_yaml(path: Path) -> Any:
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    return yaml.load(path.read_text(encoding="utf-8"), Loader=loader)


def sweep_for_leakage(payload: Any, *, location: str) -> dict[str, Any]:
    """Independent, brute-force leakage sweep over a serialised artifact.

    The whitelist projections and the runtime guard are the primary defenses. This is the third,
    deliberately dumb check: serialise to JSON text and search for every forbidden key name and
    every role value. A guard bug cannot hide from it because it does not use the guard.
    """
    text = json.dumps(payload, sort_keys=True)
    key_hits = sorted(k for k in ALL_FORBIDDEN_FIELDS if f'"{k}":' in text)
    value_hits = sorted(v for v in DOMAIN_ROLE_VALUES if f'"{v}"' in text)
    guard_error: str | None = None
    try:
        assert_no_leakage(payload, location=location)
    except LeakageError as exc:  # pragma: no cover - a failure here is a hard stop
        guard_error = str(exc)
    return {
        "location": location,
        "bytes_scanned": len(text),
        "forbidden_key_hits": key_hits,
        "forbidden_role_value_hits": value_hits,
        "guard_error": guard_error,
        "clean": not key_hits and not value_hits and guard_error is None,
    }


def main() -> int:
    t0 = time.time()
    print("=" * 78)
    print("Step 3 -- Web-RAG replay environment: index, retrieval, traces, freeze")
    print("=" * 78)

    # ---------------------------------------------------------------- 1. crawl + index
    print("\n[1/7] Crawling the frozen corpus (two-phase)")
    corpus = crawl_corpus(verify_hashes=True)
    print(f"[1/7] Building the phase-1 index (phase={PHASE_INDEX})")
    index = build_index(corpus, phase=PHASE_INDEX)

    # ---------------------------------------------------------------- 2. queries
    print("\n[2/7] Building the frozen query set")
    entities = load_yaml(ROOT / "registry" / "entities.yaml")["entities"]
    queries = build_entity_queries(entities)
    query_digest = hashlib.sha256(
        json.dumps([q.as_dict() for q in queries], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    print(f"[2/7] {len(queries)} queries over {len(entities)} entities "
          f"(digest {query_digest[:16]}...)")

    # ---------------------------------------------------------------- 3. retrieval
    print(f"\n[3/7] Retrieval (mode={RETRIEVAL_MODE}, top_k={TOP_K})")
    config = RetrieverConfig(mode=RETRIEVAL_MODE, top_k=TOP_K)
    encoder = HashedNGramEncoder()
    retriever = Retriever(index, config=config, encoder=encoder)
    results = retriever.search_all(queries, progress_every=200)
    print(f"[3/7] {len(results)} results, "
          f"{sum(len(r.candidates) for r in results)} candidates total")

    # ---------------------------------------------------------------- 4. traces
    print("\n[4/7] Recording RQ2 stage traces (read phase = live)")
    domains_raw = load_yaml(ROOT / "registry" / "domains.yaml")
    graph_edges = load_yaml(ROOT / "registry" / "authorization_graph.yaml")["edges"]
    aliases = load_yaml(ROOT / "registry" / "aliases.yaml")["aliases"]
    oracle, pipeline = build_oracle_and_pipeline(
        corpus,
        graph_edges=graph_edges,
        delegations=domains_raw["delegations"],
        entities=entities,
        aliases=aliases,
        read_phase=PHASE_LIVE,
    )
    recorder = TraceRecorder(corpus=corpus, oracle=oracle, read_phase=PHASE_LIVE)
    traces = recorder.record_all(results, pipeline, progress_every=200)
    trace_summary = summarize_traces(traces)
    for t in traces:
        validate_trace_schema(t.as_dict())
    print(f"[4/7] {len(traces)} traces validated against the RQ2 schema")
    print(f"[4/7] failure origins: {json.dumps(trace_summary['failure_origin_counts'])}")
    print(f"[4/7] scope reasons  : {json.dumps(trace_summary['attribution_scope_reasons'])}")
    # Prove the attributor is not degenerate: it must be able to name every one of the five
    # stages. This is checked here rather than assumed because the first implementation of the
    # comparison rule WAS degenerate (see web_rag/trace_recorder.py module docstring).
    reachability = assert_all_stages_reachable(recorder, results)
    print(f"[4/7] all five RQ2 stages reachable by the attributor: "
          f"{json.dumps(reachability['witness_query_per_stage'])}")

    # ---------------------------------------------------------------- 5. freeze
    print("\n[5/7] Freezing retrieval snapshots")
    per_query = [r.as_dict(ExposureTier.READER) for r in results]
    # Addressing (entity_id, action) is experiment bookkeeping the evaluator needs to join a
    # query to a ground-truth cell. Kept in a SEPARATE block from the candidate payload so the
    # reader-facing part of the file is exactly what a reader may see.
    query_index = [
        {
            "query_id": r.query_id,
            "entity_id": r.entity_id,
            "action": r.action,
            "intent": r.intent,
            "sha256": r.sha256,
        }
        for r in results
    ]
    fingerprint_material = {
        "corpus_digest": corpus.corpus_digest,
        "index_digest": index.index_digest,
        "retriever_config_digest": config.digest(),
        "encoder_id": encoder.encoder_id,
        "query_set_digest": query_digest,
        "per_query_sha256": sorted(r.sha256 for r in results),
    }
    replay_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    snapshot_doc = {
        "metadata": {
            "schema_version": "1.0",
            "builder": "workflow/12_build_web_rag_index_and_snapshots.py",
            "contract_ref": (
                "CONTRACT.md Section 6: 'Store page snapshots and SHA-256 hashes. Freeze "
                "retrieval results for primary experiments.'"
            ),
            "purpose": (
                "Frozen retrieval replay. Steps 4-8 compare ten baselines and the defense on "
                "identical evidence; without a freeze, a measured difference between two "
                "defenses would be confounded with a difference in what they were shown."
            ),
            "replay_fingerprint": replay_fingerprint,
            "fingerprint_material": {
                k: v for k, v in fingerprint_material.items() if k != "per_query_sha256"
            },
            "fingerprint_note": (
                "The fingerprint covers the corpus bytes, the index, the retriever config, the "
                "encoder id and the query set. Any drift in any of them changes it, so a stale "
                "replay fails loudly instead of silently mixing evidence."
            ),
            "integrity": {
                "per_query_hash": (
                    "SHA-256 over the canonical JSON of the ORDERED candidate list "
                    "(rank, doc_id, url, score, bm25_score, dense_score). Snippet text is "
                    "excluded: it is a derived view of page bytes already covered by "
                    "corpus_digest."
                ),
                "n_queries": len(results),
                "n_distinct_query_hashes": len({r.sha256 for r in results}),
            },
            "two_phase": {
                "index_phase": PHASE_INDEX,
                "read_phase": PHASE_LIVE,
                "note": (
                    "Candidates were ranked on phase-1 (indexed) bytes. A reader replaying this "
                    "snapshot must fetch phase-2 (live) bytes, which is what makes the Section 7 "
                    "content_change_after_indexing factor bite."
                ),
            },
            "counts": {
                "queries": len(results),
                "candidates": sum(len(r.candidates) for r in results),
                "document_slots": len(corpus),
                "manifest_pages": corpus.n_pages,
            },
        },
        "environment": describe_environment(corpus, index),
        "retriever": retriever.describe(),
        "exposure_firewall": describe_firewall(),
        "query_index": query_index,
        "snapshots": per_query,
    }

    # The reader-facing part of the file must survive the firewall.
    audits = [
        sweep_for_leakage(per_query, location="retrieval_snapshots.snapshots"),
        sweep_for_leakage(
            [t.as_dict() for t in traces],
            location="stage_traces (EVALUATOR tier -- expected to carry expectations only)",
        ),
        sweep_for_leakage(
            {k: v.as_dict() for k, v in corpus.registry.domains.items()},
            location="public_domain_registry",
        ),
        sweep_for_leakage(
            {k: v.as_dict() for k, v in corpus.registry.lifecycle.items()},
            location="derived_lifecycle_observations",
        ),
        sweep_for_leakage(public_delegation_evidence(), location="public_delegation_evidence"),
        sweep_for_leakage(
            [corpus.documents[d].public_view(PHASE_INDEX, ExposureTier.RETRIEVER)
             for d in corpus.doc_ids[:200]],
            location="retriever_tier_document_views (first 200)",
        ),
        sweep_for_leakage(
            {k: v.as_dict() for k, v in list(index.documents.items())[:200]},
            location="indexed_documents (first 200)",
        ),
    ]

    SNAPSHOT_PATH.write_text(
        json.dumps(snapshot_doc, indent=1, sort_keys=False) + "\n", encoding="utf-8"
    )
    snapshot_sha = hashlib.sha256(SNAPSHOT_PATH.read_bytes()).hexdigest()
    print(f"[5/7] wrote {SNAPSHOT_PATH.name} "
          f"({SNAPSHOT_PATH.stat().st_size / 1e6:.2f} MB, sha {snapshot_sha[:16]}...)")

    trace_doc = {
        "metadata": {
            "schema_version": "1.0",
            "builder": "workflow/12_build_web_rag_index_and_snapshots.py",
            "contract_ref": "CONTRACT.md Section 3 (RQ2)",
            "recorder": describe_recorder(),
            "pipeline_id": pipeline.pipeline_id,
            "replay_fingerprint": replay_fingerprint,
            "attributor_reachability": reachability,
            "counts": {"traces": len(traces)},
        },
        "summary": trace_summary,
        "traces": [t.as_dict() for t in traces],
    }
    TRACES_PATH.write_text(json.dumps(trace_doc, indent=1) + "\n", encoding="utf-8")
    print(f"[5/7] wrote {TRACES_PATH.name} ({TRACES_PATH.stat().st_size / 1e6:.2f} MB)")

    # ---------------------------------------------------------------- 6. leakage audit
    print("\n[6/7] Leakage audit")
    audits.append(
        sweep_for_leakage(
            json.loads(SNAPSHOT_PATH.read_text())["snapshots"],
            location="retrieval_snapshots.json (round-tripped from disk)",
        )
    )
    dirty = [a for a in audits if not a["clean"]]
    for a in audits:
        flag = "OK  " if a["clean"] else "FAIL"
        print(f"   [{flag}] {a['location']} ({a['bytes_scanned']} bytes)")
        if not a["clean"]:
            print(f"          keys={a['forbidden_key_hits']} "
                  f"values={a['forbidden_role_value_hits']} guard={a['guard_error']}")
    audit_doc = {
        "metadata": {
            "contract_ref": "CONTRACT.md Section 5.2; registry/domains.yaml metadata",
            "question": (
                "Are role, adversarial and lifecycle.prior_grants provably absent from every "
                "artifact the indexer, retriever, reader or verifier can see?"
            ),
            "method": (
                "Three independent mechanisms: (1) whitelist projections, so a new registry "
                "field is excluded by default; (2) a recursive guard on field names and enum "
                "values applied at every public return; (3) this sweep, which serialises each "
                "artifact and searches for all forbidden keys and all 13 role values WITHOUT "
                "using the guard, so a guard bug cannot hide from it."
            ),
            "forbidden_fields": sorted(ALL_FORBIDDEN_FIELDS),
            "forbidden_role_values": list(DOMAIN_ROLE_VALUES),
            "all_clean": not dirty,
        },
        "audits": audits,
    }
    (RESULTS / "leakage_audit.json").write_text(json.dumps(audit_doc, indent=2) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- 7. environment record
    print("\n[7/7] Writing environment provenance and admission-controller description")
    matrix = ExperimentMatrix.from_splits()
    env_doc = {
        "metadata": {
            "step": "Step 3 of 8 -- Controlled Web-RAG Replay Environment & Adaptive Attacker",
            "contract_ref": "CONTRACT.md Sections 3 (RQ2), 6, 7",
            "builder": "workflow/12_build_web_rag_index_and_snapshots.py",
            "replay_fingerprint": replay_fingerprint,
            "retrieval_snapshots_sha256": snapshot_sha,
            "elapsed_seconds": round(time.time() - t0, 2),
        },
        "environment": describe_environment(corpus, index),
        "retriever": retriever.describe(),
        "trace_recorder": describe_recorder(),
        "attributor_reachability": reachability,
        "trace_summary": trace_summary,
        "exposure_firewall": describe_firewall(),
        "leakage_audit_summary": {
            "n_artifacts_swept": len(audits),
            "all_clean": not dirty,
            "dirty": [a["location"] for a in dirty],
        },
        "experiment_matrix": matrix.describe(),
        "section_6_service_coverage": {
            "entity-registry": "registry/entities.yaml, registry/domains.yaml (Step 2)",
            "site-generator": "site_generator/generator.py (Step 2)",
            "private-dns": "not applicable: no network is used; hostnames are .test string keys",
            "web-server": "replaced by offline replay from frozen page bytes",
            "crawler": "web_rag/crawler_indexer.py::crawl_corpus (two-phase)",
            "indexer": "web_rag/crawler_indexer.py::build_index (phase-1 only)",
            "retriever": "web_rag/retriever.py::Retriever (BM25 + dense sketch)",
            "reranker": "web_rag/retriever.py (weighted RRF fusion)",
            "llm-reader": "Step 4 (pinned open-weight); surrogate in trace_recorder for now",
            "trace-recorder": "web_rag/trace_recorder.py::TraceRecorder",
            "aegislink-verifier": "Step 4",
            "evaluator": "Step 5",
            "note": (
                "Section 6 asks for Docker Compose with pinned image digests. Because this "
                "environment is a fully offline replay over frozen bytes -- no server, no DNS, "
                "no network -- the reproducibility guarantee is carried by the corpus digest and "
                "the replay fingerprint rather than by image pinning. Container packaging is a "
                "Step 7 deliverable and will pin the Python base image and lockfile."
            ),
        },
    }
    (RESULTS / "web_rag_environment.json").write_text(
        json.dumps(env_doc, indent=2) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 78)
    if dirty:
        print(f"FAILED: {len(dirty)} artifact(s) leaked experimenter-only data")
        return 1
    print(f"Step 3 web-RAG environment built in {time.time() - t0:.1f}s")
    print(f"  document slots      : {len(corpus)} (from {corpus.n_pages} pages)")
    print(f"  two-phase pairs     : {len(corpus.changing_doc_ids())}")
    print(f"  index vocabulary    : {index.vocabulary_size}")
    print(f"  queries frozen      : {len(results)}")
    print(f"  traces recorded     : {len(traces)}")
    print(f"  replay fingerprint  : {replay_fingerprint}")
    print(f"  leakage audit       : {len(audits)} artifacts, all clean")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
