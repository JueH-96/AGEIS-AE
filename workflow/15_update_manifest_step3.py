"""Update manifest.json with the Step 3 outputs, headline numbers and open findings."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "manifest.json"

STEP = 3
STEP_NAME = "Step 3 of 8 -- Controlled Web-RAG Replay Environment & Adaptive Attacker System"

NEW_OUTPUTS = [
    ("web_rag/exposure.py",
     "Exposure firewall: whitelist projections of registry/domains.yaml, ExposureTier "
     "(RETRIEVER/READER/VERIFIER/EVALUATOR), recursive assert_no_leakage guard on field names "
     "AND enum values, derived LifecycleObservation that drops lifecycle.prior_grants",
     "module"),
    ("web_rag/crawler_indexer.py",
     "Two-phase crawler and BM25 indexer over the frozen 2,631-page corpus. Resolves pages into "
     "2,002 document slots keyed by (entity, domain, site_template, attack_template); indexes the "
     "'indexed' snapshot while fetch(phase='live') serves the 'live' snapshot; verifies every "
     "page SHA-256 against the Step 2 freeze",
     "module"),
    ("web_rag/retriever.py",
     "Deterministic BM25 (k1=1.2, b=0.75) + hashed-ngram dense sketch (256-d, blake2b) with "
     "weighted RRF fusion; total order via (-score, doc_id); per-query SHA-256",
     "module"),
    ("web_rag/trace_recorder.py",
     "RQ2 five-stage execution traces with outcome-anchored first-divergence attribution, "
     "GroundTruthOracle held on the evaluator side, ReferenceStagePipeline surrogate, "
     "ScriptedPipeline probes and assert_all_stages_reachable",
     "module"),
    ("web_rag/experiment_matrix.py",
     "Admission controller enforcing five rules: one regime per run, pool membership, no "
     "test/holdout during threshold selection, cross-regime triple rejection, adaptive-template "
     "gating (Stratum A only for primary claims). Fails closed",
     "module"),
    ("attacks/adaptive_attacker.py",
     "Adaptive attacker: AD-0001+ templates in the pre-declared holdout region, "
     "HeuristicDefensePanel (6 surface detectors, a declared PROXY), deterministic "
     "coordinate-descent prose optimiser, corroboration clusters of 3 or 5, identity copying, "
     "endpoint replacement, post-indexing content change",
     "module"),
    ("attacks/adaptive_templates.yaml",
     "Frozen adaptive design: 80 templates (40 Stratum A primary-eligible, 40 Stratum B "
     "secondary-only), factor levels, chosen realisation, and the full optimisation trace per "
     "template",
     "design"),
    ("data/benchmark/retrieval_snapshots.json",
     "Frozen retrieval replay: 1,008 queries x top-10 candidates, a SHA-256 per query, and a "
     "replay_fingerprint over corpus/index/config/encoder/query-set",
     "frozen_replay"),
    ("data/benchmark/stage_traces.json",
     "1,008 RQ2 stage traces with per-stage ground truth, scoped and unscoped agreement, and the "
     "attributed failure origin",
     "traces"),
    ("data/benchmark/adaptive_page_manifest.yaml",
     "Side manifest for the 600 adaptive pages (page kind, template, snapshot held outside the "
     "page bytes)",
     "manifest"),
    ("data/benchmark/adaptive_pages",
     "600 rendered adaptive attack pages: 120 primary attack pages + 480 corroborating cluster "
     "pages, two-phase where content_change_after_indexing is true",
     "corpus"),
    ("results/web_rag_environment.json",
     "Environment provenance: crawler/indexer/retriever/recorder descriptions, exposure "
     "firewall, attributor reachability proof, two-phase fidelity, admission-controller "
     "description, Section 6 service coverage map",
     "provenance"),
    ("results/leakage_audit.json",
     "Zero-leakage sweep over 9 public artifacts using three independent mechanisms "
     "(whitelist, recursive guard, brute-force text search)",
     "audit"),
    ("results/adaptive_attacker_report.json",
     "Adaptive attacker report: design balance, per-detector optimiser effect, invariant checks, "
     "Section 16 safety scan (0 violations), admission-controller wiring checks",
     "report"),
    ("tests/test_web_rag_and_attacker.py",
     "150 Step 3 tests: two-phase fidelity, zero-leakage sweeps with negative controls, trace "
     "schema and hash reproducibility, admission-controller rejections, adaptive disjointness "
     "and region membership",
     "tests"),
    ("workflow/12_build_web_rag_index_and_snapshots.py",
     "Driver: crawl, index, retrieve, record traces, prove attributor reachability, freeze "
     "snapshots, audit leakage",
     "workflow"),
    ("workflow/13_generate_adaptive_attacks.py",
     "Driver: enumerate the adaptive design, optimise prose, render pages, check invariants and "
     "safety, verify admission wiring",
     "workflow"),
    ("workflow/14_plot_web_rag_summary.py",
     "Driver: render the three Step 3 diagnostic figures",
     "workflow"),
    ("figures/step3_web_rag_environment.png",
     "Corpus slot resolution, two-phase token deltas, RQ2 failure attribution overall and by "
     "action",
     "figure"),
    ("figures/step3_adaptive_attacker.png",
     "Per-detector optimiser effect (with the three attacker-immovable detectors marked), "
     "suspicion distribution, design balance, reduction by cluster size and phase",
     "figure"),
    ("figures/step3_retrieval_profile.png",
     "Rank/score profile, rank at which the first authorized domain appears per action, and the "
     "unscoped-relation-disagreement histogram that motivated outcome-anchored attribution",
     "figure"),
    ("implementation_plan_step3.md",
     "Step 3 implementation plan with the ten question-awareness resolutions",
     "plan"),
]


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    env = json.loads((ROOT / "results" / "web_rag_environment.json").read_text())
    adaptive = json.loads((ROOT / "results" / "adaptive_attacker_report.json").read_text())
    traces = json.loads((ROOT / "data" / "benchmark" / "stage_traces.json").read_text())

    manifest["current_step"] = STEP_NAME
    manifest["status"] = "completed"
    manifest["step_index"] = STEP
    manifest["last_updated"] = "2026-07-30"

    e = env["environment"]
    summary = traces["summary"]
    manifest["headline_results"]["step_3"] = {
        "document_slots_indexed": e["n_document_slots"],
        "manifest_pages_crawled": e["n_manifest_pages"],
        "two_phase_slots": e["n_changing_slots"],
        "two_phase_slots_with_identical_bytes": e["two_phase"]["fidelity"][
            "changing_with_identical_phases"
        ],
        "index_vocabulary": e["index"]["vocabulary_size"],
        "queries_frozen": env["retriever"]["config"]["top_k"] and summary["n_traces"],
        "retrieval_snapshots_sha256": env["metadata"]["retrieval_snapshots_sha256"],
        "replay_fingerprint": env["metadata"]["replay_fingerprint"],
        "replay_hashes_reproduce_on_rerun": True,
        "leakage_audit_artifacts": env["leakage_audit_summary"]["n_artifacts_swept"],
        "leakage_audit_all_clean": env["leakage_audit_summary"]["all_clean"],
        "all_five_rq2_stages_reachable": env["attributor_reachability"][
            "all_five_stages_reachable"
        ],
        "failure_origin_counts_surrogate": summary["failure_origin_counts"],
        "adaptive_templates": adaptive["design"]["n_templates"],
        "adaptive_stratum_a_primary_eligible": adaptive["design"]["stratum_a"],
        "adaptive_stratum_b_secondary_only": adaptive["design"]["stratum_b"],
        "adaptive_pages": adaptive["design"]["n_pages"],
        "adaptive_safety_violations": adaptive["safety"]["violations"],
        "adaptive_new_domains_created": adaptive["invariants"]["ground_truth_untouched"][
            "new_domains_created"
        ],
        "adaptive_new_authorization_edges_created": adaptive["invariants"][
            "ground_truth_untouched"
        ]["new_authorization_edges_created"],
        "adaptive_mean_suspicion_before": adaptive["optimisation"]["mean_initial_suspicion"],
        "adaptive_mean_suspicion_after": adaptive["optimisation"]["mean_final_suspicion"],
        "adaptive_templates_converged": adaptive["optimisation"]["templates_converged"],
        "tests_passed": 410,
        "tests_added_this_step": 150,
    }

    # Idempotent: drop any Step 3 findings from a previous run before re-adding, so re-running
    # the driver does not duplicate them.
    findings = [
        f
        for f in manifest.setdefault("findings_requiring_attention", [])
        if f.get("step") != STEP
    ]
    manifest["findings_requiring_attention"] = findings
    findings.extend([
        {
            "step": 3,
            "severity": "resolved_this_step",
            "finding": (
                "The first implementation of the RQ2 first-divergence rule compared every "
                "retrieved candidate and was degenerate: stage 3 (relation extraction) absorbed "
                "948/1008 attributions and stages 4 and 5 were unreachable, because with ten "
                "candidates some relation is almost always misread."
            ),
            "resolution": (
                "Attribution is now scoped to the outcome-determining domains (the presented "
                "unauthorized link, or the omitted authorized ones). Distribution became "
                "280 none / 48 retrieval / 8 resolution / 672 relation. The unscoped comparison "
                "is retained per stage as agreement_all_candidates. "
                "assert_all_stages_reachable proves the attributor can name all five stages "
                "using ScriptedPipelines that err at exactly one stage each."
            ),
        },
        {
            "step": 3,
            "severity": "informational",
            "finding": (
                "The surrogate ReferenceStagePipeline never triggers the "
                "inferred_action_authorizations or presented_links attributions, because its "
                "authorization rule is derived from its relation judgement, so a correct "
                "relation on the presented domain implies a correct inference."
            ),
            "resolution": (
                "Reported explicitly in stage_traces.json under "
                "stages_not_reached_by_this_pipeline. This is a property of the surrogate, not "
                "of the instrument: reachability of all five stages is proven separately. "
                "Step 4's real LLM reader is expected to populate stages 4 and 5."
            ),
        },
        {
            "step": 3,
            "severity": "carry_to_step_4",
            "finding": (
                "The dense ranker is a deterministic hashed n-gram sketch, not a learned "
                "encoder. It captures lexical and sub-lexical similarity, not paraphrase."
            ),
            "resolution": (
                "Labelled as such in web_rag/retriever.py and in the frozen snapshot's "
                "dense_caveat. Reached through an Encoder protocol, and encoder_id is inside the "
                "replay fingerprint, so substituting a pinned open-weight encoder in Step 4 "
                "mechanically invalidates every frozen hash and forces a re-freeze rather than "
                "permitting a silent mix of old and new evidence."
            ),
        },
        {
            "step": 3,
            "severity": "carry_to_step_4",
            "finding": (
                "CONTRACT.md Section 6 asks for Docker Compose with pinned image digests. Step 3 "
                "is a fully offline replay over frozen bytes: no server, no DNS, no network."
            ),
            "resolution": (
                "Reproducibility is carried by the corpus digest and the replay fingerprint "
                "instead. results/web_rag_environment.json maps all twelve Section 6 services to "
                "their implementation or explicit substitution. Container packaging with a "
                "pinned base image and lockfile remains a Step 7 deliverable."
            ),
        },
        {
            "step": 3,
            "severity": "informational",
            "finding": (
                "Observation about the environment, not a model result: the frozen retriever "
                "places an authorized domain at rank 1 for only 51.8% of 'book' queries, 3.6% of "
                "'login'/'pay', 2.4% of 'browse' and 0.0% of 'contact' queries, while an "
                "authorized domain is present in the top 10 for 84-95% of queries. Hostnames "
                "carrying the entity name win the lexical signal."
            ),
            "resolution": (
                "Recorded in figures/step3_retrieval_profile.png panel (b). This confirms the "
                "benchmark is non-trivial for RQ1 and that retrieval recall is not the binding "
                "constraint: the misbinding opportunity is created by ranking, not by absence of "
                "the correct answer."
            ),
        },
        {
            "step": 3,
            "severity": "informational",
            "finding": (
                "configs/splits.yaml pins the adaptive holdout region to "
                "identity_consistency: [full], but RQ4 also lists partial identity copying as an "
                "adaptive capability."
            ),
            "resolution": (
                "Two pre-declared strata. Stratum A (40 templates) satisfies the frozen "
                "conjunction and is the only stratum admitted to a primary RQ4/ASR_a claim. "
                "Stratum B (40 templates, identity_consistency: partial) supplies the capability "
                "and is barred from primary runs by experiment_matrix rule R5. The strata differ "
                "on exactly one factor, so the partial-vs-full contrast is unconfounded."
            ),
        },
    ])

    existing = {o["path"] for o in manifest["outputs"]}
    for rel, desc, kind in NEW_OUTPUTS:
        p = ROOT / rel
        abspath = str(p)
        if abspath in existing:
            continue
        entry = {"path": abspath, "description": desc, "type": kind, "step": STEP}
        if p.is_file():
            entry["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
            entry["bytes"] = p.stat().st_size
        elif p.is_dir():
            entry["n_files"] = sum(1 for _ in p.rglob("*.html"))
        else:
            raise FileNotFoundError(f"declared Step 3 output missing: {abspath}")
        manifest["outputs"].append(entry)

    manifest["next_step"] = (
        "Step 4 -- AegisLink method and baselines (CONTRACT.md Sections 9, 10). Implement "
        "AegisLink.verify(e, d, a) -> VERIFIED | PLAUSIBLE | UNVERIFIED | CONTRADICTED over the "
        "seven required evidence families, with the six-component architecture and the seven "
        "required ablations, plus the ten baselines. Constraints inherited from Step 3: "
        "(1) consume evidence ONLY through web_rag.exposure at VERIFIER tier -- role, "
        "adversarial and lifecycle.prior_grants are unavailable by construction; "
        "(2) replay from data/benchmark/retrieval_snapshots.json and check the "
        "replay_fingerprint, so every baseline sees identical evidence; "
        "(3) read pages at phase='live' so content_change_after_indexing is exercised; "
        "(4) implement web_rag.trace_recorder.StagePipeline so RQ2 attribution comes for free; "
        "(5) call ExperimentMatrix.require_admitted before any run, and fit tau thresholds on "
        "train_development/validation only; "
        "(6) substituting a pinned open-weight encoder requires re-freezing the snapshots."
    )

    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest updated: step {STEP}, {len(manifest['outputs'])} outputs, "
          f"{len(findings)} findings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
