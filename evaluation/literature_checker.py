"""Automated literature collision check for the AegisLink project.

Implements ``CONTRACT.md`` Section 13. The check answers exactly one question:

    Does any prior work already occupy the intended novel contribution of
    ``CONTRACT.md`` Section 4.2 -- namely, the study of *action-link misbinding*
    (a legitimate business entity bound to an unauthorized action endpoint) together
    with an *action-level authorization defense*?

Section 4.2 forbids the words "first" or "novel" in the manuscript unless this module
returns ``NO_DIRECT_COLLISION`` with evidence. The module therefore has to be able to
return ``DIRECT_COLLISION``: scores are computed by a frozen regular-expression rubric
(``evaluation/collision_rubric.yaml``) over each paper's real title and abstract, with
every matched marker emitted as evidence. No language model assigns any score, mirroring
the Section 5.2 prohibition on LLM-generated ground truth.

Design notes
------------
* Prior works are *parsed* from ``CONTRACT.md`` Sections 4.1 and 20 rather than hardcoded,
  so editing the contract propagates into the matrix instead of silently diverging.
* Paper metadata is retrieved live from the arXiv API. Identifiers that do not resolve are
  recorded as ``metadata_verified: false`` and scored at evidence tier 2 from the
  contract's own characterization. They are never presented as verified.
* The six Section 13 search concepts are executed as live web searches so that papers
  absent from the contract's reference list can still surface. Raw responses are persisted
  for provenance.

Usage
-----
    uv run python evaluation/literature_checker.py [--offline] [--no-search]

Outputs
-------
    results/literature_matrix.yaml
    results/literature_search_raw/concept_<n>.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
import yaml

# --------------------------------------------------------------------------------------
# Paths. Resolved relative to the repository root (the parent of this file's directory)
# so the module works regardless of the caller's working directory.
# --------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = REPO_ROOT / "CONTRACT.md"
RUBRIC_PATH = REPO_ROOT / "evaluation" / "collision_rubric.yaml"
MATRIX_PATH = REPO_ROOT / "results" / "literature_matrix.yaml"
SEARCH_RAW_DIR = REPO_ROOT / "results" / "literature_search_raw"

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_RATE_LIMIT_SECONDS = 3.0  # arXiv API terms of use: no more than 1 request / 3 s
ARXIV_TIMEOUT_SECONDS = 30
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

# Cap on how many previously-unknown arXiv papers discovered by the live searches get
# scored. Surfacing is uncapped and every discovered identifier is listed in the matrix;
# only full metadata retrieval + scoring is bounded, because each costs a rate-limited
# arXiv round trip. The matrix records the cap and any overflow so a truncated sweep can
# never be mistaken for an exhaustive one.
MAX_DISCOVERED_TO_SCORE = 15

SEARCH_TIMEOUT_SECONDS = 180
SEARCH_MAX_RESULTS = 10


# --------------------------------------------------------------------------------------
# Contract parsing
# --------------------------------------------------------------------------------------
@dataclass
class PriorWork:
    """One prior work declared non-novel by ``CONTRACT.md`` Section 4.1."""

    paper_id: str
    arxiv_id: str
    short_name: str
    contract_characterization: str
    contract_cited_title: str = ""
    authors: str = ""
    title: str = ""
    abstract: str = ""
    year: str = ""
    metadata_verified: bool = False
    metadata_error: str = ""
    title_similarity: float = 0.0
    title_consistent: bool = True
    scores: dict[str, int] = field(default_factory=dict)
    matched_markers: dict[str, list[str]] = field(default_factory=dict)


def _normalize_quotes(text: str) -> str:
    """Replace typographic quotes/dashes with ASCII so regexes stay simple."""
    replacements = {
        "“": '"', "”": '"', "‘": "'", "’": "'",
        "–": "-", "—": "-", "è": "e", "é": "e",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _extract_section(contract_text: str, heading_pattern: str) -> str:
    """Return the body of the first ``##``-level section whose heading matches.

    Parameters
    ----------
    contract_text : str
        Full contract markdown.
    heading_pattern : str
        Regex matched against the heading line (e.g. ``r"^#+\\s*4\\.1"``).
    """
    lines = contract_text.splitlines()
    start = None
    start_level = 0
    for i, line in enumerate(lines):
        if re.match(heading_pattern, line):
            start = i + 1
            start_level = len(line) - len(line.lstrip("#"))
            break
    if start is None:
        raise ValueError(f"Contract section not found for pattern {heading_pattern!r}")

    body: list[str] = []
    for line in lines[start:]:
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            if level <= start_level:
                break
        body.append(line)
    return "\n".join(body)


def parse_prior_works(contract_text: str) -> list[PriorWork]:
    """Parse Section 4.1 prior works, enriched with Section 20 titles/authors.

    Section 4.1 supplies the contract's own characterization of each paper and its arXiv
    identifier. Section 20 supplies the full citation. The two are joined on arXiv ID so a
    mismatch between them surfaces as a missing title rather than a silent substitution.
    """
    text = _normalize_quotes(contract_text)

    sec41 = _extract_section(text, r"^#+\s*4\.1")
    sec20 = _extract_section(text, r"^#+\s*20\.")

    # Section 20: "1. Greshake et al. "Title." arXiv:2302.12173, 2023."
    ref_by_id: dict[str, dict[str, str]] = {}
    for raw in sec20.splitlines():
        line = raw.strip()
        m_id = re.search(r"arXiv:(\d{4}\.\d{4,5})", line)
        if not (line and re.match(r"^\d+\.", line) and m_id):
            continue
        arxiv_id = m_id.group(1)
        m_title = re.search(r'"([^"]+)"', line)
        # Authors: text between the leading list number and the opening quote.
        m_auth = re.match(r"^\d+\.\s*(.+?)\s*[\"]", line)
        m_year = re.search(r"arXiv:\d{4}\.\d{4,5},\s*(\d{4})", line)
        ref_by_id[arxiv_id] = {
            "title": m_title.group(1).strip() if m_title else "",
            "authors": m_auth.group(1).strip().rstrip(".") if m_auth else "",
            "year": m_year.group(1) if m_year else "",
        }

    works: list[PriorWork] = []
    for raw in sec41.splitlines():
        line = raw.strip()
        if not line.startswith("- "):
            continue
        m_id = re.search(r"arXiv:(\d{4}\.\d{4,5})", line)
        if not m_id:
            continue
        arxiv_id = m_id.group(1)
        body = line[2:].strip().rstrip(".")

        # "Characterization (SHORTNAME): Authors, arXiv:ID"
        characterization = body.split(":")[0].strip()
        m_short = re.search(r"\(([A-Za-z][A-Za-z0-9]+)\)", characterization)
        if m_short:
            short_name = m_short.group(1)
        else:
            ref = ref_by_id.get(arxiv_id, {})
            authors = ref.get("authors", "")
            short_name = (
                re.split(r"[\s,]+", authors)[0] if authors
                else f"arXiv{arxiv_id.replace('.', '')}"
            )

        ref = ref_by_id.get(arxiv_id, {})
        works.append(
            PriorWork(
                paper_id=f"P{len(works) + 1:02d}",
                arxiv_id=arxiv_id,
                short_name=short_name,
                contract_characterization=characterization,
                contract_cited_title=ref.get("title", ""),
                authors=ref.get("authors", ""),
                title=ref.get("title", ""),
                year=ref.get("year", ""),
            )
        )
    return works


def parse_search_concepts(contract_text: str) -> list[str]:
    """Parse the Section 13 search-concept block verbatim."""
    sec13 = _extract_section(_normalize_quotes(contract_text), r"^#+\s*13\.")
    blocks = re.findall(r"```text\n(.*?)```", sec13, flags=re.DOTALL)
    if not blocks:
        raise ValueError("Section 13 search-concept block not found in CONTRACT.md")
    return [ln.strip() for ln in blocks[0].splitlines() if ln.strip()]


# --------------------------------------------------------------------------------------
# arXiv metadata retrieval
# --------------------------------------------------------------------------------------
def fetch_arxiv_metadata(arxiv_id: str) -> tuple[dict[str, str] | None, str]:
    """Retrieve title/abstract/authors for one arXiv identifier.

    Returns
    -------
    (metadata, error)
        ``metadata`` is ``None`` when the identifier does not resolve, in which case
        ``error`` explains why. A non-resolving identifier is an expected outcome, not a
        crash: the contract cites several forward-dated identifiers.
    """
    try:
        resp = requests.get(
            ARXIV_API,
            params={"id_list": arxiv_id, "max_results": 1},
            timeout=ARXIV_TIMEOUT_SECONDS,
            headers={"User-Agent": "AegisLink-literature-checker/1.0"},
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        return None, f"network_error: {type(exc).__name__}: {exc}"

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as exc:
        return None, f"xml_parse_error: {exc}"

    entries = root.findall("atom:entry", ATOM_NS)
    if not entries:
        return None, "not_found: arXiv returned zero entries for this identifier"

    entry = entries[0]

    def _txt(tag: str) -> str:
        node = entry.find(f"atom:{tag}", ATOM_NS)
        return re.sub(r"\s+", " ", node.text).strip() if node is not None and node.text else ""

    entry_id = _txt("id")
    title = _txt("title")
    summary = _txt("summary")

    # arXiv signals a bad identifier with a sentinel entry titled "Error".
    if title.lower().startswith("error") or "api/errors" in entry_id:
        return None, f"not_found: arXiv error entry ({summary[:120] or 'no detail'})"
    if arxiv_id.split("v")[0] not in entry_id:
        return None, f"identifier_mismatch: response id {entry_id!r}"

    authors = [
        a.findtext("atom:name", default="", namespaces=ATOM_NS).strip()
        for a in entry.findall("atom:author", ATOM_NS)
    ]
    published = _txt("published")
    return (
        {
            "title": title,
            "abstract": summary,
            "authors": ", ".join(x for x in authors if x),
            "year": published[:4],
            "arxiv_entry_id": entry_id,
        },
        "",
    )


# --------------------------------------------------------------------------------------
# Rubric scoring
# --------------------------------------------------------------------------------------
def load_rubric() -> dict[str, Any]:
    with RUBRIC_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def score_axis(haystack: str, axis_spec: dict[str, Any]) -> tuple[int, list[str]]:
    """Score one rubric axis against a text.

    Returns the score (``min(3, n_markers_matched)``) and the sorted names of the markers
    that fired. Emitting the marker names is what makes the score auditable: a reader can
    confirm or refute each one against the abstract.
    """
    matched: list[str] = []
    for marker_name, marker in axis_spec.get("markers", {}).items():
        for pattern in marker.get("patterns", []):
            if re.search(pattern, haystack):
                matched.append(marker_name)
                break
    return min(3, len(matched)), sorted(matched)


def build_haystack(title: str, abstract: str, fallback: str = "") -> str:
    """Lowercase, whitespace-normalized scoring text.

    ``fallback`` is the contract's own one-line characterization of the paper. It is
    concatenated in ADDITION to the retrieved title and abstract, not merely as a
    substitute when retrieval fails. This is deliberate and conservative: a marker should
    fire if EITHER the paper's own abstract OR the contract's description of it indicates
    the concept. Under-detecting a collision produces a false novelty claim, which is the
    more damaging error, so the union of evidence is the safe direction. Each record's
    ``scored_text_sources`` field states exactly which components were present.
    """
    parts = [p for p in (title, abstract, fallback) if p]
    return re.sub(r"\s+", " ", " \n ".join(parts)).lower()


# Tokens carrying no discriminative power for title matching.
_TITLE_STOPWORDS = frozenset({
    "a", "an", "the", "of", "for", "and", "or", "in", "on", "to", "with", "against",
    "via", "using", "from", "at", "by", "as", "is", "are", "be", "not", "what", "you",
    "ve", "up", "into", "onto", "our", "we",
})


def title_similarity(cited: str, retrieved: str) -> float:
    """Jaccard similarity over content tokens of two titles.

    Guards against a silent failure mode that plain existence-checking misses: an arXiv
    identifier that resolves successfully but to a DIFFERENT paper than the contract cites.
    In that case the retrieved abstract is genuine but describes the wrong work, so every
    overlap score computed from it is meaningless while still looking tier-1 verified.
    """
    def toks(s: str) -> set[str]:
        raw = re.findall(r"[a-z0-9]+", _normalize_quotes(s).lower())
        return {t for t in raw if t not in _TITLE_STOPWORDS and len(t) > 1}

    a, b = toks(cited), toks(retrieved)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# Below this Jaccard similarity, the cited and retrieved titles are treated as describing
# different works and the record is flagged for manual confirmation.
TITLE_MATCH_THRESHOLD = 0.50


def evaluate_overlap(haystack: str, rubric: dict[str, Any]) -> tuple[dict[str, int], dict[str, list[str]]]:
    scores: dict[str, int] = {}
    markers: dict[str, list[str]] = {}
    for axis_name, axis_spec in rubric["axes"].items():
        scores[axis_name], markers[axis_name] = score_axis(haystack, axis_spec)
    return scores, markers


def decide_collision(scores: dict[str, int], rubric: dict[str, Any]) -> tuple[bool, bool]:
    """Apply the frozen Section 13 collision rule and the conservative near-collision screen."""
    rule = rubric["collision_rule"]
    direct = (
        scores["problem_overlap"] >= rule["problem_threshold"]
        and scores["method_overlap"] >= rule["method_threshold"]
    )
    near_t = rubric["near_collision_rule"]["threshold"]
    near = (
        not direct
        and scores["problem_overlap"] >= near_t
        and scores["method_overlap"] >= near_t
    )
    return direct, near


def required_project_change(scores: dict[str, int], direct: bool, near: bool, short_name: str) -> str:
    """Derive the Section 13 ``required_project_change`` field deterministically."""
    if direct:
        return (
            f"BLOCKING: {short_name} directly occupies the Section 4.2 contribution. "
            "Remove all 'first'/'novel' wording, re-scope the contribution, and re-run "
            "this check before any manuscript work proceeds (Section 4.2)."
        )
    actions: list[str] = []
    if near:
        actions.append(
            "Dedicate an explicit differentiation paragraph in related work separating "
            "action-level authorization from this paper's defense scope"
        )
    if scores["attack_overlap"] >= 2:
        actions.append(
            "Reuse only as an attack component or baseline; do not let its framing "
            "redefine project scope (Sections 2, 7, 10)"
        )
    if scores["method_overlap"] >= 2:
        actions.append(
            "Implement as a reproducible baseline defense under Section 10 with identical "
            "evidence, splits, and attack budgets (Section 10)"
        )
    if scores["evaluation_overlap"] >= 2:
        actions.append(
            "Align metric naming to Section 11 definitions and note where its evaluation "
            "protocol differs to avoid implying comparability"
        )
    if not actions:
        actions.append("Cite as prior work; no project change required")
    return "; ".join(actions) + "."


# --------------------------------------------------------------------------------------
# Live search over the Section 13 concepts
# --------------------------------------------------------------------------------------
def run_search_concept(concept: str, index: int) -> dict[str, Any]:
    """Execute one Section 13 search concept via parallel-cli; persist the raw response."""
    SEARCH_RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SEARCH_RAW_DIR / f"concept_{index:02d}.json"
    record: dict[str, Any] = {
        "concept": concept,
        "raw_output_file": str(out_path.relative_to(REPO_ROOT)),
        "status": "not_run",
        "n_results": 0,
        "arxiv_ids_found": [],
    }
    cmd = [
        "parallel-cli", "search", concept,
        "--max-results", str(SEARCH_MAX_RESULTS),
        "--json", "-o", str(out_path),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=SEARCH_TIMEOUT_SECONDS, check=False,
        )
    except FileNotFoundError:
        record["status"] = "unavailable: parallel-cli not found on PATH"
        return record
    except subprocess.TimeoutExpired:
        record["status"] = f"timeout after {SEARCH_TIMEOUT_SECONDS}s"
        return record

    if proc.returncode != 0 or not out_path.exists():
        record["status"] = f"error: exit {proc.returncode}: {(proc.stderr or '')[:200]}"
        return record

    try:
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        record["status"] = f"error: unreadable response: {exc}"
        return record

    results = payload.get("results", []) or []
    record["status"] = "ok"
    record["n_results"] = len(results)
    record["result_titles"] = [(r.get("title") or "")[:160] for r in results]

    ids: list[str] = []
    for r in results:
        blob = " ".join(str(r.get(k, "")) for k in ("url", "title")) + " ".join(
            str(e) for e in (r.get("excerpts") or [])
        )
        ids.extend(re.findall(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", blob, flags=re.I))
        ids.extend(re.findall(r"arxiv[:\s]+(\d{4}\.\d{4,5})", blob, flags=re.I))
    record["arxiv_ids_found"] = sorted(set(ids))
    return record


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="AegisLink literature collision check (CONTRACT.md Section 13)")
    ap.add_argument("--offline", action="store_true",
                    help="Skip arXiv retrieval; score all priors at evidence tier 2.")
    ap.add_argument("--no-search", action="store_true",
                    help="Skip the six live Section 13 concept searches.")
    ap.add_argument("--checkpoint", default="project_start",
                    choices=["project_start", "post_pilot", "pre_full_experiments", "pre_manuscript_freeze"],
                    help="Which Section 13 checkpoint this run represents.")
    args = ap.parse_args()

    print("=" * 78)
    print("AegisLink literature collision check -- CONTRACT.md Section 13")
    print("=" * 78)

    contract_text = CONTRACT_PATH.read_text(encoding="utf-8")
    rubric = load_rubric()
    works = parse_prior_works(contract_text)
    concepts = parse_search_concepts(contract_text)

    print(f"[parse] prior works declared non-novel (Section 4.1): {len(works)}")
    print(f"[parse] search concepts (Section 13): {len(concepts)}")
    print(f"[parse] rubric version {rubric['rubric_version']}, "
          f"rule: {rubric['collision_rule']['expression']}")
    if len(works) != 8:
        print(f"[warn] expected 8 prior works from Section 4.1, parsed {len(works)}", file=sys.stderr)

    # ---- Stage 1: metadata retrieval + scoring of the declared priors -----------------
    print("\n[stage 1] retrieving arXiv metadata and scoring declared prior works")
    for i, w in enumerate(works, start=1):
        if args.offline:
            w.metadata_verified = False
            w.metadata_error = "offline mode requested; retrieval skipped"
        else:
            if i > 1:
                time.sleep(ARXIV_RATE_LIMIT_SECONDS)  # respect arXiv API terms
            meta, err = fetch_arxiv_metadata(w.arxiv_id)
            if meta:
                w.metadata_verified = True
                w.title = meta["title"] or w.title
                w.abstract = meta["abstract"]
                w.authors = meta["authors"] or w.authors
                w.year = meta["year"] or w.year
            else:
                w.metadata_verified = False
                w.metadata_error = err

        # Cross-check the retrieved title against the title the contract cites. An
        # identifier that resolves to a different paper yields a genuine-looking abstract
        # that describes the wrong work, which would corrupt every score silently.
        if w.metadata_verified and w.contract_cited_title:
            w.title_similarity = round(title_similarity(w.contract_cited_title, w.title), 3)
            w.title_consistent = w.title_similarity >= TITLE_MATCH_THRESHOLD
        else:
            w.title_similarity = 0.0
            w.title_consistent = not w.metadata_verified  # not applicable when unverified

        haystack = build_haystack(w.title, w.abstract, w.contract_characterization)
        w.scores, w.matched_markers = evaluate_overlap(haystack, rubric)
        tier = "tier_1_verified_abstract" if w.metadata_verified else "tier_2_contract_declaration"
        flag = "" if w.title_consistent else f"  [!] TITLE_MISMATCH sim={w.title_similarity}"
        print(f"  {w.paper_id} {w.short_name:<12} arXiv:{w.arxiv_id}  "
              f"verified={str(w.metadata_verified):<5} "
              f"P{w.scores['problem_overlap']} A{w.scores['attack_overlap']} "
              f"M{w.scores['method_overlap']} E{w.scores['evaluation_overlap']}  [{tier}]{flag}")

    # ---- Stage 2: live concept searches ---------------------------------------------
    search_records: list[dict[str, Any]] = []
    known_ids = {w.arxiv_id for w in works}
    discovered_ids: list[str] = []
    if args.no_search:
        print("\n[stage 2] skipped (--no-search)")
    else:
        print(f"\n[stage 2] running {len(concepts)} Section 13 search concepts")
        for idx, concept in enumerate(concepts, start=1):
            rec = run_search_concept(concept, idx)
            search_records.append(rec)
            print(f"  [{idx}/{len(concepts)}] {rec['status']:<12} "
                  f"n={rec['n_results']:<3} arxiv_ids={len(rec['arxiv_ids_found'])}  {concept[:58]}")
            for aid in rec["arxiv_ids_found"]:
                if aid not in known_ids and aid not in discovered_ids:
                    discovered_ids.append(aid)

    # ---- Stage 3: score newly discovered candidates ---------------------------------
    discovered_records: list[dict[str, Any]] = []
    to_score = discovered_ids[:MAX_DISCOVERED_TO_SCORE]
    overflow = discovered_ids[MAX_DISCOVERED_TO_SCORE:]
    if to_score:
        print(f"\n[stage 3] scoring {len(to_score)} newly discovered arXiv candidate(s)")
        if overflow:
            print(f"  [note] {len(overflow)} candidate(s) surfaced but NOT scored "
                  f"(cap={MAX_DISCOVERED_TO_SCORE}); listed in matrix as unscored")
        for j, aid in enumerate(to_score, start=1):
            if j > 1 or not args.offline:
                time.sleep(ARXIV_RATE_LIMIT_SECONDS)
            meta, err = fetch_arxiv_metadata(aid)
            if not meta:
                discovered_records.append({
                    "paper_id": f"D{j:02d}", "arxiv_id": aid,
                    "metadata_verified": False, "metadata_error": err,
                    "scored": False,
                    "note": "surfaced by Section 13 search; metadata unavailable so not scored",
                })
                print(f"  D{j:02d} arXiv:{aid}  unresolved ({err[:60]})")
                continue
            hay = build_haystack(meta["title"], meta["abstract"])
            scores, markers = evaluate_overlap(hay, rubric)
            direct, near = decide_collision(scores, rubric)
            discovered_records.append({
                "paper_id": f"D{j:02d}", "arxiv_id": aid,
                "title": meta["title"], "authors": meta["authors"], "year": meta["year"],
                "metadata_verified": True, "scored": True,
                "evidence_tier": "tier_1_verified_abstract",
                "problem_overlap": scores["problem_overlap"],
                "attack_overlap": scores["attack_overlap"],
                "method_overlap": scores["method_overlap"],
                "evaluation_overlap": scores["evaluation_overlap"],
                "matched_markers": markers,
                "direct_collision": direct,
                "near_collision": near,
                "required_project_change": required_project_change(scores, direct, near, aid),
            })
            flag = "DIRECT_COLLISION" if direct else ("near_collision" if near else "no_collision")
            print(f"  D{j:02d} arXiv:{aid}  P{scores['problem_overlap']} A{scores['attack_overlap']} "
                  f"M{scores['method_overlap']} E{scores['evaluation_overlap']}  {flag}")
    else:
        print("\n[stage 3] no new arXiv candidates discovered by the concept searches")

    # ---- Stage 4: assemble matrix ---------------------------------------------------
    prior_records: list[dict[str, Any]] = []
    for w in works:
        direct, near = decide_collision(w.scores, rubric)
        rec: dict[str, Any] = {
            "paper_id": w.paper_id,
            "short_name": w.short_name,
            "arxiv_id": w.arxiv_id,
            "title": w.title,
            "authors": w.authors,
            "year": w.year,
            "contract_characterization": w.contract_characterization,
            "contract_cited_title": w.contract_cited_title,
            "metadata_verified": w.metadata_verified,
            "evidence_tier": ("tier_1_verified_abstract" if w.metadata_verified
                              else "tier_2_contract_declaration"),
            "scored_text_sources": (
                ["retrieved_title", "retrieved_abstract", "contract_characterization"]
                if w.metadata_verified else ["contract_cited_title", "contract_characterization"]
            ),
            "title_similarity_to_contract_citation": w.title_similarity,
            "title_consistent": w.title_consistent,
            "problem_overlap": w.scores["problem_overlap"],
            "attack_overlap": w.scores["attack_overlap"],
            "method_overlap": w.scores["method_overlap"],
            "evaluation_overlap": w.scores["evaluation_overlap"],
            "matched_markers": w.matched_markers,
            "direct_collision": direct,
            "near_collision": near,
            "required_project_change": required_project_change(w.scores, direct, near, w.short_name),
        }
        if not w.metadata_verified:
            rec["metadata_error"] = w.metadata_error
            rec["unverified_caveat"] = (
                "arXiv identifier did not resolve at run time. Scored at evidence tier 2 "
                "from the contract's own characterization. MUST be rescored at tier 1 "
                "before manuscript freeze if the identifier later resolves."
            )
        if not w.title_consistent:
            rec["title_mismatch_caveat"] = (
                f"The cited title and the title retrieved for arXiv:{w.arxiv_id} share only "
                f"{w.title_similarity:.0%} of their content tokens. The identifier resolves, "
                "but possibly to a different work or a retitled version than CONTRACT.md "
                "Section 20 cites. Overlap scores for this record are computed from the "
                "RETRIEVED abstract, so they describe the retrieved paper. The citation MUST "
                "be reconciled before manuscript freeze: either correct the identifier, or "
                "correct the cited title and the Section 4.1 characterization."
            )
        prior_records.append(rec)

    collided = [r["paper_id"] for r in prior_records + discovered_records
                if r.get("direct_collision")]
    near_hits = [r["paper_id"] for r in prior_records + discovered_records
                 if r.get("near_collision")]
    verdict = "DIRECT_COLLISION" if collided else "NO_DIRECT_COLLISION"

    n_unverified = sum(1 for r in prior_records if not r["metadata_verified"])
    searches_ok = sum(1 for r in search_records if r["status"] == "ok")
    title_mismatches = [r["paper_id"] for r in prior_records if not r["title_consistent"]]

    matrix: dict[str, Any] = {
        "literature_matrix_version": "1.0",
        "contract_ref": "CONTRACT.md Section 13 (workflow), Section 4 (novelty contract)",
        "checkpoint": args.checkpoint,
        "checkpoint_schedule": [
            "project_start", "post_pilot", "pre_full_experiments", "pre_manuscript_freeze",
        ],
        "rubric": {
            "file": str(RUBRIC_PATH.relative_to(REPO_ROOT)),
            "version": rubric["rubric_version"],
            "score_rule": rubric["score_rule"],
            "collision_rule": rubric["collision_rule"]["expression"],
            "near_collision_rule": rubric["near_collision_rule"]["expression"],
            "scored_by": "frozen regular-expression markers; no LLM judgement",
        },
        "verdict": verdict,
        "direct_collision_any": bool(collided),
        "colliding_paper_ids": collided,
        "near_collision_paper_ids": near_hits,
        "citation_integrity": {
            "title_mismatch_paper_ids": title_mismatches,
            "title_match_threshold": TITLE_MATCH_THRESHOLD,
            "interpretation": (
                "Each cited arXiv identifier was resolved and its retrieved title compared "
                "against the title CONTRACT.md Section 20 cites. Listed identifiers resolve "
                "but to a title that does not match the citation. This does not change the "
                "collision verdict, which is computed from retrieved abstracts, but the "
                "citations MUST be reconciled before manuscript freeze."
            ),
        },
        "novelty_claim_permitted": not collided,
        "novelty_claim_gate": (
            "CONTRACT.md Section 4.2 permits the words 'first' or 'novel' in the manuscript "
            "only while this verdict reads NO_DIRECT_COLLISION."
        ),
        "intended_novel_contribution": (
            "Combination of: action-link misbinding definition and measurement; legitimate "
            "entity with unauthorized action endpoint (not fabricated product or rank "
            "promotion); entity-domain-action authorization model; action-risk-aware "
            "verification over browse/contact/book/login/pay; explicit preservation of "
            "authorized third-party services; fully generated auto-labelled benchmark; "
            "failure-stage attribution with security-utility evaluation under adaptive "
            "attacks (CONTRACT.md Section 4.2)."
        ),
        "counts": {
            "prior_works_declared": len(prior_records),
            "prior_works_metadata_verified": len(prior_records) - n_unverified,
            "prior_works_metadata_unverified": n_unverified,
            "search_concepts_defined": len(concepts),
            "search_concepts_executed_ok": searches_ok,
            "new_arxiv_candidates_surfaced": len(discovered_ids),
            "new_arxiv_candidates_scored": len(discovered_records),
            "new_arxiv_candidates_unscored_over_cap": len(overflow),
            "prior_works_with_title_mismatch": len(title_mismatches),
        },
        "prior_works": prior_records,
        "discovered_candidates": discovered_records,
        "discovered_unscored_over_cap": overflow,
        "search_log": search_records,
        "limitations": [
            "Overlap scoring is lexical: it detects concepts present in a title and abstract. "
            "A paper whose full text contains an action-level authorization defense that its "
            "abstract never mentions would be under-scored. Section 13 therefore requires "
            "re-running this check at all four checkpoints, and any near_collision entry is "
            "flagged for full-text differentiation.",
            "Web search coverage is not exhaustive. A NO_DIRECT_COLLISION verdict is evidence "
            "of no detected collision, not proof that none exists.",
            f"{n_unverified} of {len(prior_records)} cited identifiers did not resolve against "
            "the live arXiv API and are scored at evidence tier 2 from the contract's own "
            "characterization. Their non-collision status is provisional.",
            f"{len(title_mismatches)} of {len(prior_records)} cited identifiers resolve to a "
            "title that does not match the CONTRACT.md Section 20 citation "
            f"({', '.join(title_mismatches) if title_mismatches else 'none'}). Scores for "
            "those records describe the retrieved paper, not necessarily the intended one.",
            "Scoring text is the union of the retrieved title, the retrieved abstract, and "
            "the contract's own characterization of the paper. The union is used because "
            "under-detecting a collision would produce a false novelty claim; each record's "
            "scored_text_sources field states which components were present.",
        ],
    }

    MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MATRIX_PATH.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(matrix, fh, sort_keys=False, allow_unicode=True, width=100)

    print("\n" + "=" * 78)
    print(f"VERDICT: {verdict}")
    print(f"  prior works scored           : {len(prior_records)} "
          f"({len(prior_records) - n_unverified} tier-1 verified, {n_unverified} tier-2)")
    print(f"  direct collisions            : {len(collided)} {collided}")
    print(f"  near collisions (advisory)   : {len(near_hits)} {near_hits}")
    print(f"  new candidates surfaced      : {len(discovered_ids)} "
          f"({len(discovered_records)} scored)")
    print(f"  citation title mismatches    : {len(title_mismatches)} {title_mismatches}")
    print(f"  novelty wording permitted    : {not collided}")
    print(f"  matrix written               : {MATRIX_PATH.relative_to(REPO_ROOT)}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
