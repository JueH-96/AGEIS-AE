"""Two-phase crawler and indexer over the frozen controlled corpus.

CONTRACT.md Section 6 (controlled Web-RAG environment: ``crawler``, ``indexer`` services) and
Section 7 (the ``content_change_after_indexing`` attack factor).

The document slot
-----------------
A URL is *not* a usable index key in this corpus. 2,631 rendered pages sit on 1,376 distinct
URLs, with up to 68 pages sharing one URL, because a shared booking-provider host serves a
listing page per business. Indexing by URL would silently collapse 68 documents into one and
destroy the entity-relative structure the whole benchmark rests on.

The crawl and index unit is therefore a **document slot**, keyed by the tuple the corpus was
actually rendered at::

    (entity_id, domain_id, site_template_id, attack_template_id)

``url`` is carried verbatim as the *presented link* and never rewritten: page bytes and their
SHA-256 hashes were frozen in Step 2.

Two-phase replay
----------------
Section 7's ``content_change_after_indexing`` factor is only real if the two phases read
different bytes:

* **Phase 1, index time.** For a slot with a changing pair, the ``indexed`` page is indexed --
  the benign snapshot the crawler saw first. For a static slot, the ``static`` page is indexed.
* **Phase 2, read time.** :meth:`Corpus.fetch` with ``phase="live"`` returns the ``live`` page
  for a changing slot -- the version that carries the attack -- and the same ``static`` page
  otherwise.

So a retriever ranks a page on benign text and the reader is then handed adversarial text at
the same URL. If the two phases ever returned identical bytes for a changing slot the factor
would be inert and RQ4's "simulate content changes after an initially benign snapshot"
capability would be unmeasurable; :meth:`Corpus.verify_two_phase_fidelity` asserts otherwise
over all 629 pairs.

Everything this module hands out is projected through :mod:`web_rag.exposure` before return.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import yaml

from web_rag.exposure import (
    ExposureTier,
    PublicDomainRegistry,
    assert_no_leakage,
    load_public_domain_registry,
)

CRAWLER_VERSION = "1.0"
INDEXER_VERSION = "1.0"

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = REPO_ROOT / "data" / "benchmark"
PAGES_DIR = BENCHMARK_DIR / "pages"
PAGE_MANIFEST_PATH = BENCHMARK_DIR / "page_manifest.yaml"
PAGE_HASHES_PATH = BENCHMARK_DIR / "page_hashes.yaml"

#: Snapshot label used for pages that never change.
SNAPSHOT_STATIC = "static"
#: Snapshot indexed at phase 1.
SNAPSHOT_INDEXED = "indexed"
#: Snapshot served at phase 2.
SNAPSHOT_LIVE = "live"

PHASE_INDEX = "indexed"
PHASE_LIVE = "live"
PHASES: tuple[str, str] = (PHASE_INDEX, PHASE_LIVE)

#: The machine-readable channel the corpus publishes (CONTRACT.md Section 8 item 3). Stripped
#: from every ``RETRIEVER``/``READER``-tier text view: it is an evaluator channel.
META_CHANNEL_NAMES: tuple[str, ...] = (
    "x-aegis-entity-id",
    "x-aegis-page-id",
    "x-aegis-corpus",
)

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b.*?</\1>", re.DOTALL | re.IGNORECASE)
_META_RE = re.compile(r"<meta\b[^>]*>", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_HREF_RE = re.compile(r"""<a\b[^>]*\bhref\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_TITLE_RE = re.compile(r"<title>(.*?)</title>", re.DOTALL | re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class CorpusIntegrityError(RuntimeError):
    """Raised when the on-disk corpus disagrees with the frozen manifest or hashes."""


# ======================================================================================
# Text extraction
# ======================================================================================
def visible_text(html: str) -> str:
    """Return the human-visible text of a page.

    ``<meta>`` elements are removed *before* tag stripping, so the ``x-aegis-*`` channel can
    never reach a retriever-tier text view even though it lives in the page bytes.
    """
    body = _SCRIPT_STYLE_RE.sub(" ", html)
    body = _META_RE.sub(" ", body)
    body = _TAG_RE.sub(" ", body)
    return _WS_RE.sub(" ", body).strip()


def page_title(html: str) -> str:
    """Return the ``<title>`` text, or an empty string."""
    m = _TITLE_RE.search(html)
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", m.group(1))).strip() if m else ""


def outbound_links(html: str) -> list[str]:
    """Return every ``href`` in document order, de-duplicated, order preserved."""
    seen: set[str] = set()
    out: list[str] = []
    for href in _HREF_RE.findall(html):
        if href not in seen:
            seen.add(href)
            out.append(href)
    return out


def meta_channel(html: str) -> dict[str, str]:
    """Extract the ``x-aegis-*`` meta channel. ``EVALUATOR`` tier only."""
    out: dict[str, str] = {}
    for tag in _META_RE.findall(html):
        nm = re.search(r'name\s*=\s*["\']([^"\']+)["\']', tag, re.IGNORECASE)
        ct = re.search(r'content\s*=\s*["\']([^"\']*)["\']', tag, re.IGNORECASE)
        if nm and ct and nm.group(1) in META_CHANNEL_NAMES:
            out[nm.group(1)] = ct.group(1)
    return out


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenisation. Deterministic and dependency-free.

    Hyphenated hostnames such as ``harbour-lantern-bistro.test`` split into their component
    words, which is what makes a name-in-domain lexical signal visible to BM25 at all.
    """
    return _TOKEN_RE.findall(text.lower())


# ======================================================================================
# Records
# ======================================================================================
@dataclass(frozen=True)
class CrawledDocument:
    """One document slot with both snapshot phases resolved.

    ``indexed_html`` / ``live_html`` are the raw bytes-as-text of the two phases. For a static
    slot they are the same object. Experimenter provenance (``page_role``,
    ``attack_template_id``, ...) is *not* stored here; it stays in
    :class:`CorpusGroundTruth`, on the evaluator side of the firewall.
    """

    doc_id: str
    url: str
    domain: str
    domain_id: str
    entity_id: str
    indexed_page_id: str
    live_page_id: str
    indexed_html: str
    live_html: str
    changing: bool

    @property
    def content_changed(self) -> bool:
        """Whether the two phases differ in bytes."""
        return self.indexed_html != self.live_html

    def html(self, phase: str = PHASE_INDEX) -> str:
        if phase == PHASE_INDEX:
            return self.indexed_html
        if phase == PHASE_LIVE:
            return self.live_html
        raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")

    def text(self, phase: str = PHASE_INDEX) -> str:
        return visible_text(self.html(phase))

    def sha256(self, phase: str = PHASE_INDEX) -> str:
        return hashlib.sha256(self.html(phase).encode("utf-8")).hexdigest()

    def public_view(self, phase: str, tier: ExposureTier = ExposureTier.RETRIEVER) -> dict[str, Any]:
        """Serialise at ``tier``. Never includes the meta channel or any provenance."""
        view: dict[str, Any] = {
            "doc_id": self.doc_id,
            "url": self.url,
            "domain": self.domain,
            "domain_id": self.domain_id,
            "title": page_title(self.html(phase)),
            "text": self.text(phase),
            "outbound_links": outbound_links(self.html(phase)),
        }
        if tier is ExposureTier.EVALUATOR:
            view["entity_id"] = self.entity_id
            view["indexed_page_id"] = self.indexed_page_id
            view["live_page_id"] = self.live_page_id
            view["changing"] = self.changing
        return view


@dataclass(frozen=True)
class CorpusGroundTruth:
    """Experimenter-side provenance for one document slot. ``EVALUATOR`` tier only."""

    doc_id: str
    entity_id: str
    entity_template_id: str
    domain_id: str
    page_role: str
    site_template_id: str
    attack_template_id: str | None
    changing: bool
    indexed_page_id: str
    live_page_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "entity_id": self.entity_id,
            "entity_template_id": self.entity_template_id,
            "domain_id": self.domain_id,
            "page_role": self.page_role,
            "site_template_id": self.site_template_id,
            "attack_template_id": self.attack_template_id,
            "changing": self.changing,
            "indexed_page_id": self.indexed_page_id,
            "live_page_id": self.live_page_id,
        }


# ======================================================================================
# Crawler
# ======================================================================================
def _slot_key(entry: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(entry["entity_id"]),
        str(entry["domain_id"]),
        str(entry["site_template_id"]),
        str(entry["attack_template_id"] or "-"),
    )


def _doc_id(indexed_page_id: str) -> str:
    """Stable slot handle derived from the phase-1 page id.

    Deriving the handle from the *indexed* page means the identifier a retriever sees is the
    identifier of the page it actually indexed, and stays constant across replays.
    """
    return f"DOC{indexed_page_id.removeprefix('PG')}"


def load_page_manifest(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the frozen page manifest."""
    src = path or PAGE_MANIFEST_PATH
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    return yaml.load(src.read_text(encoding="utf-8"), Loader=loader)["pages"]


def load_page_hashes(path: Path | None = None) -> dict[str, str]:
    """Load ``page_id -> sha256`` from the frozen hash file.

    Tolerates the three shapes the freeze could plausibly take (``page_sha256`` mapping,
    ``pages`` mapping, ``pages`` list of records) rather than hard-coding one, so a Step 2
    re-freeze in a different shape fails a test rather than silently skipping verification.
    """
    src = path or PAGE_HASHES_PATH
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    raw = yaml.load(src.read_text(encoding="utf-8"), Loader=loader)
    for key in ("page_sha256", "pages", "page_hashes"):
        if isinstance(raw, dict) and key in raw:
            body = raw[key]
            break
    else:
        body = raw
    if isinstance(body, list):
        return {str(e["page_id"]): str(e["sha256"]) for e in body}
    if not isinstance(body, dict):
        raise CorpusIntegrityError(f"unrecognised page-hash file shape in {src}")
    out: dict[str, str] = {}
    for k, v in body.items():
        out[str(k)] = str(v) if isinstance(v, str) else str(v["sha256"])
    return out


@dataclass
class Corpus:
    """The crawled corpus: document slots plus the evaluator-side ground truth."""

    documents: dict[str, CrawledDocument]
    ground_truth: dict[str, CorpusGroundTruth]
    registry: PublicDomainRegistry
    n_pages: int
    crawler_version: str = CRAWLER_VERSION
    corpus_digest: str = ""

    # -- lookups ---------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self) -> Iterator[CrawledDocument]:
        yield from (self.documents[k] for k in sorted(self.documents))

    @property
    def doc_ids(self) -> list[str]:
        return sorted(self.documents)

    def fetch(self, doc_id: str, phase: str = PHASE_LIVE) -> str:
        """Fetch a document's HTML at ``phase``.

        The reader/verifier path uses ``phase="live"``; the indexer used ``phase="indexed"``.
        """
        return self.documents[doc_id].html(phase)

    def changing_doc_ids(self) -> list[str]:
        return sorted(d.doc_id for d in self if d.changing)

    def static_doc_ids(self) -> list[str]:
        return sorted(d.doc_id for d in self if not d.changing)

    def docs_for_entity(self, entity_id: str) -> list[CrawledDocument]:
        """``EVALUATOR``-tier helper: which slots concern this entity."""
        return [d for d in self if d.entity_id == entity_id]

    # -- integrity -------------------------------------------------------------------
    def verify_two_phase_fidelity(self) -> dict[str, Any]:
        """Assert the two phases are genuinely distinct where they must be.

        Returns
        -------
        dict
            Counts and the set of any offending doc ids, for the freeze record.

        Raises
        ------
        CorpusIntegrityError
            If a changing slot has identical phases (factor inert), or a static slot has
            differing phases (an unaccounted-for source of variation).
        """
        changing_identical: list[str] = []
        static_differing: list[str] = []
        for d in self:
            if d.changing and not d.content_changed:
                changing_identical.append(d.doc_id)
            if not d.changing and d.content_changed:
                static_differing.append(d.doc_id)
        if changing_identical:
            raise CorpusIntegrityError(
                f"{len(changing_identical)} slots declare content_change_after_indexing but "
                f"serve identical bytes in both phases, e.g. {changing_identical[:5]}. The "
                f"Section 7 factor would be inert."
            )
        if static_differing:
            raise CorpusIntegrityError(
                f"{len(static_differing)} static slots differ across phases, e.g. "
                f"{static_differing[:5]}."
            )
        return {
            "n_slots": len(self.documents),
            "n_changing": len(self.changing_doc_ids()),
            "n_static": len(self.static_doc_ids()),
            "changing_with_identical_phases": 0,
            "static_with_differing_phases": 0,
        }

    def two_phase_diff_report(self) -> dict[str, Any]:
        """Quantify how the live phase differs from the indexed phase.

        Reported so the ``content_change_after_indexing`` manipulation can be characterised
        rather than merely asserted: if the live snapshot only ever *added* text, a defense
        could exploit that regularity.
        """
        added = 0
        removed = 0
        both = 0
        deltas: list[int] = []
        for d in self:
            if not d.changing:
                continue
            a = set(tokenize(d.text(PHASE_INDEX)))
            b = set(tokenize(d.text(PHASE_LIVE)))
            gained, lost = b - a, a - b
            if gained and lost:
                both += 1
            elif gained:
                added += 1
            elif lost:
                removed += 1
            deltas.append(len(b) - len(a))
        n = len(deltas) or 1
        return {
            "n_changing": len(deltas),
            "live_only_added_tokens": added,
            "live_only_removed_tokens": removed,
            "live_added_and_removed": both,
            "mean_token_delta": round(sum(deltas) / n, 4),
            "min_token_delta": min(deltas) if deltas else 0,
            "max_token_delta": max(deltas) if deltas else 0,
        }


def crawl_corpus(
    *,
    manifest: Sequence[Mapping[str, Any]] | None = None,
    pages_dir: Path | None = None,
    verify_hashes: bool = True,
    registry: PublicDomainRegistry | None = None,
    progress_every: int = 500,
) -> Corpus:
    """Crawl the frozen corpus into two-phase document slots.

    Parameters
    ----------
    manifest
        Page manifest entries. Defaults to the frozen ``page_manifest.yaml``.
    pages_dir
        Root of the rendered pages. Defaults to ``data/benchmark/pages``.
    verify_hashes
        Re-hash every page and compare against ``page_hashes.yaml``. Catches silent corpus
        drift, which would invalidate every frozen retrieval snapshot.
    registry
        Public domain registry. Loaded if omitted.
    progress_every
        Emit a progress line every N pages.

    Raises
    ------
    CorpusIntegrityError
        On a missing page file, a hash mismatch, or a malformed indexed/live pairing.
    """
    entries = list(manifest) if manifest is not None else load_page_manifest()
    root = pages_dir or PAGES_DIR
    reg = registry or load_public_domain_registry()
    expected_hashes = load_page_hashes() if verify_hashes else {}

    # Group manifest entries into slots.
    slots: dict[tuple[str, str, str, str], dict[str, Mapping[str, Any]]] = {}
    for e in entries:
        slots.setdefault(_slot_key(e), {})[str(e["snapshot"])] = e

    print(f"[crawler] {len(entries)} manifest pages -> {len(slots)} document slots")

    documents: dict[str, CrawledDocument] = {}
    ground_truth: dict[str, CorpusGroundTruth] = {}
    html_cache: dict[str, str] = {}
    processed = 0

    def read_page(entry: Mapping[str, Any]) -> str:
        page_id = str(entry["page_id"])
        if page_id in html_cache:
            return html_cache[page_id]
        path = root / str(entry["entity_id"]) / f"{page_id}.html"
        if not path.is_file():
            raise CorpusIntegrityError(f"missing rendered page {path}")
        html = path.read_text(encoding="utf-8")
        if verify_hashes:
            digest = hashlib.sha256(html.encode("utf-8")).hexdigest()
            declared = str(entry.get("sha256") or expected_hashes.get(page_id, ""))
            if declared and digest != declared:
                raise CorpusIntegrityError(
                    f"hash mismatch for {page_id}: on-disk {digest} != frozen {declared}"
                )
            frozen = expected_hashes.get(page_id)
            if frozen and frozen != digest:
                raise CorpusIntegrityError(
                    f"page_hashes.yaml mismatch for {page_id}: {digest} != {frozen}"
                )
        html_cache[page_id] = html
        return html

    for key in sorted(slots):
        by_snapshot = slots[key]
        labels = set(by_snapshot)
        if labels == {SNAPSHOT_STATIC}:
            entry = by_snapshot[SNAPSHOT_STATIC]
            html = read_page(entry)
            idx_entry = live_entry = entry
            idx_html = live_html = html
            changing = False
        elif labels == {SNAPSHOT_INDEXED, SNAPSHOT_LIVE}:
            idx_entry = by_snapshot[SNAPSHOT_INDEXED]
            live_entry = by_snapshot[SNAPSHOT_LIVE]
            idx_html = read_page(idx_entry)
            live_html = read_page(live_entry)
            changing = True
        else:
            raise CorpusIntegrityError(
                f"slot {key} has snapshot labels {sorted(labels)}; expected either "
                f"{{'static'}} or {{'indexed', 'live'}}. A half-pair means the "
                f"content_change_after_indexing factor is not renderable for that slot."
            )

        doc_id = _doc_id(str(idx_entry["page_id"]))
        if doc_id in documents:
            raise CorpusIntegrityError(f"duplicate doc_id {doc_id} for slot {key}")

        documents[doc_id] = CrawledDocument(
            doc_id=doc_id,
            url=str(idx_entry["url"]),
            domain=str(idx_entry["domain"]),
            domain_id=str(idx_entry["domain_id"]),
            entity_id=str(idx_entry["entity_id"]),
            indexed_page_id=str(idx_entry["page_id"]),
            live_page_id=str(live_entry["page_id"]),
            indexed_html=idx_html,
            live_html=live_html,
            changing=changing,
        )
        ground_truth[doc_id] = CorpusGroundTruth(
            doc_id=doc_id,
            entity_id=str(idx_entry["entity_id"]),
            entity_template_id=str(idx_entry["entity_template_id"]),
            domain_id=str(idx_entry["domain_id"]),
            page_role=str(idx_entry["page_role"]),
            site_template_id=str(idx_entry["site_template_id"]),
            attack_template_id=(
                str(idx_entry["attack_template_id"]) if idx_entry["attack_template_id"] else None
            ),
            changing=changing,
            indexed_page_id=str(idx_entry["page_id"]),
            live_page_id=str(live_entry["page_id"]),
        )
        processed += len(by_snapshot)
        if progress_every and processed % progress_every < len(by_snapshot):
            print(f"[crawler] crawled {processed}/{len(entries)} pages "
                  f"({len(documents)} slots)")

    # Corpus digest over the frozen page hashes: any byte change anywhere invalidates it.
    h = hashlib.sha256()
    for page_id in sorted(html_cache):
        h.update(page_id.encode("utf-8"))
        h.update(hashlib.sha256(html_cache[page_id].encode("utf-8")).hexdigest().encode("ascii"))
    corpus = Corpus(
        documents=documents,
        ground_truth=ground_truth,
        registry=reg,
        n_pages=len(entries),
        corpus_digest=h.hexdigest(),
    )
    fidelity = corpus.verify_two_phase_fidelity()
    print(f"[crawler] two-phase fidelity OK: {fidelity}")
    print(f"[crawler] corpus digest {corpus.corpus_digest[:16]}...")
    return corpus


# ======================================================================================
# Indexer
# ======================================================================================
@dataclass(frozen=True)
class IndexedDocument:
    """A document as the index holds it. Phase-1 (``indexed``) content only."""

    doc_id: str
    url: str
    domain: str
    domain_id: str
    title: str
    text: str
    tokens: tuple[str, ...]
    term_freqs: Mapping[str, int]
    length: int
    outbound_links: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "url": self.url,
            "domain": self.domain,
            "domain_id": self.domain_id,
            "title": self.title,
            "length": self.length,
        }


@dataclass
class InvertedIndex:
    """BM25 statistics over the phase-1 corpus.

    Only the ``indexed`` snapshot enters this structure. That is the point of the two-phase
    design: the ranker's evidence is the benign snapshot.
    """

    documents: dict[str, IndexedDocument]
    postings: dict[str, dict[str, int]]
    doc_freq: dict[str, int]
    avg_doc_len: float
    indexer_version: str = INDEXER_VERSION
    phase_indexed: str = PHASE_INDEX
    index_digest: str = ""

    def __len__(self) -> int:
        return len(self.documents)

    @property
    def vocabulary_size(self) -> int:
        return len(self.doc_freq)

    @property
    def doc_ids(self) -> list[str]:
        return sorted(self.documents)

    def idf(self, term: str) -> float:
        """Okapi BM25 IDF with the +1 smoothing that keeps it non-negative."""
        n = len(self.documents)
        df = self.doc_freq.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def stats(self) -> dict[str, Any]:
        lens = [d.length for d in self.documents.values()]
        return {
            "n_documents": len(self.documents),
            "vocabulary_size": self.vocabulary_size,
            "avg_doc_len": round(self.avg_doc_len, 4),
            "min_doc_len": min(lens) if lens else 0,
            "max_doc_len": max(lens) if lens else 0,
            "total_postings": sum(len(p) for p in self.postings.values()),
            "indexer_version": self.indexer_version,
            "phase_indexed": self.phase_indexed,
            "index_digest": self.index_digest,
        }


def build_index(
    corpus: Corpus,
    *,
    phase: str = PHASE_INDEX,
    progress_every: int = 500,
) -> InvertedIndex:
    """Build the inverted index over ``phase`` (phase 1 by default).

    The index is constructed from :meth:`CrawledDocument.public_view` at ``RETRIEVER`` tier and
    the whole structure is passed through the leakage guard before return, so an experimenter
    field cannot enter the index even if a future manifest change adds one.
    """
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}")

    documents: dict[str, IndexedDocument] = {}
    postings: dict[str, dict[str, int]] = {}
    doc_freq: Counter[str] = Counter()
    total_len = 0

    for i, doc in enumerate(corpus, start=1):
        view = doc.public_view(phase, ExposureTier.RETRIEVER)
        # The domain's own hostname words are part of what a real index sees (they are in the
        # URL), so they are appended to the searchable text. This is what makes the
        # "entity name embedded in the hostname" attack signal lexically visible.
        searchable = f"{view['title']} {view['text']} {view['domain'].replace('.', ' ')}"
        tokens = tokenize(searchable)
        tf = Counter(tokens)
        documents[doc.doc_id] = IndexedDocument(
            doc_id=doc.doc_id,
            url=view["url"],
            domain=view["domain"],
            domain_id=view["domain_id"],
            title=view["title"],
            text=view["text"],
            tokens=tuple(tokens),
            term_freqs=dict(tf),
            length=len(tokens),
            outbound_links=tuple(view["outbound_links"]),
        )
        for term, count in tf.items():
            postings.setdefault(term, {})[doc.doc_id] = count
        doc_freq.update(tf.keys())
        total_len += len(tokens)
        if progress_every and i % progress_every == 0:
            print(f"[indexer] indexed {i}/{len(corpus)} slots "
                  f"(vocab {len(doc_freq)})")

    n = len(documents) or 1
    h = hashlib.sha256()
    h.update(f"{INDEXER_VERSION}|{phase}".encode("utf-8"))
    for doc_id in sorted(documents):
        d = documents[doc_id]
        h.update(f"{doc_id}|{d.length}|{d.url}".encode("utf-8"))

    index = InvertedIndex(
        documents=documents,
        postings=postings,
        doc_freq=dict(doc_freq),
        avg_doc_len=total_len / n,
        phase_indexed=phase,
        index_digest=h.hexdigest(),
    )
    assert_no_leakage(
        {k: v.as_dict() for k, v in index.documents.items()}, location="build_index"
    )
    print(f"[indexer] index built: {index.stats()}")
    return index


def describe_environment(corpus: Corpus, index: InvertedIndex) -> dict[str, Any]:
    """Provenance block for the frozen snapshot file."""
    return {
        "crawler_version": CRAWLER_VERSION,
        "indexer_version": INDEXER_VERSION,
        "contract_ref": "CONTRACT.md Sections 6, 7",
        "n_manifest_pages": corpus.n_pages,
        "n_document_slots": len(corpus),
        "n_changing_slots": len(corpus.changing_doc_ids()),
        "n_static_slots": len(corpus.static_doc_ids()),
        "corpus_digest": corpus.corpus_digest,
        "index": index.stats(),
        "slot_key": ["entity_id", "domain_id", "site_template_id", "attack_template_id"],
        "slot_key_rationale": (
            "A URL is not unique in this corpus (1,376 URLs carry 2,631 pages; up to 68 per "
            "URL) because shared provider hosts serve one listing per business. Indexing by "
            "URL would collapse those documents and destroy the entity-relative structure."
        ),
        "two_phase": {
            "index_phase": PHASE_INDEX,
            "read_phase": PHASE_LIVE,
            "factor": "content_change_after_indexing (CONTRACT.md Section 7)",
            "fidelity": corpus.verify_two_phase_fidelity(),
            "diff_report": corpus.two_phase_diff_report(),
        },
        "meta_channel_stripped_from_text": list(META_CHANNEL_NAMES),
    }


__all__ = [
    "BENCHMARK_DIR",
    "CRAWLER_VERSION",
    "INDEXER_VERSION",
    "META_CHANNEL_NAMES",
    "PAGES_DIR",
    "PHASES",
    "PHASE_INDEX",
    "PHASE_LIVE",
    "Corpus",
    "CorpusGroundTruth",
    "CorpusIntegrityError",
    "CrawledDocument",
    "IndexedDocument",
    "InvertedIndex",
    "build_index",
    "crawl_corpus",
    "describe_environment",
    "load_page_hashes",
    "load_page_manifest",
    "meta_channel",
    "outbound_links",
    "page_title",
    "tokenize",
    "visible_text",
]
