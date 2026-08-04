"""Deterministic BM25 + dense retriever and reranker over the phase-1 index.

CONTRACT.md Section 6 (``retriever``, ``reranker`` services; "Freeze retrieval results for
primary experiments").

Determinism is the requirement
------------------------------
Steps 4-8 compare ten baselines and a defense on the same evidence. If retrieval drifted
between runs, a measured difference between two defenses would be confounded with a difference
in what they were shown. Three mechanisms make this airtight:

1. **No learned weights, no network, no clock, no RNG.** BM25 statistics come from the frozen
   index; the dense encoder is a pure function of the text.
2. **Rounded scores, total order.** Every score is rounded to
   :data:`SCORE_PRECISION` decimals before ordering, and the sort key is
   ``(-score, doc_id)``. Ties therefore resolve on a stable string, not on dict iteration or
   float accumulation order.
3. **Config in the fingerprint.** :meth:`RetrieverConfig.digest` enters the frozen snapshot, so
   changing ``k1``, the fusion weights, or the encoder invalidates the freeze loudly rather
   than silently changing the evidence Step 5 reads.

On the "dense" ranker
---------------------
:class:`HashedNGramEncoder` is a *deterministic sketch*, not a learned neural encoder: signed
feature hashing of word unigrams/bigrams and character 4-grams into
:data:`DENSE_DIMENSIONS` dimensions, L2-normalised, scored by cosine. It is labelled as such
everywhere rather than described as a "dense retriever" and left to imply a trained model.

Why a sketch rather than a pinned sentence-transformer, given Section 6 asks for pinned
open-weight models? Section 6's pinning requirement is about the *inference* models that
produce the claims (the LLM reader); the retriever's binding requirement is that its output be
frozen and reproducible. A hashed sketch satisfies that with no download, no GPU
non-determinism and no version skew. The cost is stated plainly: it captures lexical and
sub-lexical similarity, not paraphrase. Because it is reached through the :class:`Encoder`
protocol and its ``encoder_id`` is in the fingerprint, Step 4 can substitute a pinned
open-weight encoder and every frozen hash will mechanically change, forcing a re-freeze
instead of permitting a silent mix of old and new evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from web_rag.crawler_indexer import InvertedIndex, tokenize
from web_rag.exposure import ExposureTier, assert_no_leakage

RETRIEVER_VERSION = "1.0"

#: Decimal places retained before ordering. 12 is well inside float64 precision for these
#: score magnitudes and well outside the accumulation noise that could reorder near-ties.
SCORE_PRECISION = 12

#: Dimensionality of the hashed sketch.
DENSE_DIMENSIONS = 256

#: Okapi BM25 parameters. The standard defaults; recorded in the fingerprint so they are
#: preregistered by construction rather than tuned later.
BM25_K1 = 1.2
BM25_B = 0.75

#: Reciprocal Rank Fusion constant. 60 is the value from the original RRF paper.
RRF_K = 60

RANKING_MODES: tuple[str, ...] = ("bm25", "dense", "hybrid_rrf")

_CHAR_NGRAM = 4


def _round(x: float) -> float:
    return round(float(x), SCORE_PRECISION)


# ======================================================================================
# Dense sketch encoder
# ======================================================================================
@runtime_checkable
class Encoder(Protocol):
    """Substitution point for a pinned open-weight encoder in a later step."""

    encoder_id: str
    dimensions: int

    def encode(self, text: str) -> dict[int, float]:
        """Return a sparse L2-normalised vector as ``{dimension: weight}``."""


@dataclass(frozen=True)
class HashedNGramEncoder:
    """Signed feature hashing of word and character n-grams into a fixed-dim sketch.

    Deterministic across processes and platforms: uses BLAKE2b with a fixed key rather than
    Python's randomised :func:`hash`, whose seed varies per interpreter run and would make
    every frozen snapshot unreproducible.
    """

    dimensions: int = DENSE_DIMENSIONS
    encoder_id: str = f"hashed-ngram-sketch-v1-d{DENSE_DIMENSIONS}"
    use_word_bigrams: bool = True
    use_char_ngrams: bool = True

    def _features(self, text: str) -> Counter[str]:
        toks = tokenize(text)
        feats: Counter[str] = Counter(f"w:{t}" for t in toks)
        if self.use_word_bigrams:
            feats.update(f"b:{a}_{b}" for a, b in zip(toks, toks[1:]))
        if self.use_char_ngrams:
            flat = " ".join(toks)
            feats.update(
                f"c:{flat[i:i + _CHAR_NGRAM]}"
                for i in range(max(0, len(flat) - _CHAR_NGRAM + 1))
            )
        return feats

    def encode(self, text: str) -> dict[int, float]:
        """Sub-linear term weighting, signed hashing, L2 normalisation."""
        vec: dict[int, float] = {}
        for feat, count in self._features(text).items():
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "big")
            dim = raw % self.dimensions
            sign = 1.0 if (raw >> 63) & 1 else -1.0
            vec[dim] = vec.get(dim, 0.0) + sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm == 0.0:
            return {}
        return {d: v / norm for d, v in sorted(vec.items())}

    def as_dict(self) -> dict[str, Any]:
        return {
            "encoder_id": self.encoder_id,
            "dimensions": self.dimensions,
            "kind": "deterministic hashed n-gram sketch (NOT a learned encoder)",
            "hash": "blake2b-64, fixed key",
            "features": ["word unigrams"]
            + (["word bigrams"] if self.use_word_bigrams else [])
            + ([f"char {_CHAR_NGRAM}-grams"] if self.use_char_ngrams else []),
            "normalisation": "L2",
            "similarity": "cosine",
        }


def cosine(a: Mapping[int, float], b: Mapping[int, float]) -> float:
    """Cosine similarity of two already-normalised sparse vectors."""
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(d, 0.0) for d, w in a.items())


# ======================================================================================
# Queries
# ======================================================================================
@dataclass(frozen=True)
class Query:
    """One entity search query.

    ``entity_id`` and ``action`` are *experiment addressing*, not evidence: they say which
    (entity, action) cell this query belongs to so the evaluator can score it. They are never
    handed to the ranker, which sees only :attr:`text`.
    """

    query_id: str
    text: str
    entity_id: str
    action: str
    intent: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Query surface per action. Deliberately natural-language and free of the label vocabulary:
#: a query containing "authorized" would hand BM25 the answer key.
ACTION_QUERY_TEMPLATES: dict[str, str] = {
    "browse": "{name} opening hours and menu",
    "contact": "{name} phone number and address",
    "book": "book a table at {name}",
    "login": "{name} customer account sign in",
    "pay": "pay {name} online",
}

#: An action-neutral navigational query, the commonest real web-RAG intent.
NAVIGATIONAL_QUERY_TEMPLATE = "{name} official website"

CANONICAL_ACTIONS: tuple[str, ...] = ("browse", "contact", "book", "login", "pay")


def build_entity_queries(
    entities: Sequence[Mapping[str, Any]],
    *,
    actions: Sequence[str] = CANONICAL_ACTIONS,
    include_navigational: bool = True,
) -> list[Query]:
    """Build the frozen query set: one navigational plus one per action, per entity.

    The query text uses the entity's canonical name only. Aliases are deliberately *not* used
    to build extra queries here: alias resolution is part of the entity-resolution stage RQ2
    measures, and pre-expanding it in the query would move that difficulty out of the system
    under test and into the harness.
    """
    queries: list[Query] = []
    for ent in entities:
        eid = str(ent["entity_id"])
        name = str(ent["canonical_name"])
        if include_navigational:
            queries.append(
                Query(
                    query_id=f"Q-{eid}-nav",
                    text=NAVIGATIONAL_QUERY_TEMPLATE.format(name=name),
                    entity_id=eid,
                    action="browse",
                    intent="navigational",
                )
            )
        for action in actions:
            queries.append(
                Query(
                    query_id=f"Q-{eid}-{action}",
                    text=ACTION_QUERY_TEMPLATES[action].format(name=name),
                    entity_id=eid,
                    action=action,
                    intent="action",
                )
            )
    return queries


# ======================================================================================
# Configuration and results
# ======================================================================================
@dataclass(frozen=True)
class RetrieverConfig:
    """Frozen retrieval configuration. Enters the replay fingerprint."""

    mode: str = "hybrid_rrf"
    top_k: int = 10
    candidate_pool: int = 50
    k1: float = BM25_K1
    b: float = BM25_B
    rrf_k: int = RRF_K
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    score_precision: int = SCORE_PRECISION
    retriever_version: str = RETRIEVER_VERSION

    def __post_init__(self) -> None:
        if self.mode not in RANKING_MODES:
            raise ValueError(f"mode must be one of {RANKING_MODES}, got {self.mode!r}")
        if self.top_k < 1 or self.candidate_pool < self.top_k:
            raise ValueError("require candidate_pool >= top_k >= 1")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class Candidate:
    """One retrieved document at ``READER`` tier. No provenance, no ground truth."""

    rank: int
    doc_id: str
    url: str
    domain: str
    domain_id: str
    title: str
    score: float
    bm25_score: float
    dense_score: float
    bm25_rank: int | None
    dense_rank: int | None
    snippet: str

    def as_dict(self, tier: ExposureTier = ExposureTier.READER) -> dict[str, Any]:
        out = {
            "rank": self.rank,
            "doc_id": self.doc_id,
            "url": self.url,
            "domain": self.domain,
            "domain_id": self.domain_id,
            "title": self.title,
            "score": self.score,
            "bm25_score": self.bm25_score,
            "dense_score": self.dense_score,
            "bm25_rank": self.bm25_rank,
            "dense_rank": self.dense_rank,
        }
        if tier in (ExposureTier.READER, ExposureTier.VERIFIER, ExposureTier.EVALUATOR):
            out["snippet"] = self.snippet
        return out

    def hashable(self) -> dict[str, Any]:
        """The subset that the per-query integrity hash covers.

        Snippet text is excluded deliberately: it is a derived view of page bytes already
        covered by ``corpus_digest``, and including it would make the hash sensitive to a
        cosmetic snippet-length change while adding no integrity guarantee.
        """
        return {
            "rank": self.rank,
            "doc_id": self.doc_id,
            "url": self.url,
            "score": self.score,
            "bm25_score": self.bm25_score,
            "dense_score": self.dense_score,
        }


@dataclass(frozen=True)
class RetrievalResult:
    """Frozen retrieval output for one query."""

    query_id: str
    query_text: str
    entity_id: str
    action: str
    intent: str
    mode: str
    candidates: tuple[Candidate, ...]
    sha256: str

    def as_dict(self, tier: ExposureTier = ExposureTier.READER) -> dict[str, Any]:
        out: dict[str, Any] = {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "mode": self.mode,
            "candidates": [c.as_dict(tier) for c in self.candidates],
            "sha256": self.sha256,
        }
        if tier is ExposureTier.EVALUATOR:
            out["entity_id"] = self.entity_id
            out["action"] = self.action
            out["intent"] = self.intent
        return out

    @property
    def doc_ids(self) -> list[str]:
        return [c.doc_id for c in self.candidates]


def candidate_hash(query_id: str, candidates: Sequence[Candidate]) -> str:
    """SHA-256 over the canonical JSON of an *ordered* candidate list."""
    payload = {
        "query_id": query_id,
        "candidates": [c.hashable() for c in candidates],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


# ======================================================================================
# Retriever
# ======================================================================================
_SNIPPET_CHARS = 320


class Retriever:
    """Deterministic BM25 / dense / hybrid retriever over a phase-1 index."""

    def __init__(
        self,
        index: InvertedIndex,
        *,
        config: RetrieverConfig | None = None,
        encoder: Encoder | None = None,
    ) -> None:
        self.index = index
        self.config = config or RetrieverConfig()
        self.encoder: Encoder = encoder or HashedNGramEncoder()
        self._doc_vectors: dict[str, dict[int, float]] = {}
        self._needs_dense = self.config.mode in ("dense", "hybrid_rrf")
        if self._needs_dense:
            self._build_dense_vectors()

    # -- setup ------------------------------------------------------------------------
    def _build_dense_vectors(self, progress_every: int = 500) -> None:
        for i, doc_id in enumerate(self.index.doc_ids, start=1):
            d = self.index.documents[doc_id]
            self._doc_vectors[doc_id] = self.encoder.encode(f"{d.title} {d.text} {d.domain}")
            if progress_every and i % progress_every == 0:
                print(f"[retriever] encoded {i}/{len(self.index)} documents")

    # -- scorers ----------------------------------------------------------------------
    def bm25_scores(self, query_text: str) -> dict[str, float]:
        """Okapi BM25 over the phase-1 index. Only documents with a match are scored."""
        k1, b = self.config.k1, self.config.b
        avg = self.index.avg_doc_len or 1.0
        scores: dict[str, float] = {}
        for term in tokenize(query_text):
            postings = self.index.postings.get(term)
            if not postings:
                continue
            idf = self.index.idf(term)
            for doc_id, tf in postings.items():
                dl = self.index.documents[doc_id].length
                denom = tf + k1 * (1.0 - b + b * dl / avg)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (tf * (k1 + 1.0)) / denom
        return {d: _round(s) for d, s in scores.items()}

    def dense_scores(self, query_text: str, restrict_to: Iterable[str] | None = None) -> dict[str, float]:
        """Cosine similarity of the query sketch against document sketches."""
        qv = self.encoder.encode(query_text)
        if not qv:
            return {}
        pool = list(restrict_to) if restrict_to is not None else self.index.doc_ids
        return {d: _round(cosine(qv, self._doc_vectors.get(d, {}))) for d in pool}

    # -- ordering ---------------------------------------------------------------------
    @staticmethod
    def _rank(scores: Mapping[str, float], *, drop_nonpositive: bool = True) -> list[str]:
        """Total order over doc ids: descending score, then ascending doc_id."""
        items = [(d, s) for d, s in scores.items() if not (drop_nonpositive and s <= 0.0)]
        items.sort(key=lambda kv: (-kv[1], kv[0]))
        return [d for d, _ in items]

    def search(self, query: Query) -> RetrievalResult:
        """Retrieve for one query and return a hashed, frozen result.

        The hybrid mode uses weighted Reciprocal Rank Fusion rather than score addition: BM25
        scores are unbounded and cosine scores live in ``[-1, 1]``, so summing them would let
        BM25 dominate by scale alone and make the "dense" contribution decorative.
        """
        cfg = self.config
        bm = self.bm25_scores(query.text)
        bm_order = self._rank(bm)
        bm_rank = {d: i + 1 for i, d in enumerate(bm_order)}

        if cfg.mode == "bm25":
            dn: dict[str, float] = {}
            dn_order: list[str] = []
            fused_order = bm_order[: cfg.candidate_pool]
            fused = {d: bm[d] for d in fused_order}
        else:
            # Dense is scored over the BM25 candidate pool plus, for pure-dense mode, the
            # whole corpus. Restricting hybrid's dense pass to the BM25 pool is the standard
            # retrieve-then-rerank arrangement and is recorded as such.
            if cfg.mode == "dense":
                dn = self.dense_scores(query.text)
            else:
                dn = self.dense_scores(query.text, restrict_to=bm_order[: cfg.candidate_pool])
            dn_order = self._rank(dn)
            if cfg.mode == "dense":
                fused_order = dn_order[: cfg.candidate_pool]
                fused = {d: dn[d] for d in fused_order}
            else:
                dn_rank_map = {d: i + 1 for i, d in enumerate(dn_order)}
                rrf: dict[str, float] = {}
                for d in set(bm_order[: cfg.candidate_pool]) | set(dn_order):
                    s = 0.0
                    if d in bm_rank:
                        s += cfg.bm25_weight / (cfg.rrf_k + bm_rank[d])
                    if d in dn_rank_map:
                        s += cfg.dense_weight / (cfg.rrf_k + dn_rank_map[d])
                    rrf[d] = _round(s)
                fused_order = self._rank(rrf)[: cfg.candidate_pool]
                fused = rrf

        dn_rank = {d: i + 1 for i, d in enumerate(dn_order)}
        candidates: list[Candidate] = []
        for i, doc_id in enumerate(fused_order[: cfg.top_k], start=1):
            d = self.index.documents[doc_id]
            candidates.append(
                Candidate(
                    rank=i,
                    doc_id=doc_id,
                    url=d.url,
                    domain=d.domain,
                    domain_id=d.domain_id,
                    title=d.title,
                    score=_round(fused.get(doc_id, 0.0)),
                    bm25_score=bm.get(doc_id, 0.0),
                    dense_score=dn.get(doc_id, 0.0),
                    bm25_rank=bm_rank.get(doc_id),
                    dense_rank=dn_rank.get(doc_id),
                    snippet=d.text[:_SNIPPET_CHARS],
                )
            )

        result = RetrievalResult(
            query_id=query.query_id,
            query_text=query.text,
            entity_id=query.entity_id,
            action=query.action,
            intent=query.intent,
            mode=cfg.mode,
            candidates=tuple(candidates),
            sha256=candidate_hash(query.query_id, candidates),
        )
        assert_no_leakage(
            result.as_dict(ExposureTier.READER), location=f"Retriever.search[{query.query_id}]"
        )
        return result

    def search_all(self, queries: Sequence[Query], *, progress_every: int = 100) -> list[RetrievalResult]:
        """Retrieve for every query, preserving input order."""
        out: list[RetrievalResult] = []
        for i, q in enumerate(queries, start=1):
            out.append(self.search(q))
            if progress_every and i % progress_every == 0:
                print(f"[retriever] retrieved {i}/{len(queries)} queries")
        return out

    # -- provenance -------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {
            "retriever_version": RETRIEVER_VERSION,
            "contract_ref": "CONTRACT.md Section 6",
            "config": self.config.as_dict(),
            "config_digest": self.config.digest(),
            "encoder": self.encoder.as_dict()
            if hasattr(self.encoder, "as_dict")
            else {"encoder_id": self.encoder.encoder_id},
            "index_digest": self.index.index_digest,
            "phase_indexed": self.index.phase_indexed,
            "determinism": [
                "no RNG, no clock, no network, no learned weights",
                f"scores rounded to {SCORE_PRECISION} dp before ordering",
                "total order via (-score, doc_id); ties resolve on doc_id",
                "blake2b fixed-key hashing, not PYTHONHASHSEED-dependent hash()",
            ],
            "fusion": (
                "weighted Reciprocal Rank Fusion (k=%d). Chosen over score addition because "
                "BM25 is unbounded while cosine is in [-1, 1]; summing would let BM25 dominate "
                "by scale alone." % self.config.rrf_k
            ),
            "dense_caveat": (
                "The dense ranker is a deterministic hashed n-gram sketch, not a learned "
                "encoder. It models lexical and sub-lexical similarity, not paraphrase. "
                "encoder_id is in the fingerprint, so substituting a pinned open-weight "
                "encoder invalidates every frozen hash and forces a re-freeze."
            ),
        }


__all__ = [
    "ACTION_QUERY_TEMPLATES",
    "BM25_B",
    "BM25_K1",
    "CANONICAL_ACTIONS",
    "DENSE_DIMENSIONS",
    "RANKING_MODES",
    "RETRIEVER_VERSION",
    "RRF_K",
    "SCORE_PRECISION",
    "Candidate",
    "Encoder",
    "HashedNGramEncoder",
    "Query",
    "RetrievalResult",
    "Retriever",
    "RetrieverConfig",
    "build_entity_queries",
    "candidate_hash",
    "cosine",
]
