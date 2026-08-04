"""B07 -- TopoGuard-style graph anomaly detection. CONTRACT.md Section 10, item 7.

Prior work: Dahal and Xiong, arXiv:2607.20437 (graph defense against split-knowledge attacks), listed
in CONTRACT.md Section 4.1 as work that cannot be claimed as novel.

The idea is unsupervised and purely structural. Build the mention graph over the retrieved candidate
set -- nodes are hosts, an edge joins two hosts when one references the other or both discuss the
same entity -- and flag nodes whose local topology is anomalous: unusually high degree into a tight
neighbourhood, membership in a dense near-clique, or a host that appears only in this one entity's
neighbourhood. Poisoning campaigns tend to introduce exactly those signatures.

What it can and cannot see
--------------------------
It detects the *Sybil* structure well: five co-registered corroborators around one listing form a
dense, entity-local subgraph, which is what this family of detector is for.

It is blind to the single-page attack. An impersonating official site or a replaced payment endpoint
adds one node with ordinary degree, structurally indistinguishable from a legitimate third-party
provider page. And because the score is a property of a *node*, it cannot express action relativity:
a host flagged for ``pay`` is flagged for ``browse`` identically.

Implemented from the described mechanism rather than from released code, so the row is labelled
"TopoGuard-style" and not a reproduction of the published system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aegislink.framework import (
    InputKind,
    InputRestriction,
    VerificationContext,
    jaccard,
    sigmoid,
)
from baselines.common import (
    BaseDefense,
    mentions_entity,
    page_text_for,
    references_domain,
)


@dataclass
class GraphAnomalyDetector(BaseDefense):
    """Structural anomaly score over the retrieved candidate mention graph."""

    defense_id: str = "B07_graph_anomaly_detector"
    label: str = "B07 graph anomaly detection (TopoGuard-style)"
    contract_ref: str = "CONTRACT.md Section 10, item 7; prior work arXiv:2607.20437"
    restrictions: tuple[InputRestriction, ...] = field(
        default_factory=lambda: (
            InputRestriction(
                InputKind.ACTION_TYPE,
                "an anomaly score is a property of a node, so it is action-independent by "
                "construction",
            ),
            InputRestriction(
                InputKind.REGISTRY_OFFICIAL,
                "unsupervised and structural: consulting an authoritative registry would make it "
                "a different (supervised) method",
            ),
        )
    )
    notes: str = (
        "Prediction before measurement: detects the Sybil corroboration arm, blind to the "
        "single-page impersonation and replaced-endpoint arms."
    )
    is_surrogate: bool = True
    surrogate_note: str = (
        "Implemented from the mechanism described in arXiv:2607.20437 (structural anomaly over the "
        "retrieved mention graph), not from released code. Labelled TopoGuard-STYLE for that "
        "reason; it is not a reproduction of the published system."
    )

    #: Text-similarity cut above which two candidate pages are treated as topologically adjacent.
    near_duplicate_threshold: float = 0.45
    w_bias: float = 1.65
    w_clique: float = -1.30
    w_entity_local: float = -0.70
    #: Near-zero on purpose. A raw-degree penalty is a POPULARITY penalty, and the most-linked node
    #: in this graph is the entity's own official site -- every partner and every forger links to
    #: it. With w_degree at -0.30 the detector rejected official links (FRR 0.57) while passing
    #: low-degree attacker pages (UALER 1.0), i.e. it had the signal exactly backwards. A
    #: split-knowledge graph defense keys on DENSITY, not on being well connected.
    w_degree: float = -0.05

    def probability(
        self, entity_id: str, domain_id: str, action: str, ctx: VerificationContext
    ) -> tuple[float, tuple[str, ...]]:
        ids = [
            d
            for d in ctx.candidate_domain_ids()
            if mentions_entity(ctx, entity_id, page_text_for(ctx, d))
        ]
        if domain_id not in ids:
            return sigmoid(self.w_bias), ("node absent from the entity's mention graph",)

        # Adjacency: reference edges plus near-duplicate-content edges.
        # Reference edges come from outbound hyperlinks (ctx.references_domain), not from
        # hostname-in-text: visible text never carries a hostname, so a text-based test would leave
        # the graph edgeless apart from near-duplicate links and the anomaly score meaningless.
        adj: dict[str, set[str]] = {d: set() for d in ids}
        for a in ids:
            for b in ids:
                if a == b:
                    continue
                if references_domain(ctx, a, b):
                    adj[a].add(b)
                    adj[b].add(a)
                sim = jaccard(ctx.shingle_set(ctx.candidate_by_domain(a).doc_id),
                              ctx.shingle_set(ctx.candidate_by_domain(b).doc_id))
                if sim >= self.near_duplicate_threshold:
                    adj[a].add(b)
                    adj[b].add(a)

        nb = adj[domain_id]
        degree = len(nb)
        # Local clustering coefficient: how near-clique the neighbourhood is.
        if degree >= 2:
            pairs = [(x, y) for i, x in enumerate(sorted(nb)) for y in sorted(nb)[i + 1 :]]
            linked = sum(1 for x, y in pairs if y in adj[x])
            clustering = linked / len(pairs) if pairs else 0.0
        else:
            clustering = 0.0

        # Entity-locality: a host whose retrieved neighbourhood is entirely this one entity's
        # pages, and which the registry does not mark as serving many listings.
        entity_local = 1.0 if (degree >= 1 and not ctx.is_shared_host(domain_id)) else 0.0

        z = (
            self.w_bias
            + self.w_clique * clustering * max(0, degree - 1)
            + self.w_entity_local * entity_local
            + self.w_degree * degree
        )
        return sigmoid(z), (
            f"degree={degree}; local clustering={clustering:.3f}; entity_local={bool(entity_local)}",
        )


def describe() -> dict[str, Any]:
    d = GraphAnomalyDetector()
    return {
        **d.metadata().as_dict(),
        "graph_construction": (
            "nodes = retrieved hosts mentioning the entity; edges = host-reference edges plus "
            f"near-duplicate content edges at shingle Jaccard >= {d.near_duplicate_threshold}"
        ),
        "features": ["local clustering coefficient (primary)", "entity locality", "degree (near-neutral)"],
        "why_degree_is_not_penalised": (
            "The most-linked node in this graph is the entity's own official site, because every "
            "partner and every forger links to it. Penalising degree made the detector reject "
            "official links and pass low-degree attacker pages -- the signal inverted. Density, "
            "not popularity, is what a split-knowledge graph defense keys on."
        ),
        "weights": {
            "bias": d.w_bias,
            "clique": d.w_clique,
            "entity_local": d.w_entity_local,
            "degree": d.w_degree,
        },
        "structural_blindness": (
            "A single-node attack (impersonating site, replaced endpoint) has ordinary local "
            "topology, and a node score cannot represent action relativity."
        ),
    }


__all__ = ["GraphAnomalyDetector", "describe"]
