"""Step 3 figures: Web-RAG replay environment and adaptive attacker diagnostics.

These are *instrument* diagnostics, not results. They characterise the benchmark environment
built in Step 3: what the two-phase corpus looks like, where the surrogate pipeline's failures
attribute, and what the adaptive optimiser could and could not move. No model has been run yet.

Run::

    uv run python workflow/14_plot_web_rag_summary.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIGS = ROOT / "figures"
BENCH = ROOT / "data" / "benchmark"
RESULTS = ROOT / "results"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})

STAGE_SHORT = {
    "none": "no failure",
    "retrieval_candidates": "retrieval",
    "resolved_entities": "entity\nresolution",
    "extracted_entity_domain_relations": "relation\nextraction",
    "inferred_action_authorizations": "authorization\ninference",
    "presented_links": "answer\ngeneration",
}
STAGE_ORDER = list(STAGE_SHORT)
ACTIONS = ["browse", "contact", "book", "login", "pay"]


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def load_yaml(p: Path):
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    return yaml.load(p.read_text(encoding="utf-8"), Loader=loader)


def fig_environment(env, traces_doc) -> None:
    """Corpus structure, two-phase deltas, retrieval rank profile, failure attribution."""
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4))

    # -- (a) corpus composition ------------------------------------------------------
    ax = axes[0, 0]
    e = env["environment"]
    bars = {
        "manifest\npages": e["n_manifest_pages"],
        "document\nslots": e["n_document_slots"],
        "two-phase\nslots": e["n_changing_slots"],
        "static\nslots": e["n_static_slots"],
    }
    cols = ["#4C72B0", "#4C72B0", "#DD8452", "#8C8C8C"]
    ax.bar(list(bars), list(bars.values()), color=cols, width=0.62)
    for i, v in enumerate(bars.values()):
        ax.text(i, v + 40, f"{v:,}", ha="center", fontsize=7.5)
    ax.set_ylabel("count")
    ax.set_ylim(0, max(bars.values()) * 1.30)
    ax.set_title("(a) Corpus resolved into document slots", fontsize=8.5, loc="left")
    ax.text(
        0.97, 0.97,
        "2,631 pages -> 2,002 slots:\n629 indexed/live pairs\ncollapse into one slot each",
        transform=ax.transAxes, fontsize=6.6, color="#444",
        ha="right", va="top",
    )

    # -- (b) two-phase token delta ---------------------------------------------------
    ax = axes[0, 1]
    tp = e["two_phase"]["diff_report"]
    kinds = ["added only", "removed only", "added &\nremoved"]
    vals = [tp["live_only_added_tokens"], tp["live_only_removed_tokens"],
            tp["live_added_and_removed"]]
    ax.bar(kinds, vals, color=["#55A868", "#C44E52", "#DD8452"], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 8, str(v), ha="center", fontsize=7.5)
    ax.set_ylabel("changing slots")
    ax.set_ylim(0, max(vals) * 1.2)
    ax.set_title("(b) How the live snapshot differs from the indexed one",
                 fontsize=8.5, loc="left")
    ax.text(
        0.03, 0.62,
        f"token delta: mean {tp['mean_token_delta']:+.1f}\n"
        f"range [{tp['min_token_delta']:+d}, {tp['max_token_delta']:+d}]\n\n"
        f"Both signs occur, so 'did the page\ngrow?' is not a free signal.",
        transform=ax.transAxes, fontsize=6.8, color="#444", va="top",
    )

    # -- (c) failure-origin attribution ----------------------------------------------
    ax = axes[1, 0]
    counts = traces_doc["summary"]["failure_origin_counts"]
    total = sum(counts.values())
    labels = [STAGE_SHORT[s] for s in STAGE_ORDER]
    vals = [counts.get(s, 0) for s in STAGE_ORDER]
    cols = ["#8C8C8C"] + ["#4C72B0"] * 5
    ax.barh(range(len(labels)), vals, color=cols, height=0.62)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("traces")
    ax.set_xlim(0, max(vals) * 1.22)
    for i, v in enumerate(vals):
        ax.text(v + total * 0.012, i, f"{v} ({v / total:.0%})", va="center", fontsize=7)
    ax.set_title("(c) RQ2 failure-stage attribution (surrogate pipeline)",
                 fontsize=8.5, loc="left")

    # -- (d) attribution by action ---------------------------------------------------
    ax = axes[1, 1]
    by_action = traces_doc["summary"]["failure_origin_by_action"]
    stages = [s for s in STAGE_ORDER if any(by_action[a].get(s, 0) for a in ACTIONS)]
    bottom = np.zeros(len(ACTIONS))
    palette = {"none": "#8C8C8C", "retrieval_candidates": "#4C72B0",
               "resolved_entities": "#DD8452",
               "extracted_entity_domain_relations": "#C44E52",
               "inferred_action_authorizations": "#55A868",
               "presented_links": "#8172B3"}
    for s in stages:
        vals = np.array([by_action[a].get(s, 0) for a in ACTIONS], dtype=float)
        ax.bar(ACTIONS, vals, bottom=bottom, label=STAGE_SHORT[s].replace("\n", " "),
               color=palette[s], width=0.66)
        bottom += vals
    ax.set_ylabel("traces")
    ax.set_title("(d) Attribution by claimed action", fontsize=8.5, loc="left")
    ax.legend(fontsize=6.2, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=2)

    fig.suptitle(
        "AegisLink Step 3 -- controlled Web-RAG replay environment (instrument diagnostics, "
        "not model results)",
        fontsize=9.5, y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"step3_web_rag_environment.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  wrote figures/step3_web_rag_environment.{png,pdf}")


def fig_adaptive(report, adaptive_doc) -> None:
    """Optimiser effect per detector, suspicion distribution, design balance."""
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4))
    opt = report["optimisation"]

    # -- (a) per-detector before/after -----------------------------------------------
    ax = axes[0, 0]
    before, after = opt["mean_detector_before"], opt["mean_detector_after"]
    keys = list(before)
    short = {
        "name_in_hostname": "name in\nhostname",
        "explicit_official_claim_keywords": "official-claim\nkeywords",
        "identity_field_mismatch": "identity\nmismatch",
        "single_source_corroboration": "single\nsource",
        "risky_endpoint_path": "risky endpoint\npath",
        "boilerplate_prose_reuse": "boilerplate\nreuse",
    }
    y = np.arange(len(keys))
    ax.barh(y - 0.19, [before[k] for k in keys], height=0.36, label="before",
            color="#C44E52")
    ax.barh(y + 0.19, [after[k] for k in keys], height=0.36, label="after",
            color="#55A868")
    ax.set_yticks(y)
    ax.set_yticklabels([short[k] for k in keys], fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlabel("mean detector suspicion")
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    ax.set_title("(a) What the prose optimiser could move", fontsize=8.5, loc="left")
    # Mark the three the attacker cannot influence.
    for i, k in enumerate(keys):
        if abs(before[k] - after[k]) < 1e-12:
            ax.text(before[k] + 0.03, i, "fixed", va="center", fontsize=6,
                    color="#777", style="italic")

    # -- (b) suspicion distribution --------------------------------------------------
    ax = axes[0, 1]
    init = [t["optimisation"]["initial_suspicion"] for t in adaptive_doc["templates"]]
    fin = [t["optimisation"]["final_suspicion"] for t in adaptive_doc["templates"]]
    bins = np.linspace(min(init + fin) - 0.01, max(init + fin) + 0.01, 22)
    ax.hist(init, bins=bins, color="#C44E52", alpha=0.72, label="before")
    ax.hist(fin, bins=bins, color="#55A868", alpha=0.72, label="after")
    ax.axvline(np.mean(init), color="#C44E52", ls="--", lw=1)
    ax.axvline(np.mean(fin), color="#55A868", ls="--", lw=1)
    ax.set_xlabel("panel suspicion score")
    ax.set_ylabel("templates")
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title(
        f"(b) Panel suspicion: {np.mean(init):.3f} -> {np.mean(fin):.3f}"
        f"  ({opt['templates_improved']}/80 improved)",
        fontsize=8.5, loc="left",
    )

    # -- (c) design balance ----------------------------------------------------------
    ax = axes[1, 0]
    a_t = [t for t in adaptive_doc["templates"] if t["stratum"] == "A"]
    b_t = [t for t in adaptive_doc["templates"] if t["stratum"] == "B"]
    ca = Counter(t["action_claim"] for t in a_t)
    cb = Counter(t["action_claim"] for t in b_t)
    x = np.arange(len(ACTIONS))
    ax.bar(x - 0.19, [ca[a] for a in ACTIONS], width=0.36,
           label="Stratum A (in region, primary-eligible)", color="#4C72B0")
    ax.bar(x + 0.19, [cb[a] for a in ACTIONS], width=0.36,
           label="Stratum B (partial identity, secondary only)", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(ACTIONS)
    ax.set_ylabel("templates")
    ax.set_ylim(0, 11)
    ax.legend(fontsize=6.2, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.15))
    ax.set_title("(c) Adaptive design balance by claimed action", fontsize=8.5, loc="left")

    # -- (d) improvement vs cluster size / content change ----------------------------
    ax = axes[1, 1]
    groups: dict[str, list[float]] = {}
    for t in adaptive_doc["templates"]:
        key = (f"{t['corroborating_sources']} src / "
               f"{'change' if t['content_change_after_indexing'] else 'static'}")
        groups.setdefault(key, []).append(t["optimisation"]["improvement"])
    order = sorted(groups)
    ax.boxplot([groups[k] for k in order], tick_labels=order, widths=0.55,
               medianprops={"color": "#C44E52"}, flierprops={"markersize": 2.5})
    ax.set_ylabel("suspicion reduction")
    ax.tick_params(axis="x", labelsize=6.2, rotation=18)
    ax.set_title("(d) Suspicion reduction by corroboration cluster and phase",
                 fontsize=8.5, loc="left")

    fig.suptitle(
        "AegisLink Step 3 -- adaptive attacker in the pre-declared holdout region "
        "(defense panel is a PROXY for the Step 4 scorer)",
        fontsize=9.5, y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"step3_adaptive_attacker.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  wrote figures/step3_adaptive_attacker.{png,pdf}")


def fig_retrieval_profile(snapshot, traces_doc) -> None:
    """What the retriever surfaces, and how often the top rank is unauthorized."""
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.3))

    # -- (a) score decay by rank -----------------------------------------------------
    ax = axes[0]
    by_rank: dict[int, list[float]] = {}
    for s in snapshot["snapshots"]:
        for c in s["candidates"]:
            by_rank.setdefault(c["rank"], []).append(c["score"])
    ranks = sorted(by_rank)
    med = [float(np.median(by_rank[r])) for r in ranks]
    lo = [float(np.percentile(by_rank[r], 10)) for r in ranks]
    hi = [float(np.percentile(by_rank[r], 90)) for r in ranks]
    ax.plot(ranks, med, "o-", ms=3, lw=1.2, color="#4C72B0")
    ax.fill_between(ranks, lo, hi, alpha=0.2, color="#4C72B0")
    ax.set_xlabel("rank")
    ax.set_ylabel("fused RRF score")
    ax.set_title("(a) Rank/score profile (median, 10-90%)", fontsize=8.5, loc="left")

    # -- (b) rank at which the first authorized domain appears -----------------------
    # More informative than a binary "is rank 1 authorized?": it shows the ranker
    # systematically places attacker-controlled hostnames above the authorized ones without
    # implying the authorized domain is absent.
    ax = axes[1]
    curves: dict[str, list[float]] = {}
    for a in ACTIONS:
        ranks = []
        for t in traces_doc["traces"]:
            if t["action"] != a:
                continue
            s1, s4 = t["stages"][0], t["stages"][3]
            doms = s1["output"]["domain_ids"]
            if not any(s4["expected"].get(d, False) for d in doms):
                continue  # nothing authorized retrievable; excluded, not counted as rank 11
            ranks.append(
                next(i + 1 for i, d in enumerate(doms) if s4["expected"].get(d, False))
            )
        n = len(ranks) or 1
        curves[a] = [sum(1 for r in ranks if r <= k) / n for k in range(1, 11)]
    colours = {"browse": "#4C72B0", "contact": "#DD8452", "book": "#55A868",
               "login": "#C44E52", "pay": "#8172B3"}
    for a in ACTIONS:
        ax.plot(range(1, 11), curves[a], "o-", ms=2.6, lw=1.1, color=colours[a], label=a)
    ax.set_xlabel("rank cutoff k")
    ax.set_ylabel("P(first authorized domain at rank <= k)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6.2, frameon=False, loc="lower right", ncol=2)
    ax.set_title(
        "(b) Where the authorized domain ranks\n"
        "     (rank 1 only for 'book'; never for 'contact')",
        fontsize=8.0, loc="left",
    )

    # -- (c) unscoped relation disagreement ------------------------------------------
    ax = axes[2]
    dis = [
        t["stages"][2]["n_disagreements_all_candidates"] for t in traces_doc["traces"]
    ]
    ax.hist(dis, bins=np.arange(-0.5, 11.5, 1), color="#8172B3", alpha=0.85)
    ax.set_xlabel("candidate relations misread (of 10)")
    ax.set_ylabel("traces")
    ax.set_title("(c) Why the unscoped rule was degenerate", fontsize=8.5, loc="left")
    ax.text(
        0.03, 0.95,
        "Comparing every candidate makes\nsome relation almost always wrong,\n"
        "so stage 3 would absorb 94% of\nattributions. Attribution is therefore\n"
        "scoped to the outcome-determining\ndomains.",
        transform=ax.transAxes, fontsize=6.3, color="#444", va="top",
    )

    fig.suptitle(
        "AegisLink Step 3 -- frozen retrieval replay: what the environment surfaces",
        fontsize=9.5, y=1.01,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"step3_retrieval_profile.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  wrote figures/step3_retrieval_profile.{png,pdf}")


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    print("Rendering Step 3 figures")
    env = load_json(RESULTS / "web_rag_environment.json")
    traces_doc = load_json(BENCH / "stage_traces.json")
    snapshot = load_json(BENCH / "retrieval_snapshots.json")
    report = load_json(RESULTS / "adaptive_attacker_report.json")
    adaptive_doc = load_yaml(ROOT / "attacks" / "adaptive_templates.yaml")

    fig_environment(env, traces_doc)
    fig_adaptive(report, adaptive_doc)
    fig_retrieval_profile(snapshot, traces_doc)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
