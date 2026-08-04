"""Step 2.8 -- summary figures for the generated controlled world.

Reads only the persisted artifacts; performs no regeneration, so the figures cannot disagree
with the frozen data.

Usage:
    uv run python workflow/11_plot_benchmark_summary.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from registry.authorization_graph import ACTIONS, load_authorization_graph  # noqa: E402
from registry.entity_generator import load_entity_world  # noqa: E402
from registry.split_builder import SPLIT_NAMES, load_splits  # noqa: E402
from site_generator.generator import (  # noqa: E402
    ATTACK_FACTORS,
    load_attack_design,
    load_page_manifest,
)

FIG_DIR = REPO_ROOT / "figures"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"       wrote figures/{stem}.png (+.pdf)")


def main() -> int:
    print("=" * 78)
    print("Step 2.8 -- benchmark summary figures")
    print("=" * 78)

    world = load_entity_world()
    graph = load_authorization_graph()
    design = load_attack_design()
    splits = load_splits()
    manifest = load_page_manifest()
    print(f"[1/4] loaded {len(world.entities)} entities, {len(graph.edges)} edges, "
          f"{len(manifest)} pages")

    # ---------------------------------------------------------------- figure 1
    # Authorization rate per action x domain role: shows action-relativity directly.
    print("[2/4] authorization structure heatmap ...")
    roles = sorted({r for m in graph.entity_domain_roles.values() for r in m.values()})
    matrix = np.zeros((len(roles), len(ACTIONS)))
    counts = np.zeros_like(matrix)
    for edge in graph.edges:
        role = graph.role_of(edge.entity_id, edge.domain_id)
        i, j = roles.index(role), ACTIONS.index(edge.action)
        counts[i, j] += 1
        matrix[i, j] += 1 if edge.authorized else 0
    rate = np.divide(matrix, counts, out=np.zeros_like(matrix), where=counts > 0)

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    im = ax.imshow(rate, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ACTIONS)), ACTIONS)
    ax.set_yticks(range(len(roles)), [r.replace("_", " ") for r in roles])
    for i in range(len(roles)):
        for j in range(len(ACTIONS)):
            ax.text(j, i, f"{rate[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="black" if 0.25 < rate[i, j] < 0.85 else "white")
    ax.set_title("Authorized(e, d, a) rate by domain role and action\n"
                 "partial rows are the point: authorization is action-relative", fontsize=8)
    fig.colorbar(im, ax=ax, label="P(authorized)", fraction=0.03)
    save(fig, "benchmark_authorization_structure")

    # ---------------------------------------------------------------- figure 2
    print("[3/4] split composition ...")
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))
    axis_names = ["entity_template", "site_template", "attack_template"]
    for ax, axis in zip(axes, axis_names):
        doc = splits["axes"][axis]
        labels = list(SPLIT_NAMES)
        sizes = [doc["split_sizes"][n] for n in labels]
        if "holdout" in doc:
            labels.append(doc["holdout_name"])
            sizes.append(doc["holdout_size"])
        colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"][: len(labels)]
        bars = ax.bar(range(len(labels)), sizes, color=colours, width=0.65)
        core = doc["core_pool_size"]
        for bar, size, name in zip(bars, sizes, labels):
            pct = f"{100 * size / core:.0f}% of core" if name in SPLIT_NAMES else "holdout"
            ax.text(bar.get_x() + bar.get_width() / 2, size, f"{size}\n{pct}",
                    ha="center", va="bottom", fontsize=6)
        ax.set_xticks(range(len(labels)), [n.replace("_", "\n") for n in labels], fontsize=6)
        ax.set_title(f"{axis.replace('_', ' ')}\ncore pool = {core}", fontsize=8)
        ax.set_ylim(0, max(sizes) * 1.35)
        ax.set_ylabel("templates" if axis == axis_names[0] else "")
    fig.suptitle("Template-level splits: exactly 50/20/30 on each core pool, "
                 "holdouts disjoint (CONTRACT.md Section 5.3)", fontsize=9)
    save(fig, "benchmark_split_composition")

    # ---------------------------------------------------------------- figure 3
    print("[4/4] attack design balance and corpus composition ...")
    by_id = design.by_id
    core = [by_id[i] for i in design.core_ids]
    holdout = [by_id[i] for i in design.adaptive_holdout_ids]

    fig, axes = plt.subplots(2, 4, figsize=(10.0, 4.6))
    for ax, factor in zip(axes.ravel(), ATTACK_FACTORS):
        levels = [str(v) for v in ATTACK_FACTORS[factor]]
        core_counts = [sum(1 for t in core if str(getattr(t, factor)) == lv) for lv in levels]
        hold_counts = [sum(1 for t in holdout if str(getattr(t, factor)) == lv) for lv in levels]
        x = np.arange(len(levels))
        ax.bar(x - 0.2, core_counts, width=0.38, label="core", color="#4C72B0")
        ax.bar(x + 0.2, hold_counts, width=0.38, label="adaptive holdout", color="#C44E52")
        ax.axhline(len(core) / len(levels), ls="--", lw=0.6, color="grey")
        ax.set_xticks(x, levels, fontsize=6)
        ax.set_title(factor.replace("_", " "), fontsize=7)
        ax.tick_params(labelsize=6)
    axes.ravel()[0].legend(fontsize=6, frameon=False)
    fig.suptitle("Section 7 attack factor marginals. Dashed line = uniform expectation for the "
                 "core pool.\nThe holdout is deliberately non-uniform: it IS the "
                 "strongest-attack region.", fontsize=9)
    fig.tight_layout()
    save(fig, "benchmark_attack_design_balance")

    role_counts = Counter(p["page_role"] for p in manifest)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    order = sorted(role_counts, key=lambda r: -role_counts[r])
    ax.barh([r.replace("_", " ") for r in order], [role_counts[r] for r in order],
            color="#4C72B0")
    for i, r in enumerate(order):
        ax.text(role_counts[r], i, f" {role_counts[r]}", va="center", fontsize=6)
    ax.set_xlabel("rendered pages")
    ax.invert_yaxis()
    ax.set_title(f"Controlled corpus: {len(manifest)} pages across all 10 Section 7 page roles",
                 fontsize=8)
    ax.set_xlim(0, max(role_counts.values()) * 1.15)
    save(fig, "benchmark_corpus_composition")

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
