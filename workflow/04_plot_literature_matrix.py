"""Visualize the frozen literature-collision matrix.

Renders the four Section 13 overlap axes for every scored paper as a heatmap, with the
collision decision rule drawn on the figure so a reader can see at a glance how far each
prior work sits from the threshold. Reads only ``results/literature_matrix.yaml`` -- no
scores are recomputed here, so the figure cannot disagree with the matrix.

Usage
-----
    uv run python workflow/04_plot_literature_matrix.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never attempt an interactive window

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

REPO_ROOT = Path(__file__).resolve().parent.parent
MATRIX_PATH = REPO_ROOT / "results" / "literature_matrix.yaml"
FIG_DIR = REPO_ROOT / "figures"

AXES = ["problem_overlap", "attack_overlap", "method_overlap", "evaluation_overlap"]
AXIS_LABELS = ["Problem\noverlap", "Attack\noverlap", "Method\noverlap", "Evaluation\noverlap"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.linewidth": 0.5,
    "savefig.dpi": 300,
})


def main() -> int:
    if not MATRIX_PATH.exists():
        print(f"[error] {MATRIX_PATH} not found; run evaluation/literature_checker.py first",
              file=sys.stderr)
        return 1

    with MATRIX_PATH.open(encoding="utf-8") as fh:
        matrix = yaml.safe_load(fh)

    priors = matrix["prior_works"]
    discovered = [r for r in matrix.get("discovered_candidates", []) if r.get("scored")]
    records = priors + discovered
    if not records:
        print("[error] matrix contains no scored records", file=sys.stderr)
        return 1

    labels = [f"{r['paper_id']}  {r.get('short_name') or r['arxiv_id']}" for r in priors]
    labels += [f"{r['paper_id']}  arXiv:{r['arxiv_id']}" for r in discovered]
    scores = np.array([[r[a] for a in AXES] for r in records], dtype=int)

    p_thresh = int(matrix["rubric"]["collision_rule"].split(">=")[1].split()[0])

    # Discrete 0-3 scale: a continuous colormap would imply precision the integer
    # marker-count scores do not have.
    cmap = LinearSegmentedColormap.from_list(
        "overlap", ["#f7f7f7", "#c6dbef", "#6baed6", "#08519c"], N=4
    )
    norm = BoundaryNorm([0, 1, 2, 3, 4], cmap.N)

    fig_h = 1.6 + 0.30 * len(records)
    fig, ax = plt.subplots(figsize=(5.6, fig_h))
    im = ax.imshow(scores, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(AXES)), AXIS_LABELS)
    ax.set_yticks(range(len(records)), labels, fontsize=7)
    ax.set_xticks(np.arange(-0.5, len(AXES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(records), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=2)

    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            v = scores[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                    color="white" if v >= 2 else "#333333",
                    fontweight="bold" if v >= p_thresh else "normal")

    # Mark the two axes that can trigger a collision; the other two cannot, per Section 13.
    for j, axis in enumerate(AXES):
        if axis in ("problem_overlap", "method_overlap"):
            ax.add_patch(plt.Rectangle(
                (j - 0.5, -0.5), 1, len(records),
                fill=False, edgecolor="#cb181d", linewidth=1.4, zorder=5,
            ))

    # Separate the declared priors from the search-discovered candidates. The distinction is
    # explained in the caption rather than inline, to keep clear of the colorbar.
    if discovered:
        ax.axhline(len(priors) - 0.5, color="#333333", linewidth=1.0, linestyle="--")

    cbar = fig.colorbar(im, ax=ax, ticks=[0.5, 1.5, 2.5, 3.5], pad=0.02, fraction=0.035)
    cbar.ax.set_yticklabels(["0\nabsent", "1", "2", "3\ncore"], fontsize=6)
    cbar.outline.set_linewidth(0.5)

    verdict = matrix["verdict"]
    ax.set_title(
        f"Literature collision matrix -- {verdict}\n"
        f"collision requires problem $\\geq$ {p_thresh} AND method $\\geq$ {p_thresh} "
        "(red outline)",
        fontsize=8.5, pad=8,
    )
    n_tier1 = matrix["counts"]["prior_works_metadata_verified"]
    caption = (
        f"Scored by frozen regex rubric v{matrix['rubric']['version']} (no LLM judgement); "
        f"{n_tier1}/{len(priors)} priors tier-1 verified against the live arXiv API. "
        f"Max observed: problem={scores[:, 0].max()}, method={scores[:, 2].max()} "
        f"(threshold {p_thresh})."
    )
    if discovered:
        caption += ("\nP-rows: priors declared in CONTRACT.md Sec. 4.1. "
                    "D-rows (below dashed line): surfaced by the Sec. 13 concept searches.")
    fig.text(0.5, 0.004, caption, ha="center", va="bottom", fontsize=5.8, color="#555555")

    fig.tight_layout(rect=(0, 0.045 if discovered else 0.025, 1, 1))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = FIG_DIR / f"literature_overlap_matrix.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"  wrote {out.relative_to(REPO_ROOT)}")
    plt.close(fig)

    print(f"  verdict           : {verdict}")
    print(f"  records plotted   : {len(records)} ({len(priors)} priors, {len(discovered)} discovered)")
    print(f"  max problem_overlap / method_overlap : {scores[:, 0].max()} / {scores[:, 2].max()}"
          f"  (threshold {p_thresh})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
