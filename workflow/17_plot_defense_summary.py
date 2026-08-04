"""Step 4 figures: the security-utility frontier, per-action behaviour, ablation effects.

Reads ``results/aegislink_defense_evaluation.json`` and writes:

``figures/step4_security_utility_frontier.{png,pdf}``
    UALER against ATPR for all 18 configurations. The plot that carries the RQ3 claim.
``figures/step4_per_action_profile.{png,pdf}``
    UALER and ATPR per action for a chosen subset, showing where action relativity bites.
``figures/step4_ablation_effects.{png,pdf}``
    Each ablation's delta against the full method, beside its pre-stated prediction.
``figures/step4_verdict_and_calibration.{png,pdf}``
    Verdict distributions and the Brier/ECE table.

Run::

    uv run python workflow/17_plot_defense_summary.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIGS = ROOT / "figures"
RESULTS = ROOT / "results"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})

ACTIONS = ("browse", "contact", "book", "login", "pay")

#: One colour per family, so the eye groups the 18 rows without a legend entry each.
FAMILY_COLOR = {
    "full": "#B03A2E",
    "ablation": "#4C72B0",
    "baseline": "#8C8C8C",
}
SHORT = {
    "aegislink_full": "AegisLink",
    "ablation_no_action_type": "-action type",
    "ablation_no_official_backlinks": "-backlinks",
    "ablation_no_source_clustering": "-clustering",
    "ablation_no_domain_lifecycle": "-lifecycle",
    "ablation_no_contradiction_edges": "-contradictions",
    "ablation_source_count_voting": "voting",
    "ablation_shared_threshold": "shared tau",
    "B01_lexical_url_rules": "B01 URL rules",
    "B02_domain_reputation": "B02 reputation",
    "B03_phishing_classifier": "B03 phishing",
    "B04_llm_as_judge": "B04 judge",
    "B05_source_count_majority": "B05 vote",
    "B06_provenance_reranker": "B06 provenance",
    "B07_graph_anomaly_detector": "B07 graph anomaly",
    "B08_reject_high_risk": "B08 block login/pay",
    "B09_official_only": "B09 official only",
    "B10_ragshield_defense": "B10 RAGShield",
}


def family_of(name: str) -> str:
    if name == "aegislink_full":
        return "full"
    return "ablation" if name.startswith("ablation_") else "baseline"


def pick_regime(rows: dict[str, Any]) -> str:
    return "validation" if "validation" in rows else next(iter(rows))


def save(fig, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"   wrote figures/{stem}.png / .pdf")


# ======================================================================================
def plot_frontier(metrics: dict[str, Any], regime: str) -> None:
    """UALER vs ATPR. Up-and-left is better; the top-left corner is the goal.

    Several configurations land on exactly the same point -- four ablations sit with the full method
    at (0, 1), and ``-backlinks`` sits exactly on top of ``B09 official only`` at (0, 0), which is
    itself a result worth seeing rather than a plotting nuisance. Coincident points are therefore
    grouped and labelled once, instead of stacking five unreadable annotations.
    """
    fig, ax = plt.subplots(figsize=(7.4, 5.0))

    # Group by proximity, not by exact equality: B05 lands at UALER 0.021 while B09 and
    # -backlinks land at 0.000, which is visually the same point but would print three labels on
    # top of each other.
    tol = 0.03
    points: list[tuple[float, float, str]] = []
    for name, rows in metrics.items():
        row = rows.get(regime)
        if not row:
            continue
        u, a = row["primary"]["UALER"], row["primary"]["ATPR"]
        if u is None or a is None:
            continue
        points.append((float(u), float(a), name))

    groups: dict[tuple[float, float], list[str]] = {}
    for u, a, name in sorted(points):
        for (gu, ga), members in groups.items():
            if abs(gu - u) <= tol and abs(ga - a) <= tol:
                members.append(name)
                break
        else:
            groups[(u, a)] = [name]

    for (u, a), names in sorted(groups.items()):
        fams = {family_of(n) for n in names}
        fam = "full" if "full" in fams else ("ablation" if "ablation" in fams else "baseline")
        ax.scatter(
            u,
            a,
            s=130 if fam == "full" else 52,
            c=FAMILY_COLOR[fam],
            marker="*" if fam == "full" else ("s" if fam == "ablation" else "o"),
            edgecolor="white",
            linewidth=0.6,
            zorder=4 if fam == "full" else 3,
        )
        label = "\n".join(SHORT.get(n, n) for n in names)
        # Push labels away from the axes so the corner clusters stay legible.
        dx = 9 if u < 0.5 else -9
        ha = "left" if u < 0.5 else "right"
        dy = -4 if a > 0.5 else 8
        va = "top" if a > 0.5 else "bottom"
        ax.annotate(
            label,
            (u, a),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=6.3,
            ha=ha,
            va=va,
            color="#222222" if fam == "full" else "#555555",
            linespacing=1.35,
        )

    ax.set_xlabel("UALER  (unauthorized action-link exposure rate; lower is better)")
    ax.set_ylabel("ATPR  (authorized third-party recall; higher is better)")
    ax.set_xlim(-0.06, 1.12)
    ax.set_ylim(-0.10, 1.14)
    ax.grid(alpha=0.16, linewidth=0.5)

    # The preregistered non-triviality gates (CONTRACT.md Section 12, baseline_not_trivial).
    ax.axvline(0.10, color="#B03A2E", linestyle=":", linewidth=0.9, zorder=1)
    ax.axhline(0.90, color="#B03A2E", linestyle=":", linewidth=0.9, zorder=1)
    ax.add_patch(
        plt.Rectangle(
            (-0.06, 0.90), 0.16, 0.24, facecolor="#B03A2E", alpha=0.05, zorder=0
        )
    )
    ax.annotate(
        "preregistered non-triviality box\nUALER <= 0.10 and ATPR >= 0.90\n"
        "no baseline reaches it",
        (0.34, 0.44),
        fontsize=6.4,
        color="#B03A2E",
        linespacing=1.45,
    )

    handles = [
        plt.Line2D([], [], marker="*", ls="", ms=11, c=FAMILY_COLOR["full"], label="AegisLink (full)"),
        plt.Line2D([], [], marker="s", ls="", ms=6, c=FAMILY_COLOR["ablation"], label="ablations (7)"),
        plt.Line2D([], [], marker="o", ls="", ms=6, c=FAMILY_COLOR["baseline"], label="baselines (10)"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.02, 0.40),
              frameon=False, fontsize=7)
    ax.set_title(
        f"Security-utility frontier ({regime} split; test never read)", fontsize=9, loc="left"
    )
    save(fig, "step4_security_utility_frontier")


def plot_per_action(metrics: dict[str, Any], regime: str) -> None:
    """Where action relativity bites: the same defense behaves differently per action."""
    show = [
        "aegislink_full",
        "ablation_no_action_type",
        "ablation_no_official_backlinks",
        "B10_ragshield_defense",
        "B09_official_only",
        "B08_reject_high_risk",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    x = np.arange(len(ACTIONS))
    width = 0.8 / len(show)

    for metric, ax in zip(("UALER", "ATPR"), axes):
        for i, name in enumerate(show):
            row = metrics.get(name, {}).get(regime)
            if not row:
                continue
            vals = [
                (row["per_action"].get(a, {}).get(metric) or 0.0) for a in ACTIONS
            ]
            fam = family_of(name)
            ax.bar(
                x + i * width - 0.4 + width / 2,
                vals,
                width=width * 0.92,
                label=SHORT.get(name, name),
                color=FAMILY_COLOR[fam],
                alpha=1.0 if fam == "full" else (0.72 if fam == "ablation" else 0.5),
                edgecolor="white",
                linewidth=0.3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(ACTIONS)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.18, linewidth=0.5)
        ax.set_title(
            f"{metric} per action" + ("  (lower better)" if metric == "UALER" else "  (higher better)"),
            fontsize=8.5,
            loc="left",
        )
    axes[0].set_ylabel("rate")
    axes[1].legend(
        frameon=False, fontsize=6.2, ncol=3, loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
    )
    fig.suptitle(
        "Action-risk relativity: browse -> pay ordering is a property of the method, not of the domain",
        fontsize=9,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    save(fig, "step4_per_action_profile")


def plot_ablations(effects: dict[str, Any]) -> None:
    """Realised effect beside the pre-stated prediction, including where it disagreed."""
    names = [k for k in effects if k.startswith("ablation_")]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    y = np.arange(len(names))

    du = [effects[n]["delta_ualer_vs_full"] or 0.0 for n in names]
    da = [effects[n]["delta_atpr_vs_full"] or 0.0 for n in names]
    ax.barh(y - 0.19, du, height=0.36, color="#B03A2E", label="delta UALER (higher = worse security)")
    ax.barh(y + 0.19, da, height=0.36, color="#4C72B0", label="delta ATPR (lower = worse utility)")

    ax.set_yticks(y)
    ax.set_yticklabels([SHORT.get(n, n) for n in names], fontsize=7)
    ax.axvline(0, color="#333333", linewidth=0.7)
    ax.grid(axis="x", alpha=0.18, linewidth=0.5)
    ax.set_xlabel("change relative to the full method")
    ax.legend(frameon=False, fontsize=6.8, loc="lower right")

    # Mark where the pre-stated prediction did not hold. This is the point of pre-stating it.
    for i, n in enumerate(names):
        holds = effects[n].get("prediction_holds")
        if holds is False:
            ax.annotate(
                "prediction did not hold",
                (max(du[i], da[i]) + 0.03, i),
                fontsize=6.2,
                color="#B03A2E",
                va="center",
            )
    ax.set_title(
        "Ablation effects vs. pre-stated predictions", fontsize=9, loc="left"
    )
    fig.tight_layout()
    save(fig, "step4_ablation_effects")


def plot_verdicts_and_calibration(metrics: dict[str, Any], regime: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))

    # -- verdict distribution for the AegisLink family ------------------------------
    verdict_order = ("VERIFIED", "PLAUSIBLE", "UNVERIFIED", "CONTRADICTED")
    palette = {
        "VERIFIED": "#2E7D32",
        "PLAUSIBLE": "#F9A825",
        "UNVERIFIED": "#8C8C8C",
        "CONTRADICTED": "#B03A2E",
    }
    fam_names = [n for n in metrics if family_of(n) in ("full", "ablation")]
    ax = axes[0]
    bottom = np.zeros(len(fam_names))
    for v in verdict_order:
        vals = []
        for n in fam_names:
            row = metrics[n].get(regime, {})
            dist = row.get("verdict_distribution", {})
            total = sum(dist.values()) or 1
            vals.append(dist.get(v, 0) / total)
        ax.barh(np.arange(len(fam_names)), vals, left=bottom, height=0.66,
                color=palette[v], label=v, edgecolor="white", linewidth=0.3)
        bottom += np.array(vals)
    ax.set_yticks(np.arange(len(fam_names)))
    ax.set_yticklabels([SHORT.get(n, n) for n in fam_names], fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of triples")
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower right")
    ax.set_title("Verdict distribution (AegisLink family)", fontsize=8.5, loc="left")

    # -- calibration ----------------------------------------------------------------
    ax = axes[1]
    names = [n for n in metrics if metrics[n].get(regime)]
    briers = [metrics[n][regime]["calibration"]["brier"] or 0.0 for n in names]
    order = np.argsort(briers)
    names = [names[i] for i in order]
    briers = [briers[i] for i in order]
    eces = [metrics[n][regime]["calibration"]["ece"] or 0.0 for n in names]
    y = np.arange(len(names))
    ax.barh(y - 0.19, briers, height=0.36, color="#4C72B0", label="Brier")
    ax.barh(y + 0.19, eces, height=0.36, color="#F9A825", label="ECE")
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT.get(n, n) for n in names], fontsize=6.4)
    ax.grid(axis="x", alpha=0.18, linewidth=0.5)
    ax.legend(frameon=False, fontsize=6.8, loc="lower right")
    ax.set_xlabel("error (lower is better)")
    ax.set_title("Authorization-probability calibration", fontsize=8.5, loc="left")

    fig.tight_layout()
    save(fig, "step4_verdict_and_calibration")


def main() -> int:
    path = RESULTS / "aegislink_defense_evaluation.json"
    if not path.is_file():
        print(f"missing {path}; run workflow/16_evaluate_aegislink_and_baselines.py first")
        return 1
    doc = json.loads(path.read_text(encoding="utf-8"))
    metrics = doc["metrics"]
    regime = pick_regime(metrics["aegislink_full"])
    print(f"Plotting Step 4 summary on the {regime} split")

    plot_frontier(metrics, regime)
    plot_per_action(metrics, regime)
    plot_ablations(doc["ablation_effects"])
    plot_verdicts_and_calibration(metrics, regime)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
