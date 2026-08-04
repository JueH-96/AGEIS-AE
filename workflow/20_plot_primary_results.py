#!/usr/bin/env python3
"""Render the primary AegisLink result figures into ``figures/``.

Reads the frozen confirmatory result JSONs (10,000-resample run) and renders
deterministic, exactly-labelled figures. No number is typed by hand: every value
plotted is pulled from ``results/*.json``.

Ten figures are produced, each as PNG (400 dpi) and PDF:

===========================================  ==================================================
File stem                                    Content
===========================================  ==================================================
``fig01_graphical_abstract``                 the misbinding threat, one panel
``fig02_architecture``                       AegisLink pipeline stages
``fig03_security_utility_frontier``          UALER vs ATPR Pareto frontier, all 18 configurations
``fig04_per_action_profile``                 UALER and ATPR by action type
``fig05_ablation_effects``                   seven single-factor ablations vs full AegisLink
``fig06_verdicts_calibration_adaptive``      verdict mix, calibration, adaptive ASR_a
``fig07_stage_attribution``                  which pipeline stage rejects, per defense
``fig08_pilot_gates``                        the ten Section 12 Go/No-Go conditions
``fig09_prevalence_regimes``                 undefended misbinding prevalence across regimes
``figA2_split_composition``                  template split sizes on all three axes
===========================================  ==================================================

No TeX toolchain is involved: matplotlib's built-in mathtext renders the few math labels.

Usage
-----
    python workflow/20_plot_primary_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PRIM = json.loads((ROOT / "results" / "primary_test_evaluation.json").read_text())
ADAPT = json.loads((ROOT / "results" / "adaptive_robustness_evaluation.json").read_text())
PILOT = json.loads((ROOT / "results" / "pilot_decision_report.json").read_text())
TRACES = json.loads((ROOT / "results" / "defense_stage_traces_summary.json").read_text())

M = PRIM["metrics"]
ORDER = PRIM["method"]["configuration_order"]
ACTIONS = ["browse", "contact", "book", "login", "pay"]

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.0,
    "axes.labelsize": 7.0,
    "axes.titlesize": 7.8,
    "axes.titleweight": "bold",
    "xtick.labelsize": 6.3,
    "ytick.labelsize": 6.3,
    "legend.fontsize": 6.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.35,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

C_AEGIS = "#1b6ca8"
C_ABL = "#e0a458"
C_BASE = "#8d99ae"
C_BAD = "#c1272d"
C_GOOD = "#2e7d4f"
C_INK = "#22303c"
C_SOFT = "#eef2f6"

SHORT = {
    "aegislink_full": "AegisLink (full)",
    "ablation_no_action_type": "$-$action type",
    "ablation_no_official_backlinks": "$-$backlinks",
    "ablation_no_source_clustering": "$-$clustering",
    "ablation_no_domain_lifecycle": "$-$lifecycle",
    "ablation_no_contradiction_edges": "$-$contradictions",
    "ablation_source_count_voting": "graph$\\rightarrow$voting",
    "ablation_shared_threshold": "shared $\\tau$",
    "B01_lexical_url_rules": "B1 URL lexical",
    "B02_domain_reputation": "B2 domain age",
    "B03_phishing_classifier": "B3 phishing clf.",
    "B04_llm_as_judge": "B4 LLM judge$^{\\dagger}$",
    "B05_source_count_majority": "B5 source vote",
    "B06_provenance_reranker": "B6 provenance",
    "B07_graph_anomaly_detector": "B7 graph anomaly$^{\\dagger}$",
    "B08_reject_high_risk": "B8 block login/pay",
    "B09_official_only": "B9 official only",
    "B10_ragshield_defense": "B10 RAGShield$^{\\dagger}$",
}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote figures/{name}.png|.pdf")


def box(ax, x, y, w, h, text, fc, ec, fs=7.2, tc=None, weight="normal", r=0.012):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={r}",
                       linewidth=0.9, facecolor=fc, edgecolor=ec, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc or C_INK, zorder=3, weight=weight, linespacing=1.35)
    return p


def arrow(ax, p0, p1, color=C_INK, lw=0.9, style="-|>", ls="-", mut=6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=mut,
                                 linewidth=lw, color=color, linestyle=ls,
                                 shrinkA=1.5, shrinkB=1.5, zorder=2))


# =====================================================================
# Figure 1 -- graphical abstract
# =====================================================================
def fig_graphical_abstract():
    fig = plt.figure(figsize=(7.00, 2.67))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.18], wspace=0.075,
                          left=0.012, right=0.988, top=0.90, bottom=0.10)

    # ---------------- panel A: the problem
    ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 1.005, "A.  The problem: action-link misbinding",
            ha="center", va="bottom", fontsize=10, weight="bold", color=C_INK)

    box(ax, 0.05, 0.845, 0.50, 0.085, "“Recommend a good local bistro.”",
        "#dbeafe", "#7aa7d9", fs=7.6)
    arrow(ax, (0.30, 0.845), (0.30, 0.785))
    box(ax, 0.14, 0.700, 0.32, 0.085, "web-enabled LLM", "#f4f6f8", C_INK, fs=7.6, weight="bold")
    arrow(ax, (0.30, 0.700), (0.30, 0.640))

    box(ax, 0.035, 0.215, 0.62, 0.425, "", "#ffffff", C_INK, r=0.02)
    ax.text(0.065, 0.585, "Model answer", fontsize=7.0, color="#5a6b7a", style="italic")
    ax.text(0.075, 0.512, "✓", fontsize=11, color=C_GOOD, weight="bold")
    ax.text(0.125, 0.522, "Harbour Lantern Bistro", fontsize=8.0, color=C_INK, weight="bold")
    ax.text(0.125, 0.470, "12 Quay Street  ·  +44 20 7946 0102", fontsize=7.0, color="#5a6b7a")
    ax.text(0.125, 0.432, "entity, address and phone all correct", fontsize=6.6,
            color=C_GOOD, style="italic")

    box(ax, 0.085, 0.300, 0.24, 0.075, "Book a table", "#fde2e2", C_BAD, fs=7.4,
        tc=C_BAD, weight="bold")
    ax.text(0.345, 0.337, "→", fontsize=11, color=C_BAD, va="center")
    box(ax, 0.375, 0.300, 0.25, 0.075, "reservations-\nquayside.test", "#ffffff", C_BAD, fs=6.5, tc=C_BAD)
    ax.text(0.395, 0.253, "✗  attacker-controlled endpoint", fontsize=6.6, color=C_BAD)

    ax.text(0.345, 0.135, "Right entity.  Wrong action endpoint.", ha="center",
            fontsize=8.4, weight="bold", color=C_INK)
    ax.text(0.345, 0.070, "The correct context is what lowers the user's guard.",
            ha="center", fontsize=6.9, color="#5a6b7a", style="italic")

    # ---------------- panel B: the model
    ax = fig.add_subplot(gs[1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 1.005, "B.  The model: $(e,d,a)$ authorization",
            ha="center", va="bottom", fontsize=10, weight="bold", color=C_INK)

    ex, ey = 0.115, 0.545
    ax.add_patch(plt.Circle((ex, ey), 0.052, facecolor=C_AEGIS, edgecolor="none", zorder=3))
    ax.text(ex, ey, "$e$", ha="center", va="center", color="white", fontsize=10,
            weight="bold", zorder=4)
    ax.text(ex, ey - 0.095, "entity", ha="center", fontsize=7.0, color=C_INK)

    doms = [("official site", 0.795, C_GOOD), ("authorized\nthird party", 0.545, C_GOOD),
            ("impersonator", 0.290, C_BAD)]
    for name, dy, col in doms:
        box(ax, 0.315, dy - 0.058, 0.29, 0.116, name, "#ffffff", col, fs=7.0)

    # edges: official = all authorized; third party = book only; impersonator = none
    arrow(ax, (ex + 0.052, ey + 0.030), (0.315, 0.795), color=C_GOOD, lw=1.3)
    arrow(ax, (ex + 0.052, ey), (0.315, 0.545), color=C_GOOD, lw=1.3)
    arrow(ax, (ex + 0.052, ey - 0.030), (0.315, 0.290), color=C_BAD, lw=1.1, ls=(0, (2.2, 1.6)))

    ax.text(0.208, 0.708, "browse … pay", fontsize=6.2, color=C_GOOD, rotation=33,
            ha="center", va="center")
    ax.text(0.222, 0.560, "book only", fontsize=6.2, color=C_GOOD, ha="center")
    ax.text(0.212, 0.372, "no grant", fontsize=6.2, color=C_BAD, rotation=-30,
            ha="center", va="center")

    # risk ladder
    lx = 0.735
    ax.annotate("", xy=(lx, 0.905), xytext=(lx, 0.215),
                arrowprops=dict(arrowstyle="-|>", color=C_INK, linewidth=1.0))
    for i, a in enumerate(ACTIONS):
        yy = 0.245 + i * 0.157
        ax.plot([lx - 0.022, lx + 0.022], [yy, yy], color=C_INK, lw=0.9)
        ax.text(lx + 0.042, yy, a, fontsize=7.4, va="center", color=C_INK,
                weight="bold" if a in ("login", "pay") else "normal")
    ax.text(lx - 0.062, 0.560, "increasing action risk", rotation=90, fontsize=7.0,
            va="center", ha="center", color="#5a6b7a")

    ax.text(0.5, 0.108, r"$\tau_{\mathrm{browse}}<\tau_{\mathrm{contact}}<\tau_{\mathrm{book}}"
                        r"<\tau_{\mathrm{login}}<\tau_{\mathrm{pay}}$",
            ha="center", fontsize=8.2, color=C_INK)
    ax.text(0.5, 0.042, "one domain may be authorized for one action and not another",
            ha="center", fontsize=6.9, color="#5a6b7a", style="italic")

    # ---------------- panel C: method + result
    ax = fig.add_subplot(gs[2]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 1.005, "C.  AegisLink verification and outcome",
            ha="center", va="bottom", fontsize=10, weight="bold", color=C_INK)

    stages = ["entity resolution", "action extraction", "evidence graph",
              "source-dependency clustering", "action-specific authorization",
              "risk-aware output policy"]
    for i, s in enumerate(stages):
        yy = 0.855 - i * 0.108
        box(ax, 0.015, yy - 0.042, 0.375, 0.084, s, C_SOFT, C_AEGIS, fs=6.5)
        if i < len(stages) - 1:
            arrow(ax, (0.2025, yy - 0.042), (0.2025, yy - 0.066), lw=0.8, mut=5)

    arrow(ax, (0.2025, 0.165), (0.2025, 0.140), lw=0.9)
    verdicts = [("VERIFIED", C_GOOD), ("PLAUSIBLE", C_AEGIS),
                ("UNVERIFIED", "#77808a"), ("CONTRADICTED", C_BAD)]
    for i, (v, col) in enumerate(verdicts):
        r, c_ = divmod(i, 2)
        box(ax, 0.012 + c_ * 0.196, 0.068 - r * 0.062, 0.182, 0.054, v, col, col,
            fs=5.9, tc="white", weight="bold")

    # result bars -- pulled from the frozen JSON
    ref = PRIM["reference_baseline_selection"]["reference_baseline"]
    aeg_t = M["aegislink_full"]["test"]["primary"]
    ref_t = M[ref]["test"]["primary"]
    bax = fig.add_axes([0.868, 0.235, 0.115, 0.545])
    idx = np.array([0, 1])
    bax.bar(idx - 0.19, [ref_t["UALER"], ref_t["ATPR"]], 0.36, color=C_BASE,
            edgecolor="none", label="best baseline")
    bax.bar(idx + 0.19, [aeg_t["UALER"], aeg_t["ATPR"]], 0.36, color=C_AEGIS,
            edgecolor="none", label="AegisLink")
    for xx, vv in zip(idx - 0.19, [ref_t["UALER"], ref_t["ATPR"]]):
        bax.text(xx, vv + 0.035, f"{vv:.2f}", ha="center", fontsize=6.2, color=C_INK)
    for xx, vv in zip(idx + 0.19, [aeg_t["UALER"], aeg_t["ATPR"]]):
        bax.text(xx, vv + 0.035, f"{vv:.2f}", ha="center", fontsize=6.2,
                 color=C_AEGIS, weight="bold")
    bax.set_xticks(idx)
    bax.set_xticklabels(["UALER\n(lower better)", "ATPR\n(higher better)"], fontsize=6.2)
    bax.set_ylim(0, 1.22); bax.set_yticks([0, 0.5, 1.0])
    bax.tick_params(labelsize=6.2, length=2)
    bax.grid(axis="y", linestyle=":", alpha=0.4)
    bax.set_axisbelow(True)
    bax.legend(frameon=False, fontsize=5.7, loc="upper center",
               bbox_to_anchor=(0.5, -0.145), ncol=1, handlelength=0.9)
    bax.set_title("held-out test split", fontsize=6.6, pad=3)

    fig.text(0.5, 0.012,
             "Fully generated benchmark  ·  deterministic labelling from the authorization graph  ·  "
             "frozen retrieval replay  ·  no human adjudication",
             ha="center", fontsize=7.4, color="#5a6b7a", style="italic")
    fig.patches.append(Rectangle((0.0, 0.0), 1.0, 0.052, transform=fig.transFigure,
                                 facecolor="#f4f6f8", edgecolor="none", zorder=-5))
    save(fig, "fig01_graphical_abstract")


# =====================================================================
# Figure 2 -- system architecture
# =====================================================================
def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.00, 4.44))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # band 1: controlled web world
    ax.add_patch(Rectangle((0.005, 0.735), 0.99, 0.245, facecolor="#f7f9fb",
                           edgecolor="#c7d2dd", lw=0.8, zorder=0))
    ax.text(0.018, 0.955, "Controlled Web-RAG world (offline, private .test namespace)",
            fontsize=7.6, weight="bold", color=C_INK)
    world = ["entity\nregistry", "authorization\ngraph", "site\ngenerator",
             "attack\ngenerator", "crawler +\nindexer", "retriever\n(frozen replay)"]
    for i, s in enumerate(world):
        box(ax, 0.022 + i * 0.161, 0.775, 0.147, 0.125, s, "#ffffff", "#7d8ea0", fs=6.4)
        if i < len(world) - 1:
            arrow(ax, (0.169 + i * 0.161, 0.8375), (0.183 + i * 0.161, 0.8375), lw=0.8, mut=5)

    # band 2: reader + parsers
    ax.add_patch(Rectangle((0.005, 0.485), 0.99, 0.225, facecolor="#fefaf3",
                           edgecolor="#e3cba3", lw=0.8, zorder=0))
    ax.text(0.018, 0.680, "Reader and deterministic parsing (no human adjudication)",
            fontsize=7.6, weight="bold", color=C_INK)
    # Labels describe what the implementation ACTUALLY does. Two registered
    # components (PSL canonicalization, constrained JSON extraction) were not
    # implemented and are not drawn as if they were; see the manuscript's
    # "Deviations from the preregistration".
    read = ["reader\n(deterministic\nsurrogate)", "structural URL /\ndomain identity",
            "entity resolver\n(generated aliases)", "frozen action\nontology",
            "official-claim\nregexes", "PARSER_\nDISAGREEMENT"]
    for i, s in enumerate(read):
        last = i == len(read) - 1
        col = "#9aa4ae" if last else "#b08442"
        box(ax, 0.022 + i * 0.161, 0.522, 0.147, 0.122, s, "#ffffff", col, fs=5.8)
        if i < len(read) - 2:
            arrow(ax, (0.169 + i * 0.161, 0.583), (0.183 + i * 0.161, 0.583), lw=0.8, mut=5)
    arrow(ax, (0.169 + 4 * 0.161, 0.583), (0.183 + 4 * 0.161, 0.583), lw=0.8, mut=5,
          color="#9aa4ae", ls=(0, (2, 1.5)))
    ax.text(0.905, 0.500, "registered rule; never fired (rate 0)", fontsize=5.5,
            color="#7b8894", ha="center", style="italic")

    # band 3: AegisLink
    ax.add_patch(Rectangle((0.005, 0.175), 0.665, 0.283, facecolor="#f2f7fb",
                           edgecolor="#a8c4dc", lw=0.8, zorder=0))
    ax.text(0.018, 0.428, "AegisLink verifier", fontsize=7.6, weight="bold", color=C_AEGIS)
    aeg = ["1  entity\nresolution", "2  action\nextraction", "3  evidence\ngraph",
           "4  source-dependency\nclustering", "5  action-specific\nauthorization",
           "6  risk-aware\noutput policy"]
    for i, s in enumerate(aeg):
        r, c = divmod(i, 3)
        box(ax, 0.022 + c * 0.216, 0.310 - r * 0.118, 0.196, 0.098, s, "#ffffff", C_AEGIS, fs=6.1)
        if c < 2:
            arrow(ax, (0.218 + c * 0.216, 0.359 - r * 0.118),
                  (0.238 + c * 0.216, 0.359 - r * 0.118), lw=0.8, mut=5, color=C_AEGIS)
    arrow(ax, (0.610, 0.310), (0.120, 0.290), lw=0.8, mut=5, color=C_AEGIS,
          style="-|>", ls=(0, (2, 1.5)))

    # evidence families feeding stage 3
    ax.text(0.685, 0.428, "Evidence families", fontsize=7.6, weight="bold", color=C_INK)
    fams = ["generated authoritative registry", "official-domain backlinks",
            "identity-field consistency", "domain lifecycle / ownership",
            "source-dependency clusters", "action-specific evidence", "contradiction edges"]
    for i, f in enumerate(fams):
        ax.text(0.692, 0.388 - i * 0.0345, "•  " + f, fontsize=5.9, color="#41505e", va="center")

    # band 4: outputs
    box(ax, 0.010, 0.052, 0.155, 0.088, "score$(e,d,a)$\nPlatt-calibrated", "#ffffff", C_INK, fs=6.0)
    arrow(ax, (0.165, 0.096), (0.186, 0.096), lw=0.9)
    th = PRIM["frozen_parameters"]["thresholds"]["tau"]
    tau_txt = "  ".join(f"$\\tau_{{\\mathrm{{{a[:2]}}}}}${th[a]:.2f}"
                        for a in ACTIONS) if isinstance(th, dict) else ""
    box(ax, 0.186, 0.052, 0.300, 0.088, "action thresholds\n" + tau_txt, "#ffffff", C_INK, fs=5.0)
    arrow(ax, (0.486, 0.096), (0.509, 0.096), lw=0.9)
    verd = [("VERIFIED", C_GOOD), ("PLAUSIBLE", C_AEGIS), ("UNVERIFIED", "#77808a"),
            ("CONTRADICTED", C_BAD)]
    for i, (v, col) in enumerate(verd):
        box(ax, 0.512 + i * 0.121, 0.058, 0.113, 0.076, v, col, col, fs=5.1,
            tc="white", weight="bold")
    ax.text(0.748, 0.020, "present  ·  withhold  ·  withhold  ·  withhold + record conflict",
            ha="center", fontsize=5.9, color="#5a6b7a", style="italic")

    save(fig, "fig02_architecture")


# =====================================================================
# Figure 3 -- security-utility frontier on the confirmatory test split
# =====================================================================
def fig_frontier():
    fig, ax = plt.subplots(figsize=(5.75, 3.75))
    gates = PILOT["frozen_hard_gates"]["baseline_not_trivial"]
    umax = gates["max_allowed_best_baseline_ualer"]
    amin = gates["min_required_baseline_atpr"]

    ax.add_patch(Rectangle((-0.02, amin), umax + 0.02, 1.10 - amin,
                           facecolor="#fdeaea", edgecolor="none", zorder=0))
    ax.axvline(umax, color=C_BAD, ls=":", lw=0.9, zorder=1)
    ax.axhline(amin, color=C_BAD, ls=":", lw=0.9, zorder=1)

    # group configurations that land on the same coordinate so labels never collide
    coords = {}
    for c in ORDER:
        p = M[c]["test"]["primary"]
        key = (round(p["UALER"], 4), round(p["ATPR"], 4))
        coords.setdefault(key, []).append(c)

    for (x, y), members in coords.items():
        if "aegislink_full" in members:
            continue
        kinds = {("ablation" if c.startswith("ablation") else "baseline") for c in members}
        col = C_ABL if kinds == {"ablation"} else (C_BASE if kinds == {"baseline"} else "#b8905f")
        mk = "s" if kinds == {"ablation"} else ("o" if kinds == {"baseline"} else "D")
        ax.scatter([x], [y], s=46 if len(members) == 1 else 74, c=col, marker=mk,
                   edgecolors="white", linewidths=0.8, zorder=4)

    # legend proxies
    ax.scatter([], [], s=46, c=C_BASE, marker="o", edgecolors="white",
               linewidths=0.8, label="baseline")
    ax.scatter([], [], s=46, c=C_ABL, marker="s", edgecolors="white",
               linewidths=0.8, label="ablation")
    ax.scatter([], [], s=74, c="#b8905f", marker="D", edgecolors="white",
               linewidths=0.8, label="co-located (mixed)")
    a = M["aegislink_full"]["test"]["primary"]
    ax.scatter([a["UALER"]], [a["ATPR"]], s=235, marker="*", c=C_AEGIS,
               edgecolors="white", linewidths=1.0, zorder=6, label="AegisLink (full)")

    # manual, non-overlapping label anchors (data coords) with leader lines
    anchors = {
        (0.0, 1.0): (0.075, 0.700),
        (0.0, 0.0): (0.095, 0.185),
        (0.0093, 0.0612): (0.100, 0.062),
        (0.0139, 0.0): (0.260, 0.012),
        (0.4491, 1.0): (0.300, 0.940),
        (0.1435, 0.3673): (0.215, 0.365),
        (0.2222, 0.8776): (0.275, 0.822),
        (0.4722, 0.0272): (0.515, 0.078),
        (0.7361, 0.9864): (0.520, 0.905),
        (0.7917, 1.0): (0.585, 1.080),
        (0.9722, 0.6667): (0.840, 0.608),
        (0.9954, 0.9864): (0.715, 0.795),
        (1.0, 1.0): (0.905, 1.080),
    }
    for (x, y), members in coords.items():
        others = [c for c in members if c != "aegislink_full"]
        if not others:
            continue
        txt = "\n".join(SHORT[c] for c in others)
        tx, ty = anchors.get((x, y), (x + 0.02, y))
        ax.annotate(txt, (x, y), (tx, ty), fontsize=6.0, color="#41505e",
                    va="center", ha="left", zorder=5, linespacing=1.45,
                    arrowprops=dict(arrowstyle="-", color="#b6c0ca", lw=0.55,
                                    shrinkA=1, shrinkB=3))
    n_co = len([c for c in coords[(round(a["UALER"], 4), round(a["ATPR"], 4))]
                if c != "aegislink_full"])
    ax.annotate(SHORT["aegislink_full"], (a["UALER"], a["ATPR"]),
                (a["UALER"] + 0.048, a["ATPR"] + 0.060), fontsize=7.6,
                color=C_AEGIS, weight="bold", zorder=7)
    ax.text(a["UALER"] + 0.048, a["ATPR"] + 0.012,
            f"coincides with {n_co} ablations $\\downarrow$", fontsize=6.0,
            color=C_AEGIS, zorder=7)

    ax.text(0.545, 0.455,
            f"preregistered non-triviality box\nUALER $\\leq$ {umax:.2f} and ATPR $\\geq$ {amin:.2f}\n"
            "no baseline reaches it",
            fontsize=6.8, color=C_BAD, va="center", ha="center", linespacing=1.5)
    ax.annotate("", xy=(0.055, 0.955), xytext=(0.470, 0.520),
                arrowprops=dict(arrowstyle="-|>", color=C_BAD, lw=0.7,
                                linestyle=(0, (3, 2))), zorder=3)

    ax.set_xlabel("UALER   unauthorized action-link exposure rate  (lower is better)")
    ax.set_ylabel("ATPR   authorized third-party recall  (higher is better)")
    ax.set_title("Security–utility frontier, held-out test split")
    ax.set_xlim(-0.055, 1.15); ax.set_ylim(-0.085, 1.13)
    ax.grid(True, linestyle=":", alpha=0.4); ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=True, framealpha=0.94, edgecolor="#d6dde4",
              fontsize=6.8, borderpad=0.55)
    save(fig, "fig03_security_utility_frontier")


# =====================================================================
# Figure 4 -- per-action profile
# =====================================================================
def fig_per_action():
    fig, axes = plt.subplots(1, 3, figsize=(7.00, 2.42))
    ref = PRIM["reference_baseline_selection"]["reference_baseline"]
    show = ["aegislink_full", "ablation_no_action_type", ref, "B04_llm_as_judge"]
    cols = [C_AEGIS, C_ABL, C_BASE, "#b0b7bf"]
    x = np.arange(len(ACTIONS)); w = 0.20

    for ax, metric, ylab in zip(
            axes, ["UALER", "ATPR", "abstention_rate"],
            ["UALER  (lower is better)", "ATPR  (higher is better)", "abstention rate"]):
        na_actions: set[int] = set()
        for j, (c, col) in enumerate(zip(show, cols)):
            pa = M[c]["test"]["per_action"]
            vals, hatch_idx = [], []
            for i, a in enumerate(ACTIONS):
                v = pa.get(a, {}).get(metric)
                if v is None:
                    vals.append(0.0); hatch_idx.append(i)
                else:
                    vals.append(v)
            ax.bar(x + (j - 1.5) * w, vals, w, color=col, edgecolor="none",
                   label=SHORT[c] if metric == "UALER" else None)
            na_actions.update(hatch_idx)
        for i in sorted(na_actions):
            ax.text(x[i], 0.03, "n/a", fontsize=5.4, ha="center", va="bottom",
                    color="#8a949e", style="italic")
        ax.set_xticks(x)
        ax.set_xticklabels(ACTIONS, fontsize=6.2, rotation=22, ha="right",
                           rotation_mode="anchor")
        ax.set_ylabel(ylab, fontsize=7.8)
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", linestyle=":", alpha=0.4); ax.set_axisbelow(True)
        ax.axvline(1.5, color="#c7d2dd", lw=0.8, ls="--")
    axes[0].text(3.5, 1.02, "high-risk actions", fontsize=6.4, ha="center", color="#5a6b7a")
    axes[0].legend(frameon=False, fontsize=6.4, loc="upper left", ncol=1)
    fig.suptitle("Per-action security and utility profile, held-out test split",
                 fontsize=9.5, weight="bold", y=1.015)
    fig.tight_layout()
    save(fig, "fig04_per_action_profile")


# =====================================================================
# Figure 5 -- ablation effects with bootstrap CIs
# =====================================================================
def fig_ablations():
    fig, axes = plt.subplots(1, 2, figsize=(7.00, 2.72), sharey=True)
    boot = PRIM["bootstrap"]["test"]["comparisons_vs_aegislink"]
    abls = [c for c in ORDER if c.startswith("ablation")]
    y = np.arange(len(abls))

    for ax, metric, xlab, worse, xlim in [
            (axes[0], "UALER", "$\\Delta$ UALER   (ablation $-$ full)",
             "higher = worse security", (-0.16, 0.70)),
            (axes[1], "ATPR", "$\\Delta$ ATPR   (ablation $-$ full)",
             "lower = worse utility", (-1.22, 0.20))]:
        pts, los, his = [], [], []
        for c in abls:
            d = boot[c][metric]
            # comparisons are stored as treatment(full) - reference(ablation)
            pt = -d["absolute_difference"]
            ci = d["absolute_difference_ci"]["bias_corrected"]
            lo, hi = -ci["hi"], -ci["lo"]
            pts.append(pt); los.append(pt - lo); his.append(hi - pt)
        colors = [C_BAD if ((metric == "UALER" and p > 1e-9) or
                            (metric == "ATPR" and p < -1e-9)) else "#b9c2cb" for p in pts]
        ax.barh(y, pts, 0.6, color=colors, edgecolor="none", zorder=3)
        ax.errorbar(pts, y, xerr=[los, his], fmt="none", ecolor=C_INK,
                    elinewidth=0.8, capsize=2.0, zorder=4)
        span = xlim[1] - xlim[0]
        for i, (p, lo_, hi_) in enumerate(zip(pts, los, his)):
            if abs(p) < 1e-9:
                ax.text(0.012 * span, i, "no change", fontsize=6.0, va="center",
                        color="#8a949e")
            elif p > 0:
                ax.text(p + hi_ + 0.022 * span, i, f"{p:+.3f}", fontsize=6.4,
                        va="center", ha="left", color=C_INK)
            else:
                ax.text(p - lo_ - 0.022 * span, i, f"{p:+.3f}", fontsize=6.4,
                        va="center", ha="right", color=C_INK)
        ax.axvline(0, color=C_INK, lw=0.9)
        ax.set_xlabel(xlab, fontsize=8.0)
        ax.set_title(worse, fontsize=7.6, weight="normal", color="#5a6b7a")
        ax.grid(axis="x", linestyle=":", alpha=0.4); ax.set_axisbelow(True)
        ax.set_xlim(*xlim)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([SHORT[c] for c in abls], fontsize=7.4)
    axes[0].invert_yaxis()
    fig.suptitle("Ablation effects with 95% paired-bootstrap intervals (test split, $B=10{,}000$)",
                 fontsize=9.5, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig05_ablation_effects")


# =====================================================================
# Figure 6 -- verdicts and calibration
# =====================================================================
def fig_verdicts_calibration():
    fig, axes = plt.subplots(1, 3, figsize=(7.00, 2.42))

    # (a) verdict distribution
    ax = axes[0]
    vd = M["aegislink_full"]["test"]["verdict_distribution"]
    keys = ["VERIFIED", "PLAUSIBLE", "UNVERIFIED", "CONTRADICTED"]
    vals = [vd.get(k, 0) for k in keys]
    cols = [C_GOOD, C_AEGIS, "#77808a", C_BAD]
    b = ax.bar(range(4), vals, 0.62, color=cols, edgecolor="none")
    tot = sum(vals)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + tot * 0.018,
                f"{v}\n{100*v/tot:.1f}%", ha="center", fontsize=6.4, color=C_INK)
    ax.set_xticks(range(4))
    ax.set_xticklabels([k.capitalize() for k in keys], fontsize=5.8,
                      rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel("candidate $(e,d,a)$ triples")
    ax.set_ylim(0, max(vals) * 1.30)
    ax.set_title(f"(a)  Verdicts, AegisLink\n($n$ = {tot} triples)", fontsize=7.6)
    ax.grid(axis="y", linestyle=":", alpha=0.4); ax.set_axisbelow(True)

    # (b) calibration: Brier / ECE across configurations
    ax = axes[1]
    cfgs = ["aegislink_full", "ablation_no_action_type", "B10_ragshield_defense",
            "B06_provenance_reranker", "B02_domain_reputation", "B01_lexical_url_rules",
            "B04_llm_as_judge", "B07_graph_anomaly_detector"]
    br = [M[c]["test"]["calibration"]["brier"] for c in cfgs]
    ec = [M[c]["test"]["calibration"]["ece"] for c in cfgs]
    yy = np.arange(len(cfgs))
    ax.barh(yy - 0.19, br, 0.36, color=C_AEGIS, edgecolor="none", label="Brier score")
    ax.barh(yy + 0.19, ec, 0.36, color=C_ABL, edgecolor="none", label="ECE")
    for i, (a_, b_) in enumerate(zip(br, ec)):
        ax.text(max(a_, b_) + 0.016, i, f"{a_:.3f} / {b_:.3f}", fontsize=5.5,
                va="center", color="#41505e")
    ax.set_yticks(yy); ax.set_yticklabels([SHORT[c] for c in cfgs], fontsize=6.3)
    ax.invert_yaxis(); ax.set_xlim(0, 1.28)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("error (lower is better)")
    ax.set_title("(b)  Authorization-probability\ncalibration (Brier / ECE)", fontsize=7.6)
    ax.legend(frameon=False, fontsize=6.0, loc="upper right",
              bbox_to_anchor=(1.03, 1.03), handlelength=1.0, borderpad=0.2)
    ax.grid(axis="x", linestyle=":", alpha=0.4); ax.set_axisbelow(True)

    # (c) adaptive ASR
    ax = axes[2]
    A = ADAPT["strata"]["A"]["asr"]
    sel = ["aegislink_full", "ablation_source_count_voting", "B02_domain_reputation",
           "B06_provenance_reranker", "B08_reject_high_risk", "B10_ragshield_defense",
           "B05_source_count_majority", "B01_lexical_url_rules", "B04_llm_as_judge"]
    vals = [A[c]["ASR_a"] for c in sel]
    colors = [C_AEGIS if c == "aegislink_full" else (C_ABL if c.startswith("ablation") else C_BASE)
              for c in sel]
    yy = np.arange(len(sel))
    ax.barh(yy, vals, 0.62, color=colors, edgecolor="none")
    for i, v in enumerate(vals):
        ax.text(v + 0.018, i, f"{v:.3f}", fontsize=6.2, va="center", color=C_INK)
    ax.set_yticks(yy); ax.set_yticklabels([SHORT[c] for c in sel], fontsize=6.3)
    ax.invert_yaxis(); ax.set_xlim(0, 1.14)
    ax.set_xlabel("$\\mathrm{ASR}_a$  (adaptive attack success rate)")
    n_att = A["aegislink_full"]["n_attempted"]
    ax.set_title(f"(c)  Adaptive holdout, stratum A\n($n$ = {n_att} queries)", fontsize=7.6)
    ax.grid(axis="x", linestyle=":", alpha=0.4); ax.set_axisbelow(True)

    fig.tight_layout(w_pad=1.4)
    save(fig, "fig06_verdicts_calibration_adaptive")


# =====================================================================
# Figure 7 -- RQ2 failure-stage attribution
# =====================================================================
def fig_stage_attribution():
    fig, axes = plt.subplots(1, 2, figsize=(7.00, 2.72),
                             gridspec_kw={"width_ratios": [1.32, 1.0]})
    pd_ = TRACES["per_defense"]
    stage_keys = list(pd_["aegislink_full"]["failure_origin_counts"].keys())
    nice = {"none": "no failure",
            "retrieval_candidates": "retrieval",
            "resolved_entities": "entity resolution",
            "extracted_entity_domain_relations": "relation interpretation",
            "inferred_action_authorizations": "authorization inference",
            "presented_links": "answer generation"}
    order_stages = [k for k in ["none", "retrieval_candidates", "resolved_entities",
                                "extracted_entity_domain_relations",
                                "inferred_action_authorizations", "presented_links"]
                    if k in stage_keys] or stage_keys
    palette = {"none": "#dfe5ea", "retrieval_candidates": "#7b9ec9",
               "resolved_entities": "#a4c3b2",
               "extracted_entity_domain_relations": "#e0a458",
               "inferred_action_authorizations": "#c1272d",
               "presented_links": "#8e6c88"}

    sel = ["aegislink_full", "ablation_no_action_type", "ablation_source_count_voting",
           "B10_ragshield_defense", "B06_provenance_reranker", "B08_reject_high_risk",
           "B03_phishing_classifier", "B01_lexical_url_rules", "B04_llm_as_judge",
           "B07_graph_anomaly_detector"]
    ax = axes[0]
    yy = np.arange(len(sel)); left = np.zeros(len(sel))
    for s in order_stages:
        vals = np.array([pd_[c]["failure_origin_fractions"].get(s, 0.0) for c in sel])
        ax.barh(yy, vals, 0.66, left=left, color=palette.get(s, "#cccccc"),
                edgecolor="white", linewidth=0.4, label=nice.get(s, s))
        left += vals
    ax.set_yticks(yy); ax.set_yticklabels([SHORT[c] for c in sel], fontsize=6.6)
    ax.invert_yaxis(); ax.set_xlim(0, 1.0)
    ax.set_xlabel("fraction of traces")
    ax.set_title("(a)  First stage diverging from ground truth", fontsize=8.0)
    ax.legend(frameon=False, fontsize=6.0, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.175))

    ax = axes[1]
    vals = [pd_[c]["traces_presenting_unauthorized_link"] for c in sel]
    n = pd_["aegislink_full"]["n_traces"]
    colors = [C_AEGIS if c == "aegislink_full" else (C_ABL if c.startswith("ablation") else C_BASE)
              for c in sel]
    ax.barh(yy, vals, 0.66, color=colors, edgecolor="none")
    for i, v in enumerate(vals):
        ax.text(v + n * 0.015, i, str(v), fontsize=6.2, va="center", color=C_INK)
    ax.set_yticks(yy); ax.set_yticklabels([]); ax.invert_yaxis()
    ax.set_xlim(0, n * 1.14)
    ax.set_xlabel(f"traces presenting $\\geq$1 unauthorized link  (of {n})")
    ax.set_title("(b)  Outcome-level exposure", fontsize=8.0)
    ax.grid(axis="x", linestyle=":", alpha=0.4); ax.set_axisbelow(True)

    fig.tight_layout()
    save(fig, "fig07_stage_attribution")


# =====================================================================
# Figure 8 -- pilot gate dashboard
# =====================================================================
def fig_gates():
    conds = PILOT["conditions_flat"]
    fig, ax = plt.subplots(figsize=(7.00, 3.55))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    gate_names = {"misbinding_reproducible": "G1  misbinding reproducible",
                  "baseline_not_trivial": "G2  baselines not trivial",
                  "aegislink_improvement": "G3  AegisLink improvement",
                  "generalization": "G4  generalization",
                  "novelty": "G5  novelty"}
    status_col = {"PASS": C_GOOD, "FAIL": C_BAD, "NOT_EVALUABLE": "#b0873a"}
    status_bg = {"PASS": "#e8f4ed", "FAIL": "#fdeaea", "NOT_EVALUABLE": "#fdf3e2"}

    ROW, HDR = 0.0505, 0.040
    y = 0.905
    prev = None
    for c in conds:
        if c["gate"] != prev:
            if prev is not None:
                y -= 0.014
            ax.text(0.010, y, gate_names.get(c["gate"], c["gate"]),
                    fontsize=7.4, weight="bold", color=C_INK, va="center")
            y -= HDR
            prev = c["gate"]
        st = c["status"]
        box(ax, 0.028, y - 0.021, 0.075, 0.042, c["condition_id"].split("_")[0],
            "#ffffff", "#c7d2dd", fs=6.3)
        obs = c["observed"]
        obs_s = f"{obs:.4g}" if isinstance(obs, float) else str(obs)
        if len(obs_s) > 14:
            obs_s = obs_s.replace("NO_DIRECT_COLLISION", "no collision")
        # description: left-aligned inside its own box so overflow is impossible
        desc = c["test"].replace("_", " ")
        if len(desc) > 68:
            desc = desc[:67].rstrip() + "\u2026"
        box(ax, 0.110, y - 0.021, 0.556, 0.042, "", "#fbfcfd", "#e2e8ee")
        ax.text(0.118, y, desc, fontsize=5.0, ha="left", va="center",
                color=C_INK, zorder=3)
        box(ax, 0.673, y - 0.021, 0.158, 0.042, "", "#ffffff", "#c7d2dd")
        ax.text(0.752, y, f"observed  {obs_s}", fontsize=5.4, ha="center",
                va="center", color=C_INK, zorder=3)
        box(ax, 0.838, y - 0.021, 0.154, 0.042, st.replace("_", " "),
            status_bg[st], status_col[st], fs=5.8, tc=status_col[st], weight="bold")
        y -= ROW

    d = PILOT["decision"]
    box(ax, 0.120, 0.006, 0.760, 0.070,
        f"Overall decision:  {d['overall_decision']}\n"
        f"{d['n_passed']} of {d['n_conditions_declared']} conditions passed, "
        f"{d['n_failed']} failed, {d['n_not_evaluable']} not evaluable "
        f"(rule: {d['combination_rule'].replace('_',' ')})",
        "#fdf3e2", "#b0873a", fs=6.0, weight="bold")
    ax.text(0.5, 0.978, "Preregistered pilot Go/No-Go gates", ha="center",
            fontsize=9.5, weight="bold", color=C_INK)
    save(fig, "fig08_pilot_gates")


# =====================================================================
# Figure 9 -- RQ1 prevalence + generalization across regimes
# =====================================================================
def fig_prevalence_regimes():
    fig, axes = plt.subplots(1, 2, figsize=(7.00, 2.60))

    ax = axes[0]
    up = PRIM["undefended_prevalence"]["per_action"]
    vals = [up[a]["point"] for a in ACTIONS]
    los = [up[a]["ci"]["bias_corrected"]["lo"] for a in ACTIONS]
    his = [up[a]["ci"]["bias_corrected"]["hi"] for a in ACTIONS]
    ns = [up[a]["n_units"] for a in ACTIONS]
    x = np.arange(len(ACTIONS))
    cols = [C_BAD if a in ("book", "login", "pay") else "#d98d90" for a in ACTIONS]
    ax.bar(x, vals, 0.6, color=cols, edgecolor="none")
    ax.errorbar(x, vals, yerr=[np.array(vals) - los, np.array(his) - vals],
                fmt="none", ecolor=C_INK, elinewidth=0.9, capsize=2.5)
    for i, (v, nn) in enumerate(zip(vals, ns)):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=6.6, color=C_INK, weight="bold")
        ax.text(i, 0.045, f"n={nn}", ha="center", fontsize=5.9, color="white")
    ax.set_xticks(x); ax.set_xticklabels(ACTIONS, fontsize=7.4)
    ax.set_ylim(0, 1.16); ax.set_ylabel("undefended UALER")
    ax.set_title("(a)  RQ1 prevalence without any defense", fontsize=8.8)
    ax.grid(axis="y", linestyle=":", alpha=0.4); ax.set_axisbelow(True)

    ax = axes[1]
    regs = ["validation", "test", "transfer_holdout"]
    reg_lbl = ["validation", "test\n(confirmatory)", "transfer holdout\n(unseen categories)"]
    ref = PRIM["reference_baseline_selection"]["reference_baseline"]
    x = np.arange(len(regs)); w = 0.185
    series = [("aegislink_full", "UALER", C_AEGIS, "AegisLink UALER"),
              ("aegislink_full", "ATPR", "#6fa8d6", "AegisLink ATPR"),
              (ref, "UALER", C_BASE, f"{SHORT[ref]} UALER"),
              (ref, "ATPR", "#c3cad2", f"{SHORT[ref]} ATPR")]
    for j, (cfg, met, col, lbl) in enumerate(series):
        vals = [M[cfg][r]["primary"][met] for r in regs]
        ax.bar(x + (j - 1.5) * w, vals, w, color=col, edgecolor="none", label=lbl)
        for xx, vv in zip(x + (j - 1.5) * w, vals):
            ax.text(xx, vv + 0.028, f"{vv:.2f}", ha="center", fontsize=5.6, color=C_INK)
    ax.set_xticks(x); ax.set_xticklabels(reg_lbl, fontsize=7.0)
    ax.set_ylim(0, 1.24); ax.set_ylabel("rate")
    ax.set_title("(b)  Generalization across evaluation regimes", fontsize=8.8)
    ax.legend(frameon=False, fontsize=6.1, ncol=2, loc="upper center")
    ax.grid(axis="y", linestyle=":", alpha=0.4); ax.set_axisbelow(True)

    fig.tight_layout()
    save(fig, "fig09_prevalence_regimes")




# =====================================================================
# Figure A2 -- clean split composition (replaces the step-2 plot whose
# suptitle collided with the subplot titles)
# =====================================================================
def fig_splits():
    S = json.loads((ROOT / "results" / "benchmark_splits_summary.json").read_text())
    axes_spec = [("entity_template", "entity template"),
                 ("site_template", "site template"),
                 ("attack_template", "attack template")]
    fig, axs = plt.subplots(1, 3, figsize=(7.00, 2.42))
    for ax, (key, nice) in zip(axs, axes_spec):
        a = S["axes"][key]
        names, vals, cols = [], [], []
        for split in ["train_development", "validation", "test"]:
            names.append(split.replace("_", "\n"))
            vals.append(a["split_sizes"][split])
            cols.append({"train_development": C_AEGIS, "validation": C_ABL,
                         "test": C_GOOD}[split])
        if a.get("holdout_name"):
            names.append(a["holdout_name"].replace("_", "\n"))
            vals.append(a["holdout_size"]); cols.append(C_BAD)
        x = np.arange(len(vals))
        ax.bar(x, vals, 0.62, color=cols, edgecolor="none")
        for i, (v, split) in enumerate(zip(vals, names)):
            frac = a["achieved_fractions"].get(split.replace("\n", "_"))
            lbl = f"{v}" + (f"\n{frac:.0%} of core" if frac else "\nholdout")
            ax.text(i, v + max(vals) * 0.035, lbl, ha="center", fontsize=6.0,
                    color=C_INK, linespacing=1.3)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=6.6)
        ax.set_ylim(0, max(vals) * 1.30)
        ax.set_ylabel("templates" if key == "entity_template" else "")
        ax.set_title(f"{nice}   (core pool = {a['core_pool_size']})", fontsize=8.2)
        ax.grid(axis="y", linestyle=":", alpha=0.4); ax.set_axisbelow(True)
    fig.suptitle("Template splits are exactly 50/20/30 on every axis; the two holdouts are disjoint",
                 fontsize=9.2, weight="bold", y=1.045)
    fig.tight_layout()
    save(fig, "figA2_split_composition")


# =====================================================================
# Entry point -- renders all ten figures, in manuscript order
# =====================================================================
FIGURE_BUILDERS = [
    ("fig01_graphical_abstract", fig_graphical_abstract),
    ("fig02_architecture", fig_architecture),
    ("fig03_security_utility_frontier", fig_frontier),
    ("fig04_per_action_profile", fig_per_action),
    ("fig05_ablation_effects", fig_ablations),
    ("fig06_verdicts_calibration_adaptive", fig_verdicts_calibration),
    ("fig07_stage_attribution", fig_stage_attribution),
    ("fig08_pilot_gates", fig_gates),
    ("fig09_prevalence_regimes", fig_prevalence_regimes),
    ("figA2_split_composition", fig_splits),
]


def main() -> int:
    """Render every primary figure; return a POSIX exit status."""
    print(f"Rendering {len(FIGURE_BUILDERS)} primary figures into {FIG} ...")
    print(f"  source: results/*.json  (B = {PRIM['metadata']['n_resamples_used']} resamples, "
          f"confirmatory = {PRIM['metadata']['is_results_run']})")
    failures = []
    for i, (stem, builder) in enumerate(FIGURE_BUILDERS, start=1):
        print(f"[{i}/{len(FIGURE_BUILDERS)}] {stem}")
        try:
            builder()
        except Exception as exc:                      # noqa: BLE001 - report, keep going
            failures.append((stem, repr(exc)))
            print(f"  ERROR rendering {stem}: {exc!r}")

    produced = sorted(p.name for p in FIG.glob("fig*.png"))
    print("-" * 70)
    print(f"figures/ now holds {len(produced)} primary PNGs: {', '.join(produced)}")
    if failures:
        print(f"FAILED: {len(failures)} figure(s)")
        for stem, err in failures:
            print(f"  {stem}: {err}")
        return 1
    print("All primary figures rendered.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
