# AegisLink — Artifact Evaluation Bundle

**One command reproduces every number, table and figure in the paper.**

```bash
make reproduce-experiment
```

That is the whole quickstart. It takes roughly 10–15 minutes on a laptop and needs
**nothing but Python** — no LaTeX, no TeX Live, no `pdflatex`, no Docker, no network access,
no API key, and no GPU.

---

## What this artifact is

AegisLink is a defense against **action-link misbinding**: a retrieval-augmented agent
identifies the right *entity* but then acts on an *endpoint that entity never authorized* —
paying at a lookalike checkout, logging in at a forged portal, booking through an
unappointed reseller. This bundle contains the benchmark, the defense, seven single-factor
ablations, ten published-baseline reimplementations, an adaptive attacker, and the full
confirmatory statistical pipeline.

It is built for an AE checker, so it is deliberately narrower than the research repository:
every LaTeX dependency has been stripped out and replaced with an Excel export.

## Requirements

* **Python ≥ 3.12**
* Seven packages: `numpy`, `pandas`, `openpyxl`, `scipy`, `matplotlib`, `pyyaml`, `pytest`

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # or: pip install -e .
make reproduce-experiment PY=.venv/bin/python
```

`make verify-env` prints the version of every package it actually imported, so a version
skew shows up before the long computation starts rather than after it.

If your interpreter is not on `PATH` as `python3`, pass it in:

```bash
make reproduce-experiment PY=/path/to/python
```

## The pipeline

`make reproduce-experiment` runs six stages in order. Each is also a standalone target.

| # | Target | What it does | Roughly |
|---|---|---|---|
| 1 | `verify-env` | Python ≥ 3.12, all imports, all input files, `CONTRACT.md` byte-identical to its immutable source, and the SHA-256 of `CONTRACT.md` / `preregistration.yaml` / `evaluation/collision_rubric.yaml` matched against the **pre-pilot freeze record** in `configs/frozen_params.json` | seconds |
| 2 | `evaluate` | Verifies the replay fingerprint, then sweeps all 18 configurations over the frozen retrieval replay: every primary and secondary endpoint, paired bootstrap CIs at B = 10,000, Holm correction within each preregistered family, and all ten Section 12 Go/No-Go conditions | ~5 min |
| 3 | `export-excel` | Builds `experimental_results.xlsx` | seconds |
| 4 | `figures` | Renders 17 manuscript figures + 4 supplementary diagnostics into `figures/`, as PNG (400 dpi) and PDF | ~1 min |
| 5 | `test` | The full `pytest` suite — including static checks that no defense imports the answer key | ~4 min |
| 6 | `summary` | Prints the headline numbers and **verifies every promised artifact actually landed**; exits non-zero if anything is missing or unpopulated | seconds |

A green `make reproduce-experiment` therefore means more than "no command crashed": stage 6
independently re-checks the artifact inventory.

**Smoke test.** `make quick` runs the same pipeline at B = 200 in about two minutes. It is
for confirming the toolchain works. It sets `is_results_run = false` and its numbers must
never be reported.

## What you get

### `experimental_results.xlsx`

Six required worksheets plus one provenance sheet. Every value is read from `results/*.json`;
no number is typed by hand anywhere in the exporter.

| Worksheet | Rows | Contents |
|---|---|---|
| `Primary_Test_Results` | 18 | All 18 configurations on the confirmatory `test` split: UALER and ATPR with 95% bootstrap CIs in both bias-corrected and percentile flavours, plus FRR, OSMR, BER, CMR, abstention rate, Brier, ECE, MCE, mean decision latency, and the verdict mix |
| `Adaptive_Attack_Results` | 36 | ASR_a, UALER and ATPR under the **600-page** adaptive attack surface, separated into stratum A (300 pages, primary-claim eligible) and stratum B (300 pages, exploratory), with deltas against the static test split |
| `Per_Action_Breakdown` | 95 | 18 configurations × 5 action types (`browse`, `contact`, `book`, `login`, `pay`) plus 5 undefended-retrieval prevalence reference rows |
| `Ablations_Analysis` | 7 | The seven single-factor ablations: signed delta against full AegisLink with CI, Holm-adjusted *p*, and a machine-checked verdict on each pre-stated prediction |
| `Baselines_Comparison` | 10 | The ten Section 10 baselines on both splits, with the Section 12 non-triviality box as a **live Excel formula** |
| `Pilot_Decision_Gates` | 10 | All ten Section 12 Go/No-Go conditions: frozen threshold, observed value, margin, three-valued status, recorded reason |
| `Run_Provenance` | 70 | Replay fingerprints, corpus and index digests, seeds, resample counts, and the SHA-256 of every governance and result artifact |

Each sheet opens with an italic note block naming the exact JSON path every column came
from, and defining every metric and convention it uses. Read those notes before reading the
numbers — several columns need them. In particular:

* **`*_ci_degenerate = TRUE`** means the bootstrap distribution was degenerate (the metric
  was pinned across every resample) and the artifact fell back to the percentile interval.
  A zero-width interval there reflects a saturated metric, **not** precision.
* **`delta_*` in `Ablations_Analysis`** is `(ablation − full AegisLink)`. A *positive*
  `delta_UALER` means the ablation is worse on security; a *negative* `delta_ATPR` means it
  is worse on utility.
* **`prediction_check`** turns the qualitative predictions registered in
  `aegislink/ablations.py` into a reproducible test. Its numeric tolerances are **conventions
  chosen by the exporter, not preregistered thresholds** — `prediction_check_rule` states the
  exact rule on every row so you can reject the convention and still read the raw deltas.
* **`Baselines_Comparison.trivially_sufficient`** is a formula, not a stored string. It reads
  as blank until you open the workbook (or recalculate it). Edit a threshold cell and the
  verdict updates — that is the audit the Section 12 gate invites.

### `figures/` — 21 figures, PNG + PDF

The **17 manuscript figures**: `fig01`–`fig09`, `figA2`, four `benchmark_*` corpus and split
figures, and three `step3_*` retrieval and attacker figures. Plus **4 supplementary
diagnostics** (`step4_*`) computed on the validation split, superseded in the paper by
`fig03`–`fig06` but kept because they are what the pilot gate decisions were read from.

Every value plotted comes from `results/*.json`. Math labels are rendered by matplotlib's
built-in mathtext — no TeX involved.

### `results/*.json`

The machine-readable artifacts, shipped pre-populated so you can inspect them *before*
re-running anything, and diff your fresh run against them afterwards. `make evaluate`
overwrites `primary_test_evaluation.json`, `adaptive_robustness_evaluation.json` and
`pilot_decision_report.json`; the rest are inputs to the export and figure stages.

## Layout

```
ae_check_repository/
├── Makefile                  # the six targets above; zero LaTeX
├── README.md                 # this file
├── CONTRACT.md               # the frozen research contract
├── preregistration.yaml      # the preregistration, frozen before the pilot ran
├── requirements.txt          # 7 packages
├── pyproject.toml            # installable; pytest config
├── conftest.py
├── aegislink/                # the defense: evidence graph, verifier, 7 ablations
├── attacks/                  # attack templates and the adaptive attacker
├── baselines/                # the 10 Section 10 baseline reimplementations
├── web_rag/                  # crawler, indexer, retriever, exposure, trace recorder
├── registry/                 # entities, domains, authorization graph, split builder
├── site_generator/           # deterministic page renderer
├── parsers/                  # fail-closed safety validator
├── evaluation/               # metrics, gate evaluator, literature collision checker
├── statistics/               # paired bootstrap, BCa intervals, Holm correction
├── experiments/              # adaptive evaluation driver
├── configs/                  # splits, frozen params + digests, benchmark freeze
├── data/benchmark/           # the frozen corpus: 2,631 pages + 600 adaptive pages,
│                             #   manifests, hashes, retrieval and stage snapshots
├── tests/                    # the pytest suite
├── workflow/                 # numbered pipeline scripts, the Excel exporter,
│                             #   the env verifier and the summary checker
├── results/                  # JSON artifacts (pre-populated; regenerated by `evaluate`)
├── figures/                  # rendered PNG + PDF
├── user_data/                # the immutable contract source CONTRACT.md is diffed against
└── claim_ledger/  manuscript/  docker/     # empty; see below
```

`claim_ledger/`, `manuscript/` and `docker/` are **required to exist** by `CONTRACT.md`
Section 17 and enforced by `tests/test_preregistration.py::test_repository_structure`. They
carry no content in this bundle. That absence is disclosed, not hidden — see below.

## Metric definitions

Quoted from `evaluation/defense_metrics.py`, which is the authoritative definition.

| Metric | Definition | Unit of the denominator | Direction |
|---|---|---|---|
| **UALER** | Unauthorized action-link responses / valid responses | **per response**, not per link — one `(query, defense)` pair | lower is better |
| **ATPR** | Retained authorized third-party links / all authorized third-party links | per authorized third-party link, **conditioned on the link having been retrieved** (a defense cannot present what retrieval never surfaced); the unconditioned value is reported alongside as `ATPR_incl_unretrieved` | higher is better |
| **ASR_a** | Responses presenting ≥ 1 adaptive attack domain / responses where ≥ 1 adaptive attack page was actually retrieved | per response, conditioned on retrieval reach; `ASR_a_unconditioned` gives the same numerator over every targeted response | lower is better |
| **FRR** | Rejected authorized links / all authorized links | per authorized link retrieved | lower is better |
| **OSMR** | Incorrect official claims / all official claims, where a claim is a presented link whose decision explicitly asserts the domain is the entity's *official* site. Presenting an authorized third party is **not** an official claim. | per official claim | lower is better |
| **BER** | Benign entity recall: responses on non-fabricated entities that presented ≥ 1 authorized link | per response, conditioned on one being retrievable | higher is better |
| **CMR** | High-confidence misbinding responses / all misbinding responses | per misbinding response | lower is better |
| **Brier / ECE / MCE** | Calibration of the confidence score: Brier score, expected and maximum calibration error over 10 equal-width bins | per scored triple | lower is better |

UALER and ATPR are the two **primary** endpoints (`CONTRACT.md` Section 11). Everything else
is secondary and is not Holm-corrected across families.

**Empty denominators are reported as `null`, never as `0.0`** — "no eligible cases" and "no
failures among eligible cases" are different findings. So `OSMR` is `null` for the six
baselines with no notion of officialness (B01, B02, B03, B05, B07, B08), and `CMR` is `null`
wherever a configuration produced no misbinding response to score. Read a `null` as *not
measured here*, not as *perfect*.

## Honest scope — what this reproduction does *not* cover

Stated here and printed again by `make summary`, because an AE checker should not have to
discover it:

* It **replays a frozen retrieval snapshot**. It does not re-crawl or re-index the web. The
  replay fingerprint binds the corpus digest, index digest, retriever configuration, encoder
  id and query set; the evaluation *refuses to run* on a mismatch, so a divergent
  environment fails loudly rather than quietly producing different numbers.
* **No language model is invoked anywhere.** The reader is a deterministic surrogate. Three
  baselines (`B04_llm_as_judge`, `B07_graph_anomaly_detector`, `B10_ragshield_defense`) are
  deterministic stand-ins rather than the published systems, flagged `is_surrogate = TRUE` in
  `Baselines_Comparison`. Read those rows as a lower bound on what the real systems achieve.
* It **does not run inside a pinned container image**. The preregistration specified Docker
  Compose with pinned digests; that was replaced by corpus-digest plus replay-fingerprint
  binding. `docker/` is empty.
* The pilot verdict is **`INCONCLUSIVE`, not `GO`**. Nine of ten Section 12 conditions PASS
  and none FAIL, but `G1.1_model_families` is **`NOT_EVALUABLE`**: the deterministic surrogate
  cannot exhibit two distinct model families. `INCONCLUSIVE` is deliberately distinguished
  from `NO_GO` because the remedy differs — complete the measurement, rather than stop or
  change the method. **It is not a GO.** See `Pilot_Decision_Gates`.
* **Five of eight evidence families are flagged degenerate** on this corpus:
  `domain_lifecycle_presence`, `ownership_change`, `prompt_injection_markers`,
  `registry_delegation_table` and `domain_age` (see
  `results/evidence_family_identifiability.json`). A degenerate signal separates the classes
  perfectly here only because the corpus lacks the benign controls that would break it — e.g.
  no benign domain changes registrant while *retaining* authorization, so ownership change has
  no false-positive region. The audit records which families AegisLink therefore **declines**
  to score on, and the corpus remediation each would need. Only
  `official_backlink_direction` and `official_registry` are non-degenerate load-bearing
  families.
* **Two of the seven pre-stated ablation predictions do not hold as stated.** `Ablations_Analysis`
  reports this rather than reconciling it: `ablation_source_count_voting` was predicted to
  raise UALER *sharply* and raises it only to 0.009 (its damage lands on ATPR instead, which
  collapses from 1.000 to 0.061), and `ablation_shared_threshold` is marked `NOT_CHECKABLE`
  because full AegisLink scores UALER = 0 on all five actions here, leaving no error for a
  shared threshold to redistribute — a floor effect, not a failed prediction.

## Uploading this as a GitHub repository

`ae_check_repository/` is already a complete repository root: no path in it escapes the
directory, and `.gitignore` is in place.

```bash
cd ae_check_repository
git init -b main
git add .
git commit -m "AegisLink artifact evaluation bundle"

# then either:
gh repo create aegislink-ae --public --source=. --push
# or, with a repository created in the web UI:
git remote add origin git@github.com:<you>/aegislink-ae.git
git push -u origin main
```

**Before pushing, check the size.** The bundle is about **87 MB across roughly 3,370 files**,
and it is not evenly distributed — four files account for 75 MB of it:

| File | Size |
|---|---|
| `data/benchmark/defense_traces.json` | 61 MB |
| `data/benchmark/retrieval_snapshots.json` | 7.8 MB |
| `data/benchmark/stage_traces.json` | 4.5 MB |
| `data/benchmark/adaptive_retrieval_snapshots.json` | 1.8 MB |

Plain `git push` works: every file is under GitHub's **100 MB hard limit**, and the repository
is well under the 1 GB soft recommendation, so **Git LFS is not required**. Be aware that
GitHub prints a warning for any file over 50 MB, so `defense_traces.json` will trigger one —
the push still succeeds. If you would rather not carry it in git history, track it with LFS
(`git lfs track "data/benchmark/defense_traces.json"`) or publish it as a release asset;
**do not simply exclude it**, because `make figures` reads the stage traces derived from it.

If you are submitting for anonymous review, note that `git init` here starts a fresh history
with no prior commits, but your `user.name` and `user.email` will still be attached to the
commit you create.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: openpyxl` | `pip install -r requirements.txt` |
| `verify-env` reports a digest mismatch | A governance file has been edited since the preregistration was frozen. Restore it; do not update the digest. |
| `evaluate` aborts on a fingerprint mismatch | The corpus or index under `data/benchmark/` has changed. Restore the shipped files. |
| `trivially_sufficient` column is blank | Expected — it is a formula. Open the workbook, or recalculate it (`soffice --headless --convert-to xlsx`). |
| `is_results_run: false` in the summary | You ran at `B != 10000` (probably via `make quick`). Re-run `make evaluate`. |
| `make` not available | Run the six `workflow/` scripts directly, in the order listed in the pipeline table. |

## Citation and governance

Every claim in the paper traces to `CONTRACT.md` (the research contract) and
`preregistration.yaml` (frozen before the pilot ran, `amendment_number: 0`). Section
references throughout the code and this README point into `CONTRACT.md`. The SHA-256 of both
documents is checked by `make verify-env` against `configs/frozen_params.json`, which was
written before any result existed.
