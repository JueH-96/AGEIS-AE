# =====================================================================
#  AegisLink -- Artifact Evaluation (AE) reproduction entry point
#
#  One command:   make reproduce-experiment
#
#  ZERO LaTeX / TeX / PDF-compiler dependencies. No target in this file
#  invokes pdflatex, xelatex, bibtex or any TeX tool, and no .tex, .cls
#  or .bst file ships in this repository. Everything an AE checker needs
#  -- the confirmatory numbers, a multi-tab Excel workbook, every figure
#  and the test suite -- is produced by Python alone.
#
#  Honest scope statement, mirrored in README.md:
#    * This reproduces the ANALYSIS from a frozen retrieval replay. The
#      replay fingerprint is verified before any analysis runs, so a
#      divergent environment fails loudly rather than quietly producing
#      different numbers.
#    * It does NOT re-crawl or re-index the corpus, and it does NOT run
#      inside a pinned container image.
#    * No language model is invoked anywhere. The reader is a
#      deterministic surrogate.
# =====================================================================

# Interpreter. Override if your virtualenv lives elsewhere:
#     make reproduce-experiment PY=python3.12
#     make reproduce-experiment PY=/path/to/.venv/bin/python
PY ?= python3

# Bootstrap resamples. 10000 is the preregistered value; anything else
# sets is_results_run = false and the run is NOT confirmatory.
RESAMPLES ?= 10000

XLSX     := experimental_results.xlsx
PRIMARY  := results/primary_test_evaluation.json
PILOT    := results/pilot_decision_report.json
ADAPTIVE := results/adaptive_robustness_evaluation.json

.DEFAULT_GOAL := help
.PHONY: help reproduce-experiment verify-env evaluate export-excel figures test \
        summary hashes quick clean clean-figures

# ---------------------------------------------------------------- help
help:
	@echo ""
	@echo "  AegisLink -- Artifact Evaluation targets"
	@echo "  ========================================"
	@echo ""
	@echo "  make reproduce-experiment   THE ONE COMMAND. Runs, in order:"
	@echo "                                verify-env -> evaluate -> export-excel"
	@echo "                                -> figures -> test -> summary"
	@echo "                              Takes roughly 10-15 minutes."
	@echo ""
	@echo "  Individual stages"
	@echo "  -----------------"
	@echo "  make verify-env      check Python >= 3.12, dependencies, inputs, and the"
	@echo "                       SHA-256 digests of CONTRACT.md and preregistration.yaml"
	@echo "                       against the pre-pilot freeze record"
	@echo "  make evaluate        re-run the confirmatory evaluation (B=$(RESAMPLES))"
	@echo "  make export-excel    build $(XLSX) (6 required worksheets + provenance)"
	@echo "  make figures         render all 17 manuscript figures + 4 supplementary"
	@echo "                       diagnostics into figures/ as PNG and PDF"
	@echo "  make test            run the pytest suite"
	@echo "  make summary         print headline numbers and verify every artifact landed"
	@echo "  make hashes          print SHA-256 of every governance and result artifact"
	@echo ""
	@echo "  Convenience"
	@echo "  -----------"
	@echo "  make quick           same pipeline at B=200. FAST BUT NOT CONFIRMATORY --"
	@echo "                       use it to smoke-test the toolchain, never to report"
	@echo "  make clean           remove generated caches and $(XLSX)"
	@echo "  make clean-figures   remove rendered figures"
	@echo ""
	@echo "  Overrides:  PY=<interpreter>   RESAMPLES=<int>"
	@echo "  Current  :  PY=$(PY)   RESAMPLES=$(RESAMPLES)"
	@echo ""

# ----------------------------------------------------- the one command
reproduce-experiment: verify-env evaluate export-excel figures test summary

# --------------------------------------------------------- verify-env
verify-env:
	@echo ""
	@echo ">>> [1/6] verify-env"
	$(PY) workflow/verify_environment.py

# ----------------------------------------------------------- evaluate
# Sweeps all 18 configurations over the frozen replay, recomputes every
# primary and secondary endpoint with paired bootstrap CIs, applies Holm
# correction within each preregistered family, and re-evaluates all ten
# Section 12 Go/No-Go conditions.
evaluate:
	@echo ""
	@echo ">>> [2/6] evaluate  (B=$(RESAMPLES); expect several minutes)"
	@test $(RESAMPLES) -eq 10000 || echo "    WARNING: B=$(RESAMPLES) != preregistered 10000; is_results_run will be false"
	$(PY) workflow/19_evaluate_primary_and_adaptive.py --resamples $(RESAMPLES)

# -------------------------------------------------------- export-excel
export-excel: $(PRIMARY)
	@echo ""
	@echo ">>> [3/6] export-excel"
	$(PY) workflow/export_excel.py

# ------------------------------------------------------------- figures
# Every value plotted is read from results/*.json; no number is typed by
# hand anywhere in these scripts.
figures: $(PRIMARY)
	@echo ""
	@echo ">>> [4/6] figures"
	$(PY) workflow/20_plot_primary_results.py
	$(PY) workflow/14_plot_web_rag_summary.py
	$(PY) workflow/17_plot_defense_summary.py
	$(PY) workflow/11_plot_benchmark_summary.py

# ---------------------------------------------------------------- test
test:
	@echo ""
	@echo ">>> [5/6] test"
	$(PY) -m pytest tests/ -q

# ------------------------------------------------------------- summary
summary:
	@echo ""
	@echo ">>> [6/6] summary"
	@$(PY) workflow/summarize_artifacts.py

# --------------------------------------------------------------- extras
hashes:
	@$(PY) -c "import hashlib, os; \
fs=['CONTRACT.md','preregistration.yaml','configs/frozen_params.json', \
    'configs/splits.yaml','configs/benchmark_freeze.json', \
    '$(PRIMARY)','$(PILOT)','$(ADAPTIVE)','$(XLSX)']; \
[print(hashlib.sha256(open(f,'rb').read()).hexdigest(), \
       str(os.path.getsize(f)).rjust(9), f) for f in fs if os.path.exists(f)]"

quick:
	@echo "*** QUICK MODE: B=200. NOT CONFIRMATORY. Do not report these numbers. ***"
	@$(MAKE) --no-print-directory reproduce-experiment RESAMPLES=200

clean:
	rm -f $(XLSX)
	rm -rf .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	@echo "[clean] removed $(XLSX), .pytest_cache and __pycache__ trees"

clean-figures:
	rm -f figures/*.png figures/*.pdf
	@echo "[clean-figures] figures/ emptied (re-render with 'make figures')"
