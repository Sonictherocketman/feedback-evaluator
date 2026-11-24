# Repo build file. Sections: getting data, running the analysis,
# plots, paper, developer checks. `make help` lists every target.
#
# Every target is pinned to the repo root (this Makefile's directory),
# so `make` works from any subdirectory.
#
# Data lives under data/ (catalog/, pdfs/, text/, feedback/,
# embeddings/, results/); overrides: DATA, PLOTS, PROMPTS, N_PAPERS,
# and per-stage *_ARGS, e.g.
#   make eval EVAL_ARGS="--model opus --resume"
#
# Dev tools (black, semgrep, pytest) live in the package's `dev` extra:
#   pip install -e "feedback_evaluator[dev]"

ROOT       := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PKG        := feedback_evaluator
PY         ?= python3
DATA       ?= $(ROOT)/data
PLOTS       ?= $(ROOT)/plots
PROMPTS    ?= $(ROOT)/prompts
EXTRACT_ARGS ?= --papers-dir $(DATA)/pdfs --text-dir $(DATA)/text
EVAL_ARGS  ?= --papers-dir $(DATA)/pdfs --system $(PROMPTS)/system.md --catalog $(PROMPTS)/catalog.yaml --feedback-dir $(DATA)/feedback --data-catalog $(DATA)/catalog/arxiv-w-s2-w-papers.$(N_PAPERS).jsonl --text-dir $(DATA)/text --model partner --resume
EMBED_ARGS ?= --feedback-dir $(DATA)/feedback --out-dir $(DATA)/embeddings
COMPARE_ARGS ?= --embeddings-file $(DATA)/embeddings/embeddings.jsonl --out-dir $(DATA)/results
N_PAPERS   ?= 200

.PHONY: help install format format-check test coverage get-archive filter-by-doi \
	subset enrich-via-s2 download-pdfs extract-text prepare \
	eval embed compare run plot-raw-citations plot-x-impact \
	plot-x-human-impact plot-human-vs-llm-impact-by-paper \
	plot-human-vs-llm-impact-per-prompt plot-paper-impact \
	plot-llm-metrics plots scan checks clean

help:
	@echo "Setup"
	@echo "  make install              install the package + dev tools (editable)"
	@echo ""
	@echo "Getting data (data/ catalog -> pdfs -> text)"
	@echo "  make get-archive          download the arXiv metadata dataset to data/catalog/"
	@echo "  make filter-by-doi        keep catalog entries that have a DOI"
	@echo "  make subset               random $(N_PAPERS) (N_PAPERS=... to override)"
	@echo "  make enrich-via-s2        enrich the subset with Semantic-Scholar metadata"
	@echo "  make download-pdfs        fetch the full-text PDFs -> data/pdfs/"
	@echo "  make extract-text         papers/*.pdf -> data/text/*.txt (feedback_evaluator extract)"
	@echo "  make prepare              get-archive -> ... -> extract-text (whole data prep)"
	@echo ""
	@echo "Running the analysis"
	@echo "  make eval                 stage 1: paper critiques      -> data/feedback/"
	@echo "  make embed                stage 2: embedding units      -> data/embeddings/"
	@echo "  make compare              stage 3: embedding overlap    -> data/results/"
	@echo "  make run                  extract-text -> eval -> embed -> compare (all)"
	@echo ""
	@echo "Plots (R)"
	@echo "  make plot-raw-citations   citations vs age -> plots/"
	@echo "  make plot-x-impact        x-LLM-impact vs overlap (from the compare report)"
	@echo "  make plot-x-human-impact  x-human-impact (citation diff) vs overlap"
	@echo "  make plot-human-vs-llm-impact-by-paper  human vs LLM impact (per-paper avg)"
	@echo "  make plot-human-vs-llm-impact-per-prompt  human vs LLM impact (per prompt)"
	@echo "  make plot-paper-impact    group paper centroids, compare impact by group"
	@echo "  make plot-llm-metrics     each LLM metric vs human impact (per paper)"
	@echo "  make plots                all plot targets"
	@echo ""
	@echo "Developer checks"
	@echo "  make format               black-format the package and tests"
	@echo "  make format-check         black --check; fails if unformatted"
	@echo "  make test                 run the pytest suite (with coverage)"
	@echo "  make coverage             tests + coverage table, and htmlcov/ report"
	@echo "  make scan                 semgrep security/bug scan"
	@echo "  make checks               all checks: format-check + test + scan"
	@echo "  make clean                remove build/bytecode/coverage artifacts"

install:
	cd $(ROOT) && $(PY) -m pip install -e "$(PKG)[dev]"

format:
	cd $(ROOT)/$(PKG) && black feedback_evaluator tests

format-check:
	cd $(ROOT)/$(PKG) && black --check feedback_evaluator tests

test:
	cd $(ROOT)/$(PKG) && $(PY) -m pytest -q

coverage:
	cd $(ROOT)/$(PKG) && $(PY) -m pytest -q \
		--cov-report=term-missing --cov-report=html:htmlcov
	@echo "html report: $(ROOT)/$(PKG)/htmlcov/index.html"

# Getting data

get-archive:
    cd $(DATA)/catalog && curl -L -o arxiv.zip \
		https://www.kaggle.com/api/v1/datasets/download/Cornell-University/arxiv && \
		unzip arxiv
	@echo "archive downloaded to data/catalog/ and unzipped"

filter-by-doi:
	cd $(DATA)/catalog && \
		jq -c 'select(.doi != null)' \
			arxiv-metadata-oai-snapshot.json > \
			non-null-doi.jsonl
	@echo "removed items without a valid DOI"

# TODO: Filter by recency (now>x>5yo?)
subset:
	cd $(DATA)/catalog && \
	cat non-null-doi.jsonl | shuf | head -n $(N_PAPERS) > subset.$(N_PAPERS).jsonl
	@echo "Truncated to $(N_PAPERS) records"

enrich-via-s2:
	cd $(DATA)/catalog && $(PY) $(ROOT)/scripts/enrich-via-s2.py \
		subset.$(N_PAPERS).jsonl \
		arxiv-w-s2.$(N_PAPERS).jsonl
	@echo "enriched dataset with metadata from S2"

download-pdfs:
	cd $(DATA)/catalog && $(PY) $(ROOT)/scripts/fetch-pdfs.py \
		arxiv-w-s2.$(N_PAPERS).jsonl \
		arxiv-w-s2-w-papers.$(N_PAPERS).jsonl \
		$(DATA)/pdfs
	@echo "downloaded the fulltext pdfs to data/pdfs"

extract-text:
	cd $(ROOT)/$(PKG) && $(PY) -m feedback_evaluator extract $(EXTRACT_ARGS)
	@echo "extracted the text from the pdfs to data/text/"

prep: get-archive filter-by-doi subset enrich-via-s2 download-pdfs extract-text
	@echo "==> done"

# Running the analysis

eval:
	cd $(ROOT)/$(PKG) && $(PY) -m feedback_evaluator eval $(EVAL_ARGS)

embed:
	cd $(ROOT)/$(PKG) && $(PY) -m feedback_evaluator embed $(EMBED_ARGS)

compare:
	cd $(ROOT)/$(PKG) && $(PY) -m feedback_evaluator compare $(COMPARE_ARGS)

run: extract-text eval embed compare
	@echo "==> done"

# Plots

plot-raw-citations:
	Rscript $(ROOT)/scripts/plot-citations.r \
		$(DATA)/catalog/arxiv-w-s2.$(N_PAPERS).jsonl \
		$(PLOTS)/citations_vs_age.png

plot-x-impact:
	@for f in $(DATA)/results/overlap_report.json $(DATA)/embeddings/embeddings.jsonl; do \
		[ -f $$f ] || { echo "missing $$f - run the pipeline up to compare first"; exit 1; }; done
	Rscript $(ROOT)/scripts/plot-x-impact-vs-overlap.r \
		$(DATA)/results/overlap_report.json \
		$(PLOTS)/x_impact_vs_overlap.png \
		$(DATA)/embeddings/embeddings.jsonl

plot-x-human-impact:
	@for f in $(DATA)/results/overlap_report.json $(DATA)/embeddings/embeddings.jsonl; do \
		[ -f $$f ] || { echo "missing $$f - run the pipeline up to compare first"; exit 1; }; done
	Rscript $(ROOT)/scripts/plot-x-human-impact-vs-overlap.r \
		$(DATA)/results/overlap_report.json \
		$(PLOTS)/x_human_impact_vs_overlap.png \
		$(DATA)/embeddings/embeddings.jsonl

plot-human-vs-llm-impact-by-paper:
	@[ -f $(DATA)/results/overlap_report.json ] || { \
		echo "missing $(DATA)/results/overlap_report.json - run make compare first"; exit 1; }
	Rscript $(ROOT)/scripts/plot-human-vs-llm-impact-by-paper.r \
		$(DATA)/results/overlap_report.json \
		$(PLOTS)/human_vs_llm_impact_by_paper.png

plot-human-vs-llm-impact-per-prompt:
	@[ -f $(DATA)/results/overlap_report.json ] || { \
		echo "missing $(DATA)/results/overlap_report.json - run make compare first"; exit 1; }
	Rscript $(ROOT)/scripts/plot-human-vs-llm-impact-per-prompt.r \
		$(DATA)/results/overlap_report.json \
		$(PLOTS)/human_vs_llm_impact_per_prompt.png

plot-paper-impact:
	@[ -f $(DATA)/embeddings/embeddings.jsonl ] || { \
		echo "missing $(DATA)/embeddings/embeddings.jsonl - run make embed first"; exit 1; }
	Rscript $(ROOT)/scripts/paper-centroids-impact.r \
		$(DATA)/embeddings/embeddings.jsonl \
		$(PLOTS)/paper_impact_clusters.png \
		$(PLOTS)/paper_impact_clusters.csv

plot-llm-metrics:
	@[ -f $(DATA)/embeddings/embeddings.jsonl ] || { \
		echo "missing $(DATA)/embeddings/embeddings.jsonl - run make embed first"; exit 1; }
	Rscript $(ROOT)/scripts/plot-llm-metrics-vs-human-impact.r \
		$(DATA)/embeddings/embeddings.jsonl \
		$(PLOTS)/llm_metrics_vs_human_impact.png

plots: plot-raw-citations plot-x-impact plot-x-human-impact \
	plot-human-vs-llm-impact-by-paper plot-human-vs-llm-impact-per-prompt \
	plot-paper-impact plot-llm-metrics

# Paper

# TODO: Typeset

# Developer checks

scan:
	@command -v semgrep >/dev/null 2>&1 || { \
		echo "semgrep not found. Install it with: $(PY) -m pip install 'semgrep'"; \
		exit 1; }
	cd $(ROOT) && semgrep --error --config auto $(PKG)

checks: format-check test scan

clean:
	cd $(ROOT) \
		&& rm -rf $(PKG)/build $(PKG)/dist $(PKG)/$(PKG).egg-info \
			.pytest_cache $(PKG)/.pytest_cache .coverage $(PKG)/.coverage \
			$(PKG)/htmlcov
	find $(ROOT) -type d -name '__pycache__' -not -path '$(ROOT)/.git/*' -exec rm -rf {} +
