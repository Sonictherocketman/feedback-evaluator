# Repo-level checks and pipeline for the feedback_evaluator package.
#
# Every target is pinned to the repo root (this Makefile's directory),
# so `make` works from any subdirectory.
#
# Dev tools (black, semgrep, pytest) live in the package's `dev` extra:
#   pip install -e "feedback_evaluator[dev]"
#
# `eval`, `embed`, `compare` are the pipeline stages (the CLI's path
# options resolve against the repo root: papers/, prompts/, ...);
# `run` runs them all in order. Override each with e.g.
# `make eval EVAL_ARGS="--model opus --resume"`.

ROOT       := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PKG        := feedback_evaluator
PY         ?= python3
EXTRACT_ARGS ?= --papers-dir $(ROOT)/papers
EVAL_ARGS  ?= --papers-dir $(ROOT)/papers --system $(ROOT)/prompts/system.md --catalog $(ROOT)/prompts/catalog.yaml --feedback-dir $(ROOT)/feedback --model partner --resume
EMBED_ARGS ?= --feedback-dir $(ROOT)/feedback --out-dir $(ROOT)/embeddings
COMPARE_ARGS ?= --embeddings-file $(ROOT)/embeddings/embeddings.jsonl --out-dir $(ROOT)/results

.PHONY: help install format format-check test coverage extract eval embed compare run scan checks clean

help:
	@echo "make install         install the package + dev tools (editable)"
	@echo "make format          black-format the package and tests"
	@echo "make format-check    black --check (no writes); fails if unformatted"
	@echo "make test            run the pytest suite (with coverage)"
	@echo "make coverage        tests + coverage table, and htmlcov/ report"
	@echo "make extract         stage 0: papers/*.pdf -> papers/text/*.txt"
	@echo "make eval            stage 1: paper critiques -> feedback/"
	@echo "make embed           stage 2: feedback lines -> embeddings/"
	@echo "make compare         stage 3: embedding overlap -> results/"
	@echo "make run             extract -> eval -> embed -> compare (all)"
	@echo "make scan            semgrep security/bug scan"
	@echo "make checks          all checks: format-check + test + scan"
	@echo "make clean           remove build/bytecode artifacts"

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

extract:
	cd $(ROOT)/$(PKG) && $(PY) -m feedback_evaluator extract $(EXTRACT_ARGS)

eval:
	cd $(ROOT)/$(PKG) && $(PY) -m feedback_evaluator eval $(EVAL_ARGS)

embed:
	cd $(ROOT)/$(PKG) && $(PY) -m feedback_evaluator embed $(EMBED_ARGS)

compare:
	cd $(ROOT)/$(PKG) && $(PY) -m feedback_evaluator compare $(COMPARE_ARGS)

run: extract eval embed compare
	@echo "==> done"

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
