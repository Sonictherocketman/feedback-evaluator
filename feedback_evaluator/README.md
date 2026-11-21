# feedback_evaluator

A small installable package that critiques academic papers with a Claude
Code agent, embeds the resulting feedback lines, and compares the
embeddings for overlap.

```
extract   convert papers/*.pdf -> papers/text/*.txt (resumable)
eval      run one Claude Code agent session per (paper, prompt)
embed     embed each prose line of the feedback via an embedding server
compare   pairwise cosine over the embeddings -> overlap report
```

## Install

```bash
pip install -e .            # or: -e ".[dev]" to include pytest + black + semgrep
```

Run everything from the repo root — path options default to
CWD-relative: `papers/`, `prompts/`, `feedback/`, `embeddings/`,
`results/`.

## Usage

```bash
feedback_evaluator extract         # papers/*.pdf -> papers/text/*.txt
feedback_evaluator eval            # 16 papers x catalog prompts -> feedback/
feedback_evaluator eval --resume --model opus
feedback_evaluator embed --embed-url http://localhost:11434/v1 --model nomic-embed-text
feedback_evaluator compare --threshold 0.9 --top 20
```

(`python -m feedback_evaluator` works too.)

- **extract** — converts each `papers/*.pdf` to
  `papers/text/<paper>.txt` (pypdf text layer; figures are dropped,
  equations come through as Unicode where the PDF's fonts allow).
  Existing non-empty files are skipped, so it's cheap to re-run as the
  corpus changes. A page that trips pypdf (e.g. a dense figure page
  hitting its token limit) is dropped with a `[warn]` count instead of
  failing the paper; only an all-failed PDF fails. Run this before
  `eval`: the model is then given the text file, not the raw PDF
  (`eval` falls back to the PDF per paper when no text file exists).
- **eval** — the agent session runs with the same process the Claude CLI
  uses (Claude Agent SDK): read-only (`dontAsk`), cwd = the directory you
  ran from, system prompt `prompts/system.md`, question prompts from
  `prompts/catalog.yaml`. Pass the tier alias `sonnet`/`opus`/`haiku`
  as `--model`; the CLI maps it to your proxy models via
  `ANTHROPIC_DEFAULT_{SONNET,OPUS,HAIKU}_MODEL` in
  `.claude/settings.local.json`. Outputs: `feedback/<paper>.md` plus
  `feedback/evaluations.jsonl` (paper_id, doi, prompt_id, feedback_id,
  feedback text, originality_score, overall_paper_score, impact_score
  per record). The paper's own **DOI** is extracted from the first two
  PDF pages (`doi.py`: `doi.org` link → `DOI:` label → first bare
  `10.xxxx/…` token) and stored as `doi` (null for arXiv-only
  preprints); it also appears in the feedback file header
  (`- DOI: …`).
- **embed** — OpenAI-compatible `POST <url>/embeddings`; Bearer auth only
  when a key is given. Env: `EMBED_URL`, `EMBED_MODEL`, `EMBED_API_KEY`.
  Outputs: `embeddings/embeddings.jsonl` (each record keeps its paper,
  feedback section, line number, line text and vector) +
  `embeddings/manifest.json`. Each record also carries the paper's
  `doi` (read from its feedback file header).
- **compare** — L2-normalized cosine for all pairs (or `--within-papers`),
  cross- vs within-paper means, top similar pairs, near-duplicates above
  `--threshold`, per-paper coherence, and a `papers_dois` map
  (paper → DOI pulled from the embedding records; pairs also carry
  `doi_a`/`doi_b`). Output: `results/overlap_report.json`.

## Layout

```
feedback_evaluator/            package: common, evaluator, embedding,
                               compare, cli (+ __main__ for -m runs)
tests/                         pytest suite (offline: the agent is never
                               called; the embed server is a local stub)
```

The `scripts/` directory at the repo root is reserved for R plotting
scripts against the results.

## Makefile (repo root)

The checks and the pipeline run from a single `Makefile` at the repo
root. First time: `make install` (installs the package with the `dev`
extra — pytest, black, semgrep — editable).

```
make install         pip install -e "feedback_evaluator[dev]"
make format          black-format the package and tests
make format-check    black --check — fails if anything is unformatted
make test            run the pytest suite
make extract         stage 0: papers/*.pdf -> papers/text/*.txt
make eval            stage 1: paper critiques  -> feedback/
make embed           stage 2: feedback lines   -> embeddings/
make compare         stage 3: embedding overlap -> results/
make run             extract -> eval -> embed -> compare (full pipeline)
make scan            semgrep --error --config auto over the package
make checks          all checks in CI order: format-check + test + scan
make clean           remove build/bytecode artifacts
```

The stages are thin wrappers over the CLI, and every Makefile target
pins its paths to the repo root — so `make` works from any
subdirectory and the pipeline writes to the usual `papers/` →
`feedback/` → `embeddings/` → `results/` flow. Extra flags go through
the `*_ARGS` variables (on any stage, or through `make run`):

```bash
make run EVAL_ARGS="--resume --model opus" \
         EMBED_ARGS="--embed-url http://localhost:11434/v1 --model nomic-embed-text" \
         COMPARE_ARGS="--threshold 0.9 --top 20"
```

## Formatting

`black`, configured in `pyproject.toml` under `[tool.black]` — 79-column
lines, Python 3.10 target, `.git`/`.venv`/egg-info excluded. After
changing any `*.py` under this directory, run `make format` and
`make test` from the repo root (also codified in the repo's CLAUDE.md).
Note: `Makefile` recipes run `black` from inside this directory so
`pyproject.toml`'s config is picked up automatically.

## Testing

From the repo root: `make test` (or `cd feedback_evaluator &&
python -m pytest -q`). The suite is fully offline — the agent SDK is
never invoked and the embedding server is a stdlib local stub. Coverage
runs with every pytest invocation (pytest-cov, config in
`pyproject.toml`); `make coverage` additionally writes an HTML report
to `feedback_evaluator/htmlcov/`.

## Security scan

`make scan` runs `semgrep --error --config auto` over the package (non-zero
exit on any finding, CI-friendly). `semgrep` comes from the `dev` extra;
if it isn't on PATH the target prints an install hint and exits 1.
