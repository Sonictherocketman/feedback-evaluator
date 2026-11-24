# feedback_evaluator

A small installable package that critiques academic papers with a Claude
Code agent, embeds the resulting feedback **sections** (one per catalogue
prompt — the unit of evaluation), and compares the embeddings for
overlap, cross-checked against each paper's citation count.

```
extract   convert papers/*.pdf -> a text dir/*.txt (resumable)
eval      run one Claude Code agent session per (paper, prompt);
          writes the markdown critique + a resumable evaluations.jsonl
embed     embed one unit per evaluations.jsonl record via a server
compare   pairwise cosine over the units -> overlap report
```

## Install

```bash
pip install -e .            # or: -e ".[dev]" to include pytest + black + semgrep
```

Run everything from the repo root. Path options default to CWD-relative,
and the Makefile pins every stage to the repo-root `data/` tree
(`data/pdfs`, `data/text`, `data/feedback`, `data/embeddings`,
`data/results`).

## Usage

```bash
feedback_evaluator extract --papers-dir data/pdfs --text-dir data/text
feedback_evaluator eval \
    --papers-dir data/pdfs \
    --data-catalog data/catalog/arxiv-w-s2-w-papers.200.jsonl \
    --text-dir data/text --model partner --resume
feedback_evaluator embed --embed-url http://localhost:11434/v1 --model nomic-embed-text
feedback_evaluator compare
```

(`python -m feedback_evaluator` works too.)

- **extract** — converts each `papers/*.pdf` to
  `<text-dir>/<paper>.txt` (pypdf text layer; figures are dropped,
  equations come through as Unicode where the PDF's fonts allow).
  Existing non-empty files are skipped, so it's cheap to re-run as the
  corpus changes. A page that trips pypdf (e.g. a dense figure page
  hitting its token limit) is dropped with a `[warn]` count instead of
  failing the paper; only an all-failed PDF fails. Run this before
  `eval`: the model is then given the text file, not the raw PDF (`eval`
  falls back to the PDF per paper when no text file exists). `--text-dir`
  must already exist — `extract` writes into it and does not create it.
- **eval** — the agent session runs with the same process the Claude CLI
  uses (Claude Agent SDK): read-only (`dontAsk`), cwd = the directory you
  ran from, system prompt `prompts/system.md`, question prompts from
  `prompts/catalog.yaml`. Pass the tier alias `sonnet`/`opus`/`haiku`
  (or a proxy model id) as `--model`; the CLI maps tier aliases to your
  proxy models via `ANTHROPIC_DEFAULT_{SONNET,OPUS,HAIKU}_MODEL`.
  Per record it outputs:
    - `feedback/<paper>.md` — one `## [<feedback_id>]` section per prompt,
      with a `- DOI: …` / `- Citations: N` / `- Model: …` header block.
    - `feedback/evaluations.jsonl` — one record per completed
      (paper, prompt): `paper_id`, `doi`, `prompt_id`, `feedback_id`,
      `feedback`, `originality_score`, `overall_paper_score`,
      `impact_score`, `citations`, `model`, … **Appended as each record
      completes** (not rewritten at the end), so a killed run loses
      nothing.
  - **Resume is automatic**: on start, prior records are loaded from
    `evaluations.jsonl` and any `(paper, prompt)` already present is
    skipped, so re-running picks up where it left off (`--resume` is kept
    for clarity/compat).
  - **DOI + citations** come from `--data-catalog` (the S2-enriched data
    pipeline's last output, one JSON object per line), joined by the PDF
    file name recorded in each row's `pdf_path`. The paper's **DOI** is
    taken from the catalogue; with no catalogue (or no row) it falls back
    to scanning the PDF's first two pages (`doi.py`: `doi.org` link →
    `DOI:` label → first bare `10.xxxx/…` token) and is null for
    arXiv-only preprints. `citations` is the paper's citation count from
    the catalogue (`citation_count`), else null.
- **embed** — OpenAI-compatible `POST <url>/embeddings`; Bearer auth only
  when a key is given. Env: `EMBED_URL`, `EMBED_MODEL`, `EMBED_API_KEY`.
  **Embeds straight from `feedback/evaluations.jsonl`** — one unit per
  record (the `feedback` field is the text sent to the server), so the
  `feedback/*.md` files — rendered only for human reference — are never
  parsed. Outputs: `embeddings/embeddings.jsonl` (each record keeps its
  `id` = `<paper_id>#<feedback_id>`, `paper`, `feedback_id`, `prompt_id`,
  `doi`, the feedback text, the three scores, `citations`, `model`, `dim`,
  and the `vector`) + `embeddings/manifest.json` (per-paper unit counts).
  Runs fail if `evaluations.jsonl` is missing or has no usable records.
- **compare** — L2-normalized cosine over units (all pairs, or
  `--within-papers` for within-paper only; all-pair mode above 10 000
  units errors and points you to `--within-papers`). Reports cosine
  stats, the **full pair list** sorted by similarity, and per-paper
  coherence. Every pair row — cross-paper **and** within-paper — carries
  its ids, `paper_a`/`paper_b`, `doi_a`/`doi_b`, `prompt_a`/`prompt_b`,
  a `same_paper` flag, a `same_prompt` flag, and the **impact metrics**:
    - `total_human_impact` — sum of the two papers' citation counts
      (a paper-level value, null only if neither paper has a count).
    - `total_llm_impact` / `x_impact` — the sum, and the absolute
      difference, of the two units' `impact_score`s (null only if either
      unit has no score).
  Within-paper rows (same paper, different prompts) are the **control**
  for prompt-variability; cross-paper same-prompt rows are the treatment
  comparison. The `same_paper` / `same_prompt` flags let the analysis
  stage (R) filter either population without re-deriving it. Output:
  `results/overlap_report.json` (no `near_duplicate_pairs`, `source`, or
  `n_lines` — the full `pairs` list is the data).
  - **`paper_pairs`** — a second, *paper-level* view: one row per pair of
    distinct papers, with the per-prompt data collapsed. Each row carries
    `avg_llm_impact_a`/`_b` (each paper's mean `impact_score` over its
    prompts), `avg_x_llm_impact` (their |diff|),
    `human_impact_a`/`_b` (each paper's citation count — a paper-level
    constant, so no averaging), `x_human_impact` (their |diff|),
    `mean_cosine` (mean overlap over
    every unit×unit combo between the two papers), and
    `total_human_impact`. Because the LLM impact is averaged over prompts,
    `avg_x_llm_impact` is continuous, which is what the `plot-x-impact`
    series uses for a smooth trend line.
  - **`per_paper`** — one entry per paper: unit count, coherence stats,
    plus `avg_llm_impact` (mean `impact_score` over the prompts — the
    same average `paper_pairs` uses) and `citations` (the paper-level
    citation count), which is what
    `plot-human-vs-llm-impact-by-paper` reads.
  - **`units`** — one row per unit (`id`, `paper`, `prompt_id`, the raw
    per-prompt `impact_score`); `plot-human-vs-llm-impact-per-prompt`
    joins this against `per_paper` citations for an unaveraged
    per-prompt view.
  - **`cosine_split`** — the cosine distribution split on the
    `same_paper` flag: `within_paper` (same paper, different prompts)
    and `between_paper` (cross-paper) statistics (the same
    mean/median/p05/p95/min/max shape as `cosine_stats`), plus
    `mean_within_vs_between` (within mean / between mean) when both
    sides are non-empty *and* the between mean is nonzero. A within
    value clearly above the between value is the signal that a paper
    looks more like itself than like other papers; the split is
    absent on whichever side has no pairs (e.g. `--within-papers` mode
    has no `between_paper`).

## Layout

```
data/                         everything under the repo root (Makefile-owned)
  catalog/                    arXiv → S2-enriched JSONL (data-acquisition)
  pdfs/                       full-text PDFs
  text/                       extracted paper text (extract stage)
  feedback/                   <paper>.md + evaluations.jsonl (eval stage)
  embeddings/                 embeddings.jsonl + manifest.json (embed)
  results/                    overlap_report.json (compare)
feedback_evaluator/            package: common, evaluator, embedding,
                               compare, doi, extract, cli (+ __main__ for -m runs)
tests/                         pytest suite (offline: the agent is never
                               called; the embed server is a local stub)
```

The `scripts/` directory at the repo root holds the data-acquisition and
R plotting scripts (`enrich-via-s2.py`, `fetch-pdfs.py`, `plot-citations.r`).

## Makefile (repo root)

The data pipeline, analysis, plots, and dev checks all run from a single
`Makefile` at the repo root. `make help` lists every target. First time:
`make install` (installs the package with the `dev` extra — pytest,
black, semgrep — editable).

```
# Setup
make install              install the package + dev tools (editable)

# Getting data (data/ catalog -> pdfs -> text)
make get-archive          download the arXiv metadata dataset to data/catalog/
make filter-by-doi        keep catalog entries that have a DOI
make subset               random $(N_PAPERS)  (N_PAPERS=... to override)
make enrich-via-s2        enrich the subset with Semantic-Scholar metadata
make download-pdfs        fetch the full-text PDFs -> data/pdfs/
make extract-text         papers/*.pdf -> data/text/*.txt (feedback_evaluator extract)
make prepare              get-archive -> ... -> extract-text (whole data prep)

# Running the analysis
make eval                 stage 1: paper critiques    -> data/feedback/
make embed                stage 2: feedback sections  -> data/embeddings/
make compare              stage 3: embedding overlap  -> data/results/
make run                  extract-text -> eval -> embed -> compare (all)

# Plots (R)
make plot-raw-citations   citations vs age -> plots/
make plot-x-impact        x-LLM-impact vs overlap (paper-level)
make plot-x-human-impact  x-human-impact (citation diff) vs overlap
make plot-human-vs-llm-impact-by-paper  human vs LLM impact (per-paper avg)
make plot-human-vs-llm-impact-per-prompt  human vs LLM impact (per prompt)
make plot-paper-impact    group paper centroids, compare impact by group
make plots                all plot targets

# Developer checks
make format               black-format the package and tests
make format-check         black --check; fails if unformatted
make test                 run the pytest suite (with coverage)
make coverage             tests + coverage table, and htmlcov/ report
make scan                 semgrep security/bug scan
make checks               all checks: format-check + test + scan
make clean                remove build/bytecode/coverage artifacts
```

Every Makefile target pins its paths to the repo root — so `make` works
from any subdirectory and writes into `data/`. Extra flags go through the
`*_ARGS` variables (on any stage, or through `make run`):

```bash
make run EVAL_ARGS="--model opus --resume" \
         EMBED_ARGS="--embed-url http://localhost:11434/v1 --model nomic-embed-text"
```

## Formatting

`black`, configured in `pyproject.toml` under `[tool.black]` — 79-column
lines, Python 3.10 target. After changing any `*.py` under this directory,
run `make format` and `make test` from the repo root (also codified in the
repo's CLAUDE.md). Note: the `Makefile` recipes run `black` from inside this
directory so `pyproject.toml`'s config is picked up automatically. Write new
code already black-clean so `make format` stays a no-op.

## Testing

From the repo root: `make test` (or `cd feedback_evaluator &&
python -m pytest -q`). The suite is fully offline — the agent SDK is
never invoked and the embedding server is a stdlib local stub. Coverage
runs with every pytest invocation (pytest-cov, config in
`pyproject.toml`); `make coverage` additionally writes an HTML report to
`feedback_evaluator/htmlcov/`. Keep overall coverage above 80%.

## Security scan

`make scan` runs `semgrep --error --config auto` over the package (non-zero
exit on any finding, CI-friendly). `semgrep` comes from the `dev` extra;
if it isn't on PATH the target prints an install hint and exits 1.
