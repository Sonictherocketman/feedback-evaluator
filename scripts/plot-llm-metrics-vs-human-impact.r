#!/usr/bin/env Rscript
# Each LLM metric against human impact, one point per paper.
#
# Standalone (touches nothing else): reads embeddings.jsonl directly and,
# per paper, averages each of the three LLM scores over the paper's prompts
# plus the paper-level citation count, giving one point per (paper, metric).
# Each metric is drawn as its own coloured series with a linear trend, so you
# can see directly which of impact / originality / overall tracks human impact
# (citations) hardest. Axes match the per-paper chart: linear 0-based metric,
# log10(citations + 1) on the y.
# Usage: Rscript plot-llm-metrics-vs-human-impact.r [embeddings.jsonl] [out.png]
# Needs only the common trio: jsonlite, dplyr, ggplot2

library(jsonlite)
library(dplyr)
library(ggplot2)

args    <- commandArgs(trailingOnly = TRUE)
jsonl   <- if (length(args) >= 1) args[1] else "data/embeddings/embeddings.jsonl"
out_png <- if (length(args) >= 2) args[2] else
  "plots/llm_metrics_vs_human_impact.png"

num <- function(v) if (length(v) == 0) NA_real_ else as.numeric(v)

recs  <- lapply(readLines(jsonl, warn = FALSE), fromJSON, simplifyVector = TRUE)
paper <- vapply(recs, function(r) r$paper, character(1))
U <- data.frame(
  paper       = paper,
  impact      = vapply(recs, function(r) num(r$impact_score), numeric(1)),
  originality = vapply(recs, function(r) num(r$originality_score), numeric(1)),
  overall     = vapply(recs, function(r) num(r$overall_paper_score), numeric(1)),
  citations   = vapply(recs, function(r) num(r$citations), numeric(1)),
  stringsAsFactors = FALSE
)

# per paper: mean of each score over its prompts; citations is constant
m <- U %>%
  group_by(paper) %>%
  summarise(
    impact      = mean(impact, na.rm = TRUE),
    originality = mean(originality, na.rm = TRUE),
    overall     = mean(overall, na.rm = TRUE),
    citations   = first(citations),
    .groups = "drop"
  )
m[is.na(m)] <- NA
n_all <- nrow(m)

# long form (melt by hand, no tidyr): one row per (paper, metric)
df <- data.frame(
  paper  = c(m$paper, m$paper, m$paper),
  metric = rep(c("impact", "originality", "overall"), each = nrow(m)),
  score  = c(m$impact, m$originality, m$overall),
  stringsAsFactors = FALSE
)
df$citations <- m$citations[match(df$paper, m$paper)]
df <- df[!is.na(df$score) & !is.na(df$citations), ]

message(sprintf("plotting %d papers (%d dropped for missing score or citations)",
                length(unique(df$paper)), n_all - length(unique(df$paper))))
max_cit <- max(df$citations, na.rm = TRUE)

p <- ggplot(df, aes(score, citations + 1, colour = metric)) +
  geom_point(alpha = 0.6, size = 1.8) +
  geom_smooth(method = "lm", formula = y ~ x, se = TRUE,
              linewidth = 0.9, alpha = 0.12) +
  facet_wrap(~ metric, ncol = 1, scales = "free_x") +
  scale_y_log10(limits = c(1, max_cit + 1)) +
  scale_colour_manual(
    name = NULL,
    values = c(impact = "firebrick", originality = "darkgreen",
               overall = "steelblue")
  ) +
  labs(
    x = "LLM metric (mean over the paper's prompts)",
    y = "human impact + 1 (citations, log scale)",
    title = "Each LLM metric vs human impact, per paper",
    subtitle = "One point per paper per metric; linear trend per metric"
  ) +
  theme_minimal() +
  theme(
    plot.background  = element_rect(fill = "white", colour = NA),
    panel.background = element_rect(fill = "white", colour = NA)
  )

ggsave(out_png, p, width = 7, height = 9, dpi = 150, bg = "white")
print(p)
unlink("Rplots.pdf")   # print() opens a pdf device in the CWD; drop it
