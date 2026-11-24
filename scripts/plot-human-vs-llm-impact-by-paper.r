#!/usr/bin/env Rscript
# Human impact (citations) vs LLM impact, one point per paper, from the
# compare stage's overlap report. The LLM side uses the same average as
# the paper-level series: mean impact_score over all of the paper's
# prompts (per_paper$avg_llm_impact). Axes: linear 0-based x, log10 on
# citations + 1 (starts at 1 = 0 citations).
# Usage: Rscript plot-human-vs-llm-impact-by-paper.r overlap_report.json [out.png]
# Needs only the common trio: jsonlite, dplyr, ggplot2

library(jsonlite)
library(dplyr)
library(ggplot2)

args    <- commandArgs(trailingOnly = TRUE)
path    <- if (length(args) >= 1) args[1] else "data/results/overlap_report.json"
out_png <- if (length(args) >= 2) args[2] else "human_vs_llm_impact_by_paper.png"

rep <- fromJSON(path, simplifyVector = TRUE)
pp  <- rep$per_paper
if (!is.data.frame(pp)) {
  # mixed keys per paper (single-unit papers lack the intra_paper_* stats,
  # some papers have null citations/impact), so jsonlite keeps per_paper
  # as a list. Per entry, null fields come back length-0 and break
  # as.data.frame, so coerce them to NA; then bind_rows unions the columns
  pp <- dplyr::bind_rows(lapply(pp, function(x) {
    x <- lapply(x, function(v) if (length(v) == 0) NA else v)
    as.data.frame(x, stringsAsFactors = FALSE)
  }))
}

df <- pp %>%
  filter(!is.na(avg_llm_impact), !is.na(citations))

n_all <- nrow(pp)
message(sprintf("plotting %d of %d papers (dropped %d with missing impact or citations)",
                nrow(df), n_all, n_all - nrow(df)))

p <- ggplot(df, aes(avg_llm_impact, citations + 1)) +
  geom_point(alpha = 0.7, size = 2, colour = "darkgreen") +
  geom_smooth(method = "lm", formula = y ~ x, se = TRUE,
              linewidth = 0.8, colour = "black") +
  scale_x_continuous(limits = c(0, max(df$avg_llm_impact, na.rm = TRUE))) +
  # citations are heavy-tailed and can hit 0: log10 scale on the raw +1,
  # ticks read 1, 10, 100... and the axis starts at 1 (0 citations)
  scale_y_log10(limits = c(1, max(df$citations, na.rm = TRUE) + 1)) +
  labs(
    x    = "avg LLM impact (mean impact_score over the paper's prompts)",
    y    = "human impact + 1 (citations, log scale)",
    title = "Human impact (citations) vs LLM impact, per paper",
    subtitle = "One point per paper; LLM impact averaged over prompts"
  ) +
  theme_minimal() +
  theme(
    plot.background  = element_rect(fill = "white", colour = NA),
    panel.background = element_rect(fill = "white", colour = NA)
  )

ggsave(out_png, p, width = 8, height = 5, dpi = 150, bg = "white")
print(p)
unlink("Rplots.pdf")   # print() opens a pdf device in the CWD; drop it
