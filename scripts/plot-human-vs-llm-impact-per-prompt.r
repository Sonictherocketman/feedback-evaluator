#!/usr/bin/env Rscript
# Human impact (citations) vs LLM impact, one point per (paper, prompt)
# — the *raw*, unaveraged version of
# plot-human-vs-llm-impact-by-paper.r. x is the per-prompt impact_score
# from the compare report's `units` section; the paper's citation count
# comes from `per_paper` (joined on paper). Points cluster in vertical
# stacks of the paper's prompts (same citations, varying score).
# Usage: Rscript plot-human-vs-llm-impact-per-prompt.r overlap_report.json [out.png]
# Needs only the common trio: jsonlite, dplyr, ggplot2

library(jsonlite)
library(dplyr)
library(ggplot2)

args    <- commandArgs(trailingOnly = TRUE)
path    <- if (length(args) >= 1) args[1] else "data/results/overlap_report.json"
out_png <- if (length(args) >= 2) args[2] else "human_vs_llm_impact_per_prompt.png"

rep <- fromJSON(path, simplifyVector = TRUE)
per <- rep$per_paper
if (!is.data.frame(per)) {
  # mixed keys per paper (single-unit papers lack the intra_paper_* stats,
  # some papers have null citations/impact), so jsonlite keeps per_paper
  # as a list. Per entry, null fields come back length-0 and break
  # as.data.frame, so coerce them to NA; then bind_rows unions the columns
  per <- dplyr::bind_rows(lapply(names(per), function(p) {
    x <- lapply(per[[p]], function(v) if (length(v) == 0) NA else v)
    d <- as.data.frame(x, stringsAsFactors = FALSE)
    d$paper <- p
    d
  }))
} else per$paper <- rownames(per)
df   <- rep$units %>%
  as.data.frame() %>%
  left_join(per %>% select(paper, citations), by = "paper") %>%
  filter(!is.na(impact_score), !is.na(citations))

n_all <- nrow(rep$units)
message(sprintf("plotting %d of %d units (dropped %d with missing score or citations)",
                nrow(df), n_all, n_all - nrow(df)))

p <- ggplot(df, aes(impact_score, citations + 1)) +
  geom_point(alpha = 0.6, size = 1.6, colour = "darkgreen") +
  scale_x_continuous(limits = c(0, max(df$impact_score, na.rm = TRUE))) +
  # citations are heavy-tailed and can hit 0: log10 scale on the raw +1,
  # ticks read 1, 10, 100... and the axis starts at 1 (0 citations)
  scale_y_log10(limits = c(1, max(df$citations, na.rm = TRUE) + 1)) +
  labs(
    x    = "LLM impact (per-prompt impact_score)",
    y    = "human impact + 1 (citations, log scale)",
    title = "Human impact (citations) vs LLM impact, per prompt",
    subtitle = "One point per (paper, prompt); citations are per-paper"
  ) +
  theme_minimal() +
  theme(
    plot.background  = element_rect(fill = "white", colour = NA),
    panel.background = element_rect(fill = "white", colour = NA)
  )

ggsave(out_png, p, width = 8, height = 5, dpi = 150, bg = "white")
print(p)
unlink("Rplots.pdf")   # print() opens a pdf device in the CWD; drop it
