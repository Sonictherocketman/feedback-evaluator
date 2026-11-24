#!/usr/bin/env Rscript
# Paper-level LLM x-impact vs embedding overlap, from the compare stage's
# overlap report. Uses the `paper_pairs` section, where each row is a pair
# of papers and x-impact is *averaged over prompts* (avg_x_llm_impact),
# so the series is continuous and the trend line is smooth. The raw
# unit-level `pairs` section stays in the report for the per-prompt data.
# Overlap is recomputed from the raw embeddings with the vectors
# mean-centered first (see paper-pair-centered-overlap.R).
# The trend line is the rolling window mean; the grey band around it is
# +/-1 std dev of that same window (the real local spread). Below the main
# panel the same spread is drawn as its own line, on shared x limits.
# Usage: Rscript plot-x-impact-vs-overlap.r overlap_report.json [out.png] [embeddings.jsonl]
# Needs jsonlite, dplyr, ggplot2, patchwork

library(jsonlite)
library(dplyr)
library(ggplot2)
library(patchwork)
# source the helper that lives next to this script (CWD when interactive)
script_path <- sub("--file=", "", grep("^--file=", commandArgs(), value = TRUE))
if (length(script_path) >= 1) {
  source(file.path(dirname(script_path[1]), "paper-pair-centered-overlap.R"))
} else {
  source("paper-pair-centered-overlap.R")
}

args      <- commandArgs(trailingOnly = TRUE)
path      <- if (length(args) >= 1) args[1] else "data/results/overlap_report.json"
out_png   <- if (length(args) >= 2) args[2] else "x_impact_vs_overlap.png"
emb_path  <- if (length(args) >= 3) args[3] else "data/embeddings/embeddings.jsonl"

pp <- fromJSON(path, simplifyVector = TRUE)$paper_pairs

# mean overlap per paper pair, centered (the report's mean_cosine is raw,
# so drop it and use the recomputed value)
ov    <- centered_paper_overlaps(emb_path)
n_all <- nrow(pp)

df <- pp %>%
  select(-mean_cosine) %>%
  filter(!is.na(avg_x_llm_impact)) %>%
  left_join(ov, by = c("paper_a", "paper_b")) %>%
  rename(mean_cosine = mean_overlap) %>%
  filter(!is.na(mean_cosine))

message(sprintf("plotting %d of %d paper-pairs (dropped %d with missing avg_x_llm_impact)",
                nrow(df), n_all, n_all - nrow(df)))

x_lab <- "avg x-LLM-impact (avg impact_a - avg impact_b, over prompts)"
xs    <- c(0, max(df$avg_x_llm_impact, na.rm = TRUE))

# One rolling window drives both panels: at each x the kk nearest
# paper-pairs (fixed count -> density-adaptive: tight in the clumps, wide
# in the gaps). The trend is the window mean; the spread is the window
# std dev (the base-R window, not the loess 95% CI of the mean).
k   <- 100
kk  <- min(k, nrow(df))
gxx <- seq(xs[1], xs[2], length.out = 200)
sel <- vapply(
  gxx,
  function(x0) order(abs(df$avg_x_llm_impact - x0))[1:kk],
  integer(kk)
)
# sel is kk x (length gxx); vector indexing with it is column-major, so
# reshape straight back into the kk x (length gxx) window matrix
vals <- matrix(df$mean_cosine[sel], nrow = kk)
ctr  <- colMeans(vals, na.rm = TRUE)
sp   <- apply(vals, 2, function(v) {
  if (length(v) >= 2) sd(v, na.rm = TRUE) else NA_real_
})
smry <- data.frame(x = gxx, ctr = ctr, lo = ctr - sp, hi = ctr + sp)

ys <- c(min(df$mean_cosine, smry$lo) - 0.05,
        max(df$mean_cosine, smry$hi) + 0.05)

p <- ggplot(df, aes(avg_x_llm_impact, mean_cosine)) +
  geom_point(alpha = 0.6, size = 1.6, colour = "steelblue") +
  # band = window mean +/- 1 std dev (the data's local spread); line = the
  # same window mean (the trend)
  geom_ribbon(
    data = smry,
    aes(x = x, ymin = lo, ymax = hi),
    inherit.aes = FALSE,
    alpha = 0.25,
    fill = "grey70",
    colour = "black",
    linewidth = 0.3
  ) +
  geom_line(
    data = smry,
    aes(x = x, y = ctr, group = 1),
    inherit.aes = FALSE,
    colour = "black",
    linewidth = 0.9
  ) +
  scale_x_continuous(limits = xs) +
  scale_y_continuous(limits = ys) +
  labs(
    x   = x_lab,
    y   = "mean overlap (centered cosine, all units between the two papers)",
    title = "Paper-level x-LLM-impact vs embedding overlap",
    subtitle = "Each point is a pair of papers; impact averaged over prompts;
                overlap = mean-centered cosine"
  ) +
  theme_minimal() +
  theme(
    plot.background  = element_rect(fill = "white", colour = NA),
    panel.background = element_rect(fill = "white", colour = NA),
    # the stacked spread panel carries the x tick labels
    axis.text.x  = element_blank(),
    axis.ticks.x = element_blank()
  )

# Sub-panel: the same local spread (window std dev) as its own line.
p_sp <- ggplot(data.frame(x = gxx, sd = sp), aes(x, sd)) +
  geom_line(colour = "grey30", linewidth = 0.7) +
  scale_x_continuous(limits = xs) +
  scale_y_continuous(limits = c(0, max(sp, na.rm = TRUE) * 1.1)) +
  labs(x = x_lab, y = "local spread (std of overlap)") +
  theme_minimal() +
  theme(
    plot.background  = element_rect(fill = "white", colour = NA),
    panel.background = element_rect(fill = "white", colour = NA)
  )

comb <- p / p_sp + plot_layout(heights = c(4, 1))
ggsave(out_png, comb, width = 8, height = 6, dpi = 150, bg = "white")
print(comb)
unlink("Rplots.pdf")   # print() opens a pdf device in the CWD; drop it
