#!/usr/bin/env Rscript
# Citation count vs. paper age (months), from the merged arXiv + S2 JSONL.
# Usage: Rscript plot_citations.R merged_pdfs.jsonl [out.png]
#   (or edit `path` below)
# install.packages(c("jsonlite", "dplyr", "ggplot2", "lubridate"))

library(jsonlite)
library(dplyr)
library(ggplot2)
library(lubridate)

args    <- commandArgs(trailingOnly = TRUE)
path    <- if (length(args) >= 1) args[1] else "merged.jsonl"
out_png <- if (length(args) >= 2) args[2] else "citations_vs_age.png"

as_of <- as.Date("2026-11-21")   # hardcoded reference date for "age"

papers <- stream_in(file(path), verbose = FALSE)

# v1 submission date, parsed from e.g. "Sun, 21 Feb 2010 20:01:21 GMT".
# Regex + month.abb avoids locale-dependent %b parsing.
parse_v1 <- function(versions) {
  s <- if (is.data.frame(versions) && nrow(versions) > 0) versions$created[1] else NA_character_
  m <- regmatches(s, regexec("(\\d{1,2}) ([A-Za-z]{3}) (\\d{4})", s))[[1]]
  if (length(m) != 4) return(as.Date(NA))
  as.Date(sprintf("%s-%02d-%02d", m[4], match(m[3], month.abb), as.integer(m[2])))
}

df <- tibble(
  id        = papers$id,
  category  = sub(" .*", "", papers$categories),          # primary category
  submitted = as.Date(vapply(papers$versions, function(v) as.character(parse_v1(v)),
                             character(1))),
  citations = papers$citation_count                       # NA where S2 had no match
) %>%
  mutate(age_months = time_length(interval(submitted, as_of), "month"))

n_all <- nrow(df)
df <- df %>% filter(!is.na(citations), !is.na(age_months))
message(sprintf("plotting %d of %d papers (dropped %d with missing citations or date)",
                nrow(df), n_all, n_all - nrow(df)))

# Citations are heavy-tailed and include zeros: plot citations + 1 on a log axis.
p <- ggplot(df, aes(age_months, citations + 1)) +
  geom_point(alpha = 0.4, size = 1.2) +
  geom_smooth(method = "loess", formula = y ~ x, se = TRUE, linewidth = 0.8) +
  scale_y_log10() +
  labs(
    x = "Age (months since arXiv v1)",
    y = "Citations + 1 (log scale)",
    title = "Citation count vs. paper age",
    subtitle = sprintf("Age as of %s; citations from Semantic Scholar at retrieval time", as_of)
  ) +
  theme_minimal() +
  theme(
    plot.background  = element_rect(fill = "white", colour = NA),
    panel.background = element_rect(fill = "white", colour = NA)
  )

# To split by field instead: p + facet_wrap(~ category)  (or color = category)

ggsave(out_png, p, width = 8, height = 5, dpi = 150, bg = "white")
print(p)
