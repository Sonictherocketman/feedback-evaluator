#!/usr/bin/env Rscript
# standalone analysis: is the LLM's `impact_score` a genuine importance
# proxy, or is its correlation with (human) citations an artifact of
# length / recency / the other score outputs?  Partial-correlation test.
#
# This is a throwaway analysis, separate from the pipeline scripts: it
# re-reads embeddings.jsonl + the S2 catalog + the extracted text, builds
# one row per paper, and asks whether impact still predicts citations
# after partialling out candidate confounds.  It writes only its own
# outputs (a text report + a csv) and touches nothing else.
#
# Question being decided (from the earlier decomposition):
#   (a) impact infers real importance  -> its citations link survives the
#       controls
#   (b) impact reads importance-rhetoric/prose -> the link collapses once
#       length etc are partialled out
#
# Controls tested, in order of interpretive weight:
#   length  (word count of extracted text)  -- "long papers read as bigger"
#   age     (years since publication)       -- "older = more time to accrue"
#   other scores (originality, overall)     -- "just the quality gestalt"
#
# Usage: Rscript impact-provenance.r [catalog.jsonl] [embeddings.jsonl]
#        [text_dir] [out.txt] [out.csv]

suppressMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
CAT <- if (length(args) >= 1) args[1] else
  "data/catalog/arxiv-w-s2-w-papers.200.jsonl"
EMB <- if (length(args) >= 2) args[2] else "data/embeddings/embeddings.jsonl"
TEX <- if (length(args) >= 3) args[3] else "data/text"
OUT <- if (length(args) >= 4) args[4] else "plots/impact_provenance.txt"
OUTC <- if (length(args) >= 5) args[5] else "plots/impact_provenance.csv"

REF_DATE <- Sys.Date()   # analysis time; controls wall-clock time-to-accrue

num <- function(v) if (length(v) == 0) NA_real_ else as.numeric(v)

recs  <- lapply(readLines(EMB, warn = FALSE), fromJSON, simplifyVector = TRUE)
paper <- vapply(recs, function(r) r$paper, character(1))
# per paper: mean of each LLM score over its prompts, and citations. Build
# each column as a *named* per-paper vector and paste them side by side on
# the (identical) paper set -- the unit-level `paper` vector must not be
# recycled into a data.frame, it would misalign the rows.
meanb <- function(f) {
  x <- tapply(vapply(recs, function(r) num(r[[f]]), numeric(1)),
              paper, function(z) mean(z, na.rm = TRUE))
  x[is.na(x)] <- NA_real_
  x
}
cols <- list(
  impact = meanb("impact_score"),
  originality = meanb("originality_score"),
  overall = meanb("overall_paper_score"),
  citations = meanb("citations")
)
# tapply names every column with the same sorted paper ids, so as.data.frame
# aligns them row-wise with rownames = those paper ids
D <- as.data.frame(cols, stringsAsFactors = FALSE)
D$paper <- rownames(D)
D$lcit <- log10(pmax(D$citations, 0) + 1)

`%||%` <- function(x, y) if (is.null(x) || (length(x) == 1 && is.na(x))) y else x

# ---- publication age from the S2 catalog ----------------------------------
# the catalog is JSON *lines*: parse per line, keep id + a publication date
clines   <- readLines(CAT, warn = FALSE)
clines_p <- lapply(clines, fromJSON, simplifyVector = TRUE)
cid      <- vapply(clines_p, function(r) r$id, character(1))
# per single-line parse `s2` is a *list* (not a data.frame), so read its
# fields directly; fall back to the first arXiv version date (data.frame)
s2dates  <- vapply(clines_p, function(r) {
  pd <- if (!is.null(r$s2)) r$s2$publicationDate else NA_character_
  if (is.null(pd) || is.na(pd)) {
    v  <- r$versions
    pd <- if (is.data.frame(v) && nrow(v) >= 1) v$created[1] else NA_character_
  }
  as.character(pd)
}, character(1))
date_by_id <- setNames(s2dates, cid)

baseid     <- sub("v[0-9]+$", "", D$paper)   # catalog id has no version suffix
pdate      <- vapply(baseid, function(b) date_by_id[[b]] %||% NA_character_,
                     character(1))
# ISO "YYYY-MM-DD" parses directly; the arXiv versions fallback is RFC-ish
# ("Mon, 26 Jul 2021 21:41:37 GMT"). as.Date on a *mixed* vector errors if
# any element is non-ISO, so parse element-by-element with a fallback.
pub <- sapply(pdate, function(s) {
  d <- tryCatch(as.Date(s), error = function(e) NA,
                warning = function(w) NA)
  if (!is.na(d)) return(d)
  m <- regmatches(s, regexec("(\\d{1,2} [A-Z][a-z]{2} \\d{4})", s))[[1]]
  if (length(m) == 3) return(as.Date(m, "%d %b %Y"))
  NA
})
pub <- as.Date(pub)
D$age_years <- as.numeric(REF_DATE - pub) / 365.25
cat(sprintf("date coverage: %d/%d papers dated (%d NA)\n",
            sum(!is.na(pub)), length(pub), sum(is.na(pub))))

# ---- text length (words) ---------------------------------------------------
txtf  <- list.files(TEX, pattern = "\\.txt$")
words <- setNames(numeric(0), character(0))
for (f in txtf) {
  # count words in the *file contents*, not the path string
  txt <- paste(readLines(file.path(TEX, f), warn = FALSE), collapse = " ")
  words[sub("\\.txt$", "", f)] <- length(strsplit(txt, "\\s+")[[1]])
}
D$length <- words[D$paper]
cat(sprintf("text-length coverage: %d/%d papers (words range %d-%d)\n",
            sum(!is.na(D$length)), nrow(D),
            min(D$length, na.rm = TRUE), max(D$length, na.rm = TRUE)))

# ---- the partial-correlation machinery (base R) ---------------------------
# rows complete across x, y and the control columns
complete_rows <- function(d, x, y, zcols) {
  keep <- !is.na(d[[x]]) & !is.na(d[[y]])
  for (z in zcols) keep <- keep & !is.na(d[[z]])
  d[keep, , drop = FALSE]
}
# partial corr of x vs y controlling for zcols: residualize both on z,
# correlate the residuals (base R throughout).
partial_corr <- function(d, x, y, zcols) {
  sub <- complete_rows(d, x, y, zcols)
  if (nrow(sub) < length(zcols) + 3) return(NA_real_)
  Zm <- as.matrix(sub[zcols])
  rx <- resid(lm(as.numeric(sub[[x]]) ~ Zm))
  ry <- resid(lm(as.numeric(sub[[y]]) ~ Zm))
  if (isTRUE(var(rx) == 0) || isTRUE(var(ry) == 0)) return(NA_real_)
  cor(rx, ry)
}
# permutation null: shuffle y, re-fit and re-correlate B times
partial_test <- function(d, x, y, zcols, B = 999) {
  sub <- complete_rows(d, x, y, zcols)
  if (nrow(sub) < length(zcols) + 3)
    return(c(obs = NA_real_, p = NA_real_))
  Zm <- as.matrix(sub[zcols])
  obs <- partial_corr(d, x, y, zcols)
  xv <- as.numeric(sub[[x]]); yv <- as.numeric(sub[[y]])
  smp <- replicate(B, {
    rx <- resid(lm(xv ~ Zm))
    ry <- resid(lm(sample(yv) ~ Zm))
    cor(rx, ry)
  })
  c(obs = obs, p = (sum(smp >= obs) + 1) / (B + 1))
}

# ---- build the report ------------------------------------------------------
lines <- character(0)
add <- function(...) {
  lines <<- c(lines, if (missing(...)) "" else sprintf(...))
  invisible()
}
sep <- function() lines <<- c(lines, strrep("-", 64))
add("Impact-provenance partial-correlation analysis")
add("reference date: %s   papers: %d   (data: %s)",
    format(REF_DATE), nrow(D), basename(EMB))
sep()
# zero-order correlations with log-citations
for (v in c("impact", "originality", "overall", "length", "age_years")) {
  rr <- cor(D[[v]], D$lcit, use = "complete.obs")
  if (is.na(rr)) rr <- 0
  add("zero-order   %-11s vs log10(cit+1) :  %+5.2f  (n=%d)", v, rr,
      sum(!is.na(D[[v]]) & !is.na(D$lcit)))
}
sep()
add("PARTIAL correlation of impact vs log10(cit+1), controlling for:")
sets <- list(
  "none (zero-order)          " = NULL,
  "length (words)             " = "length",
  "age (years since pub)      " = "age_years",
  "originality+overall        " = c("originality", "overall"),
  "length + age               " = c("length", "age_years"),
  "length+age+orig+overall    " = c("length", "age_years",
                                    "originality", "overall"))
for (nm in names(sets)) {
  cc <- if (is.null(sets[[nm]])) {
    cor(D$impact, D$lcit, use = "complete.obs")
  } else {
    partial_corr(D, "impact", "lcit", sets[[nm]])
  }
  if (is.na(cc)) cc <- 0
  add("  %-26s %+5.2f", nm, cc)
}
sep()
add("PERMUTATION p-value, full-controlled partial corr:")
full <- c("length", "age_years", "originality", "overall")
pr <- partial_test(D, "impact", "lcit", full)
add("  impact~log10cit | length,age,orig,overall : partial r=%+.2f  p=%s",
    pr["obs"], signif(pr["p"], 2))
add("")
add("READ: if the full-controlled partial corr is still clearly > 0, the")
add("impact score carries information about citations beyond length,")
add("recency and the other LLM scores (genuine importance proxy). If it")
add("collapses toward 0, its citations link was an artifact of those.")

txt <- paste(lines, collapse = "\n")
writeLines(txt, OUT)
write.csv(D, OUTC, row.names = FALSE)
cat(txt, "\n")
cat(sprintf("\nwrote %s and %s\n", OUT, OUTC))
