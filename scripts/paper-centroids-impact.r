#!/usr/bin/env Rscript
# Paper-centroid -> cluster -> impact.
#
# Rationale (the experiment's premise): the 14 catalogue prompts are
# intended as *different ways a user might phrase a request for the same
# thing* (critiques); the model's *responses* are the signal. So the
# embedding-mean of a paper's units ("the average review of this paper")
# should carry the paper-level characteristics. We ask: does k-means on
# those ~N centroids find groups whose members differ in the human
# (citation) or LLM (mean impact_score) impact?
#
# Steps
#   1. Load embeddings.jsonl; per paper: mean vector, mean impact_score
#      (two-stage: scores averaged per paper, then per cluster),
#      citation count.
#   2. Diagnostic: mean raw cosine within a paper vs between papers --
#      if the between-prompt spread rivals the paper-to-paper spread, the
#      centroid is a diluted vector (the premise is weaker than it looks).
#   3. Center the centroid matrix, PCA to 95% of variance (capped 4..30
#      components), k-means for k = 2..14, silhouette picks k.
#   4. Per cluster: mean LLM impact, median/range of citations.
#      Kruskal-Wallis + size-preserving permutation test (eta^2) on both
#      impact measures, Spearman correlations of impact vs top PCs.
#
# Outputs
#   console: diagnostics + cluster table + tests
#   [out.csv] one row per paper: cluster, llm_impact, citations, pc1..
#   [out.png] PC scatter with cluster impact labels; per-cluster impact
#             bars; paper-level impact-vs-citations coloured by cluster;
#             silhouette-vs-k
#
# TODO
#   * Leave-one-paper-out clusters to confirm the groups aren't driven by
#     a few extreme papers (n is ~58).
#   * Mclust BIC selection as a second k reference.
#   * Repeat on the impact axis only: does the *LLM impact* structure
#     match the citation structure (the core question of the whole repo)?
#
# Usage: Rscript paper-centroids-impact.r [embeddings.jsonl] [out.png] [out.csv]
# Needs jsonlite, dplyr, ggplot2, patchwork, cluster (standard R + viz trio)

suppressMessages({
  library(jsonlite); library(dplyr); library(ggplot2)
  library(patchwork); library(cluster)
})

args    <- commandArgs(trailingOnly = TRUE)
jsonl   <- if (length(args) >= 1) args[1] else "data/embeddings/embeddings.jsonl"
out_png <- if (length(args) >= 2) args[2] else "plots/paper_impact_clusters.png"
out_csv <- if (length(args) >= 3) args[3] else "plots/paper_impact_clusters.csv"

# --- load, one row per paper ----------------------------------------------
recs   <- lapply(readLines(jsonl, warn = FALSE),
                 fromJSON, simplifyVector = TRUE)
paper  <- vapply(recs, function(r) r$paper, character(1))
impact <- vapply(recs, function(r) {
  v <- r$impact_score
  if (length(v) == 0) NA_real_ else as.numeric(v)
}, numeric(1))
cite   <- vapply(recs, function(r) {
  v <- r$citations
  if (length(v) == 0) NA_real_ else as.numeric(v)
}, numeric(1))
V      <- do.call(rbind, lapply(recs, function(r) r$vector))

idx    <- split(seq_len(nrow(V)), paper)
Vc     <- t(sapply(idx, function(i) colMeans(V[i, , drop = FALSE])))
papers <- rownames(Vc)
# two-stage LLM impact: per-paper mean, missing units excluded
llm <- tapply(impact, paper, function(v) mean(v, na.rm = TRUE))
llm[is.na(llm)] <- NA_real_
CI  <- tapply(cite, paper, function(v) mean(v, na.rm = TRUE))
CI[is.na(CI)]   <- NA_real_
cat(sprintf("%d units -> %d paper centroids, dim %d\n",
            nrow(V), length(papers), ncol(V)))

# --- premise diagnostic: paper signal vs prompt spread ---------------------
G    <- tcrossprod(V)                       # raw cosines (units normalized)
w_i  <- vapply(idx, function(i) mean(G[i, i]), numeric(1))
b_i  <- vapply(idx, function(i) {
  j <- setdiff(seq_len(nrow(G)), i)
  mean(G[i, j])
}, numeric(1))
cat(sprintf("mean within-paper cosine %.3f | cross-paper %.3f (ratio %.2f)\n",
            mean(w_i), mean(b_i), mean(w_i) / mean(b_i)))

# --- PCA on the centered centroid matrix ------------------------------------
pc    <- prcomp(Vc, center = TRUE, scale. = FALSE)
cumv  <- cumsum(pc$sdev^2) / sum(pc$sdev^2)
ncomp <- min(max(which(cumv >= 0.95)[1], 4), 30)
X     <- as.matrix(pc$x[, 1:ncomp, drop = FALSE])
cat(sprintf("PCA: %d -> %d components (%.1f%% of centroid variance)\n",
            ncol(Vc), ncomp, cumv[ncomp] * 100))

# --- k-means, k by silhouette ----------------------------------------------
# (silhouette tends to keep rising as k grows with n this small, so note
# whether the argmax touches the k ceiling below)
D   <- dist(X)
npp <- nrow(X)
ks  <- 2:min(20, floor(npp / 3))   # keep clusters of at least ~3 papers
sil <- sapply(ks, function(k) {
  cl <- kmeans(X, centers = k, nstart = 25, iter.max = 150)$cluster
  mean(silhouette(as.integer(cl), D)[, 3], na.rm = TRUE)
})
bestk <- ks[which.max(sil)]
labs  <- kmeans(X, centers = bestk, nstart = 25, iter.max = 150)$cluster
if (bestk == max(ks))
  cat("NOTE: silhouette argmax is at the k ceiling; k is under-determined\n")
cat(sprintf("best k = %d (silhouette %.3f)\n", bestk, max(sil)))

# --- per-cluster impact table + tests ---------------------------------------
eta2 <- function(y, g) {
  d <- data.frame(y = y, g = g); d <- d[!is.na(d$y), ]
  gg <- split(d$y, d$g); m <- tapply(d$y, d$g, mean)
  sum(lengths(gg) * (m - mean(d$y)) ^ 2) / sum((d$y - mean(d$y)) ^ 2)
}
perm_p <- function(y, g, B = 999) {
  obs <- eta2(y, g)
  # a resample can strand the (few) NA rows as an entire cluster of their
  # own, making eta2 NA; such resamples carry >= obs by definition, so
  # counting NAs as >= keeps the test valid and non-NA
  smp <- replicate(B, eta2(y, sample(g)))
  (sum(is.na(smp) | smp >= obs) + 1) / (B + 1)
}
# per-cluster stats as a plain data.frame (tapply returns ragged dims)
summ <- do.call(rbind, lapply(sort(unique(labs)), function(g) {
  p <- papers[labs == g]
  data.frame(n = length(p), llm = mean(llm[p], na.rm = TRUE),
             cit = median(CI[p], na.rm = TRUE))
}))
rownames(summ) <- sort(unique(labs))
cat("\nCluster   papers   meanLLM   medianCit\n")
for (g in rownames(summ)) {
  c <- summ[g, "cit"]; l <- summ[g, "llm"]
  cat(sprintf("  %s         %d      %s      %s\n",
              g, summ[g, "n"],
              if (is.na(l)) "NA" else sprintf("%.2f", l),
              if (is.na(c)) "NA" else c))
}

# labs is a plain length-59 integer vector in Vc row order (one entry per
# centroid). The impact arrays are *named* by paper, so pairing them with
# labs by paper name -- NOT llm[labs], which positionally indexes the 59
# impact values by the cluster *numbers* (1..k) and silently collapses the
# data -- is essential. Order both by rownames(Vc) so they align by row.
p_order <- rownames(Vc)
llm_ord <- llm[p_order]
cit_ord <- CI[p_order]
cl_fac  <- factor(labs)
for (nm in c("LLM impact", "citations")) {
  y  <- if (nm == "LLM impact") llm_ord else cit_ord
  ok <- !is.na(y)
  r  <- tryCatch({ kw <- kruskal.test(y[ok], cl_fac[ok]); kw$p.value }
                 , error = function(e) NA_real_)
  p  <- perm_p(y, cl_fac)
  et <- eta2(y, cl_fac)
  cat(sprintf("\n%s vs cluster: eta2=%.3f  Kruskal-Wallis p=%.3f  permutation p=%.3f\n",
              nm, et, r, p))
}
for (k in 1:min(4, ncomp)) {
  for (nm in c("LLM impact", "citations")) {
    y <- if (nm == "LLM impact") llm else CI
    # complete.obs: citations has a few NA papers and cor's default
    # (everything) would return NA for the whole column
    c0 <- suppressWarnings(cor(X[, k], y, method = "spearman",
                               use = "complete.obs"))
    cat(sprintf("Spearman PC%d vs %s: %s\n", k, nm,
                if (is.na(c0)) "NA" else sprintf("%.2f", c0)))
  }
}

# --- outputs ----------------------------------------------------------------
out <- data.frame(
  paper = papers, cluster = as.character(labs),
  llm_impact = llm, citations = CI,
  pc1 = X[, 1], pc2 = X[, 2],
  stringsAsFactors = FALSE
)
write.csv(out, out_csv, row.names = FALSE)
cat(sprintf("\nwrote %d rows -> %s\n", nrow(out), out_csv))

# --- figure -----------------------------------------------------------------
whitebg <- function() theme(
  plot.background = element_rect(fill = "white", colour = NA),
  panel.background = element_rect(fill = "white", colour = NA)
)
cl_levels <- sort(as.character(unique(labs)))
fcl       <- factor(labs, levels = cl_levels)
out$cluster <- fcl

# cluster centroid positions + labels; index summ by cluster name, never
# by factor (whose integer codes would silently misalign the rows)
llm_lab <- summ[cl_levels, "llm"]
cit_lab <- summ[cl_levels, "cit"]
# lab carries the plot-level aesthetics (pc1, pc2, cluster) by name so the
# inherited aes resolves; the centroid coords are renamed from x/y.
lab <- data.frame(
  pc1 = tapply(X[, 1], fcl, mean),
  pc2 = tapply(X[, 2], fcl, mean),
  t = sprintf("%s\nLLM %s cit %s",
              cl_levels,
              ifelse(is.na(llm_lab), "NA", sprintf("%.1f", llm_lab)),
              ifelse(is.na(cit_lab), "NA", as.character(cit_lab))),
  cluster = factor(cl_levels, levels = cl_levels),
  stringsAsFactors = FALSE
)
p1 <- ggplot(out, aes(pc1, pc2, colour = cluster)) +
  geom_point(alpha = 0.8, size = 2) +
  geom_text(data = lab, aes(x = pc1, y = pc2, label = t),
            hjust = -0.15, size = 3) +
  labs(x = "PC1", y = "PC2",
       title = "Paper centroids in PC space, labelled LLM impact / citations") +
  theme_minimal() + whitebg() + theme(legend.position = "none")

p2 <- ggplot(out %>% filter(!is.na(llm_impact)),
             aes(cluster, llm_impact, fill = cluster)) +
  geom_col(alpha = 0.8) +
  labs(x = "cluster", y = "mean LLM impact",
       title = "Per-cluster impact, LLM") +
  theme_minimal() + whitebg() + theme(legend.position = "none")

p3 <- ggplot(out %>% filter(!is.na(citations)),
             aes(cluster, log10(pmax(citations, 0) + 1), fill = cluster)) +
  geom_col(alpha = 0.8) +
  labs(x = "cluster", y = "log10(median citations + 1)",
       title = "Per-cluster impact, human (citations)") +
  theme_minimal() + whitebg() + theme(legend.position = "none")

p4 <- ggplot(out %>% filter(!is.na(citations)),
             aes(llm_impact, log10(citations + 1), colour = cluster)) +
  geom_point(alpha = 0.8) +
  labs(x = "LLM impact (avg over prompts)", y = "log10(citations + 1)",
       title = "Per paper: LLM vs human impact, coloured by cluster") +
  theme_minimal() + whitebg()

p5 <- ggplot(data.frame(k = ks, sil = sil), aes(k, sil)) +
  geom_line() + geom_point() +
  geom_vline(aes(xintercept = bestk), linetype = "dashed",
             colour = "firebrick") +
  labs(x = "k", y = "avg silhouette", title = "Choosing k") +
  theme_minimal() + whitebg()

comb <- (p1 / (p2 | p3)) / (p4 | p5) +
  plot_layout(ncol = 2, heights = c(3, 2))
ggsave(out_png, comb, width = 12, height = 9, dpi = 150, bg = "white")
print(comb)
unlink("Rplots.pdf")
