# Shared helper for the *-vs-overlap plots: mean overlap per *pair of
# papers*, computed from the raw embeddings with each vector mean-centered
# (per-feature grand mean removed) before L2 normalization. Centering
# drops the common direction all text embeddings share, so the overlaps
# are more discriminative than the raw-cosine mean_cosine in the report.
# Needs only jsonlite.

centered_paper_overlaps <- function(emb_path) {
  # embeddings.jsonl is JSON *lines*; parse each line (one object, so the
  # single-document parser is happy) and pull out just the fields needed
  # (a full-frame rbind mangles column structure on single-row frames)
  recs  <- lapply(readLines(emb_path, warn = FALSE),
                  function(l) fromJSON(l, simplifyVector = TRUE))
  # single-object fields come back as scalars/vectors:
  paper <- vapply(recs, function(r) r$paper, character(1))
  V     <- do.call(rbind, lapply(recs, function(r) r$vector))

  # mean-center (subtract the grand centroid vector — the component all
  # text embeddings share), then L2-normalize each *row*; the row dot
  # product is a genuine cosine in [-1,1], just on centered vectors
  C  <- t(t(V) - colMeans(V))
  nn <- sqrt(rowSums(C^2))
  S  <- tcrossprod(C / pmax(nn, 1e-12))

  unit_idx <- split(seq_len(nrow(V)), paper)
  papers   <- names(unit_idx)
  pairs    <- combn(papers, 2)   # 2 x (n choose 2) matrix
  means    <- apply(pairs, 2, function(p) {
    ia <- as.integer(unit_idx[[p[1]]])
    ib <- as.integer(unit_idx[[p[2]]])
    mean(S[ia, ib])
  })
  data.frame(
    paper_a      = pairs[1, ],
    paper_b      = pairs[2, ],
    mean_overlap = means,
    stringsAsFactors = FALSE
  )
}
