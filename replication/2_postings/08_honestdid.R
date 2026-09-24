# 08_honestdid.R: the exact Rambachan and Roth (2023) relative-magnitudes
# bounds for the posting event study of Online Appendix II.2, computed with
# the HonestDiD package.
#
# WHAT IT COMPUTES
# The relative-magnitudes restriction bounds each post-period first
# difference of the differential trend by Mbar times the largest pre-period
# first difference, so the admissible bias of an average over many post
# months accumulates month by month. The robust 95 per cent confidence
# interval for the average post-launch coefficient is computed on a grid of
# Mbar, and the breakdown value is the smallest Mbar whose interval contains
# zero.
#
# THE REFERENCE PERIOD
# HonestDiD places the normalised zero between the last pre-period and the
# first post-period. The event study of 07 is normalised to February 2020,
# so the coefficients are re-based to the month before the post window:
# b_t - b_ref, with February 2020 entering as a pre-period at -b_ref. This is
# a linear map of the same estimates (the covariance is A V A').
#   Variant A (reported): reference November 2022, pre-periods January 2020
#     to October 2022 (34 months), post December 2022 to June 2026 (43
#     months), target the average of the 43 post months.
#   Variant B (robustness): reference March 2022, pre-periods January 2020
#     to February 2022 (26 months), post April 2022 to June 2026 (51 months),
#     target the average of December 2022 to June 2026 only.
#
# THE GRID
# The test grid for theta is [-2, 2] in 1,001 points, so it contains zero
# exactly and "includes zero" is a direct test of theta = 0. A coarse Mbar
# grid comes first; the interval between the last Mbar that excludes zero
# and the first that includes it is then refined in steps of 0.001. Variant
# B's coarse grid stops at 0.02, since its breakdown lies below 0.005.
#
# RUN TIME AND THE CACHE
# One Mbar takes about 16 CPU minutes (a linear program per possible
# location of the largest pre-period difference and per grid point), so the
# Mbar values run in parallel with parallel::mclapply; a cold run takes
# several hours on eight cores. Each finished (variant, Mbar) interval is
# appended to posting_rr_honestdid_v3_cache.csv and is not recomputed on a
# re-run; delete the cache whenever 07's estimates change.
#
# INPUT   output/results/posting_es_monthly_v3.csv, posting_es_vcov_v3.csv (07)
# OUTPUT  output/results/posting_rr_honestdid_v3.csv, posting_rr_honestdid_v3.log
# RUN     Rscript 2_postings/08_honestdid.R   (then 09 draws the figure)
# NEEDS   R (4.6.0 was used) with HonestDiD 0.2.8 from CRAN, which needs the gmp library.
# SERVES  Online Appendix II.2 (breakdown Mbar = 0.015; the interval
#         [-0.284, -0.085]) and Figure A2, panel (b)

suppressPackageStartupMessages(library(HonestDiD))

args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
tab <- file.path(dirname(here), "output", "results")

es <- read.csv(file.path(tab, "posting_es_monthly_v3.csv"), stringsAsFactors = FALSE)
V <- as.matrix(read.csv(file.path(tab, "posting_es_vcov_v3.csv"),
                        row.names = 1, check.names = FALSE))
est <- es[es$year_month != "2020-02", ]
stopifnot(identical(est$year_month, rownames(V)), identical(rownames(V), colnames(V)))
b <- setNames(est$coef, est$year_month)
months <- es$year_month                 # all 78 months, Feb 2020 included
stopifnot(length(months) == 78, length(b) == 77)

GPT <- "2022-12"
GRID_LB <- -2; GRID_UB <- 2; GRID_N <- 1001
STAGE1 <- list(
  A_pre_launch = c(0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04,
                   0.05, 0.075, 0.1, 0.15, 0.2),
  B_pre_ratehike = c(0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02))
CACHE <- file.path(tab, "posting_rr_honestdid_v3_cache.csv")
if (!file.exists(CACHE))
  write.csv(data.frame(variant = character(), Mbar = numeric(), lb = numeric(),
                       ub = numeric(), method = character()), CACHE, row.names = FALSE)
CORES <- max(1, parallel::detectCores() - 1)
LOG <- file.path(tab, "posting_rr_honestdid_v3.log")
cat("", file = LOG)
logf <- function(...) {
  msg <- sprintf(...)
  cat(msg, "\n", sep = "")
  cat(msg, "\n", sep = "", file = LOG, append = TRUE)
}

rebase <- function(ref) {
  # Map b (77, excl. Feb 2020) to c (77, excl. ref): c_t = b_t - b_ref,
  # with b_Feb2020 = 0.
  keep <- months[months != ref]
  A <- matrix(0, length(keep), length(b), dimnames = list(keep, names(b)))
  for (m in keep) {
    if (m != "2020-02") A[m, m] <- 1
    A[m, ref] <- A[m, ref] - 1
  }
  list(beta = as.vector(A %*% b), sigma = A %*% V %*% t(A), months = keep)
}

setup <- function(variant, ref) {
  r <- rebase(ref)
  pre <- r$months < ref
  post <- r$months > ref
  stopifnot(all(pre | post))
  w <- as.numeric(r$months[post] >= GPT)
  l_vec <- matrix(w / sum(w), ncol = 1)
  c(r, list(variant = variant, ref = ref, npre = sum(pre), npost = sum(post),
            nw = sum(w), l_vec = l_vec,
            theta = sum(l_vec * r$beta[post]),
            se = sqrt(as.numeric(t(l_vec) %*% r$sigma[post, post] %*% l_vec))))
}

one <- function(s, mbar) {
  t0 <- Sys.time()
  x <- createSensitivityResults_relativeMagnitudes(
    betahat = s$beta, sigma = s$sigma, numPrePeriods = s$npre,
    numPostPeriods = s$npost, l_vec = s$l_vec, Mbarvec = mbar, alpha = 0.05,
    gridPoints = GRID_N, grid.lb = GRID_LB, grid.ub = GRID_UB)
  x <- as.data.frame(x)
  cat(sprintf("    %s Mbar %.4f: [%.4f, %.4f] (%.1f min)\n", s$variant, mbar,
              x$lb, x$ub, as.numeric(difftime(Sys.time(), t0, units = "mins"))),
      file = LOG, append = TRUE)
  row <- data.frame(variant = s$variant, Mbar = mbar, lb = x$lb, ub = x$ub,
                    method = as.character(x$method))
  write.table(row, CACHE, append = TRUE, sep = ",", col.names = FALSE,
              row.names = FALSE)
  row
}

runjobs <- function(jobs) {
  cached <- read.csv(CACHE, stringsAsFactors = FALSE)
  hit <- vapply(jobs, function(j) any(cached$variant == j$v &
                                      abs(cached$Mbar - j$m) < 1e-9), logical(1))
  logf("  %d jobs, %d from the cache", length(jobs), sum(hit))
  old <- do.call(rbind, lapply(jobs[hit], function(j)
    cached[cached$variant == j$v & abs(cached$Mbar - j$m) < 1e-9, ][1, ]))
  jobs <- jobs[!hit]
  if (!length(jobs)) return(old)
  res <- parallel::mclapply(jobs, function(j) one(S[[j$v]], j$m),
                            mc.cores = CORES, mc.preschedule = FALSE)
  bad <- vapply(res, function(z) !is.data.frame(z), logical(1))
  if (any(bad)) stop("a HonestDiD job failed: ", paste(res[bad], collapse = "; "))
  rbind(old, do.call(rbind, res))
}

logf("HonestDiD relative magnitudes, posting event study (%d cores)", CORES)
S <- list(A_pre_launch = setup("A_pre_launch", "2022-11"),
          B_pre_ratehike = setup("B_pre_ratehike", "2022-03"))
for (s in S) logf("  %s: ref %s, %d pre, %d post (%d weighted), theta %.4f (SE %.4f)",
                  s$variant, s$ref, s$npre, s$npost, s$nw, s$theta, s$se)

jobs <- unlist(lapply(names(S), function(v)
  lapply(STAGE1[[v]], function(m) list(v = v, m = m))), recursive = FALSE)
sens <- runjobs(jobs)

covers <- function(d) d$lb <= 0 & d$ub >= 0
jobs2 <- list()
for (v in names(S)) {
  d <- sens[sens$variant == v, ]
  d <- d[order(d$Mbar), ]
  inc <- which(covers(d))
  if (length(inc) == 0) stop(v, ": no stage-1 Mbar includes zero; widen STAGE1")
  if (inc[1] == 1) next                     # zero inside already at Mbar = 0
  lo <- d$Mbar[inc[1] - 1]; hi <- d$Mbar[inc[1]]
  fine <- setdiff(round(seq(lo, hi, by = 0.001), 4), c(lo, hi))
  jobs2 <- c(jobs2, lapply(fine, function(m) list(v = v, m = m)))
}
if (length(jobs2)) sens <- rbind(sens, runjobs(jobs2))
sens <- sens[order(sens$variant, sens$Mbar), ]

out <- NULL
for (v in names(S)) {
  s <- S[[v]]
  d <- sens[sens$variant == v, ]
  bd <- min(d$Mbar[covers(d)])
  orig <- constructOriginalCS(betahat = s$beta, sigma = s$sigma,
                              numPrePeriods = s$npre, numPostPeriods = s$npost,
                              l_vec = s$l_vec, alpha = 0.05)
  logf("  %s: original CI [%.4f, %.4f]; robust CI at Mbar 0 [%.4f, %.4f]; breakdown Mbar %.4f",
       v, orig$lb, orig$ub, d$lb[d$Mbar == 0], d$ub[d$Mbar == 0], bd)
  meta <- data.frame(variant = v, reference = s$ref, n_pre = s$npre,
                     n_post = s$npost, n_post_weighted = s$nw,
                     theta_hat = s$theta, se_theta = s$se)
  out <- rbind(out,
    cbind(meta, Mbar = NA, lb = orig$lb, ub = orig$ub, method = "original",
          includes_zero = orig$lb <= 0 & orig$ub >= 0, breakdown_mbar = bd),
    cbind(meta, Mbar = d$Mbar, lb = d$lb, ub = d$ub, method = d$method,
          includes_zero = covers(d), breakdown_mbar = bd))
}
out$at_grid_edge <- !is.na(out$Mbar) & (out$lb <= GRID_LB + 1e-9 | out$ub >= GRID_UB - 1e-9)
write.csv(out, file.path(tab, "posting_rr_honestdid_v3.csv"), row.names = FALSE)
logf("  wrote posting_rr_honestdid_v3.csv")
