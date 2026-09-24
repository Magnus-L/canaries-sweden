# l47_posting_honestdid_v3.R: the exact Rambachan and Roth (2023)
# relative-magnitudes bounds for the posting event study of Online
# Appendix II.2, computed with the HonestDiD package.
#
# WHY THIS EXISTS
# Panel (b) of II.2 was drawn from a simplified interval written in
# src/07_robustness.py, theta +/- (1.96 SE + Mbar x Dmax). That fixes the
# bias bound at Mbar x Dmax whatever the horizon, whereas the relative-
# magnitudes set bounds each post-period FIRST DIFFERENCE by Mbar x Dmax,
# so the admissible bias of an average over many post months accumulates.
# The simplified interval is therefore narrower than the real one, not
# wider. This script replaces it with the HonestDiD computation.
#
# THE REFERENCE PERIOD
# HonestDiD places the normalised zero between the last pre-period and the
# first post-period. The event study of l46 is normalised to February
# 2020, far from either event, so the coefficients are re-based to the
# month before the post window: b_t - b_ref, with February 2020 entering
# as a pre-period at -b_ref. This is a linear map of the same estimates
# (the covariance is A V A'), equivalent to omitting the reference month
# in the regression instead of February 2020.
#
#   Variant A (reported): reference November 2022, pre-periods January
#     2020 to October 2022 (34 months), post December 2022 to June 2026
#     (43 months), target the average of the 43 post months.
#   Variant B (robustness): reference March 2022, pre-periods January 2020
#     to February 2022 (26 months), post April 2022 to June 2026 (51),
#     target the average of December 2022 to June 2026 only; April to
#     November 2022 carry zero weight but stay in the post window, so the
#     restriction still chains through them.
#
# INPUT   revision/tables/posting_es_monthly_v3.csv, posting_es_vcov_v3.csv (l46)
# OUTPUT  revision/tables/posting_rr_honestdid_v3.csv
# RUN     Rscript revision/local/l47_posting_honestdid_v3.R  (after l46; then l47b draws the figure)
# NEEDS   HonestDiD 0.2.8 (CRAN). On this Mac the install needed
#         CPLUS_INCLUDE_PATH=<CLT SDK>/usr/include/c++/v1:/opt/homebrew/include,
#         C_INCLUDE_PATH and LIBRARY_PATH to Homebrew, and
#         --with-gmp-include/--with-gmp-lib for gmp.
#
# RUN TIME AND THE GRID
# One Mbar costs about 16 CPU minutes here (C-LF test, 34 or 26 possible
# locations of the largest pre-period difference, one linear program per
# location and grid point), so the Mbar values run in parallel with
# parallel::mclapply, one value per call. The test grid for theta is
# fixed at [-2, 2] in 1,001 points, so it contains zero exactly and
# "includes zero" is a direct test of theta = 0, not an interpolation.
# Stage 1 is a coarse Mbar grid; the first probe (Mbar = 0.1 already gives
# [-0.99, 0.66]) showed the breakdown lies well below 0.1, so the coarse
# grid is dense there. Stage 2 refines in steps of 0.001 between the last
# Mbar that excludes zero and the first that includes it. The breakdown
# reported is the smallest grid value whose interval includes zero.
# Variant B's coarse grid stops at 0.02: its breakdown lies below 0.005,
# and one Mbar there took over 75 minutes at 0.015 to 0.02, so the larger
# values would add hours and nothing to the robustness check.
#
# CACHE. Each finished (variant, Mbar) interval is appended to
# posting_rr_honestdid_v3_cache.csv and is not recomputed on a re-run.
# Delete the cache whenever l46's estimates change. The first run of
# 24 September 2026 was stopped after stage 1; its intervals were seeded
# into the cache from its log, whose four decimals are exact because the
# test grid moves in steps of 0.004.

suppressPackageStartupMessages(library(HonestDiD))

args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
tab <- file.path(dirname(here), "tables")

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

logf("L47: HonestDiD relative magnitudes, posting event study (%d cores)", CORES)
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
