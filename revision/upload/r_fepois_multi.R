#!/usr/bin/env Rscript
# r_fepois_multi.R: Poisson pseudo-maximum likelihood with any list of
# treatment terms and any list of fixed effects, fitted in R with fixest and
# called from Python. Every register estimate the paper reports (Table 1,
# Figures 2 and 3, Online Appendix Part III) is fitted through this file by
# mona_common.run_fepois_multi.
#
# WHAT IT FITS
#   n_emp ~ <terms> | <fixed effects>
# with standard errors clustered on --cluster (employer_id by default). For
# the headline the terms are the Riksbank, calendar-quarter, interim and
# adoption interactions with High x Young and the effects are employer by
# month, employer by age and month by age.
#
# USAGE
#   Rscript r_fepois_multi.R --input <in.csv[.gz]> --output <out.csv>
#       --terms "a,b,c" [--fe "f1,f2,f3"] [--cluster <col>]
#       [--nrows <n>] [--nthreads <k>]
#
# INPUT COLUMNS
#   n_emp, every column named in --terms, every column named in --fe (default
#   fe_emp_bin,fe_emp_t; passed as integer codes by the Python wrapper) and
#   the cluster column.
#
# OUTPUT
#   <out.csv>        term, coef, se, pvalue, n_obs, n_emp_total, converged,
#                    elapsed_s, status; one row per term, 'dropped' when a
#                    term is absorbed by the effects.
#   <out>_vcov.csv   the clustered covariance of the terms, so that a linear
#                    combination (a level, a difference between bands, a sum
#                    of quarters) gets a standard error from the same fit.
#                    The Python wrapper copies it to the calling script's
#                    output folder as vcov_<tag>.csv.
# A failure still writes the coefficient file with a status row.
#
# READER AND THREADS
#   data.table::fread when installed; otherwise read.csv with the row count
#   from --nrows and column classes inferred from a sample. The thread count
#   comes from --nthreads, then CANARIES_R_THREADS, then 8; the Python wrapper
#   lowers it and retries when R's allocator fails. Output is ASCII only.

args <- commandArgs(trailingOnly = TRUE)
parse_arg <- function(args, key, default = NA) {
    idx <- which(args == key)
    if (length(idx) == 0) return(default)
    if (idx + 1 > length(args)) stop(sprintf("Argument %s missing value", key))
    args[idx + 1]
}

input_path  <- parse_arg(args, "--input")
output_path <- parse_arg(args, "--output")
terms_raw   <- parse_arg(args, "--terms")
cluster_col <- parse_arg(args, "--cluster", default = "employer_id")
fe_raw      <- parse_arg(args, "--fe", default = "fe_emp_bin,fe_emp_t")

if (is.na(input_path) || is.na(output_path) || is.na(terms_raw)) {
    stop("Usage: Rscript r_fepois_multi.R --input <csv> --output <csv> --terms <a,b,c> [--cluster <col>] [--fe <f1,f2>]")
}

terms <- trimws(strsplit(terms_raw, ",")[[1]])
fes   <- trimws(strsplit(fe_raw, ",")[[1]])

write_failure <- function(msg, elapsed = 0) {
    df <- data.frame(term = terms, coef = NA_real_, se = NA_real_,
                     pvalue = NA_real_, n_obs = NA_integer_,
                     n_emp_total = NA_real_, converged = FALSE,
                     elapsed_s = elapsed, status = msg,
                     stringsAsFactors = FALSE)
    write.csv(df, output_path, row.names = FALSE)
    cat(sprintf("FAIL: %s\n", msg), file = stderr())
}

ok <- suppressWarnings(suppressMessages(
    requireNamespace("fixest", quietly = TRUE)))
if (!ok) { write_failure("fixest_not_available"); quit(status = 1) }
cat(sprintf("fixest version: %s\n", as.character(packageVersion("fixest"))))

if (!file.exists(input_path)) {
    write_failure(sprintf("input_missing: %s", input_path)); quit(status = 1)
}
# READING THE EXCHANGE FILE IS WHERE THIS KEPT DYING.
#
# base read.csv parses every field as character first and then converts,
# and it grows the frame by reallocation, so peak memory runs many times
# the size of the finished object. On 21 Sep 2026 a twenty-six-million-row
# input killed R with "*** recursive gc invocation", which is the
# allocator giving up during garbage collection, on a node with 733 GB
# free. Eight fits across this round died the same way and we blamed the
# panel size.
#
# Two fixes, in order of preference. data.table::fread reads a gzipped
# file directly, in parallel, at roughly the size of the result. If it is
# not installed, read.csv with explicit colClasses skips the character
# intermediate, which is most of the cost; every column we write is
# numeric by construction, since mona_common factorises the fixed effects
# to integer codes before writing.
nrows_arg <- {
    k <- which(args == "--nrows")
    if (length(k) && length(args) > k[1])
        as.integer(args[k[1] + 1L]) else -1L
}

read_exchange <- function(path, nrows = -1L) {
    if (requireNamespace("data.table", quietly = TRUE)) {
        cat("reader: data.table::fread\n")
        return(as.data.frame(data.table::fread(path, showProgress = FALSE)))
    }
    # data.table is NOT installed on MONA and cannot be installed, so
    # this is the path that actually runs. Three things make base R's
    # reader survive a thirty-million-row frame:
    #
    #   nrows       the documented cause of "*** recursive gc invocation"
    #               is that read.csv GROWS the frame by reallocation when
    #               it does not know the length. Python knows the row
    #               count exactly and passes it, so R allocates once.
    #   colClasses  skips the character-first pass. Inferred from a
    #               sample rather than assumed, because run_fepois and
    #               run_fepois_es hand over STRING fixed effects and
    #               forcing those to numeric returns NA coefficients,
    #               which the harness caught on 21 September.
    #   quote/comment  disabling both removes per-field scanning that
    #               cannot match anything in a file we wrote ourselves.
    cat("reader: read.csv, pre-allocated\n")
    if (nrows <= 0L) {
        cat("WARNING: row count unknown, so read.csv must grow the frame\n")
        cat("  by reallocation. That is what produced *** recursive gc\n")
        cat("  invocation on 21 September 2026.\n")
    }
    hdr <- read.csv(path, nrows = 1000, stringsAsFactors = FALSE,
                    quote = "", comment.char = "")
    cls <- vapply(hdr, function(x) if (is.numeric(x)) "numeric" else
                  "character", character(1))
    rm(hdr); gc()
    read.csv(path, stringsAsFactors = FALSE, colClasses = cls,
             nrows = nrows, quote = "", comment.char = "")
}
df <- tryCatch(read_exchange(input_path, nrows_arg),
               error = function(e) { write_failure(sprintf("read_csv_failed: %s",
                   conditionMessage(e))); quit(status = 1) })

required <- c("n_emp", terms, fes, cluster_col)
missing_cols <- setdiff(required, colnames(df))
if (length(missing_cols) > 0) {
    write_failure(sprintf("missing_columns: %s",
                          paste(missing_cols, collapse = ",")))
    quit(status = 1)
}

for (fe in fes) df[[fe]] <- as.factor(df[[fe]])
n_obs <- nrow(df); n_emp_total <- sum(df$n_emp, na.rm = TRUE)
cat(sprintf("rows: %d, sum(n_emp): %.0f, terms: %d\n",
            n_obs, n_emp_total, length(terms)))

formula_str <- paste0("n_emp ~ ", paste(terms, collapse = " + "),
                      " | ", paste(fes, collapse = " + "))
cluster_formula <- as.formula(paste("~", cluster_col))

# ---------------------------------------------------------------------
# THREADS. fixest defaults to every core on the machine, and each thread
# carries its own working buffers, so peak memory scales with the core
# count inside the job's own allocation. The symptom when it runs out is
# rc=3221225477 with "*** recursive gc invocation", R's collector
# failing.
#
# This is NOT contention between lanes. An earlier version of this note
# blamed three lanes running at once; 178 log lines across the whole
# revision report the node between 362 and 738 GB free, including every
# crash, so the machine was never short. What binds is the per-job cap
# and, more than threads, the NUMBER of fixed effects: on 21 September
# a 30.5M-row fit with three effects succeeded while a 28.5M-row fit
# with four died at two threads in the same job. Drop a nested,
# redundant effect before reaching for the thread count.
#
# A modest thread count costs wall-clock and buys the fit completing.
# Override with CANARIES_R_THREADS when a lane runs alone.
# ---------------------------------------------------------------------
# The env var cannot be set from inside the MONA batch submitter, so the
# thread count has to arrive on the command line or it is never honoured.
# Precedence: --nthreads, then CANARIES_R_THREADS, then 8.
.threads <- suppressWarnings(as.integer(parse_arg(args, "--nthreads",
                                                  default = NA_character_)))
if (is.na(.threads))
    .threads <- suppressWarnings(as.integer(
        Sys.getenv("CANARIES_R_THREADS", "8")))
if (is.na(.threads) || .threads < 1) .threads <- 8
if (requireNamespace("fixest", quietly = TRUE)) {
    try(fixest::setFixest_nthreads(.threads), silent = TRUE)
}
try(data.table::setDTthreads(.threads), silent = TRUE)
cat(sprintf("threads: %d\n", .threads))

t0 <- Sys.time()
fit <- tryCatch(
    fixest::fepois(as.formula(formula_str), data = df,
                   cluster = cluster_formula),
    error = function(e) {
        write_failure(sprintf("fit_error: %s", conditionMessage(e)),
                      as.numeric(difftime(Sys.time(), t0, units = "secs")))
        quit(status = 1)
    })
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

co <- summary(fit)$coeftable

# The clustered covariance of the treatment terms, written beside the
# coefficient table as <output>_vcov.csv. A linear combination of terms
# (a net level = step during tightening + step at adoption; a difference
# between two bands; a sum of quarter terms) then gets a standard error
# from the SAME fit instead of a second trip to the lab. Added 22 Sep 2026
# when the paper needed the level after adoption relative to the pre-hike
# months and nothing exported could give its standard error. vcov(fit)
# returns the covariance under the clustering the fit was given.
vc <- tryCatch(vcov(fit), error = function(e) NULL)
if (!is.null(vc)) {
    keep <- intersect(terms, rownames(vc))
    if (length(keep) > 0) {
        vdf <- as.data.frame(vc[keep, keep, drop = FALSE])
        colnames(vdf) <- keep
        vdf <- cbind(term = keep, vdf)
        write.csv(vdf, sub("\\.csv$", "_vcov.csv", output_path),
                  row.names = FALSE)
    }
}
out_rows <- lapply(terms, function(tm) {
    if (tm %in% rownames(co)) {
        data.frame(term = tm, coef = as.numeric(co[tm, "Estimate"]),
                   se = as.numeric(co[tm, "Std. Error"]),
                   pvalue = as.numeric(co[tm, "Pr(>|z|)"]),
                   n_obs = n_obs, n_emp_total = n_emp_total,
                   converged = isTRUE(fit$convStatus),
                   elapsed_s = elapsed, status = "ok",
                   stringsAsFactors = FALSE)
    } else {
        data.frame(term = tm, coef = NA_real_, se = NA_real_,
                   pvalue = NA_real_, n_obs = n_obs,
                   n_emp_total = n_emp_total,
                   converged = isTRUE(fit$convStatus),
                   elapsed_s = elapsed, status = "dropped",
                   stringsAsFactors = FALSE)
    }
})
write.csv(do.call(rbind, out_rows), output_path, row.names = FALSE)
cat(sprintf("OK: wrote %s (elapsed %.1fs)\n", output_path, elapsed))
