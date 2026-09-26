#!/usr/bin/env Rscript
# r_fepois.R: Poisson pseudo-maximum likelihood with the two treatment terms
# of the submitted design, fitted in R with fixest and called from Python.
#
# WHAT IT FITS
#   n_emp ~ post_rb_x_high + post_gpt_x_high | fe_emp_bin + fe_emp_t
# on the exchange file mona_common.run_fepois writes, with standard errors
# clustered on the column named by --cluster (employer_id by default) and an
# optional cell weight. This is the pooled specification of the withdrawn
# occupation design and of the as-of backtest (script 45), the education
# design comparison (script 47h) and the teleworkability check (script 46).
# Every estimate the paper reports is fitted through r_fepois_multi.R.
#
# USAGE
#   Rscript r_fepois.R --input <in.csv[.gz]> --output <out.csv>
#       [--weights <col>] [--cluster <col>] [--nrows <n>] [--nthreads <k>]
#
# INPUT COLUMNS
#   n_emp             the count
#   post_rb_x_high    one from April 2022 in the top exposure quartile
#   post_gpt_x_high   one from December 2022 in the top exposure quartile
#   fe_emp_bin        employer by quartile key (integer code or string)
#   fe_emp_t          employer by month key
#   <cluster>         the cluster column
#
# OUTPUT COLUMNS
#   term, coef, se, pvalue, n_obs (input rows), n_obs_fit (rows the fit
#   used), n_emp_total, converged, elapsed_s,
#   status ('ok', 'dropped' for a term absorbed by the effects, or the
#   failure reason). A failure still writes the file, so the calling script
#   reads a status row rather than crashing.
#
# READER AND THREADS
#   data.table::fread is used when installed; otherwise read.csv with the row
#   count from --nrows and column classes inferred from a sample, which reads
#   a thirty-million-row frame at roughly the size of the result. fixest's
#   thread count comes from --nthreads, then CANARIES_R_THREADS, then 8; the
#   Python wrapper lowers it and retries when R's allocator fails.
#
# Exit code 0 on success; 1 when fixest is missing, the input is missing or
# the fit fails. Output is ASCII only.

# ----------------------------------------------------------------------
# Argument parsing (base R, no external deps)
# ----------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)

parse_arg <- function(args, key, default = NA) {
    idx <- which(args == key)
    if (length(idx) == 0) return(default)
    if (idx + 1 > length(args)) {
        stop(sprintf("Argument %s missing value", key))
    }
    return(args[idx + 1])
}

input_path  <- parse_arg(args, "--input")
output_path <- parse_arg(args, "--output")
weight_col  <- parse_arg(args, "--weights", default = NA)
cluster_col <- parse_arg(args, "--cluster", default = "employer_id")

if (is.na(input_path) || is.na(output_path)) {
    stop("Usage: Rscript r_fepois.R --input <csv> --output <csv> [--weights <col>] [--cluster <col>]")
}

# ----------------------------------------------------------------------
# Failure-result writer (always produces an output file)
# ----------------------------------------------------------------------

write_failure <- function(output_path, msg, elapsed = 0) {
    df <- data.frame(
        term       = c("post_rb_x_high", "post_gpt_x_high"),
        coef       = c(NA_real_, NA_real_),
        se         = c(NA_real_, NA_real_),
        pvalue     = c(NA_real_, NA_real_),
        n_obs      = c(NA_integer_, NA_integer_),
        n_obs_fit  = c(NA_integer_, NA_integer_),
        n_emp_total = c(NA_real_, NA_real_),
        converged  = c(FALSE, FALSE),
        elapsed_s  = c(elapsed, elapsed),
        status     = c(msg, msg),
        stringsAsFactors = FALSE
    )
    write.csv(df, output_path, row.names = FALSE)
    cat(sprintf("FAIL: %s\n", msg), file = stderr())
}

# ----------------------------------------------------------------------
# Load fixest
# ----------------------------------------------------------------------

ok <- suppressWarnings(suppressMessages(
    requireNamespace("fixest", quietly = TRUE)
))
if (!ok) {
    write_failure(output_path, "fixest_not_available")
    quit(status = 1)
}

cat(sprintf("fixest version: %s\n", as.character(packageVersion("fixest"))))

# ----------------------------------------------------------------------
# Load input
# ----------------------------------------------------------------------

if (!file.exists(input_path)) {
    write_failure(output_path, sprintf("input_missing: %s", input_path))
    quit(status = 1)
}

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

df <- tryCatch(
    read_exchange(input_path, nrows_arg),
    error = function(e) {
        write_failure(output_path, sprintf("read_csv_failed: %s", conditionMessage(e)))
        quit(status = 1)
    }
)

required_cols <- c(
    "n_emp", "post_rb_x_high", "post_gpt_x_high",
    "fe_emp_bin", "fe_emp_t", cluster_col
)
if (!is.na(weight_col)) required_cols <- c(required_cols, weight_col)

missing_cols <- setdiff(required_cols, colnames(df))
if (length(missing_cols) > 0) {
    write_failure(
        output_path,
        sprintf("missing_columns: %s", paste(missing_cols, collapse = ","))
    )
    quit(status = 1)
}

n_obs <- nrow(df)
n_emp_total <- sum(df$n_emp, na.rm = TRUE)
cat(sprintf("rows: %d, sum(n_emp): %.0f\n", n_obs, n_emp_total))

# ----------------------------------------------------------------------
# Fit fepois
# ----------------------------------------------------------------------

# fixest needs FE columns as factors (or strings, which it converts).
# Cluster variable should be a vector, passed via cluster = ~employer_id.
df$fe_emp_bin <- as.factor(df$fe_emp_bin)
df$fe_emp_t   <- as.factor(df$fe_emp_t)

formula_str <- "n_emp ~ post_rb_x_high + post_gpt_x_high | fe_emp_bin + fe_emp_t"
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
    {
        if (is.na(weight_col)) {
            fixest::fepois(
                as.formula(formula_str),
                data    = df,
                cluster = cluster_formula
            )
        } else {
            weights_formula <- as.formula(paste("~", weight_col))
            fixest::fepois(
                as.formula(formula_str),
                data    = df,
                cluster = cluster_formula,
                weights = weights_formula
            )
        }
    },
    error = function(e) {
        elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
        write_failure(
            output_path,
            sprintf("fit_error: %s", conditionMessage(e)),
            elapsed = elapsed
        )
        quit(status = 1)
    }
)

elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

# THE OBSERVATIONS THE FIT ACTUALLY USED. n_obs is the input row count,
# taken before the fit; fixest then drops the observations of any fixed-
# effect group whose outcome is zero throughout (and singletons). Until
# 26 Sep 2026 every caller reported n_obs as "cells used", so 98's
# "dropped by PPML" read 0 in every row (the ChatGPT review's finding).
# n_obs_fit is nobs(fit), the count after those removals.
n_obs_fit <- tryCatch(as.integer(nobs(fit)), error = function(e) NA_integer_)
cat(sprintf("rows used by the fit: %s of %d\n",
            ifelse(is.na(n_obs_fit), "NA", format(n_obs_fit)), n_obs))

# ----------------------------------------------------------------------
# Extract coefficients
# ----------------------------------------------------------------------

# fixest uses summary() to get SE/p with the cluster-vcov. coef() returns
# point estimates; summary()$coeftable gives a matrix with cols
# Estimate, Std. Error, z value, Pr(>|z|).
co <- summary(fit)$coeftable

terms_wanted <- c("post_rb_x_high", "post_gpt_x_high")
out_rows <- list()

for (tm in terms_wanted) {
    if (tm %in% rownames(co)) {
        out_rows[[length(out_rows) + 1]] <- data.frame(
            term       = tm,
            coef       = as.numeric(co[tm, "Estimate"]),
            se         = as.numeric(co[tm, "Std. Error"]),
            pvalue     = as.numeric(co[tm, "Pr(>|z|)"]),
            n_obs      = n_obs,
            n_obs_fit  = n_obs_fit,
            n_emp_total = n_emp_total,
            converged  = isTRUE(fit$convStatus),
            elapsed_s  = elapsed,
            status     = "ok",
            stringsAsFactors = FALSE
        )
    } else {
        # Coefficient absorbed by FEs or otherwise dropped
        out_rows[[length(out_rows) + 1]] <- data.frame(
            term       = tm,
            coef       = NA_real_,
            se         = NA_real_,
            pvalue     = NA_real_,
            n_obs      = n_obs,
            n_obs_fit  = n_obs_fit,
            n_emp_total = n_emp_total,
            converged  = isTRUE(fit$convStatus),
            elapsed_s  = elapsed,
            status     = "dropped",
            stringsAsFactors = FALSE
        )
    }
}

out <- do.call(rbind, out_rows)
write.csv(out, output_path, row.names = FALSE)

cat(sprintf("OK: wrote %s (elapsed %.1fs)\n", output_path, elapsed))
