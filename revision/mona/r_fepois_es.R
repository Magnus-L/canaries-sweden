#!/usr/bin/env Rscript
# r_fepois_es.R -- Poisson PML event study by half-year interactions.
#
# Sister script to r_fepois.R but fits the canonical event-study form
# rather than the pooled PostRB/PostGPT spec. Used by 36_eventstudy_*
# to generate per-half-year coefficients for EduQuartile (script 34
# cell panel) and PredQuartile (script 35 cell panel).
#
# Usage:
#   Rscript r_fepois_es.R --input <input.csv> --output <output.csv> \
#       [--cluster <colname>] [--ref <halfyear, e.g. 2022H1>]
#
# Required CSV columns (read from --input):
#   n_emp        -- dependent variable (count)
#   high         -- 0/1 treatment indicator (top-quartile = 1)
#   halfyear     -- half-year period string, e.g. "2022H1", "2022H2", "2023H1"
#   fe_emp_bin   -- FE 1 (employer x bin)
#   fe_emp_t     -- FE 2 (employer x year-month)
#   employer_id  -- cluster variable
#
# Output CSV columns (one row per non-reference half-year):
#   period, coef, se, pvalue, n_obs, n_emp_total, converged, elapsed_s, status

args <- commandArgs(trailingOnly = TRUE)

parse_arg <- function(args, key, default = NA) {
    idx <- which(args == key)
    if (length(idx) == 0) return(default)
    if (idx + 1 > length(args)) stop(sprintf("Argument %s missing value", key))
    return(args[idx + 1])
}

input_path  <- parse_arg(args, "--input")
output_path <- parse_arg(args, "--output")
cluster_col <- parse_arg(args, "--cluster", default = "employer_id")
ref_period  <- parse_arg(args, "--ref",     default = "2022H1")

if (is.na(input_path) || is.na(output_path)) {
    stop("Usage: Rscript r_fepois_es.R --input <csv> --output <csv> [--cluster <col>] [--ref <halfyear>]")
}

write_failure <- function(output_path, msg, elapsed = 0) {
    df <- data.frame(
        period      = NA_character_,
        coef        = NA_real_,
        se          = NA_real_,
        pvalue      = NA_real_,
        n_obs       = NA_integer_,
        n_emp_total = NA_real_,
        converged   = FALSE,
        elapsed_s   = elapsed,
        status      = msg,
        stringsAsFactors = FALSE
    )
    write.csv(df, output_path, row.names = FALSE)
    cat(sprintf("FAIL: %s\n", msg), file = stderr())
}

ok <- suppressWarnings(suppressMessages(requireNamespace("fixest", quietly = TRUE)))
if (!ok) {
    write_failure(output_path, "fixest_not_available")
    quit(status = 1)
}
cat(sprintf("fixest version: %s\n", as.character(packageVersion("fixest"))))

if (!file.exists(input_path)) {
    write_failure(output_path, sprintf("input_missing: %s", input_path))
    quit(status = 1)
}

# ---------------------------------------------------------------------
# THREADS. fixest defaults to every core, and each thread carries its own
# working buffers, so peak memory scales with the core count inside the
# job's own allocation. Symptom when it runs out: rc=3221225477 and
# "*** recursive gc invocation", R's collector failing.
#
# This is NOT contention between lanes. An earlier version of this note
# blamed three lanes at once; 178 log lines across the revision report
# the node between 362 and 738 GB free, including every crash. What
# binds is the per-job cap and, more than threads, the NUMBER of fixed
# effects: on 21 September a 30.5M-row fit with three effects succeeded
# while a 28.5M-row fit with four died at two threads in the same job.
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

required <- c("n_emp", "high", "halfyear", "fe_emp_bin", "fe_emp_t", cluster_col)
missing  <- setdiff(required, colnames(df))
if (length(missing) > 0) {
    write_failure(output_path, sprintf("missing_columns: %s", paste(missing, collapse = ",")))
    quit(status = 1)
}

n_obs       <- nrow(df)
n_emp_total <- sum(df$n_emp, na.rm = TRUE)
cat(sprintf("rows: %d, sum(n_emp): %.0f, ref_period: %s\n",
            n_obs, n_emp_total, ref_period))

if (!ref_period %in% df$halfyear) {
    write_failure(output_path,
                  sprintf("ref_period_not_in_data: %s (have: %s)",
                          ref_period, paste(unique(df$halfyear), collapse = ",")))
    quit(status = 1)
}

df$fe_emp_bin <- as.factor(df$fe_emp_bin)
df$fe_emp_t   <- as.factor(df$fe_emp_t)
df$halfyear   <- as.factor(df$halfyear)
df$halfyear   <- relevel(df$halfyear, ref = ref_period)

# Canonical event-study form using fixest::i() interaction:
#   n_emp ~ i(halfyear, high, ref = ref_period) | fe_emp_bin + fe_emp_t
formula_str <- sprintf(
    'n_emp ~ i(halfyear, high, ref = "%s") | fe_emp_bin + fe_emp_t',
    ref_period
)
cluster_formula <- as.formula(paste("~", cluster_col))

t0  <- Sys.time()
fit <- tryCatch(
    fixest::fepois(as.formula(formula_str), data = df, cluster = cluster_formula),
    error = function(e) {
        elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
        write_failure(output_path,
                      sprintf("fit_error: %s", conditionMessage(e)),
                      elapsed = elapsed)
        quit(status = 1)
    }
)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

co <- summary(fit)$coeftable
terms <- rownames(co)
# Term names look like: 'halfyear::2022H2:high', 'halfyear::2023H1:high', etc.
# Extract the period substring.
period_from_term <- function(t) {
    m <- regmatches(t, regexpr("halfyear::[^:]+", t))
    if (length(m) == 0) return(NA_character_)
    sub("halfyear::", "", m)
}
periods <- vapply(terms, period_from_term, character(1))

out <- data.frame(
    period      = periods,
    coef        = co[, "Estimate"],
    se          = co[, "Std. Error"],
    pvalue      = co[, "Pr(>|z|)"],
    n_obs       = n_obs,
    n_emp_total = n_emp_total,
    converged   = TRUE,
    elapsed_s   = elapsed,
    status      = "ok",
    stringsAsFactors = FALSE
)
# Append the reference period (zero by construction) for plotting
ref_row <- data.frame(
    period      = ref_period,
    coef        = 0.0,
    se          = 0.0,
    pvalue      = NA_real_,
    n_obs       = n_obs,
    n_emp_total = n_emp_total,
    converged   = TRUE,
    elapsed_s   = elapsed,
    status      = "reference",
    stringsAsFactors = FALSE
)
out <- rbind(out, ref_row)
out <- out[!is.na(out$period), ]
out <- out[order(out$period), ]

write.csv(out, output_path, row.names = FALSE)
cat(sprintf("OK: %d coefficients written, elapsed %.1fs\n", nrow(out), elapsed))
