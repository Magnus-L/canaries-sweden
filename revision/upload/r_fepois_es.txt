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

df <- tryCatch(
    read.csv(input_path, stringsAsFactors = FALSE),
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
