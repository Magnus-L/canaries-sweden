#!/usr/bin/env Rscript
# r_feols_es.R -- OLS+1 event study by half-year interactions.
#
# Sister to r_fepois_es.R. Same panel layout, same event-study form,
# but fits OLS on log(n_emp + 1) rather than Poisson on n_emp. The OLS+1
# spec is the workhorse in the main text, so this script generates the
# event-study coefficients in directly-comparable units to Figure 3.
#
# Usage:
#   Rscript r_feols_es.R --input <input.csv> --output <output.csv> \
#       [--cluster <colname>] [--ref <halfyear, e.g. 2022H1>]
#
# Required CSV columns (read from --input):
#   n_emp, high, halfyear, fe_emp_bin, fe_emp_t, employer_id

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
    stop("Usage: Rscript r_feols_es.R --input <csv> --output <csv> [--cluster <col>] [--ref <halfyear>]")
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

if (!ref_period %in% df$halfyear) {
    write_failure(output_path,
                  sprintf("ref_period_not_in_data: %s (have: %s)",
                          ref_period, paste(unique(df$halfyear), collapse = ",")))
    quit(status = 1)
}

# log(n_emp + 1) is the dependent var (workhorse spec)
df$ln_n_emp_p1 <- log(df$n_emp + 1)
df$fe_emp_bin <- as.factor(df$fe_emp_bin)
df$fe_emp_t   <- as.factor(df$fe_emp_t)
df$halfyear   <- as.factor(df$halfyear)
df$halfyear   <- relevel(df$halfyear, ref = ref_period)

formula_str <- sprintf(
    'ln_n_emp_p1 ~ i(halfyear, high, ref = "%s") | fe_emp_bin + fe_emp_t',
    ref_period
)
cluster_formula <- as.formula(paste("~", cluster_col))

t0  <- Sys.time()
fit <- tryCatch(
    fixest::feols(as.formula(formula_str), data = df, cluster = cluster_formula),
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
period_from_term <- function(t) {
    m <- regmatches(t, regexpr("halfyear::[^:]+", t))
    if (length(m) == 0) return(NA_character_)
    sub("halfyear::", "", m)
}
periods <- vapply(terms, period_from_term, character(1))

# feols coeftable has 'Estimate', 'Std. Error', 't value', 'Pr(>|t|)'
out <- data.frame(
    period      = periods,
    coef        = co[, "Estimate"],
    se          = co[, "Std. Error"],
    pvalue      = co[, "Pr(>|t|)"],
    n_obs       = n_obs,
    n_emp_total = n_emp_total,
    converged   = TRUE,
    elapsed_s   = elapsed,
    status      = "ok",
    stringsAsFactors = FALSE
)
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
