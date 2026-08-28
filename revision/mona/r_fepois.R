#!/usr/bin/env Rscript
# r_fepois.R -- Poisson PML wrapper for 32_mona_kauhanen_robustness.py
#
# Called via subprocess from Python. Replaces pyfixest::fepois with R's
# fixest::fepois (version 0.13.2 confirmed installed in MONA, 2026-04-28).
#
# Usage:
#   Rscript r_fepois.R --input <input.csv> --output <output.csv> \
#       [--weights <colname>] [--cluster <colname>]
#
# Required CSV columns (read from --input):
#   n_emp                -- dependent variable (count)
#   post_rb_x_high       -- treatment 1
#   post_gpt_x_high      -- treatment 2
#   fe_emp_bin           -- FE 1 (string concatenation employer x bin)
#   fe_emp_t             -- FE 2 (string concatenation employer x year-month)
#   employer_id          -- cluster variable (and used in cluster vcov)
#   <weights colname>    -- optional cell-level weight (if --weights passed)
#
# Output CSV columns (written to --output):
#   term, coef, se, pvalue, n_obs, n_emp_total, converged, elapsed_s, status
#
# Exit codes:
#   0  -- success (output CSV written)
#   1  -- fixest not available, missing input, or fit failure
#
# ASCII-only output. Errors are written to stderr; output CSV always
# written so Python can read a failure row instead of crashing.

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

df <- tryCatch(
    read.csv(input_path, stringsAsFactors = FALSE),
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
