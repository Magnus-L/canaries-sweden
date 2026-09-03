#!/usr/bin/env Rscript
# r_fepois_multi.R -- Poisson PML with an ARBITRARY list of treatment terms.
#
# Generalisation of r_fepois.R (which hard-codes the two-term PostRB/PostGPT
# formula). Needed by:
#   44_decile_gradient.py  -- nine decile x post interactions in ONE model
#   46_wfh_horserace.py    -- DAIOE terms + WFH x period terms jointly
#
# Usage:
#   Rscript r_fepois_multi.R --input in.csv --output out.csv \
#       --terms "rb_d2,gpt_d2,rb_d3" [--cluster employer_id] [--fe "fe_emp_bin,fe_emp_t"]
#
# Input CSV must contain: n_emp, every column named in --terms, every FE
# column named in --fe (default fe_emp_bin,fe_emp_t), and the cluster column.
# Output CSV: term, coef, se, pvalue, n_obs, n_emp_total, converged,
#             elapsed_s, status   (one row per term; 'dropped' if absorbed)
# ASCII-only output; a failure still writes an output CSV so Python can
# read a status row instead of crashing.

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
df <- tryCatch(read.csv(input_path, stringsAsFactors = FALSE),
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
