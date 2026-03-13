#!/usr/bin/env Rscript
# ======================================================================
#  23_mona_rambachan_roth.R
#
#  Rambachan & Roth (2023, RES) sensitivity analysis for the employment
#  event study from script 18 (corrected FE specification).
#
#  THIS SCRIPT RUNS IN SCB's MONA SECURE ENVIRONMENT.
#  It reads the event study CSV output (no register data access needed).
#
#  WHAT IT DOES:
#    1. Reads corrected_es_all_ref2022H1.csv from output_18/
#    2. For each age group, constructs the coefficient vector and
#       (diagonal) variance-covariance matrix from point estimates + SEs
#    3. Runs HonestDiD::createSensitivityResults_relativeMagnitudes()
#       to compute honest confidence intervals under relative magnitudes
#       restrictions on post-treatment trend violations
#    4. Reports the breakdown value M_bar (largest M for which the
#       honest CI excludes zero) and produces sensitivity plots
#
#  INPUT:
#    output_18/corrected_es_all_ref2022H1.csv
#      Columns: age_group, period, coef, se, pval
#      Periods: 2019H1, 2019H2, ..., 2025H1 (half-year labels)
#      Reference period: 2022H1 (coef=0, se=0, omitted category)
#
#  OUTPUT (in output_23/):
#    rambachan_roth_results.csv    -- breakdown values + honest CIs
#    rambachan_roth_<age>.png      -- sensitivity plot per age group
#    rambachan_roth_summary.txt    -- human-readable summary
#
#  METHODOLOGY:
#    We use the RELATIVE MAGNITUDES restriction (Rambachan & Roth 2023,
#    Section 3.2). This bounds the magnitude of post-treatment trend
#    violations relative to the maximum pre-treatment violation:
#
#      max_{t>=T*} |delta_t| <= M_bar * max_{t<T*} |delta_t|
#
#    where delta_t are the violations of parallel trends. The breakdown
#    value M_bar is the largest value for which the honest CI still
#    excludes zero -- i.e., how many times larger than the worst
#    pre-treatment trend violation the post-treatment violations would
#    need to be before we lose significance.
#
#    A breakdown value of, say, M_bar = 2 means the result survives
#    even if post-treatment trend violations are twice as large as
#    the worst pre-period violation. Values above 1 are reassuring.
#
#  COVARIANCE MATRIX:
#    The CSV provides only point estimates and SEs (no covariances).
#    We assume a diagonal covariance matrix: Var(beta_t) = SE_t^2,
#    Cov(beta_s, beta_t) = 0 for s != t. This is CONSERVATIVE --
#    in the actual regression, adjacent period coefficients are
#    positively correlated, so the true identified set is smaller
#    than what we compute here. Our honest CIs are therefore wider
#    (more conservative) than the exact computation.
#
#  REFERENCE:
#    Rambachan, A. and Roth, J. (2023). "A More Credible Approach to
#    Parallel Trends." Review of Economic Studies, 90(5), 2555-2591.
#
#  AUTHOR: Magnus Lodefalk / AI-Econ Lab
#  DATE:   March 2026
# ======================================================================


# ======================================================================
#  STEP 0: Install and load packages
# ======================================================================

# HonestDiD is on CRAN. If not yet installed on MONA, install it.
# If CRAN is blocked on MONA, see the fallback instructions at the
# bottom of this script for manual installation.

if (!requireNamespace("HonestDiD", quietly = TRUE)) {
  message("HonestDiD not found. Attempting CRAN install...")
  tryCatch(
    install.packages("HonestDiD", repos = "https://cran.r-project.org"),
    error = function(e) {
      message("CRAN install failed. Trying GitHub install via remotes...")
      if (!requireNamespace("remotes", quietly = TRUE)) {
        install.packages("remotes", repos = "https://cran.r-project.org")
      }
      remotes::install_github("asheshrambachan/HonestDiD")
    }
  )
}

# Load required packages
library(HonestDiD)

# ggplot2 is needed by HonestDiD for plotting; also install if missing
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2", repos = "https://cran.r-project.org")
}
library(ggplot2)


# ======================================================================
#  STEP 1: Configuration
# ======================================================================

# Input file from script 18
INPUT_DIR  <- "output_18"
INPUT_FILE <- file.path(INPUT_DIR, "corrected_es_all_ref2022H1.csv")

# Output directory
OUTPUT_DIR <- "output_23"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# Reference period (omitted dummy in the event study)
REF_PERIOD <- "2022H1"

# Treatment begins at 2022H2 (ChatGPT launched Nov 2022, so the first
# post-treatment half-year period is 2022H2 = Jul-Dec 2022)
TREATMENT_START <- "2022H2"

# All half-year periods in order (must match script 18 output)
ALL_PERIODS <- c(
  "2019H1", "2019H2",
  "2020H1", "2020H2",
  "2021H1", "2021H2",
  "2022H1", "2022H2",
  "2023H1", "2023H2",
  "2024H1", "2024H2",
  "2025H1"
)

# Age groups to analyse (all six from script 18)
AGE_GROUPS <- c("22-25", "26-30", "31-34", "35-40", "41-49", "50+")

# Significance level for confidence intervals
ALPHA <- 0.05

# Grid of M_bar values (relative magnitudes parameter)
# M_bar = 0: parallel trends hold exactly in post-period
# M_bar = 1: post-period violations at most as large as worst pre-period
# M_bar = 2: post-period violations up to 2x the worst pre-period
MBAR_GRID <- seq(from = 0, to = 4, by = 0.5)


# ======================================================================
#  STEP 2: Read the event study CSV
# ======================================================================

cat("=" , rep("=", 68), "\n", sep = "")
cat("23_mona_rambachan_roth.R\n")
cat("Rambachan-Roth (2023) sensitivity analysis for employment event study\n")
cat("=" , rep("=", 68), "\n\n", sep = "")

if (!file.exists(INPUT_FILE)) {
  stop(
    "Input file not found: ", INPUT_FILE, "\n",
    "  Run script 18 first to generate the event study coefficients.\n",
    "  Expected file: corrected_es_all_ref2022H1.csv in output_18/"
  )
}

es_data <- read.csv(INPUT_FILE, stringsAsFactors = FALSE)
cat("Loaded event study data:", nrow(es_data), "rows\n")
cat("Age groups found:", paste(unique(es_data$age_group), collapse = ", "), "\n")
cat("Periods found:", paste(sort(unique(es_data$period)), collapse = ", "), "\n\n")


# ======================================================================
#  STEP 3: Define helper functions
# ======================================================================

run_rambachan_roth <- function(es_sub, age_label) {
  # -------------------------------------------------------------------
  # Run HonestDiD relative magnitudes analysis for one age group.
  #
  # Arguments:
  #   es_sub     -- data.frame with columns: period, coef, se, pval
  #                 for ONE age group (including the reference row)
  #   age_label  -- character string for labelling output (e.g. "22-25")
  #
  # Returns:
  #   A list with: breakdown_mbar, honest_ci, sensitivity_df, or NULL on failure
  # -------------------------------------------------------------------

  cat("\n--- Age group:", age_label, "---\n")

  # Sort by period
  es_sub <- es_sub[order(es_sub$period), ]

  # Remove the reference period row (coef=0, se=0) -- HonestDiD expects
  # only the estimated coefficients, not the omitted category
  es_sub <- es_sub[es_sub$period != REF_PERIOD, ]

  # The periods in order (excluding reference)
  periods <- es_sub$period

  # Identify which coefficients are pre-treatment vs post-treatment.
  # Pre-treatment: periods before the reference period (2022H1).
  # Since reference is omitted, pre-treatment = all periods < "2022H1".
  # Post-treatment: periods >= TREATMENT_START ("2022H2").
  #
  # IMPORTANT: HonestDiD expects the indices to be 1-based positions in
  # the coefficient vector (after removing the reference period).
  is_pre  <- periods < REF_PERIOD
  is_post <- periods >= TREATMENT_START

  n_pre  <- sum(is_pre)
  n_post <- sum(is_post)

  cat("  Pre-treatment periods (", n_pre, "):",
      paste(periods[is_pre], collapse = ", "), "\n")
  cat("  Post-treatment periods (", n_post, "):",
      paste(periods[is_post], collapse = ", "), "\n")

  if (n_pre < 2) {
    cat("  WARNING: Too few pre-treatment periods for meaningful analysis.\n")
    return(NULL)
  }
  if (n_post < 1) {
    cat("  WARNING: No post-treatment periods.\n")
    return(NULL)
  }

  # Extract coefficient vector (beta_hat) -- all estimated periods
  betahat <- es_sub$coef

  # Construct diagonal variance-covariance matrix
  # (conservative: assumes no covariance between period estimates)
  sigma <- diag(es_sub$se^2)

  # Sanity check: print the coefficients
  cat("  Coefficients:\n")
  for (i in seq_along(periods)) {
    star <- ""
    if (!is.na(es_sub$pval[i])) {
      if (es_sub$pval[i] < 0.01) star <- "***"
      else if (es_sub$pval[i] < 0.05) star <- "**"
      else if (es_sub$pval[i] < 0.10) star <- "*"
    }
    cat(sprintf("    %s: %8.4f (SE = %6.4f) %s\n",
                periods[i], betahat[i], es_sub$se[i], star))
  }

  # -------------------------------------------------------------------
  # l_vec: a vector of length numPostPeriods that defines the scalar
  # parameter of interest: theta = l_vec' * tau_post, where tau_post
  # is the vector of post-treatment dynamic effects.
  #
  # For the AVERAGE post-treatment effect, we set equal weights:
  #   l_vec = (1/n_post, 1/n_post, ..., 1/n_post)
  #
  # NOTE: l_vec has length numPostPeriods (NOT the full coefficient vector).
  # The default in HonestDiD is (1, 0, ..., 0), which tests only the
  # first post-treatment period. We override this to test the average.
  # -------------------------------------------------------------------
  l_vec <- rep(1 / n_post, n_post)

  cat("  l_vec (weights for average post-treatment effect):\n")
  cat("   ", paste(round(l_vec, 3), collapse = ", "), "\n")

  # -------------------------------------------------------------------
  # numPrePeriods and numPostPeriods: HonestDiD needs to know how many
  # pre- and post-treatment periods there are. The coefficient vector
  # must be ordered: [pre-treatment periods, post-treatment periods].
  #
  # Our vector IS in this order because periods are sorted chronologically
  # and pre-treatment < reference < post-treatment.
  # -------------------------------------------------------------------

  cat("\n  Running HonestDiD relative magnitudes analysis...\n")

  # Attempt the sensitivity analysis
  tryCatch({

    # createSensitivityResults_relativeMagnitudes computes honest CIs
    # for a grid of M_bar values. It returns a data.frame with columns:
    #   Mbar, lb, ub, method
    sensitivity <- createSensitivityResults_relativeMagnitudes(
      betahat        = betahat,
      sigma          = sigma,
      numPrePeriods  = n_pre,
      numPostPeriods = n_post,
      Mbarvec        = MBAR_GRID,
      l_vec          = l_vec,
      alpha          = ALPHA
    )

    cat("  Sensitivity results:\n")
    print(sensitivity)

    # -------------------------------------------------------------------
    # Find the breakdown value: largest M_bar for which the honest CI
    # excludes zero. This tells us how much larger than the worst
    # pre-trend violation the post-trend violation would need to be
    # before we lose significance.
    # -------------------------------------------------------------------

    # For each row, check if CI excludes zero
    # CI excludes zero if: both lb and ub are on the same side of zero
    sensitivity$excludes_zero <- (sensitivity$lb > 0) | (sensitivity$ub < 0)

    # The breakdown value is the largest M_bar where CI still excludes zero
    if (any(sensitivity$excludes_zero)) {
      breakdown_mbar <- max(sensitivity$Mbar[sensitivity$excludes_zero])
    } else {
      # Even at M_bar=0, CI includes zero -- no breakdown value
      breakdown_mbar <- NA
    }

    cat(sprintf("\n  BREAKDOWN VALUE: M_bar = %s\n",
                ifelse(is.na(breakdown_mbar), "< 0 (not significant even under exact PT)",
                       sprintf("%.1f", breakdown_mbar))))

    # Also report the original (M_bar=0) CI
    orig_row <- sensitivity[sensitivity$Mbar == 0, ]
    if (nrow(orig_row) > 0) {
      cat(sprintf("  Original CI (M_bar=0): [%.4f, %.4f]\n",
                  orig_row$lb[1], orig_row$ub[1]))
    }

    # Report honest CI at M_bar = 1 (post-treatment violations same
    # magnitude as worst pre-treatment)
    m1_row <- sensitivity[sensitivity$Mbar == 1, ]
    if (nrow(m1_row) > 0) {
      cat(sprintf("  Honest CI (M_bar=1):   [%.4f, %.4f]  %s\n",
                  m1_row$lb[1], m1_row$ub[1],
                  ifelse(m1_row$excludes_zero[1],
                         "** excludes zero **", "includes zero")))
    }

    # Return results
    return(list(
      age_group      = age_label,
      breakdown_mbar = breakdown_mbar,
      sensitivity    = sensitivity,
      betahat        = betahat,
      sigma          = sigma,
      n_pre          = n_pre,
      n_post         = n_post,
      periods        = periods
    ))

  }, error = function(e) {
    cat("  ERROR in HonestDiD:", conditionMessage(e), "\n")
    cat("  This may indicate a package version issue or numerical problem.\n")
    return(NULL)
  })
}


# ======================================================================
#  STEP 4: Run analysis for each age group
# ======================================================================

all_results <- list()
summary_lines <- character()

summary_lines <- c(
  paste(rep("=", 60), collapse = ""),
  "RAMBACHAN-ROTH (2023) SENSITIVITY ANALYSIS",
  "Employment event study (script 18 output, ref = 2022H1)",
  paste(rep("=", 60), collapse = ""),
  "",
  "Method: Relative magnitudes restriction (HonestDiD package)",
  "Covariance: Diagonal (conservative -- no cross-period covariance)",
  sprintf("Significance level: alpha = %.2f", ALPHA),
  sprintf("M_bar grid: %s", paste(MBAR_GRID, collapse = ", ")),
  ""
)

for (age in AGE_GROUPS) {
  es_sub <- es_data[es_data$age_group == age, ]

  if (nrow(es_sub) == 0) {
    cat("\n  Skipping", age, "-- no data\n")
    next
  }

  result <- run_rambachan_roth(es_sub, age)

  if (!is.null(result)) {
    all_results[[age]] <- result

    # Add to summary
    summary_lines <- c(summary_lines,
      sprintf("--- %s ---", age),
      sprintf("  Pre-treatment periods:  %d", result$n_pre),
      sprintf("  Post-treatment periods: %d", result$n_post),
      sprintf("  Breakdown M_bar:        %s",
              ifelse(is.na(result$breakdown_mbar),
                     "< 0 (not significant even under exact parallel trends)",
                     sprintf("%.1f", result$breakdown_mbar)))
    )

    # Add honest CIs at key M_bar values
    for (m in c(0, 0.5, 1.0, 2.0)) {
      row <- result$sensitivity[result$sensitivity$Mbar == m, ]
      if (nrow(row) > 0) {
        summary_lines <- c(summary_lines,
          sprintf("  Honest CI (M_bar=%.1f): [%.4f, %.4f] %s",
                  m, row$lb[1], row$ub[1],
                  ifelse(row$excludes_zero[1], "", " <-- includes zero")))
      }
    }
    summary_lines <- c(summary_lines, "")
  }
}


# ======================================================================
#  STEP 5: Create sensitivity plots
# ======================================================================

cat("\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("Creating sensitivity plots\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

for (age in names(all_results)) {
  result <- all_results[[age]]
  sens   <- result$sensitivity

  # Clean age label for filenames (replace + with plus)
  age_clean <- gsub("\\+", "plus", age)

  # Build the plot manually with ggplot2
  # (HonestDiD has a built-in plot function, but we want consistent styling)

  # Average post-treatment effect (the point estimate under exact PT)
  theta_hat <- sum(result$betahat[(result$n_pre + 1):length(result$betahat)]) /
               result$n_post

  p <- ggplot(sens, aes(x = Mbar)) +
    # Honest CI band
    geom_ribbon(aes(ymin = lb, ymax = ub), fill = "#2E7D6F", alpha = 0.25) +
    # Point estimate line
    geom_hline(yintercept = theta_hat, colour = "#1B3A5C", linewidth = 0.8) +
    # Zero line
    geom_hline(yintercept = 0, colour = "grey50", linewidth = 0.5, linetype = "dashed") +
    # Labels
    labs(
      x = expression(bar(M) ~ "(relative magnitudes)"),
      y = "Average post-treatment effect",
      title = paste0("Rambachan-Roth sensitivity: age group ", age),
      subtitle = paste0(
        "Breakdown M = ",
        ifelse(is.na(result$breakdown_mbar), "< 0",
               sprintf("%.1f", result$breakdown_mbar)),
        " | Diagonal vcov (conservative)"
      )
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title    = element_text(face = "bold", size = 13),
      plot.subtitle = element_text(size = 10, colour = "grey40"),
      panel.grid.minor = element_blank()
    )

  # Add breakdown value marker if it exists
  if (!is.na(result$breakdown_mbar)) {
    p <- p + geom_vline(
      xintercept = result$breakdown_mbar,
      colour = "#E8873A", linewidth = 0.8, linetype = "dotted"
    ) +
    annotate("text",
      x = result$breakdown_mbar + 0.15,
      y = theta_hat,
      label = sprintf("Breakdown\nM = %.1f", result$breakdown_mbar),
      colour = "#E8873A", size = 3.5, hjust = 0, fontface = "bold"
    )
  }

  # Save
  out_file <- file.path(OUTPUT_DIR, paste0("rambachan_roth_", age_clean, ".png"))
  ggsave(out_file, p, width = 8, height = 5, dpi = 150)
  cat("  Saved:", out_file, "\n")
}

# --- Combined panel figure (2x3) for appendix ---
if (length(all_results) >= 2) {
  cat("  Creating combined panel figure...\n")

  # Bind all sensitivity results into one data.frame
  all_sens <- do.call(rbind, lapply(names(all_results), function(age) {
    df <- all_results[[age]]$sensitivity
    df$age_group <- age

    # Compute the average post-treatment point estimate
    r <- all_results[[age]]
    df$theta_hat <- sum(r$betahat[(r$n_pre + 1):length(r$betahat)]) / r$n_post

    # Add breakdown value for annotation
    df$breakdown <- r$breakdown_mbar

    return(df)
  }))

  # Order age groups logically
  all_sens$age_group <- factor(all_sens$age_group, levels = AGE_GROUPS)

  p_panel <- ggplot(all_sens, aes(x = Mbar)) +
    geom_ribbon(aes(ymin = lb, ymax = ub), fill = "#2E7D6F", alpha = 0.25) +
    geom_hline(aes(yintercept = theta_hat), colour = "#1B3A5C", linewidth = 0.6) +
    geom_hline(yintercept = 0, colour = "grey50", linewidth = 0.4, linetype = "dashed") +
    facet_wrap(~ age_group, scales = "free_y", ncol = 3) +
    labs(
      x = expression(bar(M) ~ "(relative magnitudes)"),
      y = "Average post-treatment effect",
      title = "Rambachan-Roth sensitivity analysis by age group"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title       = element_text(face = "bold", size = 13),
      strip.text       = element_text(face = "bold", size = 11),
      panel.grid.minor = element_blank()
    )

  # Add breakdown markers for each facet
  breakdown_df <- do.call(rbind, lapply(names(all_results), function(age) {
    r <- all_results[[age]]
    if (!is.na(r$breakdown_mbar)) {
      data.frame(
        age_group = age,
        breakdown = r$breakdown_mbar,
        stringsAsFactors = FALSE
      )
    } else {
      NULL
    }
  }))

  if (!is.null(breakdown_df) && nrow(breakdown_df) > 0) {
    breakdown_df$age_group <- factor(breakdown_df$age_group, levels = AGE_GROUPS)
    p_panel <- p_panel +
      geom_vline(
        data = breakdown_df,
        aes(xintercept = breakdown),
        colour = "#E8873A", linewidth = 0.6, linetype = "dotted"
      )
  }

  out_panel <- file.path(OUTPUT_DIR, "rambachan_roth_panel.png")
  ggsave(out_panel, p_panel, width = 12, height = 7, dpi = 150)
  cat("  Saved:", out_panel, "\n")
}


# ======================================================================
#  STEP 6: Save results CSV
# ======================================================================

cat("\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("Saving results\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

# Build a results table: one row per age group x M_bar combination
results_rows <- list()
row_idx <- 1

for (age in names(all_results)) {
  r <- all_results[[age]]
  sens <- r$sensitivity

  for (i in seq_len(nrow(sens))) {
    results_rows[[row_idx]] <- data.frame(
      age_group      = age,
      Mbar           = sens$Mbar[i],
      ci_lower       = sens$lb[i],
      ci_upper       = sens$ub[i],
      excludes_zero  = sens$excludes_zero[i],
      method         = if ("method" %in% names(sens)) sens$method[i] else "FLCI",
      breakdown_mbar = r$breakdown_mbar,
      n_pre_periods  = r$n_pre,
      n_post_periods = r$n_post,
      stringsAsFactors = FALSE
    )
    row_idx <- row_idx + 1
  }
}

if (length(results_rows) > 0) {
  results_df <- do.call(rbind, results_rows)
  out_csv <- file.path(OUTPUT_DIR, "rambachan_roth_results.csv")
  write.csv(results_df, out_csv, row.names = FALSE)
  cat("  Saved:", out_csv, "\n")
}

# Save summary text
summary_lines <- c(summary_lines,
  paste(rep("=", 60), collapse = ""),
  "INTERPRETATION GUIDE",
  paste(rep("=", 60), collapse = ""),
  "",
  "The breakdown value M_bar tells you how robust the result is to",
  "violations of parallel trends in the post-treatment period.",
  "",
  "  M_bar > 1: Result survives even if post-treatment trend violations",
  "             are LARGER than the worst pre-period violation. Strong.",
  "",
  "  M_bar ~ 1: Result survives if post-treatment violations are at most",
  "             as large as the worst pre-period violation. Moderate.",
  "",
  "  M_bar < 1: Result breaks down for violations SMALLER than those",
  "             observed pre-treatment. The result is fragile.",
  "",
  "  M_bar = NA: The result is not significant even under exact parallel",
  "              trends (M_bar = 0). No sensitivity analysis needed.",
  "",
  "NOTE: We use a diagonal covariance matrix (no cross-period covariance).",
  "This is conservative -- the true honest CIs would be narrower if we",
  "accounted for positive correlation between adjacent period estimates.",
  "The breakdown values reported here are therefore LOWER BOUNDS on the",
  "true breakdown values.",
  "",
  paste("Generated:", Sys.time())
)

out_summary <- file.path(OUTPUT_DIR, "rambachan_roth_summary.txt")
writeLines(summary_lines, out_summary)
cat("  Saved:", out_summary, "\n")

cat("\nDone. All output in:", OUTPUT_DIR, "/\n")


# ======================================================================
#  FALLBACK: If HonestDiD is not available on MONA
# ======================================================================
#
# If you cannot install HonestDiD on MONA (e.g. no internet access,
# no admin rights), here are your options:
#
# OPTION A: Manual installation
#   1. On a computer with internet, download the package:
#        download.packages("HonestDiD", destdir = ".", repos = "https://cran.r-project.org")
#      This gives you a .tar.gz file.
#   2. Transfer the .tar.gz to MONA (via the standard file import process).
#   3. Install locally:
#        install.packages("HonestDiD_X.Y.Z.tar.gz", repos = NULL, type = "source")
#   4. You also need the dependencies: CVXR, lpSolveAPI, Matrix, ggplot2.
#      Download and transfer those too.
#
# OPTION B: Simplified Python implementation
#   Script 07 (07_robustness.py) contains a simplified Rambachan-Roth
#   analysis for the posting-level event study. The same approach can be
#   adapted for the employment event study. The simplified version
#   computes:
#     CI(M_bar) = theta_hat +/- [z_0.975 * SE(theta_hat) + M_bar * Delta_max]
#   where Delta_max = max |delta_{t} - delta_{t-1}| over pre-periods.
#   This gives WIDER (more conservative) intervals than the exact HonestDiD
#   computation, which solves a linear programme. But it is straightforward
#   to implement in Python with no extra packages.
#
# OPTION C: Export the CSV and run on a non-MONA machine
#   The event study CSV (corrected_es_all_ref2022H1.csv) contains only
#   AGGREGATED coefficients -- no individual-level data. It is therefore
#   safe to export from MONA (subject to your project's disclosure rules).
#   You can then run this script on your own machine with full package
#   access. Check with your MONA project leader before exporting.
#
# ======================================================================
