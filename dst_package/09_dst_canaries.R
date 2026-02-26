#!/usr/bin/env Rscript
# ======================================================================
# 09_dst_canaries.R — Canaries test using Danish register data in DST
#
# THIS SCRIPT IS DESIGNED TO RUN IN DST's SECURE ENVIRONMENT
# It uses monthly employment register data (e.g. BFL/eIndkomst).
# Do NOT run outside DST — the data is not available externally.
#
# Purpose:
#   Test the Brynjolfsson et al. (2025) "canaries in the coal mine"
#   hypothesis with Danish monthly register data: do young workers in
#   high-AI-exposure occupations experience disproportionate employment
#   declines after ChatGPT?
#
# Specification (identical to Swedish MONA analysis):
#   ln(emp_it) = alpha_i + gamma_t
#       + beta1 * Post_RateHike * High
#       + beta2 * Post_RateHike * Young
#       + beta3 * Post_RateHike * Young * High
#       + beta4 * Post_ChatGPT * High
#       + beta5 * Post_ChatGPT * Young
#       + beta6 * Post_ChatGPT * Young * High    <-- KEY COEFFICIENT
#       + epsilon_it
#
#   Entity FE + time FE, clustered SE at entity level.
#
# Authors: Lodefalk, Engberg, Koch, Lothman
# Date: February 2026
# ======================================================================

library(data.table)
library(fixest)      # For fast fixed-effects estimation
library(ggplot2)

# ======================================================================
# CONFIGURATION — ADJUST THESE FOR YOUR DST ENVIRONMENT
# ======================================================================

# Path to your employment register extract
# (monthly individual-level data, e.g. BFL or eIndkomst)
INPUT_PATH <- "path/to/your/employment_extract.csv"  # ADJUST THIS

# Column names in your extract — adjust to match your variable names
COL_PERSON_ID  <- "PNR"           # Encrypted person ID (for dedup)
COL_YEAR_MONTH <- "PERIOD"        # Year-month (YYYY-MM or similar)
COL_DISCO4     <- "DISCO4"        # 4-digit DISCO/ISCO occupation code
COL_BIRTH_YEAR <- "FOEDSELSAAR"   # Birth year (to compute age)

# Path to DAIOE quartiles
# Option A: Use the version Mark/Sarah already have on DST
# Option B: Import daioe_quartiles.csv from this package
DAIOE_PATH <- "daioe_quartiles.csv"  # ADJUST if needed

# Occupation code column in DAIOE file
# SSYK and DISCO are both ISCO-08 based — 4-digit codes align closely.
# If Mark/Sarah already mapped DAIOE to DISCO codes, use that version
# and set this to the relevant column name.
DAIOE_OCC_COL <- "ssyk4"  # Change to "disco4" if already remapped

# Output directory
OUTPUT_DIR <- "output"
dir.create(OUTPUT_DIR, showWarnings = FALSE)

# Treatment dates
# ECB announced first hike 21 July 2022 (+50 bps); Danish markets priced it in
# immediately. Nationalbanken formally followed in September 2022 (+75 bps).
# We use the ECB date for better separation from ChatGPT (5 months vs 3).
# Swedish analysis uses Riksbanken April 2022.
RATE_HIKE      <- "2022-07"
CHATGPT_LAUNCH <- "2022-12"

# Colours (same as Swedish analysis)
DARK_BLUE <- "#1B3A5C"
ORANGE    <- "#E8873A"
TEAL      <- "#2E7D6F"
GRAY      <- "#8C8C8C"

# Spotlight occupations (DISCO/ISCO-08, 4-digit — same codes as Swedish SSYK)
DISCO_SOFTWARE <- c("2512")   # Software developers
DISCO_CUSTOMER <- c("4221",   # Contact centre agents
                     "4222",   # Helpdesk/support
                     "5230")   # Cashiers etc.

# Fine age bands matching Brynjolfsson et al.
# NOTE: ages 16-21 excluded from spotlight (included in broad canaries).
AGE_BREAKS <- c(22, 26, 31, 35, 41, 50, 70)
AGE_LABELS <- c("22-25", "26-30", "31-34", "35-40", "41-49", "50+")

# Colours for age-band trajectories (youngest = warmest)
AGE_BAND_COLORS <- c(
  "22-25" = "#E8873A",
  "26-30" = "#F0A86B",
  "31-34" = "#A8C5BC",
  "35-40" = "#5FA898",
  "41-49" = "#2E7D6F",
  "50+"   = "#1B3A5C"
)

# Normalisation base month for spotlight figures (just before ChatGPT)
BASE_MONTH <- "2022-10"


# ======================================================================
# STEP 1: LOAD AND PREPARE EMPLOYMENT DATA
# ======================================================================

cat("Loading employment data...\n")

# Read data — adjust format if parquet/SAS
df <- fread(INPUT_PATH)

cat(sprintf("  Loaded %s records\n", format(nrow(df), big.mark = ",")))
cat(sprintf("  Columns: %s\n", paste(names(df), collapse = ", ")))

# Rename to standard names
setnames(df,
  old = c(COL_PERSON_ID, COL_YEAR_MONTH, COL_DISCO4, COL_BIRTH_YEAR),
  new = c("person_id", "year_month", "occ4", "birth_year"),
  skip_absent = TRUE
)

# Parse year-month to string format YYYY-MM
df[, year_month := substr(as.character(year_month), 1, 7)]

# Compute age
df[, year := as.integer(substr(year_month, 1, 4))]
df[, age := year - as.integer(birth_year)]

# Age group: young (16-24) vs older (25+)
df[, young := as.integer(age >= 16 & age <= 24)]

# Occupation code: zero-pad to 4 digits
df[, occ4 := sprintf("%04d", as.integer(occ4))]

# Filter to working-age population
df <- df[age >= 16 & age <= 69]

# Fine age bands (for spotlight figures)
# cut with right=FALSE: [22,26), [26,31), ... = ages 22-25, 26-30, ...
# Ages 16-21 get NA (excluded from spotlight, included in broad canaries)
df[, age_band := cut(age, breaks = AGE_BREAKS,
                     labels = AGE_LABELS, right = FALSE)]

# Aggregate: occupation x fine_age_band x month (spotlight)
agg_fine <- df[!is.na(age_band),
               .(n_employed = uniqueN(person_id)),
               by = .(occ4, year_month, age_band)]

# Aggregate: occupation x age_group x month (broad canaries)
agg <- df[, .(n_employed = uniqueN(person_id)),
          by = .(occ4, year_month, young)]

cat(sprintf("  Aggregated (broad): %s cells\n", format(nrow(agg), big.mark = ",")))
cat(sprintf("  Aggregated (fine):  %s cells\n", format(nrow(agg_fine), big.mark = ",")))
cat(sprintf("  Occupations: %d\n", uniqueN(agg$occ4)))
cat(sprintf("  Months: %d\n", uniqueN(agg$year_month)))
cat(sprintf("  Period: %s to %s\n", min(agg$year_month), max(agg$year_month)))


# ======================================================================
# STEP 2: MERGE WITH DAIOE
# ======================================================================

cat("\nMerging with DAIOE...\n")

daioe <- fread(DAIOE_PATH)
daioe[, (DAIOE_OCC_COL) := sprintf("%04d", as.integer(get(DAIOE_OCC_COL)))]

# Rename DAIOE occupation column to match
if (DAIOE_OCC_COL != "occ4") {
  setnames(daioe, DAIOE_OCC_COL, "occ4")
}

merged <- merge(
  agg,
  daioe[, .(occ4, pctl_rank_genai, exposure_quartile, high_exposure)],
  by = "occ4",
  all = FALSE  # inner join
)

n_matched <- uniqueN(merged$occ4)
n_total   <- uniqueN(agg$occ4)
cat(sprintf("  Matched: %d of %d occupations (%0.0f%%)\n",
            n_matched, n_total, 100 * n_matched / n_total))


# ======================================================================
# STEP 3: TRIPLE-DIFF REGRESSION
# ======================================================================

cat("\nRunning triple-diff regression...\n")

# Prepare panel
panel <- copy(merged)
panel <- panel[n_employed > 0]
panel[, ln_emp := log(n_employed)]

# Entity = occupation x age group
panel[, entity := paste0(occ4, "_", young)]
panel[, date := as.Date(paste0(year_month, "-01"))]

# Treatment dummies
panel[, post_chatgpt  := as.integer(year_month >= CHATGPT_LAUNCH)]
panel[, post_ratehike := as.integer(year_month >= RATE_HIKE)]

# Interactions
panel[, post_high       := post_chatgpt * high_exposure]
panel[, post_young      := post_chatgpt * young]
panel[, post_young_high := post_chatgpt * young * high_exposure]

panel[, rh_high       := post_ratehike * high_exposure]
panel[, rh_young      := post_ratehike * young]
panel[, rh_young_high := post_ratehike * young * high_exposure]

# Fixed-effects regression using fixest
# Entity FE + year-month FE, clustered SE at entity level
reg <- feols(
  ln_emp ~ rh_high + rh_young + rh_young_high +
           post_high + post_young + post_young_high |
           entity + year_month,
  data    = panel,
  cluster = ~entity
)

cat("\nFull specification (entity + time FE):\n")
print(summary(reg))

# Extract the key coefficient
b3 <- coef(reg)["post_young_high"]
p3 <- pvalue(reg)["post_young_high"]
cat(sprintf("\n>>> CANARIES TEST: beta3 (Post x Young x High) = %+.4f, p = %.4f\n",
            b3, p3))
if (p3 < 0.05) {
  cat(">>> SIGNIFICANT at 5%% — evidence of canaries effect\n")
} else {
  cat(">>> NOT significant — no canaries effect detected\n")
}

# Save regression results
reg_df <- data.table(
  variable    = names(coef(reg)),
  coefficient = coef(reg),
  std_error   = se(reg),
  p_value     = pvalue(reg)
)
fwrite(reg_df, file.path(OUTPUT_DIR, "dst_canaries_regression.csv"))
cat(sprintf("\n  Saved -> dst_canaries_regression.csv\n"))


# ======================================================================
# STEP 3b: ROBUSTNESS — WITHOUT RATE HIKE INTERACTIONS
# ======================================================================
# The ECB/Nationalbanken rate hike (July/Sep 2022) is only 5 months
# before ChatGPT (Dec 2022), creating potential collinearity.
# This robustness check drops the rate hike interactions entirely.

cat("\nRunning robustness regression (no rate hike interactions)...\n")

reg_robust <- feols(
  ln_emp ~ post_high + post_young + post_young_high |
           entity + year_month,
  data    = panel,
  cluster = ~entity
)

cat("\nRobustness specification (no rate hike controls):\n")
print(summary(reg_robust))

b3r <- coef(reg_robust)["post_young_high"]
p3r <- pvalue(reg_robust)["post_young_high"]
cat(sprintf("\n>>> ROBUSTNESS: beta3 (Post x Young x High) = %+.4f, p = %.4f\n",
            b3r, p3r))

# Save robustness results
reg_robust_df <- data.table(
  variable    = names(coef(reg_robust)),
  coefficient = coef(reg_robust),
  std_error   = se(reg_robust),
  p_value     = pvalue(reg_robust)
)
fwrite(reg_robust_df, file.path(OUTPUT_DIR, "dst_canaries_regression_robust.csv"))
cat(sprintf("  Saved -> dst_canaries_regression_robust.csv\n"))


# ======================================================================
# STEP 4a: SPOTLIGHT FIGURES — SPECIFIC OCCUPATIONS BY AGE BAND
# ======================================================================

plot_spotlight <- function(agg_fine, occ_codes, occupation_label, out_filename) {
  cat(sprintf("\nPlotting spotlight: %s ...\n", occupation_label))

  sub <- agg_fine[occ4 %in% occ_codes]
  if (nrow(sub) == 0) {
    cat(sprintf("  WARNING: no data for codes %s — skipping.\n",
                paste(occ_codes, collapse = ", ")))
    return(invisible(NULL))
  }

  sub <- sub[, .(n_employed = sum(n_employed)), by = .(year_month, age_band)]
  sub[, date := as.Date(paste0(year_month, "-01"))]

  avail_base <- if (BASE_MONTH %in% sub$year_month) BASE_MONTH else min(sub$year_month)
  base_vals <- sub[year_month == avail_base, .(age_band, base_emp = n_employed)]
  valid_bands <- base_vals[base_emp > 0]$age_band

  if (length(valid_bands) == 0) {
    cat("  WARNING: no valid age bands after base-month filter.\n")
    return(invisible(NULL))
  }

  sub <- merge(sub[age_band %in% valid_bands], base_vals, by = "age_band")
  sub[, index := 100 * n_employed / base_emp]
  sub[, age_band := factor(age_band, levels = AGE_LABELS)]

  p <- ggplot(sub, aes(x = date, y = index, colour = age_band)) +
    geom_line(aes(size = age_band)) +
    geom_vline(xintercept = as.Date(paste0(RATE_HIKE, "-01")),
               colour = ORANGE, linetype = "dotted", alpha = 0.7) +
    geom_vline(xintercept = as.Date("2022-12-01"),
               colour = TEAL, linetype = "dotted", size = 1) +
    geom_hline(yintercept = 100, colour = GRAY, linetype = "dashed", alpha = 0.5) +
    scale_colour_manual(values = AGE_BAND_COLORS) +
    scale_size_manual(values = c("22-25" = 1.3, "26-30" = 0.7, "31-34" = 0.7,
                                  "35-40" = 0.7, "41-49" = 0.7, "50+" = 0.7)) +
    labs(x = "", y = sprintf("Employment index (%s = 100)", avail_base),
         title = sprintf("Employment by age group — %s\n(Register data, DISCO %s)",
                         occupation_label, paste(occ_codes, collapse = ", ")),
         colour = "Age group", size = "Age group") +
    theme_minimal(base_size = 11) +
    theme(legend.position = "bottom")

  ggsave(file.path(OUTPUT_DIR, out_filename), p,
         width = 10, height = 5, dpi = 300)
  cat(sprintf("  Saved -> %s\n", out_filename))
}

# Call spotlight for software developers
plot_spotlight(agg_fine,
               occ_codes = DISCO_SOFTWARE,
               occupation_label = "Software Developers",
               out_filename = "figA8a_dst_canaries_softwaredevelopers.png")

# Call spotlight for customer service agents
plot_spotlight(agg_fine,
               occ_codes = DISCO_CUSTOMER,
               occupation_label = "Customer Service Agents",
               out_filename = "figA8b_dst_canaries_customerservice.png")


# ======================================================================
# STEP 4b: FIGURE — BROAD EMPLOYMENT TRAJECTORIES
# ======================================================================

cat("\nPlotting canaries trajectories...\n")

# Aggregate to month x young x high_exposure
plot_data <- merged[, .(n_employed = sum(n_employed)),
                    by = .(year_month, young, high_exposure)]
plot_data[, date := as.Date(paste0(year_month, "-01"))]

# Index to earliest month = 100
base_month <- min(plot_data$year_month)
base <- plot_data[year_month == base_month,
                  .(young, high_exposure, base_emp = n_employed)]
plot_data <- merge(plot_data, base, by = c("young", "high_exposure"))
plot_data[, index := 100 * n_employed / base_emp]

# Labels for groups
plot_data[, group := ifelse(
  young == 1 & high_exposure == 1, "Young (16-24), High AI",
  ifelse(young == 1 & high_exposure == 0, "Young (16-24), Low AI",
  ifelse(young == 0 & high_exposure == 1, "Older (25+), High AI",
         "Older (25+), Low AI")))]

# Order factor for consistent legend
plot_data[, group := factor(group, levels = c(
  "Young (16-24), High AI", "Young (16-24), Low AI",
  "Older (25+), High AI", "Older (25+), Low AI"
))]

p <- ggplot(plot_data, aes(x = date, y = index,
                           colour = group, linetype = group, size = group)) +
  geom_line() +
  geom_vline(xintercept = as.Date("2022-07-01"),
             colour = ORANGE, linetype = "dotted", alpha = 0.7) +
  geom_vline(xintercept = as.Date("2022-12-01"),
             colour = TEAL, linetype = "dotted", size = 1) +
  geom_hline(yintercept = 100, colour = GRAY, linetype = "dashed", alpha = 0.5) +
  scale_colour_manual(values = c(
    "Young (16-24), High AI" = ORANGE,
    "Young (16-24), Low AI"  = ORANGE,
    "Older (25+), High AI"   = DARK_BLUE,
    "Older (25+), Low AI"    = DARK_BLUE
  )) +
  scale_linetype_manual(values = c(
    "Young (16-24), High AI" = "solid",
    "Young (16-24), Low AI"  = "dashed",
    "Older (25+), High AI"   = "solid",
    "Older (25+), Low AI"    = "dashed"
  )) +
  scale_size_manual(values = c(
    "Young (16-24), High AI" = 1.2,
    "Young (16-24), Low AI"  = 0.7,
    "Older (25+), High AI"   = 1.2,
    "Older (25+), Low AI"    = 0.7
  )) +
  labs(x = "", y = "Employment index (base month = 100)",
       title = "Monthly employment by age and AI exposure (Danish register data)",
       colour = NULL, linetype = NULL, size = NULL) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")

ggsave(file.path(OUTPUT_DIR, "figA8c_dst_canaries_economy.png"), p,
       width = 10, height = 5, dpi = 300)
cat("  Saved -> figA8c_dst_canaries_economy.png\n")


# ======================================================================
# DONE
# ======================================================================

cat("\n", strrep("=", 70), "\n")
cat("DONE. Copy the following files from output/ to your project:\n")
cat("  1. figA8a_dst_canaries_softwaredevelopers.png -> figures/\n")
cat("  2. figA8b_dst_canaries_customerservice.png    -> figures/\n")
cat("  3. figA8c_dst_canaries_economy.png            -> figures/\n")
cat("  4. dst_canaries_regression.csv                -> tables/\n")
cat("  5. dst_canaries_regression_robust.csv         -> tables/\n")
cat(strrep("=", 70), "\n")
