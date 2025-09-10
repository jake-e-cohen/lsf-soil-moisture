# ==== Imports ====
library(ggplot2)
library(tidyverse)

# ==== Set-up ====
set.seed(123)
setwd("/Users/jakecohen/Desktop/Final Code Repository/")

# ==== 1. Load Data ====
df <- read_csv("src/final_data.csv")

# ==== 2. Investigate Soil Moisture Data ====
## ==== 2.1. Summary Statistics ====
summary.stats <- df$soil.moisture %>%
  summary() %>%
  as.matrix() %>%
  t() %>%
  as.data.frame() %>%
  mutate(
    IQR = `3rd Qu.` - `1st Qu.`,
    range = Max. - Min.,
    stddev = sqrt(var(df$soil.moisture)),
    n = nrow(df)
  )

# Theoretical values if the data is normally distributed
normal.data <- rnorm(1000, summary.stats$Mean, summary.stats$stddev)

## ==== 2.2. Distribution of Soil Moisture Values ====
bw <- 2 * summary.stats$IQR / summary.stats$n ^ (1/3)
ggplot() +
  geom_histogram(
    df,
    mapping = aes(x = soil.moisture, after_stat(density)),
    fill = "#039af9",
    binwidth = bw
  ) +
  geom_density(mapping = aes(x = normal.data)) +
  labs(x = "Soil Moisture (VWC%)", y = "Density") +
  scale_x_continuous(breaks = seq(0, 1, 0.1)) +
  theme_classic()

## ==== 2.3. Normal QQ Plot ====
ggplot(df, aes(sample = soil.moisture)) +
  geom_qq() +
  geom_qq_line(color = "red") +
  labs(x = "Theoretical Quantiles", y = "Observed Quantiles") +
  theme_classic()

## ==== 2.4. 95% Confidence Intervals for Summary Stats ====
# Note we are assuming normality based on results of previous EDA
t.mean <- t.test(df$soil.moisture, alternative = "two.sided")
ci.mean <- t.mean$conf.int

# CI for standard deviation
chisq.low <- qchisq(0.05 / 2, summary.stats$n - 1, lower.tail = TRUE)
chisq.upp <- qchisq(0.05 / 2, summary.stats$n - 1, lower.tail = FALSE)

ci.stddev <- c(
  sqrt((summary.stats$n - 1) * (summary.stats$stddev) ^ 2 / chisq.upp),
  sqrt((summary.stats$n - 1) * (summary.stats$stddev) ^ 2 / chisq.low)
)
