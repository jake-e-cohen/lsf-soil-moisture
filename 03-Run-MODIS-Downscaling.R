# Config ====
library(caret)
library(ggplot2)
library(jsonlite)
library(randomForest)
library(tidyverse)

setwd("/Users/jakecohen/Desktop/Final Code Repository/")

# Load data ====
df <- readr::read_csv("/Users/jakecohen/Downloads/downscaling_data.csv")

# Create derived index for date
sin.doy <- sin(yday(df$date) * 2 * pi / 365)

# Add spatial information
long.lat <- df$.geo %>%
  lapply(fromJSON) %>%
  sapply(function(geometry) geometry[["coordinates"]]) %>%
  t()
colnames(long.lat) <- c("longitude", "latitude")

df <- df %>% subset(select = -c(.geo, date, `system:index`)) %>%
  bind_cols(long.lat, sin.doy = sin.doy)

# Split data ====
set.seed(123)
p <- 0.8
train.indices <- sample(seq(from = 1, to = nrow(df)), size = p * nrow(df))
data.train <- df[train.indices, ]
data.test <- df[-train.indices, ]

# Train model ====
train.controls <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
tuning.grid <- expand.grid(mtry = c(8, 9, 10, 11))
model <- train(
  LST ~ .,
  data = data.train,
  method = "rf",
  metric = "RMSE",
  trControl = train.controls,
  tuneGrid = tuning.grid,
  ntree = 200
)

# Make predictions ====
predicted <- predict(model, newdata = data.test)
actual <- data.test$LST

r2 <- R2(predicted, actual)
rmse <- RMSE(predicted, actual)
performance.labs <- c(
  paste0("R\u00B2: ", round(r2, 3)),
  paste0("RMSE: ", round(rmse, 3), "ºC")
) %>%
  paste(collapse = "\n")

predicted.fit <- lm(actual ~ predicted)
fit.intercept <- predicted.fit$coefficients[1]
fit.slope <- predicted.fit$coefficients[2]

plot.lims <- c(min(actual), max(actual))
ggplot(mapping = aes(x = predicted, y = actual)) +
  geom_point() +
  labs(x = "Predicted LST (ºC)", y = "Observed LST (ºC) ") +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  geom_abline(slope = fit.slope, intercept = fit.intercept,
              linetype = "dashed", color = "darkgrey") +
  geom_label(
    aes(x = plot.lims[1], y = plot.lims[2], label = performance.labs),
    fill = "skyblue",
    color = "black",
    hjust = 0, vjust = 1,
    size = 5,
    alpha = 0.7
  ) +
  theme_light() +
  theme(panel.grid = element_blank())


# Save the model ====
save(model, file = "src/LST_downscaling_model.RData")
