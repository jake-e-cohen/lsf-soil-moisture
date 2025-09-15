# ==== Imports ====
library(blockCV)
library(caret)
library(ggplot2)
library(glmnet)
library(grid)
library(gstat)
library(kernlab)
library(patchwork)
library(randomForest)
library(reshape2)
library(sf)
library(sp)
library(tidyverse)
library(vip)
library(xgboost)

# ==== Set-up ====
set.seed(123)
setwd("working_directory") # Replace with working directory
df <- read_csv("data_table") # Replace with path to full dataset

# ==== 1. Remove Variables Concerning Location ====
df <- df %>%
  select(-X, -Y, -longitude, -latitude)

# ==== 2. Investigate Multicollinearity ====
# Pearson correlation coefficients between variables
cor.matrix <- cor(df %>% select(-date))

# Remove duplicates
cor.matrix[lower.tri(cor.matrix, diag = TRUE)] <- NA
cor.matrix <- cor.matrix %>%
  melt(na.rm = TRUE)

# Investigate correlation above a certain threshold
threshold.r <- 0.9
high.cor <- cor.matrix %>%
  filter(abs(value) >= threshold.r) %>%
  sort_by(abs(.$value), decreasing = TRUE)

head(high.cor, n = 10)
nrow(high.cor)    # over 100 pairs of highly correlated variables

# ==== 3. Create Temporally Distinct Training and Testing Folds ====
testing.dates <- c("2025-07-16", "2025-07-23")

# Split the data
testing.indices <- which(df$date %in% testing.dates)
training.data <- df[-testing.indices, ] %>% select(-date)
testing.data <- df[testing.indices, ] %>% select(-date)

# ==== 4. Perform Feature Selection ====
## ==== 4.1. LASSO Regression ====
# Train/Test Split
x.train <- training.data %>% select(-soil.moisture)
x.test <- testing.data %>% select(-soil.moisture)
Y.train <- training.data$soil.moisture
Y.test <- testing.data$soil.moisture

# Build LASSO Model
lasso.model <- cv.glmnet(as.matrix(x.train), Y.train, alpha = 1)
lasso.features <- coef(lasso.model) %>%
  as.matrix() %>%
  as.data.frame() %>%
  tibble::rownames_to_column("feature") %>%
  rename(coefficient = "s0") %>%
  filter(coefficient != 0) %>%
  filter(feature != "(Intercept)") %>%
  pull("feature")

## ==== 4.2. RF Regression ====
n.trees <- 100
rf.full <- randomForest(
  soil.moisture ~ .,
  data = training.data,
  ntree = n.trees,
  importance = TRUE
)

rf.feature.importance <- rf.full$importance %>%
  as.matrix() %>%
  as.data.frame() %>%
  tibble::rownames_to_column("feature") %>%
  arrange(desc(`%IncMSE`))

# Iteratively add features to decrease MSE
rf.features <- rf.feature.importance[1, ]$feature
remaining.features <- rf.feature.importance[-1, ]$feature

mse.all <- c(Inf)
for (i in 1:length(remaining.features)) {
  # Keep track of lowest MSE and current best feature candidate
  mse.min <- Inf
  best.feature <- ""
  
  for (feature in remaining.features) {
    # Select features to build forest on
    selected.features.temp <- c(rf.features, feature)
    
    model.data.temp <- training.data %>%
      select(all_of(selected.features.temp), soil.moisture)
    
    # Build the forest
    rf.temp <- randomForest(soil.moisture ~ .,
                            data = model.data.temp,
                            ntree = n.trees)
    
    # Compare MSE
    mse <- rf.temp$mse[n.trees]
    if (mse < mse.min) {
      best.feature <- feature
      mse.min <- mse
    }
  }
  
  if (mse.min < min(mse.all)) {
    rf.features <- c(rf.features, best.feature)
    remaining.features <- remaining.features[remaining.features != best.feature]
    mse.all <- c(mse.all, mse.min)
  }
}

# ==== 5. Define Process to Train Models ====
build.model <- function(data, testing.indices, method) {
  training.data <- data[-testing.indices, ]
  testing.data <- data[testing.indices, ]

  # Create folds for expanding window CV
  dates <- unique(training.data$date)
  initialWindow <- 1 # Number of dates to initially consider
  horizon <- 1 # How many dates to add per fold
  n.folds <- length(dates) - initialWindow - horizon + 1

  train.folds <- list()
  test.folds <- list()
  for (i in 1:n.folds) {
    train.dates <- dates[1:(i + initialWindow - 1)]
    test.dates <- dates[(i + initialWindow):(i + initialWindow + horizon - 1)]
    
    train.folds[[i]] <- which(training.data$date %in% train.dates)
    test.folds[[i]] <- which(training.data$date %in% test.dates)
  }
  
  # Model configuration
  training.controls <- trainControl(
    method = "cv",
    index = train.folds,
    indexOut = test.folds
  )
  
  preProc <- if (grepl("gausspr", method)) c("center", "scale") else NULL
  
  # Perform CV to tune hyperparameters
  cv.model <- train(
    soil.moisture ~ .,
    data = training.data %>% select(-date),
    method = method,
    trControl = training.controls,
    metric = "RMSE",
    preProcess = preProc
  )
  
  # Build the final model on all training data
  model <- train(
    soil.moisture ~ .,
    data = training.data %>% select(-date),
    method = method,
    trControl = trainControl(method = "none"),
    tuneGrid = cv.model$bestTune[rep(1, 1), , drop = FALSE],
    preProcess = preProc
  )
  
  # Compute metrics of trained model
  train.predicted <- predict(model, training.data)
  train.rmse <- RMSE(train.predicted, training.data$soil.moisture)
  train.r2 <- R2(train.predicted, training.data$soil.moisture)
  train.mbe <- mean(train.predicted - training.data$soil.moisture)
  
  # Evaluate performance on testing data
  test.predicted <- predict(model, testing.data)
  test.observed <- testing.data$soil.moisture
  
  test.rmse <- RMSE(test.predicted, test.observed)
  test.r2 <- R2(test.predicted, test.observed)
  test.mbe <- mean(test.predicted - test.observed)
  
  # Return the final model and performance metrics
  list(
    model = model,
    train.results = list(
      rmse = train.rmse,
      r2 = train.r2,
      mbe = train.mbe
    ),
    test.results = list(
      rmse = test.rmse,
      r2 = test.r2,
      mbe = test.mbe,
      predictions = test.predicted,
      observations = test.observed
    )
  )
}

# ==== 6. Build the Models ====
all.features <- df
lasso.features <- df %>%
  select(all_of(lasso.features), "soil.moisture", "date")
rf.features <- df %>%
  select(all_of(rf.features), "soil.moisture", "date")

## ==== 6.1. Random Forest ====
rf.models <- list(
  all = build.model(all.features, testing.indices, "rf"),
  lasso = build.model(lasso.features, testing.indices, "rf"),
  rf = build.model(rf.features, testing.indices, "rf")
)

## ==== 6.2. XGBoost ====
xgb.models <- list(
  all = build.model(all.features, testing.indices, "xgbTree"),
  lasso = build.model(lasso.features, testing.indices, "xgbTree"),
  rf = build.model(rf.features, testing.indices, "xgbTree")
)

## ==== 6.3. Gaussian Process Regression ====
gpr.models <- list(
  all = build.model(all.features, testing.indices, "gaussprRadial"),
  lasso = build.model(lasso.features, testing.indices, "gaussprRadial"),
  rf = build.model(rf.features, testing.indices, "gaussprRadial")
)


# ==== 7. Define Process for Graphing Results ====
plot.lims <- c(min(all.features$soil.moisture), max(all.features$soil.moisture))
plot.results <- function(results) {
  # Create labels for performance metrics
  performance.labs <- c(
    paste("R\u00B2:", round(results$r2, 3)),
    paste("RMSE:", round(results$rmse, 3), "(m\u00B3/m\u00B3)"),
    paste("MBE:", round(results$mbe, 3), "(m\u00B3/m\u00B3)")
  ) %>%
    paste(collapse = "\n")
  
  # Fit line to predicted values
  predicted.fit <- lm(results$observations ~ results$predictions)
  fit.slope <- predicted.fit$coefficients[2]
  fit.intercept <- predicted.fit$coefficients[1]
  
  # Final plot
  ggplot(mapping = aes(x = results$predictions, y = results$observations)) +
    geom_point() +
    # Compare perfect fit to predicted fit
    geom_abline(slope = 1, intercept = 0, color = "red") +
    geom_abline(slope = fit.slope, intercept = fit.intercept,
                color = "darkgrey", linetype = "dashed") +
    # Aesthetics
    coord_cartesian(xlim = plot.lims, ylim = plot.lims) +
    labs(x = NULL, y = NULL) +
    geom_label(
      aes(x = plot.lims[1], y = plot.lims[2], label = performance.labs),
      fill = "skyblue",
      color = "black",
      hjust = 0,
      vjust = 1,
      size = 3,
      alpha = 0.7
    ) +
    theme_classic() +
    theme(panel.grid = element_blank())
}

# ==== 8. Plot the Results of the Models ====
graphs <- list(
  rf.all = plot.results(rf.models$all$test.results),
  rf.lasso = plot.results(rf.models$lasso$test.results),
  rf.rf = plot.results(rf.models$rf$test.results),
  xgb.all = plot.results(xgb.models$all$test.results),
  xgb.lasso = plot.results(xgb.models$lasso$test.results),
  xgb.rf = plot.results(xgb.models$rf$test.results),
  gpr.all = plot.results(gpr.models$all$test.results),
  gpr.lasso = plot.results(gpr.models$lasso$test.results),
  gpr.rf = plot.results(gpr.models$rf$test.results)
)

## ==== 8.1. Display in a 3x3 Plot ====
# Row Labels
row.labs <- list("Random Forest", "XGBoost", "Gaussian PR") %>%
  map(~ wrap_elements(
    full = textGrob(.x, rot = 90, gp = gpar(fontsize = 14)),
    ignore_tag = TRUE
  ))

# Column Labels
col.labs <- list("All Features", "LASSO Selected Features",
                 "RF Selected Features") %>%
  map(~ wrap_elements(
    full = textGrob(.x, gp = gpar(fontsize = 14)),
    ignore_tag = TRUE
  ))

# Define rows
row1 <- row.labs[[1]] + graphs$rf.all + graphs$rf.lasso + graphs$rf.rf +
  plot_layout(widths = c(0.1, 1, 1, 1))
row2 <- row.labs[[2]] + graphs$xgb.all + graphs$xgb.lasso + graphs$xgb.rf +
  plot_layout(widths = c(0.1, 1, 1, 1))
row3 <- row.labs[[3]] + graphs$gpr.all + graphs$gpr.lasso + graphs$gpr.rf +
  plot_layout(widths = c(0.1, 1, 1, 1))

# Define column
col <- plot_spacer() + col.labs[[1]] + col.labs[[2]] + col.labs[[3]] +
  plot_layout(widths = c(0.1, 1, 1, 1))

# Final graph
final.plot <- (
  row1 /
    row2 /
    row3 /
    col
) +
  plot_layout(
    widths = c(0.1, 1, 1, 1),
    heights = c(1, 1, 1, 0.1)
  ) +
  plot_annotation(
    title = "Observed vs. Model Predicted Soil Moisture (m\u00B3/m\u00B3)",
    tag_levels = "A",
    tag_prefix = "(",
    tag_suffix = ")",
    theme = theme(
      plot.title = element_text(hjust = 0.5, size = 18, face = "bold")
    )
  )

final.plot

# ==== 9. Assess Variable Importance ====
# Define variable categories
vars <- colnames(all.features)
categories <- list(
  soil = c("bd2", "bd3", "clay3", "sand2", "sand3", "soc2", "soc3"),
  climate = c("tavg_t1", "tavg_t3", "tavg_t5", "tavg_t7",
              "dew_point_t1", "dew_point_t3", "dew_point_t5", "dew_point_t7",
              "api_t1", "api_t3", "api_t5", "api_t7",
              "et_t1", "et_t3", "et_t5", "et_t7",
              "prcp_t1", "prcp_t3", "prcp_t5", "prcp_t7",
              "relative_humidity_t1", "relative_humidity_t3",
              "relative_humidity_t5", "relative_humidity_t7",
              "vpd_t1", "vpd_t3", "vpd_t5", "vpd_t7",
              "wind_speed_t1", "wind_speed_t3", "wind_speed_t5",
              "wind_speed_t7"),
  biota = c("B1", "B11", "B11_stdDev", "B12", "B12_stdDev", "B1_stdDev", "B2",
            "B2_stdDev", "B3", "B3_stdDev", "B4", "B4_stdDev", "B5",
            "B5_stdDev", "B6", "B6_stdDev", "B7", "B7_stdDev", "B8",
            "B8_stdDev", "B8A", "B8A_stdDev", "B9", "B9_stdDev", "EVI",
            "EVI_stdDev", "IRECI", "IRECI_stdDev", "NDMI", "NDMI_stdDev",
            "NDVI", "NDVI_stdDev", "NDWI", "NDWI_stdDev", "S2REP",
            "S2REP_stdDev", "LST", "n.trees", "tree.carbon", "tree.agb", "SAVI",
            "mNDWI", "BI", "GNDVI", "SAVI_stdDev", "mNDWI_stdDev", "BI_stdDev",
            "GNDVI_stdDev", "avg.tree.height", "avg.tree.diameter"),
  topography = c("AS", "EL", "FLOW", "PLAN", "PROF", "RG", "SL",
                 "TAN", "TWI", "TPI", "HS"),
  age = c("sin.doy", "cos.doy")
)

var.categories <- rep(NA, length(vars))
var.categories[vars %in% categories$soil] <- "Soil"
var.categories[vars %in% categories$climate] <- "Climate"
var.categories[vars %in% categories$biota] <- "Biota"
var.categories[vars %in% categories$topography] <- "Topography"
var.categories[vars %in% categories$age] <- "Age"
var.categories <- factor(var.categories, levels = c("Soil", "Climate", "Biota",
                                                    "Topography", "Age"))

# Function to calculate variable importance
# Returns normalized scales from FIRM method
model.vi <- function(model) {
  vi(model, method = "firm") %>%
    mutate(Importance = Importance / sum(Importance))
}

var.imp <- list(
  rf.all = model.vi(rf.models$all$model),
  rf.lasso = model.vi(rf.models$lasso$model),
  rf.rf = model.vi(rf.models$rf$model),
  xgb.all = model.vi(xgb.models$all$model),
  xgb.lasso = model.vi(xgb.models$lasso$model),
  xgb.rf = model.vi(xgb.models$rf$model),
  gpr.all = model.vi(gpr.models$all$model),
  gpr.lasso = model.vi(gpr.models$lasso$model),
  gpr.rf = model.vi(gpr.models$rf$model)
)

## ==== 10.1. Plot Variable Importance ====
# Function to plot importance (top n features)
plot.imp <- function(var.imp, n, show.legend = FALSE) {
  # Make a factor for proper sorting
  var.imp$Variable <- factor(var.imp$Variable, levels = rev(var.imp$Variable))
  var.imp$Category <- var.categories[match(var.imp$Variable, vars)]
  
  p <- ggplot(var.imp[1:min(n, nrow(var.imp)), ],
              aes(x = Variable, y = Importance, fill = Category)) +
    geom_bar(stat = "identity", show.legend = TRUE) +
    labs(x = NULL) +
    coord_flip() +
    scale_fill_manual(values = c(
      "Soil" = "tomato",
      "Climate" = "goldenrod1",
      "Biota" = "palegreen3",
      "Topography" = "steelblue",
      "Age" = "purple3"
    ), drop = FALSE) +
    theme_light()
  
  if (show.legend) {
    p <- p + theme(
      legend.position = c(.95, .05),
      legend.justification = c("right", "bottom"),
      legend.box.just = "right",
      legend.key.size = unit(10, "pt")
    )
  } else {
    p <- p + theme(
      legend.position = "none"
    )
  }
  
  p
}

# Create set of plots for all models
N <- 10
rf.varImpPlots <- list(
  all = plot.imp(var.imp$rf.all, N),
  lasso = plot.imp(var.imp$rf.lasso, N),
  rf = plot.imp(var.imp$rf.rf, N)
)

xgb.varImpPlots <- list(
  all = plot.imp(var.imp$xgb.all, N),
  lasso = plot.imp(var.imp$xgb.lasso, N),
  rf = plot.imp(var.imp$xgb.rf, N)
)

gpr.varImpPlots <- list(
  all = plot.imp(var.imp$gpr.all, N),
  lasso = plot.imp(var.imp$gpr.lasso, N),
  rf = plot.imp(var.imp$gpr.rf, N, TRUE)
)

# Arrange Final Variable Importance Plots
# Define rows
row1 <- row.labs[[1]] + rf.varImpPlots$all + rf.varImpPlots$lasso +
  rf.varImpPlots$rf +
  plot_layout(widths = c(0.1, 1, 1, 1))
row2 <- row.labs[[2]] + xgb.varImpPlots$all + xgb.varImpPlots$lasso +
  xgb.varImpPlots$rf +
  plot_layout(widths = c(0.1, 1, 1, 1))
row3 <- row.labs[[3]] + gpr.varImpPlots$all + gpr.varImpPlots$lasso +
  gpr.varImpPlots$rf +
  plot_layout(widths = c(0.1, 1, 1, 1))

# Define column
col <- plot_spacer() + col.labs[[1]] + col.labs[[2]] + col.labs[[3]] +
  plot_layout(widths = c(0.1, 1, 1, 1))

# Final graph
final.varImp.plot <- (
  row1 /
    row2 /
    row3 /
    col
) +
  plot_layout(
    widths = c(0.1, 1, 1, 1),
    heights = c(1, 1, 1, 0.1)
  ) +
  plot_annotation(
    title = "Variable Importance (Top 10 Features)",
    tag_levels = "A",
    tag_prefix = "(",
    tag_suffix = ")",
    theme = theme(
      plot.title = element_text(hjust = 0.5, size = 18, face = "bold")
    )
  )

final.varImp.plot
