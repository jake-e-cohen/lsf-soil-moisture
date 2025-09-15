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

# ==== 1. Investigate Spatial Autocorrelation of Soil Moisture ====
# Since we have multiple values at each location, we have to summarize them
spatial.sm <- df %>%
  select(X, Y, soil.moisture) %>%
  group_by(X, Y) %>%
  mutate(
    min.sm = min(soil.moisture),
    med.sm = median(soil.moisture),
    max.sm = max(soil.moisture),
    mean.sm = mean(soil.moisture)
  ) %>%
  distinct(X, Y, .keep_all = TRUE)

# Try different variogram fits
coordinates(spatial.sm) <- ~ X + Y
vgms <- list(
  min = variogram(min.sm ~ 1, spatial.sm),
  med = variogram(med.sm ~ 1, spatial.sm),
  max = variogram(max.sm ~ 1, spatial.sm),
  mean = variogram(mean.sm ~ 1, spatial.sm)
)

fits <- list(
  min = fit.variogram(vgms$min, model = vgm("Sph")),
  med = fit.variogram(vgms$med, model = vgm("Sph")),
  max = fit.variogram(vgms$max, model = vgm("Sph")),
  mean = fit.variogram(vgms$mean, model = vgm("Sph"))
)

# Use the average range as the range of autocorrelation
autocorr.range <- mean(sapply(fits, function(fit) fit$range[2])) %>% round()

## ==== 1.1. Plot Variogram Fits ====
vgm.lines <- list(
  min = variogramLine(fits$min, maxdist = max(vgms$min$dist)),
  med = variogramLine(fits$med, maxdist = max(vgms$med$dist)),
  max = variogramLine(fits$max, maxdist = max(vgms$max$dist)),
  mean = variogramLine(fits$mean, maxdist = max(vgms$mean$dist))
) %>%
  bind_rows(.id = "fit_id")

ggplot(vgm.lines, aes(x = dist, y = gamma, color = fit_id, group = fit_id)) +
  geom_line() +
  labs(
    x = "Distance (m)",
    y = "Semivariance (m\u00B3/m\u00B3)",
    color = "Variogram Fit"
  ) +
  geom_vline(xintercept = autocorr.range, linetype = "dashed", color = "gray") +
  scale_x_continuous(expand = c(0, 0), breaks = seq(0, 50, 5)) +
  scale_color_manual(values = c("red", "goldenrod1", "forestgreen", "blue")) +
  theme_classic()

# ==== 2. Remove Variables That Only Vary Temporally ====
cols <- colnames(df)[colnames(df) != "date"]

spatially.static.vars <- df %>%
  group_by(date) %>%
  mutate(across(all_of(cols), var)) %>%
  ungroup() %>%
  select(-date) %>%
  select(where(~ sum(.) == 0)) %>%
  colnames()

df <- df %>%
  select(-all_of(spatially.static.vars), -date)

# ==== 3. Investigate Multicollinearity ====
# Pearson correlation coefficients between variables
cor.matrix <- cor(df)

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

# ==== 4. Create Spatially Distinct Training/Testing Folds ====
sm.sf <- st_as_sf(df, coords = c("longitude", "latitude"), crs = 4326)
sm.sf.reproj <- st_transform(sm.sf, crs = 32618)

n.folds <- 10
blocks <- cv_spatial(
  sm.sf.reproj,
  size = autocorr.range,
  k = n.folds,
  selection = "systematic",
  flat_top = TRUE,
  plot = FALSE
)
block.folds <- blocks$folds_list %>%
  lapply(function(fold) {
    list(
      train = fold[[1]],
      test = fold[[2]]
    )
  })

# For each fold, remove training points within autocorrelation range
block.folds <- lapply(block.folds, function(fold) {
  # Define locations of training and testing points
  training.locs <- df[fold$train, ] %>%
    select(X, Y)
  testing.locs <- df[fold$test, ] %>%
    select(X, Y)
  
  # Compute Euclidean distance between all points
  distances <- bind_rows(training.locs, testing.locs) %>%
    dist() %>%
    as.matrix()
  
  # Extract distances between testing and training points only
  dist.test.to.train <- distances[
    (nrow(training.locs) + 1):(nrow(training.locs) + nrow(testing.locs)),
    1:nrow(training.locs)
  ]
  
  # Find indices of training data that are within the autocorrelation range
  in.range <- apply(dist.test.to.train, 2, function(distance) {
    any(distance < autocorr.range)
  })
  
  # Return the filtered fold
  list(
    train = fold$train[which(!in.range)],
    test = fold$test
  )
})

# Hold out one fold for testing
test.fold <- block.folds[[1]]$test
block.folds <- lapply(block.folds, function(fold) {
  list(
    train = setdiff(fold$train, test.fold),
    test = setdiff(fold$test, test.fold)
  )
})
block.folds <- block.folds[-1]

# Plot the resulting spatial split
ggplot() +
  # The blocks
  geom_sf(data = blocks$blocks$blocks,
          aes(fill = as.factor(blocks$blocks$folds)),
          color = "transparent") +
  # The sampling points
  geom_sf(data = sm.sf.reproj, size = 2) +
  labs(x = NULL, y = NULL) +
  scale_fill_brewer(palette = "Set3", name = "Fold") +
  theme_minimal() +
  theme(title = element_text(hjust = 0.5, size = 14, face = "bold"))

# ==== 5. Perform Feature Selection ====
## ==== 5.1. LASSO Regression ====
# Train/Test Split
predictors <- df %>%
  select(-soil.moisture, -X, -Y, -latitude, -longitude)
sm <- df$soil.moisture

x.train <- predictors[-test.fold, ]
x.test <- predictors[test.fold, ]
Y.train <- sm[-test.fold]
Y.test <- sm[test.fold]

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

# ==== 5.2. RF Regression ====
n.trees <- 100
train.data <- df[-test.fold, ] %>% select(-X, -Y, -latitude, -longitude)
rf.full <- randomForest(
  soil.moisture ~ .,
  data = train.data,
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
    
    model.data.temp <- train.data %>%
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

# ==== 6. Define Process to Train Models ====
build.model <- function(data, folds, method, do.kriging) {

  # Separate training and testing folds
  train.folds <- lapply(folds, function(fold) fold$train)
  test.folds <- lapply(folds, function(fold) fold$test)
  
  # Configure model
  training.controls <- trainControl(
    method = "cv",
    number = length(folds),
    index = train.folds,
    indexOut = test.folds
  )
  
  # Preprocess if necessary
  preProc <- if (grepl("gausspr", method)) c("center", "scale") else NULL
  
  # Perform CV to tune model parameters
  cv.model <- train(
    soil.moisture ~ .,
    data = data %>% select(-X, -Y),
    method = method,
    trControl = training.controls,
    metric = "RMSE",
    preProcess = preProc
  )
  
  # Build the final model on all training data
  training.indices <- unique(unlist(train.folds))
  all.training.data <- data[training.indices, ]
  
  model <- train(
    soil.moisture ~ .,
    data = all.training.data %>% select(-X, -Y),
    method = method,
    trControl = trainControl(method = "none"),
    tuneGrid = cv.model$bestTune[rep(1, 1), , drop = FALSE],
    preProcess = preProc
  )
  
  # Compute metrics on trained model
  train.predicted <- predict(model, all.training.data)
  train.observed <- all.training.data$soil.moisture
  
  train.rmse <- RMSE(train.predicted, train.observed)
  train.r2 <- R2(train.predicted, train.observed)
  train.mbe <- mean(train.predicted - train.observed)
  
  # Make predictions on testing data
  data.test <- data[test.fold, ]
  test.predicted <- predict(model, data.test)
  test.observed <- data.test$soil.moisture
  
  # Compute final performance metrics 
  test.rmse <- RMSE(test.predicted, test.observed)
  test.r2 <- R2(test.predicted, test.observed)
  test.mbe <- mean(test.predicted - test.observed)
  
  # 4.11: Return the final model and performance metrics
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

# ==== 7. Build the Models ====
# Configuration
do.kriging <- FALSE
all.features <- df %>%
  select(-latitude, -longitude)
lasso.features <- df %>%
  select(c(lasso.features, "soil.moisture", "X", "Y"))
rf.features <- df %>%
  select(c(rf.features, "soil.moisture", "X", "Y"))

## ==== 7.1. Random Forest ====
rf.models <- list(
  all = build.model(all.features, block.folds, "rf", do.kriging),
  lasso = build.model(lasso.features, block.folds, "rf", do.kriging),
  rf = build.model(rf.features, block.folds, "rf", do.kriging)
)

## ==== 7.2. XGBoost ====
xgb.models <- list(
  all = build.model(all.features, block.folds, "xgbTree", do.kriging),
  lasso = build.model(lasso.features, block.folds, "xgbTree", do.kriging),
  rf = build.model(rf.features, block.folds, "xgbTree", do.kriging)
)

## ==== 7.3. Gaussian Process Regression ====
gpr.models <- list(
  all = build.model(all.features, block.folds, "gaussprRadial", do.kriging),
  lasso = build.model(lasso.features, block.folds, "gaussprRadial", do.kriging),
  rf = build.model(rf.features, block.folds, "gaussprRadial", do.kriging)
)

# ==== 8. Define Process For Graphing Results ====
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

# ==== 9. Plot the Results of the Models ====
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

## ==== 9.1. Display in a 3x3 plot ====
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

# ==== 10. Assess Variable Importance ====
# Define variable categories
vars <- colnames(all.features %>% select(-X, -Y))
categories <- list(
  soil = c("bd2", "bd3", "clay3", "sand2", "sand3", "soc2", "soc3"),
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
                 "TAN", "TWI", "TPI", "HS")
)

var.categories <- rep(NA, length(vars))
var.categories[vars %in% categories$soil] <- "Soil"
var.categories[vars %in% categories$biota] <- "Biota"
var.categories[vars %in% categories$topography] <- "Topography"
var.categories <- factor(var.categories, levels = c("Soil", "Biota",
                                                    "Topography"))

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
      "Biota" = "palegreen3",
      "Topography" = "steelblue"
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
