# ==== Imports ====
library(jsonlite)
library(tidyverse)

# ==== Set-up ====
setwd("/Users/jakecohen/Desktop/Final Code Repository/")

# ==== 1. Import Data ====
sm <- read_csv("src/soil_moisture.csv")
trees <- read_csv("src/trees.csv")
topography <- read_csv("src/topography.csv")

# Import LST downscaling regression model
lst.model <- load("src/LST_downscaling_model.RData") %>% get()

## ==== 1.1. Combine GEE Data ====
dates <- c("2025-06-05", "2025-06-11", "2025-06-20", "2025-06-26", "2025-07-03",
           "2025-07-16", "2025-07-23")
gee <- dates %>%
  lapply(function(date) {
    df <- read_csv(paste0("src/gee/", date, ".csv")) %>%
      bind_cols(date = rep(date, nrow(.)))
  }) %>%
  reduce(bind_rows)

# ==== 2. Configure Soil Moisture Data ====
sm <- sm %>%
  select(
    date = Date,
    coordinate = Coordinate,
    soil.moisture = `Soil Moisture (VWC)`
  ) %>%
  mutate(
    date = as.character(date),
    coordinate = str_remove_all(coordinate, "[\\(\\)]"),
    soil.moisture = soil.moisture / 100    # scale to %
  )

# ==== 3. Configure GEE Data ====
## ==== 3.1. Derive Latitude and Longitude from Geometry ====
gee <- gee$.geo %>%
  lapply(fromJSON) %>%
  sapply(function(geometry) geometry[["coordinates"]]) %>%
  t() %>%
  as.data.frame() %>%
  rename(longitude = V1, latitude = V2) %>%
  bind_cols(gee)

## ==== 3.2. Encode Date as a Continuous Variable ====
gee <- gee %>%
  mutate(
    sin.doy = sin(yday(date) * 2 * pi / 365)
  ) %>%
  rename(coordinate = Code)

## ==== 3.3. Remove Data Irrelevant for Modelling ====
gee <- gee %>%
  select(-.geo, -`system:index`, -CreationDa, -Creator, -EditDate, -Editor,
         -Extra, -GlobalID, -Id)

## ==== 3.4. Add LST Data Predicted By Model ====
gee <- gee %>%
  select(B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12, EVI, NDVI, NDMI,
         temp = tavg_t1, longitude, latitude, sin.doy) %>%
  predict(lst.model, newdata = .) %>%
  bind_cols(gee, LST = .)

# ==== 4. Configure Tree Data ====
## ==== 4.1. Remove Dead Trees ====
trees <- trees %>%
  filter(!grepl("DIED", NAME))

## ==== 4.2. Add Tree Data to GEE Table ====
gee <- gee %>%
  rename(tree.codes = trees) %>%
  mutate(
    tree.codes = lapply(tree.codes, fromJSON) %>% lapply(as.numeric),
    n.trees = sapply(tree.codes, length),
    tree.carbon = tree.codes %>% map_dbl(
      ~ trees %>%
        filter(CODE %in% .x) %>%
        pull(`CARBON (kg)`) %>%
        sum()
    ),
    tree.agb = tree.codes %>% map_dbl(
      ~ trees %>%
        filter(CODE %in% .x) %>%
        pull(`AGB (kg)`) %>%
        sum()
    ),
    avg.tree.height = tree.codes %>% map_dbl(
      ~ trees %>%
        filter(CODE %in% .x) %>%
        pull(`HEIGHT (m)`) %>%
        mean()
    ) %>%
      replace(is.na(.), 0),
    avg.tree.diameter = tree.codes %>% map_dbl(
      ~ trees %>%
        filter(CODE %in% .x) %>%
        pull(`DIAMETER (cm)`) %>%
        mean()
    ) %>%
      replace(is.na(.), 0)
  ) %>%
  select(-tree.codes)

# ==== 5. Merge Final Data Table ====
df <- gee %>%
  left_join(sm, by = c("coordinate", "date")) %>%
  left_join(topography, by = "coordinate")

# ==== 6. Make Final Modifications ====
## ==== 6.1. Remove Any Rows with Null Values ====
df <- df[complete.cases(df), ]

## ==== 6.2. Remove Columns without Variation ====
df <- df[, sapply(df, function(col) {
  length(unique(col)) > 1
})]

## ==== 6.3. Split up Coordinates into X and Y Columns ====
df <- df$coordinate %>%
  sapply(str_split, ",") %>%
  sapply(as.numeric) %>%
  t() %>%
  as.data.frame() %>%
  rename(X = V1, Y = V2) %>%
  bind_cols(df, .) %>%
  subset(select = -coordinate)

rownames(df) <- NULL

# ==== 7. Export Final Data Frame ====
write_csv(df, "src/final_data.csv")
