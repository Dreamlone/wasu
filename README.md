# wasu

Water Supply Forecast competition model

## How to use this repository

This repository contains code both for preparing visualisations and for training and ap
plying predictive models.

It is recommended to start exploration with [examples](./examples) folder:

- [1_basic](./examples/1_basic) - basic scripts that prepare exploratory data visualisations
- [2_repeating](./examples/2_repeating) - repeating of last known values to generate forecasts (in two versions)
- [3_streamflow](./examples/3_streamflow) - use of aggregated statistics from USGS streamflow data 


During code execution the `plots` folder is generated. 

For example, there you can find zones for which forecast models are initialized:

![spatial_extend.png](examples%2Fplots%2Fspatial%2Fspatial_extend.png)

## Algorithms description 

This section provides explanations that explain how the algorithms work 

### Simple repeating 

In progress 

### Streamflow-based predictions

In progress 

![2005-01-22 00:00:00_site_hungry_horse_reservoir_inflow.png](examples%2Fplots%2Fusgs_streamflow_3d%2F2005-01-22%2000%3A00%3A00_site_hungry_horse_reservoir_inflow.png)

Figure. 3d plot features (`min_value` and `mean_value`), `predictions` (surface) and `actual` values (points) for training sample for
desired issue date

### SNOTEL-based predictions

![spatial_extend_snotel_fontenelle_reservoir_inflow.png](examples%2Fplots%2Fspatial_with_snotel_stations%2Fspatial_extend_snotel_fontenelle_reservoir_inflow.png)

Figure. SNOTEL stations and basin of `fontenelle_reservoir_inflow` site