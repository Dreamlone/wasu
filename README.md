![wasu_logo.png](docs%2Fwasu_logo.png)

Water Supply Forecast competition model

Competition URL: [Water Supply Forecast Rodeo](https://www.drivendata.org/competitions/group/reclamation-water-supply-forecast/)

This repository contains all the necessary materials to reproduce the results of the competition. 
This module can also be seen as a sandbox for experimentation

## How to use this repository

This repository contains code both for preparing visualisations 
and for training and applying predictive models.

Load the data and put it into data `folder`: 

- `grace_indicators` folder: (FY2009, FY2010, ...)
- `pdsi` folder: (FY2009, FY2010, ...)
- `snodas` folder: (FY2009, FY2010, ...)
- `teleconnections` folder: (mjo.txt, nino_regions_sst.txt, oni.txt, pdo.txt, pna.txt, soi.txt)
- `usgs_streamflow` folder: (FY1990, FY1991, ...)
- `geospatial.gpkg`
- `metadata_TdPVeJC.csv`
- `submission_format.csv`
- `test_monthly_naturalized_flow.csv`
- `train.csv`
- `train_monthly_naturalized_flow.csv`

After that repository is ready for experiments and data exploration
It is recommended to start exploration with [examples](./examples) folder:

- [1_basic](./examples/1_basic) - basic scripts that prepare exploratory data visualisations
- [2_repeating](./examples/2_repeating) - repeating of last known values to generate forecasts (in two versions)
- [3_streamflow](./examples/3_streamflow) - use of aggregated statistics from USGS streamflow data 
- [4_snotel](./examples/4_snotel) - use of aggregated statistics from snowpack SNOTEL data 
- [5_simple_ensemble](./examples/5_simple_ensemble) - ensemble of previous forecasts from data sources
- [6_simple_ensemble_with_smoothing](./examples/6_simple_ensemble_with_smoothing) - ensembling with smoothing
- [7_snodas](./examples/7_snodas) - use aggregated statistics of SNODAS (snow gridded) data
- [8_teleconnections](./examples/8_teleconnections) - teleconnections with snotel data
- [9_common](./examples/9_common) - complex model which use SNOTEL and PDSI data to generate predictions
- [10_common_experiment](./examples/10_common_experiment) - set of functions to provide hyperparameters search space exploration for common model 

In the folders with submit prefix placed the code for execution stage (including serialized models).

During code execution the `plots` folder is generated. 

For example, there you can find zones for which forecast models are initialized:

![spatial_extend.png](examples%2Fplots%2Fspatial%2Fspatial_extend.png)

Figure 1. Spatial polygons for river basins

## Algorithms description 

This section provides explanations that explain how the algorithms work 

### Simple repeating 

Validation years: `2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023`

- **MAE metric**: 396.65
- **MAPE metric**: 56.50
- **Symmetric MAPE metric**: 61.37
- **Quantile loss metric**: 367.66
- **Quantile loss metric (only for 0.5 quantile)**: 396.65

The simplest possible algorithm.
For provided above validation years the algorithm takes the value from 2015 and assigns it to each subsequent year.

![animas_r_at_durango_time_series_plot.png](examples%2Fplots%2Fpredictions_simple_repeating%2Fanimas_r_at_durango_time_series_plot.png)

Figure 2. Forecasts for tests years for site `animas_r_at_durango` using simple 
repeating since 2015

### Advanced repeating 

Validation years: `2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023`

- **MAE metric**: 386.02
- **MAPE metric**: 59.98
- **Symmetric MAPE metric**: 52.97
- **Quantile loss metric**: 275.82
- **Quantile loss metric (only for 0.5 quantile)**: 386.02

Uses values from the previous year for this site to be used as a forecast. 
That is, for 2005 the year 2004 will be used, for 2007 the year 2006 will be used, etc.

![animas_r_at_durango_time_series_plot.png](examples%2Fplots%2Fpredictions_advanced_repeating%2Fanimas_r_at_durango_time_series_plot.png)

Figure 3. Forecasts for tests years for site `animas_r_at_durango` using 
advanced repeating 

### Streamflow-based predictions

Validation years: `2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023`

|                  **Metric**                  | **Aggregation days 40** | **Aggregation days 80** | **Aggregation days 120** |
|:--------------------------------------------:|:-----------------------:|:-----------------------:|:------------------------:|
|                  MAE metric                  |         289.88          |         286.48          |        **286.45**        |
|                 MAPE metric                  |          43.60          |          43.72          |        **43.56**         |
|            Symmetric MAPE metric             |          38.61          |          38.77          |        **38.75**         |
|             Quantile loss metric             |         195.34          |         194.55          |        **193.70**        |
| Quantile loss metric (only for 0.5 quantile) |         289.88          |         286.48          |        **286.45**        |

This approach uses flow values aggregated over a 
specific period (for example 40, 80 or 120 days before the forecast issue date) 
to generate a forecast into the future. 

![animas_r_at_durango_time_series_plot.png](examples%2Fplots%2Fusgs_streamflow_plots%2Fanimas_r_at_durango_time_series_plot.png)

Figure 4. Representation of USGS streamflow data for `animas_r_at_durango` and actual values

![virgin_r_at_virtin_time_series_plot.png](examples%2Fplots%2Fpredictions_usgs_streamflow%2Fvirgin_r_at_virtin_time_series_plot.png)

Figure 5. Forecasts for tests years for site `virgin_r_at_virtin` using USGS 
streamflow based model (aggregation days: 120, kernel model - `QuantileRegressor`)

### SNOTEL-based predictions

|                  **Metric**                  | **Aggregation days 40** | **Aggregation days 80** | **Aggregation days 120** |
|:--------------------------------------------:|:-----------------------:|:-----------------------:|:------------------------:|
|                  MAE metric                  |       **238.13**        |         315.47          |          303.89          |
|                 MAPE metric                  |        **40.60**        |          51.64          |          49.86           |
|            Symmetric MAPE metric             |        **33.19**        |          36.84          |          36.14           |
|             Quantile loss metric             |       **151.71**        |         186.44          |          186.35          |
| Quantile loss metric (only for 0.5 quantile) |       **238.13**        |         315.47          |          303.89          |

Key features description: 

- `PREC_DAILY` - precipitation
- `TAVG_DAILY` - average daily temperature
- `TMAX_DAILY` - max temperature
- `TMIN_DAILY` - min temperature
- `WTEQ_DAILY` - snow water equivalent	

![spatial_extend_snotel_animas_r_at_durango.png](examples%2Fplots%2Fspatial_with_snotel_stations%2Fspatial_extend_snotel_animas_r_at_durango.png)

Figure 6. SNOTEL stations and basin of `fontenelle_reservoir_inflow` site

![virgin_r_at_virtin_time_series_plot.png](examples%2Fplots%2Fpredictions_snotel%2Fvirgin_r_at_virtin_time_series_plot.png)

Figure 7. Forecasts for tests years for site `virgin_r_at_virtin` using SNOTEL stations
based model (aggregation days: 40, kernel model - `QuantileRegressor`)

### Ensembling of previous predictions

Validation years: `2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023`

- **MAE metric**: 243.06
- **MAPE metric**: 39.85
- **Symmetric MAPE metric**: 33.93
- **Quantile loss metric**: 167.40
- **Quantile loss metric (only for 0.5 quantile)**: 243.06

Combination of USGS streamflow -based model prediction and SNOTEL -based prediction

![hungry_horse_reservoir_inflow_time_series_plot.png](examples%2Fplots%2Fpredictions_simple_ensemble%2Fhungry_horse_reservoir_inflow_time_series_plot.png)

Figure 8. Forecasts for tests years for site `hungry_horse_reservoir_inflow` using simple ensemble

### Ensembling of previous predictions (with smoothing)

Validation years: `2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023`

- **MAE metric**: 240.41
- **MAPE metric**: 39.60
- **Symmetric MAPE metric**: 33.63
- **Quantile loss metric**: 165.20
- **Quantile loss metric (only for 0.5 quantile)**: 240.41

![hungry_horse_reservoir_inflow_time_series_plot.png](examples%2Fplots%2Fpredictions_simple_ensemble_smoothing%2Fhungry_horse_reservoir_inflow_time_series_plot.png)

Figure 9. Forecasts for tests years for site `hungry_horse_reservoir_inflow` using simple ensemble with smoothing

### SNODAS-based model

|                  **Metric**                  | **Aggregation days 40** | **Aggregation days 80** | **Aggregation days 120** |
|:--------------------------------------------:|:-----------------------:|:-----------------------:|:------------------------:|
|                  MAE metric                  |         220.73          |         216.84          |        **215.10**        |
|                 MAPE metric                  |          36.13          |          32.95          |        **32.26**         |
|            Symmetric MAPE metric             |          31.36          |          29.99          |        **30.22**         |
|             Quantile loss metric             |         146.95          |         140.51          |        **137.92**        |
| Quantile loss metric (only for 0.5 quantile) |         220.73          |         216.84          |        **215.10**        |


Modeled snow layer thickness, total of snow layers

Data from SNODAS files:
- Non-snow accumulation, 24-hour total
- Snow accumulation, 24-hour total
- Modeled snow layer thickness, total of snow layers
- Modeled average temperature, SWE-weighted average of snow layers, 24-hour average
- Modeled blowing snow sublimation, 24-hour total
- Modeled melt, bottom of snow layers, 24-hour total
- Modeled snowpack sublimation, 24-hour total

Data preprocessing for SNODAS is divided into two steps: 

1. **Archive unpacking**: Archives with `.dat` and `.txt` files are transformed into geotiff files
2. **Data extraction**: For each site id and for each datetime stamp, information is extracted and written as a `.csv` file

![snow_accumulation.gif](docs%2Fimages%2Fsnow_accumulation.gif)

Animation 1. Snow accumulation per days for site `hungry_horse_reservoir_inflow`. Units: `Kilograms per square meter / 10`

![animas_r_at_durango_time_series_plot.png](examples%2Fplots%2Fpredictions_snodas%2Fanimas_r_at_durango_time_series_plot.png)

Figure 10. Forecasts for tests years for site `animas_r_at_durango` using SNODAS-based model (aggregation days: 120, kernel model - `QuantileRegressor`)

### Complex model ver 1

Validation years: `2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023`

- **MAE metric**: 214.44
- **MAPE metric**: 31.16
- **Symmetric MAPE metric**: 30.54
- **Quantile loss metric**: 137.65
- **Quantile loss metric (only for 0.5 quantile)**: 214.44
- 
### Complex model ver 2

Validation years: `2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023`

- **MAE metric**: 197.09
- **MAPE metric**: 30.71
- **Symmetric MAPE metric**: 29.03
- **Quantile loss metric**: 132.90
- **Quantile loss metric (only for 0.5 quantile)**: 197.09
- 
Since SNODAS and SNOTEL data are compatible in the terms of feature engineering is was decided to use only SNOTEL 
data because it is much easier to process (Figure 11)

![pueblo_reservoir_inflow_ts_mean_Modeled snow water equivalent, total of snow layers.png](examples%2Fplots%2Fsnodas_investigation%2Fpueblo_reservoir_inflow_ts_mean_Modeled%20snow%20water%20equivalent%2C%20total%20of%20snow%20layers.png)

Figure 11. SNODAS and SNOTEL data comparison vs target for site `pueblo_reservoir_inflow`

The graph shows that snowpack does not completely determine target. 
Therefore, it was decided to include an additional parameter, PDSI, 
in the model to account for soil characteristics. 


### Complex model (optimized)

Common model. Final model metrics. 

- **MAE metric**: 177.17
- **MAPE metric**: 27.97
- **Symmetric MAPE metric**: 26.30
- **Quantile loss metric**: 120.78
- **Quantile loss metric (only for 0.5 quantile)**: 177.17

To find the optimal configuration of hyperparameters (`days SNOTEL short`, `days SNOTEL long`, `days PDSI`), a brute force algorithm 
was applied. The figures below show the results of 
calculations for two loss functions: Quantile loss and MAE (Figure 12 and Figure 13)

![22_mae_virgin_r_at_virtin.png](docs%2Fimages%2F22_mae_virgin_r_at_virtin.png)

Figure 12. Exploration of MAE landscape for common model for `virgin_r_at_virtin` site with constant `days SNOTEL short`=22 parameter.
Optimal configuration for this site: `days SNOTEL short`=22, `days SNOTEL long`=148, `days PDSI`=124)

![22_quantile_virgin_r_at_virtin.png](docs%2Fimages%2F22_quantile_virgin_r_at_virtin.png)

Figure 13. Exploration of Quantile loss landscape for common model for `virgin_r_at_virtin` site with constant `days SNOTEL short`=22 parameter.
Optimal configuration for this site: `days SNOTEL short`=22, `days SNOTEL long`=108, `days PDSI`=92)

## Metric on validation sample changes through model versions

![compare_approaches.png](examples%2Fplots%2Fcompare_approaches.png)
