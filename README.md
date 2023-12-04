# wasu

Water Supply Forecast competition model

## How to use this repository

This repository contains code both for preparing visualis
ations and for training and ap
plying predictive models.

It is recommended to start exploration with [examples](./examples) folder:

- [1_basic](./examples/1_basic) - basic scripts that prepare exploratory data visualisations
- [2_repeating](./examples/2_repeating) - repeating of last known values to generate forecasts (in two versions)
- [3_streamflow](./examples/3_streamflow) - use of aggregated statistics from USGS streamflow data 
- [4_snotel](./examples/4_snotel) - use of aggregated statistics from snowpack SNOTEL data 


During code execution the `plots` folder is generated. 

For example, there you can find zones for which forecast models are initialized:

![spatial_extend.png](examples%2Fplots%2Fspatial%2Fspatial_extend.png)

## Algorithms description 

This section provides explanations that explain how the algorithms work 

### Simple repeating 

**Public Averaged Mean Quantile Loss**: 330.1185

The simplest possible algorithm. Test years on which the evaluation is performed:

- 2005
- 2007
- 2009
- 2011
- 2013
- 2015
- 2017
- 2019
- 2021
- 2023

The algorithm takes the value from 2004 and assigns it to each subsequent year.

![animas_r_at_durango_time_series_plot.png](examples%2Fplots%2Fpredictions_simple_repeating%2Fanimas_r_at_durango_time_series_plot.png)

Figure. Forecasts for tests years for site `animas_r_at_durango` using simple repeating since 2004

### Advanced repeating 

**Public Averaged Mean Quantile Loss**: 283.6057

Uses values from the previous year for this site to be used as a forecast. That is, for 2005 the year 2004 will be used, for 2007 the year 2006 will be used.

![animas_r_at_durango_time_series_plot.png](examples%2Fplots%2Fpredictions_advanced_repeating%2Fanimas_r_at_durango_time_series_plot.png)

Figure. Forecasts for tests years for site `animas_r_at_durango` using advanced repeating 

### Streamflow-based predictions

**Public Averaged Mean Quantile Loss**: 207.7306 

This approach uses flow values aggregated over a 
specific period (typically 180 days before the forecast issue date) 
to generate a forecast into the future. 

![2005-01-22 00:00:00_site_hungry_horse_reservoir_inflow.png](examples%2Fplots%2Fusgs_streamflow_3d%2F2005-01-22%2000%3A00%3A00_site_hungry_horse_reservoir_inflow.png)

Figure. 3d plot features (`min_value` and `mean_value`), `predictions` (surface) and `actual` values (points) for training sample for
desired issue date

![animas_r_at_durango_time_series_plot.png](examples%2Fplots%2Fpredictions_usgs_streamflow%2Fanimas_r_at_durango_time_series_plot.png)

### SNOTEL-based predictions

**Public Averaged Mean Quantile Loss**: 186.9168

Key features description: 

- `PREC_DAILY` - precipitation
- `TAVG_DAILY` - average daily temperature
- `TMAX_DAILY` - max temperature
- `TMIN_DAILY` - min temperature
- `WTEQ_DAILY` - snow water equivalent	

![spatial_extend_snotel_fontenelle_reservoir_inflow.png](examples%2Fplots%2Fspatial_with_snotel_stations%2Fspatial_extend_snotel_fontenelle_reservoir_inflow.png)

Figure. SNOTEL stations and basin of `fontenelle_reservoir_inflow` site