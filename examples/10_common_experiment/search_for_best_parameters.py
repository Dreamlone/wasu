from pathlib import Path

import warnings

import numpy as np
import pandas as pd

from wasu.development.validation import smape

warnings.filterwarnings('ignore')

SITES = ['hungry_horse_reservoir_inflow',
         'snake_r_nr_heise', 'pueblo_reservoir_inflow',
         'sweetwater_r_nr_alcova', 'missouri_r_at_toston',
         'animas_r_at_durango', 'yampa_r_nr_maybell', 'libby_reservoir_inflow', 'boise_r_nr_boise',
         'green_r_bl_howard_a_hanson_dam', 'taylor_park_reservoir_inflow',
         'dillon_reservoir_inflow', 'ruedi_reservoir_inflow',
         'fontenelle_reservoir_inflow', 'weber_r_nr_oakley',
         'san_joaquin_river_millerton_reservoir', 'merced_river_yosemite_at_pohono_bridge',
         'american_river_folsom_lake', 'colville_r_at_kettle_falls',
         'stehekin_r_at_stehekin', 'detroit_lake_inflow', 'virgin_r_at_virtin',
         'skagit_ross_reservoir', 'boysen_reservoir_inflow', 'pecos_r_nr_pecos',
         'owyhee_r_bl_owyhee_dam']


def create_optimal_surfaces_plots(report_with_metrics: pd.DataFrame):
    """ Draw a 3d plot """
    pass


def calculate_metric(dataframe: pd.DataFrame, metric_name: str):
    if metric_name == 'SMAPE':
        smape_metric = smape(y_true=np.array(dataframe['actual'], dtype=float),
                             y_pred=np.array(dataframe['volume_50'], dtype=float))
        return smape_metric

    return 0.1


def search_for_optimum(metric_name: str = 'SMAPE'):
    path_to_results = Path('./validation').resolve()
    files = list(path_to_results.iterdir())

    report_with_metrics = []
    for file in files:
        name = file.name
        name = name.split('.csv')[0]
        name_split = name.split('_')
        method = name_split[0]
        snotel_short = int(name_split[1])
        snotel_long = int(name_split[2])
        pdsi_days = int(name_split[3])
        results = [method, snotel_short, snotel_long, pdsi_days]

        # Load file - calculate metrics and store result into big dataframe with overall results
        predicted = pd.read_csv(file)
        for site in SITES:
            site_df = predicted[predicted['site_id'] == site]

            metric = calculate_metric(site_df, metric_name)
            results.append(metric)

        # Calculate average in the end
        results.append(calculate_metric(predicted, metric_name))
        report_with_metrics.append(results)

    columns = ['Method', 'SNOTEL short days', 'SNOTEL long days',
               'PDSI days', 'hungry_horse_reservoir_inflow',
               'snake_r_nr_heise', 'pueblo_reservoir_inflow',
               'sweetwater_r_nr_alcova', 'missouri_r_at_toston',
               'animas_r_at_durango', 'yampa_r_nr_maybell', 'libby_reservoir_inflow', 'boise_r_nr_boise',
               'green_r_bl_howard_a_hanson_dam', 'taylor_park_reservoir_inflow',
               'dillon_reservoir_inflow', 'ruedi_reservoir_inflow',
               'fontenelle_reservoir_inflow', 'weber_r_nr_oakley',
               'san_joaquin_river_millerton_reservoir', 'merced_river_yosemite_at_pohono_bridge',
               'american_river_folsom_lake', 'colville_r_at_kettle_falls',
               'stehekin_r_at_stehekin', 'detroit_lake_inflow', 'virgin_r_at_virtin',
               'skagit_ross_reservoir', 'boysen_reservoir_inflow', 'pecos_r_nr_pecos',
               'owyhee_r_bl_owyhee_dam', 'average']
    report_with_metrics = pd.DataFrame(report_with_metrics, columns=columns)

    # Start 3d plotting the results
    create_optimal_surfaces_plots(report_with_metrics)

    # Find the best solution per site and for all cases


if __name__ == '__main__':
    search_for_optimum()
