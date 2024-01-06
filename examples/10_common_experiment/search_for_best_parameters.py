from pathlib import Path

import warnings
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler

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


def create_optimal_surfaces_plots(report_with_metrics: pd.DataFrame, best_solutions: pd.DataFrame):
    """ Draw a 3d plot """
    plots_folder = Path('./optimum').resolve()
    plots_folder.mkdir(parents=True, exist_ok=True)

    for site in SITES:
        site_df = best_solutions[best_solutions['site'] == site]
        pdsi_days_optimal = site_df.iloc[0]['SNOTEL short days']

        pdsi_fixed = report_with_metrics[report_with_metrics['SNOTEL short days'] == pdsi_days_optimal]
        reg = GaussianProcessRegressor(alpha=0.4)
        features = ['PDSI days', 'SNOTEL long days']
        scaler = StandardScaler()
        scaler.fit(np.array(pdsi_fixed[features]))
        reg.fit(scaler.transform(np.array(pdsi_fixed[features])),
                np.array(pdsi_fixed[site]))

        first_feature_simulated = np.arange(min(pdsi_fixed['PDSI days']) + 1,
                                            max(pdsi_fixed['PDSI days']) - 1, 0.1)
        second_feature_simulated = np.arange(min(pdsi_fixed['SNOTEL long days']) + 1,
                                             max(pdsi_fixed['SNOTEL long days']) - 1, 0.06)
        features_ = []
        for long_snotel in second_feature_simulated:
            constant_cpu = [long_snotel] * len(first_feature_simulated)
            features_.append(pd.DataFrame({'PDSI days': first_feature_simulated,
                                           'SNOTEL long days': constant_cpu}))
        features_ = pd.concat(features_)
        features_['predicted'] = reg.predict(scaler.transform(np.array(features_[features])))

        fig = plt.figure(figsize=(20, 9))
        # First plot
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(np.array(features_['PDSI days']),
                   np.array(features_['SNOTEL long days']),
                   np.array(features_['predicted']),
                   c='Grey', s=2, linewidth=0, alpha=0.1)
        ax.scatter(np.array(site_df['PDSI days']),
                   np.array(site_df['SNOTEL long days']),
                   np.array(site_df['metric']),
                   s=150, edgecolors='red', c='white',
                   alpha=0.8, linewidth=0.9)
        surf = ax.scatter(np.array(pdsi_fixed['PDSI days']),
                          np.array(pdsi_fixed['SNOTEL long days']),
                          np.array(pdsi_fixed[site]),
                          c=np.array(pdsi_fixed[site]), cmap='Reds', s=35,
                          linewidth=0.3, alpha=0.9, edgecolors='black')
        cb = fig.colorbar(surf, shrink=0.3, aspect=10)
        cb.set_label('Metric for optimization', fontsize=12)

        ax.view_init(5, 100)
        ax.set_xlabel('PDSI days', fontsize=13)
        ax.set_ylabel('SNOTEL long days', fontsize=13)
        ax.set_zlabel(f'Metric', fontsize=13)

        # Second plot
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(np.array(features_['PDSI days']),
                   np.array(features_['SNOTEL long days']),
                   np.array(features_['predicted']),
                   c='Grey', s=2, linewidth=0, alpha=0.1)
        ax.scatter(np.array(site_df['PDSI days']),
                   np.array(site_df['SNOTEL long days']),
                   np.array(site_df['metric']),
                   s=150, edgecolors='red', c='white',
                   alpha=0.8, linewidth=0.9)
        ax.scatter(np.array(pdsi_fixed['PDSI days']),
                   np.array(pdsi_fixed['SNOTEL long days']),
                   np.array(pdsi_fixed[site]),
                   c=np.array(pdsi_fixed[site]), cmap='Reds', s=35,
                   linewidth=0.3, alpha=0.9,
                   edgecolors='black')
        # 30, 80
        ax.view_init(40, 110)
        ax.set_xlabel('PDSI days', fontsize=13)
        ax.set_ylabel('SNOTEL long days', fontsize=13)
        ax.set_zlabel('Metric', fontsize=13)
        fig.suptitle(f'Site {site}. SNOTEL short days {pdsi_days_optimal}', fontsize=15)
        fig.savefig(Path(plots_folder, f'{site}.png'), dpi=300, bbox_inches='tight')
        plt.close()


def find_best_solution(report_with_metrics: pd.DataFrame):
    report_with_metrics = report_with_metrics.reset_index()

    best_metrics = []
    report = []
    for site_id in SITES:
        row_number = report_with_metrics[site_id].argmin()

        best_solution_per_site = report_with_metrics.iloc[row_number]
        logger.debug(f'{best_solution_per_site["SNOTEL short days"]} -'
                     f' {best_solution_per_site["SNOTEL long days"]} -'
                     f' {best_solution_per_site["PDSI days"]}. Metric: {best_solution_per_site[site_id]:.2f},'
                     f' Site: {site_id}')
        best_metrics.append(best_solution_per_site[site_id])
        report.append([site_id, best_solution_per_site["SNOTEL short days"], best_solution_per_site["SNOTEL long days"],
                       best_solution_per_site["PDSI days"], best_solution_per_site[site_id]])

    row_number = report_with_metrics['average'].argmin()
    best_solution_per_site = report_with_metrics.iloc[row_number]
    logger.debug(f'{best_solution_per_site["SNOTEL short days"]} -'
                 f' {best_solution_per_site["SNOTEL long days"]} -'
                 f' {best_solution_per_site["PDSI days"]}. Metric: {best_solution_per_site["average"]:.2f},'
                 f' AVERAGE')

    mean_best_metric = np.mean(np.array(best_metrics))
    logger.info(f'Mean metric according to best values: {mean_best_metric:.2f}')
    report.append(['average', best_solution_per_site["SNOTEL short days"], best_solution_per_site["SNOTEL long days"],
                   best_solution_per_site["PDSI days"], best_solution_per_site['average']])
    report = pd.DataFrame(report, columns=['site', 'SNOTEL short days', 'SNOTEL long days', 'PDSI days', 'metric'])

    return report


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

    # Find the best solution per site and for all cases
    best_solutions = find_best_solution(report_with_metrics)

    # Start 3d plotting the results
    create_optimal_surfaces_plots(report_with_metrics, best_solutions)


if __name__ == '__main__':
    search_for_optimum()
