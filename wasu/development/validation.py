import datetime
from pathlib import Path
from typing import List, Union, Dict

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_pinball_loss

from wasu.development.paths import path_to_examples_folder
from wasu.metrics import compute_quantile_loss


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ Symmetric mean absolute percentage error, % """

    numerator = 2 * np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    result = numerator / denominator
    result[np.isnan(result)] = 0.0
    return float(np.mean(100 * result))


class ModelValidation:
    """ Class for calculating standard regression metrics for desired dates

    :param issue_dates: list with dictionaries with information about issue dates since which it is required to generate
    a forecast for desired years
    """

    def __init__(self, issue_dates: Union[List[Dict], None] = None,
                 years_to_validate: Union[List[int], None] = None,
                 sites_to_validate: Union[List[str], None] = None,
                 folder_for_plots: str = 'default_model'):

        self.issue_dates = issue_dates
        if self.issue_dates is None:
            self.issue_dates = [{'month': 1, 'day': 15}, {'month': 2, 'day': 15},
                                {'month': 3, 'day': 15}, {'month': 4, 'day': 15}]

        self.years_to_validate = years_to_validate
        if self.years_to_validate is None:
            self.years_to_validate = [2020, 2021, 2022]

        self.sites_to_validate = sites_to_validate
        if self.sites_to_validate is None:
            self.sites_to_validate = ['hungry_horse_reservoir_inflow', 'snake_r_nr_heise', 'pueblo_reservoir_inflow',
                                      'sweetwater_r_nr_alcova', 'missouri_r_at_toston', 'animas_r_at_durango',
                                      'yampa_r_nr_maybell', 'libby_reservoir_inflow', 'boise_r_nr_boise',
                                      'green_r_bl_howard_a_hanson_dam', 'taylor_park_reservoir_inflow',
                                      'dillon_reservoir_inflow', 'ruedi_reservoir_inflow',
                                      'fontenelle_reservoir_inflow', 'weber_r_nr_oakley',
                                      'san_joaquin_river_millerton_reservoir', 'merced_river_yosemite_at_pohono_bridge',
                                      'american_river_folsom_lake', 'colville_r_at_kettle_falls',
                                      'stehekin_r_at_stehekin', 'detroit_lake_inflow', 'virgin_r_at_virtin',
                                      'skagit_ross_reservoir', 'boysen_reservoir_inflow', 'pecos_r_nr_pecos',
                                      'owyhee_r_bl_owyhee_dam']

        self.folder_for_plots = folder_for_plots

    def generate_submission_format(self):
        df = []
        for site_to_validate in self.sites_to_validate:
            logger.debug(f'Generate submission for validation for site {site_to_validate} ')
            for year in self.years_to_validate:
                for issue_date in self.issue_dates:
                    dt = datetime.datetime(year=year, month=issue_date['month'], day=issue_date['day'])
                    df.append(pd.DataFrame({'site_id': [site_to_validate], 'issue_date': [dt.strftime('%Y-%m-%d')],
                                            'volume_10': [0], 'volume_50': [0], 'volume_90': [0]}))

        df = pd.concat(df)
        df['issue_date'] = pd.to_datetime(df['issue_date'])
        return df

    def compare_dataframes(self, predicted: pd.DataFrame, train_data: pd.DataFrame,
                           save_predicted_vs_actual_into_file: Union[Path, str, None] = None):
        """ Take predictions and compare it with known values """
        all_values = []
        for site_to_validate in self.sites_to_validate:
            site_actual = train_data[train_data['site_id'] == site_to_validate]
            site_actual['year_as_int'] = pd.to_datetime(site_actual['year']).dt.year

            site_predict = predicted[predicted['site_id'] == site_to_validate]
            site_predict['year_as_int'] = pd.to_datetime(site_predict['issue_date']).dt.year

            # Create dataframe with actual and predicted values
            df_for_comparison = []
            for year in self.years_to_validate:
                actual_per_year = site_actual[site_actual['year_as_int'] == year]
                predict_per_year = site_predict[site_predict['year_as_int'] == year]

                actual_value = actual_per_year['volume'].values[0]
                predict_per_year['actual'] = actual_value

                df_for_comparison.append(predict_per_year)

            df_for_comparison = pd.concat(df_for_comparison)
            all_values.append(df_for_comparison)
            self._print_metrics(df_for_comparison, site_to_validate)
            self._make_plots(df_for_comparison, site_to_validate)

        # Calculate mean metrics for all sites at once
        all_values = pd.concat(all_values)
        self._print_metrics(all_values, 'all')
        self._make_plots(all_values, 'all')
        if save_predicted_vs_actual_into_file is not None:
            if isinstance(save_predicted_vs_actual_into_file, str) is True:
                save_predicted_vs_actual_into_file = Path(save_predicted_vs_actual_into_file)
            save_predicted_vs_actual_into_file = save_predicted_vs_actual_into_file.resolve()

            save_predicted_vs_actual_into_file.parent.mkdir(parents=True, exist_ok=True)
            all_values.to_csv(save_predicted_vs_actual_into_file, index=False)

    @staticmethod
    def _print_metrics(df_for_comparison: pd.DataFrame, site_to_validate: str):
        """ Calculate and display regression metrics """
        logger.info(f'Print metrics for {site_to_validate} site(s)')

        mae_metric = mean_absolute_error(y_true=np.array(df_for_comparison['actual']),
                                         y_pred=np.array(df_for_comparison['volume_50']))
        logger.info(f'MAE metric: {mae_metric:.2f}')

        mape_metric = mean_absolute_percentage_error(y_true=np.array(df_for_comparison['actual']),
                                                     y_pred=np.array(df_for_comparison['volume_50'])) * 100
        logger.info(f'MAPE metric: {mape_metric:.2f}')

        smape_metric = smape(y_true=np.array(df_for_comparison['actual'], dtype=float),
                             y_pred=np.array(df_for_comparison['volume_50'], dtype=float))

        logger.info(f'Symmetric MAPE metric: {smape_metric:.2f}')

        metric_low = compute_quantile_loss(y_true=np.array(df_for_comparison['actual']),
                                           y_pred=np.array(df_for_comparison['volume_10']), quantile=0.1)
        metric_mean = compute_quantile_loss(y_true=np.array(df_for_comparison['actual']),
                                            y_pred=np.array(df_for_comparison['volume_50']), quantile=0.5)
        metric_high = compute_quantile_loss(y_true=np.array(df_for_comparison['actual']),
                                            y_pred=np.array(df_for_comparison['volume_90']), quantile=0.9)
        quantile_loss = (metric_low + metric_mean + metric_high) / 3

        logger.info(f'Quantile loss metric: {quantile_loss:.2f}. 0.5 loss: {metric_mean:.2f}')
        logger.warning('--------------------------------------------')

    def _make_plots(self, df_for_comparison: pd.DataFrame, site_to_validate: str):
        logger.info(f'Make plots for {site_to_validate} site(s)')

        # Generate bi-plot first
        plots_folder = Path(path_to_examples_folder(), 'plots_validation', self.folder_for_plots)
        plots_folder.mkdir(exist_ok=True, parents=True)

        fig_size = (12.0, 10.0)
        fig, ax = plt.subplots(figsize=fig_size)

        ax.scatter(df_for_comparison['actual'], df_for_comparison['volume_50'],
                   c='blue', alpha=0.5, s=45, edgecolors={'#BCE7FF'})
        ax.plot([min(df_for_comparison['actual']), max(df_for_comparison['actual'])],
                [min(df_for_comparison['actual']), max(df_for_comparison['actual'])],
                c='black', alpha=0.5, linewidth=2)
        ax.set_xlabel(f'Actual. Site {site_to_validate}', fontsize=13)
        ax.set_ylabel(f'Predicted. Site {site_to_validate}', fontsize=13)
        plt.grid()
        plt.title(f'Validation on {self.years_to_validate} years')
        plt.savefig(Path(plots_folder, f'site_{site_to_validate}_biplot.png'))
        plt.close()

        ########################
        # Barplot with metrics #
        ########################
        df_for_comparison['dayofyear'] = pd.to_datetime(df_for_comparison['issue_date']).dt.dayofyear
        df_for_vis = []
        visited_days = []
        for dayofyear in list(df_for_comparison['dayofyear'].unique()):
            need_to_skip = False
            for visited_day in visited_days:
                min_th = visited_day - 3
                max_th = visited_day + 3
                if min_th <= dayofyear <= max_th:
                    need_to_skip = True
            if need_to_skip is True:
                continue

            # Clip
            day_df = df_for_comparison[df_for_comparison['dayofyear'] >= dayofyear - 3]
            day_df = day_df[day_df['dayofyear'] <= dayofyear + 3]

            smape_metric = smape(y_true=np.array(day_df['actual'], dtype=float),
                                 y_pred=np.array(day_df['volume_50'], dtype=float))
            df_for_vis.append(pd.DataFrame({'dayofyear': [dayofyear], 'SMAPE': [smape_metric]}))
            visited_days.append(dayofyear)

        df_for_vis = pd.concat(df_for_vis)

        fig_size = (12.0, 7.0)
        fig, ax = plt.subplots(figsize=fig_size)
        plt.bar(df_for_vis['dayofyear'], df_for_vis['SMAPE'], width=5, color='blue', alpha=0.2, edgecolor='#BCE7FF')
        plt.xlabel('Day of the year', fontsize=13)
        plt.ylabel('SMAPE of the forecast', fontsize=13)
        plt.savefig(Path(plots_folder, f'site_{site_to_validate}_barplot.png'))
        plt.close()
