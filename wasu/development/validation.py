import datetime
from typing import List, Union, Dict

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


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
                 sites_to_validate: Union[List[str], None] = None):
        test_years = {2005, 2007, 2009, 2011, 2013, 2015, 2017, 2019, 2021, 2023}

        self.issue_dates = issue_dates
        if self.issue_dates is None:
            self.issue_dates = [{'month': 1, 'day': 15}, {'month': 2, 'day': 15}, {'month': 3, 'day': 15},
                                {'month': 4, 'day': 15}]

        self.years_to_validate = years_to_validate
        if self.years_to_validate is None:
            self.years_to_validate = [2014, 2016, 2018, 2020]
        else:
            if len(set(self.years_to_validate).intersection(test_years)) > 0:
                raise ValueError(f'Can not validate if the year is in test data: {test_years}')

        self.sites_to_validate = sites_to_validate
        if self.sites_to_validate is None:
            self.sites_to_validate = ['hungry_horse_reservoir_inflow', 'snake_r_nr_heise', 'pueblo_reservoir_inflow',
                                      'sweetwater_r_nr_alcova', 'missouri_r_at_toston', 'animas_r_at_durango',
                                      'yampa_r_nr_maybell', 'libby_reservoir_inflow']

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

    def compare_dataframes(self, predicted: pd.DataFrame, train_data: pd.DataFrame):
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

        # Calculate mean metrics for all sites at once
        all_values = pd.concat(all_values)
        self._print_metrics(all_values, 'all')

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
        logger.warning('--------------------------------------------')

    @staticmethod
    def _make_plots(df_for_comparison: pd.DataFrame, site_to_validate: str):
        logger.info(f'Make plots for {site_to_validate} site(s)')
        # TODO implement it
        raise NotImplementedError()
