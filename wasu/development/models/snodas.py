import datetime
import pickle
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from loguru import logger
from matplotlib import pyplot as plt
from pandas import Timestamp
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_error

from wasu.development.data.snodas import collect_snodas_data_for_site
from wasu.development.models.create import ModelsCreator
from wasu.development.models.date_utils import generate_datetime_into_julian, get_julian_date_from_datetime
from wasu.development.models.snotel import SnotelFlowRegression
from wasu.development.models.train_model import TrainModel
from wasu.development.validation import smape
from wasu.development.vis.visualization import created_spatial_plot


def _aggregate_features(features_columns: List, agg_snodas: pd.DataFrame):
    agg_snodas = agg_snodas[features_columns]
    agg_snodas = agg_snodas.dropna()

    dataset = pd.DataFrame()
    for feature in features_columns:
        mean_value = agg_snodas[feature].mean()
        min_value = agg_snodas[feature].min()
        max_value = agg_snodas[feature].max()

        dataset[f'mean_{feature}'] = [mean_value]
        dataset[f'min_{feature}'] = [min_value]
        dataset[f'max_{feature}'] = [max_value]

    return dataset


class SnodasRegression(TrainModel):
    """ Create forecasts based on SNODAS data """

    def __init__(self, train_df: pd.DataFrame, aggregation_days: int = 90):
        super().__init__(train_df)
        self.backup_model = SnotelFlowRegression(train_df, aggregation_days=180)

        self.lower_ratio = 0.3
        self.above_ratio = 0.3
        # Use only available data
        self.base_features_columns = ['Snow accumulation, 24-hour total',
                                      'Non-snow accumulation, 24-hour total',
                                      'Modeled snow water equivalent, total of snow layers',
                                      'Modeled snow layer thickness, total of snow layers']
        self.features_columns = []
        for column in self.base_features_columns:
            self.features_columns.append(f'mean_{column}')

        self.all_features = ['day_of_year']
        for column in self.features_columns:
            self.all_features.extend([f'mean_{column}', f'min_{column}', f'max_{column}'])
        self.vis = False

        self.train_df = self.train_df.dropna()
        self.train_df['year_as_int'] = self.train_df['year'].dt.year

        self.name = f'snodas_{aggregation_days}'
        self.aggregation_days = aggregation_days
        self.model_folder = ModelsCreator(self.name).model_folder()

    def load_data_from_kwargs(self, kwargs):
        metadata: pd.DataFrame = kwargs['metadata']
        path_to_snodas: Path = kwargs['path_to_snodas']

        self.vis: bool = kwargs.get('vis')
        if self.vis is None:
            self.vis = False

        return metadata, path_to_snodas

    def fit_main_model(self, site: str, snodas_df: pd.DataFrame, site_df: pd.DataFrame):
        dataframe_for_model_fit = self._collect_data_for_model_fit(snodas_df, site_df)

        # Fit model
        for alpha in [0.1, 0.5, 0.9]:
            reg = LGBMRegressor(objective='quantile', random_state=2023, alpha=alpha, verbose=-1)
            # reg = QuantileRegressor(quantile=alpha, solver='highs-ds')
            df = dataframe_for_model_fit[dataframe_for_model_fit['issue_date'].dt.year < 2020]
            if alpha == 0.9:
                target = np.array(df['target']) * 1.05
            elif alpha == 0.5:
                target = np.array(df['target'])
            else:
                target = np.array(df['target']) * 0.95
            #
            # reg = RandomForestRegressor(max_depth=3)
            reg.fit(np.array(df[self.all_features]), target)

            train_predicted = reg.predict(np.array(dataframe_for_model_fit[self.all_features]))
            mae_metric = mean_absolute_error(y_pred=train_predicted,
                                             y_true=np.array(dataframe_for_model_fit['target'], dtype=float))
            logger.debug(f'Train model for alpha {alpha}. Length: {len(dataframe_for_model_fit)}. MAE: {mae_metric}')

            if self.vis is True:
                created_spatial_plot(dataframe_for_model_fit, reg, self.all_features,
                                     self.name, f'{site}_alpha_{alpha}.png',
                                     f'SNODAS regression model {site}. Alpha: {alpha}')
            file_name = f'model_{site}_{str(alpha).replace(".", "_")}.pkl'
            model_path = Path(self.model_folder, file_name)
            with open(model_path, 'wb') as pkl:
                pickle.dump(reg, pkl)

        # Make plots
        for col in self.all_features:
            plt.scatter(dataframe_for_model_fit['issue_date'], dataframe_for_model_fit['target'], c='blue',
                        label='Target')
            plt.scatter(dataframe_for_model_fit['issue_date'], dataframe_for_model_fit[col], c='orange')

            for alpha in [0.1, 0.5, 0.9]:
                # Load model
                file_name = f'model_{site}_{str(alpha).replace(".", "_")}.pkl'
                with open(Path(self.model_folder, file_name), 'rb') as pkl:
                    model = pickle.load(pkl)

                train_predicted = model.predict(np.array(dataframe_for_model_fit[self.all_features]))
                if alpha == 0.5:
                    plt.scatter(dataframe_for_model_fit['issue_date'], train_predicted, c='green', label='Predict')
                else:
                    plt.plot(dataframe_for_model_fit['issue_date'], train_predicted, c='green', alpha=0.5)
            plt.title(col)
            plt.legend()
            plt.show()

    def fit(self, submission_format: pd.DataFrame, **kwargs) -> Union[str, Path]:
        """ Fit new model based on snotel """
        metadata, path_to_snodas = self.load_data_from_kwargs(kwargs)

        for site in list(submission_format['site_id'].unique()):
            logger.info(f'Train SNOTEL {self.name} model for site: {site}')
            snodas_df = collect_snodas_data_for_site(path_to_snodas, site)

            site_df = self.train_df[self.train_df['site_id'] == site]
            site_df = site_df.sort_values(by='year')
            if snodas_df is None or len(snodas_df) < 1:
                raise ValueError(f'Can not obtain SNODAS data for site {site}')
            else:
                self.fit_main_model(site, snodas_df, site_df)

        return self.model_folder

    def _collect_data_for_model_fit(self, snodas_df: pd.DataFrame, site_df: pd.DataFrame):
        snodas_df = generate_datetime_into_julian(dataframe=snodas_df, datetime_column='datetime',
                                                  julian_column='julian_datetime', round_julian=3)

        dataframe_for_model_fit = []
        for year in list(set(snodas_df['datetime'].dt.year)):
            train_for_issue = site_df[site_df['year_as_int'] == year]
            if len(train_for_issue) < 1:
                continue
            target_value = train_for_issue['volume'].values[0]

            for day_of_year in np.arange(1, 220, step=15):

                # Start aggregation - define borders
                issue_date = datetime.datetime.strptime(f'{year} {day_of_year}', '%Y %j')

                aggregation_end_julian = get_julian_date_from_datetime(current_date=issue_date)
                aggregation_start_julian = get_julian_date_from_datetime(current_date=issue_date,
                                                                         offset_days=self.aggregation_days)

                agg_snodas = snodas_df[snodas_df['julian_datetime'] >= aggregation_start_julian]
                agg_snodas = agg_snodas[agg_snodas['julian_datetime'] < aggregation_end_julian]
                if len(agg_snodas) < 1:
                    continue

                dataset = _aggregate_features(self.features_columns, agg_snodas)
                dataset['target'] = target_value
                dataset['issue_date'] = issue_date
                dataset['day_of_year'] = day_of_year
                dataframe_for_model_fit.append(dataset)

        dataframe_for_model_fit = pd.concat(dataframe_for_model_fit)
        return dataframe_for_model_fit

    def predict(self, submission_format: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """ Make predictions based on SNODAS data """
        metadata, path_to_snodas = self.load_data_from_kwargs(kwargs)

        submit = []
        for site in list(submission_format['site_id'].unique()):
            submission_site = submission_format[submission_format['site_id'] == site]
            logger.info(f'Generate prediction for site: {site}')

            submission_site = self.predict_main_model(site, submission_site, path_to_snodas)
            submit.append(submission_site)

        submit = pd.concat(submit)
        return submit

    def _collect_data_for_model_predict(self, snodas_df: pd.DataFrame, submission_site: pd.DataFrame):
        snodas_df = generate_datetime_into_julian(dataframe=snodas_df, datetime_column='datetime',
                                                  julian_column='julian_datetime', round_julian=3)
        collected_data = []
        for row_id, row in submission_site.iterrows():
            issue_date = row['issue_date']

            aggregation_end_julian = get_julian_date_from_datetime(current_date=issue_date)
            aggregation_start_julian = get_julian_date_from_datetime(current_date=issue_date,
                                                                     offset_days=self.aggregation_days)

            agg_streamflow = snodas_df[snodas_df['julian_datetime'] >= aggregation_start_julian]
            agg_streamflow = agg_streamflow[agg_streamflow['julian_datetime'] < aggregation_end_julian]

            dataset = _aggregate_features(self.features_columns, agg_streamflow)
            dataset['issue_date'] = issue_date
            dataset['day_of_year'] = issue_date.dayofyear

            collected_data.append(dataset)

        collected_data = pd.concat(collected_data)
        return collected_data

    def predict_main_model(self, site: str, submission_site: pd.DataFrame, path_to_snotel: Union[str, Path]):
        snodas_df = collect_snodas_data_for_site(path_to_snotel, site)

        dataframe_for_model_predict = self._collect_data_for_model_predict(snodas_df, submission_site)

        for alpha, column_name in zip([0.1, 0.5, 0.9], ['volume_10', 'volume_50', 'volume_90']):
            file_name = f'model_{site}_{str(alpha).replace(".", "_")}.pkl'
            model_path = Path(self.model_folder, file_name)

            with open(model_path, 'rb') as pkl:
                model = pickle.load(pkl)

            predicted = model.predict(np.array(dataframe_for_model_predict[self.all_features]))
            submission_site[column_name] = predicted

        return submission_site