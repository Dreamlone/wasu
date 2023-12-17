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
from wasu.development.data.snotel import collect_snotel_data_for_site
from wasu.development.models.create import ModelsCreator
from wasu.development.models.date_utils import generate_datetime_into_julian, get_julian_date_from_datetime
from wasu.development.models.snotel import SnotelFlowRegression
from wasu.development.models.train_model import TrainModel
from wasu.development.validation import smape
from wasu.development.vis.visualization import created_spatial_plot


def _aggregate_features_snodas(agg_snodas: pd.DataFrame):
    features_columns = ['mean_Snow accumulation, 24-hour total',
                        'mean_Non-snow accumulation, 24-hour total',
                        'mean_Modeled snow water equivalent, total of snow layers',
                        'mean_Modeled snow layer thickness, total of snow layers']
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


def _aggregate_features_snotel(agg_snotel: pd.DataFrame):
    agg_snotel = agg_snotel[['PREC_DAILY', 'TAVG_DAILY', 'TMAX_DAILY', 'TMIN_DAILY', 'WTEQ_DAILY']]
    agg_snotel = agg_snotel.dropna()

    dataset = pd.DataFrame()
    for feature in ['PREC_DAILY', 'TAVG_DAILY', 'TMAX_DAILY', 'TMIN_DAILY', 'WTEQ_DAILY']:
        mean_value = agg_snotel[feature].mean()
        sum_value = agg_snotel[feature].sum()
        min_value = agg_snotel[feature].min()
        max_value = agg_snotel[feature].max()

        dataset[f'mean_{feature}'] = [mean_value]
        dataset[f'sum_{feature}'] = [sum_value]
        dataset[f'min_{feature}'] = [min_value]
        dataset[f'max_{feature}'] = [max_value]

    return dataset


def aggregate_data_for_issue_date(issue_date, day_of_year, dataframe, aggregation_days,
                                  target_value: Union[float, None], label: str) -> Union[pd.DataFrame, None]:
    aggregation_end_julian = get_julian_date_from_datetime(current_date=issue_date)
    aggregation_start_julian = get_julian_date_from_datetime(current_date=issue_date,
                                                             offset_days=aggregation_days)

    aggregated = dataframe[dataframe['julian_datetime'] >= aggregation_start_julian]
    aggregated = aggregated[aggregated['julian_datetime'] < aggregation_end_julian]
    if len(aggregated) < 1:
        return None

    if label == 'snodas':
        dataset = _aggregate_features_snodas(aggregated)
    elif label == 'snotel':
        dataset = _aggregate_features_snotel(aggregated)
    else:
        raise ValueError(f'Not supported label {label}')

    if target_value is not None:
        dataset['target'] = target_value
    dataset['issue_date'] = issue_date
    dataset['day_of_year'] = day_of_year
    return dataset


class CommonRegression(TrainModel):
    """ Create forecasts based on SNODAS, SNOTEL and all possible data sources """

    def __init__(self, train_df: pd.DataFrame,
                 method: str = 'lg',
                 aggregation_days_snodas: int = 90,
                 aggregation_days_snotel: int = 90):
        super().__init__(train_df)

        # Use only available data
        self.all_features = []

        self.vis = False

        self.train_df = self.train_df.dropna()
        self.train_df['year_as_int'] = self.train_df['year'].dt.year

        self.name = f'common_{method}_{aggregation_days_snodas}_{aggregation_days_snotel}'
        self.aggregation_days_snodas = aggregation_days_snodas
        self.aggregation_days_snotel = aggregation_days_snotel
        self.model_folder = ModelsCreator(self.name).model_folder()
        self.method = method

    def load_data_from_kwargs(self, kwargs):
        metadata: pd.DataFrame = kwargs['metadata']
        path_to_snodas: Path = kwargs['path_to_snodas']
        path_to_snotel: Path = kwargs['path_to_snotel']

        self.vis: bool = kwargs.get('vis')
        if self.vis is None:
            self.vis = False

        return metadata, path_to_snodas, path_to_snotel

    def fit_main_model(self, site: str, snodas_df: pd.DataFrame, snotel_df: pd.DataFrame, site_df: pd.DataFrame):
        dataframe_for_model_fit = self._collect_data_for_model_fit(snodas_df, snotel_df, site_df)
        features_columns = list(dataframe_for_model_fit.columns)
        for i in ['issue_date', 'target']:
            features_columns.remove(i)
        # Order must be the same
        features_columns.sort()

        # Fit model
        for alpha in [0.1, 0.5, 0.9]:
            if self.method == 'lg':
                reg = LGBMRegressor(objective='quantile', random_state=2023, alpha=alpha, verbose=-1)
            elif self.method == 'linear':
                reg = QuantileRegressor(quantile=alpha, solver='highs-ds', alpha=0.1)
            elif self.method == 'forest':
                reg = RandomForestRegressor()
            else:
                raise NotImplementedError()

            df_fit = dataframe_for_model_fit[dataframe_for_model_fit['issue_date'].dt.year < 2020]
            df_test = dataframe_for_model_fit[dataframe_for_model_fit['issue_date'].dt.year >= 2020]
            if alpha == 0.9:
                target = np.array(df_fit['target']) * 1.2
            elif alpha == 0.5:
                target = np.array(df_fit['target'])
            else:
                target = np.array(df_fit['target']) * 0.95

            reg.fit(np.array(df_fit[features_columns]), target)

            train_predicted = reg.predict(np.array(df_fit[features_columns]))
            test_predicted = reg.predict(np.array(df_test[features_columns]))
            mae_metric_fit = mean_absolute_error(y_pred=train_predicted,
                                                 y_true=np.array(target, dtype=float))
            mae_metric_test = mean_absolute_error(y_pred=test_predicted, y_true=np.array(df_test['target'], dtype=float))
            logger.debug(f'Train model for alpha {alpha}. Length: {len(dataframe_for_model_fit)}. '
                         f'MAE train: {mae_metric_fit:.2f}. MAE test: {mae_metric_test:.2f}')

            if self.vis is True:
                created_spatial_plot(dataframe_for_model_fit, reg, features_columns,
                                     self.name, f'{site}_alpha_{alpha}.png',
                                     f'SNODAS regression model {site}. Alpha: {alpha}')
            file_name = f'model_{site}_{str(alpha).replace(".", "_")}.pkl'
            model_path = Path(self.model_folder, file_name)
            with open(model_path, 'wb') as pkl:
                pickle.dump(reg, pkl)

        if self.vis is True:
            plt.scatter(dataframe_for_model_fit['issue_date'], dataframe_for_model_fit['target'],
                        c='blue', label='Target')

            for alpha in [0.1, 0.5, 0.9]:
                # Load model
                file_name = f'model_{site}_{str(alpha).replace(".", "_")}.pkl'
                with open(Path(self.model_folder, file_name), 'rb') as pkl:
                    model = pickle.load(pkl)

                train_predicted = model.predict(np.array(dataframe_for_model_fit[features_columns]))
                if alpha == 0.5:
                    plt.scatter(dataframe_for_model_fit['issue_date'], train_predicted, c='green', label='Predict')
                else:
                    plt.plot(dataframe_for_model_fit['issue_date'], train_predicted, c='green', alpha=0.5)
            plt.title(site)
            plt.legend()
            plt.show()

    def fit(self, submission_format: pd.DataFrame, **kwargs) -> Union[str, Path]:
        """ Fit new model based on snotel """
        metadata, path_to_snodas, path_to_snotel = self.load_data_from_kwargs(kwargs)

        for site in list(submission_format['site_id'].unique()):
            logger.info(f'Train SNOTEL {self.name} model for site: {site}')
            snodas_df = collect_snodas_data_for_site(path_to_snodas, site)
            snotel_df = collect_snotel_data_for_site(path_to_snotel, site)

            site_df = self.train_df[self.train_df['site_id'] == site]
            site_df = site_df.sort_values(by='year')
            self.fit_main_model(site, snodas_df, snotel_df, site_df)

        return self.model_folder

    def _collect_data_for_model_fit(self, snodas_df: pd.DataFrame, snotel_df: pd.DataFrame, site_df: pd.DataFrame):
        snodas_df = generate_datetime_into_julian(dataframe=snodas_df, datetime_column='datetime',
                                                  julian_column='julian_datetime', round_julian=3)
        snotel_df = generate_datetime_into_julian(dataframe=snotel_df, datetime_column='date',
                                                  julian_column='julian_datetime', round_julian=3)

        dataframe_for_model_fit = []
        for year in list(set(snodas_df['datetime'].dt.year)):
            train_for_issue = site_df[site_df['year_as_int'] == year]
            if len(train_for_issue) < 1:
                continue
            target_value = train_for_issue['volume'].values[0]

            for day_of_year in np.arange(1, 220, step=15):

                # Aggregate common dataset with common features
                issue_date = datetime.datetime.strptime(f'{year} {day_of_year}', '%Y %j')

                current_dataset = None
                for df, aggregation_days, label in zip([snodas_df, snotel_df],
                                                       [self.aggregation_days_snodas, self.aggregation_days_snotel],
                                                       ['snodas', 'snotel']):
                    dataset = aggregate_data_for_issue_date(issue_date, day_of_year, df, aggregation_days,
                                                            target_value, label)
                    if dataset is None:
                        # Slip this part
                        continue

                    if current_dataset is None:
                        # First iteration
                        current_dataset = dataset
                    else:
                        current_dataset = current_dataset.merge(dataset.drop(columns=['target', 'day_of_year']),
                                                                on='issue_date')

                dataframe_for_model_fit.append(current_dataset)

        dataframe_for_model_fit = pd.concat(dataframe_for_model_fit)
        dataframe_for_model_fit = dataframe_for_model_fit.dropna()
        return dataframe_for_model_fit

    def predict(self, submission_format: pd.DataFrame, **kwargs) -> pd.DataFrame:
        metadata, path_to_snodas, path_to_snotel = self.load_data_from_kwargs(kwargs)

        submit = []
        for site in list(submission_format['site_id'].unique()):
            submission_site = submission_format[submission_format['site_id'] == site]
            logger.info(f'Generate prediction for site: {site}')

            submission_site = self.predict_main_model(site, submission_site, path_to_snodas, path_to_snotel)
            logger.debug(f'Size of dataframe: {len(submission_site)}')

            # Correct according to train data (predictions can not be higher or lower)
            site_train_data = self.train_df[self.train_df['site_id'] == site]
            min_value = min(site_train_data['volume'])
            max_value = max(site_train_data['volume'])
            for col in ['volume_10', 'volume_50', 'volume_90']:
                submission_site[col][submission_site[col] <= min_value] = min_value
                submission_site[col][submission_site[col] >= max_value] = max_value

            logger.debug(f'Size of dataframe: {len(submission_site)}')

            submit.append(submission_site)

        submit = pd.concat(submit)
        return submit

    def _collect_data_for_model_predict(self, snodas_df: pd.DataFrame, snotel_df, submission_site: pd.DataFrame):
        snodas_df = generate_datetime_into_julian(dataframe=snodas_df, datetime_column='datetime',
                                                  julian_column='julian_datetime', round_julian=3)
        snotel_df = generate_datetime_into_julian(dataframe=snotel_df, datetime_column='date',
                                                  julian_column='julian_datetime', round_julian=3)
        collected_data = []
        for row_id, row in submission_site.iterrows():
            issue_date = row['issue_date']

            current_dataset = None
            for df, aggregation_days, label in zip([snodas_df, snotel_df],
                                                   [self.aggregation_days_snodas, self.aggregation_days_snotel],
                                                   ['snodas', 'snotel']):
                dataset = aggregate_data_for_issue_date(issue_date, issue_date.dayofyear, df, aggregation_days,
                                                        None, label)

                if current_dataset is None:
                    # First iteration
                    current_dataset = dataset
                else:
                    current_dataset = current_dataset.merge(dataset.drop(columns=['day_of_year']),
                                                            on='issue_date')

            if current_dataset is None or len(current_dataset) < 1:
                a = 0
            collected_data.append(current_dataset)

        collected_data = pd.concat(collected_data)
        return collected_data

    def predict_main_model(self, site: str, submission_site: pd.DataFrame, path_to_snodas: Union[str, Path],
                           path_to_snotel: Union[str, Path]):
        snodas_df = collect_snodas_data_for_site(path_to_snodas, site)
        snotel_df = collect_snotel_data_for_site(path_to_snotel, site)

        dataframe_for_model_predict = self._collect_data_for_model_predict(snodas_df, snotel_df, submission_site)
        features_columns = list(dataframe_for_model_predict.columns)
        features_columns.remove('issue_date')
        features_columns.sort()

        for alpha, column_name in zip([0.1, 0.5, 0.9], ['volume_10', 'volume_50', 'volume_90']):
            file_name = f'model_{site}_{str(alpha).replace(".", "_")}.pkl'
            model_path = Path(self.model_folder, file_name)

            with open(model_path, 'rb') as pkl:
                model = pickle.load(pkl)

            predicted = model.predict(np.array(dataframe_for_model_predict[features_columns]))
            submission_site[column_name] = predicted

        return submission_site
