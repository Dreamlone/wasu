import datetime
import pickle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from loguru import logger
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.preprocessing import StandardScaler

from wasu.development.data.snotel import collect_snotel_data_for_site
from wasu.development.models.create import ModelsCreator
from wasu.development.models.date_utils import generate_datetime_into_julian, get_julian_date_from_datetime
from wasu.development.models.repeating import AdvancedRepeatingTrainModel
from wasu.development.models.train_model import TrainModel
from wasu.development.validation import smape
from wasu.development.vis.visualization import created_spatial_plot


def _aggregate_features(agg_snotel: pd.DataFrame):
    agg_snotel = agg_snotel[['PREC_DAILY', 'TAVG_DAILY', 'TMAX_DAILY', 'TMIN_DAILY', 'WTEQ_DAILY', 'date']]
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


class SnotelFlowRegression(TrainModel):
    """ Create forecasts based on streamflow USGS data """

    def __init__(self, train_df: pd.DataFrame, aggregation_days: int = 90,
                 enable_spatial_aggregation: bool = True, collect_only_in_basin: bool = False,
                 train_test_split_year: int = 2015):
        super().__init__(train_df)
        self.backup_model = AdvancedRepeatingTrainModel(train_df)

        # One month
        self.aggregation_days = aggregation_days
        self.features_columns = ['mean_PREC_DAILY', 'mean_TAVG_DAILY', 'mean_TMAX_DAILY', 'mean_TMIN_DAILY',
                                 'mean_WTEQ_DAILY', 'sum_WTEQ_DAILY', 'min_WTEQ_DAILY', 'max_WTEQ_DAILY',
                                 'day_of_year']

        self.statistics = []
        self.vis = False
        self.train_df = self.train_df.dropna()
        self.train_df['year_as_int'] = self.train_df['year'].dt.year

        self.enable_spatial_aggregation = enable_spatial_aggregation
        self.collect_only_in_basin = collect_only_in_basin

        self.name = f'snotel_{aggregation_days}_spatial_{self.enable_spatial_aggregation}_basin_{self.collect_only_in_basin}'
        self.model_folder = ModelsCreator(self.name).model_folder()
        self.train_test_split_year = train_test_split_year

    def load_data_from_kwargs(self, kwargs):
        metadata: pd.DataFrame = kwargs['metadata']
        path_to_snotel: Path = kwargs['path_to_snotel']

        self.vis: bool = kwargs.get('vis')
        if self.vis is None:
            self.vis = False

        return metadata, path_to_snotel

    def fit(self, submission_format: pd.DataFrame, **kwargs) -> Union[str, Path]:
        """ Fit new model based on snotel """
        metadata, path_to_snotel = self.load_data_from_kwargs(kwargs)

        for site in list(submission_format['site_id'].unique()):
            logger.info(f'Train SNOTEL {self.name} model for site: {site}')
            snotel_df = collect_snotel_data_for_site(path_to_snotel, site,
                                                     collect_only_in_basin=self.collect_only_in_basin)

            site_df = self.train_df[self.train_df['site_id'] == site]
            site_df = site_df.sort_values(by='year')
            if snotel_df is None or len(snotel_df) < 1:
                raise ValueError(f'Can not obtain snotel data for site {site}')
            else:
                self.fit_main_model(site, snotel_df, site_df)

        return self.model_folder

    def fit_main_model(self, site: str, snotel_df: pd.DataFrame, site_df: pd.DataFrame):
        dataframe_for_model_fit = self._collect_data_for_model_fit(snotel_df, site_df)
        dataframe_for_model_fit = dataframe_for_model_fit[dataframe_for_model_fit['issue_date'].dt.year <= self.train_test_split_year]

        # Fit model
        for alpha in [0.1, 0.5, 0.9]:
            if alpha == 0.5:
                reg = QuantileRegressor(quantile=alpha, solver='highs-ds', alpha=0.17)
            else:
                reg = QuantileRegressor(quantile=alpha, solver='highs-ds', alpha=0.08)

            scaler = StandardScaler()
            scaled_train = scaler.fit_transform(np.array(dataframe_for_model_fit[self.features_columns]))

            reg.fit(scaled_train, np.array(dataframe_for_model_fit['target']))

            logger.debug(f'Train model for alpha {alpha}. Length: {len(dataframe_for_model_fit)}')

            if self.vis is True:
                created_spatial_plot(dataframe_for_model_fit, reg, self.features_columns,
                                     self.name, f'{site}_alpha_{alpha}.png',
                                     f'SNOTEL regression model {site}. Alpha: {alpha}')

            file_name = f'model_{site}_{str(alpha).replace(".", "_")}.pkl'
            model_path = Path(self.model_folder, file_name)
            with open(model_path, 'wb') as pkl:
                pickle.dump(reg, pkl)

            scaler_name = f'scaler_{site}_{str(alpha).replace(".", "_")}.pkl'
            scaler_path = Path(self.model_folder, scaler_name)
            with open(scaler_path, 'wb') as pkl:
                pickle.dump(scaler, pkl)

    def _collect_data_for_model_fit(self, snotel_df: pd.DataFrame, site_df: pd.DataFrame):

        if self.enable_spatial_aggregation is not None and self.enable_spatial_aggregation is True:
            # Calculate mean values per basin
            snotel_df = snotel_df.groupby('date').agg({'PREC_DAILY': 'mean', 'TAVG_DAILY': 'mean',
                                                       'TMAX_DAILY': 'mean', 'TMIN_DAILY': 'mean',
                                                       'WTEQ_DAILY': 'mean'})
            snotel_df = snotel_df.reset_index()
            snotel_df = snotel_df.dropna()

        snotel_df = generate_datetime_into_julian(dataframe=snotel_df, datetime_column='date',
                                                  julian_column='julian_datetime', round_julian=3)

        dataframe_for_model_fit = []
        for year in list(set(snotel_df['date'].dt.year)):
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

                agg_snotel = snotel_df[snotel_df['julian_datetime'] >= aggregation_start_julian]
                agg_snotel = agg_snotel[agg_snotel['julian_datetime'] < aggregation_end_julian]
                if len(agg_snotel) < 1:
                    continue

                dataset = _aggregate_features(agg_snotel)
                dataset['target'] = target_value
                dataset['issue_date'] = issue_date
                dataset['day_of_year'] = day_of_year
                dataframe_for_model_fit.append(dataset)

        dataframe_for_model_fit = pd.concat(dataframe_for_model_fit)
        return dataframe_for_model_fit

    def predict(self, submission_format: pd.DataFrame, **kwargs) -> pd.DataFrame:
        metadata, path_to_snotel = self.load_data_from_kwargs(kwargs)

        submit = []
        for site in list(submission_format['site_id'].unique()):
            submission_site = submission_format[submission_format['site_id'] == site]
            logger.info(f'Generate prediction for site: {site}')

            submission_site = self.predict_main_model(site, submission_site, path_to_snotel)
            submit.append(submission_site)

        submit = pd.concat(submit)
        return submit

    def _collect_data_for_model_predict(self, snotel_df: pd.DataFrame, submission_site: pd.DataFrame):
        snotel_df = generate_datetime_into_julian(dataframe=snotel_df, datetime_column='date',
                                                  julian_column='julian_datetime', round_julian=3)
        collected_data = []
        for row_id, row in submission_site.iterrows():
            issue_date = row['issue_date']

            aggregation_end_julian = get_julian_date_from_datetime(current_date=issue_date)
            aggregation_start_julian = get_julian_date_from_datetime(current_date=issue_date,
                                                                     offset_days=self.aggregation_days)

            agg_streamflow = snotel_df[snotel_df['julian_datetime'] >= aggregation_start_julian]
            agg_streamflow = agg_streamflow[agg_streamflow['julian_datetime'] < aggregation_end_julian]

            dataset = _aggregate_features(agg_streamflow)
            dataset['issue_date'] = issue_date
            dataset['day_of_year'] = issue_date.dayofyear

            collected_data.append(dataset)

        collected_data = pd.concat(collected_data)
        return collected_data

    def predict_main_model(self, site: str, submission_site: pd.DataFrame, path_to_snotel: Union[str, Path]):
        snotel_df = collect_snotel_data_for_site(path_to_snotel, site,
                                                 collect_only_in_basin=self.collect_only_in_basin)

        dataframe_for_model_predict = self._collect_data_for_model_predict(snotel_df, submission_site)

        for alpha, column_name in zip([0.1, 0.5, 0.9], ['volume_10', 'volume_50', 'volume_90']):
            scaler_name = f'scaler_{site}_{str(alpha).replace(".", "_")}.pkl'
            scaler_path = Path(self.model_folder, scaler_name)

            with open(scaler_path, 'rb') as pkl:
                scaler = pickle.load(pkl)

            file_name = f'model_{site}_{str(alpha).replace(".", "_")}.pkl'
            model_path = Path(self.model_folder, file_name)

            with open(model_path, 'rb') as pkl:
                model = pickle.load(pkl)

            predicted = model.predict(scaler.transform(np.array(dataframe_for_model_predict[self.features_columns])))
            submission_site[column_name] = predicted

        return submission_site
