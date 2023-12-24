import datetime
import pickle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from loguru import logger

from wasu.development.data.streamflow import collect_usgs_streamflow_time_series_for_site
from wasu.development.models.create import ModelsCreator
from wasu.development.models.custom.advanced_repeating import AdvancedRepeatingCustomModel
from wasu.development.models.date_utils import generate_datetime_into_julian, get_julian_date_from_datetime
from wasu.development.models.repeating import AdvancedRepeatingTrainModel
from wasu.development.models.train_model import TrainModel
from wasu.development.validation import smape


def _aggregate_features(agg_streamflow: pd.DataFrame):
    # Collect features for ML model
    mean_value = agg_streamflow['00060_Mean'].mean()
    sum_value = agg_streamflow['00060_Mean'].sum()
    min_value = agg_streamflow['00060_Mean'].min()
    max_value = agg_streamflow['00060_Mean'].max()

    dataset = pd.DataFrame({'mean_value': [mean_value], 'sum_value': [sum_value], 'min_value': [min_value],
                            'max_value': [max_value]})

    return dataset


class StreamFlowRegression(TrainModel):
    """ Create forecasts based on streamflow USGS data """

    def __init__(self, train_df: pd.DataFrame, aggregation_days: int = 180):
        super().__init__(train_df)
        self.backup_model = AdvancedRepeatingTrainModel(train_df)

        self.lower_ratio = 0.3
        self.above_ratio = 0.3
        # Days for parameters aggregation
        self.aggregation_days = aggregation_days
        self.features_columns = ['mean_value', 'sum_value', 'min_value', 'max_value', 'day_of_year']

        self.statistics = []
        self.vis = False
        self.vis_folder_name = 'usgs_streamflow_3d'
        self.name = f'streamflow_{aggregation_days}'
        self.model_folder = ModelsCreator(self.name).model_folder()

        self.train_df = self.train_df.dropna()
        self.train_df['year_as_int'] = self.train_df['year'].dt.year

    def load_data_from_kwargs(self, kwargs):
        metadata: pd.DataFrame = kwargs['metadata']
        path_to_streamflow: Path = kwargs['path_to_streamflow']

        self.vis: bool = kwargs.get('vis')
        if self.vis is None:
            self.vis = False

        return metadata, path_to_streamflow

    def fit(self, submission_format: pd.DataFrame, **kwargs) -> Union[str, Path]:
        """ Train model """
        metadata, path_to_streamflow = self.load_data_from_kwargs(kwargs)

        for site in list(submission_format['site_id'].unique()):
            # Try to collect features for main model
            logger.info(f'Train {self.name} model for site: {site}')
            streamflow_df = collect_usgs_streamflow_time_series_for_site(path_to_streamflow, site)

            site_df = self.train_df[self.train_df['site_id'] == site]
            site_df = site_df.sort_values(by='year')

            if streamflow_df is None or len(streamflow_df) < 1:
                logger.warning(f'Can not obtain streamflow data for site {site}. Train advanced repeating model')
                self.fit_backup_model(site, site_df)
            else:
                self.fit_main_model(site, streamflow_df, site_df)

        return self.model_folder

    def fit_backup_model(self, site: str, site_df: pd.DataFrame):
        """ Fit Advanced repeating model """
        for alpha in [0.1, 0.5, 0.9]:
            model = AdvancedRepeatingCustomModel(site, alpha)
            model.fit(site_df)

            file_name = f'backup_model_{site}_{str(alpha).replace(".", "_")}.pkl'
            model_path = Path(self.model_folder, file_name)
            with open(model_path, 'wb') as pkl:
                pickle.dump(model, pkl)

    def fit_main_model(self, site: str, streamflow_df: pd.DataFrame, site_df: pd.DataFrame):
        """ Fit main model and save it """
        dataframe_for_model_fit = self._collect_data_for_model_fit(streamflow_df, site_df)

        # Fit model
        for alpha in [0.1, 0.5, 0.9]:
            reg = LGBMRegressor(objective='quantile', random_state=2023, alpha=alpha,
                                min_data_in_leaf=10, min_child_samples=10, verbose=-1)
            reg.fit(np.array(dataframe_for_model_fit[self.features_columns]),
                    np.array(dataframe_for_model_fit['target']))

            smape_metric = smape(y_true=reg.predict(np.array(dataframe_for_model_fit[self.features_columns])),
                                 y_pred=np.array(dataframe_for_model_fit['target'], dtype=float))
            logger.debug(f'Train model for alpha {alpha}. Length: {len(dataframe_for_model_fit)}. SMAPE: {smape_metric}')

            file_name = f'model_{site}_{str(alpha).replace(".", "_")}.pkl'
            model_path = Path(self.model_folder, file_name)
            with open(model_path, 'wb') as pkl:
                pickle.dump(reg, pkl)

    def predict(self, submission_format: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """ Generate prediction model """
        metadata, path_to_streamflow = self.load_data_from_kwargs(kwargs)

        submit = []
        for site in list(submission_format['site_id'].unique()):
            submission_site = submission_format[submission_format['site_id'] == site]
            logger.info(f'Generate prediction for site: {site}')

            # Determine which model to use - main or backup
            main_model_file_name = f'model_{site}_0_5.pkl'
            if Path(self.model_folder, main_model_file_name).is_file() is False:
                # Generate backup forecast
                submission_site = self.predict_backup_model(site, submission_site)
            else:
                # Generate main model forecast
                submission_site = self.predict_main_model(site, submission_site, path_to_streamflow)

            submit.append(self.adjust_forecast(site, submission_site))

        submit = pd.concat(submit)
        return submit

    def predict_backup_model(self, site: str, submission_site: pd.DataFrame):
        for alpha, column_name in zip([0.1, 0.5, 0.9], ['volume_10', 'volume_50', 'volume_90']):
            file_name = f'backup_model_{site}_{str(alpha).replace(".", "_")}.pkl'
            with open(Path(self.model_folder, file_name), 'rb') as pkl:
                model = pickle.load(pkl)

            predicted = model.predict(submission_site)
            submission_site[column_name] = predicted

        return submission_site

    def predict_main_model(self, site: str, submission_site: pd.DataFrame, path_to_streamflow: Union[str, Path]):

        streamflow_df = collect_usgs_streamflow_time_series_for_site(path_to_streamflow, site)

        dataframe_for_model_predict = self._collect_data_for_model_predict(streamflow_df, submission_site)

        for alpha, column_name in zip([0.1, 0.5, 0.9], ['volume_10', 'volume_50', 'volume_90']):
            file_name = f'model_{site}_{str(alpha).replace(".", "_")}.pkl'
            model_path = Path(self.model_folder, file_name)

            with open(model_path, 'rb') as pkl:
                model = pickle.load(pkl)

            predicted = model.predict(np.array(dataframe_for_model_predict[self.features_columns]))
            submission_site[column_name] = predicted

        return submission_site

    def _collect_data_for_model_fit(self, streamflow_df: pd.DataFrame, site_df: pd.DataFrame):
        """ Collect dataframe for site model fitting """
        streamflow_df = generate_datetime_into_julian(dataframe=streamflow_df, datetime_column='datetime',
                                                      julian_column='julian_datetime', round_julian=3)

        dataframe_for_model_fit = []
        for year in list(set(streamflow_df['datetime'].dt.year)):
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

                agg_streamflow = streamflow_df[streamflow_df['julian_datetime'] >= aggregation_start_julian]
                agg_streamflow = agg_streamflow[agg_streamflow['julian_datetime'] < aggregation_end_julian]
                if len(agg_streamflow) < 1:
                    continue

                dataset = _aggregate_features(agg_streamflow)
                dataset['target'] = target_value
                dataset['issue_date'] = issue_date
                dataset['day_of_year'] = day_of_year
                dataframe_for_model_fit.append(dataset)

        dataframe_for_model_fit = pd.concat(dataframe_for_model_fit)
        return dataframe_for_model_fit

    def _collect_data_for_model_predict(self, streamflow_df: pd.DataFrame, submission_site: pd.DataFrame):
        streamflow_df = generate_datetime_into_julian(dataframe=streamflow_df, datetime_column='datetime',
                                                      julian_column='julian_datetime', round_julian=3)

        collected_data = []
        for row_id, row in submission_site.iterrows():
            issue_date = row['issue_date']

            aggregation_end_julian = get_julian_date_from_datetime(current_date=issue_date)
            aggregation_start_julian = get_julian_date_from_datetime(current_date=issue_date,
                                                                     offset_days=self.aggregation_days)

            agg_streamflow = streamflow_df[streamflow_df['julian_datetime'] >= aggregation_start_julian]
            agg_streamflow = agg_streamflow[agg_streamflow['julian_datetime'] < aggregation_end_julian]

            dataset = _aggregate_features(agg_streamflow)
            dataset['issue_date'] = issue_date
            dataset['day_of_year'] = issue_date.dayofyear

            collected_data.append(dataset)

        collected_data = pd.concat(collected_data)
        return collected_data
