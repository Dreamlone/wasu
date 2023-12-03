from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from pandas import Timestamp
from sklearn.ensemble import RandomForestRegressor

from wasu.development.data import collect_usgs_streamflow_time_series_for_site
from wasu.development.models.date_utils import generate_datetime_into_julian, get_julian_date_from_datetime
from wasu.development.models.repeating import AdvancedRepeatingTrainModel
from wasu.development.models.train_model import TrainModel
from wasu.development.vis.visualization import created_spatial_plot


class StreamFlowRegression(TrainModel):
    """ Create forecasts based on streamflow USGS data """

    def __init__(self, train_df: pd.DataFrame):
        super().__init__(train_df)
        self.backup_model = AdvancedRepeatingTrainModel(train_df)

        self.lower_ratio = 0.3
        self.above_ratio = 0.3
        # Days for parameters aggregation
        self.aggregation_days = 180
        self.features_columns = ['mean_value', 'sum_value', 'min_value', 'max_value']

        self.statistics = []
        self.vis = False
        self.vis_folder_name = 'usgs_streamflow_3d'

    def predict(self, submission_format: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.train_df = self.train_df.dropna()

        metadata: pd.DataFrame = kwargs['metadata']
        path_to_streamflow: Path = kwargs['path_to_streamflow']
        self.vis: bool = kwargs.get('vis')
        if self.vis is None:
            self.vis = False

        df_to_send = []
        # For every site
        for site in list(submission_format['site_id'].unique()):
            streamflow_df = collect_usgs_streamflow_time_series_for_site(path_to_streamflow, site)
            submission_site = submission_format[submission_format['site_id'] == site]
            site_df = self.train_df[self.train_df['site_id'] == site]
            site_df = site_df.sort_values(by='year')

            if streamflow_df is None or len(streamflow_df) < 1:
                logger.warning(f'Can not obtain streamflow data for site {site}. Apply Advanced repeating')
                submit = self.backup_model.generate_forecasts_for_site(historical_values=site_df,
                                                                       submission_site=submission_site)
                df_to_send.append(submit)
                self.statistics.append([site, 'backup'])
                continue

            metadata_site = metadata[metadata['site_id'] == site]
            season_start_month = metadata_site['season_start_month'].values[0]
            season_end_month = metadata_site['season_end_month'].values[0]

            logger.info(f'Generating forecast for site {site}. Start season month: {season_start_month}. '
                        f'End season month: {season_end_month}')

            try:
                submit = self._generate_forecasts_for_site(historical_values=site_df,
                                                           streamflow_df=streamflow_df,
                                                           submission_site=submission_site,
                                                           site_id=site)
                df_to_send.append(submit)
                self.statistics.append([site, 'streamflow model'])
            except Exception as ex:
                logger.warning(f'Can not generate forecast for site {site} due to {ex}. Apply Advanced repeating')
                submit = self.backup_model.generate_forecasts_for_site(historical_values=site_df,
                                                                       submission_site=submission_site)
                df_to_send.append(submit)
                self.statistics.append([site, 'backup'])

        for case in self.statistics:
            print(f'Site: {case[0]}. Model: {case[1]}')

        df_to_send = pd.concat(df_to_send)
        return df_to_send

    def _generate_forecasts_for_site(self, historical_values: pd.DataFrame,
                                     streamflow_df: pd.DataFrame, submission_site: pd.DataFrame,
                                     site_id: str):
        streamflow_df = generate_datetime_into_julian(dataframe=streamflow_df, datetime_column='datetime',
                                                      julian_column='julian_datetime', round_julian=3)

        submit = []
        for row_id, row in submission_site.iterrows():
            ############################################
            # GENERATE FORECAST FOR CURRENT ISSUE DATE #
            ############################################

            # Get information about datetime
            issue_date = row['issue_date']
            issue_date_julian = get_julian_date_from_datetime(current_date=issue_date, round_julian=3)
            issue_date_agg_start_julian = get_julian_date_from_datetime(current_date=issue_date,
                                                                        offset_days=self.aggregation_days)

            # Volume from the previous year
            known_historical_values = historical_values[historical_values['year'] < issue_date]
            known_streamflow_values = streamflow_df[streamflow_df['julian_datetime'] < issue_date_julian]

            # Add information about year in the dataframe
            known_streamflow_values['year'] = known_streamflow_values['datetime'].dt.year
            known_historical_values['year_number'] = known_historical_values['year'].dt.year

            # Fit model
            model, min_target, max_target = self._fit_model_based_on_historical_data(known_historical_values=known_historical_values,
                                                                                     known_streamflow_values=known_streamflow_values,
                                                                                     issue_date=issue_date, site_id=site_id)

            # Generate forecast
            agg_streamflow = known_streamflow_values[known_streamflow_values['julian_datetime'] >= issue_date_agg_start_julian]
            agg_streamflow = agg_streamflow[agg_streamflow['julian_datetime'] < issue_date_julian]
            dataset = self.__aggregate_features(agg_streamflow)

            predicted = model.predict(dataset[self.features_columns])[0]

            # Clip to borders
            if predicted > max_target:
                predicted = max_target
            if predicted < min_target:
                predicted = min_target

            row['volume_10'] = predicted - (predicted * self.lower_ratio)
            row['volume_50'] = predicted
            row['volume_90'] = predicted + (predicted * self.above_ratio)

            submit.append(pd.DataFrame(row).T)

        submit = pd.concat(submit)
        return submit

    def _fit_model_based_on_historical_data(self, known_historical_values: pd.DataFrame,
                                            known_streamflow_values: pd.DataFrame, issue_date: Timestamp,
                                            site_id: str):

        dataframe_for_model_fitting = []
        for historical_year in list(known_historical_values['year'].dt.year):
            # For each year in the past collect data for model fitting
            if historical_year not in list(known_streamflow_values['year'].unique()):
                continue
            # One year in advance for aggregation
            if historical_year - 1 not in list(known_streamflow_values['year'].unique()):
                continue

            # Find that day in the past (and min and max borders for data aggregation)
            years_offset = issue_date.year - historical_year
            aggregation_end_julian = get_julian_date_from_datetime(current_date=issue_date, offset_years=years_offset)
            aggregation_start_julian = get_julian_date_from_datetime(current_date=issue_date, offset_years=years_offset,
                                                                     offset_days=self.aggregation_days)

            agg_streamflow = known_streamflow_values[known_streamflow_values['julian_datetime'] >= aggregation_start_julian]
            agg_streamflow = agg_streamflow[agg_streamflow['julian_datetime'] < aggregation_end_julian]

            dataset = self.__aggregate_features(agg_streamflow)

            # Add target (for prediction)
            historical_data_for_year = known_historical_values[known_historical_values['year_number'] == historical_year]
            target = historical_data_for_year['volume'].values[0]

            dataset['historical_year'] = historical_year
            dataset['target'] = target
            dataframe_for_model_fitting.append(dataset)

        dataframe_for_model_fitting = pd.concat(dataframe_for_model_fitting)
        dataframe_for_model_fitting = dataframe_for_model_fitting.dropna()

        reg = RandomForestRegressor(n_estimators=15)
        reg.fit(dataframe_for_model_fitting[self.features_columns], dataframe_for_model_fitting['target'])
        min_target, max_target = min(dataframe_for_model_fitting['target']), max(dataframe_for_model_fitting['target'])
        if self.vis is True:
            # Make visualization 3d plot
            file_name = f'{issue_date}_site_{site_id}.png'
            title = f'USGS streamflow model for site {site_id} and issue date {issue_date}'
            created_spatial_plot(dataframe_for_model_fitting, reg, self.features_columns, self.vis_folder_name,
                                 file_name, title)

        return reg, min_target, max_target

    @staticmethod
    def __aggregate_features(agg_streamflow: pd.DataFrame):
        # Collect features for ML model
        mean_value = agg_streamflow['00060_Mean'].mean()
        sum_value = agg_streamflow['00060_Mean'].sum()
        min_value = agg_streamflow['00060_Mean'].min()
        max_value = agg_streamflow['00060_Mean'].max()

        dataset = pd.DataFrame({'mean_value': [mean_value], 'sum_value': [sum_value], 'min_value': [min_value],
                                'max_value': [max_value]})

        return dataset
