from abc import abstractmethod
from pathlib import Path
from typing import Union

import pandas as pd
from loguru import logger

from wasu.development.main import collect_usgs_streamflow_time_series_for_site
from wasu.development.output import Output


class TrainModel:
    """ Class for predicting values on train test sample """

    def __init__(self, train_df: pd.DataFrame):
        self.train_df = train_df
        self.output = Output()

    @abstractmethod
    def predict(self, submission_format: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplementedError(f'Abstract method')

    def save_predictions_as_submit(self, predicted: pd.DataFrame, path: Union[None, str, Path] = None):
        """ Save results into desired submission format

        :param predicted: table with predicted values. Must contain the following columns:
            [site_id, issue_date, volume_10, volume_50, volume_90]
        :param path: path to the file where to save the results
        """
        if set(list(predicted.columns)) != {'site_id', 'issue_date', 'volume_10', 'volume_50', 'volume_90'}:
            raise ValueError(f'Columns in the dataframe with predicted values are not compatible with submission '
                             f'format! Please reduce columns to [site_id, issue_date, volume_10, volume_50, volume_90]')

        if path is None:
            path = 'default_submission.csv'
        if isinstance(path, str):
            path = Path(path).resolve()

        # Generate index for tables to identify "site - issue date" pair
        df = self.output.output_example
        df['index'] = (df['site_id'].astype(str) + pd.to_datetime(df['issue_date']).dt.strftime('%Y-%m-%d'))
        predicted['index'] = (predicted['site_id'].astype(str) + pd.to_datetime(predicted['issue_date']).dt.strftime('%Y-%m-%d'))

        submit = df[['index']].merge(predicted, on='index')
        submit = submit.drop(columns=['index'])
        submit['issue_date'] = pd.to_datetime(predicted['issue_date']).dt.strftime('%Y-%m-%d')
        submit.to_csv(path, index=False)


class SimpleRepeatingTrainModel(TrainModel):
    """ Repeat last known values in 2004 year """

    def __init__(self, train_df: pd.DataFrame):
        super().__init__(train_df)

        self.last_year = 2004
        self.lower_ratio = 0.1
        self.above_ratio = 0.1

    def predict(self, submission_format: pd.DataFrame, **kwargs):
        self.train_df = self.train_df.dropna()

        df_to_send = []
        # For every site provide calculations
        for site in list(submission_format['site_id'].unique()):
            submission_site = submission_format[submission_format['site_id'] == site]

            # Get last known volume
            site_df = self.train_df[self.train_df['site_id'] == site]
            last_known_value = site_df[site_df['year'].dt.year == self.last_year]
            predicted_income = last_known_value['volume'].values[0]

            lower_predicted_income = predicted_income - (predicted_income * self.lower_ratio)
            above_predicted_income = predicted_income + (predicted_income * self.above_ratio)

            submission_site['volume_10'] = lower_predicted_income
            submission_site['volume_50'] = predicted_income
            submission_site['volume_90'] = above_predicted_income

            df_to_send.append(submission_site)

        df_to_send = pd.concat(df_to_send)
        return df_to_send


class AdvancedRepeatingTrainModel(TrainModel):
    """ Repeat last known volume based on some test sample information """

    def __init__(self, train_df: pd.DataFrame):
        super().__init__(train_df)

        self.lower_ratio = 0.3
        self.above_ratio = 0.3

    def predict(self, submission_format: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.train_df = self.train_df.dropna()

        metadata: pd.DataFrame = kwargs['metadata']
        df_to_send = []
        # For every site
        for site in list(submission_format['site_id'].unique()):
            metadata_site = metadata[metadata['site_id'] == site]
            season_start_month = metadata_site['season_start_month'].values[0]
            season_end_month = metadata_site['season_end_month'].values[0]

            logger.info(f'Generating forecast for site {site}. Start season month: {season_start_month}. '
                        f'End season month: {season_end_month}')

            submission_site = submission_format[submission_format['site_id'] == site]
            site_df = self.train_df[self.train_df['site_id'] == site]
            site_df = site_df.sort_values(by='year')

            submit = self._generate_forecasts_for_site(historical_values=site_df,
                                                       submission_site=submission_site)

            df_to_send.append(submit)

        df_to_send = pd.concat(df_to_send)
        return df_to_send

    def _generate_forecasts_for_site(self, historical_values: pd.DataFrame, submission_site: pd.DataFrame):
        """ Generate forecasts for desired site """
        submit = []
        for row_id, row in submission_site.iterrows():
            # For each datetime label
            issue_year = row['issue_date'].year

            # Volume from the previous year
            last_known_value = historical_values[historical_values['year'].dt.year == issue_year - 1]
            previous_year_value = last_known_value['volume'].values[0]

            row['volume_10'] = previous_year_value - (previous_year_value * self.lower_ratio)
            row['volume_50'] = previous_year_value
            row['volume_90'] = previous_year_value + (previous_year_value * self.above_ratio)

            submit.append(pd.DataFrame(row).T)

        submit = pd.concat(submit)
        return submit


class StreamFlowRegression(TrainModel):
    """ Create forecasts based on streamflow USGS data """

    def __init__(self, train_df: pd.DataFrame):
        super().__init__(train_df)

        self.lower_ratio = 0.3
        self.above_ratio = 0.3

    def predict(self, submission_format: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.train_df = self.train_df.dropna()

        metadata: pd.DataFrame = kwargs['metadata']
        path_to_streamflow: Path = kwargs['path_to_streamflow']

        df_to_send = []
        # For every site
        for site in list(submission_format['site_id'].unique()):
            streamflow_df = collect_usgs_streamflow_time_series_for_site(path_to_streamflow, site)
            if streamflow_df is None:
                # TODO Generate alternative forecast in that case
                continue

            metadata_site = metadata[metadata['site_id'] == site]
            season_start_month = metadata_site['season_start_month'].values[0]
            season_end_month = metadata_site['season_end_month'].values[0]

            logger.info(f'Generating forecast for site {site}. Start season month: {season_start_month}. '
                        f'End season month: {season_end_month}')

            submission_site = submission_format[submission_format['site_id'] == site]
            site_df = self.train_df[self.train_df['site_id'] == site]
            site_df = site_df.sort_values(by='year')

            submit = self._generate_forecasts_for_site(historical_values=site_df,
                                                       streamflow_df=streamflow_df,
                                                       submission_site=submission_site)

            df_to_send.append(submit)

        df_to_send = pd.concat(df_to_send)
        return df_to_send

    def _generate_forecasts_for_site(self, historical_values: pd.DataFrame,
                                     streamflow_df: pd.DataFrame, submission_site: pd.DataFrame):
        streamflow_df['julian_datetime'] = pd.DatetimeIndex(streamflow_df['datetime']).to_julian_date()
        streamflow_df['julian_datetime'] = streamflow_df['julian_datetime'] .round(3)

        submit = []
        for row_id, row in submission_site.iterrows():
            # For each datetime label
            issue_date = row['issue_date']

            # Get also issue date in julian
            issue_date_df = pd.DataFrame({'issue_date': [issue_date]})
            issue_date_df['julian_datetime'] = pd.DatetimeIndex(issue_date_df['issue_date']).to_julian_date()
            issue_date_df['julian_datetime'] = issue_date_df['julian_datetime'].round(3)
            issue_date_julian = issue_date_df['julian_datetime'].values[0]

            # Volume from the previous year
            known_historical_values = historical_values[historical_values['year'] < issue_date]
            known_streamflow_values = streamflow_df[streamflow_df['julian_datetime'] < issue_date_julian]

            # Aggregate the data
            known_streamflow_values['year'] = known_streamflow_values['datetime'].dt.year
            dataframe_for_model_fitting = []
            for historical_year in list(known_historical_values['year'].dt.year):
                if historical_year not in list(known_streamflow_values['year'].unique()):
                    continue

                same_datetime_in_past = []
                print(issue_date.dayofyear, issue_date.day, issue_date)

            row['volume_10'] = 0
            row['volume_50'] = 0
            row['volume_90'] = 0

            submit.append(pd.DataFrame(row).T)

        submit = pd.concat(submit)
        return submit
