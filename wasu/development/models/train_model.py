from abc import abstractmethod
from pathlib import Path
from typing import Union

import pandas as pd
from loguru import logger
from sklearn.linear_model import LinearRegression

from wasu.development.main import collect_usgs_streamflow_time_series_for_site
from wasu.development.models.date_utils import generate_datetime_into_julian, get_julian_date_from_datetime
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
        submit['issue_date'] = pd.to_datetime(submit['issue_date']).dt.strftime('%Y-%m-%d')
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

            submit = self.generate_forecasts_for_site(historical_values=site_df,
                                                      submission_site=submission_site)

            df_to_send.append(submit)

        df_to_send = pd.concat(df_to_send)
        return df_to_send

    def generate_forecasts_for_site(self, historical_values: pd.DataFrame, submission_site: pd.DataFrame):
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
        self.backup_model = AdvancedRepeatingTrainModel(train_df)

        self.lower_ratio = 0.3
        self.above_ratio = 0.3
        # One month
        self.aggregation_days = 45
        self.features_columns = ['mean_value', 'sum_value', 'min_value', 'max_value']

    def predict(self, submission_format: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.train_df = self.train_df.dropna()

        metadata: pd.DataFrame = kwargs['metadata']
        path_to_streamflow: Path = kwargs['path_to_streamflow']

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

            metadata_site = metadata[metadata['site_id'] == site]
            season_start_month = metadata_site['season_start_month'].values[0]
            season_end_month = metadata_site['season_end_month'].values[0]

            logger.info(f'Generating forecast for site {site}. Start season month: {season_start_month}. '
                        f'End season month: {season_end_month}')

            try:
                submit = self._generate_forecasts_for_site(historical_values=site_df,
                                                           streamflow_df=streamflow_df,
                                                           submission_site=submission_site)
                df_to_send.append(submit)
            except Exception as ex:
                logger.warning(f'Can not generate forecast for site {site} due to {ex}. Apply Advanced repeating')
                submit = self.backup_model.generate_forecasts_for_site(historical_values=site_df,
                                                                       submission_site=submission_site)
                df_to_send.append(submit)

        df_to_send = pd.concat(df_to_send)
        return df_to_send

    def _generate_forecasts_for_site(self, historical_values: pd.DataFrame,
                                     streamflow_df: pd.DataFrame, submission_site: pd.DataFrame):
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
                                                                                     issue_date=issue_date)

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
                                            known_streamflow_values: pd.DataFrame, issue_date):

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

        reg = LinearRegression()
        reg.fit(dataframe_for_model_fitting[self.features_columns], dataframe_for_model_fitting['target'])
        min_target, max_target = min(dataframe_for_model_fitting['target']), max(dataframe_for_model_fitting['target'])
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
