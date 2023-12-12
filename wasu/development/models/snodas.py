from pathlib import Path

import pandas as pd
from loguru import logger
from pandas import Timestamp
from sklearn.ensemble import RandomForestRegressor

from wasu.development.data.snodas import collect_snodas_data_for_site
from wasu.development.models.date_utils import generate_datetime_into_julian, get_julian_date_from_datetime
from wasu.development.models.snotel import SnotelFlowRegression
from wasu.development.models.train_model import TrainModel


class SnodasRegression(TrainModel):
    """ Create forecasts based on SNODAS data """

    def __init__(self, train_df: pd.DataFrame):
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

        self.all_features = []
        for column in self.features_columns:
            self.all_features.extend([f'mean_{column}', f'min_{column}', f'max_{column}'])
        self.statistics = []
        self.vis = False

    def predict(self, submission_format: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """ Make predictions based on SNODAS data """
        self.train_df = self.train_df.dropna()

        metadata: pd.DataFrame = kwargs['metadata']
        path_to_snodas: Path = kwargs['path_to_snodas']
        self.vis: bool = kwargs.get('vis')
        if self.vis is None:
            self.vis = False

        df_to_send = []
        # For every site
        for site in list(submission_format['site_id'].unique()):
            snodas_df = collect_snodas_data_for_site(path_to_snodas, site)

            submission_site = submission_format[submission_format['site_id'] == site]
            site_df = self.train_df[self.train_df['site_id'] == site]
            site_df = site_df.sort_values(by='year')
            if snodas_df is None or len(snodas_df) < 1:
                logger.warning(f'Can not obtain SNODAS data for site {site}. Apply SNOTEL-based model prediction')
                submit = self.backup_model.generate_forecasts_for_site(site=site, submission_site=submission_site)
                df_to_send.append(submit)
                self.statistics.append([site, 'backup'])
                continue

            metadata_site = metadata[metadata['site_id'] == site]
            season_start_month = metadata_site['season_start_month'].values[0]
            season_end_month = metadata_site['season_end_month'].values[0]

            logger.info(f'Generating forecast for site {site}. Start season month: {season_start_month}. '
                        f'End season month: {season_end_month}')

            try:
                submit = self._generate_forecasts_for_site(historical_values=site_df, snodas_df=snodas_df,
                                                           submission_site=submission_site, site_id=site)
                df_to_send.append(submit)
                self.statistics.append([site, 'snodas model'])
            except Exception as ex:
                logger.warning(f'Can not generate forecast for site {site} due to {ex}. Apply SNOTEL-based model prediction')
                submit = self.backup_model.generate_forecasts_for_site(site=site, submission_site=submission_site)
                df_to_send.append(submit)
                self.statistics.append([site, 'backup'])

        for case in self.statistics:
            print(f'Site: {case[0]}. Model: {case[1]}')

        df_to_send = pd.concat(df_to_send)
        return df_to_send

    def _generate_forecasts_for_site(self, historical_values: pd.DataFrame,
                                     snodas_df: pd.DataFrame, submission_site: pd.DataFrame, site_id: str):
        snodas_df = generate_datetime_into_julian(dataframe=snodas_df, datetime_column='datetime',
                                                  julian_column='julian_datetime', round_julian=3)

        submit = []
        for row_id, row in submission_site.iterrows():
            ############################################
            # GENERATE FORECAST FOR CURRENT ISSUE DATE #
            ############################################

            # Get information about datetime
            issue_date = row['issue_date']
            issue_date_julian = get_julian_date_from_datetime(current_date=issue_date, round_julian=3)
            aggregation_days = self._choose_aggregation_period_based_of_the_day_of_year(issue_date)
            issue_date_agg_start_julian = get_julian_date_from_datetime(current_date=issue_date,
                                                                        offset_days=aggregation_days)

            # Volume from the previous year
            known_historical_values = historical_values[historical_values['year'] < issue_date]
            known_snodas_values = snodas_df[snodas_df['julian_datetime'] < issue_date_julian]

            # Add information about year in the dataframe
            known_snodas_values['year'] = known_snodas_values['datetime'].dt.year
            known_historical_values['year_number'] = known_historical_values['year'].dt.year

            # Fit model
            model, min_target, max_target = self._fit_model_based_on_historical_data(known_historical_values=known_historical_values,
                                                                                     known_snodas_values=known_snodas_values,
                                                                                     issue_date=issue_date,
                                                                                     site_id=site_id)

            # Generate forecast
            agg_snodas = known_snodas_values[known_snodas_values['julian_datetime'] >= issue_date_agg_start_julian]
            agg_snodas = agg_snodas[agg_snodas['julian_datetime'] < issue_date_julian]
            dataset = self.__aggregate_features(agg_snodas)

            predicted = model.predict(dataset[self.all_features])[0]

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
                                            known_snodas_values: pd.DataFrame,
                                            issue_date: Timestamp,
                                            site_id: str):

        dataframe_for_model_fitting = []
        for historical_year in list(known_historical_values['year'].dt.year):
            # For each year in the past collect data for model fitting
            if historical_year not in list(known_snodas_values['year'].unique()):
                continue
            # One year in advance for aggregation
            if historical_year - 1 not in list(known_snodas_values['year'].unique()):
                continue

            # Find that day in the past (and min and max borders for data aggregation)
            years_offset = issue_date.year - historical_year
            aggregation_days = self._choose_aggregation_period_based_of_the_day_of_year(issue_date)
            aggregation_end_julian = get_julian_date_from_datetime(current_date=issue_date, offset_years=years_offset)
            aggregation_start_julian = get_julian_date_from_datetime(current_date=issue_date, offset_years=years_offset,
                                                                     offset_days=aggregation_days)

            agg_snodas = known_snodas_values[known_snodas_values['julian_datetime'] >= aggregation_start_julian]
            agg_snodas = agg_snodas[agg_snodas['julian_datetime'] < aggregation_end_julian]

            dataset = self.__aggregate_features(agg_snodas)

            # Add target (for prediction)
            historical_data_for_year = known_historical_values[known_historical_values['year_number'] == historical_year]
            target = historical_data_for_year['volume'].values[0]

            dataset['historical_year'] = historical_year
            dataset['target'] = target
            dataframe_for_model_fitting.append(dataset)

        dataframe_for_model_fitting = pd.concat(dataframe_for_model_fitting)
        dataframe_for_model_fitting = dataframe_for_model_fitting.dropna()

        reg = RandomForestRegressor(n_estimators=50, random_state=2023)
        reg.fit(dataframe_for_model_fitting[self.all_features], dataframe_for_model_fitting['target'])
        min_target, max_target = min(dataframe_for_model_fitting['target']), max(dataframe_for_model_fitting['target'])
        return reg, min_target, max_target

    def __aggregate_features(self, agg_snodas: pd.DataFrame):
        # Collect statistics
        agg_snodas = agg_snodas[self.features_columns]
        agg_snodas = agg_snodas.dropna()

        dataset = pd.DataFrame()
        for feature in self.features_columns:
            mean_value = agg_snodas[feature].mean()
            min_value = agg_snodas[feature].min()
            max_value = agg_snodas[feature].max()

            dataset[f'mean_{feature}'] = [mean_value]
            dataset[f'min_{feature}'] = [min_value]
            dataset[f'max_{feature}'] = [max_value]

        return dataset

    def _choose_aggregation_period_based_of_the_day_of_year(self, issue_date: Timestamp):
        dayofyear = issue_date.dayofyear

        if 0 <= dayofyear <= 30:
            return 60
        elif 30 < dayofyear <= 60:
            return 115
        elif 60 < dayofyear <= 90:
            return 150
        else:
            return 150
