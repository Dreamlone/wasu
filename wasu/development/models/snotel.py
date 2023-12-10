from pathlib import Path

import pandas as pd
from loguru import logger
from pandas import Timestamp
from sklearn.ensemble import RandomForestRegressor

from wasu.development.data.snotel import collect_snotel_data_for_site
from wasu.development.models.date_utils import generate_datetime_into_julian, get_julian_date_from_datetime
from wasu.development.models.repeating import AdvancedRepeatingTrainModel
from wasu.development.models.train_model import TrainModel
from wasu.development.paths import path_to_data_folder


class SnotelFlowRegression(TrainModel):
    """ Create forecasts based on streamflow USGS data """

    def __init__(self, train_df: pd.DataFrame, aggregation_days: int = 90):
        super().__init__(train_df)
        self.backup_model = AdvancedRepeatingTrainModel(train_df)

        self.lower_ratio = 0.3
        self.above_ratio = 0.3
        # One month
        self.aggregation_days = aggregation_days
        self.features_columns = ['mean_PREC_DAILY', 'mean_TAVG_DAILY', 'mean_TMAX_DAILY', 'mean_TMIN_DAILY',
                                 'mean_WTEQ_DAILY',
                                 'sum_PREC_DAILY', 'sum_TAVG_DAILY', 'sum_TMAX_DAILY', 'sum_TMIN_DAILY',
                                 'sum_WTEQ_DAILY',
                                 'max_PREC_DAILY', 'max_TAVG_DAILY', 'max_TMAX_DAILY', 'max_TMIN_DAILY',
                                 'max_WTEQ_DAILY',
                                 'min_PREC_DAILY', 'min_TAVG_DAILY', 'min_TMAX_DAILY', 'min_TMIN_DAILY',
                                 'min_WTEQ_DAILY']

        self.statistics = []
        self.vis = False

    def predict(self, submission_format: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """ Make predictions based on SNOTEL data

        :param submission_format: pandas  table with example of output
        :param kwargs: additional parameters
            - enable_spatial_aggregation - aggregated data from all stations per datetime label or not
            - collect_only_in_basin - is there a need to use only stations which is included in the basin of site or all
        """
        self.train_df = self.train_df.dropna()

        metadata: pd.DataFrame = kwargs['metadata']
        path_to_snotel: Path = kwargs['path_to_snotel']
        enable_spatial_aggregation: Path = kwargs.get('enable_spatial_aggregation')
        collect_only_in_basin = kwargs.get('collect_only_in_basin')
        if collect_only_in_basin is None:
            collect_only_in_basin = True
        self.vis: bool = kwargs.get('vis')
        if self.vis is None:
            self.vis = False

        df_to_send = []
        # For every site
        for site in list(submission_format['site_id'].unique()):
            snotel_df = collect_snotel_data_for_site(path_to_snotel, site, collect_only_in_basin=collect_only_in_basin)

            submission_site = submission_format[submission_format['site_id'] == site]
            site_df = self.train_df[self.train_df['site_id'] == site]
            site_df = site_df.sort_values(by='year')

            if snotel_df is None or len(snotel_df) < 1:
                logger.warning(f'Can not obtain SNOTEL data for site {site}. Apply Advanced repeating')
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
                if enable_spatial_aggregation is not None and enable_spatial_aggregation is True:
                    # Calculate mean values per basin
                    logger.info(f'Len before aggregation: {len(snotel_df)}')
                    snotel_df = snotel_df.groupby('date').agg({'PREC_DAILY': 'mean', 'TAVG_DAILY': 'mean',
                                                               'TMAX_DAILY': 'mean', 'TMIN_DAILY': 'mean',
                                                               'WTEQ_DAILY': 'mean'})
                    snotel_df = snotel_df.reset_index()
                    logger.info(f'Len after aggregation: {len(snotel_df)}')
                    snotel_df = snotel_df.dropna()
                submit = self._generate_forecasts_for_site(historical_values=site_df, snotel_df=snotel_df,
                                                           submission_site=submission_site, site_id=site)
                df_to_send.append(submit)
                self.statistics.append([site, 'snotel model'])
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

    def generate_forecasts_for_site(self, site: str, submission_site: pd.DataFrame) -> pd.DataFrame:
        """ Generate SNOTEL forecast with default parameters """
        path_to_snotel = Path(path_to_data_folder(), 'snodas')
        enable_spatial_aggregation = True

        snotel_df = collect_snotel_data_for_site(path_to_snotel, site, collect_only_in_basin=False)

        site_df = self.train_df[self.train_df['site_id'] == site]
        site_df = site_df.sort_values(by='year')

        if snotel_df is None or len(snotel_df) < 1:
            logger.warning(f'Can not obtain SNOTEL data for site {site}. Apply Advanced repeating')
            submit = self.backup_model.generate_forecasts_for_site(historical_values=site_df,
                                                                   submission_site=submission_site)
            return submit

        try:
            if enable_spatial_aggregation is not None and enable_spatial_aggregation is True:
                snotel_df = snotel_df.groupby('date').agg({'PREC_DAILY': 'mean', 'TAVG_DAILY': 'mean',
                                                           'TMAX_DAILY': 'mean', 'TMIN_DAILY': 'mean',
                                                           'WTEQ_DAILY': 'mean'})
                snotel_df = snotel_df.reset_index()
                snotel_df = snotel_df.dropna()
            submit = self._generate_forecasts_for_site(historical_values=site_df, snotel_df=snotel_df,
                                                       submission_site=submission_site, site_id=site)
            return submit
        except Exception as ex:
            logger.warning(f'Can not generate forecast for site {site} due to {ex}. Apply Advanced repeating')
            submit = self.backup_model.generate_forecasts_for_site(historical_values=site_df,
                                                                   submission_site=submission_site)
            return submit

    def _generate_forecasts_for_site(self, historical_values: pd.DataFrame,
                                     snotel_df: pd.DataFrame, submission_site: pd.DataFrame, site_id: str):
        snotel_df = generate_datetime_into_julian(dataframe=snotel_df, datetime_column='date',
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
            known_snotel_values = snotel_df[snotel_df['julian_datetime'] < issue_date_julian]
            known_snotel_values = known_snotel_values[['PREC_DAILY', 'TAVG_DAILY', 'TMAX_DAILY',
                                                       'TMIN_DAILY', 'WTEQ_DAILY', 'date', 'julian_datetime']]

            # Add information about year in the dataframe
            known_snotel_values['year'] = known_snotel_values['date'].dt.year
            known_historical_values['year_number'] = known_historical_values['year'].dt.year

            # Fit model
            model, min_target, max_target = self._fit_model_based_on_historical_data(known_historical_values=known_historical_values,
                                                                                     known_snotel_values=known_snotel_values,
                                                                                     issue_date=issue_date,
                                                                                     site_id=site_id)

            # Generate forecast
            agg_snotel = known_snotel_values[known_snotel_values['julian_datetime'] >= issue_date_agg_start_julian]
            agg_snotel = agg_snotel[agg_snotel['julian_datetime'] < issue_date_julian]
            dataset = self.__aggregate_features(agg_snotel)

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
                                            known_snotel_values: pd.DataFrame,
                                            issue_date: Timestamp,
                                            site_id: str):

        dataframe_for_model_fitting = []
        for historical_year in list(known_historical_values['year'].dt.year):
            # For each year in the past collect data for model fitting
            if historical_year not in list(known_snotel_values['year'].unique()):
                continue
            # One year in advance for aggregation
            if historical_year - 1 not in list(known_snotel_values['year'].unique()):
                continue

            # Find that day in the past (and min and max borders for data aggregation)
            years_offset = issue_date.year - historical_year
            aggregation_end_julian = get_julian_date_from_datetime(current_date=issue_date, offset_years=years_offset)
            aggregation_start_julian = get_julian_date_from_datetime(current_date=issue_date, offset_years=years_offset,
                                                                     offset_days=self.aggregation_days)

            agg_snotel = known_snotel_values[known_snotel_values['julian_datetime'] >= aggregation_start_julian]
            agg_snotel = agg_snotel[agg_snotel['julian_datetime'] < aggregation_end_julian]

            dataset = self.__aggregate_features(agg_snotel)

            # Add target (for prediction)
            historical_data_for_year = known_historical_values[known_historical_values['year_number'] == historical_year]
            target = historical_data_for_year['volume'].values[0]

            dataset['historical_year'] = historical_year
            dataset['target'] = target
            dataframe_for_model_fitting.append(dataset)

        dataframe_for_model_fitting = pd.concat(dataframe_for_model_fitting)
        dataframe_for_model_fitting = dataframe_for_model_fitting.dropna()

        reg = RandomForestRegressor(n_estimators=40)
        reg.fit(dataframe_for_model_fitting[self.features_columns], dataframe_for_model_fitting['target'])
        min_target, max_target = min(dataframe_for_model_fitting['target']), max(dataframe_for_model_fitting['target'])
        return reg, min_target, max_target

    @staticmethod
    def __aggregate_features(agg_snotel: pd.DataFrame):
        # Collect statistics
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
