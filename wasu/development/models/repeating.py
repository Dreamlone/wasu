import pandas as pd
from loguru import logger

from wasu.development.models.train_model import TrainModel


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
