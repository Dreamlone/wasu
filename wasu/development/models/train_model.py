from abc import abstractmethod
from pathlib import Path
from typing import Union

import pandas as pd

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
        df['index'] = (df['site_id'].astype(str) + df['issue_date'].astype(str))
        predicted['index'] = (predicted['site_id'].astype(str) + predicted['issue_date'].astype(str))

        submit = df[['index']].merge(predicted, on='index')
        submit = submit.drop(columns=['index'])
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

    def predict(self, submission_format: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.train_df = self.train_df.dropna()

        monthly_df: pd.DataFrame = kwargs['monthly_df']
        monthly_df = monthly_df.sort_values(by=['site_id', 'forecast_year', 'year'])

        df_to_send = []
        # For every site
        for site in list(submission_format['site_id'].unique()):
            submission_site = submission_format[submission_format['site_id'] == site]
            site_df = self.train_df[self.train_df['site_id'] == site]
            site_df = site_df.sort_values(by='year')

            monthly_site_df = monthly_df[monthly_df['site_id'] == site]
            monthly_site_df = monthly_site_df.dropna()
            adjusted_df = []
            for row_id, row in monthly_site_df.iterrows():
                if row['month'] in [10, 11, 12]:
                    row['year'] = row['year'] + pd.DateOffset(years=1)
                    adjusted_df.append(pd.DataFrame(row).T)
            adjusted_df = pd.concat(adjusted_df)
            cumulative = adjusted_df.groupby(['year']).agg({'volume': 'sum'}).reset_index()

            last_known_value = site_df[site_df['year'].dt.year == 2004]
            predicted_income = last_known_value['volume'].values[0]
            lower_predicted_income = predicted_income - (predicted_income * 0.1)
            above_predicted_income = predicted_income + (predicted_income * 0.1)

            submission_site['volume_10'] = lower_predicted_income
            submission_site['volume_50'] = predicted_income
            submission_site['volume_90'] = above_predicted_income

            df_to_send.append(submission_site)

        df_to_send = pd.concat(df_to_send)
        return df_to_send
