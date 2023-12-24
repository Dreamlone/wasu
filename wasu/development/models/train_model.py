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
    def fit(self, submission_format: pd.DataFrame, **kwargs) -> Union[str, Path]:
        raise NotImplementedError(f'Abstract method')

    @abstractmethod
    def predict(self, submission_format: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplementedError(f'Abstract method')

    def save_predictions_as_submit(self, predicted: pd.DataFrame, path: Union[None, str, Path] = None,
                                   submission_format: Union[pd.DataFrame, None] = None):
        """ Save results into desired submission format

        :param predicted: table with predicted values. Must contain the following columns:
            [site_id, issue_date, volume_10, volume_50, volume_90]
        :param path: path to the file where to save the results
        :parma submission_format: table with submission to imitate
        """
        if set(list(predicted.columns)) != {'site_id', 'issue_date', 'volume_10', 'volume_50', 'volume_90'}:
            raise ValueError(f'Columns in the dataframe with predicted values are not compatible with submission '
                             f'format! Please reduce columns to [site_id, issue_date, volume_10, volume_50, volume_90]')

        if path is None:
            path = 'default_submission.csv'
        if isinstance(path, str):
            path = Path(path).resolve()
        base_dir = path.parent
        base_dir.mkdir(exist_ok=True, parents=True)

        # Generate index for tables to identify "site - issue date" pair
        df = self.output.output_example
        if submission_format is not None:
            df = submission_format
        df['index'] = (df['site_id'].astype(str) + pd.to_datetime(df['issue_date']).dt.strftime('%Y-%m-%d'))
        predicted['index'] = (predicted['site_id'].astype(str) + pd.to_datetime(predicted['issue_date']).dt.strftime('%Y-%m-%d'))

        submit = df[['index']].merge(predicted, on='index')
        submit = submit.drop(columns=['index'])
        submit['issue_date'] = pd.to_datetime(submit['issue_date']).dt.strftime('%Y-%m-%d')
        submit.to_csv(path, index=False)

    def adjust_forecast(self, site: str, submission_site: pd.DataFrame):
        site_train_data = self.train_df[self.train_df['site_id'] == site]
        min_value = min(site_train_data['volume'])
        max_value = max(site_train_data['volume'])
        mean_value = site_train_data['volume'].mean()

        submission_site['volume_10'][submission_site['volume_10'] <= min_value] = min_value * 0.8
        submission_site['volume_10'][submission_site['volume_10'] >= max_value] = min_value * 0.8

        submission_site['volume_50'][submission_site['volume_50'] <= min_value] = mean_value
        submission_site['volume_50'][submission_site['volume_50'] >= max_value] = mean_value

        submission_site['volume_90'][submission_site['volume_90'] <= min_value] = max_value * 1.2
        submission_site['volume_90'][submission_site['volume_90'] >= max_value] = max_value * 1.2

        submission_site_updated = []
        for row_id, row in submission_site.iterrows():
            # Final forecast adjustment
            current_min = row.volume_10
            current_max = row.volume_90
            current_mean = row.volume_50
            if current_min > current_max:
                row.volume_10 = current_max
                row.volume_90 = current_min
            if current_mean <= row.volume_10 or current_mean >= row.volume_90:
                row.volume_50 = (row.volume_10 + row.volume_90) / 2
            submission_site_updated.append(pd.DataFrame(row).T)

        submission_site_updated = pd.concat(submission_site_updated)
        return submission_site_updated
