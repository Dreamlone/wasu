import numpy as np
import pandas as pd


class AdvancedRepeatingCustomModel:
    """ Model imitates signature of sklearn models """

    def __init__(self, site_id: str, alpha: float):
        self.site_id = site_id
        self.alpha = alpha
        self.train_df = None

        self.lower_ratio = 0.3
        self.above_ratio = 0.3

    def fit(self, train_df: pd.DataFrame):
        """ Store historical values in the memory """
        self.train_df = train_df

    def predict(self, submission_site: pd.DataFrame):
        """ Apply  """
        submit = []
        for row_id, row in submission_site.iterrows():
            # For each datetime label
            issue_year = row['issue_date'].year

            # Volume from the previous year
            last_known_value = self.train_df[self.train_df['year'].dt.year == issue_year - 1]
            if last_known_value is None or len(last_known_value) < 1:
                # Check two years ago
                last_known_value = self.train_df[self.train_df['year'].dt.year == issue_year - 2]
            if last_known_value is None or len(last_known_value) < 1:
                last_known_value = self.train_df[self.train_df['year'].dt.year == issue_year - 3]
            previous_year_value = last_known_value['volume'].values[0]

            if self.alpha == 0.1:
                predicted = previous_year_value - (previous_year_value * self.lower_ratio)
            elif self.alpha == 0.5:
                predicted = previous_year_value
            else:
                predicted = previous_year_value + (previous_year_value * self.above_ratio)

            submit.append(predicted)

        return np.array(submit)
