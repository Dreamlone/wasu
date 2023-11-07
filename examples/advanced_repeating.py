from pathlib import Path

import warnings

import pandas as pd

from wasu.development.models.train_model import AdvancedRepeatingTrainModel

warnings.filterwarnings('ignore')


def generate_forecast_repeating_last_value_using_test_sample():
    train_df = pd.read_csv(Path('../data/train.csv'), parse_dates=['year'])
    submission_format = pd.read_csv(Path('../data/submission_format.csv'), parse_dates=['issue_date'])

    # Create dataframe with monthly volumes
    parse_dates = ['forecast_year', 'year']
    test_monthly_df = pd.read_csv(Path('../data/test_monthly_naturalized_flow.csv'), parse_dates=parse_dates)
    train_monthly_df = pd.read_csv(Path('../data/train_monthly_naturalized_flow.csv'), parse_dates=parse_dates)
    monthly_df = pd.concat([train_monthly_df, test_monthly_df])

    # Define simple model to make a predictions
    model = AdvancedRepeatingTrainModel(train_df=train_df)
    predicted = model.predict(submission_format, monthly_df=monthly_df)

    # Save into file
    model.save_predictions_as_submit(predicted, path='./results/advanced_repeating_07_11_2023.csv')


if __name__ == '__main__':
    generate_forecast_repeating_last_value_using_test_sample()
