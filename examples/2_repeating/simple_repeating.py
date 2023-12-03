from pathlib import Path

import warnings

import pandas as pd

from wasu.development.models.repeating import SimpleRepeatingTrainModel
from wasu.development.vis.visualization import TimeSeriesPlot

warnings.filterwarnings('ignore')


def generate_forecast_repeating_last_value():
    train_df = pd.read_csv(Path('../../data/train.csv'), parse_dates=['year'])
    submission_format = pd.read_csv(Path('../../data/submission_format.csv'), parse_dates=['issue_date'])

    # Define simple model to make a predictions
    model = SimpleRepeatingTrainModel(train_df=train_df)
    predicted = model.predict(submission_format)

    # Save into file
    model.save_predictions_as_submit(predicted, path='./results/simple_repeating_07_11_2023.csv')

    TimeSeriesPlot().predicted_time_series(predicted)


if __name__ == '__main__':
    generate_forecast_repeating_last_value()
