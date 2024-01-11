from pathlib import Path

import warnings

import pandas as pd

from wasu.development.models.repeating import SimpleRepeatingTrainModel
from wasu.development.validation import ModelValidation
from wasu.development.vis.visualization import TimeSeriesPlot

warnings.filterwarnings('ignore')


def generate_forecast_repeating_last_value():
    validation_year = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    train_df = pd.read_csv(Path('../../data/train.csv'), parse_dates=['year'])
    # submission_format = pd.read_csv(Path('../../data/submission_format.csv'), parse_dates=['issue_date'])
    validator = ModelValidation(folder_for_plots='simple_repeating',
                                years_to_validate=validation_year)
    submission_format = validator.generate_submission_format()

    # Define simple model to make a predictions
    model = SimpleRepeatingTrainModel(train_df=train_df, last_year=2015)
    predicted = model.predict(submission_format)

    # Save into file
    model.save_predictions_as_submit(predicted, path='./results/simple_repeating.csv')
    validator.compare_dataframes(predicted, train_df,
                                 save_predicted_vs_actual_into_file='./validation/simple_repeating.csv')

    TimeSeriesPlot(test_years=validation_year).predicted_time_series(predicted,
                                                                     plots_folder_name='predictions_simple_repeating')


if __name__ == '__main__':
    generate_forecast_repeating_last_value()
