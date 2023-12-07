from pathlib import Path

import warnings

import pandas as pd

from wasu.development.models.repeating import SimpleRepeatingTrainModel
from wasu.development.validation import ModelValidation
from wasu.development.vis.visualization import TimeSeriesPlot

warnings.filterwarnings('ignore')


def validate_simple_model():
    validator = ModelValidation(folder_for_plots='simple_repeating')
    train_df = pd.read_csv(Path('../../data/train.csv'), parse_dates=['year'])
    submission_format = validator.generate_submission_format()

    # Define simple model to make a predictions
    model = SimpleRepeatingTrainModel(train_df=train_df)
    predicted = model.predict(submission_format)

    validator.compare_dataframes(predicted, train_df)


if __name__ == '__main__':
    validate_simple_model()
