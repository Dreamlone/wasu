from pathlib import Path

import warnings

import pandas as pd

from wasu.development.models.train_model import AdvancedRepeatingTrainModel, StreamFlowRegression
from wasu.development.vis.visualization import TimeSeriesPlot

warnings.filterwarnings('ignore')


def generate_forecast_based_on_streamflow():
    train_df = pd.read_csv(Path('../data/train.csv'), parse_dates=['year'])
    submission_format = pd.read_csv(Path('../data/submission_format.csv'), parse_dates=['issue_date'])

    # Load dataframe with metadata
    metadata = pd.read_csv(Path('../data/metadata_TdPVeJC.csv'))
    path_to_streamflow = Path('../data/usgs_streamflow').resolve()

    model = StreamFlowRegression(train_df=train_df)
    predicted = model.predict(submission_format, metadata=metadata, path_to_streamflow=path_to_streamflow)

    # Save into file
    model.save_predictions_as_submit(predicted, path='./results/advanced_repeating_07_11_2023.csv')

    TimeSeriesPlot().predicted_time_series(predicted)


if __name__ == '__main__':
    generate_forecast_based_on_streamflow()
