from pathlib import Path

import warnings

import pandas as pd

from wasu.development.models.snotel import SnotelFlowRegression
from wasu.development.vis.visualization import TimeSeriesPlot

warnings.filterwarnings('ignore')


def generate_forecast_based_on_snotel():
    train_df = pd.read_csv(Path('../../data/train.csv'), parse_dates=['year'])
    submission_format = pd.read_csv(Path('../../data/submission_format.csv'), parse_dates=['issue_date'])

    # Load dataframe with metadata
    metadata = pd.read_csv(Path('../../data/metadata_TdPVeJC.csv'))
    path_to_snotel = Path('../../data/snotel').resolve()

    model = SnotelFlowRegression(train_df=train_df, aggregation_days=50)
    predicted = model.predict(submission_format, metadata=metadata, path_to_snotel=path_to_snotel,
                              enable_spatial_aggregation=True, collect_only_in_basin=True)

    # Save into file
    model.save_predictions_as_submit(predicted, path='./results/snotel_50_basin.csv')

    TimeSeriesPlot().predicted_time_series(predicted, plots_folder_name='predictions_snotel')


if __name__ == '__main__':
    generate_forecast_based_on_snotel()
