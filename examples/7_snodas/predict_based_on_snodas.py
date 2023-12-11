from pathlib import Path

import warnings

import pandas as pd

from wasu.development.models.snodas import SnodasRegression
from wasu.development.models.snotel import SnotelFlowRegression
from wasu.development.vis.visualization import TimeSeriesPlot

warnings.filterwarnings('ignore')


def generate_forecast_based_on_snodas():
    train_df = pd.read_csv(Path('../../data/train.csv'), parse_dates=['year'])
    submission_format = pd.read_csv(Path('../../data/submission_format.csv'), parse_dates=['issue_date'])

    # Load dataframe with metadata
    metadata = pd.read_csv(Path('../../data/metadata_TdPVeJC.csv'))
    path_to_snodas = Path('../../data/snodas_csv').resolve()

    model = SnodasRegression(train_df=train_df, aggregation_days=180)
    predicted = model.predict(submission_format, metadata=metadata, path_to_snodas=path_to_snodas)

    # Save into file
    model.save_predictions_as_submit(predicted, path='./results/snodas_180.csv')

    TimeSeriesPlot().predicted_time_series(predicted, plots_folder_name='predictions_snodas')


if __name__ == '__main__':
    generate_forecast_based_on_snodas()
