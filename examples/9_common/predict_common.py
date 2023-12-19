from pathlib import Path

import warnings

import pandas as pd

from wasu.development.models.common import CommonRegression
from wasu.development.models.snotel import SnotelFlowRegression
from wasu.development.vis.visualization import TimeSeriesPlot

warnings.filterwarnings('ignore')


def generate_forecast_based_on_snotel():
    method = 'linear'
    aggregation_days_snodas = 28
    aggregation_days_snotel = 80
    train_df = pd.read_csv(Path('../../data/train.csv'), parse_dates=['year'])
    submission_format = pd.read_csv(Path('../../data/submission_format.csv'), parse_dates=['issue_date'])

    # Load dataframe with metadata
    metadata = pd.read_csv(Path('../../data/metadata_TdPVeJC.csv'))
    path_to_snotel = Path('../../data/snotel').resolve()
    path_to_snodas = Path('../../data/snodas_csv').resolve()
    path_to_teleconnections = Path('../../data/teleconnections').resolve()

    model = CommonRegression(train_df=train_df, method=method,
                             aggregation_days_snodas=aggregation_days_snodas,
                             aggregation_days_snotel=aggregation_days_snotel)
    predicted = model.predict(submission_format, metadata=metadata, path_to_snotel=path_to_snotel,
                              path_to_snodas=path_to_snodas, path_to_teleconnections=path_to_teleconnections)

    # Save into file
    model.save_predictions_as_submit(predicted,
                                     path=f'./results/{method}_{aggregation_days_snodas}_{aggregation_days_snotel}.csv')

    TimeSeriesPlot().predicted_time_series(predicted, plots_folder_name='predictions_common')


if __name__ == '__main__':
    generate_forecast_based_on_snotel()
