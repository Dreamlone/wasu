from pathlib import Path

import warnings

import pandas as pd

from wasu.development.models.common import CommonRegression
from wasu.development.validation import ModelValidation
from wasu.development.vis.visualization import TimeSeriesPlot

warnings.filterwarnings('ignore')


def generate_forecast_based_on_snotel():
    method = 'linear'
    aggregation_days_snodas = 21
    aggregation_days_snotel = 120
    aggregation_days_pdsi = 124
    train_df = pd.read_csv(Path('../../data/train.csv'), parse_dates=['year'])
    submission_format = ModelValidation(folder_for_plots='common').generate_submission_format()

    # Load dataframe with metadata
    metadata = pd.read_csv(Path('../../data/metadata_TdPVeJC.csv'))
    path_to_snotel = Path('../../data/snotel').resolve()
    path_to_snodas = Path('../../data/snodas_csv').resolve()
    path_to_teleconnections = Path('../../data/teleconnections').resolve()
    path_to_pdsi = Path('../../data/pdsi_csv').resolve()

    model = CommonRegression(train_df=train_df, method=method,
                             aggregation_days_snodas=aggregation_days_snodas,
                             aggregation_days_snotel=aggregation_days_snotel,
                             aggregation_days_pdsi=aggregation_days_pdsi)
    predicted = model.predict(submission_format, metadata=metadata, path_to_snotel=path_to_snotel,
                              path_to_snodas=path_to_snodas, path_to_teleconnections=path_to_teleconnections,
                              path_to_pdsi=path_to_pdsi)

    # Save into file
    model.save_predictions_as_submit(predicted,
                                     path=f'./results/{method}_{aggregation_days_snodas}_{aggregation_days_snotel}_{aggregation_days_pdsi}.csv')

    TimeSeriesPlot().predicted_time_series(predicted, plots_folder_name='predictions_common')


if __name__ == '__main__':
    generate_forecast_based_on_snotel()
