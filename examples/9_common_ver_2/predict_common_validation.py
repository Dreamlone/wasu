from pathlib import Path

import warnings

import pandas as pd

from wasu.development.models.common import CommonRegression
from wasu.development.validation import ModelValidation
from wasu.development.vis.visualization import TimeSeriesPlot

warnings.filterwarnings('ignore')


def generate_forecast_based_on_snotel():
    method = 'linear'
    aggregation_days_snotel_short = 40
    aggregation_days_snotel_long = 120
    aggregation_days_pdsi = 120
    validation_year = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    validator = ModelValidation(folder_for_plots='common', years_to_validate=validation_year)

    train_df = pd.read_csv(Path('../../data/train.csv'), parse_dates=['year'])
    submission_format = validator.generate_submission_format()

    # Load dataframe with metadata
    metadata = pd.read_csv(Path('../../data/metadata_TdPVeJC.csv'))
    path_to_snotel = Path('../../data/snotel').resolve()
    path_to_snodas = Path('../../data/snodas_csv').resolve()
    path_to_pdsi = Path('../../data/pdsi_csv').resolve()

    model = CommonRegression(train_df=train_df, method=method,
                             aggregation_days_snotel_short=aggregation_days_snotel_short,
                             aggregation_days_snotel_long=aggregation_days_snotel_long,
                             aggregation_days_pdsi=aggregation_days_pdsi)
    predicted = model.predict(submission_format, metadata=metadata, path_to_snotel=path_to_snotel,
                              path_to_snodas=path_to_snodas,
                              path_to_pdsi=path_to_pdsi)

    file = f'./validation/{method}_{aggregation_days_snotel_short}_{aggregation_days_snotel_long}_{aggregation_days_pdsi}.csv'
    validator.compare_dataframes(predicted, train_df, save_predicted_vs_actual_into_file=file)
    TimeSeriesPlot(validation_year).predicted_time_series(predicted, plots_folder_name='predictions_snotel')


if __name__ == '__main__':
    generate_forecast_based_on_snotel()
