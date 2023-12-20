from pathlib import Path

import warnings

import pandas as pd

from wasu.development.models.common import CommonRegression
from wasu.development.validation import ModelValidation

warnings.filterwarnings('ignore')


def generate_forecast_based_on_snotel():
    method = 'linear'
    aggregation_days_snodas = 5
    aggregation_days_snotel = 30
    validator = ModelValidation(folder_for_plots='common')

    train_df = pd.read_csv(Path('../../data/train.csv'), parse_dates=['year'])
    submission_format = validator.generate_submission_format()

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

    validator.compare_dataframes(predicted, train_df)
    model.save_predictions_as_submit(predicted, path=f'./validation/{method}_{aggregation_days_snodas}_{aggregation_days_snotel}.csv',
                                     submission_format=submission_format)


if __name__ == '__main__':
    generate_forecast_based_on_snotel()
