from pathlib import Path

import warnings

import pandas as pd
from loguru import logger

from wasu.development.models.common import CommonRegression
from wasu.development.validation import ModelValidation

warnings.filterwarnings('ignore')


def generate_forecast_based_on_snotel():
    method = 'linear'
    validator = ModelValidation(folder_for_plots='common',  years_to_validate=[2019, 2020, 2021, 2022, 2023])

    train_df = pd.read_csv(Path('../../data/train.csv'), parse_dates=['year'])
    submission_format = validator.generate_submission_format()

    # Load dataframe with metadata
    metadata = pd.read_csv(Path('../../data/metadata_TdPVeJC.csv'))
    path_to_snotel = Path('../../data/snotel').resolve()
    path_to_snodas = Path('../../data/snodas_csv').resolve()
    path_to_pdsi = Path('../../data/pdsi_csv').resolve()

    for aggregation_days_snodas in [10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50]:
        for aggregation_days_snotel in [88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152]:
            for aggregation_days_pdsi in [88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152]:
                name = f'{method}_{aggregation_days_snodas}_{aggregation_days_snotel}_{aggregation_days_pdsi}'
                file = Path(f'./validation/{name}.csv').resolve()
                if file.is_file() is True:
                    logger.info(f'Skip file {name} because it is already generated')

                model = CommonRegression(train_df=train_df, method=method,
                                         aggregation_days_snodas=aggregation_days_snodas,
                                         aggregation_days_snotel=aggregation_days_snotel,
                                         aggregation_days_pdsi=aggregation_days_pdsi)
                predicted = model.predict(submission_format, metadata=metadata, path_to_snotel=path_to_snotel,
                                          path_to_snodas=path_to_snodas,
                                          path_to_pdsi=path_to_pdsi)

                validator.compare_dataframes(predicted, train_df,
                                             save_predicted_vs_actual_into_file=file)


if __name__ == '__main__':
    generate_forecast_based_on_snotel()
