from pathlib import Path

import warnings

import pandas as pd
from loguru import logger

from wasu.development.models.common import CommonRegression

warnings.filterwarnings('ignore')


def train_common_models_with_different_hyperparameters():
    """ Train model and save it """
    train_df = pd.read_csv(Path('../../data/train.csv'), parse_dates=['year'])
    submission_format = pd.read_csv(Path('../../data/submission_format.csv'), parse_dates=['issue_date'])

    # Load dataframe with metadata
    metadata = pd.read_csv(Path('../../data/metadata_TdPVeJC.csv'))
    path_to_snodas = Path('../../data/snodas_csv').resolve()
    path_to_snotel = Path('../../data/snotel').resolve()
    path_to_pdsi = Path('../../data/pdsi_csv').resolve()

    for aggregation_days_snodas in [10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50]:
        for aggregation_days_snotel in [88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152]:
            for aggregation_days_pdsi in [88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152]:
                try:
                    model = CommonRegression(train_df=train_df, method='linear',
                                             aggregation_days_snodas=aggregation_days_snodas,
                                             aggregation_days_snotel=aggregation_days_snotel,
                                             aggregation_days_pdsi=aggregation_days_pdsi)

                    model.fit(submission_format, metadata=metadata,
                              path_to_snotel=path_to_snotel, path_to_snodas=path_to_snodas, path_to_pdsi=path_to_pdsi,
                              vis=False)
                    logger.info(f'EXPERIMENT. Successfully fit model {aggregation_days_snodas, aggregation_days_snotel, aggregation_days_pdsi}')
                except Exception as ex:
                    logger.info(f'EXPERIMENT. Failed to fit model {aggregation_days_snodas, aggregation_days_snotel, aggregation_days_pdsi}')


if __name__ == '__main__':
    train_common_models_with_different_hyperparameters()
