from pathlib import Path

import warnings

import pandas as pd

from wasu.development.models.common import CommonRegression

warnings.filterwarnings('ignore')


def train_common_model():
    """ Train model and save it """
    train_df = pd.read_csv(Path('../../data/train.csv'), parse_dates=['year'])
    submission_format = pd.read_csv(Path('../../data/submission_format.csv'), parse_dates=['issue_date'])

    # Load dataframe with metadata
    metadata = pd.read_csv(Path('../../data/metadata_TdPVeJC.csv'))
    path_to_snodas = Path('../../data/snodas_csv').resolve()
    path_to_snotel = Path('../../data/snotel').resolve()
    path_to_teleconnections = Path('../../data/teleconnections').resolve()

    model = CommonRegression(train_df=train_df, method='linear',
                             aggregation_days_snodas=25,
                             aggregation_days_snotel=80)

    model.fit(submission_format, metadata=metadata,
              path_to_snotel=path_to_snotel, path_to_snodas=path_to_snodas,
              path_to_teleconnections=path_to_teleconnections,
              vis=False)


if __name__ == '__main__':
    train_common_model()
