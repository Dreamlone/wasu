from pathlib import Path

import warnings

import pandas as pd

from wasu.development.models.snodas import SnodasRegression

warnings.filterwarnings('ignore')


def train_snodas_model():
    """ Train model and save it """
    train_df = pd.read_csv(Path('../../data/train.csv'), parse_dates=['year'])
    submission_format = pd.read_csv(Path('../../data/submission_format.csv'), parse_dates=['issue_date'])

    # Load dataframe with metadata
    metadata = pd.read_csv(Path('../../data/metadata_TdPVeJC.csv'))
    path_to_snodas = Path('../../data/snodas_csv').resolve()

    model = SnodasRegression(train_df=train_df, aggregation_days=180)
    model.fit(submission_format, metadata=metadata, path_to_snodas=path_to_snodas, vis=True)


if __name__ == '__main__':
    train_snodas_model()
