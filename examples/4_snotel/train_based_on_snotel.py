from pathlib import Path

import warnings

import pandas as pd

from wasu.development.models.snotel import SnotelFlowRegression

warnings.filterwarnings('ignore')


def train_snotel_model():
    """ Train model and save it """
    train_df = pd.read_csv(Path('../../data/train.csv'), parse_dates=['year'])
    submission_format = pd.read_csv(Path('../../data/submission_format.csv'), parse_dates=['issue_date'])

    # Load dataframe with metadata
    metadata = pd.read_csv(Path('../../data/metadata_TdPVeJC.csv'))
    path_to_snotel = Path('../../data/snotel').resolve()

    model = SnotelFlowRegression(train_df=train_df,
                                 aggregation_days=172,
                                 enable_spatial_aggregation=True,
                                 collect_only_in_basin=False)
    model.fit(submission_format, metadata=metadata, path_to_snotel=path_to_snotel, vis=False)


if __name__ == '__main__':
    train_snotel_model()
