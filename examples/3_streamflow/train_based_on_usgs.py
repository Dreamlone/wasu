from pathlib import Path

import warnings

import pandas as pd

from wasu.development.models.streamflow import StreamFlowRegression

warnings.filterwarnings('ignore')


def train_streamflow_model():
    """ Train model and save it """
    train_df = pd.read_csv(Path('../../data/train.csv'), parse_dates=['year'])
    submission_format = pd.read_csv(Path('../../data/submission_format.csv'), parse_dates=['issue_date'])

    # Load dataframe with metadata
    metadata = pd.read_csv(Path('../../data/metadata_TdPVeJC.csv'))
    path_to_streamflow = Path('../../data/usgs_streamflow').resolve()

    model = StreamFlowRegression(train_df=train_df, aggregation_days=45)
    model.fit(submission_format=submission_format,
              metadata=metadata, path_to_streamflow=path_to_streamflow, vis=False)


if __name__ == '__main__':
    train_streamflow_model()
