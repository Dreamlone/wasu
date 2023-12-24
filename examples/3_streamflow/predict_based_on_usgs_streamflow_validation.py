from pathlib import Path

import warnings

import pandas as pd

from wasu.development.models.streamflow import StreamFlowRegression
from wasu.development.validation import ModelValidation

warnings.filterwarnings('ignore')


def generate_forecast_based_on_streamflow():
    validator = ModelValidation(folder_for_plots='usgs_streamflow')
    train_df = pd.read_csv(Path('../../data/train.csv'), parse_dates=['year'])
    submission_format = validator.generate_submission_format()

    # Load dataframe with metadata
    metadata = pd.read_csv(Path('../../data/metadata_TdPVeJC.csv'))
    path_to_streamflow = Path('../../data/usgs_streamflow').resolve()

    model = StreamFlowRegression(train_df=train_df, aggregation_days=45)
    predicted = model.predict(submission_format, metadata=metadata, path_to_streamflow=path_to_streamflow, vis=False)

    validator.compare_dataframes(predicted, train_df)
    model.save_predictions_as_submit(predicted, path='./validation/usgs_streamflow_200.csv',
                                     submission_format=submission_format)


if __name__ == '__main__':
    generate_forecast_based_on_streamflow()
