from pathlib import Path

import warnings

import pandas as pd

from wasu.development.models.snotel import SnotelFlowRegression
from wasu.development.models.streamflow import StreamFlowRegression
from wasu.development.validation import ModelValidation
from wasu.development.vis.visualization import TimeSeriesPlot

warnings.filterwarnings('ignore')


def generate_forecast_based_on_snotel():
    validator = ModelValidation()

    train_df = pd.read_csv(Path('../../data/train.csv'), parse_dates=['year'])
    submission_format = validator.generate_submission_format()

    # Load dataframe with metadata
    metadata = pd.read_csv(Path('../../data/metadata_TdPVeJC.csv'))
    path_to_snotel = Path('../../data/snotel').resolve()

    model = SnotelFlowRegression(train_df=train_df, aggregation_days=110)
    predicted = model.predict(submission_format, metadata=metadata, path_to_snotel=path_to_snotel,
                              enable_spatial_aggregation=False, collect_only_in_basin=False)

    validator.compare_dataframes(predicted, train_df)
    model.save_predictions_as_submit(predicted, path='./validation/snotel_110_all_stations_without_aggregation.csv',
                                     submission_format=submission_format)


if __name__ == '__main__':
    generate_forecast_based_on_snotel()
