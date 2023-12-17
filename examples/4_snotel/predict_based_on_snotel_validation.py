from pathlib import Path

import warnings

import pandas as pd

from wasu.development.models.snotel import SnotelFlowRegression
from wasu.development.validation import ModelValidation

warnings.filterwarnings('ignore')


def generate_forecast_based_on_snotel():
    validator = ModelValidation(folder_for_plots='snotel')

    train_df = pd.read_csv(Path('../../data/train.csv'), parse_dates=['year'])
    submission_format = validator.generate_submission_format()

    # Load dataframe with metadata
    metadata = pd.read_csv(Path('../../data/metadata_TdPVeJC.csv'))
    path_to_snotel = Path('../../data/snotel').resolve()

    model = SnotelFlowRegression(train_df=train_df, aggregation_days=150,
                                 enable_spatial_aggregation=True, collect_only_in_basin=False)
    predicted = model.predict(submission_format, metadata=metadata, path_to_snotel=path_to_snotel)

    validator.compare_dataframes(predicted, train_df)
    model.save_predictions_as_submit(predicted, path='./validation/snotel_150_all.csv',
                                     submission_format=submission_format)


if __name__ == '__main__':
    generate_forecast_based_on_snotel()
