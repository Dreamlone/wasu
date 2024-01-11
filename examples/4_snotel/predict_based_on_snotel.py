from pathlib import Path

import warnings

import pandas as pd

from wasu.development.models.snotel import SnotelFlowRegression
from wasu.development.validation import ModelValidation
from wasu.development.vis.visualization import TimeSeriesPlot

warnings.filterwarnings('ignore')


def generate_forecast_based_on_snotel():
    aggregation_days = 40
    validation_year = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    validator = ModelValidation(folder_for_plots='snotel', years_to_validate=validation_year)

    train_df = pd.read_csv(Path('../../data/train.csv'), parse_dates=['year'])
    submission_format = validator.generate_submission_format()

    # Load dataframe with metadata
    metadata = pd.read_csv(Path('../../data/metadata_TdPVeJC.csv'))
    path_to_snotel = Path('../../data/snotel').resolve()

    model = SnotelFlowRegression(train_df=train_df, aggregation_days=aggregation_days,
                                 enable_spatial_aggregation=True, collect_only_in_basin=False)
    predicted = model.predict(submission_format, metadata=metadata, path_to_snotel=path_to_snotel)

    validator.compare_dataframes(predicted, train_df)
    model.save_predictions_as_submit(predicted, path=f'./results/snotel_{aggregation_days}.csv',
                                     submission_format=submission_format)

    validator.compare_dataframes(predicted, train_df,
                                 save_predicted_vs_actual_into_file=f'./validation/snotel_{aggregation_days}.csv')
    TimeSeriesPlot(validation_year).predicted_time_series(predicted, plots_folder_name='predictions_snotel')


if __name__ == '__main__':
    generate_forecast_based_on_snotel()
