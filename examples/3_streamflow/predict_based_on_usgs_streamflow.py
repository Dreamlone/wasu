from pathlib import Path

import warnings

import pandas as pd

from wasu.development.models.streamflow import StreamFlowRegression
from wasu.development.validation import ModelValidation
from wasu.development.vis.visualization import TimeSeriesPlot

warnings.filterwarnings('ignore')


def generate_forecast_based_on_streamflow():
    aggregation_days = 40
    validation_year = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    train_df = pd.read_csv(Path('../../data/train.csv'), parse_dates=['year'])
    # submission_format = pd.read_csv(Path('../../data/submission_format.csv'), parse_dates=['issue_date'])
    validator = ModelValidation(folder_for_plots='simple_repeating',
                                years_to_validate=validation_year)
    submission_format = validator.generate_submission_format()

    # Load dataframe with metadata
    metadata = pd.read_csv(Path('../../data/metadata_TdPVeJC.csv'))
    path_to_streamflow = Path('../../data/usgs_streamflow').resolve()

    model = StreamFlowRegression(train_df=train_df, aggregation_days=aggregation_days)
    predicted = model.predict(submission_format, metadata=metadata, path_to_streamflow=path_to_streamflow, vis=False)

    # Save into file
    model.save_predictions_as_submit(predicted, path=f'./results/usgs_streamflow_{aggregation_days}.csv')
    validator.compare_dataframes(predicted, train_df,
                                 save_predicted_vs_actual_into_file=f'./validation/usgs_streamflow_{aggregation_days}.csv')
    TimeSeriesPlot(validation_year).predicted_time_series(predicted, plots_folder_name='predictions_usgs_streamflow')


if __name__ == '__main__':
    generate_forecast_based_on_streamflow()
