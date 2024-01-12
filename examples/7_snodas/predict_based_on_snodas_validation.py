from pathlib import Path

import warnings

import pandas as pd

from wasu.development.models.snodas import SnodasRegression
from wasu.development.validation import ModelValidation
from wasu.development.vis.visualization import TimeSeriesPlot

warnings.filterwarnings('ignore')


def generate_forecast_based_on_snodas():
    aggregation_days = 120
    validation_year = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    validator = ModelValidation(folder_for_plots='snodas', years_to_validate=validation_year)
    train_df = pd.read_csv(Path('../../data/train.csv'), parse_dates=['year'])
    submission_format = validator.generate_submission_format()

    # Load dataframe with metadata
    metadata = pd.read_csv(Path('../../data/metadata_TdPVeJC.csv'))
    path_to_snodas = Path('../../data/snodas_csv').resolve()

    model = SnodasRegression(train_df=train_df, aggregation_days=aggregation_days, train_test_split_year=2015)
    predicted = model.predict(submission_format, metadata=metadata, path_to_snodas=path_to_snodas, vis=False)

    validator.compare_dataframes(predicted, train_df,
                                 save_predicted_vs_actual_into_file=f'./validation/snodas_{aggregation_days}.csv')
    TimeSeriesPlot(validation_year).predicted_time_series(predicted, plots_folder_name='predictions_snodas')


if __name__ == '__main__':
    generate_forecast_based_on_snodas()
