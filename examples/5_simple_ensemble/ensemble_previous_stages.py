from pathlib import Path

import numpy as np
import pandas as pd

import warnings

from loguru import logger

from wasu.development.validation import ModelValidation
from wasu.development.vis.visualization import TimeSeriesPlot

warnings.filterwarnings('ignore')


def ensemble_from_files(path: str):
    """ Collect predictions from the files and collect information about them into one prediction """
    files_to_ensemble = ['../3_streamflow/validation/usgs_streamflow_120.csv',
                         '../4_snotel/validation/snotel_40.csv']
    validation_year = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    validator = ModelValidation(folder_for_plots='simple_ensemble', years_to_validate=validation_year)

    # Load tables from csv files
    dataframes = []
    for file in files_to_ensemble:
        file = Path(file).resolve()

        dataframes.append(pd.read_csv(file))

    first_submit = dataframes[0]
    corrected_response = []
    for row_id in range(len(first_submit)):
        logger.debug(f'Assimilate data for row {row_id}')

        predicted_values = []
        for df in dataframes:
            current_record = df.iloc[row_id]
            predicted_values.append(current_record.volume_50)
        predicted_values = np.array(predicted_values)

        mean_value = np.median(np.array(predicted_values))
        adjust_ratio = 0.2
        dataset = pd.DataFrame({'site_id': [first_submit.iloc[row_id].site_id],
                                'issue_date': [first_submit.iloc[row_id].issue_date],
                                'volume_10': [np.percentile(predicted_values, 10) - (mean_value * adjust_ratio)],
                                'volume_50': [mean_value],
                                'volume_90': [np.percentile(predicted_values, 90) + (mean_value * adjust_ratio)]})
        corrected_response.append(dataset)

    corrected_response = pd.concat(corrected_response)
    path = Path(path).resolve()
    base_dir = path.parent
    base_dir.mkdir(exist_ok=True, parents=True)
    corrected_response.to_csv(path, index=False)

    train_df = pd.read_csv(Path('../../data/train.csv'), parse_dates=['year'])
    validator.compare_dataframes(corrected_response, train_df)
    TimeSeriesPlot(validation_year).predicted_time_series(corrected_response,
                                                          plots_folder_name='predictions_simple_ensemble')


if __name__ == '__main__':
    ensemble_from_files('results/simple_ensemble.csv')
