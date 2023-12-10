from pathlib import Path

import numpy as np
import pandas as pd

import warnings
from loguru import logger

from wasu.development.models.train_model import TrainModel
from wasu.development.vis.visualization import TimeSeriesPlot

warnings.filterwarnings('ignore')


def smoothing(dataframe_with_predictions: pd.DataFrame) -> pd.DataFrame:
    logger.info(f'Start smoothing process. Lenght of the dataframe: {len(dataframe_with_predictions)}')
    smoothed_df = []
    for year in list(set(pd.to_datetime(dataframe_with_predictions['issue_date']).dt.year)):
        # Process every year
        year_df = dataframe_with_predictions[pd.to_datetime(dataframe_with_predictions['issue_date']).dt.year == year]

        for site in list(year_df['site_id'].unique()):
            # And every site
            site_df = year_df[year_df['site_id'] == site]

            for target in ['volume_10', 'volume_50', 'volume_90']:
                # Process every target column
                site_df[target] = site_df[target].rolling(2).mean()
                site_df = site_df.fillna(method='backfill')

            smoothed_df.append(site_df)

    smoothed_df = pd.concat(smoothed_df)
    logger.info(f'Finished smoothing process. Lenght of the dataframe: {len(smoothed_df)}')
    return smoothed_df


def ensemble_from_files(path: str):
    """ Collect predictions from the files and collect information about them into one prediction """
    files_to_ensemble = ['../4_snotel/results/snotel_50_basin.csv',
                         '../7_snodas/results/snodas_50.csv',
                         '../4_snotel/results/snotel_180_all_stations.csv']

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
        adjust_ratio = 0.3
        dataset = pd.DataFrame({'site_id': [first_submit.iloc[row_id].site_id],
                                'issue_date': [first_submit.iloc[row_id].issue_date],
                                'volume_10': [np.percentile(predicted_values, 10) -
                                              (np.percentile(predicted_values, 10) * adjust_ratio)],
                                'volume_50': [mean_value],
                                'volume_90': [np.percentile(predicted_values, 90) +
                                              (np.percentile(predicted_values, 90) * adjust_ratio)]})
        corrected_response.append(dataset)

    corrected_response = pd.concat(corrected_response)
    corrected_response = smoothing(corrected_response)

    TrainModel(pd.DataFrame()).save_predictions_as_submit(corrected_response, path=path)
    TimeSeriesPlot().predicted_time_series(corrected_response,
                                           plots_folder_name='predictions_first_ensemble_with_smoothing')


if __name__ == '__main__':
    ensemble_from_files('results/first_ensemble_smooth_10_12_2023.csv')
