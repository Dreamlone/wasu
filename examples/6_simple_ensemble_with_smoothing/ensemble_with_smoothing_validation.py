from pathlib import Path

import numpy as np
import pandas as pd

import warnings
from loguru import logger

from wasu.development.models.train_model import TrainModel
from wasu.development.validation import ModelValidation
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
                site_df[target] = site_df[target].rolling(3).mean()
                site_df = site_df.fillna(method='backfill')

            smoothed_df.append(site_df)

    smoothed_df = pd.concat(smoothed_df)
    logger.info(f'Finished smoothing process. Lenght of the dataframe: {len(smoothed_df)}')
    return smoothed_df


def ensemble_from_files():
    files_to_ensemble = ['../7_snodas/validation/snodas.csv',
                         '../4_snotel/validation/snotel_50_basin.csv',
                         '../4_snotel/validation/snotel_50_all.csv']
    validator = ModelValidation(folder_for_plots='ensemble')

    dataframes = []
    for file in files_to_ensemble:
        file = Path(file).resolve()
        dataframes.append(pd.read_csv(file))

    first_submit = dataframes[0]
    corrected_response = []
    for row_id in range(len(first_submit)):
        logger.debug(f'Assimilate data for row {row_id}')

        predicted_low = []
        predicted_values = []
        predicted_up = []
        for df in dataframes:
            current_record = df.iloc[row_id]
            predicted_low.append(current_record.volume_10)
            predicted_values.append(current_record.volume_50)
            predicted_up.append(current_record.volume_90)
        predicted_low = np.array(predicted_low)
        predicted_values = np.array(predicted_values)
        predicted_up = np.array(predicted_up)

        mean_value = np.median(np.array(predicted_values))
        adjust_ratio = 0.0
        dataset = pd.DataFrame({'site_id': [first_submit.iloc[row_id].site_id],
                                'issue_date': [first_submit.iloc[row_id].issue_date],
                                'volume_10': [np.median(predicted_low) - (np.median(predicted_low) * adjust_ratio)],
                                'volume_50': [mean_value],
                                'volume_90': [np.median(predicted_up) + (np.median(predicted_up) * adjust_ratio)]})
        corrected_response.append(dataset)

    corrected_response = pd.concat(corrected_response)
    corrected_response = smoothing(corrected_response)

    train_df = pd.read_csv(Path('../../data/train.csv'), parse_dates=['year'])
    validator.compare_dataframes(corrected_response, train_df)


if __name__ == '__main__':
    ensemble_from_files()
