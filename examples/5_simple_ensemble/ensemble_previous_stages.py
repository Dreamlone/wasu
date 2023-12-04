from pathlib import Path

import contextily as cx
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import warnings

from geopandas import GeoDataFrame
from loguru import logger

from wasu.development.data import collect_snotel_data_for_site, prepare_points_layer
from wasu.development.paths import path_to_plots_folder
from wasu.development.vis.visualization import TimeSeriesPlot

warnings.filterwarnings('ignore')


def ensemble_from_files(path: str):
    """ Collect predictions from the files and collect information about them into one prediction """
    files_to_ensemble = ['../2_repeating/results/advanced_repeating_07_11_2023.csv',
                         '../3_streamflow/results/usgs_streamflow_27_11_2023.csv',
                         '../4_snotel/results/snotel_03_12_2023.csv']
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

        dataset = pd.DataFrame({'site_id': [first_submit.iloc[row_id].site_id],
                                'issue_date': [first_submit.iloc[row_id].issue_date],
                                'volume_10': [min(predicted_values)],
                                'volume_50': [np.mean(np.array(predicted_values))],
                                'volume_90': [max(predicted_values)]})
        corrected_response.append(dataset)

    corrected_response = pd.concat(corrected_response)
    path = Path(path).resolve()
    base_dir = path.parent
    base_dir.mkdir(exist_ok=True, parents=True)
    corrected_response.to_csv(path, index=False)

    TimeSeriesPlot().predicted_time_series(corrected_response, plots_folder_name='predictions_first_ensemble')


if __name__ == '__main__':
    ensemble_from_files('./results/first_ensemble_04_12_2023.csv')