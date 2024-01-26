from pathlib import Path

import numpy as np
import pandas as pd

import warnings
from loguru import logger
from matplotlib import pyplot as plt

from wasu.development.models.train_model import TrainModel
from wasu.development.paths import path_to_examples_folder
from wasu.development.vis.visualization import TimeSeriesPlot

warnings.filterwarnings('ignore')

COLORS = ['blue', 'green', 'orange', 'red']

SITES_TO_SHOW = ['hungry_horse_reservoir_inflow', 'snake_r_nr_heise', 'pueblo_reservoir_inflow',
                 'sweetwater_r_nr_alcova', 'missouri_r_at_toston', 'animas_r_at_durango',
                 'yampa_r_nr_maybell', 'libby_reservoir_inflow', 'boise_r_nr_boise',
                 'green_r_bl_howard_a_hanson_dam', 'taylor_park_reservoir_inflow',
                 'dillon_reservoir_inflow', 'ruedi_reservoir_inflow',
                 'fontenelle_reservoir_inflow', 'weber_r_nr_oakley',
                 'san_joaquin_river_millerton_reservoir', 'merced_river_yosemite_at_pohono_bridge',
                 'american_river_folsom_lake', 'colville_r_at_kettle_falls',
                 'stehekin_r_at_stehekin', 'detroit_lake_inflow', 'virgin_r_at_virtin',
                 'skagit_ross_reservoir', 'boysen_reservoir_inflow', 'pecos_r_nr_pecos',
                 'owyhee_r_bl_owyhee_dam']

def compare_predictions():
    """ Create plots about different predictions

    Example:
        '../3_streamflow/results/usgs_streamflow_170.csv', '../4_snotel/results/snotel_170_all.csv'
    """
    files_to_ensemble = ['../6_simple_ensemble_with_smoothing/results/best_result.csv',
                         '../9_common/results/forest_14_90_60.csv']

    plots_folder = Path(path_to_examples_folder(), 'plots_comparison')
    plots_folder.mkdir(exist_ok=True, parents=True)

    for site in SITES_TO_SHOW:
        # Generate plot for current site
        logger.info(f'Generate plot for site {site}')

        fig_size = (17.0, 8.0)
        fig, ax = plt.subplots(figsize=fig_size)

        for file_id, file in enumerate(files_to_ensemble):
            file = Path(file).resolve()
            df = pd.read_csv(file, parse_dates=['issue_date'])

            site_df = df[df['site_id'] == site]

            color = COLORS[file_id]
            ax.plot(site_df['issue_date'], site_df['volume_10'], '--', c=color, alpha=0.7)
            ax.plot(site_df['issue_date'], site_df['volume_50'], '-ok', c=color, alpha=1,
                    label=f'{file.name.split(".csv")[0]}', linewidth=3)
            ax.plot(site_df['issue_date'], site_df['volume_90'], '--', c=color, alpha=0.7)

        ax.set_xlabel('Datetime', fontsize=13)
        ax.set_ylabel(f'Flow values', fontsize=13)

        plt.grid()
        plt.legend()
        plt.title(f'Site {site}')
        plt.savefig(Path(plots_folder, f'{site}.png'))
        plt.close()


if __name__ == '__main__':
    compare_predictions()
