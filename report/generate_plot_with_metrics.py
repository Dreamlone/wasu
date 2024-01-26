from pathlib import Path

import contextily as cx
import geopandas
import matplotlib.pyplot as plt

import warnings

import pandas as pd

from wasu.development.paths import path_to_plots_folder

warnings.filterwarnings('ignore')


def compare_approaches():
    plots_folder = path_to_plots_folder()
    plots_folder.mkdir(exist_ok=True, parents=True)

    df = pd.DataFrame({'Mean average quantile loss': [367.66, 275.82, 193.70, 151.71, 167.40, 165.20, 137.92, 132.90, 120.78],
                       'Method': ['Simple repeating', 'Advanced repeating',
                                  'Streamflow-based', 'SNOTEL', 'Ensemble',
                                  'Ensemble (smoothed)', 'SNODAS', 'Complex',
                                  'Complex (optimized)']})

    fig_size = (8.0, 3.0)
    fig, ax = plt.subplots(figsize=fig_size)
    plt.grid()
    plt.scatter(df['Method'], df['Mean average quantile loss'],
                alpha=1, s=100, c='blue', edgecolor='black')
    plt.xlabel('Model')
    plt.ylabel('Mean average quantile loss')
    plt.xticks(rotation=75)

    plt.savefig(Path(plots_folder, f'compare_approaches.png'), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    compare_approaches()
