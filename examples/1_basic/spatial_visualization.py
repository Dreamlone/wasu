from pathlib import Path

import contextily as cx
import geopandas
import matplotlib.pyplot as plt

import warnings

from wasu.development.paths import path_to_plots_folder

warnings.filterwarnings('ignore')


def visualize_data(path_to_file: Path = Path('../../data/geospatial.gpkg')):
    plots_folder = Path(path_to_plots_folder(), 'spatial')
    plots_folder.mkdir(exist_ok=True, parents=True)

    spatial_objects = geopandas.read_file(path_to_file)

    fig_size = (12.0, 10.0)
    fig, ax = plt.subplots(figsize=fig_size)
    ax = spatial_objects.plot(ax=ax, column='area', legend=False, color='blue', zorder=1, alpha=0.3)
    cx.add_basemap(ax, crs=spatial_objects.crs.to_string(), source=cx.providers.CartoDB.Voyager)

    plt.savefig(Path(plots_folder, f'spatial_extend.png'))
    plt.close()

    print(f'Columns in the spatial dataframe: {list(spatial_objects.columns)}')
    print(spatial_objects.head(5))


if __name__ == '__main__':
    visualize_data()
