from pathlib import Path

import contextily as cx
import geopandas
import matplotlib.pyplot as plt
import pandas as pd

import warnings

from geopandas import GeoDataFrame
from loguru import logger

from wasu.development.data import collect_snotel_data_for_site, prepare_points_layer
from wasu.development.paths import path_to_plots_folder

warnings.filterwarnings('ignore')


def show_spatial_plot():
    plots_folder = Path(path_to_plots_folder(), 'spatial_with_snotel_stations')
    plots_folder.mkdir(exist_ok=True, parents=True)

    spatial_objects = geopandas.read_file(Path('../../data/geospatial.gpkg'))

    for row_id, spatial_object in spatial_objects.iterrows():
        site_id = spatial_object.site_id
        spatial_df = GeoDataFrame(crs="EPSG:4326", geometry=[spatial_object.geometry])
        df = collect_snotel_data_for_site(path_to_folder=Path('../../data/snotel'), site_id=site_id,
                                          collect_only_in_basin=True)
        df = df[['station', 'latitude', 'longitude']].drop_duplicates()

        df = prepare_points_layer(df)

        fig_size = (12.0, 10.0)
        fig, ax = plt.subplots(figsize=fig_size)
        ax = df.plot(ax=ax, color='#ffffff', edgecolor='black')
        for x, y, label in zip(df.geometry.x, df.geometry.y, df.station):
            ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points", fontsize=5)

        ax = spatial_df.plot(ax=ax, legend=False, color='blue', zorder=1, alpha=0.3)
        cx.add_basemap(ax, crs=spatial_objects.crs.to_string(), source=cx.providers.CartoDB.Voyager)

        plt.savefig(Path(plots_folder, f'spatial_extend_snotel_{site_id}.png'))
        plt.close()

        print(f'Columns in the spatial dataframe: {list(spatial_objects.columns)}')
        print(spatial_objects.head(5))


if __name__ == '__main__':
    show_spatial_plot()
