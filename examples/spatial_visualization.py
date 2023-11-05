from pathlib import Path

import contextily as cx
import geopandas
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


def visualize_data(path_to_file=Path('../data/geospatial.gpkg')):
    spatial_objects = geopandas.read_file(path_to_file)

    ax = spatial_objects.plot(column='area', legend=False, color='blue', zorder=1)
    cx.add_basemap(ax, crs=spatial_objects.crs.to_string(), source=cx.providers.CartoDB.Voyager)
    plt.show()

    print(f'Columns in the spatial dataframe: {list(spatial_objects.columns)}')
    print(spatial_objects.head(5))


if __name__ == '__main__':
    visualize_data()
