from pathlib import Path

import warnings

import geopandas
import rasterio
from loguru import logger
from rasterio.mask import mask
from shapely.geometry import mapping

warnings.filterwarnings('ignore')


def prepare_grace_data_as_table():
    spatial_objects = geopandas.read_file(Path('../../data/geospatial.gpkg'))

    grace_path = Path('../../data/grace_indicators').resolve()
    grace_results = Path('../../data/grace_csv').resolve()
    grace_results.mkdir(parents=True, exist_ok=True)

    years = list(grace_path.iterdir())
    years.sort()
    for row_id, spatial_object in spatial_objects.iterrows():
        site_id = spatial_object.site_id
        site_geom = spatial_objects[spatial_objects['site_id'] == site_id]
        site_geometry = site_geom.geometry.values[0]
        geometry = [mapping(site_geometry)]

        for folder in years:
            files_in_folder = list(folder.iterdir())
            if len(files_in_folder) < 1:
                continue

            logger.info(f'{site_id}. Process folder {folder.name}')
            for netcdf_file in files_in_folder:
                with rasterio.open(netcdf_file) as src:
                    out_image, _ = mask(src, geometry, crop=False, nodata=-100.0)
                    clipped_matrix = out_image[0, :, :]



if __name__ == '__main__':
    prepare_grace_data_as_table()
