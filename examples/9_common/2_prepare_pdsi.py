from pathlib import Path

import warnings

import geopandas
import numpy as np
import pandas as pd
import rasterio
from loguru import logger
from rasterio.mask import mask
from shapely.geometry import mapping

warnings.filterwarnings('ignore')


def extract_data_from_netcdf_file(geometry, netcdf_file) -> pd.DataFrame:
    netcdf_file_name = netcdf_file.name.split('_')
    first_date = netcdf_file_name[1]
    second_date = netcdf_file_name[2].split('.nc')[0]
    datetime_labels = pd.date_range(first_date, second_date, freq='5D')

    dataframe = []
    with rasterio.open(netcdf_file) as src:
        out_image, _ = mask(src, geometry, crop=False, nodata=32767.0)
        for time_id in range(len(out_image)):
            clipped_matrix = out_image[time_id, :, :]
            filtered_values = np.ravel(clipped_matrix)[np.ravel(clipped_matrix) < 32766]

            dataframe.append([np.nanmean(filtered_values), np.nansum(filtered_values), np.nanstd(filtered_values)])

    dataframe = pd.DataFrame(dataframe, columns=['Mean_PDSI', 'Sum_PDSI', 'std_PDSI'])
    dataframe['datetime'] = datetime_labels

    return dataframe


def prepare_pdsi_data_as_table():
    spatial_objects = geopandas.read_file(Path('../../data/geospatial.gpkg'))

    pdsi_path = Path('../../data/pdsi').resolve()
    pdsi_results = Path('../../data/pdsi_csv').resolve()
    pdsi_results.mkdir(parents=True, exist_ok=True)

    years = list(pdsi_path.iterdir())
    years.sort()
    for row_id, spatial_object in spatial_objects.iterrows():
        site_id = spatial_object.site_id
        site_geom = spatial_objects[spatial_objects['site_id'] == site_id]
        site_geometry = site_geom.geometry.values[0]
        geometry = [mapping(site_geometry)]

        dataframe_for_site = []
        for folder in years:
            files_in_folder = list(folder.iterdir())
            if len(files_in_folder) < 1:
                continue

            logger.debug(f'{site_id}. Process folder {folder.name}')
            for netcdf_file in files_in_folder:
                df = extract_data_from_netcdf_file(geometry, netcdf_file)
                dataframe_for_site.append(df)

        dataframe_for_site = pd.concat(dataframe_for_site)
        dataframe_for_site['datetime'] = pd.to_datetime(dataframe_for_site['datetime'])
        dataframe_for_site = dataframe_for_site.sort_values(by='datetime')

        dataframe_for_site.to_csv(Path(pdsi_results, f'{site_id}.csv'), index=False)
        logger.info(f'Successfully saved result for site {site_id}')


if __name__ == '__main__':
    prepare_pdsi_data_as_table()
