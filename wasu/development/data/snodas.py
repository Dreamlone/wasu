import datetime
import gzip
import os
import shutil
import subprocess
from pathlib import Path

import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from loguru import logger
from matplotlib import cm
from rasterio.mask import mask
from shapely.geometry import mapping

from wasu.development.paths import path_to_data_folder, path_to_examples_folder


def _unpack_data(archive_files: list, final_path: Path):
    ######################
    # PROCESS TEXT FILES #
    ######################
    for file in archive_files:
        if '.txt' in str(file.name):
            unpacked_text_file = Path(final_path, f'{file.name.split(".")[0]}.txt')
            with gzip.open(file, 'rb') as f_in:
                with open(unpacked_text_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

    #####################
    # PROCESS DAT FILES #
    #####################
    for file in archive_files:
        if '.dat' in str(file.name):
            unpacked_dat_file = Path(final_path, f'{file.name.split(".")[0]}.dat')
            with gzip.open(file, 'rb') as f_in:
                with open(unpacked_dat_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)


def _transform_dat_file_into_geotiff(txt_file: Path):
    product = None
    product_code = None
    data_units = None
    with open(txt_file, 'r') as fh:
        for line in fh:
            if 'Description' in line:
                product = line.split('Description: ')[-1]
            if 'Data units' in line:
                data_units = line.split('Data units: ')[-1]
            if 'Product code' in line:
                product_code = line.split('Product code: ')[-1]

    geotiff_file = Path(txt_file.parent, f'{txt_file.name.split(".")[0]}.tif')
    with rasterio.open(txt_file) as src:
        src_profile = src.profile
        src_profile.update(driver='GTiff', compress='lzw')

        with rasterio.open(geotiff_file, 'w', **src_profile) as dst:
            dst.write(src.read())

    # Remove old files
    os.remove(str(txt_file))
    os.remove(str(Path(txt_file.parent, f'{txt_file.name.split(".")[0]}.dat')))
    if product is None or product_code is None or data_units is None:
        raise ValueError(f'Can not define product name for dat file SNODAS: {txt_file}')
    return geotiff_file, product, product_code, data_units


def extract_values_by_extend_through_files(raster_path: Path, site_geometry, site_id, product, date_info, vis):
    geometry = [mapping(site_geometry)]

    # It's important not to set crop as True because it distort output
    with rasterio.open(raster_path) as src:
        out_image, _ = mask(src, geometry, crop=False, nodata=-100.0)
        clipped_matrix = out_image[0, :, :]

    filtered_values = np.ravel(clipped_matrix)[np.ravel(clipped_matrix) > -100]

    if vis is True and site_id == 'hungry_horse_reservoir_inflow':
        plots_folder = Path(path_to_examples_folder(), f'SNODAS_{site_id}')
        if plots_folder.is_dir() is False:
            plots_folder.mkdir(exist_ok=True, parents=True)

        fig_size = (12.0, 12.0)
        fig, ax = plt.subplots(figsize=fig_size)
        masked_array = np.ma.masked_where(clipped_matrix == -100.0, clipped_matrix)
        if product == 'Modeled average temperature, SWE-weighted average of snow layers, 24-hour average':
            cmap = cm.get_cmap('coolwarm')
        else:
            cmap = cm.get_cmap('Blues')
        cmap.set_bad(color='#C0C0C0')
        if product == 'Snow accumulation, 24-hour total':
            plt.imshow(masked_array, interpolation='nearest', cmap=cmap, vmin=0, vmax=1500)
            plt.ylim(500, 700)
            plt.xlim(1250, 1430)
            plt.colorbar()
            plt.title(product, fontsize=14)
            plt.savefig(Path(plots_folder, f'{product}_{date_info}.png'))
            plt.close()
        else:
            pass

    return filtered_values


def collect_snodas_data_for_site(path_to_folder: Path, site_id: str, vis: bool = False):
    """ Collect SNODAS data for desired site """
    # First - get information about site extend
    spatial_objects = geopandas.read_file(Path(path_to_data_folder(), 'geospatial.gpkg'))
    spatial_objects = spatial_objects[spatial_objects['site_id'] == site_id]
    site_geometry = spatial_objects.geometry[0]
    metadata_file = Path(path_to_folder, 'metadata.csv')

    metadata_df = pd.read_csv(metadata_file, parse_dates=['datetime'])
    metadata_df = metadata_df.sort_values(by='datetime')

    path_to_folder = path_to_folder.resolve()

    all_years = list(path_to_folder.iterdir())
    all_years.sort()
    for year_folder in all_years:
        if len(list(year_folder.iterdir())) < 1:
            # Skip empty folders
            continue

        for date_label in list(year_folder.iterdir()):
            date_info = date_label.name.split('_')[1]
            date_info = datetime.datetime.strptime(date_info, '%Y%m%d')
            logger.debug(f'Read SNODAS from geotiff files for datetime {date_info}')

            # Read geotiff files
            for geotiff_file in list(date_label.iterdir()):
                extract_values_by_extend_through_files(geotiff_file, site_geometry, site_id, product, date_info, vis)


def preprocess_snodas_data(path_to_folder: Path, new_folder_name: str = 'snodas_unpacked'):
    info_df = []
    path_to_folder = path_to_folder.resolve()

    # Create folder to unpack
    folder_to_unpack_files = Path(path_to_folder.parent, new_folder_name)
    folder_to_unpack_files.mkdir(exist_ok=True, parents=True)
    metadata_file = Path(folder_to_unpack_files, 'metadata.csv')

    all_years = list(path_to_folder.iterdir())
    all_years.sort()
    metadata_df = None
    if metadata_file.is_file() is True:
        # File is already exists - read it and add into stack
        metadata_df = pd.read_csv(metadata_file)
        info_df = [metadata_df]

    for year_folder in all_years:
        logger.info(f'SNODAS preprocess folder {year_folder}')
        for archive in list(year_folder.iterdir()):
            logger.debug(f'SNODAS preprocess archive {archive}')

            ##################
            # Unpack archive #
            ##################
            archive_base_name = archive.name.split('.')[0]
            date_info = archive_base_name.split('_')[-1]
            date_info = datetime.datetime.strptime(date_info, '%Y%m%d')
            if metadata_df is not None:
                # Check if that archive was already processed
                if archive.name in list(metadata_df['archive_name'].unique()):
                    logger.debug(f'SNODAS preprocess archive {archive}. Was already processed - skip it')
                    continue

            path_to_extract = Path(folder_to_unpack_files, year_folder.name, archive_base_name)
            path_to_extract.mkdir(exist_ok=True, parents=True)

            code = subprocess.call(['tar', '-xvf', str(archive), '-C', str(path_to_extract)])
            if code != 0:
                logger.warning(f'Can not open archive file {archive}. Code {code}')
                # Remove prepared folder
                shutil.rmtree(path_to_extract)
                continue

            final_path = Path(folder_to_unpack_files, year_folder.name, f"{archive.name.split('.')[0]}_processed")
            final_path.mkdir(exist_ok=True, parents=True)

            archive_files = list(path_to_extract.iterdir())
            _unpack_data(archive_files, final_path)
            shutil.rmtree(path_to_extract)

            ####################################
            # Transform dat files into geotiff #
            ####################################
            for file in list(final_path.iterdir()):
                if '.txt' not in file.name:
                    continue

                # Transform into geotiff and save files
                geotiff_file, product, product_code, data_units = _transform_dat_file_into_geotiff(file)
                product = product.replace('\n', '')
                product_code = product_code.replace('\n', '')
                data_units = data_units.replace('\n', '')

                info_df.append(pd.DataFrame({'archive_name': archive.name, 'datetime': [date_info], 'product': [product],
                                             'product_code': [product_code], 'data_units': [data_units]}))

        # Save metadata after processing one year
        if len(info_df) > 1:
            pd.concat(info_df).to_csv(metadata_file, index=False)

    info_df = pd.concat(info_df)
    info_df.to_csv(metadata_file, index=False)
