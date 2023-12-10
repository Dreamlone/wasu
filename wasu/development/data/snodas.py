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

SITES = ['hungry_horse_reservoir_inflow', 'snake_r_nr_heise', 'pueblo_reservoir_inflow',
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
        plots_folder = Path(path_to_examples_folder(), f'snodas_{site_id}')
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
            plt.imshow(masked_array, interpolation='nearest', cmap=cmap, vmin=0, vmax=100)
            plt.ylim(500, 700)
            plt.xlim(1250, 1430)
            plt.colorbar()
            plt.title(f'{product} {date_info}', fontsize=14)
            plt.savefig(Path(plots_folder, f'{product}_{date_info}.png'))
            plt.close()
        elif product == 'Modeled average temperature, SWE-weighted average of snow layers, 24-hour average':
            plt.imshow(masked_array, interpolation='nearest', cmap=cmap, vmin=250, vmax=300)
            plt.ylim(500, 700)
            plt.xlim(1250, 1430)
            plt.colorbar()
            plt.title(f'{product} {date_info}', fontsize=14)
            plt.savefig(Path(plots_folder, f'{product}_{date_info}.png'))
            plt.close()
        else:
            pass

    return filtered_values


def extract_data_from_geotiff_files_into_csv(path_to_folder: Path, folder_for_csv: Path, vis: bool = False):
    """
    STAGE 2
    Transform SNODAS data per sites: extract values and calculate statistics and save results as csv files
    """
    folder_for_csv = folder_for_csv.resolve()
    folder_for_csv.mkdir(exist_ok=True, parents=True)

    # First - get information about site extend
    spatial_objects = geopandas.read_file(Path(path_to_data_folder(), 'geospatial.gpkg'))
    metadata_file = Path(path_to_folder, 'metadata.csv')

    metadata_df = pd.read_csv(metadata_file, parse_dates=['datetime'])
    metadata_df = metadata_df.sort_values(by='datetime')

    for site_id in SITES:
        site_name_df = Path(folder_for_csv, f'{site_id}.csv')
        if site_name_df.is_file() is True:
            logger.info(f'Skip collecting data for site {site_id} because it is already stored')
            continue

        # Prepare data for every site
        site_geom = spatial_objects[spatial_objects['site_id'] == site_id]
        if len(site_geom) < 1:
            logger.info(f'Skip site {site_id} because there is no geometry data for it')
            continue

        site_geometry = site_geom.geometry.values[0]

        all_info_per_site = []
        for date_info in list(metadata_df['datetime'].unique()):
            try:
                date_df = metadata_df[metadata_df['datetime'] == date_info]
                logger.debug(f'Site: {site_id}, Assimilate geotiff files for {date_info}')

                datetime_site_df = pd.DataFrame({'datetime': [date_info]})
                for row_id, row in date_df.iterrows():
                    geotiff_file = row.geotiff
                    product_name = row['product']
                    # Product melt rate
                    if product_name == 'Modeled melt rate, bottom of snow layers, 24-hour total':
                        product_name = 'Modeled melt rate, bottom of snow layers'

                    # Snow accumulation
                    if product_name == 'Scaled Snow accumulation, 24-hour total':
                        product_name = 'Snow accumulation, 24-hour total'
                    if product_name == 'Scaled Snow accumulation 3 hour forecast, 24-hour total':
                        product_name = 'Snow accumulation, 24-hour total'

                    # Non snow accumulation
                    if product_name == 'Scaled Non-snow accumulation, 24-hour total':
                        product_name = 'Non-snow accumulation, 24-hour total'
                    if product_name == 'Scaled Non-snow accumulation 3 hour forecast, 24-hour total':
                        product_name = 'Non-snow accumulation, 24-hour total'

                    vals = extract_values_by_extend_through_files(geotiff_file, site_geometry, site_id, product_name,
                                                                  date_info, vis)
                    # Calculate statistics for current datetime
                    datetime_site_df[f'mean_{product_name}'] = np.nanmean(vals)
                    datetime_site_df[f'sum_{product_name}'] = np.nansum(vals)
                    datetime_site_df[f'std_{product_name}'] = np.nanstd(vals)

                all_info_per_site.append(datetime_site_df)
            except Exception as ex:
                logger.warning(F'Can not assimilate data due to {ex}. Skip')

        all_info_per_site = pd.concat(all_info_per_site)
        all_info_per_site.to_csv(site_name_df, index=False)


def unpack_snodas_data_into_geotiff(path_to_folder: Path, folder_to_unpack_files: Path):
    """
    STAGE 1
    Unpack archives with SNODAS data and save results into new folder with geotiff files
    """
    info_df = []
    path_to_folder = path_to_folder.resolve()

    # Create folder to unpack
    folder_to_unpack_files = folder_to_unpack_files.resolve()
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
                try:
                    if '.txt' not in file.name:
                        continue

                    # Transform into geotiff and save files
                    geotiff_file, product, product_code, data_units = _transform_dat_file_into_geotiff(file)
                    product = product.replace('\n', '')
                    product_code = product_code.replace('\n', '')
                    data_units = data_units.replace('\n', '')

                    info_df.append(pd.DataFrame({'archive_name': archive.name, 'datetime': [date_info],
                                                 'product': [product],
                                                 'product_code': [product_code], 'data_units': [data_units],
                                                 'geotiff': [geotiff_file]}))
                except Exception as ex:
                    logger.warning(f'Can not process SNODAS file {file} due to {ex}')

            # After processing archive - save data
            if len(info_df) > 1:
                updated_df = pd.concat(info_df)
                updated_df = pd.to_datetime(updated_df['datetime']).dt.strftime('%Y-%m-%d')

                updated_df.to_csv(metadata_file, index=False)

    info_df = pd.concat(info_df)
    info_df.to_csv(metadata_file, index=False)


def collect_snodas_data_for_site(path_to_snodas: Path, site: str):
    """ Read prepared csv file for desired site """
    path_to_site = Path(path_to_snodas, f'{site}.csv')
    if path_to_site.is_file() is False:
        return None

    dataframe = pd.read_csv(path_to_site, parse_dates=['datetime'])

    return dataframe
