import gzip
import shutil
import subprocess
from pathlib import Path

import numpy as np
import xarray
from loguru import logger


def _unpack_data(archive_files: list, final_path: Path):
    ######################
    # PROCESS TEXT FILES #
    ######################
    for file in archive_files:
        if '.txt' in str(file.name):
            unpacked_text_file = Path(final_path, f'{file.name.split(".")[0]}.hdr')
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


def collect_snodas_data_for_site(path_to_folder: Path, site_id: str):
    """ Collect SNODAS data for desired site """
    path_to_folder = path_to_folder.resolve()

    # Create folder to unpack
    folder_to_unpack_files = Path(path_to_folder.parent, 'snodas_unpacked')
    folder_to_unpack_files.mkdir(exist_ok=True, parents=True)

    all_years = list(path_to_folder.iterdir())
    all_years.sort()

    for year_folder in all_years[-2:]:
        for archive in list(year_folder.iterdir()):

            path_to_extract = Path(folder_to_unpack_files, year_folder.name, archive.name.split('.')[0])
            path_to_extract.mkdir(exist_ok=True, parents=True)

            code = subprocess.call(['tar', '-xvf', str(archive), '-C', str(path_to_extract)])
            if code != 0:
                logger.warning(f'Can not open archive file {archive}. Code {code}')
                # Remove prepared folder
                shutil.rmtree(path_to_extract)
                continue

            final_path = Path(folder_to_unpack_files, year_folder.name,
                              f"{archive.name.split('.')[0]}_processed")
            final_path.mkdir(exist_ok=True, parents=True)

            archive_files = list(path_to_extract.iterdir())
            # _unpack_data(archive_files, final_path)
            shutil.rmtree(path_to_extract)

            for file in list(final_path.iterdir()):
                if '.dat' not in file.name:
                    continue

                geotiff_file = Path(file.parent, f'{file.name.split(".")[0]}.tif')
                print(f'gdal_translate -of GTiff {file} {geotiff_file}')
                code = subprocess.call(['gdal_translate', '-of', 'GTiff', str(file), str(geotiff_file)])

                a = 0
