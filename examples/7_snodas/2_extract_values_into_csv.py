from pathlib import Path

import warnings


from wasu.development.data.snodas import extract_data_from_geotiff_files_into_csv

warnings.filterwarnings('ignore')


def extract_data_from_geotiff():
    extract_data_from_geotiff_files_into_csv(path_to_folder=Path('../../data/snodas_unpacked'),
                                             folder_for_csv=Path('../../data/snodas_csv'),
                                             vis=False)


if __name__ == '__main__':
    extract_data_from_geotiff()
